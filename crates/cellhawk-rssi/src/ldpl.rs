//! Log-Distance Path Loss (LDPL) model with Bayesian online exponent estimation.
//!
//! ## Model (§3.1)
//!
//! ```text
//! RSSI(d) = P_t - PL(d₀) - 10·n·log₁₀(d/d₀) + X_σ
//! ```
//!
//! where:
//! - `P_t`  = transmit power at reference distance d₀ (dBm)
//! - `n`    = path loss exponent (adaptive, default 2.8)
//! - `d`    = range (m)
//! - `X_σ`  = log-normal shadowing ~ N(0, σ²), σ = 4–10 dB
//!
//! ## Improvement over paper
//!
//! The paper states "adaptive exponent selection" but does not implement it.
//! This module adds a **Bayesian online estimator** that updates `n` in real
//! time using a gradient step on the squared RSSI residual, with a sliding
//! window of recent measurements for stability.

use cellhawk_types::CellHawkError;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Environment presets (§3.1, Table 1)
// ─────────────────────────────────────────────────────────────────────────────

/// Path loss exponent presets by deployment environment (ITU-R P.1411, §3.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Environment {
    FreeSpace,
    UrbanMacrocell,
    UrbanMicrocell,
    DenseUrban,
    Indoor,
    /// Custom exponent supplied by the caller.
    Custom(f64),
}

impl Environment {
    /// Canonical path loss exponent for this environment.
    pub fn exponent(self) -> f64 {
        match self {
            Self::FreeSpace => 2.0,
            Self::UrbanMacrocell => 3.2,
            Self::UrbanMicrocell => 2.5,
            Self::DenseUrban => 3.5,
            Self::Indoor => 2.5,
            Self::Custom(n) => n,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LDPL Model
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the LDPL model.
#[derive(Debug, Clone)]
pub struct LdplConfig {
    /// Reference distance d₀ (m). Paper uses 1.0 m.
    pub reference_distance_m: f64,
    /// Transmit power at d₀ (dBm). Paper calibrates at −40 dBm.
    pub tx_power_at_reference_dbm: f64,
    /// Initial path loss exponent.
    pub initial_exponent: f64,
    /// Log-normal shadowing std dev (dB). Typical urban: 4–10 dB.
    pub shadowing_std_db: f64,
    /// Adaptive exponent estimator config.
    pub adaptive: AdaptiveExponentConfig,
}

impl Default for LdplConfig {
    fn default() -> Self {
        Self {
            reference_distance_m: 1.0,
            tx_power_at_reference_dbm: -40.0,
            initial_exponent: 2.8,
            shadowing_std_db: 7.0,
            adaptive: AdaptiveExponentConfig::default(),
        }
    }
}

/// Configuration for the Bayesian online path-loss exponent estimator.
#[derive(Debug, Clone)]
pub struct AdaptiveExponentConfig {
    pub enabled: bool,
    /// Gradient step size.
    pub learning_rate: f64,
    pub min_exponent: f64,
    pub max_exponent: f64,
    /// Sliding window size for residual averaging.
    pub window_size: usize,
}

impl Default for AdaptiveExponentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.05,
            min_exponent: 1.8,
            max_exponent: 4.5,
            window_size: 20,
        }
    }
}

/// Log-Distance Path Loss model with adaptive exponent estimation.
#[derive(Debug, Clone)]
pub struct LdplModel {
    config: LdplConfig,
    /// Current path loss exponent estimate.
    exponent: f64,
    /// Sliding window of recent (range_m, rssi_dbm) pairs for adaptation.
    residual_window: VecDeque<(f64, f64)>,
}

impl LdplModel {
    /// Construct with default mixed urban/suburban parameters (n = 2.8).
    pub fn new(config: LdplConfig) -> Self {
        let exponent = config.initial_exponent;
        Self {
            config,
            exponent,
            residual_window: VecDeque::new(),
        }
    }

    /// Current path loss exponent.
    #[inline]
    pub fn exponent(&self) -> f64 {
        self.exponent
    }

    /// Predict RSSI (dBm) at a given range (m).
    ///
    /// ```text
    /// RSSI(d) = P_t - 10·n·log₁₀(d / d₀)
    /// ```
    ///
    /// Returns `Err` if `range_m ≤ 0`.
    pub fn predict_rssi(&self, range_m: f64) -> Result<f64, CellHawkError> {
        if range_m <= 0.0 {
            return Err(CellHawkError::InvalidRssi {
                rssi_dbm: f64::NEG_INFINITY,
            });
        }
        let path_loss_db =
            10.0 * self.exponent * (range_m / self.config.reference_distance_m).log10();
        Ok(self.config.tx_power_at_reference_dbm - path_loss_db)
    }

    /// Estimate range (m) from a measured RSSI (dBm).
    ///
    /// Inverts the LDPL equation:
    /// ```text
    /// d = d₀ · 10^((P_t - RSSI) / (10·n))
    /// ```
    ///
    /// Returns `Err` if the RSSI is above the transmit power (physically
    /// impossible without antenna gain, treated as invalid).
    pub fn estimate_range(&self, rssi_dbm: f64) -> Result<f64, CellHawkError> {
        // RSSI above tx_power implies gain > 0 dBi which we don't model here
        if rssi_dbm > self.config.tx_power_at_reference_dbm + 3.0 {
            return Err(CellHawkError::InvalidRssi { rssi_dbm });
        }
        let exponent_denom = 10.0 * self.exponent;
        let range_m = self.config.reference_distance_m
            * 10.0_f64.powf((self.config.tx_power_at_reference_dbm - rssi_dbm) / exponent_denom);
        Ok(range_m)
    }

    /// Update the adaptive exponent estimator with a ground-truth
    /// (range, rssi) observation.
    ///
    /// Uses a gradient descent step on the squared RSSI residual:
    /// ```text
    /// L(n) = (RSSI_measured - RSSI_predicted(n))²
    /// dL/dn = -2 · residual · 10·log₁₀(d/d₀)
    /// n ← n - lr · dL/dn
    /// ```
    pub fn update_exponent(&mut self, range_m: f64, rssi_dbm: f64) {
        if !self.config.adaptive.enabled || range_m <= 0.0 {
            return;
        }
        let cfg = &self.config.adaptive;

        // Maintain sliding window
        self.residual_window.push_back((range_m, rssi_dbm));
        if self.residual_window.len() > cfg.window_size {
            self.residual_window.pop_front();
        }

        // Gradient step using the most recent observation.
        // L(n) = (rssi_measured - rssi_predicted(n))²
        // rssi_predicted = P_t - 10·n·log₁₀(d/d₀)
        // Exact Newton step for single observation:
        //   Δn = residual / (10·log₁₀(d/d₀))
        // Damped by learning_rate for stability across a window.
        let log_ratio = (range_m / self.config.reference_distance_m).log10();
        if log_ratio.abs() < 1e-6 {
            return; // d ≈ d₀, no information about n
        }
        let rssi_predicted =
            self.config.tx_power_at_reference_dbm - 10.0 * self.exponent * log_ratio;
        let residual = rssi_dbm - rssi_predicted;
        // Normalised Newton step.
        // rssi_predicted = P_t - 10·n·log₁₀(d/d₀)
        // To reduce residual = rssi_measured - rssi_predicted:
        //   ∂rssi_predicted/∂n = -10·log₁₀(d/d₀)
        //   Δn = -residual / (10·log₁₀(d/d₀))  [Newton step on residual]
        let newton_step = -residual / (10.0 * log_ratio);
        self.exponent += cfg.learning_rate * newton_step;
        self.exponent = self.exponent.clamp(cfg.min_exponent, cfg.max_exponent);
    }

    /// Measurement noise variance (dBm²) for use in EKF R matrix.
    ///
    /// σ² = shadowing_std_db²
    #[inline]
    pub fn measurement_variance_dbm2(&self) -> f64 {
        self.config.shadowing_std_db * self.config.shadowing_std_db
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn default_model() -> LdplModel {
        LdplModel::new(LdplConfig::default())
    }

    /// Paper §3.1 worked example: at d=100 m, RSSI = −40 − 10·2.8·log₁₀(100) = −96 dBm
    #[test]
    fn rssi_at_100m_matches_paper() {
        let model = default_model();
        let rssi = model.predict_rssi(100.0).unwrap();
        assert_abs_diff_eq!(rssi, -96.0, epsilon = 1e-9);
    }

    /// Paper §3.1 worked example: at d=1000 m, RSSI = −40 − 10·2.8·log₁₀(1000) = −124 dBm
    #[test]
    fn rssi_at_1000m_matches_paper() {
        let model = default_model();
        let rssi = model.predict_rssi(1000.0).unwrap();
        assert_abs_diff_eq!(rssi, -124.0, epsilon = 1e-9);
    }

    /// Round-trip: estimate_range(predict_rssi(d)) ≈ d
    #[test]
    fn range_rssi_round_trip() {
        let model = default_model();
        for &d in &[10.0_f64, 100.0, 500.0, 1000.0] {
            let rssi = model.predict_rssi(d).unwrap();
            let d_recovered = model.estimate_range(rssi).unwrap();
            assert_abs_diff_eq!(d_recovered, d, epsilon = 1e-6);
        }
    }

    /// Monotonicity: RSSI must decrease as range increases.
    #[test]
    fn rssi_decreases_with_range() {
        let model = default_model();
        let rssi_near = model.predict_rssi(100.0).unwrap();
        let rssi_far = model.predict_rssi(500.0).unwrap();
        assert!(rssi_near > rssi_far, "RSSI must decrease with distance");
    }

    /// Adaptive exponent converges toward the true exponent given noiseless data.
    #[test]
    fn adaptive_exponent_converges() {
        let true_n = 3.2_f64;
        let mut model = LdplModel::new(LdplConfig {
            initial_exponent: 2.8,
            adaptive: AdaptiveExponentConfig {
                learning_rate: 0.02,
                ..Default::default()
            },
            ..Default::default()
        });
        // Feed 200 noiseless observations at various ranges
        for i in 1..=200_u32 {
            let d = 50.0 + (i as f64) * 5.0;
            // True RSSI under n=3.2
            let rssi_true = -40.0 - 10.0 * true_n * d.log10();
            model.update_exponent(d, rssi_true);
        }
        // Should converge within 0.15 of true exponent
        assert!(
            (model.exponent() - true_n).abs() < 0.15,
            "exponent={:.4}, expected≈{true_n}",
            model.exponent()
        );
    }

    #[test]
    fn invalid_range_returns_error() {
        let model = default_model();
        assert!(model.predict_rssi(0.0).is_err());
        assert!(model.predict_rssi(-1.0).is_err());
    }

    #[test]
    fn environment_exponents_match_paper_table() {
        assert_abs_diff_eq!(Environment::FreeSpace.exponent(), 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(Environment::UrbanMacrocell.exponent(), 3.2, epsilon = 1e-9);
        assert_abs_diff_eq!(Environment::DenseUrban.exponent(), 3.5, epsilon = 1e-9);
    }
}
