//! Rician fading channel model for RSSI correction and NLoS detection (§3.3).
//!
//! ## Rician K-factor
//!
//! ```text
//! K = P_direct / P_scattered
//! K_dB = 10·log₁₀(K)
//! ```
//!
//! The Rician PDF of the envelope amplitude `r` given a direct component `ν`
//! and scatter power `2σ²` is:
//!
//! ```text
//! f(r) = (r/σ²) · exp(-(r² + ν²)/(2σ²)) · I₀(r·ν/σ²)
//! ```
//!
//! For EKF purposes we need the **mean** and **variance** of the received
//! power in dBm, which we derive from the K-factor.
//!
//! ## NLoS bias correction
//!
//! NLoS conditions add an excess path loss `Δ_NLoS` (dB) on top of the LDPL
//! prediction. We estimate this bias from the K-factor:
//!
//! ```text
//! Δ_NLoS ≈ -10·log₁₀(K/(K+1))   [dB, always ≥ 0]
//! ```
//!
//! This is the power penalty from losing the direct component.

use cellhawk_types::CellHawkError;

// ─────────────────────────────────────────────────────────────────────────────
// Rician Fading Model
// ─────────────────────────────────────────────────────────────────────────────

/// Rician fading channel model.
///
/// Provides:
/// 1. NLoS bias correction for RSSI measurements.
/// 2. Measurement noise variance inflation based on K-factor.
/// 3. LoS/NLoS classification threshold.
#[derive(Debug, Clone)]
pub struct RicianFadingModel {
    /// Rician K-factor in dB. Range: 3–12 dB in simulation (§3.3).
    k_factor_db: f64,
}

impl RicianFadingModel {
    /// Construct with a K-factor in dB.
    ///
    /// # Panics
    /// Does not panic; clamps K to a physically meaningful range.
    pub fn new(k_factor_db: f64) -> Self {
        Self { k_factor_db }
    }

    /// Construct from a linear K-factor ratio.
    pub fn from_linear(k_linear: f64) -> Result<Self, CellHawkError> {
        if k_linear <= 0.0 {
            return Err(CellHawkError::InvalidRssi { rssi_dbm: k_linear });
        }
        Ok(Self {
            k_factor_db: 10.0 * k_linear.log10(),
        })
    }

    /// K-factor in dB.
    #[inline]
    pub fn k_factor_db(&self) -> f64 {
        self.k_factor_db
    }

    /// K-factor as a linear ratio.
    #[inline]
    pub fn k_factor_linear(&self) -> f64 {
        // Clamp to prevent overflow: K > 100 dB is effectively perfect LoS
        10.0_f64.powf(self.k_factor_db.min(100.0) / 10.0)
    }

    /// NLoS excess path loss bias (dB, always ≥ 0).
    ///
    /// This is the power penalty from the absence of a direct LoS component:
    /// ```text
    /// Δ_NLoS = -10·log₁₀(K/(K+1))
    /// ```
    ///
    /// At K = 6 dB (K_lin ≈ 4): Δ ≈ 0.97 dB (mild)
    /// At K = 0 dB (K_lin = 1):  Δ ≈ 3.01 dB (moderate)
    /// At K → −∞ (Rayleigh):     Δ → ∞ (severe NLoS)
    pub fn nlos_bias_db(&self) -> f64 {
        let k = self.k_factor_linear();
        -10.0 * (k / (k + 1.0)).log10()
    }

    /// Apply NLoS bias correction to a raw RSSI measurement.
    ///
    /// Adds back the estimated NLoS power penalty so the corrected RSSI
    /// better reflects the free-space path loss.
    ///
    /// ```text
    /// RSSI_corrected = RSSI_measured + Δ_NLoS
    /// ```
    pub fn correct_rssi(&self, rssi_dbm: f64) -> f64 {
        rssi_dbm + self.nlos_bias_db()
    }

    /// Measurement noise variance inflation factor (dimensionless).
    ///
    /// As K decreases (more NLoS), fading variance increases. We model the
    /// additional variance as proportional to `1/(K+1)` in linear scale,
    /// converted to dB²:
    ///
    /// ```text
    /// σ²_fading ≈ (10/ln(10))² · 1/(K+1)   [dB²]
    /// ```
    pub fn variance_inflation_db2(&self) -> f64 {
        let k = self.k_factor_linear();
        let scale = 10.0 / std::f64::consts::LN_10; // ≈ 4.343
        scale * scale / (k + 1.0)
    }

    /// Classify the link as LoS or NLoS based on K-factor threshold.
    ///
    /// K < 0 dB → NLoS dominant (approaches Rayleigh fading, §3.3).
    #[inline]
    pub fn is_nlos(&self) -> bool {
        self.k_factor_db < 0.0
    }

    /// Probability that the link is NLoS, modelled as a sigmoid on K_dB.
    ///
    /// Returns 0.0 for strong LoS (K >> 0) and 1.0 for strong NLoS (K << 0).
    pub fn nlos_probability(&self) -> f64 {
        // Sigmoid centred at K=0 dB with scale factor 0.5
        1.0 / (1.0 + (0.5 * self.k_factor_db).exp())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// At K → ∞ (perfect LoS), NLoS bias → 0.
    #[test]
    fn nlos_bias_vanishes_at_high_k() {
        let model = RicianFadingModel::new(30.0); // K = 30 dB ≈ 1000 linear
        assert!(
            model.nlos_bias_db() < 0.05,
            "bias={:.4}",
            model.nlos_bias_db()
        );
    }

    /// At K = 0 dB (K_lin = 1), bias = -10·log₁₀(0.5) ≈ 3.01 dB.
    #[test]
    fn nlos_bias_at_k_zero_db() {
        let model = RicianFadingModel::new(0.0);
        assert_abs_diff_eq!(model.nlos_bias_db(), 3.0103, epsilon = 1e-3);
    }

    /// NLoS bias is always non-negative (power can only be lost, not gained).
    #[test]
    fn nlos_bias_is_non_negative() {
        for k_db in [-10.0_f64, -3.0, 0.0, 3.0, 6.0, 12.0, 20.0] {
            let model = RicianFadingModel::new(k_db);
            assert!(
                model.nlos_bias_db() >= 0.0,
                "k_db={k_db}, bias={:.4}",
                model.nlos_bias_db()
            );
        }
    }

    /// Variance inflation decreases as K increases (better LoS → less fading).
    #[test]
    fn variance_inflation_decreases_with_k() {
        let low_k = RicianFadingModel::new(3.0);
        let high_k = RicianFadingModel::new(12.0);
        assert!(
            low_k.variance_inflation_db2() > high_k.variance_inflation_db2(),
            "low_k var={:.4}, high_k var={:.4}",
            low_k.variance_inflation_db2(),
            high_k.variance_inflation_db2()
        );
    }

    /// NLoS classification: K < 0 dB → NLoS.
    #[test]
    fn nlos_classification() {
        assert!(RicianFadingModel::new(-1.0).is_nlos());
        assert!(!RicianFadingModel::new(0.0).is_nlos());
        assert!(!RicianFadingModel::new(6.0).is_nlos());
    }

    /// NLoS probability is 0.5 at K = 0 dB (sigmoid midpoint).
    #[test]
    fn nlos_probability_at_k_zero() {
        let model = RicianFadingModel::new(0.0);
        assert_abs_diff_eq!(model.nlos_probability(), 0.5, epsilon = 1e-9);
    }

    /// Round-trip: from_linear(k_linear).k_factor_linear() ≈ k_linear.
    #[test]
    fn linear_round_trip() {
        let k_lin = 4.0_f64;
        let model = RicianFadingModel::from_linear(k_lin).unwrap();
        assert_abs_diff_eq!(model.k_factor_linear(), k_lin, epsilon = 1e-9);
    }
}
