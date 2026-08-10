//! Jammer-to-Noise Ratio (JNR) estimator using the Minimum Statistics algorithm.
//!
//! ## Algorithm (Doblinger 1995, adapted for SDR)
//!
//! The noise floor is estimated as the minimum power observed over a sliding
//! window of recent power estimates, corrected for the statistical bias of the
//! minimum of a chi-squared random variable:
//!
//! ```text
//! P_noise(k) = min{ P(k-W), …, P(k) } · β
//! JNR(k)     = max(0,  P(k) / P_noise(k) − 1)   [linear]
//! JNR_dB(k)  = 10 · log₁₀(JNR(k) + 1)
//! ```
//!
//! where `β ≥ 1` is the bias correction factor (default 1.5, empirical for
//! chi-squared with 2 DOF at window size 100).
//!
//! ## Why minimum statistics?
//!
//! Unlike a simple moving average, minimum statistics tracks the noise floor
//! even when a strong jammer is present — the minimum of the window will
//! reflect the quietest recent period, which is the best estimate of the
//! true noise floor.

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the JNR estimator.
#[derive(Debug, Clone)]
pub struct JnrEstimatorConfig {
    /// Noise floor window size (number of power estimates).
    ///
    /// Larger windows give a more stable noise floor estimate but respond
    /// more slowly to changes in the jamming environment.
    /// Default: 100 estimates ≈ 10 seconds at 10 Hz.
    pub noise_window: usize,
    /// Bias correction factor for the minimum statistics estimator.
    ///
    /// Compensates for the downward bias of the minimum of a chi-squared
    /// random variable.  Empirical value: 1.5 for 2-DOF chi-squared.
    pub bias_correction: f64,
    /// Maximum JNR output (dB).  Clamps extreme values from hardware glitches.
    pub max_jnr_db: f64,
    /// Minimum power (linear) below which the noise floor is considered
    /// unreliable (avoids division by near-zero).
    pub min_noise_floor_linear: f64,
}

impl Default for JnrEstimatorConfig {
    fn default() -> Self {
        Self {
            noise_window: 100,
            bias_correction: 1.5,
            max_jnr_db: 40.0,
            min_noise_floor_linear: 1e-12,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JnrEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum-statistics JNR estimator.
///
/// Accepts mean power estimates (dBm) from [`PowerEstimator`] and outputs
/// JNR in dB for the EKF tier arbiter.
#[derive(Debug)]
pub struct JnrEstimator {
    config: JnrEstimatorConfig,
    /// Sliding window of recent power estimates in linear scale.
    noise_window: VecDeque<f64>,
    /// Cached noise floor estimate (linear).
    noise_floor: f64,
    /// Last JNR estimate (dB).
    last_jnr_db: f64,
}

impl JnrEstimator {
    pub fn new(config: JnrEstimatorConfig) -> Self {
        Self {
            noise_floor: config.min_noise_floor_linear,
            noise_window: VecDeque::with_capacity(config.noise_window + 1),
            last_jnr_db: 0.0,
            config,
        }
    }

    /// Update with a new mean power estimate (dBm) and return JNR in dB.
    ///
    /// The power estimate is typically the output of [`PowerEstimator::update`].
    pub fn update(&mut self, mean_power_dbm: f64) -> f64 {
        // Convert dBm → linear for minimum statistics
        let p_linear = if mean_power_dbm <= -200.0 {
            self.config.min_noise_floor_linear
        } else {
            10.0_f64.powf(mean_power_dbm / 10.0)
        };

        // Update noise floor window
        self.noise_window.push_back(p_linear);
        if self.noise_window.len() > self.config.noise_window {
            self.noise_window.pop_front();
        }

        // Noise floor = minimum of window × bias correction
        let min_power = self
            .noise_window
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        self.noise_floor =
            (min_power * self.config.bias_correction).max(self.config.min_noise_floor_linear);

        // JNR = (P_signal / P_noise) − 1, floored at 0
        let jnr_linear = (p_linear / self.noise_floor - 1.0).max(0.0);
        let jnr_db = if jnr_linear < 1e-10 {
            0.0
        } else {
            (10.0 * (jnr_linear + 1.0).log10()).min(self.config.max_jnr_db)
        };

        self.last_jnr_db = jnr_db;
        jnr_db
    }

    /// Last computed JNR (dB) without updating.
    #[inline]
    pub fn last_jnr_db(&self) -> f64 {
        self.last_jnr_db
    }

    /// Current noise floor estimate (linear).
    #[inline]
    pub fn noise_floor_linear(&self) -> f64 {
        self.noise_floor
    }

    /// Current noise floor estimate (dBm).
    pub fn noise_floor_dbm(&self) -> f64 {
        if self.noise_floor < 1e-20 {
            return -200.0;
        }
        10.0 * self.noise_floor.log10()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator() -> JnrEstimator {
        JnrEstimator::new(JnrEstimatorConfig::default())
    }

    #[test]
    fn initial_jnr_is_zero() {
        let est = make_estimator();
        assert_eq!(est.last_jnr_db(), 0.0);
    }

    #[test]
    fn constant_power_gives_near_zero_jnr() {
        // When signal power equals noise floor, JNR ≈ 0 dB
        let mut est = make_estimator();
        for _ in 0..200 {
            let jnr = est.update(-60.0); // constant −60 dBm
            let _ = jnr;
        }
        // After window fills, noise floor ≈ signal power → JNR ≈ 0
        assert!(
            est.last_jnr_db() < 5.0,
            "constant power JNR={:.2}",
            est.last_jnr_db()
        );
    }

    #[test]
    fn strong_signal_above_noise_gives_positive_jnr() {
        let mut est = make_estimator();
        // Establish noise floor at −80 dBm
        for _ in 0..150 {
            est.update(-80.0);
        }
        // Inject strong signal at −40 dBm (40 dB above noise floor)
        let jnr = est.update(-40.0);
        assert!(
            jnr > 10.0,
            "strong signal must give JNR > 10 dB, got {:.2}",
            jnr
        );
    }

    #[test]
    fn jnr_clamped_to_max() {
        let cfg = JnrEstimatorConfig {
            max_jnr_db: 30.0,
            ..Default::default()
        };
        let mut est = JnrEstimator::new(cfg);
        // Establish very low noise floor
        for _ in 0..150 {
            est.update(-120.0);
        }
        // Inject extremely strong signal
        let jnr = est.update(0.0);
        assert!(
            jnr <= 30.0,
            "JNR must be clamped to max_jnr_db, got {:.2}",
            jnr
        );
    }

    #[test]
    fn jnr_is_non_negative() {
        let mut est = make_estimator();
        for p in [-120.0_f64, -100.0, -80.0, -60.0, -40.0, -20.0, 0.0] {
            let jnr = est.update(p);
            assert!(
                jnr >= 0.0,
                "JNR must be non-negative, got {:.2} at p={p}",
                jnr
            );
        }
    }

    #[test]
    fn noise_floor_tracks_minimum() {
        let mut est = JnrEstimator::new(JnrEstimatorConfig {
            noise_window: 10,
            bias_correction: 1.0, // disable bias correction for this test
            ..Default::default()
        });
        // Feed 10 samples at −60 dBm
        for _ in 0..10 {
            est.update(-60.0);
        }
        let floor_before = est.noise_floor_dbm();
        // Feed 10 samples at −80 dBm (lower power)
        for _ in 0..10 {
            est.update(-80.0);
        }
        let floor_after = est.noise_floor_dbm();
        // Noise floor should have dropped toward −80 dBm
        assert!(
            floor_after < floor_before,
            "floor_before={floor_before:.1} floor_after={floor_after:.1}"
        );
    }

    #[test]
    fn tier1_threshold_reachable_with_strong_jamming() {
        use cellhawk_types::NavigationTier;
        let mut est = make_estimator();
        // Establish noise floor
        for _ in 0..150 {
            est.update(-90.0);
        }
        // Strong jammer: −30 dBm (60 dB above noise floor)
        let jnr = est.update(-30.0);
        assert!(
            jnr >= NavigationTier::TIER1_JNR_THRESHOLD_DB,
            "strong jammer must exceed Tier1 threshold ({} dB), got {:.2}",
            NavigationTier::TIER1_JNR_THRESHOLD_DB,
            jnr
        );
    }
}
