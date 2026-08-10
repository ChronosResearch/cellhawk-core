//! JNR-driven dynamic covariance scaling with sigmoid ramp (§3.4).
//!
//! ## Smooth Handover
//!
//! The paper specifies a 5-cycle covariance ramp to prevent EKF divergence
//! at tier transition boundaries. We implement this as a **sigmoid** rather
//! than a linear ramp, which provides:
//!
//! - Zero derivative at both ends (no abrupt covariance jumps)
//! - Monotonic transition
//! - Configurable ramp duration
//!
//! The trust weight α for a sensor source transitions from 1.0 (full trust)
//! to 0.0 (zero trust) over `ramp_cycles` EKF update cycles:
//!
//! ```text
//! α(t) = 1 / (1 + exp(k · (t - t_mid)))
//! ```
//!
//! where `k` is chosen so that α(0) ≈ 0.99 and α(ramp_cycles) ≈ 0.01.

use cellhawk_types::NavigationTier;

// ─────────────────────────────────────────────────────────────────────────────
// Per-source trust weights
// ─────────────────────────────────────────────────────────────────────────────

/// Trust weights for each navigation source, in [0.0, 1.0].
#[derive(Debug, Clone, Copy)]
pub struct SourceWeights {
    /// GNSS pseudo-range trust weight.
    pub gnss: f64,
    /// Cellular RSSI multilateration trust weight.
    pub cellular: f64,
    /// Visual SLAM odometry trust weight.
    pub slam: f64,
}

impl SourceWeights {
    /// Full trust to GNSS, secondary to cellular and SLAM (Tier 1).
    pub const TIER1: Self = Self {
        gnss: 1.0,
        cellular: 0.3,
        slam: 0.2,
    };
    /// Zero GNSS, full cellular, SLAM as velocity constraint (Tier 2).
    pub const TIER2: Self = Self {
        gnss: 0.0,
        cellular: 1.0,
        slam: 0.4,
    };
    /// Zero GNSS and cellular, full SLAM (Tier 3).
    pub const TIER3: Self = Self {
        gnss: 0.0,
        cellular: 0.0,
        slam: 1.0,
    };

    /// Target weights for a given tier.
    pub fn target_for_tier(tier: NavigationTier) -> Self {
        match tier {
            NavigationTier::GnssActive => Self::TIER1,
            NavigationTier::CellularRssi => Self::TIER2,
            NavigationTier::VisualSlam => Self::TIER3,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Covariance Scaler
// ─────────────────────────────────────────────────────────────────────────────

/// JNR-driven dynamic covariance scaler with sigmoid ramp.
///
/// Maintains the current source trust weights and smoothly transitions them
/// when the active navigation tier changes.
#[derive(Debug, Clone)]
pub struct CovarianceScaler {
    /// Current (possibly mid-transition) trust weights.
    current_weights: SourceWeights,
    /// Target weights for the new tier.
    target_weights: SourceWeights,
    /// Source weights at the start of the current transition.
    start_weights: SourceWeights,
    /// Number of EKF cycles over which to ramp (paper: 5 cycles).
    ramp_cycles: usize,
    /// Current cycle within the ramp [0, ramp_cycles].
    ramp_progress: usize,
    /// Whether a transition is currently in progress.
    transitioning: bool,
    /// Active tier.
    active_tier: NavigationTier,
}

impl CovarianceScaler {
    /// Construct with initial tier and ramp duration.
    pub fn new(initial_tier: NavigationTier, ramp_cycles: usize) -> Self {
        let weights = SourceWeights::target_for_tier(initial_tier);
        Self {
            current_weights: weights,
            target_weights: weights,
            start_weights: weights,
            ramp_cycles,
            ramp_progress: 0,
            transitioning: false,
            active_tier: initial_tier,
        }
    }

    /// Current source trust weights (may be mid-transition).
    #[inline]
    pub fn weights(&self) -> SourceWeights {
        self.current_weights
    }

    /// Active navigation tier.
    #[inline]
    pub fn active_tier(&self) -> NavigationTier {
        self.active_tier
    }

    /// Whether a tier transition is currently ramping.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        self.transitioning
    }

    /// Request a transition to a new tier.
    ///
    /// If the tier is unchanged, this is a no-op.
    /// If a transition is already in progress, the new target overrides it
    /// (the ramp restarts from the current weights).
    pub fn request_tier(&mut self, new_tier: NavigationTier) {
        if new_tier == self.active_tier && !self.transitioning {
            return;
        }
        self.active_tier = new_tier;
        self.start_weights = self.current_weights;
        self.target_weights = SourceWeights::target_for_tier(new_tier);
        self.ramp_progress = 0;
        self.transitioning = true;
    }

    /// Advance the ramp by one EKF cycle.
    ///
    /// Must be called once per EKF update step. Updates `current_weights`
    /// using a sigmoid interpolation between `start_weights` and
    /// `target_weights`.
    pub fn step(&mut self) {
        if !self.transitioning {
            return;
        }

        self.ramp_progress += 1;
        let alpha = sigmoid_ramp(self.ramp_progress, self.ramp_cycles);

        self.current_weights = SourceWeights {
            gnss: lerp(self.start_weights.gnss, self.target_weights.gnss, alpha),
            cellular: lerp(
                self.start_weights.cellular,
                self.target_weights.cellular,
                alpha,
            ),
            slam: lerp(self.start_weights.slam, self.target_weights.slam, alpha),
        };

        if self.ramp_progress >= self.ramp_cycles {
            self.current_weights = self.target_weights;
            self.transitioning = false;
        }
    }

    /// Scale a measurement noise variance by the inverse of the source weight.
    ///
    /// When weight → 0, the variance → ∞ (measurement is ignored by EKF).
    /// When weight = 1, the variance is unchanged.
    ///
    /// ```text
    /// R_scaled = R_base / max(weight, ε)
    /// ```
    pub fn scale_measurement_noise(&self, base_variance: f64, source_weight: f64) -> f64 {
        let w = source_weight.max(1e-6);
        base_variance / w
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid ramp helper
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid interpolation factor for smooth tier handover.
///
/// Returns a value in [0, 1] that follows a sigmoid curve from 0 to 1
/// as `cycle` goes from 0 to `total_cycles`.
///
/// The sigmoid is parameterised so that:
/// - At cycle=0:            α ≈ 0.01
/// - At cycle=total/2:      α = 0.50
/// - At cycle=total_cycles: α ≈ 0.99
fn sigmoid_ramp(cycle: usize, total_cycles: usize) -> f64 {
    if total_cycles == 0 {
        return 1.0;
    }
    // Map cycle to [-6, +6] range for sigmoid
    let t = (cycle as f64 / total_cycles as f64) * 12.0 - 6.0;
    1.0 / (1.0 + (-t).exp())
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sigmoid_ramp_endpoints() {
        // At cycle=0, α should be close to 0
        assert!(sigmoid_ramp(0, 5) < 0.05);
        // At cycle=total, α should be close to 1
        assert!(sigmoid_ramp(5, 5) > 0.95);
        // Midpoint should be 0.5
        assert_abs_diff_eq!(sigmoid_ramp(5, 10), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_ramp_is_monotonic() {
        let total = 10;
        let mut prev = sigmoid_ramp(0, total);
        for c in 1..=total {
            let curr = sigmoid_ramp(c, total);
            assert!(curr >= prev, "sigmoid not monotonic at cycle {c}");
            prev = curr;
        }
    }

    #[test]
    fn tier1_weights_are_gnss_dominant() {
        let w = SourceWeights::TIER1;
        assert!(w.gnss > w.cellular);
        assert!(w.gnss > w.slam);
    }

    #[test]
    fn tier2_weights_zero_gnss() {
        let w = SourceWeights::TIER2;
        assert_abs_diff_eq!(w.gnss, 0.0, epsilon = 1e-9);
        assert!(w.cellular > 0.0);
    }

    #[test]
    fn tier3_weights_zero_gnss_and_cellular() {
        let w = SourceWeights::TIER3;
        assert_abs_diff_eq!(w.gnss, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(w.cellular, 0.0, epsilon = 1e-9);
        assert!(w.slam > 0.0);
    }

    #[test]
    fn transition_completes_after_ramp_cycles() {
        let mut scaler = CovarianceScaler::new(NavigationTier::GnssActive, 5);
        scaler.request_tier(NavigationTier::CellularRssi);
        assert!(scaler.is_transitioning());

        for _ in 0..5 {
            scaler.step();
        }
        assert!(!scaler.is_transitioning());
        assert_abs_diff_eq!(scaler.weights().gnss, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(scaler.weights().cellular, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn gnss_weight_decreases_during_transition() {
        let mut scaler = CovarianceScaler::new(NavigationTier::GnssActive, 10);
        scaler.request_tier(NavigationTier::CellularRssi);

        let mut prev_gnss = scaler.weights().gnss;
        for _ in 0..10 {
            scaler.step();
            let curr_gnss = scaler.weights().gnss;
            assert!(
                curr_gnss <= prev_gnss + 1e-9,
                "GNSS weight must not increase during T1→T2"
            );
            prev_gnss = curr_gnss;
        }
    }

    #[test]
    fn scale_measurement_noise_zero_weight_gives_large_variance() {
        let scaler = CovarianceScaler::new(NavigationTier::GnssActive, 5);
        let scaled = scaler.scale_measurement_noise(100.0, 0.0);
        assert!(
            scaled > 1e7,
            "zero weight should inflate variance to near-infinity"
        );
    }

    #[test]
    fn scale_measurement_noise_full_weight_unchanged() {
        let scaler = CovarianceScaler::new(NavigationTier::GnssActive, 5);
        let scaled = scaler.scale_measurement_noise(100.0, 1.0);
        assert_abs_diff_eq!(scaled, 100.0, epsilon = 1e-9);
    }
}
