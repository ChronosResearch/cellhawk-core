//! Navigation tier arbitration with GPS spoofing detection (§3.4, §9 gap fix).
//!
//! ## Tier Arbitration
//!
//! The arbiter consumes real-time JNR measurements and drives the
//! `CovarianceScaler` to request tier transitions. It also implements
//! **GPS spoofing detection** — a known gap in the paper (§9, limitation 5).
//!
//! ## Spoofing Detection
//!
//! When Tier 1 (GNSS) is active, the arbiter cross-checks the GNSS position
//! against the cellular RSSI multilateration position. If the horizontal
//! discrepancy exceeds a configurable threshold, a spoofing alert is raised
//! and the system transitions to Tier 2.
//!
//! This is a lightweight consistency check, not a full spoofing detector,
//! but it closes the most obvious attack vector described in §9.

use cellhawk_types::{CellHawkError, EnuPosition, NavigationTier};

use crate::covariance::CovarianceScaler;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the tier arbiter.
#[derive(Debug, Clone)]
pub struct TierArbiterConfig {
    /// JNR threshold for Tier 1 → 2 handover (dB). Paper: 6 dB.
    pub tier1_jnr_threshold_db: f64,
    /// JNR threshold for Tier 2 → 3 handover (dB). Paper: 19 dB.
    pub tier2_jnr_threshold_db: f64,
    /// Hysteresis band (dB) to prevent rapid tier oscillation.
    pub hysteresis_db: f64,
    /// Spoofing detection: GNSS vs cellular position discrepancy threshold (m).
    pub spoofing_threshold_m: f64,
    /// Number of consecutive spoofing detections before triggering alert.
    pub spoofing_confirmation_count: usize,
    /// Ramp cycles for smooth handover (paper: 5 cycles).
    pub ramp_cycles: usize,
}

impl Default for TierArbiterConfig {
    fn default() -> Self {
        Self {
            tier1_jnr_threshold_db: NavigationTier::TIER1_JNR_THRESHOLD_DB,
            tier2_jnr_threshold_db: NavigationTier::TIER2_JNR_THRESHOLD_DB,
            hysteresis_db: 1.5,
            spoofing_threshold_m: 150.0,
            spoofing_confirmation_count: 3,
            ramp_cycles: 5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier Arbiter
// ─────────────────────────────────────────────────────────────────────────────

/// Navigation tier arbiter.
///
/// Consumes JNR measurements and optional GNSS/cellular position pairs,
/// drives the `CovarianceScaler`, and raises spoofing alerts.
#[derive(Debug)]
pub struct TierArbiter {
    config: TierArbiterConfig,
    scaler: CovarianceScaler,
    /// Consecutive spoofing detection count.
    spoofing_count: usize,
    /// Whether spoofing is currently suspected.
    spoofing_suspected: bool,
    /// Last JNR measurement (dB).
    last_jnr_db: f64,
}

impl TierArbiter {
    pub fn new(config: TierArbiterConfig) -> Self {
        let scaler = CovarianceScaler::new(NavigationTier::GnssActive, config.ramp_cycles);
        Self {
            config,
            scaler,
            spoofing_count: 0,
            spoofing_suspected: false,
            last_jnr_db: 0.0,
        }
    }

    /// Current active navigation tier.
    #[inline]
    pub fn active_tier(&self) -> NavigationTier {
        self.scaler.active_tier()
    }

    /// Whether GPS spoofing is currently suspected.
    #[inline]
    pub fn spoofing_suspected(&self) -> bool {
        self.spoofing_suspected
    }

    /// Reference to the covariance scaler (for EKF measurement noise scaling).
    #[inline]
    pub fn scaler(&self) -> &CovarianceScaler {
        &self.scaler
    }

    /// Mutable reference to the covariance scaler.
    #[inline]
    pub fn scaler_mut(&mut self) -> &mut CovarianceScaler {
        &mut self.scaler
    }

    /// Update the arbiter with a new JNR measurement.
    ///
    /// Applies hysteresis to prevent rapid tier oscillation:
    /// - Upgrade (lower tier number): requires JNR to drop below threshold − hysteresis
    /// - Downgrade (higher tier number): requires JNR to exceed threshold + hysteresis
    pub fn update_jnr(&mut self, jnr_db: f64) {
        self.last_jnr_db = jnr_db;
        let current = self.scaler.active_tier();
        let new_tier = self.tier_with_hysteresis(jnr_db, current);

        if new_tier != current {
            self.scaler.request_tier(new_tier);
        }
        self.scaler.step();
    }

    /// Cross-check GNSS position against cellular position for spoofing detection.
    ///
    /// Only meaningful when Tier 1 (GNSS) is active and a cellular fix is
    /// available. Returns `Err(GpsSpoofingSuspected)` if the discrepancy
    /// exceeds the configured threshold for `spoofing_confirmation_count`
    /// consecutive calls.
    pub fn check_spoofing(
        &mut self,
        gnss_pos: &EnuPosition,
        cellular_pos: &EnuPosition,
    ) -> Result<(), CellHawkError> {
        if self.scaler.active_tier() != NavigationTier::GnssActive {
            // Only check when GNSS is the primary source
            self.spoofing_count = 0;
            return Ok(());
        }

        let residual_m = gnss_pos.horizontal_distance_to(cellular_pos);

        if residual_m > self.config.spoofing_threshold_m {
            self.spoofing_count += 1;
            if self.spoofing_count >= self.config.spoofing_confirmation_count {
                self.spoofing_suspected = true;
                // Force transition to Tier 2
                self.scaler.request_tier(NavigationTier::CellularRssi);
                return Err(CellHawkError::GpsSpoofingSuspected {
                    residual_m,
                    threshold_m: self.config.spoofing_threshold_m,
                });
            }
        } else {
            // Reset counter on a clean check
            self.spoofing_count = 0;
            self.spoofing_suspected = false;
        }

        Ok(())
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Determine the target tier from JNR with hysteresis.
    fn tier_with_hysteresis(&self, jnr_db: f64, current: NavigationTier) -> NavigationTier {
        let h = self.config.hysteresis_db;
        match current {
            NavigationTier::GnssActive => {
                // Downgrade to Tier 2 only if JNR exceeds threshold + hysteresis
                if jnr_db >= self.config.tier1_jnr_threshold_db + h {
                    NavigationTier::CellularRssi
                } else {
                    NavigationTier::GnssActive
                }
            }
            NavigationTier::CellularRssi => {
                if jnr_db >= self.config.tier2_jnr_threshold_db + h {
                    NavigationTier::VisualSlam
                } else if jnr_db < self.config.tier1_jnr_threshold_db - h {
                    NavigationTier::GnssActive
                } else {
                    NavigationTier::CellularRssi
                }
            }
            NavigationTier::VisualSlam => {
                // Upgrade to Tier 2 only if JNR drops below threshold − hysteresis
                if jnr_db < self.config.tier2_jnr_threshold_db - h {
                    NavigationTier::CellularRssi
                } else {
                    NavigationTier::VisualSlam
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_arbiter() -> TierArbiter {
        TierArbiter::new(TierArbiterConfig::default())
    }

    #[test]
    fn starts_in_tier1() {
        let arbiter = default_arbiter();
        assert_eq!(arbiter.active_tier(), NavigationTier::GnssActive);
    }

    #[test]
    fn transitions_to_tier2_above_threshold() {
        let mut arbiter = default_arbiter();
        // JNR = 8 dB > 6 + 1.5 = 7.5 dB → should trigger T1→T2
        for _ in 0..10 {
            arbiter.update_jnr(8.0);
        }
        assert_eq!(arbiter.active_tier(), NavigationTier::CellularRssi);
    }

    #[test]
    fn hysteresis_prevents_oscillation_at_boundary() {
        let mut arbiter = default_arbiter();
        // JNR = 6.5 dB — above T1 threshold (6.0) but below T1+hysteresis (7.5)
        for _ in 0..20 {
            arbiter.update_jnr(6.5);
        }
        // Should remain in Tier 1 due to hysteresis
        assert_eq!(arbiter.active_tier(), NavigationTier::GnssActive);
    }

    #[test]
    fn transitions_to_tier3_above_tier2_threshold() {
        let mut arbiter = default_arbiter();
        // First get to Tier 2
        for _ in 0..10 {
            arbiter.update_jnr(8.0);
        }
        assert_eq!(arbiter.active_tier(), NavigationTier::CellularRssi);
        // Now push to Tier 3: JNR = 21 dB > 19 + 1.5 = 20.5 dB
        for _ in 0..10 {
            arbiter.update_jnr(21.0);
        }
        assert_eq!(arbiter.active_tier(), NavigationTier::VisualSlam);
    }

    #[test]
    fn recovers_from_tier3_to_tier2_when_jamming_clears() {
        let mut arbiter = default_arbiter();
        // Drive to Tier 3
        for _ in 0..20 {
            arbiter.update_jnr(25.0);
        }
        assert_eq!(arbiter.active_tier(), NavigationTier::VisualSlam);
        // JNR drops to 16 dB < 19 − 1.5 = 17.5 dB → back to Tier 2
        for _ in 0..20 {
            arbiter.update_jnr(16.0);
        }
        assert_eq!(arbiter.active_tier(), NavigationTier::CellularRssi);
    }

    #[test]
    fn spoofing_detected_after_confirmation_count() {
        let mut arbiter = default_arbiter();
        let gnss = EnuPosition {
            east_m: 0.0,
            north_m: 0.0,
            up_m: 50.0,
        };
        // Cellular position 200 m away — exceeds 150 m threshold
        let cellular = EnuPosition {
            east_m: 200.0,
            north_m: 0.0,
            up_m: 50.0,
        };

        // First 2 checks: no alert yet (confirmation_count = 3)
        for _ in 0..2 {
            let _ = arbiter.check_spoofing(&gnss, &cellular);
        }
        assert!(!arbiter.spoofing_suspected());

        // Third check: alert triggered
        let result = arbiter.check_spoofing(&gnss, &cellular);
        assert!(result.is_err());
        assert!(arbiter.spoofing_suspected());
        assert_eq!(arbiter.active_tier(), NavigationTier::CellularRssi);
    }

    #[test]
    fn spoofing_counter_resets_on_clean_check() {
        let mut arbiter = default_arbiter();
        let gnss = EnuPosition {
            east_m: 0.0,
            north_m: 0.0,
            up_m: 50.0,
        };
        let cellular_bad = EnuPosition {
            east_m: 200.0,
            north_m: 0.0,
            up_m: 50.0,
        };
        let cellular_ok = EnuPosition {
            east_m: 5.0,
            north_m: 0.0,
            up_m: 50.0,
        };

        // Two bad checks
        for _ in 0..2 {
            let _ = arbiter.check_spoofing(&gnss, &cellular_bad);
        }
        // One clean check — resets counter
        arbiter.check_spoofing(&gnss, &cellular_ok).unwrap();
        // Two more bad checks — should not trigger (counter was reset)
        for _ in 0..2 {
            let _ = arbiter.check_spoofing(&gnss, &cellular_bad);
        }
        assert!(!arbiter.spoofing_suspected());
    }
}
