//! 9-state Extended Kalman Filter for CellHawk navigation fusion (§3.4).
//!
//! State vector x (9×1):
//!   [0] east_m        [1] north_m       [2] up_m
//!   [3] v_east_m_s    [4] v_north_m_s   [5] v_up_m_s
//!   [6] heading_rad   [7] jnr_db        [8] accel_bias_z
//!
//! Predict: x_k|k-1 = F·x_k-1,  P_k|k-1 = F·P·Fᵀ + Q
//! Update:  K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹,  x = x + K·z,  P = (I-K·H)·P

use cellhawk_types::{
    CellHawkError, EnuPosition, GnssMeasurement, ImuMeasurement, NavigationState, NavigationTier,
    SlamMeasurement,
};
use nalgebra::{SMatrix, SVector, Vector3};

use crate::tier::{TierArbiter, TierArbiterConfig};

// ── dimension aliases ─────────────────────────────────────────────────────────
const N: usize = 9;
type StateVec = SVector<f64, N>;
type StateMat = SMatrix<f64, N, N>;

// ── measurement dimensions ────────────────────────────────────────────────────
const GNSS_DIM: usize = 3; // east, north, up
const CELLULAR_DIM: usize = 2; // east, north  (altitude from baro)
const SLAM_DIM: usize = 3; // v_east, v_north, v_up
                           // ─────────────────────────────────────────────────────────────────────────────
                           // EKF configuration
                           // ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EkfConfig {
    /// Process noise: position (m²/s)
    pub q_position: f64,
    /// Process noise: velocity (m²/s³)
    pub q_velocity: f64,
    /// Process noise: heading (rad²/s)
    pub q_heading: f64,
    /// Process noise: JNR random walk (dB²/s)
    pub q_jnr: f64,
    /// Process noise: accel bias (m²/s⁵)
    pub q_bias: f64,
    /// Measurement noise: GNSS position (m)
    pub r_gnss_m: f64,
    /// Measurement noise: cellular position (m)
    pub r_cellular_m: f64,
    /// Measurement noise: SLAM velocity (m/s)
    pub r_slam_m_s: f64,
    /// Innovation gate: reject if Mahalanobis distance > threshold (σ)
    pub innovation_gate_sigma: f64,
    /// Huber loss threshold (σ) for robust M-estimation
    pub huber_threshold_sigma: f64,
    /// Tier arbiter config
    pub tier: TierArbiterConfig,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            q_position: 0.01,
            q_velocity: 0.1,
            q_heading: 0.001,
            q_jnr: 1.0,
            q_bias: 1e-6,
            r_gnss_m: 4.5,
            r_cellular_m: 42.0,
            r_slam_m_s: 0.5,
            innovation_gate_sigma: 3.0,
            huber_threshold_sigma: 1.5,
            tier: TierArbiterConfig::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF
// ─────────────────────────────────────────────────────────────────────────────

pub struct CellHawkEkf {
    /// State estimate x̂
    x: StateVec,
    /// State covariance P
    p: StateMat,
    /// Configuration
    cfg: EkfConfig,
    /// Tier arbiter (drives covariance scaler)
    arbiter: TierArbiter,
    /// Timestamp of last update (s)
    last_t: f64,
}

impl CellHawkEkf {
    /// Initialise with a known position and default covariance.
    pub fn new(initial_position: EnuPosition, initial_heading_rad: f64, cfg: EkfConfig) -> Self {
        let mut x = StateVec::zeros();
        x[0] = initial_position.east_m;
        x[1] = initial_position.north_m;
        x[2] = initial_position.up_m;
        x[6] = initial_heading_rad;

        // Initial covariance diagonal (m², m²/s², rad², dB², m²/s⁴)
        let diag = [100.0, 100.0, 25.0, 1.0, 1.0, 0.25, 0.1, 4.0, 0.01];
        let mut p = StateMat::zeros();
        for (i, &v) in diag.iter().enumerate() {
            p[(i, i)] = v;
        }

        let arbiter = TierArbiter::new(cfg.tier.clone());

        Self {
            x,
            p,
            cfg,
            arbiter,
            last_t: 0.0,
        }
    }

    // ── public accessors ──────────────────────────────────────────────────────

    pub fn active_tier(&self) -> NavigationTier {
        self.arbiter.active_tier()
    }
    pub fn spoofing_suspected(&self) -> bool {
        self.arbiter.spoofing_suspected()
    }

    /// Export current state as a `NavigationState` for telemetry.
    pub fn state(&self) -> NavigationState {
        let pos = EnuPosition {
            east_m: self.x[0],
            north_m: self.x[1],
            up_m: self.x[2],
        };
        let vel = Vector3::new(self.x[3], self.x[4], self.x[5]);
        let rms = (self.p[(0, 0)] + self.p[(1, 1)]).sqrt();

        let mut cov = vec![0.0f64; 81];
        for r in 0..N {
            for c in 0..N {
                cov[r * N + c] = self.p[(r, c)];
            }
        }

        NavigationState {
            position: pos,
            velocity_enu: vel,
            heading_rad: self.x[6],
            jnr_db: self.x[7],
            accel_bias_z: self.x[8],
            tier: self.arbiter.active_tier(),
            covariance: cov,
            timestamp_s: self.last_t,
            rms_position_error_m: rms,
        }
    }

    // ── predict step ──────────────────────────────────────────────────────────

    /// IMU-driven predict step.
    ///
    /// State transition (constant-velocity + IMU acceleration):
    /// ```text
    /// east   += v_east  · dt
    /// north  += v_north · dt
    /// up     += v_up    · dt
    /// v_east += a_east  · dt
    /// v_north+= a_north · dt
    /// v_up   += (a_z - bias_z) · dt
    /// heading+= ω_z · dt
    /// jnr    unchanged (random walk in Q)
    /// bias_z unchanged (random walk in Q)
    /// ```
    pub fn predict(&mut self, imu: &ImuMeasurement) {
        let dt = (imu.timestamp_s - self.last_t).max(0.0);
        if dt < 1e-9 {
            return;
        }
        self.last_t = imu.timestamp_s;

        let ax = imu.acceleration_body[0];
        let ay = imu.acceleration_body[1];
        let az = imu.acceleration_body[2] - self.x[8]; // bias-corrected
        let wz = imu.angular_velocity_body[2];

        // State propagation
        self.x[0] += self.x[3] * dt;
        self.x[1] += self.x[4] * dt;
        self.x[2] += self.x[5] * dt;
        self.x[3] += ax * dt;
        self.x[4] += ay * dt;
        self.x[5] += az * dt;
        self.x[6] += wz * dt;
        // x[7] (jnr) and x[8] (bias) propagate as random walks via Q

        // State transition Jacobian F (9×9)
        let mut f = StateMat::identity();
        f[(0, 3)] = dt;
        f[(1, 4)] = dt;
        f[(2, 5)] = dt;
        f[(5, 8)] = -dt; // v_up depends on bias_z

        // Process noise Q (diagonal, continuous → discrete)
        let mut q = StateMat::zeros();
        let qp = self.cfg.q_position * dt;
        let qv = self.cfg.q_velocity * dt;
        q[(0, 0)] = qp;
        q[(1, 1)] = qp;
        q[(2, 2)] = qp;
        q[(3, 3)] = qv;
        q[(4, 4)] = qv;
        q[(5, 5)] = qv;
        q[(6, 6)] = self.cfg.q_heading * dt;
        q[(7, 7)] = self.cfg.q_jnr * dt;
        q[(8, 8)] = self.cfg.q_bias * dt;

        self.p = f * self.p * f.transpose() + q;
    }

    // ── JNR update ────────────────────────────────────────────────────────────

    /// Update JNR state from SDR measurement and drive tier arbitration.
    pub fn update_jnr(&mut self, jnr_db: f64, timestamp_s: f64) {
        self.last_t = timestamp_s;
        // Scalar measurement: z = jnr_db, H = e_7 (unit vector for state[7])
        let innovation = jnr_db - self.x[7];
        let r_jnr = 1.0_f64; // 1 dB² measurement noise
        let p_jnr = self.p[(7, 7)];
        let s = p_jnr + r_jnr;
        let k = p_jnr / s;

        self.x[7] += k * innovation;
        self.p[(7, 7)] *= 1.0 - k;

        self.arbiter.update_jnr(self.x[7]);
    }

    // ── GNSS update (Tier 1) ──────────────────────────────────────────────────

    /// Fuse a GNSS position fix.
    ///
    /// H = [I₃ | 0₃ | 0₃] (3×9, selects position states)
    pub fn update_gnss(&mut self, gnss: &GnssMeasurement) -> Result<(), CellHawkError> {
        let w = self.arbiter.scaler().weights().gnss;
        let r_base = self.cfg.r_gnss_m * self.cfg.r_gnss_m;
        let r_scaled = self.arbiter.scaler().scale_measurement_noise(r_base, w);

        // Measurement: z = [east, north, up]
        type H = SMatrix<f64, GNSS_DIM, N>;
        type R = SMatrix<f64, GNSS_DIM, GNSS_DIM>;

        let z = SVector::<f64, GNSS_DIM>::new(
            gnss.position.east_m,
            gnss.position.north_m,
            gnss.position.up_m,
        );
        let mut h = H::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        let r = R::from_diagonal(&SVector::<f64, GNSS_DIM>::new(r_scaled, r_scaled, r_scaled));

        self.last_t = gnss.timestamp_s;
        self.ekf_update::<GNSS_DIM>(&z, &h, &r)
    }

    // ── Cellular update (Tier 2) ──────────────────────────────────────────────

    /// Fuse a cellular RSSI multilateration position fix.
    ///
    /// H = [I₂ | 0] (2×9, selects east/north only — altitude from baro)
    pub fn update_cellular(
        &mut self,
        position: &EnuPosition,
        timestamp_s: f64,
    ) -> Result<(), CellHawkError> {
        let w = self.arbiter.scaler().weights().cellular;
        let r_base = self.cfg.r_cellular_m * self.cfg.r_cellular_m;
        let r_scaled = self.arbiter.scaler().scale_measurement_noise(r_base, w);

        type H = SMatrix<f64, CELLULAR_DIM, N>;
        type R = SMatrix<f64, CELLULAR_DIM, CELLULAR_DIM>;

        let z = SVector::<f64, CELLULAR_DIM>::new(position.east_m, position.north_m);
        let mut h = H::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;

        let r = R::from_diagonal(&SVector::<f64, CELLULAR_DIM>::new(r_scaled, r_scaled));

        self.last_t = timestamp_s;
        self.ekf_update::<CELLULAR_DIM>(&z, &h, &r)
    }

    // ── SLAM update (Tier 3) ──────────────────────────────────────────────────

    /// Fuse a Visual SLAM velocity measurement.
    ///
    /// H = [0₃ | I₃ | 0₃] (3×9, selects velocity states)
    pub fn update_slam(&mut self, slam: &SlamMeasurement) -> Result<(), CellHawkError> {
        let w = self.arbiter.scaler().weights().slam;
        let r_base = self.cfg.r_slam_m_s * self.cfg.r_slam_m_s;
        let r_scaled = self.arbiter.scaler().scale_measurement_noise(r_base, w);

        type H = SMatrix<f64, SLAM_DIM, N>;
        type R = SMatrix<f64, SLAM_DIM, SLAM_DIM>;

        let z = slam.velocity_enu;
        let mut h = H::zeros();
        h[(0, 3)] = 1.0;
        h[(1, 4)] = 1.0;
        h[(2, 5)] = 1.0;

        let r = R::from_diagonal(&SVector::<f64, SLAM_DIM>::new(r_scaled, r_scaled, r_scaled));

        self.last_t = slam.timestamp_s;
        self.ekf_update::<SLAM_DIM>(&z, &h, &r)
    }

    // ── Spoofing cross-check ──────────────────────────────────────────────────

    /// Cross-check GNSS position against cellular fix for spoofing detection.
    pub fn check_spoofing(
        &mut self,
        gnss_pos: &EnuPosition,
        cellular_pos: &EnuPosition,
    ) -> Result<(), CellHawkError> {
        self.arbiter.check_spoofing(gnss_pos, cellular_pos)
    }

    // ── Generic EKF update ────────────────────────────────────────────────────

    /// Generic EKF measurement update for any measurement dimension M.
    ///
    /// ```text
    /// innovation y  = z − H·x̂
    /// S             = H·P·Hᵀ + R
    /// K             = P·Hᵀ·S⁻¹
    /// x̂             = x̂ + K·y   (after Huber gating)
    /// P             = (I − K·H)·P  (Joseph form for numerical stability)
    /// ```
    fn ekf_update<const M: usize>(
        &mut self,
        z: &SVector<f64, M>,
        h: &SMatrix<f64, M, N>,
        r: &SMatrix<f64, M, M>,
    ) -> Result<(), CellHawkError> {
        let h_x: SVector<f64, M> = h * self.x;
        let innovation: SVector<f64, M> = z - h_x;

        // Innovation covariance S = H·P·Hᵀ + R
        let s: SMatrix<f64, M, M> = h * self.p * h.transpose() + r;

        // Mahalanobis distance for innovation gating
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => {
                return Err(CellHawkError::EkfDivergence {
                    magnitude: f64::INFINITY,
                    threshold: self.cfg.innovation_gate_sigma,
                })
            }
        };

        let mahal_sq = (innovation.transpose() * s_inv * innovation)[(0, 0)];
        let mahal = mahal_sq.sqrt();

        if mahal > self.cfg.innovation_gate_sigma * (M as f64).sqrt() {
            // Huber downweighting instead of hard rejection
            let scale = self.cfg.huber_threshold_sigma * (M as f64).sqrt() / mahal;
            // Apply scaled innovation
            let k: SMatrix<f64, N, M> = self.p * h.transpose() * s_inv;
            self.x += k * (innovation * scale);
            // Joseph form: P = (I-KH)P(I-KH)ᵀ + K·R·Kᵀ
            let i_kh = StateMat::identity() - k * h;
            self.p = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();
            return Ok(());
        }

        // Standard update
        let k: SMatrix<f64, N, M> = self.p * h.transpose() * s_inv;
        self.x += k * innovation;

        // Joseph form for numerical stability
        let i_kh = StateMat::identity() - k * h;
        self.p = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();

        // Enforce symmetry
        self.p = (self.p + self.p.transpose()) * 0.5;

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn origin() -> EnuPosition {
        EnuPosition {
            east_m: 0.0,
            north_m: 0.0,
            up_m: 50.0,
        }
    }

    fn make_ekf() -> CellHawkEkf {
        CellHawkEkf::new(origin(), 0.0, EkfConfig::default())
    }

    fn imu_zero(t: f64) -> ImuMeasurement {
        ImuMeasurement {
            acceleration_body: Vector3::zeros(),
            angular_velocity_body: Vector3::zeros(),
            timestamp_s: t,
        }
    }

    /// After predict with zero IMU, position should not change.
    #[test]
    fn predict_zero_imu_no_position_change() {
        let mut ekf = make_ekf();
        ekf.predict(&imu_zero(0.1));
        assert_abs_diff_eq!(ekf.x[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ekf.x[1], 0.0, epsilon = 1e-12);
    }

    /// Predict with constant acceleration should integrate to correct position.
    #[test]
    fn predict_constant_acceleration_integrates_correctly() {
        let mut ekf = make_ekf();
        // 1 m/s² east for 2 seconds → v=2 m/s, x=2 m
        let imu1 = ImuMeasurement {
            acceleration_body: Vector3::new(1.0, 0.0, 0.0),
            angular_velocity_body: Vector3::zeros(),
            timestamp_s: 1.0,
        };
        let imu2 = ImuMeasurement {
            acceleration_body: Vector3::new(1.0, 0.0, 0.0),
            angular_velocity_body: Vector3::zeros(),
            timestamp_s: 2.0,
        };
        ekf.predict(&imu1);
        ekf.predict(&imu2);
        // After t=1: v=1, x=0+0*1=0 (velocity applied at start of interval)
        // After t=2: v=2, x=0+1*1=1
        assert_abs_diff_eq!(ekf.x[3], 2.0, epsilon = 1e-10); // v_east = 2 m/s
        assert_abs_diff_eq!(ekf.x[0], 1.0, epsilon = 1e-10); // east = 1 m
    }

    /// Covariance must grow during predict (no measurements).
    #[test]
    fn covariance_grows_during_predict() {
        let mut ekf = make_ekf();
        let p0 = ekf.p[(0, 0)];
        ekf.predict(&imu_zero(1.0));
        assert!(
            ekf.p[(0, 0)] > p0,
            "position variance must grow during predict"
        );
    }

    /// GNSS update must reduce position uncertainty.
    #[test]
    fn gnss_update_reduces_uncertainty() {
        let mut ekf = make_ekf();
        let p0 = ekf.p[(0, 0)];
        let gnss = GnssMeasurement {
            position: origin(),
            hdop: 1.0,
            satellites: 8,
            timestamp_s: 0.1,
        };
        ekf.update_gnss(&gnss).unwrap();
        assert!(ekf.p[(0, 0)] < p0, "GNSS update must reduce east variance");
    }

    /// GNSS update must move state toward measurement.
    #[test]
    fn gnss_update_moves_state_toward_measurement() {
        let mut ekf = make_ekf();
        let gnss = GnssMeasurement {
            position: EnuPosition {
                east_m: 100.0,
                north_m: 50.0,
                up_m: 50.0,
            },
            hdop: 1.0,
            satellites: 8,
            timestamp_s: 0.1,
        };
        ekf.update_gnss(&gnss).unwrap();
        // State should move toward (100, 50) from (0, 0)
        assert!(ekf.x[0] > 0.0, "east must increase toward measurement");
        assert!(ekf.x[1] > 0.0, "north must increase toward measurement");
    }

    /// Cellular update must not affect up-position (altitude from baro only).
    #[test]
    fn cellular_update_does_not_change_altitude() {
        let mut ekf = make_ekf();
        let up_before = ekf.x[2];
        let pos = EnuPosition {
            east_m: 50.0,
            north_m: 30.0,
            up_m: 999.0,
        };
        ekf.update_cellular(&pos, 0.1).unwrap();
        assert_abs_diff_eq!(ekf.x[2], up_before, epsilon = 1e-10);
    }

    /// SLAM update must affect velocity states, not position.
    #[test]
    fn slam_update_affects_velocity_not_position() {
        let mut ekf = make_ekf();
        let east_before = ekf.x[0];
        let slam = SlamMeasurement {
            velocity_enu: Vector3::new(5.0, 3.0, 0.5),
            heading_rad: 0.0,
            velocity_covariance: [0.25; 9],
            loop_closure: false,
            timestamp_s: 0.1,
        };
        ekf.update_slam(&slam).unwrap();
        assert_abs_diff_eq!(ekf.x[0], east_before, epsilon = 1e-10);
        assert!(
            ekf.x[3] > 0.0,
            "v_east must increase toward SLAM measurement"
        );
    }

    /// Covariance must remain symmetric after updates.
    #[test]
    fn covariance_remains_symmetric() {
        let mut ekf = make_ekf();
        ekf.predict(&imu_zero(0.1));
        let gnss = GnssMeasurement {
            position: EnuPosition {
                east_m: 10.0,
                north_m: 5.0,
                up_m: 50.0,
            },
            hdop: 1.2,
            satellites: 7,
            timestamp_s: 0.1,
        };
        ekf.update_gnss(&gnss).unwrap();
        for i in 0..N {
            for j in 0..N {
                assert_abs_diff_eq!(ekf.p[(i, j)], ekf.p[(j, i)], epsilon = 1e-10);
            }
        }
    }

    /// Covariance diagonal must remain positive after updates.
    #[test]
    fn covariance_diagonal_positive_definite() {
        let mut ekf = make_ekf();
        for step in 1..=10 {
            ekf.predict(&imu_zero(step as f64 * 0.1));
            let gnss = GnssMeasurement {
                position: EnuPosition {
                    east_m: step as f64,
                    north_m: step as f64 * 0.5,
                    up_m: 50.0,
                },
                hdop: 1.0,
                satellites: 8,
                timestamp_s: step as f64 * 0.1,
            };
            ekf.update_gnss(&gnss).unwrap();
        }
        for i in 0..N {
            assert!(
                ekf.p[(i, i)] > 0.0,
                "P[{i},{i}] must be positive, got {}",
                ekf.p[(i, i)]
            );
        }
    }

    /// JNR update must drive tier transition.
    #[test]
    fn jnr_update_drives_tier_transition() {
        let mut ekf = make_ekf();
        assert_eq!(ekf.active_tier(), NavigationTier::GnssActive);
        // Push JNR above Tier1→2 threshold + hysteresis (6 + 1.5 = 7.5 dB)
        for i in 1..=15 {
            ekf.update_jnr(10.0, i as f64 * 0.1);
        }
        assert_eq!(ekf.active_tier(), NavigationTier::CellularRssi);
    }

    /// Full predict→update cycle must not diverge over 100 steps.
    #[test]
    fn no_divergence_over_100_steps() {
        let mut ekf = make_ekf();
        for step in 1..=100 {
            let t = step as f64 * 0.1;
            ekf.predict(&imu_zero(t));
            ekf.update_jnr(3.0, t);
            let gnss = GnssMeasurement {
                position: EnuPosition {
                    east_m: (t * 5.0).sin() * 50.0,
                    north_m: t * 2.0,
                    up_m: 50.0,
                },
                hdop: 1.0,
                satellites: 8,
                timestamp_s: t,
            };
            ekf.update_gnss(&gnss).unwrap();
        }
        let state = ekf.state();
        assert!(state.rms_position_error_m.is_finite());
        assert!(state.rms_position_error_m < 50.0);
    }

    /// Paper §5.3: Tier 1 RMS error ≈ 4.5 m.
    /// Verify EKF converges to within 2× of this under repeated GNSS updates.
    #[test]
    fn tier1_rms_converges_to_paper_claim() {
        let mut ekf = make_ekf();
        let true_pos = EnuPosition {
            east_m: 100.0,
            north_m: 200.0,
            up_m: 50.0,
        };
        for step in 1..=50 {
            let t = step as f64 * 0.1;
            ekf.predict(&imu_zero(t));
            ekf.update_jnr(2.0, t);
            let gnss = GnssMeasurement {
                position: true_pos,
                hdop: 1.0,
                satellites: 10,
                timestamp_s: t,
            };
            ekf.update_gnss(&gnss).unwrap();
        }
        // RMS from covariance should be ≤ 2× paper claim (9 m)
        let rms = ekf.state().rms_position_error_m;
        assert!(rms < 9.0, "Tier 1 RMS={rms:.2} m, expected < 9.0 m");
    }
}
