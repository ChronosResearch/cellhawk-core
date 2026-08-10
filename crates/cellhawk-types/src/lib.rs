//! # cellhawk-types
//!
//! Canonical domain types shared across the entire CellHawk navigation stack.
//!
//! All physical quantities use SI units unless explicitly noted:
//! - distances: metres
//! - velocities: m/s
//! - angles: radians
//! - power ratios: dB (linear where noted)
//! - time: seconds

use nalgebra::{Matrix2, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Navigation Tier
// ─────────────────────────────────────────────────────────────────────────────

/// The three-tier navigation hierarchy defined in §3.4 of the paper.
///
/// Tier transitions are governed by real-time JNR measurements at the SDR
/// front-end. The EKF innovation covariance matrix is dynamically scaled
/// based on the active tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum NavigationTier {
    /// **Tier 1 — GNSS Active** (JNR < 6 dB)
    ///
    /// GNSS pseudo-ranges receive full trust weighting. Cellular RSSI and
    /// Visual SLAM are fused as secondary constraints.
    /// Simulated RMS error: ~4.5 m.
    GnssActive = 1,

    /// **Tier 2 — Cellular RSSI Primary** (6 dB ≤ JNR < 19 dB)
    ///
    /// GNSS is zero-weighted. Cellular RSSI multilateration is primary.
    /// Visual SLAM fused as complementary velocity constraint.
    /// Simulated RMS error: ~42 m (favourable conditions).
    CellularRssi = 2,

    /// **Tier 3 — Visual SLAM Only** (JNR ≥ 19 dB)
    ///
    /// Both GNSS and cellular metrics are zero-weighted. EKF relies solely
    /// on Visual SLAM odometry with IMU integration.
    /// Simulated RMS error: ~12 m (terrain-rich environments).
    VisualSlam = 3,
}

impl NavigationTier {
    /// JNR threshold (dB) above which this tier becomes active.
    ///
    /// - Tier 1 → 2 handover: JNR ≥ 6 dB  (GPS degraded)
    /// - Tier 2 → 3 handover: JNR ≥ 19 dB (cellular overwhelmed, ~79× noise)
    pub const TIER1_JNR_THRESHOLD_DB: f64 = 6.0;
    pub const TIER2_JNR_THRESHOLD_DB: f64 = 19.0;

    /// Determine the appropriate tier from a JNR measurement.
    #[inline]
    pub fn from_jnr_db(jnr_db: f64) -> Self {
        if jnr_db < Self::TIER1_JNR_THRESHOLD_DB {
            Self::GnssActive
        } else if jnr_db < Self::TIER2_JNR_THRESHOLD_DB {
            Self::CellularRssi
        } else {
            Self::VisualSlam
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geodetic / Cartesian Position
// ─────────────────────────────────────────────────────────────────────────────

/// WGS-84 geodetic coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeodeticPosition {
    /// Latitude in decimal degrees (−90 … +90).
    pub latitude_deg: f64,
    /// Longitude in decimal degrees (−180 … +180).
    pub longitude_deg: f64,
    /// Altitude above WGS-84 ellipsoid in metres.
    pub altitude_m: f64,
}

/// Local East-North-Up (ENU) Cartesian position relative to a reference origin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnuPosition {
    pub east_m: f64,
    pub north_m: f64,
    pub up_m: f64,
}

impl EnuPosition {
    /// Euclidean distance to another ENU position (metres).
    #[inline]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let de = self.east_m - other.east_m;
        let dn = self.north_m - other.north_m;
        let du = self.up_m - other.up_m;
        (de * de + dn * dn + du * du).sqrt()
    }

    /// 2-D horizontal distance (ignores altitude).
    #[inline]
    pub fn horizontal_distance_to(&self, other: &Self) -> f64 {
        let de = self.east_m - other.east_m;
        let dn = self.north_m - other.north_m;
        (de * de + dn * dn).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cellular Tower Measurement
// ─────────────────────────────────────────────────────────────────────────────

/// A single RSSI measurement from one 4G/LTE tower, as sampled by the SDR
/// front-end at 10 Hz (§2.1, §3.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TowerMeasurement {
    /// Unique tower identifier (e.g. Cell-ID from OpenCellID database).
    pub tower_id: u64,
    /// Known ENU position of the tower (from tower database).
    pub tower_position: EnuPosition,
    /// Measured RSSI in dBm.
    pub rssi_dbm: f64,
    /// Transmit power of the tower in dBm (from database or estimated).
    pub tx_power_dbm: f64,
    /// Rician K-factor estimate for this link (dB). Used for fading model.
    /// Range: 3–12 dB in simulation; < 0 dB for dense NLoS.
    pub rician_k_db: f64,
    /// Timestamp of this measurement (seconds since mission epoch).
    pub timestamp_s: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// GNSS Measurement
// ─────────────────────────────────────────────────────────────────────────────

/// A GNSS position fix, including quality indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnssMeasurement {
    /// Estimated ENU position.
    pub position: EnuPosition,
    /// Horizontal dilution of precision (dimensionless).
    pub hdop: f64,
    /// Number of satellites used in the fix.
    pub satellites: u8,
    /// Timestamp (seconds since mission epoch).
    pub timestamp_s: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Visual SLAM Measurement
// ─────────────────────────────────────────────────────────────────────────────

/// Velocity and heading estimate from the Visual SLAM / optical odometry
/// pipeline (§3.4, Tier 3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamMeasurement {
    /// Velocity vector in ENU frame (m/s).
    pub velocity_enu: Vector3<f64>,
    /// Heading estimate (radians, 0 = North, clockwise positive).
    pub heading_rad: f64,
    /// Covariance of the velocity estimate (3×3, diagonal assumed).
    pub velocity_covariance: [f64; 9],
    /// Whether a loop closure was detected this frame (improves accuracy).
    pub loop_closure: bool,
    /// Timestamp (seconds since mission epoch).
    pub timestamp_s: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// IMU Measurement
// ─────────────────────────────────────────────────────────────────────────────

/// Raw IMU data used for EKF state propagation between measurement updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuMeasurement {
    /// Linear acceleration in body frame (m/s²).
    pub acceleration_body: Vector3<f64>,
    /// Angular velocity in body frame (rad/s).
    pub angular_velocity_body: Vector3<f64>,
    /// Timestamp (seconds since mission epoch).
    pub timestamp_s: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Full Navigation State (EKF state vector)
// ─────────────────────────────────────────────────────────────────────────────

/// The 9-dimensional EKF state vector.
///
/// State: [east, north, up, v_east, v_north, v_up, heading, jnr_db, accel_bias_z]
///
/// This extends the paper's description to include velocity and bias states,
/// enabling proper IMU pre-integration and drift compensation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationState {
    /// ENU position (m).
    pub position: EnuPosition,
    /// ENU velocity (m/s).
    pub velocity_enu: Vector3<f64>,
    /// Heading (radians, 0 = North).
    pub heading_rad: f64,
    /// Current JNR estimate (dB) from SDR front-end.
    pub jnr_db: f64,
    /// Z-axis accelerometer bias estimate (m/s²).
    pub accel_bias_z: f64,
    /// Active navigation tier.
    pub tier: NavigationTier,
    /// 9×9 state covariance matrix (row-major, 81 elements).
    pub covariance: Vec<f64>,
    /// Timestamp of this state estimate (seconds since mission epoch).
    pub timestamp_s: f64,
    /// RMS position error estimate from the EKF (m).
    pub rms_position_error_m: f64,
}

impl NavigationState {
    /// Extract the 2-D horizontal position as a `Vector2`.
    #[inline]
    pub fn position_2d(&self) -> Vector2<f64> {
        Vector2::new(self.position.east_m, self.position.north_m)
    }

    /// Extract the 2×2 horizontal position covariance sub-matrix.
    pub fn horizontal_covariance(&self) -> Matrix2<f64> {
        // Indices 0,1 are east, north in the 9-state vector
        assert!(
            self.covariance.len() == 81,
            "covariance must have 81 elements"
        );
        Matrix2::new(
            self.covariance[0],
            self.covariance[1],
            self.covariance[9],
            self.covariance[10],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Danger Grid Entry (§6.1)
// ─────────────────────────────────────────────────────────────────────────────

/// A hazard broadcast by any drone in the swarm to the shared Danger Grid.
///
/// Analogous to stigmergic pheromone trails (§6.1). Stored in Redis GEO index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DangerGridEntry {
    /// Originating drone node ID.
    pub drone_id: u32,
    /// ENU position of the hazard.
    pub position: EnuPosition,
    /// Threat severity [0.0, 1.0].
    pub severity: f64,
    /// Threat category.
    pub threat_type: ThreatType,
    /// Time-to-live in seconds (entries decay after this period).
    pub ttl_s: f64,
    /// Timestamp of the event (seconds since mission epoch).
    pub timestamp_s: f64,
}

/// Classification of threat types recorded in the Danger Grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatType {
    /// RF jamming dome detected.
    RfJamming,
    /// GPS spoofing signal detected.
    GpsSpoofing,
    /// Adversarial hunter drone proximity.
    HunterDrone,
    /// Physical obstacle (near-miss).
    PhysicalObstacle,
    /// Communication link degradation.
    CommsDegradation,
}

// ─────────────────────────────────────────────────────────────────────────────
// CORTEX DQN State Vector (§4.1)
// ─────────────────────────────────────────────────────────────────────────────

/// The 19-dimensional continuous state vector fed to the CORTEX DQN at 10 Hz.
///
/// Components (§4.1):
/// - 8 LiDAR sector distances (45° sectors, 0 = North, clockwise)
/// - barometric altitude
/// - destination bearing
/// - velocity (east, north, up)
/// - battery voltage
/// - real-time JNR from SDR front-end
/// - tier index (1, 2, or 3) — added improvement over paper
/// - heading error to waypoint — added improvement over paper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortexStateVector {
    /// LiDAR proximity distances for 8 sectors (m). Index 0 = North, clockwise.
    pub lidar_sectors_m: [f64; 8],
    /// Barometric altitude (m AGL).
    pub baro_altitude_m: f64,
    /// Bearing to destination waypoint (radians).
    pub destination_bearing_rad: f64,
    /// ENU velocity components (m/s).
    pub velocity_enu: [f64; 3],
    /// Battery voltage (V).
    pub battery_voltage_v: f64,
    /// Real-time JNR from SDR front-end (dB).
    pub jnr_db: f64,
    /// Active navigation tier (1, 2, or 3).
    pub tier: u8,
    /// Heading error to current waypoint (radians, signed).
    pub heading_error_rad: f64,
}

impl CortexStateVector {
    /// Dimension of the state vector (must equal 19 per §4.1).
    pub const DIM: usize = 19;

    /// Flatten to a fixed-size array for neural network input.
    pub fn to_array(&self) -> [f64; Self::DIM] {
        [
            self.lidar_sectors_m[0],
            self.lidar_sectors_m[1],
            self.lidar_sectors_m[2],
            self.lidar_sectors_m[3],
            self.lidar_sectors_m[4],
            self.lidar_sectors_m[5],
            self.lidar_sectors_m[6],
            self.lidar_sectors_m[7],
            self.baro_altitude_m,
            self.destination_bearing_rad,
            self.velocity_enu[0],
            self.velocity_enu[1],
            self.velocity_enu[2],
            self.battery_voltage_v,
            self.jnr_db,
            self.tier as f64,
            self.heading_error_rad,
            // Slots 17-18: reserved for future sensor channels
            0.0,
            0.0,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level error type for the CellHawk navigation stack.
#[derive(Debug, Error)]
pub enum CellHawkError {
    #[error("Insufficient tower measurements: need ≥{required}, got {got}")]
    InsufficientTowers { required: usize, got: usize },

    #[error(
        "EKF divergence detected: innovation magnitude {magnitude:.2} exceeds {threshold:.2}σ"
    )]
    EkfDivergence { magnitude: f64, threshold: f64 },

    #[error("RSSI measurement out of valid range: {rssi_dbm:.1} dBm")]
    InvalidRssi { rssi_dbm: f64 },

    #[error("Multilateration solver failed to converge after {iterations} iterations")]
    SolverNonConvergence { iterations: usize },

    #[error("GPS spoofing suspected: RSSI cross-check residual {residual_m:.1} m exceeds threshold {threshold_m:.1} m")]
    GpsSpoofingSuspected { residual_m: f64, threshold_m: f64 },

    #[error("Navigation tier unavailable: {tier:?}")]
    TierUnavailable { tier: NavigationTier },

    #[error("Tower database error: {0}")]
    TowerDatabase(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_from_jnr_boundaries() {
        // Exactly at thresholds
        assert_eq!(NavigationTier::from_jnr_db(0.0), NavigationTier::GnssActive);
        assert_eq!(
            NavigationTier::from_jnr_db(5.99),
            NavigationTier::GnssActive
        );
        assert_eq!(
            NavigationTier::from_jnr_db(6.0),
            NavigationTier::CellularRssi
        );
        assert_eq!(
            NavigationTier::from_jnr_db(18.99),
            NavigationTier::CellularRssi
        );
        assert_eq!(
            NavigationTier::from_jnr_db(19.0),
            NavigationTier::VisualSlam
        );
        assert_eq!(
            NavigationTier::from_jnr_db(30.0),
            NavigationTier::VisualSlam
        );
    }

    #[test]
    fn tier_ordering_is_severity_ascending() {
        assert!(NavigationTier::GnssActive < NavigationTier::CellularRssi);
        assert!(NavigationTier::CellularRssi < NavigationTier::VisualSlam);
    }

    #[test]
    fn enu_distance_is_symmetric() {
        let a = EnuPosition {
            east_m: 0.0,
            north_m: 0.0,
            up_m: 0.0,
        };
        let b = EnuPosition {
            east_m: 3.0,
            north_m: 4.0,
            up_m: 0.0,
        };
        let dist = a.distance_to(&b);
        assert!((dist - 5.0).abs() < 1e-10, "3-4-5 triangle: got {dist}");
        assert!((a.distance_to(&b) - b.distance_to(&a)).abs() < 1e-12);
    }

    #[test]
    fn cortex_state_vector_dim() {
        let sv = CortexStateVector {
            lidar_sectors_m: [10.0; 8],
            baro_altitude_m: 50.0,
            destination_bearing_rad: 1.0,
            velocity_enu: [5.0, 0.0, 0.0],
            battery_voltage_v: 14.8,
            jnr_db: 3.0,
            tier: 1,
            heading_error_rad: 0.1,
        };
        assert_eq!(sv.to_array().len(), CortexStateVector::DIM);
    }

    #[test]
    fn navigation_state_horizontal_covariance_symmetry() {
        let mut cov = vec![0.0f64; 81];
        // Set P[0,0]=4, P[0,1]=1, P[1,0]=1, P[1,1]=9
        cov[0] = 4.0;
        cov[1] = 1.0;
        cov[9] = 1.0;
        cov[10] = 9.0;
        let state = NavigationState {
            position: EnuPosition {
                east_m: 0.0,
                north_m: 0.0,
                up_m: 0.0,
            },
            velocity_enu: Vector3::zeros(),
            heading_rad: 0.0,
            jnr_db: 0.0,
            accel_bias_z: 0.0,
            tier: NavigationTier::GnssActive,
            covariance: cov,
            timestamp_s: 0.0,
            rms_position_error_m: 0.0,
        };
        let h = state.horizontal_covariance();
        assert_eq!(h[(0, 0)], 4.0);
        assert_eq!(h[(1, 1)], 9.0);
        assert_eq!(h[(0, 1)], h[(1, 0)]);
    }
}
