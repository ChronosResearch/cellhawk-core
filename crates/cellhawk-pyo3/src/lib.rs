//! PyO3 Python bindings for the CellHawk EKF.
//!
//! Exposes [`CellHawkEkf`] as a Python class `cellhawk_pyo3.CellHawkEkf`.
//!
//! ## Usage (Python)
//!
//! ```python
//! import cellhawk_pyo3 as ch
//!
//! ekf = ch.CellHawkEkf(east_m=0.0, north_m=0.0, up_m=50.0, heading_rad=0.0)
//!
//! # IMU predict step
//! ekf.predict(ax=0.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=0.1)
//!
//! # JNR update (from SDR front-end)
//! ekf.update_jnr(jnr_db=3.0, timestamp_s=0.1)
//!
//! # GNSS update
//! ekf.update_gnss(east_m=10.0, north_m=5.0, up_m=50.0,
//!                 hdop=1.0, satellites=8, timestamp_s=0.1)
//!
//! state = ekf.state()   # returns dict
//! print(state["tier"], state["rms_position_error_m"])
//! ```
//!
//! ## Error handling
//!
//! All fallible methods raise `RuntimeError` on EKF divergence or spoofing
//! detection.  The caller should catch these and take appropriate action
//! (e.g. force tier downgrade, alert GCS).

use cellhawk_ekf::filter::{CellHawkEkf, EkfConfig};
use cellhawk_types::{
    EnuPosition, GnssMeasurement, ImuMeasurement, NavigationTier, SlamMeasurement,
};
use nalgebra::Vector3;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ─────────────────────────────────────────────────────────────────────────────
// Python module entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Python module `cellhawk_pyo3`.
#[pymodule]
fn cellhawk_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEkf>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// PyEkf — Python-callable wrapper around CellHawkEkf
// ─────────────────────────────────────────────────────────────────────────────

/// Python-callable 9-state Extended Kalman Filter.
///
/// All physical quantities use SI units (metres, m/s, radians, dB, seconds).
#[pyclass(name = "CellHawkEkf")]
pub struct PyEkf {
    inner: CellHawkEkf,
}

#[pymethods]
impl PyEkf {
    /// Create a new EKF instance at the given initial position.
    ///
    /// Args:
    ///     east_m:       Initial east position (m ENU).
    ///     north_m:      Initial north position (m ENU).
    ///     up_m:         Initial altitude (m ENU).
    ///     heading_rad:  Initial heading (radians, 0 = North).
    #[new]
    #[pyo3(signature = (east_m=0.0, north_m=0.0, up_m=50.0, heading_rad=0.0))]
    fn new(east_m: f64, north_m: f64, up_m: f64, heading_rad: f64) -> Self {
        let pos = EnuPosition {
            east_m,
            north_m,
            up_m,
        };
        Self {
            inner: CellHawkEkf::new(pos, heading_rad, EkfConfig::default()),
        }
    }

    /// IMU-driven predict step.
    ///
    /// Args:
    ///     ax, ay, az:   Body-frame acceleration (m/s²).
    ///     wz:           Body-frame yaw rate (rad/s).
    ///     timestamp_s:  Measurement timestamp (seconds since mission epoch).
    #[pyo3(signature = (ax, ay, az, wz, timestamp_s))]
    fn predict(&mut self, ax: f64, ay: f64, az: f64, wz: f64, timestamp_s: f64) {
        let imu = ImuMeasurement {
            acceleration_body: Vector3::new(ax, ay, az),
            angular_velocity_body: Vector3::new(0.0, 0.0, wz),
            timestamp_s,
        };
        self.inner.predict(&imu);
    }

    /// Update JNR state from SDR measurement and drive tier arbitration.
    ///
    /// Args:
    ///     jnr_db:      Jammer-to-Noise Ratio (dB) from SDR front-end.
    ///     timestamp_s: Measurement timestamp.
    #[pyo3(signature = (jnr_db, timestamp_s))]
    fn update_jnr(&mut self, jnr_db: f64, timestamp_s: f64) {
        self.inner.update_jnr(jnr_db, timestamp_s);
    }

    /// Fuse a GNSS position fix (Tier 1).
    ///
    /// Raises:
    ///     RuntimeError: On EKF divergence.
    #[pyo3(signature = (east_m, north_m, up_m, hdop, satellites, timestamp_s))]
    fn update_gnss(
        &mut self,
        east_m: f64,
        north_m: f64,
        up_m: f64,
        hdop: f64,
        satellites: u8,
        timestamp_s: f64,
    ) -> PyResult<()> {
        let gnss = GnssMeasurement {
            position: EnuPosition {
                east_m,
                north_m,
                up_m,
            },
            hdop,
            satellites,
            timestamp_s,
        };
        self.inner.update_gnss(&gnss).map_err(rust_err_to_py)
    }

    /// Fuse a cellular RSSI multilateration fix (Tier 2).
    ///
    /// Only east/north are fused; altitude comes from barometer.
    ///
    /// Raises:
    ///     RuntimeError: On EKF divergence.
    #[pyo3(signature = (east_m, north_m, timestamp_s))]
    fn update_cellular(&mut self, east_m: f64, north_m: f64, timestamp_s: f64) -> PyResult<()> {
        let pos = EnuPosition {
            east_m,
            north_m,
            up_m: 0.0,
        };
        self.inner
            .update_cellular(&pos, timestamp_s)
            .map_err(rust_err_to_py)
    }

    /// Fuse a Visual SLAM velocity measurement (Tier 3).
    ///
    /// Raises:
    ///     RuntimeError: On EKF divergence.
    #[pyo3(signature = (v_east, v_north, v_up, heading_rad, timestamp_s))]
    fn update_slam(
        &mut self,
        v_east: f64,
        v_north: f64,
        v_up: f64,
        heading_rad: f64,
        timestamp_s: f64,
    ) -> PyResult<()> {
        let slam = SlamMeasurement {
            velocity_enu: Vector3::new(v_east, v_north, v_up),
            heading_rad,
            velocity_covariance: [0.25; 9],
            loop_closure: false,
            timestamp_s,
        };
        self.inner.update_slam(&slam).map_err(rust_err_to_py)
    }

    /// Cross-check GNSS vs cellular position for GPS spoofing detection.
    ///
    /// Raises:
    ///     RuntimeError: When spoofing is confirmed (3 consecutive detections).
    #[pyo3(signature = (gnss_east, gnss_north, cell_east, cell_north))]
    fn check_spoofing(
        &mut self,
        gnss_east: f64,
        gnss_north: f64,
        cell_east: f64,
        cell_north: f64,
    ) -> PyResult<()> {
        let gnss_pos = EnuPosition {
            east_m: gnss_east,
            north_m: gnss_north,
            up_m: 0.0,
        };
        let cell_pos = EnuPosition {
            east_m: cell_east,
            north_m: cell_north,
            up_m: 0.0,
        };
        self.inner
            .check_spoofing(&gnss_pos, &cell_pos)
            .map_err(rust_err_to_py)
    }

    /// Export the current navigation state as a Python dict.
    ///
    /// Returns:
    ///     dict with keys: east_m, north_m, up_m, v_east_m_s, v_north_m_s,
    ///     v_up_m_s, heading_rad, jnr_db, accel_bias_z, tier (int 1-3),
    ///     rms_position_error_m, timestamp_s, spoofing_suspected (bool).
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.state();
        let d = PyDict::new_bound(py);
        d.set_item("east_m", s.position.east_m)?;
        d.set_item("north_m", s.position.north_m)?;
        d.set_item("up_m", s.position.up_m)?;
        d.set_item("v_east_m_s", s.velocity_enu[0])?;
        d.set_item("v_north_m_s", s.velocity_enu[1])?;
        d.set_item("v_up_m_s", s.velocity_enu[2])?;
        d.set_item("heading_rad", s.heading_rad)?;
        d.set_item("jnr_db", s.jnr_db)?;
        d.set_item("accel_bias_z", s.accel_bias_z)?;
        d.set_item("tier", tier_to_int(s.tier))?;
        d.set_item("rms_position_error_m", s.rms_position_error_m)?;
        d.set_item("timestamp_s", s.timestamp_s)?;
        d.set_item("spoofing_suspected", self.inner.spoofing_suspected())?;
        Ok(d)
    }

    /// Active navigation tier (1 = GNSS, 2 = Cellular, 3 = SLAM).
    fn active_tier(&self) -> u8 {
        tier_to_int(self.inner.active_tier())
    }

    /// Whether GPS spoofing is currently suspected.
    fn spoofing_suspected(&self) -> bool {
        self.inner.spoofing_suspected()
    }

    fn __repr__(&self) -> String {
        let s = self.inner.state();
        format!(
            "CellHawkEkf(tier={}, pos=({:.1},{:.1},{:.1}), rms={:.2}m)",
            tier_to_int(s.tier),
            s.position.east_m,
            s.position.north_m,
            s.position.up_m,
            s.rms_position_error_m,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn tier_to_int(tier: NavigationTier) -> u8 {
    match tier {
        NavigationTier::GnssActive => 1,
        NavigationTier::CellularRssi => 2,
        NavigationTier::VisualSlam => 3,
    }
}

fn rust_err_to_py(e: cellhawk_types::CellHawkError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Rust-side unit tests
//
// These tests call CellHawkEkf directly (no PyO3 runtime needed) so they
// compile and link cleanly under `cargo test`.  The Python-level behaviour
// is covered by tests/test_pyo3_bridge.py (requires `maturin develop`).
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use cellhawk_ekf::filter::{CellHawkEkf, EkfConfig};
    use cellhawk_types::{EnuPosition, ImuMeasurement, NavigationTier};
    use nalgebra::Vector3;

    fn make_ekf() -> CellHawkEkf {
        CellHawkEkf::new(
            EnuPosition {
                east_m: 0.0,
                north_m: 0.0,
                up_m: 50.0,
            },
            0.0,
            EkfConfig::default(),
        )
    }

    fn imu_zero(t: f64) -> ImuMeasurement {
        ImuMeasurement {
            acceleration_body: Vector3::zeros(),
            angular_velocity_body: Vector3::zeros(),
            timestamp_s: t,
        }
    }

    #[test]
    fn wrapper_tier_mapping_gnss_active() {
        let ekf = make_ekf();
        assert_eq!(ekf.active_tier(), NavigationTier::GnssActive);
    }

    #[test]
    fn wrapper_predict_then_jnr_drives_tier2() {
        let mut ekf = make_ekf();
        for i in 1..=20 {
            ekf.predict(&imu_zero(i as f64 * 0.1));
            ekf.update_jnr(10.0, i as f64 * 0.1);
        }
        assert_eq!(ekf.active_tier(), NavigationTier::CellularRssi);
    }

    #[test]
    fn wrapper_spoofing_not_suspected_initially() {
        let ekf = make_ekf();
        assert!(!ekf.spoofing_suspected());
    }

    #[test]
    fn wrapper_spoofing_detected_after_3_checks() {
        let mut ekf = make_ekf();
        let gnss = EnuPosition {
            east_m: 0.0,
            north_m: 0.0,
            up_m: 0.0,
        };
        let cell = EnuPosition {
            east_m: 200.0,
            north_m: 0.0,
            up_m: 0.0,
        };
        for _ in 0..3 {
            let _ = ekf.check_spoofing(&gnss, &cell);
        }
        assert!(ekf.spoofing_suspected());
    }

    #[test]
    fn tier_to_int_mapping_is_correct() {
        assert_eq!(super::tier_to_int(NavigationTier::GnssActive), 1);
        assert_eq!(super::tier_to_int(NavigationTier::CellularRssi), 2);
        assert_eq!(super::tier_to_int(NavigationTier::VisualSlam), 3);
    }
}
