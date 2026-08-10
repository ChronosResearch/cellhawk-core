"""PyO3 bridge integration tests.

These tests exercise the Python-callable CellHawkEkf wrapper built by maturin.
They are skipped automatically if the extension has not been compiled yet.

To build locally:
    maturin develop --manifest-path crates/cellhawk-pyo3/Cargo.toml

In CI the Python job runs `maturin develop` before pytest, so these tests
always execute in the pipeline.
"""
from __future__ import annotations

import math
import pytest

# ── Conditional import — skip gracefully if not built ────────────────────────
try:
    import cellhawk_pyo3 as ch
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="cellhawk_pyo3 not built — run: maturin develop --manifest-path crates/cellhawk-pyo3/Cargo.toml",
)


# ─────────────────────────────────────────────────────────────────────────────
# Construction and repr
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_default_construction(self) -> None:
        ekf = ch.CellHawkEkf()
        assert ekf is not None

    def test_custom_initial_position(self) -> None:
        ekf = ch.CellHawkEkf(east_m=100.0, north_m=200.0, up_m=75.0, heading_rad=1.57)
        state = ekf.state()
        assert abs(state["east_m"]  - 100.0) < 1e-9
        assert abs(state["north_m"] - 200.0) < 1e-9
        assert abs(state["up_m"]    - 75.0)  < 1e-9

    def test_repr_contains_tier(self) -> None:
        ekf = ch.CellHawkEkf()
        r = repr(ekf)
        assert "tier=1" in r

    def test_version_attribute_present(self) -> None:
        assert hasattr(ch, "__version__")
        assert isinstance(ch.__version__, str)


# ─────────────────────────────────────────────────────────────────────────────
# Predict step
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_zero_imu_no_position_change(self) -> None:
        ekf = ch.CellHawkEkf(east_m=0.0, north_m=0.0, up_m=50.0)
        ekf.predict(ax=0.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=0.1)
        state = ekf.state()
        assert abs(state["east_m"])  < 1e-9
        assert abs(state["north_m"]) < 1e-9

    def test_predict_constant_acceleration_integrates(self) -> None:
        ekf = ch.CellHawkEkf()
        ekf.predict(ax=1.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=1.0)
        ekf.predict(ax=1.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=2.0)
        state = ekf.state()
        # After 2 steps: v_east = 2 m/s, east = 1 m (velocity applied at start)
        assert abs(state["v_east_m_s"] - 2.0) < 1e-9
        assert abs(state["east_m"]     - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# JNR update and tier transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestJnrAndTier:
    def test_starts_in_tier1(self) -> None:
        ekf = ch.CellHawkEkf()
        assert ekf.active_tier() == 1

    def test_high_jnr_drives_tier2(self) -> None:
        ekf = ch.CellHawkEkf()
        for i in range(1, 21):
            ekf.update_jnr(jnr_db=10.0, timestamp_s=i * 0.1)
        assert ekf.active_tier() == 2

    def test_very_high_jnr_drives_tier3(self) -> None:
        ekf = ch.CellHawkEkf()
        for i in range(1, 31):
            ekf.update_jnr(jnr_db=25.0, timestamp_s=i * 0.1)
        assert ekf.active_tier() == 3

    def test_tier_reflected_in_state_dict(self) -> None:
        ekf = ch.CellHawkEkf()
        for i in range(1, 21):
            ekf.update_jnr(jnr_db=10.0, timestamp_s=i * 0.1)
        state = ekf.state()
        assert state["tier"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# GNSS update
# ─────────────────────────────────────────────────────────────────────────────

class TestGnssUpdate:
    def test_gnss_moves_state_toward_measurement(self) -> None:
        ekf = ch.CellHawkEkf(east_m=0.0, north_m=0.0, up_m=50.0)
        ekf.update_gnss(east_m=100.0, north_m=50.0, up_m=50.0,
                        hdop=1.0, satellites=8, timestamp_s=0.1)
        state = ekf.state()
        assert state["east_m"]  > 0.0
        assert state["north_m"] > 0.0

    def test_gnss_reduces_rms_error(self) -> None:
        ekf = ch.CellHawkEkf()
        rms_before = ekf.state()["rms_position_error_m"]
        for i in range(1, 20):
            ekf.predict(ax=0.0, ay=0.0, az=0.0, wz=0.0, timestamp_s=i * 0.1)
            ekf.update_gnss(east_m=0.0, north_m=0.0, up_m=50.0,
                            hdop=1.0, satellites=10, timestamp_s=i * 0.1)
        rms_after = ekf.state()["rms_position_error_m"]
        assert rms_after < rms_before


# ─────────────────────────────────────────────────────────────────────────────
# Spoofing detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSpoofingDetection:
    def test_no_spoofing_initially(self) -> None:
        ekf = ch.CellHawkEkf()
        assert not ekf.spoofing_suspected()

    def test_spoofing_detected_after_3_bad_checks(self) -> None:
        ekf = ch.CellHawkEkf()
        # 200 m discrepancy > 150 m threshold
        for _ in range(3):
            try:
                ekf.check_spoofing(gnss_east=0.0, gnss_north=0.0,
                                   cell_east=200.0, cell_north=0.0)
            except RuntimeError:
                pass  # expected on 3rd check
        assert ekf.spoofing_suspected()

    def test_spoofing_raises_runtime_error(self) -> None:
        ekf = ch.CellHawkEkf()
        with pytest.raises(RuntimeError, match="spoofing"):
            for _ in range(3):
                ekf.check_spoofing(0.0, 0.0, 200.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# State dict completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestStateDict:
    _REQUIRED_KEYS = {
        "east_m", "north_m", "up_m",
        "v_east_m_s", "v_north_m_s", "v_up_m_s",
        "heading_rad", "jnr_db", "accel_bias_z",
        "tier", "rms_position_error_m", "timestamp_s",
        "spoofing_suspected",
    }

    def test_all_keys_present(self) -> None:
        ekf = ch.CellHawkEkf()
        state = ekf.state()
        assert self._REQUIRED_KEYS.issubset(state.keys())

    def test_tier_is_integer(self) -> None:
        ekf = ch.CellHawkEkf()
        assert isinstance(ekf.state()["tier"], int)

    def test_spoofing_suspected_is_bool(self) -> None:
        ekf = ch.CellHawkEkf()
        assert isinstance(ekf.state()["spoofing_suspected"], bool)

    def test_rms_error_is_finite_and_positive(self) -> None:
        ekf = ch.CellHawkEkf()
        rms = ekf.state()["rms_position_error_m"]
        assert math.isfinite(rms)
        assert rms > 0.0
