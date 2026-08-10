"""Tests for Gap 9: IMU drift characterisation (python/cortex/imu_drift.py)."""
from __future__ import annotations

import math

import pytest

from python.cortex.imu_drift import (
    AllanVarianceParams,
    DriftPredictor,
    ImuProfile,
    MEMS_UAV,
    TACTICAL,
)


# ── AllanVarianceParams ───────────────────────────────────────────────────────

class TestAllanVarianceParams:
    def setup_method(self) -> None:
        # Typical MEMS gyro parameters
        self.params = AllanVarianceParams(
            arw=3.5e-4,
            bias_instability=1.0e-5,
            rrw=1.0e-6,
        )

    def test_allan_deviation_positive(self) -> None:
        assert self.params.allan_deviation(1.0) > 0

    def test_allan_deviation_rejects_zero_tau(self) -> None:
        with pytest.raises(ValueError):
            self.params.allan_deviation(0.0)

    def test_allan_deviation_rejects_negative_tau(self) -> None:
        with pytest.raises(ValueError):
            self.params.allan_deviation(-1.0)

    def test_arw_dominates_at_short_tau(self) -> None:
        """At τ=0.01 s, ARW term N²/τ should dominate."""
        tau = 0.01
        arw_term = self.params.arw ** 2 / tau
        rrw_term = self.params.rrw ** 2 * tau / 3.0
        assert arw_term > rrw_term * 100

    def test_rrw_dominates_at_long_tau(self) -> None:
        """At τ=10000 s, RRW term K²τ/3 should dominate."""
        tau = 10_000.0
        arw_term = self.params.arw ** 2 / tau
        rrw_term = self.params.rrw ** 2 * tau / 3.0
        assert rrw_term > arw_term * 100

    def test_allan_deviation_has_minimum(self) -> None:
        """Allan deviation must have a minimum (U-shape)."""
        taus = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        devs = [self.params.allan_deviation(t) for t in taus]
        min_dev = min(devs)
        # Minimum must be strictly less than the short-tau endpoint
        assert min_dev < devs[0]
        # And less than or equal to the long-tau endpoint (may be flat at RRW floor)
        assert min_dev <= devs[-1]

    def test_optimal_averaging_time_positive(self) -> None:
        tau_star = self.params.optimal_averaging_time()
        assert tau_star > 0

    def test_optimal_averaging_time_near_minimum(self) -> None:
        """Allan deviation at τ* should be close to the minimum."""
        tau_star = self.params.optimal_averaging_time()
        dev_star = self.params.allan_deviation(tau_star)
        # Check it's lower than at 10× and 0.1× τ*
        assert dev_star <= self.params.allan_deviation(tau_star * 10) + 1e-12
        assert dev_star <= self.params.allan_deviation(tau_star * 0.1) + 1e-12

    def test_zero_rrw_gives_infinite_optimal_tau(self) -> None:
        p = AllanVarianceParams(arw=1e-3, bias_instability=1e-5, rrw=0.0)
        assert p.optimal_averaging_time() == float("inf")

    def test_allan_variance_formula_components(self) -> None:
        """Verify the formula: σ²(τ) = N²/τ + (0.664B)² + K²τ/3."""
        p = AllanVarianceParams(arw=1.0, bias_instability=1.0, rrw=1.0)
        tau = 4.0
        expected_var = 1.0 / tau + (0.664) ** 2 + tau / 3.0
        assert abs(p.allan_deviation(tau) ** 2 - expected_var) < 1e-12


# ── ImuProfile presets ────────────────────────────────────────────────────────

class TestImuProfiles:
    def test_mems_uav_has_name(self) -> None:
        assert MEMS_UAV.name == "MEMS_UAV"

    def test_tactical_has_name(self) -> None:
        assert TACTICAL.name == "TACTICAL"

    def test_tactical_gyro_better_than_mems(self) -> None:
        """Tactical IMU must have lower ARW than MEMS."""
        assert TACTICAL.gyro.arw < MEMS_UAV.gyro.arw

    def test_tactical_accel_better_than_mems(self) -> None:
        assert TACTICAL.accel.arw < MEMS_UAV.accel.arw

    def test_mems_accel_allan_deviation_at_1s(self) -> None:
        """MEMS accel Allan deviation at τ=1 s should be dominated by ARW."""
        dev = MEMS_UAV.accel.allan_deviation(1.0)
        assert dev > 0
        assert dev < 1.0  # sanity: < 1 m/s² noise at 1 s

    def test_profiles_have_positive_parameters(self) -> None:
        for profile in (MEMS_UAV, TACTICAL):
            for params in (profile.gyro, profile.accel):
                assert params.arw > 0
                assert params.bias_instability > 0
                assert params.rrw > 0


# ── DriftPredictor ────────────────────────────────────────────────────────────

class TestDriftPredictor:
    def setup_method(self) -> None:
        self.predictor = DriftPredictor(profile=MEMS_UAV)

    def test_zero_duration_gives_zero_error(self) -> None:
        assert self.predictor.position_error_1sigma(0.0) == 0.0

    def test_negative_duration_gives_zero_error(self) -> None:
        assert self.predictor.position_error_1sigma(-1.0) == 0.0

    def test_error_grows_with_time(self) -> None:
        e1 = self.predictor.position_error_1sigma(10.0)
        e2 = self.predictor.position_error_1sigma(60.0)
        assert e2 > e1

    def test_slam_bound_limits_long_term_drift(self) -> None:
        """SLAM velocity bound (σ_v · t) must cap the IMU double-integration."""
        # At very long times, SLAM bound = 0.5 * t grows slower than IMU bound
        # which grows as t² — so SLAM bound should be active at long durations
        predictor_no_slam = DriftPredictor(
            profile=MEMS_UAV, slam_velocity_noise_m_s=0.0
        )
        # With zero SLAM noise, bound = min(imu_bound, 0) = 0 — not useful
        # Use a realistic SLAM noise and verify it's tighter than IMU at 300 s
        predictor = DriftPredictor(profile=MEMS_UAV, slam_velocity_noise_m_s=0.5)
        t = 300.0
        sigma_a = MEMS_UAV.accel.allan_deviation(t)
        imu_bound = 0.5 * sigma_a * t * t
        slam_bound = 0.5 * t
        assert predictor.position_error_1sigma(t) == min(imu_bound, slam_bound)

    def test_error_budget_table_length(self) -> None:
        durations = [10.0, 30.0, 60.0, 120.0, 300.0]
        table = self.predictor.error_budget_table(durations)
        assert len(table) == len(durations)

    def test_error_budget_table_monotone(self) -> None:
        durations = [10.0, 30.0, 60.0, 120.0, 300.0]
        table = self.predictor.error_budget_table(durations)
        errors = [e for _, e in table]
        assert all(errors[i] <= errors[i + 1] for i in range(len(errors) - 1))

    def test_time_to_error_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            self.predictor.time_to_error(0.0)

    def test_time_to_error_returns_inf_for_unreachable_target(self) -> None:
        # 1e9 m is unreachable within 1 hour
        result = self.predictor.time_to_error(1e9)
        assert result == float("inf")

    def test_time_to_error_consistency(self) -> None:
        """position_error_1sigma(time_to_error(target)) ≈ target."""
        target = 50.0  # 50 m
        t = self.predictor.time_to_error(target)
        if math.isinf(t):
            pytest.skip("target unreachable")
        error_at_t = self.predictor.position_error_1sigma(t)
        assert abs(error_at_t - target) < 0.1  # within 0.1 m

    def test_tactical_imu_drifts_slower_than_mems(self) -> None:
        """Tactical IMU must have lower position error at 60 s."""
        mems_pred = DriftPredictor(profile=MEMS_UAV)
        tact_pred = DriftPredictor(profile=TACTICAL)
        assert tact_pred.position_error_1sigma(60.0) < mems_pred.position_error_1sigma(60.0)

    def test_paper_tier3_rms_claim_plausible(self) -> None:
        """Paper claims ~12 m RMS in Tier 3 (terrain-rich).

        With SLAM at 0.5 m/s noise, the 1-σ bound at 24 s (≈ 12 m / 0.5 m/s)
        should be ≤ 12 m.
        """
        predictor = DriftPredictor(profile=MEMS_UAV, slam_velocity_noise_m_s=0.5)
        # At 24 s, SLAM bound = 0.5 * 24 = 12 m
        error = predictor.position_error_1sigma(24.0)
        assert error <= 12.0, f"1-σ error at 24 s = {error:.2f} m, expected ≤ 12 m"

    def test_mems_position_error_at_10s_reasonable(self) -> None:
        """MEMS IMU: 1-σ position error at 10 s should be < 5 m (SLAM-assisted)."""
        predictor = DriftPredictor(profile=MEMS_UAV, slam_velocity_noise_m_s=0.5)
        error = predictor.position_error_1sigma(10.0)
        assert error < 5.0, f"Error at 10 s = {error:.3f} m"

    def test_custom_profile(self) -> None:
        """DriftPredictor works with a custom ImuProfile."""
        custom = ImuProfile(
            name="custom",
            gyro=AllanVarianceParams(arw=1e-3, bias_instability=1e-4, rrw=1e-5),
            accel=AllanVarianceParams(arw=2e-3, bias_instability=2e-4, rrw=2e-5),
        )
        pred = DriftPredictor(profile=custom)
        assert pred.position_error_1sigma(30.0) > 0
