"""IMU drift characterisation via Allan variance model (Gap 9).

The paper (§3.4, §9) acknowledges that prolonged Tier 3 (Visual SLAM only)
operation leads to unbounded IMU integration drift.  This module provides:

  ImuDriftModel — Allan variance noise model for a MEMS IMU.
  DriftPredictor — integrates the model over time to bound position error.

Allan variance parameters (IEEE Std 952-1997):
  - Angle Random Walk (ARW, N):  white noise on gyro output  [rad/√s]
  - Bias Instability (B):        flicker noise floor          [rad/s]
  - Rate Random Walk (RRW, K):   random walk on gyro bias     [rad/s/√s]

Position error bound during GNSS denial:
  σ_pos(t) ≈ ½ · a_rms · t²   (double-integration of accel noise)

where a_rms is derived from the accelerometer ARW and bias instability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Allan variance parameters ─────────────────────────────────────────────────

@dataclass(frozen=True)
class AllanVarianceParams:
    """Allan variance noise parameters for one IMU axis.

    All values are in SI units (rad or m/s²).

    Attributes:
        arw:   Angle/Velocity Random Walk coefficient N  [rad/√s or m/s²/√s].
               Dominates at short averaging times (τ < 1 s).
        bias_instability: Bias instability B [rad/s or m/s²].
               Minimum of the Allan deviation curve.
        rrw:   Rate/Acceleration Random Walk K [rad/s/√s or m/s²/√s³].
               Dominates at long averaging times.
    """
    arw:               float   # N  — white noise coefficient
    bias_instability:  float   # B  — flicker noise floor
    rrw:               float   # K  — random walk coefficient

    def allan_deviation(self, tau: float) -> float:
        """Allan deviation σ(τ) for averaging time τ (seconds).

        σ²(τ) = N²/τ  +  (0.664·B)²  +  K²·τ/3

        The 0.664 factor converts bias instability to Allan deviation units
        (IEEE Std 952-1997, §C.3).
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        variance = (
            self.arw ** 2 / tau
            + (0.664 * self.bias_instability) ** 2
            + self.rrw ** 2 * tau / 3.0
        )
        return math.sqrt(variance)

    def optimal_averaging_time(self) -> float:
        """τ* that minimises Allan deviation (bias instability floor).

        τ* = N / (K · √(1/3))  — from dσ²/dτ = 0
        """
        if self.rrw == 0:
            return float("inf")
        return self.arw / (self.rrw * math.sqrt(1.0 / 3.0))


# ── Preset IMU profiles ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImuProfile:
    """Allan variance parameters for a complete 6-DOF IMU.

    Attributes:
        name:  Human-readable identifier.
        gyro:  Gyroscope Allan variance parameters [rad/√s, rad/s, rad/s/√s].
        accel: Accelerometer Allan variance parameters [m/s²/√s, m/s², m/s²/√s³].
    """
    name:  str
    gyro:  AllanVarianceParams
    accel: AllanVarianceParams


# MEMS-grade IMU (e.g. ICM-42688-P class) — typical for small UAVs
MEMS_UAV = ImuProfile(
    name="MEMS_UAV",
    gyro=AllanVarianceParams(
        arw=3.5e-4,           # 0.021 °/√hr → rad/√s
        bias_instability=1.0e-5,  # 0.002 °/s
        rrw=1.0e-6,
    ),
    accel=AllanVarianceParams(
        arw=1.5e-3,           # 90 µg/√Hz → m/s²/√s
        bias_instability=5.0e-5,  # ~5 µg
        rrw=2.0e-5,
    ),
)

# Tactical-grade IMU (e.g. ADIS16488 class) — higher-end UAV
TACTICAL = ImuProfile(
    name="TACTICAL",
    gyro=AllanVarianceParams(
        arw=6.0e-5,
        bias_instability=3.0e-6,
        rrw=1.0e-7,
    ),
    accel=AllanVarianceParams(
        arw=2.0e-4,
        bias_instability=5.0e-6,
        rrw=1.0e-6,
    ),
)


# ── Drift predictor ───────────────────────────────────────────────────────────

@dataclass
class DriftPredictor:
    """Predicts 1-σ position error growth during GNSS-denied Tier 3 operation.

    Uses the double-integration model:
      σ_pos(t) = ½ · σ_accel(τ=t) · t²

    where σ_accel(τ) is the accelerometer Allan deviation at averaging time τ=t.
    This is a conservative bound — Visual SLAM velocity updates (Tier 3) will
    partially correct the drift, but we model the worst case (SLAM unavailable).

    Args:
        profile:       IMU noise profile.
        slam_velocity_noise_m_s: 1-σ SLAM velocity noise (m/s).  When > 0,
                       SLAM updates reduce the drift rate.
    """
    profile:                  ImuProfile
    slam_velocity_noise_m_s:  float = 0.5   # paper §3.4: r_slam = 0.5 m/s

    def position_error_1sigma(self, denial_duration_s: float) -> float:
        """1-σ position error (m) after `denial_duration_s` seconds of GNSS denial.

        Without SLAM: σ_pos = ½ · σ_accel(t) · t²
        With SLAM:    SLAM velocity updates bound the drift to
                      σ_pos ≈ σ_v · t  (velocity random walk)
        Returns the minimum of the two bounds.
        """
        if denial_duration_s <= 0:
            return 0.0
        t = denial_duration_s
        # IMU-only bound (double integration)
        sigma_a = self.profile.accel.allan_deviation(t)
        imu_bound = 0.5 * sigma_a * t * t
        # SLAM-assisted bound (velocity integration)
        slam_bound = self.slam_velocity_noise_m_s * t
        return min(imu_bound, slam_bound)

    def time_to_error(self, target_error_m: float) -> float:
        """Seconds until 1-σ position error exceeds `target_error_m`.

        Uses bisection search over [0, 3600] s.
        Returns inf if the error never reaches the target within 1 hour.
        """
        if target_error_m <= 0:
            raise ValueError("target_error_m must be positive")
        if self.position_error_1sigma(3600.0) < target_error_m:
            return float("inf")
        lo, hi = 0.0, 3600.0
        for _ in range(60):  # 60 bisection steps → < 0.001 s precision
            mid = (lo + hi) / 2.0
            if self.position_error_1sigma(mid) < target_error_m:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def error_budget_table(self, durations_s: list[float]) -> list[tuple[float, float]]:
        """Return [(duration_s, error_1sigma_m)] for a list of durations."""
        return [(t, self.position_error_1sigma(t)) for t in durations_s]
