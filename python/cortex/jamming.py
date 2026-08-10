"""Spatial Electronic Attack (EA) jamming dome model (§2.1, §4.3 gap fix).

## Physical model

A jamming dome is a bounded spherical region of elevated JNR centred on a
ground-based jammer.  The JNR experienced by a drone at position (e, n) is:

    JNR(e, n) = peak_jnr_db · ramp(t) · spatial_falloff(dist)

where:
    spatial_falloff(dist) = max(0, 1 − dist / radius_m)   [linear taper]
    ramp(t)               = sigmoid over ramp_steps        [100 ms rise time]

Multiple overlapping domes are supported; the drone experiences the maximum
JNR across all domes (worst-case jamming).

## Integration with CortexEnvironment

Replace the uniform `_jnr_sample()` call with `JammingDomeField.step()`.
The dome field is re-spawned on each episode reset using the curriculum
level's `jnr_max_db` as the peak JNR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# JammingDome
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JammingDome:
    """A single bounded jamming dome.

    Args:
        center_east_m:  Dome centre east coordinate (m ENU).
        center_north_m: Dome centre north coordinate (m ENU).
        radius_m:       Dome radius (m).  JNR = 0 outside this boundary.
        peak_jnr_db:    Peak JNR at the dome centre (dB).
        ramp_steps:     Number of 100 ms control steps for the JNR to ramp
                        from 0 to peak when entering the dome.  Default 1
                        (instantaneous, matching the paper's 100 ms claim at
                        10 Hz).  Set > 1 for a smoother ramp.
    """
    center_east_m:  float
    center_north_m: float
    radius_m:       float
    peak_jnr_db:    float
    ramp_steps:     int = 1

    def distance_to(self, east_m: float, north_m: float) -> float:
        """Euclidean distance from a point to the dome centre (m)."""
        return math.hypot(east_m - self.center_east_m, north_m - self.center_north_m)

    def spatial_falloff(self, east_m: float, north_m: float) -> float:
        """Linear spatial taper: 1.0 at centre, 0.0 at boundary, 0.0 outside."""
        dist = self.distance_to(east_m, north_m)
        return max(0.0, 1.0 - dist / self.radius_m)


# ─────────────────────────────────────────────────────────────────────────────
# JammingDomeField
# ─────────────────────────────────────────────────────────────────────────────

class JammingDomeField:
    """Manages a collection of jamming domes and tracks per-dome ramp state.

    The ramp state for each dome is a float in [0, 1]:
    - Increases by 1/ramp_steps per step when the drone is inside the dome.
    - Decreases by 1/ramp_steps per step when the drone is outside.

    This models the 100 ms activation ramp described in the paper (§2.1).

    Args:
        domes:            List of jamming domes in the arena.
        background_jnr:   Ambient JNR outside all domes (dB).  Represents
                          broadband noise floor.  Default 0.5 dB.
        noise_std:        Gaussian noise added to the JNR output (dB).
                          Models measurement uncertainty.  Default 0.5 dB.
    """

    def __init__(
        self,
        domes: list[JammingDome],
        background_jnr: float = 0.5,
        noise_std: float = 0.5,
    ) -> None:
        self._domes = domes
        self._background = background_jnr
        self._noise_std = noise_std
        # Per-dome ramp state [0, 1]
        self._ramp: list[float] = [0.0] * len(domes)

    @property
    def domes(self) -> list[JammingDome]:
        return list(self._domes)

    def step(self, east_m: float, north_m: float, rng_gauss: float = 0.0) -> float:
        """Advance ramp states and return the JNR at the given position.

        Args:
            east_m, north_m: Current drone position (m ENU).
            rng_gauss:       Pre-sampled N(0,1) variate for noise injection.
                             Pass 0.0 for deterministic output (tests).

        Returns:
            JNR in dB (≥ 0).
        """
        max_jnr = self._background

        for i, dome in enumerate(self._domes):
            inside = dome.distance_to(east_m, north_m) < dome.radius_m
            step_size = 1.0 / max(dome.ramp_steps, 1)

            if inside:
                self._ramp[i] = min(1.0, self._ramp[i] + step_size)
                jnr = dome.peak_jnr_db * self._ramp[i] * dome.spatial_falloff(east_m, north_m)
            else:
                self._ramp[i] = max(0.0, self._ramp[i] - step_size)
                # Ramp-down: drone just exited — JNR decays toward 0
                jnr = dome.peak_jnr_db * self._ramp[i]

            max_jnr = max(max_jnr, jnr)

        # Add measurement noise
        noisy = max_jnr + self._noise_std * rng_gauss
        return max(0.0, noisy)

    def reset(self) -> None:
        """Reset all ramp states to 0 (call on episode reset)."""
        self._ramp = [0.0] * len(self._domes)

    def jnr_at_centre(self, dome_idx: int) -> float:
        """Peak JNR at the centre of dome `dome_idx` (fully ramped)."""
        if dome_idx >= len(self._domes):
            return 0.0
        dome = self._domes[dome_idx]
        return dome.peak_jnr_db

    @classmethod
    def from_level_config(
        cls,
        peak_jnr_db: float,
        arena_size_m: float = 600.0,
        n_domes: int = 1,
        dome_radius_fraction: float = 0.15,
        rng_seed: int = 42,
    ) -> "JammingDomeField":
        """Factory: create a dome field from curriculum level parameters.

        Places `n_domes` domes at random positions within the arena.
        Dome radius = arena_size_m × dome_radius_fraction.

        Args:
            peak_jnr_db:          Peak JNR at dome centres (dB).
            arena_size_m:         Side length of the square arena (m).
            n_domes:              Number of domes to place.
            dome_radius_fraction: Dome radius as a fraction of arena size.
            rng_seed:             Seed for reproducible dome placement.
        """
        import random
        rng = random.Random(rng_seed)
        radius = arena_size_m * dome_radius_fraction
        margin = radius + 20.0  # keep domes away from arena edges

        domes = [
            JammingDome(
                center_east_m=rng.uniform(margin, arena_size_m - margin),
                center_north_m=rng.uniform(margin, arena_size_m - margin),
                radius_m=radius,
                peak_jnr_db=peak_jnr_db,
                ramp_steps=1,
            )
            for _ in range(n_domes)
        ]
        return cls(domes)
