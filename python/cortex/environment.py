"""Simulation environment for CORTEX DQN training (§4.2, §4.3, §4.4).

Models:
- 3-D terrain with configurable obstacle density
- Wind disturbance (sustained + gust, §4.3 levels)
- Electronic Attack jamming dome with configurable JNR
- Adversarial hunter drone with True Proportional Navigation (§4.4)
- 8-sector LiDAR proximity model
- Reward shaping: progress toward waypoint, collision penalty, survival bonus
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .dqn import STATE_DIM, ACTION_DIM
from .curriculum import LevelConfig, LEVELS
from .jamming import JammingDome, JammingDomeField

# Optional terrain grid — imported lazily to avoid hard dependency
try:
    from python.terrain.osm_loader import TerrainGrid, OsmTerrainLoader
    _TERRAIN_AVAILABLE = True
except ImportError:
    _TERRAIN_AVAILABLE = False


# ── Action → heading/altitude delta mapping ───────────────────────────────────
# 0=hover, 1=N, 2=E, 3=S, 4=W, 5=climb, 6=descend, 7=evade-left, 8=evade-right
_ACTION_HEADING_DELTA = [0.0, 0.0, math.pi/2, math.pi, -math.pi/2, 0.0, 0.0, -math.pi/4, math.pi/4]
_ACTION_ALT_DELTA     = [0.0, 0.0, 0.0,       0.0,     0.0,        2.0, -2.0, 0.0,        0.0]
_DT = 0.1  # 10 Hz control loop (§4.2)


@dataclass
class Obstacle:
    east_m:  float
    north_m: float
    radius_m: float


@dataclass
class EnvState:
    east_m:      float = 0.0
    north_m:     float = 0.0
    alt_m:       float = 50.0
    heading_rad: float = 0.0
    speed_m_s:   float = 10.0
    battery_v:   float = 14.8
    jnr_db:      float = 0.0
    step:        int   = 0


class CortexEnvironment:
    """Simulation environment for one drone agent.

    Args:
        level_config:   Curriculum difficulty level config.
        waypoint:       Target (east_m, north_m, alt_m).
        max_steps:      Episode length limit.
        seed:           RNG seed for reproducibility.
    """

    LIDAR_RANGE_M   = 50.0
    COLLISION_RADIUS = 3.0   # metres — drone body radius
    BATTERY_DRAIN_PER_STEP = 0.001  # V per step

    def __init__(
        self,
        level_config: LevelConfig | None = None,
        waypoint: tuple[float, float, float] = (500.0, 500.0, 50.0),
        max_steps: int = 1000,
        seed: int = 42,
        terrain_grid: object | None = None,
    ) -> None:
        self._cfg     = level_config or LEVELS[1]
        self._wp      = waypoint
        self._max_steps = max_steps
        self._rng     = random.Random(seed)
        self._obstacles: list[Obstacle] = []
        self._hunter: Optional[_HunterDrone] = None
        self._dome_field: Optional[JammingDomeField] = None
        self._terrain_grid = terrain_grid  # Optional[TerrainGrid]
        self._state   = EnvState()
        self.reset()

    def reset(self) -> Tensor:
        self._state = EnvState(
            east_m=self._rng.uniform(-10.0, 10.0),
            north_m=self._rng.uniform(-10.0, 10.0),
            alt_m=50.0,
            heading_rad=self._rng.uniform(-math.pi, math.pi),
            speed_m_s=10.0,
            battery_v=14.8,
            jnr_db=self._rng.uniform(0.0, self._cfg.jnr_max_db * 0.3),
        )
        self._obstacles = self._spawn_obstacles()
        self._hunter = _HunterDrone(
            east_m=self._rng.uniform(200.0, 400.0),
            north_m=self._rng.uniform(200.0, 400.0),
            speed_m_s=self._state.speed_m_s * 1.2,  # 20% advantage §4.4
        )
        self._dome_field = self._spawn_dome_field()
        return self._observe()

    def step(self, action: int) -> tuple[Tensor, float, bool, dict]:
        """Advance one 100 ms control step.

        Returns:
            (next_obs, reward, done, info)
        """
        s = self._state
        s.step += 1

        # ── Apply action ──────────────────────────────────────────────────────
        s.heading_rad += _ACTION_HEADING_DELTA[action]
        s.alt_m       = max(5.0, s.alt_m + _ACTION_ALT_DELTA[action])
        wind_e, wind_n = self._wind_disturbance()
        s.east_m  += (s.speed_m_s * math.cos(s.heading_rad) + wind_e) * _DT
        s.north_m += (s.speed_m_s * math.sin(s.heading_rad) + wind_n) * _DT
        s.battery_v -= self.BATTERY_DRAIN_PER_STEP
        s.jnr_db = self._jnr_sample()

        # ── Hunter drone step ─────────────────────────────────────────────────
        hunter_close = False
        if self._hunter:
            self._hunter.step_tpn(s.east_m, s.north_m)
            hunter_close = self._hunter.distance_to(s.east_m, s.north_m) < 10.0

        # ── Termination conditions ────────────────────────────────────────────
        collision = self._check_collision()
        reached   = self._distance_to_waypoint() < 10.0
        timeout   = s.step >= self._max_steps
        dead      = collision or hunter_close or s.battery_v < 10.0
        done      = dead or reached or timeout

        # ── Reward ────────────────────────────────────────────────────────────
        reward = self._compute_reward(reached, dead)

        info = {
            "reached": reached,
            "collision": collision,
            "hunter_close": hunter_close,
            "step": s.step,
            "jnr_db": s.jnr_db,
        }
        return self._observe(), reward, done, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _observe(self) -> Tensor:
        s = self._state
        lidar = self._lidar_scan()
        wp_e, wp_n, wp_a = self._wp
        bearing = math.atan2(wp_n - s.north_m, wp_e - s.east_m)
        heading_err = _wrap_angle(bearing - s.heading_rad)

        obs = [
            *lidar,                          # 8 sectors
            s.alt_m,                         # baro altitude
            bearing,                         # destination bearing
            s.speed_m_s * math.cos(s.heading_rad),  # v_east
            s.speed_m_s * math.sin(s.heading_rad),  # v_north
            0.0,                             # v_up (simplified)
            s.battery_v,
            s.jnr_db,
            float(self._active_tier()),      # tier index
            heading_err,                     # heading error
            0.0, 0.0,                        # reserved slots 17-18
        ]
        assert len(obs) == STATE_DIM
        return torch.tensor(obs, dtype=torch.float32)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lidar_scan(self) -> list[float]:
        """8-sector LiDAR: returns distance to nearest obstacle per sector."""
        s = self._state
        sectors = [self.LIDAR_RANGE_M] * 8
        for obs in self._obstacles:
            de = obs.east_m - s.east_m
            dn = obs.north_m - s.north_m
            dist = math.hypot(de, dn) - obs.radius_m
            if dist > self.LIDAR_RANGE_M:
                continue
            angle = math.atan2(dn, de)
            sector = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
            sectors[sector] = min(sectors[sector], max(0.0, dist))
        return sectors

    def _wind_disturbance(self) -> tuple[float, float]:
        sustained = self._cfg.wind_sustained_m_s
        gust = self._rng.uniform(0.0, self._cfg.wind_gust_m_s - sustained) if self._cfg.wind_gust_m_s > sustained else 0.0
        mag = sustained + gust
        angle = self._rng.uniform(0.0, 2 * math.pi)
        return mag * math.cos(angle), mag * math.sin(angle)

    def _jnr_sample(self) -> float:
        """Return JNR at the drone's current position using the dome field.

        Falls back to uniform sampling if no dome field is active (e.g. during
        the very first step before reset completes).
        """
        if self._dome_field is not None:
            return self._dome_field.step(
                self._state.east_m,
                self._state.north_m,
                rng_gauss=self._rng.gauss(0.0, 1.0),
            )
        # Fallback: uniform sampling (pre-reset state)
        return self._rng.uniform(0.0, self._cfg.jnr_max_db)

    def _active_tier(self) -> int:
        jnr = self._state.jnr_db
        if jnr < 6.0:   return 1
        if jnr < 19.0:  return 2
        return 3

    def _distance_to_waypoint(self) -> float:
        wp_e, wp_n, _ = self._wp
        return math.hypot(wp_e - self._state.east_m, wp_n - self._state.north_m)

    def _check_collision(self) -> bool:
        s = self._state
        return any(
            math.hypot(o.east_m - s.east_m, o.north_m - s.north_m) < (o.radius_m + self.COLLISION_RADIUS)
            for o in self._obstacles
        )

    def _compute_reward(self, reached: bool, dead: bool) -> float:
        if reached: return 1.0
        if dead:    return -1.0
        # Progress reward: normalised reduction in distance to waypoint
        max_dist = math.hypot(*self._wp[:2]) + 100.0
        progress = 1.0 - self._distance_to_waypoint() / max_dist
        return 0.01 * progress

    def _spawn_dome_field(self) -> JammingDomeField:
        """Spawn a dome field for this episode using the curriculum level config."""
        return JammingDomeField.from_level_config(
            peak_jnr_db=self._cfg.jnr_max_db,
            arena_size_m=600.0,
            n_domes=1,
            dome_radius_fraction=0.15,
            rng_seed=self._rng.randint(0, 2**31),
        )

    def _spawn_obstacles(self) -> list[Obstacle]:
        """Spawn obstacles from terrain grid if available, else random."""
        if self._terrain_grid is not None:
            # Use real OSM building footprints
            raw = self._terrain_grid.to_environment_obstacles()  # type: ignore[union-attr]
            return [Obstacle(east_m=o["east_m"], north_m=o["north_m"], radius_m=o["radius_m"]) for o in raw]
        # Fallback: random obstacle spawning
        area = 600.0 * 600.0
        n = int(self._cfg.obstacle_density * area / 100.0)  # density in features per 100 m²
        return [
            Obstacle(
                east_m=self._rng.uniform(20.0, 580.0),
                north_m=self._rng.uniform(20.0, 580.0),
                radius_m=self._rng.uniform(3.0, 15.0),
            )
            for _ in range(n)
        ]


class _HunterDrone:
    """Adversarial hunter with True Proportional Navigation (§4.4)."""

    N_PRIME = 3.0  # navigation constant

    def __init__(self, east_m: float, north_m: float, speed_m_s: float) -> None:
        self.east_m   = east_m
        self.north_m  = north_m
        self.speed_m_s = speed_m_s
        self._prev_los_angle: float | None = None

    def step_tpn(self, target_e: float, target_n: float) -> None:
        """Advance hunter one step using TPN guidance law (§4.4)."""
        de = target_e - self.east_m
        dn = target_n - self.north_m
        los_angle = math.atan2(dn, de)

        if self._prev_los_angle is not None:
            los_rate = _wrap_angle(los_angle - self._prev_los_angle) / _DT
            # TPN: a_c = N' · V_c · λ̇  (perpendicular to LOS)
            closing_v = self.speed_m_s
            lat_accel = self.N_PRIME * closing_v * los_rate
            # Commanded heading = LOS + lateral correction
            commanded = los_angle + math.atan2(lat_accel * _DT, closing_v)
        else:
            commanded = los_angle

        self._prev_los_angle = los_angle
        self.east_m  += self.speed_m_s * math.cos(commanded) * _DT
        self.north_m += self.speed_m_s * math.sin(commanded) * _DT

    def distance_to(self, east_m: float, north_m: float) -> float:
        return math.hypot(east_m - self.east_m, north_m - self.north_m)


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
