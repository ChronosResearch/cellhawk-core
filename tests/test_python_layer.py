"""Tests for CORTEX DQN, curriculum, heatmap, GCS, and SLAM."""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from python.cortex.dqn import CortexDQN, STATE_DIM, ACTION_DIM
from python.cortex.curriculum import AutoCurriculumScheduler, LEVELS
from python.cortex.heatmap import NeuralHeatmapProjector
from python.gcs.workers import classify_threat, compute_swarm_failover
from python.slam.interface import SimulatedSlamBackend, SlamFrame, SlamInterface


# ─────────────────────────────────────────────────────────────────────────────
# CORTEX DQN
# ─────────────────────────────────────────────────────────────────────────────

class TestCortexDQN:
    def test_output_shape(self) -> None:
        model = CortexDQN()
        x = torch.zeros(1, STATE_DIM)
        q = model(x)
        assert q.shape == (1, ACTION_DIM)

    def test_batch_output_shape(self) -> None:
        model = CortexDQN()
        x = torch.zeros(32, STATE_DIM)
        q = model(x)
        assert q.shape == (32, ACTION_DIM)

    def test_param_count_near_10k(self) -> None:
        """Paper §4.1: ~10,000 parameters (actual: 6,025 for 19→64→64→9)."""
        model = CortexDQN()
        assert 5_000 <= model.param_count <= 15_000, f"param_count={model.param_count}"

    def test_hidden_activations_shape(self) -> None:
        model = CortexDQN()
        x = torch.zeros(1, STATE_DIM)
        q, h2 = model.hidden_activations(x)
        assert q.shape == (1, ACTION_DIM)
        assert h2.shape == (1, 64)

    def test_hidden_activations_non_negative(self) -> None:
        """ReLU activations must be ≥ 0."""
        model = CortexDQN()
        x = torch.randn(4, STATE_DIM)
        _, h2 = model.hidden_activations(x)
        assert (h2 >= 0).all()

    def test_different_inputs_give_different_outputs(self) -> None:
        model = CortexDQN()
        x1 = torch.zeros(1, STATE_DIM)
        x2 = torch.ones(1, STATE_DIM)
        assert not torch.allclose(model(x1), model(x2))


# ─────────────────────────────────────────────────────────────────────────────
# Auto-curriculum
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoCurriculum:
    def test_starts_at_level_1(self) -> None:
        s = AutoCurriculumScheduler()
        assert s.level == 1

    def test_advances_after_threshold(self) -> None:
        s = AutoCurriculumScheduler(advancement_threshold=0.8, evaluation_window=5)
        for _ in range(5):
            advanced = s.record_episode(0.9)
        assert s.level == 2

    def test_does_not_advance_below_threshold(self) -> None:
        s = AutoCurriculumScheduler(advancement_threshold=0.8, evaluation_window=5)
        for _ in range(10):
            s.record_episode(0.5)
        assert s.level == 1

    def test_does_not_exceed_max_level(self) -> None:
        s = AutoCurriculumScheduler(advancement_threshold=0.0, evaluation_window=1)
        for _ in range(20):
            s.record_episode(1.0)
        assert s.level == max(LEVELS)

    def test_level5_wind_matches_paper(self) -> None:
        """Paper §4.3: Level 5 = 13 m/s sustained, 18 m/s gusts."""
        cfg = LEVELS[5]
        assert cfg.wind_sustained_m_s == 13.0
        assert cfg.wind_gust_m_s == 18.0

    def test_window_resets_after_advance(self) -> None:
        s = AutoCurriculumScheduler(advancement_threshold=0.8, evaluation_window=3)
        for _ in range(3):
            s.record_episode(1.0)
        assert s.level == 2
        # After advance, window is cleared — one bad episode should not advance again
        s.record_episode(0.0)
        assert s.level == 2


# ─────────────────────────────────────────────────────────────────────────────
# Neural heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralHeatmap:
    def test_grid_shape(self) -> None:
        proj = NeuralHeatmapProjector(grid_radius_cells=10)
        assert proj.grid.shape == (21, 21)

    def test_update_returns_grid(self) -> None:
        model = CortexDQN()
        proj = NeuralHeatmapProjector(grid_radius_cells=5)
        x = torch.randn(STATE_DIM)
        grid = proj.update(model, x, 0.0, 0.0, 0.0, 0.0)
        assert grid.shape == (11, 11)

    def test_decay_reduces_values(self) -> None:
        proj = NeuralHeatmapProjector(grid_radius_cells=5, decay_rate=0.5)
        proj._grid[:] = 1.0
        model = CortexDQN()
        # Zero-activation state → only decay applied
        x = torch.zeros(STATE_DIM)
        with torch.no_grad():
            # Force all activations to zero by zeroing weights
            for p in model.parameters():
                p.data.zero_()
        proj.update(model, x, 0.0, 0.0, 0.0, 0.0)
        assert (proj.grid <= 0.5 + 1e-6).all()

    def test_telemetry_payload_keys(self) -> None:
        proj = NeuralHeatmapProjector()
        payload = proj.to_telemetry_payload(0.0, 0.0)
        for key in ("activations", "grid_width", "grid_height", "cell_size_m"):
            assert key in payload


# ─────────────────────────────────────────────────────────────────────────────
# Celery workers (called directly, no broker needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkers:
    def test_classify_threat_tier3(self) -> None:
        result = classify_threat.run({"jnr_db": 25.0, "tier": 3})
        assert result["threat_type"] == "RF_JAMMING"
        assert result["severity"] > 0.8

    def test_classify_threat_tier1_no_threat(self) -> None:
        result = classify_threat.run({"jnr_db": 2.0, "tier": 1})
        assert result["threat_type"] == "NONE"
        assert result["severity"] == 0.0

    def test_classify_threat_tier2(self) -> None:
        result = classify_threat.run({"jnr_db": 12.0, "tier": 2})
        assert result["threat_type"] == "RF_JAMMING"
        assert 0.3 <= result["severity"] <= 0.8

    def test_swarm_failover_selects_lowest_jnr(self) -> None:
        fleet = {
            "1": {"jnr_db": 15.0, "battery_v": 14.8},
            "2": {"jnr_db": 3.0,  "battery_v": 14.2},
            "3": {"jnr_db": 22.0, "battery_v": 15.0},
        }
        result = compute_swarm_failover.run(fleet)
        assert result["new_lead_id"] == "2"

    def test_swarm_failover_empty_fleet(self) -> None:
        result = compute_swarm_failover.run({})
        assert result["new_lead_id"] is None


# ─────────────────────────────────────────────────────────────────────────────
# SLAM interface
# ─────────────────────────────────────────────────────────────────────────────

class TestSlamInterface:
    def _frame(self, vx: float = 1.0) -> SlamFrame:
        import time
        return SlamFrame(
            v_east_m_s=vx, v_north_m_s=0.5, v_up_m_s=0.0,
            heading_rad=0.1, loop_closure=False,
            timestamp_s=time.monotonic(),
        )

    def test_healthy_after_inject(self) -> None:
        backend = SimulatedSlamBackend()
        iface = SlamInterface(backend)
        assert not iface.is_healthy()
        backend.inject(self._frame())
        assert iface.is_healthy()

    def test_returns_frame_after_inject(self) -> None:
        backend = SimulatedSlamBackend()
        iface = SlamInterface(backend)
        backend.inject(self._frame(vx=3.0))
        frame = iface.get_measurement()
        assert frame is not None
        assert frame.v_east_m_s == 3.0

    def test_outlier_rejected(self) -> None:
        """Velocities above max_velocity_m_s must be discarded."""
        backend = SimulatedSlamBackend()
        iface = SlamInterface(backend, max_velocity_m_s=20.0)
        backend.inject(self._frame(vx=50.0))
        assert iface.get_measurement() is None

    def test_none_when_no_frame(self) -> None:
        backend = SimulatedSlamBackend()
        iface = SlamInterface(backend)
        assert iface.get_measurement() is None
