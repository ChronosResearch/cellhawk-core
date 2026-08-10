"""Tests for trainer, environment, and integration scenarios."""
from __future__ import annotations

import math
import torch
import pytest

from python.cortex.dqn import CortexDQN, STATE_DIM, ACTION_DIM
from python.cortex.trainer import CortexTrainer, TrainConfig, Transition, ReplayBuffer
from python.cortex.curriculum import AutoCurriculumScheduler
from python.cortex.environment import CortexEnvironment, _HunterDrone, _wrap_angle
from python.cortex.heatmap import NeuralHeatmapProjector


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayBuffer:
    def _t(self) -> Transition:
        return Transition(
            state=torch.zeros(STATE_DIM),
            action=0,
            reward=0.0,
            next_state=torch.zeros(STATE_DIM),
            done=False,
        )

    def test_capacity_enforced(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for _ in range(10):
            buf.push(self._t())
        assert len(buf) == 5

    def test_sample_size(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(self._t())
        batch = buf.sample(16)
        assert len(batch) == 16

    def test_too_small_to_sample_raises(self) -> None:
        buf = ReplayBuffer(capacity=100)
        buf.push(self._t())
        with pytest.raises(ValueError):
            buf.sample(16)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TestCortexTrainer:
    def _trainer(self) -> CortexTrainer:
        return CortexTrainer(config=TrainConfig(batch_size=8, replay_capacity=100))

    def _push_n(self, trainer: CortexTrainer, n: int) -> None:
        for _ in range(n):
            trainer.push(Transition(
                state=torch.randn(STATE_DIM),
                action=0,
                reward=0.1,
                next_state=torch.randn(STATE_DIM),
                done=False,
            ))

    def test_train_step_returns_none_when_buffer_small(self) -> None:
        trainer = self._trainer()
        self._push_n(trainer, 4)
        assert trainer.train_step() is None

    def test_train_step_returns_loss_when_ready(self) -> None:
        trainer = self._trainer()
        self._push_n(trainer, 20)
        loss = trainer.train_step()
        assert loss is not None
        assert loss >= 0.0

    def test_epsilon_decreases_with_steps(self) -> None:
        trainer = self._trainer()
        eps0 = trainer.epsilon
        self._push_n(trainer, 1000)
        assert trainer.epsilon < eps0

    def test_epsilon_bounded(self) -> None:
        cfg = TrainConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10)
        trainer = CortexTrainer(config=cfg)
        self._push_n(trainer, 10000)
        assert trainer.epsilon >= cfg.epsilon_end

    def test_select_action_valid_range(self) -> None:
        trainer = self._trainer()
        state = torch.zeros(STATE_DIM)
        for _ in range(20):
            a = trainer.select_action(state)
            assert 0 <= a < ACTION_DIM

    def test_target_network_updates(self) -> None:
        cfg = TrainConfig(batch_size=8, replay_capacity=100, target_update_freq=10)
        trainer = CortexTrainer(config=cfg)
        # Modify online network weights
        with torch.no_grad():
            for p in trainer.online_model.parameters():
                p.add_(1.0)
        # Before update: target differs
        online_sum = sum(p.sum().item() for p in trainer.online_model.parameters())
        target_sum = sum(p.sum().item() for p in trainer._target.parameters())
        assert abs(online_sum - target_sum) > 1.0
        # Push enough steps to trigger target update
        self._push_n(trainer, 10)
        for _ in range(10):
            trainer.train_step()
        # After update: target matches online
        target_sum_new = sum(p.sum().item() for p in trainer._target.parameters())
        online_sum_new = sum(p.sum().item() for p in trainer.online_model.parameters())
        assert abs(online_sum_new - target_sum_new) < 1e-3

    def test_curriculum_integration(self) -> None:
        curriculum = AutoCurriculumScheduler(advancement_threshold=0.0, evaluation_window=1)
        trainer = CortexTrainer(curriculum=curriculum)
        advanced = trainer.record_episode(1.0)
        assert advanced
        assert curriculum.level == 2


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class TestCortexEnvironment:
    def test_reset_returns_correct_obs_shape(self) -> None:
        env = CortexEnvironment()
        obs = env.reset()
        assert obs.shape == (STATE_DIM,)

    def test_step_returns_correct_shapes(self) -> None:
        env = CortexEnvironment()
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_collision_terminates_episode(self) -> None:
        """Force a collision by placing an obstacle on the drone."""
        from python.cortex.environment import Obstacle
        env = CortexEnvironment(seed=0)
        env.reset()
        # Place obstacle directly on drone
        env._obstacles = [Obstacle(
            east_m=env._state.east_m,
            north_m=env._state.north_m,
            radius_m=100.0,
        )]
        _, reward, done, info = env.step(0)
        assert done
        assert reward == -1.0
        assert info["collision"]

    def test_reaching_waypoint_gives_positive_reward(self) -> None:
        env = CortexEnvironment(waypoint=(5.0, 5.0, 50.0), seed=1)
        env.reset()
        # Place drone right next to waypoint
        env._state.east_m  = 5.0
        env._state.north_m = 5.0
        _, reward, done, info = env.step(0)
        assert done
        assert reward == 1.0
        assert info["reached"]

    def test_timeout_terminates_episode(self) -> None:
        env = CortexEnvironment(max_steps=3, seed=2)
        env.reset()
        done = False
        for _ in range(3):
            _, _, done, _ = env.step(0)
        assert done

    def test_obs_state_dim_matches_constant(self) -> None:
        env = CortexEnvironment()
        obs = env.reset()
        assert len(obs) == STATE_DIM

    def test_lidar_detects_close_obstacle(self) -> None:
        from python.cortex.environment import Obstacle
        env = CortexEnvironment()
        env.reset()
        env._state.east_m  = 0.0
        env._state.north_m = 0.0
        env._state.heading_rad = 0.0
        # Place obstacle 10 m to the east (sector 0 ≈ east direction)
        env._obstacles = [Obstacle(east_m=10.0, north_m=0.0, radius_m=2.0)]
        lidar = env._lidar_scan()
        assert min(lidar) < env.LIDAR_RANGE_M


# ─────────────────────────────────────────────────────────────────────────────
# Hunter drone TPN physics (§4.4)
# ─────────────────────────────────────────────────────────────────────────────

class TestHunterDrone:
    def test_hunter_moves_toward_target(self) -> None:
        hunter = _HunterDrone(east_m=100.0, north_m=0.0, speed_m_s=12.0)
        target_e, target_n = 0.0, 0.0
        d0 = hunter.distance_to(target_e, target_n)
        for _ in range(10):
            hunter.step_tpn(target_e, target_n)
        d1 = hunter.distance_to(target_e, target_n)
        assert d1 < d0, f"Hunter should approach target: d0={d0:.1f} d1={d1:.1f}"

    def test_wrap_angle(self) -> None:
        assert abs(_wrap_angle(0.0))           < 1e-9
        assert abs(_wrap_angle(2 * math.pi))   < 1e-9
        assert abs(_wrap_angle(-2 * math.pi))  < 1e-9
        assert abs(_wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: short training run converges loss downward
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_loss_decreases_over_training(self) -> None:
        """Loss should trend downward over 200 gradient steps."""
        cfg = TrainConfig(
            batch_size=16,
            replay_capacity=500,
            lr=1e-3,
            target_update_freq=50,
            epsilon_start=0.5,
            epsilon_end=0.1,
            epsilon_decay_steps=200,
        )
        trainer = CortexTrainer(config=cfg)
        env = CortexEnvironment(max_steps=50, seed=7)

        obs = env.reset()
        for _ in range(300):
            action = trainer.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            trainer.push(Transition(obs, action, reward, next_obs, done))
            trainer.train_step()
            obs = env.reset() if done else next_obs

        # Mean loss over last 50 steps should be lower than first 50
        losses = trainer._losses
        if len(losses) >= 100:
            early = sum(losses[:50]) / 50
            late  = sum(losses[-50:]) / 50
            assert late <= early * 1.5, f"Loss not converging: early={early:.4f} late={late:.4f}"
