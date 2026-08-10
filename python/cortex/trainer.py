"""CORTEX DQN training loop with experience replay and target network (§4.1, [7]).

Implements:
- Epsilon-greedy exploration with linear decay
- Experience replay buffer
- Hard target network update every N steps
- Auto-curriculum integration (§4.3)
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .dqn import CortexDQN, STATE_DIM, ACTION_DIM
from .curriculum import AutoCurriculumScheduler


class Transition(NamedTuple):
    state:      Tensor   # (STATE_DIM,)
    action:     int
    reward:     float
    next_state: Tensor   # (STATE_DIM,)
    done:       bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class TrainConfig:
    lr:                  float = 1e-4
    gamma:               float = 0.99
    epsilon_start:       float = 1.0
    epsilon_end:         float = 0.05
    epsilon_decay_steps: int   = 50_000
    batch_size:          int   = 64
    replay_capacity:     int   = 100_000
    target_update_freq:  int   = 1_000


class CortexTrainer:
    """DQN trainer with target network and auto-curriculum.

    Args:
        config:     Training hyperparameters.
        curriculum: Auto-curriculum scheduler (optional).
        device:     Torch device string.
    """

    def __init__(
        self,
        config: TrainConfig | None = None,
        curriculum: AutoCurriculumScheduler | None = None,
        device: str = "cpu",
    ) -> None:
        self._cfg = config or TrainConfig()
        self._curriculum = curriculum
        self._device = torch.device(device)

        self._online  = CortexDQN().to(self._device)
        self._target  = CortexDQN().to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._opt    = optim.Adam(self._online.parameters(), lr=self._cfg.lr)
        self._buf    = ReplayBuffer(self._cfg.replay_capacity)
        self._steps  = 0
        self._losses: list[float] = []

    @property
    def online_model(self) -> CortexDQN:
        return self._online

    @property
    def epsilon(self) -> float:
        frac = min(self._steps / self._cfg.epsilon_decay_steps, 1.0)
        return self._cfg.epsilon_start + frac * (self._cfg.epsilon_end - self._cfg.epsilon_start)

    def select_action(self, state: Tensor) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        self._online.eval()
        with torch.no_grad():
            q = self._online(state.unsqueeze(0).to(self._device))
        return int(q.argmax(dim=1).item())

    def push(self, t: Transition) -> None:
        self._buf.push(t)
        self._steps += 1

    def train_step(self) -> float | None:
        """Sample a mini-batch and perform one gradient step.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self._buf) < self._cfg.batch_size:
            return None

        batch = self._buf.sample(self._cfg.batch_size)
        states      = torch.stack([t.state      for t in batch]).to(self._device)
        next_states = torch.stack([t.next_state for t in batch]).to(self._device)
        actions     = torch.tensor([t.action for t in batch], dtype=torch.long,  device=self._device)
        rewards     = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self._device)
        dones       = torch.tensor([t.done   for t in batch], dtype=torch.float32, device=self._device)

        # Current Q-values for taken actions
        self._online.train()
        q_current = self._online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman)
        with torch.no_grad():
            q_next = self._target(next_states).max(dim=1).values
            q_target = rewards + self._cfg.gamma * q_next * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_current, q_target)

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), max_norm=10.0)
        self._opt.step()

        loss_val = float(loss.item())
        self._losses.append(loss_val)

        # Hard target update
        if self._steps % self._cfg.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

        return loss_val

    def record_episode(self, normalised_reward: float) -> bool:
        """Forward episode reward to curriculum scheduler.

        Returns True if difficulty level was advanced.
        """
        if self._curriculum is not None:
            return self._curriculum.record_episode(normalised_reward)
        return False

    def mean_loss(self, window: int = 100) -> float:
        if not self._losses:
            return 0.0
        recent = self._losses[-window:]
        return sum(recent) / len(recent)
