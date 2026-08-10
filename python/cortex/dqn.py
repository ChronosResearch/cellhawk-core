"""CORTEX v2.0 Deep Q-Network architecture (§4.1).

State:  19-dimensional continuous vector (§4.1)
Output: 9 discrete navigational intent actions
Params: ~10,000 (sub-ms inference on Jetson Orin Nano)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Action space ──────────────────────────────────────────────────────────────

# 9 discrete navigational intent vectors (§4.1):
# 0=hover, 1-4=cardinal headings, 5=climb, 6=descend, 7=evade-left, 8=evade-right
ACTION_DIM = 9
STATE_DIM  = 19  # §4.1


class CortexDQN(nn.Module):
    """Compact DQN: two FC layers of 64 neurons each (~10k params).

    Intentionally small so TensorRT inference runs in < 1 ms on edge
    hardware without GPU acceleration (§2.1, §4.1).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

        # Xavier init for stable early training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Return Q-values for each action.

        Args:
            x: State tensor of shape (batch, STATE_DIM).

        Returns:
            Q-values of shape (batch, ACTION_DIM).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def hidden_activations(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return Q-values and second-layer activations for heatmap projection (§4.5).

        Args:
            x: State tensor of shape (batch, STATE_DIM).

        Returns:
            (q_values, layer2_activations) — layer2 has shape (batch, 64).
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.out(h2), h2

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
