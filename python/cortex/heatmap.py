"""Neural heatmap projection from DQN hidden-layer activations (§4.5).

Extracts second FC-layer activations (64 neurons) and projects them
onto a 2-D geospatial grid centred on the drone's current position.
Activation magnitude correlates with perceived risk in each terrain sector.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .dqn import CortexDQN


class NeuralHeatmapProjector:
    """Project DQN layer-2 activations onto a geospatial grid.

    Args:
        grid_resolution_m: Cell size in metres (default 5 m, §4.5 config).
        grid_radius_cells: Half-width of the grid in cells.
        decay_rate:        Per-frame exponential decay of the heatmap.
    """

    def __init__(
        self,
        grid_resolution_m: float = 5.0,
        grid_radius_cells: int = 20,
        decay_rate: float = 0.95,
    ) -> None:
        self._res = grid_resolution_m
        self._r = grid_radius_cells
        self._decay = decay_rate
        size = 2 * grid_radius_cells + 1
        self._grid = np.zeros((size, size), dtype=np.float32)

    @property
    def grid(self) -> np.ndarray:
        """Current heatmap grid (2-D float32 array)."""
        return self._grid

    def update(
        self,
        model: CortexDQN,
        state: Tensor,
        drone_east_m: float,
        drone_north_m: float,
        origin_east_m: float,
        origin_north_m: float,
    ) -> np.ndarray:
        """Compute activations and splat them onto the grid.

        The 64 neurons are mapped to 64 spatial directions evenly distributed
        around the drone.  Each neuron's activation magnitude is added to the
        grid cell it points toward at a distance proportional to its magnitude.

        Returns:
            Updated heatmap grid.
        """
        model.eval()
        with torch.no_grad():
            _, activations = model.hidden_activations(state.unsqueeze(0))
        acts: np.ndarray = activations.squeeze(0).cpu().numpy()  # (64,)

        # Decay existing heatmap
        self._grid *= self._decay

        # Drone position in grid coordinates
        cx = int((drone_east_m  - origin_east_m)  / self._res) + self._r
        cy = int((drone_north_m - origin_north_m) / self._res) + self._r

        n_neurons = len(acts)
        for idx, magnitude in enumerate(acts):
            if magnitude < 1e-4:
                continue
            angle = 2.0 * np.pi * idx / n_neurons
            # Project at distance = magnitude (clamped to grid radius)
            dist_cells = min(int(magnitude), self._r)
            gx = cx + int(dist_cells * np.cos(angle))
            gy = cy + int(dist_cells * np.sin(angle))
            size = self._grid.shape[0]
            if 0 <= gx < size and 0 <= gy < size:
                self._grid[gy, gx] += float(magnitude)

        return self._grid

    def to_telemetry_payload(
        self,
        origin_east_m: float,
        origin_north_m: float,
    ) -> dict:
        """Serialise heatmap for Protobuf HeatmapFrame telemetry."""
        return {
            "activations": self._grid.flatten().tolist(),
            "grid_width": self._grid.shape[1],
            "grid_height": self._grid.shape[0],
            "grid_origin_east_m": origin_east_m,
            "grid_origin_north_m": origin_north_m,
            "cell_size_m": self._res,
        }
