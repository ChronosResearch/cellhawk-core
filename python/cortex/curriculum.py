"""Auto-curriculum scheduler for CORTEX DQN training (§4.3).

Difficulty advances when the agent's mean reward over the evaluation
window exceeds the advancement threshold.  Five levels map directly
to the table in §4.3.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LevelConfig:
    wind_sustained_m_s: float
    wind_gust_m_s: float
    obstacle_density: float   # features/m²
    jnr_max_db: float


# §4.3 table — exact values from paper
LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(0.0,  5.0,  0.10,  5.0),
    2: LevelConfig(5.0,  8.0,  0.20, 10.0),
    3: LevelConfig(8.0, 12.0,  0.35, 15.0),
    4: LevelConfig(10.0, 15.0, 0.50, 20.0),
    5: LevelConfig(13.0, 18.0, 0.70, 35.0),
}


class AutoCurriculumScheduler:
    """Advance difficulty when rolling mean reward exceeds threshold.

    Args:
        advancement_threshold: Mean reward required to advance (0–1).
        evaluation_window:     Number of episodes in rolling window.
    """

    def __init__(
        self,
        advancement_threshold: float = 0.85,
        evaluation_window: int = 50,
    ) -> None:
        self._threshold = advancement_threshold
        self._window: deque[float] = deque(maxlen=evaluation_window)
        self._level = 1

    @property
    def level(self) -> int:
        return self._level

    @property
    def config(self) -> LevelConfig:
        return LEVELS[self._level]

    def record_episode(self, normalised_reward: float) -> bool:
        """Record episode reward and advance level if ready.

        Args:
            normalised_reward: Episode reward normalised to [0, 1].

        Returns:
            True if the level was advanced this call.
        """
        self._window.append(normalised_reward)
        if (
            self._level < max(LEVELS)
            and len(self._window) == self._window.maxlen
            and self._mean_reward() >= self._threshold
        ):
            self._level += 1
            self._window.clear()
            return True
        return False

    def _mean_reward(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0
