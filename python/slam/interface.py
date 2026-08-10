"""Visual SLAM interface — ORB-SLAM2 compatible (§3.4, Tier 3).

Wraps any SLAM backend that exposes a velocity + heading estimate
and presents a clean SlamMeasurement stream for EKF fusion.

In production this connects to an ORB-SLAM2 ROS topic or shared-memory
buffer.  For simulation it accepts synthetic velocity inputs directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SlamFrame:
    """Raw output from the SLAM backend."""
    v_east_m_s:  float
    v_north_m_s: float
    v_up_m_s:    float
    heading_rad: float
    loop_closure: bool
    timestamp_s: float
    # Velocity covariance diagonal (3 values: east, north, up)
    velocity_cov: tuple[float, float, float] = (0.25, 0.25, 0.25)


class SlamBackend(Protocol):
    """Protocol any SLAM backend must satisfy."""
    def latest_frame(self) -> SlamFrame | None: ...
    def is_healthy(self) -> bool: ...


class SimulatedSlamBackend:
    """Synthetic SLAM backend for simulation and unit tests.

    Accepts velocity injections; reports healthy when at least one
    frame has been injected in the last 500 ms.
    """

    def __init__(self) -> None:
        self._frame: SlamFrame | None = None

    def inject(self, frame: SlamFrame) -> None:
        self._frame = frame

    def latest_frame(self) -> SlamFrame | None:
        return self._frame

    def is_healthy(self) -> bool:
        if self._frame is None:
            return False
        return (time.monotonic() - self._frame.timestamp_s) < 0.5


class SlamInterface:
    """Thin adapter between a SLAM backend and the EKF measurement format.

    Args:
        backend: Any object satisfying the SlamBackend protocol.
        max_velocity_m_s: Sanity-check upper bound; frames exceeding this
                          are discarded as outliers.
    """

    def __init__(
        self,
        backend: SlamBackend,
        max_velocity_m_s: float = 30.0,
    ) -> None:
        self._backend = backend
        self._max_v = max_velocity_m_s

    def is_healthy(self) -> bool:
        return self._backend.is_healthy()

    def get_measurement(self) -> SlamFrame | None:
        """Return the latest validated SLAM frame, or None if unavailable."""
        frame = self._backend.latest_frame()
        if frame is None:
            return None
        speed = (frame.v_east_m_s**2 + frame.v_north_m_s**2 + frame.v_up_m_s**2) ** 0.5
        if speed > self._max_v:
            return None  # outlier rejection
        return frame
