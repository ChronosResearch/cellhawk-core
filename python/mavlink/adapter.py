"""PX4/MAVLink adapter: translates CORTEX DQN intent actions into MAVLink
commands sent to a PX4 flight controller (§4.2).

## Architecture

The DQN operates at 10 Hz and outputs one of 9 discrete navigational intent
actions.  This adapter translates each intent into a MAVLink
SET_POSITION_TARGET_LOCAL_NED message (MAV_FRAME_LOCAL_NED, type-mask
selects velocity control) and sends it to the flight controller.

The PID attitude controller on PX4 runs at 400 Hz and handles the
low-level motor mixing — the adapter only needs to send 10 Hz velocity
setpoints.

## Action mapping (§4.2)

    0 = hover          → zero velocity setpoint
    1 = north          → +v_north
    2 = east           → +v_east
    3 = south          → -v_north
    4 = west           → -v_east
    5 = climb          → +v_up
    6 = descend        → -v_up
    7 = evade-left     → lateral velocity perpendicular to heading (CCW)
    8 = evade-right    → lateral velocity perpendicular to heading (CW)

## Connection

In production: UDP connection to PX4 SITL or real hardware via
`udp:127.0.0.1:14550` (GCS port) or `serial:/dev/ttyUSB0:57600`.

For tests: `SimulatedMavlinkAdapter` records all sent commands without
requiring a flight controller.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ─────────────────────────────────────────────────────────────────────────────
# Velocity setpoint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VelocitySetpoint:
    """NED velocity setpoint sent to PX4 via MAVLink.

    All values in m/s.  NED frame: +x = North, +y = East, +z = Down.
    """
    vx_north_m_s: float = 0.0
    vy_east_m_s:  float = 0.0
    vz_down_m_s:  float = 0.0   # positive = descend
    yaw_rate_rad_s: float = 0.0
    timestamp_s:  float = field(default_factory=time.monotonic)

    @property
    def speed_m_s(self) -> float:
        return math.hypot(self.vx_north_m_s, self.vy_east_m_s)


# ─────────────────────────────────────────────────────────────────────────────
# MavlinkAdapter protocol
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class MavlinkAdapter(Protocol):
    """Protocol any MAVLink backend must satisfy."""

    def send_velocity_setpoint(self, sp: VelocitySetpoint) -> None:
        """Send a velocity setpoint to the flight controller."""
        ...

    def is_connected(self) -> bool:
        """Whether the MAVLink link is active."""
        ...

    def arm(self) -> bool:
        """Arm the vehicle.  Returns True on success."""
        ...

    def disarm(self) -> bool:
        """Disarm the vehicle.  Returns True on success."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Action → velocity translation
# ─────────────────────────────────────────────────────────────────────────────

# Cruise speed used for cardinal direction actions (m/s)
_CRUISE_SPEED_M_S = 10.0
# Vertical speed for climb/descend actions (m/s)
_VERTICAL_SPEED_M_S = 2.0
# Evade lateral speed (m/s)
_EVADE_SPEED_M_S = 8.0


def action_to_setpoint(
    action: int,
    heading_rad: float,
    cruise_speed_m_s: float = _CRUISE_SPEED_M_S,
    vertical_speed_m_s: float = _VERTICAL_SPEED_M_S,
    evade_speed_m_s: float = _EVADE_SPEED_M_S,
) -> VelocitySetpoint:
    """Translate a DQN action index into a NED velocity setpoint.

    Args:
        action:           DQN action index [0, 8].
        heading_rad:      Current drone heading (radians, 0 = North).
        cruise_speed_m_s: Speed for cardinal direction actions.
        vertical_speed_m_s: Speed for climb/descend actions.
        evade_speed_m_s:  Speed for evasion actions.

    Returns:
        VelocitySetpoint in NED frame.

    Raises:
        ValueError: If action is outside [0, 8].
    """
    if not 0 <= action <= 8:
        raise ValueError(f"action must be in [0, 8], got {action}")

    v = cruise_speed_m_s
    h = heading_rad

    # Perpendicular heading for evasion (±90° from current heading)
    evade_left_heading  = h + math.pi / 2.0
    evade_right_heading = h - math.pi / 2.0

    match action:
        case 0:  # hover
            return VelocitySetpoint(0.0, 0.0, 0.0)
        case 1:  # north
            return VelocitySetpoint(v, 0.0, 0.0)
        case 2:  # east
            return VelocitySetpoint(0.0, v, 0.0)
        case 3:  # south
            return VelocitySetpoint(-v, 0.0, 0.0)
        case 4:  # west
            return VelocitySetpoint(0.0, -v, 0.0)
        case 5:  # climb (NED: negative vz = up)
            return VelocitySetpoint(0.0, 0.0, -vertical_speed_m_s)
        case 6:  # descend
            return VelocitySetpoint(0.0, 0.0, vertical_speed_m_s)
        case 7:  # evade-left (perpendicular CCW)
            return VelocitySetpoint(
                evade_speed_m_s * math.cos(evade_left_heading),
                evade_speed_m_s * math.sin(evade_left_heading),
                0.0,
            )
        case 8:  # evade-right (perpendicular CW)
            return VelocitySetpoint(
                evade_speed_m_s * math.cos(evade_right_heading),
                evade_speed_m_s * math.sin(evade_right_heading),
                0.0,
            )
        case _:
            raise ValueError(f"unreachable action {action}")


# ─────────────────────────────────────────────────────────────────────────────
# SimulatedMavlinkAdapter — for tests and SITL without a real connection
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedMavlinkAdapter:
    """Records all sent commands; no real MAVLink connection required.

    Used in unit tests and CI.  Thread-safe for single-threaded test use.
    """

    def __init__(self) -> None:
        self._commands: list[VelocitySetpoint] = []
        self._armed = False
        self._connected = True

    def send_velocity_setpoint(self, sp: VelocitySetpoint) -> None:
        self._commands.append(sp)

    def is_connected(self) -> bool:
        return self._connected

    def arm(self) -> bool:
        self._armed = True
        return True

    def disarm(self) -> bool:
        self._armed = False
        return True

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def commands(self) -> list[VelocitySetpoint]:
        return list(self._commands)

    @property
    def last_command(self) -> VelocitySetpoint | None:
        return self._commands[-1] if self._commands else None

    def set_connected(self, connected: bool) -> None:
        self._connected = connected


# ─────────────────────────────────────────────────────────────────────────────
# Px4Adapter — real PX4 connection via pymavlink
# ─────────────────────────────────────────────────────────────────────────────

class Px4Adapter:
    """Production PX4 adapter using pymavlink.

    Connects to PX4 via UDP (SITL) or serial (hardware).

    Args:
        connection_string: pymavlink connection string, e.g.
            ``"udp:127.0.0.1:14550"`` for SITL or
            ``"serial:/dev/ttyUSB0:57600"`` for hardware.
        source_system:     MAVLink system ID for this GCS (default 255).
        target_system:     MAVLink system ID of the vehicle (default 1).
        target_component:  MAVLink component ID (default 1 = autopilot).
        connect_timeout_s: Seconds to wait for heartbeat on connect.
    """

    # MAVLink type-mask: ignore position + acceleration, use velocity + yaw_rate
    # Bit layout (SET_POSITION_TARGET_LOCAL_NED):
    #   bits 0-2  (pos x,y,z)   = 1 → ignore
    #   bits 3-5  (vel x,y,z)   = 0 → use
    #   bits 6-8  (acc x,y,z)   = 1 → ignore
    #   bit  10   (yaw)         = 1 → ignore
    #   bit  11   (yaw_rate)    = 0 → use
    # Result: 0b010111000111 = 0x5C7
    _TYPE_MASK_VELOCITY_ONLY = 0b010111000111  # 0x5C7

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        source_system: int = 255,
        target_system: int = 1,
        target_component: int = 1,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self._conn_str = connection_string
        self._src_sys = source_system
        self._tgt_sys = target_system
        self._tgt_comp = target_component
        self._timeout = connect_timeout_s
        self._mav: object | None = None
        self._connected = False

    def connect(self) -> bool:
        """Establish MAVLink connection and wait for heartbeat.

        Returns True if a heartbeat was received within the timeout.
        """
        try:
            from pymavlink import mavutil  # type: ignore[import]
            self._mav = mavutil.mavlink_connection(
                self._conn_str,
                source_system=self._src_sys,
            )
            # Wait for heartbeat to confirm vehicle is alive
            msg = self._mav.wait_heartbeat(timeout=self._timeout)  # type: ignore[union-attr]
            self._connected = msg is not None
            return self._connected
        except Exception:
            self._connected = False
            return False

    def send_velocity_setpoint(self, sp: VelocitySetpoint) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED with velocity-only type mask."""
        if self._mav is None:
            raise RuntimeError("Px4Adapter not connected — call connect() first")
        self._mav.mav.set_position_target_local_ned_send(  # type: ignore[union-attr]
            int(sp.timestamp_s * 1000) & 0xFFFFFFFF,  # time_boot_ms
            self._tgt_sys,
            self._tgt_comp,
            1,   # MAV_FRAME_LOCAL_NED
            self._TYPE_MASK_VELOCITY_ONLY,
            0.0, 0.0, 0.0,                    # position (ignored)
            sp.vx_north_m_s,                  # vx (North)
            sp.vy_east_m_s,                   # vy (East)
            sp.vz_down_m_s,                   # vz (Down, positive = descend)
            0.0, 0.0, 0.0,                    # acceleration (ignored)
            0.0,                              # yaw (ignored)
            sp.yaw_rate_rad_s,                # yaw_rate
        )

    def is_connected(self) -> bool:
        return self._connected

    def arm(self) -> bool:
        """Send MAV_CMD_COMPONENT_ARM_DISARM (arm=1)."""
        if self._mav is None:
            return False
        try:
            self._mav.arducopter_arm()  # type: ignore[union-attr]
            self._mav.motors_armed_wait()  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def disarm(self) -> bool:
        """Send MAV_CMD_COMPONENT_ARM_DISARM (arm=0)."""
        if self._mav is None:
            return False
        try:
            self._mav.arducopter_disarm()  # type: ignore[union-attr]
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# IntentDispatcher — 10 Hz DQN → MAVLink bridge
# ─────────────────────────────────────────────────────────────────────────────

class IntentDispatcher:
    """Bridges the 10 Hz DQN action loop to the MAVLink adapter.

    Translates each DQN action into a velocity setpoint and dispatches it.
    Tracks dispatch statistics for health monitoring.

    Args:
        adapter:           Any MavlinkAdapter implementation.
        cruise_speed_m_s:  Cruise speed for cardinal actions (m/s).
    """

    def __init__(
        self,
        adapter: MavlinkAdapter,
        cruise_speed_m_s: float = _CRUISE_SPEED_M_S,
    ) -> None:
        self._adapter = adapter
        self._cruise = cruise_speed_m_s
        self._dispatch_count = 0
        self._error_count = 0
        self._last_action: int | None = None

    def dispatch(self, action: int, heading_rad: float) -> bool:
        """Translate action and send to flight controller.

        Args:
            action:      DQN action index [0, 8].
            heading_rad: Current drone heading (radians).

        Returns:
            True if the command was sent successfully.
        """
        if not self._adapter.is_connected():
            self._error_count += 1
            return False
        try:
            sp = action_to_setpoint(action, heading_rad, self._cruise)
            self._adapter.send_velocity_setpoint(sp)
            self._dispatch_count += 1
            self._last_action = action
            return True
        except Exception:
            self._error_count += 1
            return False

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def last_action(self) -> int | None:
        return self._last_action
