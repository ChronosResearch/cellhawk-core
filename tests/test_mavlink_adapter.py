"""Tests for the PX4/MAVLink adapter (Gap 3).

Verifies:
- All 9 DQN actions produce correct NED velocity setpoints
- IntentDispatcher tracks dispatch/error counts
- SimulatedMavlinkAdapter records commands faithfully
- Px4Adapter type-mask constant is correct
- MavlinkAdapter protocol is satisfied by SimulatedMavlinkAdapter
"""
from __future__ import annotations

import math
import pytest

from python.mavlink.adapter import (
    VelocitySetpoint,
    action_to_setpoint,
    SimulatedMavlinkAdapter,
    IntentDispatcher,
    Px4Adapter,
    MavlinkAdapter,
    _CRUISE_SPEED_M_S,
    _VERTICAL_SPEED_M_S,
    _EVADE_SPEED_M_S,
)


# ─────────────────────────────────────────────────────────────────────────────
# action_to_setpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestActionToSetpoint:
    def test_hover_gives_zero_velocity(self) -> None:
        sp = action_to_setpoint(0, heading_rad=0.0)
        assert sp.vx_north_m_s == 0.0
        assert sp.vy_east_m_s  == 0.0
        assert sp.vz_down_m_s  == 0.0

    def test_north_action_gives_positive_vx(self) -> None:
        sp = action_to_setpoint(1, heading_rad=0.0)
        assert sp.vx_north_m_s == _CRUISE_SPEED_M_S
        assert sp.vy_east_m_s  == 0.0

    def test_east_action_gives_positive_vy(self) -> None:
        sp = action_to_setpoint(2, heading_rad=0.0)
        assert sp.vy_east_m_s  == _CRUISE_SPEED_M_S
        assert sp.vx_north_m_s == 0.0

    def test_south_action_gives_negative_vx(self) -> None:
        sp = action_to_setpoint(3, heading_rad=0.0)
        assert sp.vx_north_m_s == -_CRUISE_SPEED_M_S

    def test_west_action_gives_negative_vy(self) -> None:
        sp = action_to_setpoint(4, heading_rad=0.0)
        assert sp.vy_east_m_s == -_CRUISE_SPEED_M_S

    def test_climb_gives_negative_vz(self) -> None:
        # NED: negative vz = upward
        sp = action_to_setpoint(5, heading_rad=0.0)
        assert sp.vz_down_m_s == -_VERTICAL_SPEED_M_S

    def test_descend_gives_positive_vz(self) -> None:
        sp = action_to_setpoint(6, heading_rad=0.0)
        assert sp.vz_down_m_s == _VERTICAL_SPEED_M_S

    def test_evade_left_is_perpendicular_to_heading(self) -> None:
        heading = math.pi / 4  # 45° NE
        sp = action_to_setpoint(7, heading_rad=heading)
        # Evade-left = heading + 90°
        expected_heading = heading + math.pi / 2
        expected_vx = _EVADE_SPEED_M_S * math.cos(expected_heading)
        expected_vy = _EVADE_SPEED_M_S * math.sin(expected_heading)
        assert abs(sp.vx_north_m_s - expected_vx) < 1e-9
        assert abs(sp.vy_east_m_s  - expected_vy) < 1e-9

    def test_evade_right_is_perpendicular_to_heading(self) -> None:
        heading = math.pi / 4
        sp = action_to_setpoint(8, heading_rad=heading)
        expected_heading = heading - math.pi / 2
        expected_vx = _EVADE_SPEED_M_S * math.cos(expected_heading)
        expected_vy = _EVADE_SPEED_M_S * math.sin(expected_heading)
        assert abs(sp.vx_north_m_s - expected_vx) < 1e-9
        assert abs(sp.vy_east_m_s  - expected_vy) < 1e-9

    def test_invalid_action_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            action_to_setpoint(9, heading_rad=0.0)
        with pytest.raises(ValueError):
            action_to_setpoint(-1, heading_rad=0.0)

    def test_all_9_actions_produce_setpoints(self) -> None:
        for action in range(9):
            sp = action_to_setpoint(action, heading_rad=0.0)
            assert isinstance(sp, VelocitySetpoint)

    def test_evade_speed_is_less_than_cruise(self) -> None:
        """Evasion is slower than cruise to allow tighter turns."""
        assert _EVADE_SPEED_M_S < _CRUISE_SPEED_M_S


# ─────────────────────────────────────────────────────────────────────────────
# VelocitySetpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestVelocitySetpoint:
    def test_speed_property_is_horizontal_magnitude(self) -> None:
        sp = VelocitySetpoint(vx_north_m_s=3.0, vy_east_m_s=4.0, vz_down_m_s=10.0)
        assert abs(sp.speed_m_s - 5.0) < 1e-9

    def test_hover_speed_is_zero(self) -> None:
        sp = VelocitySetpoint()
        assert sp.speed_m_s == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SimulatedMavlinkAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulatedMavlinkAdapter:
    def test_satisfies_protocol(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        assert isinstance(adapter, MavlinkAdapter)

    def test_records_sent_commands(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        sp = VelocitySetpoint(vx_north_m_s=5.0)
        adapter.send_velocity_setpoint(sp)
        assert len(adapter.commands) == 1
        assert adapter.last_command == sp

    def test_arm_disarm(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        assert not adapter.is_armed
        assert adapter.arm()
        assert adapter.is_armed
        assert adapter.disarm()
        assert not adapter.is_armed

    def test_connected_by_default(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        assert adapter.is_connected()

    def test_set_connected_false(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        adapter.set_connected(False)
        assert not adapter.is_connected()

    def test_commands_returns_copy(self) -> None:
        adapter = SimulatedMavlinkAdapter()
        adapter.send_velocity_setpoint(VelocitySetpoint())
        cmds = adapter.commands
        cmds.clear()
        assert len(adapter.commands) == 1  # original unaffected


# ─────────────────────────────────────────────────────────────────────────────
# IntentDispatcher
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentDispatcher:
    def _dispatcher(self) -> tuple[IntentDispatcher, SimulatedMavlinkAdapter]:
        adapter = SimulatedMavlinkAdapter()
        return IntentDispatcher(adapter), adapter

    def test_dispatch_sends_command(self) -> None:
        dispatcher, adapter = self._dispatcher()
        ok = dispatcher.dispatch(action=1, heading_rad=0.0)
        assert ok
        assert len(adapter.commands) == 1

    def test_dispatch_count_increments(self) -> None:
        dispatcher, _ = self._dispatcher()
        for _ in range(5):
            dispatcher.dispatch(0, 0.0)
        assert dispatcher.dispatch_count == 5

    def test_error_count_when_disconnected(self) -> None:
        dispatcher, adapter = self._dispatcher()
        adapter.set_connected(False)
        ok = dispatcher.dispatch(0, 0.0)
        assert not ok
        assert dispatcher.error_count == 1
        assert dispatcher.dispatch_count == 0

    def test_last_action_tracked(self) -> None:
        dispatcher, _ = self._dispatcher()
        dispatcher.dispatch(3, 0.0)
        assert dispatcher.last_action == 3

    def test_all_actions_dispatch_successfully(self) -> None:
        dispatcher, adapter = self._dispatcher()
        for action in range(9):
            assert dispatcher.dispatch(action, heading_rad=0.0)
        assert len(adapter.commands) == 9


# ─────────────────────────────────────────────────────────────────────────────
# Px4Adapter — static properties (no real connection needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPx4AdapterStatic:
    def test_type_mask_uses_velocity_only(self) -> None:
        mask = Px4Adapter._TYPE_MASK_VELOCITY_ONLY
        # Position bits (0,1,2) must be 1 → ignore position
        assert mask & 0b111 == 0b111, f"position bits must be 1 (ignore), got {mask & 7}"
        # Velocity bits (3,4,5) must be 0 → use velocity
        assert (mask >> 3) & 0b111 == 0, f"velocity bits must be 0 (use), got {(mask>>3)&7}"

    def test_not_connected_before_connect(self) -> None:
        adapter = Px4Adapter()
        assert not adapter.is_connected()
