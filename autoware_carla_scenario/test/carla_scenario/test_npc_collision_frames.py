"""A contact's start frame is its earliest event's, whatever order the two
actors' sensors deliver them in."""

from __future__ import annotations

from types import SimpleNamespace as NS

from autoware_carla_scenario.npc_collisions import REPEAT_S, NpcCollisionMonitor


def _event(frame: int, other_id: int, role: str) -> NS:
    other = NS(id=other_id, type_id="vehicle.x", attributes={"role_name": role})
    return NS(frame=frame, other_actor=other, normal_impulse=NS(x=3.0, y=0.0, z=0.0))


def _monitor() -> NpcCollisionMonitor:
    monitor = NpcCollisionMonitor()
    monitor._frame = 12
    monitor._names = {1: "a", 2: "b"}
    return monitor


def test_a_later_frame_heard_first_does_not_start_the_contact() -> None:
    monitor = _monitor()
    monitor._on_event(1, _event(12, 2, "b"))
    monitor._on_event(2, _event(11, 1, "a"))
    (start,) = monitor._collisions(0.6)
    assert start.phase == "start"
    assert start.start_frame == 11


def test_an_earlier_frame_heard_a_tick_late_moves_the_contact_back() -> None:
    monitor = _monitor()
    monitor._on_event(1, _event(12, 2, "b"))
    assert [r.start_frame for r in monitor._collisions(0.6)] == [12]
    monitor._on_event(2, _event(11, 1, "a"))
    assert monitor._collisions(0.65) == []
    monitor._frame = 40
    (end,) = monitor._collisions(0.65 + REPEAT_S + 0.1)
    assert end.phase == "end"
    assert end.start_frame == 11
