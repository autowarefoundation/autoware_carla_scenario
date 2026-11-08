"""A walker on CARLA's walker controller turns back once and finishes its walk (#2003).

Two faults of :mod:`autoware_carla_scenario.entity._walker_ai`, on a fake
world with a navigation mesh (no CARLA):

- A walker turned back at the kerb was handed back to its controller half a
  metre behind the kerb, before it reached the place it was turning back to,
  and its turn-back was never cleared: the controller walked it forward, it
  stepped off and turned back again -- a start and a stop of the controller
  about every tick, long after the threat had gone.
- Within ``REACHED_M`` (2 m) of its last target its top speed was set to 0,
  while its walk only counted as arrived within ``WALK_ARRIVAL_M`` (1 m) of
  the end: the walker stood 2 m short, ``walking`` stayed true and the route
  never finished.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import SpawnTransform
from autoware_carla_scenario.entity import _walker_ai, pedestrian_entity
from autoware_carla_scenario.entity.pedestrian_entity import (
    WALK_ARRIVAL_M,
    PedestrianEntity,
    PedestrianEntityConfig,
)
from autoware_carla_scenario.trajectory.model import ResolvedTrajectory

_DT_S = 0.05
_WALKER_ID = 2003
#: The road the walks cross: x from 10 m to 20 m.
_ROAD = (10.0, 20.0)


class _Snapshot:
    class timestamp:  # noqa: N801 -- carla's attribute name
        delta_seconds = _DT_S


class _Controller:
    """CARLA's walker controller: walks the walker to its target at its top
    speed while it runs."""

    is_alive = True

    def __init__(self, walker: "_Walker") -> None:
        self.walker = walker
        self.running = False
        self.target: Optional[Any] = None
        self.speed = 0.0
        self.calls: list[str] = []

    def start(self) -> None:
        self.running = True
        self.calls.append("start")

    def stop(self) -> None:
        self.running = False
        self.calls.append("stop")

    def go_to_location(self, location: Any) -> None:
        self.target = location

    def set_max_speed(self, speed: float) -> None:
        self.speed = speed

    def destroy(self) -> None:
        self.calls.append("destroy")


class _World:
    """A world with a navigation mesh, which hands out the walker's controller."""

    id = 2003

    def __init__(self, walker: "_Walker") -> None:
        self.walker = walker

    def get_random_location_from_navigation(self) -> Any:
        return carla.Location()

    def get_blueprint_library(self) -> Any:
        return self

    def find(self, name: str) -> str:
        return name

    def spawn_actor(self, blueprint: Any, transform: Any, attach_to: Any = None) -> Any:
        self.walker.controller = _Controller(self.walker)
        return self.walker.controller

    def set_pedestrians_cross_factor(self, factor: float) -> None:
        pass

    def get_snapshot(self) -> Any:
        return _Snapshot()

    def get_actors(self) -> Any:
        raise RuntimeError("no actors in a test world")


class _Walker:
    """A walker moved by its controller while it runs, else by its steps."""

    id = _WALKER_ID

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.controller: Optional[_Controller] = None
        self.world = _World(self)
        self.step_x = 0.0
        self.step_y = 0.0
        #: What it was told of its physics and its collisions, in order.
        self.physics: list[tuple[str, bool]] = []

    def get_world(self) -> _World:
        return self.world

    def get_location(self) -> Any:
        return carla.Location(x=self.x, y=self.y, z=0.0)

    def get_transform(self) -> Any:
        return carla.Transform(self.get_location(), carla.Rotation())

    def apply_control(self, control: Any) -> None:
        mps = float(control.speed) / pedestrian_entity.walker_speed_command(1.0)
        self.step_x = float(control.direction.x) * mps
        self.step_y = float(control.direction.y) * mps

    def set_simulate_physics(self, enabled: bool) -> None:
        self.physics.append(("physics", enabled))

    def set_collisions(self, enabled: bool) -> None:
        self.physics.append(("collisions", enabled))

    def tick(self) -> None:
        controller = self.controller
        if controller is not None and controller.running:
            target = controller.target
            if target is not None:
                dx, dy = target.x - self.x, target.y - self.y
                gap = math.hypot(dx, dy)
                reach = min(gap, controller.speed * _DT_S)
                if gap > 1e-6:
                    self.x += dx / gap * reach
                    self.y += dy / gap * reach
        else:
            self.x += self.step_x * _DT_S
            self.y += self.step_y * _DT_S
        self.step_x = self.step_y = 0.0


@pytest.fixture(autouse=True)
def _forget() -> Iterator[None]:
    _walker_ai.forget_all()
    _walker_ai._NAVMESH.clear()
    yield
    _walker_ai.forget_all()
    _walker_ai._NAVMESH.clear()


def _straight(length: float) -> ResolvedTrajectory:
    return ResolvedTrajectory([0.0, length], [0.0, 0.0], [0.0, 0.0], [None, None])


def _road_kind(x: float, y: float) -> str:
    return "road" if _ROAD[0] <= x <= _ROAD[1] else "walk"


class _Crossing:
    """How a walk across the road went."""

    def __init__(self, walker: _Walker) -> None:
        self.walker = walker
        #: How many times it was put on or taken off its controller.
        self.flips = 0
        #: How many times it turned back.
        self.turn_backs = 0
        #: When (s after it first stepped off) it was past the road, if ever.
        self.crossed_s: Optional[float] = None


#: A threat by the time since the walker first stepped off (s) and whether it
#: is asked of the walker standing (behind the kerb) or walking.
_ThreatOf = Callable[[float, bool], bool]


def _cross(
    monkeypatch: pytest.MonkeyPatch,
    seconds: float,
    threat_of: _ThreatOf,
    blocked_back: bool = False,
) -> _Crossing:
    """Walk a walker 30 m straight across the road for *seconds*, a vehicle
    threatening it by *threat_of*; *blocked_back*: its steps back to the
    kerb go nowhere (something stands behind it)."""
    monkeypatch.setattr(_walker_ai, "_kind_at", _road_kind)
    clock = {"t": 0.0, "off": -1.0}

    def threat(walker: Any, location: Any, turn_back: Any, standing: bool) -> str:
        off = clock["off"]
        if off < 0.0 or not threat_of(clock["t"] - off, standing):
            return ""
        return "vehicle 9 3 m away"

    monkeypatch.setattr(_walker_ai, "_threat", threat)
    path = _straight(30.0)
    walker = _Walker()
    crossing = _Crossing(walker)
    here = 0.0
    turning = False
    for tick in range(int(seconds / _DT_S)):
        clock["t"] = tick * _DT_S
        location = walker.get_location()
        here = path.project(location.x, location.y, near=here)
        driven = _walker_ai.drive(
            walker,  # type: ignore[arg-type]
            path,
            here,
            1.4,
            location,
            1.0,
            (10.0, None, 3.0),
        )
        state = _walker_ai._DRIVEN[_WALKER_ID]
        if state.stepping and clock["off"] < 0.0:
            clock["off"] = clock["t"]
        if state.turning_back and not turning:
            crossing.turn_backs += 1
        turning = state.turning_back
        if not driven:
            pedestrian_entity.steer_walker(
                walker,  # type: ignore[arg-type]
                location,
                path,
                here,
                1.4,
            )
        if blocked_back and state.turning_back:
            walker.step_x = walker.step_y = 0.0
        walker.tick()
        past = _ROAD[1] + _walker_ai.HAND_BACK_PAST_M
        if crossing.crossed_s is None and walker.x > past:
            crossing.crossed_s = clock["t"] - clock["off"]
    assert walker.controller is not None
    crossing.flips = len(walker.controller.calls)
    return crossing


def _between(start_s: float, end_s: float, standing_too: bool) -> _ThreatOf:
    """A threat from *start_s* to *end_s* after the walker first steps off,
    to the walker walking (and *standing_too*: standing)."""

    def threat_of(since_s: float, standing: bool) -> bool:
        if standing and not standing_too:
            return False
        return start_s <= since_s <= end_s

    return threat_of


def test_a_walker_turned_back_at_the_kerb_crosses_once_the_threat_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossing = _cross(monkeypatch, 40.0, _between(0.3, 10.3, standing_too=False))
    driven = _walker_ai._DRIVEN[_WALKER_ID]
    # Over the road and back on its controller on the far side, put on and
    # taken off it a bounded number of times -- not every tick.
    assert crossing.crossed_s is not None and crossing.crossed_s > 10.3
    assert not driven.stepping and driven.running
    assert crossing.flips <= 4
    assert not driven.turning_back
    assert not driven.waiting_back
    assert driven.back_to is None


def test_a_walker_turned_back_waits_behind_the_kerb_at_most_a_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A threat that never goes: it waits behind the kerb for
    # KERB_WAIT_MAX_S, then crosses all the same and does not turn back
    # again.
    seconds = _walker_ai.KERB_WAIT_MAX_S + 30.0
    crossing = _cross(monkeypatch, seconds, _between(0.3, 1e9, standing_too=True))
    driven = _walker_ai._DRIVEN[_WALKER_ID]
    assert crossing.crossed_s is not None
    assert crossing.crossed_s < _walker_ai.KERB_WAIT_MAX_S + 15.0
    assert crossing.turn_backs == 1
    assert not driven.stepping
    assert crossing.flips <= 4


def test_a_walker_threatened_only_as_it_walks_turns_back_at_most_a_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Clear as it stands behind the kerb, a threat whenever it walks (a time
    # to collision): it turns back, steps off and turns back again -- for
    # KERB_WAIT_MAX_S in all, not each wait, then it crosses.
    seconds = _walker_ai.KERB_WAIT_MAX_S + 30.0
    crossing = _cross(monkeypatch, seconds, _between(0.3, 1e9, standing_too=False))
    driven = _walker_ai._DRIVEN[_WALKER_ID]
    assert crossing.turn_backs > 1
    assert crossing.crossed_s is not None
    # No sooner than KERB_WAIT_MAX_S after its first turn back (0.3 s after
    # it stepped off; a tick's tolerance): each tick counted once.
    assert crossing.crossed_s >= 0.3 + _walker_ai.KERB_WAIT_MAX_S - _DT_S
    assert crossing.crossed_s < _walker_ai.KERB_WAIT_MAX_S + 15.0
    assert not driven.stepping
    assert crossing.flips <= 4


def test_a_walker_still_waiting_behind_the_kerb_has_not_crossed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seconds = _walker_ai.KERB_WAIT_MAX_S - 20.0
    crossing = _cross(monkeypatch, seconds, _between(0.3, 1e9, standing_too=True))
    driven = _walker_ai._DRIVEN[_WALKER_ID]
    assert driven.waiting_back and driven.stepping
    assert crossing.walker.x < _ROAD[0]
    assert crossing.flips == 2  # started once, stopped once at the kerb


def test_a_walker_whose_steps_back_are_blocked_waits_where_it_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Something stands behind it: it gets no nearer the kerb, so after
    # BACK_STALL_S it waits where it stands, and crosses once the threat
    # has gone.
    lasting = _between(0.3, 1e9, standing_too=True)
    stalled = _cross(monkeypatch, 15.0, lasting, blocked_back=True)
    driven = _walker_ai._DRIVEN[_WALKER_ID]
    assert driven.waiting_back and not driven.turning_back
    assert stalled.walker.x > _ROAD[0] - 0.5  # not back behind the kerb
    _walker_ai.forget_all()
    passing = _between(0.3, 3.0, standing_too=True)
    crossing = _cross(monkeypatch, 40.0, passing, blocked_back=True)
    assert crossing.crossed_s is not None
    assert not _walker_ai._DRIVEN[_WALKER_ID].stepping


def _walk(entity: PedestrianEntity, walker: _Walker, seconds: float) -> Optional[float]:
    for tick in range(int(seconds / _DT_S)):
        entity.walk_on()
        walker.tick()
        if not entity.walking:
            return tick * _DT_S
    return None


def _entity(walker: _Walker) -> PedestrianEntity:
    entity = PedestrianEntity(
        PedestrianEntityConfig("ped", SpawnTransform(carla.Transform()))
    )
    entity._walker = walker  # type: ignore[assignment]
    return entity


def test_a_walk_on_the_controller_finishes_where_it_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_walker_ai, "_kind_at", lambda x, y: "walk")
    walker = _Walker()
    entity = _entity(walker)
    entity.walk_along(_straight(30.0), 1.4)
    arrived = _walk(entity, walker, 60.0)
    assert arrived is not None
    # Held 2 m short of the end, as near as the controller takes it: short of
    # WALK_ARRIVAL_M, and arrived all the same.
    assert walker.x < 30.0 - WALK_ARRIVAL_M
    assert walker.x >= 30.0 - _walker_ai.REACHED_M - 0.1
    assert walker.controller is not None
    assert walker.controller.calls[-2:] == ["stop", "destroy"]
    assert _WALKER_ID not in _walker_ai._DRIVEN
    assert walker.step_x == 0.0 and walker.step_y == 0.0
    # Its physics and collisions, which its controller set off, on again.
    assert walker.physics[-2:] == [("physics", True), ("collisions", True)]


def test_a_walk_whose_end_was_moved_off_the_road_finishes_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Its end on the road, which its cross factor keeps it off: sent to the
    # nearest walkway instead, 5 m back and 4 m aside, and held there.
    monkeypatch.setattr(
        _walker_ai, "_kind_at", lambda x, y: "road" if x > 27.0 else "walk"
    )
    monkeypatch.setattr(_walker_ai, "_nearest_walk", lambda x, y: (25.0, 4.0))
    walker = _Walker()
    entity = _entity(walker)
    entity.walk_along(_straight(30.0), 1.4)
    arrived = _walk(entity, walker, 60.0)
    assert arrived is not None
    assert math.hypot(walker.x - 25.0, walker.y - 4.0) <= _walker_ai.REACHED_M
    assert walker.controller is not None
    assert walker.controller.calls[-2:] == ["stop", "destroy"]
    assert _WALKER_ID not in _walker_ai._DRIVEN


def test_a_walker_held_by_its_timing_short_of_its_end_is_still_walking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Held mid-walk (a top speed of 0) is not arriving: only at its last
    # target.
    monkeypatch.setattr(_walker_ai, "_kind_at", lambda x, y: "walk")
    walker = _Walker()
    entity = _entity(walker)
    entity.walk_along(_straight(30.0), 1.4)
    for _ in range(int(5.0 / _DT_S)):
        entity.walk_on(0.0)
        walker.tick()
    assert entity.walking
    assert _WALKER_ID in _walker_ai._DRIVEN
    assert walker.x < 1.0
