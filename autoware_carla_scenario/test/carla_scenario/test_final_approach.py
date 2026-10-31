"""A vehicle whose given path ends before a dead end drives its last metres
itself (traffic.final_approach), instead of being held short of its goal."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.traffic.final_approach import FinalApproach

from .test_routing_trajectory_action import _DT
from .test_routing_trajectory_action import _Actor as _Vehicle
from .test_traffic_backend import _Actor, _on_autopilot, _road, _stand


def test_it_drives_the_path_to_its_end_and_stops_there() -> None:
    vehicle = _Vehicle(speed=6.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path)  # type: ignore[arg-type]
    assert approach.length == pytest.approx(20.0)
    elapsed = 0.0
    while elapsed < 30.0:
        approach.step(_DT, elapsed)
        vehicle.step(_DT, driven=True)
        elapsed += _DT
    assert approach.done
    assert vehicle.x == pytest.approx(20.0, abs=2.0)
    assert vehicle.speed < 0.5


class _Approach:
    """Stands in for FinalApproach: counts its steps."""

    made: List["_Approach"] = []

    def __init__(self, actor: Any, start: Any, points: List[Any]) -> None:
        self.points = points
        self.steps = 0
        self.length = 10.0
        _Approach.made.append(self)

    def step(self, dt: float, elapsed: float) -> None:
        self.steps += 1


def test_a_vehicle_on_a_path_ending_before_a_dead_end_drives_on_to_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    ego = _Actor(1)
    ego.attributes = {"role_name": "Ego"}
    ego.location = carla.Location(0.0, 0.0, 0.0)
    backend, _ = _on_autopilot(ego)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15)]
    backend._remember_path(ego, path)  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO):
        _stand(backend, 5.0)
    # Off autopilot (so never destroyed), not braked: driven by the approach.
    assert ego.autopilot == [(True, 8123), (False, 8123)]
    assert ego.controls == []
    (approach,) = _Approach.made
    assert approach.points == path[1:]
    assert approach.steps > 0
    said = [r for r in caplog.records if "drives the last" in r.getMessage()]
    assert len(said) == 1 and said[0].getMessage().startswith("Ego:")


def test_a_roaming_vehicle_is_still_held(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    npc = _Actor(2)
    backend, _ = _on_autopilot(npc)
    _stand(backend, 5.0)
    assert _Approach.made == []
    assert [(c.throttle, c.brake) for c in npc.controls] == [(0.0, 1.0)]


def test_a_path_given_before_the_vehicle_is_kept_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ego is given its route at its init, before the backend starts and
    keeps it: that path is the one its final approach drives."""
    import autoware_carla_scenario.traffic.traffic_manager as tm_module
    from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
    from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend

    from .test_traffic_backend import _Client, _World

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _Approach)
    _road(monkeypatch, 20.0, dead=True)
    ego = _Actor(1)
    ego.location = carla.Location(0.0, 0.0, 0.0)
    backend = TrafficManagerBackend(
        TrafficManagerBackendConfig(port=8123), client=_Client()
    )
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15)]
    backend._remember_path(ego, path)  # type: ignore[arg-type]
    backend.start(_World([ego]))
    _stand(backend, 5.0)
    (approach,) = _Approach.made
    assert approach.points == path[1:]
    assert ego.controls == []


def test_a_lead_out_does_not_stop_at_its_end() -> None:
    """Driving a vehicle out of a junction it hands it on at speed."""
    vehicle = _Vehicle(speed=4.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    elapsed = 0.0
    while not approach.done and elapsed < 30.0:
        approach.step(_DT, elapsed)
        vehicle.step(_DT, driven=True)
        elapsed += _DT
    assert approach.done
    assert vehicle.x == pytest.approx(19.0, abs=2.0)
    assert vehicle.speed > 2.0


class _DoneApproach(_Approach):
    def __init__(
        self,
        actor: Any,
        start: Any,
        points: List[Any],
        stop: bool = True,
        speed: float = 4.0,
    ) -> None:
        super().__init__(actor, start, points)
        self.stop = stop
        self.speed = speed
        self.done = False

    def step(self, dt: float, elapsed: float) -> None:
        super().step(dt, elapsed)
        self.done = self.steps >= 3


def test_a_vehicle_led_out_of_its_junction_is_handed_back_with_its_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off autopilot while it is driven out of the junction's mouth, then on
    autopilot again with the rest of its path."""
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    _Approach.made = []
    monkeypatch.setattr(tm_module, "FinalApproach", _DoneApproach)
    _road(monkeypatch, 500.0, dead=False)
    npc = _Actor(4)
    npc.attributes = {"role_name": "npc4"}
    npc.location = carla.Location(0.0, 0.0, 0.0)
    npc.get_world = lambda: None  # type: ignore[attr-defined]
    backend, client = _on_autopilot(npc)
    # The junction spans x 8-12: the route's point at 10 is in it.
    from .test_traffic_backend import _Waypoint

    def waypoint(location: Any) -> Any:
        found = _Waypoint(500.0, False)
        found.is_junction = 8.0 <= location.x <= 12.0  # type: ignore[attr-defined]
        return found

    junction = SimpleNamespace(get_waypoint=waypoint)
    monkeypatch.setattr(tm_module, "run_carla_map", lambda world: junction)
    route = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20, 25)]
    path = route[2:]
    backend._remember_path(npc, path)  # type: ignore[arg-type]
    backend._lead_out(npc, carla.Location(0.0, 0.0, 0.0), route, path)  # type: ignore[arg-type]
    (approach,) = _Approach.made
    assert approach.points == route[:3] and not approach.stop  # type: ignore[attr-defined]
    assert npc.autopilot[-1] == (False, 8123)
    npc.location = carla.Location(15.0, 0.0, 0.0)
    _stand(backend, 2.0)
    assert npc.autopilot[-1] == (True, 8123)
    assert backend._kept[4].approach is None
    assert client.tm.paths[-1] == (4, route[3:])


def test_a_vehicle_driven_out_of_its_junction_waits_for_one_in_its_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It brakes for a vehicle in its way; after LEAD_OUT_WAIT_S it is handed
    on all the same.  A final approach waits as long as it takes."""
    from autoware_carla_scenario.traffic import final_approach as fa

    monkeypatch.setattr(
        fa.FinalApproach, "_in_way", lambda self, kind, reach, hw: kind == "vehicle"
    )
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    for stop in (False, True):
        vehicle = _Vehicle(speed=0.0)
        approach = FinalApproach(
            vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=stop
        )  # type: ignore[arg-type]
        elapsed = 0.0
        while not approach.done and elapsed < 2 * fa.LEAD_OUT_WAIT_S:
            approach.step(_DT, elapsed)
            vehicle.step(_DT, driven=True)
            elapsed += _DT
        assert vehicle.x == pytest.approx(0.0, abs=0.5)
        if stop:
            assert not approach.done
        else:
            assert approach.done
            assert elapsed == pytest.approx(fa.LEAD_OUT_WAIT_S, abs=2 * _DT)


def test_a_vehicle_led_out_watches_the_path_it_turns_along(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning right out of its junction, it does not stop for a walker on the
    corner straight ahead of it, off its path (vehicle_56 waited 30 s on
    pedestrian_66, which waited on the ego, which waited on it); it does for
    one on its path, round the turn."""
    import math

    from autoware_carla_scenario.utils import clearance

    walkers: List[tuple[float, float]] = []

    def first_in_way(world, self_id, x, y, heading_deg, kind, distance, half_width):
        if kind != "walker":
            return None
        c, s_ = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
        for wx, wy in walkers:
            ahead = (wx - x) * c + (wy - y) * s_
            side = -(wx - x) * s_ + (wy - y) * c
            if 0.0 <= ahead <= distance and abs(side) <= half_width + 0.3:
                return SimpleNamespace(id=7, x=wx, y=wy, ahead=ahead, side=side)
        return None

    monkeypatch.setattr(clearance, "first_box_in_way", first_in_way)
    # Heading east (+x), then turning right: in CARLA, towards +y.
    path = [carla.Location(2.0, 0.0, 0.0), carla.Location(4.0, 1.0, 0.0)]
    path += [carla.Location(5.0, 3.0 + 2.0 * k, 0.0) for k in range(4)]
    vehicle = _Vehicle(speed=1.0)
    vehicle.get_world = lambda: None  # type: ignore[attr-defined]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    # Straight on from where it starts, 6 m ahead: the corner, off its path.
    walkers.append((6.0, -0.5))
    assert not approach._in_way("walker", 8.0, 1.5)
    # Round the turn, on its path.
    walkers.append((5.0, 4.0))
    assert approach._in_way("walker", 8.0, 1.5)


def test_one_behind_the_vehicle_is_not_in_its_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path whose first point lies behind where the vehicle stands (its
    route's next point, short of it) looks back along its first piece: the
    vehicle behind (vehicle_56, behind vehicle_58, each waiting on the other)
    is not in its way."""
    from autoware_carla_scenario.utils import clearance

    # Its centre 2 m behind the vehicle's: corners near, still behind it.
    behind_it = SimpleNamespace(id=9, x=-2.0, y=-1.5)

    def first_in_way(world, self_id, x, y, heading_deg, kind, distance, half_width):
        return behind_it if kind == "vehicle" else None

    monkeypatch.setattr(clearance, "first_box_in_way", first_in_way)
    vehicle = _Vehicle(speed=0.0)  # at the origin, facing +x
    vehicle.get_world = lambda: None  # type: ignore[attr-defined]
    path = [carla.Location(-1.0, 0.0, 0.0), carla.Location(10.0, 0.0, 0.0)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    assert not approach._in_way("vehicle", 8.0, 1.3)
    behind_it.x = 3.0  # ahead of it after all
    assert approach._in_way("vehicle", 8.0, 1.3)


def test_a_vehicle_led_out_keeps_less_room_for_a_walker_than_one_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driving out of its junction at a crawl, it keeps
    LEAD_OUT_WALKER_CLEARANCE_M to each side for a walker, not
    PASSING_CLEARANCE_M."""
    from autoware_carla_scenario.actions._road_threat import PASSING_CLEARANCE_M
    from autoware_carla_scenario.traffic import final_approach as fa

    asked: List[tuple[str, float]] = []

    def in_way(self, kind, reach, half_width):
        asked.append((kind, half_width))
        return False

    monkeypatch.setattr(fa.FinalApproach, "_in_way", in_way)
    vehicle = _Vehicle(speed=1.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    approach.step(_DT, 0.0)
    walker = dict(asked)["walker"]
    assert walker == pytest.approx(1.0 + fa.LEAD_OUT_WALKER_CLEARANCE_M)
    assert fa.LEAD_OUT_WALKER_CLEARANCE_M < PASSING_CLEARANCE_M


def test_a_walker_walking_off_the_way_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A walker beside the path, off the vehicle's body, walking away from
    it (pedestrian_66, off the corner vehicle_56 turned round) is leaving the
    way; standing there, or walking towards the path, it is in it."""
    from autoware_carla_scenario.utils import clearance

    walker = SimpleNamespace(id=5, x=3.0, y=1.4, ahead=3.0, side=1.4)
    velocity = SimpleNamespace(x=0.0, y=1.2, z=0.0)  # +y: away from the path
    world = SimpleNamespace(
        get_actor=lambda actor_id: SimpleNamespace(get_velocity=lambda: velocity)
    )
    monkeypatch.setattr(
        clearance,
        "first_box_in_way",
        lambda *args: walker if args[5] == "walker" else None,
    )
    vehicle = _Vehicle(speed=1.0)  # half width 1.0
    vehicle.get_world = lambda: world  # type: ignore[attr-defined]
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    assert not approach._in_way("walker", 6.0, 1.5)
    velocity.y = 0.0  # standing
    assert approach._in_way("walker", 6.0, 1.5)
    velocity.y = -1.2  # towards the path
    assert approach._in_way("walker", 6.0, 1.5)
    walker.side = walker.y = 0.6  # on the body's width: in the way however it walks
    velocity.y = 1.2
    assert approach._in_way("walker", 6.0, 1.5)


class _Clock:
    """A world whose simulated clock the test sets, holding one walker."""

    def __init__(self, walker: Any) -> None:
        self.elapsed = 10.0
        self.walker = walker

    def get_snapshot(self) -> Any:
        return SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=self.elapsed))

    def get_actor(self, actor_id: int) -> Any:
        return self.walker


def _ai_walker(y: float) -> Any:
    """A walker on CARLA's controller at ``(3, y)``: it reports standing,
    whether or not it walks."""
    walker = SimpleNamespace(
        id=66,
        type_id="walker.pedestrian.0001",
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        at=SimpleNamespace(x=3.0, y=y, z=0.0),
    )
    walker.get_location = lambda: walker.at
    return walker


def _ahead(monkeypatch: pytest.MonkeyPatch, world: _Clock) -> Any:
    """``_in_way_ahead`` along +x from the origin, the walker found beside
    the path where it stands now; the vehicle's body 1 m to each side."""
    from autoware_carla_scenario.actions import _trajectory_run
    from autoware_carla_scenario.utils import clearance

    def found(*args: Any) -> Any:
        at = world.walker.at
        return SimpleNamespace(id=66, x=at.x, y=at.y, ahead=at.x, side=at.y)

    monkeypatch.setattr(clearance, "first_box_in_way", found)
    # Looked up, so the leaving tests fail on their assertions where walkers
    # are not measured.
    forget = getattr(_trajectory_run, "forget_walkers", None)
    if forget is not None:
        forget()

    def look() -> int:
        return _trajectory_run._in_way_ahead(
            world,  # type: ignore[arg-type]
            1,
            0.0,
            0.0,
            0.0,
            "walker",
            5.0,
            2.0,
            behind=(0.0, 0.0, 0.0, 0.0),
            body_half_width=1.0,
        )

    return look


def test_a_walker_on_its_controller_walking_off_the_way_is_measured_leaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A walker on CARLA's controller reports standing while it walks: it is
    measured from where it stood the tick before, and walking off the way it
    is not waited for (#929).  Two reading it on one tick read the same."""
    world = _Clock(_ai_walker(1.5))
    look = _ahead(monkeypatch, world)
    assert look() == 66  # nothing to measure from yet: it reports standing
    world.elapsed += 0.05
    world.walker.at = SimpleNamespace(x=3.0, y=1.57, z=0.0)  # 1.4 m/s away
    assert look() == -1
    assert look() == -1  # read again on the same tick: the same


def test_a_walker_on_its_controller_standing_in_the_way_is_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _Clock(_ai_walker(1.5))
    look = _ahead(monkeypatch, world)
    assert look() == 66
    world.elapsed += 0.05
    assert look() == 66


def test_a_teleported_walker_is_not_read_as_leaving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A walker put elsewhere between two ticks (faster than any walk) did
    not walk there: it is not leaving, and is waited for."""
    world = _Clock(_ai_walker(1.5))
    look = _ahead(monkeypatch, world)
    assert look() == 66
    world.elapsed += 0.05
    world.walker.at = SimpleNamespace(x=3.0, y=4.0, z=0.0)  # 50 m/s: put there
    assert look() == 66


def test_a_walker_on_its_controller_looks_as_far_ahead_as_it_walks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_vehicle_before_walker`` looks a second's walk ahead: for a walker on
    CARLA's controller, which reports standing, that walk is measured."""
    from autoware_carla_scenario.actions import _trajectory_run
    from autoware_carla_scenario.utils import clearance

    reaches: List[float] = []

    def found(*args: Any) -> Any:
        reaches.append(float(args[6]))
        return None

    monkeypatch.setattr(clearance, "first_box_in_way", found)
    forget = getattr(_trajectory_run, "forget_walkers", None)
    if forget is not None:
        forget()
    walker = _ai_walker(1.5)
    world = _Clock(walker)
    walker.get_world = lambda: world
    walker.get_transform = lambda: SimpleNamespace(
        location=walker.at, rotation=SimpleNamespace(yaw=90.0)
    )
    walker.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=0.3, y=0.3, z=0.9))
    assert not _trajectory_run._vehicle_before_walker(walker)
    world.elapsed += 0.05
    walker.at = SimpleNamespace(x=3.0, y=1.57, z=0.0)  # 1.4 m/s
    assert not _trajectory_run._vehicle_before_walker(walker)
    assert reaches == pytest.approx([1.8, 3.2])


def test_a_vehicles_velocity_is_what_carla_reports() -> None:
    """Only walkers are measured: a vehicle reads what CARLA reports, however
    far it has gone between two ticks."""
    from autoware_carla_scenario.actions import _trajectory_run

    _trajectory_run.forget_walkers()
    vehicle = SimpleNamespace(
        id=7,
        type_id="vehicle.mini",
        get_velocity=lambda: SimpleNamespace(x=4.0, y=-1.0, z=0.0),
        get_location=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    world = _Clock(vehicle)
    first = _trajectory_run._measured_velocity(world, vehicle)  # type: ignore[arg-type]
    world.elapsed += 0.05
    second = _trajectory_run._measured_velocity(world, vehicle)  # type: ignore[arg-type]
    assert (first.x, first.y) == (second.x, second.y) == (4.0, -1.0)


def test_the_walkers_measured_are_forgotten_when_a_run_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.actions import _trajectory_run

    world = _Clock(_ai_walker(1.5))
    look = _ahead(monkeypatch, world)
    look()
    _trajectory_run.forget_walkers()
    world.elapsed += 0.05
    world.walker.at = SimpleNamespace(x=3.0, y=1.57, z=0.0)
    assert look() == 66  # nothing to measure from: a new run's first read


def test_a_path_crossing_to_the_lane_beside_marks_a_lane_change() -> None:
    """Where a path steps sideways by a lane (2 m or more) it crosses to the
    lane beside: marked, with the side (CARLA: +y is south, so a step to +y
    going east is to the right); a gentle curve is not."""
    from autoware_carla_scenario.traffic.traffic_manager import _lane_change_marks

    east = [carla.Location(float(x), 0.0, 0.0) for x in (0, 2, 4)]
    across = [carla.Location(6.0, 3.5, 0.0), carla.Location(8.0, 3.5, 0.0)]
    marks = _lane_change_marks(east + across)
    assert [(m.x, m.y, right) for m, right in marks] == [(4.0, 0.0, True)]
    curve = [carla.Location(float(x), 0.1 * x * x, 0.0) for x in range(6)]
    assert _lane_change_marks(curve) == []
    # A turn, its junction points left out: a long step to the side and on
    # another way, not a lane change.
    turn = [carla.Location(float(x), 0.0, 0.0) for x in (0, 2, 4)] + [
        carla.Location(20.0, 16.0 + float(y), 0.0) for y in (0, 2, 4)
    ]
    assert _lane_change_marks(turn) == []


class _LaneChangingTrafficManager:
    """Records the bool each ``force_lane_change`` is sent."""

    def __init__(self) -> None:
        self.lane_changes: List[bool] = []

    def force_lane_change(self, vehicle: Any, to_the_right: bool) -> None:
        self.lane_changes.append(to_the_right)


@pytest.mark.parametrize(("side", "expected"), [(3.5, True), (-3.5, False)])
def test_a_path_crossing_to_the_lane_beside_sends_carlas_bool(
    side: float, expected: bool
) -> None:
    """CARLA's ``force_lane_change`` takes ``True`` for the right
    (``LocalizationStage``), as :meth:`LaneChangeDirection.to_carla_bool`
    sends it: a path stepping to +y going east (the right, in CARLA's
    left-handed world) sends ``True``, one stepping to -y ``False``."""
    from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
    from autoware_carla_scenario.traffic.traffic_manager import (
        TrafficManagerBackend,
        _Kept,
        _lane_change_marks,
    )

    from .test_traffic_backend import _Client

    tm = _LaneChangingTrafficManager()
    backend = TrafficManagerBackend(
        TrafficManagerBackendConfig(port=8123),
        client=_Client(tm),  # type: ignore[arg-type]
    )
    npc = _Actor(5)
    npc.location = carla.Location(0.0, 0.0, 0.0)
    kept = _Kept(npc)  # type: ignore[arg-type]
    east = [carla.Location(float(x), 0.0, 0.0) for x in (0, 2, 4)]
    across = [carla.Location(6.0, side, 0.0), carla.Location(8.0, side, 0.0)]
    kept.lane_changes = _lane_change_marks(east + across)
    backend._change_lanes_on_path(kept)
    assert tm.lane_changes == [expected]
    assert kept.lane_changes == []


def test_a_vehicle_wedged_on_a_kerb_is_given_full_throttle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Throttle on, nothing in its way, and making no way for 1.5 s (an 11 m
    bus's corner on the kerb, vehicle_41): full throttle; moving, as asked."""
    vehicle = _Vehicle(speed=0.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path)  # type: ignore[arg-type]
    elapsed = 0.0
    with caplog.at_level(logging.INFO):
        while elapsed < 2.0:
            approach.step(_DT, elapsed)  # the vehicle does not move
            elapsed += _DT
    assert vehicle.control.throttle == pytest.approx(1.0)
    assert any("wedged" in r.getMessage() for r in caplog.records)
    while elapsed < 6.0:
        approach.step(_DT, elapsed)
        vehicle.step(_DT, driven=True)
        elapsed += _DT
    assert vehicle.control.throttle < 1.0 or vehicle.control.brake > 0.0


def test_a_vehicle_wedged_beyond_full_throttle_stops_where_it_is(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Not over the kerb at full throttle either (the recording's path ran
    over it): after a few seconds it stops there, done, and says so."""
    vehicle = _Vehicle(speed=0.0)
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path)  # type: ignore[arg-type]
    elapsed = 0.0
    with caplog.at_level(logging.WARNING):
        while elapsed < 6.0 and not approach.done:
            approach.step(_DT, elapsed)  # it never moves
            elapsed += _DT
    assert approach.done
    assert vehicle.control.brake == pytest.approx(1.0)
    assert any("stops there" in r.getMessage() for r in caplog.records)


def test_a_path_whose_corners_would_stand_on_a_sidewalk_is_moved_off_it() -> None:
    """A sidewalk from y = 1.5 on: a 2.5 m wide bus driven along y = 0.5 would
    have its corners on it, so its points -- its goal too -- move over (up to
    2.5 m) till they are off; one clear of it is left be."""
    from autoware_carla_scenario.utils import clearance

    class _Map:
        def get_waypoint(
            self, location: Any, project_to_road: bool, lane_type: Any
        ) -> Any:
            kind = (
                carla.LaneType.Sidewalk if location.y > 1.5 else carla.LaneType.Driving
            )
            return SimpleNamespace(lane_type=kind)

    points = [(float(x), 0.5, 0.0) for x in range(2, 21, 2)]
    moved = clearance.off_the_kerb(_Map(), points, 5.5, 1.25, 0.0, 0.5)
    assert all(y <= 0.25 + 1e-6 for _x, y, _z in moved)
    clear = [(float(x), -1.0, 0.0) for x in range(2, 21, 2)]
    assert clearance.off_the_kerb(_Map(), clear, 5.5, 1.25, 0.0, -1.0) == clear
