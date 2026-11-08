"""A lane change asked for at a given place and over a given length, on the
TrafficManager backend: a straight road along +x, the lane to the left at
y = -3.5 (CARLA's world is left-handed)."""

from __future__ import annotations

import math
from typing import Any, Optional

import pytest

from autoware_carla_scenario import LaneChangeAction
from autoware_carla_scenario.traffic import LaneChangeDirection
from autoware_carla_scenario.traffic.driven import BackendDriven
from autoware_carla_scenario.traffic.traffic_manager import (
    TrafficManagerBackend,
    _Kept,
)

LANE_WIDTH_M = 3.5
LANE_CHANGE_BOTH = 3
LANE_CHANGE_NONE = 0


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = 0.0

    def distance(self, other: "_Vector") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class _Transform:
    def __init__(self, location: _Vector) -> None:
        self.location = location

    def get_forward_vector(self) -> _Vector:
        return _Vector(1.0, 0.0)

    def get_right_vector(self) -> _Vector:
        return _Vector(0.0, 1.0)


class _Box:
    extent = _Vector(2.0, 1.0)


class _Waypoint:
    road_id = 1
    is_junction = False

    def __init__(self, y: float, x: float, lane_change: int, lane_id: int = -1) -> None:
        self.lane_id = lane_id
        self.lane_change = lane_change
        self.transform = _Transform(_Vector(x, y))
        self._x = x

    def get_left_lane(self) -> "_Waypoint":
        return _Waypoint(-LANE_WIDTH_M, self._x, self.lane_change, lane_id=-2)

    def get_right_lane(self) -> None:
        return None


class _Map:
    def __init__(self, lane_change: int) -> None:
        self.lane_change = lane_change

    def get_waypoint(
        self, location: _Vector, project_to_road: bool = True
    ) -> _Waypoint:
        return _Waypoint(0.0, location.x, self.lane_change)


class _Actor:
    bounding_box = _Box()
    is_alive = True

    def __init__(self, actor_id: int, x: float, y: float = 0.0) -> None:
        self.id = actor_id
        self.location = _Vector(x, y)
        self.velocity = _Vector()
        self.attributes = {"role_name": "npc"}

    def get_location(self) -> _Vector:
        return self.location

    def get_transform(self) -> _Transform:
        return _Transform(self.location)

    def get_velocity(self) -> _Vector:
        return self.velocity

    def get_speed_limit(self) -> float:
        return 50.0


class _Actors(list):
    def filter(self, pattern: str) -> "_Actors":
        return self


class _World:
    def __init__(self, actors: list[_Actor], lane_change: int) -> None:
        self.actors = _Actors(actors)
        self.map = _Map(lane_change)

    def get_map(self) -> _Map:
        return self.map

    def get_actors(self) -> _Actors:
        return self.actors


class _TrafficManager:
    def __init__(self) -> None:
        self.auto: list[bool] = []
        self.forced: list[bool] = []
        self.offsets: list[float] = []
        self.speeds: list[float] = []
        self.differences: list[float] = []

    def auto_lane_change(self, actor: Any, enable: bool) -> None:
        self.auto.append(enable)

    def force_lane_change(self, actor: Any, to_right: bool) -> None:
        self.forced.append(to_right)

    def vehicle_lane_offset(self, actor: Any, offset: float) -> None:
        self.offsets.append(offset)

    def set_desired_speed(self, actor: Any, speed: float) -> None:
        self.speeds.append(speed)

    def vehicle_percentage_speed_difference(self, actor: Any, pct: float) -> None:
        self.differences.append(pct)

    def set_path(self, actor: Any, path: list) -> None:
        pass


class _Client:
    def __init__(self, tm: _TrafficManager) -> None:
        self.tm = tm

    def get_trafficmanager(self, port: int) -> _TrafficManager:
        return self.tm


class _Vehicle(BackendDriven):
    def __init__(self, actor: _Actor) -> None:
        self.actor = actor


class _Run:
    def __init__(
        self, lane_change: int = LANE_CHANGE_BOTH, other: Optional[_Actor] = None
    ) -> None:
        self.tm = _TrafficManager()
        self.backend = TrafficManagerBackend()
        self.backend._client = _Client(self.tm)
        self.actor = _Actor(1, 0.0)
        self.entity = _Vehicle(self.actor)
        actors = [self.actor]
        if other is not None:
            actors.append(other)
        self.world = _World(actors, lane_change)

    def ask(self, start_after_m: float, length_m: float) -> None:
        self.backend.change_lane(
            self.entity, self.world, LaneChangeDirection.LEFT, start_after_m, length_m
        )

    def drive_to(self, x: float, y: float = 0.0) -> None:
        self.actor.location = _Vector(x, y)
        self.backend._carry_out_lane_changes(self.world)


class TestStartAfter:
    def test_it_waits_for_the_distance_driven(self) -> None:
        run = _Run()
        run.ask(20.0, 0.0)
        assert run.tm.forced == []
        assert run.entity._lane_change_target is None
        run.drive_to(19.0)
        assert run.tm.forced == []
        run.drive_to(21.0)
        assert run.tm.forced == [False]
        assert run.entity._lane_change_target == (1, -2)

    def test_without_a_length_the_crossing_is_the_traffic_managers(self) -> None:
        run = _Run()
        run.ask(5.0, 0.0)
        run.drive_to(6.0)
        run.drive_to(7.0)
        assert run.tm.offsets == []
        assert run.backend._lane_change_requests == []

    def test_no_options_force_at_once_with_no_offset(self) -> None:
        run = _Run()
        run.ask(0.0, 0.0)
        assert run.tm.forced == [False]
        assert run.tm.offsets == []
        assert run.backend._lane_change_requests == []


class TestOwnLaneChanges:
    def test_they_are_off_from_the_request_to_the_end_of_the_crossing(self) -> None:
        run = _Run()
        run.ask(10.0, 20.0)
        assert run.tm.auto == [False]
        run.drive_to(11.0)
        assert run.tm.auto == [False]
        run.drive_to(40.0)
        assert run.tm.auto == [False, True]

    def test_without_a_length_they_are_back_once_it_is_told(self) -> None:
        run = _Run()
        run.ask(10.0, 0.0)
        run.drive_to(11.0)
        assert run.tm.auto == [False, True]

    def test_no_options_leave_them_alone(self) -> None:
        run = _Run()
        run.ask(0.0, 0.0)
        assert run.tm.auto == []


class _RoutedRun(_Run):
    """A run whose vehicle is kept (as on autopilot) and can be given a route
    along the road, with no map: the route is its points."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        from types import SimpleNamespace  # noqa: PLC0415

        import autoware_carla_scenario.coordinate.map_manager as map_manager  # noqa: PLC0415
        import autoware_carla_scenario.coordinate.transform as transform  # noqa: PLC0415
        import autoware_carla_scenario.traffic.traffic_manager as traffic_manager  # noqa: PLC0415

        monkeypatch.setattr(transform, "to_carla_location", lambda pose: pose)
        monkeypatch.setattr(
            map_manager.MapManager,
            "get_instance",
            staticmethod(
                lambda: SimpleNamespace(
                    lanelet_map=None, routing_graph=None, carla_map=None
                )
            ),
        )
        monkeypatch.setattr(
            traffic_manager, "_clear_of_junctions", lambda m, points: list(points)
        )
        monkeypatch.setattr(traffic_manager, "_in_junction", lambda m, p: False)
        monkeypatch.setattr(traffic_manager, "_at_junction_mouth", lambda m, p: False)
        monkeypatch.setattr(traffic_manager, "_lane_change_marks", lambda path: [])
        self.monkeypatch = monkeypatch
        kept = _Kept(self.actor)
        kept.check_due = 1.0e9  # Never looked at for standing or a dead end.
        self.backend._kept[self.actor.id] = kept

    def route(self, *xs: float) -> None:
        """Give it the route from where it stands through each of *xs*."""
        from types import SimpleNamespace  # noqa: PLC0415

        import autoware_carla_scenario.odd.route_plan as route_plan  # noqa: PLC0415

        here = self.actor.location
        points = [_Vector(here.x, here.y)] + [_Vector(x) for x in xs]
        self.monkeypatch.setattr(route_plan, "route_poses", lambda *a, **k: points)
        self.backend.follow_route(
            self.entity,
            self.world,  # type: ignore[arg-type]
            [SimpleNamespace(), SimpleNamespace()],  # type: ignore[list-item]
        )

    def tick(self) -> None:
        """A backend tick: the route ends where its path is used up."""
        self.backend._keep_scenario_vehicles(self.world, 0.0)  # type: ignore[arg-type]


class TestOwnLaneChangesWithARoute:
    """A route and a crossing each turn the TrafficManager's own lane changes
    off; they come back only once neither holds (#2005)."""

    def test_a_route_used_up_gives_them_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _RoutedRun(monkeypatch)
        run.route(10.0, 20.0)
        assert run.tm.auto == [False]
        run.drive_to(10.0)
        run.tick()
        assert run.tm.auto == [False]
        run.drive_to(20.0)
        run.tick()
        assert run.tm.auto == [False, True]
        run.tick()
        assert run.tm.auto == [False, True]

    def test_a_spent_route_leaves_a_later_crossing_locked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _RoutedRun(monkeypatch)
        run.route(10.0, 20.0)
        run.drive_to(10.0)
        run.tick()
        run.drive_to(20.0)
        run.tick()
        assert run.tm.auto == [False, True]
        run.ask(10.0, 20.0)
        assert run.tm.auto == [False, True, False]
        run.tick()
        run.drive_to(31.0)
        run.tick()
        assert run.tm.auto == [False, True, False]
        run.drive_to(60.0)
        run.tick()
        assert run.tm.auto == [False, True, False, True]

    @pytest.mark.parametrize(("start_after_m", "length_m"), [(10.0, 0.0), (0.0, 20.0)])
    def test_a_turn_used_up_leaves_a_crossing_locked(
        self, monkeypatch: pytest.MonkeyPatch, start_after_m: float, length_m: float
    ) -> None:
        """A turn's path, driven to its end, is no route to end."""
        run = _RoutedRun(monkeypatch)
        run.backend._remember_path(run.actor, [_Vector(-10.0), _Vector(0.0)])  # type: ignore[list-item]
        run.tick()
        assert run.backend._kept[1].progress == 2
        run.ask(start_after_m, length_m)
        assert run.tm.auto == [False]
        run.tick()
        assert run.tm.auto == [False]

    def test_the_end_of_a_crossing_leaves_a_later_route_locked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _RoutedRun(monkeypatch)
        run.ask(10.0, 20.0)
        run.route(50.0, 60.0)
        assert run.tm.auto == [False, False]
        run.drive_to(11.0)
        run.drive_to(40.0)
        assert run.backend._lane_change_requests == []
        run.tick()
        assert run.tm.auto == [False, False]
        run.drive_to(50.0)
        run.tick()
        run.drive_to(60.0)
        run.tick()
        assert run.tm.auto == [False, False, True]

    def test_the_end_of_a_route_leaves_a_later_crossing_locked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _RoutedRun(monkeypatch)
        run.route(10.0, 20.0)
        run.drive_to(5.0)
        run.ask(5.0, 30.0)
        assert run.tm.auto == [False]  # Off already.
        run.drive_to(10.0)
        run.tick()
        run.drive_to(20.0)
        run.tick()
        assert run.tm.auto == [False]
        assert run.backend._lane_change_requests != []
        run.drive_to(60.0)
        assert run.tm.auto == [False, True]

    def test_end_route_leaves_a_crossing_locked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _RoutedRun(monkeypatch)
        run.route(50.0, 60.0)
        run.ask(10.0, 20.0)
        run.backend.end_route(run.entity, run.world)  # type: ignore[arg-type]
        assert run.tm.auto == [False]
        run.drive_to(11.0)
        run.drive_to(40.0)
        assert run.tm.auto == [False, True]


class TestLength:
    def test_the_offset_goes_from_where_it_is_to_nothing(self) -> None:
        run = _Run()
        run.ask(0.0, 40.0)
        # Keeps it where it is, right of the lane it goes to.
        assert run.tm.offsets == [pytest.approx(LANE_WIDTH_M)]
        assert run.tm.forced == [False]
        run.drive_to(20.0, -LANE_WIDTH_M / 2)
        # Half the arc length, as the crow flies from the last tick.
        travelled = math.hypot(20.0, LANE_WIDTH_M / 2)
        expected = LANE_WIDTH_M * 0.5 * (1.0 + math.cos(math.pi * travelled / 40.0))
        assert run.tm.offsets[-1] == pytest.approx(expected)
        run.drive_to(41.0, -LANE_WIDTH_M)
        assert run.tm.offsets[-1] == 0.0
        assert run.backend._lane_change_requests == []

    def test_it_stops_crossing_while_it_stands(self) -> None:
        run = _Run()
        run.ask(0.0, 40.0)
        run.drive_to(20.0, -1.0)
        before = run.tm.offsets[-1]
        run.drive_to(20.0, -1.0)
        assert run.tm.offsets[-1] == pytest.approx(before)

    def test_no_offset_where_the_marking_forbids_it(self) -> None:
        run = _Run(lane_change=LANE_CHANGE_NONE)
        run.ask(0.0, 40.0)
        assert run.tm.offsets == []
        assert run.tm.forced == [False]
        assert run.backend._lane_change_requests == []

    def test_a_crossing_the_traffic_manager_refused_is_given_up(self) -> None:
        run = _Run()
        run.ask(0.0, 40.0)
        # Carried away from the lane it should go to.
        run.drive_to(5.0, 1.0)
        assert run.tm.offsets[-1] == 0.0
        assert run.backend._lane_change_requests == []

    def test_asking_again_replaces_the_request(self) -> None:
        run = _Run()
        run.ask(0.0, 40.0)
        run.ask(100.0, 0.0)
        assert run.tm.offsets[-1] == 0.0
        assert len(run.backend._lane_change_requests) == 1


class TestWayAhead:
    def test_it_slows_for_one_ahead_in_the_lane_it_leaves(self) -> None:
        run = _Run(other=_Actor(2, 15.0, 0.0))
        run.backend._exact_kmh[1] = 30.0
        run.ask(0.0, 40.0)
        run.drive_to(1.0)
        gap = 14.0 - 2.0 - 2.0 - 3.0
        assert run.tm.speeds[-1] == pytest.approx(
            math.sqrt(2.0 * 3.0 * gap) * 3.6, abs=0.01
        )

    def test_it_slows_for_one_ahead_in_the_lane_it_goes_to(self) -> None:
        run = _Run(other=_Actor(2, 15.0, -LANE_WIDTH_M))
        run.backend._exact_kmh[1] = 30.0
        run.ask(0.0, 40.0)
        run.drive_to(1.0)
        assert len(run.tm.speeds) == 1

    def test_one_in_the_lane_on_the_other_side_is_not_in_the_way(self) -> None:
        run = _Run(other=_Actor(2, 15.0, LANE_WIDTH_M))
        run.backend._exact_kmh[1] = 30.0
        run.ask(0.0, 40.0)
        run.drive_to(1.0)
        assert run.tm.speeds == []

    def test_it_gets_its_speed_back(self) -> None:
        other = _Actor(2, 15.0, 0.0)
        run = _Run(other=other)
        run.backend._exact_kmh[1] = 30.0
        run.ask(0.0, 40.0)
        run.drive_to(1.0)
        other.location = _Vector(200.0, 0.0)
        run.drive_to(2.0)
        assert run.tm.speeds[-1] == 30.0


class TestAction:
    @pytest.mark.parametrize(
        "kwargs", [{"start_after_m": -1.0}, {"lane_change_length_m": -1.0}]
    )
    def test_a_negative_distance_is_refused(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            LaneChangeAction("npc", LaneChangeDirection.LEFT, **kwargs)
