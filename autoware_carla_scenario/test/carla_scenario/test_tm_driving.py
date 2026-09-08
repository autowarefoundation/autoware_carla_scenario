"""Manoeuvres for a TrafficManager-driven vehicle, and who performs them.

The action names an intent and the entity carries it out, so these are the
entity's tests: what reaches the TrafficManager, and what counts as finished.
None of it had a test while it lived in the action, which is why the behaviour
is pinned here rather than assumed.

CARLA is faked -- the manoeuvre is a single TrafficManager call and the
completion check is arithmetic on a transform, so a live server would only make
the test slower.
"""

from __future__ import annotations

from typing import Optional

import pytest

from autoware_carla_scenario.entity.tm_driving import (
    LaneChangeDirection,
    TrafficManagerDriven,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Rotation:
    def __init__(self, yaw: float) -> None:
        self.yaw = yaw


class _Location:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y

    def distance(self, other: "_Location") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class _Transform:
    def __init__(self, location: _Location, yaw: float = 0.0) -> None:
        self.location = location
        self.rotation = _Rotation(yaw)


class _Waypoint:
    def __init__(
        self,
        road_id: int,
        lane_id: int,
        location: Optional[_Location] = None,
        yaw: float = 0.0,
        left: Optional["_Waypoint"] = None,
        right: Optional["_Waypoint"] = None,
    ) -> None:
        self.road_id = road_id
        self.lane_id = lane_id
        self.transform = _Transform(location or _Location(), yaw)
        self._left = left
        self._right = right

    def get_left_lane(self) -> Optional["_Waypoint"]:
        return self._left

    def get_right_lane(self) -> Optional["_Waypoint"]:
        return self._right


class _Map:
    def __init__(self, waypoint: Optional[_Waypoint]) -> None:
        self._waypoint = waypoint
        self.queries = 0

    def get_waypoint(self, location, project_to_road=True):  # noqa: ANN001, ARG002
        self.queries += 1
        return self._waypoint


class _World:
    def __init__(self, carla_map: _Map) -> None:
        self._map = carla_map

    def get_map(self) -> _Map:
        return self._map


class _TrafficManager:
    def __init__(self) -> None:
        self.lane_changes: list[tuple[object, bool]] = []
        self.paths: list[tuple[object, list]] = []

    def force_lane_change(self, actor, to_right: bool) -> None:  # noqa: ANN001
        self.lane_changes.append((actor, to_right))

    def set_path(self, actor, path: list) -> None:  # noqa: ANN001
        self.paths.append((actor, path))


class _Client:
    def __init__(self, tm: _TrafficManager) -> None:
        self._tm = tm
        self.ports: list[int] = []

    def get_trafficmanager(self, port: int) -> _TrafficManager:
        self.ports.append(port)
        return self._tm


class _Actor:
    def __init__(self, transform: _Transform) -> None:
        self._transform = transform

    def get_location(self) -> _Location:
        return self._transform.location

    def get_transform(self) -> _Transform:
        return self._transform


class _Vehicle(TrafficManagerDriven):
    """The smallest thing the mixin needs: something with an ``actor``."""

    def __init__(self, actor: Optional[_Actor] = None) -> None:
        self.actor = actor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _vehicle_on(road: int, lane: int, *, left=None, right=None, yaw: float = 0.0):
    """A vehicle sitting on ``(road, lane)``, with the given neighbours."""
    here = _Waypoint(road, lane, _Location(0.0, 0.0), yaw, left=left, right=right)
    actor = _Actor(_Transform(_Location(0.0, 0.0), yaw))
    vehicle = _Vehicle(actor)
    tm = _TrafficManager()
    vehicle.set_client(_Client(tm), tm_port=8123)
    return vehicle, tm, _World(_Map(here))


class TestChangeLane:
    def test_right_is_true_and_left_is_false(self) -> None:
        """CARLA's ``force_lane_change`` takes a bool, and the sense matters.

        Inverting it sends every vehicle the wrong way while every test that
        only checks "a lane change was requested" still passes.
        """
        for direction, expected in (
            (LaneChangeDirection.RIGHT, True),
            (LaneChangeDirection.LEFT, False),
        ):
            neighbour = _Waypoint(1, 2)
            vehicle, tm, world = _vehicle_on(1, 1, left=neighbour, right=neighbour)
            vehicle.change_lane(world, direction)
            assert [sent for _, sent in tm.lane_changes] == [expected]

    def test_the_port_the_entity_was_given_is_the_one_used(self) -> None:
        vehicle, _tm, world = _vehicle_on(1, 1, right=_Waypoint(1, 2))
        vehicle.change_lane(world, LaneChangeDirection.RIGHT)
        assert vehicle._tm_client.ports == [8123]

    def test_the_lane_aimed_at_is_recorded(self) -> None:
        """Completion is the *target* lane, not "some other lane id".

        Lane ids are scoped to a road, so a vehicle that reaches a continuation
        road without moving sideways gets a different ``(road_id, lane_id)``.
        """
        vehicle, _tm, world = _vehicle_on(1, 1, right=_Waypoint(1, 2))
        vehicle.change_lane(world, LaneChangeDirection.RIGHT)
        assert vehicle._lane_change_target == (1, 2)

    def test_a_lane_change_with_nowhere_to_go_is_still_requested(self) -> None:
        """No neighbour means no target, and the manoeuvre never finishes.

        The request still goes out: refusing it here would differ from what the
        TrafficManager does with it, and the honest report is "asked for, never
        completed".
        """
        vehicle, tm, world = _vehicle_on(1, 1)
        vehicle.change_lane(world, LaneChangeDirection.LEFT)
        assert vehicle._lane_change_target is None
        assert len(tm.lane_changes) == 1
        assert vehicle.lane_change_finished(world) is False

    def test_without_a_client_nothing_is_sent_and_nothing_raises(self) -> None:
        """An entity nobody injected a client into says so rather than crashing."""
        vehicle = _Vehicle(_Actor(_Transform(_Location())))
        vehicle.change_lane(_World(_Map(_Waypoint(1, 1))), LaneChangeDirection.LEFT)
        assert vehicle.lane_change_finished(_World(_Map(None))) is False

    def test_without_an_actor_nothing_is_sent(self) -> None:
        vehicle = _Vehicle(actor=None)
        tm = _TrafficManager()
        vehicle.set_client(_Client(tm))
        vehicle.change_lane(_World(_Map(_Waypoint(1, 1))), LaneChangeDirection.LEFT)
        assert tm.lane_changes == []


class TestLaneChangeFinished:
    """Settled means on the target lane, centred on it, and pointing along it."""

    @staticmethod
    def _mid_manoeuvre(
        target_road: int, target_lane: int, *, offset: float, yaw: float
    ):
        vehicle = _Vehicle(_Actor(_Transform(_Location(0.0, 0.0), yaw)))
        vehicle._lane_change_target = (target_road, target_lane)
        vehicle._lane_change_map = _Map(
            _Waypoint(target_road, target_lane, _Location(offset, 0.0), 0.0)
        )
        return vehicle

    def test_settled_when_on_the_lane_centred_and_aligned(self) -> None:
        vehicle = self._mid_manoeuvre(1, 2, offset=0.1, yaw=1.0)
        assert vehicle.lane_change_finished(_World(_Map(None))) is True

    def test_a_different_lane_is_not_the_target_lane(self) -> None:
        vehicle = self._mid_manoeuvre(1, 2, offset=0.0, yaw=0.0)
        vehicle._lane_change_map = _Map(_Waypoint(1, 3, _Location(), 0.0))
        assert vehicle.lane_change_finished(_World(_Map(None))) is False

    def test_still_diagonal_across_the_boundary_is_not_finished(self) -> None:
        """The centre has crossed but the vehicle has not straightened up.

        Calling this finished lets a reaction fire mid-manoeuvre.
        """
        vehicle = self._mid_manoeuvre(1, 2, offset=0.0, yaw=45.0)
        assert vehicle.lane_change_finished(_World(_Map(None))) is False

    def test_too_far_from_the_lane_centre_is_not_finished(self) -> None:
        vehicle = self._mid_manoeuvre(1, 2, offset=5.0, yaw=0.0)
        assert vehicle.lane_change_finished(_World(_Map(None))) is False

    def test_a_manoeuvre_never_started_never_finishes(self) -> None:
        assert _Vehicle().lane_change_finished(_World(_Map(None))) is False


class TestEntitiesThatAreNotTrafficManagerDriven:
    """They refuse rather than sending a command that would do nothing."""

    def test_an_autoware_ego_refuses_a_forced_lane_change(self, caplog) -> None:
        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.entity import AutowareEgoEntity

        entity = AutowareEgoEntity(bridge=FakeAutowareBridge())
        with caplog.at_level("WARNING"):
            entity.change_lane(_World(_Map(None)), LaneChangeDirection.LEFT)
        assert "not driven by the TrafficManager" in caplog.text

    def test_a_driver_ego_refuses_a_forced_lane_change(self, caplog) -> None:
        from autoware_carla_scenario.driver import DriverClientConfig
        from autoware_carla_scenario.entity import CarlaDriverEntity

        entity = CarlaDriverEntity(DriverClientConfig())
        with caplog.at_level("WARNING"):
            entity.change_lane(_World(_Map(None)), LaneChangeDirection.RIGHT)
        assert "not driven by the TrafficManager" in caplog.text


class TestTheActionDelegates:
    """The action names the entity; the entity performs the manoeuvre."""

    def test_execute_asks_the_named_entity(self) -> None:
        from autoware_carla_scenario.actions import LaneChangeAction
        from autoware_carla_scenario.entity.registry import (
            clear_entities,
            register_entity,
        )

        asked: list[LaneChangeDirection] = []

        class _Recording(_Vehicle):
            def change_lane(self, world, direction) -> None:  # noqa: ANN001
                asked.append(direction)

            def lane_change_finished(self, world) -> bool:  # noqa: ANN001
                return True

        clear_entities()
        register_entity("npc1", _Recording())
        try:
            action = LaneChangeAction(
                "npc1", LaneChangeDirection.LEFT, client=None, label="lc"
            )
            world = _World(_Map(None))
            action.execute(world)
            assert asked == [LaneChangeDirection.LEFT]
            assert action.is_finished(world, 1.0) is True
        finally:
            clear_entities()

    def test_an_unknown_entity_is_reported_and_never_finishes(self, caplog) -> None:
        from autoware_carla_scenario.actions import LaneChangeAction
        from autoware_carla_scenario.entity.registry import clear_entities

        clear_entities()
        action = LaneChangeAction(
            "ghost", LaneChangeDirection.LEFT, client=None, label="lc"
        )
        world = _World(_Map(None))
        with caplog.at_level("WARNING"):
            action.execute(world)
        assert "not found" in caplog.text
        assert action.is_finished(world, 1.0) is False


@pytest.mark.parametrize(
    "direction", [LaneChangeDirection.LEFT, LaneChangeDirection.RIGHT]
)
def test_the_neighbour_looked_at_is_the_one_asked_for(direction) -> None:  # noqa: ANN001
    """Left must read the left neighbour, right the right one."""
    left = _Waypoint(1, 9)
    right = _Waypoint(1, 7)
    vehicle, _tm, world = _vehicle_on(1, 8, left=left, right=right)

    vehicle.change_lane(world, direction)

    expected = (1, 9) if direction is LaneChangeDirection.LEFT else (1, 7)
    assert vehicle._lane_change_target == expected


# ---------------------------------------------------------------------------
# Turning at a junction
# ---------------------------------------------------------------------------


class _TurnWaypoint(_Waypoint):
    """A waypoint that knows what comes next, so a route can be walked."""

    def __init__(self, *args, is_junction: bool = False, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.is_junction = is_junction
        self._next: list["_TurnWaypoint"] = []

    def then(self, *waypoints: "_TurnWaypoint") -> "_TurnWaypoint":
        self._next = list(waypoints)
        return self

    def next(self, step: float):  # noqa: ANN201, ARG002
        return list(self._next)


def _straight_run(yaw: float, count: int, *, y: float = 0.0, is_junction: bool = False):
    """A chain of *count* waypoints all heading *yaw*, offset to *y*.

    The offset is what tells the branches apart in the route that comes out:
    the path is a list of locations, so the branch that was chosen has to be
    readable from them.
    """
    chain = [
        _TurnWaypoint(1, 1, _Location(float(i), y), yaw, is_junction=is_junction)
        for i in range(count)
    ]
    for a, b in zip(chain, chain[1:]):
        a.then(b)
    return chain


class TestTurnAtJunction:
    """The route is worked out from the map, then handed to the TrafficManager."""

    @staticmethod
    def _map_with_a_fork():
        """A road that reaches a junction offering a left and a right branch.

        Exit headings are what the branches are told apart by: CARLA's yaw is
        clockwise-positive, so -90 is a left turn and +90 a right one.
        """
        left_branch = _straight_run(-90.0, 12, y=-5.0, is_junction=True)
        right_branch = _straight_run(90.0, 12, y=5.0, is_junction=True)
        for branch in (left_branch, right_branch):
            for wp in branch[-6:]:
                wp.is_junction = False
        pre = _TurnWaypoint(1, 1, _Location(0.0, 0.0), 0.0)
        pre.then(left_branch[0], right_branch[0])
        start = _TurnWaypoint(1, 1, _Location(-2.0, 0.0), 0.0).then(pre)
        return start

    def _vehicle(self):
        actor = _Actor(_Transform(_Location(0.0, 0.0), 0.0))
        vehicle = _Vehicle(actor)
        tm = _TrafficManager()
        vehicle.set_client(_Client(tm))
        return vehicle, tm, _World(_Map(self._map_with_a_fork()))

    def test_each_direction_takes_the_branch_that_bends_that_way(self) -> None:
        """The branch is chosen by exit heading, so both must be checked.

        A picker that always returned the first branch would satisfy a test
        that only asked for a left turn.
        """
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        for direction, expected_y in (
            (TurnDirection.LEFT, -5.0),
            (TurnDirection.RIGHT, 5.0),
        ):
            vehicle, tm, world = self._vehicle()
            vehicle.turn_at_junction(world, direction, post_junction_distance=4.0)

            assert len(tm.paths) == 1, direction
            path = tm.paths[0][1]
            assert path, direction
            assert {location.y for location in path} == {expected_y}, direction

    def test_a_route_that_cannot_be_found_sends_nothing(self) -> None:
        """A junction that is not there is reported, not faked."""
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        actor = _Actor(_Transform(_Location(0.0, 0.0), 0.0))
        vehicle = _Vehicle(actor)
        tm = _TrafficManager()
        vehicle.set_client(_Client(tm))
        # A road that simply ends: nothing ahead, so no junction and no route.
        dead_end = _TurnWaypoint(1, 1, _Location(), 0.0)
        vehicle.turn_at_junction(_World(_Map(dead_end)), TurnDirection.RIGHT)
        assert tm.paths == []

    def test_without_a_client_nothing_is_sent(self) -> None:
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        vehicle = _Vehicle(_Actor(_Transform(_Location())))
        vehicle.turn_at_junction(
            _World(_Map(self._map_with_a_fork())), TurnDirection.LEFT
        )  # must not raise

    def test_an_autoware_ego_refuses_a_turn_route(self, caplog) -> None:
        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.entity import AutowareEgoEntity
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        entity = AutowareEgoEntity(bridge=FakeAutowareBridge())
        with caplog.at_level("WARNING"):
            entity.turn_at_junction(_World(_Map(None)), TurnDirection.LEFT)
        assert "not driven by the TrafficManager" in caplog.text


class TestTheTurnActionDelegates:
    def test_execute_asks_the_named_entity_with_its_tuning(self) -> None:
        from autoware_carla_scenario.actions import TurnAction
        from autoware_carla_scenario.entity.registry import (
            clear_entities,
            register_entity,
        )
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        seen: list[tuple] = []

        class _Recording(_Vehicle):
            def turn_at_junction(self, world, direction, **kwargs) -> None:  # noqa: ANN001, ANN003
                seen.append((direction, kwargs))

        clear_entities()
        register_entity("npc1", _Recording())
        try:
            TurnAction(
                "npc1",
                TurnDirection.RIGHT,
                client=None,
                label="t",
                search_distance=42.0,
            ).execute(_World(_Map(None)))
        finally:
            clear_entities()

        assert seen[0][0] is TurnDirection.RIGHT
        assert seen[0][1]["search_distance"] == 42.0

    def test_an_unknown_entity_is_reported(self, caplog) -> None:
        from autoware_carla_scenario.actions import TurnAction
        from autoware_carla_scenario.entity.registry import clear_entities
        from autoware_carla_scenario.entity.tm_driving import TurnDirection

        clear_entities()
        with caplog.at_level("WARNING"):
            TurnAction("ghost", TurnDirection.LEFT, client=None, label="t").execute(
                _World(_Map(None))
            )
        assert "not found" in caplog.text
