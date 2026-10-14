"""A scenario that is a particular drive gets that drive, not the shortest one.

A goal alone is planned for by whatever route reaches it soonest. That is the
wrong route for a run whose point is the road it takes -- one rendered from a
recorded drive, where a shortcut leaves the rendered world entirely. These
cover the way a scenario says which drive it meant: waypoints, carried from the
config or derived from the route the scenario already asserts, snapped like the
goal and handed over with it.
"""

from __future__ import annotations

import math

import pytest

from autoware_carla_scenario.autoware_bridge import BridgePose, FakeAutowareBridge
from autoware_carla_scenario.coordinate import CarlaWorldPose, Lanelet2Pose


class TestTheBridgeCarriesThem:
    """The contract's third field: the way there, beside where and from where."""

    def test_a_mission_without_waypoints_leaves_the_way_there_open(self) -> None:
        bridge = FakeAutowareBridge()
        pose = BridgePose.from_yaw(0.0, 0.0, 0.0, 0.0)

        bridge.configure(pose, pose)

        assert bridge.configured_waypoints == []

    def test_the_waypoints_arrive_in_the_order_they_were_given(self) -> None:
        bridge = FakeAutowareBridge()
        start = BridgePose.from_yaw(0.0, 0.0, 0.0, 0.0)
        goal = BridgePose.from_yaw(90.0, 0.0, 0.0, 0.0)
        via = [BridgePose.from_yaw(x, 0.0, 0.0, 0.0) for x in (10.0, 20.0, 30.0)]

        bridge.configure(start, goal, via)

        assert [p.position.x for p in bridge.configured_waypoints] == [10.0, 20.0, 30.0]


class TestDerivingThemFromTheRoute:
    """A scenario that lists its route should not have to list it twice."""

    def test_the_asserted_route_becomes_the_route_asked_for(self) -> None:
        scenario = _scenario()

        scenario.derive_waypoints_from_route([11, 22, 33, 44])

        # The last is the goal, not a waypoint: routing to it is the point.
        assert [p.lanelet_id for p in scenario.waypoint_poses] == [11, 22, 33]

    def test_waypoints_named_elsewhere_win(self) -> None:
        """Deriving must not overwrite what a config or a caller already said."""
        scenario = _scenario()
        scenario.waypoint_poses = [Lanelet2Pose(lanelet_id=99, s=0.0)]

        scenario.derive_waypoints_from_route([11, 22, 33])

        assert [p.lanelet_id for p in scenario.waypoint_poses] == [99]

    @pytest.mark.parametrize("route", [[], [42]], ids=["empty", "goal-only"])
    def test_a_route_that_names_no_way_there_derives_nothing(self, route) -> None:
        """A drive with no intermediate road named is a legitimate drive."""
        scenario = _scenario()

        scenario.derive_waypoints_from_route(route)

        assert scenario.waypoint_poses == []

    def test_the_last_lanelet_stays_when_the_goal_is_somewhere_else(self) -> None:
        """A route's end is only the goal's when the goal says so.

        `derive_goal_from_route` keeps a goal the config named rather than
        overwriting it with the route's end.  When it did that, the route's last
        lanelet is an ordinary stretch of the way there -- dropping it would let
        Autoware reach the configured goal without ever driving it.
        """
        scenario = _scenario()
        scenario.goal_pose = Lanelet2Pose(lanelet_id=44, s=0.0)

        scenario.derive_waypoints_from_route([11, 22, 33])

        assert [p.lanelet_id for p in scenario.waypoint_poses] == [11, 22, 33]

    def test_the_last_lanelet_goes_when_it_is_the_goal(self) -> None:
        """The ordinary case: routing to it is the point, so it is not a via."""
        scenario = _scenario()
        scenario.goal_pose = Lanelet2Pose(lanelet_id=33, s=0.0)

        scenario.derive_waypoints_from_route([11, 22, 33])

        assert [p.lanelet_id for p in scenario.waypoint_poses] == [11, 22]

    def test_a_single_lanelet_route_is_a_way_there_when_the_goal_is_elsewhere(
        self,
    ) -> None:
        """One named road is still a road the run was meant to take."""
        scenario = _scenario()
        scenario.goal_pose = Lanelet2Pose(lanelet_id=99, s=0.0)

        scenario.derive_waypoints_from_route([42])

        assert [p.lanelet_id for p in scenario.waypoint_poses] == [42]


def _scenario():
    """A minimal concrete scenario; only its ego config is exercised here."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform

    class _Scenario(BaseScenario):
        def setup(self) -> None: ...

        def is_done(self) -> bool:
            return False

    ego = EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        )
    )
    return _Scenario(ego)


def test_the_entity_snaps_each_waypoint_as_the_lanelet_pose_it_was_written_as(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waypoints go through the same snap as the goal, in the same frame.

    A waypoint snapped as its OpenDRIVE projection would carry the road
    reference line's heading, which on a left-hand-traffic map converted from
    Lanelet2 points back down the lane -- the mission planner reads that as a
    request to traverse the lane backwards.
    """
    from autoware_carla_scenario.coordinate import OpenDrivePose
    from autoware_carla_scenario.entity import AutowareEgoEntity

    seen: list[object] = []

    def _snap(pose, world, ground_projection):
        seen.append(pose)
        return CarlaWorldPose(x=float(len(seen)), y=0.0, z=0.0, yaw=0.0)

    monkeypatch.setattr("autoware_carla_scenario.coordinate.snap_to_carla_road", _snap)
    monkeypatch.setattr(
        "autoware_carla_scenario.entity.bridged_ego.to_map_frame",
        lambda p: BridgePose.from_yaw(p.x, p.y, p.z, math.radians(p.yaw)),
    )

    bridge = FakeAutowareBridge()
    entity = AutowareEgoEntity(None, bridge=bridge)
    via = [Lanelet2Pose(lanelet_id=1, s=0.0), Lanelet2Pose(lanelet_id=2, s=0.0)]
    goal = Lanelet2Pose(lanelet_id=3, s=7.0)

    entity.route_to(object(), goal, waypoints=via)  # type: ignore[arg-type]

    # The goal is snapped last, after the waypoints it is the end of.
    assert seen == [via[0], via[1], goal]
    assert not any(isinstance(p, OpenDrivePose) for p in seen)


class TestTheRoutingActionHandsThemOn:
    """Waypoints reach the entity only when the scenario named some."""

    @staticmethod
    def _execute(monkeypatch: pytest.MonkeyPatch, entity, waypoints) -> None:
        from autoware_carla_scenario.actions.routing import RoutingAction
        from autoware_carla_scenario.entity import registry

        # An entity of neither kind (a duck-typed one) is registered as both,
        # so the action's typed lookups find it.
        monkeypatch.setattr(registry, "_VEHICLES", {})
        monkeypatch.setattr(registry, "_PEDESTRIANS", {})
        registry.register_entity("ego", entity)
        RoutingAction(
            "ego", Lanelet2Pose(lanelet_id=3, s=7.0), waypoints=waypoints
        ).execute(object())  # type: ignore[arg-type]

    def test_an_entity_with_the_route_to_it_had_before_waypoints_still_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """route_to() is overridable; an override written before waypoints
        existed must keep working for every run that names none."""
        routed: list[object] = []

        class _LegacyEntity:
            def route_to(
                self, world, goal, *, initial_pose=None, ground_projection=None
            ):
                routed.append(goal)

        self._execute(monkeypatch, _LegacyEntity(), ())

        assert routed == [Lanelet2Pose(lanelet_id=3, s=7.0)]

    def test_named_waypoints_are_handed_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []

        class _Entity:
            def route_to(
                self,
                world,
                goal,
                *,
                initial_pose=None,
                ground_projection=None,
                waypoints=(),
            ):
                seen.extend(waypoints)

        via = (Lanelet2Pose(lanelet_id=1, s=0.0),)
        self._execute(monkeypatch, _Entity(), via)

        assert seen == list(via)

    def test_an_entity_without_route_to_still_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entity that cannot be routed fails as the call always did."""

        class _Npc:
            pass

        with pytest.raises(AttributeError, match="route_to"):
            self._execute(monkeypatch, _Npc(), ())

    def test_a_pedestrian_is_found_when_no_vehicle_has_the_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pedestrian is kept as a pedestrian only: the second lookup."""
        from autoware_carla_scenario.entity import registry
        from autoware_carla_scenario.entity.pedestrian_entity import (
            PedestrianEntity,
        )

        routed: list[object] = []

        class _RoutedWalker(PedestrianEntity):
            def route_to(
                self, world, goal, *, initial_pose=None, ground_projection=None
            ):
                routed.append(goal)

        walker = _RoutedWalker.__new__(_RoutedWalker)
        self._execute(monkeypatch, walker, ())

        assert "ego" not in registry._VEHICLES
        assert routed == [Lanelet2Pose(lanelet_id=3, s=7.0)]

    def test_an_entity_not_registered_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario.actions.routing import RoutingAction
        from autoware_carla_scenario.entity import registry

        monkeypatch.setattr(registry, "_VEHICLES", {})
        monkeypatch.setattr(registry, "_PEDESTRIANS", {})
        RoutingAction("npc1", Lanelet2Pose(lanelet_id=3, s=7.0)).execute(
            object()  # type: ignore[arg-type]
        )
        assert "entity 'npc1' not found" in caplog.text
