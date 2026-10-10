"""Poses and progress measured along a logical scenario's route.

On the synthetic crossroads of :mod:`._route_maps`, the route used throughout
is the W approach straight through the junction:

* ``W_IN1 W_IN2 J(W->E) E_OUT1``, starting 40 m into ``W_IN1`` (x = -70) --
  route s 0 -- entering the junction at route s 60 (x = -10), leaving it at 80
  (x = 10), and ending at 130 (x = 60);
* the lane on its right (y = -5.25) is ``W_OUT_LANE*``, the opposite road on
  its left (y = +1.75) ``W_BACK*``; the W leg's crosswalk is at x = -15 and
  the E leg's at x = 15.

The map is loaded as the :class:`MapManager`'s, with the Lanelet2 and CARLA
frames aligned except for CARLA's flipped y.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    ComparisonRule,
    FollowTrajectoryAction,
    Lanelet2Pose,
    MapPose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteProgressCondition,
    RouteRoadsidePose,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryVertex,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity
from autoware_carla_scenario.route import (
    RouteMatch,
    RouteSegmentMatch,
    clear_scenario_route,
    parse_route_search,
    set_scenario_route,
)
from autoware_carla_scenario.route.frame import RouteFrame, ego_placement
from autoware_carla_scenario.route.positions import resolve_route_pose
from autoware_carla_scenario.route.search import find_route_matches
from autoware_carla_scenario.trajectory.resolve import resolve_trajectory

from . import _route_maps as rm
from .test_follow_trajectory_action import _Actor, _Entity, _run, _World

_STRAIGHT = {
    "segments": [
        {"kind": "lane", "length": {"max": 60}},
        {"kind": "junction", "turn": "straight", "traffic_light": "yes"},
        {"kind": "lane", "length": {"max": 50}},
    ]
}


def _approx_pose(
    lanelet_id: int, s: float, t: float = 0.0, heading: float = 0.0
) -> Any:
    """A Lanelet2 pose compared with its s approximately."""
    return Lanelet2Pose(lanelet_id, pytest.approx(s), t, heading)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def road() -> tuple[Any, Any]:
    lanelet_map = rm.crossroads()
    return lanelet_map, rm.routing_graph(lanelet_map)


@pytest.fixture(scope="module")
def straight(road: tuple[Any, Any]) -> RouteMatch:
    (match,) = find_route_matches(parse_route_search(_STRAIGHT), *road)
    return match


@pytest.fixture
def frame(road: tuple[Any, Any], straight: RouteMatch) -> RouteFrame:
    return RouteFrame(straight, *road)


@pytest.fixture
def loaded(road: tuple[Any, Any], straight: RouteMatch) -> Iterator[None]:
    """The crossroads as the loaded map, and the straight route as the scenario's."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map, manager._routing_graph = road
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    set_scenario_route(straight, *road)
    yield
    clear_scenario_route()
    MapManager._instance = saved


def _one_way(road: tuple[Any, Any]) -> RouteFrame:
    match = RouteMatch(
        (rm.ONE_WAY1, rm.ONE_WAY2),
        0.0,
        60.0,
        (RouteSegmentMatch("lane", 0.0, 120.0, (rm.ONE_WAY1, rm.ONE_WAY2)),),
    )
    return RouteFrame(match, *road)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


class TestTheFrame:
    def test_route_s_is_metres_from_where_the_route_starts(
        self, frame: RouteFrame
    ) -> None:
        assert frame.length == pytest.approx(130.0)
        assert frame.locate(0.0) == (rm.W_IN1, pytest.approx(40.0))
        assert frame.locate(65.0) == (rm.J[("W", "E")], pytest.approx(5.0))
        x, y, heading = frame.point(65.0)
        assert (x, y, heading) == pytest.approx((-5.0, -1.75, 0.0), abs=1e-6)

    def test_before_its_start_and_past_its_end_the_lane_goes_on(
        self, frame: RouteFrame
    ) -> None:
        assert frame.locate(-30.0) == (rm.W_IN1, pytest.approx(10.0))
        assert frame.locate(150.0) == (rm.E_OUT2, pytest.approx(20.0))
        with pytest.raises(ValueError, match="no lanelet precedes"):
            frame.locate(-50.0)

    def test_projection_is_abreast_whichever_lane(self, frame: RouteFrame) -> None:
        assert frame.project(-30.0, -1.75)[0] == pytest.approx(40.0)
        # On the lane beside the route: the same route s.
        s, gap = frame.project(-30.0, -5.25)
        assert s == pytest.approx(40.0) and gap == pytest.approx(3.5)

    def test_ego_placement(self, frame: RouteFrame) -> None:
        spawn, goal = ego_placement(frame, 5.0, True, 10.0)
        assert spawn == (rm.W_IN1, pytest.approx(45.0))
        assert goal == (rm.E_OUT1, pytest.approx(40.0))
        with pytest.raises(ValueError, match="past the end"):
            ego_placement(frame, 200.0, False, 0.0)
        with pytest.raises(ValueError, match="not ahead of the spawn"):
            ego_placement(frame, 125.0, True, 10.0)

    def test_a_lanelet_the_map_lacks(self, road: tuple[Any, Any]) -> None:
        match = RouteMatch(
            (999999,), 0.0, 1.0, (RouteSegmentMatch("lane", 0, 1, (999999,)),)
        )
        with pytest.raises(ValueError, match="does not have"):
            RouteFrame(match, *road)


# ---------------------------------------------------------------------------
# Each route pose
# ---------------------------------------------------------------------------


class TestRouteLanePose:
    def test_from_an_anchor(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteLanePose(ds=-15.0, anchor="junction:0:entry"), frame
        )
        assert pose == _approx_pose(rm.W_IN2, 35.0, 0.0, 0.0)

    def test_from_the_ego(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteLanePose(ds=5.0, offset=0.5, yaw=0.1), frame, 20.0
        )
        assert pose == _approx_pose(rm.W_IN2, 15.0, 0.5, 0.1)

    def test_across_lanes(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteLanePose(ds=10.0, d_lane=-1, anchor="start"), frame
        )
        assert isinstance(pose, Lanelet2Pose)
        assert pose.lanelet_id == rm.W_OUT_LANE2 and pose.s == pytest.approx(0.0)
        with pytest.raises(ValueError, match="no lane to its left"):
            resolve_route_pose(RouteLanePose(d_lane=1, anchor="start"), frame)

    def test_without_an_anchor_it_needs_the_ego(self, frame: RouteFrame) -> None:
        with pytest.raises(ValueError, match="measured from the ego"):
            resolve_route_pose(RouteLanePose(ds=1.0), frame)

    def test_an_anchor_the_route_lacks(self, frame: RouteFrame) -> None:
        with pytest.raises(ValueError, match="no junction 1"):
            resolve_route_pose(RouteLanePose(anchor="junction:1:entry"), frame)

    def test_the_pose_refuses_a_bad_value(self) -> None:
        with pytest.raises(ValueError, match="whole number"):
            RouteLanePose(d_lane=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not a route anchor"):
            RouteLanePose(anchor="middle")


class TestRouteOppositePose:
    def test_the_lane_abreast_facing_its_own_way(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(RouteOppositePose(ds=0.0, anchor="start"), frame)
        # Route s 0 is x = -70: W_BACK1 runs x -60 -> -110.
        assert pose == _approx_pose(rm.W_BACK1, 10.0, 0.0, 0.0)

    def test_a_lane_the_opposite_road_lacks(self, frame: RouteFrame) -> None:
        with pytest.raises(ValueError, match="has 1 lane"):
            resolve_route_pose(RouteOppositePose(lane=2, anchor="start"), frame)

    def test_a_one_way_road(self, road: tuple[Any, Any]) -> None:
        with pytest.raises(ValueError, match="no opposite-direction lane"):
            resolve_route_pose(RouteOppositePose(anchor="start"), _one_way(road))


class TestRouteCrossingPose:
    def test_traffic_that_crosses_first(self, frame: RouteFrame) -> None:
        # From the ego's left: the N road, whose straight way crosses it.
        pose = resolve_route_pose(
            RouteCrossingPose(junction=0, approach="left", distance=-20.0), frame
        )
        assert pose == _approx_pose(rm.N_IN, 80.0, 0.0, 0.0)

    def test_by_turn(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteCrossingPose(approach="opposite", turn="left", distance=3.0), frame
        )
        assert pose == _approx_pose(rm.J[("E", "S")], 3.0)
        right = resolve_route_pose(
            RouteCrossingPose(approach="right", turn="right", distance=0.0), frame
        )
        assert right.lanelet_id == rm.J[("S", "E")]  # type: ignore[union-attr]

    def test_a_junction_the_route_lacks(self, frame: RouteFrame) -> None:
        with pytest.raises(ValueError, match="no junction 1"):
            resolve_route_pose(RouteCrossingPose(junction=1), frame)

    def test_a_bad_approach(self) -> None:
        with pytest.raises(ValueError, match="approach"):
            RouteCrossingPose(approach="behind")


class TestRouteCrosswalkPose:
    def test_at_the_kerb_on_the_right_facing_across(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteCrosswalkPose(junction=0, leg="entry", side="right", yaw=0.0), frame
        )
        assert isinstance(pose, MapPose)
        assert (pose.x, pose.y) == pytest.approx((-15.0, -7.0))
        assert pose.yaw == pytest.approx(math.pi / 2)

    def test_on_the_pavement_and_part_way_across(self, frame: RouteFrame) -> None:
        behind = resolve_route_pose(
            RouteCrosswalkPose(leg="entry", side="right", along=-2.0), frame
        )
        assert (behind.x, behind.y) == pytest.approx((-15.0, -9.0))  # type: ignore[union-attr]
        across = resolve_route_pose(
            RouteCrosswalkPose(leg="exit", side="left", along=3.0), frame
        )
        assert (across.x, across.y) == pytest.approx((15.0, 4.0))  # type: ignore[union-attr]
        assert across.yaw is None  # type: ignore[union-attr]

    def test_a_leg_with_no_crosswalk(self, road: tuple[Any, Any]) -> None:
        (right,) = find_route_matches(
            parse_route_search(
                {
                    "segments": [
                        {"kind": "lane", "length": {"max": 60}},
                        {"kind": "junction", "turn": "right", "traffic_light": "yes"},
                        {"kind": "lane", "length": {"max": 20}},
                    ]
                }
            ),
            *road,
        )
        with pytest.raises(ValueError, match="no crosswalk across its exit leg"):
            resolve_route_pose(RouteCrosswalkPose(leg="exit"), RouteFrame(right, *road))


class TestRouteRoadsidePose:
    def test_beyond_every_lane_on_that_side(self, frame: RouteFrame) -> None:
        right = resolve_route_pose(
            RouteRoadsidePose(ds=10.0, side="right", kerb_distance=1.0, anchor="start"),
            frame,
        )
        # Past the lane beside the route, whose outer edge is y = -7.
        assert (right.x, right.y) == pytest.approx((-60.0, -8.0))  # type: ignore[union-attr]
        left = resolve_route_pose(
            RouteRoadsidePose(ds=10.0, side="left", kerb_distance=1.0, anchor="start"),
            frame,
        )
        # Past the opposite road, whose outer edge is y = 3.5.
        assert (left.x, left.y) == pytest.approx((-60.0, 4.5))  # type: ignore[union-attr]

    def test_on_the_road_with_a_negative_distance(self, frame: RouteFrame) -> None:
        pose = resolve_route_pose(
            RouteRoadsidePose(
                side="right", kerb_distance=-0.5, anchor="start", yaw=0.0
            ),
            frame,
        )
        assert (pose.x, pose.y) == pytest.approx((-70.0, -6.5))  # type: ignore[union-attr]
        assert pose.yaw == pytest.approx(0.0)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Progress, and trajectories placed when the action starts
# ---------------------------------------------------------------------------


class _Ego(_Actor):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x=x, y=-y)  # CARLA's y is South
        self.attributes = {"role_name": "Ego"}

    def move(self, x: float, y: float) -> None:
        self.x, self.y = x, -y


class _RouteWorld(_World):
    def __init__(self, *actors: Any) -> None:
        self.actors = list(actors)

    def get_actors(self) -> list[Any]:
        return self.actors


class TestRouteProgress:
    def test_progress_across_a_lane_change(self, loaded: None) -> None:
        ego = _Ego(-30.0, -1.75)
        world = _RouteWorld(ego)
        condition = RouteProgressCondition(
            value=-15.0, anchor="junction:0:entry", label="near_junction"
        )
        assert condition.check(world, 0.0) is None
        assert condition.progress == pytest.approx(40.0)
        # Onto the lane beside the route, and 10 m on.
        ego.move(-20.0, -5.25)
        result = condition.check(world, 1.0)
        assert result is not None and result.passed
        assert condition.progress == pytest.approx(50.0)

    def test_rules_and_entities(self, loaded: None) -> None:
        ego = _Ego(-30.0, -1.75)
        npc = _Actor(x=0.0, y=1.75)  # in the junction, westbound
        world = _RouteWorld(ego, npc)
        assert (
            RouteProgressCondition(
                value=50.0, rule=ComparisonRule.LESS_THAN, label="x"
            ).check(world, 0.0)
            is not None
        )
        assert (
            RouteProgressCondition("npc1", value=69.0, label="npc").check(world, 0.0)
            is not None
        )
        assert RouteProgressCondition("npc9", label="missing").check(world, 0.0) is None

    def test_without_a_route(self) -> None:
        clear_scenario_route()
        with pytest.raises(ValueError, match="has none"):
            RouteProgressCondition(label="x").check(_RouteWorld(), 0.0)

    def test_a_bad_anchor_is_refused_up_front(self) -> None:
        with pytest.raises(ValueError, match="not a route anchor"):
            RouteProgressCondition(anchor="junction:0", label="x")


class TestTrajectoriesOnTheRoute:
    def test_vertices_placed_against_the_ego(self, loaded: None) -> None:
        path = Trajectory(
            "cut_in",
            [
                TrajectoryVertex(RouteLanePose(ds=10.0, d_lane=-1)),
                TrajectoryVertex(RouteLanePose(ds=30.0)),
                TrajectoryVertex(RouteCrossingPose(approach="left", distance=-5.0)),
            ],
        )
        assert path.is_relative
        ego = CarlaWorldPose(-30.0, 1.75, 0.0, 0.0)
        resolved = resolve_trajectory(path, reference=lambda ref: ego)
        assert (resolved.xs[0], resolved.ys[0]) == pytest.approx((-20.0, 5.25))
        assert (resolved.xs[1], resolved.ys[1]) == pytest.approx((0.0, 1.75))
        assert (resolved.xs[2], resolved.ys[2]) == pytest.approx((-1.75, -15.0))

    def test_a_vertex_the_map_cannot_give_names_itself(self, loaded: None) -> None:
        path = Trajectory(
            "p",
            [
                TrajectoryVertex(RouteLanePose(anchor="start")),
                TrajectoryVertex(RouteOppositePose(lane=3, anchor="start")),
            ],
        )
        with pytest.raises(ValueError, match="vertex 1: .*no lane 3"):
            resolve_trajectory(path, reference=lambda ref: CarlaWorldPose(0, 0, 0, 0))


class TestAppearOnStart:
    """A hidden entity enters on its first vertex when its action starts."""

    @staticmethod
    def _path() -> Trajectory:
        return Trajectory(
            "enter",
            [
                TrajectoryVertex(RouteLanePose(ds=-40.0, anchor="junction:0:entry")),
                TrajectoryVertex(
                    RouteLanePose(ds=-20.0, anchor="junction:0:entry"),
                    RouteProgressCondition(value=30.0, label="ego_at_30"),
                ),
                TrajectoryVertex(RouteLanePose(ds=10.0, anchor="junction:0:entry")),
            ],
        )

    def test_it_is_refused_with_hiding_outside(self) -> None:
        with pytest.raises(ValueError, match="choose one"):
            FollowTrajectoryAction(
                "npc1",
                self._path(),
                appear_on_start=True,
                hidden_outside_trajectory=True,
            )

    def test_a_vehicle_appears_moving_then_follows(self, loaded: None) -> None:
        actor = _Actor(x=0.0, y=0.0)
        actor.z = -500.0
        actor.physics = False
        entity = _Entity(actor)
        register_entity("npc1", entity)
        ego = _Ego(-60.0, -1.75)  # route s 10
        try:
            action = FollowTrajectoryAction(
                "npc1",
                self._path(),
                following_mode=TrajectoryFollowingMode.FOLLOW,
                speed=5.0,
                appear_on_start=True,
            )
            world = _RouteWorld(ego, actor)
            action.tick(world, 0.0)
            assert actor.physics is True
            # The first vertex: 40 m before the junction, x = -50, facing East.
            assert (actor.x, actor.y) == pytest.approx((-50.0, 1.75))
            assert actor.yaw == pytest.approx(0.0)
            assert actor.speed == pytest.approx(5.0)
            # It drives to the second vertex and waits for the ego there.
            elapsed = 0.05
            for _ in range(200):
                action.tick(world, elapsed)
                actor.step(0.05, driven=True)
                elapsed += 0.05
            assert action.held_vertex == 1
            assert actor.x == pytest.approx(-30.0, abs=1.5)
            ego.move(-40.0, -1.75)  # route s 30: the vertex is departed
            for _ in range(200):
                action.tick(world, elapsed)
                actor.step(0.05, driven=True)
                elapsed += 0.05
            assert action.held_vertex is None
            assert actor.x > -5.0
        finally:
            unregister_entity("npc1")

    def test_a_walker_appears_walking(self, loaded: None) -> None:
        walker = _Actor(type_id="walker.pedestrian.0015")
        walker.physics = False
        register_entity("walker1", _Entity(walker))
        try:
            path = Trajectory(
                "cross",
                [
                    TrajectoryVertex(RouteCrosswalkPose(leg="exit", side="right")),
                    TrajectoryVertex(
                        RouteCrosswalkPose(leg="exit", side="right", along=14.0)
                    ),
                ],
            )
            action = FollowTrajectoryAction(
                "walker1",
                path,
                following_mode=TrajectoryFollowingMode.FOLLOW,
                speed=1.2,
                appear_on_start=True,
            )
            action.tick(_RouteWorld(walker), 0.0)
            assert walker.physics is True
            # The E crosswalk's end on the ego's right, raised by half a height.
            assert (walker.x, walker.y, walker.z) == pytest.approx((15.0, 7.0, 0.9))
            assert walker.walker_controls and walker.walker_controls[0].speed > 0.0
        finally:
            unregister_entity("walker1")


def test_the_simulated_vehicle_helpers_are_shared() -> None:
    """The action tests above drive the same fake as test_follow_trajectory_action."""
    assert callable(_run)


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def staggered() -> tuple[Any, Any, RouteFrame]:
    """One lanelet eastbound x 0..100, the other way split at x 120 and 40."""
    builder = rm._Builder()
    half = rm._HALF
    builder.lanelet(1, rm._line((0.0, -half), (100.0, -half)))
    builder.lanelet(2, rm._line((100.0, -half), (200.0, -half)))
    builder.lanelet(11, rm._line((200.0, half), (120.0, half)))
    builder.lanelet(12, rm._line((120.0, half), (40.0, half)))
    builder.lanelet(13, rm._line((40.0, half), (-60.0, half)))
    lanelet_map = builder.map
    graph = rm.routing_graph(lanelet_map)
    (match,) = find_route_matches(
        parse_route_search({"segments": [{"kind": "lane", "opposite_lane": "yes"}]}),
        lanelet_map,
        graph,
    )
    return lanelet_map, graph, RouteFrame(match, lanelet_map, graph)


class TestOppositeLaneletsSplitElsewhere:
    @pytest.mark.parametrize(
        ("ds", "lanelet_id", "s"),
        [(10.0, 13, 30.0), (50.0, 12, 70.0), (150.0, 11, 50.0)],
    )
    def test_the_opposite_pose_is_abreast(
        self,
        staggered: tuple[Any, Any, RouteFrame],
        ds: float,
        lanelet_id: int,
        s: float,
    ) -> None:
        pose = resolve_route_pose(
            RouteOppositePose(ds=ds, anchor="start"), staggered[2]
        )
        assert pose == _approx_pose(lanelet_id, s)

    def test_the_roadside_is_abreast(
        self, staggered: tuple[Any, Any, RouteFrame]
    ) -> None:
        pose = resolve_route_pose(
            RouteRoadsidePose(ds=10.0, side="left", kerb_distance=1.0, anchor="start"),
            staggered[2],
        )
        assert (pose.x, pose.y) == pytest.approx((10.0, 4.5))  # type: ignore[union-attr]


class TestProgressIsNotStale:
    def test_a_teleported_entity_is_found_at_once(self, loaded: None) -> None:
        ego = _Ego(-60.0, -1.75)  # route s 10
        world = _RouteWorld(ego)
        condition = RouteProgressCondition(value=1000.0, label="far")
        condition.check(world, 0.0)
        assert condition.progress == pytest.approx(10.0)
        ego.move(50.0, -1.75)  # route s 120, in one tick
        condition.check(world, 0.1)
        assert condition.progress == pytest.approx(120.0)

    def test_a_hidden_entity_has_no_progress(self, loaded: None) -> None:
        npc = _Actor(x=50.0, y=1.75)  # route s 120 ...
        world = _RouteWorld(npc)
        condition = RouteProgressCondition("npc1", value=0.0, label="npc")
        assert condition.check(world, 0.0) is not None
        npc.z = -500.0  # ... parked under the map
        assert condition.check(world, 0.1) is None
        assert condition.progress is None
