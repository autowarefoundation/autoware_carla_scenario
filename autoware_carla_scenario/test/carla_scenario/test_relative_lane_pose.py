"""Trajectory vertices given relative to an entity, in lane coordinates.

A :class:`RelativeLanePose` is OpenSCENARIO's ``RelativeLanePosition``: so many
metres along the reference entity's lane, so many lanes across, so far from the
centreline.  These tests work on a small synthetic Lanelet2 map --

* two lanes of three 10 m lanelets each, running East: the right lane
  ``R0 R1 R2`` (centreline y = 0) and the left lane ``L0 L1 L2`` (y = 3.5),
  lane-changeable between them except alongside ``R0``, where the line is solid;
* a branch ``B`` that leaves ``R1``'s end bending 45 degrees to the right, with
  a lower id than ``R2`` so that choosing by id alone would take it --

first the arithmetic, then whole trajectories placed in CARLA coordinates, then
the action following one against the simulated vehicle of
``test_follow_trajectory_action``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.core
import lanelet2.routing
import lanelet2.traffic_rules

from autoware_carla_scenario import (
    CarlaWorldPose,
    ElapsedTimeCondition,
    FollowTrajectoryAction,
    Lanelet2Pose,
    RelativeLanePose,
    Trajectory,
    TrajectoryTiming,
    TrajectoryVertex,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity
from autoware_carla_scenario.trajectory.relative_lane import (
    advance_along_lane,
    reference_lane_pose,
    relative_lane_pose,
    shift_lanes,
)
from autoware_carla_scenario.trajectory.resolve import resolve_trajectory

from .test_follow_trajectory_action import _Actor, _Entity, _run, _World

B, R0, R1, R2, L0, L1, L2 = 100, 101, 102, 103, 111, 112, 113

_LANE = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes"}


def _map() -> Any:
    """The two lanes and the branch described in the module docstring."""
    from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

    next_id = iter(range(1000, 100000))

    def points(coords: list[tuple[float, float]]) -> list[Any]:
        return [Point3d(next(next_id), x, y, 0.0) for x, y in coords]

    # The three lines that bound the two lanes, as shared points every 10 m.
    lines = {y: points([(10.0 * i, y) for i in range(4)]) for y in (-1.75, 1.75, 5.25)}

    def bound(y: float, i: int, subtype: str = "dashed") -> Any:
        return LineString3d(
            next(next_id),
            lines[y][i : i + 2],
            AttributeMap({"type": "line_thin", "subtype": subtype}),
        )

    lanelet_map = LaneletMap()
    middle = [bound(1.75, i, "solid" if i == 0 else "dashed") for i in range(3)]
    for i, (right_id, left_id) in enumerate(zip((R0, R1, R2), (L0, L1, L2))):
        lanelet_map.add(
            Lanelet(right_id, middle[i], bound(-1.75, i), AttributeMap(_LANE))
        )
        lanelet_map.add(
            Lanelet(left_id, bound(5.25, i), middle[i], AttributeMap(_LANE))
        )

    # The branch: from R1's end, 10 m at 45 degrees to the right.
    step = 10.0 / math.sqrt(2.0)
    start_left, start_right = lines[1.75][2], lines[-1.75][2]
    branch_left = LineString3d(
        next(next_id),
        [start_left, *points([(20.0 + step, 1.75 - step)])],
    )
    branch_right = LineString3d(
        next(next_id),
        [start_right, *points([(20.0 + step, -1.75 - step)])],
    )
    lanelet_map.add(Lanelet(B, branch_left, branch_right, AttributeMap(_LANE)))
    return lanelet_map


def _graph(lanelet_map: Any) -> Any:
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    return lanelet2.routing.RoutingGraph(lanelet_map, rules)


@pytest.fixture(scope="module")
def road() -> tuple[Any, Any]:
    lanelet_map = _map()
    return lanelet_map, _graph(lanelet_map)


@pytest.fixture
def loaded(road: tuple[Any, Any]) -> Iterator[None]:
    """The synthetic map as the loaded one, Lanelet2 and CARLA frames aligned."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map, manager._routing_graph = road
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


# ---------------------------------------------------------------------------
# The pose itself
# ---------------------------------------------------------------------------


class TestThePose:
    def test_defaults_are_where_the_entity_itself_is(self) -> None:
        pose = RelativeLanePose()
        assert (pose.ds, pose.offset, pose.d_lane) == (0.0, 0.0, 0)
        assert pose.yaw is None and pose.entity_ref is None

    def test_d_lane_is_a_whole_number(self) -> None:
        assert RelativeLanePose(d_lane=2.0).d_lane == 2  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="d_lane"):
            RelativeLanePose(d_lane=1.5)  # type: ignore[arg-type]

    def test_a_trajectory_with_one_is_relative(self) -> None:
        mixed = Trajectory(
            "mixed",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0)),
                TrajectoryVertex(RelativeLanePose(10.0)),
            ],
        )
        assert mixed.is_relative
        absolute = Trajectory(
            "absolute",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0)),
                TrajectoryVertex(CarlaWorldPose(1.0, 0.0, 0.0)),
            ],
        )
        assert not absolute.is_relative


# ---------------------------------------------------------------------------
# The arithmetic, on the map frame
# ---------------------------------------------------------------------------


class TestWhereTheReferenceIs:
    def test_a_point_on_a_lane(self, road: tuple[Any, Any]) -> None:
        pose = reference_lane_pose(road[0], 15.0, 0.5, 0.0)
        assert pose.lanelet_id == R1
        assert pose.s == pytest.approx(5.0)
        assert pose.t == pytest.approx(0.5)

    def test_where_lanelets_overlap_the_heading_chooses(
        self, road: tuple[Any, Any]
    ) -> None:
        # Just past the fork, inside both R2 and the branch.
        assert reference_lane_pose(road[0], 20.5, -0.2, 0.0).lanelet_id == R2
        assert reference_lane_pose(road[0], 20.5, -0.2, -math.pi / 4).lanelet_id == B


class TestAlongTheLane:
    def test_within_one_lanelet(self, road: tuple[Any, Any]) -> None:
        assert advance_along_lane(*road, R0, 2.0, 5.0) == (R0, pytest.approx(7.0))

    def test_into_the_lanelet_that_follows(self, road: tuple[Any, Any]) -> None:
        assert advance_along_lane(*road, L0, 5.0, 12.0) == (L1, pytest.approx(7.0))
        assert advance_along_lane(*road, L0, 5.0, 22.0) == (L2, pytest.approx(7.0))

    def test_at_a_fork_the_straightest_is_taken(self, road: tuple[Any, Any]) -> None:
        # B has the lower id; R2 goes straight on.
        assert advance_along_lane(*road, R1, 5.0, 10.0) == (R2, pytest.approx(5.0))

    def test_a_negative_ds_goes_back(self, road: tuple[Any, Any]) -> None:
        assert advance_along_lane(*road, R1, 2.0, -5.0) == (R0, pytest.approx(7.0))
        assert advance_along_lane(*road, R2, 1.0, -21.0) == (R0, pytest.approx(0.0))

    def test_off_the_end_of_the_map(self, road: tuple[Any, Any]) -> None:
        with pytest.raises(ValueError, match=f"past the end of lanelet {L2}"):
            advance_along_lane(*road, L0, 0.0, 31.0)
        with pytest.raises(ValueError, match=f"before the start of lanelet {R0}"):
            advance_along_lane(*road, R1, 2.0, -13.0)


class TestAcrossLanes:
    def test_one_lane_to_the_left_and_back(self, road: tuple[Any, Any]) -> None:
        assert shift_lanes(*road, R1, 4.0, 1) == (L1, pytest.approx(4.0))
        assert shift_lanes(*road, L1, 4.0, -1) == (R1, pytest.approx(4.0))

    def test_across_a_solid_line_too(self, road: tuple[Any, Any]) -> None:
        # Not lane-changeable, but still the next lane over.
        assert shift_lanes(*road, R0, 4.0, 1) == (L0, pytest.approx(4.0))

    def test_no_lane_there(self, road: tuple[Any, Any]) -> None:
        with pytest.raises(ValueError, match=f"lanelet {R1} has no lane to its right"):
            shift_lanes(*road, R1, 4.0, -1)
        with pytest.raises(
            ValueError, match=f"lanelet {L1} has no lane to its left .after 1 of 2"
        ):
            shift_lanes(*road, R1, 4.0, 2)


class TestAltogether:
    def test_ds_then_d_lane_then_offset(self, road: tuple[Any, Any]) -> None:
        reference = Lanelet2Pose(R0, 5.0, t=-0.4, heading=0.2)
        pose = relative_lane_pose(
            *road, reference, RelativeLanePose(ds=12.0, offset=0.5, d_lane=1, yaw=0.1)
        )
        # The reference's own offset and heading do not carry over.
        assert (pose.lanelet_id, pose.t, pose.heading) == (L1, 0.5, 0.1)
        assert pose.s == pytest.approx(7.0)

    def test_an_unstated_yaw_is_along_the_lane(self, road: tuple[Any, Any]) -> None:
        pose = relative_lane_pose(*road, Lanelet2Pose(R0, 5.0), RelativeLanePose(1.0))
        assert pose.heading == 0.0


# ---------------------------------------------------------------------------
# Whole trajectories, in CARLA coordinates
# ---------------------------------------------------------------------------


def _at(x: float, y: float, yaw: float = 0.0) -> Any:
    """A reference lookup that finds every entity at the CARLA pose given."""
    return lambda entity_ref: CarlaWorldPose(x, y, 0.0, yaw=yaw)


class TestResolution:
    def test_offset_left_is_north_which_is_carla_minus_y(self, loaded: None) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose(0.0, offset=1.0)),
                TrajectoryVertex(RelativeLanePose(5.0, offset=-1.0)),
            ],
        )
        resolved = resolve_trajectory(trajectory, reference=_at(2.0, 0.0))
        assert (resolved.xs[0], resolved.ys[0]) == (
            pytest.approx(2.0),
            pytest.approx(-1.0),
        )
        assert (resolved.xs[1], resolved.ys[1]) == (
            pytest.approx(7.0),
            pytest.approx(1.0),
        )

    def test_d_lane_left_is_carla_minus_y(self, loaded: None) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose()),
                TrajectoryVertex(RelativeLanePose(15.0, d_lane=1)),
            ],
        )
        resolved = resolve_trajectory(trajectory, reference=_at(2.0, 0.0))
        assert (resolved.xs[1], resolved.ys[1]) == (
            pytest.approx(17.0),
            pytest.approx(-3.5),
        )

    def test_a_stated_yaw_is_relative_to_the_lane(self, loaded: None) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose(yaw=math.radians(10.0))),
                TrajectoryVertex(RelativeLanePose(5.0)),
            ],
        )
        resolved = resolve_trajectory(trajectory, reference=_at(2.0, 0.0))
        # Ten degrees to the left is ten degrees anticlockwise: -10 in CARLA.
        assert resolved.yaws[0] == pytest.approx(-10.0)
        assert resolved.yaws[1] is None

    def test_mixed_with_absolute_vertices(self, loaded: None) -> None:
        trajectory = Trajectory(
            "mixed",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 0.0), 0.0),
                TrajectoryVertex(RelativeLanePose(10.0, d_lane=1), 1.0),
                TrajectoryVertex(Lanelet2Pose(L2, 5.0), 2.0),
            ],
        )
        resolved = resolve_trajectory(trajectory, reference=_at(5.0, 0.0))
        assert list(zip(resolved.xs, resolved.ys)) == [
            (0.0, 0.0),
            (pytest.approx(15.0), pytest.approx(-3.5)),
            (pytest.approx(25.0), pytest.approx(-3.5)),
        ]
        assert resolved.times == [0.0, 1.0, 2.0]

    def test_each_entity_ref_is_looked_up(self, loaded: None) -> None:
        asked: list[Any] = []

        def reference(entity_ref: Any) -> CarlaWorldPose:
            asked.append(entity_ref)
            x = 2.0 if entity_ref is None else 22.0
            return CarlaWorldPose(x, 0.0, 0.0)

        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose()),
                TrajectoryVertex(RelativeLanePose(1.0)),
                TrajectoryVertex(RelativeLanePose(entity_ref="ego")),
            ],
        )
        resolved = resolve_trajectory(trajectory, reference=reference)
        assert resolved.xs == [
            pytest.approx(2.0),
            pytest.approx(3.0),
            pytest.approx(22.0),
        ]
        # Once per reference entity, whatever the number of vertices.
        assert asked == [None, "ego"]

    def test_an_error_names_the_vertex(self, loaded: None) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose()),
                TrajectoryVertex(RelativeLanePose(d_lane=-1)),
            ],
        )
        with pytest.raises(ValueError, match="'t', vertex 1: d_lane=-1"):
            resolve_trajectory(trajectory, reference=_at(2.0, 0.0))

    def test_no_reference_no_relative_pose(self, loaded: None) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(RelativeLanePose()),
                TrajectoryVertex(RelativeLanePose(1.0)),
            ],
        )
        with pytest.raises(ValueError, match="no reference pose"):
            resolve_trajectory(trajectory)


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


@pytest.fixture
def npc(loaded: None) -> Iterator[_Entity]:
    registered = _Entity(_Actor(x=5.0, y=0.0))
    register_entity("npc1", registered)
    yield registered
    unregister_entity("npc1")


def _overtake() -> Trajectory:
    """Pull out into the left lane over 10 m and drive on in it, at 10 m/s."""
    return Trajectory(
        "overtake",
        [
            TrajectoryVertex(RelativeLanePose(), 0.0),
            TrajectoryVertex(RelativeLanePose(10.0, d_lane=1), 1.0),
            TrajectoryVertex(RelativeLanePose(20.0, d_lane=1), 2.0),
        ],
    )


class TestTheAction:
    def test_it_follows_a_path_placed_where_the_entity_started(
        self, npc: _Entity
    ) -> None:
        action = FollowTrajectoryAction("npc1", _overtake(), TrajectoryTiming())
        _run(action, npc.actor, 2.5)
        assert action.finished
        assert (npc.actor.x, npc.actor.y) == (pytest.approx(25.0), pytest.approx(-3.5))

    def test_a_relative_vertex_can_hold_it(self, npc: _Entity) -> None:
        """A gate on a relative vertex holds the entity where the vertex was placed."""
        action = FollowTrajectoryAction(
            "npc1",
            _overtake().gated({1: ElapsedTimeCondition(2.0, label="go")}),
            TrajectoryTiming(),
        )
        _run(action, npc.actor, 1.5)
        # Placed against where npc1 started (5 m along R0): 10 m on, one lane left.
        assert (npc.actor.x, npc.actor.y) == (pytest.approx(15.0), pytest.approx(-3.5))
        assert action.held_vertex == 1
        _run(action, npc.actor, 2.0, start=1.5)
        assert action.finished
        assert (npc.actor.x, npc.actor.y) == (pytest.approx(25.0), pytest.approx(-3.5))

    def test_it_is_placed_once_at_the_start_not_every_tick(self, npc: _Entity) -> None:
        ego = _Entity(_Actor(x=2.0, y=0.0))
        register_entity("ego", ego)
        try:
            ahead_of_ego = Trajectory(
                "ahead",
                [
                    TrajectoryVertex(CarlaWorldPose(5.0, 0.0, 0.0), 0.0),
                    TrajectoryVertex(RelativeLanePose(20.0, entity_ref="ego"), 1.0),
                ],
            )
            action = FollowTrajectoryAction("npc1", ahead_of_ego, TrajectoryTiming())
            _run(action, npc.actor, 0.5)
            ego.actor.x = 12.0  # the reference moves on; the path does not
            _run(action, npc.actor, 1.0, start=0.5)
        finally:
            unregister_entity("ego")
        assert action.finished
        assert npc.actor.x == pytest.approx(22.0)

    def test_it_waits_for_a_reference_that_is_not_there_yet(self, npc: _Entity) -> None:
        trajectory = Trajectory(
            "behind",
            [
                TrajectoryVertex(CarlaWorldPose(5.0, 0.0, 0.0), 0.0),
                TrajectoryVertex(RelativeLanePose(-2.0, entity_ref="late"), 1.0),
            ],
        )
        action = FollowTrajectoryAction("npc1", trajectory, TrajectoryTiming())
        _run(action, npc.actor, 0.2)
        assert npc.actor.teleports == 0
        assert npc.released == 0

        late = _Entity(_Actor(x=20.0, y=0.0))
        register_entity("late", late)
        try:
            _run(action, npc.actor, 1.5, start=0.2)
        finally:
            unregister_entity("late")
        assert action.finished
        assert npc.actor.x == pytest.approx(18.0)

    def test_a_repeat_waits_for_a_reference_gone_since_the_last_run(
        self, npc: _Entity
    ) -> None:
        """The last run's arrival does not end a run still waiting to start."""
        trajectory = Trajectory(
            "hop",
            [
                TrajectoryVertex(RelativeLanePose(entity_ref="lead"), 0.0),
                TrajectoryVertex(RelativeLanePose(5.0, entity_ref="lead"), 0.5),
            ],
        )
        action = FollowTrajectoryAction(
            "npc1", trajectory, TrajectoryTiming(), once=False
        )
        world = _World()
        register_entity("lead", _Entity(_Actor(x=20.0, y=0.0)))
        elapsed = 0.0
        try:
            while not action.finished and elapsed < 2.0:
                action.tick(world, elapsed)
                npc.actor.step(0.05, driven=False)
                elapsed += 0.05
        finally:
            unregister_entity("lead")
        assert action.finished
        stopped = npc.actor.x
        # The lead is gone: the next runs wait, and nothing reports an end.
        for _ in range(20):
            action.tick(world, elapsed)
            npc.actor.step(0.05, driven=False)
            elapsed += 0.05
            assert not action.finished
        assert npc.actor.x == pytest.approx(stopped)

    def test_a_missing_reference_is_warned_about_once(
        self, npc: _Entity, caplog: pytest.LogCaptureFixture
    ) -> None:
        trajectory = Trajectory(
            "behind",
            [
                TrajectoryVertex(CarlaWorldPose(5.0, 0.0, 0.0), 0.0),
                TrajectoryVertex(RelativeLanePose(-2.0, entity_ref="nobody"), 1.0),
            ],
        )
        action = FollowTrajectoryAction("npc1", trajectory, TrajectoryTiming())
        with caplog.at_level("WARNING"):
            _run(action, npc.actor, 1.0)
        assert sum("'nobody'" in r.getMessage() for r in caplog.records) == 1

    def test_a_repeated_run_is_placed_afresh(self, npc: _Entity) -> None:
        trajectory = Trajectory(
            "hop",
            [
                TrajectoryVertex(RelativeLanePose(), 0.0),
                TrajectoryVertex(RelativeLanePose(5.0), 0.5),
            ],
        )
        action = FollowTrajectoryAction(
            "npc1", trajectory, TrajectoryTiming(), once=False
        )
        world = _World()
        elapsed = 0.0
        ends: list[float] = []
        while len(ends) < 2 and elapsed < 5.0:
            was = action.finished
            action.tick(world, elapsed)
            npc.actor.step(0.05, driven=False)
            elapsed += 0.05
            if action.finished and not was:
                ends.append(npc.actor.x)
        # Each run starts where the last one stopped and goes 5 m further.
        assert ends == [pytest.approx(10.0), pytest.approx(15.0)]
