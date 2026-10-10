"""The trajectory model: what a trajectory may be, and where it puts an entity.

Everything here is arithmetic on resolved CARLA world points -- the geometry
the follow-trajectory action asks every tick -- so it runs without a simulator.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryTimeCondition,
    TrajectoryTiming,
    TrajectoryVertex,
)
from autoware_carla_scenario.trajectory.model import ResolvedTrajectory
from autoware_carla_scenario.trajectory.resolve import (
    map_pose_to_carla,
    resolve_trajectory,
)


def _at(time: Optional[float]) -> Optional[TrajectoryTimeCondition]:
    """A vertex's time, as the condition it departs on."""
    return None if time is None else TrajectoryTimeCondition(time)


def _straight(times=(0.0, 1.0, 2.0), step=10.0, yaws=None) -> ResolvedTrajectory:
    """Vertices *step* m apart along CARLA +x."""
    count = len(times) if times is not None else 3
    return ResolvedTrajectory(
        xs=[step * index for index in range(count)],
        ys=[0.0] * count,
        zs=[0.0] * count,
        yaws=yaws if yaws is not None else [None] * count,
        times=list(times) if times is not None else None,
    )


# ---------------------------------------------------------------------------
# What a trajectory may be
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_a_bare_time_is_refused_saying_how_to_write_it(self) -> None:
        with pytest.raises(TypeError, match=r"advance=TrajectoryTimeCondition\(1.5\)"):
            TrajectoryVertex(MapPose(0.0, 0.0), 1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"advance=TrajectoryTimeCondition\(2.0\)"):
            TrajectoryVertex(MapPose(0.0, 0.0), time=2.0)  # type: ignore[call-arg]

    def test_its_time_is_its_time_condition(self) -> None:
        vertex = TrajectoryVertex(MapPose(0.0, 0.0), _at(1.5))
        assert vertex.time == 1.5 and vertex.gate is None
        assert vertex == TrajectoryVertex(MapPose(0.0, 0.0), _at(1.5))
        assert hash(vertex) == hash(TrajectoryVertex(MapPose(0.0, 0.0), _at(1.5)))
        assert TrajectoryVertex(MapPose(0.0, 0.0)).time is None

    def test_it_needs_two_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            Trajectory("t", [TrajectoryVertex(MapPose(0.0, 0.0), _at(0.0))])

    def test_timing_is_all_or_nothing(self) -> None:
        with pytest.raises(ValueError, match="every vertex"):
            Trajectory(
                "t",
                [
                    TrajectoryVertex(MapPose(0.0, 0.0), _at(0.0)),
                    TrajectoryVertex(MapPose(1.0, 0.0)),
                ],
            )

    def test_times_may_not_go_back(self) -> None:
        with pytest.raises(ValueError, match="decrease"):
            Trajectory(
                "t",
                [
                    TrajectoryVertex(MapPose(0.0, 0.0), _at(1.0)),
                    TrajectoryVertex(MapPose(1.0, 0.0), _at(0.5)),
                ],
            )

    def test_a_closed_trajectory_carries_no_times(self) -> None:
        with pytest.raises(ValueError, match="closed"):
            Trajectory(
                "t",
                [
                    TrajectoryVertex(MapPose(0.0, 0.0), _at(0.0)),
                    TrajectoryVertex(MapPose(1.0, 0.0), _at(1.0)),
                ],
                closed=True,
            )

    def test_is_timed(self) -> None:
        timed = Trajectory(
            "t",
            [
                TrajectoryVertex(MapPose(0.0, 0.0), _at(0.0)),
                TrajectoryVertex(MapPose(1.0, 0.0), _at(1.0)),
            ],
        )
        untimed = Trajectory(
            "t",
            [TrajectoryVertex(MapPose(0.0, 0.0)), TrajectoryVertex(MapPose(1.0, 0.0))],
        )
        assert timed.is_timed and not untimed.is_timed


class TestTiming:
    def test_relative_counts_from_the_action(self) -> None:
        timing = TrajectoryTiming(ReferenceContext.RELATIVE)
        assert timing.trajectory_time(12.0, action_start=10.0) == pytest.approx(2.0)

    def test_absolute_counts_from_the_scenario(self) -> None:
        timing = TrajectoryTiming(ReferenceContext.ABSOLUTE)
        assert timing.trajectory_time(12.0, action_start=10.0) == pytest.approx(12.0)

    def test_scale_and_offset_follow_openscenario(self) -> None:
        """A vertex at ``τ`` is reached at ``τ * scale + offset``."""
        timing = TrajectoryTiming(ReferenceContext.ABSOLUTE, scale=2.0, offset=1.0)
        assert timing.trajectory_time(2.0 * 3.0 + 1.0, 0.0) == pytest.approx(3.0)

    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            TrajectoryTiming(scale=0.0)


# ---------------------------------------------------------------------------
# Where a trajectory puts the entity
# ---------------------------------------------------------------------------


class TestAtTime:
    def test_between_vertices_it_interpolates(self) -> None:
        sample = _straight().at_time(0.5)
        assert (sample.x, sample.y) == pytest.approx((5.0, 0.0))

    def test_the_velocity_is_the_segments(self) -> None:
        sample = _straight(times=(0.0, 2.0, 4.0)).at_time(1.0)
        assert (sample.vx, sample.vy) == pytest.approx((5.0, 0.0))
        assert sample.speed == pytest.approx(5.0)

    def test_before_the_start_it_waits_at_rest(self) -> None:
        sample = _straight(times=(1.0, 2.0, 3.0)).at_time(0.0)
        assert (sample.x, sample.speed) == pytest.approx((0.0, 0.0))

    def test_after_the_end_it_stays_at_rest(self) -> None:
        sample = _straight().at_time(99.0)
        assert (sample.x, sample.speed) == pytest.approx((20.0, 0.0))

    def test_a_heading_turns_the_short_way(self) -> None:
        """170 to -170 degrees is a 20 degree turn through 180, not 340 back."""
        trajectory = _straight(times=(0.0, 1.0, 2.0), yaws=[170.0, -170.0, -170.0])
        assert abs(trajectory.at_time(0.5).yaw) == pytest.approx(180.0)

    def test_an_untimed_trajectory_has_no_clock(self) -> None:
        with pytest.raises(ValueError, match="timed"):
            _straight(times=None).at_time(0.0)


class TestHeadings:
    def test_a_missing_heading_follows_the_path(self) -> None:
        """CARLA's yaw is clockwise from +x, and +y is South: +y is 90 degrees."""
        trajectory = ResolvedTrajectory(
            xs=[0.0, 0.0], ys=[0.0, 10.0], zs=[0.0, 0.0], yaws=[None, None]
        )
        assert trajectory.at_distance(5.0).yaw == pytest.approx(90.0)

    def test_a_stop_keeps_the_heading_it_stopped_with(self) -> None:
        trajectory = ResolvedTrajectory(
            xs=[0.0, 0.0, 10.0, 10.0],
            ys=[10.0, 0.0, 0.0, 0.0],
            zs=[0.0] * 4,
            yaws=[None] * 4,
            times=[0.0, 1.0, 2.0, 3.0],
        )
        assert trajectory.at_time(2.5).yaw == pytest.approx(0.0)


class TestAlongThePath:
    def test_at_distance(self) -> None:
        sample = _straight().at_distance(15.0, speed=3.0)
        assert (sample.x, sample.vx, sample.vy) == pytest.approx((15.0, 3.0, 0.0))

    def test_an_open_path_clamps(self) -> None:
        assert _straight().at_distance(100.0).x == pytest.approx(20.0)

    def test_a_closed_path_wraps(self) -> None:
        square = ResolvedTrajectory(
            xs=[0.0, 10.0, 10.0, 0.0],
            ys=[0.0, 0.0, 10.0, 10.0],
            zs=[0.0] * 4,
            yaws=[None] * 4,
            closed=True,
        )
        assert square.length == pytest.approx(40.0)
        sample = square.at_distance(45.0)
        assert (sample.x, sample.y) == pytest.approx((5.0, 0.0))

    def test_time_and_distance_convert_both_ways(self) -> None:
        trajectory = _straight(times=(0.0, 1.0, 3.0))
        assert trajectory.distance_at_time(2.0) == pytest.approx(15.0)
        assert trajectory.time_at_distance(15.0) == pytest.approx(2.0)

    def test_project_finds_the_nearest_point(self) -> None:
        assert _straight().project(12.0, 3.0) == pytest.approx(12.0)

    def test_project_near_picks_the_pass_the_entity_is_on(self) -> None:
        """A U-turn passes (5, 0) twice; the window tells the passes apart."""
        u_turn = ResolvedTrajectory(
            xs=[0.0, 50.0, 50.0, 0.0],
            ys=[0.0, 0.0, 1.0, 1.0],
            zs=[0.0] * 4,
            yaws=[None] * 4,
        )
        assert u_turn.project(5.0, 0.5, near=96.0, window=10.0) == pytest.approx(96.0)
        assert u_turn.project(5.0, 0.4, near=5.0, window=10.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Placing a trajectory in CARLA
# ---------------------------------------------------------------------------


class TestResolve:
    def test_a_map_pose_goes_through_the_projector_offset(self) -> None:
        """The inverse of to_map_frame: subtract the offset, flip y and yaw."""
        x, y, z, yaw = map_pose_to_carla(
            MapPose(1010.0, 2020.0, yaw=math.pi / 2, z=45.0),
            mgrs_offset=(1000.0, 2000.0),
            z_offset=40.0,
        )
        assert (x, y, z, yaw) == pytest.approx((10.0, -20.0, 5.0, -90.0))

    def test_an_unstated_height_and_heading_stay_unstated(self) -> None:
        _, _, z, yaw = map_pose_to_carla(MapPose(0.0, 0.0), (0.0, 0.0), 0.0)
        assert z is None and yaw is None

    def test_a_vertex_without_a_height_stands_on_the_ground(self) -> None:
        trajectory = Trajectory(
            "t",
            [
                TrajectoryVertex(CarlaWorldPose(0.0, 0.0, 7.0), _at(0.0)),
                TrajectoryVertex(MapPose(10.0, 0.0), _at(1.0)),
            ],
        )

        def resolve(position):
            if isinstance(position, MapPose):
                return position.x, -position.y, None, None
            return position.x, position.y, position.z, position.yaw

        resolved = resolve_trajectory(trajectory, lambda x, y: 2.5, resolve=resolve)
        assert resolved.zs == pytest.approx([7.0, 2.5])
        assert resolved.times == [0.0, 1.0]
