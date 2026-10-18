"""Unit tests for the ego's debug markers, which the trajectory log records."""

from __future__ import annotations

from types import SimpleNamespace

from autoware_carla_scenario.debug_markers import (
    TRAJECTORY_COLOR,
    DebugMarkers,
    DebugPolyline,
    plan_markers,
)
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.entity.ego import EgoVehicle
from autoware_carla_scenario.scenario_runner import _ego_markers


def _markers(*points: tuple[float, float, float]) -> DebugMarkers:
    markers = DebugMarkers()
    markers.add_polyline(DebugPolyline(list(points)))
    return markers


# ---------------------------------------------------------------------------
# The markers
# ---------------------------------------------------------------------------


def test_a_plan_is_in_carla_s_world_a_little_above_the_road() -> None:
    plan = Trajectory.empty()
    plan.append(0, Pose.from_xyz_yaw(1.0, 2.0, 3.0, 0.0))
    plan.append(100_000, Pose.from_xyz_yaw(4.0, -5.0, 3.0, 0.0))

    (polyline,) = plan_markers(plan).polylines

    # The local frame is CARLA's with y flipped.
    assert [(x, y) for x, y, _ in polyline.points] == [(1.0, -2.0), (4.0, 5.0)]
    assert all(z > 3.0 for _, _, z in polyline.points)
    assert polyline.color == TRAJECTORY_COLOR


def test_a_polyline_without_points_is_not_kept() -> None:
    markers = plan_markers(Trajectory.empty())

    assert not markers
    assert markers.polylines == []


def test_an_ego_shows_nothing_by_default() -> None:
    assert not EgoVehicle().debug_markers()


def test_the_runner_takes_what_the_ego_shows() -> None:
    markers = _markers((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    ego = SimpleNamespace(debug_markers=lambda: markers)
    assert _ego_markers(ego) is markers  # type: ignore[arg-type]


def test_an_ego_whose_markers_fail_does_not_fail_the_run() -> None:
    def _fail() -> DebugMarkers:
        raise RuntimeError("boom")

    assert _ego_markers(SimpleNamespace(debug_markers=_fail)) is None  # type: ignore[arg-type]
