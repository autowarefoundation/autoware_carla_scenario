"""Unit tests for the ego's debug markers and their way into the replay video."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioRunner
from autoware_carla_scenario.constants import FIXED_DELTA_SECONDS
from autoware_carla_scenario.debug_markers import (
    DRIVEN_PATH_COLOR,
    TRAJECTORY_COLOR,
    DebugMarkerLog,
    DebugMarkers,
    DebugPolyline,
    DrivenPathDrawer,
    draw_debug_markers,
    plan_markers,
)
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.entity.ego import EgoVehicle
from autoware_carla_scenario.scenario_runner import (
    _DEBUG_MARKER_LIFE_TIME,
    _record_debug_markers,
)


def _markers(*points: tuple[float, float, float]) -> DebugMarkers:
    markers = DebugMarkers()
    markers.add_polyline(DebugPolyline(list(points)))
    return markers


class _DrawingWorld:
    """A world that records what is drawn on it and when it ticks, in order."""

    def __init__(self) -> None:
        self.events: List[Any] = []
        self.debug = MagicMock()
        self.debug.draw_line.side_effect = lambda *a, **k: self.events.append(
            ("line", a, k)
        )
        self.debug.draw_point.side_effect = lambda *a, **k: self.events.append(
            ("point", a, k)
        )
        self.frame = 0

    def tick(self) -> None:
        self.frame += 1
        self.events.append(("tick", self.frame))

    def get_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(frame=self.frame)

    def get_settings(self) -> MagicMock:
        return MagicMock()

    def apply_settings(self, settings: Any) -> None:
        del settings


# ---------------------------------------------------------------------------
# The markers
# ---------------------------------------------------------------------------


def test_a_plan_is_drawn_in_carla_s_world_a_little_above_the_road() -> None:
    plan = Trajectory.empty()
    plan.append(0, Pose.from_xyz_yaw(1.0, 2.0, 3.0, 0.0))
    plan.append(100_000, Pose.from_xyz_yaw(4.0, -5.0, 3.0, 0.0))

    (polyline,) = plan_markers(plan).polylines

    # The local frame is CARLA's with y flipped.
    assert [(x, y) for x, y, _ in polyline.points] == [(1.0, -2.0), (4.0, 5.0)]
    assert all(z > 3.0 for _, _, z in polyline.points)
    assert polyline.color == TRAJECTORY_COLOR


def test_a_polyline_without_points_is_not_drawn() -> None:
    markers = plan_markers(Trajectory.empty())

    assert not markers
    assert markers.polylines == []


def test_a_polyline_is_drawn_segment_by_segment_with_a_dot_at_each_vertex() -> None:
    world = _DrawingWorld()
    markers = DebugMarkers()
    markers.add_polyline(
        DebugPolyline(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)],
            color=(1, 2, 3),
            thickness=0.2,
            point_size=0.05,
        )
    )

    draw_debug_markers(world, markers, 0.075)  # type: ignore[arg-type]

    lines = [e for e in world.events if e[0] == "line"]
    points = [e for e in world.events if e[0] == "point"]
    assert len(lines) == 2
    assert len(points) == 3
    _, (begin, end), kwargs = lines[1]
    assert (begin.x, begin.y, end.x, end.y) == (1.0, 0.0, 2.0, 1.0)
    assert kwargs["thickness"] == 0.2
    assert kwargs["life_time"] == 0.075
    color = kwargs["color"]
    assert (color.r, color.g, color.b) == (1, 2, 3)
    assert points[0][2]["size"] == 0.05


def test_a_polyline_without_a_point_size_has_no_dots() -> None:
    world = _DrawingWorld()

    draw_debug_markers(
        world,  # type: ignore[arg-type]
        _markers((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        0.075,
    )

    assert [e[0] for e in world.events] == ["line"]


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------


def test_the_log_keeps_nothing_before_it_starts() -> None:
    log = DebugMarkerLog()

    log.record(5, _markers((0.0, 0.0, 0.0)))

    assert not log.active
    assert log.at(5) is None


def test_the_log_counts_frames_from_its_start() -> None:
    log = DebugMarkerLog()
    log.start(100)
    markers = _markers((0.0, 0.0, 0.0))

    log.record(103, markers)

    assert log.at(3) is markers
    assert log.at(103) is None


def test_the_log_keeps_no_frame_without_markers() -> None:
    log = DebugMarkerLog()
    log.start(0)

    log.record(1, DebugMarkers())

    assert log.at(1) is None


def test_the_log_keeps_nothing_once_stopped() -> None:
    log = DebugMarkerLog()
    log.start(0)
    kept = _markers((0.0, 0.0, 0.0))
    log.record(1, kept)

    log.stop()
    log.record(2, _markers((1.0, 0.0, 0.0)))

    assert not log.active
    assert log.at(1) is kept
    assert log.at(2) is None


def test_the_span_counts_the_frames_holding_the_very_same_markers() -> None:
    log = DebugMarkerLog()
    log.start(0)
    plan = _markers((0.0, 0.0, 0.0))
    other = _markers((0.0, 0.0, 0.0))
    for frame in (1, 2, 3):
        log.record(frame, plan)
    log.record(4, other)

    assert log.span(1) == 3
    assert log.span(2) == 2
    assert log.span(4) == 1
    assert log.span(5) == 0


# ---------------------------------------------------------------------------
# The ego and the runner
# ---------------------------------------------------------------------------


def test_an_ego_shows_nothing_by_default() -> None:
    assert not EgoVehicle().debug_markers()


def test_the_runner_records_the_frame_the_world_ticked_to() -> None:
    world = _DrawingWorld()
    world.frame = 42
    log = DebugMarkerLog()
    log.start(40)
    markers = _markers((0.0, 0.0, 0.0))
    ego = SimpleNamespace(debug_markers=lambda: markers)

    _record_debug_markers(world, ego, log)  # type: ignore[arg-type]

    assert log.at(2) is markers


def test_an_ego_whose_markers_fail_does_not_fail_the_run() -> None:
    world = _DrawingWorld()
    log = DebugMarkerLog()
    log.start(0)

    def _fail() -> DebugMarkers:
        raise RuntimeError("boom")

    _record_debug_markers(
        world,  # type: ignore[arg-type]
        SimpleNamespace(debug_markers=_fail),  # type: ignore[arg-type]
        log,
    )

    # The log stops there, rather than failing every tick after.
    assert not log.active
    assert log.at(0) is None


def test_the_markers_last_one_frame_of_the_replay() -> None:
    assert FIXED_DELTA_SECONDS < _DEBUG_MARKER_LIFE_TIME < 2 * FIXED_DELTA_SECONDS


def _replay(
    tmp_path: Path, log: DebugMarkerLog, ego_x_per_tick: float = 0.0
) -> _DrawingWorld:
    """Render a recording of 1 warm-up and 3 scenario ticks; the drawn world."""
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock())
    world = _DrawingWorld()
    runner._client = MagicMock()  # noqa: SLF001
    runner._client.get_world.return_value = world  # noqa: SLF001
    ego = MagicMock()
    ego.get_location.side_effect = lambda: SimpleNamespace(
        x=ego_x_per_tick * world.frame, y=0.0, z=0.0
    )

    recorder = MagicMock()
    recorder.write_frame.side_effect = lambda *a: world.events.append(
        ("frame", world.frame)
    )
    with (
        patch("autoware_carla_scenario.scenario_runner._destroy_all_dynamic_actors"),
        patch(
            "autoware_carla_scenario.scenario_runner.find_actor_by_role_name",
            return_value=ego,
        ),
        patch(
            "autoware_carla_scenario.scenario_runner.CameraRecorder",
            return_value=recorder,
        ),
    ):
        runner._render_video_from_recording(  # noqa: SLF001
            log_path=tmp_path / "run.log",
            mp4_path=tmp_path / "run.mp4",
            spectator_config=MagicMock(),
            tick_count=3,
            warmup_ticks=1,
            scenario_name="S",
            debug_log=log,
        )
    return world


def test_the_replay_draws_each_frame_s_markers_before_the_tick_that_shows_it(
    tmp_path: Path,
) -> None:
    log = DebugMarkerLog()
    log.start(0)
    # Frame 1 is the tick that spawns the replayed actors, before the camera.
    log.record(1, _markers((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    log.record(3, _markers((3.0, 0.0, 0.0), (3.0, 1.0, 0.0)))

    # An ego standing still traces no path.
    world = _replay(tmp_path, log)

    lines = [
        (e[0], e[1][0].x) if e[0] == "line" else e
        for e in world.events
        if e[0] in ("line", "tick")
    ]
    assert lines == [
        ("tick", 1),
        ("tick", 2),
        ("line", 3.0),
        ("tick", 3),
        ("tick", 4),
    ]
    line = next(e for e in world.events if e[0] == "line")
    assert line[2]["life_time"] == pytest.approx(_DEBUG_MARKER_LIFE_TIME)


def test_markers_held_for_several_frames_are_drawn_once_to_last_them_all(
    tmp_path: Path,
) -> None:
    log = DebugMarkerLog()
    log.start(0)
    plan = _markers((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    for frame in (2, 3, 4):
        log.record(frame, plan)

    world = _replay(tmp_path, log)

    lines = [e for e in world.events if e[0] == "line"]
    assert len(lines) == 1
    assert lines[0][2]["life_time"] == pytest.approx(
        2 * FIXED_DELTA_SECONDS + _DEBUG_MARKER_LIFE_TIME
    )


def test_the_replay_traces_the_path_the_ego_drove_to_the_end_of_the_video(
    tmp_path: Path,
) -> None:
    world = _replay(tmp_path, DebugMarkerLog(), ego_x_per_tick=2.0)

    segments = [(e[1][0].x, e[1][1].x, e[2]) for e in world.events if e[0] == "line"]
    # One segment per tick after the first, from where the ego was to where
    # it is, each lasting the ticks still to come.
    assert [(begin, end) for begin, end, _ in segments] == [
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, 8.0),
    ]
    life_times = [kwargs["life_time"] for _, _, kwargs in segments]
    assert life_times == pytest.approx(
        [
            (3 - index) * FIXED_DELTA_SECONDS + _DEBUG_MARKER_LIFE_TIME
            for index in range(3)
        ]
    )
    color = segments[0][2]["color"]
    assert (color.r, color.g, color.b) == DRIVEN_PATH_COLOR


# ---------------------------------------------------------------------------
# The driven path
# ---------------------------------------------------------------------------


def _at(x: float, y: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def test_the_driven_path_starts_at_the_first_location_it_is_given() -> None:
    world = _DrawingWorld()
    path = DrivenPathDrawer()

    path.update(world, _at(0.0), 1.0)  # type: ignore[arg-type]

    assert world.events == []


def test_a_vehicle_standing_still_adds_no_segment() -> None:
    world = _DrawingWorld()
    path = DrivenPathDrawer(step_m=0.5)

    for x in (0.0, 0.1, 0.2, 0.3):
        path.update(world, _at(x), 1.0)  # type: ignore[arg-type]

    assert world.events == []


def test_the_path_is_drawn_once_the_vehicle_has_moved_a_step() -> None:
    world = _DrawingWorld()
    path = DrivenPathDrawer(step_m=0.5)

    for x in (0.0, 0.3, 0.6, 0.9, 1.2):
        path.update(world, _at(x), 4.0)  # type: ignore[arg-type]

    lines = [e for e in world.events if e[0] == "line"]
    # carla.Location holds float32.
    assert [(round(e[1][0].x, 5), round(e[1][1].x, 5)) for e in lines] == [
        (0.0, 0.6),
        (0.6, 1.2),
    ]
    assert all(e[2]["life_time"] == 4.0 for e in lines)
    # Drawn a little above the road, as the plan is.
    assert all(e[1][0].z > 0.0 for e in lines)


def test_a_vehicle_placed_elsewhere_starts_a_new_piece_of_path() -> None:
    world = _DrawingWorld()
    path = DrivenPathDrawer(step_m=0.5, max_jump_m=10.0)

    for x in (0.0, 1.0, 100.0, 101.0):
        path.update(world, _at(x), 1.0)  # type: ignore[arg-type]

    lines = [e for e in world.events if e[0] == "line"]
    assert [(e[1][0].x, e[1][1].x) for e in lines] == [(0.0, 1.0), (100.0, 101.0)]
