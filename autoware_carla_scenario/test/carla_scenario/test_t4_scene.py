"""Trajectory logs as T4Scenes (``t4_scene``, docs/t4_scene.md): the frames,
the time grid, the objects, the lights on their lanelets, the map resource and
the command line -- all from logs written here, without CARLA."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import lanelet2
import numpy as np
import pytest

from autoware_carla_scenario.coordinate.projection import load_lanelet2, map_origin
from autoware_carla_scenario.t4_scene import (
    MAP_SUFFIX,
    SCENE_SUFFIX,
    T4SceneExport,
    trajectory_log_to_t4_scene,
    write_t4_scene,
)
from autoware_carla_scenario.t4_scene.__main__ import main
from autoware_carla_scenario.t4_scene._proto import t4_scene_pb2 as pb
from autoware_carla_scenario.t4_scene.convert import _stop_lines
from autoware_carla_scenario.t4_scene.map_resource import (
    CENTERLINE_INVERTED_TAG,
    resource_digest,
)

DATA = Path(__file__).resolve().parents[3] / "data"
OSM = DATA / "nishishinjuku.osm"
XODR = DATA / "nishishinjuku_carla.xodr"

EGO = {
    "type": "actor",
    "id": 1,
    "type_id": "vehicle.lincoln.mkz",
    "role": "ego",
    "extent": [2.4, 1.0, 0.75],
    "center": [0.1, 0.0, 0.7],
}
CAR = {
    "type": "actor",
    "id": 2,
    "type_id": "vehicle.tesla.model3",
    "role": "npc",
    "extent": [2.0, 0.9, 0.7],
    "center": [0.2, 0.1, 0.7],
}
WALKER = {
    "type": "actor",
    "id": 3,
    "type_id": "walker.pedestrian.0001",
    "role": "",
    "extent": [0.3, 0.3, 0.9],
    "center": [0.0, 0.0, 0.0],
}


def _state(
    actor_id: int,
    x: float,
    y: float,
    yaw: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> list:
    return [actor_id, x, y, 0.5, 0.0, 0.0, yaw, vx, vy, 0.0]


def _tick(frame: int, *states: list, lights: Optional[dict[str, str]] = None) -> dict:
    tick: dict[str, Any] = {
        "type": "tick",
        "frame": frame,
        "t": round((frame - 100) * 0.05, 3),
        "actors": list(states),
    }
    if lights:
        tick["lights"] = lights
    return tick


def _map_record(
    mgrs_offset: tuple[float, float] = (1000.0, 2000.0), z_offset: float = 10.0
) -> dict:
    return {
        "type": "map",
        "name": "Test",
        "source": None,
        "lanelet2": None,
        "opendrive": None,
        "projector_info": None,
        "projector_type": None,
        "mgrs_offset": list(mgrs_offset),
        "z_offset": z_offset,
    }


def _write_log(path: Path, *records: dict, map_record: Optional[dict] = None) -> Path:
    lines = [
        {"type": "world", "schema_version": 2, "map": "Carla/Maps/Test", "episode": 7},
        map_record if map_record is not None else _map_record(),
        *records,
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _numeric(array: pb.NumericArray) -> np.ma.MaskedArray:
    dtypes: dict[int, np.dtype[Any]] = {
        pb.NumericArray.BOOL: np.dtype(np.bool_),
        pb.NumericArray.INT64: np.dtype("<i8"),
        pb.NumericArray.FLOAT64: np.dtype("<f8"),
    }
    dtype = dtypes[array.dtype]
    data = np.frombuffer(array.data, dtype=dtype).reshape(tuple(array.shape))
    mask = (
        np.frombuffer(array.mask, dtype=np.bool_).reshape(data.shape)
        if array.HasField("mask")
        else np.zeros(data.shape, dtype=bool)
    )
    return np.ma.MaskedArray(data, mask=mask)


# -- frames and time ------------------------------------------------------


def test_ego_is_its_rear_axle_in_the_map_frame(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        # Facing CARLA's +y (south): the map's -y.
        _tick(100, _state(1, 10.0, 20.0, yaw=90.0)),
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    pose = _numeric(scene.ego.poses)[0]
    # Origin (10, 20) -> map (1010, 1980); the rear axle 1.2 m behind it, along -y.
    np.testing.assert_allclose(pose, [1010.0, 1980.0 + 1.2, -math.pi / 2], atol=1e-9)
    pose_3d = _numeric(scene.ego.poses_3d)[0]
    np.testing.assert_allclose(pose_3d[:3, 3], [1010.0, 1981.2, 10.5], atol=1e-9)
    np.testing.assert_allclose(
        pose_3d[:3, :3] @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12
    )
    shape = scene.ego.shape
    assert (shape.length, shape.width) == pytest.approx((4.8, 2.0))
    # From the rear axle (-1.2) to the box's front (0.1 + 2.4).
    assert shape.base_link_to_front == pytest.approx(3.7)
    assert shape.base_link_to_rear_axle == 0.0
    assert not shape.HasField("wheel_base")


@pytest.mark.parametrize(
    ("passed", "tag"),
    [
        (True, "result:passed"),
        (False, "result:failed"),
        ("false", "result:failed"),
        ("true", "result:failed"),
        (1, "result:failed"),
        ([0], "result:failed"),
        (None, "result:failed"),
    ],
)
def test_only_a_boolean_true_tags_the_scene_passed(
    tmp_path: Path, passed: Any, tag: str
) -> None:
    # bool("false") is True: a truthy value must not tag a run as passed.
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        _tick(100, _state(1, 10.0, 20.0)),
        {"type": "result", "passed": passed},
    )
    tags = list(
        trajectory_log_to_t4_scene(log, with_map=False).scene.scene_metadata.tags
    )
    assert tag in tags
    assert [t for t in tags if t.startswith("result:")] == [tag]


def test_ego_velocity_flips_y_and_speed_is_along_the_heading(tmp_path: Path) -> None:
    # CARLA -y: the map's +y, the way it faces.
    state = _state(1, 0.0, 0.0, yaw=-90.0, vy=-5.0)
    log = _write_log(tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, state))
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    np.testing.assert_allclose(
        _numeric(scene.ego.velocities)[0], [0.0, 5.0], atol=1e-12
    )
    assert _numeric(scene.ego.longitudinal_speeds)[0] == pytest.approx(5.0)


#: A kinematic bicycle turning left: its rear axle (``base_link``) moves along
#: its heading at _SPEED, turning at _YAW_RATE (rad/s, the map's
#: counter-clockwise); its actor origin is _AXLE_TO_ORIGIN ahead of it.
_SPEED, _WHEEL_BASE, _STEER = 8.0, 2.8, 0.1
_YAW_RATE = _SPEED * math.tan(_STEER) / _WHEEL_BASE
_AXLE_TO_ORIGIN = 1.4


def _bicycle_turn(
    frames: int, yaw_rate: float = _YAW_RATE, *, with_angular: bool = True
) -> tuple[list[dict], list[np.ndarray]]:
    """The turn's ticks, as the recorder writes them (CARLA's frame: ``y``
    and the yaw flip, the angular velocity an axial vector in deg/s), and
    each tick's base_link velocity in the map frame."""
    ticks: list[dict] = []
    truths: list[np.ndarray] = []
    for k in range(frames):
        yaw = 0.3 + yaw_rate * 0.05 * k
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        # The rear axle on its arc (radius _SPEED / yaw_rate), exactly.
        if yaw_rate:
            radius = _SPEED / yaw_rate
            rear = radius * np.array(
                [math.sin(yaw) - math.sin(0.3), math.cos(0.3) - math.cos(yaw)]
            )
        else:
            rear = _SPEED * 0.05 * k * heading
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        origin = rear + _AXLE_TO_ORIGIN * heading
        # v_origin = v_rear + w x (d * heading) = v_rear + w * d * left.
        v_origin = _SPEED * heading + yaw_rate * _AXLE_TO_ORIGIN * left
        tick = _tick(
            100 + k,
            _state(
                1,
                float(origin[0]),
                -float(origin[1]),
                yaw=-math.degrees(yaw),
                vx=float(v_origin[0]),
                vy=-float(v_origin[1]),
            ),
        )
        if with_angular:
            tick["angular_velocity"] = {"1": [0.0, 0.0, -math.degrees(yaw_rate)]}
        ticks.append(tick)
        truths.append(_SPEED * heading)
    return ticks, truths


def _turn_scene(tmp_path: Path, ticks: list[dict]) -> pb.T4Scene:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        {"type": "rear_axle", "id": 1, "offset": -_AXLE_TO_ORIGIN},
        *ticks,
    )
    return trajectory_log_to_t4_scene(log, with_map=False).scene


def test_a_turning_egos_velocity_is_its_rear_axles(tmp_path: Path) -> None:
    ticks, truths = _bicycle_turn(11)
    scene = _turn_scene(tmp_path, ticks)
    velocities = _numeric(scene.ego.velocities)
    headings = _numeric(scene.ego.poses)[:, 2]
    assert len(velocities) == 6 and not velocities.mask.any()
    for row, velocity in enumerate(velocities):
        left = np.array([-math.sin(headings[row]), math.cos(headings[row])])
        # base_link does not slide sideways, and moves at the rear axle's speed.
        assert float(velocity @ left) == pytest.approx(0.0, abs=1e-9)
        assert float(np.linalg.norm(velocity)) == pytest.approx(_SPEED)
        np.testing.assert_allclose(velocity, truths[2 * row], atol=1e-9)
    np.testing.assert_allclose(
        _numeric(scene.ego.longitudinal_speeds), _SPEED, atol=1e-9
    )
    # The body angular velocity, counter-clockwise in the right-handed frame.
    np.testing.assert_allclose(
        _numeric(scene.ego.angular_velocities),
        np.tile([0.0, 0.0, _YAW_RATE], (6, 1)),
        atol=1e-12,
    )


def test_the_turns_origin_velocity_is_its_positions_rate() -> None:
    # The setup is a rigid body, not the converter's formula restated: the
    # origin velocity the ticks carry is what its positions do between them.
    ticks, _ = _bicycle_turn(5)
    positions = np.array([t["actors"][0][1:3] for t in ticks])
    for k in range(1, len(ticks) - 1):
        rate = (positions[k + 1] - positions[k - 1]) / 0.1
        np.testing.assert_allclose(ticks[k]["actors"][0][7:9], rate, atol=1e-3)


def test_a_straight_egos_velocity_is_unchanged(tmp_path: Path) -> None:
    ticks, truths = _bicycle_turn(3, yaw_rate=0.0)
    scene = _turn_scene(tmp_path, ticks)
    np.testing.assert_allclose(_numeric(scene.ego.velocities)[0], truths[0], atol=1e-12)
    assert _numeric(scene.ego.longitudinal_speeds)[0] == pytest.approx(_SPEED)


def test_a_log_without_angular_velocity_keeps_the_origins_velocity(
    tmp_path: Path,
) -> None:
    ticks, _ = _bicycle_turn(3, with_angular=False)
    scene = _turn_scene(tmp_path, ticks)
    origin = ticks[0]["actors"][0]
    np.testing.assert_allclose(
        _numeric(scene.ego.velocities)[0], [origin[7], -origin[8]], atol=1e-12
    )
    assert _numeric(scene.ego.longitudinal_speeds)[0] == pytest.approx(_SPEED)
    assert not scene.ego.HasField("angular_velocities")


def test_the_axial_flip_is_the_driver_observations() -> None:
    from autoware_carla_scenario.driver.observation import _to_local_axial
    from autoware_carla_scenario.t4_scene.convert import _FLIP_AXIAL

    vector = (1.5, -2.0, 3.25)
    np.testing.assert_array_equal(
        np.array(vector) * _FLIP_AXIAL, _to_local_axial(*vector)
    )


def test_rear_axle_offset_can_be_given(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    scene = trajectory_log_to_t4_scene(
        log, with_map=False, rear_axle_offset_m=-1.5
    ).scene
    assert _numeric(scene.ego.poses)[0][0] == pytest.approx(1000.0 - 1.5)
    assert scene.ego.shape.base_link_to_front == pytest.approx(0.1 + 2.4 + 1.5)


def test_the_rear_axle_is_the_one_the_run_recorded(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        {"type": "rear_axle", "id": 1, "offset": -1.5},
        _tick(100, _state(1, 0.0, 0.0)),
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    assert _numeric(scene.ego.poses)[0][0] == pytest.approx(1000.0 - 1.5)
    assert scene.ego.shape.base_link_to_front == pytest.approx(0.1 + 2.4 + 1.5)
    # Given, it wins over the log's.
    given = trajectory_log_to_t4_scene(
        log, with_map=False, rear_axle_offset_m=-1.0
    ).scene
    assert _numeric(given.ego.poses)[0][0] == pytest.approx(1000.0 - 1.0)


def test_the_ego_shows_the_turn_indicators_the_log_records(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        _tick(100, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
        # Written after the tick they belong to; the car's are not the ego's.
        {"type": "turn_indicators", "frame": 101, "t": 0.05, "states": {"2": "right"}},
        _tick(101, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
        _tick(102, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
        {"type": "turn_indicators", "frame": 102, "t": 0.1, "states": {"1": "off"}},
        _tick(103, _state(1, 0.0, 0.0)),
        {"type": "turn_indicators", "frame": 103, "t": 0.15, "states": {"1": "left"}},
        _tick(104, _state(1, 0.0, 0.0)),
        _tick(105, _state(1, 0.0, 0.0)),
        {"type": "turn_indicators", "frame": 105, "t": 0.25, "states": {"1": "hazard"}},
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False, step_s=0.05).scene
    indicators = _numeric(scene.ego.turn_indicators)
    # Unknown until the ego's first record; then disabled, left, left, hazard.
    assert list(np.ma.getmaskarray(indicators)) == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert list(indicators.compressed()) == [1, 2, 2, 4]


def test_a_log_without_turn_indicators_leaves_them_out(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    assert not scene.ego.HasField("turn_indicators")


def test_rows_are_a_uniform_grid_and_a_missing_tick_is_unavailable(
    tmp_path: Path,
) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        *[
            _tick(
                frame, _state(1, float(frame), 0.0), _state(2, float(frame) + 5.0, 0.0)
            )
            for frame in (100, 101, 102, 103, 106, 107, 108)
        ],
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False, step_s=0.1).scene
    assert list(_numeric(scene.frame_indices)) == [100, 102, 104, 106, 108]
    assert list(_numeric(scene.timestamps_ns)) == [
        0,
        100_000_000,
        200_000_000,
        300_000_000,
        400_000_000,
    ]
    assert list(_numeric(scene.ego.available)) == [True, True, False, True, True]
    assert list(np.ma.getmaskarray(_numeric(scene.ego.poses))[:, 0]) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert list(_numeric(scene.annotations.available)) == [
        True,
        True,
        False,
        True,
        True,
    ]
    assert list(_numeric(scene.annotations.box_offsets)) == [0, 1, 2, 2, 3, 4]
    assert scene.scene_metadata.scene_id == "Run"
    assert scene.scene_metadata.recording_id == "7"


def test_step_must_be_a_multiple_of_the_tick(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="multiple"):
        trajectory_log_to_t4_scene(log, with_map=False, step_s=0.07)


def test_current_index_counts_from_the_end_when_negative(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        *[_tick(frame, _state(1, 0.0, 0.0)) for frame in range(100, 110)],
    )
    assert (
        trajectory_log_to_t4_scene(
            log, with_map=False, current_index=-1
        ).scene.current_index
        == 4
    )
    with pytest.raises(ValueError, match="current_index"):
        trajectory_log_to_t4_scene(log, with_map=False, current_index=5)


def test_a_log_without_an_ego_or_a_tick_is_refused(tmp_path: Path) -> None:
    no_ego = _write_log(
        tmp_path / "A.trajectory.jsonl", CAR, _tick(100, _state(2, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="no ego"):
        trajectory_log_to_t4_scene(no_ego, with_map=False)
    no_tick = _write_log(tmp_path / "B.trajectory.jsonl", EGO)
    with pytest.raises(ValueError, match="no tick"):
        trajectory_log_to_t4_scene(no_tick, with_map=False)


def test_a_killed_runs_cut_off_last_tick_is_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log = _write_log(
        tmp_path / "Killed.trajectory.jsonl",
        EGO,
        *[_tick(frame, _state(1, 0.0, 0.0)) for frame in range(100, 110)],
    )
    complete = json.dumps(_tick(110, _state(1, 0.0, 0.0)))
    with log.open("a") as stream:
        stream.write(complete[: len(complete) // 2])  # no newline: cut off
    with caplog.at_level("WARNING"):
        export = trajectory_log_to_t4_scene(log, step_s=0.05, with_map=False)
    assert list(_numeric(export.scene.frame_indices)) == list(range(100, 110))
    assert any(
        message.startswith(f"{log}:14: the last line is not JSON")
        for message in caplog.messages
    )


def test_a_cut_off_last_line_before_blank_lines_is_left_out(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Killed.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    with log.open("a") as stream:
        stream.write('{"type": "tick", "frame": 101\n\n')
    export = trajectory_log_to_t4_scene(log, step_s=0.05, with_map=False)
    assert list(_numeric(export.scene.frame_indices)) == [100]


def test_a_line_that_is_not_json_in_the_middle_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "Corrupt.trajectory.jsonl"
    _write_log(log, EGO, _tick(100, _state(1, 0.0, 0.0)))
    lines = log.read_text().splitlines(keepends=True)
    lines.insert(3, '{"type": "tick", "frame": 1\n')
    lines.append(json.dumps(_tick(101, _state(1, 0.0, 0.0))) + "\n")
    log.write_text("".join(lines))
    with pytest.raises(ValueError, match=rf"^{re.escape(str(log))}:4: not a JSON line"):
        trajectory_log_to_t4_scene(log, with_map=False)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"type": "tick", "frame": 101, "t": 0.05}, "missing field 'actors'"),
        ({"type": "rear_axle", "id": 1, "offset": None}, "NoneType"),
        ({"type": "tick", "frame": 101, "t": 0.05, "actors": [[1]]}, ""),
        (
            {"type": "tick", "frame": 101, "t": 0.05, "actors": [], "lights": [1]},
            "items",
        ),
    ],
    ids=["tick-without-actors", "null-rear-axle-offset", "short-state", "lights-list"],
)
def test_a_record_that_cannot_be_used_is_refused_with_its_line(
    tmp_path: Path, record: dict, message: str
) -> None:
    log = _write_log(
        tmp_path / "Bad.trajectory.jsonl",
        EGO,
        _tick(100, _state(1, 0.0, 0.0)),
        record,
        _tick(102, _state(1, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match=rf"^{re.escape(str(log))}:5: ") as caught:
        trajectory_log_to_t4_scene(log, step_s=0.05, with_map=False)
    assert message in str(caught.value)
    assert caught.value.__cause__ is not None


# -- objects ----------------------------------------------------------------


def test_objects_are_boxes_in_the_map_frame(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        WALKER,
        _tick(
            100,
            _state(1, 0.0, 0.0),
            _state(2, 10.0, 5.0, yaw=90.0, vx=3.0),
            _state(3, -4.0, 2.0),
        ),
    )
    values = trajectory_log_to_t4_scene(
        log, with_map=False
    ).scene.annotations.values.boxes
    assert [track.value for track in values.track_ids.values] == ["2", "3"]
    assert list(_numeric(values.labels)) == [1, 7]
    # The car faces the map's -y: its box centre (0.2, 0.1) ahead/right in
    # CARLA is (0.2 ahead, 0.1 right) = map (-0.1, -0.2) from its origin.
    np.testing.assert_allclose(
        _numeric(values.positions)[0], [1010.0 - 0.1, 1995.0 - 0.2, 11.2], atol=1e-9
    )
    np.testing.assert_allclose(_numeric(values.dimensions)[0], [1.8, 4.0, 1.4])
    assert _numeric(values.headings)[0] == pytest.approx(-math.pi / 2)
    np.testing.assert_allclose(_numeric(values.velocities)[0], [3.0, 0.0])
    quaternion = _numeric(values.orientations)[0]
    np.testing.assert_allclose(
        quaternion,
        [0.0, 0.0, -math.sin(math.pi / 4), math.cos(math.pi / 4)],
        atol=1e-12,
    )


def test_objects_beyond_the_distance_are_left_out(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        _tick(100, _state(1, 0.0, 0.0), _state(2, 150.0, 0.0)),
    )
    near = trajectory_log_to_t4_scene(log, with_map=False).scene
    assert list(_numeric(near.annotations.box_offsets)) == [0, 0]
    everyone = trajectory_log_to_t4_scene(
        log, with_map=False, max_distance_m=None
    ).scene
    assert list(_numeric(everyone.annotations.box_offsets)) == [0, 1]


def test_an_entity_is_on_the_scene_from_its_spawn_to_its_despawn(
    tmp_path: Path,
) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        *[
            _tick(frame, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0))
            for frame in range(100, 108)
        ],
        {
            "type": "spawn",
            "frame": 102,
            "t": 0.1,
            "label": "s",
            "entity": "npc",
            "id": 2,
        },
        {
            "type": "despawn",
            "frame": 106,
            "t": 0.3,
            "label": "d",
            "entity": "npc",
            "id": 2,
            "reason": "action",
        },
    )
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    # Parked out of sight up to its spawn frame (102), gone from its despawn's.
    assert list(np.diff(_numeric(scene.annotations.box_offsets))) == [0, 0, 1, 0]


def test_an_entity_parked_under_the_map_is_off_the_scene(tmp_path: Path) -> None:
    """A hidden road user is parked 500 m under the map with no despawn
    recorded (#844): it is no box while it is down there."""
    hidden = _state(2, 5.0, 0.0)
    hidden[3] = -499.5
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        _tick(100, _state(1, 0.0, 0.0), hidden),
        _tick(102, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
    )
    for max_distance_m in (50.0, None):
        scene = trajectory_log_to_t4_scene(
            log, with_map=False, max_distance_m=max_distance_m
        ).scene
        assert list(_numeric(scene.annotations.box_offsets)) == [0, 0, 1]


def test_a_row_without_the_ego_has_no_objects_near_it(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        _tick(100, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
        _tick(102, _state(2, 5.0, 0.0)),
    )
    near = trajectory_log_to_t4_scene(log, with_map=False).scene
    assert list(_numeric(near.annotations.available)) == [True, False]
    assert list(_numeric(near.annotations.box_offsets)) == [0, 1, 1]
    everyone = trajectory_log_to_t4_scene(
        log, with_map=False, max_distance_m=None
    ).scene
    assert list(_numeric(everyone.annotations.available)) == [True, True]
    assert list(_numeric(everyone.annotations.box_offsets)) == [0, 1, 2]


def test_a_second_ego_role_actor_is_a_road_user(tmp_path: Path) -> None:
    hero = {**CAR, "role": "hero"}
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        hero,
        _tick(100, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
    )
    values = trajectory_log_to_t4_scene(
        log, with_map=False
    ).scene.annotations.values.boxes
    assert [track.value for track in values.track_ids.values] == ["2"]


# -- the map and its lights ---------------------------------------------------


@pytest.fixture(scope="module")
def fixture_map() -> tuple[Any, tuple[float, float]]:
    """The nishishinjuku map as the run reads it, and its MGRS offset."""
    (lat, lon, alt), _ = map_origin(OSM, XODR)
    lanelet_map, projector, _ = load_lanelet2(OSM, lanelet2.io.Origin(lat, lon, alt))
    offset = projector.forward(lanelet2.core.GPSPoint(lat, lon, alt))
    return lanelet_map, (offset.x, offset.y)


@pytest.fixture(scope="module")
def map_log(
    tmp_path_factory: pytest.TempPathFactory,
    fixture_map: tuple[Any, tuple[float, float]],
) -> tuple[Path, int]:
    """A run on nishishinjuku with one light, whose stop waypoint is at a
    governed lanelet's stop line; and that lanelet's id."""
    lanelet_map, (ox, oy) = fixture_map
    line = _stop_lines(lanelet_map)[0]
    stop = line.segments[0].mean(axis=0)
    # The map frame back to CARLA's: y and yaw flip.
    x, y, yaw = stop[0] - ox, -(stop[1] - oy), -math.degrees(line.heading)
    record = _map_record((ox, oy), 0.0)
    record["lanelet2"] = {"path": str(OSM), "sha256": None, "bytes": 0}
    record["opendrive"] = {"path": str(XODR), "sha256": None, "bytes": 0}
    log = _write_log(
        tmp_path_factory.mktemp("map") / "OnMap.trajectory.jsonl",
        {
            "type": "traffic_light",
            "id": 9,
            "opendrive_id": "1",
            "location": [x, y, 0.0],
            "boxes": [],
            "stops": [[x, y, 0.0, yaw]],
        },
        EGO,
        _tick(100, _state(1, x, y), lights={"9": "Red"}),
        _tick(101, _state(1, x, y)),
        _tick(102, _state(1, x, y), lights={"9": "Green"}),
        map_record=record,
    )
    return log, line.lanelet_id


@pytest.fixture(scope="module")
def map_export(map_log: tuple[Path, int]) -> T4SceneExport:
    """The map log converted at the tick, once for the tests that read it."""
    return trajectory_log_to_t4_scene(map_log[0], step_s=0.05)


def test_the_map_is_a_resource_the_scene_names(map_export: T4SceneExport) -> None:
    export = map_export
    resource = export.map_resource
    assert resource is not None
    assert resource.encoding == pb.MapResource.LANELET2_OSM
    assert resource.projection == pb.MapResource.LOCAL_METERS
    assert resource.sha256 == resource_digest(resource.data)
    reference = export.scene.map
    assert reference.sha256 == resource.sha256
    assert reference.scope == pb.MapReference.FULL
    assert reference.has_elevation
    assert reference.map_id.startswith("lanelet2-sha256:")
    # Both in Autoware's map frame: registered alike, at the identity.
    for registered in (export.scene, reference):
        assert registered.global_origin == pb.Pose2D(cos_yaw=1.0)
        rotation = registered.global_origin_3d.rotation
        assert (rotation.m00, rotation.m11, rotation.m22) == (1.0, 1.0, 1.0)
        assert registered.global_origin_3d.translation == pb.Vector3()


def _group_rows(lights: Any) -> list[dict[int, tuple[int, int, int]]]:
    """Each row's groups: regulatory element id -> its one lamp's (colour,
    shape, status)."""
    offsets = list(_numeric(lights.group_offsets))
    groups = _numeric(lights.values.group_ids).tolist()
    elements = list(_numeric(lights.values.element_offsets))
    lamps = list(
        zip(
            _numeric(lights.values.element_colors).tolist(),
            _numeric(lights.values.element_shapes).tolist(),
            _numeric(lights.values.element_statuses).tolist(),
        )
    )
    rows = []
    for a, b in zip(offsets, offsets[1:]):
        row = {}
        for g in range(a, b):
            assert elements[g + 1] - elements[g] == 1  # one round lamp
            row[groups[g]] = lamps[elements[g]]
        rows.append(row)
    return rows


def _elements_of(lanelet_map: Any, lanelet_id: int) -> tuple[int, ...]:
    return next(
        line.traffic_lights
        for line in _stop_lines(lanelet_map)
        if line.lanelet_id == lanelet_id
    )


def test_a_light_colours_the_regulatory_element_of_the_lanelet_it_governs(
    map_log: tuple[Path, int],
    map_export: T4SceneExport,
    fixture_map: tuple[Any, tuple[float, float]],
) -> None:
    element = _elements_of(fixture_map[0], map_log[1])[0]
    lights = map_export.scene.traffic_lights
    rows = _group_rows(lights)
    # Red, carried over the tick without a change, then green: a round lamp, on.
    assert [row[element] for row in rows] == [(1, 1, 2), (1, 1, 2), (3, 1, 2)]
    assert list(_numeric(lights.available)) == [True, True, True]
    np.testing.assert_array_equal(_numeric(lights.values.element_confidences), 1.0)


@pytest.mark.parametrize(("before_m", "matched"), [(6.0, True), (None, False)])
def test_a_light_whose_stop_waypoint_is_before_the_line_colours_its_lanelet(
    tmp_path: Path,
    fixture_map: tuple[Any, tuple[float, float]],
    before_m: Optional[float],
    matched: bool,
) -> None:
    lanelet_map, (ox, oy) = fixture_map
    line = _stop_lines(lanelet_map)[0]
    forward = np.array([math.cos(line.heading), math.sin(line.heading)])
    # CARLA's stop waypoint, in the lane before the map's line; or on no lane.
    stop = (
        line.segments[0].mean(axis=0) - before_m * forward
        if before_m is not None
        else np.array([ox + 5000.0, oy + 5000.0])
    )
    x, y, yaw = stop[0] - ox, -(stop[1] - oy), -math.degrees(line.heading)
    record = _map_record((ox, oy), 0.0)
    # No OpenDRIVE: no controller ties the light to a lanelet.
    record["lanelet2"] = {"path": str(OSM), "sha256": None, "bytes": 0}
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        {
            "type": "traffic_light",
            "id": 9,
            "location": [x, y, 0.0],
            "boxes": [],
            "stops": [[x, y, 0.0, yaw]],
        },
        EGO,
        _tick(100, _state(1, x, y), lights={"9": "Red"}),
        map_record=record,
    )
    rows = _group_rows(trajectory_log_to_t4_scene(log).scene.traffic_lights)
    element = _elements_of(lanelet_map, line.lanelet_id)[0]
    assert (element in rows[0]) is matched


def test_a_light_its_controller_switches_colours_the_lanelet_wherever_it_stands(
    tmp_path: Path, fixture_map: tuple[Any, tuple[float, float]]
) -> None:
    from autoware_carla_scenario.maps.road_network_cache import open_road_network
    from autoware_carla_scenario.utils.traffic_light import (
        get_signal_ids_for_controller,
        lanelet2_traffic_light_id_to_opendrive_controller_id,
    )

    lanelet_map, (ox, oy) = fixture_map
    network = open_road_network(XODR)
    lanelet_id, signal = next(
        (line.lanelet_id, signals[0])
        for line in _stop_lines(lanelet_map)
        for element in line.traffic_lights
        if (
            controller := lanelet2_traffic_light_id_to_opendrive_controller_id(
                network, element
            )
        )
        is not None
        and (signals := get_signal_ids_for_controller(network, controller))
    )
    record = _map_record((ox, oy), 0.0)
    record["lanelet2"] = {"path": str(OSM), "sha256": None, "bytes": 0}
    record["opendrive"] = {"path": str(XODR), "sha256": None, "bytes": 0}
    # A kilometre from any stop line: only its controller ties it to the lanelet.
    far = [5000.0, 5000.0, 0.0]
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        {
            "type": "traffic_light",
            "id": 9,
            "opendrive_id": signal,
            "location": far,
            "boxes": [],
            "stops": [[*far, 0.0]],
        },
        EGO,
        _tick(100, _state(1, 5000.0, 5000.0), lights={"9": "Yellow"}),
        map_record=record,
    )
    lights = trajectory_log_to_t4_scene(log).scene.traffic_lights
    element = _elements_of(lanelet_map, lanelet_id)[0]
    assert _group_rows(lights)[0][element] == (2, 1, 2)


def test_the_resource_keeps_exact_local_coordinates(
    map_export: T4SceneExport, fixture_map: tuple[Any, tuple[float, float]]
) -> None:
    resource = map_export.map_resource
    assert resource is not None
    lanelet_map, _ = fixture_map
    root = ET.fromstring(resource.data)
    projector = lanelet2.projection.MercatorProjector(lanelet2.io.Origin(0, 0))
    nodes = root.findall("node")
    assert len(nodes) == len(lanelet_map.pointLayer)
    for node in nodes[:: max(1, len(nodes) // 200)]:
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        point = lanelet_map.pointLayer[int(node.attrib["id"])]
        assert float(tags["local_x"]) == point.x
        assert float(tags["local_y"]) == point.y
        assert float(tags["ele"]) == point.z
        # The auxiliary lat/lon project back onto the same point.
        back = projector.forward(
            lanelet2.core.GPSPoint(
                float(node.attrib["lat"]), float(node.attrib["lon"]), 0.0
            )
        )
        assert (back.x, back.y) == pytest.approx((point.x, point.y), abs=1e-6)
    for relation in root.findall("relation"):
        if relation.find("member[@role='centerline']") is not None:
            assert relation.find(f"tag[@k='{CENTERLINE_INVERTED_TAG}']") is not None


def _drive_along(
    lanelet_map: Any,
    offset: tuple[float, float],
    legs: list[tuple[int, float, float]],
    spacing_m: float = 0.5,
) -> list[dict]:
    """Ticks of an ego whose origin (its rear axle: ``rear_axle`` 0) runs along
    the centrelines of *legs*, ``(lanelet id, from, to)`` as fractions of each
    one's length, a point every *spacing_m*, in CARLA's frame."""
    ox, oy = offset
    points: list[np.ndarray] = []
    for lanelet_id, start, end in legs:
        centre = np.array(
            [(p.x, p.y) for p in lanelet_map.laneletLayer[lanelet_id].centerline]
        )
        s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
        )
        for at in np.arange(start * s[-1], end * s[-1], spacing_m):
            points.append(np.array([np.interp(at, s, centre[:, k]) for k in range(2)]))
    ticks = []
    for i, point in enumerate(points):
        ahead = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
        yaw = -math.degrees(math.atan2(ahead[1], ahead[0]))
        ticks.append(_tick(100 + i, _state(1, point[0] - ox, -(point[1] - oy), yaw)))
    return ticks


@pytest.fixture(scope="module")
def route_log(
    tmp_path_factory: pytest.TempPathFactory,
    fixture_map: tuple[Any, tuple[float, float]],
) -> Path:
    """A run on nishishinjuku: along lanelet 4, a lane change to 98 beside it,
    and on into 98's successor 102."""
    lanelet_map, (ox, oy) = fixture_map
    record = _map_record((ox, oy), 0.0)
    record["lanelet2"] = {"path": str(OSM), "sha256": None, "bytes": 0}
    record["opendrive"] = {"path": str(XODR), "sha256": None, "bytes": 0}
    return _write_log(
        tmp_path_factory.mktemp("route") / "Drive.trajectory.jsonl",
        EGO,
        {"type": "rear_axle", "id": 1, "offset": 0.0},
        *_drive_along(
            lanelet_map, (ox, oy), [(4, 0.0, 0.4), (98, 0.6, 1.0), (102, 0.0, 1.0)]
        ),
        map_record=record,
    )


def test_the_route_is_the_lanelets_the_ego_drove_along(route_log: Path) -> None:
    scene = trajectory_log_to_t4_scene(route_log).scene
    rows = scene.routes.rows
    assert len(rows) == len(_numeric(scene.timestamps_ns))
    for row in rows:
        assert row.HasField("value")
        assert [list(seg.lane_ids) for seg in row.value.segments] == [[4], [98], [102]]
        assert not any(seg.HasField("preferred_lane_id") for seg in row.value.segments)


def test_the_route_can_be_left_out(route_log: Path, tmp_path: Path) -> None:
    assert not trajectory_log_to_t4_scene(route_log, with_route=False).scene.HasField(
        "routes"
    )
    assert not trajectory_log_to_t4_scene(route_log, with_map=False).scene.HasField(
        "routes"
    )
    assert main([str(route_log), "-o", str(tmp_path), "--no-route"]) == 0
    written = pb.T4Scene.FromString((tmp_path / f"Drive{SCENE_SUFFIX}").read_bytes())
    assert not written.HasField("routes")


def test_an_ego_that_stood_still_has_no_route(map_export: T4SceneExport) -> None:
    assert not map_export.scene.HasField("routes")


def test_a_map_that_is_not_here_is_left_out(tmp_path: Path) -> None:
    record = _map_record()
    record["lanelet2"] = {
        "path": str(tmp_path / "gone.osm"),
        "sha256": None,
        "bytes": 0,
    }
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        _tick(100, _state(1, 0.0, 0.0)),
        map_record=record,
    )
    export = trajectory_log_to_t4_scene(log)
    assert export.map_resource is None
    assert not export.scene.HasField("map")
    assert not export.scene.HasField("traffic_lights")


# -- files and the command line ----------------------------------------------


def test_the_contract_keeps_the_devkits_descriptor_name() -> None:
    """The same name and package as the devkit's own bindings, so both load
    into one process."""
    assert pb.DESCRIPTOR.name == "t4_scene.proto"
    assert pb.DESCRIPTOR.package == "t4.devkit.experimental.v1"


def test_write_names_the_files_after_the_run(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    scene_path, map_path = write_t4_scene(log, tmp_path / "out", with_map=False)
    assert scene_path == tmp_path / "out" / f"Run{SCENE_SUFFIX}"
    assert map_path is None
    scene = pb.T4Scene.FromString(scene_path.read_bytes())
    assert scene.scene_metadata.scene_id == "Run"
    # The same log gives the same bytes.
    first = scene_path.read_bytes()
    write_t4_scene(log, tmp_path / "out", with_map=False)
    assert scene_path.read_bytes() == first


def test_cli_converts_every_log_under_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("A", "B"):
        (tmp_path / "runs" / name).mkdir(parents=True)
        _write_log(
            tmp_path / "runs" / name / f"{name}.trajectory.jsonl",
            EGO,
            _tick(100, _state(1, 0.0, 0.0)),
        )
    assert main([str(tmp_path / "runs"), "--no-map"]) == 0
    assert (tmp_path / "runs" / "A" / f"A{SCENE_SUFFIX}").is_file()
    assert (tmp_path / "runs" / "B" / f"B{SCENE_SUFFIX}").is_file()
    assert not (tmp_path / "runs" / "A" / f"A{MAP_SUFFIX}").exists()
    assert "B.t4scene.pb" in capsys.readouterr().out


def test_cli_refuses_two_logs_of_one_stem_into_one_directory(tmp_path: Path) -> None:
    for case in ("0", "1"):
        (tmp_path / case).mkdir()
        _write_log(
            tmp_path / case / "Run.trajectory.jsonl",
            EGO,
            _tick(100, _state(1, 0.0, 0.0)),
        )
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--no-map", "-o", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    assert main([str(tmp_path), "--no-map"]) == 0


def test_cli_goes_on_past_a_malformed_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "A.trajectory.jsonl"
    _write_log(broken, EGO, {"type": "tick", "frame": 100, "t": 0.0})  # no actors
    _write_log(tmp_path / "B.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0)))
    assert main([str(tmp_path), "--no-map"]) == 1
    assert (tmp_path / f"B{SCENE_SUFFIX}").is_file()
    assert "A.trajectory.jsonl" in capsys.readouterr().err


def test_cli_fails_on_a_log_it_cannot_convert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _write_log(
        tmp_path / "Bad.trajectory.jsonl", CAR, _tick(100, _state(2, 0.0, 0.0))
    )
    assert main([str(bad), "--no-map"]) == 1
    assert "no ego" in capsys.readouterr().err


# -- the command line's logs and its stems, pinned before they moved ----------


def _recording_cli(
    monkeypatch: pytest.MonkeyPatch, failing: tuple[str, ...] = ()
) -> list[tuple[Path, Optional[Path]]]:
    """Replace the conversion the command calls: each log it converts, in
    order, with the output directory; a log named in *failing* raises."""
    import autoware_carla_scenario.t4_scene.__main__ as cli

    calls: list[tuple[Path, Optional[Path]]] = []

    def convert(log: Path, output_dir: Optional[Path], **_: Any) -> tuple[Path, Any]:
        calls.append((log, output_dir))
        if log.name in failing:
            raise ValueError(f"cannot convert {log.name}")
        resource = None if "nomap" in log.name else Path(f"{log}.map")
        return Path(f"{log}.scene"), resource

    monkeypatch.setattr(cli, "write_t4_scene", convert)
    return calls


def test_cli_takes_a_log_as_named_and_every_log_under_a_directory_sorted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _recording_cli(monkeypatch)
    runs = tmp_path / "runs"
    for name in ("b/Z", "a/Y", "a/sub/X", "c/W", "Ä/V"):
        (runs / name).parent.mkdir(parents=True, exist_ok=True)
        (runs / f"{name}.trajectory.jsonl").write_text("")
    # Not logs: no suffix, another suffix, a directory named like a log.
    (runs / "a" / "notes.txt").write_text("")
    (runs / "a" / "Y.trajectory.json").write_text("")
    (runs / "d" / "dir.trajectory.jsonl").mkdir(parents=True)
    (runs / "d" / "dir.trajectory.jsonl" / "In.trajectory.jsonl").write_text("")
    given = tmp_path / "any-name.txt"  # a file is taken as it is named
    missing = tmp_path / "missing.trajectory.jsonl"  # and so is a missing one
    code = main([str(given), str(runs), str(missing), str(given), "--no-map"])
    assert code == 0
    assert [log for log, _ in calls] == [
        given,
        runs / "a" / "Y.trajectory.jsonl",
        runs / "a" / "sub" / "X.trajectory.jsonl",
        runs / "b" / "Z.trajectory.jsonl",
        runs / "c" / "W.trajectory.jsonl",
        runs / "d" / "dir.trajectory.jsonl",
        runs / "d" / "dir.trajectory.jsonl" / "In.trajectory.jsonl",
        runs / "Ä" / "V.trajectory.jsonl",
        missing,
        given,
    ]
    assert all(output is None for _, output in calls)
    out = capsys.readouterr().out
    assert out == "".join(f"{log}.scene (map: {log}.map)\n" for log, _ in calls)


def test_cli_reports_each_log_it_cannot_convert_and_goes_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _recording_cli(monkeypatch, failing=("B.trajectory.jsonl",))
    for name in ("A", "B", "C-nomap"):
        (tmp_path / f"{name}.trajectory.jsonl").write_text("")
    assert main([str(tmp_path), "-o", str(tmp_path / "out")]) == 1
    assert [output for _, output in calls] == [tmp_path / "out"] * 3
    captured = capsys.readouterr()
    a, b, c = (tmp_path / f"{n}.trajectory.jsonl" for n in ("A", "B", "C-nomap"))
    assert captured.out == f"{a}.scene (map: {a}.map)\n{c}.scene\n"
    assert captured.err == f"{b}: cannot convert B.trajectory.jsonl\n"


def _cli_error(capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    """The last line the command writes before it exits with status 2."""
    with pytest.raises(SystemExit) as raised:
        main(argv)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: scenario-t4scene ")
    return captured.err.splitlines(keepends=True)[-1]


def test_cli_refuses_an_empty_directory_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _recording_cli(monkeypatch)
    (tmp_path / "empty").mkdir()
    line = _cli_error(capsys, [str(tmp_path / "empty"), "--no-map"])
    assert line == "scenario-t4scene: error: no trajectory log found\n"
    assert calls == []


@pytest.mark.parametrize(
    ("names", "first", "second", "stem"),
    [
        (["0/Run.trajectory.jsonl", "1/Run.trajectory.jsonl"], 0, 1, "Run"),
        (
            [
                "0/Run.trajectory.jsonl",
                "1/Run.trajectory.jsonl",
                "2/Run.trajectory.jsonl",
            ],
            0,
            1,
            "Run",
        ),
        (
            ["0/Run.trajectory.jsonl", "1/Other.trajectory.jsonl", "2/Run.json"],
            0,
            2,
            "Run",
        ),
        (["0/日本.trajectory.jsonl", "1/日本.trajectory.jsonl"], 0, 1, "日本"),
        (["0/a.b.trajectory.jsonl", "1/a.b.c"], 0, 1, "a.b"),
    ],
    ids=["two", "three", "other-suffix", "non-ascii", "dotted"],
)
def test_cli_refuses_two_logs_of_one_stem_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    names: list[str],
    first: int,
    second: int,
    stem: str,
) -> None:
    calls = _recording_cli(monkeypatch)
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    out = tmp_path / "out dir"
    line = _cli_error(capsys, [*map(str, paths), "-o", str(out), "--no-map"])
    assert line == (
        f"scenario-t4scene: error: {paths[first]} and {paths[second]} would both "
        f"be written to {out / stem}.*; convert them without -o\n"
    )
    assert calls == []
    # Beside each log, they do not collide.
    assert main([*map(str, paths), "--no-map"]) == 0
    assert [log for log, _ in calls] == paths


def test_cli_refuses_one_log_named_twice_into_one_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _recording_cli(monkeypatch)
    log = tmp_path / "Run.trajectory.jsonl"
    log.write_text("")
    line = _cli_error(capsys, [str(log), str(tmp_path), "-o", "rel"])
    assert line == (
        f"scenario-t4scene: error: {log} and {log} would both be written to "
        "rel/Run.*; convert them without -o\n"
    )


def test_cli_checks_its_options_before_it_looks_for_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _recording_cli(monkeypatch)
    (tmp_path / "empty").mkdir()
    line = _cli_error(capsys, [str(tmp_path / "empty"), "--step=0.03"])
    assert line == (
        "scenario-t4scene: error: --step=0.03 is not a positive multiple of the "
        "0.05 s tick\n"
    )


# -- as each run ends (t4scene.enabled) ---------------------------------------


def test_the_exporter_writes_the_scene_beside_the_log(tmp_path: Path) -> None:
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        *[_tick(frame, _state(1, 0.0, 0.0)) for frame in range(100, 104)],
    )
    scene = T4SceneExporter(step_s=0.05, max_distance_m=None).export(log)
    assert scene == tmp_path / f"Run{SCENE_SUFFIX}"
    frames = _numeric(pb.T4Scene.FromString(scene.read_bytes()).frame_indices)
    assert list(frames) == [100, 101, 102, 103]


def test_the_exporter_logs_a_log_it_cannot_convert_and_goes_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    bad = _write_log(
        tmp_path / "Bad.trajectory.jsonl", CAR, _tick(100, _state(2, 0.0, 0.0))
    )
    with caplog.at_level("WARNING"):
        assert T4SceneExporter().export(bad) is None
    assert "Bad.trajectory.jsonl" in caplog.text
    assert not (tmp_path / f"Bad{SCENE_SUFFIX}").exists()


def test_the_shipped_config_writes_no_scene_and_enabling_it_does() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    plain = _shipped([])
    assert plain.t4scene.enabled is False
    assert run.build_t4scene_exporter(plain) is None
    enabled = _shipped(["t4scene.enabled=true", "t4scene.max_distance_m=null"])
    exporter = run.build_t4scene_exporter(plain, enabled)
    assert exporter is not None
    assert (exporter.step_s, exporter.max_distance_m) == (0.1, None)


def test_a_step_that_does_not_fit_the_tick_is_refused_before_anything_runs() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    with pytest.raises(ValueError, match="t4scene.step_s"):
        run.build_t4scene_exporter(
            _shipped(["t4scene.enabled=true", "t4scene.step_s=0.07"])
        )


# Hydra reads ``true`` as a bool, which float() would take for a 1 s step:
# every scene would be written at 1 s with no error (#1502).
@pytest.mark.parametrize("override", ["true", "false", "'0.1'", "[0.1]"])
def test_a_step_that_is_not_a_number_is_refused_before_anything_runs(
    override: str,
) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    cfg = _shipped(["t4scene.enabled=true", f"t4scene.step_s={override}"])
    with pytest.raises(ValueError, match=r"^t4scene\.step_s must be a number"):
        run.build_t4scene_exporter(cfg)
    with pytest.raises(run.RunConfigError, match=r"^t4scene\.step_s must be a number"):
        run.check_run_config(cfg)


@pytest.mark.parametrize("step_s", [True, False, "0.1", None, [0.1]])
def test_the_exporter_refuses_a_step_that_is_not_a_number(step_s: Any) -> None:
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    with pytest.raises(ValueError, match=r"^t4scene\.step_s must be a number"):
        T4SceneExporter(step_s=step_s)


@pytest.mark.parametrize("step_s", [0.07, 0.0, -0.1, math.nan, math.inf])
def test_the_exporter_refuses_a_step_that_does_not_fit_the_tick(step_s: float) -> None:
    """Refused when it is built, naming the key, not per run as a logged
    warning of the best-effort export."""
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    with pytest.raises(
        ValueError, match=r"^t4scene\.step_s=.* is not a positive multiple"
    ):
        T4SceneExporter(step_s=step_s)


@pytest.mark.parametrize("step_s", [0.05, 0.1, 1, 0.15])
def test_the_exporter_takes_a_multiple_of_the_tick(step_s: float) -> None:
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    exporter = T4SceneExporter(step_s=step_s)
    assert exporter.step_s == float(step_s)
    assert type(exporter.step_s) is float


def test_the_runner_takes_a_whole_number_step() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    exporter = run.build_t4scene_exporter(
        _shipped(["t4scene.enabled=true", "t4scene.step_s=1"])
    )
    assert exporter is not None
    assert exporter.step_s == 1.0


@pytest.mark.parametrize("override", ["'100'", "[100]"])
def test_a_distance_that_is_not_a_number_is_refused_before_anything_runs(
    override: str,
) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    with pytest.raises(ValueError, match="t4scene.max_distance_m must be positive"):
        run.build_t4scene_exporter(
            _shipped(["t4scene.enabled=true", f"t4scene.max_distance_m={override}"])
        )


@pytest.mark.parametrize("seconds", [math.inf, -math.inf, math.nan])
def test_tick_multiple_refuses_a_value_that_is_not_finite(seconds: float) -> None:
    """#1384: a ValueError naming the key, not round()'s OverflowError (for
    infinity) or its unnamed ValueError (for NaN)."""
    from autoware_carla_scenario.constants import tick_multiple

    with pytest.raises(
        ValueError,
        match=rf"^t4scene\.step_s={seconds} is not a positive multiple of the 0\.05 s tick$",
    ):
        tick_multiple(seconds, "t4scene.step_s")


@pytest.mark.parametrize("override", ["inf", "nan"])
def test_the_run_check_refuses_a_step_that_is_not_finite(override: str) -> None:
    """#1384: refused as a RunConfigError naming the key, as a lanelet sweep and
    a batch refuse it, rather than an OverflowError out of the check."""
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    with pytest.raises(run.RunConfigError, match=r"^t4scene\.step_s="):
        run.check_run_config(
            _shipped(["t4scene.enabled=true", f"t4scene.step_s={override}"])
        )


# 0, a negative distance and NaN would keep no road user, infinity is what
# null says: each is refused, by every entry point, before a scene is written.
_BAD_DISTANCES = (0.0, -100.0, math.nan, math.inf)
_BAD_OVERRIDES = ("0", "-100", "nan", "inf")


def _one_car_log(tmp_path: Path) -> Path:
    return _write_log(
        tmp_path / "Run.trajectory.jsonl",
        EGO,
        CAR,
        _tick(100, _state(1, 0.0, 0.0), _state(2, 5.0, 0.0)),
    )


def _objects(scene: pb.T4Scene) -> int:
    return int(_numeric(scene.annotations.box_offsets).compressed()[-1])


@pytest.mark.parametrize("distance", _BAD_DISTANCES)
def test_the_converter_refuses_a_distance_that_is_not_positive_and_finite(
    tmp_path: Path, distance: float
) -> None:
    log = _one_car_log(tmp_path)
    with pytest.raises(ValueError, match="max_distance_m must be positive and finite"):
        trajectory_log_to_t4_scene(log, with_map=False, max_distance_m=distance)
    with pytest.raises(ValueError, match="max_distance_m must be positive and finite"):
        write_t4_scene(log, with_map=False, max_distance_m=distance)
    assert not (tmp_path / f"Run{SCENE_SUFFIX}").exists()


@pytest.mark.parametrize("distance", [10.0, 100, None])
def test_the_converter_takes_a_positive_distance_or_none(
    tmp_path: Path, distance: Optional[float]
) -> None:
    scene = trajectory_log_to_t4_scene(
        _one_car_log(tmp_path), with_map=False, max_distance_m=distance
    ).scene
    assert _objects(scene) == 1


@pytest.mark.parametrize("distance", _BAD_DISTANCES)
def test_the_exporter_refuses_a_distance_that_is_not_positive_and_finite(
    distance: float,
) -> None:
    from autoware_carla_scenario.t4_scene import T4SceneExporter

    with pytest.raises(ValueError, match="t4scene.max_distance_m must be positive"):
        T4SceneExporter(max_distance_m=distance)


# Hydra reads ``true`` as a bool, which float() would take for 1 m.
@pytest.mark.parametrize("override", (*_BAD_OVERRIDES, "true"))
def test_a_distance_that_is_not_positive_and_finite_is_refused_before_anything_runs(
    override: str,
) -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    with pytest.raises(ValueError, match="t4scene.max_distance_m must be positive"):
        run.build_t4scene_exporter(
            _shipped(["t4scene.enabled=true", f"t4scene.max_distance_m={override}"])
        )


def test_the_runner_takes_a_positive_distance() -> None:
    from autoware_carla_scenario.examples import run

    from .test_video_removed import _shipped

    exporter = run.build_t4scene_exporter(
        _shipped(["t4scene.enabled=true", "t4scene.max_distance_m=30"])
    )
    assert exporter is not None
    assert exporter.max_distance_m == 30.0


@pytest.mark.parametrize("override", _BAD_OVERRIDES)
def test_cli_refuses_a_distance_that_is_not_positive_and_finite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: str
) -> None:
    log = _one_car_log(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main([str(log), "--no-map", f"--max-distance={override}"])
    assert raised.value.code == 2
    assert "--max-distance must be positive and finite" in capsys.readouterr().err
    assert not (tmp_path / f"Run{SCENE_SUFFIX}").exists()


@pytest.mark.parametrize("override", ["0", "-0.1", "0.07", "nan", "inf"])
def test_cli_refuses_a_step_that_is_not_a_multiple_of_the_tick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: str
) -> None:
    """#1384: refused up front (2), as --max-distance is, not once per log (1)."""
    log = _one_car_log(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main([str(log), "--no-map", f"--step={override}"])
    assert raised.value.code == 2
    assert "is not a positive multiple of the 0.05 s tick" in (
        err := capsys.readouterr().err
    )
    assert "--step=" in err
    assert not (tmp_path / f"Run{SCENE_SUFFIX}").exists()


@pytest.mark.parametrize("option", [["--max-distance", "10"], ["--all-objects"]])
def test_cli_takes_a_positive_distance_or_every_object(
    tmp_path: Path, option: list[str]
) -> None:
    log = _one_car_log(tmp_path)
    assert main([str(log), "--no-map", *option]) == 0
    scene = pb.T4Scene.FromString((tmp_path / f"Run{SCENE_SUFFIX}").read_bytes())
    assert _objects(scene) == 1


_BAD_OFFSETS = (math.nan, math.inf, -math.inf)


@pytest.mark.parametrize("offset", _BAD_OFFSETS)
def test_the_converter_refuses_a_rear_axle_offset_that_is_not_finite(
    tmp_path: Path, offset: float
) -> None:
    log = _one_car_log(tmp_path)
    with pytest.raises(ValueError, match="rear_axle_offset_m must be finite"):
        trajectory_log_to_t4_scene(log, with_map=False, rear_axle_offset_m=offset)
    with pytest.raises(ValueError, match="rear_axle_offset_m must be finite"):
        write_t4_scene(log, with_map=False, rear_axle_offset_m=offset)
    assert not (tmp_path / f"Run{SCENE_SUFFIX}").exists()


# Behind the origin (the usual case), at it, or ahead of it: all places.
@pytest.mark.parametrize("offset", [-1.5, 0.0, 0.5, None])
def test_the_converter_takes_any_finite_rear_axle_offset(
    tmp_path: Path, offset: Optional[float]
) -> None:
    scene = trajectory_log_to_t4_scene(
        _one_car_log(tmp_path), with_map=False, rear_axle_offset_m=offset
    ).scene
    assert np.isfinite(_numeric(scene.ego.poses).compressed()).all()
    assert math.isfinite(scene.ego.shape.base_link_to_front)


@pytest.mark.parametrize("offset", ["NaN", "Infinity"])
def test_a_log_whose_rear_axle_record_is_not_finite_is_refused(
    tmp_path: Path, offset: str
) -> None:
    log = tmp_path / "Run.trajectory.jsonl"
    _write_log(log, EGO, _tick(100, _state(1, 0.0, 0.0)))
    # json.dumps would write the bare NaN token, which is not JSON; a string
    # is what float() reads as NaN.
    with log.open("a") as stream:
        stream.write(json.dumps({"type": "rear_axle", "id": 1, "offset": offset}))
    with pytest.raises(
        ValueError, match=r"Run.trajectory.jsonl:\d+: the rear_axle record of actor 1"
    ):
        trajectory_log_to_t4_scene(log, with_map=False)


def test_the_live_scene_refuses_a_rear_axle_offset_that_is_not_finite() -> None:
    from autoware_carla_scenario.t4_scene.live import LiveT4Scene

    with pytest.raises(ValueError, match="rear_axle_offset_m must be finite"):
        LiveT4Scene(1, rear_axle_offset_m=math.nan)
    assert LiveT4Scene(1, rear_axle_offset_m=-1.5).rear_axle_offset_m == -1.5


@pytest.mark.parametrize("override", ["nan", "inf", "-inf"])
def test_cli_refuses_a_rear_axle_offset_that_is_not_finite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: str
) -> None:
    log = _one_car_log(tmp_path)
    with pytest.raises(SystemExit) as raised:
        main([str(log), "--no-map", f"--rear-axle-offset={override}"])
    assert raised.value.code == 2
    assert "--rear-axle-offset must be finite" in capsys.readouterr().err
    assert not (tmp_path / f"Run{SCENE_SUFFIX}").exists()


@pytest.mark.parametrize("override", ["-1.5", "0", "0.5"])
def test_cli_takes_any_finite_rear_axle_offset(tmp_path: Path, override: str) -> None:
    log = _one_car_log(tmp_path)
    assert main([str(log), "--no-map", f"--rear-axle-offset={override}"]) == 0
    scene = pb.T4Scene.FromString((tmp_path / f"Run{SCENE_SUFFIX}").read_bytes())
    assert math.isfinite(scene.ego.shape.base_link_to_front)


def test_the_runner_hands_each_closed_log_to_the_exporter(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from .test_rerun import _FakeRecorder
    from .test_scenario_runner import (
        TestRunScenarioRegistersTheDefaultTimeout,
        _make_runner,
        _Stop,
    )

    runner = _make_runner(None)
    assert runner.t4scene is None
    runner.output_dir = tmp_path
    runner._world = MagicMock()
    exporter = MagicMock()
    runner.t4scene = exporter
    scenario = TestRunScenarioRegistersTheDefaultTimeout._scenario()
    type(scenario).__name__ = "CutInScenario"
    with (
        patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
        patch("autoware_carla_scenario.scenario_runner._ScenarioClock"),
        patch(
            "autoware_carla_scenario.scenario_runner.TrajectoryRecorder", _FakeRecorder
        ),
        pytest.raises(_Stop),
    ):
        runner.run_scenario(scenario, reset_world=False)
    # Even a run that raised: its log is what shows what happened.
    [call] = exporter.export.call_args_list
    (path,) = call.args
    assert path.parent == tmp_path and path.name.endswith(".trajectory.jsonl")


def test_the_queue_hands_its_exporter_to_the_runner() -> None:
    from unittest.mock import MagicMock, patch

    from autoware_carla_scenario import ScenarioQueue

    exporter = MagicMock()
    queue = ScenarioQueue(server=MagicMock(), t4scene=exporter)
    with (
        patch("autoware_carla_scenario.scenario_queue.ScenarioRunner") as runner_cls,
        patch("autoware_carla_scenario.scenario_queue.MapManager"),
    ):
        queue.start()
    assert runner_cls.return_value.t4scene is exporter
