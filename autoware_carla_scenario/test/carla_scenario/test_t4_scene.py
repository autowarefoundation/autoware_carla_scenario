"""Trajectory logs as T4Scenes (``t4_scene``, docs/t4_scene.md): the frames,
the time grid, the objects, the lights on their lanelets, the map resource and
the command line -- all from logs written here, without CARLA."""

from __future__ import annotations

import json
import math
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
    actor_id: int, x: float, y: float, yaw: float = 0.0, vx: float = 0.0
) -> list:
    return [actor_id, x, y, 0.5, 0.0, 0.0, yaw, vx, 0.0, 0.0]


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


def test_ego_velocity_flips_y_and_speed_is_along_the_heading(tmp_path: Path) -> None:
    state = _state(1, 0.0, 0.0, yaw=-90.0)
    state[7:10] = [0.0, -5.0, 0.0]  # CARLA -y: the map's +y, the way it faces
    log = _write_log(tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, state))
    scene = trajectory_log_to_t4_scene(log, with_map=False).scene
    np.testing.assert_allclose(
        _numeric(scene.ego.velocities)[0], [0.0, 5.0], atol=1e-12
    )
    assert _numeric(scene.ego.longitudinal_speeds)[0] == pytest.approx(5.0)


def test_rear_axle_offset_can_be_given(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", EGO, _tick(100, _state(1, 0.0, 0.0))
    )
    scene = trajectory_log_to_t4_scene(
        log, with_map=False, rear_axle_offset_m=-1.5
    ).scene
    assert _numeric(scene.ego.poses)[0][0] == pytest.approx(1000.0 - 1.5)
    assert scene.ego.shape.base_link_to_front == pytest.approx(0.1 + 2.4 + 1.5)


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
    assert list(_numeric(scene.annotations.offsets)) == [0, 1, 2, 2, 3, 4]
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
    values = trajectory_log_to_t4_scene(log, with_map=False).scene.annotations.values
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
    assert list(_numeric(near.annotations.offsets)) == [0, 0]
    everyone = trajectory_log_to_t4_scene(
        log, with_map=False, max_distance_m=None
    ).scene
    assert list(_numeric(everyone.annotations.offsets)) == [0, 1]


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
    assert list(np.diff(_numeric(scene.annotations.offsets))) == [0, 1, 1, 0]


# -- the map and its lights ---------------------------------------------------


@pytest.fixture(scope="module")
def fixture_map() -> tuple[Any, tuple[float, float]]:
    """The nishishinjuku map as the run reads it, and its MGRS offset."""
    (lat, lon, alt), _ = map_origin(OSM, XODR)
    lanelet_map, projector, _ = load_lanelet2(OSM, lanelet2.io.Origin(lat, lon))
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
    lanelet_id, stop, heading = next(iter(_stop_lines(lanelet_map)))
    # The map frame back to CARLA's: y and yaw flip.
    x, y, yaw = stop[0] - ox, -(stop[1] - oy), -math.degrees(heading)
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
    return log, lanelet_id


def test_the_map_is_a_resource_the_scene_names(map_log: tuple[Path, int]) -> None:
    log, _ = map_log
    export = trajectory_log_to_t4_scene(log)
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


def test_a_light_colours_the_lanelet_it_governs(map_log: tuple[Path, int]) -> None:
    log, lanelet_id = map_log
    lights = trajectory_log_to_t4_scene(log, step_s=0.05).scene.traffic_lights
    offsets = list(_numeric(lights.lane_offsets))
    lane_ids = _numeric(lights.values.lane_ids)
    states = _numeric(lights.values.lane_states)
    rows = [
        dict(zip(lane_ids[a:b].tolist(), states[a:b].tolist()))
        for a, b in zip(offsets, offsets[1:])
    ]
    # Red, carried over the tick without a change, then green.
    assert [row[lanelet_id] for row in rows] == [1, 1, 3]
    assert list(_numeric(lights.available)) == [True, True, True]
    assert list(_numeric(lights.lane_available)) == [True, True, True]
    # The map's regulatory elements and their bulbs are not observed: no raw
    # groups, and said so, rather than an observed empty set of them.
    assert not lights.values.HasField("group_ids")
    assert not lights.values.HasField("element_offsets")
    assert list(_numeric(lights.group_offsets)) == [0, 0, 0, 0]
    assert list(_numeric(lights.group_available)) == [False, False, False]


def test_the_resource_keeps_exact_local_coordinates(
    map_log: tuple[Path, int], fixture_map: tuple[Any, tuple[float, float]]
) -> None:
    log, _ = map_log
    resource = trajectory_log_to_t4_scene(log).map_resource
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


def test_cli_fails_on_a_log_it_cannot_convert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = _write_log(
        tmp_path / "Bad.trajectory.jsonl", CAR, _tick(100, _state(2, 0.0, 0.0))
    )
    assert main([str(bad), "--no-map"]) == 1
    assert "no ego" in capsys.readouterr().err
