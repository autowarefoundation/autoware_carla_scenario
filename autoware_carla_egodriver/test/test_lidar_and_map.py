"""What the runtime adds to the alpasim contract: LiDAR sweeps and map-file stop lines.

The map set here is written by hand -- a manifest, roadgen's IR and two traces --
so the reader is pinned without roadgen; the scenario framework's tests write
real sets with roadgen and resolve lights against them end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autoware_carla_egodriver import protocol
from autoware_carla_egodriver.driver import DriveContext, SessionState
from autoware_carla_egodriver.geometry import Pose
from autoware_carla_egodriver.hdmap import MapFiles
from autoware_carla_egodriver.protocol import (
    CarlaRendererData,
    LidarSweep,
    StopPoint,
    TrafficLight,
    TrafficLightState,
    Vec3,
    pack_lidar_sweep,
    unpack_lidar_points,
)

# ---------------------------------------------------------------------------
# LiDAR
# ---------------------------------------------------------------------------


def _context(
    data: CarlaRendererData, map_files: MapFiles | None = None
) -> DriveContext:
    session = SessionState(uuid="s", seed=0, scene_id="Town", cameras={})
    return DriveContext(session, 0, 100_000, data, map=map_files)


def test_a_sweep_round_trips_through_the_renderer_payload() -> None:
    points = np.random.default_rng(0).normal(size=(1000, 4)).astype(np.float32)
    sweep = pack_lidar_sweep("lidar_top", 123, points, Pose.identity().to_proto())
    payload = protocol.pack_renderer_data(CarlaRendererData(lidar=[sweep]))

    restored = protocol.unpack_renderer_data(payload)

    assert restored is not None
    out = unpack_lidar_points(restored.lidar[0])
    np.testing.assert_array_equal(out, points)
    assert out.flags.writeable
    from_context = _context(restored).lidar_points()
    assert from_context is not None
    np.testing.assert_array_equal(from_context, points)


def test_a_truncated_sweep_is_refused_rather_than_read_short() -> None:
    sweep = pack_lidar_sweep(
        "lidar_top", 0, np.zeros((10, 4)), Pose.identity().to_proto()
    )
    broken = LidarSweep()
    broken.CopyFrom(sweep)
    broken.points_xyzi = sweep.points_xyzi[:-4]
    with pytest.raises(ValueError, match="declares 10 points"):
        unpack_lidar_points(broken)


def test_packing_refuses_the_wrong_number_of_columns() -> None:
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        pack_lidar_sweep("lidar_top", 0, np.zeros((3, 5)), Pose.identity().to_proto())


def test_lidar_points_refuses_to_guess_between_several_sensors() -> None:
    pose = Pose.identity().to_proto()
    data = CarlaRendererData(
        lidar=[
            pack_lidar_sweep("lidar_top", 0, np.zeros((1, 4)), pose),
            pack_lidar_sweep("lidar_rear", 0, np.ones((2, 4)), pose),
        ]
    )
    ctx = _context(data)
    with pytest.raises(ValueError, match="name one"):
        ctx.lidar_sweep()
    points = ctx.lidar_points("lidar_rear")
    assert points is not None and points.shape == (2, 4)
    assert ctx.lidar_sweep("lidar_side") is None
    assert _context(CarlaRendererData()).lidar_points() is None


# ---------------------------------------------------------------------------
# Map files
# ---------------------------------------------------------------------------


def _write_set(directory: Path) -> Path:
    """A one-light map set: OpenDRIVE signal 5 on lane 3/0/-1, as Lanelet2."""
    directory.mkdir(parents=True)
    files = {
        "manifest.json": {
            "map_id": "Town-000000000000",
            "map_name": "Town",
            "source": "map.xodr",
            "ir": "map.ir.json",
            "read_trace": "map.roadgen.xodr.read.trace.json",
            "formats": {
                "lanelet2": {
                    "path": "lanelet2_map.osm",
                    "trace": "lanelet2_map.osm.trace.json",
                }
            },
        },
        "map.ir.json": {
            "rules": [{"id": "rule/light/5", "objects": ["object/light/5"]}],
        },
        "map.roadgen.xodr.read.trace.json": {
            "links": [
                {"ref": "signal:5", "ir": "object/light/5"},
                {"ref": "lane:3/0/-1", "ir": "lane/3/0/-1"},
            ]
        },
        "lanelet2_map.osm.trace.json": {
            "links": [
                {"ir": "lane/3/0/-1", "ref": "relation:100", "role": "lanelet"},
                {"ir": "lane/3/0/-1", "ref": "way:101", "role": "left_bound"},
                {"ir": "rule/light/5", "ref": "relation:200"},
                {"ir": "object/light/5", "ref": "way:300"},
            ]
        },
    }
    for name, content in files.items():
        (directory / name).write_text(json.dumps(content))
    (directory / "map.xodr").write_text("<OpenDRIVE/>")
    (directory / "lanelet2_map.osm").write_text("<osm/>")
    return directory


def _light(signal: str = "5", lane_id: int = -1) -> TrafficLight:
    return TrafficLight(
        opendrive_id=signal,
        state=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        stop_points=[
            StopPoint(
                road_id=3,
                section_id=0,
                lane_id=lane_id,
                position_local=Vec3(x=9.0, y=-1.5),
            )
        ],
    )


def test_a_light_resolves_to_the_formats_own_elements(tmp_path: Path) -> None:
    map_files = MapFiles.open(tmp_path, _write_set(tmp_path / "Town-000000000000").name)
    [stop] = map_files.stop_lines([_light()])

    assert stop.traffic_light_id == "5"
    assert stop.state == TrafficLightState.TRAFFIC_LIGHT_STATE_RED
    assert stop.position_local.tolist() == [9.0, -1.5, 0.0]
    # The lanelet only, not its boundaries; the kind prefix stripped.
    assert stop.lane_ids == ("100",)
    assert stop.rule_ids == ("200",)
    assert stop.light_ids == ("300",)


def test_a_light_the_map_lacks_still_reports_where_to_stop(tmp_path: Path) -> None:
    map_files = MapFiles(_write_set(tmp_path / "set"))
    [stop] = map_files.stop_lines([_light(signal="99", lane_id=-7)])
    assert stop.lane_ids == () and stop.rule_ids == () and stop.light_ids == ()
    assert stop.position_local.tolist() == [9.0, -1.5, 0.0]


def test_a_missing_set_or_format_says_what_to_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="map_dir"):
        MapFiles.open(tmp_path, "Town-000000000000")
    directory = _write_set(tmp_path / "set")
    with pytest.raises(ValueError, match="map_formats"):
        MapFiles(directory, format="clipgt")


def test_stop_lines_need_a_map_and_lights(tmp_path: Path) -> None:
    map_files = MapFiles(_write_set(tmp_path / "set"))
    assert _context(CarlaRendererData(traffic_lights=[_light()])).stop_lines() == []
    assert _context(CarlaRendererData(), map_files).stop_lines() == []
    [stop] = _context(
        CarlaRendererData(traffic_lights=[_light()]), map_files
    ).stop_lines()
    assert stop.lane_ids == ("100",)
