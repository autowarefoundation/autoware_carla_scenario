"""The helpers #2398 shares give what each inline copy they replace gave.

Each test keeps the replaced expression, as it was written, and compares it
with the helper over random inputs and the edge cases: the XY flip between
CARLA's frame and the Lanelet2 map's, the CARLA pose the trajectory follower
reads, the tick length with and without a fallback, the projector descriptor
the trajectory log records and the dotted keys a composition error suggests.
"""

from __future__ import annotations

import math
import random
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import yaml
from carla_driver_interface.geometry import Pose

from autoware_carla_scenario.constants import (
    FIXED_DELTA_SECONDS,
    tick_seconds,
    tick_seconds_or_fixed,
)
from autoware_carla_scenario.coordinate.projection import read_projector_info
from autoware_carla_scenario.coordinate.transform import (
    carla_xy_to_map,
    map_xy_to_carla,
)
from autoware_carla_scenario.driver.observation import to_local_pose
from autoware_carla_scenario.maps.resolver import projector_info_path
from autoware_carla_scenario.odd.controls import config_keys


def _bits(value: float) -> bytes:
    """*value*'s IEEE 754 bits: equal ones are the same float, signed zero
    and all."""
    return struct.pack("<d", value)


def _same(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return [_bits(v) for v in a] == [_bits(v) for v in b]


def _points(seed: int) -> list[tuple[float, float, float, float]]:
    rng = random.Random(seed)
    points = []
    for _ in range(2000):
        x, y = rng.uniform(-1e4, 1e4), rng.uniform(-1e4, 1e4)
        offset = (rng.uniform(-1e6, 1e6), rng.uniform(-1e7, 1e7))
        points.append((x, y, *offset))
    # The edges: a point on the offset itself (a zero whose sign the two
    # spellings of the flip could tell apart), zeros, and huge values.
    points += [
        (0.0, 0.0, 0.0, 0.0),
        (-0.0, -0.0, 0.0, 0.0),
        (5.0, 7.0, 5.0, 7.0),
        (-5.0, -7.0, 5.0, -7.0),
        (1e300, -1e300, 1e300, 1e300),
    ]
    return points


def test_carla_xy_to_map_is_the_inline_flip_it_replaces() -> None:
    for x, y, offset_x, offset_y in _points(2398):
        # measures, routing, relative_lane, route_progress, _walker_ai, and
        # transform's own _carla_to_lanelet2_frame, to_map_frame and
        # _carla_to_lanelet2.
        inline = (x + offset_x, -y + offset_y)
        assert _same(carla_xy_to_map(x, y, (offset_x, offset_y)), inline)


def test_map_xy_to_carla_is_the_inline_flip_it_replaces() -> None:
    for x, y, offset_x, offset_y in _points(2399):
        # measures, entity_area_distance, resolve.map_pose_to_carla,
        # _walker_ai, and transform's own _lanelet2_to_carla.
        inline = (x - offset_x, -(y - offset_y))
        assert _same(map_xy_to_carla(x, y, (offset_x, offset_y)), inline)


def test_the_two_flips_undo_each_other() -> None:
    for x, y, offset_x, offset_y in _points(2400)[:2000]:
        offset = (offset_x, offset_y)
        back = map_xy_to_carla(*carla_xy_to_map(x, y, offset), offset)
        assert back == pytest.approx((x, y), abs=1e-6)


def _transform(x: float, y: float, z: float, yaw: float) -> Any:
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(yaw=yaw, pitch=0.0, roll=0.0),
    )


def test_to_local_pose_is_the_follower_pose_it_replaces() -> None:
    # actions._trajectory_run (FOLLOW of a vehicle) and traffic.final_approach
    # handed the follower this pose of the actor's transform.
    rng = random.Random(2401)
    cases = [(0.0, 0.0, 0.0, 0.0), (-0.0, -0.0, -0.0, -180.0), (1.0, 2.0, 3.0, 720.0)]
    cases += [
        (
            rng.uniform(-1e4, 1e4),
            rng.uniform(-1e4, 1e4),
            rng.uniform(-100.0, 100.0),
            rng.uniform(-720.0, 720.0),
        )
        for _ in range(500)
    ]
    for x, y, z, yaw in cases:
        transform = _transform(x, y, z, yaw)
        inline = Pose.from_xyz_yaw(
            transform.location.x,
            -transform.location.y,
            transform.location.z,
            -math.radians(transform.rotation.yaw),
        )
        assert np.array_equal(to_local_pose(transform).as_matrix(), inline.as_matrix())


class _World:
    def __init__(self, delta: Any = None, error: Optional[BaseException] = None):
        self._delta = delta
        self._error = error

    def get_snapshot(self) -> Any:
        if self._error is not None:
            raise self._error
        return SimpleNamespace(timestamp=SimpleNamespace(delta_seconds=self._delta))


def test_tick_seconds_reads_the_snapshot_and_raises_what_it_raises() -> None:
    # actions._trajectory_run and actions.set_speed had no fallback.
    assert tick_seconds(_World(0.1)) == 0.1  # type: ignore[arg-type]
    assert tick_seconds(_World(0.0)) == 0.0  # type: ignore[arg-type]
    for error in (AttributeError("no snapshot"), RuntimeError("server gone")):
        with pytest.raises(type(error)):
            tick_seconds(_World(error=error))  # type: ignore[arg-type]


def test_tick_seconds_or_fixed_falls_back_where_the_copies_did() -> None:
    # traffic.traffic_manager fell back on FIXED_DELTA_SECONDS, and
    # entity._walker_ai on a 0.05 of its own: the same value.
    assert FIXED_DELTA_SECONDS == 0.05
    assert tick_seconds_or_fixed(_World(0.1)) == 0.1  # type: ignore[arg-type]
    assert tick_seconds_or_fixed(_World(0.0)) == 0.0  # type: ignore[arg-type]
    for error in (AttributeError("no snapshot"), RuntimeError("server gone")):
        world = _World(error=error)
        assert tick_seconds_or_fixed(world) == 0.05  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tick_seconds_or_fixed(_World(error=ValueError("other")))  # type: ignore[arg-type]


def _old_recorder_projector_info(lanelet2_path: Path) -> Optional[dict[str, Any]]:
    """trajectory_recorder._projector_info before #2398, behind the
    ``_metadata`` wrapper that turned any error into None."""
    try:
        info_path = projector_info_path(Path(lanelet2_path).parent)
        if info_path is None:
            return None
        info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else None
    except Exception:  # noqa: BLE001 - what _metadata did
        return None


@pytest.mark.parametrize(
    "body",
    [
        None,  # no descriptor
        b"projector_type: MGRS\nvertical_datum: WGS84\n",
        b"{}\n",  # an empty mapping is recorded as one
        b"",  # an empty file: None, not a mapping
        b"- MGRS\n",
        b"MGRS\n",
        b"projector_type: [MGRS\n",  # unparsable
        b"projector_type: MGRS # \x93\n",  # not UTF-8
        b"map_origin: {latitude: 35.6, longitude: 139.7}\n",
    ],
)
def test_the_recorded_projector_info_is_what_the_recorder_read(
    tmp_path: Path, body: Optional[bytes]
) -> None:
    osm = tmp_path / "lanelet2_map.osm"
    osm.write_bytes(b"<osm/>")
    if body is not None:
        (tmp_path / "map_projector_info.yaml").write_bytes(body)
    assert read_projector_info(osm) == _old_recorder_projector_info(osm)


def test_a_descriptor_linked_from_outside_the_map_is_none(tmp_path: Path) -> None:
    directory = tmp_path / "map"
    directory.mkdir()
    osm = directory / "lanelet2_map.osm"
    osm.write_bytes(b"<osm/>")
    (tmp_path / "outside.yaml").write_text("projector_type: MGRS\n")
    (directory / "map_projector_info.yaml").symlink_to(tmp_path / "outside.yaml")
    assert read_projector_info(osm) is None
    assert _old_recorder_projector_info(osm) is None


def _old_flatten(node: Any, prefix: str = "") -> list[str]:
    """examples.compose_errors._flatten before #2398."""
    if not isinstance(node, dict):
        return []
    keys: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.append(path)
        keys += _old_flatten(value, path)
    return keys


def test_config_keys_is_the_composition_error_s_flatten() -> None:
    # What examples.run._to_dict hands it: OmegaConf.to_container's plain
    # dicts and lists, scalars and None.
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(
        OmegaConf.create(
            {
                "ego": {"initial_speed_kmh": 30, "route": [1, 2, {"x": 1}]},
                "traffic": {"backend": None, "sumo": {"step": 0.05}},
                3: "int key",
                "empty": {},
            }
        ),
        resolve=True,
    )
    assert config_keys(container, "") == _old_flatten(container, "")
    assert config_keys([1, 2], "") == _old_flatten([1, 2], "") == []
    assert config_keys("text") == _old_flatten("text") == []
