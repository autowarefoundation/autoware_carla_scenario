"""Unit tests for TrajectoryRecorder (no CARLA server)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import jsonschema
import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.conditions.base import ConditionStatus, ScenarioResult
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.debug_markers import DebugMarkers, DebugPolyline
from autoware_carla_scenario.driver.geometry import Pose, Trajectory
from autoware_carla_scenario.ego_plan import EgoPlan, PlanMarkers, ego_plan_markers
from autoware_carla_scenario.trajectory_recorder import (
    DRAFT7_SCHEMA_PATH,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    TrajectoryRecorder,
)


def _vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _rot(yaw: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw)


class _Actor:
    def __init__(self, actor_id: int, type_id: str, role: str = "") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self.bounding_box = SimpleNamespace(
            extent=_vec(2.45, 1.07, 0.77), location=_vec(0.0, 0.0, 0.7)
        )

    def get_control(self) -> SimpleNamespace:
        return SimpleNamespace(throttle=0.5, steer=0.0, brake=0.0, gear=1)


class _ActorSnapshot:
    def __init__(
        self, actor_id: int, x: float, vx: float, wz: float = 0.0, z: float = 0.0
    ) -> None:
        self.id = actor_id
        self._x, self._vx, self._wz, self._z = x, vx, wz, z

    def get_transform(self) -> SimpleNamespace:
        return SimpleNamespace(
            location=_vec(self._x, 1.0, self._z), rotation=_rot(90.0)
        )

    def get_velocity(self) -> SimpleNamespace:
        return _vec(self._vx)

    def get_angular_velocity(self) -> SimpleNamespace:
        return _vec(0.0, 0.0, self._wz)


class _Light:
    def __init__(self, light_id: int) -> None:
        self.id = light_id
        # A plain int, as typesafe_carla's TrafficLight.state gives.
        self.state = int(carla.TrafficLightState.Red)

    def get_location(self) -> SimpleNamespace:
        return _vec(10.0, 20.0, 0.0)

    def get_light_boxes(self) -> list:
        return [
            SimpleNamespace(
                location=_vec(10.0, 20.0, 5.0),
                extent=_vec(0.2, 0.3, 0.6),
                rotation=_rot(180.0),
            )
        ]

    def get_stop_waypoints(self) -> list:
        return [
            SimpleNamespace(
                transform=SimpleNamespace(
                    location=_vec(5.0, 18.0, 0.0), rotation=_rot(-90.0)
                )
            )
        ]

    def get_opendrive_id(self) -> str:
        return "42"


class _World:
    def __init__(self) -> None:
        self.id = 7
        self.actors = {
            1: _Actor(1, "vehicle.lincoln.mkz", "Ego"),
            2: _Actor(2, "sensor.camera.rgb"),
        }
        self.light = _Light(9)
        self.snapshot: list[_ActorSnapshot] = []
        self.frame = 100

    #: What get_vehicles_light_states() gives; None: a world that cannot say.
    vehicle_light_states: dict[int, int] | None = None

    def get_vehicles_light_states(self) -> dict[int, int]:
        if self.vehicle_light_states is None:
            raise RuntimeError("no vehicle light states in this world")
        return dict(self.vehicle_light_states)

    def get_map(self) -> SimpleNamespace:
        return SimpleNamespace(name="Carla/Maps/Town10HD_Opt")

    def get_actors(self, ids: list[int] | None = None):
        if ids is None:
            actors = MagicMock()
            actors.filter.return_value = [self.light]
            return actors
        return [self.actors[i] for i in ids if i in self.actors]

    def get_snapshot(self) -> "_Snapshot":
        return _Snapshot(self.snapshot, self.frame)


class _Snapshot:
    """Like carla.WorldSnapshot: iterable (repeatedly) over actor snapshots."""

    def __init__(self, actors: list[_ActorSnapshot], frame: int) -> None:
        self._actors = actors
        self.frame = frame
        # 20 Hz, the framework's fixed step.
        self.timestamp = SimpleNamespace(elapsed_seconds=0.05 * frame)

    def __iter__(self):
        return iter(self._actors)


@cache
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


@cache
def _draft7_validator() -> jsonschema.Draft7Validator:
    return jsonschema.Draft7Validator(json.loads(DRAFT7_SCHEMA_PATH.read_text()))


def _lines(path: Path) -> list[dict]:
    """The file's records, each checked against both published schemas."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        _validator().validate(record)
        _draft7_validator().validate(record)
    return records


def test_the_schema_is_a_valid_2020_12_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(_validator().schema)
    assert _validator().schema["x-schema-version"] == SCHEMA_VERSION


def _draft7_script() -> Any:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "trajectory_schema_draft7.py"
    )
    spec = importlib.util.spec_from_file_location("trajectory_schema_draft7", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_draft_7_copy_is_generated_from_the_2020_12_schema() -> None:
    jsonschema.Draft7Validator.check_schema(_draft7_validator().schema)
    assert DRAFT7_SCHEMA_PATH.read_text() == _draft7_script().render(
        json.loads(SCHEMA_PATH.read_text())
    ), "run autoware_carla_scenario/scripts/trajectory_schema_draft7.py"


def test_the_draft_7_conversion_repoints_only_references() -> None:
    literal = "#/$defs/A"
    schema: dict[str, Any] = {
        "$defs": {"A": {"type": "string"}},
        "oneOf": [{"$ref": "#/$defs/A"}],
        "discriminator": {"propertyName": "type", "mapping": {"a": "#/$defs/A"}},
        "properties": {
            "c": {"const": literal},
            "e": {"enum": [literal], "default": literal, "examples": [literal]},
            "t": {"prefixItems": [{"$ref": "#/$defs/A"}], "items": {"type": "null"}},
        },
    }
    out = _draft7_script().to_draft7(schema)
    assert out["definitions"] == {"A": {"type": "string"}}
    assert out["oneOf"] == [{"$ref": "#/definitions/A"}]
    assert out["discriminator"]["mapping"] == {"a": "#/definitions/A"}
    assert out["properties"]["c"] == {"const": literal}
    assert out["properties"]["e"] == schema["properties"]["e"]
    assert out["properties"]["t"] == {
        "items": [{"$ref": "#/definitions/A"}],
        "additionalItems": {"type": "null"},
    }


def test_every_record_type_is_in_the_discriminator() -> None:
    schema = _validator().schema
    mapping = schema["discriminator"]["mapping"]
    assert [r["$ref"] for r in schema["oneOf"]] == list(mapping.values())
    for name, target in mapping.items():
        definition = schema["$defs"][target.rsplit("/", 1)[-1]]
        assert definition["properties"]["type"] == {"const": name}


def _resolve(schema: dict, pointer: str) -> object:
    """The node *pointer* (a local ``#/...`` JSON pointer) names in *schema*."""
    assert pointer.startswith("#/"), pointer
    node: object = schema
    for token in pointer[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        else:
            assert isinstance(node, dict) and token in node, pointer
            node = node[token]
    return node


def _pointers(node: object) -> list[str]:
    """Every ``$ref`` and ``discriminator.mapping`` value under *node*."""
    found: list[str] = []
    if isinstance(node, list):
        for item in node:
            found += _pointers(item)
    elif isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            found.append(node["$ref"])
        discriminator = node.get("discriminator")
        if isinstance(discriminator, dict):
            found += list(discriminator.get("mapping", {}).values())
        for value in node.values():
            found += _pointers(value)
    return found


@pytest.mark.parametrize(
    "path", [SCHEMA_PATH, DRAFT7_SCHEMA_PATH], ids=lambda p: p.name
)
def test_every_pointer_in_the_schema_resolves(path: Path) -> None:
    schema = json.loads(path.read_text())
    pointers = _pointers(schema)
    mapping = list(schema["discriminator"]["mapping"].values())
    assert mapping and set(mapping) <= set(pointers)
    for pointer in pointers:
        assert isinstance(_resolve(schema, pointer), dict), pointer


def test_an_episode_id_carla_hands_over_signed_is_written_unsigned(
    tmp_path: Path,
) -> None:
    # CARLA's episode id is a random uint64; its Python API gives one past
    # 2^63 as a negative int, which a reader of the log rejects as a u64.
    world = _World()
    world.id = -6682850984959527471
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    recorder.close()
    assert _lines(recorder.path)[0]["episode"] == 2**64 - 6682850984959527471


def test_it_writes_the_world_actors_and_each_tick(tmp_path: Path) -> None:
    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "out" / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0), _ActorSnapshot(2, 0.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1
    world.snapshot = [
        _ActorSnapshot(1, 0.5, 10.0, wz=-12.3456),
        _ActorSnapshot(2, 0.5, 10.0, wz=7.0),
    ]
    world.light.state = int(carla.TrafficLightState.Green)
    recorder.record(world, 0.05)
    recorder.close()

    records = _lines(recorder.path)
    assert [r["type"] for r in records] == [
        "world",
        "map",
        "traffic_light",
        "actor",
        "rear_axle",
        "tick",
        "tick",
    ]
    assert records[0] == {
        "type": "world",
        "schema_version": 2,
        "map": "Carla/Maps/Town10HD_Opt",
        "episode": 7,
    }
    assert records[2]["opendrive_id"] == "42"
    assert records[2]["boxes"] == [[10.0, 20.0, 5.0, 0.2, 0.3, 0.6, 180.0]]
    assert records[2]["stops"] == [[5.0, 18.0, 0.0, -90.0]]
    # Only vehicles/walkers are recorded; the camera is looked up once and skipped.
    assert records[3]["id"] == 1 and records[3]["role"] == "Ego"
    # The ego's rear axle: half its box behind, the vehicle having no wheels.
    assert records[4] == {"type": "rear_axle", "id": 1, "offset": -1.225}
    first, second = records[5], records[6]
    assert first["actors"] == [[1, 0.0, 1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0]]
    assert first["control"] == {"1": [0.5, 0.0, 0.0, 1]}
    assert first["angular_velocity"] == {"1": [0.0, 0.0, 0.0]}
    assert first["lights"] == {"9": "Red"}
    assert second["t"] == 0.05 and second["frame"] == 101
    assert second["actors"][0][7] == 10.0
    # The ego's angular velocity as CARLA gives it (deg/s), to the thousandth.
    assert second["angular_velocity"] == {"1": [0.0, 0.0, -12.346]}
    # Only the light states that changed since the last written tick.
    assert second["lights"] == {"9": "Green"}
    # orjson writes what the stdlib wrote before it, byte for byte.
    for line in recorder.path.read_text().splitlines():
        assert line == json.dumps(json.loads(line), separators=(",", ":"))


class _UnnamedLight(_Light):
    def get_opendrive_id(self) -> str:
        raise RuntimeError("no signal")


def test_the_rear_axle_is_the_one_a_driver_entity_offered(tmp_path: Path) -> None:
    from autoware_carla_scenario import t4

    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    t4.offer_driver_sensors(-1.4)
    try:
        recorder.record(world, 0.0)
    finally:
        t4.withdraw_driver_sensors()
    recorder.close()
    rear_axles = [r for r in _lines(recorder.path) if r["type"] == "rear_axle"]
    assert rear_axles == [{"type": "rear_axle", "id": 1, "offset": -1.4}]


def test_records_without_an_ego_or_a_light_change_follow_the_schema(
    tmp_path: Path,
) -> None:
    world = _World()
    world.actors = {
        3: _Actor(3, "walker.pedestrian.0001"),
        4: _Actor(4, "vehicle.tesla.model3"),
    }
    world.light = _UnnamedLight(9)
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(3, 0.0, 0.0), _ActorSnapshot(4, 0.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1
    recorder.record(world, 0.05)
    recorder.close()

    records = _lines(recorder.path)
    assert records[2]["opendrive_id"] == ""
    assert [r["type_id"] for r in records if r["type"] == "actor"] == [
        "walker.pedestrian.0001",
        "vehicle.tesla.model3",
    ]
    first, second = (r for r in records if r["type"] == "tick")
    assert first["control"] == {} and first["lights"] == {"9": "Red"}
    assert "angular_velocity" not in first  # an ego's only
    assert "lights" not in second  # the light did not change


_TICK = {
    "type": "tick",
    "frame": 101,
    "t": 0.05,
    "actors": [[1, 0.0, 1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0]],
    "control": {"1": [0.5, 0.0, 0.0, 1]},
}


@pytest.mark.parametrize(
    "record",
    [
        {**_TICK, "type": "frame"},  # an unknown record type
        {**_TICK, "actors": [[1, 0.0, 1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0]]},  # short
        {**_TICK, "lights": {"9": "Blue"}},  # no such light state
        {**_TICK, "control": {"ego": [0.5, 0.0, 0.0, 1]}},  # a key that is no id
        {**_TICK, "accel": [0.0, 0.0, 0.0]},  # a field the format does not have
        {**_TICK, "angular_velocity": {"1": [0.0, 0.0]}},  # not a vector
        {**_TICK, "angular_velocity": {"ego": [0.0, 0.0, 1.0]}},  # a key that is no id
        {**_TICK, "frame": 2**53},  # an integer a double cannot hold exactly
        {**_TICK, "frame": -1},  # LibCarla's frame is unsigned
        {"type": "world", "map": "m", "episode": 2**64},  # past a uint64
        {"type": "world", "map": "m", "episode": 1, "schema_version": 0},
        {
            "type": "condition",
            "frame": 1,
            "t": 0.0,
            "role": "maybe",
            "index": 0,
            "label": "x",
            "satisfied": True,
        },  # neither pass nor fail
        {
            "type": "markers",
            "frame": 1,
            "t": 0.0,
            "actor": 1,
            "polylines": [
                {
                    "points": [[0.0, 0.0]],
                    "color": [0, 255, 0],
                    "thickness": 0.1,
                    "point_size": 0.0,
                }
            ],
        },  # a point of two coordinates
        {
            "type": "markers",
            "frame": 1,
            "t": 0.0,
            "actor": 1,
            "polylines": [
                {
                    "points": [],
                    "color": [0, 256, 0],
                    "thickness": 0.1,
                    "point_size": 0.0,
                }
            ],
        },  # no such colour
    ],
)
def test_the_schema_rejects_lines_the_recorder_does_not_write(record: dict) -> None:
    assert not _validator().is_valid(record)


def test_a_tick_with_or_without_an_angular_velocity_follows_the_schema() -> None:
    # Without: a file written before ticks carried it.
    for tick in (_TICK, {**_TICK, "angular_velocity": {"1": [0.1, -0.2, 5.0]}}):
        _validator().validate(tick)
        _draft7_validator().validate(tick)


def test_ticks_without_vehicles_are_not_written(tmp_path: Path) -> None:
    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(2, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.close()

    assert [r["type"] for r in _lines(recorder.path)] == [
        "world",
        "map",
        "traffic_light",
    ]


class _TimedOutWorld(_World):
    def get_snapshot(self) -> _Snapshot:
        raise RuntimeError("time-out")


def test_an_error_stops_recording_without_raising(tmp_path: Path) -> None:
    world = _TimedOutWorld()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)

    recorder.record(world, 0.0)  # must not raise

    assert not recorder.active
    recorder.record(world, 0.05)  # and stays off


class _FailingFile:
    def write(self, data: bytes) -> int:
        raise OSError("No space left on device")

    def close(self) -> None:
        raise OSError("No space left on device")


def test_a_failing_write_and_close_do_not_raise(tmp_path: Path) -> None:
    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    recorder._out = _FailingFile()  # type: ignore[assignment]
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]

    recorder.record(world, 0.0)  # the write fails, then so does the close

    assert not recorder.active
    recorder.close()  # and closing again is a no-op


def test_a_file_of_the_first_version_still_follows_the_schema() -> None:
    # What a recorder before version 2 wrote: no schema_version, no new records.
    for record in [
        {
            "type": "world",
            "map": "Carla/Maps/Town10HD_Opt",
            "episode": 1687550558397697221,
        },
        _TICK,
    ]:
        _validator().validate(record)
        _draft7_validator().validate(record)


# ---------------------------------------------------------------------------
# Version 2: the map, the warm-up, actors that go, markers, conditions, result
# ---------------------------------------------------------------------------


def test_the_map_record_names_and_hashes_the_files_the_run_read(tmp_path: Path) -> None:
    osm = tmp_path / "lanelet2_map.osm"
    osm.write_bytes(b"<osm/>")
    (tmp_path / "map_projector_info.yaml").write_text(
        "projector_type: MGRS\nvertical_datum: WGS84\nmgrs_grid: 54SUE\n"
    )
    xodr = tmp_path / "map.xodr"
    xodr.write_bytes(b"<OpenDRIVE/>")
    recorder = TrajectoryRecorder(
        tmp_path / "s.trajectory.jsonl",
        "git+https://example.org/maps@abc#Town10HD_Opt",
        osm,
        xodr,
        None,
    )
    recorder.start(_World())
    recorder.close()

    [record] = [r for r in _lines(recorder.path) if r["type"] == "map"]
    assert record["name"] == "Town10HD_Opt"
    assert record["source"] == "git+https://example.org/maps@abc#Town10HD_Opt"
    assert record["lanelet2"] == {
        "path": str(osm.resolve()),
        "sha256": hashlib.sha256(b"<osm/>").hexdigest(),
        "bytes": 6,
    }
    assert record["opendrive"]["sha256"] == hashlib.sha256(b"<OpenDRIVE/>").hexdigest()
    assert record["projector_info"] == {
        "projector_type": "MGRS",
        "vertical_datum": "WGS84",
        "mgrs_grid": "54SUE",
    }
    assert record["projector_type"] is None


def test_the_map_record_carries_the_loaded_lanelet2_maps_offsets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        MapManager,
        "get_instance",
        classmethod(
            lambda cls: SimpleNamespace(
                mgrs_offset=(81655.0, 50137.5), z_offset=-1.25, carla_map=None
            )
        ),
    )
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(_World())
    recorder.close()
    [record] = [r for r in _lines(recorder.path) if r["type"] == "map"]
    assert record["mgrs_offset"] == [81655.0, 50137.5]
    assert record["z_offset"] == -1.25
    assert record["lanelet2"] is None and record["source"] is None


def _run(tmp_path: Path) -> tuple[TrajectoryRecorder, _World]:
    world = _World()
    world.actors[3] = _Actor(3, "walker.pedestrian.0001")
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    recorder.record_warmup(66, 34, 100)
    return recorder, world


def _plan(*xs: float) -> DebugMarkers:
    markers = DebugMarkers()
    markers.add_polyline(
        DebugPolyline([(x, 1.0, 0.3) for x in xs], (0, 255, 0), 0.1, 0.1)
    )
    return markers


def test_the_warmup_record_says_where_the_recording_and_the_file_start(
    tmp_path,
) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.close()
    records = _lines(recorder.path)
    assert [r["type"] for r in records][:5] == [
        "world",
        "map",
        "traffic_light",
        "warmup",
        "actor",
    ]
    assert records[3] == {
        "type": "warmup",
        "ticks": 66,
        "start_frame": 34,
        "end_frame": 100,
    }


def test_no_recording_means_no_start_frame(tmp_path) -> None:
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(_World())
    recorder.record_warmup(0, -1, 100)
    recorder.close()
    [record] = [r for r in _lines(recorder.path) if r["type"] == "warmup"]
    assert record["start_frame"] is None


def test_actors_that_come_later_and_go_are_written_when_they_do(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1  # a pedestrian spawned
    world.snapshot = [_ActorSnapshot(1, 0.5, 0.0), _ActorSnapshot(3, 5.0, 0.0)]
    recorder.record(world, 0.05)
    world.frame += 1  # and destroyed
    world.snapshot = [_ActorSnapshot(1, 1.0, 0.0)]
    recorder.record(world, 0.1)
    recorder.close()

    records = [r for r in _lines(recorder.path) if r["type"] != "traffic_light"]
    kinds = [(r["type"], r.get("id")) for r in records[3:]]
    assert kinds == [
        ("actor", 1),
        ("rear_axle", 1),
        ("tick", None),
        ("actor", 3),
        ("tick", None),
        ("actor_gone", 3),
        ("tick", None),
    ]
    gone = records[-2]
    assert gone == {"type": "actor_gone", "id": 3, "frame": 102, "t": 0.1}
    assert [row[0] for row in records[-1]["actors"]] == [1]


def test_markers_are_written_when_they_change(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    plan = _plan(0.0, 5.0)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    for tick, handed in enumerate(
        [plan, plan, _plan(1.0), DebugMarkers(), DebugMarkers()]
    ):
        world.frame = 100 + tick
        recorder.record(world, 0.05 * tick)
        recorder.record_markers(handed)
    recorder.close()

    markers = [r for r in _lines(recorder.path) if r["type"] == "markers"]
    # The same plan twice is written once, no markers at all once, as none.
    assert [(m["frame"], len(m["polylines"])) for m in markers] == [
        (100, 1),
        (102, 1),
        (103, 0),
    ]
    assert markers[0]["actor"] == 1
    assert markers[0]["polylines"] == [
        {
            "points": [[0.0, 1.0, 0.3], [5.0, 1.0, 0.3]],
            "color": [0, 255, 0],
            "thickness": 0.1,
            "point_size": 0.1,
        }
    ]


def _ego_plan(step: int, *xs: float) -> PlanMarkers:
    markers = PlanMarkers(
        EgoPlan(
            step=step,
            planned_at_us=step * 100_000,
            points=[(0.1 * (i + 1), x, 1.0, 0.0, -90.0) for i, x in enumerate(xs)],
            policy_name="fake",
            inference_seconds=0.0123456,
            turn_indicator=2,
            debug_scalars={"speed": 3.14159, "nan": float("nan")},
        )
    )
    markers.add_polyline(
        DebugPolyline([(x, 1.0, 0.3) for x in xs], (0, 255, 0), 0.1, 0.1)
    )
    return markers


def test_a_plan_is_written_after_its_markers_once_per_plan(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    first, second = _ego_plan(1, 0.0, 5.0), _ego_plan(2, 1.0)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    for tick, handed in enumerate([first, first, second, _plan(2.0)]):
        world.frame = 100 + tick
        recorder.record(world, 0.05 * tick)
        recorder.record_markers(handed)
    recorder.close()

    records = [r for r in _lines(recorder.path) if r["type"] in ("markers", "plan")]
    # Each plan right after its markers; markers that carry no plan, alone.
    assert [(r["type"], r["frame"]) for r in records] == [
        ("markers", 100),
        ("plan", 100),
        ("markers", 102),
        ("plan", 102),
        ("markers", 103),
    ]
    assert records[1] == {
        "type": "plan",
        "frame": 100,
        "t": 0.0,
        "actor": 1,
        "step": 1,
        "planned_at_us": 100_000,
        "policy": "fake",
        "inference_seconds": 0.012,
        "turn_indicator": 2,
        "debug_scalars": {"speed": 3.142, "nan": None},
        "points": [[0.1, 0.0, 1.0, 0.0, -90.0], [0.2, 5.0, 1.0, 0.0, -90.0]],
    }


def test_a_plan_with_nothing_to_draw_is_still_written(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    empty = PlanMarkers(
        EgoPlan(
            step=3,
            planned_at_us=0,
            points=[],
            policy_name="",
            inference_seconds=float("nan"),
            turn_indicator=1,
            debug_scalars={},
        )
    )
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.record_markers(empty)
    recorder.close()

    records = [r for r in _lines(recorder.path) if r["type"] in ("markers", "plan")]
    # Its markers clear the drawing; the plan, its command and all, is kept.
    assert [r["type"] for r in records] == ["markers", "plan"]
    assert records[0]["polylines"] == []
    assert records[1]["points"] == [] and records[1]["turn_indicator"] == 1
    assert records[1]["inference_seconds"] is None  # NaN: what JSON cannot hold


def test_a_plan_leaves_out_a_point_that_is_not_finite() -> None:
    trajectory = Trajectory.empty()
    for i, x in enumerate([1.0, float("nan"), 3.0]):
        trajectory.append(i * 100_000, Pose.from_xyz_yaw(x, 2.0, 0.0, 0.0))
    markers = ego_plan_markers(trajectory, 0, 1)
    assert [p[1] for p in markers.plan.points] == [1.0, 3.0]
    assert [p[2] for p in markers.plan.points] == [-2.0, -2.0]  # y flipped back
    assert [x for x, _, _ in markers.polylines[0].points] == [1.0, 3.0]


@pytest.mark.parametrize(
    "change",
    [
        {"points": [[0.1, 0.0, 1.0, 0.0]]},  # a point without its yaw
        {"step": 0},  # plans count from 1
        {"turn_indicator": -1},
        {"debug_scalars": {"speed": "fast"}},
        {"planned_at_us": -1},
    ],
)
def test_the_schema_rejects_a_plan_the_recorder_does_not_write(change: dict) -> None:
    plan = {
        "type": "plan",
        "frame": 1,
        "t": 0.0,
        "actor": 1,
        "step": 1,
        "planned_at_us": 0,
        "policy": "",
        "inference_seconds": 0.0,
        "turn_indicator": 0,
        "debug_scalars": {},
        "points": [[0.1, 0.0, 1.0, 0.0, 0.0]],
    }
    assert _validator().is_valid(plan)
    assert not _validator().is_valid({**plan, **change})


def test_actions_are_written_when_first_seen_and_when_their_state_changes(
    tmp_path,
) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    states = [
        "standbyState",
        "standbyState",
        "startTransition",
        "runningState",
        "runningState",
        "endTransition",
        "completeState",
    ]
    for tick, state in enumerate(states):
        world.frame = 100 + tick
        recorder.record(world, 0.05 * tick)
        recorder.record_action("post_tick", 0, "cut_in", "LaneChangeAction", state)
        recorder.record_action(
            "pre_tick", 0, "lead_speed", "SetSpeedAction", "runningState"
        )
    recorder.close()

    actions = [r for r in _lines(recorder.path) if r["type"] == "action"]
    assert [(a["frame"], a["phase"], a["state"]) for a in actions] == [
        (100, "post_tick", "standbyState"),
        (100, "pre_tick", "runningState"),
        (102, "post_tick", "startTransition"),
        (103, "post_tick", "runningState"),
        (105, "post_tick", "endTransition"),
        (106, "post_tick", "completeState"),
    ]
    assert actions[2] == {
        "type": "action",
        "frame": 102,
        "t": 0.1,
        "phase": "post_tick",
        "index": 0,
        "label": "cut_in",
        "action_type": "LaneChangeAction",
        "state": "startTransition",
    }


def test_an_adversity_nade_sampled_is_written_on_its_tick(tmp_path) -> None:
    from autoware_carla_scenario.adversity_event import AdversityEvent

    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    world.frame = 140
    recorder.record(world, 2.0)
    event = AdversityEvent(
        "npc1", "brake_check", "any", 0.01, 0.2, 0.05, True, ["npc2"], "avoid"
    )
    recorder.record_adversity("nade", event)
    recorder.close()

    (written,) = [r for r in _lines(recorder.path) if r["type"] == "adversity"]
    assert written == {
        "type": "adversity",
        "frame": 140,
        "t": 2.0,
        "label": "nade",
        "agent": "npc1",
        "kind": "brake_check",
        "location": "any",
        "probability": 0.01,
        "is_probability": 0.2,
        "weight": 0.05,
        "avoidable": True,
        "victims": ["npc2"],
        "response": "avoid",
    }


def test_a_spawn_and_a_despawn_are_written_on_their_tick(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    world.frame = 140
    recorder.record(world, 2.0)
    recorder.record_spawn("npc1_enters", "npc1", 7)
    recorder.record_spawn("npc2_enters", "npc2", -1)
    recorder.record_despawn("npc1_leaves", "npc1", 7)
    recorder.close()

    written = [r for r in _lines(recorder.path) if r["type"] in ("spawn", "despawn")]
    assert written == [
        {
            "type": "spawn",
            "frame": 140,
            "t": 2.0,
            "label": "npc1_enters",
            "entity": "npc1",
            "id": 7,
        },
        {
            "type": "spawn",
            "frame": 140,
            "t": 2.0,
            "label": "npc2_enters",
            "entity": "npc2",
            "id": None,
        },
        {
            "type": "despawn",
            "frame": 140,
            "t": 2.0,
            "label": "npc1_leaves",
            "entity": "npc1",
            "id": 7,
            "reason": "action",
        },
    ]


def test_a_spawn_is_not_written_when_not_recording(tmp_path) -> None:
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.record_spawn("npc1_enters", "npc1", 7)  # never started: nothing
    assert not recorder.path.exists()


def test_conditions_are_written_when_first_checked_and_when_they_change(
    tmp_path,
) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    for tick, (arrived, collided) in enumerate(
        [(False, False), (False, False), (True, False), (True, False)]
    ):
        world.frame = 100 + tick
        recorder.record(world, 0.05 * tick)
        recorder.record_condition("pass", 0, "arrived", arrived)
        recorder.record_condition("fail", 0, "collision", collided)
    recorder.close()

    conditions = [r for r in _lines(recorder.path) if r["type"] == "condition"]
    assert [(c["frame"], c["role"], c["satisfied"]) for c in conditions] == [
        (100, "pass", False),
        (100, "fail", False),
        (102, "pass", True),
    ]
    assert conditions[2] == {
        "type": "condition",
        "frame": 102,
        "t": 0.1,
        "role": "pass",
        "index": 0,
        "label": "arrived",
        "satisfied": True,
    }


def test_the_verdict_is_the_last_line(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.record_result(
        ScenarioResult(
            passed=False,
            message="Timeout after 10.00s",
            elapsed_seconds=10.0000001,
            condition_statuses=[
                ConditionStatus(
                    label="fail[0](timeout)",
                    satisfied=True,
                    message="Timeout",
                    condition_type="TimeoutCondition",
                    role="fail",
                    details={"timeout_seconds": 10.0},
                )
            ],
        )
    )
    recorder.close()
    result = _lines(recorder.path)[-1]
    assert result == {
        "type": "result",
        "passed": False,
        "message": "Timeout after 10.00s",
        "elapsed_seconds": 10.0,
        "conditions": [
            {
                "label": "fail[0](timeout)",
                "role": "fail",
                "condition_type": "TimeoutCondition",
                "satisfied": True,
                "message": "Timeout",
            }
        ],
    }


def test_the_new_records_never_fail_the_run(tmp_path) -> None:
    recorder, world = _run(tmp_path)
    recorder._out = _FailingFile()  # type: ignore[assignment]
    recorder.record_markers(_plan(0.0))  # must not raise
    assert not recorder.active
    recorder.record_condition("pass", 0, "x", True)
    recorder.record_result(MagicMock())
    recorder.record_warmup(1, 0, 1)
    recorder.record_spawn("s", "npc1", 1)
    recorder.record_despawn("d", "npc1", 1)


def test_a_map_file_that_cannot_be_read_leaves_its_field_null(tmp_path) -> None:
    # A Lanelet2 path that is gone by the time the run starts, an OpenDRIVE
    # that is a directory: the log goes on, without their digests.
    recorder = TrajectoryRecorder(
        tmp_path / "s.trajectory.jsonl",
        None,
        tmp_path / "gone" / "lanelet2_map.osm",
        tmp_path,
        None,
    )
    world = _World()
    recorder.start(world)
    assert recorder.active
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.close()
    records = _lines(recorder.path)
    [record] = [r for r in records if r["type"] == "map"]
    assert record["lanelet2"] is None and record["opendrive"] is None
    assert records[-1]["type"] == "tick"


def test_a_projector_file_json_cannot_hold_is_left_out(tmp_path) -> None:
    osm = tmp_path / "lanelet2_map.osm"
    osm.write_bytes(b"<osm/>")
    # YAML reads a !!set as a Python set, which JSON has no type for.
    (tmp_path / "map_projector_info.yaml").write_text(
        "projector_type: MGRS\ngrids: !!set {54SUE, 54SVE}\n"
    )
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl", None, osm)
    recorder.start(_World())
    assert recorder.active
    recorder.close()
    [record] = [r for r in _lines(recorder.path) if r["type"] == "map"]
    assert record["projector_info"] is None
    assert record["lanelet2"]["bytes"] == 6


def test_an_unparsable_projector_file_is_left_out(tmp_path) -> None:
    osm = tmp_path / "lanelet2_map.osm"
    osm.write_bytes(b"<osm/>")
    (tmp_path / "map_projector_info.yaml").write_text("projector_type: [MGRS\n")
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl", None, osm)
    recorder.start(_World())
    recorder.close()
    [record] = [r for r in _lines(recorder.path) if r["type"] == "map"]
    assert record["projector_info"] is None and record["lanelet2"] is not None


def test_turn_indicators_are_written_when_they_change(tmp_path: Path) -> None:
    recorder, world = _run(tmp_path)
    left = int(carla.VehicleLightState.LeftBlinker)
    right = int(carla.VehicleLightState.RightBlinker)
    lights = {1: 0}
    world.vehicle_light_states = lights
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0), _ActorSnapshot(3, 5.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1  # unchanged: nothing written
    recorder.record(world, 0.05)
    world.frame += 1
    lights[1] = left | int(carla.VehicleLightState.Position)  # other lights don't count
    recorder.record(world, 0.1)
    world.frame += 1
    lights[1] = left | right
    recorder.record(world, 0.15)
    recorder.close()

    records = [r for r in _lines(recorder.path) if r["type"] == "turn_indicators"]
    # The walker (3) has none; the vehicle's first state is written, then each change.
    assert records == [
        {"type": "turn_indicators", "frame": 100, "t": 0.0, "states": {"1": "off"}},
        {"type": "turn_indicators", "frame": 102, "t": 0.1, "states": {"1": "left"}},
        {"type": "turn_indicators", "frame": 103, "t": 0.15, "states": {"1": "hazard"}},
    ]
    kinds = [r["type"] for r in _lines(recorder.path)]
    # Each right after the tick it belongs to.
    assert kinds[kinds.index("turn_indicators") - 1] == "tick"


def test_a_world_without_light_states_leaves_the_turn_indicators_out(
    tmp_path: Path,
) -> None:
    recorder, world = _run(tmp_path)

    world.vehicle_light_states = None
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1
    recorder.record(world, 0.05)
    recorder.close()
    kinds = [r["type"] for r in _lines(recorder.path)]
    assert "turn_indicators" not in kinds and kinds.count("tick") == 2


def _velocities(path: Path, actor_id: int) -> list[list[float]]:
    """*actor_id*'s ``[vx, vy, vz]`` on each tick written."""
    return [
        row[7:10]
        for r in _lines(path)
        if r["type"] == "tick"
        for row in r["actors"]
        if row[0] == actor_id
    ]


def _walk(
    recorder: TrajectoryRecorder,
    world: _World,
    steps: list[tuple[float, float]],
    vehicle_x: float = 0.0,
) -> None:
    """Record a tick per step (20 Hz): the walker (3) at ``(x, z)``, reporting
    0 as one on CARLA's walker controller does, and the ego (1) reporting 7
    m/s while it moves 10."""
    for n, (x, z) in enumerate(steps):
        world.snapshot = [
            _ActorSnapshot(1, vehicle_x + 0.5 * n, 7.0),
            _ActorSnapshot(3, x, 0.0, z=z),
        ]
        recorder.record(world, 0.05 * (world.frame - 100))
        world.frame += 1


def test_a_walker_on_its_controller_is_recorded_walking(tmp_path: Path) -> None:
    """CARLA's walker controller places its walker, which reports 0 while it
    walks: its velocity is measured from where it stood the tick before
    (#924); a vehicle's is the one CARLA reports."""
    recorder, world = _run(tmp_path)
    _walk(recorder, world, [(0.0, 0.0), (0.07, 0.0), (0.14, 0.0)])
    recorder.close()
    walker = _velocities(recorder.path, 3)
    assert walker[0] == [0.0, 0.0, 0.0]  # nothing to measure from yet
    for vx, vy, vz in walker[1:]:
        assert vx == pytest.approx(1.4, abs=1e-3) and vy == vz == 0.0
    assert _velocities(recorder.path, 1) == [[7.0, 0.0, 0.0]] * 3


def test_a_walker_brought_in_from_under_the_map_is_no_spike(
    tmp_path: Path,
) -> None:
    """A walker parked under the map and brought in has not walked there: the
    tick it appears reads what CARLA reports, the next its walk."""
    recorder, world = _run(tmp_path)
    _walk(recorder, world, [(0.0, -500.0), (0.0, -500.0), (5.0, 0.0), (5.07, 0.0)])
    recorder.close()
    walker = _velocities(recorder.path, 3)
    assert walker[:3] == [[0.0, 0.0, 0.0]] * 3
    assert walker[3][0] == pytest.approx(1.4, abs=1e-3)


def test_a_rerun_measures_its_walkers_afresh(tmp_path: Path) -> None:
    """Started again, the recorder has forgotten where the walkers stood: a
    walker first read on the frame it was last read on in the run before is
    not given that run's velocity."""
    recorder, world = _run(tmp_path)
    _walk(recorder, world, [(0.0, 0.0), (0.07, 0.0)])
    recorder.close()
    world.frame = 101
    recorder.start(world)
    _walk(recorder, world, [(3.0, 0.0)])
    recorder.close()
    assert _velocities(recorder.path, 3) == [[0.0, 0.0, 0.0]]
