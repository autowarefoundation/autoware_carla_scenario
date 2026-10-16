"""Unit tests for TrajectoryRecorder (no CARLA server)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import jsonschema
import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.trajectory_recorder import SCHEMA_PATH, TrajectoryRecorder


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
    def __init__(self, actor_id: int, x: float, vx: float) -> None:
        self.id = actor_id
        self._x, self._vx = x, vx

    def get_transform(self) -> SimpleNamespace:
        return SimpleNamespace(location=_vec(self._x, 1.0, 0.0), rotation=_rot(90.0))

    def get_velocity(self) -> SimpleNamespace:
        return _vec(self._vx)


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

    def __iter__(self):
        return iter(self._actors)


@cache
def _validator() -> jsonschema.Draft7Validator:
    return jsonschema.Draft7Validator(json.loads(SCHEMA_PATH.read_text()))


def _lines(path: Path) -> list[dict]:
    """The file's records, each checked against the published schema."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        _validator().validate(record)
    return records


def test_the_schema_is_a_valid_draft_7_schema() -> None:
    jsonschema.Draft7Validator.check_schema(_validator().schema)


def test_it_writes_the_world_actors_and_each_tick(tmp_path: Path) -> None:
    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "out" / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(1, 0.0, 0.0), _ActorSnapshot(2, 0.0, 0.0)]
    recorder.record(world, 0.0)
    world.frame += 1
    world.snapshot = [_ActorSnapshot(1, 0.5, 10.0), _ActorSnapshot(2, 0.5, 10.0)]
    world.light.state = int(carla.TrafficLightState.Green)
    recorder.record(world, 0.05)
    recorder.close()

    records = _lines(recorder.path)
    assert [r["type"] for r in records] == [
        "world",
        "traffic_light",
        "actor",
        "tick",
        "tick",
    ]
    assert records[0] == {
        "type": "world",
        "map": "Carla/Maps/Town10HD_Opt",
        "episode": 7,
    }
    assert records[1]["opendrive_id"] == "42"
    assert records[1]["boxes"] == [[10.0, 20.0, 5.0, 0.2, 0.3, 0.6, 180.0]]
    assert records[1]["stops"] == [[5.0, 18.0, 0.0, -90.0]]
    # Only vehicles/walkers are recorded; the camera is looked up once and skipped.
    assert records[2]["id"] == 1 and records[2]["role"] == "Ego"
    first, second = records[3], records[4]
    assert first["actors"] == [[1, 0.0, 1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0]]
    assert first["control"] == {"1": [0.5, 0.0, 0.0, 1]}
    assert first["lights"] == {"9": "Red"}
    assert second["t"] == 0.05 and second["frame"] == 101
    assert second["actors"][0][7] == 10.0
    # Only the light states that changed since the last written tick.
    assert second["lights"] == {"9": "Green"}


class _UnnamedLight(_Light):
    def get_opendrive_id(self) -> str:
        raise RuntimeError("no signal")


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
    assert records[1]["opendrive_id"] == ""
    assert [r["type_id"] for r in records if r["type"] == "actor"] == [
        "walker.pedestrian.0001",
        "vehicle.tesla.model3",
    ]
    first, second = (r for r in records if r["type"] == "tick")
    assert first["control"] == {} and first["lights"] == {"9": "Red"}
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
        {**_TICK, "frame": 2**53},  # an integer a double cannot hold exactly
        {**_TICK, "frame": -1},  # LibCarla's frame is unsigned
    ],
)
def test_the_schema_rejects_lines_the_recorder_does_not_write(record: dict) -> None:
    assert not _validator().is_valid(record)


def test_ticks_without_vehicles_are_not_written(tmp_path: Path) -> None:
    world = _World()
    recorder = TrajectoryRecorder(tmp_path / "s.trajectory.jsonl")
    recorder.start(world)
    world.snapshot = [_ActorSnapshot(2, 0.0, 0.0)]
    recorder.record(world, 0.0)
    recorder.close()

    assert [r["type"] for r in _lines(recorder.path)] == ["world", "traffic_light"]


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
    def write(self, text: str) -> int:
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
