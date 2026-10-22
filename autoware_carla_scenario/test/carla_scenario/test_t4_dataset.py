"""T4 datasets of runs (``t4.enabled``): the settings, the recorder on a fake
world, and the dataset it writes, read back with t4-devkit -- TIER IV's own
reader, at its latest release (v0.8.0) -- and passed through its sanity
checker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario import t4
from autoware_carla_scenario.driver.observation import rear_axle_offset
from autoware_carla_scenario.sensor import carla_lidar
from autoware_carla_scenario.sensor.carla_camera import CarlaCameraSensorConfig
from autoware_carla_scenario.t4 import (
    T4CameraConfig,
    T4Recorder,
    T4Settings,
    T4UnavailableError,
    category_of,
    dataset_path,
)

writer_module = pytest.importorskip("autoware_carla_scenario_t4")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_the_defaults_are_a_top_lidar_and_a_front_camera_at_10_hz() -> None:
    settings = T4Settings()
    assert [c.channel for c in settings.lidars] == ["LIDAR_TOP"]
    assert [c.channel for c in settings.cameras] == ["CAM_FRONT"]
    assert settings.interval_ticks == 2


def test_settings_read_a_config_section() -> None:
    settings = T4Settings.from_config(
        {
            "enabled": True,
            "frequency_hz": 5,
            "max_distance_m": None,
            "lidars": [{"channel": "LIDAR_CONCAT", "channels": 32, "position_z": 1.9}],
            "cameras": [],
        }
    )
    assert settings.interval_ticks == 4
    assert settings.max_distance_m is None
    assert settings.lidars[0].channel == "LIDAR_CONCAT"
    assert settings.lidars[0].sensor.channels == 32
    assert settings.cameras == ()


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"frequency_hz": 3.0}, "whole ticks"),
        ({"frequency_hz": 0.0}, "positive"),
        ({"lidars": [], "cameras": []}, "at least one"),
        ({"cameras": [{"channel": "LIDAR_TOP"}]}, "used twice"),
        ({"lidars": [{"channel": "L", "beams": 64}]}, "no setting beams"),
        ({"jpeg_quality": 0}, "1-100"),
        ({"max_distance_m": -1.0}, "positive"),
    ],
)
def test_settings_refuse_what_cannot_be_recorded(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        T4Settings.from_config(config)


@pytest.mark.parametrize(
    ("type_id", "base_type", "wheels", "category"),
    [
        ("walker.pedestrian.0001", "", 0, "pedestrian"),
        ("vehicle.tesla.model3", "car", 4, "car"),
        ("vehicle.mercedes.sprinter", "Van", 4, "car"),
        ("vehicle.carlamotors.firetruck", "truck", 4, "truck"),
        ("vehicle.mitsubishi.fusorosa", "bus", 4, "bus"),
        ("vehicle.diamondback.century", "bicycle", 2, "bicycle"),
        ("vehicle.yamaha.yzf", "", 2, "motorcycle"),
        ("vehicle.unknown", "", 4, "car"),
    ],
)
def test_blueprints_map_to_t4_categories(
    type_id: str, base_type: str, wheels: int, category: str
) -> None:
    assert category_of(type_id, base_type, wheels) == category


def test_a_run_refuses_t4_without_the_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(t4.importlib, "import_module", missing)
    assert run.build_t4_recorder(OmegaConf.create({"t4": {"enabled": False}})) is None
    with pytest.raises(T4UnavailableError, match=r"autoware-carla-scenario\[t4\]"):
        run.build_t4_recorder(OmegaConf.create({"t4": {"enabled": True}}))


# ---------------------------------------------------------------------------
# A fake world
# ---------------------------------------------------------------------------


def _xyz(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _transform(x: float, y: float, z: float, yaw: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        location=_xyz(x, y, z), rotation=SimpleNamespace(roll=0.0, pitch=0.0, yaw=yaw)
    )


class _Actor:
    def __init__(
        self,
        actor_id: int,
        type_id: str,
        extent: tuple[float, float, float],
        attributes: Optional[dict[str, str]] = None,
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = attributes or {}
        self.bounding_box = SimpleNamespace(
            location=_xyz(0.0, 0.0, extent[2]), extent=_xyz(*extent)
        )

    def get_physics_control(self) -> Any:
        raise RuntimeError("no physics in a fake")

    def get_control(self) -> SimpleNamespace:
        return SimpleNamespace(throttle=0.5, brake=0.0, steer=0.1, reverse=False)


class _ActorSnapshot:
    def __init__(self, actor_id: int, transform: Any, velocity: Any) -> None:
        self.id = actor_id
        self._transform = transform
        self._velocity = velocity

    def get_transform(self) -> Any:
        return self._transform

    def get_velocity(self) -> Any:
        return self._velocity

    def get_angular_velocity(self) -> Any:
        return _xyz()

    def get_acceleration(self) -> Any:
        return _xyz()


class _Snapshot(list):
    def __init__(self, frame: int, actors: list[_ActorSnapshot]) -> None:
        super().__init__(actors)
        self.frame = frame

    def find(self, actor_id: int) -> Optional[_ActorSnapshot]:
        return next((a for a in self if a.id == actor_id), None)


class _Sensor:
    def __init__(self, blueprint: str) -> None:
        self.blueprint = blueprint
        self.listener: Any = None
        self.destroyed = False

    def listen(self, callback: Any) -> None:
        self.listener = callback

    def stop(self) -> None:
        pass

    def destroy(self) -> None:
        self.destroyed = True


class _Blueprint:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value

    def has_attribute(self, name: str) -> bool:
        return True


class _World:
    """The ego driving east at 10 m/s, a car keeping 10 m ahead in the next
    lane to its right, and a pedestrian far away."""

    def __init__(self) -> None:
        self.frame = 100
        self.ego = _Actor(
            1, "vehicle.lincoln.mkz", (2.4, 1.0, 0.75), {"role_name": "ego"}
        )
        self.actors = {
            1: self.ego,
            2: _Actor(2, "vehicle.tesla.model3", (2.0, 0.9, 0.7), {"base_type": "car"}),
            3: _Actor(3, "walker.pedestrian.0001", (0.3, 0.3, 0.9)),
            9: _Actor(9, "traffic.traffic_light", (0.5, 0.5, 2.0)),
        }
        self.sensors: list[_Sensor] = []

    def ego_x(self, frame: int) -> float:
        return 0.5 * (frame - 100)

    def get_snapshot(self) -> _Snapshot:
        x = self.ego_x(self.frame)
        return _Snapshot(
            self.frame,
            [
                _ActorSnapshot(1, _transform(x, 0.0, 0.0), _xyz(10.0)),
                _ActorSnapshot(2, _transform(x + 10.0, 3.5, 0.0), _xyz(10.0)),
                _ActorSnapshot(3, _transform(500.0, 0.0, 0.0), _xyz()),
                _ActorSnapshot(9, _transform(0.0, 0.0, 0.0), _xyz()),
            ],
        )

    def get_actors(self, ids: list[int]) -> list[_Actor]:
        return [self.actors[i] for i in ids if i in self.actors]

    def get_blueprint_library(self) -> SimpleNamespace:
        return SimpleNamespace(find=_Blueprint)

    def spawn_actor(
        self, blueprint: _Blueprint, transform: Any, attach_to: Any
    ) -> _Sensor:
        sensor = _Sensor(blueprint.name)
        self.sensors.append(sensor)
        return sensor

    def get_weather(self) -> SimpleNamespace:
        return SimpleNamespace(precipitation=0.0, fog_density=0.0, fog_distance=0.0)

    def sense(self) -> None:
        """What the sensors deliver on this frame."""
        for sensor in self.sensors:
            if sensor.blueprint == "sensor.lidar.ray_cast":
                # In the LiDAR's (left-handed) frame, 2.2 m up on the ego: two
                # points on the car ahead, one on nothing.
                points = np.array(
                    [
                        [10.0, 3.5, -1.5, 0.9],
                        [10.5, 3.3, -1.2, 0.8],
                        [30.0, 0.0, 0.0, 0.1],
                    ],
                    dtype=np.float32,
                )
                sensor.listener(
                    SimpleNamespace(frame=self.frame, raw_data=points.tobytes())
                )
            else:
                pixels = np.full((48, 64, 4), 200, dtype=np.uint8)
                sensor.listener(
                    SimpleNamespace(
                        frame=self.frame, width=64, height=48, raw_data=pixels.tobytes()
                    )
                )


OFFSETS = (1000.0, 2000.0, 30.0)


def _record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, _World]:
    """Ten ticks of the fake world at 10 Hz (a sweep every other tick)."""
    monkeypatch.setattr(t4, "_map_offsets", lambda: OFFSETS)
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lanelet2 = tmp_path / "map" / "lanelet2_map.osm"
    lanelet2.parent.mkdir()
    lanelet2.write_text("<osm/>")
    world = _World()
    settings = T4Settings(
        cameras=(
            T4CameraConfig(
                sensor=CarlaCameraSensorConfig(
                    image_width=64,
                    image_height=48,
                    fov=90.0,
                    position_x=1.0,
                    position_z=1.6,
                )
            ),
        )
    )
    recording = T4Recorder(settings).start(
        tmp_path / "out",
        "CutInScenario",
        world,  # type: ignore[arg-type]
        world.ego,  # type: ignore[arg-type]
        100,
        "CutInScenario",
        "nishishinjuku",
        lanelet2,
    )
    assert recording is not None
    for frame in range(101, 111):
        world.frame = frame
        if frame % 2 == 1:  # the sensors' ticks, from the one after they were spawned
            world.sense()
        recording.record(world)  # type: ignore[arg-type]
    assert recording.active
    assert recording.close() == dataset_path(tmp_path / "out", "CutInScenario")
    assert all(sensor.destroyed for sensor in world.sensors)
    return dataset_path(tmp_path / "out", "CutInScenario"), world


def _table(root: Path, name: str) -> list[dict[str, Any]]:
    return json.loads((root / "0" / "annotation" / f"{name}.json").read_text())


def test_a_key_frame_is_written_for_every_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _record(tmp_path, monkeypatch)
    samples = _table(root, "sample")
    assert len(samples) == 5
    stamps = [s["timestamp"] for s in samples]
    assert [b - a for a, b in zip(stamps, stamps[1:])] == [100_000] * 4
    channels = sorted(
        {sd["filename"].split("/")[1] for sd in _table(root, "sample_data")}
    )
    assert channels == ["CAM_FRONT", "LIDAR_TOP"]
    assert (root / "0" / "map" / "lanelet2_map.osm").read_text() == "<osm/>"


def test_the_ego_pose_is_its_rear_axle_in_the_map_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, world = _record(tmp_path, monkeypatch)
    first = _table(root, "ego_pose")[0]
    rear = rear_axle_offset(world.ego)  # type: ignore[arg-type]
    np.testing.assert_allclose(
        first["translation"],
        [world.ego_x(101) + rear + OFFSETS[0], OFFSETS[1], OFFSETS[2]],
        atol=1e-9,
    )
    np.testing.assert_allclose(first["rotation"], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(first["twist"], [10.0, 0, 0, 0, 0, 0], atol=1e-9)


def test_the_boxes_are_the_road_users_near_the_ego(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, world = _record(tmp_path, monkeypatch)
    annotations = _table(root, "sample_annotation")
    # The car in every frame; the pedestrian 500 m away and the light never.
    assert len(annotations) == 5
    (instance,) = _table(root, "instance")
    assert instance["nbr_annotations"] == 5
    (category,) = _table(root, "category")
    assert category["name"] == "car"
    first = annotations[0]
    np.testing.assert_allclose(
        first["translation"],
        [world.ego_x(101) + 10.0 + OFFSETS[0], -3.5 + OFFSETS[1], 0.7 + OFFSETS[2]],
        atol=1e-9,
    )
    assert first["size"] == [1.8, 4.0, 1.4]  # width, length, height
    assert first["num_lidar_pts"] == 2


def test_t4_devkit_loads_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t4_devkit = pytest.importorskip("t4_devkit")
    from t4_devkit.dataclass import LidarPointCloud  # noqa: PLC0415

    root, _ = _record(tmp_path, monkeypatch)
    dataset = t4_devkit.T4Devkit(str(root))
    assert dataset.version == "0"
    assert len(dataset.sample) == 5

    sample = dataset.sample[0]
    path, boxes, intrinsic = dataset.get_sample_data(sample.data["LIDAR_TOP"])
    assert intrinsic is None
    (box,) = boxes
    assert box.semantic_label.name == "car"
    assert box.num_points == 2
    # In the LiDAR's frame: 10 m ahead, 3.5 m to the right, 1.5 m below it.
    np.testing.assert_allclose(box.position, [10.0, -3.5, -1.5], atol=1e-6)
    points = LidarPointCloud.from_file(path).points
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[:3, 0], [10.0, -3.5, -1.5], atol=1e-6)

    path, boxes, intrinsic = dataset.get_sample_data(sample.data["CAM_FRONT"])
    np.testing.assert_allclose(
        intrinsic, [[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]]
    )
    assert Path(path).read_bytes()[:2] == b"\xff\xd8"
    # The car is ahead and to the right: in the optical frame, +z and +x.
    (box,) = boxes
    assert box.position[2] > 0.0 and box.position[0] > 0.0


def test_the_dataset_passes_t4_devkits_sanity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("t4_devkit")
    from t4_devkit.sanity import sanity_check  # noqa: PLC0415

    root, _ = _record(tmp_path, monkeypatch)
    result = sanity_check(str(root))
    assert result.is_passed()
    reports = result.reports
    assert reports
    errors = [r for r in reports if r.severity.is_error() and r.status.name == "FAILED"]
    assert errors == []
    warnings = {r.id for r in reports if r.status.name == "FAILED"}
    # A simulated run has no ROS bag and no point cloud map.
    assert warnings <= {"STR005", "STR009"}


def test_a_recording_that_cannot_start_is_none(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (dataset_path(out, "Run") / "0" / "annotation").mkdir(parents=True)
    world = _World()
    recording = T4Recorder().start(out, "Run", world, world.ego, 100)  # type: ignore[arg-type]
    assert recording is None
    assert world.sensors == []
