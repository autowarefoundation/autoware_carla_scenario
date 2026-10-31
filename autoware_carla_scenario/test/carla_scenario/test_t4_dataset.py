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
from autoware_carla_scenario.driver.base import (
    DriverCameraConfig,
    DriverClientConfig,
    DriverLidarConfig,
)
from autoware_carla_scenario.driver.observation import rear_axle_offset
from autoware_carla_scenario.sensor import carla_lidar
from autoware_carla_scenario.t4 import (
    LIDAR_CHANNEL,
    T4Recorder,
    T4Recording,
    T4Settings,
    T4UnavailableError,
    camera_channel,
    category_of,
    dataset_path,
)

writer_module = pytest.importorskip("autoware_carla_scenario_t4")


@pytest.fixture(autouse=True)
def _no_driver_sensors_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The offer of a driver entity's sensors is process-global: no test starts
    with one left by an earlier test in the same worker."""
    monkeypatch.setattr(t4, "_driver_rear_axle", None)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _driver(**overrides: Any) -> DriverClientConfig:
    """A driver policy's config: a roof LiDAR and a small front camera."""
    values: dict[str, Any] = {
        "lidars": (DriverLidarConfig(logical_id="lidar_top", position_z=2.2),),
        "cameras": (
            DriverCameraConfig(
                logical_id="cam_front",
                image_width=64,
                image_height=48,
                fov=90.0,
                position_x=1.0,
                position_z=1.6,
            ),
        ),
        "policy_timestep_s": 0.1,
        **overrides,
    }
    return DriverClientConfig(**values)


def test_settings_take_the_driver_policys_sensors_and_step() -> None:
    settings = T4Settings.from_driver(
        _driver(policy_timestep_s=0.2, image_quality=70),
        {"enabled": True, "max_distance_m": None, "workers": 2},
    )
    assert [c.position_z for c in settings.lidars] == [2.2]
    assert [logical_id for logical_id, _ in settings.cameras] == ["cam_front"]
    assert settings.interval_ticks == 4
    assert settings.jpeg_quality == 70
    assert settings.max_distance_m is None
    assert settings.workers == 2
    assert camera_channel("cam_front") == "CAM_FRONT"


@pytest.mark.parametrize(
    ("driver", "config", "match"),
    [
        ({"lidars": (), "cameras": ()}, {}, "needs a sensor"),
        ({}, {"max_distance_m": -1.0}, "positive"),
        ({}, {"workers": 0}, "positive"),
    ],
)
def test_settings_refuse_what_cannot_be_recorded(
    driver: dict, config: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        T4Settings.from_driver(_driver(**driver), config)


def test_a_camera_lens_comes_from_the_driver_config() -> None:
    from ._ftheta_lens import COSMOS3_FRONT_WIDE

    config = DriverClientConfig.from_mapping(
        {
            "cameras": [
                {
                    "logical_id": "camera_front_wide_120fov",
                    "image_width": 832,
                    "image_height": 480,
                    "lens": COSMOS3_FRONT_WIDE,
                },
                {"logical_id": "cam_pinhole"},
            ]
        }
    )
    settings = T4Settings.from_driver(config, {})
    ((logical_id, lens),) = settings.lenses
    assert logical_id == "camera_front_wide_120fov"
    assert lens.max_angle == pytest.approx(1.226783919)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"model": "kannala_brandt"}, "lens.model"),
        ({"principal_point": [1.0]}, "principal_point"),
        ({"angle_to_pixeldist_poly": []}, "both"),
        ({"max_angle": 0.0}, "max_angle"),
        ({"reference_poly": "fw"}, "reference_poly"),
    ],
)
def test_a_lens_out_of_its_domain_is_refused(change: dict, message: str) -> None:
    from autoware_carla_scenario.driver.base import DriverCameraLens

    from ._ftheta_lens import COSMOS3_FRONT_WIDE

    with pytest.raises(ValueError, match=message):
        DriverCameraLens.from_mapping({**COSMOS3_FRONT_WIDE, **change})


def _opencv_project(intrinsic: Any, distortion: Any, rays: np.ndarray) -> np.ndarray:
    """OpenCV's projection (cv2.projectPoints) of optical-frame rays, written out."""
    k1, k2, p1, p2, k3, k4, k5, k6 = distortion
    x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    r2 = x * x + y * y
    f = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (
        1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    )
    xd, yd = x * f, y * f
    (fx, s, cx), (_, fy, cy), _ = intrinsic
    return np.stack([fx * xd + s * yd + cx, fy * yd + cy], 1)


def test_an_ftheta_lens_is_written_as_opencvs_rational_model() -> None:
    """T4's cameras are OpenCV's: the fitted model lands a ray where the lens
    does, to a fraction of a pixel."""
    from autoware_carla_scenario.driver.base import DriverCameraLens

    from ._ftheta_lens import COSMOS3_FRONT_WIDE

    lens = DriverCameraLens.from_mapping(COSMOS3_FRONT_WIDE)
    intrinsic, distortion, mean_px, max_px = t4.opencv_of_ftheta(lens, 832, 480)
    assert len(distortion) == 8 and distortion[2:4] == [0.0, 0.0]
    (fx, skew, cx), (_, fy, cy), _ = intrinsic
    assert (cx, cy) == pytest.approx(tuple(COSMOS3_FRONT_WIDE["principal_point"]))
    assert fy == pytest.approx(411.286296) and fx == pytest.approx(
        411.286296 * 1.00234926
    )
    assert skew == 0.0
    assert mean_px < 0.2 and max_px < 3.0

    # Rays the lens sees, projected by the lens and by the fitted model.
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.0, np.radians(60.0), 2000)
    phi = rng.uniform(-np.pi, np.pi, 2000)
    rays = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1
    )
    radius = np.polynomial.polynomial.polyval(theta, lens.angle_to_pixeldist_poly)
    c = lens.linear_cde[0]
    lens_px = np.stack([c * radius * np.cos(phi) + cx, radius * np.sin(phi) + cy], 1)
    error = np.hypot(*(_opencv_project(intrinsic, distortion, rays) - lens_px).T)
    assert np.median(error) < 0.2 and error.max() < 3.0


def test_a_skewed_lens_has_no_opencv_model() -> None:
    from autoware_carla_scenario.driver.base import DriverCameraLens

    from ._ftheta_lens import COSMOS3_FRONT_WIDE

    lens = DriverCameraLens.from_mapping(
        {**COSMOS3_FRONT_WIDE, "linear_cde": [1.0, 0.0, 0.1]}
    )
    with pytest.raises(ValueError, match="no OpenCV model"):
        t4.opencv_of_ftheta(lens, 832, 480)


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
    enabled = OmegaConf.create({"t4": {"enabled": True}, "driver": {}})
    with pytest.raises(T4UnavailableError, match=r"autoware-carla-scenario\[t4\]"):
        run.build_t4_recorder(enabled)


def test_a_run_builds_the_recorder_from_its_driver_config() -> None:
    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    cfg = OmegaConf.create(
        {
            "t4": {"enabled": True, "max_distance_m": 80.0},
            "driver": {
                "policy_timestep_s": 0.05,
                "lidars": [{"logical_id": "lidar_top", "position_z": 1.9}],
            },
        }
    )
    recorder = run.build_t4_recorder(cfg)
    assert recorder is not None
    assert recorder.settings.interval_ticks == 1
    assert recorder.settings.max_distance_m == 80.0
    assert [c.position_z for c in recorder.settings.lidars] == [1.9]
    with pytest.raises(ValueError, match="'driver' config group"):
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


def _start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    world: "_World",
    driver: Optional[DriverClientConfig] = None,
    section: Optional[dict[str, Any]] = None,
) -> T4Recording:
    monkeypatch.setattr(t4, "_map_offset", lambda: OFFSETS)
    monkeypatch.setattr(carla_lidar, "_SWEEP_TIMEOUT", 0.05)
    lanelet2 = tmp_path / "map" / "lanelet2_map.osm"
    lanelet2.parent.mkdir()
    lanelet2.write_text("<osm/>")
    recording = T4Recorder(
        T4Settings.from_driver(driver or _driver(), section or {})
    ).start(
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
    return recording


def _record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    driver: Optional[DriverClientConfig] = None,
    section: Optional[dict[str, Any]] = None,
    world: Optional[_World] = None,
) -> tuple[Path, _World]:
    """Ten ticks of the fake world at 10 Hz (a sweep every other tick), with
    the recorder's own sensors on the ego."""
    world = world or _World()
    recording = _start(tmp_path, monkeypatch, world, driver, section)
    assert not recording.fed_by_driver
    assert len(world.sensors) == 2
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
    assert channels == ["CAM_FRONT", LIDAR_CHANNEL]
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


class _HidingWorld(_World):
    """The fake world with two more cars parked out of it, 500 m below the
    ground (a hidden spawn, a trajectory entity before its start): one is
    revealed 20 m ahead of the ego from frame 104 to 108 and hidden again
    after, the other stays hidden."""

    def __init__(self) -> None:
        super().__init__()
        self.actors[4] = _Actor(4, "vehicle.audi.tt", (2.0, 0.9, 0.7))
        self.actors[5] = _Actor(5, "vehicle.audi.tt", (2.0, 0.9, 0.7))

    def revealed(self, frame: int) -> bool:
        return 104 <= frame <= 108

    def get_snapshot(self) -> _Snapshot:
        snapshot = super().get_snapshot()
        x = self.ego_x(self.frame)
        z = 0.0 if self.revealed(self.frame) else -500.0
        snapshot.append(_ActorSnapshot(4, _transform(x + 20.0, 0.0, z), _xyz(10.0)))
        snapshot.append(_ActorSnapshot(5, _transform(x + 20.0, 5.0, -500.0), _xyz()))
        return snapshot


def _boxes_by_instance(root: Path) -> dict[str, list[dict[str, Any]]]:
    boxes: dict[str, list[dict[str, Any]]] = {}
    for annotation in _table(root, "sample_annotation"):
        boxes.setdefault(annotation["instance_token"], []).append(annotation)
    return boxes


@pytest.mark.parametrize("section", [{"max_distance_m": None}, {}])
def test_a_hidden_actor_gets_no_box_until_it_is_revealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: dict[str, Any]
) -> None:
    """An actor parked out of the world is not on the scene: no box, even with
    no distance limit (#908); revealed, it is annotated from the key frame it
    appears on, as one instance, until it is hidden again."""
    root, world = _record(tmp_path, monkeypatch, section=section, world=_HidingWorld())
    annotations = _table(root, "sample_annotation")
    # Every box is on the ground (the map's elevation offset, give or take a box).
    assert all(abs(a["translation"][2] - OFFSETS[2]) < 5.0 for a in annotations)
    instances = {i["token"]: i for i in _table(root, "instance")}
    # The car in every frame, the revealed car on two, the pedestrian 500 m
    # away only without a limit; the car that stays hidden never.
    unlimited = section.get("max_distance_m", 100.0) is None
    assert len(instances) == (3 if unlimited else 2)
    assert len(annotations) == (12 if unlimited else 7)
    sample_frame = {
        s["token"]: 101 + 2 * n
        for n, s in enumerate(
            sorted(_table(root, "sample"), key=lambda s: s["timestamp"])
        )
    }
    (revealed,) = [
        boxes for boxes in _boxes_by_instance(root).values() if len(boxes) == 2
    ]
    assert sorted(sample_frame[a["sample_token"]] for a in revealed) == [105, 107]
    first, last = sorted(revealed, key=lambda a: sample_frame[a["sample_token"]])
    np.testing.assert_allclose(
        first["translation"][:2],
        [world.ego_x(105) + 20.0 + OFFSETS[0], OFFSETS[1]],
        atol=1e-9,
    )
    instance = instances[first["instance_token"]]
    assert instance["nbr_annotations"] == 2
    assert (instance["first_annotation_token"], instance["last_annotation_token"]) == (
        first["token"],
        last["token"],
    )
    assert (first["prev"], first["next"]) == ("", last["token"])
    assert (last["prev"], last["next"]) == (first["token"], "")


def test_t4_devkit_loads_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t4_devkit = pytest.importorskip("t4_devkit")
    from t4_devkit.dataclass import LidarPointCloud  # noqa: PLC0415

    root, world = _record(tmp_path, monkeypatch)
    rear = rear_axle_offset(world.ego)  # type: ignore[arg-type]
    dataset = t4_devkit.T4Devkit(str(root))
    assert dataset.version == "0"
    assert len(dataset.sample) == 5

    sample = dataset.sample[0]
    path, boxes, intrinsic = dataset.get_sample_data(sample.data[LIDAR_CHANNEL])
    assert intrinsic is None
    (box,) = boxes
    assert box.semantic_label.name == "car"
    assert box.num_points == 2
    # In the ego's frame (LIDAR_CONCAT's): 10 m ahead of the actor, so
    # 10 - rear ahead of the rear axle, 3.5 m to the right, 0.7 m up.
    np.testing.assert_allclose(box.position, [10.0 - rear, -3.5, 0.7], atol=1e-6)
    points = LidarPointCloud.from_file(path).points
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points[:3, 0], [10.0 - rear, -3.5, 0.7], atol=1e-5)

    path, boxes, intrinsic = dataset.get_sample_data(sample.data["CAM_FRONT"])
    np.testing.assert_allclose(
        intrinsic, [[32.0, 0.0, 32.0], [0.0, 32.0, 24.0], [0.0, 0.0, 1.0]]
    )
    assert Path(path).read_bytes()[:2] == b"\xff\xd8"
    # The car is ahead and to the right: in the optical frame, +z and +x.
    (box,) = boxes
    assert box.position[2] > 0.0 and box.position[0] > 0.0


def test_a_lens_camera_is_calibrated_with_its_opencv_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.driver.base import DriverCameraLens

    from ._ftheta_lens import COSMOS3_FRONT_WIDE

    lens = DriverCameraLens.from_mapping(COSMOS3_FRONT_WIDE)
    camera = DriverCameraConfig(
        logical_id="cam_front", image_width=832, image_height=480, lens=lens
    )
    root, _ = _record(tmp_path, monkeypatch, _driver(cameras=(camera,)))
    intrinsic, distortion, _, _ = t4.opencv_of_ftheta(lens, 832, 480)
    sensors = {s["token"]: s["channel"] for s in _table(root, "sensor")}
    (calibrated,) = [
        c
        for c in _table(root, "calibrated_sensor")
        if sensors[c["sensor_token"]] == "CAM_FRONT"
    ]
    np.testing.assert_allclose(calibrated["camera_intrinsic"], intrinsic)
    np.testing.assert_allclose(calibrated["camera_distortion"], distortion)


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
    recorder = T4Recorder(T4Settings.from_driver(_driver(), {}))
    recording = recorder.start(out, "Run", world, world.ego, 100)  # type: ignore[arg-type]
    assert recording is None
    assert world.sensors == []


def test_a_driver_policys_ego_feeds_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a driver policy's ego, the dataset puts no sensor on it: it takes
    the sweeps (rig frame) and the JPEG frames the policy is sent."""
    world = _World()
    t4.offer_driver_sensors(-1.25)
    try:
        recording = _start(tmp_path, monkeypatch, world)
    finally:
        t4.withdraw_driver_sensors()
    assert recording.fed_by_driver
    assert t4.active_recording() is recording
    assert world.sensors == []
    jpeg = _jpeg()
    for frame in range(101, 111):
        world.frame = frame
        if frame % 2 == 1:  # the policy's steps
            recording.feed_lidar(
                frame, np.array([[11.25, -3.5, 0.7, 0.9]], dtype=np.float32)
            )
            recording.feed_camera_jpeg(frame, "cam_front", jpeg)
        recording.record(world)  # type: ignore[arg-type]
    root = recording.close()
    assert root is not None
    assert t4.active_recording() is None

    assert len(_table(root, "sample")) == 5
    first = _table(root, "ego_pose")[0]
    np.testing.assert_allclose(
        first["translation"][0], world.ego_x(101) - 1.25 + OFFSETS[0], atol=1e-9
    )
    (camera,) = [s for s in _table(root, "sample_data") if s["fileformat"] == "jpg"][:1]
    assert (root / "0" / camera["filename"]).read_bytes() == jpeg
    assert (camera["width"], camera["height"]) == (64, 48)
    assert [a["num_lidar_pts"] for a in _table(root, "sample_annotation")] == [1] * 5


def test_a_recording_after_a_driver_fed_one_is_not_fed_by_a_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver entity withdraws its offer when its scenario ends, so the next
    recording in the process puts sensors on the ego itself."""
    (tmp_path / "fed").mkdir()
    (tmp_path / "plain").mkdir()
    fed_world = _World()
    t4.offer_driver_sensors(-1.25)
    try:
        fed = _start(tmp_path / "fed", monkeypatch, fed_world)
    finally:
        t4.withdraw_driver_sensors()
    assert fed.fed_by_driver
    fed.close()
    world = _World()
    recording = _start(tmp_path / "plain", monkeypatch, world)
    assert not recording.fed_by_driver
    assert len(world.sensors) == 2


def _jpeg() -> bytes:
    import cv2  # noqa: PLC0415

    ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()
