"""T4 datasets of runs (``t4.enabled``; the ``t4`` extra).

With ``t4.enabled=true`` each run is also written as a T4 dataset, TIER IV's
nuScenes-style format that `t4-devkit <https://github.com/tier4/t4-devkit>`_
reads, in ``{stem}.t4/0/`` beside the run's other files.  The recorder puts
sensors of its own on the ego -- the LiDARs and cameras ``t4.lidars`` and
``t4.cameras`` list, whatever the ego's entity is -- and makes a key frame of
every sweep: the ego's pose, a sweep of every LiDAR, a frame of every camera,
and the box of every vehicle and pedestrian, which CARLA knows exactly.

The dataset is written by ``autoware-carla-scenario-t4``, a Rust writer with
a PyO3 binding (``autoware_carla_scenario_t4.T4Writer``): this module only
reads the world and the sensors and hands them over.  The writer encodes and
writes the sensor files on threads of its own while the run goes on.

Frames (docs/t4_dataset.md):

* the map frame is Autoware's: the Lanelet2 map's (CARLA's with ``y``
  flipped, plus the map's MGRS and elevation offsets), or CARLA's with ``y``
  flipped when no Lanelet2 map is loaded;
* the ego's frame (``base_link``) is at its rear axle, x forward, y left, z up;
* a LiDAR's points are in its own frame (CARLA's sensor frame with ``y``
  flipped), a camera's frame is the optical one (x right, y down, z forward).

Recording is best effort, like the trajectory log: an error is logged and the
dataset stops, it never fails the scenario.
"""

from __future__ import annotations

import importlib
import logging
import math
import shutil
import threading
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
from carla_driver_interface.geometry import Pose, body_to_optical

from .constants import FIXED_DELTA_SECONDS
from .coordinate.map_manager import MapManager
from .driver.observation import rear_axle_offset, sensor_pose_in_rig
from .sensor.carla_camera import BLUEPRINT_ATTRIBUTES, CarlaCameraSensorConfig
from .sensor.carla_lidar import CarlaLidarSensor, CarlaLidarSensorConfig

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "DATASET_SUFFIX",
    "INSTALL_HINT",
    "T4CameraConfig",
    "T4LidarConfig",
    "T4Recorder",
    "T4Recording",
    "T4Settings",
    "T4UnavailableError",
    "category_of",
    "dataset_path",
    "require_writer",
]

logger = logging.getLogger(__name__)

#: How to install what this module needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[t4]'"
#: What a run's dataset is named: ``{stem}.t4``, its one version ``0`` inside.
DATASET_SUFFIX = ".t4"
#: The package that writes the datasets.
_WRITER_MODULE = "autoware_carla_scenario_t4"
#: The dataset version a run writes (T4's ``<dataset id>/<version>/``).
_VERSION = "0"
#: How long a frame waits for a camera's image, in seconds.
_FRAME_TIMEOUT_S = 1.0
#: How far below the interval a sensor's ``sensor_tick`` is set; see
#: ``sensor.carla_lidar._SENSOR_TICK_MARGIN_S``, which the LiDARs use.
_SENSOR_TICK_MARGIN_S = 1e-6
#: The actors annotated, by blueprint prefix.
_ANNOTATED_KINDS = ("vehicle.", "walker.")
#: A vehicle blueprint's ``base_type`` -> its T4 category.
_VEHICLE_CATEGORIES = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


class T4UnavailableError(RuntimeError):
    """``t4.enabled`` is set, but the writer is not installed."""


def require_writer() -> ModuleType:
    """Return the writer package, or raise :class:`T4UnavailableError` saying
    how to install it.

    Called before a run starts, so a run asking for datasets it cannot write
    is refused before it spends any time on CARLA.
    """
    try:
        return importlib.import_module(_WRITER_MODULE)
    except ImportError as exc:
        raise T4UnavailableError(
            "t4.enabled is set, but autoware-carla-scenario-t4, which writes "
            f"the T4 datasets, is not installed: {INSTALL_HINT} "
            "(or set t4.enabled=false)"
        ) from exc


def dataset_path(directory: Path, stem: str) -> Path:
    """``{directory}/{stem}.t4``: the run's dataset (its version ``0`` inside)."""
    return Path(directory) / f"{stem}{DATASET_SUFFIX}"


@dataclass(frozen=True)
class T4LidarConfig:
    """A LiDAR the recorder puts on the ego: its channel and its sensor."""

    channel: str = "LIDAR_TOP"
    sensor: CarlaLidarSensorConfig = field(
        default_factory=lambda: CarlaLidarSensorConfig(position_z=2.2)
    )


@dataclass(frozen=True)
class T4CameraConfig:
    """A camera the recorder puts on the ego: its channel and its sensor.

    The camera takes a frame with every key frame: its ``fps`` is not used.
    """

    channel: str = "CAM_FRONT"
    sensor: CarlaCameraSensorConfig = field(
        default_factory=lambda: CarlaCameraSensorConfig(
            image_width=1280, image_height=720, fov=90.0, position_x=1.0, position_z=1.6
        )
    )


def _sensor_config(kind: type, channel_default: str, entry: Mapping[str, Any]) -> Any:
    """``entry`` (``{channel, <sensor fields>}``) as a :class:`T4LidarConfig`
    or :class:`T4CameraConfig` (*kind*)."""
    values = dict(entry)
    channel = str(values.pop("channel", channel_default))
    sensor_type = (
        CarlaLidarSensorConfig if kind is T4LidarConfig else CarlaCameraSensorConfig
    )
    known = {f.name for f in fields(sensor_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"t4: {channel} has no setting {', '.join(unknown)} "
            f"(a {sensor_type.__name__} has {', '.join(sorted(known))})"
        )
    return kind(channel=channel, sensor=sensor_type(**values))


@dataclass(frozen=True)
class T4Settings:
    """What a run's dataset holds, and how often (the ``t4`` config)."""

    #: Key frames a second; a whole number of ticks apart.
    frequency_hz: float = 10.0
    lidars: tuple[T4LidarConfig, ...] = (T4LidarConfig(),)
    cameras: tuple[T4CameraConfig, ...] = (T4CameraConfig(),)
    #: Annotate the road users within this distance of the ego; None for all.
    max_distance_m: Optional[float] = 100.0
    #: The camera frames' JPEG quality, 1-100.
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not self.lidars and not self.cameras:
            raise ValueError("t4: a dataset needs at least one LiDAR or camera")
        channels = [c.channel for c in self.lidars] + [c.channel for c in self.cameras]
        duplicates = sorted({c for c in channels if channels.count(c) > 1})
        if duplicates:
            raise ValueError(f"t4: channel {', '.join(duplicates)} is used twice")
        self.interval_ticks  # noqa: B018 - validates frequency_hz
        if self.max_distance_m is not None and self.max_distance_m <= 0.0:
            raise ValueError("t4.max_distance_m must be positive (or null)")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("t4.jpeg_quality must be 1-100")

    @property
    def interval_ticks(self) -> int:
        """World ticks from one key frame to the next."""
        if self.frequency_hz <= 0.0:
            raise ValueError("t4.frequency_hz must be positive")
        ticks = round(1.0 / (self.frequency_hz * FIXED_DELTA_SECONDS))
        if (
            ticks < 1
            or abs(ticks * self.frequency_hz * FIXED_DELTA_SECONDS - 1.0) > 1e-6
        ):
            raise ValueError(
                f"t4.frequency_hz must divide the {1.0 / FIXED_DELTA_SECONDS:g} Hz "
                f"tick rate into whole ticks, got {self.frequency_hz}"
            )
        return int(ticks)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "T4Settings":
        """The settings a ``t4`` config section gives (``enabled`` aside)."""
        values: dict[str, Any] = {}
        if config.get("frequency_hz") is not None:
            values["frequency_hz"] = float(config["frequency_hz"])
        if "max_distance_m" in config:
            distance = config["max_distance_m"]
            values["max_distance_m"] = None if distance is None else float(distance)
        if config.get("jpeg_quality") is not None:
            values["jpeg_quality"] = int(config["jpeg_quality"])
        if config.get("lidars") is not None:
            values["lidars"] = tuple(
                _sensor_config(T4LidarConfig, "LIDAR_TOP", entry)
                for entry in config["lidars"]
            )
        if config.get("cameras") is not None:
            values["cameras"] = tuple(
                _sensor_config(T4CameraConfig, "CAM_FRONT", entry)
                for entry in config["cameras"]
            )
        return cls(**values)


def category_of(type_id: str, base_type: str = "", wheels: int = 0) -> str:
    """The T4 category of a CARLA blueprint: ``pedestrian`` for a walker; a
    vehicle's from its ``base_type`` (``car``, ``truck``, ``bus``,
    ``motorcycle``, ``bicycle``), else ``motorcycle`` on two wheels, else
    ``car``."""
    if type_id.startswith("walker."):
        return "pedestrian"
    category = _VEHICLE_CATEGORIES.get(base_type.lower())
    if category is not None:
        return category
    return "motorcycle" if wheels == 2 else "car"


def _attribute(actor: Any, name: str) -> str:
    try:
        return str(actor.attributes.get(name, ""))
    except Exception:  # noqa: BLE001 - a missing attribute is an empty one
        return ""


def _carla_pose(transform: Any, offsets: tuple[float, float, float]) -> Pose:
    """A CARLA transform as a pose in the map frame (right-handed): ``y``
    flipped, CARLA's ``yaw * pitch * roll`` rebuilt with the yaw and pitch
    negated (``driver.observation.camera_extrinsics_to_rig``), and the map's
    offsets added."""
    loc, rot = transform.location, transform.rotation
    rotation = (
        Pose.from_axis_angle(2, -math.radians(rot.yaw))
        @ Pose.from_axis_angle(1, -math.radians(rot.pitch))
        @ Pose.from_axis_angle(0, math.radians(rot.roll))
    )
    return Pose(
        np.array(
            [loc.x + offsets[0], -loc.y + offsets[1], loc.z + offsets[2]],
            dtype=np.float64,
        ),
        rotation.quat_xyzw,
    )


def _wxyz(pose: Pose) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in pose.quat_xyzw)
    return (w, x, y, z)


def _xyz(values: Any) -> tuple[float, float, float]:
    """The first three of *values* (a sequence or an array) as floats."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _mirrored(vector: Any) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """A CARLA world vector in the map frame (``y`` flipped)."""
    return np.array([vector.x, -vector.y, vector.z], dtype=np.float64)


def _mirrored_rate(vector: Any) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """A CARLA angular velocity (deg/s) in the map frame, in rad/s: a
    pseudovector, so the flip of ``y`` negates ``x`` and ``z`` instead."""
    return np.radians(np.array([-vector.x, vector.y, -vector.z], dtype=np.float64))


def _map_offsets() -> tuple[float, float, float]:
    """The loaded Lanelet2 map's MGRS and elevation offsets, or zeros."""
    mm = MapManager.get_instance()
    try:
        ox, oy = mm.mgrs_offset
        return (float(ox), float(oy), float(mm.z_offset))
    except RuntimeError:  # no Lanelet2 map loaded
        return (0.0, 0.0, 0.0)


@dataclass
class _ActorInfo:
    """What a recorded actor's boxes need that does not change."""

    category: str
    name: str
    #: The box's centre in the actor's frame (right-handed) and its size
    #: ``(width, length, height)``.
    center: np.ndarray[tuple[int], np.dtype[np.float64]]
    size: tuple[float, float, float]


class _FrameCamera:
    """A ``sensor.camera.rgb`` that keeps its newest frames by CARLA frame."""

    def __init__(self, config: CarlaCameraSensorConfig, interval_ticks: int) -> None:
        self.config = config
        self._interval_ticks = interval_ticks
        self._sensor: Optional[Any] = None
        self._frames: dict[
            int, np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
        ] = {}
        self._arrived = threading.Condition()

    def attach(self, world: Any, actor: Any) -> None:
        import typesafe_carla.carla as _carla  # noqa: PLC0415

        cfg = self.config
        blueprint = world.get_blueprint_library().find(cfg.sensor_type)
        blueprint.set_attribute("image_size_x", str(cfg.image_width))
        blueprint.set_attribute("image_size_y", str(cfg.image_height))
        blueprint.set_attribute("fov", str(cfg.fov))
        if self._interval_ticks > 1:
            interval_s = self._interval_ticks * FIXED_DELTA_SECONDS
            blueprint.set_attribute(
                "sensor_tick", str(interval_s - _SENSOR_TICK_MARGIN_S)
            )
        for name in BLUEPRINT_ATTRIBUTES:
            value = getattr(cfg, name)
            if value is not None and blueprint.has_attribute(name):
                blueprint.set_attribute(name, str(value))
        transform = _carla.Transform(
            _carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            _carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )
        self._sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
        self._sensor.listen(self._receive)

    def _receive(self, data: Any) -> None:
        """On a CARLA thread: copy the BGRA frame out, keep the newest few."""
        try:
            image = data.as_image() if hasattr(data, "as_image") else data
            pixels = (
                np.frombuffer(image.raw_data, dtype=np.uint8)
                .reshape(int(image.height), int(image.width), 4)
                .copy()
            )
            with self._arrived:
                self._frames[int(image.frame)] = pixels
                for old in sorted(self._frames)[:-4]:
                    del self._frames[old]
                self._arrived.notify_all()
        except Exception:  # noqa: BLE001 - a sensor thread must not die
            logger.exception("T4: failed to receive a camera frame")

    def newest_frame(self) -> Optional[int]:
        """The CARLA frame of the newest image received, if any."""
        with self._arrived:
            return max(self._frames) if self._frames else None

    def image_at(
        self, frame: int, timeout_s: float = _FRAME_TIMEOUT_S
    ) -> Optional[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
        """The frame taken on *frame*, waiting up to *timeout_s* for it."""
        deadline = time.monotonic() + timeout_s
        with self._arrived:
            while frame not in self._frames:
                if any(f > frame for f in self._frames):
                    return None  # skipped: a later one came first
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._arrived.wait(remaining)
            return self._frames.pop(frame)

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
                self._sensor.destroy()
            finally:
                self._sensor = None


class T4Recording:
    """One run's dataset: the sensors on its ego, and the writer.

    Made by :meth:`T4Recorder.start`; :meth:`record` after every world tick,
    :meth:`close` when the run ends.
    """

    def __init__(
        self,
        writer_module: ModuleType,
        settings: T4Settings,
        path: Path,
        world: "carla.World",
        ego: "carla.Actor",
        start_frame: int,
        *,
        scene_name: str,
        description: str = "",
        location: str = "",
        lanelet2_path: Optional[Path] = None,
    ) -> None:
        self.path = Path(path)
        self.start_frame = int(start_frame)
        self._settings = settings
        self._ego = ego
        self._ego_id = int(ego.id)
        self._offsets = _map_offsets()
        self._actors: dict[int, Optional[_ActorInfo]] = {}
        #: World snapshots of the last ticks, by frame: a sweep can be from a
        #: tick before the one it is read on.
        self._snapshots: dict[int, Any] = {}
        self._last_frame = -1
        self._active = True
        self._lidars: list[tuple[str, CarlaLidarSensor]] = []
        self._cameras: list[tuple[str, _FrameCamera]] = []

        started = time.time()
        self._epoch_us = int(started * 1e6)
        captured = datetime.fromtimestamp(started, tz=timezone.utc)
        self._writer: Any = writer_module.T4Writer(
            self.path / _VERSION,
            scene_name=scene_name,
            scene_description=description,
            logfile=f"{scene_name}.log",
            vehicle=str(ego.type_id),
            date_captured=captured.strftime("%Y-%m-%d-%H-%M-%S"),
            location=location,
            token_seed=f"{scene_name}@{self._epoch_us}",
            jpeg_quality=settings.jpeg_quality,
        )
        if lanelet2_path is not None and Path(lanelet2_path).is_file():
            self._writer.add_lanelet2_map(Path(lanelet2_path))
            from .coordinate.projection import PROJECTOR_INFO_FILENAME  # noqa: PLC0415

            info = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
            if info.is_file():
                shutil.copy(
                    info, self.path / _VERSION / "map" / PROJECTOR_INFO_FILENAME
                )

        self._rear_axle = rear_axle_offset(ego)
        interval = settings.interval_ticks
        try:
            for lidar_config in settings.lidars:
                cfg = lidar_config.sensor
                mount = sensor_pose_in_rig(
                    cfg.position_x,
                    cfg.position_y,
                    cfg.position_z,
                    cfg.roll,
                    cfg.pitch,
                    cfg.yaw,
                    self._rear_axle,
                )
                self._writer.add_sensor(
                    lidar_config.channel,
                    "lidar",
                    _xyz(mount.position),
                    _wxyz(mount),
                )
                lidar = CarlaLidarSensor(
                    cfg, sweep_interval_s=interval * FIXED_DELTA_SECONDS
                )
                lidar.attach(world, ego)
                self._lidars.append((lidar_config.channel, lidar))
            for camera_config in settings.cameras:
                cam = camera_config.sensor
                body = sensor_pose_in_rig(
                    cam.position_x,
                    cam.position_y,
                    cam.position_z,
                    cam.roll,
                    cam.pitch,
                    cam.yaw,
                    self._rear_axle,
                )
                optical = body_to_optical(body)
                self._writer.add_sensor(
                    camera_config.channel,
                    "camera",
                    _xyz(optical.position),
                    _wxyz(optical),
                    camera_intrinsic=[
                        [cam.fx, 0.0, cam.image_width / 2.0],
                        [0.0, cam.fx, cam.image_height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    camera_distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
                )
                camera = _FrameCamera(cam, interval)
                camera.attach(world, ego)
                self._cameras.append((camera_config.channel, camera))
        except Exception:
            self._destroy_sensors()
            raise
        logger.info(
            "T4 dataset: %d LiDAR(s) and %d camera(s) on the ego, %g Hz -> %s",
            len(self._lidars),
            len(self._cameras),
            settings.frequency_hz,
            self.path,
        )

    @property
    def active(self) -> bool:
        """Whether frames are still being recorded."""
        return self._active

    @property
    def frames(self) -> int:
        """The key frames recorded so far."""
        return int(self._writer.frames)

    def record(self, world: "carla.World") -> None:
        """Record the tick the world was just stepped to, when it completes a
        key frame; an error stops the dataset (logged), never the run."""
        if not self._active:
            return
        try:
            self._record(world)
        except Exception:
            logger.exception("T4 dataset: recording stopped (%s)", self.path)
            self._active = False

    def _record(self, world: "carla.World") -> None:
        snapshot = world.get_snapshot()
        frame = int(snapshot.frame)
        self._snapshots[frame] = snapshot
        horizon = frame - 2 * self._settings.interval_ticks
        for old in [f for f in self._snapshots if f < horizon]:
            del self._snapshots[old]

        sweeps: dict[str, Any] = {}
        key_frame: Optional[int] = None
        for channel, lidar in self._lidars:
            sweep = lidar.latest_sweep(frame)
            if sweep is None or sweep[0] <= self._last_frame:
                continue
            sweeps[channel] = sweep
            key_frame = sweep[0] if key_frame is None else min(key_frame, sweep[0])
        if not self._lidars:
            # Cameras alone: the first one's frames are the key frames.
            newest = self._cameras[0][1].newest_frame()
            if newest is not None and newest > self._last_frame:
                key_frame = newest
        if key_frame is None or key_frame < self.start_frame:
            return
        # A LiDAR whose sweep is later than the key frame's joins the next one.
        sweeps = {c: s for c, s in sweeps.items() if s[0] == key_frame}
        key_snapshot = self._snapshots.get(key_frame)
        if key_snapshot is None:
            logger.debug("T4 dataset: no snapshot of frame %d; skipped", key_frame)
            return
        self._last_frame = key_frame
        self._write_frame(world, key_frame, key_snapshot, sweeps)

    def _timestamp_us(self, frame: int) -> int:
        return self._epoch_us + int(
            round((frame - self.start_frame) * FIXED_DELTA_SECONDS * 1e6)
        )

    def _write_frame(
        self, world: "carla.World", frame: int, snapshot: Any, sweeps: dict[str, Any]
    ) -> None:
        ego_snapshot = snapshot.find(self._ego_id)
        if ego_snapshot is None:
            return
        actor_pose = _carla_pose(ego_snapshot.get_transform(), self._offsets)
        base_link = actor_pose @ Pose.from_xyz_yaw(self._rear_axle, 0.0, 0.0, 0.0)
        to_local = base_link.rotation.inv()
        omega = _mirrored_rate(ego_snapshot.get_angular_velocity())
        lever = base_link.position - actor_pose.position
        velocity = _mirrored(ego_snapshot.get_velocity()) + np.cross(omega, lever)
        v_local = to_local.apply(velocity)
        w_local = to_local.apply(omega)
        a_local = to_local.apply(_mirrored(ego_snapshot.get_acceleration()))

        writer = self._writer
        writer.begin_frame(
            self._timestamp_us(frame),
            _xyz(base_link.position),
            _wxyz(base_link),
            twist=(
                *_xyz(v_local),
                float(w_local[2]),
                float(w_local[1]),
                float(w_local[0]),
            ),
            acceleration=_xyz(a_local),
        )
        try:
            self._annotate(world, snapshot, base_link)
            for channel, (_, raw) in sweeps.items():
                points = np.array(raw, dtype=np.float32, copy=True).reshape(-1, 4)
                points[:, 1] *= -1.0  # CARLA's sensor frame is left-handed
                writer.add_lidar(channel, points)
            for channel, camera in self._cameras:
                image = camera.image_at(frame)
                if image is None:
                    logger.debug("T4 dataset: no %s frame for frame %d", channel, frame)
                    continue
                writer.add_camera(channel, image, "bgra")
            self._vehicle_state(float(np.linalg.norm(velocity)))
        finally:
            writer.end_frame()

    def _annotate(self, world: "carla.World", snapshot: Any, base_link: Pose) -> None:
        new = [s.id for s in snapshot if s.id not in self._actors]
        if new:
            for actor in world.get_actors(new):
                self._actors[int(actor.id)] = self._actor_info(actor)
            for actor_id in new:  # gone before it could be looked up
                self._actors.setdefault(int(actor_id), None)
        limit = self._settings.max_distance_m
        for s in snapshot:
            actor_id = int(s.id)
            info = self._actors.get(actor_id)
            if info is None or actor_id == self._ego_id:
                continue
            pose = _carla_pose(s.get_transform(), self._offsets)
            center = pose.position + pose.rotation.apply(info.center)
            if (
                limit is not None
                and np.linalg.norm(center - base_link.position) > limit
            ):
                continue
            self._writer.add_annotation(
                actor_id,
                info.category,
                _xyz(center),
                info.size,
                _wxyz(pose),
                velocity=_xyz(_mirrored(s.get_velocity())),
                acceleration=_xyz(_mirrored(s.get_acceleration())),
                instance_name=info.name,
            )

    @staticmethod
    def _actor_info(actor: Any) -> Optional[_ActorInfo]:
        type_id = str(actor.type_id)
        if not type_id.startswith(_ANNOTATED_KINDS):
            return None
        try:
            wheels = int(_attribute(actor, "number_of_wheels") or 0)
        except ValueError:
            wheels = 0
        box = actor.bounding_box
        role = _attribute(actor, "role_name")
        return _ActorInfo(
            category=category_of(type_id, _attribute(actor, "base_type"), wheels),
            name=f"{role or type_id}#{actor.id}",
            center=np.array(
                [box.location.x, -box.location.y, box.location.z], dtype=np.float64
            ),
            size=(2.0 * box.extent.y, 2.0 * box.extent.x, 2.0 * box.extent.z),
        )

    def _vehicle_state(self, speed: float) -> None:
        try:
            control = self._ego.get_control()
        except Exception:  # noqa: BLE001 - a walker or a gone ego has none
            return
        self._writer.add_vehicle_state(
            accel_pedal=100.0 * float(control.throttle),
            brake_pedal=100.0 * float(control.brake),
            # CARLA steers right for positive; the frame is left-positive.
            steer_pedal=-100.0 * float(control.steer),
            shift_state="REVERSE" if bool(control.reverse) else "FORWARD",
            speed=speed,
        )

    def _destroy_sensors(self) -> None:
        for channel, lidar in self._lidars:
            try:
                lidar.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning(
                    "T4 dataset: destroying %s failed", channel, exc_info=True
                )
        for channel, camera in self._cameras:
            try:
                camera.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning(
                    "T4 dataset: destroying %s failed", channel, exc_info=True
                )
        self._lidars = []
        self._cameras = []

    def close(self) -> Optional[Path]:
        """Remove the sensors and finish the dataset; its path, or ``None``
        when it holds nothing or could not be finished (logged)."""
        self._active = False
        self._destroy_sensors()
        self._snapshots = {}
        try:
            summary = self._writer.finish()
        except Exception as error:  # noqa: BLE001 - best effort, like the log
            logger.warning("T4 dataset: not written to %s: %s", self.path, error)
            return None
        logger.info(
            "T4 dataset: %d frame(s), %d box(es) of %d road user(s) written to %s",
            summary["samples"],
            summary["sample_annotations"],
            summary["instances"],
            self.path,
        )
        return self.path


class T4Recorder:
    """Writes each run of a batch as a T4 dataset (``t4.enabled``).

    Args:
        settings: The sensors and the rate; :class:`T4Settings`' defaults
            when None.
        writer: The writer package; imported (:func:`require_writer`) when
            None.

    Raises:
        T4UnavailableError: The writer is not installed.
    """

    def __init__(
        self, settings: Optional[T4Settings] = None, writer: Optional[ModuleType] = None
    ) -> None:
        self.settings = settings if settings is not None else T4Settings()
        self._writer = writer if writer is not None else require_writer()

    def start(
        self,
        directory: Path,
        stem: str,
        world: "carla.World",
        ego: "carla.Actor",
        start_frame: int,
        description: str = "",
        location: str = "",
        lanelet2_path: Optional[Path] = None,
    ) -> Optional[T4Recording]:
        """Put the sensors on *ego* and start ``{directory}/{stem}.t4``, from
        CARLA frame *start_frame* (the scenario clock's start); ``None``, with
        a warning, when it cannot be started."""
        try:
            return T4Recording(
                self._writer,
                self.settings,
                dataset_path(directory, stem),
                world,
                ego,
                start_frame,
                scene_name=stem,
                description=description,
                location=location,
                lanelet2_path=lanelet2_path,
            )
        except Exception:
            logger.exception(
                "T4 dataset: could not start %s", dataset_path(directory, stem)
            )
            return None
