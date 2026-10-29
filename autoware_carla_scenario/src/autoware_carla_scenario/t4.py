"""T4 datasets of runs (``t4.enabled``; the ``t4`` extra).

With ``t4.enabled=true`` each run is also written as a T4 dataset, TIER IV's
nuScenes-style format that `t4-devkit <https://github.com/tier4/t4-devkit>`_
reads, in ``{stem}.t4/0/`` beside the run's other files.  Its sensors are the
driver policy's (``driver.lidars``, ``driver.cameras``), and a key frame is a
policy step (``driver.policy_timestep_s``): the ego's pose, the LiDARs' sweeps
as one ``LIDAR_CONCAT`` cloud in the ego's frame, a frame of every camera, and
the box of every vehicle and pedestrian near the ego, which CARLA knows
exactly.

Where the data comes from:

* an ego driven by a driver policy (``ego.entity=carla_driver``) feeds the
  dataset what it sends the policy (:meth:`T4Recording.feed_lidar`,
  :meth:`T4Recording.feed_camera_jpeg`) -- the sweeps already in the rig
  frame and the frames already JPEG -- so the dataset renders and encodes
  nothing of its own;
* any other ego gets the same sensors put on it by the recorder, at the same
  rate.

The dataset is written by ``autoware-carla-scenario-t4``, a Rust writer with
a PyO3 binding (``autoware_carla_scenario_t4.T4Writer``), which runs on every
key frame: it turns the CARLA states into the map frame, keeps the tables, and
hands the files to threads of its own, which write the sweeps, encode the raw
camera frames and count each box's points.  This module only reads the world
and hands it over; what the dataset holds (:class:`T4Settings`), its names
and categories and the OpenCV model of an F-theta camera are
:mod:`autoware_carla_scenario.t4_settings`'s, which this module re-exports
and the library check compiles (#731).

Frames (docs/t4_dataset.md): the map frame is Autoware's (the Lanelet2 map's:
CARLA's with ``y`` flipped, plus the map's MGRS and elevation offsets); the
ego's frame (``base_link``) is at its rear axle, as the driver policy's rig.

Recording is best effort, like the trajectory log: an error is logged and the
dataset stops, it never fails the scenario.
"""

from __future__ import annotations

import importlib
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from carla_driver_interface.geometry import body_to_optical

from .constants import FIXED_DELTA_SECONDS
from .coordinate.map_manager import MapManager
from .driver.observation import (
    lidar_points_to_rig,
    rear_axle_offset,
    sensor_pose_in_rig,
)
from .sensor.carla_camera import BLUEPRINT_ATTRIBUTES, CarlaCameraSensorConfig
from .sensor.carla_lidar import CarlaLidarSensor
from .t4_settings import (
    DATASET_SUFFIX,
    LIDAR_CHANNEL,
    T4Settings,
    camera_channel,
    category_of,
    dataset_path,
    opencv_of_ftheta,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "DATASET_SUFFIX",
    "INSTALL_HINT",
    "LIDAR_CHANNEL",
    "T4Recorder",
    "T4Recording",
    "T4Settings",
    "T4UnavailableError",
    "active_recording",
    "camera_channel",
    "category_of",
    "dataset_path",
    "opencv_of_ftheta",
    "offer_driver_sensors",
    "offered_rear_axle",
    "require_writer",
    "withdraw_driver_sensors",
]

logger = logging.getLogger(__name__)

#: How to install what this module needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[t4]'"
#: The package that writes the datasets.
_WRITER_MODULE = "autoware_carla_scenario_t4"
#: The dataset version a run writes (T4's ``<dataset id>/<version>/``).
_VERSION = "0"
#: How long a key frame waits for a camera's image, in seconds.
_FRAME_TIMEOUT_S = 1.0
#: How far below the interval a sensor's ``sensor_tick`` is set; see
#: ``sensor.carla_lidar._SENSOR_TICK_MARGIN_S``, which the LiDARs use.
_SENSOR_TICK_MARGIN_S = 1e-6
#: The actors annotated, by blueprint prefix.
_ANNOTATED_KINDS = ("vehicle.", "walker.")

_lock = threading.Lock()
#: The recording of the run in progress, which a driver entity feeds.
_active: Optional["T4Recording"] = None
#: The rear-axle offset of a driver entity whose sensors feed the dataset, while
#: its scenario runs; None when no such entity is running.
_driver_rear_axle: Optional[float] = None


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


def active_recording() -> Optional["T4Recording"]:
    """The recording of the run in progress, if one is fed by a driver entity."""
    return _active


def offer_driver_sensors(rear_axle_offset_m: float) -> None:
    """A driver entity's scenario starts with its sensors attached and its
    rig origin *rear_axle_offset_m* from the actor: a dataset started now is
    fed by it rather than putting sensors of its own on the ego."""
    global _driver_rear_axle
    with _lock:
        _driver_rear_axle = float(rear_axle_offset_m)


def offered_rear_axle() -> Optional[float]:
    """The rig origin's offset from the actor a driver entity offered
    (:func:`offer_driver_sensors`), or ``None`` without one: where the run
    takes the ego's rear axle to be."""
    with _lock:
        return _driver_rear_axle


def withdraw_driver_sensors() -> None:
    """The driver entity's scenario has ended: its sensors are gone."""
    global _driver_rear_axle
    with _lock:
        _driver_rear_axle = None


def _attribute(actor: Any, name: str) -> str:
    try:
        return str(actor.attributes.get(name, ""))
    except Exception:  # noqa: BLE001 - a missing attribute is an empty one
        return ""


def _xyz(values: Any) -> tuple[float, float, float]:
    """The first three of *values* (a sequence or an array) as floats."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _wxyz(quat_xyzw: Any) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return (w, x, y, z)


def _map_offset() -> tuple[float, float, float]:
    """The loaded Lanelet2 map's MGRS and elevation offsets, or zeros."""
    mm = MapManager.get_instance()
    try:
        ox, oy = mm.mgrs_offset
        return (float(ox), float(oy), float(mm.z_offset))
    except RuntimeError:  # no Lanelet2 map loaded
        return (0.0, 0.0, 0.0)


class _FrameCamera:
    """A ``sensor.camera.rgb`` that keeps its newest frames by CARLA frame."""

    def __init__(self, config: CarlaCameraSensorConfig, interval_ticks: int) -> None:
        self.config = config
        self._interval_ticks = interval_ticks
        self._sensor: Optional[Any] = None
        self._frames: dict[int, np.ndarray[Any, np.dtype[np.uint8]]] = {}
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
        """On a CARLA thread: copy the BGRA frame out of CARLA's buffer, which
        is reused, and keep the newest few only."""
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
        except Exception:
            logger.exception("T4: failed to receive a camera frame")

    def newest_frame(self) -> Optional[int]:
        """The CARLA frame of the newest image received, if any."""
        with self._arrived:
            return max(self._frames) if self._frames else None

    def image_at(
        self, frame: int, timeout_s: float = _FRAME_TIMEOUT_S
    ) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
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
    """One run's dataset: the writer, and the sensors that feed it.

    Made by :meth:`T4Recorder.start`; :meth:`record` after every world tick
    (once the ego has had its tick), :meth:`close` when the run ends.
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
        #: Ids of the actors seen, and whether each is annotated.
        self._seen: dict[int, bool] = {}
        #: World snapshots of the last ticks, by frame: a sweep of the
        #: recorder's own LiDARs can be from a tick before it is read on.
        self._snapshots: dict[int, Any] = {}
        self._last_frame = -1
        self._active = True
        self._lidars: list[tuple[CarlaLidarSensor, Any]] = []
        self._cameras: list[tuple[str, _FrameCamera]] = []
        #: Each camera's resolution, by logical id.
        self._resolutions = {
            logical_id: (cam.image_width, cam.image_height)
            for logical_id, cam in settings.cameras
        }
        #: What a driver entity fed for a frame: ``(frame, sweeps, frames)``.
        self._fed: Optional[tuple[int, list[Any], list[tuple[str, bytes]]]] = None

        with _lock:
            driver_rear_axle = _driver_rear_axle
        self.fed_by_driver = driver_rear_axle is not None
        if driver_rear_axle is not None:
            self._rear_axle = driver_rear_axle
        else:
            self._rear_axle = rear_axle_offset(ego, settings.rear_axle_offset_m)

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
            map_offset=_map_offset(),
            rear_axle_offset_m=self._rear_axle,
            max_distance_m=settings.max_distance_m,
            jpeg_quality=settings.jpeg_quality,
            workers=settings.workers,
        )
        if lanelet2_path is not None and Path(lanelet2_path).is_file():
            self._writer.add_lanelet2_map(Path(lanelet2_path))
            from .coordinate.projection import PROJECTOR_INFO_FILENAME  # noqa: PLC0415

            info = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
            if info.is_file():
                shutil.copy(
                    info, self.path / _VERSION / "map" / PROJECTOR_INFO_FILENAME
                )

        self._declare_sensors()
        if not self.fed_by_driver:
            self._attach_sensors(world, ego)
        logger.info(
            "T4 dataset: %d LiDAR(s) and %d camera(s) %s, every %d tick(s) -> %s",
            len(settings.lidars),
            len(settings.cameras),
            "fed by the driver policy's" if self.fed_by_driver else "on the ego",
            settings.interval_ticks,
            self.path,
        )

    def _declare_sensors(self) -> None:
        """Declare the channels: the LiDARs' cloud in the ego's frame, and each
        camera's optical frame on the ego with its pinhole."""
        if self._settings.lidars:
            self._writer.add_sensor(
                LIDAR_CHANNEL, "lidar", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
            )
        lenses = dict(self._settings.lenses)
        for logical_id, cam in self._settings.cameras:
            intrinsic = [
                [cam.fx, 0.0, cam.image_width / 2.0],
                [0.0, cam.fx, cam.image_height / 2.0],
                [0.0, 0.0, 1.0],
            ]
            distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
            lens = lenses.get(logical_id)
            if lens is not None:
                intrinsic, distortion, mean_px, max_px = opencv_of_ftheta(
                    lens, cam.image_width, cam.image_height
                )
                logger.info(
                    "T4: camera %s is an F-theta lens; written as OpenCV's rational "
                    "model fitted to it (%.3f px on average, %.2f px at most)",
                    logical_id,
                    mean_px,
                    max_px,
                )
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
                camera_channel(logical_id),
                "camera",
                _xyz(optical.position),
                _wxyz(optical.quat_xyzw),
                camera_intrinsic=intrinsic,
                camera_distortion=distortion,
            )

    def _attach_sensors(self, world: "carla.World", ego: "carla.Actor") -> None:
        """Put the driver policy's sensors on an ego that has none of its own."""
        interval = self._settings.interval_ticks
        try:
            for cfg in self._settings.lidars:
                mount = sensor_pose_in_rig(
                    cfg.position_x,
                    cfg.position_y,
                    cfg.position_z,
                    cfg.roll,
                    cfg.pitch,
                    cfg.yaw,
                    self._rear_axle,
                )
                lidar = CarlaLidarSensor(
                    cfg, sweep_interval_s=interval * FIXED_DELTA_SECONDS
                )
                lidar.attach(world, ego)
                self._lidars.append((lidar, mount))
            for logical_id, cam in self._settings.cameras:
                camera = _FrameCamera(cam, interval)
                camera.attach(world, ego)
                self._cameras.append((camera_channel(logical_id), camera))
        except Exception:
            self._destroy_sensors()
            raise

    @property
    def active(self) -> bool:
        """Whether frames are still being recorded."""
        return self._active

    @property
    def frames(self) -> int:
        """The key frames recorded so far."""
        return int(self._writer.frames)

    # -- What a driver entity feeds -----------------------------------------

    def _fed_for(self, frame: int) -> tuple[int, list[Any], list[tuple[str, bytes]]]:
        if self._fed is None or self._fed[0] != frame:
            self._fed = (frame, [], [])
        return self._fed

    def feed_lidar(self, frame: int, points_in_rig: Any) -> None:
        """A sweep the driver policy got on *frame*: ``[N, 4]`` float32 in the
        rig frame (the ego's ``base_link``)."""
        if self._active:
            self._fed_for(frame)[1].append(points_in_rig)

    def feed_camera_jpeg(self, frame: int, logical_id: str, jpeg: bytes) -> None:
        """A camera frame the driver policy got on *frame*, as it got it."""
        if self._active:
            self._fed_for(frame)[2].append((logical_id, jpeg))

    # -- Every tick -----------------------------------------------------------

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
        if self.fed_by_driver:
            fed, self._fed = self._fed, None
            if fed is None or fed[0] != frame or frame < self.start_frame:
                return
            self._write_frame(world, frame, snapshot, fed[1], fed[2], [])
            return

        self._snapshots[frame] = snapshot
        horizon = frame - 2 * self._settings.interval_ticks
        for old in [f for f in self._snapshots if f < horizon]:
            del self._snapshots[old]
        key_frame: Optional[int] = None
        sweeps: list[tuple[int, Any, Any]] = []
        for lidar, mount in self._lidars:
            sweep = lidar.latest_sweep(frame)
            if sweep is not None and sweep[0] > self._last_frame:
                sweeps.append((sweep[0], sweep[1], mount))
                key_frame = sweep[0] if key_frame is None else min(key_frame, sweep[0])
        if not self._lidars:
            # Cameras alone: the first one's frames are the key frames.
            newest = self._cameras[0][1].newest_frame()
            if newest is not None and newest > self._last_frame:
                key_frame = newest
        if key_frame is None or key_frame < self.start_frame:
            return
        key_snapshot = self._snapshots.get(key_frame)
        if key_snapshot is None:
            logger.debug("T4 dataset: no snapshot of frame %d; skipped", key_frame)
            return
        self._last_frame = key_frame
        # A LiDAR whose sweep is later than the key frame's joins the next one.
        points = [
            lidar_points_to_rig(raw, mount)
            for f, raw, mount in sweeps
            if f == key_frame
        ]
        raw_frames = []
        for channel, camera in self._cameras:
            image = camera.image_at(key_frame)
            if image is None:
                logger.debug("T4 dataset: no %s frame for frame %d", channel, key_frame)
            else:
                raw_frames.append((channel, image))
        self._write_frame(world, key_frame, key_snapshot, points, [], raw_frames)

    def _write_frame(
        self,
        world: "carla.World",
        frame: int,
        snapshot: Any,
        points: list[Any],
        jpegs: list[tuple[str, bytes]],
        raw_frames: list[tuple[str, Any]],
    ) -> None:
        ego = snapshot.find(self._ego_id)
        if ego is None:
            return
        writer = self._writer
        tf, v = ego.get_transform(), ego.get_velocity()
        w, a = ego.get_angular_velocity(), ego.get_acceleration()
        loc, rot = tf.location, tf.rotation
        writer.begin_frame_carla(
            self._epoch_us
            + int(round((frame - self.start_frame) * FIXED_DELTA_SECONDS * 1e6)),
            (loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw),
            (v.x, v.y, v.z),
            (w.x, w.y, w.z),
            (a.x, a.y, a.z),
        )
        try:
            self._annotate(world, snapshot)
            for cloud in points:
                writer.add_lidar(LIDAR_CHANNEL, cloud)
            for logical_id, jpeg in jpegs:
                resolution = self._resolutions.get(logical_id)
                if resolution is not None:
                    writer.add_camera_jpeg(
                        camera_channel(logical_id), jpeg, *resolution
                    )
            for channel, image in raw_frames:
                writer.add_camera(channel, image, "bgra")
            self._vehicle_state(float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)))
        finally:
            writer.end_frame()

    def _annotate(self, world: "carla.World", snapshot: Any) -> None:
        """Hand the writer the vehicles' and pedestrians' states: one array,
        which it turns into boxes in the map frame."""
        seen = self._seen
        new = [s.id for s in snapshot if s.id not in seen]
        if new:
            for actor in world.get_actors(new):
                seen[int(actor.id)] = self._register(actor)
            for actor_id in new:  # gone before it could be looked up
                seen.setdefault(int(actor_id), False)
        ids: list[int] = []
        rows: list[tuple[float, ...]] = []
        for s in snapshot:
            if not seen.get(s.id):
                continue
            tf, v = s.get_transform(), s.get_velocity()
            loc, rot = tf.location, tf.rotation
            ids.append(s.id)
            rows.append(
                (loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw, v.x, v.y, v.z)
            )
        if ids:
            self._writer.add_carla_boxes(
                np.asarray(ids, dtype=np.uint64),
                np.asarray(rows, dtype=np.float64).reshape(-1, 9),
            )

    def _register(self, actor: Any) -> bool:
        """Register *actor* with the writer when it is annotated: a vehicle or
        pedestrian other than the ego."""
        type_id = str(actor.type_id)
        if int(actor.id) == self._ego_id or not type_id.startswith(_ANNOTATED_KINDS):
            return False
        try:
            wheels = int(_attribute(actor, "number_of_wheels") or 0)
        except ValueError:
            wheels = 0
        box = actor.bounding_box
        role = _attribute(actor, "role_name")
        self._writer.register_actor(
            int(actor.id),
            category_of(type_id, _attribute(actor, "base_type"), wheels),
            (box.location.x, -box.location.y, box.location.z),
            (2.0 * box.extent.y, 2.0 * box.extent.x, 2.0 * box.extent.z),
            f"{role or type_id}#{actor.id}",
        )
        return True

    def _vehicle_state(self, speed: float) -> None:
        try:
            control = self._ego.get_control()
        except Exception:  # noqa: BLE001 - a gone ego has none
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
        for lidar, _ in self._lidars:
            try:
                lidar.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("T4 dataset: destroying a LiDAR failed", exc_info=True)
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
        """Remove the recorder's sensors and finish the dataset; its path, or
        ``None`` when it holds nothing or could not be finished (logged)."""
        global _active
        with _lock:
            if _active is self:
                _active = None
        self._active = False
        self._destroy_sensors()
        self._snapshots = {}
        self._fed = None
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
        settings: The sensors and the rate (:meth:`T4Settings.from_driver`).
        writer: The writer package; imported (:func:`require_writer`) when
            None.

    Raises:
        T4UnavailableError: The writer is not installed.
    """

    def __init__(
        self, settings: T4Settings, writer: Optional[ModuleType] = None
    ) -> None:
        self.settings = settings
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
        """Start ``{directory}/{stem}.t4`` from CARLA frame *start_frame* (the
        scenario clock's start), fed by the ego's driver policy or with sensors
        of its own on *ego*; ``None``, with a warning, when it cannot start."""
        global _active
        try:
            recording = T4Recording(
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
        with _lock:
            _active = recording
        return recording
