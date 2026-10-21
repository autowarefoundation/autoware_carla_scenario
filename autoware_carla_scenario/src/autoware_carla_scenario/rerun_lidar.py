"""The ego's sensors in a run's Rerun recording (``rerun.lidar``, ``rerun.cameras``).

A run's Rerun recording (``{stem}.rrd``, :mod:`.rerun`) is made from its
trajectory log after the run, and the log holds no sensor data.  With
``rerun.lidar=true`` the sweeps the ego's driver policy is sent, and with
``rerun.cameras=true`` the camera frames it is sent, are written during the run
instead, into ``{stem}.sensors.rrd``: the same application and recording id as
the converter's (``autoware_carla_scenario``, the stem), on its timelines
(``frame``, the CARLA frame; ``sim_time``, the scenario clock) and in its world
frame (CARLA's with ``y`` flipped).  Once the log is converted,
:func:`merge_into` merges them into ``{stem}.rrd`` (``rerun rrd merge``: one
recording, the converter's blueprint kept) and removes ``{stem}.sensors.rrd``.
Two files of one recording id are not merged by the viewer: it registers each
by the id, and the second replaces the first.

Each LiDAR is ``world/lidar/<logical id>``, a ``Points3D`` coloured by height,
inside the converter's 3D view (``world/**``).  A sweep is the points the
policy got, carried into the world with the ego's pose it got them with.

Each camera is ``world/ego/pose/cameras/<logical id>``: its mount on the ego
(``world/ego/pose`` is the ego actor's transform), a pinhole of its field of
view and the JPEG frames the policy got, so the 3D view draws its frustum with
the frame in it, and a 2D view of the entity shows the frames.

Needs the Rerun SDK of the converter's minor version
(``pip install 'autoware-carla-scenario[rerun-lidar]'``), imported only when a
recording starts.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .constants import FIXED_DELTA_SECONDS

__all__ = [
    "APPLICATION_ID",
    "SENSORS_SUFFIX",
    "SensorRecorder",
    "INSTALL_HINT",
    "LIDAR_SUFFIX",
    "CameraMount",
    "LidarRecorder",
    "LidarRerunUnavailableError",
    "active_recorder",
    "lidar_recording_path",
    "merge_into",
    "require_sdk",
]

logger = logging.getLogger(__name__)

#: The converter's application id, so the viewer merges the two recordings.
APPLICATION_ID = "autoware_carla_scenario"
#: What a run's sensor recording is named: ``{stem}.sensors.rrd``.
SENSORS_SUFFIX = ".sensors.rrd"
#: The name it had when it held the LiDAR alone.
LIDAR_SUFFIX = SENSORS_SUFFIX
#: How to install what this module needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[rerun-lidar]'"

#: Heights (m, above the ego's rig) the colours run over, blue to red.
_COLOUR_LOW_M = -2.5
_COLOUR_HIGH_M = 4.0

_lock = threading.Lock()
_active: Optional["SensorRecorder"] = None


class LidarRerunUnavailableError(RuntimeError):
    """``rerun.lidar`` or ``rerun.cameras`` is set, but the Rerun SDK is not installed."""


def require_sdk() -> Any:
    """Return the ``rerun`` module, or raise :class:`LidarRerunUnavailableError`."""
    try:
        import rerun  # noqa: PLC0415
    except ImportError as exc:
        raise LidarRerunUnavailableError(
            "rerun.lidar or rerun.cameras is set, but the Rerun SDK, which "
            f"records the sensors, is not installed: {INSTALL_HINT} (or set "
            "rerun.lidar=false rerun.cameras=false)"
        ) from exc
    return rerun


def lidar_recording_path(directory: Path, stem: str) -> Path:
    """``{directory}/{stem}.sensors.rrd``."""
    return Path(directory) / f"{stem}{SENSORS_SUFFIX}"


def merge_into(recording: Path, lidar: Path) -> Path:
    """Merge the LiDAR recording *lidar* into the run's *recording*, in place,
    with the Rerun SDK's CLI (``rerun rrd merge``), and remove *lidar*.

    Raises:
        LidarRerunUnavailableError: The SDK's CLI is not installed.
        subprocess.CalledProcessError: The merge failed; both files are kept.
    """
    spec = importlib.util.find_spec("rerun_cli")
    if spec is None or spec.origin is None:
        raise LidarRerunUnavailableError(
            f"merging the sensor recording needs the Rerun SDK's CLI: {INSTALL_HINT}"
        )
    cli = Path(spec.origin).parent / "rerun"
    recording, lidar = Path(recording), Path(lidar)
    merged = recording.with_name(f".{recording.name}.merging")
    try:
        subprocess.run(
            [str(cli), "rrd", "merge", str(recording), str(lidar), "-o", str(merged)],
            check=True,
            capture_output=True,
        )
        os.replace(merged, recording)
    finally:
        merged.unlink(missing_ok=True)
    lidar.unlink()
    return recording


def active_recorder() -> Optional["SensorRecorder"]:
    """The recorder of the run in progress, if it records its sensors."""
    return _active


class SensorRecorder:
    """Writes one run's sensors into ``{stem}.sensors.rrd``.

    Args:
        path: The recording to write.
        recording_id: The run's file stem, the converter's recording id.
        start_frame: The CARLA frame the scenario clock starts at
            (``sim_time`` 0); data before it, the warm-up's, is not written.
        max_points: Keep at most this many points of a sweep (an even
            subsample); 0 keeps them all.
        sdk: The ``rerun`` module; imported when ``None``.
        lidar: Whether the LiDAR sweeps are recorded (``rerun.lidar``).
        cameras: Whether the camera frames are recorded (``rerun.cameras``).
    """

    def __init__(
        self,
        path: Path,
        recording_id: str,
        start_frame: int,
        max_points: int = 0,
        sdk: Any = None,
        *,
        lidar: bool = True,
        cameras: bool = False,
    ) -> None:
        if max_points < 0:
            raise ValueError("rerun.lidar_max_points must not be negative")
        self._rr = sdk if sdk is not None else require_sdk()
        self.path = Path(path)
        self.recording_id = recording_id
        self.start_frame = int(start_frame)
        self.max_points = int(max_points)
        self.records_lidar = bool(lidar)
        self.records_cameras = bool(cameras)
        self.sweeps = 0
        self.frames = 0
        self._mounted: set[str] = set()
        self._stream: Any = self._rr.RecordingStream(
            APPLICATION_ID, recording_id=recording_id
        )
        self._stream.save(str(self.path))

    def start(self) -> "SensorRecorder":
        """Make this the recorder :func:`active_recorder` returns."""
        global _active
        with _lock:
            _active = self
        return self

    def log_sweep(
        self,
        lidar_id: str,
        frame: int,
        points_in_rig: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        rig_rotation: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        rig_position: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> bool:
        """Write one sweep, ``[N, >=3]`` in the rig frame, at *frame*.

        *rig_rotation* (3x3) and *rig_position* place the rig in the world
        frame.  Returns whether it was written (not a warm-up sweep).
        """
        if frame < self.start_frame:
            return False
        rig = np.asarray(points_in_rig, dtype=np.float32)[:, :3]
        if self.max_points and len(rig) > self.max_points:
            rig = rig[np.linspace(0, len(rig) - 1, self.max_points).astype(np.int64)]
        world = rig @ np.asarray(rig_rotation, dtype=np.float32).T
        world += np.asarray(rig_position, dtype=np.float32)
        rr = self._rr
        self._at(frame)
        self._stream.log(
            f"world/lidar/{lidar_id}",
            rr.Points3D(world, colors=_height_colours(rig[:, 2]), radii=0.04),
        )
        self.sweeps += 1
        return True

    def log_camera_frame(
        self,
        camera_id: str,
        frame: int,
        jpeg: bytes,
        mount: "CameraMount",
    ) -> bool:
        """Write one JPEG frame of a camera at *frame*; its mount and pinhole
        the first time.  Returns whether it was written (not a warm-up frame).
        """
        if frame < self.start_frame:
            return False
        rr = self._rr
        entity = f"world/ego/pose/cameras/{camera_id}"
        if camera_id not in self._mounted:
            self._mounted.add(camera_id)
            self._stream.log(
                entity,
                rr.Transform3D(
                    translation=mount.translation(), mat3x3=mount.rotation()
                ),
                rr.Pinhole(
                    focal_length=mount.focal_length_px(),
                    width=mount.width,
                    height=mount.height,
                    camera_xyz=rr.ViewCoordinates.FLU,
                    image_plane_distance=2.0,
                ),
                static=True,
            )
        self._at(frame)
        self._stream.log(
            entity, rr.EncodedImage(contents=bytes(jpeg), media_type="image/jpeg")
        )
        self.frames += 1
        return True

    def _at(self, frame: int) -> None:
        self._stream.set_time("frame", sequence=int(frame))
        self._stream.set_time(
            "sim_time", duration=(frame - self.start_frame) * FIXED_DELTA_SECONDS
        )

    def close(self) -> Path:
        """Flush and close the recording; no longer the active one.  Its path."""
        global _active
        with _lock:
            if _active is self:
                _active = None
        try:
            self._stream.flush()
        finally:
            self._stream.disconnect()
        logger.info(
            "Sensor recording: %d LiDAR sweep(s) and %d camera frame(s) written to %s",
            self.sweeps,
            self.frames,
            self.path,
        )
        return self.path


#: The name it had when it recorded the LiDAR alone.
LidarRecorder = SensorRecorder


class CameraMount:
    """A camera's mount on the ego, as ``driver.cameras`` gives it (CARLA's
    convention: x forward, y right, z up, degrees), and its image."""

    def __init__(
        self,
        width: int,
        height: int,
        fov: float,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov = float(fov)
        self.position = position
        self.roll, self.pitch, self.yaw = rotation_deg

    def translation(self) -> list[float]:
        """In the recording's right-handed frame: CARLA's ``y`` negated."""
        x, y, z = self.position
        return [float(x), -float(y), float(z)]

    def rotation(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """CARLA's ``yaw * pitch * roll``, rebuilt right-handed: yaw and pitch
        negated, roll kept (as the driver's camera extrinsics)."""
        yaw, pitch, roll = (
            np.radians(-self.yaw),
            np.radians(-self.pitch),
            np.radians(self.roll),
        )
        cz, sz, cy, sy, cx, sx = (
            np.cos(yaw),
            np.sin(yaw),
            np.cos(pitch),
            np.sin(pitch),
            np.cos(roll),
            np.sin(roll),
        )
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        return rz @ ry @ rx

    def focal_length_px(self) -> float:
        """CARLA's horizontal field of view as a focal length in pixels."""
        return self.width / (2.0 * float(np.tan(np.radians(self.fov) / 2.0)))


def _height_colours(
    heights: np.ndarray[tuple[int], np.dtype[np.float32]],
) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
    """Blue (low) through green to red (high), over the ego's surroundings."""
    t = np.clip((heights - _COLOUR_LOW_M) / (_COLOUR_HIGH_M - _COLOUR_LOW_M), 0.0, 1.0)
    colours = np.empty((len(t), 3), dtype=np.uint8)
    colours[:, 0] = (255 * np.clip(2.0 * t - 1.0, 0.0, 1.0)).astype(np.uint8)
    colours[:, 1] = (255 * (1.0 - np.abs(2.0 * t - 1.0))).astype(np.uint8)
    colours[:, 2] = (255 * np.clip(1.0 - 2.0 * t, 0.0, 1.0)).astype(np.uint8)
    return colours
