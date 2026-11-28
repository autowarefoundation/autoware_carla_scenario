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
policy got, carried into the world with the ego's full pose (pitch and roll
too) on the tick it was taken.

Each camera is ``world/ego/pose/cameras/<logical id>``: its mount on the ego,
a pinhole of its field of view and the JPEG frames the policy got, so the 3D
view draws its frustum with the frame in it, and a 2D view of the entity shows
the frames.  The converter writes ``world/ego/pose`` with the ego actor's
location and yaw alone, so ``world/ego/pose/cameras`` carries the rest of its
rotation, its pitch and roll, on every tick (:meth:`SensorRecorder.log_ego_tilt`).

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
from .rerun import RerunUnavailableError

# What a run's sensor recording is named, ``{stem}.sensors.rrd``, and where it
# is: the path rule is the checked run_outputs's (#1678); the names they had
# here stay importable from here.
from .run_outputs import SENSORS_SUFFIX as SENSORS_SUFFIX
from .run_outputs import sensors_recording_path as lidar_recording_path

# The scratch files a merge and an extraction write through, which
# run_outputs.remove_run_outputs removes when a killed run leaves one.
from .run_outputs import extracting_path, merging_path

# A camera's mount, the colours of a sweep, the sensors' entities and the
# parse of ``rerun rrd print`` are the checked rerun_sensor_mount's (#1688);
# the names they had here stay importable from here.
from .rerun_sensor_mount import _CHUNK_LINE as _CHUNK_LINE
from .rerun_sensor_mount import _COLOUR_HIGH_M as _COLOUR_HIGH_M
from .rerun_sensor_mount import _COLOUR_LOW_M as _COLOUR_LOW_M
from .rerun_sensor_mount import SENSOR_ENTITIES as SENSOR_ENTITIES
from .rerun_sensor_mount import CameraMount as CameraMount
from .rerun_sensor_mount import _height_colours as _height_colours
from .rerun_sensor_mount import _is_sensor_entity as _is_sensor_entity
from .rerun_sensor_mount import chunk_entities as chunk_entities

# The scan for the sensors' entity paths is the checked rerun_sensor_mount's
# too (#1697); the names it had here stay importable from here.
from .rerun_sensor_mount import _SCAN_BLOCK as _SCAN_BLOCK
from .rerun_sensor_mount import carries_sensors as carries_sensors

__all__ = [
    "APPLICATION_ID",
    "SENSORS_SUFFIX",
    "SensorRecorder",
    "INSTALL_HINT",
    "LIDAR_SUFFIX",
    "CameraMount",
    "LidarRecorder",
    "LidarRerunUnavailableError",
    "RerunCliError",
    "RerunCliTimeoutError",
    "SENSOR_ENTITIES",
    "active_recorder",
    "carries_sensors",
    "chunk_entities",
    "ego_tilt",
    "extract_sensors",
    "lidar_recording_path",
    "merge_into",
    "require_sdk",
]

logger = logging.getLogger(__name__)

#: The converter's application id, so the viewer merges the two recordings.
APPLICATION_ID = "autoware_carla_scenario"
#: The name it had when it held the LiDAR alone.
LIDAR_SUFFIX = SENSORS_SUFFIX
#: How to install what this module needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[rerun-lidar]'"
#: How long one call of the Rerun SDK's CLI may take: merging or filtering a
#: long run's recording reads and writes all of it, so this is generous; a
#: hung call must not block the converter's one worker, and with it
#: ``RerunConverter.close()``, forever (#2182).
CLI_TIMEOUT_S = 600.0
_lock = threading.Lock()
_active: Optional["SensorRecorder"] = None


class LidarRerunUnavailableError(RerunUnavailableError):
    """``rerun.lidar`` or ``rerun.cameras`` is set, but the Rerun SDK is not installed.

    A :class:`~autoware_carla_scenario.rerun.RerunUnavailableError` (and so a
    ``RuntimeError``): an extra the run asks for is missing, which the
    ``scenario`` CLI and the lanelet sweeper refuse with exit 2 (#2089).
    """


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


def _describe(cmd: Any, what: str, stderr: Any) -> str:
    """``rerun ...`` *what*, and what it wrote on stderr (decoded)."""
    argv = cmd if isinstance(cmd, (list, tuple)) else [cmd]
    message = f"`{' '.join(str(arg) for arg in argv)}` {what}"
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    stderr = (stderr or "").strip()
    return f"{message}: {stderr}" if stderr else f"{message} (no stderr)"


class RerunCliError(subprocess.CalledProcessError):
    """A call of the Rerun SDK's CLI failed: a ``CalledProcessError`` whose
    message names the command and holds what it wrote on stderr."""

    def __str__(self) -> str:
        return _describe(self.cmd, f"exited with status {self.returncode}", self.stderr)


class RerunCliTimeoutError(subprocess.TimeoutExpired):
    """A call of the Rerun SDK's CLI took longer than :data:`CLI_TIMEOUT_S`
    and was killed: a ``TimeoutExpired`` whose message names the command and
    holds what it wrote on stderr."""

    def __str__(self) -> str:
        return _describe(self.cmd, f"timed out after {self.timeout:g} s", self.stderr)


def _run_cli(argv: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """Run the Rerun SDK's CLI with *argv*, its output captured, for at most
    :data:`CLI_TIMEOUT_S` (read here, so that it stays patchable).

    Raises:
        RerunCliError: It exited non-zero.
        RerunCliTimeoutError: It took too long (and was killed).
    """
    try:
        return subprocess.run(
            argv, check=True, capture_output=True, timeout=CLI_TIMEOUT_S, **kwargs
        )
    except subprocess.CalledProcessError as error:
        raise RerunCliError(
            error.returncode, error.cmd, error.output, error.stderr
        ) from None
    except subprocess.TimeoutExpired as error:
        raise RerunCliTimeoutError(
            error.cmd, error.timeout, error.output, error.stderr
        ) from None


def _cli() -> Path:
    """The Rerun SDK's CLI (``rerun``), or :class:`LidarRerunUnavailableError`."""
    spec = importlib.util.find_spec("rerun_cli")
    if spec is None or spec.origin is None:
        raise LidarRerunUnavailableError(
            f"merging or keeping sensor recordings needs the Rerun SDK's CLI: {INSTALL_HINT}"
        )
    return Path(spec.origin).parent / "rerun"


def merge_into(recording: Path, lidar: Path) -> Path:
    """Merge the LiDAR recording *lidar* into the run's *recording*, in place,
    with the Rerun SDK's CLI (``rerun rrd merge``), and remove *lidar*.

    Raises:
        LidarRerunUnavailableError: The SDK's CLI is not installed.
        RerunCliError: The merge failed (a ``CalledProcessError``, its
            message the command and its stderr); both files are kept.
        RerunCliTimeoutError: The merge took longer than
            :data:`CLI_TIMEOUT_S` (a ``TimeoutExpired``); both files are kept.
    """
    cli = _cli()
    recording, lidar = Path(recording), Path(lidar)
    merged = merging_path(recording)
    try:
        _run_cli(
            [str(cli), "rrd", "merge", str(recording), str(lidar), "-o", str(merged)]
        )
        os.replace(merged, recording)
    finally:
        merged.unlink(missing_ok=True)
    lidar.unlink()
    return recording


def extract_sensors(recording: Path, sensors: Path) -> bool:
    """Write the sensor entities of *recording* (:data:`SENSOR_ENTITIES`) into
    *sensors*, a recording :func:`merge_into` can merge into another, with the
    Rerun SDK's CLI (``rerun rrd print`` lists the entities, ``rerun rrd
    filter`` drops the others).  *recording* is not changed.

    Returns whether it holds any; when it does not, *sensors* is not written.

    Raises:
        LidarRerunUnavailableError: The SDK's CLI is not installed.
        RerunCliError: Reading or writing a recording failed (a
            ``CalledProcessError``, its message the command and its stderr);
            *sensors* is not written.
        RerunCliTimeoutError: Reading or writing it took longer than
            :data:`CLI_TIMEOUT_S` (a ``TimeoutExpired``); *sensors* is not
            written.
        RuntimeError: ``rerun rrd print`` lists the chunks in a form this
            does not read (or lists none); *sensors* is not written.
    """
    cli = _cli()
    recording, sensors = Path(recording), Path(sensors)
    printed = _run_cli(
        [str(cli), "rrd", "print", str(recording)],
        encoding="utf-8",
        errors="replace",
    ).stdout
    # _CHUNK_LINE is read here, so that it stays patchable on this module.
    read, unread = chunk_entities(printed, _CHUNK_LINE)
    if unread or not read:
        # Another SDK's output: dropping too little would duplicate data on
        # a merge, and finding no sensors would lose them.
        raise RuntimeError(
            f"cannot read the entities of {recording} from `rerun rrd print`"
            + (f": {unread!r}" if unread else ": no chunks")
        )
    entities = set(read)
    if not any(_is_sensor_entity(entity) for entity in entities):
        return False
    dropped = sorted(entity for entity in entities if not _is_sensor_entity(entity))
    partial = extracting_path(sensors)
    try:
        _run_cli(
            [
                str(cli),
                "rrd",
                "filter",
                *(arg for entity in dropped for arg in ("--drop-entity", entity)),
                str(recording),
                "-o",
                str(partial),
            ]
        )
        os.replace(partial, sensors)
    finally:
        partial.unlink(missing_ok=True)
    return True


def ego_tilt(
    roll_deg: float, pitch_deg: float
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """The ego's rotation without its yaw, in the recording's right-handed
    frame, from CARLA's roll and pitch (degrees): the full rotation is the
    yaw-only one (``world/ego/pose``) times this.  CARLA's ``yaw * pitch *
    roll`` rebuilt right-handed is ``Rz(-yaw) Ry(-pitch) Rx(roll)``, as a
    camera's mount (:meth:`CameraMount.rotation`)."""
    return CameraMount(0, 0, 0.0, rotation_deg=(roll_deg, pitch_deg, 0.0)).rotation()


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
        try:
            # The converter's chunks are on `frame` and `sim_time` alone: the
            # SDK's wall-clock `log_time` would put the sensors alone on a
            # timeline of their own.
            self._stream.set_log_time_enabled(False)
            self._stream.save(str(self.path))
        except BaseException:
            # A recorder that cannot start leaves no stream running (#2485,
            # #2617): nothing else would ever close it.
            stream, self._stream = self._stream, None
            try:
                stream.disconnect()
            finally:
                del stream
                _cleanup_orphaned_recordings(self._rr)
            raise

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

        A camera whose image is no pinhole (an F-theta lens,
        :attr:`CameraMount.pinhole`) gets its mount alone: Rerun has no model
        of such a lens, and a pinhole of its ``fov`` would draw its frustum
        and project its image wrong.
        """
        if frame < self.start_frame:
            return False
        rr = self._rr
        entity = f"world/ego/pose/cameras/{camera_id}"
        if camera_id not in self._mounted:
            self._mounted.add(camera_id)
            components: list[Any] = [
                rr.Transform3D(translation=mount.translation(), mat3x3=mount.rotation())
            ]
            if mount.pinhole:
                components.append(
                    rr.Pinhole(
                        focal_length=mount.focal_length_px(),
                        width=mount.width,
                        height=mount.height,
                        camera_xyz=rr.ViewCoordinates.FLU,
                        image_plane_distance=2.0,
                    )
                )
            self._stream.log(entity, *components, static=True)
        self._at(frame)
        self._stream.log(
            entity, rr.EncodedImage(contents=bytes(jpeg), media_type="image/jpeg")
        )
        self.frames += 1
        return True

    def log_ego_tilt(self, frame: int, roll_deg: float, pitch_deg: float) -> bool:
        """Write the ego's pitch and roll on *frame* (CARLA's, degrees) as the
        transform of ``world/ego/pose/cameras``, the cameras' parent, so they
        are drawn with the ego's full rotation: the converter's
        ``world/ego/pose`` has its yaw alone.  Only when the cameras are
        recorded; returns whether it was written (not a warm-up tick)."""
        if not self.records_cameras or frame < self.start_frame:
            return False
        self._at(frame)
        self._stream.log(
            "world/ego/pose/cameras",
            self._rr.Transform3D(mat3x3=ego_tilt(roll_deg, pitch_deg)),
        )
        return True

    def _at(self, frame: int) -> None:
        self._stream.set_time("frame", sequence=int(frame))
        self._stream.set_time(
            "sim_time", duration=(frame - self.start_frame) * FIXED_DELTA_SECONDS
        )

    def close(self) -> Path:
        """Flush and close the recording; no longer the active one.  Its path.

        The stream is dropped and the SDK told to clean up the recordings no
        longer referenced: Rerun 0.38 never joins a closed stream's threads
        otherwise, two of them a run for the life of the process (#2485).
        """
        global _active
        with _lock:
            if _active is self:
                _active = None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.flush()
            finally:
                stream.disconnect()
                del stream
                _cleanup_orphaned_recordings(self._rr)
        logger.info(
            "Sensor recording: %d LiDAR sweep(s) and %d camera frame(s) written to %s",
            self.sweeps,
            self.frames,
            self.path,
        )
        return self.path


#: One sweep at a time: each puts back the global recordings it found.
_cleanup_lock = threading.Lock()


def _cleanup_orphaned_recordings(sdk: Any) -> None:
    """End the threads of the recordings nothing references any more, with
    the SDK's ``flush_and_cleanup_orphaned_recordings`` (Rerun 0.38); nothing
    with an SDK without it.

    The sweep also unsets the process's global recordings (``rr.init``'s), as
    ``rr.init`` wants it to: a policy or scenario logging to its own global
    recording in the same process would lose everything it logged after the
    run's sensors were closed.  They are put back.  The other recordings
    (the converter's, another run's sensors, a thread's own) are only
    flushed: something still references them."""
    bindings = getattr(sdk, "bindings", None)
    cleanup = getattr(bindings, "flush_and_cleanup_orphaned_recordings", None)
    if cleanup is None:
        return
    slots: list[tuple[Any, Any]] = [
        (
            getattr(bindings, f"get_global_{kind}_recording", None),
            getattr(bindings, f"set_global_{kind}_recording", None),
        )
        for kind in ("data", "blueprint")
    ]
    slots = [(get, put) for get, put in slots if get is not None and put is not None]
    with _cleanup_lock:
        held = [get() for get, _ in slots]
        try:
            cleanup()
        finally:
            for (get, put), recording in zip(slots, held):
                if recording is not None and get() is None:
                    put(recording)


#: The name it had when it recorded the LiDAR alone.
LidarRecorder = SensorRecorder
