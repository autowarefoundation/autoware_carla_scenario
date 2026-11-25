"""Abstract base class and configuration dataclass for camera sensors.

This module defines the simulator-agnostic camera interface.  Concrete
implementations (e.g. :class:`CarlaCameraSensor`) subclass
:class:`CameraSensorBase` and extend :class:`CameraSensorConfig` with
simulator-specific attributes.
"""

from __future__ import annotations

import math
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np

from ..constants import FIXED_DELTA_SECONDS

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

#: How far below its interval a sensor's ``sensor_tick`` is set.  CARLA sums the
#: ticks' float deltas and fires the sensor once the sum reaches ``sensor_tick``;
#: two 0.05 s ticks sum to a hair under 0.1 s, so a ``sensor_tick`` of exactly 0.1
#: fires one tick late every time it is reached.  What is left over carries to
#: the next firing, so the margin only adds up, a tick in every
#: ``interval / margin`` firings -- one a tick early, which a reader taking the
#: newest frame it has takes in its stride.
SENSOR_TICK_MARGIN_S: float = 1e-6


def sensor_tick_s(interval_s: float) -> float:
    """The ``sensor_tick`` for a sensor meant to fire every *interval_s* seconds.

    :data:`SENSOR_TICK_MARGIN_S` below *interval_s* when that is longer than
    one tick; *interval_s* itself when the sensor fires on every tick anyway.
    """
    if interval_s > FIXED_DELTA_SECONDS + 1e-9:
        return interval_s - SENSOR_TICK_MARGIN_S
    return interval_s


def focal_length_px(width: float, fov_deg: float) -> float:
    """The focal length, in pixels, of a pinhole camera *width* pixels wide
    whose horizontal field of view is *fov_deg* degrees: CARLA's camera, whose
    pixels are square, so it is the vertical one too.

    The one formula behind :attr:`CameraSensorConfig.fx` (and so the T4
    dataset's camera intrinsics) and the Rerun recording's
    :meth:`~autoware_carla_scenario.rerun_sensor_mount.CameraMount.focal_length_px`.
    """
    return width / (2.0 * math.tan(math.radians(fov_deg / 2.0)))


def sensor_interval_ticks(interval_s: float) -> int:
    """The ticks a sensor set to fire every *interval_s* seconds waits at most
    from one firing to the next (at least 1): its :func:`sensor_tick_s`, rounded
    up to whole ticks."""
    ticks = math.ceil(sensor_tick_s(interval_s) / FIXED_DELTA_SECONDS - 1e-9)
    return max(1, int(ticks))


def frame_ticks(fps: float) -> tuple[int, int]:
    """The fewest and the most ticks from one frame to the next of a camera
    set to *fps* (positive): a sensor fires at most once a tick, on a tick.

    ``(1, 1)`` from one frame a tick up; ``(n, n)`` when ``1 / fps`` is *n*
    whole ticks; otherwise the two whole numbers of ticks around it, which
    the frames alternate between -- on average *fps*, with a tick of jitter
    (#2499).
    """
    period = 1.0 / (fps * FIXED_DELTA_SECONDS)
    if period <= 1.0 + 1e-9:
        return (1, 1)
    nearest = round(period)
    if abs(period - nearest) <= 1e-9:
        return (int(nearest), int(nearest))
    return (int(math.floor(period)), int(math.ceil(period)))


def frame_rate_text(fps: float) -> str:
    """The rate a camera set to *fps* delivers frames at, as the tick allows:
    ``"10 fps"``, ``"20 fps (30 fps set; one frame a tick)"``, or
    ``"15 fps on average (a frame every 1-2 ticks)"`` (:func:`frame_ticks`)."""
    fewest, most = frame_ticks(fps)
    if fewest != most:
        return f"{fps:g} fps on average (a frame every {fewest}-{most} ticks)"
    effective = 1.0 / (fewest * FIXED_DELTA_SECONDS)
    if abs(effective - fps) <= 1e-6 * effective:
        return f"{effective:g} fps"
    return f"{effective:g} fps ({fps:g} fps set; one frame a tick)"


#: What a sensor's queue holds: a camera's frames, a LiDAR's sweeps.
_Item = TypeVar("_Item")


def put_dropping_oldest(items: queue.Queue[_Item], item: _Item) -> None:
    """Queue *item* into the bounded *items*, dropping the oldest item queued
    while it is full: a sensor's listener, on CARLA's delivery thread, never
    blocks, and never drops the newest data."""
    while True:
        try:
            items.put_nowait(item)
            return
        except queue.Full:
            try:
                items.get_nowait()
            except queue.Empty:
                pass


def drain(items: queue.Queue[_Item]) -> None:
    """Throw away everything queued in *items*, never waiting: a sensor
    destroyed leaves none of its data for the one attached next (#2265)."""
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


@dataclass(frozen=True)
class CameraSensorConfig:
    """Simulator-agnostic camera configuration.

    Groups the parameters common to *any* pinhole camera sensor:

    * **Resolution** -- ``image_width`` and ``image_height`` in pixels.
    * **Field of view** -- horizontal ``fov`` in degrees.
    * **Frame rate** -- ``fps`` (frames per second).
    * **Extrinsics** -- 6-DOF mount ``(position_x, position_y, position_z)``
      in metres and ``(roll, pitch, yaw)`` in degrees, relative to the CARLA
      actor origin (the vehicle's centre, where CARLA attaches the sensor), in
      CARLA's convention (x forward, y right, z up, degrees), as a LiDAR's
      are.  This is not Autoware's ``base_link`` (the rear axle, y left): the
      values go to ``carla.Location`` and ``carla.Rotation`` unchanged.  The
      server mounts the sensor with them as Unreal's ``FRotator``, whose signs
      the angle fields below give.

    Intrinsic matrix entries (``fx``, ``fy``, ``cx``, ``cy``) are derived
    automatically from resolution and FOV via :attr:`intrinsic_matrix`.
    """

    # -- Resolution -----------------------------------------------------------
    image_width: int = 1920
    """Image width in pixels."""

    image_height: int = 1080
    """Image height in pixels."""

    # -- Optics ---------------------------------------------------------------
    fov: float = 90.0
    """Horizontal field of view in degrees."""

    # -- Timing ---------------------------------------------------------------
    fps: float = 20.0
    """Sensor capture rate in frames per second."""

    # -- Extrinsics (CARLA actor origin -> camera, CARLA convention) ---------
    position_x: float = 0.0
    """Forward offset from the CARLA actor origin in metres (positive = forward)."""

    position_y: float = 0.0
    """Lateral offset from the CARLA actor origin in metres (positive = right)."""

    position_z: float = 0.0
    """Vertical offset from the CARLA actor origin in metres (positive = up)."""

    roll: float = 0.0
    """CARLA roll about the forward (x) axis in degrees (positive = right side down)."""

    pitch: float = 0.0
    """CARLA pitch about the lateral (y) axis in degrees (negative = look down)."""

    yaw: float = 0.0
    """CARLA yaw about the vertical (z) axis in degrees (positive = turn right)."""

    # -- Derived intrinsic helpers -------------------------------------------

    @property
    def fx(self) -> float:
        """Horizontal focal length in pixels derived from FOV and width
        (:func:`focal_length_px`)."""
        return focal_length_px(self.image_width, self.fov)

    @property
    def fy(self) -> float:
        """Vertical focal length in pixels (square-pixel assumption)."""
        return self.fx

    @property
    def cx(self) -> float:
        """Principal point x-coordinate (image centre)."""
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        """Principal point y-coordinate (image centre)."""
        return self.image_height / 2.0

    @property
    def intrinsic_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """3x3 pinhole camera intrinsic matrix (K).

        .. math::

            K = \\begin{bmatrix}
                f_x & 0   & c_x \\\\
                0   & f_y & c_y \\\\
                0   & 0   & 1
            \\end{bmatrix}
        """
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


class CameraSensorBase(ABC):
    """Abstract base for camera sensors attached to a CARLA actor.

    Subclasses must implement :meth:`attach` and :meth:`destroy`.

    Args:
        config: Camera configuration dataclass.
    """

    _config: CameraSensorConfig
    _attached: bool

    def __init__(self, config: CameraSensorConfig) -> None:
        self._config = config
        self._attached = False

    @property
    def config(self) -> CameraSensorConfig:
        """Return the camera configuration."""
        return self._config

    @property
    def attached(self) -> bool:
        """Whether the sensor is currently attached to an actor."""
        return self._attached

    @abstractmethod
    def attach(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Attach the sensor to *actor* in *world*.

        Args:
            world: The CARLA world instance.
            actor: The actor to attach the sensor to.
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """Destroy the sensor actor and release resources."""
        ...

    @abstractmethod
    def get_image(
        self,
    ) -> Optional[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]]:
        """Return the latest captured frame as an HxWx3 BGR array, or ``None``.

        Returns:
            A NumPy array of shape ``(height, width, 3)`` in BGR order, or
            ``None`` if no frame is available yet.
        """
        ...
