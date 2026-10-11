"""CARLA ray-cast LiDAR, one full sweep per simulation tick.

Sweeps arrive asynchronously, like camera frames, into a queue the caller drains
once per policy step; :meth:`CarlaLidarSensor.get_sweep` hands back the raw
``[N, 4]`` buffer in the sensor's own (left-handed) frame, and the conversion into
the rig frame is left to whoever sends it -- only a sweep actually sent is worth
converting.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..constants import FIXED_DELTA_SECONDS

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["CarlaLidarSensor", "CarlaLidarSensorConfig"]

#: How long :meth:`CarlaLidarSensor.get_sweep` waits for the sweep of the frame asked
#: for, in seconds.
_SWEEP_TIMEOUT: float = 1.0


@dataclass(frozen=True)
class CarlaLidarSensorConfig:
    """Blueprint attributes and mount of one ``sensor.lidar.ray_cast``.

    The mount is relative to the vehicle actor, in CARLA's convention (x forward,
    y right, z up, degrees), as for a camera.  The rotation frequency is not
    configurable: it is pinned to one revolution per simulation tick, so every
    sweep covers the full 360 degrees.  Left at CARLA's default, a sweep would be
    whatever arc the beam covered during one tick, and a policy trained on whole
    sweeps would see a pie slice.  Each sweep therefore holds about
    ``points_per_second * FIXED_DELTA_SECONDS`` points.
    """

    channels: int = 64
    range_m: float = 100.0
    points_per_second: int = 1_200_000
    upper_fov_deg: float = 15.0
    lower_fov_deg: float = -25.0
    #: CARLA drops this fraction of points at random by default (0.45); a policy
    #: expecting a real sensor's density wants none of that.
    dropoff_general_rate: float = 0.0
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 2.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if self.channels < 1:
            raise ValueError("LiDAR channels must be positive")
        if self.range_m <= 0.0 or self.points_per_second <= 0:
            raise ValueError("LiDAR range_m and points_per_second must be positive")
        if self.lower_fov_deg >= self.upper_fov_deg:
            raise ValueError("LiDAR lower_fov_deg must be below upper_fov_deg")


class CarlaLidarSensor:
    """A ray-cast LiDAR attached to a CARLA actor.

    Args:
        config: Blueprint attributes and mount.
    """

    def __init__(self, config: CarlaLidarSensorConfig) -> None:
        self._config = config
        self._sensor: Optional["carla.Actor"] = None
        #: ``(frame, raw [N, 4] float32)``, oldest first.  Unbounded: the listen
        #: callback must never block, and :meth:`get_sweep` drains it every call.
        self._sweeps: "queue.Queue[Tuple[int, NDArray[np.float32]]]" = queue.Queue()

    @property
    def config(self) -> CarlaLidarSensorConfig:
        """Return the sensor configuration."""
        return self._config

    def attach(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Spawn the LiDAR and attach it to *actor*.

        Raises:
            RuntimeError: If the sensor is already attached.
        """
        if self._sensor is not None:
            raise RuntimeError("CarlaLidarSensor is already attached")

        import typesafe_carla.carla as _carla  # noqa: PLC0415

        cfg = self._config
        blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        blueprint.set_attribute("channels", str(cfg.channels))
        blueprint.set_attribute("range", str(cfg.range_m))
        blueprint.set_attribute("points_per_second", str(cfg.points_per_second))
        blueprint.set_attribute("rotation_frequency", str(1.0 / FIXED_DELTA_SECONDS))
        blueprint.set_attribute("upper_fov", str(cfg.upper_fov_deg))
        blueprint.set_attribute("lower_fov", str(cfg.lower_fov_deg))
        if blueprint.has_attribute("dropoff_general_rate"):
            blueprint.set_attribute(
                "dropoff_general_rate", str(cfg.dropoff_general_rate)
            )
        transform = _carla.Transform(
            _carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            _carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )
        self._sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
        self._sensor.listen(self._receive)
        logger.info(
            "CarlaLidarSensor attached: %d channels, %.0f m, %d pts/s -> %s",
            cfg.channels,
            cfg.range_m,
            cfg.points_per_second,
            actor.type_id,
        )

    def _receive(self, measurement: Any) -> None:
        """Queue the raw buffer; copying it is all the callback thread does."""
        try:
            raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
            self._sweeps.put((int(measurement.frame), raw.reshape(-1, 4).copy()))
        except Exception:  # noqa: BLE001 - a sensor thread must not die
            logger.exception("failed to receive a LiDAR sweep")

    def destroy(self) -> None:
        """Stop and destroy the CARLA sensor actor."""
        if self._sensor is not None:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None

    def get_sweep(self, frame: int) -> Optional[NDArray[np.float32]]:
        """The sweep of simulator frame *frame*, raw ``[N, 4]`` in the sensor frame.

        Waits up to :data:`_SWEEP_TIMEOUT` seconds for it; older sweeps queued meanwhile
        are dropped.  ``None`` when it does not arrive in time -- an older sweep would
        be a stale one, and the policy is better told nothing than told the past.
        """
        deadline = time.monotonic() + _SWEEP_TIMEOUT
        newest: Optional[Tuple[int, NDArray[np.float32]]] = None
        while newest is None or newest[0] < frame:
            remaining = deadline - time.monotonic()
            try:
                newest = (
                    self._sweeps.get(timeout=remaining)
                    if remaining > 0.0
                    else self._sweeps.get_nowait()
                )
            except queue.Empty:
                logger.warning("no LiDAR sweep for frame %d in time; skipped", frame)
                return None
        while True:  # a later frame already queued wins
            try:
                newest = self._sweeps.get_nowait()
            except queue.Empty:
                break
        if newest[0] != frame:
            logger.debug("LiDAR delivered frame %d for frame %d", newest[0], frame)
        return newest[1]
