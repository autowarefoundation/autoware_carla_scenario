"""CARLA ray-cast LiDAR: one full revolution per sweep, a sweep every so many ticks.

Sweeps arrive asynchronously, like camera frames, into a queue the caller drains
once per policy step; :meth:`CarlaLidarSensor.latest_sweep` hands back the newest
raw ``[N, 4]`` buffer in the sensor's own (left-handed) frame, with the frame it
was taken on, and the conversion into the rig frame is left to whoever sends it
-- only a sweep actually sent is worth converting.

A sweep costs the server one ray per point, every tick it is taken on: at a
million points a second that is most of what a tick costs.  A LiDAR read once a
policy step is therefore swept once a policy step (``sweep_interval_s``), not on
the ticks between, whose sweeps nobody would read.

The sweep a caller is handed carries the world's rain and fog as they are when
it is taken (:mod:`.lidar_weather`).
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from ..constants import FIXED_DELTA_SECONDS, tick_multiple
from ..seed import resolve_seed
from .base import sensor_tick_s
from .lidar_weather import CARLA_FOG_DENSITY_SCALE, LidarWeather, apply_weather

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["CarlaLidarSensor", "CarlaLidarSensorConfig"]

#: How long :meth:`CarlaLidarSensor.latest_sweep` waits for a sweep that is due, in
#: seconds.
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
    #: How much of the world's rain and fog the sweeps carry
    #: (:mod:`~autoware_carla_scenario.sensor.lidar_weather`): 0 none, as CARLA
    #: casts them; 1 the model as tuned; more exaggerates it.
    weather_noise: float = 1.0
    #: Seed of the weather noise, so a run repeats its sweeps; ``None`` takes
    #: the run's seed (:mod:`~autoware_carla_scenario.seed`) when it attaches.
    noise_seed: Optional[int] = None
    #: The server's ``carla.Weather.FogDensityScale``, which the LiDAR's fog
    #: follows.  A scenario run sets it from ``server.fog_density_scale``;
    #: built by hand, match it to the server's.
    fog_density_scale: float = CARLA_FOG_DENSITY_SCALE
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
        if self.weather_noise < 0.0 or (
            self.noise_seed is not None and self.noise_seed < 0
        ):
            raise ValueError("LiDAR weather_noise and noise_seed must not be negative")
        if self.fog_density_scale <= 0.0:
            raise ValueError("LiDAR fog_density_scale must be positive")


class CarlaLidarSensor:
    """A ray-cast LiDAR attached to a CARLA actor.

    Args:
        config: Blueprint attributes and mount.
        sweep_interval_s: Simulated seconds between sweeps, a positive whole
            number of ticks; one tick (the default) sweeps on every tick.  Each
            sweep is still one whole revolution, taken within its tick.

    Raises:
        ValueError: If *sweep_interval_s* is not a positive whole number of ticks.
    """

    _config: CarlaLidarSensorConfig
    _sensor: Optional["carla.Actor"]
    #: ``(frame, raw [N, 4] float32)``, oldest first.  Unbounded: the listen
    #: callback must never block, and :meth:`latest_sweep` drains it every call.
    _sweeps: queue.Queue[Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]]
    _interval_ticks: int
    #: The newest sweep received, kept until a newer one arrives.
    _newest: Optional[Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]]
    #: The world whose weather the sweeps are degraded by, once attached.
    _world: Optional["carla.World"]
    #: The frame of the sweep the weather was last put on (in :attr:`_newest`).
    _weathered_frame: int
    #: The weather noise's seed: the config's, or the run's when it attaches.
    _noise_seed: int

    def __init__(
        self,
        config: CarlaLidarSensorConfig,
        sweep_interval_s: float = FIXED_DELTA_SECONDS,
    ) -> None:
        ticks = tick_multiple(sweep_interval_s, "LiDAR sweep_interval_s")
        self._config = config
        self._sensor = None
        self._sweeps = queue.Queue()
        self._interval_ticks = ticks
        self._newest = None
        self._world = None
        self._weathered_frame = -1
        self._noise_seed = resolve_seed(config.noise_seed)

    @property
    def sweep_interval_ticks(self) -> int:
        """Ticks from one sweep to the next."""
        return self._interval_ticks

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
        self._noise_seed = resolve_seed(cfg.noise_seed)
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
        if self._interval_ticks > 1:
            interval_s = self._interval_ticks * FIXED_DELTA_SECONDS
            blueprint.set_attribute("sensor_tick", str(sensor_tick_s(interval_s)))
        transform = _carla.Transform(
            _carla.Location(x=cfg.position_x, y=cfg.position_y, z=cfg.position_z),
            _carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw),
        )
        self._sensor = world.spawn_actor(blueprint, transform, attach_to=actor)
        self._world = world
        self._sensor.listen(self._on_sensor_data)
        logger.info(
            "CarlaLidarSensor attached: %d channels, %.0f m, %d pts/s, "
            "a sweep every %d tick(s) -> %s",
            cfg.channels,
            cfg.range_m,
            cfg.points_per_second,
            self._interval_ticks,
            actor.type_id,
        )

    def _on_sensor_data(
        self, data: Union["carla.LidarMeasurement", "carla.SensorData"]
    ) -> None:
        """The sensor's listener, on a CARLA thread: hands the sweep to :meth:`_receive`.

        CARLA's Python API, and typesafe_carla's CPython package, call it with
        the ``carla.LidarMeasurement`` itself.  typesafe_carla's Codon library
        calls it with the ``carla.SensorData``, which ``as_lidar()`` turns into
        the measurement; the static check sees that branch, and Python, whose
        ``LidarMeasurement`` has no ``as_lidar``, never takes it.
        """
        if hasattr(data, "as_lidar"):
            self._receive(data.as_lidar())
        else:
            self._receive(data)

    def _receive(self, measurement: "carla.LidarMeasurement") -> None:
        """Queue the raw buffer; copying it is all the callback thread does."""
        try:
            if TYPE_CHECKING:
                # typesafe_carla's Codon raw_data is a view, and Codon's
                # frombuffer takes the bytes (a str).
                raw = np.frombuffer(measurement.raw_data.tobytes(), dtype=np.float32)
            else:
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
        self._world = None

    def latest_sweep(
        self, frame: int
    ) -> Optional[Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]]:
        """The newest sweep as of simulator frame *frame*: ``(its frame, raw [N, 4])``.

        The buffer is in the sensor frame, with the world's rain and fog on it
        (:meth:`_weather`).  A sweep is due on *frame* when the
        newest one is a whole interval old; then this waits up to
        :data:`_SWEEP_TIMEOUT` seconds for it.  Between sweeps the newest one is
        the answer, taken on an earlier tick of the interval.  ``None`` when
        nothing within the last interval arrived -- an older sweep would be a
        stale one, and the policy is better told nothing than told the past.
        """
        deadline = time.monotonic() + _SWEEP_TIMEOUT
        self._take_queued()
        while self._due(frame):
            remaining = deadline - time.monotonic()
            try:
                self._keep(
                    self._sweeps.get(timeout=remaining)
                    if remaining > 0.0
                    else self._sweeps.get_nowait()
                )
            except queue.Empty:
                logger.warning("no LiDAR sweep for frame %d in time; skipped", frame)
                break
            self._take_queued()
        newest = self._newest
        if newest is None or newest[0] <= frame - self._interval_ticks:
            return None
        if newest[0] != frame:
            logger.debug("LiDAR swept frame %d for frame %d", newest[0], frame)
        return self._weathered_sweep(newest)

    def get_sweep(
        self, frame: int
    ) -> Optional[np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """:meth:`latest_sweep`'s raw buffer alone."""
        sweep = self.latest_sweep(frame)
        if sweep is None:
            return None
        return sweep[1]

    def _weathered_sweep(
        self, sweep: Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]
    ) -> Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """*sweep* (:attr:`_newest`) with the weather on it, put on once."""
        if self._weathered_frame == sweep[0]:
            return sweep
        self._weathered_frame = sweep[0]
        weather = self._weather()
        if weather is None:
            return sweep
        result = (
            sweep[0],
            apply_weather(
                sweep[1],
                weather,
                self._config.weather_noise,
                self._noise_seed,
                sweep[0],
            ),
        )
        self._newest = result
        return result

    def _weather(self) -> Optional[LidarWeather]:
        """The world's rain and fog now; ``None`` when no noise is to be put on."""
        world = self._world
        if world is None or self._config.weather_noise <= 0.0:
            return None
        try:
            weather = world.get_weather()
        except Exception:  # noqa: BLE001 - a lost reading costs the noise, not the sweep
            logger.exception("failed to read the weather for the LiDAR noise")
            return None
        return LidarWeather(
            precipitation=float(weather.precipitation),
            fog_density=float(weather.fog_density),
            fog_distance=float(weather.fog_distance),
            fog_density_scale=self._config.fog_density_scale,
        )

    def _due(self, frame: int) -> bool:
        """Whether a sweep newer than the newest one is due by *frame*."""
        newest = self._newest
        if newest is None:
            return True
        return newest[0] + self._interval_ticks <= frame

    def _keep(
        self, sweep: Tuple[int, np.ndarray[tuple[int, int], np.dtype[np.float32]]]
    ) -> None:
        newest = self._newest
        if newest is None or sweep[0] >= newest[0]:
            self._newest = sweep

    def _take_queued(self) -> None:
        """Move every sweep already queued into :attr:`_newest`; never waits."""
        while True:
            try:
                sweep = self._sweeps.get_nowait()
            except queue.Empty:
                break
            self._keep(sweep)
