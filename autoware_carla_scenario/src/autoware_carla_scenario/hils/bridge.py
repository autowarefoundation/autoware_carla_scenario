"""The HILS bridge: follows a run's ego in CARLA and streams it to devices under test.

It is a CARLA client of its own, beside the scenario runner, on the HILS PC
next to the ECUs.  It never ticks the world: it waits for the runner's ticks
(``World.wait_for_tick``), so it works in synchronous mode as well, and it
survives the runner's episodes -- when the ego goes away it waits for the
next one and attaches its sensors to that.

Each tick:

1. The ego's state (:class:`~.signals.EgoState`, from the world snapshot, the
   map's geo-reference and the vehicle's control) becomes the signals the
   message devices' periodic messages are updated with (``Periodic.update``).
2. What the sensor devices' sensors measured since the last tick is sent:
   LiDAR revolutions through ``lidar:MODEL`` (the sensor's UDP packets),
   camera frames (``v4l2://`` and friends).
3. With ``clock: simulation``, the shared ``hilspy.Clock`` advances by the
   simulated time that passed, which sends the periodic messages and LiDAR
   packets that fall due within it.

The sensors are CARLA's, or carla_canvas's when the configuration names a
``canvas`` file: :func:`~autoware_carla_scenario.canvas.use_canvas` points
typesafe_carla's sensor plugin at it before anything listens, and its
asynchronous rendering (``renderer.async``, carla_canvas's default) follows
the runner's ticks like this client does.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..canvas import use_canvas
from ..entity_role import find_actor_in_world
from .config import DeviceSpec, HilsConfig, MessageSpec, SensorSpec
from .measurements import (
    LidarRevolution,
    camera_frame,
    lidar_points,
    lidar_scan_fields,
)
from .signals import EgoState, ego_signals

if TYPE_CHECKING:
    import hilspy
    import typesafe_carla.carla as carla

__all__ = ["INSTALL_HINT", "HilsBridge", "HilsUnavailableError"]

logger = logging.getLogger(__name__)

#: How to install what the bridge needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[hils]'"

#: Seconds between scans of the world for the ego while there is none.
_EGO_SCAN_INTERVAL_S = 1.0

#: Seconds between attempts to reach a CARLA server that is not up yet.
_CONNECT_RETRY_S = 2.0

#: Sends a sensor device may have waiting with ``clock: wall`` before the
#: oldest is dropped: a device slower than its sensor gets the newest data.
_WALL_QUEUE = 2

#: A failing send is logged the first time and then once in this many.
_LOG_EVERY = 100

OpenChannel = Callable[..., "hilspy.Channel"]


class HilsUnavailableError(RuntimeError):
    """hilspy (the ``hils`` extra) is not installed."""


def _require_hilspy() -> ModuleType:
    try:
        import hilspy  # noqa: PLC0415
    except ImportError as exc:
        raise HilsUnavailableError(
            f"the HILS bridge needs hilspy: {INSTALL_HINT}"
        ) from exc
    return hilspy


def _vector(v: Any) -> Tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


class _Failures:
    """Logs a device's failing sends without flooding the log."""

    def __init__(self, device: str) -> None:
        self._device = device
        self._count = 0

    def report(self, what: str, exc: BaseException) -> None:
        if self._count % _LOG_EVERY == 0:
            logger.warning(
                "%s: %s failed (%d so far): %s",
                self._device,
                what,
                self._count + 1,
                exc,
            )
        self._count += 1


class _MessageDevice:
    """A device sent periodic messages filled from the ego's signals."""

    def __init__(self, spec: DeviceSpec, channel: "hilspy.Channel") -> None:
        self.spec = spec
        self.channel = channel
        self._periodics: List[Tuple[MessageSpec, "hilspy.Periodic"]] = []
        self._errors: Dict[str, Optional[str]] = {}

    def update(self, signals: Dict[str, float]) -> None:
        """Start the messages on the first call; update their fields after."""
        if not self._periodics:
            for message in self.spec.messages:
                periodic = self.channel.every(
                    message.period, message.name, message.values(signals)
                )
                self._periodics.append((message, periodic))
                logger.info(
                    "%s: %s every %.3f s", self.spec.name, message.name, periodic.period
                )
            return
        for message, periodic in self._periodics:
            periodic.update(message.name, message.values(signals))
            error = periodic.last_error
            if error != self._errors.get(message.name):
                self._errors[message.name] = error
                if error:
                    logger.warning("%s: %s: %s", self.spec.name, message.name, error)

    def close(self) -> None:
        for _, periodic in self._periodics:
            periodic.stop()
        self._periodics.clear()


class _SensorDevice:
    """A device sent what a sensor on the ego measures.

    The sensor's callback (a CARLA thread) only copies the data into a queue;
    :meth:`flush` turns it into messages and sends them -- on the bridge's
    thread with a simulated clock, so sends interleave with its advances in
    time order, or on the device's own thread with the wall clock, as a paced
    LiDAR send takes as long as the revolution it streams.
    """

    def __init__(
        self,
        spec: DeviceSpec,
        sensor_spec: SensorSpec,
        channel: "hilspy.Channel",
        hilspy: ModuleType,
        threaded: bool,
    ) -> None:
        self.spec = spec
        self.sensor_spec = sensor_spec
        self.channel = channel
        self._hilspy = hilspy
        self._actor: Optional["carla.Actor"] = None
        self._revolution: Optional[LidarRevolution] = None
        self._received: "queue.Queue[Tuple[float, Any, int, int]]" = queue.Queue()
        self._failures = _Failures(spec.name)
        self._outbox: Optional["queue.Queue[Optional[Any]]"] = None
        self._thread: Optional[threading.Thread] = None
        if threaded:
            self._outbox = queue.Queue(maxsize=_WALL_QUEUE)
            self._thread = threading.Thread(
                target=self._send_loop, name=f"hils-{spec.name}", daemon=True
            )
            self._thread.start()

    @property
    def attached(self) -> bool:
        return self._actor is not None

    def attach(
        self, carla_module: Any, world: "carla.World", ego: "carla.Actor", dt: float
    ) -> None:
        """Spawn the sensor on *ego* and listen; *dt* is the world's tick length."""
        spec = self.sensor_spec
        blueprint = world.get_blueprint_library().find(spec.blueprint)
        for key, value in spec.attributes.items():
            blueprint.set_attribute(key, value)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", self.spec.name)
        x, y, z, roll, pitch, yaw = spec.transform
        transform = carla_module.Transform(
            carla_module.Location(x=x, y=y, z=z),
            carla_module.Rotation(roll=roll, pitch=pitch, yaw=yaw),
        )
        actor = world.spawn_actor(blueprint, transform, attach_to=ego)
        if spec.is_lidar:
            attributes = actor.attributes
            frequency = (
                float(attributes["rotation_frequency"])
                if "rotation_frequency" in attributes
                else 1.0 / dt
            )
            self._revolution = LidarRevolution(frequency, dt)
        actor.listen(self._on_data)
        self._actor = actor
        logger.info("%s: %s attached to ego %d", self.spec.name, spec.blueprint, ego.id)

    def detach(self) -> None:
        actor, self._actor = self._actor, None
        if actor is None:
            return
        try:
            actor.stop()
            if actor.is_alive:
                actor.destroy()
        except Exception:  # noqa: BLE001 - the ego (and its sensors) may be gone
            logger.debug(
                "%s: the sensor was gone already", self.spec.name, exc_info=True
            )

    def _on_data(self, data: Any) -> None:
        """The sensor's listener, on a CARLA thread: copy, queue, return."""
        try:
            raw = np.frombuffer(data.raw_data, dtype=np.uint8).copy()
            width = int(getattr(data, "width", 0))
            height = int(getattr(data, "height", 0))
            self._received.put((float(data.timestamp), raw, width, height))
        except Exception:  # noqa: BLE001 - a sensor thread must not die
            logger.exception("%s: failed to receive a measurement", self.spec.name)

    def _message(self, measured: Tuple[float, Any, int, int]) -> Optional[Any]:
        timestamp, raw, width, height = measured
        spec = self.sensor_spec
        if self._revolution is not None:
            revolution = self._revolution.add(timestamp, lidar_points(raw))
            if revolution is None:
                return None
            return ("scan", lidar_scan_fields(revolution, spec.intensity_scale))
        frame = camera_frame(raw, width, height, spec.encoding)
        return self._hilspy.Message(data=frame)

    def _send(self, message: Any) -> None:
        try:
            if isinstance(message, tuple):
                self.channel.send(*message)
            else:
                self.channel.send(message)
        except Exception as exc:  # noqa: BLE001 - one device must not stop the others
            self._failures.report("send", exc)

    def flush(self) -> None:
        """Send what was measured since the last flush."""
        while True:
            try:
                measured = self._received.get_nowait()
            except queue.Empty:
                return
            try:
                message = self._message(measured)
            except Exception as exc:  # noqa: BLE001
                self._failures.report("conversion", exc)
                continue
            if message is None:
                continue
            if self._outbox is None:
                self._send(message)
                continue
            while True:
                try:
                    self._outbox.put_nowait(message)
                    break
                except queue.Full:
                    try:
                        self._outbox.get_nowait()
                    except queue.Empty:
                        pass

    def _send_loop(self) -> None:
        assert self._outbox is not None
        while True:
            message = self._outbox.get()
            if message is None:
                return
            self._send(message)

    def close(self) -> None:
        self.detach()
        if self._outbox is not None and self._thread is not None:
            while True:
                try:
                    self._outbox.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._outbox.get_nowait()
                    except queue.Empty:
                        pass
            self._thread.join(timeout=5.0)


class HilsBridge:
    """Streams a run's ego to the devices *config* describes.

    Args:
        config: The bridge's configuration.
        carla_module: typesafe_carla's ``carla`` module (imported when ``None``).
        open_channel: Opens a device's channel (``hilspy.open`` when ``None``).

    Raises:
        HilsUnavailableError: hilspy is not installed.
        CanvasUnavailableError: ``canvas`` is set, but carla_canvas is not
            installed or the file is missing.
    """

    def __init__(
        self,
        config: HilsConfig,
        carla_module: Optional[Any] = None,
        open_channel: Optional[OpenChannel] = None,
    ) -> None:
        self.config = config
        self._hilspy = _require_hilspy()
        # Before typesafe_carla is imported and before any sensor listens:
        # typesafe_carla loads its sensor plugin then.
        use_canvas(config.canvas)
        if carla_module is None:
            import typesafe_carla.carla as _carla  # noqa: PLC0415

            carla_module = _carla
        self._carla: Any = carla_module
        self._open: OpenChannel = open_channel or self._hilspy.open
        self.clock: Optional["hilspy.Clock"] = (
            self._hilspy.Clock(start=time.time())
            if config.clock == "simulation"
            else None
        )
        self._messages: List[_MessageDevice] = []
        self._sensors: List[_SensorDevice] = []
        self._client: Optional["carla.Client"] = None
        self._world: Optional["carla.World"] = None
        self._map: Optional["carla.Map"] = None
        self._ego: Optional["carla.Actor"] = None
        self._last_time: Optional[float] = None
        self._next_scan = 0.0
        self._stop = threading.Event()

    # -- set-up --------------------------------------------------------------

    def open_devices(self) -> None:
        """Open every device's channel."""
        for spec in self.config.devices:
            options: Dict[str, Any] = dict(spec.options)
            if self.clock is not None:
                options["clock"] = self.clock
            channel = self._open(spec.url, spec.codec, **options)
            logger.info("%s: %s (%s)", spec.name, spec.url, spec.codec)
            if spec.sensor is not None:
                self._sensors.append(
                    _SensorDevice(
                        spec, spec.sensor, channel, self._hilspy, self.clock is None
                    )
                )
            else:
                self._messages.append(_MessageDevice(spec, channel))

    def connect(self) -> None:
        """Connect to CARLA."""
        cfg = self.config
        self._client = self._carla.Client(cfg.host, cfg.port)
        self._client.set_timeout(cfg.timeout)
        self._world = self._client.get_world()
        logger.info("connected to CARLA at %s:%d", cfg.host, cfg.port)

    # -- the ego -------------------------------------------------------------

    @property
    def ego(self) -> Optional["carla.Actor"]:
        """The ego the bridge follows, once found."""
        return self._ego

    def _find_ego(self, snapshot: "carla.WorldSnapshot") -> None:
        now = time.monotonic()
        if now < self._next_scan:
            return
        self._next_scan = now + _EGO_SCAN_INTERVAL_S
        assert self._world is not None
        ego = find_actor_in_world(self._world, self.config.role_name)
        if ego is None:
            return
        self._ego = ego
        self._map = self._world.get_map()
        dt = float(snapshot.delta_seconds) or 0.05
        logger.info("following ego %d (%s)", ego.id, ego.type_id)
        for device in self._sensors:
            try:
                device.attach(self._carla, self._world, ego, dt)
            except Exception:
                logger.exception("%s: failed to attach the sensor", device.spec.name)

    def _lose_ego(self) -> None:
        logger.info("the ego is gone; waiting for the next one")
        self._ego = None
        for device in self._sensors:
            device.detach()

    def _state(self, snapshot: "carla.WorldSnapshot") -> Optional[EgoState]:
        assert self._ego is not None and self._map is not None
        found = snapshot.find(self._ego.id)
        if found is None:
            return None
        transform = found.get_transform()
        geo = self._map.transform_to_geolocation(transform.location)
        vehicle = (
            self._ego.as_vehicle() if hasattr(self._ego, "as_vehicle") else self._ego
        )
        control = vehicle.get_control()
        rotation = transform.rotation
        return EgoState(
            time=float(snapshot.timestamp.elapsed_seconds),
            frame=int(snapshot.frame),
            location=_vector(transform.location),
            rotation=(float(rotation.pitch), float(rotation.yaw), float(rotation.roll)),
            velocity=_vector(found.get_velocity()),
            acceleration=_vector(found.get_acceleration()),
            angular_velocity=_vector(found.get_angular_velocity()),
            geolocation=(
                float(geo.latitude),
                float(geo.longitude),
                float(geo.altitude),
            ),
            throttle=float(control.throttle),
            brake=float(control.brake),
            steer=float(control.steer),
            hand_brake=bool(control.hand_brake),
            reverse=bool(control.reverse),
            gear=int(control.gear),
        )

    # -- the loop ------------------------------------------------------------

    def step(self) -> bool:
        """Wait for the next tick and serve it; ``False`` when none came in time."""
        assert self._world is not None
        try:
            snapshot = self._world.wait_for_tick(self.config.timeout)
        except RuntimeError as exc:
            # No tick in time: the runner is between runs, or loaded another
            # map (a new episode, whose world object is a new one).
            logger.debug("no tick: %s", exc)
            if self._client is not None:
                self._world = self._client.get_world()
            if self._ego is not None and not self._ego.is_alive:
                self._lose_ego()
            return False
        if self._ego is None:
            self._find_ego(snapshot)
        if self._ego is not None:
            state = self._state(snapshot)
            if state is None:
                self._lose_ego()
            else:
                signals = ego_signals(state)
                for message_device in self._messages:
                    try:
                        message_device.update(signals)
                    except Exception:
                        logger.exception(
                            "%s: failed to update", message_device.spec.name
                        )
        for sensor_device in self._sensors:
            sensor_device.flush()
        self._advance(snapshot)
        return True

    def _advance(self, snapshot: "carla.WorldSnapshot") -> None:
        """Advance the clock by the simulated time since the last tick."""
        elapsed = float(snapshot.timestamp.elapsed_seconds)
        last, self._last_time = self._last_time, elapsed
        if self.clock is None or last is None:
            return
        dt = elapsed - last
        if dt <= 0.0:  # a new episode restarted the world's clock
            dt = float(snapshot.delta_seconds)
        if dt > 0.0:
            self.clock.advance(dt)

    def run(self) -> None:
        """Connect, open the devices and serve ticks until :meth:`stop`.

        A CARLA server that is not up yet is waited for.
        """
        attempts = 0
        while self._world is None and not self._stop.is_set():
            try:
                self.connect()
            except RuntimeError as exc:
                logger.log(
                    logging.DEBUG if attempts else logging.WARNING,
                    "no CARLA at %s:%d (%s); retrying every %.0f s",
                    self.config.host,
                    self.config.port,
                    exc,
                    _CONNECT_RETRY_S,
                )
                attempts += 1
                self._stop.wait(_CONNECT_RETRY_S)
        if self._stop.is_set():
            return
        if not self._messages and not self._sensors:
            self.open_devices()
        logger.info("waiting for ego %r", self.config.role_name)
        waited_since = time.monotonic()
        while not self._stop.is_set():
            self.step()
            if self._ego is not None:
                waited_since = time.monotonic()
            elif time.monotonic() - waited_since > self.config.ego_wait > 0.0:
                logger.warning(
                    "no ego %r for %.0f s; still waiting",
                    self.config.role_name,
                    self.config.ego_wait,
                )
                waited_since = time.monotonic()

    def stop(self) -> None:
        """Have :meth:`run` return after the tick it is serving."""
        self._stop.set()

    def close(self) -> None:
        """Destroy the sensors and close every channel."""
        for message_device in self._messages:
            message_device.close()
            message_device.channel.close()
        for sensor_device in self._sensors:
            sensor_device.close()
            sensor_device.channel.close()
        self._messages.clear()
        self._sensors.clear()
        self._ego = None
