"""The HILS bridge: a resident process between scenario runs and the ECUs under test.

It stays up on the HILS PC next to the ECUs and is a CARLA client of its own,
beside the scenario runner.  It never ticks the world: it waits for the
runner's ticks (``World.wait_for_tick``), so it works in synchronous mode too.

**Lifecycle.**  The configuration's ``lifecycle`` steps (send a message, wait
for an ECU's message, sleep) run at four points:

* ``startup`` -- when the bridge starts, its devices just opened;
* ``scenario_start`` -- when a scenario starts.  A run with
  ``ego.entity=hils`` says so over the link (:mod:`.link`, :mod:`.protocol`)
  and waits for the steps -- the ECUs initialized -- before its clock starts;
  without one, the ego appearing starts it.  The periodic messages start and
  the sensors are attached first, so the ECUs have their inputs;
* ``scenario_end`` -- when it ends: the run's ``end``, the run going away, or
  the ego disappearing; then the periodic messages stop and the sensors go;
* ``shutdown`` -- when the bridge stops (Ctrl-C, SIGTERM).

**Each tick** of a scenario: the ego's state (:class:`~.signals.EgoState`)
becomes the signals the periodic messages are updated with; what the sensors
measured is sent (LiDAR revolutions as a model's UDP packets, camera frames);
with ``clock: simulation`` the shared ``hilspy.Clock`` advances by the
simulated time that passed, sending what falls due in it.  While steps run the
world is not followed and the clock follows the wall clock, so periodic
messages keep flowing to an ECU being waited on.

**ECU to simulation.**  Devices with ``inputs`` are received on a thread each;
an input message's fields become command fields (:data:`~.protocol.COMMAND_FIELDS`)
merged into the ECUs' command, which is forwarded to the run as it changes.
The run's ego entity applies it to the ego (:mod:`~autoware_carla_scenario.entity.hils_entity`).

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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..canvas import use_canvas
from ..entity_role import find_actor_in_world
from .config import DeviceSpec, HilsConfig, MessageSpec, SensorSpec, StepSpec
from .link import Event, RunnerLink
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

__all__ = ["INSTALL_HINT", "HilsBridge", "HilsStepError", "HilsUnavailableError"]

logger = logging.getLogger(__name__)

#: How to install what the bridge needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[hils]'"

#: Seconds between scans of the world for the ego while there is none.
_EGO_SCAN_INTERVAL_S = 1.0

#: Seconds between attempts to reach a CARLA server that is not up yet.
_CONNECT_RETRY_S = 2.0

#: How long the loop waits for a tick before it looks at the runs' requests.
_TICK_POLL_S = 0.1

#: How often a step waiting (for an ECU, or a sleep) looks again, seconds.
_STEP_POLL_S = 0.01

#: How long a receiving thread waits for a message before it looks at whether
#: it is to stop, seconds.
_RECV_POLL_S = 0.2

#: Sends a sensor device may have waiting with ``clock: wall`` before the
#: oldest is dropped: a device slower than its sensor gets the newest data.
_WALL_QUEUE = 2

#: A failing send is logged the first time and then once in this many.
_LOG_EVERY = 100

#: A ``disconnect``'s direction (:data:`~.protocol.LINK_DIRECTIONS`) as the
#: device channel's: the bridge sends the ECU its inputs and receives its output.
_CHANNEL_DIRECTIONS: Dict[str, Literal["send", "recv", "both"]] = {
    "to_ecu": "send",
    "from_ecu": "recv",
    "both": "both",
}

#: A channel's two sides, which cuts stop separately.
_SIDES: Tuple[Literal["send", "recv"], ...] = ("send", "recv")

OpenChannel = Callable[..., "hilspy.Channel"]


class HilsUnavailableError(RuntimeError):
    """hilspy (the ``hils`` extra) is not installed."""


class HilsStepError(RuntimeError):
    """A lifecycle step failed: a send was refused, or an ECU did not answer."""


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


def _matches(fields: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Whether a decoded message's *fields* have the *expected* values."""
    for key, want in expected.items():
        if key not in fields:
            return False
        got = fields[key]
        numbers = (int, float)
        if isinstance(want, numbers) and isinstance(got, numbers):
            if abs(float(got) - float(want)) > 1e-9:
                return False
        elif got != want:
            return False
    return True


class _Cut(NamedTuple):
    """Data a scenario stops (:meth:`HilsBridge.disconnect`)."""

    #: The device, or ``None`` for every device.
    device: Optional[str]
    #: The names of the messages, or ``None`` for everything.
    messages: Optional[FrozenSet[str]]
    #: The channel's direction: ``send``, ``recv`` or ``both``.
    direction: str


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


class _Receiver:
    """Receives what an ECU sends on a device, on a thread of its own.

    Each decoded message is remembered (the newest of each name, with when it
    came, for a ``wait`` step) and handed to *on_message* (the inputs).
    """

    def __init__(
        self,
        spec: DeviceSpec,
        channel: "hilspy.Channel",
        on_message: Callable[[DeviceSpec, Any], None],
    ) -> None:
        self.spec = spec
        self._channel = channel
        self._on_message = on_message
        self._latest: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._receive, name=f"hils-rx-{spec.name}", daemon=True
        )
        self._thread.start()

    def _receive(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._channel.recv(_RECV_POLL_S)
            except Exception as exc:  # noqa: BLE001 - closed, or a broken transport
                if not self._stop.is_set():
                    logger.warning("%s: receiving stopped: %s", self.spec.name, exc)
                return
            if message is None or message.name is None:
                continue
            fields = dict(message.fields)
            with self._lock:
                self._latest[message.name] = (time.monotonic(), fields)
            try:
                self._on_message(self.spec, message)
            except Exception:  # noqa: BLE001 - a receiving thread must not die
                logger.exception(
                    "%s: failed to handle %s", self.spec.name, message.name
                )

    def seen(self, name: str, match: Mapping[str, Any], since: float) -> bool:
        """Whether *name* with the *match*ing fields came at or after *since*."""
        with self._lock:
            latest = self._latest.get(name)
        return latest is not None and latest[0] >= since and _matches(latest[1], match)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class _MessageDevice:
    """A device sent periodic messages filled from the ego's signals, while a
    scenario runs."""

    def __init__(self, spec: DeviceSpec, channel: "hilspy.Channel") -> None:
        self.spec = spec
        self.channel = channel
        self._periodics: List[Tuple[MessageSpec, "hilspy.Periodic"]] = []
        self._errors: Dict[str, Optional[str]] = {}

    def start(self, signals: Mapping[str, float]) -> None:
        """Start sending the messages, filled from *signals*."""
        self.stop()
        for message in self.spec.messages:
            periodic = self.channel.every(
                message.period, message.name, message.values(signals)
            )
            self._periodics.append((message, periodic))
            logger.info(
                "%s: %s every %.3f s", self.spec.name, message.name, periodic.period
            )

    def update(self, signals: Mapping[str, float]) -> None:
        """Refill the messages from *signals*."""
        for message, periodic in self._periodics:
            periodic.update(message.name, message.values(signals))
            error = periodic.last_error
            if error != self._errors.get(message.name):
                self._errors[message.name] = error
                if error:
                    logger.warning("%s: %s: %s", self.spec.name, message.name, error)

    def stop(self) -> None:
        """Stop sending the messages."""
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
    """The resident bridge the configuration *config* describes.

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
        self._channels: Dict[str, "hilspy.Channel"] = {}
        self._messages: List[_MessageDevice] = []
        self._sensors: List[_SensorDevice] = []
        self._receivers: Dict[str, _Receiver] = {}
        self._events: "queue.Queue[Event]" = queue.Queue()
        self._link: Optional[RunnerLink] = None
        self._client: Optional["carla.Client"] = None
        self._world: Optional["carla.World"] = None
        self._map: Optional["carla.Map"] = None
        self._ego: Optional["carla.Actor"] = None
        self._signals: Dict[str, float] = {}
        self._last_time: Optional[float] = None
        self._last_tick = time.monotonic()
        self._next_scan = 0.0
        self._started = False
        #: Whether a scenario runs (its scenario_start steps ran).
        self._active = False
        #: Whether a run is connected: then the run, not the ego appearing,
        #: starts scenarios.
        self._session = False
        #: Whether the ECUs' commands go to the run (after its ``ready``).
        self._forwarding = False
        self._command: Dict[str, float] = {}
        self._command_lock = threading.Lock()
        #: What the scenario stops (:meth:`disconnect`), by cut id.
        self._cuts: Dict[str, _Cut] = {}
        self._stop = threading.Event()

    # -- set-up and shut-down ------------------------------------------------

    def open_devices(self) -> None:
        """Open every device's channel, and receive on those that need it."""
        waited = {
            step.device
            for phase in self.config.lifecycle.values()
            for step in phase
            if step.kind == "wait"
        }
        for spec in self.config.devices:
            options: Dict[str, Any] = dict(spec.options)
            if self.clock is not None:
                options["clock"] = self.clock
            channel = self._open(spec.url, spec.codec, **options)
            self._channels[spec.name] = channel
            logger.info("%s: %s (%s)", spec.name, spec.url, spec.codec)
            if spec.sensor is not None:
                self._sensors.append(
                    _SensorDevice(
                        spec, spec.sensor, channel, self._hilspy, self.clock is None
                    )
                )
                continue
            if spec.messages:
                self._messages.append(_MessageDevice(spec, channel))
            if spec.inputs or spec.name in waited:
                self._receivers[spec.name] = _Receiver(spec, channel, self._on_input)

    def startup(self) -> None:
        """Open the devices and run the ``startup`` steps.

        Raises:
            HilsStepError: A startup step failed.
        """
        if self._started:
            return
        self._started = True
        self.open_devices()
        self._run_steps("startup", strict=True)
        if self.config.runner_listen:
            self._link = RunnerLink(self.config.runner_listen, self._events)
            self._link.start()

    @property
    def link_address(self) -> Optional[Tuple[str, int]]:
        """Where runs reach the bridge, once started; ``None`` without a link."""
        return self._link.address if self._link is not None else None

    def connect(self) -> None:
        """Connect to CARLA."""
        cfg = self.config
        self._client = self._carla.Client(cfg.host, cfg.port)
        self._client.set_timeout(cfg.timeout)
        self._world = self._client.get_world()
        logger.info("connected to CARLA at %s:%d", cfg.host, cfg.port)

    def close(self) -> None:
        """End a running scenario, run the ``shutdown`` steps, close everything."""
        if self._active:
            self._finish()
        if self._started:
            self._run_steps("shutdown", strict=False)
        if self._link is not None:
            self._link.close()
            self._link = None
        for receiver in self._receivers.values():
            receiver.close()
        for sensor_device in self._sensors:
            sensor_device.close()
        for message_device in self._messages:
            message_device.stop()
        for channel in self._channels.values():
            channel.close()
        self._receivers.clear()
        self._sensors.clear()
        self._messages.clear()
        self._channels.clear()
        self._started = False

    # -- lifecycle steps -----------------------------------------------------

    def _run_steps(self, phase: str, strict: bool) -> None:
        """Run *phase*'s steps; the first failure raises when *strict*, else
        each is logged and the rest still run."""
        steps = self.config.steps(phase)
        if not steps:
            return
        logger.info("%s: %d step(s)", phase, len(steps))
        since = time.monotonic()
        try:
            for step in steps:
                try:
                    self._run_step(step, since)
                except HilsStepError as exc:
                    if strict:
                        raise HilsStepError(f"{phase}: {exc}") from exc
                    logger.warning("%s: %s", phase, exc)
        finally:
            # The clock followed the wall clock meanwhile: the next tick must
            # not add the world time that passed as well.
            self._last_time = None

    def _run_step(self, step: StepSpec, since: float) -> None:
        logger.info("  %s", step.describe())
        if step.kind == "sleep":
            self._pump(lambda: False, step.seconds)
            return
        if step.kind == "wait":
            receiver = self._receivers[step.device]
            assert step.name is not None
            name: str = step.name
            if not self._pump(
                lambda: receiver.seen(name, step.match, since), step.seconds
            ):
                expected = f" with {dict(step.match)}" if step.match else ""
                raise HilsStepError(
                    f"{step.device} sent no {step.name}{expected} within {step.seconds} s"
                )
            return
        channel = self._channels[step.device]
        try:
            if step.name is None:
                channel.send(self._hilspy.Message(data=step.data, id=step.frame_id))
            else:
                values = {k: v.resolve(self._signals) for k, v in step.fields.items()}
                channel.send(step.name, values)
        except Exception as exc:  # noqa: BLE001 - hilspy.Error, or a missing signal
            raise HilsStepError(f"{step.describe()} failed: {exc}") from exc

    def _pump(self, done: Callable[[], bool], seconds: float) -> bool:
        """Wait up to *seconds* for *done*; the simulated clock follows the wall
        clock meanwhile.  Whether *done* came true."""
        start = last = time.monotonic()
        while True:
            if done():
                return True
            now = time.monotonic()
            if now - start >= seconds:
                return False
            time.sleep(min(_STEP_POLL_S, seconds - (now - start)))
            now = time.monotonic()
            if self.clock is not None:
                self.clock.advance(now - last)
            last = now

    # -- scenarios -----------------------------------------------------------

    @property
    def ego(self) -> Optional["carla.Actor"]:
        """The ego of the scenario running, if any."""
        return self._ego

    @property
    def active(self) -> bool:
        """Whether a scenario runs."""
        return self._active

    def _begin(self, ego: "carla.Actor", scenario: str) -> None:
        """Start a scenario for *ego*: periodic messages, sensors, then the
        ``scenario_start`` steps.

        Raises:
            HilsStepError: A scenario_start step failed (the scenario is still
                running, and ends as any other).
        """
        assert self._world is not None
        self._ego = ego
        self._active = True
        self._map = self._world.get_map()
        snapshot = self._world.get_snapshot()
        state = self._state(snapshot)
        if state is None:
            state = EgoState(
                time=float(snapshot.timestamp.elapsed_seconds),
                frame=int(snapshot.frame),
                location=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0),
            )
        self._signals = ego_signals(state)
        logger.info(
            "scenario %sstarts: ego %d (%s)",
            f"{scenario!r} " if scenario else "",
            ego.id,
            ego.type_id,
        )
        for message_device in self._messages:
            message_device.start(self._signals)
        dt = float(snapshot.delta_seconds) or 0.05
        for sensor_device in self._sensors:
            try:
                sensor_device.attach(self._carla, self._world, ego, dt)
            except Exception:
                logger.exception(
                    "%s: failed to attach the sensor", sensor_device.spec.name
                )
        self._run_steps("scenario_start", strict=True)

    def _finish(self) -> None:
        """End the scenario: the ``scenario_end`` steps, then stop the periodic
        messages and remove the sensors."""
        self._forwarding = False
        self._run_steps("scenario_end", strict=False)
        for message_device in self._messages:
            message_device.stop()
        for sensor_device in self._sensors:
            sensor_device.detach()
        with self._command_lock:
            self._command = {}
        self._reconnect_all()
        self._ego = None
        self._active = False
        logger.info("scenario ended")

    def disconnect(
        self,
        cut_id: str,
        device: Optional[str] = None,
        messages: Optional[Sequence[str]] = None,
        direction: str = "both",
    ) -> None:
        """Stop data, as a break in the communication path would, until
        :meth:`reconnect` with *cut_id* or the scenario's end: *messages* (by
        name; everything when ``None``) of *device* (every device when
        ``None``), in *direction* (:data:`~.protocol.LINK_DIRECTIONS`).

        Cuts add up: a device stops what any of its cuts stops, and a cut
        given again under its *cut_id* is replaced.

        Raises:
            ValueError: No scenario runs, or no such device or direction.
        """
        if not self._active:
            raise ValueError("no scenario runs")
        if direction not in _CHANNEL_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {', '.join(_CHANNEL_DIRECTIONS)}, "
                f"got {direction!r}"
            )
        if device is not None and device not in self._channels:
            raise ValueError(
                f"no device {device!r}; the devices are "
                f"{', '.join(sorted(self._channels)) or 'none'}"
            )
        names = None if messages is None else frozenset(str(m) for m in messages)
        self._cuts[cut_id] = _Cut(device, names, _CHANNEL_DIRECTIONS[direction])
        logger.warning(
            "%s: %s stopped (%s)",
            device or "every device",
            "every message" if names is None else ", ".join(sorted(names)),
            direction,
        )
        self._apply_cuts()

    def reconnect(self, cut_id: str) -> None:
        """Carry what the cut *cut_id* stopped again (nothing if there is none)."""
        if self._cuts.pop(cut_id, None) is not None:
            logger.info("cut %r restored", cut_id)
            self._apply_cuts()

    def _apply_cuts(self) -> None:
        """Set every device's channel to stop what its cuts stop together."""
        for name, channel in self._channels.items():
            for side in _SIDES:
                cuts = [
                    c
                    for c in self._cuts.values()
                    if c.device in (None, name) and c.direction in (side, "both")
                ]
                if not cuts:
                    if channel.disconnected(side):
                        channel.reconnect(direction=side)
                elif any(c.messages is None for c in cuts):
                    channel.disconnect(direction=side)
                else:
                    stopped = sorted(set().union(*(c.messages or () for c in cuts)))
                    channel.disconnect(direction=side, messages=stopped)

    def _reconnect_all(self) -> None:
        """Carry everything again."""
        if self._cuts:
            self._cuts.clear()
            logger.info("every cut restored")
        self._apply_cuts()

    def _on_input(self, spec: DeviceSpec, message: Any) -> None:
        """A receiving thread's message: merge its command fields, forward them."""
        values: Dict[str, float] = {}
        for input_spec in spec.inputs:
            if input_spec.name == message.name:
                values.update(input_spec.values(message.fields))
        if not values:
            return
        with self._command_lock:
            self._command.update(values)
            command = dict(self._command)
        if self._forwarding and self._link is not None:
            self._link.send(
                {"type": "command", "mode": self.config.command_mode, **command}
            )

    @property
    def command(self) -> Dict[str, float]:
        """The ECUs' latest command fields."""
        with self._command_lock:
            return dict(self._command)

    def _handle(self, event: Event) -> None:
        """A run's request (:class:`~.link.RunnerLink`)."""
        kind, message = event
        link = self._link
        if kind == "connect":
            self._session = True
            return
        if kind == "gone":
            self._session = False
            if self._active:
                logger.warning("the run went away during its scenario")
                self._finish()
            return
        assert link is not None
        if kind == "end":
            if self._active:
                self._finish()
            link.send({"type": "ended"})
            return
        if kind == "disconnect":
            device = message.get("device")
            messages = message.get("messages")
            try:
                if messages is not None and not isinstance(messages, list):
                    raise ValueError("messages must be a list of names")
                self.disconnect(
                    str(message.get("id", "")),
                    None if device is None else str(device),
                    messages,
                    str(message.get("direction", "both")),
                )
            except ValueError as exc:
                link.send({"type": "error", "message": f"disconnect: {exc}"})
            return
        if kind == "reconnect":
            self.reconnect(str(message.get("id", "")))
            return
        # start
        if self._active:
            self._finish()
        if self._client is not None:
            # The run may have loaded another map: a new world.
            self._world = self._client.get_world()
        assert self._world is not None
        ego_id = message.get("ego_id")
        ego = self._world.get_actor(int(ego_id)) if isinstance(ego_id, int) else None
        if ego is None:
            link.send({"type": "error", "message": f"no ego with actor id {ego_id}"})
            return
        try:
            self._begin(ego, str(message.get("scenario", "")))
        except HilsStepError as exc:
            logger.error("%s", exc)
            link.send({"type": "error", "message": str(exc)})
            return
        link.send({"type": "ready"})
        self._forwarding = True
        command = self.command
        if command:
            link.send({"type": "command", "mode": self.config.command_mode, **command})

    def _find_ego(self) -> None:
        """Without a run connected, the ego appearing starts a scenario."""
        now = time.monotonic()
        if now < self._next_scan:
            return
        self._next_scan = now + _EGO_SCAN_INTERVAL_S
        assert self._world is not None
        ego = find_actor_in_world(self._world, self.config.role_name)
        if ego is None:
            return
        try:
            self._begin(ego, "")
        except HilsStepError as exc:
            logger.error("%s", exc)

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

    def handle_events(self) -> None:
        """Serve the runs' requests that arrived."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(event)
            except Exception:
                logger.exception("failed to serve a run's %s", event[0])

    def step(self, timeout: Optional[float] = None) -> bool:
        """Wait up to *timeout* s (the configuration's when ``None``) for the next
        tick and serve it; ``False`` when none came."""
        assert self._world is not None
        try:
            snapshot = self._world.wait_for_tick(
                self.config.timeout if timeout is None else timeout
            )
        except RuntimeError as exc:
            if time.monotonic() - self._last_tick >= self.config.timeout:
                # Quiet for long: the runner is between runs, or loaded another
                # map (a new episode, whose world object is a new one).
                logger.debug("no tick: %s", exc)
                self._last_tick = time.monotonic()
                if self._client is not None:
                    self._world = self._client.get_world()
                if self._ego is not None and not self._ego.is_alive:
                    logger.info("the ego is gone")
                    self._finish()
            return False
        self._last_tick = time.monotonic()
        if not self._active and not self._session:
            self._find_ego()
        if self._active and self._ego is not None:
            state = self._state(snapshot)
            if state is None:
                logger.info("the ego is gone")
                self._finish()
            else:
                self._signals = ego_signals(state)
                for message_device in self._messages:
                    try:
                        message_device.update(self._signals)
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
        """Start up, connect, and serve runs and ticks until :meth:`stop`.

        A CARLA server that is not up yet is waited for (the startup steps
        have run by then).

        Raises:
            HilsStepError: A startup step failed.
        """
        self.startup()
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
        logger.info("ready: waiting for scenarios (ego %r)", self.config.role_name)
        while not self._stop.is_set():
            self.handle_events()
            self.step(_TICK_POLL_S)

    def stop(self) -> None:
        """Have :meth:`run` return after the tick it is serving."""
        self._stop.set()
