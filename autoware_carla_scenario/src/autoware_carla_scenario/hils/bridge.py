"""The HILS bridge: a resident process between scenario runs and the ECUs under test.

It stays up on the HILS PC next to the ECUs and is a CARLA client of its own,
beside the scenario runner.  It never ticks the world: it waits for the
runner's ticks (``World.wait_for_tick``), so it works in synchronous mode too.

**Lifecycle.**  The configuration's ``lifecycle`` steps (send a message, wait
for an ECU's message, sleep) run at five points:

* ``startup`` -- when the bridge starts, its devices just opened;
* ``scenario_start`` -- when a scenario starts.  A run with
  ``ego.entity=hils`` says so over the link (:mod:`.link`, :mod:`.protocol`)
  and waits for the steps -- the ECUs initialized -- before its clock starts;
  without one, the ego appearing starts it -- not the ego of a run that went
  away, which its failure or end can leave in the world a while.  The
  periodic messages start first, so the ECUs have their inputs; the sensors
  are attached after the steps, so that nothing they measured while the
  steps waited (nothing is sent meanwhile) reaches the ECUs late, or piles
  up in memory;
* ``warmup`` -- right after, with the sensors attached and the world followed
  (the run ticks it, its vehicles held on their brakes): the ECUs getting
  ready to drive on what they measure.  The run waits for these steps too
  before its clock starts -- its warm-up, as for an ego whose stack has to
  localize, route and engage;
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

import functools
import logging
import queue
import threading
import time
from collections import deque
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from ..canvas import use_canvas
from ..constants import FIXED_DELTA_SECONDS
from ..entity_role import find_actor_in_world
from ..sensor.base import put_dropping_oldest
from .config import DeviceSpec, HilsConfig, MessageSpec, SensorSpec, StepSpec
from .link import Event, RunnerLink

# Re-exported: the cuts' sides and what they stop together are the checked
# .cuts' (#2386).
from .cuts import _DIRECTION_SIDES as _DIRECTION_SIDES
from .cuts import _SIDES as _SIDES
from .cuts import Side as Side
from .cuts import _Cut as _Cut
from .cuts import cut_refusal, messages_refusal, stopped_messages
from .measurements import (
    LidarRevolution,
    camera_frame,
    lidar_points,
    lidar_scan_fields,
    ray_times,
    spin_lidar_points,
    steady_revolution_start,
)
from .signals import EgoState, ego_signals

if TYPE_CHECKING:
    import hilspy
    import typesafe_carla.carla as carla

__all__ = [
    "INSTALL_HINT",
    "HilsBridge",
    "HilsListenError",
    "HilsOpenError",
    "HilsStepError",
    "HilsUnavailableError",
]

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

#: How long a sensor's :meth:`_SensorDevice.detach` waits for a send under
#: way, seconds: a device that hangs in a send must not keep the scenario
#: from ending (nor the bridge from closing).
_SEND_WAIT_S = 5.0

#: A failing send is logged the first time and then once in this many.
_LOG_EVERY = 100

#: How many of the newest messages of each name a receiver keeps for a
#: ``wait`` it was not told of (:class:`_Receiver`); those it was told of are
#: followed exactly, however many messages come.
_HISTORY = 256

OpenChannel = Callable[..., "hilspy.Channel"]


class HilsUnavailableError(RuntimeError):
    """hilspy (the ``hils`` extra) is not installed."""


class HilsStepError(RuntimeError):
    """A lifecycle step failed: a send was refused, or an ECU did not answer."""


class HilsListenError(RuntimeError):
    """The bridge could not listen for runs at ``runner.listen`` (the port is
    taken, or the address is not the machine's); raised before any startup
    step is sent."""


class HilsOpenError(RuntimeError):
    """A device's channel could not be opened (a ``vcan`` interface that is
    not there, a codec file that is missing, an option its codec does not
    take); raised before any startup step is sent, the devices opened
    before it closed again."""


class _Stopping(HilsStepError):
    """A step that waits was cut short: the bridge is stopping
    (:meth:`HilsBridge.stop`)."""


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

    Each decoded message is handed to *on_message* (the inputs) and remembered
    for the ``wait`` steps (:meth:`seen`): for each of the *patterns* (a
    message's name and the field values a ``wait`` step looks for), when a
    message last matched it, so that a matching message counts however many
    messages of its name follow it; and, for any other name and fields asked
    about, the newest :data:`_HISTORY` messages of each name (#2277).
    """

    def __init__(
        self,
        spec: DeviceSpec,
        channel: "hilspy.Channel",
        on_message: Callable[[DeviceSpec, Any], None],
        patterns: Sequence[Tuple[str, Mapping[str, Any]]] = (),
    ) -> None:
        self.spec = spec
        self._channel = channel
        self._on_message = on_message
        #: Each pattern, and when a message last matched it (``None``: never).
        self._patterns: List[Tuple[str, Dict[str, Any], Optional[float]]] = []
        for name, match in patterns:
            entry = (name, dict(match), None)
            if all(p[:2] != entry[:2] for p in self._patterns):
                self._patterns.append(entry)
        self._history: Dict[str, Deque[Tuple[float, Dict[str, Any]]]] = {}
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
            self._remember(str(message.name), fields, time.monotonic())
            try:
                self._on_message(self.spec, message)
            except Exception:  # noqa: BLE001 - a receiving thread must not die
                logger.exception(
                    "%s: failed to handle %s", self.spec.name, message.name
                )

    def _remember(self, name: str, fields: Dict[str, Any], now: float) -> None:
        with self._lock:
            for i, (pattern, match, _) in enumerate(self._patterns):
                if pattern == name and _matches(fields, match):
                    self._patterns[i] = (pattern, match, now)
            history = self._history.get(name)
            if history is None:
                history = self._history[name] = deque(maxlen=_HISTORY)
            history.append((now, fields))

    def seen(self, name: str, match: Mapping[str, Any], since: float) -> bool:
        """Whether a message *name* with the *match*ing fields came at or after
        *since* (any such message, not only the newest of its name)."""
        wanted = dict(match)
        with self._lock:
            for pattern, expected, last in self._patterns:
                if pattern == name and expected == wanted:
                    return last is not None and last >= since
            history = list(self._history.get(name, ()))
        return any(t >= since and _matches(f, wanted) for t, f in history)

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

    Nothing a detached sensor measured is sent once :meth:`detach` returns.
    Each :meth:`attach` and :meth:`detach` starts a new *generation*; the
    callback tags what it queues with the generation it was attached in, and
    every send checks the tag against the current generation while holding
    :attr:`_send_lock`, which :meth:`detach` takes once it moved to the next
    generation.  So a measurement still queued, one waiting in the outbox, or
    one a callback in flight queues after :meth:`detach` drained the queues
    (CARLA may still be inside the callback when ``stop()`` returns) is
    dropped, and a send already under way finishes before :meth:`detach`
    returns, unless it hangs longer than :data:`_SEND_WAIT_S`.  A plain drain
    could not promise that: a callback can put after it.  :attr:`_convert_lock` guards the conversion (a LiDAR's revolution)
    apart from the send, so that with the wall clock :meth:`flush` never
    waits for the device's thread.  The two locks are leaves, never held
    together nor while taking the bridge's locks (:meth:`HilsBridge._finish`
    releases ``_forward_lock`` before it detaches the sensors).
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
        #: The ego the sensor rides (for its motion while a LiDAR turns).
        self._ego: Optional["carla.Actor"] = None
        #: With the wall clock, a LiDAR's arcs not yet sent whole: id -> (when it
        #: arrived, the sensor's motion then); its points carry the id (column 4).
        self._arcs: Dict[int, Tuple[float, Tuple[float, ...]]] = {}
        self._next_arc = 0
        #: When the last revolution sent started (wall clock), for the next.
        self._last_start: Optional[float] = None
        #: Gathers a LiDAR's arcs into revolutions (under :attr:`_convert_lock`).
        self._revolution: Optional[LidarRevolution] = None
        #: The attach generation (:meth:`attach`, :meth:`detach`); written by
        #: the bridge's thread only, under :attr:`_generation_lock`.
        self._generation = 0
        self._generation_lock = threading.Lock()
        #: Held while a measurement is converted and sent, and by
        #: :meth:`detach` to wait for a send under way.
        self._send_lock = threading.Lock()
        #: Held while a measurement is converted into a message.
        self._convert_lock = threading.Lock()
        #: What the callback copied, tagged with its generation.
        self._received: "queue.Queue[Tuple[int, float, Any, Any, int, int]]" = (
            queue.Queue()
        )
        self._failures = _Failures(spec.name)
        #: With the wall clock, the messages for the device's thread, tagged
        #: with their generation (``None`` stops the thread).
        self._outbox: Optional["queue.Queue[Optional[Tuple[int, Any]]]"] = None
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
        self._ego = ego
        # Until it is the device's, a failure must not leave the sensor on the
        # ego, measuring for nobody: the caller only logs it (#1478).
        try:
            generation = self._next_generation()
            revolution: Optional[LidarRevolution] = None
            if spec.is_lidar:
                attributes = actor.attributes
                frequency = (
                    float(attributes["rotation_frequency"])
                    if "rotation_frequency" in attributes
                    else 1.0 / dt
                )
                # A revolution sweeps the field of view (360 degrees, CARLA's
                # default); the config refused a sensor_tick.
                fov = (
                    float(attributes["horizontal_fov"])
                    if "horizontal_fov" in attributes
                    else 360.0
                )
                revolution = LidarRevolution(frequency, dt, fov)
            with self._convert_lock:
                self._revolution = revolution
                self._last_start = None
            actor.listen(functools.partial(self._on_data, generation))
        except BaseException:
            _discard(actor, self.spec.name)
            raise
        self._actor = actor
        logger.info("%s: %s attached to ego %d", self.spec.name, spec.blueprint, ego.id)

    def _next_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _current(self, generation: int) -> bool:
        """Whether *generation* is the sensor's current one."""
        with self._generation_lock:
            return generation == self._generation

    def detach(self) -> None:
        """Remove the sensor; nothing it measured is sent once this returns."""
        # First, so that whatever is queued, and whatever a callback still
        # queues, is stale for every send that has not checked yet ...
        self._next_generation()
        # ... then wait for a conversion and a send that checked already.
        with self._convert_lock:
            self._revolution = None
        if self._send_lock.acquire(timeout=_SEND_WAIT_S):
            self._send_lock.release()
        else:
            logger.warning(
                "%s: a send still under way after %.0f s; detaching anyway",
                self.spec.name,
                _SEND_WAIT_S,
            )
        actor, self._actor = self._actor, None
        if actor is not None:
            # The ego (and its sensors) may be gone: not worth a warning.
            _discard(
                actor, self.spec.name, logging.DEBUG, "the sensor was gone already"
            )
        # Not needed for correctness (the generation is): what is stale need
        # not wait in memory for the next flush.
        _drain_queue(self._received)
        if self._outbox is not None:
            _drain_queue(self._outbox)

    def _on_data(self, generation: int, data: Any) -> None:
        """The sensor's listener, on a CARLA thread: copy, queue, return."""
        try:
            raw = np.frombuffer(data.raw_data, dtype=np.uint8).copy()
            width = int(getattr(data, "width", 0))
            height = int(getattr(data, "height", 0))
            arc = None
            if self._outbox is not None and self._revolution is not None:
                # The wall clock's moment of the measurement (give or take the
                # callback's delay) and how the sensor moved then.
                arc = (time.time(), self._sensor_motion(data))
            self._received.put(
                (generation, float(data.timestamp), arc, raw, width, height)
            )
        except Exception:  # noqa: BLE001 - a sensor thread must not die
            logger.exception("%s: failed to receive a measurement", self.spec.name)

    def _sensor_motion(self, data: Any) -> Tuple[float, ...]:
        """The sensor's velocity and angular velocity (its frame; m/s, rad/s) as *data*
        was measured.

        The ego's motion from the client's last snapshot (no call to the
        server), carried to the sensor's mount -- the ego origin's velocity
        plus ``w x r``, *r* the sensor's mount (its spawn offset on the ego)
        turned into world axes by the ego's rotation (#2499) -- and turned into
        the sensor's frame by the measurement's transform; still when it cannot
        be read.  *r* is not the measurement's location less the ego's: the
        snapshot may be a tick off the measurement, and the ego's travel in
        that tick would join the lever arm.
        """
        ego = self._ego
        if ego is None:
            return (0.0,) * 6
        try:
            v = ego.get_velocity()
            w = ego.get_angular_velocity()
            ego_rotation = np.asarray(
                ego.get_transform().get_matrix(), dtype=np.float64
            )[:3, :3]
            inverse = np.asarray(data.transform.get_inverse_matrix(), dtype=np.float64)[
                :3, :3
            ]
        except Exception:  # noqa: BLE001 - the ego may be gone; send the points as measured
            return (0.0,) * 6
        # CARLA's angular velocity is in deg/s, about the world axes.
        w_world = np.radians(np.array([w.x, w.y, w.z], dtype=np.float64))
        x, y, z = self.sensor_spec.transform[:3]
        lever = ego_rotation @ np.array([x, y, z], dtype=np.float64)
        # CARLA's axes are left-handed and its angular velocity follows the
        # left-hand rule, so the cross product's components are the usual
        # ones: a yaw rate (z) moves a mount ahead (+x) to the right (+y).
        v_world = np.array([v.x, v.y, v.z], dtype=np.float64) + np.cross(w_world, lever)
        vs = inverse @ v_world
        # A rotation turns the angular velocity as it does a vector.
        ws = inverse @ w_world
        return (*(float(c) for c in vs), *(float(c) for c in ws))

    def _message(self, measured: Tuple[float, Any, Any, int, int]) -> Optional[Any]:
        timestamp, arc, raw, width, height = measured
        spec = self.sensor_spec
        if self._revolution is not None:
            points = lidar_points(raw)
            if arc is not None:
                points = np.column_stack(
                    [points, np.full(len(points), self._next_arc, np.float32)]
                )
                self._arcs[self._next_arc] = arc
                self._next_arc += 1
            revolution = self._revolution.add(timestamp, points)
            if revolution is None:
                return None
            if arc is None:
                return ("scan", lidar_scan_fields(revolution, spec.intensity_scale))
            # Date the revolution by when CARLA measured it, not by its send, and
            # re-observe it as a turning sensor would (see spin_lidar_points).
            spun, start = self._spin(revolution)
            fields: Dict[str, Any] = dict(lidar_scan_fields(spun, spec.intensity_scale))
            fields["timestamp"] = start
            return ("scan", fields)
        frame = camera_frame(raw, width, height, spec.encoding)
        return self._hilspy.Message(data=frame)

    def _spin(self, revolution: np.ndarray) -> Tuple[np.ndarray, float]:
        """*revolution*'s points (tagged with their arc) as a spinning sensor's,
        and the time its first ray fires."""
        ids = revolution[:, 4].astype(np.int64)
        known = np.array(sorted(self._arcs), np.int64)
        index = np.searchsorted(known, ids)
        arcs = [self._arcs[int(k)] for k in known]
        measured = np.array([a[0] for a in arcs])[index]
        motion = np.array([a[1] for a in arcs])[index]
        period = self._revolution.period  # type: ignore[union-attr]
        rays = ray_times(revolution, period)
        # The start that centres the rays' times on the measuring times, kept
        # on the sensor's period.
        start = steady_revolution_start(
            float(np.mean(measured - rays)), self._last_start, period
        )
        self._last_start = start
        # Arcs before the last one in this revolution are sent whole.
        for k in [k for k in self._arcs if k < int(ids.max())]:
            del self._arcs[k]
        return spin_lidar_points(revolution, measured, motion, start, rays), start

    def _send(self, message: Any) -> None:
        try:
            if isinstance(message, tuple):
                self.channel.send(*message)
            else:
                self.channel.send(message)
        except Exception as exc:  # noqa: BLE001 - one device must not stop the others
            self._failures.report("send", exc)

    def flush(self) -> None:
        """Send what the attached sensor measured since the last flush; what
        a sensor detached since measured is dropped."""
        while True:
            try:
                generation, timestamp, arrived, raw, width, height = (
                    self._received.get_nowait()
                )
            except queue.Empty:
                return
            with self._convert_lock:
                if not self._current(generation):
                    continue
                try:
                    message = self._message((timestamp, arrived, raw, width, height))
                except Exception as exc:  # noqa: BLE001
                    self._failures.report("conversion", exc)
                    continue
            if message is None:
                continue
            if self._outbox is not None:
                put_dropping_oldest(self._outbox, (generation, message))
                continue
            self._send_current(generation, message)

    def _send_loop(self) -> None:
        assert self._outbox is not None
        while True:
            item = self._outbox.get()
            if item is None:
                return
            self._send_current(*item)

    def _send_current(self, generation: int, message: Any) -> None:
        """Send *message* unless its sensor was detached since it measured it."""
        with self._send_lock:
            if self._current(generation):
                self._send(message)

    def close(self) -> None:
        self.detach()
        if self._outbox is not None and self._thread is not None:
            put_dropping_oldest(self._outbox, None)
            self._thread.join(timeout=5.0)


def _discard(
    actor: Any,
    name: str,
    level: int = logging.WARNING,
    failure: str = "failed to destroy the sensor",
) -> None:
    """Stop and destroy a sensor *actor* (by default one that never became a
    device's); a failure is logged at *level*, as *failure*."""
    try:
        actor.stop()
        if actor.is_alive:
            actor.destroy()
    except Exception:  # noqa: BLE001 - the failure being raised matters more
        logger.log(level, "%s: %s", name, failure, exc_info=True)


def _drain_queue(items: "queue.Queue[Any]") -> None:
    """Empty *items* without waiting."""
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


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
        #: When the bridge was last without an ego from then on (it reached
        #: CARLA, or a scenario ended; wall clock), and whether that wait was
        #: warned about once it passed ``ego.wait`` (:meth:`_check_ego_wait`).
        self._without_ego_since = time.monotonic()
        self._without_ego_warned = False
        self._next_scan = 0.0
        self._started = False
        #: Whether a scenario runs (its scenario_start steps ran).
        self._active = False
        #: Whether a run is connected: then the run, not the ego appearing,
        #: starts scenarios.
        self._session = False
        #: The actor id of the last run's ego (its ``start``), while that
        #: actor lives.  Once the run is gone (it failed, or ended) its ego
        #: can stay in the world a while: it does not start a scenario of the
        #: bridge's own.  Forgotten once the actor is gone, so that an id
        #: CARLA hands out again is not ignored.
        self._run_ego: Optional[int] = None
        #: Whether the ECUs' commands go to the run (after its ``ready``).
        self._forwarding = False
        #: The ECUs' command fields merged, and how many times they changed
        #: (under :attr:`_command_lock`, which is never held while sending).
        self._command: Dict[str, float] = {}
        self._command_version = 0
        self._command_lock = threading.Lock()
        #: Serializes forwarding the command (:meth:`_forward`): whoever holds
        #: it sends the newest merged command, so the last one the run gets is
        #: the newest.  Taken before :attr:`_command_lock` and the link's own
        #: lock, never while holding either.
        self._forward_lock = threading.Lock()
        #: The :attr:`_command_version` the run was last sent (under
        #: :attr:`_forward_lock`).
        self._forwarded_version = -1
        #: What the scenario stops (:meth:`disconnect`), by cut id.
        self._cuts: Dict[str, _Cut] = {}
        #: What each channel side was last set to stop (:meth:`_apply_cuts`):
        #: these names (none when absent), or everything (``None``).
        self._applied: Dict[Tuple[str, Side], Optional[FrozenSet[str]]] = {}
        #: The channel sides whose last change failed: what they stop is not
        #: known, so :meth:`_apply_cuts` sets them again.
        self._unsettled: Set[Tuple[str, Side]] = set()
        self._stop = threading.Event()

    # -- set-up and shut-down ------------------------------------------------

    def open_devices(self) -> None:
        """Open every device's channel, and receive on those that need it.

        Raises:
            HilsOpenError: A device's channel could not be opened (hilspy's
                error; an ``OSError``: a missing interface or file; a
                ``TypeError`` or ``ValueError``: an option hilspy refuses);
                the devices opened before it stay open, for :meth:`close`.
        """
        waited: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
        for phase in self.config.lifecycle.values():
            for step in phase:
                if step.kind == "wait" and step.name is not None:
                    waited.setdefault(step.device, []).append((step.name, step.match))
        for spec in self.config.devices:
            options: Dict[str, Any] = dict(spec.options)
            if self.clock is not None:
                options["clock"] = self.clock
            try:
                channel = self._open(spec.url, spec.codec, **options)
            except (self._hilspy.Error, OSError, TypeError, ValueError) as exc:
                raise HilsOpenError(f"{spec.name}: {spec.url}: {exc}") from exc
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
                self._receivers[spec.name] = _Receiver(
                    spec, channel, self._on_input, waited.get(spec.name, ())
                )

    def startup(self) -> None:
        """Listen for runs, open the devices and run the ``startup`` steps,
        then serve the runs.

        The link's address is taken first, so that one the bridge cannot
        listen at fails before anything reaches the ECUs (nor do the
        ``shutdown`` steps run then: no startup step did, #2277).  So does a
        device that cannot be opened: the devices opened before it are closed
        again, and the bridge is not started (#2557).

        Raises:
            HilsListenError: The bridge cannot listen at ``runner.listen``.
            HilsOpenError: A device's channel could not be opened.
            HilsStepError: A startup step failed.
        """
        if self._started:
            return
        address = self.config.runner_listen
        if address and self._link is None:
            try:
                self._link = RunnerLink(address, self._events)
            except OSError as exc:
                raise HilsListenError(
                    f"runner.listen {address}: {exc.strerror or exc}"
                ) from exc
        try:
            self.open_devices()
        except BaseException:
            # Nothing was sent: what opened closes again, no shutdown step.
            self._close_all()
            raise
        self._started = True
        self._run_steps("startup", strict=True)
        if self._link is not None:
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
        if not self._active:
            self._wait_for_ego()

    def close(self) -> None:
        """End a running scenario, run the ``shutdown`` steps (once started),
        close everything.

        Everything is closed even when ending the scenario or a step fails
        (#2557); such a failure is raised after."""
        try:
            try:
                if self._active:
                    self._finish()
            finally:
                if self._started:
                    self._run_steps("shutdown", strict=False)
        finally:
            # A scenario whose end failed is over all the same: a second
            # close() does not end it again on closed devices.
            self._active = False
            self._ego = None
            self._started = False
            self._close_all()

    def _close_all(self) -> None:
        """Close the link, the receivers, the sensors, the periodic messages and
        the channels -- each, even when one before it fails (logged)."""
        link, self._link = self._link, None
        closers: List[Tuple[str, Callable[[], None]]] = []
        if link is not None:
            closers.append(("the run link", link.close))
        closers += [(name, r.close) for name, r in self._receivers.items()]
        closers += [(s.spec.name, s.close) for s in self._sensors]
        closers += [(m.spec.name, m.stop) for m in self._messages]
        closers += [(name, c.close) for name, c in self._channels.items()]
        self._receivers.clear()
        self._sensors.clear()
        self._messages.clear()
        self._channels.clear()
        for name, close in closers:
            try:
                close()
            except Exception:  # noqa: BLE001 - the others must still close
                logger.exception("%s: failed to close", name)

    # -- lifecycle steps -----------------------------------------------------

    def _run_steps(self, phase: str, strict: bool) -> None:
        """Run *phase*'s steps; the first failure raises when *strict*, else
        each is logged -- an unexpected error too -- and the rest still run.  The ``warmup`` steps follow
        the world while they wait (:meth:`_pump`).

        Once the bridge is stopping (:meth:`stop`), a step that waits (a
        ``wait`` or a ``sleep``) ends at once, as a failure of its own
        (:class:`_Stopping`), but in the ``shutdown`` phase, whose steps run
        whole: the bridge is stopping then."""
        live = phase == "warmup"
        stoppable = phase != "shutdown"
        steps = self.config.steps(phase)
        if not steps:
            return
        logger.info("%s: %d step(s)", phase, len(steps))
        since = time.monotonic()
        try:
            for step in steps:
                try:
                    self._run_step(step, since, live, stoppable)
                except _Stopping as exc:
                    if strict:
                        raise _Stopping(f"{phase}: {exc}") from exc
                    logger.info("%s: %s skipped (stopping)", phase, step.describe())
                except HilsStepError as exc:
                    if strict:
                        raise HilsStepError(f"{phase}: {exc}") from exc
                    logger.warning("%s: %s", phase, exc)
                except Exception:
                    if strict:
                        raise
                    # Not a step's own failure: the other steps still run.
                    logger.exception("%s: %s failed", phase, step.describe())
        finally:
            if not live:
                # The clock followed the wall clock meanwhile: the next tick
                # must not add the world time that passed as well.
                self._last_time = None

    def _run_step(
        self,
        step: StepSpec,
        since: float,
        live: bool = False,
        stoppable: bool = True,
    ) -> None:
        logger.info("  %s", step.describe())
        if step.kind == "sleep":
            self._pump(lambda: False, step.seconds, live, stoppable)
            return
        if step.kind == "wait":
            receiver = self._receivers.get(step.device)
            if receiver is None:
                raise HilsStepError(f"{step.describe()}: {step.device} is not open")
            assert step.name is not None
            name: str = step.name
            if not self._pump(
                lambda: receiver.seen(name, step.match, since),
                step.seconds,
                live,
                stoppable,
            ):
                expected = f" with {dict(step.match)}" if step.match else ""
                raise HilsStepError(
                    f"{step.device} sent no {step.name}{expected} within {step.seconds} s"
                )
            return
        channel = self._channels.get(step.device)
        if channel is None:
            raise HilsStepError(f"{step.describe()}: {step.device} is not open")
        try:
            if step.name is None:
                channel.send(self._hilspy.Message(data=step.data, id=step.frame_id))
            else:
                values = {k: v.resolve(self._signals) for k, v in step.fields.items()}
                channel.send(step.name, values)
        except Exception as exc:  # noqa: BLE001 - hilspy.Error, or a missing signal
            raise HilsStepError(f"{step.describe()} failed: {exc}") from exc

    def _pump(
        self,
        done: Callable[[], bool],
        seconds: float,
        live: bool = False,
        stoppable: bool = True,
    ) -> bool:
        """Wait up to *seconds* for *done*; the simulated clock follows the wall
        clock meanwhile, or, *live*, the world's ticks are served
        (:meth:`step`).  Whether *done* came true.

        Raises:
            HilsStepError: *live*, and the scenario ended meanwhile (its ego
                went away).
            _Stopping: *stoppable*, and the bridge is stopping (:meth:`stop`):
                a SIGTERM does not wait out a long step (#2277).
        """
        start = last = time.monotonic()
        while True:
            if done():
                return True
            if stoppable and self._stop.is_set():
                raise _Stopping("the bridge is stopping")
            now = time.monotonic()
            if now - start >= seconds:
                return False
            if live:
                self.step(min(_TICK_POLL_S, seconds - (now - start)))
                if not self._active:
                    raise HilsStepError("the scenario ended")
                continue
            pause = min(_STEP_POLL_S, seconds - (now - start))
            if stoppable:
                self._stop.wait(pause)
            else:
                time.sleep(pause)
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
        """Start a scenario for *ego*: periodic messages, the
        ``scenario_start`` steps, the sensors, then the ``warmup`` steps.

        The sensors are attached once the steps are over, failed or not:
        nothing a sensor measures is sent until the scenario is live (the
        next :meth:`step` flushes it), and the steps can wait for seconds.
        A sensor attached before them would queue all it measured meanwhile
        -- unbounded -- for the first flush to send stale.  Nothing in a step
        needs a sensor.

        The ``warmup`` steps wait on what the ECUs make of the sensors, so
        they run with the world followed (:meth:`step`), as a scenario runs.

        No channel stops data when a scenario begins: what a previous
        scenario's cuts stopped is carried again first -- every channel side,
        those an earlier end failed to restore included (#2444).

        Raises:
            HilsStepError: A channel still stops data, or CARLA cannot be
                read (its map, its snapshot: the server is down or
                restarting, #2557) -- the scenario does not start -- or a
                scenario_start or warmup step failed (the scenario is still
                running, and ends as any other).
        """
        assert self._world is not None
        try:
            self._reconnect_all()
        except Exception as exc:  # noqa: BLE001 - refused, as a failed step
            raise HilsStepError(
                "the data a previous scenario cut cannot be carried again "
                f"({', '.join(self._still_cut())}: {exc}); the scenario does "
                "not start"
            ) from exc
        # Read before the scenario is active: a CARLA that fails here leaves
        # no scenario the run (or the bridge) waits on.
        try:
            self._map = self._world.get_map()
            snapshot = self._world.get_snapshot()
            self._ego = ego
            state = self._state(snapshot)
        except RuntimeError as exc:
            self._ego = None
            raise HilsStepError(
                f"CARLA cannot be read ({exc}); the scenario does not start"
            ) from exc
        self._active = True
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
        dt = float(snapshot.delta_seconds) or FIXED_DELTA_SECONDS
        try:
            self._run_steps("scenario_start", strict=True)
        finally:
            # A failed step leaves the scenario running: it needs its sensors.
            for sensor_device in self._sensors:
                try:
                    sensor_device.attach(self._carla, self._world, ego, dt)
                except Exception:
                    logger.exception(
                        "%s: failed to attach the sensor", sensor_device.spec.name
                    )
        self._run_steps("warmup", strict=True)

    def _finish(self) -> None:
        """End the scenario: remove the sensors, carry what its cuts stopped
        again, so that the ``scenario_end`` steps reach the ECUs (and hear
        back from them), run those steps, then stop the periodic messages.

        The sensors go first, as :meth:`_begin` attaches them last: the steps
        can wait for seconds, during which nothing is flushed, and a sensor
        still listening meanwhile would queue all it measured -- unbounded --
        only to be dropped when it is removed.  Nothing in a step needs a
        sensor, and nothing it measured is sent once it is removed."""
        with self._forward_lock:
            # A command being sent finishes first: none follows ``ended``.
            self._forwarding = False
        for sensor_device in self._sensors:
            try:
                sensor_device.detach()
            except Exception:  # noqa: BLE001 - the scenario must still end
                logger.exception(
                    "%s: failed to remove the sensor", sensor_device.spec.name
                )
        try:
            self._reconnect_all()
        except Exception:  # noqa: BLE001 - the scenario must still end
            logger.exception("failed to restore the cut data")
        self._run_steps("scenario_end", strict=False)
        for message_device in self._messages:
            message_device.stop()
        with self._command_lock:
            self._command = {}
            self._command_version += 1
        self._ego = None
        self._active = False
        self._wait_for_ego()
        logger.info("scenario ended")

    def _wait_for_ego(self) -> None:
        """Start a wait without an ego, from now."""
        self._without_ego_since = time.monotonic()
        self._without_ego_warned = False

    def _check_ego_wait(self) -> None:
        """Warn, once a wait, when no scenario's ego came for ``ego.wait`` s
        (wall clock, as the run's own timeouts); the bridge keeps waiting."""
        if self._active or self._without_ego_warned:
            return
        waited = time.monotonic() - self._without_ego_since
        if waited < self.config.ego_wait:
            return
        self._without_ego_warned = True
        logger.warning(
            "no ego %r for %.0f s (ego.wait); still waiting for one",
            self.config.role_name,
            waited,
        )

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
            ValueError: No scenario runs, no such device or direction,
                messages that are not a list of names or name none, or the
                device's channel refused the cut (which is then not made).
        """
        if not self._active:
            raise ValueError("no scenario runs")
        refusal = cut_refusal(direction, device, list(self._channels))
        if refusal is not None:
            raise ValueError(refusal)
        if messages is not None and (
            isinstance(messages, str) or not isinstance(messages, (list, tuple))
        ):
            raise ValueError("messages must be a list of names")
        names = None if messages is None else frozenset(str(m) for m in messages)
        refusal = messages_refusal(names)
        if refusal is not None:
            raise ValueError(refusal)
        saved = dict(self._cuts)
        self._cuts[cut_id] = _Cut(device, names, _DIRECTION_SIDES[direction])
        try:
            self._apply_cuts()
        except self._hilspy.Error as exc:
            # Undone, so the cut does not fail every later change of the cuts.
            self._cuts = saved
            try:
                self._apply_cuts()
            except Exception:  # noqa: BLE001 - the run must still hear of it
                # A side that still fails (this one, or one an earlier change
                # left unsettled) is set again by the next change (#2444).
                logger.exception("failed to undo the refused cut")
            raise ValueError(str(exc)) from exc
        logger.warning(
            "%s: %s stopped (%s)",
            device or "every device",
            "every message" if names is None else ", ".join(sorted(names)),
            direction,
        )

    def reconnect(self, cut_id: str) -> None:
        """Carry what the cut *cut_id* stopped again (nothing if there is none)."""
        if self._cuts.pop(cut_id, None) is not None:
            logger.info("cut %r restored", cut_id)
            self._apply_cuts()

    def _apply_cuts(self) -> None:
        """Set every device's channel to stop what its cuts stop together,
        changing only the sides whose combined cut changed -- or whose last
        change failed, which is tried again.

        Every side is set, even when one before it fails (#2444): a channel
        left cut would stop its data in the scenarios that follow.  A side
        that fails is set again by the next call.

        Raises:
            Exception: The first side's failure, once every side was tried
                (the others are logged).
        """
        cuts = list(self._cuts.values())
        errors: List[Exception] = []
        for name, channel in self._channels.items():
            for side in _SIDES:
                # Nothing (an empty set), everything (None), or these names.
                stopped = stopped_messages(cuts, name, side)
                if (name, side) not in self._unsettled and self._applied.get(
                    (name, side), frozenset()
                ) == stopped:
                    continue
                try:
                    if stopped is None:
                        channel.disconnect(direction=side)
                    elif stopped:
                        channel.disconnect(direction=side, messages=sorted(stopped))
                    else:
                        channel.reconnect(direction=side)
                except Exception as exc:  # noqa: BLE001 - the others still
                    # What the side stops is not known: set it next time.
                    self._unsettled.add((name, side))
                    if errors:
                        logger.error(
                            "%s: failed to set what is stopped (%s): %s",
                            name,
                            side,
                            exc,
                        )
                    errors.append(exc)
                    continue
                self._unsettled.discard((name, side))
                self._applied[(name, side)] = stopped
        if errors:
            raise errors[0]

    def _reconnect_all(self) -> None:
        """Carry everything again: every channel side, even when one fails.

        Raises:
            Exception: A channel side failed to carry its data again
                (:meth:`_apply_cuts`); the others were.
        """
        if self._cuts:
            self._cuts.clear()
            logger.info("every cut restored")
        self._apply_cuts()

    def _still_cut(self) -> List[str]:
        """The channel sides that may still stop data (``"device (side)"``):
        those set to stop some, and those whose last change failed."""
        return [
            f"{name} ({side})"
            for name, side in sorted(set(self._applied) | self._unsettled)
            if (name, side) in self._unsettled
            or self._applied[(name, side)] != frozenset()
        ]

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
            self._command_version += 1
        self._forward()

    def _forward(self, start: bool = False) -> None:
        """Send the run the newest merged command, unless it has it already;
        with *start*, start forwarding first (after ``ready``).

        The receiving threads of several devices call this at once: a thread
        merged its fields, but another may send before it does.  So the
        command is copied while :attr:`_forward_lock` is held, right before
        the send, and a thread whose fields went out with another's newer copy
        sends nothing -- the last command sent is always the newest.
        :attr:`_command_lock` is not held while sending: a slow link must not
        hold up merging, :attr:`command` or :meth:`_finish`.
        """
        with self._forward_lock:
            if start:
                self._forwarding = True
            link = self._link
            if not self._forwarding or link is None:
                return
            with self._command_lock:
                version = self._command_version
                if version == self._forwarded_version or not self._command:
                    return
                command = dict(self._command)
            link.send({"type": "command", "mode": self.config.command_mode, **command})
            self._forwarded_version = version

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
            try:
                self.disconnect(
                    str(message.get("id", "")),
                    None if device is None else str(device),
                    message.get("messages"),
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
        ego_id = message.get("ego_id")
        self._run_ego = ego_id if isinstance(ego_id, int) else None
        try:
            if self._client is not None:
                # The run may have loaded another map: a new world.
                self._world = self._client.get_world()
            assert self._world is not None
            ego = (
                self._world.get_actor(int(ego_id)) if isinstance(ego_id, int) else None
            )
        except RuntimeError as exc:  # the server is down or restarting (#2557)
            logger.error("CARLA cannot be read: %s", exc)
            link.send({"type": "error", "message": f"CARLA cannot be read: {exc}"})
            return
        if ego is None:
            link.send({"type": "error", "message": f"no ego with actor id {ego_id}"})
            return
        try:
            self._begin(ego, str(message.get("scenario", "")))
        except HilsStepError as exc:
            # A stop (SIGTERM) is no step failure: the run is still told.
            logger.log(
                logging.INFO if isinstance(exc, _Stopping) else logging.ERROR,
                "%s",
                exc,
            )
            link.send({"type": "error", "message": str(exc)})
            return
        link.send({"type": "ready"})
        # Through _forward, so that this command does not overtake a newer
        # one a receiving thread forwards as soon as forwarding starts.
        self._forward(start=True)

    def _find_ego(self) -> None:
        """Without a run connected, the ego appearing starts a scenario -- not
        the ego of a run that went away."""
        now = time.monotonic()
        if now < self._next_scan:
            return
        self._next_scan = now + _EGO_SCAN_INTERVAL_S
        assert self._world is not None
        try:
            ego = find_actor_in_world(self._world, self.config.role_name)
        except RuntimeError as exc:  # the server is down or restarting (#2557)
            logger.debug("no actors to look for the ego in: %s", exc)
            return
        if ego is None or ego.id == self._run_ego:
            return
        try:
            self._begin(ego, "")
        except HilsStepError as exc:
            logger.log(
                logging.INFO if isinstance(exc, _Stopping) else logging.ERROR,
                "%s",
                exc,
            )

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
                if self._refresh_world():
                    logger.info("the ego is gone")
                    self._finish()
            self._check_ego_wait()
            return False
        self._last_tick = time.monotonic()
        if not self._active and not self._session:
            if self._run_ego is not None and snapshot.find(self._run_ego) is None:
                self._run_ego = None
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
        self._check_ego_wait()
        return True

    def _refresh_world(self) -> bool:
        """After a quiet period, take the server's world anew (the runner may
        have loaded another map); whether the scenario's ego is gone.

        A server that cannot be reached (down, or restarting) is still waited
        for (#2557): the world held is kept, the ego taken to be there, and
        the next quiet period tries again."""
        try:
            if self._client is not None:
                self._world = self._client.get_world()
            return self._ego is not None and not self._ego.is_alive
        except RuntimeError as exc:
            logger.debug("CARLA cannot be reached: %s; still waiting", exc)
            return False

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

        A :meth:`stop` during the startup steps ends them (:meth:`_pump`), and
        the bridge returns without reaching CARLA; :meth:`close` still runs
        the ``shutdown`` steps.

        Once it is up, a CARLA server that goes down or restarts is waited
        for again (#2557): the quiet period's refresh of the world, and the
        scans for an ego, try again later.

        Raises:
            HilsListenError: The bridge cannot listen at ``runner.listen``.
            HilsOpenError: A device's channel could not be opened.
            HilsStepError: A startup step failed.
        """
        try:
            self.startup()
        except _Stopping:
            logger.info("stopped during the startup steps")
            return
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
        """Have :meth:`run` return after the tick it is serving; a lifecycle
        step waiting meanwhile (but a ``shutdown`` one) ends at once."""
        self._stop.set()
