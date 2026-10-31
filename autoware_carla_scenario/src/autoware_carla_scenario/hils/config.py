"""The HILS bridge's configuration: which CARLA, which ego, which devices.

One YAML file says everything (``scenario-hils --example`` prints a commented
one)::

    carla: {host: localhost, port: 2000, timeout: 10.0}
    ego: {role_name: Ego, wait: 120.0}
    canvas: canvas.yaml          # optional: carla_canvas renders the sensors
    clock: simulation            # or wall
    devices:
      vehicle_can:               # periodic messages filled from the ego's signals
        url: can://vcan0
        codec: dbc:vehicle.dbc
        messages:
          - name: VehicleSpeed
            period: 0.01         # omitted: the DBC's GenMsgCycleTime
            fields: {Speed: $speed_kmh, Gear: D}
      lidar_top:                 # a sensor on the ego, each measurement forwarded
        url: udp://192.168.1.201:2368
        codec: lidar:pandar_xt32
        sensor:
          blueprint: sensor.lidar.ray_cast
          transform: {z: 2.0}
          attributes: {channels: 32, rotation_frequency: 10}

A device is a hilspy channel (``url`` + ``codec`` + ``options``, as
``hilspy.open`` takes them) with ``messages`` (periodic, while a scenario runs)
and ``inputs`` (ECU messages that carry the command driving the ego), or a
``sensor``.  A field is a constant, ``$signal`` (one of
:data:`~.signals.SIGNALS`) or ``{signal: name, scale: k, offset: c}``; ``$$``
starts a constant beginning with ``$``.  ``runner.listen`` is where runs with
``ego.entity=hils`` reach the bridge, ``commands.mode`` how the ECUs' command
drives the ego, and ``lifecycle`` the steps run at :data:`PHASES`.  Relative files (the configuration's ``canvas``, a codec's file,
a ``profiles`` option) are relative to the configuration file.  Unknown keys
are refused, so a typo fails loudly instead of being ignored.

Names, addresses and raw bytes are text, and a value YAML reads as something
else is refused rather than turned back into text: unquoted, ``0102`` is the
octal number 66, ``0x10`` is 16 and ``On`` is ``True``, and what was written is
lost.  Quote them (``data: "02 10 01"``, ``fields: {"On": 1}``).  Message,
input and step names also take a whole number (RTCM's ``1005``).
:func:`load_config` further refuses, anywhere in the file, the unquoted YAML
1.1 forms that do not mean what they look like (octal ``0102``, sexagesimal
``1:30``, ``yes``/``no``/``on``/``off``), so ``Gear: Off`` in a field value is
refused too instead of becoming ``False``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..constants import EGO_ROLE_NAME
from .measurements import CAMERA_ENCODINGS
from .protocol import COMMAND_FIELDS, COMMAND_MODES, DEFAULT_PORT, command_value
from .signals import SIGNALS

logger = logging.getLogger(__name__)

#: A command field an ECU keeps sending non-finite is logged the first time and
#: then once in this many.
_NON_FINITE_LOG_EVERY = 100

__all__ = [
    "CLOCKS",
    "PHASES",
    "DeviceSpec",
    "FieldSource",
    "HilsConfig",
    "HilsConfigError",
    "InputSource",
    "InputSpec",
    "MessageSpec",
    "SensorSpec",
    "StepSpec",
    "load_config",
    "parse_config",
]

#: A constant field value.
Constant = Union[str, int, float, bool]

#: What the devices' time is:
#:
#: * ``"simulation"`` -- a ``hilspy.Clock`` advanced by the world's ticks:
#:   periodic messages and a LiDAR's packets go out as simulated time passes
#:   them, so a world slower (or faster) than real time stays consistent.
#: * ``"wall"`` -- real time: periodic messages from hilspy's threads.
CLOCKS: Tuple[str, ...] = ("simulation", "wall")

#: The sensors a device can carry: a ray-cast LiDAR (with a ``lidar:``
#: codec) and the cameras (whose frames are BGRA images).
LIDAR_BLUEPRINT = "sensor.lidar.ray_cast"
CAMERA_PREFIX = "sensor.camera."

_TRANSFORM_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")

#: When the bridge runs a lifecycle's steps:
#:
#: * ``"startup"`` -- once, when the bridge starts (its devices just opened).
#: * ``"scenario_start"`` -- when a scenario starts: a run's ``start``, or the
#:   ego appearing.  The run waits for them (the ECUs being initialized).
#: * ``"scenario_end"`` -- when the scenario ends: a run's ``end``, the run
#:   going away, or the ego disappearing.
#: * ``"shutdown"`` -- once, when the bridge stops (Ctrl-C, SIGTERM).
PHASES: Tuple[str, ...] = ("startup", "scenario_start", "scenario_end", "shutdown")

#: The phases with no ego, whose steps cannot use a signal.
_EGOLESS_PHASES = ("startup", "shutdown")


class HilsConfigError(ValueError):
    """The configuration is not one the bridge can run."""


@dataclass(frozen=True)
class FieldSource:
    """Where one field's value comes from: a constant, or a scaled signal."""

    signal: Optional[str] = None
    constant: Optional[Constant] = None
    scale: float = 1.0
    offset: float = 0.0

    def resolve(self, signals: Mapping[str, float]) -> Constant:
        """The field's value, given the ego's *signals*."""
        if self.signal is None:
            assert self.constant is not None
            return self.constant
        return signals[self.signal] * self.scale + self.offset


@dataclass(frozen=True)
class MessageSpec:
    """A message a device sends periodically.

    Attributes:
        name: The codec's message name (a DBC message, ``PVTGeodetic``, ``GGA``).
        fields: Each field's source.
        period: Seconds between sends; ``None`` takes the message's cycle time
            from the network description (``dbc:`` / ``fibex:``).
    """

    name: str
    fields: Mapping[str, FieldSource]
    period: Optional[float] = None

    def values(self, signals: Mapping[str, float]) -> Dict[str, Constant]:
        """The fields' values, given the ego's *signals*."""
        return {name: source.resolve(signals) for name, source in self.fields.items()}


@dataclass(frozen=True)
class InputSource:
    """Where one command field comes from: an ECU message's field (scaled), or
    a constant."""

    field: Optional[str] = None
    constant: Optional[float] = None
    scale: float = 1.0
    offset: float = 0.0

    def resolve(self, fields: Mapping[str, Any]) -> Optional[float]:
        """The command field's value from a decoded message's *fields*;
        ``None`` when the message lacks the field, it is not a number, or it
        (scaled) is not finite (NaN, an infinity: a decoded IEEE-float signal,
        an overflowing scale) or too large for CARLA's 32-bit floats
        (:func:`~.protocol.command_value`), which CARLA must not get."""
        value = self.scaled(fields)
        return None if value is None else command_value(value)

    def scaled(self, fields: Mapping[str, Any]) -> Optional[float]:
        """:meth:`resolve`, but a non-finite or out-of-range value is returned
        (not ``None``), for the caller to report; a number too large for a
        float is ``inf``."""
        if self.field is None:
            return self.constant
        value = fields.get(self.field)
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except OverflowError:
            number = math.inf if value > 0 else -math.inf
        return number * self.scale + self.offset


@dataclass(frozen=True)
class InputSpec:
    """An ECU message that carries (part of) the command driving the ego.

    Attributes:
        name: The message's name, as the device's codec decodes it.
        command: Each command field (:data:`~.protocol.COMMAND_FIELDS` of the
            configuration's mode) and where it comes from.
    """

    name: str
    command: Mapping[str, InputSource]
    #: How many non-finite values each command field has dropped (the log's
    #: rate limit).
    _dropped: Dict[str, int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def values(self, fields: Mapping[str, Any]) -> Dict[str, float]:
        """The command fields a decoded message's *fields* give.

        A field whose value is not finite, or too large for CARLA's 32-bit
        floats, is left out, as a missing one is:
        the ego keeps the field's previous value.  It is logged the first time
        and then once in a hundred."""
        values: Dict[str, float] = {}
        for key, source in self.command.items():
            value = source.scaled(fields)
            if value is None:
                continue
            number = command_value(value)
            if number is None:
                count = self._dropped.get(key, 0)
                if count % _NON_FINITE_LOG_EVERY == 0:
                    logger.warning(
                        "%s: dropped a non-finite or out-of-range %s "
                        "(%r from %s; %d so far)",
                        self.name,
                        key,
                        value,
                        source.field,
                        count + 1,
                    )
                self._dropped[key] = count + 1
                continue
            values[key] = number
        return values


@dataclass(frozen=True)
class StepSpec:
    """One step of a lifecycle phase.

    Attributes:
        kind: ``"send"`` (a message, once), ``"wait"`` (for an ECU's message
            whose fields match, up to :attr:`timeout`) or ``"sleep"``.
        device: The device sent to or waited on.
        name: The message's name; ``None`` for a raw frame (:attr:`data`).
        fields: What a ``send`` fills the message with.
        match: The field values a ``wait`` waits for.
        data: A raw frame's bytes, sent as is.
        frame_id: A raw frame's identifier (CAN).
        seconds: A ``sleep``'s length, or a ``wait``'s timeout.
    """

    kind: str
    device: str = ""
    name: Optional[str] = None
    fields: Mapping[str, FieldSource] = field(default_factory=dict)
    match: Mapping[str, Constant] = field(default_factory=dict)
    data: Optional[bytes] = None
    frame_id: Optional[int] = None
    seconds: float = 0.0

    def describe(self) -> str:
        if self.kind == "sleep":
            return f"sleep {self.seconds} s"
        what = self.name if self.name is not None else f"frame {self.frame_id}"
        if self.kind == "wait":
            return f"wait for {what} from {self.device} (up to {self.seconds} s)"
        return f"send {what} to {self.device}"


@dataclass(frozen=True)
class SensorSpec:
    """A CARLA sensor the bridge attaches to the ego, forwarding what it measures.

    Attributes:
        blueprint: ``sensor.lidar.ray_cast`` or a ``sensor.camera.*``.
        transform: ``(x, y, z, roll, pitch, yaw)`` relative to the ego, in
            CARLA's convention (metres, degrees; y right).
        attributes: Blueprint attributes, as strings.
        encoding: A camera frame's encoding (:data:`CAMERA_ENCODINGS`).
        intensity_scale: A LiDAR's intensity (0 to 1) times this is the
            reflectivity byte sent.
    """

    blueprint: str
    transform: Tuple[float, float, float, float, float, float] = (0.0,) * 6
    attributes: Mapping[str, str] = field(default_factory=dict)
    encoding: str = "bgra"
    intensity_scale: float = 255.0

    @property
    def is_lidar(self) -> bool:
        return self.blueprint == LIDAR_BLUEPRINT


@dataclass(frozen=True)
class DeviceSpec:
    """One hilspy channel toward a device under test.

    Attributes:
        name: The device's name (logs; a sensor's ``role_name``, which
            carla_canvas's ``sensors`` rules can match).
        url: hilspy's transport URL.
        codec: hilspy's codec (``raw`` when omitted).
        options: hilspy's codec options.
        messages: Periodic messages, sent while a scenario runs.
        sensor: The sensor whose measurements the device is sent.
        inputs: The ECU messages received on it that drive the ego.
    """

    name: str
    url: str
    codec: str = "raw"
    options: Mapping[str, Any] = field(default_factory=dict)
    messages: Tuple[MessageSpec, ...] = ()
    sensor: Optional[SensorSpec] = None
    inputs: Tuple[InputSpec, ...] = ()


@dataclass(frozen=True)
class HilsConfig:
    """The whole configuration (see the module's documentation)."""

    devices: Tuple[DeviceSpec, ...]
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    role_name: str = str(EGO_ROLE_NAME)
    ego_wait: float = 120.0
    canvas: Optional[Path] = None
    clock: str = "simulation"
    #: Where runs reach the bridge (``host:port``); ``""`` takes no run.
    runner_listen: str = f"0.0.0.0:{DEFAULT_PORT}"
    #: How the ECUs' command drives the ego (:data:`~.protocol.COMMAND_MODES`).
    command_mode: str = "control"
    #: Each phase's steps (:data:`PHASES`).
    lifecycle: Mapping[str, Tuple[StepSpec, ...]] = field(default_factory=dict)

    def steps(self, phase: str) -> Tuple[StepSpec, ...]:
        """The steps of *phase*."""
        return tuple(self.lifecycle.get(phase, ()))


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HilsConfigError(f"{where}: expected a mapping, got {value!r}")
    return value


def _only(section: Mapping[str, Any], allowed: Tuple[str, ...], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise HilsConfigError(
            f"{where}: unknown key(s) {', '.join(map(str, unknown))}; "
            f"expected {', '.join(allowed)}"
        )


def _text(value: Any, where: str, *, number: bool = False) -> str:
    """*value*, which must be a string: YAML reads an unquoted ``0102``,
    ``0x10``, ``1.10`` or ``On`` as a number or a bool, whose text is lost.

    With *number*, an integer is also taken, as its decimal text: a message
    name can be a number (RTCM's ``1005``), and :func:`load_config` has
    already refused the YAML 1.1 forms (octal ``0102``) whose number is not
    the one written.
    """
    if isinstance(value, str):
        return value
    if number and isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise HilsConfigError(
        f"{where}: expected text, got the {type(value).__name__} {value!r}; "
        "YAML reads an unquoted number or yes/no/on/off as that type, so quote it "
        '(e.g. "0102")'
    )


def _keys(section: Mapping[Any, Any], where: str) -> Dict[str, Any]:
    """*section* with its keys checked to be strings (:func:`_text`)."""
    for key in section:
        if not isinstance(key, str):
            _text(key, f"{where}: key {key!r}")
    return dict(section)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HilsConfigError(f"{where}: expected a number, got {value!r}")
    try:
        return float(value)
    except OverflowError:
        raise HilsConfigError(
            f"{where}: expected a number a float can hold, got {value!r}"
        ) from None


def _finite(value: Any, where: str) -> float:
    number = _number(value, where)
    if not math.isfinite(number):
        raise HilsConfigError(f"{where}: expected a finite number, got {value!r}")
    return number


def _seconds(value: Any, where: str, *, zero: bool = False) -> float:
    """A duration: a finite number, positive (or, with *zero*, not negative)."""
    seconds = _finite(value, where)
    if seconds < 0.0 or (seconds == 0.0 and not zero):
        raise HilsConfigError(
            f"{where}: must {'not be negative' if zero else 'be positive'}, "
            f"got {value!r}"
        )
    return seconds


def _port(value: Any, where: str) -> int:
    """A TCP port: an integer (not a bool) in 1..65535."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise HilsConfigError(f"{where}: expected an integer port, got {value!r}")
    if not 1 <= value <= 65535:
        raise HilsConfigError(f"{where}: must be within 1-65535, got {value!r}")
    return value


def _signal(name: str, where: str) -> str:
    if name not in SIGNALS:
        raise HilsConfigError(
            f"{where}: unknown signal {name!r}; signals: {', '.join(SIGNALS)}"
        )
    return name


def _field(value: Any, where: str) -> FieldSource:
    if isinstance(value, Mapping):
        _only(value, ("signal", "scale", "offset"), where)
        if "signal" not in value:
            raise HilsConfigError(f"{where}: a mapping names a signal")
        return FieldSource(
            signal=_signal(_text(value["signal"], f"{where}.signal"), where),
            scale=_finite(value.get("scale", 1.0), f"{where}.scale"),
            offset=_finite(value.get("offset", 0.0), f"{where}.offset"),
        )
    if isinstance(value, str):
        if value.startswith("$$"):
            return FieldSource(constant=value[1:])
        if value.startswith("$"):
            return FieldSource(signal=_signal(value[1:], where))
        return FieldSource(constant=value)
    if isinstance(value, (bool, int, float)):
        return FieldSource(constant=value)
    raise HilsConfigError(
        f"{where}: expected a constant, $signal or {{signal: ...}}, got {value!r}"
    )


def _message(value: Any, where: str) -> MessageSpec:
    section = _mapping(value, where)
    _only(section, ("name", "period", "fields"), where)
    if not section.get("name"):
        raise HilsConfigError(f"{where}: a message has a name")
    period = section.get("period")
    if period is not None:
        period = _seconds(period, f"{where}.period")
    fields = _keys(
        _mapping(section.get("fields"), f"{where}.fields"), f"{where}.fields"
    )
    return MessageSpec(
        name=_text(section["name"], f"{where}.name", number=True),
        period=period,
        fields={k: _field(v, f"{where}.fields.{k}") for k, v in fields.items()},
    )


def _sensor(value: Any, codec: str, where: str) -> SensorSpec:
    section = _mapping(value, where)
    _only(
        section,
        ("blueprint", "transform", "attributes", "encoding", "intensity_scale"),
        where,
    )
    blueprint = _text(section.get("blueprint", ""), f"{where}.blueprint")
    transform = _mapping(section.get("transform"), f"{where}.transform")
    _only(transform, _TRANSFORM_KEYS, f"{where}.transform")
    spec = SensorSpec(
        blueprint=blueprint,
        transform=tuple(  # type: ignore[arg-type]
            _finite(transform.get(k, 0.0), f"{where}.transform.{k}")
            for k in _TRANSFORM_KEYS
        ),
        attributes={
            str(k): str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in _mapping(
                section.get("attributes"), f"{where}.attributes"
            ).items()
        },
        encoding=_text(section.get("encoding", "bgra"), f"{where}.encoding"),
        intensity_scale=_finite(
            section.get("intensity_scale", 255.0), f"{where}.intensity_scale"
        ),
    )
    codec_name = codec.split(":", 1)[0]
    if spec.is_lidar:
        if codec_name != "lidar":
            raise HilsConfigError(
                f"{where}: a {LIDAR_BLUEPRINT} is sent through a lidar:MODEL codec, "
                f"not {codec!r}"
            )
    elif blueprint.startswith(CAMERA_PREFIX):
        if codec_name != "raw":
            raise HilsConfigError(
                f"{where}: camera frames are sent raw (codec raw), not {codec!r}"
            )
        if spec.encoding not in CAMERA_ENCODINGS:
            raise HilsConfigError(
                f"{where}.encoding: {spec.encoding!r} is not one of "
                f"{', '.join(CAMERA_ENCODINGS)}"
            )
    else:
        raise HilsConfigError(
            f"{where}.blueprint: {blueprint!r} is neither {LIDAR_BLUEPRINT} nor a "
            f"{CAMERA_PREFIX}* camera"
        )
    return spec


def _relative_file(value: str, base: Path) -> str:
    """*value* made absolute when it names a file relative to *base*."""
    path = Path(value).expanduser()
    if not path.is_absolute() and (base / path).is_file():
        return str((base / path).resolve())
    return value


def _codec(value: Any, base: Path, where: str) -> str:
    codec = _text(value, where)
    name, sep, argument = codec.partition(":")
    if sep and argument:
        return f"{name}:{_relative_file(argument, base)}"
    return codec


def _input_source(value: Any, where: str) -> InputSource:
    if isinstance(value, Mapping):
        _only(value, ("field", "scale", "offset"), where)
        if "field" not in value:
            raise HilsConfigError(f"{where}: a mapping names a field")
        return InputSource(
            field=_text(value["field"], f"{where}.field"),
            scale=_finite(value.get("scale", 1.0), f"{where}.scale"),
            offset=_finite(value.get("offset", 0.0), f"{where}.offset"),
        )
    if isinstance(value, str):
        return InputSource(field=value)
    if isinstance(value, (bool, int, float)):
        return InputSource(constant=_finite(float(value), where))
    raise HilsConfigError(
        f"{where}: expected a field name, a number or {{field: ...}}, got {value!r}"
    )


def _input(value: Any, mode: str, where: str) -> InputSpec:
    section = _mapping(value, where)
    _only(section, ("name", "command"), where)
    if not section.get("name"):
        raise HilsConfigError(f"{where}: an input has a name")
    command = _mapping(section.get("command"), f"{where}.command")
    if not command:
        raise HilsConfigError(f"{where}.command: names no command field")
    allowed = COMMAND_FIELDS[mode]
    for key in command:
        if key not in allowed:
            raise HilsConfigError(
                f"{where}.command: {key!r} is not a field of a {mode} command; "
                f"expected {', '.join(allowed)}"
            )
    return InputSpec(
        name=_text(section["name"], f"{where}.name", number=True),
        command={
            str(k): _input_source(v, f"{where}.command.{k}") for k, v in command.items()
        },
    )


_STEP_KINDS = ("send", "wait", "sleep")


def _step(
    value: Any, phase: str, devices: Mapping[str, DeviceSpec], where: str
) -> StepSpec:
    section = _mapping(value, where)
    kinds = [k for k in _STEP_KINDS if k in section]
    if len(kinds) != 1:
        raise HilsConfigError(
            f"{where}: a step is one of {', '.join(_STEP_KINDS)}, got {sorted(section)}"
        )
    kind = kinds[0]
    if kind == "sleep":
        _only(section, ("sleep",), where)
        seconds = _seconds(section["sleep"], f"{where}.sleep", zero=True)
        return StepSpec(kind="sleep", seconds=seconds)
    device = _text(section[kind], f"{where}.{kind}")
    if device not in devices:
        raise HilsConfigError(
            f"{where}: no device {device!r}; devices: {', '.join(devices)}"
        )
    if devices[device].sensor is not None:
        raise HilsConfigError(f"{where}: {device} is a sensor device")
    name = section.get("name")
    if kind == "wait":
        _only(section, ("wait", "name", "fields", "timeout"), where)
        if not name:
            raise HilsConfigError(f"{where}: a wait names the message")
        name = _text(name, f"{where}.name", number=True)
        match = _keys(
            _mapping(section.get("fields"), f"{where}.fields"), f"{where}.fields"
        )
        for key, expected in match.items():
            if not isinstance(expected, (str, int, float, bool)):
                raise HilsConfigError(
                    f"{where}.fields.{key}: expected a constant, got {expected!r}"
                )
        timeout = _seconds(section.get("timeout", 10.0), f"{where}.timeout")
        return StepSpec(
            kind="wait",
            device=device,
            name=name,
            match=match,
            seconds=timeout,
        )
    _only(section, ("send", "name", "fields", "data", "id"), where)
    if bool(name) == ("data" in section):
        raise HilsConfigError(
            f"{where}: a send has either a name (and fields) or raw data"
        )
    if "data" in section:
        data = _data(section["data"], f"{where}.data")
        frame_id = section.get("id")
        if frame_id is not None and (
            isinstance(frame_id, bool) or not isinstance(frame_id, int)
        ):
            raise HilsConfigError(f"{where}.id: expected an integer")
        return StepSpec(kind="send", device=device, data=data, frame_id=frame_id)
    name = _text(name, f"{where}.name", number=True)
    fields = {
        k: _field(v, f"{where}.fields.{k}")
        for k, v in _keys(
            _mapping(section.get("fields"), f"{where}.fields"), f"{where}.fields"
        ).items()
    }
    if phase in _EGOLESS_PHASES:
        signals = [k for k, source in fields.items() if source.signal is not None]
        if signals:
            raise HilsConfigError(
                f"{where}.fields: {', '.join(signals)} use(s) a signal, but there "
                f"is no ego at {phase}"
            )
    return StepSpec(kind="send", device=device, name=name, fields=fields)


def _data(raw: Any, where: str) -> bytes:
    """A raw frame's bytes: a quoted hex string or a list of byte values."""
    if isinstance(raw, list):
        if not all(
            isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= 0xFF
            for b in raw
        ):
            raise HilsConfigError(
                f"{where}: a list holds byte values (0-255), got {raw!r}"
            )
        return bytes(raw)
    if not isinstance(raw, str):
        raise HilsConfigError(
            f"{where}: expected the bytes as a quoted hex string "
            f'(data: "02 10 01") or a list (data: [0x02, 0x10, 0x01]), got the '
            f"{type(raw).__name__} {raw!r}; YAML reads unquoted digits as a "
            "number (0102 is octal 66, 0x10 is 16), so quote it"
        )
    try:
        return bytes.fromhex(raw)
    except ValueError as exc:
        raise HilsConfigError(f"{where}: not hexadecimal bytes") from exc


def _lifecycle(
    value: Any, devices: Mapping[str, DeviceSpec]
) -> Dict[str, Tuple[StepSpec, ...]]:
    section = _mapping(value, "lifecycle")
    _only(section, PHASES, "lifecycle")
    phases: Dict[str, Tuple[StepSpec, ...]] = {}
    for phase in PHASES:
        steps = section.get(phase) or []
        if not isinstance(steps, list):
            raise HilsConfigError(f"lifecycle.{phase}: expected a list of steps")
        phases[phase] = tuple(
            _step(step, phase, devices, f"lifecycle.{phase}[{i}]")
            for i, step in enumerate(steps)
        )
    return phases


def _device(name: str, value: Any, base: Path, mode: str) -> DeviceSpec:
    where = f"devices.{name}"
    section = _mapping(value, where)
    _only(section, ("url", "codec", "options", "messages", "sensor", "inputs"), where)
    if not section.get("url"):
        raise HilsConfigError(f"{where}: a device has a url")
    codec = _codec(section.get("codec", "raw"), base, f"{where}.codec")
    options = {
        str(k): _relative_file(v, base) if k == "profiles" and isinstance(v, str) else v
        for k, v in _mapping(section.get("options"), f"{where}.options").items()
    }
    messages_value = section.get("messages") or []
    if not isinstance(messages_value, list):
        raise HilsConfigError(f"{where}.messages: expected a list")
    messages = tuple(
        _message(m, f"{where}.messages[{i}]") for i, m in enumerate(messages_value)
    )
    sensor = (
        _sensor(section["sensor"], codec, f"{where}.sensor")
        if section.get("sensor") is not None
        else None
    )
    inputs_value = section.get("inputs") or []
    if not isinstance(inputs_value, list):
        raise HilsConfigError(f"{where}.inputs: expected a list")
    inputs = tuple(
        _input(m, mode, f"{where}.inputs[{i}]") for i, m in enumerate(inputs_value)
    )
    if sensor is not None and (messages or inputs):
        raise HilsConfigError(
            f"{where}: a device with a sensor has no messages or inputs"
        )
    return DeviceSpec(
        name=name,
        url=_text(section["url"], f"{where}.url"),
        codec=codec,
        options=options,
        messages=messages,
        sensor=sensor,
        inputs=inputs,
    )


def parse_config(document: Any, base: Union[str, Path] = ".") -> HilsConfig:
    """The configuration *document* (a parsed YAML mapping) describes.

    Args:
        document: The parsed file.
        base: The directory relative files are resolved against.

    Raises:
        HilsConfigError: The document is not a configuration the bridge can run.
    """
    base = Path(base)
    root = _mapping(document, "configuration")
    _only(
        root,
        (
            "carla",
            "ego",
            "canvas",
            "clock",
            "runner",
            "commands",
            "devices",
            "lifecycle",
        ),
        "configuration",
    )
    carla = _mapping(root.get("carla"), "carla")
    _only(carla, ("host", "port", "timeout"), "carla")
    ego = _mapping(root.get("ego"), "ego")
    _only(ego, ("role_name", "wait"), "ego")
    clock = _text(root.get("clock", "simulation"), "clock")
    if clock not in CLOCKS:
        raise HilsConfigError(f"clock: {clock!r} is not one of {', '.join(CLOCKS)}")
    canvas: Optional[Path] = None
    if root.get("canvas"):
        canvas = (base / Path(_text(root["canvas"], "canvas")).expanduser()).resolve()
    runner = _mapping(root.get("runner"), "runner")
    _only(runner, ("listen",), "runner")
    listen = runner.get("listen", f"0.0.0.0:{DEFAULT_PORT}")
    if listen is not None and not isinstance(listen, str):
        raise HilsConfigError(
            f"runner.listen: expected host:port text, got {listen!r}; quote it "
            f'(listen: "0.0.0.0:{DEFAULT_PORT}"), or null for no run link'
        )
    listen = "" if listen is None else listen
    commands = _mapping(root.get("commands"), "commands")
    _only(commands, ("mode",), "commands")
    mode = _text(commands.get("mode", "control"), "commands.mode")
    if mode not in COMMAND_MODES:
        raise HilsConfigError(
            f"commands.mode: {mode!r} is not one of {', '.join(COMMAND_MODES)}"
        )
    devices_section = _mapping(root.get("devices"), "devices")
    if not devices_section:
        raise HilsConfigError("devices: the configuration names no device")
    devices = {
        k: _device(k, v, base, mode)
        for k, v in _keys(devices_section, "devices").items()
    }
    return HilsConfig(
        devices=tuple(devices.values()),
        runner_listen=listen,
        command_mode=mode,
        lifecycle=_lifecycle(root.get("lifecycle"), devices),
        host=_text(carla.get("host", "localhost"), "carla.host"),
        port=_port(carla.get("port", 2000), "carla.port"),
        timeout=_seconds(carla.get("timeout", 10.0), "carla.timeout"),
        role_name=_text(ego.get("role_name", str(EGO_ROLE_NAME)), "ego.role_name"),
        ego_wait=_seconds(ego.get("wait", 120.0), "ego.wait", zero=True),
        canvas=canvas,
        clock=clock,
    )


def load_config(path: Union[str, Path]) -> HilsConfig:
    """The configuration in the YAML file *path*.

    Raises:
        HilsConfigError: The file is missing or not a configuration.
    """
    import yaml  # noqa: PLC0415

    path = Path(path).expanduser()
    if not path.is_file():
        raise HilsConfigError(f"{path} is not a file")
    loader = yaml.SafeLoader(path.read_text(encoding="utf-8"))
    try:
        node = loader.get_single_node()
        if node is not None:
            _refuse_yaml11_scalars(node, path)
        document = None if node is None else loader.construct_document(node)
    except yaml.YAMLError as exc:
        raise HilsConfigError(f"{path}: {exc}") from exc
    finally:
        loader.dispose()
    return parse_config(document, path.resolve().parent)


_YAML_INT = "tag:yaml.org,2002:int"
_YAML_FLOAT = "tag:yaml.org,2002:float"
_YAML_BOOL = "tag:yaml.org,2002:bool"


def _yaml11_reading(tag: str, text: str) -> Optional[str]:
    """What YAML 1.1 makes of the plain scalar *text* (resolved to *tag*) when
    that is not what it looks like, and how to write it instead; else None.

    These are the forms YAML 1.2 reads differently: ``0102`` (octal 66),
    ``1:30`` (sexagesimal 90) and ``yes``/``no``/``on``/``off`` (booleans).
    ``0x7DF``, ``1_000``, ``1.5`` and ``true`` mean what they say.
    """
    digits = text.lstrip("+-").replace("_", "")
    if tag == _YAML_BOOL and text.lower() in ("yes", "no", "on", "off"):
        truth = text.lower() in ("yes", "on")
        return f"the boolean {truth}; write {str(truth).lower()}"
    if tag in (_YAML_INT, _YAML_FLOAT) and ":" in digits:
        return "a sexagesimal (base 60) number; write it in decimal"
    if (
        tag == _YAML_INT
        and len(digits) > 1
        and digits[0] == "0"
        and digits[1] in "01234567"
    ):
        value = int(digits, 8) * (-1 if text.startswith("-") else 1)
        return f"the octal number {value}; write {value} or {value:#x}"
    return None


def _refuse_yaml11_scalars(node: Any, file: Path) -> None:
    """Refuse the plain scalars under *node* that YAML 1.1 reads as something
    other than what they look like (:func:`_yaml11_reading`), naming each one's
    place in the configuration and the file.

    ``yaml.safe_load`` is YAML 1.1: unquoted, ``Gear: Off`` is ``False`` and
    ``data: 0102`` is 66, in a field value or a codec option as much as in a
    name, and nothing after loading can tell ``66`` from ``0102``.
    """
    import yaml  # noqa: PLC0415

    seen: set[int] = set()

    def check(scalar: Any, where: str) -> None:
        if scalar.style is not None:  # quoted, a block scalar, or flow text
            return
        reading = _yaml11_reading(scalar.tag, scalar.value)
        if reading is not None:
            mark = scalar.start_mark
            raise HilsConfigError(
                f"{file}:{mark.line + 1}:{mark.column + 1}: {where or 'configuration'}: "
                f"unquoted, YAML reads {scalar.value} as {reading}, or quote it "
                f'("{scalar.value}") if it is text'
            )

    def walk(item: Any, where: str) -> None:
        if id(item) in seen:  # an alias
            return
        seen.add(id(item))
        if isinstance(item, yaml.ScalarNode):
            check(item, where)
        elif isinstance(item, yaml.SequenceNode):
            for index, child in enumerate(item.value):
                walk(child, f"{where}[{index}]")
        elif isinstance(item, yaml.MappingNode):
            for key, child in item.value:
                if isinstance(key, yaml.ScalarNode):
                    check(key, f"{where}: key" if where else "key")
                    walk(child, f"{where}.{key.value}" if where else key.value)
                else:
                    walk(key, where)
                    walk(child, where)

    walk(node, "")
