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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..constants import EGO_ROLE_NAME
from .measurements import CAMERA_ENCODINGS
from .protocol import COMMAND_FIELDS, COMMAND_MODES, DEFAULT_PORT
from .signals import SIGNALS

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
        ``None`` when the message lacks the field or it is not a number."""
        if self.field is None:
            return self.constant
        value = fields.get(self.field)
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)):
            return None
        return float(value) * self.scale + self.offset


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

    def values(self, fields: Mapping[str, Any]) -> Dict[str, float]:
        """The command fields a decoded message's *fields* give."""
        values: Dict[str, float] = {}
        for key, source in self.command.items():
            value = source.resolve(fields)
            if value is not None:
                values[key] = value
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


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HilsConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


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
            signal=_signal(str(value["signal"]), where),
            scale=_number(value.get("scale", 1.0), f"{where}.scale"),
            offset=_number(value.get("offset", 0.0), f"{where}.offset"),
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
        period = _number(period, f"{where}.period")
        if period <= 0.0:
            raise HilsConfigError(f"{where}.period: must be positive")
    fields = _mapping(section.get("fields"), f"{where}.fields")
    return MessageSpec(
        name=str(section["name"]),
        period=period,
        fields={str(k): _field(v, f"{where}.fields.{k}") for k, v in fields.items()},
    )


def _sensor(value: Any, codec: str, where: str) -> SensorSpec:
    section = _mapping(value, where)
    _only(
        section,
        ("blueprint", "transform", "attributes", "encoding", "intensity_scale"),
        where,
    )
    blueprint = str(section.get("blueprint", ""))
    transform = _mapping(section.get("transform"), f"{where}.transform")
    _only(transform, _TRANSFORM_KEYS, f"{where}.transform")
    spec = SensorSpec(
        blueprint=blueprint,
        transform=tuple(  # type: ignore[arg-type]
            _number(transform.get(k, 0.0), f"{where}.transform.{k}")
            for k in _TRANSFORM_KEYS
        ),
        attributes={
            str(k): str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in _mapping(
                section.get("attributes"), f"{where}.attributes"
            ).items()
        },
        encoding=str(section.get("encoding", "bgra")),
        intensity_scale=_number(
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


def _codec(value: Any, base: Path) -> str:
    codec = str(value)
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
            field=str(value["field"]),
            scale=_number(value.get("scale", 1.0), f"{where}.scale"),
            offset=_number(value.get("offset", 0.0), f"{where}.offset"),
        )
    if isinstance(value, str):
        return InputSource(field=value)
    if isinstance(value, (bool, int, float)):
        return InputSource(constant=float(value))
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
        name=str(section["name"]),
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
        seconds = _number(section["sleep"], f"{where}.sleep")
        if seconds < 0.0:
            raise HilsConfigError(f"{where}.sleep: must not be negative")
        return StepSpec(kind="sleep", seconds=seconds)
    device = str(section[kind])
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
        match = _mapping(section.get("fields"), f"{where}.fields")
        for key, expected in match.items():
            if not isinstance(expected, (str, int, float, bool)):
                raise HilsConfigError(
                    f"{where}.fields.{key}: expected a constant, got {expected!r}"
                )
        timeout = _number(section.get("timeout", 10.0), f"{where}.timeout")
        if timeout <= 0.0:
            raise HilsConfigError(f"{where}.timeout: must be positive")
        return StepSpec(
            kind="wait",
            device=device,
            name=str(name),
            match=dict(match),
            seconds=timeout,
        )
    _only(section, ("send", "name", "fields", "data", "id"), where)
    if bool(name) == ("data" in section):
        raise HilsConfigError(
            f"{where}: a send has either a name (and fields) or raw data"
        )
    if "data" in section:
        try:
            data = bytes.fromhex(str(section["data"]))
        except ValueError as exc:
            raise HilsConfigError(f"{where}.data: not hexadecimal bytes") from exc
        frame_id = section.get("id")
        if frame_id is not None and (
            isinstance(frame_id, bool) or not isinstance(frame_id, int)
        ):
            raise HilsConfigError(f"{where}.id: expected an integer")
        return StepSpec(kind="send", device=device, data=data, frame_id=frame_id)
    fields = {
        str(k): _field(v, f"{where}.fields.{k}")
        for k, v in _mapping(section.get("fields"), f"{where}.fields").items()
    }
    if phase in _EGOLESS_PHASES:
        signals = [k for k, source in fields.items() if source.signal is not None]
        if signals:
            raise HilsConfigError(
                f"{where}.fields: {', '.join(signals)} use(s) a signal, but there "
                f"is no ego at {phase}"
            )
    return StepSpec(kind="send", device=device, name=str(name), fields=fields)


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
    codec = _codec(section.get("codec", "raw"), base)
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
        url=str(section["url"]),
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
    clock = str(root.get("clock", "simulation"))
    if clock not in CLOCKS:
        raise HilsConfigError(f"clock: {clock!r} is not one of {', '.join(CLOCKS)}")
    canvas: Optional[Path] = None
    if root.get("canvas"):
        canvas = (base / Path(str(root["canvas"])).expanduser()).resolve()
    runner = _mapping(root.get("runner"), "runner")
    _only(runner, ("listen",), "runner")
    listen = runner.get("listen", f"0.0.0.0:{DEFAULT_PORT}")
    listen = "" if listen is None else str(listen)
    commands = _mapping(root.get("commands"), "commands")
    _only(commands, ("mode",), "commands")
    mode = str(commands.get("mode", "control"))
    if mode not in COMMAND_MODES:
        raise HilsConfigError(
            f"commands.mode: {mode!r} is not one of {', '.join(COMMAND_MODES)}"
        )
    devices_section = _mapping(root.get("devices"), "devices")
    if not devices_section:
        raise HilsConfigError("devices: the configuration names no device")
    devices = {
        str(k): _device(str(k), v, base, mode) for k, v in devices_section.items()
    }
    return HilsConfig(
        devices=tuple(devices.values()),
        runner_listen=listen,
        command_mode=mode,
        lifecycle=_lifecycle(root.get("lifecycle"), devices),
        host=str(carla.get("host", "localhost")),
        port=int(_number(carla.get("port", 2000), "carla.port")),
        timeout=_number(carla.get("timeout", 10.0), "carla.timeout"),
        role_name=str(ego.get("role_name", EGO_ROLE_NAME)),
        ego_wait=_number(ego.get("wait", 120.0), "ego.wait"),
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
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HilsConfigError(f"{path}: {exc}") from exc
    return parse_config(document, path.resolve().parent)
