"""The plain-typed half of :mod:`.config`: a device's sensor, and the names a
configuration chooses from.

:class:`SensorSpec` (the CARLA sensor a device carries), the checks of a
LiDAR's blueprint attributes, the resolution of a file relative to the
configuration, the check of a field's signal name (:func:`checked_signal`),
:data:`CLOCKS`, :data:`PHASES` and :class:`HilsConfigError`.
Nothing here reads YAML or a document of any shape, so the library check
compiles it (docs/typecheck.md); :mod:`.config` uses it and re-exports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple

from .signals import SIGNALS

__all__ = [
    "CAMERA_PREFIX",
    "CLOCKS",
    "LIDAR_BLUEPRINT",
    "PHASES",
    "HilsConfigError",
    "SensorSpec",
    "checked_signal",
]

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

#: When the bridge runs a lifecycle's steps:
#:
#: * ``"startup"`` -- once, when the bridge starts (its devices just opened).
#: * ``"scenario_start"`` -- when a scenario starts: a run's ``start``, or the
#:   ego appearing.  The run waits for them (the ECUs being initialized).
#: * ``"warmup"`` -- right after, with the sensors attached and the world
#:   followed: the ECUs getting ready to drive on what they measure (an
#:   automated driving stack localizing, routing and engaging).  The run
#:   waits for them too: its warm-up, as an ego's stack's
#:   (:attr:`~autoware_carla_scenario.entity.EgoVehicle.is_initialized`).
#: * ``"scenario_end"`` -- when the scenario ends: a run's ``end``, the run
#:   going away, or the ego disappearing.
#: * ``"shutdown"`` -- once, when the bridge stops (Ctrl-C, SIGTERM).
PHASES: Tuple[str, ...] = (
    "startup",
    "scenario_start",
    "warmup",
    "scenario_end",
    "shutdown",
)


class HilsConfigError(ValueError):
    """The configuration is not one the bridge can run."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class SensorSpec:
    """A CARLA sensor the bridge attaches to the ego, forwarding what it measures.

    Attributes:
        blueprint: ``sensor.lidar.ray_cast`` or a ``sensor.camera.*``.
        transform: ``(x, y, z, roll, pitch, yaw)`` relative to the ego, in
            CARLA's convention (metres, degrees; y right).
        attributes: Blueprint attributes, as strings.
        encoding: A camera frame's encoding
            (:data:`~.measurements.CAMERA_ENCODINGS`).
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


def _attribute_number(attributes: Mapping[str, str], key: str, where: str) -> float:
    """The blueprint attribute *key* of a sensor as a finite number."""
    text = attributes[key]
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise HilsConfigError(f"{where}.{key}: expected a finite number, got {text!r}")
    return number


#: The LiDAR attributes CARLA takes as a proportion of the points, from 0 to
#: 1: ``dropoff_general_rate`` (the share of points dropped at random) and
#: ``dropoff_zero_intensity`` (the probability a point of zero intensity is
#: dropped).  CARLA clamps neither, so 45 empties every sweep and a negative
#: one drops nothing, both without a word (#1770).  ``dropoff_intensity_limit``
#: is a threshold, not a proportion, and is passed as it is.
_LIDAR_PROPORTIONS: Tuple[str, ...] = ("dropoff_general_rate", "dropoff_zero_intensity")


def _lidar_attributes(attributes: Mapping[str, str], where: str) -> None:
    """Refuse the LiDAR attributes the bridge cannot gather revolutions for,
    and the drop-off proportions (:data:`_LIDAR_PROPORTIONS`) outside [0, 1].

    The bridge gathers a LiDAR's per-tick arcs into revolutions
    (:class:`~autoware_carla_scenario.hils.lidar_revolution.LidarRevolution`).
    A revolution is one sweep of ``horizontal_fov`` degrees, which may be
    anything in (0, 360], at ``rotation_frequency`` revolutions a second, a
    finite positive number.  ``sensor_tick`` must be left unset (or 0): CARLA
    sweeps only one world tick per measurement however long the
    ``sensor_tick``, so a LiDAR measured less often than every tick turns
    slower than its ``rotation_frequency``, and no device turns that way.
    """
    if "horizontal_fov" in attributes:
        fov = _attribute_number(attributes, "horizontal_fov", where)
        if not 0.0 < fov <= 360.0:
            raise HilsConfigError(
                f"{where}.horizontal_fov: must be in (0, 360] degrees, "
                f"got {attributes['horizontal_fov']!r}"
            )
    if "rotation_frequency" in attributes:
        frequency = _attribute_number(attributes, "rotation_frequency", where)
        if frequency <= 0.0:
            raise HilsConfigError(
                f"{where}.rotation_frequency: must be positive, "
                f"got {attributes['rotation_frequency']!r}"
            )
    if "sensor_tick" in attributes:
        if _attribute_number(attributes, "sensor_tick", where) != 0.0:
            raise HilsConfigError(
                f"{where}.sensor_tick: a LiDAR measures every world tick; "
                f"leave sensor_tick unset (or 0), got {attributes['sensor_tick']!r} "
                "(CARLA sweeps only one world tick per measurement, so with a "
                "sensor_tick it turns slower than its rotation_frequency)"
            )
    for key in _LIDAR_PROPORTIONS:
        if key in attributes:
            proportion = _attribute_number(attributes, key, where)
            if not 0.0 <= proportion <= 1.0:
                raise HilsConfigError(
                    f"{where}.{key}: must be in [0, 1] (a proportion of the "
                    f"points), got {attributes[key]!r}"
                )


def _relative_file(value: str, base: Path) -> str:
    """*value* with ``~`` expanded, and made absolute when it names a file
    relative to *base*.

    hilspy expands no ``~``, so an expanded path is returned even when it is
    absolute; any other absolute *value* is returned as it is, not normalized
    (a registered codec's argument may not be a path), and so is a relative
    *value* that names no file under *base* (a LiDAR codec's model name).
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path) if value.startswith("~") else value
    if (base / path).is_file():
        return str((base / path).resolve())
    return value


def checked_signal(name: str, where: str) -> str:
    """*name*, which must be one of :data:`~.signals.SIGNALS` (a field's
    ``$signal`` or ``{signal: name}``), or a :class:`HilsConfigError` naming
    *where* and the signals there are."""
    if name not in SIGNALS:
        raise HilsConfigError(
            f"{where}: unknown signal {name!r}; signals: {', '.join(SIGNALS)}"
        )
    return name
