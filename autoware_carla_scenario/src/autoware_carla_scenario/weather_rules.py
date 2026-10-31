"""The rules the weather's effects follow: the road's grip and how far drivers see.

A run sets the weather; what it does to the road and to the drivers' sight
follows from it, as a range a run's quantile picks a point of
(:func:`~autoware_carla_scenario.actions.environment.tire_friction`,
:func:`~autoware_carla_scenario.visibility.visibility`).  Which range is
written as fuzzy rules (``weather_rules.yaml`` next to this module), which
openodd reads and infers (:mod:`openodd.rules`, its docs/rules.md): they are
the CARLA backend's own (:attr:`~.odd.backend.CarlaBackend.rules`).

* a **quantity** (``tire_friction``, ``visibility``) is one or more rule
  bases, whose ranges ``combine`` (``min`` or ``max``) joins;
* a **rule base** reads some of the weather's fields (:data:`WEATHER_INPUTS`),
  each sorted into fuzzy terms, and its rules give the ``low`` and ``high``
  ends of the range per combination of terms.

A run reads other rules with ``environment.rules`` (:func:`use_weather_rules`):
a file -- a path or an ``https://`` URL -- merged over the default rules (it
need only name what it changes, and may ``extends`` others), or, for rules
that differ by vehicle::

    environment:
      rules:
        default: weather/measured.yaml          # every vehicle no entry names
        vehicles:                               # the first that matches
          ego: weather/ego_measured.yaml        # the ego (role_name "Ego")
          "vehicle.mini.*": weather/mini.yaml   # a blueprint pattern

Each vehicle's file is read over the run's default rules.  The grip is a
vehicle's (:func:`vehicle_range`); how far drivers see is the default rules'.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from openodd import Rules, RulesError

if TYPE_CHECKING:
    from .odd.backend import CarlaBackend

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RULES",
    "EGO_KEY",
    "QUANTITIES",
    "WEATHER_INPUTS",
    "RulesError",
    "default_rules",
    "rule_values",
    "use_weather_rules",
    "vehicle_range",
    "vehicle_rules_in_use",
    "weather_range",
    "weather_rule_files",
]

#: The rules a run follows unless it names others.
DEFAULT_RULES: Path = Path(__file__).resolve().parent / "weather_rules.yaml"

#: The quantities the framework reads a range of.
QUANTITIES: tuple[str, ...] = ("tire_friction", "visibility")

#: The weather's fields a rule base may read, as CARLA's ``WeatherParameters``
#: name them, and ``water``: the larger of ``wetness`` and
#: ``precipitation_deposits``, which the road's grip follows.
WEATHER_INPUTS: tuple[str, ...] = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
    "fog_distance",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "water",
)

#: The ``vehicles`` key naming the ego (``role_name`` ``Ego``).
EGO_KEY = "ego"
_EGO_ROLE = "Ego"

#: The packaged rules, read once.
_DEFAULT: Optional[Rules] = None
#: Blueprint pattern (or :data:`EGO_KEY`) -> its rules, for the run.
_VEHICLE_RULES: list[tuple[str, Rules]] = []


def default_rules() -> Rules:
    """The rules :data:`DEFAULT_RULES` holds, read once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Rules.read(DEFAULT_RULES)
    return _DEFAULT


def rule_values(values: Mapping[str, float]) -> dict[str, float]:
    """What the rules read of the weather *values*: those, and ``water``
    from ``wetness`` and ``precipitation_deposits`` when not given."""
    out = {str(k): float(v) for k, v in values.items()}
    if "water" not in out and ("wetness" in out or "precipitation_deposits" in out):
        out["water"] = max(
            out.get("wetness", 0.0), out.get("precipitation_deposits", 0.0)
        )
    return out


def _backend() -> "CarlaBackend":
    # Inside a function: odd.backend imports the probes, which import the
    # environment action, which imports this module.
    from .odd.backend import CARLA_BACKEND  # noqa: PLC0415

    return CARLA_BACKEND


def use_weather_rules(spec: Union[str, Path, Mapping[str, Any], None] = None) -> None:
    """Follow *spec*'s rules from now on: ``None`` the default ones, a file
    (a path or an ``https://`` URL) merged over them, or ``{default,
    vehicles}`` (the module docstring).  The runner calls this for every run
    with its config's ``environment.rules``, so one run's do not reach the next.

    Raises:
        RulesError: When a file cannot be read, its rules are not well formed,
            or *spec* is neither a file nor ``{default, vehicles}``.
    """
    backend = _backend()
    default: Union[str, Path, None]
    vehicles: Mapping[Any, Any]
    if spec is None or isinstance(spec, (str, Path)):
        default, vehicles = spec, {}
    elif isinstance(spec, Mapping):
        unknown = sorted(set(map(str, spec)) - {"default", "vehicles"})
        if unknown:
            raise RulesError(f"environment.rules: unknown fields {unknown}")
        default = spec.get("default")
        vehicles = spec.get("vehicles") or {}
        if not isinstance(vehicles, Mapping):
            raise RulesError("environment.rules.vehicles: expected pattern -> file")
    else:
        raise RulesError(
            f"environment.rules: expected a file or {{default, vehicles}}, got {spec!r}"
        )
    _VEHICLE_RULES.clear()
    followed = backend.use_rules(None if default is None else str(default))
    base = followed if followed is not None else default_rules()
    for pattern, location in vehicles.items():
        if not isinstance(location, (str, Path)):
            raise RulesError(f"environment.rules.vehicles.{pattern}: expected a file")
        rules = base.read_over(str(location))
        rules.check_inputs(WEATHER_INPUTS)
        _VEHICLE_RULES.append((str(pattern), rules))
    if default is not None or _VEHICLE_RULES:
        logger.info(
            "Following the weather rules of %s",
            ", ".join(path for path, _sha in weather_rule_files()),
        )


def vehicle_rules_in_use() -> bool:
    """Whether the run's rules differ by vehicle."""
    return len(_VEHICLE_RULES) > 0


def weather_rule_files() -> list[tuple[str, str]]:
    """``(path or URL, sha256)`` of every rule file the run follows, the
    default rules' first: what a run records of them."""
    seen: list[tuple[str, str]] = []
    active = _backend().active_rules
    for rules in [active, *(r for _p, r in _VEHICLE_RULES)]:
        if rules is None:
            continue
        for entry in rules.files:
            if entry not in seen:
                seen.append(entry)
    return seen


def weather_range(quantity: str, inputs: dict[str, float]) -> tuple[float, float]:
    """The ``(low, high)`` range of *quantity* in the weather *inputs*, by the
    rules the run follows (every vehicle's but those ``vehicles`` names)."""
    return _backend().derived_range(quantity, inputs)


def _rules_for(type_id: str, role: str) -> Optional[Rules]:
    for pattern, rules in _VEHICLE_RULES:
        if pattern.lower() == EGO_KEY:
            if role == _EGO_ROLE:
                return rules
        elif fnmatch.fnmatchcase(type_id, pattern):
            return rules
    return None


def vehicle_range(
    quantity: str, inputs: dict[str, float], type_id: str, role: str
) -> tuple[float, float]:
    """The ``(low, high)`` range of *quantity* in the weather *inputs* for the
    vehicle of blueprint *type_id* and ``role_name`` *role* (``""``: none):
    the rules of the first ``vehicles`` entry it matches, else
    :func:`weather_range`'s."""
    rules = _rules_for(type_id, role)
    if rules is None:
        return weather_range(quantity, inputs)
    return rules.range(quantity, rule_values(inputs))
