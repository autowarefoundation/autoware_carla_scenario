"""The rules the weather's effects follow: CARLA's weather, the road's grip and how far drivers see.

A run is given conditions -- the rainfall rate, the visibility through the
fog, the time of day: the causes, concepts of the taxonomy an ODD is written
in.  What they do follows from them, as a distribution a run's quantile picks
the value of: the CARLA weather settings that render them, the road's grip
(:func:`~autoware_carla_scenario.actions.environment.tire_friction`) and how
far drivers see (:func:`~autoware_carla_scenario.visibility.visibility`).
Which distribution is written as conditional distributions, a Bayesian
network (``weather_rules.yaml`` next to this module, version 3), which
openodd reads (:mod:`openodd.rules`, its docs/rules.md): they are the CARLA
backend's own (:attr:`~.odd.backend.CarlaBackend.rules`).

* a **quantity** (``carla_precipitation``, ``tire_friction``,
  ``visibility``, ...: :data:`QUANTITIES`) is one or more parts; at a run's
  quantile, each part gives its value and ``combine`` (``min`` or ``max``)
  joins them;
* a **part** reads some causes (:data:`WEATHER_INPUTS`) or other quantities,
  and gives a distribution (``uniform``, ``triangular`` or ``normal``) at
  points of them;
* so the rules chain: the rainfall rate gives CARLA's precipitation, which
  gives its wetness, which gives the grip; the visibility gives
  ``sight_speed``, the speed a driver keeps
  (:func:`~autoware_carla_scenario.visibility.sight_speed_at`).

**The network is worked out once, before a run** (:func:`resolve_environment`):
the causes a config names, and any CARLA field it sets itself, give every
CARLA field and the visibility the run then applies as numbers.  The sampler
does the same for the cases it draws.  Only what depends on a vehicle is
worked out during the run: its grip, from the conditions and the CARLA
weather already set (:func:`vehicle_at`), and the speed each traffic driver
keeps.

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
vehicle's (:func:`vehicle_range`); everything else is the default rules'.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from openodd import Rules, RulesError

from .weather_inputs import CARLA_QUANTITIES as CARLA_QUANTITIES
from .weather_inputs import CAUSE_SETTINGS as CAUSE_SETTINGS
from .weather_inputs import EGO_KEY as EGO_KEY
from .weather_inputs import QUANTITIES as QUANTITIES
from .weather_inputs import WEATHER_INPUTS as WEATHER_INPUTS
from .weather_inputs import rule_values as rule_values
from .weather_inputs import vehicle_entry

if TYPE_CHECKING:
    from .odd.backend import CarlaBackend

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_QUANTITIES",
    "CAUSE_SETTINGS",
    "DEFAULT_RULES",
    "RESOLVED_QUANTITIES",
    "EGO_KEY",
    "QUANTITIES",
    "WEATHER_INPUTS",
    "RulesError",
    "check_weather_rules",
    "custom_rules_in_use",
    "default_rules",
    "resolve_environment",
    "rule_values",
    "use_weather_rules",
    "vehicle_at",
    "vehicle_range",
    "vehicle_rules_in_use",
    "weather_at",
    "weather_range",
    "weather_rule_files",
]

#: The rules a run follows unless it names others.
DEFAULT_RULES: Path = Path(__file__).resolve().parent / "weather_rules.yaml"

#: The packaged rules, read once.
_DEFAULT: Optional[Rules] = None
#: Blueprint pattern (or :data:`EGO_KEY`) -> its rules, for the run.
_VEHICLE_RULES: list[tuple[str, Rules]] = []
#: Whether the run follows a rule file of its own over the default rules
#: (``environment.rules``, or its ``default``).
_RULE_FILE: bool = False


def default_rules() -> Rules:
    """The rules :data:`DEFAULT_RULES` holds, read once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Rules.read(DEFAULT_RULES)
    return _DEFAULT


def _backend() -> "CarlaBackend":
    # Inside a function: odd.backend imports the probes, which import the
    # environment action, which imports this module.
    from .odd.backend import CARLA_BACKEND  # noqa: PLC0415

    return CARLA_BACKEND


def check_weather_rules(
    spec: Any,
) -> tuple[Union[str, Path, None], list[tuple[str, Union[str, Path]]]]:
    """*spec*'s rule files, its form checked but no file read: the default
    rules' (``None``: the packaged ones alone) and every ``vehicles`` entry's
    ``(pattern, file)``, in order.  What :func:`use_weather_rules` reads, and
    what the runner refuses a batch for before it runs.

    *spec* is ``None``, a file (a ``str`` path or ``https://`` URL, or a
    :class:`~pathlib.Path`), or a mapping of ``default`` (``None`` or a
    file) and ``vehicles`` (``None`` or a mapping of pattern to file).

    Raises:
        RulesError: When *spec*, its ``default`` or a ``vehicles`` entry is
            not of that form, or it names a field other than those two.
    """
    if spec is None or isinstance(spec, (str, Path)):
        return spec, []
    if not isinstance(spec, Mapping):
        raise RulesError(
            f"environment.rules: expected a file or {{default, vehicles}}, got {spec!r}"
        )
    unknown = sorted(set(map(str, spec)) - {"default", "vehicles"})
    if unknown:
        raise RulesError(f"environment.rules: unknown fields {unknown}")
    default = spec.get("default")
    if default is not None and not isinstance(default, (str, Path)):
        # Not str() of it: a number, or a pinned {url, sha256} (which only
        # ``extends`` takes), would be read as a path that does not exist.
        raise RulesError(f"environment.rules.default: expected a file, got {default!r}")
    vehicles = spec.get("vehicles")
    if vehicles is None:
        vehicles = {}
    elif not isinstance(vehicles, Mapping):
        raise RulesError("environment.rules.vehicles: expected pattern -> file")
    entries: list[tuple[str, Union[str, Path]]] = []
    for pattern, location in vehicles.items():
        if not isinstance(location, (str, Path)):
            raise RulesError(f"environment.rules.vehicles.{pattern}: expected a file")
        entries.append((str(pattern), location))
    return default, entries


def use_weather_rules(spec: Union[str, Path, Mapping[str, Any], None] = None) -> None:
    """Follow *spec*'s rules from now on: ``None`` the default ones, a file
    (a path or an ``https://`` URL) merged over them, or ``{default,
    vehicles}`` (the module docstring).  The runner calls this at the start
    of every run with the rules its scenario was built with (its config's
    ``environment.rules``, or ``None``), so one run's do not reach the next,
    and a batch, which builds every case before any runs, runs each under
    its own (#1936).

    Raises:
        RulesError: When a file cannot be read, its rules are not well formed,
            or *spec* is neither a file nor ``{default, vehicles}`` of files
            (:func:`check_weather_rules`).
    """
    global _RULE_FILE
    backend = _backend()
    # Its form first, so a malformed spec changes nothing.
    default, vehicles = check_weather_rules(spec)
    _VEHICLE_RULES.clear()
    _RULE_FILE = False
    followed = backend.use_rules(None if default is None else str(default))
    _RULE_FILE = default is not None
    base = followed if followed is not None else default_rules()
    for pattern, location in vehicles:
        rules = base.read_over(str(location))
        rules.check_inputs(list(WEATHER_INPUTS))
        _VEHICLE_RULES.append((pattern, rules))
    if default is not None or _VEHICLE_RULES:
        logger.info(
            "Following the weather rules of %s",
            ", ".join(path for path, _sha in weather_rule_files()),
        )


def vehicle_rules_in_use() -> bool:
    """Whether the run's rules differ by vehicle."""
    return len(_VEHICLE_RULES) > 0


def custom_rules_in_use() -> bool:
    """Whether the run follows rules other than the default ones -- a rule
    file of its own (``environment.rules``), or rules by vehicle -- so its
    grip may differ from the one the default rules give a dry road
    (CARLA's own): an ``EnvironmentAction`` then gives every vehicle its
    rules' grip whatever it sets (#1937)."""
    return _RULE_FILE or len(_VEHICLE_RULES) > 0


def _with_labels(
    inputs: Mapping[str, Any], labels: Optional[Mapping[str, str]]
) -> dict[str, Any]:
    values: dict[str, Any] = dict(inputs)
    if labels is not None:
        values.update(labels)
    return values


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


def weather_range(quantity: str, inputs: Mapping[str, Any]) -> tuple[float, float]:
    """The ``(low, high)`` range of *quantity* at *inputs* -- causes (a number,
    or the time of day's label) and quantities -- by the rules the run follows
    (every vehicle's but those ``vehicles`` names)."""
    return _backend().derived_range(quantity, inputs)


def weather_at(
    quantity: str,
    inputs: Mapping[str, Any],
    quantile: float,
    labels: Optional[Mapping[str, str]] = None,
) -> float:
    """The value of *quantity* at *quantile* (0-1, clamped) of its
    distribution at *inputs* (as :func:`weather_range` reads them), by the
    rules the run follows (every vehicle's but those ``vehicles`` names).

    *labels* are more inputs, the categorical ones (the time of day's
    label): what a caller that keeps numbers and labels apart (the checked
    ``actions.environment``) gives them as.
    """
    return _backend().derived_at(quantity, _with_labels(inputs, labels), quantile)


def _rules_for(type_id: str, role: str) -> Optional[Rules]:
    # Which entry is the checked weather_inputs's (vehicle_entry()).
    index = vehicle_entry([pattern for pattern, _ in _VEHICLE_RULES], type_id, role)
    return None if index < 0 else _VEHICLE_RULES[index][1]


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


def vehicle_at(
    quantity: str,
    inputs: Mapping[str, Any],
    quantile: float,
    type_id: str,
    role: str,
    labels: Optional[Mapping[str, str]] = None,
) -> float:
    """The value of *quantity* at *quantile* (0-1, clamped) in the weather
    *inputs* (and *labels*, as :func:`weather_at` reads them) for the
    vehicle of blueprint *type_id* and ``role_name`` *role* (``""``: none):
    by the rules of the first ``vehicles`` entry it matches, else
    :func:`weather_at`'s."""
    rules = _rules_for(type_id, role)
    if rules is None:
        return weather_at(quantity, inputs, quantile, labels)
    values = _backend().rule_values(_with_labels(inputs, labels))
    return rules.quantile(quantity, values, quantile)


#: The quantities worked out before a run (:func:`resolve_environment`):
#: CARLA's weather fields and how far drivers see.  The grip and the speed a
#: driver keeps depend on the vehicle, and are worked out during it.
RESOLVED_QUANTITIES: tuple[str, ...] = (*CARLA_QUANTITIES.values(), "visibility")

_FIELD_OF: dict[str, str] = {q: field for field, q in CARLA_QUANTITIES.items()}


def resolve_environment(
    settings: Mapping[str, Any],
    quantiles: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """The ``environment`` *settings* with the weather rules worked out: what
    a run applies, as numbers.

    The causes *settings* names (:data:`CAUSE_SETTINGS`: ``rainfall_rate_mm_h``,
    ``fog_visibility_range_m``, ``time_of_day``, ``wind_speed_kmh``,
    ``cloud_cover``) and the CARLA fields it sets
    itself (``precipitation``, ``fog_density``, ...) are kept; every other
    CARLA field that follows from them, and ``visibility_m``, are added, each
    at its quantile in *quantiles* (by quantity; ``visibility`` also from
    ``visibility_quantile`` in *settings*), the median when none is given.  A
    field nothing given leads to is left out: the world keeps its own.

    The visibility follows the causes alone, so a config that sets CARLA's
    fields but names no cause gets none: its drivers see as they always did
    (a warning says so), and the ODD's weather is not measured.

    Raises:
        RulesError: When the rules cannot give a quantity at the values given
            (a cause they do not know, a label they are not written at).
    """
    rules = _backend().active_rules
    out: dict[str, Any] = dict(settings)
    if rules is None:
        return out
    wanted = dict(quantiles or {})
    if "visibility" not in wanted and settings.get("visibility_quantile") is not None:
        wanted["visibility"] = float(settings["visibility_quantile"])
    # What is known: the causes named, and the CARLA fields set.
    known: dict[str, Any] = {}
    for key, cause in CAUSE_SETTINGS.items():
        if settings.get(key) is not None:
            known[cause] = settings[key]
    causes = set(known)
    for field, quantity in CARLA_QUANTITIES.items():
        if settings.get(field) is not None:
            known[quantity] = float(settings[field])
    given = set(known)
    if settings.get("visibility_m") is not None:
        given.add("visibility")
    for quantity in rules.order:
        if quantity not in RESOLVED_QUANTITIES or quantity in given:
            continue
        reads = set(rules.inputs(quantity))
        # A quantity follows what is known: a cause named, or a quantity set
        # or worked out; the visibility follows the causes alone.
        led = reads & causes if quantity == "visibility" else reads & set(known)
        if not led:
            continue
        value = rules.quantile(quantity, known, wanted.get(quantity, 0.5), wanted)
        known[quantity] = value
        if quantity == "visibility":
            out["visibility_m"] = value
        else:
            out[_FIELD_OF[quantity]] = value
    if (
        not causes
        and any(settings.get(f) is not None for f in CARLA_QUANTITIES)
        and settings.get("visibility_m") is None
    ):
        logger.warning(
            "environment sets CARLA's weather but names no condition (%s): how "
            "far drivers see does not follow it, and the ODD's weather is not "
            "measured",
            ", ".join(CAUSE_SETTINGS),
        )
    return out
