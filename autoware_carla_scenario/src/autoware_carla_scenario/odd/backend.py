"""This framework as an openodd backend: what measures an ODD in CARLA.

`openodd <https://github.com/hakuturu583/openodd_py>`_ reads, judges,
versions and samples ODDs; what it leaves to an application is how the
attributes are measured and set. Here:

* **probes** -- the built-in probes a binding file names (``probe:
  speed_limit_kph``), each reading the CARLA world (:mod:`.probes`);
* **readers** -- ``measure: <key>``: the running scenario's measure *key*
  (:func:`~.scenario_measure.scenario_measure`);
* **knobs** -- how a run sets an attribute: the conditions (the rainfall
  rate, the visibility through the fog, the time of day), CARLA's weather
  and how far drivers see, through the ``environment`` config, and the mean and spread
  of each traffic parameter, through the ``traffic`` config
  (:mod:`.traffic_probes`), for the built-in probes that read them
  (:data:`DEFAULT_KNOBS`), and a scenario's ``controls`` for the measures its
  parameters set;
* **where the ego is** -- in the world (the metres a tick stands for in
  coverage) and on Earth (the positions of an exported COD);
* **rules** -- what the conditions do: CARLA's weather, the grip and the
  visibility, a Bayesian network openodd reads
  (:mod:`autoware_carla_scenario.weather_rules`).

:class:`CarlaBackend` derives openodd's :class:`~openodd.Backend` for these;
everything else -- the model, the reader, coverage, the sampler -- is
openodd's.  Importing :mod:`autoware_carla_scenario.odd` makes this openodd's default
backend, so ``openodd.load_odd_binding`` and this package's own read the same.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from openodd import Backend, OddKnob, Rules, set_default_backend
from openodd.model import _missing

from .. import weather_rules
from . import probes
from .scenario_measure import ScenarioMeasure, scenario_measure
from .traffic_probes import TRAFFIC_KNOBS, TRAFFIC_PROBES

__all__ = [
    "BUILT_IN_PROBES",
    "CACHE_ENV",
    "CARLA_BACKEND",
    "DEFAULT_KNOBS",
    "UNSUPPORTED_CONCEPTS",
    "CarlaBackend",
    "sync_cache_env",
    "weather_effects",
]

#: Built-in probe name -> the unit it returns.
_PROBE_UNITS: dict[str, str] = {
    "ego_speed_kph": "km/h",
    "speed_limit_kph": "km/h",
    "lanelet_speed_limit_kph": "km/h",
    "visibility": "m",
    "rainfall_rate_mm_h": "mm/h",
    "fog_visibility_range_m": "m",
    "wind_speed_kmh": "km/h",
    **{
        name: ""
        for name in (
            "in_junction",
            "junction_type",
            "lane_count",
            "lanelet_location",
            "lanelet_subtype",
            "road_type",
            "drivable_area_type",
            "time_of_day",
            "cloud_cover",
            "road_surface_condition",
            "nearby_vehicle_count",
            "illumination",
            "rain",
            "fog",
            "tire_friction",
            "road_wetness",
            "wind",
            "cloudiness",
            "traffic_density",
            "pedestrian_nearby",
        )
    },
}


class _BuiltInProbes(Mapping[str, tuple[Callable[[Any], Any], str]]):
    """Built-in probe name -> (probe, unit), the probe looked up in
    :mod:`.probes` when asked for, so a probe replaced there (a test's
    monkeypatch, a plugin) is the one an ODD read afterwards uses."""

    def __getitem__(self, name: str) -> tuple[Callable[[Any], Any], str]:
        if name in TRAFFIC_PROBES:
            return TRAFFIC_PROBES[name]
        if name not in _PROBE_UNITS:
            raise KeyError(name)
        return getattr(probes, name), _PROBE_UNITS[name]

    def __iter__(self) -> Iterator[str]:
        yield from _PROBE_UNITS
        yield from TRAFFIC_PROBES

    def __len__(self) -> int:
        return len(_PROBE_UNITS) + len(TRAFFIC_PROBES)


#: Built-in probe name -> (probe, the unit it returns).
BUILT_IN_PROBES: Mapping[str, tuple[Callable[[Any], Any], str]] = _BuiltInProbes()

#: Knobs for the built-in probes a run can set, through the ``environment``
#: config (:class:`~autoware_carla_scenario.EnvironmentAction`).
#:
#: * The conditions -- :func:`~.probes.rainfall_rate_mm_h`,
#:   :func:`~.probes.fog_visibility_range_m`, :func:`~.probes.time_of_day`:
#:   the default ODD's -- are written as they are drawn
#:   (``environment.rainfall_rate_mm_h``, ...); CARLA's weather and how far
#:   drivers see follow from them by the weather rules, which the sampler
#:   draws with each case (:func:`weather_effects`) and the runner works out
#:   for anything the case leaves
#:   (:func:`~autoware_carla_scenario.weather_rules.resolve_environment`).
#: * CARLA's own intensities -- :func:`~.probes.rain`, :func:`~.probes.fog`,
#:   :func:`~.probes.road_wetness`, ... -- are written to CARLA's fields, by
#:   level (:func:`~.probes.intensity_level`, read backwards); what follows
#:   from them in the rules (the cloudiness and standing water that come
#:   with the precipitation, the grip that follows the water) is worked out
#:   the same way.  The rain and fog knobs set the LiDARs' noise too
#:   (:mod:`autoware_carla_scenario.sensor.lidar_weather`).
#: * The visibility knob names how far drivers see outright, in metres,
#:   whatever the weather; the tire-friction knob names the grip outright,
#:   whatever the water: the vehicles' wheel friction, which CARLA's physics
#:   alone reads.
#:
#: The value a weather intensity knob draws for each of
#: :func:`~.probes.intensity_level`'s levels, so a drawn value reads back as
#: its bucket.
_INTENSITY_VALUES: dict[str, Any] = {
    "none": 0,
    "light": [1, 30],
    "moderate": [30, 70],
    "heavy": [70, 100],
}

DEFAULT_KNOBS: dict[Callable[..., Any], OddKnob] = {
    # The conditions, as the taxonomy names them (the default ODD's).
    probes.rainfall_rate_mm_h: OddKnob(
        "environment.rainfall_rate_mm_h", range=(0.0, 30.0)
    ),
    probes.fog_visibility_range_m: OddKnob(
        "environment.fog_visibility_range_m", range=(15.0, 5000.0)
    ),
    probes.time_of_day: OddKnob(
        "environment.time_of_day",
        values={label: label for label in probes.TIMES_OF_DAY},
    ),
    probes.wind_speed_kmh: OddKnob("environment.wind_speed_kmh", range=(0.0, 100.0)),
    probes.cloud_cover: OddKnob(
        "environment.cloud_cover",
        values={label: label for label in probes.CLOUD_COVERS},
    ),
    # CARLA's own fields.
    probes.rain: OddKnob(
        "environment.precipitation",
        values=_INTENSITY_VALUES,
    ),
    probes.fog: OddKnob(
        "environment.fog_density",
        values=_INTENSITY_VALUES,
    ),
    probes.illumination: OddKnob(
        "environment.sun_altitude_angle",
        values={
            "day": [15, 90],
            "low_sun": [0, 15],
            "twilight": [-6, 0],
            "night": [-90, -6],
        },
    ),
    probes.visibility: OddKnob(
        "environment.visibility_m",
        range=(15.0, 400.0),
    ),
    # The wetness, which the grip follows (with the standing water, the
    # wetter of the two).
    probes.road_wetness: OddKnob(
        "environment.wetness",
        values=_INTENSITY_VALUES,
    ),
    probes.tire_friction: OddKnob(
        "environment.tire_friction",
        range=(0.3, 1.1),
    ),
    probes.wind: OddKnob(
        "environment.wind_intensity",
        values=_INTENSITY_VALUES,
    ),
    probes.cloudiness: OddKnob(
        "environment.cloudiness",
        values=_INTENSITY_VALUES,
    ),
    # The mean and spread of every traffic parameter (.traffic_probes).
    **TRAFFIC_KNOBS,
}


#: Rule input (a cause) -> the knob (of :data:`DEFAULT_KNOBS`) that writes it.
_RULE_INPUT_KNOBS: dict[str, str] = {
    cause: f"environment.{setting}"
    for setting, cause in weather_rules.CAUSE_SETTINGS.items()
}

#: CARLA's weather fields given as a share, 0-100: what an effect writing
#: one is held to.
_PERCENT_FIELDS: frozenset[str] = frozenset(
    {
        "precipitation",
        "cloudiness",
        "precipitation_deposits",
        "wetness",
        "wind_intensity",
        "fog_density",
    }
)

_EC = "environment_conditions.environment_conditions"

#: What CARLA cannot bring about or measure: an ODD that measures one, or a
#: module that restricts one (or a concept defined by one, the snow
#: intensity), is refused (openodd's ``Backend.unsupported``).
UNSUPPORTED_CONCEPTS: dict[str, str] = {
    f"{_EC}.weather.snow.snowfall_rate": "CARLA renders no snow",
    f"{_EC}.temperature": "CARLA has no temperature",
}


def weather_effects(odd: Any, rules: Optional[Rules] = None) -> list[OddKnob]:
    """The CARLA weather fields a case drawn from *odd* sets through the
    weather *rules* (the backend's): an effect (``OddKnob(quantity=...)``) for
    each field that follows from a condition *odd* names -- an attribute that
    is a cause's concept -- so the sampler draws it with the case, at a
    quantile of its own the flaky search can move.  A field nothing *odd*
    names leads to is left to the world."""
    rules = rules if rules is not None else CARLA_BACKEND.active_rules
    if rules is None:
        return []
    names = {a.name for a in odd.attributes}
    causes = {
        name
        for name, (concept, _unit) in rules.input_concepts.items()
        if concept in names
    }
    if not causes:
        return []
    led: set[str] = set()
    for quantity in rules.order:
        reads = set(rules.inputs(quantity))
        if reads & (causes | led):
            led.add(quantity)
    effects: list[OddKnob] = []
    for setting, quantity in weather_rules.CARLA_QUANTITIES.items():
        if quantity in led and quantity in rules.quantities:
            limits = (0.0, 100.0) if setting in _PERCENT_FIELDS else None
            effects.append(
                OddKnob(f"environment.{setting}", quantity=quantity, limits=limits)
            )
    return effects


def _measure(key: str) -> tuple[ScenarioMeasure, str]:
    probe = scenario_measure(key)
    return probe, probe.unit


def _knob_for(attribute: Any, controls: Mapping[str, OddKnob]) -> Optional[OddKnob]:
    """A scenario measure is set by the scenario's control of that measure --
    but for one that writes a node of the rules (``quantity``), which is the
    scenario's effect, drawn with every case whatever the ODD names."""
    probe = getattr(attribute, "probe", None)
    if isinstance(probe, ScenarioMeasure):
        control = controls.get(probe.key)
        if control is not None and control.quantity is None:
            return control
    return None


@dataclass
class CarlaBackend(Backend):
    """openodd's backend for a CARLA run (see the module docstring)."""

    name: str = "CARLA"
    probes: Mapping[str, tuple[Callable[[Any], Any], str]] = field(
        default_factory=lambda: BUILT_IN_PROBES
    )
    readers: Mapping[str, Callable[[str], tuple[Any, str]]] = field(
        default_factory=lambda: {"measure": _measure}
    )
    knobs: Mapping[Any, OddKnob] = field(default_factory=lambda: dict(DEFAULT_KNOBS))
    knob_for: Optional[Callable[[Any, Mapping[str, OddKnob]], Optional[OddKnob]]] = (
        _knob_for
    )
    # What the conditions do: CARLA's weather, the grip and the visibility
    # (:mod:`..weather_rules`).
    rules: Optional[Rules] = field(default_factory=weather_rules.default_rules)
    rule_inputs: Sequence[str] = weather_rules.WEATHER_INPUTS
    # The knob whose written value each input is, so a tool can follow an
    # ODD attribute through its knob to the rules (Backend.manifest).
    rule_input_knobs: Mapping[str, str] = field(
        default_factory=lambda: dict(_RULE_INPUT_KNOBS)
    )
    # What CARLA cannot bring about (snow, the temperature).
    unsupported: Mapping[str, str] = field(
        default_factory=lambda: dict(UNSUPPORTED_CONCEPTS)
    )

    def position(self, context: Any) -> Optional[tuple[float, float, float]]:
        """The ego's location in the CARLA world, in metres."""
        return probes.ego_position(context)

    def geolocation(self, context: Any) -> Optional[tuple[float, float]]:
        """The ego's latitude and longitude, from the map's geoReference."""
        return probes.ego_geolocation(context)


#: The CARLA backend.
CARLA_BACKEND = CarlaBackend()

#: Where this framework's git sources were cached before openodd: honoured
#: when set (as ``$OPENODD_CACHE``).
CACHE_ENV = "AUTOWARE_CARLA_SCENARIO_ODD_CACHE"


def sync_cache_env() -> None:
    """Point openodd's cache at ours when ours is the one set."""
    ours = os.environ.get(CACHE_ENV)
    if ours:
        os.environ["OPENODD_CACHE"] = ours


def _missing_on_lanelet(lanelet: Any, lanelet_map: Any, routing_graph: Any) -> None:
    """On a planned route too (:mod:`.route`), a concept nothing measures is missing."""
    return None


_missing.on_lanelet = _missing_on_lanelet  # type: ignore[attr-defined]

set_default_backend(CARLA_BACKEND)
