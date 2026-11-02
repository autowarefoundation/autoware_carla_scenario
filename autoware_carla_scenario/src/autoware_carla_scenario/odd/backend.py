"""This framework as an openodd backend: what measures an ODD in CARLA.

`openodd <https://github.com/hakuturu583/openodd_py>`_ reads, judges,
versions and samples ODDs; what it leaves to an application is how the
attributes are measured and set. Here:

* **probes** -- the built-in probes a binding file names (``probe:
  speed_limit_kph``), each reading the CARLA world (:mod:`.probes`);
* **readers** -- ``measure: <key>``: the running scenario's measure *key*
  (:func:`~.scenario_measure.scenario_measure`);
* **knobs** -- how a run sets an attribute: the weather, the sun and how far
  drivers see, through the ``environment`` config, and the mean and spread
  of each traffic parameter, through the ``traffic`` config
  (:mod:`.traffic_probes`), for the built-in probes that read them
  (:data:`DEFAULT_KNOBS`), and a scenario's ``controls`` for the measures its
  parameters set;
* **where the ego is** -- in the world (the metres a tick stands for in
  coverage) and on Earth (the positions of an exported COD);
* **rules** -- the grip and the visibility the weather leaves, conditional
  distributions openodd reads (:mod:`autoware_carla_scenario.weather_rules`).

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
    "CarlaBackend",
    "sync_cache_env",
]

#: Built-in probe name -> the unit it returns.
_PROBE_UNITS: dict[str, str] = {
    "ego_speed_kph": "km/h",
    "speed_limit_kph": "km/h",
    "lanelet_speed_limit_kph": "km/h",
    "visibility": "m",
    **{
        name: ""
        for name in (
            "in_junction",
            "lane_count",
            "lanelet_location",
            "lanelet_subtype",
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
#: config (:class:`~autoware_carla_scenario.EnvironmentAction`).  The ranges
#: are the probes' own levels (:func:`~.probes.intensity_level`,
#: :func:`~.probes.illumination_level`), read backwards.  The rain and fog
#: knobs set the LiDARs' noise too
#: (:mod:`autoware_carla_scenario.sensor.lidar_weather`).  The rain, fog and
#: illumination knobs set how far drivers see as well: the visibility follows
#: the weather (:func:`~autoware_carla_scenario.visibility.visibility_range`),
#: and a traffic backend that models drivers' sight (``sumo``) holds its
#: traffic to it.  The visibility knob names it outright, in metres, whatever
#: the weather.  The road-wetness knob sets the road's water, which the
#: vehicles' grip follows (:func:`~autoware_carla_scenario.actions.environment.tire_friction_range`),
#: and the tire-friction knob names the grip outright, whatever the water:
#: the vehicles' wheel friction, which CARLA's physics alone reads.
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
    # environment.rain rather than environment.precipitation: precipitation
    # alone barely shows in a camera image, so a "heavy rain" case would look
    # like a dry day (EnvironmentAction's ``rain``).
    probes.rain: OddKnob(
        "environment.rain",
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
    # The wetness, which the grip follows (with rain's standing water, the
    # larger of the two).
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


#: Rule input -> the knob (of :data:`DEFAULT_KNOBS`) that writes it.
#: ``environment.rain`` writes ``precipitation`` (and the ``wetness`` and
#: standing water CARLA's rain presets pair with it).
_RULE_INPUT_KNOBS: dict[str, str] = {
    "precipitation": "environment.rain",
    "fog_density": "environment.fog_density",
    "sun_altitude_angle": "environment.sun_altitude_angle",
    "wetness": "environment.wetness",
    "wind_intensity": "environment.wind_intensity",
    "cloudiness": "environment.cloudiness",
}


def _measure(key: str) -> tuple[ScenarioMeasure, str]:
    probe = scenario_measure(key)
    return probe, probe.unit


def _knob_for(attribute: Any, controls: Mapping[str, OddKnob]) -> Optional[OddKnob]:
    """A scenario measure is set by the scenario's control of that measure."""
    probe = getattr(attribute, "probe", None)
    if isinstance(probe, ScenarioMeasure):
        return controls.get(probe.key)
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
    # The grip and the visibility the weather leaves (:mod:`..weather_rules`).
    rules: Optional[Rules] = field(default_factory=weather_rules.default_rules)
    rule_inputs: Sequence[str] = weather_rules.WEATHER_INPUTS
    # The knob whose written value each input is, so a tool can follow an
    # ODD attribute through its knob to the rules (Backend.manifest).
    rule_input_knobs: Mapping[str, str] = field(
        default_factory=lambda: dict(_RULE_INPUT_KNOBS)
    )
    rule_input_text: Mapping[str, str] = field(
        default_factory=lambda: {
            "water": "The larger of wetness and precipitation_deposits: the "
            "road's water, which the grip follows.",
        }
    )

    def rule_values(self, values: Mapping[str, float]) -> dict[str, float]:
        """The weather's fields, and ``water``, which the grip follows."""
        return weather_rules.rule_values(values)

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
