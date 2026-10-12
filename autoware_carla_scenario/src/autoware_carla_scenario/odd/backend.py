"""The framework's openodd_py backend: how a CARLA run measures and sets an ODD.

openodd_py reads, judges and samples ODDs; what measures their attributes is
the application's :class:`openodd.Backend`.  :class:`CarlaBackend` is this
framework's:

* **probes** -- the built-in probes (:mod:`.probes`) by the names a binding
  file gives them (``probe: speed_limit_kph``), with the units they read in;
  ``probe: package.module:function`` imports any other;
* **readers** -- ``measure: vehicle_ahead_gap_m`` reads the running
  scenario's measure (:func:`~.scenario_measure.scenario_measure`);
* **knobs** -- how a run sets an attribute, for the sampler: the weather and
  the sun through the ``environment`` config (:data:`DEFAULT_KNOBS`), and an
  attribute read by a scenario measure through the scenario's ``controls``
  for that measure.

:data:`CARLA_BACKEND` is the one the framework reads its ODDs with, and
openodd_py's default backend once :mod:`autoware_carla_scenario.odd` is
imported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import openodd

from . import probes
from .scenario_measure import ScenarioMeasure, scenario_measure

__all__ = ["CARLA_BACKEND", "DEFAULT_KNOBS", "PROBES", "CarlaBackend"]

#: Built-in probe name -> (probe, the unit it returns).
PROBES: dict[str, tuple[Callable[[Any], Any], str]] = {
    "ego_speed_kph": (probes.ego_speed_kph, "km/h"),
    "speed_limit_kph": (probes.speed_limit_kph, "km/h"),
    "lanelet_speed_limit_kph": (probes.lanelet_speed_limit_kph, "km/h"),
    **{
        name: (getattr(probes, name), "")
        for name in (
            "in_junction",
            "lane_count",
            "lanelet_location",
            "lanelet_subtype",
            "illumination",
            "rain",
            "fog",
            "traffic_density",
            "pedestrian_nearby",
        )
    },
}

#: Knobs for the built-in probes a run can set, through the ``environment``
#: config (:class:`~autoware_carla_scenario.EnvironmentAction`).  The ranges
#: are the probes' own levels (:func:`~.probes.intensity_level`,
#: :func:`~.probes.illumination_level`), read backwards.
DEFAULT_KNOBS: dict[Callable[..., Any], openodd.OddKnob] = {
    probes.rain: openodd.OddKnob(
        "environment.precipitation",
        values={"none": 0, "light": [1, 30], "moderate": [30, 70], "heavy": [70, 100]},
    ),
    probes.fog: openodd.OddKnob(
        "environment.fog_density",
        values={"none": 0, "light": [1, 30], "moderate": [30, 70], "heavy": [70, 100]},
    ),
    probes.illumination: openodd.OddKnob(
        "environment.sun_altitude_angle",
        values={
            "day": [15, 90],
            "low_sun": [0, 15],
            "twilight": [-6, 0],
            "night": [-90, -6],
        },
    ),
}


def _read_measure(key: str) -> tuple[Callable[[Any], Any], str]:
    """``measure: key``: the running scenario's measure, in its own unit."""
    probe = scenario_measure(key)
    return probe, probe.unit


@dataclass
class CarlaBackend(openodd.Backend):
    """openodd_py's backend for a CARLA run (see the module docstring)."""

    name: str = "carla"
    probes: Mapping[str, tuple[Callable[[Any], Any], str]] = field(
        default_factory=lambda: dict(PROBES)
    )
    readers: Mapping[str, Callable[[str], tuple[Callable[[Any], Any], str]]] = field(
        default_factory=lambda: {"measure": _read_measure}
    )
    knobs: Mapping[Any, openodd.OddKnob] = field(
        default_factory=lambda: dict(DEFAULT_KNOBS)
    )

    def probe(self, spec: str) -> tuple[Callable[[Any], Any], str]:
        """A probe and its unit; a built-in one as :mod:`.probes` holds it now."""
        if spec in PROBES and self.probes.get(spec) == PROBES[spec]:
            return getattr(probes, spec), PROBES[spec][1]
        return super().probe(spec)

    def knob(
        self, attribute: Any, controls: Mapping[str, openodd.OddKnob]
    ) -> Optional[openodd.OddKnob]:
        """How a run sets *attribute*: openodd_py's lookup, and for an ODD
        written in Python, the scenario's control of the measure its probe reads."""
        knob = super().knob(attribute, controls)
        if knob is None and isinstance(attribute.probe, ScenarioMeasure):
            knob = controls.get(attribute.probe.key)
        return knob


#: The backend the framework reads, measures and samples its ODDs with.
CARLA_BACKEND = CarlaBackend()
