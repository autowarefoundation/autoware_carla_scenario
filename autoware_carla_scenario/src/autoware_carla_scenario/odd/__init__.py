"""Operational Design Domains: what the runs are measured against.

The ODD model, its OpenODD reader and its sampler are openodd_py's
(``openodd``); this package is the framework's side of it: the backend that
measures and sets an ODD in a CARLA run (:class:`CarlaBackend`, made
openodd_py's default backend on import), its probes, the ODDs a run can be
given by name (:func:`resolve_odd`), and the route check.

An ODD is written in Python (:class:`OddDefinition`, checked with Codon like a
scenario: ``scenario-odd check``) or in ASAM OpenODD 1.0 YAML with a binding
file naming its probes (:func:`load_odd_binding`, :func:`load_openodd`),
which builds the same objects.  The runner samples the ODD's attributes on
every tick: their buckets are the ODD coverage, and the ODD's modules decide
which ticks were outside it.  Before a run, :func:`plan_route_coverage` checks
the ego's planned route against it on the Lanelet2 map alone.  See
``docs/odd.md``.
"""

import openodd
from openodd import (
    UNDECIDED,
    OddAttribute,
    OddCondition,
    OddDefinition,
    OddModule,
    OddVerdict,
    all_of,
    any_of,
    module_holds,
    read_probe,
)

from . import probes
from .backend import CARLA_BACKEND, CarlaBackend
from .loader import (
    GitSource,
    GitSourceError,
    OpenOddError,
    load_odd_binding,
    load_odd_file,
    load_openodd,
)
from .probes import (
    ILLUMINATION_LEVELS,
    INTENSITY_LEVELS,
    NEARBY_RADIUS_M,
    TRAFFIC_DENSITY_LEVELS,
    ego_speed_kph,
    fog,
    illumination,
    in_junction,
    lane_count,
    lanelet_location,
    lanelet_speed_limit_kph,
    lanelet_subtype,
    pedestrian_nearby,
    rain,
    reset_probes,
    speed_limit_kph,
    traffic_density,
)
from .scenario_measure import ScenarioMeasure, scenario_measure
from .route import (
    PlannedRoute,
    RouteCoverage,
    RouteCoverageSummary,
    RouteError,
    RouteLanelet,
    combine_route_coverage,
    plan_route,
    plan_route_coverage,
)
from .sampler import (
    DEFAULT_KNOBS,
    OddDraws,
    OddKnob,
    OddSample,
    OddSampler,
    knobs_from_mapping,
)
from .registry import (
    DEFAULT_ODD,
    ENTRY_POINT_GROUP,
    default_odd,
    import_callable,
    odd_names,
    register_odd,
    resolve_odd,
)

openodd.set_default_backend(CARLA_BACKEND)

__all__ = [
    "CARLA_BACKEND",
    "CarlaBackend",
    "ILLUMINATION_LEVELS",
    "INTENSITY_LEVELS",
    "NEARBY_RADIUS_M",
    "TRAFFIC_DENSITY_LEVELS",
    "DEFAULT_KNOBS",
    "DEFAULT_ODD",
    "ENTRY_POINT_GROUP",
    "OddAttribute",
    "OddCondition",
    "OddDefinition",
    "OddDraws",
    "OddKnob",
    "OddModule",
    "OddSample",
    "OddSampler",
    "OddVerdict",
    "PlannedRoute",
    "RouteCoverage",
    "RouteCoverageSummary",
    "RouteError",
    "RouteLanelet",
    "UNDECIDED",
    "combine_route_coverage",
    "plan_route",
    "plan_route_coverage",
    "all_of",
    "any_of",
    "default_odd",
    "ego_speed_kph",
    "fog",
    "illumination",
    "in_junction",
    "knobs_from_mapping",
    "lane_count",
    "lanelet_location",
    "lanelet_speed_limit_kph",
    "lanelet_subtype",
    "GitSource",
    "GitSourceError",
    "OpenOddError",
    "load_odd_binding",
    "load_odd_file",
    "load_openodd",
    "module_holds",
    "import_callable",
    "odd_names",
    "pedestrian_nearby",
    "probes",
    "read_probe",
    "rain",
    "register_odd",
    "reset_probes",
    "resolve_odd",
    "speed_limit_kph",
    "traffic_density",
    "ScenarioMeasure",
    "scenario_measure",
]
