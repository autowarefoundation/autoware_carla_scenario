"""Operational Design Domains: what the runs are measured against.

An ODD is written in Python (:class:`OddDefinition`, checked with Codon like a
scenario: ``scenario-odd check``) or in ASAM OpenODD 1.0 YAML with a binding
file naming its probes (:func:`load_odd_binding`, :func:`load_openodd`),
which builds the same objects.  The runner samples the ODD's attributes on
every tick: their buckets are the ODD coverage, and the ODD's modules decide
which ticks were outside it.  See ``docs/odd.md``.
"""

from . import probes
from .openodd import OpenOddError, load_odd_binding, load_odd_file, load_openodd
from .sources import GitSource, GitSourceError
from .model import (
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
from .registry import (
    DEFAULT_ODD,
    ENTRY_POINT_GROUP,
    default_odd,
    import_callable,
    odd_names,
    register_odd,
    resolve_odd,
)

__all__ = [
    "ILLUMINATION_LEVELS",
    "INTENSITY_LEVELS",
    "NEARBY_RADIUS_M",
    "TRAFFIC_DENSITY_LEVELS",
    "DEFAULT_ODD",
    "ENTRY_POINT_GROUP",
    "OddAttribute",
    "OddCondition",
    "OddDefinition",
    "OddModule",
    "OddVerdict",
    "all_of",
    "any_of",
    "default_odd",
    "ego_speed_kph",
    "fog",
    "illumination",
    "in_junction",
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
]
