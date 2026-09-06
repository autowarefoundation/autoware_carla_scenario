"""Which projection a Lanelet2 map wants, and the projector that reads it.

Lanelet2 stores geography, not metres: a map is read *through* a projection,
and reading one map two different ways puts the same lanelet in two different
places.  That is not a rounding difference.  The MGRS correction the coordinate
stack applies is a single translation, so it only holds while the whole map
projects into one continuous MGRS square; a map whose ``geoReference`` sits on a
100 km grid line -- or on the lon 0 UTM zone boundary, where the CARLA towns put
theirs -- has half its geometry land tens of kilometres from the rest.

So the choice lives here, in one module, rather than in each place that happens
to load a map.  Two do: :class:`~...coordinate.map_manager.MapManager` during a
run, and ``sweeper.map_loader`` before CARLA is even started.  They resolve the
spawn a scenario uses -- the sweeper picks the lanelet, the run poses the
vehicle on it -- so a disagreement between them is a vehicle placed somewhere
the sweep never looked.

Which projection a map wants is not a guess.  Autoware ships it beside the map
in ``map_projector_info.yaml``, and that is what this module reads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

# autoware_lanelet2_extension_python must be imported before lanelet2 to register
# Autoware-specific regulatory elements (road_marking, detection_area, etc.)
from autoware_lanelet2_extension_python.projection import (
    MGRSProjector,
    TransverseMercatorProjector,
)
import lanelet2.io
import lanelet2.projection
import yaml

logger = logging.getLogger(__name__)

#: Autoware's projection descriptor, written next to the Lanelet2 map.
PROJECTOR_INFO_FILENAME = "map_projector_info.yaml"

#: Lanelet2 projector to read each Autoware ``projector_type`` with.
#:
#: ``Local`` maps store the map frame in the ``local_x``/``local_y`` tags and
#: derive their lat/lon from it; ``UtmProjector`` at the map origin reproduces
#: those metres exactly (checked against a CARLA town to the centimetre), and
#: it is the only one of these that stays continuous across a UTM zone
#: boundary, which is where a CARLA geoReference of (0, 0) sits.
PROJECTOR_BY_AUTOWARE_TYPE = {
    "Local": "utm",
    "LocalCartesian": "local_cartesian",
    "LocalCartesianUTM": "local_cartesian",
    "MGRS": "mgrs",
    "TransverseMercator": "transverse_mercator",
}

#: Used when no descriptor is found, which is what every map did before this
#: module read one.
DEFAULT_PROJECTOR = "mgrs"


def read_projector_type(lanelet2_path: Path) -> Optional[str]:
    """Return the projector named by the map's ``map_projector_info.yaml``.

    ``None`` when there is no descriptor beside the map, or it names a
    projection this module cannot read the map with.
    """
    info_path = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
    if not info_path.is_file():
        return None
    try:
        info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read %s; falling back", info_path, exc_info=True)
        return None
    if not isinstance(info, dict):
        # yaml.safe_load is happy with a list or a bare scalar at the root, and
        # .get() on either is an AttributeError rather than the fallback.
        logger.warning(
            "%s is not a mapping (%s); falling back",
            info_path,
            type(info).__name__,
        )
        return None
    autoware_type = info.get("projector_type")
    projector = (
        PROJECTOR_BY_AUTOWARE_TYPE.get(autoware_type)
        if isinstance(autoware_type, str)
        else None
    )
    if projector is None:
        logger.warning(
            "%s names projector_type %r, which this module does not read; "
            "falling back to %s",
            info_path,
            autoware_type,
            DEFAULT_PROJECTOR,
        )
        return None
    logger.info("%s says projector_type %s", info_path.name, autoware_type)
    return projector


def make_projector(projector_type: str, origin: "lanelet2.io.Origin") -> Any:
    """Return the Lanelet2 projector named by *projector_type*, at *origin*."""
    if projector_type == "mgrs":
        return MGRSProjector(origin)
    if projector_type == "utm":
        return lanelet2.projection.UtmProjector(origin)
    if projector_type == "local_cartesian":
        return lanelet2.projection.LocalCartesianProjector(origin)
    if projector_type == "transverse_mercator":
        return TransverseMercatorProjector(origin)
    raise ValueError(
        f"Unknown projector_type {projector_type!r}; expected one of "
        f"{sorted(set(PROJECTOR_BY_AUTOWARE_TYPE.values()))}"
    )


def resolve_projector(
    lanelet2_path: Path,
    origin: "lanelet2.io.Origin",
    projector_type: Optional[str] = None,
) -> tuple[Any, str]:
    """Return the projector to read *lanelet2_path* with, and its name.

    *projector_type* forces the choice; leave it ``None`` to take it from the
    map's own descriptor, and ``mgrs`` when there is no descriptor.
    """
    if projector_type is None:
        projector_type = read_projector_type(lanelet2_path) or DEFAULT_PROJECTOR
    return make_projector(projector_type, origin), projector_type
