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
in ``map_projector_info.yaml``, and that is what this module reads -- both the
projection and, when the descriptor carries one, the origin it is anchored at.
"""

from __future__ import annotations

import logging
import re
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
import yaml  # type: ignore[import-untyped]

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


def _projector_info(lanelet2_path: Path) -> dict[str, Any]:
    """Return the map's descriptor as a mapping, empty when there is none.

    Both things read out of it -- the projection and the origin -- come through
    here, so "is there a descriptor, and is it usable" is decided once.
    """
    info_path = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
    if not info_path.is_file():
        return {}
    try:
        info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        logger.warning("Could not read %s; falling back", info_path, exc_info=True)
        return {}
    if not isinstance(info, dict):
        # yaml.safe_load is happy with a list or a bare scalar at the root, and
        # .get() on either is an AttributeError rather than the fallback.
        logger.warning(
            "%s is not a mapping (%s); falling back",
            info_path,
            type(info).__name__,
        )
        return {}
    return info


def read_projector_type(lanelet2_path: Path) -> Optional[str]:
    """Return the projector named by the map's ``map_projector_info.yaml``.

    ``None`` when there is no descriptor beside the map, or it names a
    projection this module cannot read the map with.
    """
    info_path = Path(lanelet2_path).parent / PROJECTOR_INFO_FILENAME
    info = _projector_info(Path(lanelet2_path))
    if not info:
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


#: Where a CARLA town's Lanelet2 map is anchored.  Its nodes carry lat/lon
#: derived from ``local_x``/``local_y`` against an origin of (0, 0), which is
#: what ``projector_type: Local`` means and what the town's own OpenDRIVE
#: ``geoReference`` says too (``+lat_0=0 +lon_0=0``).
NULL_ISLAND = (0.0, 0.0, 0.0)


def read_map_origin(lanelet2_path: Path) -> Optional[tuple[float, float, float]]:
    """Return the ``map_origin`` in the map's descriptor, if it names one.

    Autoware writes an origin beside maps whose projection needs one --
    ``LocalCartesian`` and ``TransverseMercator``.  A ``Local`` or ``MGRS`` map
    carries none, because neither projection is anchored by one.

    Returns:
        ``(latitude, longitude, altitude)``, or ``None`` when the descriptor is
        missing, unreadable, or names no origin.
    """
    origin = _projector_info(Path(lanelet2_path)).get("map_origin")
    if not isinstance(origin, dict):
        return None
    try:
        return (
            float(origin["latitude"]),
            float(origin["longitude"]),
            float(origin.get("altitude", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "%s beside %s has a map_origin this module cannot read",
            PROJECTOR_INFO_FILENAME,
            lanelet2_path,
        )
        return None


def _parse_geo_reference(xodr_content: str) -> tuple[float, float, float]:
    """Extract (lat, lon, alt) from the geoReference PROJ string in an XODR file.

    Parameters
    ----------
    xodr_content:
        Full text content of the .xodr file.

    Returns
    -------
    tuple[float, float, float]
        (latitude, longitude, altitude).  Altitude defaults to 0.0 if absent.

    Raises
    ------
    ValueError
        If lat_0 or lon_0 cannot be found.
    """
    # Extract the geoReference element content
    geo_ref_match = re.search(
        r"<geoReference>\s*<!\[CDATA\[(.*?)\]\]>\s*</geoReference>",
        xodr_content,
        re.DOTALL,
    )
    if geo_ref_match is None:
        # Fallback: try without CDATA wrapper
        geo_ref_match = re.search(
            r"<geoReference>(.*?)</geoReference>",
            xodr_content,
            re.DOTALL,
        )
    if geo_ref_match is None:
        raise ValueError("No <geoReference> element found in XODR file.")

    proj_string = geo_ref_match.group(1)

    lat_match = re.search(r"\+lat_0=([-\d.]+)", proj_string)
    lon_match = re.search(r"\+lon_0=([-\d.]+)", proj_string)

    if lat_match is None:
        raise ValueError(f"Could not find +lat_0 in geoReference: {proj_string!r}")
    if lon_match is None:
        raise ValueError(f"Could not find +lon_0 in geoReference: {proj_string!r}")

    lat = float(lat_match.group(1))
    lon = float(lon_match.group(1))

    alt_match = re.search(r"\+h_0=([-\d.]+)", proj_string)
    alt = float(alt_match.group(1)) if alt_match else 0.0

    return lat, lon, alt


def map_origin(
    lanelet2_path: Path, xodr_path: Optional[Path] = None
) -> tuple[tuple[float, float, float], str]:
    """Return where a map is anchored, and what said so.

    In order of authority: the map's own ``map_projector_info.yaml``, which is
    what Autoware itself reads; the OpenDRIVE ``geoReference``, which describes
    the same world when there is one; and (0, 0), which is where a map recorded
    on a CARLA town sits and is the only remaining answer for a map with no
    OpenDRIVE and no stated origin.

    This lives beside :func:`resolve_projector` because the two answer halves of
    one question -- how to read a map's geometry -- and both loaders in this
    framework have to give the same answer to both.  The sweeper picks the
    lanelet a vehicle spawns on and the run poses the vehicle on it; anchoring
    them differently puts the vehicle somewhere the sweep never looked.

    Raises:
        ValueError: If a given OpenDRIVE file lacks a valid ``geoReference``.
    """
    stated = read_map_origin(lanelet2_path)
    if stated is not None:
        return stated, PROJECTOR_INFO_FILENAME
    if xodr_path is not None:
        path = Path(xodr_path)
        return _parse_geo_reference(path.read_text(encoding="utf-8")), path.name
    return NULL_ISLAND, "the CARLA town default"


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
