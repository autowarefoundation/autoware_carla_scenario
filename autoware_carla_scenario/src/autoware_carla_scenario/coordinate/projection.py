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
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# autoware_lanelet2_extension_python must be imported before lanelet2.  Its
# `regulatory_elements` module is what registers the Autoware-specific ones
# (road_marking, detection_area, ...) with the Lanelet2 parser -- without it an
# Autoware map fails to load.  Every loader in this package goes through this
# module, so importing it here covers all of them.
import autoware_lanelet2_extension_python.regulatory_elements  # noqa: F401
from autoware_lanelet2_extension_python.projection import (
    MGRSProjector,
    TransverseMercatorProjector,
)
import lanelet2.core
import lanelet2.io
import lanelet2.projection

logger = logging.getLogger(__name__)

#: Autoware's projection descriptor, written next to the Lanelet2 map.
PROJECTOR_INFO_FILENAME = "map_projector_info.yaml"

#: Lanelet2 projector to read each Autoware ``projector_type`` with.
#:
#: ``Local`` maps store the map frame in the ``local_x``/``local_y`` tags, and
#: :data:`LOCAL_XY` reads it from there, as Autoware's own loader does.
PROJECTOR_BY_AUTOWARE_TYPE = {
    "Local": "local_xy",
    "LocalCartesian": "local_cartesian",
    "LocalCartesianUTM": "local_cartesian",
    "MGRS": "mgrs",
    "TransverseMercator": "transverse_mercator",
}

#: Used when no descriptor is found, which is what every map did before this
#: module read one.
DEFAULT_PROJECTOR = "mgrs"

#: Reads each node at its ``local_x``/``local_y`` (and ``local_z``, when it has
#: one), the map frame itself.  A node without them is read from its lat/lon
#: through ``UtmProjector`` at the map origin -- what ``utm`` reads every node
#: from, and what this reading used to be.  UTM reproduces the local tags only
#: where the lat/lon were themselves derived from them with UTM (the official
#: CARLA towns), but it is the only projection here that stays continuous
#: across a UTM zone boundary, which is where a CARLA geoReference of (0, 0)
#: sits.
LOCAL_XY = "local_xy"

#: Farther apart than this (m), a ``local_xy`` map's local tags and lat/lon
#: are warned about: a tool that reads the other one places the map elsewhere.
LOCAL_READINGS_TOLERANCE = 0.01


def _projector_info(lanelet2_path: Path) -> dict[str, Any]:
    """Return the map's descriptor as a mapping, empty when there is none.

    Both things read out of it -- the projection and the origin -- come through
    here, so "is there a descriptor, and is it usable" is decided once.

    yaml is imported here, the one place that reads it: the check
    (docs/typecheck.md) compiles neither this nor the two readers, which
    hand out YAML values of any type.
    """
    import yaml  # type: ignore[import-untyped]

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

    # str() of the str a matched group is: Codon 0.19 cannot settle the type
    # of a group (`Optional[str]`) that is then used as a str.
    proj_string = str(geo_ref_match.group(1))

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
    if TYPE_CHECKING:
        # read_map_origin() reads the descriptor's YAML values (`Any`), which
        # Codon cannot express (UNCALLED in typecheck/library.py): the check
        # takes a map that states no origin.
        stated: Optional[tuple[float, float, float]] = None
    else:
        stated = read_map_origin(lanelet2_path)
    if stated is not None:
        origin: tuple[float, float, float] = stated
        return origin, PROJECTOR_INFO_FILENAME
    if xodr_path is not None:
        path = Path(xodr_path)
        return _parse_geo_reference(path.read_text(encoding="utf-8")), path.name
    return NULL_ISLAND, "the CARLA town default"


def make_projector(
    projector_type: str, origin: "lanelet2.io.Origin"
) -> lanelet2.projection.Projector:
    """Return the Lanelet2 projector named by *projector_type*, at *origin*.

    Every one of them is a ``lanelet::Projector`` in C++, which is what
    ``lanelet2.io.load()`` takes; Python's binding registers none as derived
    from ``Projector`` (nor do lanelet2's stubs), hence the ignores.  The
    check's lanelet2 model derives them as C++ does.
    """
    if projector_type == "mgrs":
        return MGRSProjector(origin)
    if projector_type in ("utm", LOCAL_XY):
        return lanelet2.projection.UtmProjector(origin)  # type: ignore[return-value]
    if projector_type == "local_cartesian":
        return lanelet2.projection.LocalCartesianProjector(origin)  # type: ignore[return-value]
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
) -> tuple[lanelet2.projection.Projector, str]:
    """Return the projector to read *lanelet2_path* with, and its name.

    *projector_type* forces the choice; leave it ``None`` to take it from the
    map's own descriptor, and ``mgrs`` when there is no descriptor.
    """
    if projector_type is not None:
        name: str = projector_type
    elif TYPE_CHECKING:
        # read_projector_type() reads the descriptor's YAML values (`Any`),
        # which Codon cannot express (UNCALLED in typecheck/library.py): the
        # check takes a map without one.
        name = DEFAULT_PROJECTOR
    else:
        name = read_projector_type(lanelet2_path) or DEFAULT_PROJECTOR
    return make_projector(name, origin), name


def _read_local_tags(
    lanelet_map: lanelet2.core.LaneletMap, lanelet2_path: Path
) -> lanelet2.core.LaneletMap:
    """Return *lanelet_map*, read from lat/lon, with its points at their local tags.

    Every point carrying ``local_x``/``local_y`` moves there (``local_z`` too,
    when it has one); one without them keeps its lat/lon reading.  How far the
    two readings disagree is logged, as a warning beyond
    :data:`LOCAL_READINGS_TOLERANCE`.

    The map is rebuilt from its primitives once they move, so no spatial index
    built from where they were is left to answer a search.
    """
    tagged = 0
    untagged = 0
    worst = 0.0
    for point in lanelet_map.pointLayer:
        attributes = point.attributes
        if "local_x" not in attributes or "local_y" not in attributes:
            untagged += 1
            continue
        tagged += 1
        x = float(attributes["local_x"])
        y = float(attributes["local_y"])
        worst = max(worst, math.hypot(point.x - x, point.y - y))
        point.x = x
        point.y = y
        if "local_z" in attributes:
            point.z = float(attributes["local_z"])

    if tagged == 0:
        logger.warning(
            "%s is a Local map, but no node carries local_x/local_y; "
            "reading every node from its lat/lon",
            lanelet2_path.name,
        )
        return lanelet_map
    logger.info(
        "Reading %s from local_x/local_y (%d nodes; %d without them from "
        "their lat/lon)",
        lanelet2_path.name,
        tagged,
        untagged,
    )
    if worst > LOCAL_READINGS_TOLERANCE:
        logger.warning(
            "%s: the lat/lon of its nodes and their local_x/local_y disagree "
            "by up to %.3f m; read from local_x/local_y (projector_type utm "
            "reads the lat/lon)",
            lanelet2_path.name,
            worst,
        )

    rebuilt = lanelet2.core.LaneletMap()
    for layer in (
        lanelet_map.pointLayer,
        lanelet_map.lineStringLayer,
        lanelet_map.polygonLayer,
        lanelet_map.laneletLayer,
        lanelet_map.areaLayer,
        lanelet_map.regulatoryElementLayer,
    ):
        for element in layer:
            rebuilt.add(element)
    return rebuilt


def load_lanelet2(
    lanelet2_path: Path,
    origin: "lanelet2.io.Origin",
    projector_type: Optional[str] = None,
) -> tuple[lanelet2.core.LaneletMap, lanelet2.projection.Projector, str]:
    """Load *lanelet2_path*, anchored at *origin*; the one way this package does.

    The projector is :func:`resolve_projector`'s; under :data:`LOCAL_XY` (a
    ``projector_type: Local`` map) the points are then moved to their local
    tags.

    Returns:
        The map, the projector its lat/lon were read with, and its name.

    Raises:
        ValueError: For a *projector_type* this module does not know.
    """
    path = Path(lanelet2_path)
    projector, name = resolve_projector(path, origin, projector_type)
    lanelet_map = lanelet2.io.load(str(path), projector)
    read = _read_local_tags(lanelet_map, path) if name == LOCAL_XY else lanelet_map
    return read, projector, name
