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
#: :func:`load_lanelet2` reads it from there (:data:`LOCAL_XY`).  The projector
#: named here reads their lat/lon instead: what a node without those tags
#: falls back to, and the whole map under :data:`UTM_LATLON`.  ``UtmProjector``
#: at the map origin reproduces the local tags only where the lat/lon were
#: themselves derived from them with UTM (the official CARLA towns), but it is
#: the only one of these that stays continuous across a UTM zone boundary,
#: which is where a CARLA geoReference of (0, 0) sits.
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

#: How a ``projector_type: Local`` map is read (``map.local_projection``):
#: from each node's ``local_x``/``local_y`` (and ``local_z``, when present),
#: which is the map frame itself and what Autoware's own loader reads ...
LOCAL_XY = "local_xy"
#: ... or from each node's lat/lon through the ``Local`` projector above,
#: which is what this module did before it read the local tags.
UTM_LATLON = "utm_latlon"
LOCAL_PROJECTIONS = (LOCAL_XY, UTM_LATLON)
DEFAULT_LOCAL_PROJECTION = LOCAL_XY

#: Farther apart than this (m), the two readings of a ``Local`` map are
#: warned about: a map whose lat/lon and local tags disagree reads differently
#: in a tool that takes the other one.
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


def is_local_map(lanelet2_path: Path) -> bool:
    """Whether the map's ``map_projector_info.yaml`` says ``projector_type: Local``."""
    return _projector_info(Path(lanelet2_path)).get("projector_type") == "Local"


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
    if projector_type == "utm":
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


def resolve_local_projection(local_projection: Optional[str] = None) -> str:
    """Return how a ``Local`` map is read: *local_projection*, or the default.

    Raises:
        ValueError: If *local_projection* is neither of :data:`LOCAL_PROJECTIONS`.
    """
    if local_projection is None:
        return DEFAULT_LOCAL_PROJECTION
    if local_projection not in LOCAL_PROJECTIONS:
        raise ValueError(
            f"Unknown local_projection {local_projection!r}; expected one of "
            f"{list(LOCAL_PROJECTIONS)}"
        )
    return local_projection


def _read_local_tags(
    lanelet_map: lanelet2.core.LaneletMap, lanelet2_path: Path, reading: str
) -> lanelet2.core.LaneletMap:
    """Return *lanelet_map* read the way *reading* says, and log which.

    The map comes in read from lat/lon.  Under :data:`LOCAL_XY` every point
    carrying ``local_x``/``local_y`` is moved there (``local_z`` too, when it
    has one); one without them keeps its lat/lon reading.  Under either, the
    two readings are compared wherever both exist, and a disagreement beyond
    :data:`LOCAL_READINGS_TOLERANCE` is warned about.

    The map is rebuilt from its primitives after they move, so no spatial
    index built from where they were is left to answer a search.
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
        if reading == LOCAL_XY:
            point.x = x
            point.y = y
            if "local_z" in attributes:
                point.z = float(attributes["local_z"])

    if reading == LOCAL_XY and tagged == 0:
        logger.warning(
            "%s is a Local map, but no node carries local_x/local_y; "
            "reading every node from its lat/lon",
            lanelet2_path.name,
        )
    elif reading == LOCAL_XY:
        logger.info(
            "Reading the Local map %s from local_x/local_y (%d nodes; "
            "%d without them from their lat/lon)",
            lanelet2_path.name,
            tagged,
            untagged,
        )
    else:
        logger.info(
            "Reading the Local map %s from lat/lon through UTM (%s)",
            lanelet2_path.name,
            UTM_LATLON,
        )
    if worst > LOCAL_READINGS_TOLERANCE:
        logger.warning(
            "%s: the lat/lon of its nodes and their local_x/local_y disagree "
            "by up to %.3f m; read with %s",
            lanelet2_path.name,
            worst,
            reading,
        )
    if reading != LOCAL_XY or tagged == 0:
        return lanelet_map

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
    local_projection: Optional[str] = None,
) -> tuple[lanelet2.core.LaneletMap, lanelet2.projection.Projector, str]:
    """Load *lanelet2_path*, anchored at *origin*; the one way this package does.

    The projector is :func:`resolve_projector`'s.  A map whose descriptor says
    ``projector_type: Local`` is then read from its ``local_x``/``local_y``
    tags, or from its lat/lon, as *local_projection* says
    (:data:`LOCAL_PROJECTIONS`; :data:`DEFAULT_LOCAL_PROJECTION` when unset).
    A *projector_type* given names the projection to read lat/lon with, so it
    turns the local tags off.

    Returns:
        The map, the projector it was read with, and the reading's name: the
        projector's, or :data:`LOCAL_XY` for a map read from its local tags.

    Raises:
        ValueError: For a *projector_type* or *local_projection* this module
            does not know.
    """
    path = Path(lanelet2_path)
    reading = resolve_local_projection(local_projection)
    projector, name = resolve_projector(path, origin, projector_type)
    lanelet_map = lanelet2.io.load(str(path), projector)
    if projector_type is not None:
        return lanelet_map, projector, name
    if TYPE_CHECKING:
        # is_local_map() reads the descriptor's YAML values (`Any`), and
        # _read_local_tags() moves points and rebuilds the map, which the
        # lanelet2 model has no setters for (UNCALLED in typecheck/library.py):
        # the check takes a map that is not Local.
        local = False
    else:
        local = is_local_map(path)
    if not local:
        return lanelet_map, projector, name
    if TYPE_CHECKING:
        read = lanelet_map
    else:
        read = _read_local_tags(lanelet_map, path, reading)
    return read, projector, (LOCAL_XY if reading == LOCAL_XY else name)
