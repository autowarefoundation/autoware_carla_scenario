"""Lightweight Lanelet2 map loader for pre-simulation use.

The sweeper runs *before* CARLA is started, so we cannot rely on
:class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
(which needs a CARLA world for z-offset computation).  This module
provides a minimal loader that needs the Lanelet2 ``.osm`` and, at most, an
OpenDRIVE ``.xodr`` to read the map's origin out of.

The OpenDRIVE is optional, because a scenario is written against Lanelet2:
lanelets are what a spawn, a goal and a constraint search all name, and reading
them needs a projection and an origin rather than a road network.  An Autoware
map states both beside itself -- ``map_projector_info.yaml`` -- and a map
recorded on a CARLA town is anchored at (0, 0), which is what its town's own
``geoReference`` says as well.  So the OpenDRIVE is asked for when it is there
and inferred from when it is not; a *run* still needs one, and gets it from
CARLA (see :mod:`autoware_carla_scenario.maps.opendrive`).

It is lighter, not different: the projection comes from the same
:mod:`~autoware_carla_scenario.coordinate.projection` module ``MapManager``
uses.  These two loaders decide one thing between them -- the sweeper picks the
lanelet a vehicle spawns on, the run poses the vehicle on it -- so reading the
same map two ways would place a vehicle somewhere the sweep never looked.  On a
CARLA town, whose geoReference sits on the lon 0 UTM zone boundary, that gap is
tens of kilometres.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lanelet2.core
import lanelet2.io

from ..coordinate.projection import map_origin, resolve_projector

if TYPE_CHECKING:
    from ..maps import MapPaths

logger = logging.getLogger(__name__)


def load_lanelet2_map(
    lanelet2_path: str | Path,
    xodr_path: str | Path | None = None,
    projector_type: str | None = None,
) -> Any:
    """Load a Lanelet2 map, anchored at the origin its map states.

    This is a lightweight alternative to :meth:`MapManager.initialize` that
    does *not* require a CARLA connection or RoadNetwork loading.

    Args:
        lanelet2_path: Path to the Lanelet2 ``.osm`` file.
        xodr_path: Path to the OpenDRIVE ``.xodr`` file, read only for its
            ``geoReference``.  Optional: see
            :func:`~autoware_carla_scenario.coordinate.projection.map_origin`
            for what is used instead, and the module docstring for why a
            scenario can be written without one.
        projector_type: Which projection to read the map with.  Leave it unset
            to take it from the map's own ``map_projector_info.yaml``, exactly
            as ``MapManager`` does.

    Returns:
        A ``lanelet2.core.LaneletMap`` instance.

    Raises:
        FileNotFoundError: If a named file does not exist.
        ValueError: If a given XODR file lacks a valid ``geoReference``.
    """
    lanelet2_path = Path(lanelet2_path)
    xodr_path = Path(xodr_path) if xodr_path is not None else None

    if not lanelet2_path.exists():
        raise FileNotFoundError(f"Lanelet2 file not found: {lanelet2_path}")
    if xodr_path is not None and not xodr_path.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {xodr_path}")

    (lat, lon, _alt), anchored_by = map_origin(lanelet2_path, xodr_path)
    logger.info(
        "Anchoring %s at (%s, %s) from %s", lanelet2_path.name, lat, lon, anchored_by
    )

    origin = lanelet2.io.Origin(lat, lon)
    projector, projector_type = resolve_projector(lanelet2_path, origin, projector_type)
    lanelet_map = lanelet2.io.load(str(lanelet2_path), projector)

    logger.info(
        "Loaded Lanelet2 map from %s with the %s projection (%d lanelets)",
        lanelet2_path,
        projector_type,
        len(list(lanelet_map.laneletLayer)),
    )
    return lanelet_map


def load_map(paths: "MapPaths") -> Any:
    """Load the Lanelet2 map a resolved ``map`` config names.

    The one way this framework turns a resolved map config into a parsed map.
    The sweeper, the sweep resolver and the editor's preview all load the same
    map for the same purpose -- deciding which lanelet a scenario means -- so
    they ask the same way, with the same three inputs.

    Raises:
        FileNotFoundError: If *paths* names no Lanelet2 map, or it is missing.
    """
    if paths.lanelet2_path is None:
        raise FileNotFoundError(
            "This scenario names no Lanelet2 map. Set map.source, or "
            "map.lanelet2_path, in the config."
        )
    return load_lanelet2_map(paths.lanelet2_path, paths.xodr_path, paths.projector_type)
