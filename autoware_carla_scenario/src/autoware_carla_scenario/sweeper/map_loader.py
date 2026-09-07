"""Lightweight Lanelet2 map loader for pre-simulation use.

The sweeper runs *before* CARLA is started, so we cannot rely on
:class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
(which needs a CARLA world for z-offset computation).  This module
provides a minimal loader that only needs the Lanelet2 ``.osm`` and
OpenDRIVE ``.xodr`` file paths.

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
from typing import Any

import lanelet2.core
import lanelet2.io

from ..coordinate.map_manager import _parse_geo_reference
from ..coordinate.projection import resolve_projector

logger = logging.getLogger(__name__)


def load_lanelet2_map(
    lanelet2_path: str | Path,
    xodr_path: str | Path,
    projector_type: str | None = None,
) -> Any:
    """Load a Lanelet2 map using the geoReference from an XODR file.

    This is a lightweight alternative to :meth:`MapManager.initialize` that
    does *not* require a CARLA connection or RoadNetwork loading.

    Args:
        lanelet2_path: Path to the Lanelet2 ``.osm`` file.
        xodr_path: Path to the OpenDRIVE ``.xodr`` file (used only for the
            ``geoReference`` PROJ string).
        projector_type: Which projection to read the map with.  Leave it unset
            to take it from the map's own ``map_projector_info.yaml``, exactly
            as ``MapManager`` does.

    Returns:
        A ``lanelet2.core.LaneletMap`` instance.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If the XODR file lacks a valid ``geoReference``.
    """
    lanelet2_path = Path(lanelet2_path)
    xodr_path = Path(xodr_path)

    if not lanelet2_path.exists():
        raise FileNotFoundError(f"Lanelet2 file not found: {lanelet2_path}")
    if not xodr_path.exists():
        raise FileNotFoundError(f"OpenDRIVE file not found: {xodr_path}")

    xodr_content = xodr_path.read_text(encoding="utf-8")
    lat, lon, _alt = _parse_geo_reference(xodr_content)

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
