"""HD maps that live in a git repository rather than beside the scenario.

A scenario names its map with one URI -- see
:class:`~autoware_carla_scenario.maps.source.MapSource` -- and everything else
follows: the repository is cloned into a local map root, the map's files are
found in it, and the OpenDRIVE that CARLA needs but the repository does not
carry is fetched from CARLA.  All of it is cached.

The map root defaults to ``~/autoware_data/maps``, where Autoware's own setup
downloads map datasets, so a map its ``demo_artifacts`` role already fetched is
used as it sits::

    from autoware_carla_scenario.maps import resolve_map

    hd_map = resolve_map(
        "git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
        "@splatsim#autoware_maps/Town10HD_Opt"
    )
    hd_map.lanelet2_path  # -> .../autoware_maps/Town10HD_Opt/lanelet2_map.osm

Nothing here imports CARLA or Lanelet2, so the editor can use it.
"""

from __future__ import annotations

from .cache import GitMapCache, MapCacheError, map_root
from .catalogue import KNOWN_REPOSITORIES, MapEntry, MapRepository, list_maps
from .config import MapPaths, resolve_map_paths
from .opendrive import OpenDriveUnavailable, capture_opendrive, ensure_xodr
from .resolver import (
    MapResolutionError,
    ResolvedMap,
    cached_map,
    pin_source,
    resolve_map,
)
from .source import MapSource, MapSourceError

__all__ = [
    "KNOWN_REPOSITORIES",
    "GitMapCache",
    "MapCacheError",
    "MapEntry",
    "MapPaths",
    "MapRepository",
    "MapResolutionError",
    "MapSource",
    "MapSourceError",
    "OpenDriveUnavailable",
    "ResolvedMap",
    "cached_map",
    "capture_opendrive",
    "ensure_xodr",
    "list_maps",
    "map_root",
    "pin_source",
    "resolve_map",
    "resolve_map_paths",
]
