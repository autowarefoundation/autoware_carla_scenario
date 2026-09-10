"""What maps a repository offers.

An HD map repository is a directory tree, not a list, and the person choosing a
map knows the repository rather than the path inside it.  So the editor asks
this module what is in there and shows the answer; a map is any directory
holding a Lanelet2 ``.osm``, which is the same rule
:mod:`~autoware_carla_scenario.maps.resolver` resolves one by.

Listing costs a clone of the repository's commit and tree objects and nothing
else -- no file content, no LFS -- and that clone is the same cache entry the
map itself is later resolved out of, so browsing a repository and then picking
a map from it downloads the map once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .cache import GitMapCache
from .resolver import PREFERRED_LANELET2_NAME
from .source import MapSource

logger = logging.getLogger(__name__)

__all__ = ["KNOWN_REPOSITORIES", "MapEntry", "MapRepository", "list_maps"]


@dataclass(frozen=True)
class MapRepository:
    """A map repository worth offering by name.

    Attributes:
        label: How it reads in a list.
        uri: The source URI for the repository, with no map path.
        description: One line on what is in it.
    """

    label: str
    uri: str
    description: str = ""


#: Repositories the editor offers before anyone has typed a URL.  A short list
#: on purpose: it is a starting point for browsing, not a registry, and any
#: repository can be pasted in beside it.
KNOWN_REPOSITORIES: tuple[MapRepository, ...] = (
    MapRepository(
        label="AutowareFoundation / carla-ue5-maps",
        uri=(
            "git+https://huggingface.co/datasets/AutowareFoundation/"
            "carla-ue5-maps@splatsim"
        ),
        description=(
            "Autoware maps for the CARLA 0.10 towns, one directory per world. "
            "The same dataset Autoware's own setup downloads."
        ),
    ),
)


@dataclass(frozen=True)
class MapEntry:
    """One map a repository offers.

    Attributes:
        source: A source naming this map, ready to store on a scenario.
        name: The directory's name, e.g. ``Town10HD_Opt``.
        path: Where it sits in the repository.
        lanelet2_name: The ``.osm`` found there.
        has_xodr: Whether the repository ships OpenDRIVE for it.  Most do not:
            see :mod:`~autoware_carla_scenario.maps.opendrive`.
        has_projector_info: Whether ``map_projector_info.yaml`` sits beside it.
    """

    source: MapSource
    name: str
    path: str
    lanelet2_name: str
    has_xodr: bool = False
    has_projector_info: bool = False

    @property
    def uri(self) -> str:
        """The source URI for this map, for a form field or an override."""
        return self.source.uri


def list_maps(
    source: "str | MapSource",
    *,
    cache: Optional[GitMapCache] = None,
    refresh: bool = False,
) -> list[MapEntry]:
    """Return every map in *source*'s repository, in path order.

    The source's own ``path`` narrows the search rather than selecting a single
    map, so browsing ``...#autoware_maps`` lists the maps under it.

    Args:
        source: A map source URI, or an already-parsed :class:`MapSource`.
        cache: The cache to read and fill.  Defaults to the shared one.
        refresh: Fetch the repository again even when it is already cached.

    Returns:
        One :class:`MapEntry` per directory holding a Lanelet2 map.

    Raises:
        MapSourceError: If *source* is not a usable URI.
        MapCacheError: If the repository could not be cloned or updated.
    """
    parsed = source if isinstance(source, MapSource) else MapSource.parse(source)
    files = (cache or GitMapCache()).list_files(parsed.with_path(""), refresh=refresh)

    prefix = f"{parsed.path}/" if parsed.path else ""
    by_directory: dict[str, list[str]] = {}
    for path in files:
        if not path.startswith(prefix):
            continue
        # The plain dirname of every file, which is *not*
        # `MapSource.directory` -- that one answers "which directory does this
        # source name", and for `.../TownB.xodr` the answer is TownB, not the
        # file's parent. Here every file is being bucketed by where it sits.
        head, separator, name = path.rpartition("/")
        by_directory.setdefault(head if separator else "", []).append(name)

    entries: list[MapEntry] = []
    for directory, names in sorted(by_directory.items()):
        osm = sorted(name for name in names if name.endswith(".osm"))
        if not osm:
            continue
        chosen = PREFERRED_LANELET2_NAME if PREFERRED_LANELET2_NAME in osm else osm[0]
        at_directory = parsed.with_path(directory)
        entries.append(
            MapEntry(
                source=at_directory,
                name=at_directory.name,
                path=directory,
                lanelet2_name=chosen,
                has_xodr=any(name.endswith(".xodr") for name in names),
                has_projector_info="map_projector_info.yaml" in names,
            )
        )
    return entries
