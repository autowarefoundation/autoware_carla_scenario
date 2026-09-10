"""Turn a ``map`` config into the paths the rest of the framework reads.

The sweeper before CARLA is started, the runner as it builds a queue, and the
editor's preview all ask the same question of the same ``map`` group:
``map.source`` names a git repository, ``map.lanelet2_path`` and
``map.xodr_path`` name files directly.

An explicitly configured path always wins over the source, which is what keeps
``map.xodr_path=/tmp/patched.xodr`` working as an override: the source says
where the map *lives*, not what a run is forbidden to replace.

Nothing here imports CARLA, Lanelet2 or Hydra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .resolver import ResolvedMap, cached_map, resolve_map

logger = logging.getLogger(__name__)

__all__ = ["MapPaths", "resolve_map_paths"]


def _setting(map_config: Any, key: str) -> Optional[str]:
    """Read one ``map`` setting off a config, whatever shape it has.

    The same map is described by a :class:`~...scenario_config.MapConfig`, by an
    OmegaConf node, and by the editor's :class:`~...authoring.models.MapRef`;
    reading a key off all three is the only thing this module needs from them.
    """
    if map_config is None:
        return None
    if hasattr(map_config, key):
        value = getattr(map_config, key)
    else:
        try:
            value = map_config[key]  # OmegaConf nodes and plain dicts
        except (KeyError, TypeError):
            return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class MapPaths:
    """Where a scenario's map files are, once the config has been resolved.

    Attributes:
        name: The CARLA map name the scenario runs on.
        lanelet2_path: The Lanelet2 map, or ``None`` when the config names none.
        xodr_path: The OpenDRIVE file, or ``None`` when it has yet to be taken
            from CARLA -- see :func:`~...maps.opendrive.ensure_xodr`.
        resolved: The remote map this came from, when ``map.source`` was set.
        projector_type: The projection override the config carries, if any.
        xodr_is_derived: Whether :attr:`xodr_path` was read back out of CARLA
            rather than named by the config or shipped by the repository.
    """

    name: str = ""
    lanelet2_path: Optional[Path] = None
    xodr_path: Optional[Path] = None
    resolved: Optional[ResolvedMap] = None
    projector_type: Optional[str] = None
    xodr_is_derived: bool = False

    @property
    def from_source(self) -> bool:
        """Whether these paths came out of a map repository."""
        return self.resolved is not None

    @property
    def derived_xodr(self) -> Optional[Path]:
        """Where an OpenDRIVE fetched from CARLA should be written.

        ``None`` for a map that is not backed by a repository: there is no
        cache entry to put one in, and such a scenario names its own file.
        """
        if self.resolved is None:
            return None
        return self.resolved.derived_xodr(self.name)

    @property
    def install_xodr(self) -> Optional[Path]:
        """The OpenDRIVE that has to be written into the CARLA installation.

        A map describing roads CARLA does not ship has to replace the
        simulator's own road network before the world is loaded.  One read back
        *from* CARLA never does.
        """
        return None if self.xodr_is_derived else self.xodr_path

    @property
    def opendrive_path(self) -> Optional[Path]:
        """The OpenDRIVE to read geometry from, without installing it.

        Either one already derived from CARLA, or where to put one when the run
        reaches a loaded world.
        """
        if self.xodr_is_derived:
            return self.xodr_path
        return self.derived_xodr if self.xodr_path is None else None


def resolve_map_paths(
    map_config: Any,
    *,
    allow_fetch: bool = True,
    refresh: bool = False,
) -> MapPaths:
    """Return the local map files *map_config* names.

    A config naming neither a source nor any file comes back empty rather than
    raising: a scenario is edited into shape long before it has a map.

    Args:
        map_config: The ``map`` config group, in any of the shapes
            :func:`_setting` reads.
        allow_fetch: Whether a source that is not cached yet may be cloned.
            Left off, an uncached source contributes no paths -- which is what
            the editor wants on a render, where fetching is a button rather
            than a side effect of drawing the page.
        refresh: Fetch the map repository again even when it is already cached.

    Raises:
        MapSourceError, MapCacheError, MapResolutionError: If ``map.source`` is
            set and cannot be resolved.
    """
    source = _setting(map_config, "source")
    lanelet2 = _setting(map_config, "lanelet2_path")
    xodr = _setting(map_config, "xodr_path")
    name = _setting(map_config, "name") or ""
    projector_type = _setting(map_config, "projector_type")

    resolved: Optional[ResolvedMap] = None
    if source is not None:
        resolved = (
            resolve_map(source, refresh=refresh)
            if allow_fetch or refresh
            else cached_map(source)
        )
        if resolved is not None:
            name = name or resolved.name

    return MapPaths(
        name=name,
        lanelet2_path=(
            Path(lanelet2).expanduser()
            if lanelet2
            else (resolved.lanelet2_path if resolved else None)
        ),
        xodr_path=(
            Path(xodr).expanduser()
            if xodr
            else (resolved.xodr_path if resolved else None)
        ),
        resolved=resolved,
        projector_type=projector_type,
        xodr_is_derived=(
            not xodr and resolved is not None and resolved.xodr_is_derived
        ),
    )
