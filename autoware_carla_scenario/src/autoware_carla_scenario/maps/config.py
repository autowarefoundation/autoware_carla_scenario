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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from ..config_flags import check_flag
from .resolver import (
    ResolvedMap,
    _short_map_name,
    cached_map,
    resolve_directory,
    resolve_map,
)
from .source import redacted_url

logger = logging.getLogger(__name__)

__all__ = ["MapPaths", "map_setting", "resolve_map_paths"]


def map_setting(map_config: Any, key: str) -> Optional[str]:
    """Read one ``map`` setting off a config, whatever shape it has.

    The same map is described by a :class:`~...scenario_config.MapConfig`, by an
    OmegaConf node, and by the editor's :class:`~...authoring.models.MapRef`;
    reading a key off all three is the only thing this module needs from them.
    With *key* absent or null the value is ``None``; text is returned
    stripped, and ``None`` for a blank one -- the editor's ``MapRef`` holds
    these settings as text fields, where clearing one leaves ``""``.  A number
    is read as its text (``map.name=2024`` names the map ``"2024"``).

    Raises:
        ValueError: naming ``map.<key>`` and the value, when it is a boolean
            (``map.xodr_path=false`` is not a file named ``False``) or neither
            text nor a number (a list, a mapping) -- refused here rather than
            failing later as a file or repository that does not exist (#2165).
    """
    value = _value(map_config, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"map.{key} must be text, got {value!r}")
    text = str(value).strip()
    return text or None


def _value(map_config: Any, key: str) -> Any:
    """One ``map`` setting off a config of any shape (:func:`map_setting`), as
    it holds it; ``None`` when it holds none."""
    if map_config is None:
        return None
    if hasattr(map_config, key):
        return getattr(map_config, key)
    try:
        return map_config[key]  # OmegaConf nodes and plain dicts
    except (KeyError, TypeError):
        return None


def _flag(map_config: Any, key: str) -> bool:
    """Read one boolean ``map`` setting, off when it is absent or null, and
    otherwise as every config switch is read
    (:func:`~...config_flags.check_flag`): a boolean, ``"true"``/``"false"``
    (``yes``/``no``, ``on``/``off``, ``1``/``0``) or the integer 1 or 0 --
    a typo (``"ture"``), ``2`` or ``"enabled"`` is refused, naming
    ``map.<key>``, rather than read as off (#2143).
    """
    value = _value(map_config, key)
    if value is None:
        return False
    return check_flag(f"map.{key}", value)


@dataclass(frozen=True)
class MapPaths:
    """Where a scenario's map files are, once the config has been resolved.

    Attributes:
        name: The CARLA map name the scenario runs on.
        lanelet2_path: The Lanelet2 map, or ``None`` when the config names none.
        xodr_path: The OpenDRIVE file, or ``None`` when it has yet to be taken
            from CARLA -- see :func:`~...maps.opendrive.ensure_xodr`.
        resolved: The map this came from, when ``map.directory`` or
            ``map.source`` was set.
        projector_type: The projection override the config carries, if any.
        xodr_is_derived: Whether :attr:`xodr_path` was read back out of CARLA
            rather than named by the config or shipped by the repository.
        overwrite_xodr: Whether :attr:`install_xodr` may replace the road
            network CARLA ships.  Off unless ``map.overwrite_xodr`` asks for it.
        pack: The content pack the CARLA map is mounted from (``map.pack``),
            ``OWNER/REPO/PACK@VERSION`` once resolved, or ``None``.
        pack_sha256: That pack's artifact, so the pack the run mounts is the
            one its map files were read from.
    """

    name: str = ""
    lanelet2_path: Optional[Path] = None
    xodr_path: Optional[Path] = None
    resolved: Optional[ResolvedMap] = None
    projector_type: Optional[str] = None
    xodr_is_derived: bool = False
    overwrite_xodr: bool = False
    pack: Optional[str] = None
    pack_sha256: Optional[str] = None

    @property
    def derived_xodr(self) -> Optional[Path]:
        """Where an OpenDRIVE fetched from CARLA should be written.

        ``None`` for a map named only by file paths: there is no cache entry
        to put one in, and such a scenario names its own file.
        """
        if self.resolved is None:
            return None
        return self.resolved.derived_xodr(self.name)

    @property
    def install_xodr(self) -> Optional[Path]:
        """The OpenDRIVE the run reads, and installs when asked to.

        A map describing roads CARLA does not ship can replace the simulator's
        own road network before the world is loaded, but only when
        ``map.overwrite_xodr`` says so -- see :attr:`overwrite_xodr`.  One read
        back *from* CARLA is never a candidate.
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

    ``map.directory`` names a map on this machine and is read in place -- see
    :func:`~...maps.resolver.resolve_directory`.  It wins over ``map.source``,
    so a run can be pointed at a local copy of a map without first clearing
    the source its map group selects.  ``lanelet2_path`` and ``xodr_path``
    still win over both.

    ``map.pack`` names the CARLA content pack the map is mounted from
    (:class:`~...maps.pack.MapPack`).  The pack is fetched, and its Lanelet2
    map read as ``map.directory`` would read it -- winning over ``map.source``,
    but not over ``map.directory`` -- and its OpenDRIVE is the road network
    unless ``xodr_path`` or the map directory names one.  Without
    *allow_fetch* the pack is not fetched, and only the reference is kept.

    Args:
        map_config: The ``map`` config group, in any of the shapes
            :func:`map_setting` reads.
        allow_fetch: Whether a source that is not cached yet may be cloned.
            Left off, an uncached source contributes no paths -- which is what
            the editor wants on a render, where fetching is a button rather
            than a side effect of drawing the page.
        refresh: Fetch the map repository again even when it is already cached.

    Raises:
        ValueError: If a setting is a boolean or neither text nor a number
            (:func:`map_setting`), or ``map.overwrite_xodr`` is not a switch
            (:func:`_flag`), naming the key and the value; or
            ``map.pack_sha256`` is set without ``map.pack``.
        MapSourceError, MapCacheError, MapResolutionError: If ``map.source``
            or ``map.directory`` is set and cannot be resolved.
        MapPackError: If ``map.pack`` is set and the pack cannot be fetched
            or does not carry the map (:mod:`~...maps.pack`).
    """
    directory = map_setting(map_config, "directory")
    source = map_setting(map_config, "source")
    name = map_setting(map_config, "name") or ""
    pack = map_setting(map_config, "pack")
    pack_sha256 = map_setting(map_config, "pack_sha256")
    pack_xodr: Optional[Path] = None
    if pack is None and pack_sha256 is not None:
        # Nothing to pin: the run would go ahead unpinned (#2198).
        raise ValueError(
            "map.pack_sha256 pins the pack map.pack names, but map.pack is "
            f"not set: got map.pack_sha256={pack_sha256!r}"
        )
    if pack is not None:
        # Imported here: the pack module and carla-pack are needed only by a
        # config naming a pack.
        from .pack import MapPack  # noqa: PLC0415

        map_pack = MapPack(pack, sha256=pack_sha256)
        if allow_fetch or refresh:
            files = map_pack.files(name)
            pack, pack_sha256 = files.reference, files.sha256 or pack_sha256
            name = name or files.map_name
            pack_xodr = files.xodr_path
            if directory is None and files.directory is not None:
                if source is not None:
                    logger.info(
                        "map.pack %s takes precedence over map.source %s",
                        pack,
                        redacted_url(source),
                    )
                directory, source = str(files.directory), None
    paths = _resolve(
        directory=directory,
        source=source,
        lanelet2=map_setting(map_config, "lanelet2_path"),
        xodr=map_setting(map_config, "xodr_path"),
        name=name,
        projector_type=map_setting(map_config, "projector_type"),
        overwrite_xodr=_flag(map_config, "overwrite_xodr"),
        allow_fetch=allow_fetch,
        refresh=refresh,
    )
    if pack is None:
        return paths
    if pack_xodr is not None and (paths.xodr_path is None or paths.xodr_is_derived):
        # The OpenDRIVE the pack's map was cooked from: what CARLA runs, so
        # it is read rather than one read back out of the loaded world.
        paths = replace(paths, xodr_path=pack_xodr, xodr_is_derived=False)
    return replace(paths, pack=pack, pack_sha256=pack_sha256)


def _resolve(
    directory: Optional[str],
    source: Optional[str],
    lanelet2: Optional[str],
    xodr: Optional[str],
    name: str,
    projector_type: Optional[str],
    overwrite_xodr: bool,
    allow_fetch: bool,
    refresh: bool,
) -> MapPaths:
    """Resolve the map settings :func:`resolve_map_paths` read off a config.

    Each string setting is ``None`` when the config leaves it unset (a blank
    one reads as unset too); *name* is ``""`` then.  The other arguments are
    :func:`resolve_map_paths`'s.
    """
    resolved: Optional[ResolvedMap] = None
    if directory is not None:
        if source is not None:
            logger.info(
                "map.directory %s takes precedence over map.source %s",
                directory,
                redacted_url(source),
            )
        local = resolve_directory(directory)
        resolved = local
        if not name:
            name = local.name
    elif source is not None:
        if allow_fetch or refresh:
            resolved = resolve_map(source, refresh=refresh)
        else:
            resolved = cached_map(source)
        if resolved is not None and not name:
            name = resolved.name

    if (
        resolved is not None
        and name
        and _short_map_name(name) != _short_map_name(resolved.name)
        and (resolved.xodr_path is None or resolved.xodr_is_derived)
    ):
        # The resolver looks for an OpenDRIVE taken from CARLA under the map
        # directory's name; a run on a differently named CARLA map wrote it
        # under that name instead (MapPaths.derived_xodr).  One captured under
        # the directory's name is another town's, so it is never used here.
        captured = resolved.cached_derived_xodr(name)
        resolved = replace(
            resolved, xodr_path=captured, xodr_is_derived=captured is not None
        )

    lanelet2_path: Optional[Path] = None
    if lanelet2:
        lanelet2_path = Path(lanelet2).expanduser()
    elif resolved is not None:
        lanelet2_path = resolved.lanelet2_path
    xodr_path: Optional[Path] = None
    xodr_is_derived = False
    if xodr:
        xodr_path = Path(xodr).expanduser()
    elif resolved is not None:
        xodr_path = resolved.xodr_path
        xodr_is_derived = resolved.xodr_is_derived

    return MapPaths(
        name=name,
        lanelet2_path=lanelet2_path,
        xodr_path=xodr_path,
        resolved=resolved,
        projector_type=projector_type,
        xodr_is_derived=xodr_is_derived,
        overwrite_xodr=overwrite_xodr,
    )
