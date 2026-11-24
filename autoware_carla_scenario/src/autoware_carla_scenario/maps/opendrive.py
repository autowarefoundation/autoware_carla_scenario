"""Where the OpenDRIVE for a remote Lanelet2 map comes from: CARLA.

An HD map published for Autoware ships a Lanelet2 map and stops there -- the
road network belongs to the CARLA asset, not to the map repository.  But a run
needs OpenDRIVE for the road geometry
:class:`~autoware_carla_scenario.coordinate.map_manager.MapManager` converts
poses through, so it is taken from CARLA itself, in the order that costs least:
one already cached or shipped; the asset inside a local CARLA installation,
named by the ``<MAP_NAME>_PATH`` variable ``load_map_by_overwriting_xodr``
writes to; a running server; or the world a run has already loaded.

Whatever it comes from is written into the map root's ``derived`` directory, so
the answer is found without CARLA from then on.
"""

from __future__ import annotations

import logging
import re
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .cache import _copy_atomically, _write_atomically
from .resolver import ResolvedMap, _short_map_name

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "OpenDriveUnavailable",
    "capture_opendrive",
    "ensure_xodr",
    "install_xodr",
    "installed_xodr",
    "map_asset_env_var",
    "map_asset_path",
    "original_xodr_path",
]

#: Default CARLA RPC endpoint, matching
#: :class:`~autoware_carla_scenario.scenario_config.ServerConfig`.
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2000

#: Seconds the client waits for the server.  Loading a town is slow, so this is
#: well above the CARLA client default of 5.
DEFAULT_TIMEOUT = 60.0

#: Suffix of the copy of a CARLA installation's own ``.xodr`` kept beside it
#: before :func:`install_xodr` first overwrites it (#2446).
ORIGINAL_SUFFIX = ".orig"


class OpenDriveUnavailable(RuntimeError):
    """No OpenDRIVE could be found or fetched for a map."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def map_asset_env_var(map_name: str) -> str:
    """Convert a CamelCase map name to its ``UPPER_SNAKE_CASE_PATH`` variable.

    The variable names the ``.xodr`` inside the CARLA installation -- both where
    this module reads one from and where ``load_map_by_overwriting_xodr`` writes
    one to.  ``NishishinjukuMap`` -> ``NISHISHINJUKU_MAP_PATH``, ``Town10HD_Opt``
    -> ``TOWN10_HD_OPT_PATH``.  A full asset path is named by its town:
    ``/Game/Carla/Maps/Town10HD_Opt`` -> ``TOWN10_HD_OPT_PATH`` too (#1559).
    """
    # Insert underscore between a lowercase/digit and the following uppercase letter
    snake = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", _short_map_name(map_name))
    return snake.upper() + "_PATH"


def map_asset_path(map_name: str) -> Optional[Path]:
    """Return the path the ``<MAP_NAME>_PATH`` variable names for *map_name*.

    Surrounding whitespace is stripped and a leading ``~`` expanded, so every
    reader of the variable agrees on the file.  ``None`` when the variable is
    unset or blank; whether the file exists is left to the caller.
    """
    import os  # noqa: PLC0415 -- read at call time, not at import time

    configured = os.environ.get(map_asset_env_var(map_name), "").strip()
    if not configured:
        return None
    return Path(configured).expanduser()


def original_xodr_path(installed: Path) -> Path:
    """Return where the original of the installation's ``.xodr`` *installed*
    is kept once :func:`install_xodr` has overwritten it: beside it, with
    :data:`ORIGINAL_SUFFIX` appended (``Town10HD_Opt.xodr.orig``)."""
    return installed.with_name(installed.name + ORIGINAL_SUFFIX)


def installed_xodr(map_name: str) -> Optional[Path]:
    """Return the CARLA installation's ``.xodr`` for *map_name*, if reachable.

    The installation's *original* road network: once a run with
    ``map.overwrite_xodr`` has installed another map's OpenDRIVE over the file
    (:func:`install_xodr`), the copy of what was there first
    (:func:`original_xodr_path`), so that a copy derived from the
    installation is CARLA's map, not the last run's (#2446).

    ``None`` when the variable is unset or points at a file that is not there --
    a workstation with no CARLA installed is the normal case for the editor.
    """
    path = map_asset_path(map_name)
    if path is None:
        return None
    original = original_xodr_path(path)
    if original.is_file():
        return original
    return path if path.is_file() else None


def install_xodr(xodr_path: Path, destination: Path) -> bool:
    """Install *xodr_path* as the CARLA installation's ``.xodr`` *destination*;
    whether it was written.

    What ``load_map_by_overwriting_xodr`` (and the lanelet check of
    ``tools.detect_no_3d_model_lanelets``) does to the installation (#2446):

    * Before the file is first overwritten, its original is kept beside it
      (:func:`original_xodr_path`), once: an existing copy is never replaced,
      so it stays CARLA's own road network whatever runs install later, and
      :func:`installed_xodr` derives copies from it.  The installed file
      itself stays overwritten after the run.
    * The file is replaced in one rename (``maps.cache._copy_atomically``):
      another process on the same installation -- a task of the same SLURM
      batch -- loading the map meanwhile reads the old file or the new one,
      never a part.
    * Nothing is written when the installed file already holds these bytes,
      as it does on every reload of a batch's map.

    Raises:
        FileNotFoundError: If *xodr_path* does not exist.
    """
    if destination.is_file():
        if _same_bytes(xodr_path, destination):
            return False
        original = original_xodr_path(destination)
        if not original.exists() and _copy_atomically(destination, original, True):
            logger.info(
                "Kept the CARLA installation's original OpenDRIVE as %s", original
            )
    _copy_atomically(xodr_path, destination)
    return True


def _same_bytes(first: Path, second: Path) -> bool:
    """Whether the files *first* and *second* hold the same bytes."""
    if first.stat().st_size != second.stat().st_size:
        return False
    return first.read_bytes() == second.read_bytes()


def capture_opendrive(world: carla.World, destination: Path) -> Path:
    """Write *world*'s OpenDRIVE to *destination*, and return it.

    The cheapest of the four sources: a run already has the town loaded.  It
    lives beside the other three so a run fills the same cache file the editor's
    Fetch button does.  Written atomically: runs in other processes (the tasks
    of a SLURM batch, docs/slurm.md) may capture the same map meanwhile, and
    none of them may read half a file (``maps.cache._write_atomically``).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(destination, str(world.get_map().to_opendrive()))
    return destination


def _from_server(
    map_name: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    load: bool = True,
) -> str:
    """Return *map_name*'s OpenDRIVE as read from a running CARLA server.

    Args:
        load: Load *map_name* when the server currently has another world open.
            Left off, a mismatch is an error rather than a world reload, which
            is what a caller sharing the server with a run wants.

    Raises:
        OpenDriveUnavailable: If the CARLA client is not installed, the server
            cannot be reached, or it does not have that map.
    """
    try:
        import typesafe_carla.carla as carla  # noqa: PLC0415 -- optional, and heavy
    except ImportError as exc:  # pragma: no cover -- depends on the extra
        raise OpenDriveUnavailable(
            "The CARLA Python client is not installed, so OpenDRIVE cannot be "
            "fetched from a server. Install it with the `carla` extra, or set "
            f"{map_asset_env_var(map_name)} to the .xodr in a CARLA installation."
        ) from exc

    wanted = _short_map_name(map_name)
    # Nothing listening is said here, before the CARLA client is made: on
    # arm64 its get_world() on a refused connection can abort the process
    # (std::terminate, not a RuntimeError), and with it the editor serving
    # the request.
    try:
        socket.create_connection((host, port), timeout=min(timeout, 2.0)).close()
    except OSError as exc:
        raise OpenDriveUnavailable(
            f"Could not read OpenDRIVE from the CARLA server at {host}:{port}: "
            f"nothing accepts a connection there ({exc})."
        ) from exc
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        # Once: CARLA builds the whole map, road network and all, on every
        # get_map() call (hundreds of MB on a large map).
        current = world.get_map()
        if _short_map_name(current.name) != wanted:
            if not load:
                raise OpenDriveUnavailable(
                    f"The CARLA server at {host}:{port} has "
                    f"{_short_map_name(current.name)} open, not {wanted}."
                )
            available = {_short_map_name(name) for name in client.get_available_maps()}
            if wanted not in available:
                raise OpenDriveUnavailable(
                    f"The CARLA server at {host}:{port} does not have a map "
                    f"named {wanted}. It offers: {', '.join(sorted(available))}."
                )
            logger.info("Loading %s on %s:%s to read its OpenDRIVE", wanted, host, port)
            world = client.load_world(wanted)
            current = world.get_map()
        return str(current.to_opendrive())
    except OpenDriveUnavailable:
        raise
    except RuntimeError as exc:  # CARLA raises bare RuntimeErrors for transport
        raise OpenDriveUnavailable(
            f"Could not read OpenDRIVE from the CARLA server at {host}:{port}: {exc}"
        ) from exc


def ensure_xodr(
    resolved: ResolvedMap,
    *,
    map_name: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    allow_server: bool = True,
    load: bool = True,
    refresh: bool = False,
) -> Path:
    """Return the OpenDRIVE for *resolved*, fetching and caching it if needed.

    Args:
        resolved: The map whose OpenDRIVE is wanted.
        map_name: The CARLA map to read it from.  Defaults to the map
            directory's own name, which is how a published map is named.
        host: CARLA RPC host.
        port: CARLA RPC port.
        timeout: Seconds to wait for the server.
        allow_server: Whether a running CARLA server may be contacted.  Left
            off, only a file already on disk is accepted.
        load: Passed to :func:`_from_server`.
        refresh: Fetch again even when a cached ``.xodr`` is already there.

    Returns:
        Path to the OpenDRIVE file, inside the map cache.

    Raises:
        OpenDriveUnavailable: If no OpenDRIVE could be found or fetched.
    """
    name = map_name or resolved.name
    if not refresh:
        # A derived file is only this CARLA map's when it was captured under
        # its name; the one resolve_map found is named after the directory.
        if resolved.xodr_path is not None and (
            not resolved.xodr_is_derived
            or _short_map_name(name) == _short_map_name(resolved.name)
        ):
            return resolved.xodr_path
        cached = resolved.cached_derived_xodr(name)
        if cached is not None:
            return cached

    destination = resolved.derived_xodr(name)
    destination.parent.mkdir(parents=True, exist_ok=True)

    installed = installed_xodr(name)
    if installed is not None:
        # Copied whole, as the fetch below writes: another process resolving
        # the same map reads the old file or the new one, never a part.
        # Byte for byte, mode and mtime as well: load_map_by_overwriting_xodr
        # may copy this file back over the installation's.  The installation's
        # original, once a run has overwritten it (installed_xodr, #2446).
        _copy_atomically(installed, destination)
        logger.info("Took OpenDRIVE for %s from %s", name, installed)
        return destination

    if not allow_server:
        raise OpenDriveUnavailable(
            f"No OpenDRIVE for {name}. The map repository ships none, and "
            f"{map_asset_env_var(name)} does not name one in a CARLA "
            "installation."
        )

    text = _from_server(name, host=host, port=port, timeout=timeout, load=load)
    # Atomically (#1264): another process resolving the same map -- a task of
    # the same SLURM batch, the editor -- may read the file meanwhile.
    _write_atomically(destination, text)
    logger.info("Fetched OpenDRIVE for %s from %s:%s", name, host, port)
    return destination
