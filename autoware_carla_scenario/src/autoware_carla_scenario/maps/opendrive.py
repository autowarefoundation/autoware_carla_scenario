"""Where the OpenDRIVE for a remote Lanelet2 map comes from: CARLA.

An HD map published for Autoware ships a Lanelet2 ``.osm`` and the projection
it was written in, and stops there -- the road network the simulator drives on
belongs to the CARLA asset, not to the map repository.  But this framework
needs OpenDRIVE for two things a scenario cannot do without: the
``geoReference`` that says where on Earth the Lanelet2 map is (see
:mod:`~autoware_carla_scenario.sweeper.map_loader`), and the road geometry
:class:`~autoware_carla_scenario.coordinate.map_manager.MapManager` converts
poses through.

So the OpenDRIVE is taken from CARLA itself, in the order that costs least:

1. one already in the map directory, or already derived into the cache;
2. the asset file inside a local CARLA installation, named by the same
   ``<MAP_NAME>_PATH`` environment variable
   :meth:`~autoware_carla_scenario.scenario_runner.ScenarioRunner.load_map_by_overwriting_xodr`
   writes to;
3. a running server, over the CARLA Python API --
   ``world.get_map().to_opendrive()``.

Whatever it comes from is written into the cache entry's ``derived`` directory,
so the answer is found at step 1 from then on and neither CARLA nor the network
is needed to open the scenario again.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from .resolver import ResolvedMap

logger = logging.getLogger(__name__)

__all__ = [
    "OpenDriveUnavailable",
    "capture_opendrive",
    "ensure_xodr",
    "installed_xodr",
    "map_asset_env_var",
    "opendrive_from_server",
]

#: Default CARLA RPC endpoint, matching
#: :class:`~autoware_carla_scenario.scenario_config.ServerConfig`.
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 2000

#: Seconds the client waits for the server.  Loading a town is slow, so this is
#: well above the CARLA client default of 5.
DEFAULT_TIMEOUT = 60.0


class OpenDriveUnavailable(RuntimeError):
    """No OpenDRIVE could be found or fetched for a map."""


def map_asset_env_var(map_name: str) -> str:
    """Convert a CamelCase map name to its ``UPPER_SNAKE_CASE_PATH`` variable.

    The variable names the ``.xodr`` inside the CARLA installation, which is
    both where this module reads one from and where
    ``load_map_by_overwriting_xodr`` writes one to.

    Examples::

        map_asset_env_var("NishishinjukuMap")  # -> "NISHISHINJUKU_MAP_PATH"
        map_asset_env_var("Town01")            # -> "TOWN01_PATH"
        map_asset_env_var("Town10HD_Opt")      # -> "TOWN10_HD_OPT_PATH"

    Args:
        map_name: CamelCase CARLA map name.

    Returns:
        The derived environment variable name.
    """
    # Insert underscore between a lowercase/digit and the following uppercase letter
    snake = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", map_name)
    return snake.upper() + "_PATH"


def installed_xodr(map_name: str) -> Optional[Path]:
    """Return the CARLA installation's ``.xodr`` for *map_name*, if reachable.

    ``None`` when the variable is unset or points at a file that is not there --
    a workstation with no CARLA installed is the normal case for the editor.
    """
    import os  # noqa: PLC0415 -- read at call time, not at import time

    configured = os.environ.get(map_asset_env_var(map_name), "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    return path if path.is_file() else None


def _short_map_name(name: str) -> str:
    """Return the bare town name from whatever CARLA calls a map."""
    return name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def capture_opendrive(world: Any, destination: Path) -> Path:
    """Write *world*'s OpenDRIVE to *destination*, and return it.

    The fourth way a map can get its OpenDRIVE, and the cheapest: a run already
    has the town loaded, so the road network is right there.  It lives here
    beside the other three so that "where a derived OpenDRIVE comes from and
    where it is written" stays one answer -- a run fills the same cache file
    the editor's Fetch button does, and the sweeper finds it either way.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(world.get_map().to_opendrive()), encoding="utf-8")
    return destination


def opendrive_from_server(
    map_name: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    load: bool = True,
) -> str:
    """Return *map_name*'s OpenDRIVE as read from a running CARLA server.

    Args:
        map_name: The CARLA map, e.g. ``Town10HD_Opt``.
        host: CARLA RPC host.
        port: CARLA RPC port.
        timeout: Seconds to wait for the server.
        load: Load *map_name* when the server currently has another world open.
            Left off, a mismatch is an error rather than a world reload, which
            is what a caller sharing the server with a run wants.

    Returns:
        The OpenDRIVE document as text.

    Raises:
        OpenDriveUnavailable: If the CARLA client is not installed, the server
            cannot be reached, or it does not have that map.
    """
    try:
        import carla  # noqa: PLC0415 -- optional, and heavy
    except ImportError as exc:  # pragma: no cover -- depends on the extra
        raise OpenDriveUnavailable(
            "The CARLA Python client is not installed, so OpenDRIVE cannot be "
            "fetched from a server. Install it with the `carla` extra, or set "
            f"{map_asset_env_var(map_name)} to the .xodr in a CARLA installation."
        ) from exc

    wanted = _short_map_name(map_name)
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        if _short_map_name(world.get_map().name) != wanted:
            if not load:
                raise OpenDriveUnavailable(
                    f"The CARLA server at {host}:{port} has "
                    f"{_short_map_name(world.get_map().name)} open, not {wanted}."
                )
            available = {_short_map_name(name) for name in client.get_available_maps()}
            if wanted not in available:
                raise OpenDriveUnavailable(
                    f"The CARLA server at {host}:{port} does not have a map "
                    f"named {wanted}. It offers: {', '.join(sorted(available))}."
                )
            logger.info("Loading %s on %s:%s to read its OpenDRIVE", wanted, host, port)
            world = client.load_world(wanted)
        return str(world.get_map().to_opendrive())
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
        load: Passed to :func:`opendrive_from_server`.
        refresh: Fetch again even when a cached ``.xodr`` is already there.

    Returns:
        Path to the OpenDRIVE file, inside the map cache.

    Raises:
        OpenDriveUnavailable: If no OpenDRIVE could be found or fetched.
    """
    name = map_name or resolved.name
    if not refresh and resolved.xodr_path is not None:
        return resolved.xodr_path

    destination = resolved.derived_xodr(name)
    destination.parent.mkdir(parents=True, exist_ok=True)

    installed = installed_xodr(name)
    if installed is not None:
        shutil.copy2(installed, destination)
        logger.info("Took OpenDRIVE for %s from %s", name, installed)
        return destination

    if not allow_server:
        raise OpenDriveUnavailable(
            f"No OpenDRIVE for {name}. The map repository ships none, and "
            f"{map_asset_env_var(name)} does not name one in a CARLA "
            "installation."
        )

    text = opendrive_from_server(name, host=host, port=port, timeout=timeout, load=load)
    destination.write_text(text, encoding="utf-8")
    logger.info("Fetched OpenDRIVE for %s from %s:%s", name, host, port)
    return destination
