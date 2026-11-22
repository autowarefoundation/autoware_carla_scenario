"""The world's map as a CommonRoad scenario, converted by roadgen and cached.

The CommonRoad metrics (:mod:`autoware_carla_scenario.commonroad`) measure the
road users on the map's lanelets, which the worker reads from a CommonRoad
2020a scenario.  roadgen (``Map.export_commonroad``, roadgen 0.42.0 or later)
writes one from the world's OpenDRIVE, in the OpenDRIVE's inertial frame:
CARLA's ``(x, y)`` is that frame's ``(x, -y)``
(:func:`~autoware_carla_scenario.commonroad.frame.to_commonroad`).

A map is converted once per machine: the file is cached under a directory
named after a hash of the OpenDRIVE and roadgen's version.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from ..utils.cache_dir import user_cache_dir
from ..utils.opendrive import sanitize_opendrive

__all__ = ["MAP_FILE", "commonroad_map", "default_cache_dir"]

logger = logging.getLogger(__name__)

#: The scenario's file in its cache directory.
MAP_FILE = "map.commonroad.xml"
#: The least roadgen that writes CommonRoad.
_ROADGEN_MIN = (0, 42, 0)
_INSTALL = (
    "the CommonRoad metrics convert the map with roadgen>=0.42.0: install the "
    "`map` extra (autoware-carla-scenario[map])"
)


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/autoware_carla_scenario/commonroad``, or under
    ``~/.cache`` without an absolute one (:func:`.user_cache_dir`)."""
    return user_cache_dir() / "autoware_carla_scenario" / "commonroad"


def _version(text: str) -> tuple[int, ...]:
    parts = []
    for part in text.split(".")[:3]:
        digits = "".join(c for c in part if c.isascii() and c.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def _cache_place(
    opendrive: str, version: str, cache_dir: Optional[Path]
) -> tuple[Path, str]:
    """Where *opendrive* (OpenDRIVE text) converted by roadgen *version* is
    cached: the cache's root (*cache_dir*, or :func:`default_cache_dir`) and
    the name of the map's directory under it, a hash of the two."""
    root = default_cache_dir()
    if cache_dir is not None:
        root = cache_dir
    key = hashlib.sha256(f"roadgen={version}\0{opendrive}".encode()).hexdigest()[:16]
    return root, key


def commonroad_map(opendrive: str, cache_dir: Optional[Path] = None) -> Path:
    """The CommonRoad scenario of *opendrive* (OpenDRIVE text), converted or
    cached; its path.

    Raises:
        ImportError: roadgen is not installed, or too old to write CommonRoad.
    """
    try:
        import roadgen  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    version = str(getattr(roadgen, "__version__", "0"))
    # Compared as lists, as tuples compare (Codon's _version() is a list).
    if list(_version(version)) < list(_ROADGEN_MIN) or not hasattr(
        roadgen.Map, "export_commonroad"
    ):
        raise ImportError(f"roadgen {version} does not write CommonRoad: {_INSTALL}")
    root, key = _cache_place(opendrive, version, cache_dir)
    target = root / key
    path = target / MAP_FILE
    if path.is_file():
        return path

    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
    try:
        xodr = tmp / "map.xodr"
        xodr.write_text(sanitize_opendrive(opendrive), encoding="utf-8")
        road_map = roadgen.read_opendrive(str(xodr))
        for warning in road_map.read_warnings():
            logger.debug("roadgen read: %s", warning)
        # Fixed id and date, so the file is the map's alone.
        road_map.export_commonroad(
            str(tmp / MAP_FILE), benchmark_id="ZAM_Carla-1_1_T-1", date="2026-01-01"
        )
        for warning in road_map.commonroad_warnings():
            logger.debug("roadgen CommonRoad: %s", warning)
        try:
            os.rename(tmp, target)  # atomic; another process may have won
        except OSError:
            if not path.is_file():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    logger.info("converted the map to CommonRoad: %s", path)
    return path
