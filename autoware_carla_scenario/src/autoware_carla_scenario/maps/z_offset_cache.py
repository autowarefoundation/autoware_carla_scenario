"""Remember a map's Lanelet2-to-CARLA vertical offset across runs.

:meth:`~..coordinate.map_manager.MapManager.initialize` reads the offset off
every CARLA spawn point, each projected onto its nearest lanelet: on a large map
(Odaiba, 2,717 spawn points, 6,460 lanelets) that is some 40 seconds of every
run, for a number that only changes with the map. It is kept under the map
cache's ``.derived`` directory, keyed by the content of the two map files and by
the CARLA map whose spawn points it was read from, so an edited map file is a
new key rather than a stale value.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

from .cache import DERIVED_SUBDIR, map_root

logger = logging.getLogger(__name__)

__all__ = ["cached_z_offset", "z_offset_key"]

#: Bumped whenever the computation changes, so values from the old one are not reused.
_METHOD_VERSION = 1


#: Each file's digest by what it is now (``(path, device, inode, size,
#: modification time)``): a queue re-initialises the map manager for every
#: run, and hashing a city-sized map twice each time is most of a second.
_DIGESTS: dict[tuple[str, int, int, int, int], str] = {}


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (
        str(path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if key in _DIGESTS:
        return _DIGESTS[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(1 << 20)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(1 << 20)
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def z_offset_key(
    lanelet2_path: Path, xodr_path: Path, carla_map_name: str, projection: str = ""
) -> str:
    """The cache key: both files' content, the CARLA map, the projection, the method.

    *projection* names how the Lanelet2 map was projected (projector, origin):
    the spawn points are matched to lanelets in that frame, so another
    projection of the same file may well give another offset.
    """
    digest = hashlib.sha256()
    for part in (
        f"v{_METHOD_VERSION}",
        _file_sha256(lanelet2_path),
        _file_sha256(xodr_path),
        carla_map_name,
        projection,
    ):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _entry_json(
    value: float,
    lanelet2_path: Path,
    xodr_path: Path,
    carla_map_name: str,
    projection: str,
) -> str:
    """The entry's JSON: ``z_offset`` a number, the rest strings.

    A mapping of a float and strings has no type for the check, so the
    strings are dumped as one object and ``z_offset`` is put in front of them,
    which is what ``json.dumps`` of the whole mapping writes, byte for byte.
    """
    described = json.dumps(
        {
            "lanelet2": str(lanelet2_path),
            "xodr": str(xodr_path),
            "carla_map": carla_map_name,
            "projection": projection,
        }
    )
    return '{"z_offset": ' + json.dumps(value) + ", " + described[1:]


def cached_z_offset(
    lanelet2_path: Path,
    xodr_path: Path,
    carla_map_name: str,
    compute: Callable[[], float],
    *,
    projection: str = "",
    root: Path | None = None,
) -> float:
    """The offset remembered for these map files, else *compute*'s, remembered.

    A cache that cannot be read or written is not an error: the offset is then
    computed, as it was before there was a cache.
    """
    key = z_offset_key(lanelet2_path, xodr_path, carla_map_name, projection)
    entry = (root or map_root() / DERIVED_SUBDIR / "z_offset") / f"{key}.json"
    try:
        value = float(json.loads(entry.read_text(encoding="utf-8"))["z_offset"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    else:
        logger.info("Lanelet2 z offset %.3f m from the cache (%s)", value, entry.name)
        return value

    value = compute()
    # Per process, so two runs storing the same key never write one file; the
    # rename is atomic, so a reader sees one whole entry or none.
    temporary = entry.with_name(f"{entry.name}.{os.getpid()}.tmp")
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            _entry_json(value, lanelet2_path, xodr_path, carla_map_name, projection),
            encoding="utf-8",
        )
        temporary.replace(entry)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        logger.warning(
            "Could not cache the Lanelet2 z offset at %s", entry, exc_info=True
        )
    return value
