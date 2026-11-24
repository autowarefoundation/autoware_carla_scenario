"""Remember a map's pyxodr roads across runs.

The first read of :attr:`~..coordinate.map_manager.MapManager.road_network`
builds every road of the OpenDRIVE map, its reference line, lane sections and
lanes: on a large map (Odaiba, 3,890 roads) that is some 15 seconds of every
run, for objects that only change with the map. They are kept under the map
cache's ``.derived`` directory, keyed by the content of the OpenDRIVE file and
by what pyxodr was asked to build from it, so an edited map file is a new key
rather than a stale value.

An entry is a pickle of the network's roads that refers to the XML by position:
each lxml element is stored as its index in the document's ``root.iter()``,
and the document is parsed afresh from the file when the entry is read. So the
XML is never stored, and every element a loaded road holds is one of
``RoadNetwork.root``'s tree, as it is in a built one.

Unpickling runs code, and a map cache may be a tree other users can reach
(a group-shared ``AUTOWARE_DATA_DIR``, #2325), so an entry is read only when
both it and its directory belong to the current user and neither is writable
by the group or others, and only the classes and functions this module writes
(:data:`_ALLOWED_GLOBALS`) are resolved from it.  An entry refused for either
reason is a cache miss, with a warning: the roads are built, and stored again
when the directory is the user's own.  A directory of the user's that the group
may write to (made under a umask of ``002``) is closed to it when an entry is
stored, so it is refused once, not on every run; and where the directory is
another user's (a map root shared with a group), this user's entries go to a
sibling of their own, ``road_network-<uid>`` (:func:`_own_directory`).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import logging
import os
import pickle
import stat
import tempfile
import time
import weakref
from collections.abc import Iterable
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree
from pyxodr.road_objects.network import RoadNetwork

from .cache import DERIVED_SUBDIR, map_root

logger = logging.getLogger(__name__)

__all__ = ["load_roads", "open_road_network"]

#: Bumped whenever this module's format or the pyxodr objects it stores change,
#: so entries written the old way are not reused.
_METHOD_VERSION = 1

#: The sha256 of the bytes each network open_road_network() opened was parsed from.
_DIGESTS: weakref.WeakKeyDictionary[RoadNetwork, str] = weakref.WeakKeyDictionary()


@functools.cache
def _pyxodr_version() -> str:
    try:
        return version("pyxodr")
    except PackageNotFoundError:
        return "unknown"


def _road_network_key(network: RoadNetwork, content_digest: str) -> str:
    """The cache key: the file's content, pyxodr and its options, the method."""
    digest = hashlib.sha256()
    for part in (
        f"v{_METHOD_VERSION}",
        _pyxodr_version(),
        repr(float(network.resolution)),
        repr(sorted(network.ignored_lane_types)),
        content_digest,
    ):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def open_road_network(xodr_path: Path) -> RoadNetwork:
    """Parse *xodr_path* as pyxodr does, its roads not built yet.

    The file is read once: the network is parsed from the same bytes its cache
    key is taken from, so the key always describes the document it is used with.
    Its options (``resolution``, ``ignored_lane_types``) are read into the key
    when the roads are loaded.
    """
    content = xodr_path.read_bytes()
    # pyxodr hands its argument to lxml.etree.parse, which takes a file object.
    network = RoadNetwork(io.BytesIO(content))  # type: ignore[arg-type]
    _DIGESTS[network] = hashlib.sha256(content).hexdigest()
    return network


def _is_node(value: object) -> bool:
    """A pyxodr object (a road, lane section, lane...), its enums aside."""
    return type(value).__module__.startswith("pyxodr.") and not isinstance(value, Enum)


def _nodes(roads: Iterable[object]) -> list[Any]:
    """Every pyxodr object reachable from *roads*, each once."""
    seen: set[int] = set()
    nodes: list[Any] = []
    pending: list[object] = list(roads)
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif _is_node(value) and id(value) not in seen:
            seen.add(id(value))
            nodes.append(value)
            pending.extend(vars(value).values())
    return nodes


class _Refused(pickle.UnpicklingError):
    """An entry this module will not read: whose file, or directory, may have
    been written by someone else, or that names a class it never writes.
    *foreign_directory*: its directory is another user's, so an entry stored
    there would be refused again."""

    def __init__(self, reason: str, *, foreign_directory: bool = False) -> None:
        super().__init__(reason)
        self.foreign_directory = foreign_directory


def _distrust(info: os.stat_result, what: str) -> str | None:
    """Why the file or directory *info* describes may have been written by
    someone else, or None: it is another user's, or the group or others may
    write to it."""
    uid = os.getuid()
    if info.st_uid != uid:
        return f"{what} is owned by uid {info.st_uid}, not {uid}"
    if info.st_mode & _GROUP_OTHER_WRITE:
        return f"{what} is writable by others (mode {stat.S_IMODE(info.st_mode):o})"
    return None


_GROUP_OTHER_WRITE = stat.S_IWGRP | stat.S_IWOTH


def _own_directory(directory: Path) -> Path:
    """*directory*, or where this user's entries go when it is another user's:
    ``<directory>-<uid>`` beside it, so that users sharing a map root each
    cache their own roads instead of rebuilding them on every run."""
    try:
        owner = os.stat(directory).st_uid
    except OSError:
        return directory  # made by _dump, as the user's
    uid = os.getuid()
    if owner == uid:
        return directory
    return directory.with_name(f"{directory.name}-{uid}")


def _shell(cls: type[Any]) -> object:
    """An instance of *cls* with no state yet: the state is set after."""
    return object.__new__(cls)


def _view(base: np.ndarray, offset: int, shape: Any, strides: Any) -> np.ndarray:
    return np.ndarray(shape, base.dtype, buffer=base, offset=offset, strides=strides)


class _Pickler(pickle.Pickler):
    """Writes elements as their index in the tree, views as views of their base.

    *shelled* are the pyxodr objects written state first: an object is written
    as an empty shell (:func:`_shell`) and its state written after every shell,
    so a long chain of linked lanes is never pickled one recursion level per
    link.
    """

    def __init__(
        self, file: io.BufferedIOBase, elements: dict[Any, int], shelled: set[int]
    ) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._elements = elements
        self._shelled = shelled

    def persistent_id(self, obj: Any) -> int | None:
        if isinstance(obj, etree._Element):
            return self._elements[obj]
        return None

    def reducer_override(self, obj: Any) -> Any:
        if id(obj) in self._shelled:
            return _shell, (type(obj),)
        # pyxodr slices lane sections and lanes out of their road's lines: kept
        # as views of the one base, not as copies of their own.
        if isinstance(obj, np.ndarray):
            base = obj.base
            if (
                isinstance(base, np.ndarray)
                and base.base is None
                and base.flags.c_contiguous
                and base.dtype == obj.dtype
                and not obj.dtype.hasobject
            ):
                offset = (
                    obj.__array_interface__["data"][0]
                    - base.__array_interface__["data"][0]
                )
                return _view, (base, offset, obj.shape, obj.strides)
        return NotImplemented


#: Every global an entry may name, as ``(module, name)``: what :class:`_Pickler`
#: writes (read off a written entry with ``pickletools``) -- the shells and
#: views above, numpy's arrays (and its scalars, which a road's state may
#: hold), and pyxodr's roads, lane sections, lanes and their enums.
_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        (__name__, "_shell"),
        (__name__, "_view"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        # numpy 1.x's names of the same (numpy>=1.21 is allowed), so that an
        # entry written there is not refused on every run.
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("pyxodr.road_objects.lane", "ConnectionPosition"),
        ("pyxodr.road_objects.lane", "Lane"),
        ("pyxodr.road_objects.lane", "LaneOrientation"),
        ("pyxodr.road_objects.lane", "TrafficOrientation"),
        ("pyxodr.road_objects.lane_section", "LaneSection"),
        ("pyxodr.road_objects.road", "Road"),
    }
)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: io.BufferedIOBase, elements: list[Any]) -> None:
        super().__init__(file)
        self._elements = elements

    def persistent_load(self, pid: Any) -> Any:
        return self._elements[pid]

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise _Refused(f"it names {module}.{name}, which the cache never writes")
        return super().find_class(module, name)


def _force(roads: Iterable[Any]) -> None:
    """Build what pyxodr builds only for linked roads: lane sections, lanes.

    So that a loaded network needs no building. One that raises (a zero-length
    lane, say) is left unbuilt, to raise where it is used, as it does unloaded.
    """
    for road in roads:
        try:
            sections = road.lane_sections
        except Exception:  # noqa: BLE001 -- raised again where it is used
            continue
        for section in sections:
            for side in ("left_lanes", "right_lanes"):
                with contextlib.suppress(Exception):
                    getattr(section, side)


def _dump(network: RoadNetwork, entry: Path) -> None:
    elements = {element: i for i, element in enumerate(network.root.iter())}
    nodes = _nodes(network.road_ids_to_object.values())
    # Through a scratch file mkstemp names in the entry's directory, as
    # maps.cache._write_atomically writes: two writers storing the same key --
    # other processes, or other hosts sharing the map root -- never write one
    # file (#2165).  The rename is atomic, so a reader sees one whole entry or
    # none, and the scratch file is removed when the write fails.
    # A directory its group or others may write to is never read from
    # (_load): made 0o755 (less the umask), and one of the user's that is open
    # to writing -- made before, under a umask of 002 -- is closed, so that
    # what is stored now is read on the next run.
    entry.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    info = os.stat(entry.parent)
    if info.st_uid == os.getuid() and info.st_mode & _GROUP_OTHER_WRITE:
        os.chmod(entry.parent, stat.S_IMODE(info.st_mode) & ~_GROUP_OTHER_WRITE)
    fd, name = tempfile.mkstemp(
        prefix=f".{entry.name}.", suffix=".tmp", dir=str(entry.parent)
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            _Pickler(stream, elements, set(map(id, nodes))).dump(
                (
                    len(elements),
                    nodes,
                    [vars(node) for node in nodes],
                    network.road_ids_to_object,
                )
            )
        # What open("wb") makes under the usual umask, not mkstemp's 0o600.
        os.chmod(name, 0o644)
        os.replace(name, entry)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise


def _load(network: RoadNetwork, entry: Path) -> dict[str, Any]:
    """The roads *entry* holds for *network*.

    Raises:
        FileNotFoundError: There is no entry.
        _Refused: The entry, or its directory, may have been written by
            another user, or it names a class the cache never writes.
    """
    # Opened first, so that a missing entry is a plain miss, and not through
    # a symlink; the checks are of the file opened.
    fd = os.open(entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        directory = os.stat(entry.parent)
        problem = _distrust(directory, "its directory")
        if problem is not None:
            raise _Refused(problem, foreign_directory=directory.st_uid != os.getuid())
        problem = _distrust(os.fstat(stream.fileno()), "the entry")
        if problem is not None:
            raise _Refused(problem)
        elements = list(network.root.iter())
        count, nodes, states, roads = _Unpickler(stream, elements).load()  # type: ignore[arg-type]
    if count != len(elements) or len(nodes) != len(states):
        raise ValueError("the entry is not of this document")
    for node, state in zip(nodes, states):
        vars(node).update(state)
    if not isinstance(roads, dict):
        raise TypeError("the entry holds no roads")
    return roads


def load_roads(network: RoadNetwork, *, root: Path | None = None) -> None:
    """Fill *network*'s roads: from the cache, else built by pyxodr and stored.

    Either way the roads are linked, as ``get_roads()`` links them; once loaded,
    ``get_roads()`` must not be called on the network, which would link them a
    second time, and this does nothing. A network open_road_network() did not
    open is only built.
    A cache that cannot be read or written is not an error: the roads are then
    built, as they were before there was a cache.
    """
    if network.road_ids_to_object:
        return
    content_digest = _DIGESTS.get(network)
    if content_digest is None:
        network.get_roads()
        return
    key = _road_network_key(network, content_digest)
    directory = _own_directory(root or map_root() / DERIVED_SUBDIR / "road_network")
    entry = directory / f"{key}.pickle"

    started = time.monotonic()
    store = True
    try:
        roads = _load(network, entry)
    except FileNotFoundError:
        pass
    except _Refused as refused:
        logger.warning("Not reading the cached roads at %s: %s", entry, refused)
        # Not stored where the next run would refuse them again.
        store = not refused.foreign_directory
    except Exception:  # noqa: BLE001 -- any unreadable entry is a miss
        logger.warning("Could not read the cached roads at %s", entry, exc_info=True)
    else:
        network.road_ids_to_object = roads
        logger.info(
            "Loaded %d OpenDRIVE roads from the cache in %.1f s (%s)",
            len(roads),
            time.monotonic() - started,
            entry.name,
        )
        return

    network.get_roads()
    _force(network.road_ids_to_object.values())
    logger.info(
        "Built %d OpenDRIVE roads in %.1f s",
        len(network.road_ids_to_object),
        time.monotonic() - started,
    )
    if not store:
        return
    try:
        _dump(network, entry)
    except Exception:  # noqa: BLE001 -- the roads are built, caching is a bonus
        logger.warning(
            "Could not cache the OpenDRIVE roads at %s", entry, exc_info=True
        )
