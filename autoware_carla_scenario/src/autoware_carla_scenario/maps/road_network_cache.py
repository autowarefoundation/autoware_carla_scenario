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

Unpickling runs code. An entry is only ever read from the user's own map
cache, written by this module; nothing else should be put there.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import logging
import os
import pickle
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


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: io.BufferedIOBase, elements: list[Any]) -> None:
        super().__init__(file)
        self._elements = elements

    def persistent_load(self, pid: Any) -> Any:
        return self._elements[pid]


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
    # Per process, so two runs storing the same key never write one file; the
    # rename is atomic, so a reader sees one whole entry or none.
    temporary = entry.with_name(f"{entry.name}.{os.getpid()}.tmp")
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            _Pickler(stream, elements, set(map(id, nodes))).dump(
                (
                    len(elements),
                    nodes,
                    [vars(node) for node in nodes],
                    network.road_ids_to_object,
                )
            )
        temporary.replace(entry)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _load(network: RoadNetwork, entry: Path) -> dict[str, Any]:
    elements = list(network.root.iter())
    with entry.open("rb") as stream:
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
    entry = (root or map_root() / DERIVED_SUBDIR / "road_network") / f"{key}.pickle"

    started = time.monotonic()
    try:
        roads = _load(network, entry)
    except FileNotFoundError:
        pass
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
    try:
        _dump(network, entry)
    except Exception:  # noqa: BLE001 -- the roads are built, caching is a bonus
        logger.warning(
            "Could not cache the OpenDRIVE roads at %s", entry, exc_info=True
        )
