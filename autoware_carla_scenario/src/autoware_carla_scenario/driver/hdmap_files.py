"""The names of a map set's files, and the id a map set is written under.

What :mod:`.hdmap_export` writes and reads that needs neither roadgen nor the
file system: the file names a set holds, where each format is written, and
:func:`map_id_for`. Kept apart so the static check compiles them
(:mod:`.hdmap_export` itself is not checked); it re-exports every name here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Mapping, Optional, Tuple

__all__ = [
    "IR_FILE",
    "MANIFEST_FILE",
    "MAP_FORMATS",
    "PROJECTOR_FILE",
    "READ_FILE",
    "READ_TRACE_FILE",
    "SOURCE_FILE",
    "map_id_for",
]

#: The file listing what a map set holds; written last, so a set without it is
#: incomplete.
MANIFEST_FILE = "manifest.json"
SOURCE_FILE = "map.xodr"
IR_FILE = "map.ir.json"
#: What roadgen actually read: the source with CARLA's departures fixed. Kept,
#: because the read trace names it and records its digest.
READ_FILE = "map.roadgen.xodr"
READ_TRACE_FILE = "map.roadgen.xodr.read.trace.json"
#: The projection a given Lanelet2 map is read with, copied when it sits beside it.
PROJECTOR_FILE = "map_projector_info.yaml"

#: Where each format is written, relative to the set's directory.
MAP_FORMATS: Dict[str, str] = {
    "lanelet2": "lanelet2_map.osm",
    "opendrive": "roadgen.xodr",
    "osm": "openstreetmap.osm",
    "clipgt": "clipgt",
    # A CommonRoad 2020a scenario: the lanelets, signs, lights and
    # intersections, with the planning problem its schema requires.
    "commonroad": "commonroad.xml",
    # The run's Lanelet2 map as a tier4/e2e-devkit MapResource, for a policy
    # reading T4Scenes (``driver.t4_scene``); never converted by roadgen.
    "t4": "t4_map.pb",
}


def map_id_for(
    map_name: str, opendrive: str, provided: Optional[Mapping[str, bytes]] = None
) -> str:
    """``<map name>-<first 12 hex digits of a SHA-256 of the map's content>``.

    The content is the OpenDRIVE, and with ``provided`` -- format name to the
    SHA-256 of a file given rather than converted -- those digests too; without
    it the id digests the OpenDRIVE alone. Either way the id names the map's
    content, so a driver holding an older copy of a map with the same name (or
    one given another Lanelet2 file) cannot mistake it for this one.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", map_name.rsplit("/", 1)[-1]) or "map"
    digest = hashlib.sha256(opendrive.encode())
    # An explicit branch rather than ``provided or {}``, which the check cannot
    # type; an empty mapping and None both add nothing.
    entries: List[Tuple[str, bytes]] = []
    if provided is not None:
        entries = sorted(provided.items())
    for format_name, file_digest in entries:
        digest.update(b"\0" + format_name.encode() + b"\0" + file_digest)
    return f"{name}-{digest.hexdigest()[:12]}"
