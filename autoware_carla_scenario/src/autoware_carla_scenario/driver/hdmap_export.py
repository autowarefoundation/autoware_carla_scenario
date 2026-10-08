"""Write a world's map as files, once, for driver policies to read.

alpasim's services read a scene's vector map from its artifact, never off the
wire; this is the CARLA counterpart. :class:`CarlaDriverEntity` takes the
world's OpenDRIVE (``carla.Map.to_opendrive()``), has roadgen convert it into
whatever format the policy reads, and writes the result under
``<map_dir>/<map_id>/`` (``driver.map_dir``). Each policy step then carries only
what changes -- the traffic lights -- which ``autoware_carla_egodriver.hdmap``
resolves against these files.

Every set also holds the OpenDRIVE itself, roadgen's IR and two kinds of
roadgen trace: the read trace says which IR element each OpenDRIVE road, lane
and signal became, and each format's trace which element of the format each IR
element became. Chained, they are how a light named by OpenDRIVE finds its
stop line in, say, a Lanelet2 map.

Needs roadgen (the ``map`` extra, ``autoware-carla-scenario[map]``), imported
only when a map is written: a policy reading the files does not need it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

from ..utils.opendrive import sanitize_opendrive

__all__ = ["MANIFEST_FILE", "MAP_FORMATS", "export_map", "map_id_for"]

logger = logging.getLogger(__name__)

#: The file listing what a map set holds; written last, so a set without it is
#: incomplete.
MANIFEST_FILE = "manifest.json"
SOURCE_FILE = "map.xodr"
IR_FILE = "map.ir.json"
#: What roadgen actually read: the source with CARLA's departures fixed. Kept,
#: because the read trace names it and records its digest.
READ_FILE = "map.roadgen.xodr"
READ_TRACE_FILE = "map.roadgen.xodr.read.trace.json"

#: Where each format is written, relative to the set's directory.
MAP_FORMATS: Dict[str, str] = {
    "lanelet2": "lanelet2_map.osm",
    "opendrive": "roadgen.xodr",
    "osm": "openstreetmap.osm",
    "clipgt": "clipgt",
}


def map_id_for(map_name: str, opendrive: str) -> str:
    """``<map name>-<first 12 hex digits of the OpenDRIVE's SHA-256>``.

    The digest makes the id name the map's content, so a driver holding an
    older copy of a map with the same name cannot mistake it for this one.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", map_name.rsplit("/", 1)[-1]) or "map"
    return f"{name}-{hashlib.sha256(opendrive.encode()).hexdigest()[:12]}"


def export_map(
    opendrive: str,
    map_name: str,
    map_dir: Union[str, Path],
    formats: Iterable[str] = ("lanelet2",),
) -> str:
    """Write ``opendrive`` as ``formats`` under ``<map_dir>/<map_id>/``; returns the id.

    A complete set already holding every format is reused as it is. A set is
    built in a scratch directory beside its destination and moved into place
    whole, so a reader never sees one half written.
    """
    formats = tuple(dict.fromkeys(formats))
    unknown = sorted(set(formats) - set(MAP_FORMATS))
    if unknown:
        raise ValueError(
            f"unknown map format(s) {unknown}; known: {sorted(MAP_FORMATS)}"
        )
    map_id = map_id_for(map_name, opendrive)
    root = Path(map_dir)
    target = root / map_id
    if _has_formats(target, formats):
        logger.info("map %s already written to %s", map_id, target)
        return map_id

    try:
        import roadgen  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "writing a map needs roadgen: install the `map` extra "
            "(autoware-carla-scenario[map])"
        ) from exc

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{map_id}.") as scratch:
        out = Path(scratch)
        (out / SOURCE_FILE).write_text(opendrive, encoding="utf-8")
        sanitized = out / READ_FILE
        sanitized.write_text(sanitize_opendrive(opendrive), encoding="utf-8")
        world_map = roadgen.read_opendrive(str(sanitized))
        for warning in world_map.read_warnings():
            logger.info("roadgen read %s: %s", map_name, warning)

        world_map.export_ir(str(out / IR_FILE))
        world_map.write_read_trace(str(out / READ_TRACE_FILE))
        written = {name: _export(world_map, name, out) for name in formats}
        manifest = {
            "map_id": map_id,
            "map_name": map_name,
            "roadgen_version": getattr(roadgen, "__version__", ""),
            "source": SOURCE_FILE,
            "ir": IR_FILE,
            "read_trace": READ_TRACE_FILE,
            "formats": written,
        }
        (out / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=1), encoding="utf-8"
        )

        if target.exists():
            shutil.rmtree(target)
        out.rename(target)
        # The scratch directory is gone; recreate it so the context manager's
        # cleanup has something to remove.
        out.mkdir()
    logger.info("wrote map %s (%s) to %s", map_id, ", ".join(formats), target)
    return map_id


def _export(world_map: Any, name: str, out: Path) -> Dict[str, str]:
    """Write one format; returns its file and trace, relative to ``out``."""
    path = MAP_FORMATS[name]
    if name == "clipgt":
        clip_id = world_map.export_clipgt(str(out / path))
        return {"path": path, "trace": f"{path}/{clip_id}.clipgt.trace.json"}
    getattr(world_map, f"export_{name}")(str(out / path))
    return {"path": path, "trace": f"{path}.trace.json"}


def _has_formats(directory: Path, formats: Tuple[str, ...]) -> bool:
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return set(formats) <= set(manifest.get("formats", {}))
