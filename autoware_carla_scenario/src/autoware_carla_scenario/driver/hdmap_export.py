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

A Lanelet2 map can be given instead of converted (``driver.lanelet2_path``):
the file (and the ``map_projector_info.yaml`` beside it, if any) is copied into
the set as it is, with no trace, so a policy still gets
each light's stop point but not the lanelet it lies on. A set whose every format
is given that way is written without roadgen, and holds no IR or read trace.

Needs roadgen (the ``map`` extra, ``autoware-carla-scenario[map]``), imported
only when a map is converted: a policy reading the files does not need it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

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
#: The projection a given Lanelet2 map is read with, copied when it sits beside it.
PROJECTOR_FILE = "map_projector_info.yaml"

#: Where each format is written, relative to the set's directory.
MAP_FORMATS: Dict[str, str] = {
    "lanelet2": "lanelet2_map.osm",
    "opendrive": "roadgen.xodr",
    "osm": "openstreetmap.osm",
    "clipgt": "clipgt",
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
    for format_name, file_digest in sorted((provided or {}).items()):
        digest.update(b"\0" + format_name.encode() + b"\0" + file_digest)
    return f"{name}-{digest.hexdigest()[:12]}"


def export_map(
    opendrive: str,
    map_name: str,
    map_dir: Union[str, Path],
    formats: Iterable[str] = ("lanelet2",),
    lanelet2_path: Union[str, Path, None] = None,
) -> str:
    """Write ``opendrive`` as ``formats`` under ``<map_dir>/<map_id>/``; returns the id.

    With ``lanelet2_path``, that file is the set's Lanelet2 map, copied as it
    is rather than converted by roadgen (and ``lanelet2`` is written even if
    ``formats`` leaves it out), together with the ``map_projector_info.yaml``
    beside it, if any. A complete set already holding every format is reused
    as it is. A set is built in a scratch directory beside its destination and
    moved into place whole, so a reader never sees one half written.
    """
    provided: Dict[str, Path] = {}
    if lanelet2_path is not None:
        provided["lanelet2"] = Path(lanelet2_path).expanduser()
    formats = tuple(dict.fromkeys((*formats, *provided)))
    unknown = sorted(set(formats) - set(MAP_FORMATS))
    if unknown:
        raise ValueError(
            f"unknown map format(s) {unknown}; known: {sorted(MAP_FORMATS)}"
        )
    for name, path in provided.items():
        if not path.is_file():
            raise FileNotFoundError(f"the {name} map {path} is not a file")
    map_id = map_id_for(
        map_name, opendrive, {name: _file_digest(p) for name, p in provided.items()}
    )
    root = Path(map_dir)
    target = root / map_id
    if _has_formats(target, formats):
        logger.info("map %s already written to %s", map_id, target)
        return map_id

    converted = [name for name in formats if name not in provided]
    roadgen = _import_roadgen(converted) if converted else None

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{map_id}.") as scratch:
        out = Path(scratch)
        (out / SOURCE_FILE).write_text(opendrive, encoding="utf-8")
        manifest: Dict[str, Any] = {
            "map_id": map_id,
            "map_name": map_name,
            "roadgen_version": "",
            "source": SOURCE_FILE,
            "ir": None,
            "read_trace": None,
        }
        written: Dict[str, Dict[str, Any]] = {}
        if roadgen is not None:
            sanitized = out / READ_FILE
            sanitized.write_text(sanitize_opendrive(opendrive), encoding="utf-8")
            world_map = roadgen.read_opendrive(str(sanitized))
            for warning in world_map.read_warnings():
                logger.info("roadgen read %s: %s", map_name, warning)

            world_map.export_ir(str(out / IR_FILE))
            world_map.write_read_trace(str(out / READ_TRACE_FILE))
            manifest["roadgen_version"] = getattr(roadgen, "__version__", "")
            manifest["ir"] = IR_FILE
            manifest["read_trace"] = READ_TRACE_FILE
            written = {name: _export(world_map, name, out) for name in converted}
        for name, path in provided.items():
            written[name] = _copy_provided(path, out / MAP_FORMATS[name])
        manifest["formats"] = {name: written[name] for name in formats}
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


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def _copy_provided(source: Path, destination: Path) -> Dict[str, Any]:
    """Copy a given map file, and the projector beside it; its manifest entry."""
    shutil.copyfile(source, destination)
    entry: Dict[str, Any] = {
        "path": destination.name,
        "trace": None,
        "provided": True,
    }
    projector = source.parent / PROJECTOR_FILE
    if projector.is_file():
        shutil.copyfile(projector, destination.parent / PROJECTOR_FILE)
        entry["projector"] = PROJECTOR_FILE
    return entry


def _import_roadgen(converted: Iterable[str]) -> Any:
    try:
        import roadgen  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"converting the map to {sorted(converted)} needs roadgen: install the "
            "`map` extra (autoware-carla-scenario[map])"
        ) from exc
    return roadgen


def _export(world_map: Any, name: str, out: Path) -> Dict[str, Optional[str]]:
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
