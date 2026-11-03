"""Write a world's map as files, once, for driver policies to read.

alpasim's services read a scene's vector map from its artifact, never off the
wire; this is the CARLA counterpart. :class:`CarlaDriverEntity` takes the
world's OpenDRIVE (``carla.Map.to_opendrive()``), has roadgen convert it into
whatever format the policy reads, and writes the result under
``<map_dir>/<map_id>/`` (``driver.map_dir``). Each policy step then carries only
what changes -- the traffic lights -- which ``carla_driver_interface.hdmap``
resolves against these files.

Every set also holds the OpenDRIVE itself, roadgen's IR and two kinds of
roadgen trace: the read trace says which IR element each OpenDRIVE road, lane
and signal became, and each format's trace which element of the format each IR
element became. Chained, they are how a light named by OpenDRIVE finds its
stop line in, say, a Lanelet2 map.

A Lanelet2 map can be given instead of converted (``driver.lanelet2_path``):
the file (and the ``map_projector_info.yaml`` beside it, if any) is copied into
the set as it is, with an empty trace; ``carla_driver_interface.hdmap`` (2.3
and later) then finds each light on the map itself, as the first traffic-light
element on or ahead of its stop point. A set whose every format
is given that way is written without roadgen, its IR and read trace empty too:
every set has the same shape, so any reader of these files opens it.

Needs roadgen (the ``map`` extra, ``autoware-carla-scenario[map]``), imported
only when a map is converted: a policy reading the files does not need it.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Set, Tuple, Union

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
    for format_name, file_digest in sorted((provided or {}).items()):
        digest.update(b"\0" + format_name.encode() + b"\0" + file_digest)
    return f"{name}-{digest.hexdigest()[:12]}"


def export_map(
    opendrive: str,
    map_name: str,
    map_dir: Union[str, Path],
    formats: Iterable[str] = ("lanelet2",),
    lanelet2_path: Union[str, Path, None] = None,
    t4_map: Optional[bytes] = None,
) -> str:
    """Write ``opendrive`` as ``formats`` under ``<map_dir>/<map_id>/``; returns the id.

    With ``lanelet2_path``, that file is the set's Lanelet2 map, copied as it
    is rather than converted by roadgen (and ``lanelet2`` is written even if
    ``formats`` leaves it out), together with the ``map_projector_info.yaml``
    beside it, if any. With ``t4_map``, a serialized ``MapResource`` of the
    run's Lanelet2 map, the set holds it as its ``t4`` format (and ``t4`` is
    written even if ``formats`` leaves it out); without, ``t4`` cannot be
    written. A complete set already holding every format is reused
    as it is. One lacking some is built again holding the formats it has and
    those asked for: a set only ever grows, its id unchanged. A set is built in
    a scratch directory beside its destination and moved into place whole, so
    a reader never sees one half written.
    """
    provided: Dict[str, Path] = {}
    if lanelet2_path is not None:
        provided["lanelet2"] = Path(lanelet2_path).expanduser()
    given: Dict[str, bytes] = {} if t4_map is None else {"t4": t4_map}
    formats = tuple(dict.fromkeys((*formats, *provided, *given)))
    if "t4" in formats and "t4" not in given:
        raise ValueError(
            "the t4 map format is the run's Lanelet2 map as the T4Scenes reference "
            "it; it is written with driver.t4_scene, not converted"
        )
    unknown = sorted(set(formats) - set(MAP_FORMATS))
    if unknown:
        raise ValueError(
            f"unknown map format(s) {unknown}; known: {sorted(MAP_FORMATS)}"
        )
    for name, path in provided.items():
        if not path.is_file():
            raise FileNotFoundError(f"the {name} map {path} is not a file")
    map_id = map_id_for(
        map_name,
        opendrive,
        {
            **{name: _file_digest(p) for name, p in provided.items()},
            **{name: hashlib.sha256(data).digest() for name, data in given.items()},
        },
    )
    root = Path(map_dir)
    target = root / map_id
    present = _formats_in(target)
    if set(formats) <= set(present):
        logger.info("map %s already written to %s", map_id, target)
        return map_id

    # The id names the content, not the formats, so a set grows by the formats
    # a run asks for and never loses one an earlier run wrote -- nor one a run
    # killed while replacing the set left aside in its scratch directory (only
    # read: it may be a live run's). Every run of one id gives the same files
    # (they are in the id): whatever the set holds that this run does not give
    # was converted, and is converted again, so that every format's trace
    # leads into the same IR. A format this version cannot write (a newer
    # version's, or one the installed roadgen lacks an exporter for) is
    # dropped, with a warning.
    for leftover in sorted(root.glob(f".{map_id}.*/old")):
        present = {**present, **_formats_in(leftover)}
    dropped: Set[str] = set()
    present = _known(present, dropped, map_id)
    root.mkdir(parents=True, exist_ok=True)
    last_error: Optional[OSError] = None
    requested = set(formats)
    for _ in range(len(MAP_FORMATS) + 2):
        formats = tuple(dict.fromkeys((*formats, *present)))
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{map_id}.") as scratch:
            out = Path(scratch) / "set"
            aside = Path(scratch) / "old"
            formats = _build(
                out, map_id, map_name, opendrive, formats, requested, provided, given
            )
            dropped |= set(present) - set(formats)

            # Runs in other processes may write the same set meanwhile (the
            # tasks of a SLURM batch, docs/slurm.md). Theirs is kept if it
            # holds every format this one does -- the id is the content's;
            # one holding formats this one lacks is built again with them.
            present = _known(_formats_in(target), dropped, map_id)
            if set(formats) <= set(present):
                logger.info("map %s written to %s meanwhile", map_id, target)
                return map_id
            if set(present) <= set(formats):
                # Each rename moves a complete set: a reader never sees one half
                # written (only, between the two, none).
                try:
                    if target.exists():
                        target.rename(aside)
                    if set(_known(_formats_in(aside), dropped, map_id)) <= set(formats):
                        out.rename(target)
                        logger.info(
                            "wrote map %s (%s) to %s",
                            map_id,
                            ", ".join(formats),
                            target,
                        )
                        return map_id
                    aside.rename(target)  # grown meanwhile: put it back
                except OSError as error:
                    if aside.exists() and not target.exists():
                        try:
                            aside.rename(target)  # never lose the old set
                        except OSError:
                            pass
                    # Only another run taking or filling the set's place is
                    # retried; anything else (permissions, a read-only file
                    # system) would fail again.
                    if error.errno not in _RACES:
                        raise
                    last_error = error
            present = _known(
                {**_formats_in(target), **_formats_in(aside)}, dropped, map_id
            )
    raise RuntimeError(
        f"map {map_id} kept changing in {target} while written"
    ) from last_error


#: What a rename fails with when another run moved the set or put one in place.
_RACES = frozenset({errno.ENOENT, errno.EEXIST, errno.ENOTEMPTY})


def _known(
    formats: Mapping[str, Any], dropped: Set[str], map_id: str
) -> Dict[str, Any]:
    """``formats`` without those this version cannot write, warning of each."""
    unknown = sorted(n for n in formats if n not in MAP_FORMATS or n in dropped)
    for name in unknown:
        if name not in dropped:
            logger.warning(
                "map %s holds the format %r, which this version cannot write; "
                "it is dropped when the set is written again",
                map_id,
                name,
            )
            dropped.add(name)
    return {n: e for n, e in formats.items() if n in MAP_FORMATS and n not in dropped}


def _build(
    out: Path,
    map_id: str,
    map_name: str,
    opendrive: str,
    formats: Tuple[str, ...],
    required: Set[str],
    provided: Mapping[str, Path],
    given: Mapping[str, bytes],
) -> Tuple[str, ...]:
    """Write the set holding ``formats`` to the new directory ``out``.

    Returns the formats written: those the installed roadgen has no exporter
    for are left out with a warning, unless ``required``. The manifest is
    written last, so a set without it is incomplete.
    """
    converted = [name for name in formats if name not in provided and name not in given]
    roadgen = _import_roadgen(converted) if converted else None
    out.mkdir()
    (out / SOURCE_FILE).write_text(opendrive, encoding="utf-8")
    manifest: Dict[str, Any] = {
        "map_id": map_id,
        "map_name": map_name,
        "roadgen_version": "",
        "source": SOURCE_FILE,
        "ir": IR_FILE,
        "read_trace": READ_TRACE_FILE,
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
        for name in converted:
            if name in required or _can_export(world_map, name):
                written[name] = _export(world_map, name, out)
            else:
                logger.warning(
                    "map %s holds the format %r, which the installed roadgen "
                    "cannot write; it is dropped",
                    map_id,
                    name,
                )
    else:
        _write_json(out / IR_FILE, {})
        _write_json(out / READ_TRACE_FILE, _EMPTY_TRACE)
    for name, path in provided.items():
        written[name] = _copy_provided(path, out / MAP_FORMATS[name])
    for name, data in given.items():
        written[name] = _write_given(data, out / MAP_FORMATS[name])
    formats = tuple(name for name in formats if name in written)
    manifest["formats"] = {name: written[name] for name in formats}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return formats


def _can_export(world_map: Any, name: str) -> bool:
    return hasattr(world_map, f"export_{name}")


_EMPTY_TRACE: Dict[str, Any] = {"links": []}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def _copy_provided(source: Path, destination: Path) -> Dict[str, Any]:
    """Copy a given map file, and the projector beside it; its manifest entry.

    The trace is empty: nothing relates a given file to the OpenDRIVE.
    """
    shutil.copyfile(source, destination)
    trace = f"{destination.name}.trace.json"
    _write_json(destination.parent / trace, _EMPTY_TRACE)
    entry: Dict[str, Any] = {
        "path": destination.name,
        "trace": trace,
        "provided": True,
    }
    projector = source.parent / PROJECTOR_FILE
    if projector.is_file():
        shutil.copyfile(projector, destination.parent / PROJECTOR_FILE)
        entry["projector"] = PROJECTOR_FILE
    return entry


def _write_given(data: bytes, destination: Path) -> Dict[str, Any]:
    """Write a format given as bytes; its manifest entry, with an empty trace."""
    destination.write_bytes(data)
    trace = f"{destination.name}.trace.json"
    _write_json(destination.parent / trace, _EMPTY_TRACE)
    return {"path": destination.name, "trace": trace, "provided": True}


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


def _formats_in(directory: Path) -> Dict[str, Any]:
    """The formats a complete set in ``directory`` holds; none if there is none."""
    try:
        manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    formats = manifest.get("formats") if isinstance(manifest, dict) else None
    return dict(formats) if isinstance(formats, dict) else {}
