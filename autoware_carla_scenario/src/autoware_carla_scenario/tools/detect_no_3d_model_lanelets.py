"""Detect lanelets without a matching 3D model in CARLA.

For each lanelet in the Lanelet2 map, this script:

1. Computes the midpoint position (s = length / 2, t = 0) on the centerline.
2. Converts the position to CARLA world coordinates.
3. Performs a downward ground projection (ray cast) at that position.
4. Records the lanelet ID if no ground hit is detected.

The detected lanelet IDs are written to the map configuration YAML file
under ``no_3d_model_lanelet_ids`` -- in its ``map:`` section when it has one,
at the top level otherwise.  The file is edited as text, so comments and
layout survive; when that cannot be done safely the file is left untouched.

The map files are resolved from the configuration's ``map`` settings the
way ``uv run scenario`` resolves them
(:func:`~autoware_carla_scenario.maps.config.resolve_map_paths`):
``lanelet2_path`` and ``xodr_path`` (``~`` and ``${oc.env:...}`` work),
``directory``, ``source``, ``projector_type`` and ``overwrite_xodr``.  A map
that carries no OpenDRIVE (e.g. ``town10hd_opt.yaml``, which has only
``map.source``) has it read back out of the loaded world and cached, as a run
does.

Usage, from the repository root (where ``nishishinjuku.yaml``'s default
``data/`` paths resolve)::

    uv run detect-no-3d-model \\
        autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map/nishishinjuku.yaml

    # With verbose logging
    uv run detect-no-3d-model -v \\
        autoware_carla_scenario/src/autoware_carla_scenario/examples/conf/map/town10hd_opt.yaml

Coordinate systems
------------------
Lanelet2 (MGRS absolute):  x = East, y = North, z = Up
CARLA world (left-hand):   x = East, y = South, z = Up

    carla_x =   (mgrs_x - mgrs_offset_x)
    carla_y = -(mgrs_y - mgrs_offset_y)
    carla_z =   mgrs_z - z_offset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import copy
import os
import shutil
import tempfile
from collections.abc import Sequence
from typing import Any, NoReturn

import typesafe_carla.carla as carla
import lanelet2.geometry
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from tqdm import tqdm

from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.transform import (
    _interpolate_at_s as _interpolate_at_s_4,
)
from autoware_carla_scenario.maps import (
    MapCacheError,
    MapPaths,
    MapResolutionError,
    MapSourceError,
    capture_opendrive,
    resolve_map_paths,
)
from autoware_carla_scenario.maps.opendrive import (
    map_asset_env_var,
    map_asset_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main detection logic
# ---------------------------------------------------------------------------


def detect_no_3d_model_lanelets(
    world: carla.World,
    lanelet_map: lanelet2.core.LaneletMap,
    mgrs_offset: tuple[float, float],
    z_offset: float,
    *,
    ray_distance_upper: float = 5.0,
    ray_distance_lower: float = 5.0,
) -> list[int]:
    """Return sorted list of lanelet IDs that have no ground geometry in CARLA.

    For each lanelet, the midpoint (s = length / 2, t = 0) is computed and
    converted to CARLA world coordinates.  A downward ray is cast from
    ``z_estimate + ray_distance_upper`` with a total search distance of
    ``ray_distance_upper + ray_distance_lower``.

    - If the ray hits nothing (``result is None``), the lanelet is recorded.
    - If the ray hits geometry with ``CityObjectLabel.NONE``, a warning is
      logged but the lanelet is **not** recorded (geometry exists, just no
      semantic tag).
    - If ``ground_projection`` raises ``RuntimeError`` (mesh not loaded),
      the lanelet is recorded.

    Parameters
    ----------
    world:
        An active ``carla.World`` instance.
    lanelet_map:
        A loaded ``lanelet2.LaneletMap`` instance.
    mgrs_offset:
        ``(offset_x, offset_y)`` for converting MGRS to XODR coordinates.
    z_offset:
        ``lanelet2_z - carla_z`` vertical offset.
    ray_distance_upper:
        Search range (m) above the estimated z.
    ray_distance_lower:
        Search range (m) below the estimated z.

    Returns
    -------
    list[int]
        Sorted lanelet IDs where ground projection found no hit.
    """
    no_model_ids: list[int] = []
    total = len(lanelet_map.laneletLayer)

    for lanelet in tqdm(
        lanelet_map.laneletLayer, total=total, desc="Checking lanelets"
    ):
        length = lanelet2.geometry.length2d(lanelet)
        if length < 1e-6:
            logger.warning("Lanelet %d has near-zero length, skipping", lanelet.id)
            continue

        points = [(p.x, p.y, p.z) for p in lanelet.centerline]
        s_mid = length / 2.0
        mgrs_x, mgrs_y, mgrs_z, _ = _interpolate_at_s_4(points, s_mid)

        # MGRS -> CARLA world coordinates
        carla_x = mgrs_x - mgrs_offset[0]
        carla_y = -(mgrs_y - mgrs_offset[1])
        carla_z = mgrs_z - z_offset

        origin = carla.Location(
            x=carla_x,
            y=carla_y,
            z=carla_z + ray_distance_upper,
        )
        search_distance = ray_distance_upper + ray_distance_lower

        try:
            result = world.ground_projection(origin, search_distance)
        except AttributeError:
            raise RuntimeError(
                "world.ground_projection() is not available in this CARLA version. "
                "A CARLA build that supports ground_projection is required."
            ) from None
        except RuntimeError as exc:
            logger.warning(
                "ground_projection raised RuntimeError for lanelet %d "
                "at (%.2f, %.2f, %.2f): %s",
                lanelet.id,
                carla_x,
                carla_y,
                carla_z,
                exc,
            )
            no_model_ids.append(lanelet.id)
            continue

        if result is None:
            logger.info(
                "No ground hit for lanelet %d at (%.2f, %.2f, %.2f)",
                lanelet.id,
                carla_x,
                carla_y,
                carla_z,
            )
            no_model_ids.append(lanelet.id)
            continue

        if result.label == carla.CityObjectLabel.NONE:
            logger.warning(
                "Ground hit at lanelet %d has CityObjectLabel.NONE "
                "at (%.2f, %.2f); geometry exists but has no semantic tag",
                lanelet.id,
                carla_x,
                carla_y,
            )

        logger.debug(
            "Lanelet %d: hit z=%.3f (estimate=%.3f, delta=%.3f)",
            lanelet.id,
            result.location.z,
            carla_z,
            result.location.z - carla_z,
        )

    return sorted(no_model_ids)


# ---------------------------------------------------------------------------
# YAML update
# ---------------------------------------------------------------------------

#: The key the detected lanelet IDs are written to.
NO_3D_MODEL_KEY = "no_3d_model_lanelet_ids"


class YamlUpdateError(ValueError):
    """The map configuration YAML cannot be updated safely as text."""


def _scalar_key(node: yaml.Node) -> str | None:
    """Return the string value of a plain or quoted string key, else ``None``."""
    if isinstance(node, yaml.ScalarNode) and node.tag == "tag:yaml.org,2002:str":
        return str(node.value)
    return None


def _find_key(
    mapping: yaml.MappingNode, key: str
) -> tuple[yaml.Node, yaml.Node] | None:
    """Return the ``(key, value)`` node pair for *key* in *mapping*, if any."""
    matches = [(k, v) for k, v in mapping.value if _scalar_key(k) == key]
    if len(matches) > 1:
        raise YamlUpdateError(f"duplicate key {key!r}")
    return matches[0] if matches else None


def _content_end(node: yaml.Node, text: str) -> int:
    """Return the index just past the last character of *node*'s own text.

    A block collection's ``end_mark`` lies at the next token, past any blank
    lines and comments that follow it, so for those the end of the last
    child is used instead.  A block scalar (``|`` or ``>``) ends past the
    line breaks (and blank lines) after its last line, so those are left out.
    """
    if isinstance(node, yaml.CollectionNode) and not node.flow_style and node.value:
        last = node.value[-1]
        return _content_end(
            last[1] if isinstance(node, yaml.MappingNode) else last, text
        )
    end = int(node.end_mark.index)
    if isinstance(node, yaml.ScalarNode) and node.style in ("|", ">"):
        start = int(node.start_mark.index)
        while end > start and text[end - 1] in " \t\r\n":
            end -= 1
    return end


def _format_value(
    lanelet_ids: Sequence[int], indent: str, newline: str, *, flow: bool
) -> str:
    """Render ``no_3d_model_lanelet_ids: <ids>`` starting at the key."""
    if not lanelet_ids or flow:
        items = ", ".join(str(int(lid)) for lid in lanelet_ids)
        return f"{NO_3D_MODEL_KEY}: [{items}]"
    lines = [f"{NO_3D_MODEL_KEY}:"]
    lines.extend(f"{indent}  - {int(lid)}" for lid in lanelet_ids)
    return newline.join(lines)


def _target_path(data: dict[Any, Any]) -> tuple[str, ...]:
    """Return where the key lives: under ``map:`` if present, else top level.

    This mirrors ``_map_section``, which reads the map settings from
    ``raw.get("map", raw)``.
    """
    return ("map",) if "map" in data else ()


def update_no_3d_model_lanelet_ids_text(text: str, lanelet_ids: Sequence[int]) -> str:
    """Return *text* with ``no_3d_model_lanelet_ids`` set to *lanelet_ids*.

    The key is set in the ``map:`` section when the document has one and at
    the top level otherwise -- the same place ``_resolve_map_config`` reads
    the map settings from.  The edit is textual, so comments, ``${...}``
    interpolations, layout and line endings (LF or CRLF) survive:

    * an existing value in any style (a block list, indented or not, a flow
      list, an empty value, ``null``, a scalar) is replaced -- a non-empty
      flow list stays a flow list, anything else becomes a block list
      (``[]`` when *lanelet_ids* is empty);
    * a missing key is added after the last entry of its section.

    The result is parsed again and must equal the original document with
    only that one value changed.

    Raises
    ------
    YamlUpdateError
        If *text* is not a YAML mapping, the section to edit is not a block
        mapping (e.g. ``map: {name: X}``), the key is duplicated, or the edit
        would change anything else in the document.
    """
    ids = [int(lid) for lid in lanelet_ids]
    newline = "\r\n" if "\r\n" in text else "\n"

    try:
        original = yaml.safe_load(text)
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise YamlUpdateError(f"not valid YAML: {exc}") from exc
    if not isinstance(original, dict) or not isinstance(root, yaml.MappingNode):
        raise YamlUpdateError("expected a YAML mapping at the top level")

    path = _target_path(original)
    label = ".".join((*path, NO_3D_MODEL_KEY))
    section: yaml.Node = root
    for name in path:
        found = (
            _find_key(section, name) if isinstance(section, yaml.MappingNode) else None
        )
        if found is None:  # safe_load found it, so the composer does too
            raise YamlUpdateError(f"cannot set {label}: {name!r} not found")
        section = found[1]
    if not isinstance(section, yaml.MappingNode):
        raise YamlUpdateError(
            f"cannot set {label}: {'.'.join(path)!r} is not a mapping"
        )
    if section.flow_style:
        where = repr(".".join(path)) if path else "the top level"
        raise YamlUpdateError(
            f"cannot set {label}: {where} is a flow mapping ({{...}}), which "
            "cannot be edited as text; rewrite it as a block mapping"
        )

    first_key = section.value[0][0]
    indent = " " * int(first_key.start_mark.column)
    existing = _find_key(section, NO_3D_MODEL_KEY)
    if existing is not None:
        key_node, value_node = existing
        start = int(key_node.start_mark.index)
        if value_node.start_mark.index < key_node.end_mark.index:
            # The composer hands back the anchored node for an alias.
            raise YamlUpdateError(
                f"cannot set {label}: its value is an alias (*name); "
                "replace it with a list"
            )
        end = max(_content_end(value_node, text), int(key_node.end_mark.index))
        # A non-empty flow list stays one; ``[]`` is only a placeholder.
        flow = bool(
            isinstance(value_node, yaml.SequenceNode)
            and value_node.flow_style
            and value_node.value
        )
        replacement = _format_value(ids, indent, newline, flow=flow)
        result = text[:start] + replacement + text[end:]
    else:
        # After the line holding the section's last value.
        last_end = _content_end(section.value[-1][1], text)
        line_end = text.find("\n", last_end)
        entry = indent + _format_value(ids, indent, newline, flow=False) + newline
        if line_end == -1:
            prefix = text if text.endswith(("\n", "\r")) else text + newline
            result = prefix + entry
        else:
            result = text[: line_end + 1] + entry + text[line_end + 1 :]

    expected = copy.deepcopy(original)
    target = expected
    for name in path:
        target = target[name]
    target[NO_3D_MODEL_KEY] = ids
    try:
        reparsed = yaml.safe_load(result)
    except yaml.YAMLError as exc:
        raise YamlUpdateError(
            f"setting {label} as text would produce invalid YAML: {exc}"
        ) from exc
    if reparsed != expected:
        raise YamlUpdateError(
            f"setting {label} as text would change more than that value"
        )
    return result


def _update_yaml(yaml_path: Path, lanelet_ids: list[int]) -> None:
    """Set ``no_3d_model_lanelet_ids`` in the map configuration YAML.

    The file is edited as text (see :func:`update_no_3d_model_lanelet_ids_text`)
    and replaced atomically (a symlink's target, not the link itself); when
    the edit cannot be made safely, :class:`YamlUpdateError` is raised and
    the file is left untouched.
    """
    raw = yaml_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise YamlUpdateError(f"{yaml_path}: not UTF-8: {exc}") from exc
    try:
        updated = update_no_3d_model_lanelet_ids_text(text, lanelet_ids)
    except YamlUpdateError as exc:
        raise YamlUpdateError(f"{yaml_path}: {exc}") from exc
    if updated == text:
        return

    _write_atomically(Path(os.path.realpath(yaml_path)), updated.encode("utf-8"))


def _write_atomically(path: Path, data: bytes) -> None:
    """Replace *path*'s content with *data* through a temporary file.

    *path* must already be resolved: replacing a symlink would turn it into
    a regular file instead of writing through it.  The temporary file gets
    *path*'s mode and, where permitted, its owner.  When the directory does
    not allow creating it, *path* is written in place instead.
    """
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except PermissionError:
        path.write_bytes(data)
        return
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, tmp_path)
        stat = path.stat()
        try:
            os.chown(tmp_path, stat.st_uid, stat.st_gid)
        except OSError:
            pass  # Only root (or the owner, for the group) may do this.
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Map resolution
# ---------------------------------------------------------------------------


class MapConfigError(ValueError):
    """The map configuration cannot be read or does not name a usable map.

    The message is a single line, which :func:`main` prints as it is.
    """


def _one_line(exc: BaseException) -> str:
    """Return *exc*'s message on one line (YAML errors span several)."""
    return " ".join(str(exc).split()) or type(exc).__name__


def _map_section(yaml_path: Path) -> DictConfig:
    """Load *yaml_path* and return its ``map:`` section, or the whole file.

    The YAML may have the settings nested under a ``map:`` key (Hydra package
    style, as in ``examples/conf/map/*.yaml``) or at the top level.
    """
    try:
        raw = OmegaConf.load(yaml_path)
    except (OSError, UnicodeDecodeError) as exc:
        raise MapConfigError(f"{yaml_path}: cannot read: {_one_line(exc)}") from exc
    except (yaml.YAMLError, OmegaConfBaseException) as exc:
        raise MapConfigError(f"{yaml_path}: not valid YAML: {_one_line(exc)}") from exc
    if not isinstance(raw, DictConfig):
        raise MapConfigError(f"{yaml_path}: expected a YAML mapping, got a sequence")
    try:
        map_cfg = raw.get("map", raw)
    except OmegaConfBaseException as exc:
        raise MapConfigError(f"{yaml_path}: {_one_line(exc)}") from exc
    if not isinstance(map_cfg, DictConfig):
        raise MapConfigError(
            f"{yaml_path}: expected 'map' to be a mapping, "
            f"got {type(map_cfg).__name__}"
        )
    return map_cfg


def _resolve_map_config(yaml_path: Path, *, allow_fetch: bool = True) -> MapPaths:
    """Return the map files the config at *yaml_path* names, without CARLA.

    The ``map`` settings are resolved by
    :func:`~autoware_carla_scenario.maps.config.resolve_map_paths`, exactly as
    ``uv run scenario`` resolves them: ``map.lanelet2_path`` and
    ``map.xodr_path`` (``~`` expanded, ``${oc.env:...}`` interpolated),
    ``map.directory``, ``map.source``, ``map.projector_type`` and
    ``map.overwrite_xodr``.

    A map with no OpenDRIVE of its own (a ``map.source`` or ``map.directory``
    that ships none) is accepted: :func:`_opendrive_for` reads it back out of
    the loaded world, as the runner does.

    Args:
        yaml_path: The map configuration YAML.
        allow_fetch: Whether a ``map.source`` that is not cached yet may be
            cloned.

    Raises:
        MapConfigError: If the file cannot be read or parsed, or does not name
            a usable map.
    """
    map_cfg = _map_section(yaml_path)
    try:
        paths = resolve_map_paths(map_cfg, allow_fetch=allow_fetch)
    except (
        MapSourceError,
        MapCacheError,
        MapResolutionError,
        OmegaConfBaseException,
    ) as exc:
        raise MapConfigError(f"{yaml_path}: {_one_line(exc)}") from exc

    if not paths.name:
        raise MapConfigError(f"{yaml_path}: map.name is not set")
    if paths.lanelet2_path is None:
        raise MapConfigError(
            f"{yaml_path}: names no Lanelet2 map; set map.lanelet2_path, "
            "map.directory or map.source"
        )
    if not paths.lanelet2_path.is_file():
        raise MapConfigError(
            f"{yaml_path}: Lanelet2 file not found: {paths.lanelet2_path}"
        )
    install = paths.install_xodr
    if install is not None and not install.is_file():
        raise MapConfigError(f"{yaml_path}: XODR file not found: {install}")
    if install is None and paths.opendrive_path is None:
        raise MapConfigError(
            f"{yaml_path}: names no OpenDRIVE; set map.xodr_path, or "
            "map.directory or map.source to read one back from CARLA"
        )
    if paths.overwrite_xodr and install is None:
        raise MapConfigError(
            f"{yaml_path}: map.overwrite_xodr needs an OpenDRIVE to install; "
            "set map.xodr_path or leave overwrite_xodr off"
        )
    return paths


def _load_world(client: carla.Client, paths: MapPaths) -> carla.World:
    """Load the map in CARLA the way the runner does.

    With ``map.overwrite_xodr`` the OpenDRIVE is installed into the CARLA
    asset first (``ScenarioRunner.load_map_by_overwriting_xodr``); otherwise
    the map is loaded by name with the road network CARLA ships
    (``ScenarioRunner.load_map_by_name``).

    Raises:
        MapConfigError: If the OpenDRIVE is to be installed and the
            ``<MAP_NAME>_PATH`` variable naming its destination is not set.
    """
    install = paths.install_xodr
    if paths.overwrite_xodr and install is not None:
        env_var = map_asset_env_var(paths.name)
        dest = map_asset_path(paths.name)
        if dest is None:
            raise MapConfigError(
                f"map.overwrite_xodr is set but {env_var} is not; set it to the "
                f"internal .xodr file of {paths.name} inside the CARLA installation"
            )
        logger.info("Overwriting CARLA XODR: %s -> %s", install, dest)
        shutil.copy2(install, dest)
    logger.info("Loading CARLA map %s ...", paths.name)
    return client.load_world(paths.name)


def _opendrive_for(paths: MapPaths, world: carla.World) -> Path:
    """Return the OpenDRIVE to read geometry from, once *world* is loaded.

    The configured (or shipped) file when there is one; otherwise the one
    read back out of CARLA into the map cache, written from *world* the first
    time -- what ``ScenarioQueue`` does for a map that carries no OpenDRIVE.
    """
    if paths.install_xodr is not None:
        return paths.install_xodr
    destination = paths.opendrive_path
    if destination is None:  # _resolve_map_config refuses such a config
        raise MapConfigError(f"{paths.name}: no OpenDRIVE to read")
    if destination.is_file():
        return destination
    written = capture_opendrive(world, destination)
    logger.info("Wrote the loaded world's OpenDRIVE to %s", written)
    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _fail(message: str) -> NoReturn:
    """Print *message* as one line on stderr and exit with status 1."""
    print(f"detect-no-3d-model: error: {message}", file=sys.stderr)
    sys.exit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for detecting lanelets without a CARLA 3D model."""
    parser = argparse.ArgumentParser(
        prog="detect-no-3d-model",
        description=(
            "Detect lanelets without a matching 3D model in CARLA and "
            "write the IDs to a map configuration YAML file.  "
            "The map files are resolved from the config's map settings "
            "(lanelet2_path, xodr_path, directory, source) as "
            "`uv run scenario` resolves them."
        ),
    )
    parser.add_argument(
        "yaml_path",
        type=Path,
        help=(
            "Path to the map configuration YAML (e.g. autoware_carla_scenario/"
            "src/autoware_carla_scenario/examples/conf/map/nishishinjuku.yaml)"
        ),
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="CARLA server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )
    parser.add_argument(
        "--ray-upper",
        type=float,
        default=5.0,
        help="Ray search range above z estimate in metres (default: 5.0)",
    )
    parser.add_argument(
        "--ray-lower",
        type=float,
        default=5.0,
        help="Ray search range below z estimate in metres (default: 5.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="CARLA client timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Resolve the map files from the YAML, before connecting to CARLA
    # ------------------------------------------------------------------
    if not args.yaml_path.is_file():
        _fail(f"YAML config not found: {args.yaml_path}")
    try:
        paths = _resolve_map_config(args.yaml_path)
    except MapConfigError as exc:
        _fail(str(exc))
    lanelet2_path = paths.lanelet2_path
    assert lanelet2_path is not None  # _resolve_map_config checked it
    logger.info("Map name: %s", paths.name)
    logger.info("XODR:     %s", paths.install_xodr or paths.opendrive_path)
    logger.info("Lanelet2: %s", lanelet2_path)

    # ------------------------------------------------------------------
    # Connect to CARLA and load the map
    # ------------------------------------------------------------------
    logger.info("Connecting to CARLA at %s:%d ...", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    try:
        version = client.get_server_version()
        logger.info("Connected to CARLA server version %s", version)
    except RuntimeError as exc:
        _fail(f"cannot connect to the CARLA server: {_one_line(exc)}")

    try:
        world = _load_world(client, paths)
        xodr_path = _opendrive_for(paths, world)
    except MapConfigError as exc:
        _fail(str(exc))
    except (OSError, RuntimeError) as exc:
        _fail(f"cannot load CARLA map {paths.name}: {_one_line(exc)}")

    # ------------------------------------------------------------------
    # Initialize MapManager (loads lanelet2, computes mgrs_offset & z_offset)
    # ------------------------------------------------------------------
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(
        xodr_path=xodr_path,
        lanelet2_path=lanelet2_path,
        carla_world=world,
        projector_type=paths.projector_type,
    )
    logger.info(
        "Loaded Lanelet2 map with %d lanelets",
        len(mm.lanelet_map.laneletLayer),
    )
    logger.info("MGRS offset: (%.4f, %.4f)", mm.mgrs_offset[0], mm.mgrs_offset[1])
    logger.info("z_offset: %.4f", mm.z_offset)

    # ------------------------------------------------------------------
    # Detect lanelets without 3D model
    # ------------------------------------------------------------------
    no_model_ids = detect_no_3d_model_lanelets(
        world=world,
        lanelet_map=mm.lanelet_map,
        mgrs_offset=mm.mgrs_offset,
        z_offset=mm.z_offset,
        ray_distance_upper=args.ray_upper,
        ray_distance_lower=args.ray_lower,
    )

    total = len(mm.lanelet_map.laneletLayer)
    logger.info(
        "Detection complete: %d / %d lanelets have no 3D model",
        len(no_model_ids),
        total,
    )
    if no_model_ids:
        logger.info("Lanelet IDs without 3D model: %s", no_model_ids)

    # ------------------------------------------------------------------
    # Update YAML config
    # ------------------------------------------------------------------
    try:
        _update_yaml(args.yaml_path, no_model_ids)
    except YamlUpdateError as exc:
        logger.error("%s; the file was not changed", exc)
        logger.error("Set %s by hand to: %s", NO_3D_MODEL_KEY, no_model_ids)
        sys.exit(1)
    logger.info("Updated %s with %d lanelet IDs", args.yaml_path, len(no_model_ids))


if __name__ == "__main__":
    main()
