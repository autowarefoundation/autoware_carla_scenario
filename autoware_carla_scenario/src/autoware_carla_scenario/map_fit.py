"""Whether a scenario config fits the map it is composed on (#2253).

A config written for one map's lanelet ids -- the shipped ``adversity``
cases, on Town10HD_Opt -- does not fit another: on it those ids name other
lanelets or none.  Two things say so, both before anything starts and
without CARLA or Lanelet2:

* ``scenario.required_map``, the ``map.name`` the config was written for
  (:func:`~autoware_carla_scenario.authoring.hydra_config.required_map_refusal`);
* a lanelet id the config names (:func:`config_lanelet_ids`) that is no
  lanelet of the Lanelet2 map it resolved (:func:`lanelet_refusal`, over
  :func:`map_lanelet_ids`).

A run refuses a config that does not fit (``examples.run``:
``check_scenario_maps``, ``check_config_lanelets``), and so does a batch of
scenarios named one by one.  A batch whose scenarios a glob picked
(``scenario='*/*'``, ``'adversity/*'``) skips them instead, with a warning
naming each and why, and runs the rest (:func:`skip_off_map_scenarios`): a
glob over every scenario cannot fit every map.  The ``scenario`` CLI's batch
(and so a SLURM batch), ``scenario-expand`` and the viewer's launcher all
select through it.

Light: nothing here imports CARLA, Lanelet2 or the runner, so the viewer can
use it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from .route.model import _whole_number

__all__ = [
    "NothingFitsError",
    "config_lanelet_ids",
    "lanelet_refusal",
    "map_fit_refusal",
    "map_lanelet_ids",
    "skip_off_map_scenarios",
]

logger = logging.getLogger(__name__)


class NothingFitsError(ValueError):
    """Every scenario a glob picked was skipped: none fits the map."""


def _lanelet_id(key: str, value: Any) -> int:
    """The lanelet id *value* of *key* names, checked as every whole-number
    setting is (:func:`~autoware_carla_scenario.odd.sampler.whole_number_setting`):
    ``242``, ``242.0`` or ``"242"`` -- not ``242.7``, which ``int()`` alone
    truncated to 242 and so spawned the ego on another lanelet, ``abc``, a
    boolean or null (#2089).

    Raises:
        ValueError: naming *key*.
    """
    if value is None:
        raise ValueError(f"{key} must be a lanelet id (a whole number), got None")
    lanelet_id = _whole_number(key, value)
    if lanelet_id is None:  # A blank string, which no whole number is.
        raise ValueError(f"{key} must be a whole number, got {value!r}")
    return lanelet_id


def _waypoint_lanelet_ids(ego_cfg: Any) -> list[int]:
    """``ego.waypoint_lanelet_ids`` of the ``ego`` group *ego_cfg*, each
    checked (:func:`_lanelet_id`); empty when absent, null or empty.

    Raises:
        ValueError: naming the key (and the item), when it is not a list or
            an item is not a lanelet id.
    """
    value = ego_cfg.get("waypoint_lanelet_ids")
    if value is None:
        return []
    if not isinstance(value, (ListConfig, list, tuple)):
        raise ValueError(
            f"ego.waypoint_lanelet_ids must be a list of lanelet ids, got {value!r}"
        )
    return [
        _lanelet_id(f"ego.waypoint_lanelet_ids[{i}]", item)
        for i, item in enumerate(value)
    ]


#: The ``scenario`` keys a built-in scenario names a lanelet by, by scenario
#: name: each is checked against the resolved map (:func:`config_lanelet_ids`)
#: beside the ego's, unless the scenario's ``kind`` is one that does not read
#: it.  ``adversity``'s NPC (or pedestrian) lanelet, which ``nade`` ignores.
_SCENARIO_LANELET_KEYS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "adversity": (("npc_lanelet_id", ("nade",)),),
}


def config_lanelet_ids(cfg: Any) -> list[tuple[str, int]]:
    """The lanelet ids *cfg* names, each with its key, in the order given.

    The ego's -- ``ego.spawn_lanelet_id``, ``ego.goal_lanelet_id`` and
    ``ego.waypoint_lanelet_ids[<i>]`` -- and a built-in scenario's own
    (:data:`_SCENARIO_LANELET_KEYS`).  Each is read as building the run reads
    it (:func:`_lanelet_id`); one that is absent or null is left out.

    Raises:
        ValueError: naming the key, when a value is not a lanelet id.
    """
    named: list[tuple[str, int]] = []
    ego_cfg = cfg.get("ego")
    if isinstance(ego_cfg, (DictConfig, Mapping)):
        for key in ("spawn_lanelet_id", "goal_lanelet_id"):
            value = ego_cfg.get(key)
            if value is not None:
                named.append((f"ego.{key}", _lanelet_id(f"ego.{key}", value)))
        for i, lanelet_id in enumerate(_waypoint_lanelet_ids(ego_cfg)):
            named.append((f"ego.waypoint_lanelet_ids[{i}]", lanelet_id))
    scenario = cfg.get("scenario")
    if isinstance(scenario, (DictConfig, Mapping)):
        keys = _SCENARIO_LANELET_KEYS.get(str(scenario.get("name") or ""), ())
        for key, ignored_by in keys:
            if str(scenario.get("kind") or "") in ignored_by:
                continue
            value = scenario.get(key)
            if value is not None:
                named.append((f"scenario.{key}", _lanelet_id(f"scenario.{key}", value)))
    return named


#: The lanelet ids of each Lanelet2 file :func:`map_lanelet_ids` scanned, by
#: path, with the file's size and modification time: a batch's cases, and a
#: sweep's, share one map.
_MAP_LANELET_IDS: dict[str, tuple[int, int, frozenset[int]]] = {}


def map_lanelet_ids(lanelet2_path: object) -> frozenset[int] | None:
    """The ids of the lanelets of the Lanelet2 file *lanelet2_path*, or ``None``.

    Read off the ``.osm`` as XML -- each ``relation`` tagged ``type=lanelet``
    that is not marked ``action=delete``, as Lanelet2 reads them -- not loaded
    with Lanelet2: no projection, no routing graph, nothing the run loads
    again, so the check costs a scan of the file, once a file
    (:data:`_MAP_LANELET_IDS`).  ``None`` when *lanelet2_path* is not a path
    (``None``: the config names no Lanelet2 map), there is no such file, it
    does not parse, or it has no lanelet at all (not a Lanelet2 map this can
    read): nothing to check against, and the run reports the map as it
    always did.
    """
    if not isinstance(lanelet2_path, (str, Path)):
        return None
    path = Path(lanelet2_path)
    try:
        stat = path.stat()
    except OSError as error:
        # A relative path is read against the working directory: say so, as
        # the check then passes every lanelet (#2253).
        logger.debug("Not checking lanelet ids against %s: %s", path, error)
        return None
    key = str(path.resolve())
    cached = _MAP_LANELET_IDS.get(key)
    if (
        cached is not None
        and cached[0] == stat.st_size
        and cached[1] == stat.st_mtime_ns
    ):
        return cached[2]
    import xml.etree.ElementTree as ElementTree  # noqa: PLC0415

    ids: set[int] = set()
    try:
        for _event, element in ElementTree.iterparse(str(path), events=("end",)):
            # Lanelet2 skips an element JOSM marked deleted (action=delete).
            if element.tag == "relation" and element.get("action") != "delete":
                for tag in element.iter("tag"):
                    if tag.get("k") == "type" and tag.get("v") == "lanelet":
                        try:
                            ids.add(int(element.get("id") or ""))
                        except ValueError:
                            pass
                        break
                element.clear()
            elif element.tag in ("node", "way", "relation"):
                element.clear()
    except (OSError, ElementTree.ParseError):
        logger.debug("could not scan %s for its lanelets", path, exc_info=True)
        return None
    if not ids:
        return None
    found = frozenset(ids)
    _MAP_LANELET_IDS[key] = (stat.st_size, stat.st_mtime_ns, found)
    return found


def lanelet_refusal(
    cfg: Any, map_paths: Any, *, ignore: frozenset[str] = frozenset()
) -> str | None:
    """Say which lanelets *cfg* names that *map_paths*' map does not have.

    Each lanelet id *cfg* names (:func:`config_lanelet_ids`), but those of the
    keys in *ignore*, is looked up among the lanelets of *map_paths*' Lanelet2
    file (:func:`map_lanelet_ids`).  ``None`` when every one is there, or the
    map has no Lanelet2 file this can read (nothing to check against).

    Raises:
        ValueError: naming the key, when a value is not a lanelet id.
    """
    lanelet2_path = getattr(map_paths, "lanelet2_path", None)
    known = map_lanelet_ids(lanelet2_path)
    if known is None:
        return None
    missing = [
        (key, lanelet_id)
        for key, lanelet_id in config_lanelet_ids(cfg)
        if lanelet_id not in known
        # ``ego.waypoint_lanelet_ids[<i>]`` is set by its list's key.
        and key not in ignore
        and key.partition("[")[0] not in ignore
    ]
    if not missing:
        return None
    listed = ", ".join(f"{key}={lanelet_id}" for key, lanelet_id in missing)
    map_name = getattr(map_paths, "name", None)
    return (
        f"no lanelet of map {map_name!r} ({lanelet2_path}) has the id {listed}: "
        "the config names lanelets of another map.  Run it on the map its "
        "lanelet ids were written for (map=<group>), or name lanelets of this one"
    )


#: The lanelet key beside a spawn's ``s`` key, which a ``stop_line_offset``
#: binding that walks back off its pick moves too
#: (``sweeper.bindings._walk_back_lanelet_key``).
_WALK_BACK_SIBLINGS = {
    "s": "lanelet_id",
    "spawn_s": "spawn_lanelet_id",
    "goal_s": "goal_lanelet_id",
}


def _swept_lanelet_keys(cfg: Any) -> frozenset[str]:
    """The lanelet keys *cfg*'s sweep sets for each of its cases.

    A ``sweep.constraints`` key (``ego.spawn_lanelet_id``) is picked on the
    map by the sweep, a ``sweep.bindings`` key (``ego.goal_lanelet_id``,
    ``route_through``) is derived from each pick, and the lanelet key beside
    a bound spawn's ``s`` (``ego.goal_s`` -> ``ego.goal_lanelet_id``) is moved
    when the binding walks back off the pick, as a ``sweep.route`` picks the
    ego's spawn and goal: the value the config writes there is not what the
    cases run with.  Empty when the sweep is not a mapping (the run refuses
    it itself).
    """
    try:
        sweep = OmegaConf.select(cfg, "sweep") if isinstance(cfg, DictConfig) else None
        if sweep is None and isinstance(cfg, Mapping):
            sweep = cfg.get("sweep")
        if not isinstance(sweep, (DictConfig, Mapping)):
            return frozenset()
        keys: set[str] = set()
        constraints = sweep.get("constraints")
        if isinstance(constraints, (DictConfig, Mapping)):
            keys.update(str(key) for key in constraints)
        bindings = sweep.get("bindings")
        if isinstance(bindings, (DictConfig, Mapping)):
            for key in bindings:
                prefix, _, leaf = str(key).rpartition(".")
                keys.add(str(key))
                sibling = _WALK_BACK_SIBLINGS.get(leaf)
                if sibling is not None:
                    keys.add(f"{prefix}.{sibling}" if prefix else sibling)
        if sweep.get("route"):
            keys.update(("ego.spawn_lanelet_id", "ego.goal_lanelet_id"))
        return frozenset(keys)
    except Exception:  # noqa: BLE001 -- an interpolation, ...: the run reports it
        logger.debug("cannot read the sweep's keys", exc_info=True)
        return frozenset()


def _map_key(map_cfg: Any) -> str:
    """*map_cfg* as text, the same for the same map settings."""
    if isinstance(map_cfg, DictConfig):
        map_cfg = OmegaConf.to_container(map_cfg, resolve=True)
    return json.dumps(map_cfg, sort_keys=True, default=str)


def map_fit_refusal(
    cfg: Any,
    resolve_map: Callable[[Any], Any],
    *,
    resolved: dict[str, Any] | None = None,
    expanding: bool = False,
) -> str | None:
    """Say why *cfg* does not fit the map it is composed on, or ``None``.

    Its ``scenario.required_map`` first
    (:func:`~autoware_carla_scenario.authoring.hydra_config.required_map_refusal`),
    then its lanelet ids, on the map *resolve_map* resolves its ``map`` to
    (:func:`lanelet_refusal`).  With *expanding* (its sweep is to be expanded)
    the lanelet keys its sweep sets are not checked
    (:func:`_swept_lanelet_keys`): the sweep picks them on the map.

    What cannot be read -- a map that does not resolve, a lanelet id that is
    not a whole number -- is no refusal here: the run refuses it itself, with
    its own message.

    Args:
        cfg: The composed config.
        resolve_map: Resolves a ``map`` group to its files
            (:func:`~autoware_carla_scenario.maps.resolve_map_paths`).
        resolved: The maps resolved so far, by their settings, filled here:
            the scenarios of a batch share one map.
        expanding: Whether the config's sweep is expanded into its cases.
    """
    from .authoring.hydra_config import required_map_refusal  # noqa: PLC0415

    refusal = required_map_refusal(cfg)
    if refusal is not None:
        return refusal
    maps = {} if resolved is None else resolved
    try:
        map_cfg = cfg.get("map")
        key = _map_key(map_cfg)
        if key not in maps:
            try:
                maps[key] = resolve_map(map_cfg)
            except Exception:  # noqa: BLE001 -- the run reports it
                logger.debug("cannot resolve the map", exc_info=True)
                maps[key] = None
        map_paths = maps[key]
        if map_paths is None:
            return None
        ignore = _swept_lanelet_keys(cfg) if expanding else frozenset()
        return lanelet_refusal(cfg, map_paths, ignore=ignore)
    except Exception:  # noqa: BLE001 -- the run reports it
        logger.debug("cannot check the config's lanelets", exc_info=True)
        return None


def skip_off_map_scenarios(
    pattern: str,
    names: list[str],
    compose: Callable[[str], Any],
    resolve_map: Callable[[Any], Any],
    *,
    expanding: bool = False,
) -> list[str]:
    """The scenarios the glob *pattern* picked (*names*) that fit the map.

    Each is composed (*compose*: its name to its config, or ``None`` when it
    does not compose) and asked :func:`map_fit_refusal`; one that does not fit
    is skipped with a warning naming it and why, and the rest are kept, in
    order.  One that does not compose is kept: the run reports it.  Only a
    glob's scenarios are skipped: a scenario named alone is refused, as it
    always was.

    Raises:
        NothingFitsError: when every scenario was skipped, naming them.
    """
    resolved: dict[str, Any] = {}
    kept: list[str] = []
    skipped: list[str] = []
    map_name: Any = None
    for name in names:
        cfg = compose(name)
        if cfg is None:
            kept.append(name)
            continue
        refusal = map_fit_refusal(
            cfg, resolve_map, resolved=resolved, expanding=expanding
        )
        if refusal is None:
            kept.append(name)
            continue
        if map_name is None:
            try:
                map_name = (cfg.get("map") or {}).get("name")
            except Exception:  # noqa: BLE001 -- only for the message
                map_name = None
        logger.warning(
            "Skipping %s (picked by scenario=%s): it does not fit the map: %s",
            name,
            pattern,
            refusal,
        )
        skipped.append(name)
    if skipped:
        logger.warning(
            "scenario=%s: skipped %d of %d scenario(s) that do not fit the map: %s",
            pattern,
            len(skipped),
            len(names),
            ", ".join(skipped),
        )
    if names and not kept:
        raise NothingFitsError(
            f"no scenario that scenario={pattern} matches fits map "
            f"{map_name!r}: all {len(skipped)} were skipped "
            f"({', '.join(skipped)}).  Pick the map they were written for "
            "(map=<group>), or other scenarios"
        )
    return kept
