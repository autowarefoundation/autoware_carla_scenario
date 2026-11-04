"""Render a :class:`ScenarioDocument` as the framework's Hydra scenario config.

An authored scenario is configured exactly like a hand-written one: a
``# @package _global_`` YAML that sets the ``scenario`` group, the ``ego`` group
and -- when an entity spawns by constraint search -- the ``sweep`` section the
existing :class:`~autoware_carla_scenario.sweeper.LaneletConstraintSweeper`
already understands.  Nothing here is editor-specific; the file this module
produces is one a human could have written.

Spawn keys
----------
The ego's spawn reaches the runner through the framework's own
``ego.spawn_lanelet_id`` / ``ego.spawn_s`` keys, which is what the sweeper
overrides today, with its lateral offset and heading as ``ego.spawn_t`` /
``ego.spawn_heading``.  Other entities get a declared ``scenario.spawn_overrides``
sub-tree so that they are addressable by the same plain ``key=value`` override
mechanism -- the keys are written into the generated YAML precisely so that
Hydra's struct mode accepts them.

Ego keys
--------
The ego's goal travels the same way as its spawn, as ``ego.goal_lanelet_id`` /
``ego.goal_s``, and which stack drives it as ``ego.entity``.  Those are the keys
the runner turns into the ego's configuration, so a document that sends the ego
somewhere has to write them rather than hand the goal over some editor-only
channel.  The goal keys are emitted only when the document has a goal: an ego
the TrafficManager drives may have no destination, and a document being edited
into shape should leave the ``ego`` group's own ``null`` in place rather than a
lanelet nobody chose.

Route keys
----------
A logical scenario -- one with a route search (:attr:`ScenarioDocument.route`)
-- renders its search as ``sweep.route`` and declares the ``scenario.route``
keys a concrete match is carried in (:data:`ROUTE_CONFIG_DEFAULTS`).
``scenario-expand`` (and the sweeper) find the matches and write those keys,
and the ego's spawn and goal, per match; a run that was not expanded leaves
them empty and the scenario searches the map it runs on itself.

Map group
---------
The framework's ``config.yaml`` composes the ``scenario`` group *before* the
``map`` group, so whatever map group a run selects overrides the ``map`` keys a
scenario config writes -- and a scenario config cannot select another one
(Hydra refuses ``override /map`` from a group composed before ``map``).  That
is harmless for a document on a built-in group, which names the group's own
map.  A document that names its map by ``source`` or ``directory`` describes a
map no built-in group does, so an export ships it as a map group of its own
(:func:`build_map_group_config`, ``conf/map/<own_map_group>.yaml``) that sets
every ``map`` key, and the package is run with ``map=<own_map_group>``
(:func:`run_map_group`).  A run that composes another map group anyway --
the default one, when ``map=<own_map_group>`` is left out -- is refused before
anything starts (:func:`own_map_mismatch`, :func:`scenario_map_refusal`), on
every path that composes it: a run, a batch, a sweep, ``scenario-expand``, the
viewer's sweep resolver and ``scenario-odd route``.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import Entity, LaneletSlot, ScenarioDocument
from .persistence import dump_yaml

_logger = logging.getLogger(__name__)

__all__ = [
    "PACKAGE_GLOBAL_HEADER",
    "ROUTE_CONFIG_DEFAULTS",
    "build_map_group_config",
    "build_scenario_config",
    "dump_map_group_config",
    "dump_scenario_config",
    "own_map_group",
    "own_map_mismatch",
    "param_override_key",
    "run_map_group",
    "scenario_map_refusal",
    "slot_lanelet_key",
    "spawn_lanelet_key",
    "spawn_s_key",
    "swept_entity",
    "swept_slot",
]

#: Hydra needs this on the first line for a config that writes into the root.
PACKAGE_GLOBAL_HEADER = "# @package _global_"

#: The ``scenario.route`` keys a route match is carried in, empty: declared so
#: Hydra's struct mode accepts the overrides an expansion writes (see
#: :meth:`autoware_carla_scenario.route.model.RouteMatch.to_config`).
ROUTE_CONFIG_DEFAULTS: dict[str, Any] = {
    "index": 0,
    "lanelet_ids": [],
    "start_s": 0.0,
    "end_s": 0.0,
    "segment_kinds": [],
    "segment_ends": [],
    "segment_lanelet_counts": [],
    "junction_turns": [],
}


def spawn_lanelet_key(entity: Entity) -> str:
    """Return the Hydra key holding *entity*'s spawn lanelet ID."""
    if entity.kind == "ego":
        return "ego.spawn_lanelet_id"
    return f"scenario.spawn_overrides.{entity.id}.lanelet_id"


def spawn_s_key(entity: Entity) -> str:
    """Return the Hydra key holding *entity*'s longitudinal spawn offset."""
    if entity.kind == "ego":
        return "ego.spawn_s"
    return f"scenario.spawn_overrides.{entity.id}.s"


def param_override_key(node_id: str, field: str) -> str:
    """Return the Hydra key holding one searched action/condition parameter.

    Declared as its own sub-tree for the same reason spawns are: the sweeper
    drives a run with plain ``key=value`` overrides, so a lanelet it chooses has
    to be *addressable*, and Hydra's struct mode only accepts a key the exported
    YAML already declares.
    """
    return f"scenario.param_overrides.{node_id}.{field}"


def slot_lanelet_key(document: ScenarioDocument, slot: LaneletSlot) -> str | None:
    """Return the Hydra key holding *slot*'s lanelet id, or ``None``.

    ``None`` means the slot has no override channel and so cannot be swept: the
    goal of a vehicle that is not the ego, which the export does not render at
    all and :mod:`.validator` reports separately.
    """
    entity = document.entity(slot.owner_id)
    if slot.field == "spawn" and entity is not None:
        return spawn_lanelet_key(entity)
    if slot.field == "goal":
        return (
            "ego.goal_lanelet_id"
            if entity is not None and entity.kind == "ego"
            else None
        )
    return param_override_key(slot.owner_id, slot.field)


def swept_slot(document: ScenarioDocument) -> LaneletSlot | None:
    """Return the lanelet slot the sweep drives, if any.

    The lanelet-constraint sweeper enumerates a single target key per run, so
    at most one lanelet in a scenario can be searched for; the first searched
    slot in document order wins and :mod:`.validator` flags the rest.
    """
    for slot in document.searched_lanelet_slots():
        if slot_lanelet_key(document, slot) is not None:
            return slot
    return None


def swept_entity(document: ScenarioDocument) -> Entity | None:
    """Return the entity whose *spawn* drives the sweep, if any.

    A narrower question than :func:`swept_slot` since a search may now name a
    goal or a condition's lanelet instead, and the answer is ``None`` when it
    does: no entity's spawn is being enumerated then.
    """
    slot = swept_slot(document)
    if slot is None or slot.field != "spawn":
        return None
    return document.entity(slot.owner_id)


def _entity_spawn_overrides(document: ScenarioDocument) -> dict[str, Any]:
    """Return the declared ``scenario.spawn_overrides`` tree for non-ego entities."""
    return {
        entity.id: {
            "lanelet_id": entity.spawn.lanelet_id,
            "s": entity.spawn.s.value,
        }
        for entity in document.entities
        if entity.kind != "ego"
    }


def _param_override_defaults(document: ScenarioDocument) -> dict[str, Any]:
    """Return the declared ``scenario.param_overrides`` tree, node by node.

    Only searched and derived parameters are declared.  A pinned lanelet already reaches the
    runtime in the document itself, and declaring a key for every lanelet a
    primitive can name would fill the config with addresses nothing writes to.
    """
    tree: dict[str, Any] = {}
    for slot in document.lanelet_slots():
        if slot.field in ("spawn", "goal") or not (slot.searching or slot.deriving):
            continue
        tree.setdefault(slot.owner_id, {})[slot.field] = slot.lanelet_id
    return tree


def _map_overrides(document: ScenarioDocument) -> dict[str, Any]:
    """Return the ``map`` keys the document actually pins.

    Only non-empty values are emitted so that everything else still falls
    through to whichever built-in ``map`` group the run selects -- notably
    ``no_3d_model_lanelet_ids``, which sweep constraints reference through
    ``${map.no_3d_model_lanelet_ids}``.

    A document that names its own map by ``source`` or ``directory`` is the
    exception: a built-in group describes a *different* map, and inheriting
    that map's exclusion list would hand this scenario a set of lanelet ids
    that mean nothing on the map it actually runs on.  So the list is written
    out even when it is empty.  The ``map`` group is composed after the
    scenario, though, so these keys alone do not keep a built-in group's
    paths, name and list out of the run: that is what the document's own map
    group (:func:`build_map_group_config`) is for.
    """
    map_ref = document.map
    overrides: dict[str, Any] = {}
    if map_ref.name:
        overrides["name"] = map_ref.name
    if map_ref.source:
        overrides["source"] = map_ref.source
    if map_ref.directory:
        overrides["directory"] = map_ref.directory
    if map_ref.xodr_path:
        overrides["xodr_path"] = map_ref.xodr_path
    if map_ref.lanelet2_path:
        overrides["lanelet2_path"] = map_ref.lanelet2_path
    if map_ref.no_3d_model_lanelet_ids or map_ref.source or map_ref.directory:
        overrides["no_3d_model_lanelet_ids"] = list(map_ref.no_3d_model_lanelet_ids)
    return overrides


def own_map_group(document: ScenarioDocument) -> str | None:
    """Return the name of the map group an export of *document* ships, if any.

    Only a document that names its map by ``source`` or ``directory`` gets one
    (see the module docstring); one on a built-in group runs on that group and
    ships none.  The name is derived from the document's id -- by convention
    unique among installed scenario packages, though nothing enforces it: two
    packages shipping the same id collide on their ``conf/scenario`` files and
    on this group alike -- and its ``_map`` suffix keeps it apart from the
    built-in groups, named after the map they describe: a scenario called
    ``town10hd_opt`` must not shadow that group.
    """
    map_ref = document.map
    if not (map_ref.source or map_ref.directory):
        return None
    return f"{document.id}_map"


def run_map_group(document: ScenarioDocument) -> str:
    """Return the ``map=<group>`` a run of *document* selects."""
    return own_map_group(document) or document.map.group


def _map_value(map_config: Any, key: str) -> str | None:
    """``map.<key>`` of a composed ``map`` node (or a dict), stripped; ``None``
    when unset or empty -- as :func:`~...maps.config.resolve_map_paths` reads it."""
    if map_config is None:
        return None
    try:
        value = map_config.get(key)
    except AttributeError:
        return None
    if value is None:
        return None
    return str(value).strip() or None


def own_map_mismatch(document: ScenarioDocument, map_config: Any) -> str | None:
    """Say how the run's composed *map_config* is not *document*'s own map.

    ``None`` when it is, and for a document on a built-in group, which has no
    own map group (:func:`own_map_group`): the group it names is the map.

    The document's map group is composed after the scenario, so a run that
    leaves ``map=<own_map_group>`` out composes the default group instead, and
    that group's keys win over the ones the scenario's config writes.  Two
    things identify the map a run is on, and both are compared with the
    document:

    * ``map.name``, the CARLA world.  Every map group sets it, so a group
      describing another map -- the default one -- always differs here.
    * Where the map comes from: ``map.directory``, which wins over
      ``map.source`` (:func:`~...maps.config.resolve_map_paths`), else
      ``map.source``.  A directory the document does not name is accepted: no
      built-in group sets one, so it is the run's own local copy of the map
      (``map.directory=...``, as ``resolve_map_paths`` documents).  A source
      that is not the document's is another map, or another revision of it,
      whose lanelet ids are not the ones the document names; two spellings
      of one source are the same (:func:`_same_source`).

    ``map.lanelet2_path`` and ``map.xodr_path`` are not compared: an explicit
    file wins over the source by design, which is how a run points at a local
    or patched copy of the same map (``map.xodr_path=/tmp/patched.xodr``),
    and a group that sets them for another map sets that map's name as well.

    Returns:
        What differs, naming the keys, or ``None``.
    """
    if own_map_group(document) is None:
        return None
    map_ref = document.map
    want_name = (map_ref.name or "").strip() or None
    name = _map_value(map_config, "name")
    if want_name is not None and name != want_name:
        return f"map.name is {name!r}, not the document's {want_name!r}"
    want_directory = (map_ref.directory or "").strip() or None
    want_source = (map_ref.source or "").strip() or None
    directory = _map_value(map_config, "directory")
    source = _map_value(map_config, "source")
    if directory is not None:
        # The document's own, or the run's local copy of its map.
        return None
    if want_directory is not None:
        return (
            f"map.directory is unset (map.source {source!r}), not the "
            f"document's {want_directory!r}"
        )
    if not _same_source(source, want_source):
        return f"map.source is {source!r}, not the document's {want_source!r}"
    return None


def _same_source(source: str | None, want: str | None) -> bool:
    """Whether two ``map.source`` values name the same map: compared as
    :class:`~...maps.source.MapSource` parses them, so two spellings of one
    source (``git+...@<ref>#<path>``, a browser URL, a trailing slash) agree;
    as text when either does not parse."""
    if source is None or want is None:
        return source == want
    from ..maps.source import MapSource, MapSourceError  # noqa: PLC0415

    try:
        return MapSource.parse(source) == MapSource.parse(want)
    except MapSourceError:
        return source == want


def scenario_map_refusal(
    cfg: Any, documents: dict[str, ScenarioDocument | None] | None = None
) -> str | None:
    """Say why the composed run config *cfg* is refused for its map, or ``None``.

    The scenario's document is the one its package registered with its builder
    (:func:`~autoware_carla_scenario.registry.register_scenario_document`),
    so the scenario packages must be loaded first
    (:func:`~autoware_carla_scenario.registry.load_scenario_plugins`);
    the composed ``map`` is compared with it by :func:`own_map_mismatch`.  A
    scenario with no registered document, or one on a built-in group, is not
    refused; neither is one whose document cannot be loaded, which its builder
    refuses with its own message.  Nothing is resolved, fetched or read but
    the document.

    Args:
        cfg: The composed config (a ``DictConfig`` or a dict).
        documents: The documents already loaded, by scenario name: filled
            here, so a batch loads each scenario's once.

    Returns:
        The refusal, naming what differs and the ``map=<group>`` to run the
        scenario with, or ``None``.
    """
    from ..registry import get_scenario_document  # noqa: PLC0415

    scenario_name = (cfg.get("scenario") or {}).get("name")
    if not scenario_name:
        return None
    key = str(scenario_name)
    loaded = {} if documents is None else documents
    if key not in loaded:
        try:
            loaded[key] = get_scenario_document(key)
        except Exception:  # noqa: BLE001 - the builder refuses it, by name
            _logger.debug("scenario %s: its document did not load", key, exc_info=True)
            loaded[key] = None
    document = loaded[key]
    if document is None:
        return None
    mismatch = own_map_mismatch(document, cfg.get("map"))
    if mismatch is None:
        return None
    return (
        f"scenario {document.id!r} runs on the map its document names, but "
        f"{mismatch}.  Run it with map={run_map_group(document)}, the map "
        "group its package ships"
    )


def build_map_group_config(document: ScenarioDocument) -> dict[str, Any] | None:
    """Return the document's own ``map`` group as a plain dict, or ``None``.

    ``None`` for a document on a built-in group (:func:`own_map_group`).
    Otherwise every key a ``map`` group can set is written, the ones the
    document leaves empty as ``null`` (``overwrite_xodr`` as ``false``):
    selecting this group replaces the built-in one, and a key it left out would
    fall back to the base config, not to anything about this map.
    """
    if own_map_group(document) is None:
        return None
    map_ref = document.map
    return {
        "map": {
            "name": map_ref.name,
            "source": map_ref.source or None,
            "directory": map_ref.directory or None,
            "lanelet2_path": map_ref.lanelet2_path or None,
            "xodr_path": map_ref.xodr_path or None,
            "projector_type": None,
            # Never on by default: installing an OpenDRIVE replaces the road
            # network of the CARLA installation itself, an opt-in only the
            # Nishishinjuku group makes, for assets that ship without one.  A
            # document's xodr_path is what the run reads geometry from; a run
            # that does want it installed asks with map.overwrite_xodr=true.
            "overwrite_xodr": False,
            "no_3d_model_lanelet_ids": list(map_ref.no_3d_model_lanelet_ids),
        }
    }


def dump_map_group_config(document: ScenarioDocument) -> str | None:
    """Return the document's own ``map`` group as YAML text, or ``None``."""
    config = build_map_group_config(document)
    if config is None:
        return None
    return f"{PACKAGE_GLOBAL_HEADER}\n{dump_yaml(config)}"


def build_scenario_config(
    document: ScenarioDocument, *, document_path: str | None = None
) -> dict[str, Any]:
    """Return the Hydra scenario config for *document* as a plain dict.

    Args:
        document: The scenario to render.
        document_path: Value for ``scenario.document_path``.  Omitted when
            ``None``, which is the normal case: an exported package resolves
            the path from its own location instead of hard-coding it.
    """
    scenario: dict[str, Any] = {
        "name": document.id,
        "timeout_seconds": document.timeout_seconds,
    }
    if document_path is not None:
        scenario["document_path"] = document_path
    overrides = _entity_spawn_overrides(document)
    if overrides:
        scenario["spawn_overrides"] = overrides

    config: dict[str, Any] = {"scenario": scenario}

    map_overrides = _map_overrides(document)
    if map_overrides:
        config["map"] = map_overrides

    ego = document.ego
    if ego is not None:
        ego_overrides: dict[str, Any] = {
            "vehicle_type": ego.vehicle_type,
            "initial_speed_kmh": ego.initial_speed_kmh,
            "spawn_lanelet_id": ego.spawn.lanelet_id,
            "spawn_s": ego.spawn.s.value,
            # Across the lane and turned against it, as an NPC's spawn is: the
            # runner builds the ego's pose from these keys alone.
            "spawn_t": ego.spawn.t,
            "spawn_heading": ego.spawn.heading,
            "entity": ego.driven_by,
        }
        if ego.goal is not None:
            ego_overrides["goal_lanelet_id"] = ego.goal.lanelet_id
            ego_overrides["goal_s"] = ego.goal.s
        config["ego"] = ego_overrides

    # Every searched parameter gets its key declared, whether or not it is the
    # one this run sweeps: a key Hydra's struct mode has never seen is a key an
    # override cannot set, and which slot the sweeper drives is decided per run.
    param_defaults = _param_override_defaults(document)
    if param_defaults:
        scenario["param_overrides"] = param_defaults

    if document.route is not None:
        # A logical scenario: the route search drives the expansion, and the
        # match it picks reaches the run through these keys.
        scenario["route"] = {
            key: list(value) if isinstance(value, list) else value
            for key, value in ROUTE_CONFIG_DEFAULTS.items()
        }
        config["sweep"] = {"route": document.route.to_sweep_dict()}
        return config

    slot = swept_slot(document)
    if slot is not None:
        target_key = slot_lanelet_key(document, slot)
        assert target_key is not None  # noqa: S101 -- swept_slot only returns keyed slots
        sweep: dict[str, Any] = {
            "constraints": {target_key: slot.choice.sweep_constraint_dicts()}
        }
        # The offset binding belongs to the spawn whose lanelet is being
        # enumerated: it derives "15 m before *this* lanelet's stop line", so it
        # says nothing when the sweep is driving some other slot.
        target = document.entity(slot.owner_id) if slot.field == "spawn" else None
        bindings: dict[str, Any] = {}
        if (
            target is not None
            and target.spawn.s.mode == "derived"
            and target.spawn.s.binding is not None
        ):
            bindings[spawn_s_key(target)] = target.spawn.s.binding.to_sweep_dict()
        # Every lanelet derived from the pick is a binding onto its own key:
        # the sweeper resolves each one against the lanelet it picked.
        for derived in document.derived_lanelet_slots():
            key = slot_lanelet_key(document, derived)
            if key is not None and derived.choice.binding is not None:
                bindings[key] = derived.choice.binding.to_sweep_dict()
        if bindings:
            sweep["bindings"] = bindings
        config["sweep"] = sweep

    return config


def dump_scenario_config(
    document: ScenarioDocument, *, document_path: str | None = None
) -> str:
    """Return the Hydra scenario config for *document* as YAML text."""
    body = dump_yaml(build_scenario_config(document, document_path=document_path))
    return f"{PACKAGE_GLOBAL_HEADER}\n{body}"
