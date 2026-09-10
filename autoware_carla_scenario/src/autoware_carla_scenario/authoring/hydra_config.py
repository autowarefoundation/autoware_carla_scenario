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
overrides today.  Other entities get a declared ``scenario.spawn_overrides``
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
"""

from __future__ import annotations

from typing import Any

from .models import Entity, LaneletSlot, ScenarioDocument
from .persistence import dump_yaml

__all__ = [
    "PACKAGE_GLOBAL_HEADER",
    "build_scenario_config",
    "dump_scenario_config",
    "param_override_key",
    "slot_lanelet_key",
    "spawn_lanelet_key",
    "spawn_s_key",
    "swept_entity",
    "swept_slot",
]

#: Hydra needs this on the first line for a config that writes into the root.
PACKAGE_GLOBAL_HEADER = "# @package _global_"


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

    Only searched parameters are declared.  A pinned lanelet already reaches the
    runtime in the document itself, and declaring a key for every lanelet a
    primitive can name would fill the config with addresses nothing writes to.
    """
    tree: dict[str, Any] = {}
    for slot in document.lanelet_slots():
        if slot.field in ("spawn", "goal") or not slot.searching:
            continue
        tree.setdefault(slot.owner_id, {})[slot.field] = slot.lanelet_id
    return tree


def _map_overrides(document: ScenarioDocument) -> dict[str, Any]:
    """Return the ``map`` keys the document actually pins.

    Only non-empty values are emitted so that everything else still falls
    through to whichever built-in ``map`` group the run selects -- notably
    ``no_3d_model_lanelet_ids``, which sweep constraints reference through
    ``${map.no_3d_model_lanelet_ids}``.

    A document that names its own map by ``source`` is the exception: the group
    a run selects then describes a *different* map, and inheriting that map's
    exclusion list would hand this scenario a set of lanelet ids that mean
    nothing on the map it actually runs on.  So the list is written out even
    when it is empty, which is what stops it being inherited.
    """
    map_ref = document.map
    overrides: dict[str, Any] = {}
    if map_ref.name:
        overrides["name"] = map_ref.name
    if map_ref.source:
        overrides["source"] = map_ref.source
    if map_ref.xodr_path:
        overrides["xodr_path"] = map_ref.xodr_path
    if map_ref.lanelet2_path:
        overrides["lanelet2_path"] = map_ref.lanelet2_path
    if map_ref.no_3d_model_lanelet_ids or map_ref.source:
        overrides["no_3d_model_lanelet_ids"] = list(map_ref.no_3d_model_lanelet_ids)
    return overrides


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
        if (
            target is not None
            and target.spawn.s.mode == "derived"
            and target.spawn.s.binding is not None
        ):
            sweep["bindings"] = {
                spawn_s_key(target): target.spawn.s.binding.to_sweep_dict()
            }
        config["sweep"] = sweep

    return config


def dump_scenario_config(
    document: ScenarioDocument, *, document_path: str | None = None
) -> str:
    """Return the Hydra scenario config for *document* as YAML text."""
    body = dump_yaml(build_scenario_config(document, document_path=document_path))
    return f"{PACKAGE_GLOBAL_HEADER}\n{body}"
