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
"""

from __future__ import annotations

from typing import Any

from .models import Entity, ScenarioDocument
from .persistence import dump_yaml

__all__ = [
    "PACKAGE_GLOBAL_HEADER",
    "build_scenario_config",
    "dump_scenario_config",
    "spawn_lanelet_key",
    "spawn_s_key",
    "swept_entity",
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


def swept_entity(document: ScenarioDocument) -> Entity | None:
    """Return the entity whose spawn drives the sweep, if any.

    The lanelet-constraint sweeper enumerates a single target key per run, so
    at most one entity can spawn by constraint search in a given scenario; the
    first such entity wins and :mod:`.validator` flags the rest.
    """
    return next(
        (e for e in document.entities if e.spawn.mode == "constraint_search"), None
    )


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


def _map_overrides(document: ScenarioDocument) -> dict[str, Any]:
    """Return the ``map`` keys the document actually pins.

    Only non-empty values are emitted so that everything else still falls
    through to whichever built-in ``map`` group the run selects -- notably
    ``no_3d_model_lanelet_ids``, which sweep constraints reference through
    ``${map.no_3d_model_lanelet_ids}``.
    """
    map_ref = document.map
    overrides: dict[str, Any] = {}
    if map_ref.name:
        overrides["name"] = map_ref.name
    if map_ref.xodr_path:
        overrides["xodr_path"] = map_ref.xodr_path
    if map_ref.lanelet2_path:
        overrides["lanelet2_path"] = map_ref.lanelet2_path
    if map_ref.no_3d_model_lanelet_ids:
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
        config["ego"] = {
            "vehicle_type": ego.vehicle_type,
            "initial_speed_kmh": ego.initial_speed_kmh,
            "spawn_lanelet_id": ego.spawn.lanelet_id,
            "spawn_s": ego.spawn.s.value,
        }

    target = swept_entity(document)
    if target is not None:
        sweep: dict[str, Any] = {
            "constraints": {
                spawn_lanelet_key(target): target.spawn.sweep_constraint_dicts()
            }
        }
        if target.spawn.s.mode == "derived" and target.spawn.s.binding is not None:
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
