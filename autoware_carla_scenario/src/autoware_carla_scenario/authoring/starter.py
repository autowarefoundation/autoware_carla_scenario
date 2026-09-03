"""Starter documents -- what "New scenario" gives you.

A blank canvas is a bad first experience for a declarative editor: nothing on
screen explains what a swimlane, a trigger condition or a spawn constraint is.
:func:`new_document` therefore returns a small but *complete and valid* cut-in
scenario -- an ego, one NPC, a triggered lane change, and PASS/FAIL assertions --
that a user can run, then edit into whatever they actually wanted.
"""

from __future__ import annotations

from .models import (
    ActionNode,
    Assertions,
    BindingRef,
    ConditionNode,
    ConstraintNode,
    Entity,
    MapRef,
    ScenarioDocument,
    SpawnSpec,
    SValue,
)
from .registry import default_params, get_action_spec, get_condition_spec

__all__ = ["blank_document", "new_document"]

#: Defaults for the built-in Nishishinjuku map, matching ``conf/map/nishishinjuku.yaml``.
_DEFAULT_MAP = MapRef(
    group="nishishinjuku",
    name="NishishinjukuMap",
    xodr_path=("autoware_lanelet2_to_opendrive/test/data/nishishinjuku_carla.xodr"),
    lanelet2_path="autoware_lanelet2_to_opendrive/test/data/nishishinjuku.osm",
)


def _condition(type_id: str, **params: object) -> ConditionNode:
    """Return a condition node seeded with its spec defaults plus *params*."""
    spec = get_condition_spec(type_id)
    assert spec is not None  # noqa: S101 -- built-in types are always registered
    merged = default_params(spec.fields)
    merged.update(params)
    return ConditionNode(type=type_id, params=merged)


def _action(type_id: str, actor: str, title: str, **params: object) -> ActionNode:
    """Return an action node seeded with its spec defaults plus *params*."""
    spec = get_action_spec(type_id)
    assert spec is not None  # noqa: S101 -- built-in types are always registered
    merged = default_params(spec.fields)
    merged.update(params)
    return ActionNode(
        type=type_id,
        title=title,
        actor=actor,
        params=merged,
        timing=spec.default_timing,
    )


def blank_document(
    scenario_id: str = "new_scenario", title: str = ""
) -> ScenarioDocument:
    """Return the smallest document that still validates: an ego and a timeout."""
    document = ScenarioDocument(
        id=scenario_id,
        title=title or "New scenario",
        map=_DEFAULT_MAP.model_copy(deep=True),
        entities=[
            Entity(
                id="ego",
                kind="ego",
                title="Ego",
                spawn=SpawnSpec(mode="fixed", lanelet_id=183, s=SValue(value=0.0)),
            )
        ],
        assertions=Assertions(
            **{
                "pass": [_condition("elapsed_time", duration_seconds=10.0)],
                "fail": [_condition("collision")],
            }
        ),
    )
    document.sync_layout()
    return document


def new_document(
    scenario_id: str = "cut_in", title: str = "Cut in"
) -> ScenarioDocument:
    """Return a runnable cut-in scenario to start editing from.

    The NPC spawns by constraint search (a lane with a left neighbour, long
    enough, not in a junction, not on the map's exclusion list) with its
    longitudinal offset derived from the nearest stop line -- i.e. every spawn
    feature the editor offers is already switched on and visible.
    """
    ego = Entity(
        id="ego",
        kind="ego",
        title="Ego",
        initial_speed_kmh=10.0,
        spawn=SpawnSpec(mode="fixed", lanelet_id=183, s=SValue(value=0.0)),
    )
    npc = Entity(
        id="npc1",
        kind="vehicle",
        title="NPC1",
        initial_speed_kmh=25.0,
        spawn=SpawnSpec(
            mode="constraint_search",
            lanelet_id=183,
            s=SValue(
                mode="derived",
                value=10.0,
                binding=BindingRef(type="stop_line_offset", params={"offset": 15.0}),
            ),
            constraints=[
                ConstraintNode(
                    type="and",
                    constraints=[
                        ConstraintNode(type="has_adjacent", params={"value": "left"}),
                        ConstraintNode(
                            type="lanelet_length",
                            params={"rule": "greater_than_or_equal", "value": 10.0},
                        ),
                        ConstraintNode(
                            type="not",
                            constraints=[ConstraintNode(type="is_junction")],
                        ),
                        ConstraintNode(
                            type="not",
                            constraints=[
                                ConstraintNode(
                                    type="in_set",
                                    params={"values": "${map.no_3d_model_lanelet_ids}"},
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
    )

    lane_change = _action("lane_change", actor="npc1", title="Cut in", direction="left")
    lane_change.trigger = ConditionNode(
        type="all",
        children=[
            _condition(
                "entity_distance",
                source="npc1",
                target="ego",
                rule="less_than",
                distance=20.0,
            ),
            _condition(
                "ttc", source="npc1", target="ego", rule="less_than", seconds=4.0
            ),
        ],
    )

    document = ScenarioDocument(
        id=scenario_id,
        title=title,
        description="NPC1 cuts into the ego lane once it is close and closing.",
        timeout_seconds=30.0,
        map=_DEFAULT_MAP.model_copy(deep=True),
        entities=[ego, npc],
        actions=[lane_change],
        assertions=Assertions(
            **{
                "pass": [
                    ConditionNode(
                        type="sticky",
                        children=[
                            _condition(
                                "entity_lane_position",
                                entity="npc1",
                                lanelet_id=183,
                                lane_id=None,
                            )
                        ],
                    )
                ],
                "fail": [
                    _condition("collision"),
                    _condition("timeout", timeout_seconds=30.0),
                ],
            }
        ),
    )
    document.sync_layout()
    return document
