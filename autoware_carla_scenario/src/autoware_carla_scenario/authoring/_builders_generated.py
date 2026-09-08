"""Builders derived from the runtime constructors -- GENERATED, DO NOT EDIT.

Every function here is rendered from the ``target`` of a spec in
:mod:`autoware_carla_scenario.authoring.registry` and the signature of the
runtime class it names.  Editing this file by hand is pointless: the next
regeneration overwrites it, and CI fails when the two disagree.

Regenerate with::

    uv run python -m autoware_carla_scenario.authoring.codegen

Builders that are *not* a plain constructor call -- the ones that assemble a
pose, or pick between two constructors -- stay hand-written in
:mod:`autoware_carla_scenario.authoring.builders`.

Module-level imports stay free of CARLA and lanelet2, exactly as in the
hand-written module: the editor process imports this package without a
simulator, so every runtime import happens inside a builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..actions import BaseAction
    from ..conditions import BaseCondition
    from .compiler import BuildContext, CompiledAction, CompiledCondition

__all__ = [
    "build_and_condition",
    "build_or_condition",
    "build_not_condition",
    "build_persistent_condition",
    "build_sticky_condition",
    "build_entity_existence_condition",
    "build_waypoint_condition",
    "build_speed_condition",
    "build_standstill_condition",
    "build_entity_distance_condition",
    "build_ttc_condition",
    "build_action_state_condition",
    "build_always_true_condition",
    "build_collision_condition",
    "build_elapsed_time_condition",
    "build_timeout_condition",
    "build_traffic_signal_condition",
    "build_traffic_signal_action",
    "build_lane_change_action",
    "build_turn_action",
]


def build_and_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`AndCondition`."""
    from ..conditions import AndCondition  # noqa: PLC0415

    return AndCondition(
        conditions=children,
        label=compiled.label,
    )


def build_or_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`OrCondition`."""
    from ..conditions import OrCondition  # noqa: PLC0415

    return OrCondition(
        conditions=children,
        label=compiled.label,
    )


def build_not_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`NotCondition`."""
    from ..conditions import NotCondition  # noqa: PLC0415

    return NotCondition(
        condition=children[0],
        label=compiled.label,
    )


def build_persistent_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`PersistentCondition`."""
    from ..conditions import PersistentCondition  # noqa: PLC0415

    params = compiled.params
    return PersistentCondition(
        condition=children[0],
        duration=params["duration"],
        label=compiled.label,
    )


def build_sticky_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`StickyCondition`."""
    from ..conditions import StickyCondition  # noqa: PLC0415

    return StickyCondition(
        condition=children[0],
        label=compiled.label,
    )


def build_entity_existence_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityExistenceCondition`."""
    from ..conditions import EntityExistenceCondition  # noqa: PLC0415

    params = compiled.params
    return EntityExistenceCondition(
        entity_name=str(params["entity"]),
        label=compiled.label,
    )


def build_waypoint_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`WaypointCondition`."""
    from ..conditions import WaypointCondition  # noqa: PLC0415

    params = compiled.params
    return WaypointCondition(
        entity_name=str(params["entity"]),
        distance=params["distance"],
        label=compiled.label,
    )


def build_speed_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`SpeedCondition`."""
    from ..conditions import SpeedCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415
    from ..conditions import SpeedDirection  # noqa: PLC0415

    params = compiled.params
    return SpeedCondition(
        entity_name=str(params["entity"]),
        value=params["value"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        direction=SpeedDirection[str(params["direction"])],
        label=compiled.label,
    )


def build_standstill_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`StandstillCondition`."""
    from ..conditions import StandstillCondition  # noqa: PLC0415

    params = compiled.params
    return StandstillCondition(
        entity_name=str(params["entity"]),
        duration=params["duration"],
        speed_threshold=params["speed_threshold"],
        label=compiled.label,
    )


def build_entity_distance_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityDistanceCondition`."""
    from ..conditions import EntityDistanceCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    return EntityDistanceCondition(
        source=str(params["source"]),
        target=str(params["target"]),
        value=params["distance"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        label=compiled.label,
    )


def build_ttc_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeToCollisionCondition`."""
    from ..conditions import TimeToCollisionCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    return TimeToCollisionCondition(
        source=str(params["source"]),
        target=str(params["target"]),
        value=params["seconds"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        label=compiled.label,
    )


def build_action_state_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`ActionStateCondition`."""
    from ..conditions import ActionStateCondition  # noqa: PLC0415

    params = compiled.params
    return ActionStateCondition(
        action_id=str(params["action"]),
        state=str(params["state"]),
        actions=ctx.actions,
        label=compiled.label,
    )


def build_always_true_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`AlwaysTrueCondition`."""
    from ..conditions import AlwaysTrueCondition  # noqa: PLC0415

    return AlwaysTrueCondition(
        label=compiled.label,
    )


def build_collision_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`CollisionCondition`."""
    from ..conditions import CollisionCondition  # noqa: PLC0415

    params = compiled.params
    return CollisionCondition(
        min_impulse=params["min_impulse"],
        label=compiled.label,
    )


def build_elapsed_time_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`ElapsedTimeCondition`."""
    from ..conditions import ElapsedTimeCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    return ElapsedTimeCondition(
        duration_seconds=params["duration_seconds"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        label=compiled.label,
    )


def build_timeout_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeoutCondition`."""
    from ..conditions import TimeoutCondition  # noqa: PLC0415

    params = compiled.params
    return TimeoutCondition(
        timeout_seconds=params["timeout_seconds"],
        label=compiled.label,
    )


def build_traffic_signal_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TrafficSignalCondition`."""
    from ..conditions import TrafficSignalCondition  # noqa: PLC0415
    import carla  # noqa: PLC0415

    params = compiled.params
    return TrafficSignalCondition(
        lanelet2_regulatory_element_id=params["lanelet2_regulatory_element_id"],
        expected_state=getattr(carla.TrafficLightState, str(params["state"])),
        label=compiled.label,
    )


def build_traffic_signal_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficSignalAction`."""
    from ..actions import TrafficSignalAction  # noqa: PLC0415
    from ..actions import TrafficLightTarget  # noqa: PLC0415
    import carla  # noqa: PLC0415

    params = compiled.params
    lanelet2_traffic_light_ids: Any
    if params["target"] == "all":
        lanelet2_traffic_light_ids = TrafficLightTarget.ALL
    elif params["target"] == "ids":
        lanelet2_traffic_light_ids = params["lanelet2_traffic_light_ids"]
    else:  # pragma: no cover -- the generator checks the cases are exhaustive
        raise ValueError(f"traffic_signal: unknown target {params['target']!r}")
    return TrafficSignalAction(
        state=getattr(carla.TrafficLightState, str(params["state"])),
        lanelet2_traffic_light_ids=lanelet2_traffic_light_ids,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        freeze=params["freeze"],
    )


def build_lane_change_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`LaneChangeAction`."""
    from ..actions import LaneChangeAction  # noqa: PLC0415
    from ..actions import LaneChangeDirection  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return LaneChangeAction(
        entity_name=compiled.actor_role,
        direction=LaneChangeDirection[str(params["direction"]).upper()],
        client=ctx.client,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        tm_port=ctx.tm_port,
    )


def build_turn_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TurnAction`."""
    from ..actions import TurnAction  # noqa: PLC0415
    from ..actions import TurnDirection  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return TurnAction(
        entity_name=compiled.actor_role,
        direction=TurnDirection[str(params["direction"]).upper()],
        client=ctx.client,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        search_distance=params["search_distance"],
        tm_port=ctx.tm_port,
    )
