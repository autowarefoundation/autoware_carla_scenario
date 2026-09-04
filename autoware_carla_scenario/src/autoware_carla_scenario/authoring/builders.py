"""Factories that turn a compiled plan into the framework's runtime objects.

Every function here returns an object that already existed in
``autoware_carla_scenario`` -- ``LaneChangeAction``, ``AndCondition``,
``StickyCondition`` and friends.  The editor contributes no runtime of its own;
it only decides which of these to build and with what arguments.

Module-level imports are kept free of CARLA and lanelet2 so that
:mod:`autoware_carla_scenario.authoring` can be imported in the editor process.
The heavy imports happen inside the builders, which only run inside
:meth:`~autoware_carla_scenario.declarative.DeclarativeScenario.setup` -- i.e.
against a live CARLA world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..actions import BaseAction
    from ..conditions import BaseCondition
    from .compiler import BuildContext, CompiledAction, CompiledCondition

__all__ = [
    "instantiate_action",
    "instantiate_condition",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rule(name: Any) -> Any:
    """Return the :class:`ComparisonRule` member named *name*."""
    from ..conditions.comparison import ComparisonRule  # noqa: PLC0415

    return ComparisonRule[str(name).upper()]


def _traffic_light_state(name: Any) -> Any:
    """Return the ``carla.TrafficLightState`` member named *name*."""
    import carla  # noqa: PLC0415

    return getattr(carla.TrafficLightState, str(name))


def _opendrive_lane_for_lanelet(lanelet_id: int) -> tuple[str, int]:
    """Return the OpenDRIVE ``(road_id, lane_id)`` a lanelet occupies.

    Scenario authors think in lanelets; ``EntityLanePositionCondition`` speaks
    OpenDRIVE.  Both halves are carried because **an OpenDRIVE road is not a
    lane**: on the nishishinjuku fixture, lanelets 183 and 184 are lanes 2 and 1
    of the same road 80.  Resolving only the road would turn "the entity is on
    lanelet 183" into "the entity is anywhere on road 80", which is also true
    while it sits in the neighbouring lane -- so a cut-in scenario would pass
    without the cut-in ever happening.
    """
    from ..coordinate import Lanelet2Pose, to_opendrive  # noqa: PLC0415

    pose = to_opendrive(Lanelet2Pose(lanelet_id=int(lanelet_id), s=0.0))
    return pose.road_id, int(pose.lane_id)


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------


def build_and_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`AndCondition` over *children*."""
    from ..conditions import AndCondition  # noqa: PLC0415

    return AndCondition(children, label=compiled.label)


def build_or_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`OrCondition` over *children*."""
    from ..conditions import OrCondition  # noqa: PLC0415

    return OrCondition(children, label=compiled.label)


def build_not_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`NotCondition` around the single child."""
    from ..conditions import NotCondition  # noqa: PLC0415

    return NotCondition(children[0], label=compiled.label)


def build_sticky_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`StickyCondition` around the single child."""
    from ..conditions import StickyCondition  # noqa: PLC0415

    return StickyCondition(children[0], label=compiled.label)


def build_persistent_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`PersistentCondition` around the single child."""
    from ..conditions import PersistentCondition  # noqa: PLC0415

    return PersistentCondition(
        children[0],
        duration=float(compiled.params["duration"]),
        label=compiled.label,
    )


def build_entity_distance_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityDistanceCondition`."""
    from ..conditions import EntityDistanceCondition  # noqa: PLC0415

    params = compiled.params
    return EntityDistanceCondition(
        source=params["source"],
        target=params["target"],
        value=float(params["distance"]),
        rule=_rule(params["rule"]),
        label=compiled.label,
    )


def build_ttc_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeToCollisionCondition`."""
    from ..conditions import TimeToCollisionCondition  # noqa: PLC0415

    params = compiled.params
    return TimeToCollisionCondition(
        source=params["source"],
        target=params["target"],
        value=float(params["seconds"]),
        rule=_rule(params["rule"]),
        label=compiled.label,
    )


def build_speed_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`SpeedCondition`."""
    from ..conditions import SpeedCondition, SpeedDirection  # noqa: PLC0415

    params = compiled.params
    return SpeedCondition(
        entity_name=params["entity"],
        value=float(params["value"]),
        rule=_rule(params["rule"]),
        direction=SpeedDirection[str(params.get("direction") or "MAGNITUDE")],
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
        entity_name=params["entity"],
        duration=float(params["duration"]),
        speed_threshold=float(params.get("speed_threshold") or 0.1),
        label=compiled.label,
    )


def build_entity_lane_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition` from a Lanelet2 reference.

    A lanelet names one lane, so both halves of the OpenDRIVE address are
    derived from it and neither is left for the author to contradict.
    """
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415

    params = compiled.params
    road_id, lane_id = _opendrive_lane_for_lanelet(int(params["lanelet_id"]))
    return EntityLanePositionCondition(
        entity_name=params["entity"],
        road_id=road_id,
        lane_id=lane_id,
        label=compiled.label,
    )


def build_entity_road_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition` from an OpenDRIVE address.

    Road and lane are passed through untouched -- this is the frame the runtime
    already speaks, so nothing is resolved and nothing can disagree.
    """
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415

    params = compiled.params
    lane_id = params.get("lane_id")
    return EntityLanePositionCondition(
        entity_name=params["entity"],
        road_id=str(params["road_id"]),
        lane_id=int(lane_id) if lane_id is not None else None,
        label=compiled.label,
    )


def build_entity_existence_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityExistenceCondition`."""
    from ..conditions import EntityExistenceCondition  # noqa: PLC0415

    return EntityExistenceCondition(
        entity_name=compiled.params["entity"], label=compiled.label
    )


def build_waypoint_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`WaypointCondition`."""
    from ..conditions import WaypointCheckType, WaypointCondition  # noqa: PLC0415

    params = compiled.params
    return WaypointCondition(
        entity_name=params["entity"],
        distance=float(params["distance"]),
        check_type=WaypointCheckType.IS_EMPTY,
        label=compiled.label,
    )


def build_elapsed_time_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`ElapsedTimeCondition`."""
    from ..conditions import ElapsedTimeCondition  # noqa: PLC0415

    params = compiled.params
    return ElapsedTimeCondition(
        duration_seconds=float(params["duration_seconds"]),
        rule=_rule(params["rule"]),
        label=compiled.label,
    )


def build_timeout_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeoutCondition`."""
    from ..conditions import TimeoutCondition  # noqa: PLC0415

    return TimeoutCondition(
        float(compiled.params["timeout_seconds"]), label=compiled.label
    )


def build_collision_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`CollisionCondition`."""
    from ..conditions import CollisionCondition  # noqa: PLC0415

    return CollisionCondition(
        min_impulse=float(compiled.params.get("min_impulse") or 0.0),
        label=compiled.label,
    )


def build_traffic_signal_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TrafficSignalCondition`."""
    from ..conditions import TrafficSignalCondition  # noqa: PLC0415

    params = compiled.params
    return TrafficSignalCondition(
        lanelet2_regulatory_element_id=int(params["lanelet2_regulatory_element_id"]),
        expected_state=_traffic_light_state(params["state"]),
        label=compiled.label,
    )


def build_action_state_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`ActionStateCondition` watching a named action."""
    from ..conditions import ActionStateCondition  # noqa: PLC0415

    return ActionStateCondition(
        action_id=str(compiled.params["action"]),
        state=str(compiled.params["state"]),
        # The live mapping, not a copy: the action being watched may not have
        # been instantiated yet when this condition is built.
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

    return AlwaysTrueCondition(label=compiled.label)


# ---------------------------------------------------------------------------
# Action builders
# ---------------------------------------------------------------------------


def build_lane_change_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`LaneChangeAction`."""
    from ..actions import LaneChangeAction, LaneChangeDirection  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    return LaneChangeAction(
        entity_name=compiled.actor_role,
        direction=LaneChangeDirection(str(compiled.params["direction"])),
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
    from ..actions import TurnAction, TurnDirection  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    return TurnAction(
        entity_name=compiled.actor_role,
        direction=TurnDirection(str(compiled.params["direction"])),
        client=ctx.client,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        search_distance=float(compiled.params.get("search_distance") or 200.0),
        tm_port=ctx.tm_port,
    )


def build_traffic_signal_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficSignalAction`."""
    from ..actions import TrafficLightTarget, TrafficSignalAction  # noqa: PLC0415

    params = compiled.params
    if str(params.get("target")) == "all":
        target: Any = TrafficLightTarget.ALL
    else:
        target = [int(v) for v in params.get("lanelet2_traffic_light_ids") or []]
    return TrafficSignalAction(
        state=_traffic_light_state(params["state"]),
        lanelet2_traffic_light_ids=target,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        freeze=bool(params.get("freeze", True)),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _resolve(builder_name: str) -> Callable[..., Any]:
    """Return the builder function a spec names.

    Raises:
        LookupError: If the spec names a builder this module does not define,
            which means a primitive was registered without one.
    """
    builder = globals().get(builder_name)
    if builder is None or not callable(builder):
        raise LookupError(
            f"No builder named {builder_name!r} in "
            f"autoware_carla_scenario.authoring.builders."
        )
    return builder


def instantiate_condition(
    compiled: "CompiledCondition", ctx: "BuildContext"
) -> "BaseCondition":
    """Recursively build the runtime condition tree for *compiled*."""
    children = [instantiate_condition(c, ctx) for c in compiled.children]
    return _resolve(compiled.spec.builder)(compiled, children, ctx)


def instantiate_action(compiled: "CompiledAction", ctx: "BuildContext") -> "BaseAction":
    """Build the runtime action for *compiled*, including its trigger."""
    from ..actions import TickTiming  # noqa: PLC0415

    condition = (
        instantiate_condition(compiled.trigger, ctx)
        if compiled.trigger is not None
        else None
    )
    timing = TickTiming(compiled.node.timing)
    return _resolve(compiled.spec.builder)(compiled, condition, timing, ctx)
