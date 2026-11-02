"""Builders derived from the runtime constructors -- GENERATED, DO NOT EDIT.

Every function here is rendered from the ``target`` of a spec in
:mod:`autoware_carla_scenario.authoring.registry` and the signature of the
runtime class it names.  Editing this file by hand is pointless: the next
regeneration overwrites it, and CI fails when the two disagree.

Regenerate with::

    uv run python -m autoware_carla_scenario.authoring.codegen

An argument the signature alone cannot describe -- a pose assembled from two
fields and a literal, a list of comparison rules built from four bounds -- is
described by the spec's ``builds`` and assembled in the prelude below, so that
these builders are generated and checked like every other one.

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
    "ONCE_ACTIONS",
    "OVERRIDING_ACTIONS",
    "build_and_condition",
    "build_or_condition",
    "build_not_condition",
    "build_persistent_condition",
    "build_sticky_condition",
    "build_acceleration_condition",
    "build_entity_existence_condition",
    "build_lane_change_settled_condition",
    "build_waypoint_condition",
    "build_entity_lane_of_condition",
    "build_off_road_fraction_condition",
    "build_entity_lane_position_condition",
    "build_entity_road_position_condition",
    "build_route_progress_condition",
    "build_speed_condition",
    "build_standstill_condition",
    "build_temporary_stop_condition",
    "build_temporary_stop_at_stop_lines_condition",
    "build_turn_indicator_condition",
    "build_adversity_active_condition",
    "build_entity_distance_condition",
    "build_distance_condition",
    "build_relative_speed_condition",
    "build_entity_count_condition",
    "build_ttc_condition",
    "build_ttc_to_position_condition",
    "build_time_headway_condition",
    "build_action_state_condition",
    "build_always_true_condition",
    "build_collision_condition",
    "build_elapsed_time_condition",
    "build_traffic_signal_controller_condition",
    "build_metric_condition",
    "build_place_clear_condition",
    "build_timeout_condition",
    "build_traffic_signal_condition",
    "build_trajectory_time_condition",
    "build_adversarial_brake_action",
    "build_adversarial_cut_in_action",
    "build_adversarial_merge_action",
    "build_adversarial_traffic_action",
    "build_construction_zone_action",
    "build_head_on_action",
    "build_ignore_leader_action",
    "build_jaywalk_action",
    "build_lane_change_abort_action",
    "build_pedestrian_run_red_light_action",
    "build_run_traffic_rule_action",
    "build_stalled_object_action",
    "build_stop_crossing_action",
    "build_urban_construction_zone_action",
    "build_traffic_signal_controller_action",
    "build_traffic_signal_action",
    "build_environment_action",
    "build_traffic_sink_action",
    "build_traffic_source_action",
    "build_play_motion_action",
    "build_smart_pedestrian_action",
    "build_walk_straight_action",
    "build_traffic_parameters_action",
    "build_communication_loss_action",
    "build_turn_indicator_action",
    "build_despawn_action",
    "build_lane_change_action",
    "build_routing_action",
    "build_set_speed_action",
    "build_spawn_action",
    "build_turn_action",
    "build_vehicle_parameters_action",
]

#: The action types whose constructor takes ``overrides``: those that command
#: an entity's control domains, whose card offers *Override a running action*.
OVERRIDING_ACTIONS: frozenset[str] = frozenset(
    {
        "adversarial_brake",
        "adversarial_cut_in",
        "adversarial_merge",
        "head_on",
        "ignore_leader",
        "jaywalk",
        "lane_change_abort",
        "pedestrian_run_red_light",
        "run_traffic_rule",
        "stalled_object",
        "stop_crossing",
        "play_motion",
        "smart_pedestrian",
        "walk_straight",
        "turn_indicator",
        "lane_change",
        "routing",
        "set_speed",
        "turn",
    }
)

#: The action types whose constructor takes ``once``: those that can fire more
#: than once, whose card offers *Fire only once*.  Any other fires once.
ONCE_ACTIONS: frozenset[str] = frozenset(
    {
        "adversarial_brake",
        "adversarial_cut_in",
        "adversarial_merge",
        "head_on",
        "ignore_leader",
        "jaywalk",
        "lane_change_abort",
        "pedestrian_run_red_light",
        "run_traffic_rule",
        "stop_crossing",
        "traffic_signal_controller",
        "traffic_signal",
        "environment",
        "traffic_sink",
        "traffic_source",
        "play_motion",
        "smart_pedestrian",
        "walk_straight",
        "traffic_parameters",
        "communication_loss",
        "turn_indicator",
        "despawn",
        "lane_change",
        "routing",
        "set_speed",
        "spawn",
        "turn",
        "vehicle_parameters",
    }
)


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

    params = compiled.params
    return NotCondition(
        condition=children[0],
        label=compiled.label,
        holds_when_pending=params["holds_when_pending"],
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


def build_acceleration_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`AccelerationCondition`."""
    from ..conditions import AccelerationCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415
    from ..conditions import AccelerationDirection  # noqa: PLC0415

    params = compiled.params
    return AccelerationCondition(
        entity_name=str(params["entity"]),
        value=params["value"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        direction=AccelerationDirection[str(params["direction"])],
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


def build_lane_change_settled_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`LaneChangeSettledCondition`."""
    from ..conditions import LaneChangeSettledCondition  # noqa: PLC0415

    params = compiled.params
    return LaneChangeSettledCondition(
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


def build_entity_lane_of_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLaneOfCondition`."""
    from ..conditions import EntityLaneOfCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    params = compiled.params
    position = Lanelet2Pose(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
    )
    return EntityLaneOfCondition(
        entity_name=str(params["entity"]),
        position=position,
        lane=str(params["lane"]),
        label=compiled.label,
    )


def build_off_road_fraction_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`OffRoadFractionCondition`."""
    from ..conditions import OffRoadFractionCondition  # noqa: PLC0415

    params = compiled.params
    return OffRoadFractionCondition(
        fraction=params["fraction"],
        entity_name=(str(params["entity"]) if params["entity"] is not None else None),
        label=compiled.label,
    )


def build_entity_lane_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition`."""
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415
    from ..conditions.comparison import ScalarComparisonRule  # noqa: PLC0415
    from ..conditions.comparison import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    position = Lanelet2Pose(
        s=0.0,
        lanelet_id=params["lanelet_id"],
    )
    rules: list[Any] = []
    if params.get("s_min") is not None:
        rules.append(
            ScalarComparisonRule(
                field="s",
                rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                value=params["s_min"],
            )
        )
    if params.get("s_max") is not None:
        rules.append(
            ScalarComparisonRule(
                field="s",
                rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                value=params["s_max"],
            )
        )
    if params.get("t_min") is not None:
        rules.append(
            ScalarComparisonRule(
                field="t",
                rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                value=params["t_min"],
            )
        )
    if params.get("t_max") is not None:
        rules.append(
            ScalarComparisonRule(
                field="t",
                rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                value=params["t_max"],
            )
        )
    return EntityLanePositionCondition(
        entity_name=str(params["entity"]),
        position=position,
        rules=rules,
        label=compiled.label,
    )


def build_entity_road_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityLanePositionCondition`."""
    from ..conditions import EntityLanePositionCondition  # noqa: PLC0415
    from ..coordinate import OpenDrivePose  # noqa: PLC0415
    from ..conditions.comparison import ScalarComparisonRule  # noqa: PLC0415
    from ..conditions.comparison import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    position = OpenDrivePose(
        s=0.0,
        road_id=params["road_id"],
        lane_id=params["lane_id"],
    )
    rules: list[Any] = []
    if params.get("s_min") is not None:
        rules.append(
            ScalarComparisonRule(
                field="s",
                rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                value=params["s_min"],
            )
        )
    if params.get("s_max") is not None:
        rules.append(
            ScalarComparisonRule(
                field="s",
                rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                value=params["s_max"],
            )
        )
    if params.get("t_min") is not None:
        rules.append(
            ScalarComparisonRule(
                field="t",
                rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                value=params["t_min"],
            )
        )
    if params.get("t_max") is not None:
        rules.append(
            ScalarComparisonRule(
                field="t",
                rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                value=params["t_max"],
            )
        )
    return EntityLanePositionCondition(
        entity_name=str(params["entity"]),
        position=position,
        rules=rules,
        label=compiled.label,
    )


def build_route_progress_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`RouteProgressCondition`."""
    from ..conditions import RouteProgressCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    return RouteProgressCondition(
        entity_name=(str(params["entity"]) if params["entity"] is not None else None),
        value=params["value"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        anchor=(str(params["anchor"]) if params["anchor"] is not None else None),
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


def build_temporary_stop_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TemporaryStopCondition`."""
    from ..conditions import TemporaryStopCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    params = compiled.params
    stop_positions: list[Any] = []
    for item in params["stop_lanelets"]:
        stop_positions.append(
            Lanelet2Pose(
                lanelet_id=item,
                s=0.0,
            )
        )
    return TemporaryStopCondition(
        entity_name=str(params["entity"]),
        stop_positions=stop_positions,
        s_margin=params["s_margin"],
        speed_threshold=params["speed_threshold"],
        stop_duration=params["stop_duration"],
        label=compiled.label,
    )


def build_temporary_stop_at_stop_lines_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TemporaryStopCondition`."""
    from ..conditions import TemporaryStopCondition  # noqa: PLC0415
    from ..coordinate import get_stop_line_poses_with_following  # noqa: PLC0415

    params = compiled.params
    stop_positions = get_stop_line_poses_with_following(
        lanelet_id=params["from_lanelet"],
        depth=params["depth"],
    )
    return TemporaryStopCondition(
        entity_name=str(params["entity"]),
        stop_positions=stop_positions,
        s_margin=params["s_margin"],
        speed_threshold=params["speed_threshold"],
        stop_duration=params["stop_duration"],
        label=compiled.label,
    )


def build_turn_indicator_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TurnIndicatorCondition`."""
    from ..conditions import TurnIndicatorCondition  # noqa: PLC0415
    from ..conditions import TurnIndicatorState  # noqa: PLC0415

    params = compiled.params
    return TurnIndicatorCondition(
        entity_name=str(params["entity"]),
        state=TurnIndicatorState[str(params["state"])],
        label=compiled.label,
    )


def build_adversity_active_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`AdversityActiveCondition`."""
    from ..conditions.adversity_active import AdversityActiveCondition  # noqa: PLC0415
    from ..conditions import entity_filter_from_fields  # noqa: PLC0415
    from ..conditions.adversity_active import adversity_kinds_from_field  # noqa: PLC0415

    params = compiled.params
    entities = entity_filter_from_fields(
        names=params["names"],
        kind=params["kind"],
        reference=params["reference"],
        min_distance_m=params["min_distance_m"],
        max_distance_m=params["max_distance_m"],
        min_ahead_m=params["min_ahead_m"],
        max_ahead_m=params["max_ahead_m"],
        min_left_m=params["min_left_m"],
        max_left_m=params["max_left_m"],
        lanelet_ids=params["lanelet_ids"],
        include_reference=params["include_reference"],
    )
    kinds = adversity_kinds_from_field(
        kind=params["adversity"],
    )
    return AdversityActiveCondition(
        entities=entities,
        kinds=kinds,
        include_responses=params["include_responses"],
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
    from ..conditions import RelativeDistanceType  # noqa: PLC0415
    from ..conditions import DistanceCoordinateSystem  # noqa: PLC0415

    params = compiled.params
    return EntityDistanceCondition(
        source=str(params["source"]),
        target=str(params["target"]),
        value=params["distance"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        distance_type=RelativeDistanceType[str(params["distance_type"])],
        edge_to_edge=params["edge_to_edge"],
        coordinate_system=DistanceCoordinateSystem[str(params["measure"]).upper()],
        label=compiled.label,
        signed=params["signed"],
    )


def build_distance_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityPositionDistanceCondition`."""
    from ..conditions import EntityPositionDistanceCondition  # noqa: PLC0415
    from ..trajectory.authoring import authored_place  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    position = authored_place(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
        route_vertex=params["route_pose"],
    )
    return EntityPositionDistanceCondition(
        entity_name=str(params["entity"]),
        position=position,
        value=params["distance"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        label=compiled.label,
    )


def build_relative_speed_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`RelativeSpeedCondition`."""
    from ..conditions import RelativeSpeedCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415
    from ..conditions import SpeedDirection  # noqa: PLC0415

    params = compiled.params
    return RelativeSpeedCondition(
        entity_name=str(params["entity"]),
        reference_entity_name=str(params["reference"]),
        value=params["value"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        direction=SpeedDirection[str(params["direction"])],
        label=compiled.label,
    )


def build_entity_count_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build an :class:`EntityCountCondition`."""
    from ..conditions import EntityCountCondition  # noqa: PLC0415
    from ..conditions import entity_filter_from_fields  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    entities = entity_filter_from_fields(
        names=params["names"],
        kind=params["kind"],
        reference=params["reference"],
        min_distance_m=params["min_distance_m"],
        max_distance_m=params["max_distance_m"],
        min_ahead_m=params["min_ahead_m"],
        max_ahead_m=params["max_ahead_m"],
        min_left_m=params["min_left_m"],
        max_left_m=params["max_left_m"],
        lanelet_ids=params["lanelet_ids"],
        include_reference=params["include_reference"],
    )
    return EntityCountCondition(
        entities=entities,
        rule=ComparisonRule[str(params["rule"]).upper()],
        count=params["count"],
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
    from ..conditions import DistanceCoordinateSystem  # noqa: PLC0415

    params = compiled.params
    return TimeToCollisionCondition(
        source=str(params["source"]),
        target=str(params["target"]),
        value=params["seconds"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        edge_to_edge=params["edge_to_edge"],
        coordinate_system=DistanceCoordinateSystem[str(params["measure"]).upper()],
        label=compiled.label,
    )


def build_ttc_to_position_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeToCollisionCondition`."""
    from ..conditions import TimeToCollisionCondition  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    position = Lanelet2Pose(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
    )
    return TimeToCollisionCondition(
        source=str(params["entity"]),
        value=params["seconds"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        position=position,
        edge_to_edge=params["edge_to_edge"],
        label=compiled.label,
    )


def build_time_headway_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TimeHeadwayCondition`."""
    from ..conditions import TimeHeadwayCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415
    from ..conditions import DistanceCoordinateSystem  # noqa: PLC0415

    params = compiled.params
    return TimeHeadwayCondition(
        source=str(params["entity"]),
        target=str(params["target"]),
        value=params["seconds"],
        rule=ComparisonRule[str(params["rule"]).upper()],
        coordinate_system=DistanceCoordinateSystem[str(params["measure"]).upper()],
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
    from ..conditions import CollisionTargetType  # noqa: PLC0415

    params = compiled.params
    return CollisionCondition(
        min_impulse=params["min_impulse"],
        target=(str(params["target"]) if params["target"] is not None else None),
        target_type=CollisionTargetType[str(params["target_type"])],
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


def build_traffic_signal_controller_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TrafficSignalControllerCondition`."""
    from ..conditions import TrafficSignalControllerCondition  # noqa: PLC0415

    params = compiled.params
    return TrafficSignalControllerCondition(
        controller=str(params["controller"]),
        phase=str(params["signal_phase"]),
        label=compiled.label,
    )


def build_metric_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`MetricCondition`."""
    from ..conditions import MetricCondition  # noqa: PLC0415
    from ..conditions import ComparisonRule  # noqa: PLC0415

    params = compiled.params
    return MetricCondition(
        metric=str(params["metric"]),
        rule=ComparisonRule[str(params["rule"]).upper()],
        value=params["value"],
        max_ng_seconds=params["max_ng_seconds"],
        label=compiled.label,
    )


def build_place_clear_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`PlaceClearCondition`."""
    from ..conditions import PlaceClearCondition  # noqa: PLC0415
    from ..trajectory.authoring import authored_place  # noqa: PLC0415

    params = compiled.params
    position = authored_place(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
        t=params["t"],
        heading=params["heading"],
        route_vertex=params["route_pose"],
    )
    return PlaceClearCondition(
        position=position,
        length=params["length"],
        width=params["width"],
        margin=params["margin"],
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
    import typesafe_carla.carla as carla  # noqa: PLC0415

    params = compiled.params
    return TrafficSignalCondition(
        lanelet2_regulatory_element_id=params["lanelet2_regulatory_element_id"],
        expected_state=getattr(carla.TrafficLightState, str(params["state"])),
        label=compiled.label,
    )


def build_trajectory_time_condition(
    compiled: "CompiledCondition",
    children: "list[BaseCondition]",
    ctx: "BuildContext",
) -> "BaseCondition":
    """Build a :class:`TrajectoryTimeCondition`."""
    from ..conditions import TrajectoryTimeCondition  # noqa: PLC0415

    params = compiled.params
    return TrajectoryTimeCondition(
        time=params["time"],
        label=compiled.label,
    )


def build_adversarial_brake_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`AdversarialBrakeAction`."""
    from ..actions.adversarial import AdversarialBrakeAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return AdversarialBrakeAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_adversarial_cut_in_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`AdversarialCutInAction`."""
    from ..actions.adversarial import AdversarialCutInAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return AdversarialCutInAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_adversarial_merge_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`AdversarialMergeAction`."""
    from ..actions.adversarial import AdversarialMergeAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return AdversarialMergeAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_adversarial_traffic_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`AdversarialTrafficAction`."""
    from ..actions.adversarial_traffic import AdversarialTrafficAction  # noqa: PLC0415
    from ..adversity.nade import NadeSettings  # noqa: PLC0415
    from ..actions.adversarial_traffic import AdversityKind  # noqa: PLC0415
    from ..adversity.location import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial_traffic import NadeAdversity  # noqa: PLC0415
    from ..actions.adversarial_traffic import TrajectoryFollowingMode  # noqa: PLC0415

    params = compiled.params
    settings = NadeSettings(
        max_controlled=params["max_controlled"],
        control_radius=params["control_radius"],
        magnitude=params["magnitude"],
        max_is_probability=params["max_is_probability"],
        fixed_is_probability=params["fixed_is_probability"],
        cooldown=params["cooldown"],
        challenge_npcs=params["challenge_npcs"],
    )
    adversities: list[Any] = []
    for item in params["adversities"]:
        adversities.append(
            NadeAdversity(
                kind=AdversityKind[str(item["kind"]).upper()],
                location=AdversityLocation[str(item["location"]).upper()],
                probability=item["probability"],
            )
        )
    return AdversarialTrafficAction(
        adversities=adversities,
        condition=condition,
        timing=timing,
        label=compiled.label,
        seed=params["seed"],
        settings=settings,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_construction_zone_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`ConstructionZoneAction`."""
    from ..actions.construction_zone import ConstructionZoneAction  # noqa: PLC0415
    from ..adversity.construction import ConstructionSettings  # noqa: PLC0415

    params = compiled.params
    settings = ConstructionSettings(
        construction_type=params["construction_type"],
        spacing=params["spacing"],
        work_zone_offset=params["work_zone_offset"],
        taper_length_in=params["taper_length_in"],
        taper_length_out=params["taper_length_out"],
        taper_type=params["taper_type"],
        warning_zone_length=params["warning_zone_length"],
        buffer_zone_length=params["buffer_zone_length"],
        termination_zone_length=params["termination_zone_length"],
        side=params["side"],
    )
    return ConstructionZoneAction(
        lanelet_id=params["lanelet_id"],
        start_s=params["start_s"],
        end_s=params["end_s"],
        settings=settings,
        start_time=params["start_time"],
        end_time=params["end_time"],
        close_lane=params["close_lane"],
        clear_traffic=params["clear_traffic"],
        condition=condition,
        timing=timing,
        label=compiled.label,
    )


def build_head_on_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`HeadOnAction`."""
    from ..actions.adversarial import HeadOnAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return HeadOnAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_ignore_leader_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`IgnoreLeaderAction`."""
    from ..actions.adversarial import IgnoreLeaderAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return IgnoreLeaderAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_jaywalk_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`JaywalkAction`."""
    from ..actions.adversarial import JaywalkAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return JaywalkAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_lane_change_abort_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`LaneChangeAbortAction`."""
    from ..actions.adversarial import LaneChangeAbortAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return LaneChangeAbortAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_pedestrian_run_red_light_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`PedestrianRunRedLightAction`."""
    from ..actions.adversarial import PedestrianRunRedLightAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return PedestrianRunRedLightAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_run_traffic_rule_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`RunTrafficRuleAction`."""
    from ..actions.adversarial import RunTrafficRuleAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return RunTrafficRuleAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_stalled_object_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`StalledObjectAction`."""
    from ..actions.stalled_object import StalledObjectAction  # noqa: PLC0415
    from ..coordinate import Lanelet2Pose  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    pose = Lanelet2Pose(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
        t=params["t"],
        heading=params["heading"],
    )
    return StalledObjectAction(
        entity_name=compiled.actor_role,
        pose=pose,
        start_time=params["start_time"],
        end_time=params["end_time"],
        condition=condition,
        timing=timing,
        label=compiled.label,
        overrides=compiled.node.overrides,
    )


def build_stop_crossing_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`StopCrossingAction`."""
    from ..actions.adversarial import StopCrossingAction  # noqa: PLC0415
    from ..actions.adversarial import AdversityLocation  # noqa: PLC0415
    from ..actions.adversarial import TrajectoryFollowingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return StopCrossingAction(
        entity_name=compiled.actor_role,
        location=AdversityLocation[str(params["location"]).upper()],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        following_mode=TrajectoryFollowingMode[str(params["following_mode"]).upper()],
    )


def build_urban_construction_zone_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`ConstructionZoneAction`."""
    from ..actions.construction_zone import ConstructionZoneAction  # noqa: PLC0415
    from ..adversity.construction import ConstructionSettings  # noqa: PLC0415

    params = compiled.params
    settings = ConstructionSettings(
        layout="urban",
        warning_sign_offset=2.0,
        construction_type=params["construction_type"],
        spacing=params["spacing"],
        warning_zone_length=params["warning_zone_length"],
        merge_distance=params["merge_distance"],
        merge_signs=params["merge_signs"],
        side=params["side"],
    )
    return ConstructionZoneAction(
        lanelet_id=params["lanelet_id"],
        start_s=params["start_s"],
        end_s=params["end_s"],
        settings=settings,
        start_time=params["start_time"],
        end_time=params["end_time"],
        close_lane=params["close_lane"],
        clear_traffic=params["clear_traffic"],
        condition=condition,
        timing=timing,
        label=compiled.label,
    )


def build_traffic_signal_controller_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficSignalControllerAction`."""
    from ..actions import TrafficSignalControllerAction  # noqa: PLC0415

    params = compiled.params
    return TrafficSignalControllerAction(
        controller=str(params["controller"]),
        phase=str(params["signal_phase"]),
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
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
    import typesafe_carla.carla as carla  # noqa: PLC0415

    params = compiled.params
    lanelet2_traffic_light_ids: Any
    if params["target"] == "all":
        lanelet2_traffic_light_ids = TrafficLightTarget.ALL
    elif params["target"] == "ids":
        lanelet2_traffic_light_ids = params["lanelet2_traffic_light_ids"]
    elif params["target"] == "route":
        lanelet2_traffic_light_ids = TrafficLightTarget.ROUTE
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
        route_junction=params["route_junction"],
        route_approach=str(params["route_approach"]),
        route_turn=str(params["route_turn"]),
    )


def build_environment_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build an :class:`EnvironmentAction`."""
    from ..actions import EnvironmentAction  # noqa: PLC0415

    params = compiled.params
    return EnvironmentAction(
        cloudiness=params["cloudiness"],
        precipitation=params["precipitation"],
        precipitation_deposits=params["precipitation_deposits"],
        wetness=params["wetness"],
        wind_intensity=params["wind_intensity"],
        fog_density=params["fog_density"],
        fog_distance=params["fog_distance"],
        sun_altitude_angle=params["sun_altitude_angle"],
        sun_azimuth_angle=params["sun_azimuth_angle"],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
    )


def build_traffic_sink_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficSinkAction`."""
    from ..actions import TrafficSinkAction  # noqa: PLC0415
    from ..actions.background_traffic import constraints_from_text  # noqa: PLC0415

    params = compiled.params
    constraints = constraints_from_text(
        text=params["constraints"],
    )
    return TrafficSinkAction(
        constraints=constraints,
        label=compiled.label,
        condition=condition,
        timing=timing,
        once=compiled.node.once,
    )


def build_traffic_source_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficSourceAction`."""
    from ..actions import TrafficSourceAction  # noqa: PLC0415
    from ..actions.background_traffic import constraints_from_text  # noqa: PLC0415

    params = compiled.params
    constraints = constraints_from_text(
        text=params["constraints"],
    )
    return TrafficSourceAction(
        constraints=constraints,
        initial_vehicles=params["initial_vehicles"],
        vehicles_per_minute=params["vehicles_per_minute"],
        max_vehicles=params["max_vehicles"],
        speed_kmh=params["speed_kmh"],
        min_gap_m=params["min_gap_m"],
        seed=params["seed"],
        label=compiled.label,
        condition=condition,
        timing=timing,
        once=compiled.node.once,
    )


def build_play_motion_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`PlayMotionAction`."""
    from ..actions import PlayMotionAction  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return PlayMotionAction(
        entity_name=compiled.actor_role,
        motion_path=str(params["motion_path"]),
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        fps=params["fps"],
        loop=params["loop"],
        duration_s=params["duration_s"],
        upper_body_only=params["upper_body_only"],
        overrides=compiled.node.overrides,
    )


def build_smart_pedestrian_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`SmartPedestrianControllerAction`."""
    from ..actions import SmartPedestrianControllerAction  # noqa: PLC0415
    from ..actions import TimedPrompt  # noqa: PLC0415
    from ..actions import WalkWaypoint  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    prompts: list[Any] = []
    for item in params["prompts"]:
        prompts.append(
            TimedPrompt(
                time_s=item["time_s"],
                text=str(item["text"]),
            )
        )
    waypoints: list[Any] = []
    for item in params["waypoints"]:
        waypoints.append(
            WalkWaypoint(
                x=item["x"],
                y=item["y"],
            )
        )
    return SmartPedestrianControllerAction(
        entity_name=compiled.actor_role,
        prompt=str(params["prompt"]),
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        prompts=prompts,
        waypoints=waypoints,
        speed_ms=params["speed_ms"],
        keyframes=(str(params["poses"]) if params["poses"] is not None else None),
        duration_s=params["duration_s"],
        seed=params["seed"],
        overrides=compiled.node.overrides,
    )


def build_walk_straight_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`WalkStraightAction`."""
    from ..actions import WalkStraightAction  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return WalkStraightAction(
        entity_name=compiled.actor_role,
        speed_ms=params["speed_ms"],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        turn_back_ttc_s=params["turn_back_ttc_s"],
        turn_back_distance_m=params["turn_back_distance_m"],
        turn_back_within_m=params["turn_back_within_m"],
    )


def build_traffic_parameters_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficParametersAction`."""
    from ..actions import TrafficParametersAction  # noqa: PLC0415
    from ..traffic.authoring import authored_vehicle_parameters  # noqa: PLC0415
    from ..traffic.authoring import authored_pedestrian_parameters  # noqa: PLC0415
    from ..traffic.authoring import authored_bicycle_parameters  # noqa: PLC0415

    params = compiled.params
    vehicles = authored_vehicle_parameters(
        speed_factor_mean=params["speed_factor_mean"],
        speed_factor_std=params["speed_factor_std"],
        min_gap_m_mean=params["min_gap_m_mean"],
        min_gap_m_std=params["min_gap_m_std"],
        ignore_lights_mean=params["ignore_lights_mean"],
        ignore_lights_std=params["ignore_lights_std"],
        ignore_signs_mean=params["ignore_signs_mean"],
        ignore_signs_std=params["ignore_signs_std"],
        ignore_vehicles_mean=params["ignore_vehicles_mean"],
        ignore_vehicles_std=params["ignore_vehicles_std"],
        ignore_walkers_mean=params["ignore_walkers_mean"],
        ignore_walkers_std=params["ignore_walkers_std"],
        random_lane_changes_mean=params["random_lane_changes_mean"],
        random_lane_changes_std=params["random_lane_changes_std"],
        accel_mean=params["accel_mean"],
        accel_std=params["accel_std"],
        decel_mean=params["decel_mean"],
        decel_std=params["decel_std"],
        reaction_time_s_mean=params["reaction_time_s_mean"],
        reaction_time_s_std=params["reaction_time_s_std"],
        imperfection_mean=params["imperfection_mean"],
        imperfection_std=params["imperfection_std"],
        max_lateral_speed_mps_mean=params["max_lateral_speed_mps_mean"],
        max_lateral_speed_mps_std=params["max_lateral_speed_mps_std"],
        max_lateral_accel_mean=params["max_lateral_accel_mean"],
        max_lateral_accel_std=params["max_lateral_accel_std"],
        lane_changes=params["lane_changes"],
        follow_visibility=params["follow_visibility"],
    )
    pedestrians = authored_pedestrian_parameters(
        walking_speed_mps_mean=params["walking_speed_mps_mean"],
        walking_speed_mps_std=params["walking_speed_mps_std"],
        pedestrian_min_gap_m_mean=params["pedestrian_min_gap_m_mean"],
        pedestrian_min_gap_m_std=params["pedestrian_min_gap_m_std"],
        red_light_violation_s_mean=params["red_light_violation_s_mean"],
        red_light_violation_s_std=params["red_light_violation_s_std"],
    )
    bicycles = authored_bicycle_parameters(
        bicycle_max_speed_mps_mean=params["bicycle_max_speed_mps_mean"],
        bicycle_max_speed_mps_std=params["bicycle_max_speed_mps_std"],
        bicycle_accel_mean=params["bicycle_accel_mean"],
        bicycle_accel_std=params["bicycle_accel_std"],
        bicycle_decel_mean=params["bicycle_decel_mean"],
        bicycle_decel_std=params["bicycle_decel_std"],
        bicycle_min_gap_m_mean=params["bicycle_min_gap_m_mean"],
        bicycle_min_gap_m_std=params["bicycle_min_gap_m_std"],
        bicycle_reaction_time_s_mean=params["bicycle_reaction_time_s_mean"],
        bicycle_reaction_time_s_std=params["bicycle_reaction_time_s_std"],
        bicycle_imperfection_mean=params["bicycle_imperfection_mean"],
        bicycle_imperfection_std=params["bicycle_imperfection_std"],
        bicycle_max_lateral_speed_mps_mean=params["bicycle_max_lateral_speed_mps_mean"],
        bicycle_max_lateral_speed_mps_std=params["bicycle_max_lateral_speed_mps_std"],
        bicycle_red_light_violation_s_mean=params["bicycle_red_light_violation_s_mean"],
        bicycle_red_light_violation_s_std=params["bicycle_red_light_violation_s_std"],
    )
    return TrafficParametersAction(
        vehicles=vehicles,
        pedestrians=pedestrians,
        bicycles=bicycles,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
    )


def build_communication_loss_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`CommunicationLossAction`."""
    from ..actions import CommunicationLossAction  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return CommunicationLossAction(
        device=(str(params["device"]) if params["device"] is not None else None),
        messages=(str(params["messages"]) if params["messages"] is not None else None),
        direction=str(params["direction"]),
        duration=params["duration"],
        entity_name=compiled.actor_role,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
    )


def build_turn_indicator_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TurnIndicatorAction`."""
    from ..actions import TurnIndicatorAction  # noqa: PLC0415
    from ..utils.turn_indicator import TurnIndicatorState  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return TurnIndicatorAction(
        entity_name=compiled.actor_role,
        state=TurnIndicatorState[str(params["state"])],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
    )


def build_despawn_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`DespawnAction`."""
    from ..actions import DespawnAction  # noqa: PLC0415

    params = compiled.params
    return DespawnAction(
        entity_name=compiled.actor_role,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        name=(str(params["name"]) if params["name"] is not None else None),
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
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
    )


def build_routing_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`RoutingAction`."""
    from ..actions import RoutingAction  # noqa: PLC0415
    from ..trajectory.authoring import authored_goal  # noqa: PLC0415
    from ..trajectory.authoring import authored_waypoints  # noqa: PLC0415
    from ..trajectory.authoring import authored_route  # noqa: PLC0415
    from .builders import instantiate_advance_conditions  # noqa: PLC0415
    from ..trajectory.authoring import authored_timing  # noqa: PLC0415
    from ..actions import RoutingMode  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    goal = authored_goal(
        path_source=params["path_source"],
        lanelet_id=params["goal_lanelet_id"],
        s=params["goal_s"],
    )
    waypoints = authored_waypoints(
        path_source=params["path_source"],
        lanelet_ids=params["waypoint_lanelet_ids"],
    )
    trajectory = authored_route(
        path_source=params["path_source"],
        vertices=params["vertices"],
        lanelet_ids=params["lanelet_ids"],
        speed_kmh=params["speed_kmh"],
        lateral_offset_m=params["lateral_offset_m"],
        relative_vertices=params["relative_vertices"],
        reference_entity=params["reference_entity"],
        route_vertices=params["route_vertices"],
        advance=instantiate_advance_conditions(compiled, ctx),
    )
    time_reference = authored_timing(
        time_domain=params["time_domain"],
        scale=params["time_scale"],
        offset=params["time_offset"],
    )
    return RoutingAction(
        entity_name=compiled.actor_role,
        goal=goal,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        waypoints=waypoints,
        walking_speed=params["walking_speed"],
        cross_factor=params["cross_factor"],
        turn_back_ttc_s=params["turn_back_ttc_s"],
        turn_back_distance_m=params["turn_back_distance_m"],
        turn_back_within_m=params["turn_back_within_m"],
        trajectory=trajectory,
        mode=RoutingMode[str(params["mode"]).upper()],
        time_reference=time_reference,
        initial_distance_offset=params["initial_distance_offset"],
        speed=params["speed_ms"],
        stop_at_goal=params["stop_at_goal"],
    )


def build_set_speed_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`SetSpeedAction`."""
    from ..actions import SetSpeedAction  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    return SetSpeedAction(
        entity_name=compiled.actor_role,
        target_speed_kmh=params["target_speed_kmh"],
        rate_kmh_s=params["rate_kmh_s"],
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
    )


def build_spawn_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`SpawnAction`."""
    from ..actions import SpawnAction  # noqa: PLC0415
    from ..trajectory.authoring import authored_place  # noqa: PLC0415

    params = compiled.params
    position = authored_place(
        lanelet_id=params["lanelet_id"],
        s=params["s"],
        t=params["t"],
        heading=params["heading"],
        route_vertex=params["route_pose"],
    )
    return SpawnAction(
        entity_name=compiled.actor_role,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        position=position,
        shift_max=params["shift_max"],
        speed=params["speed_ms"],
        name=(str(params["name"]) if params["name"] is not None else None),
        blueprint=str(params["blueprint"]),
        on_ground=params["on_ground"],
        lanes=params["lanes"],
        length=params["length"],
        width=params["width"],
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
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
        overrides=compiled.node.overrides,
        search_distance=params["search_distance"],
    )


def build_vehicle_parameters_action(
    compiled: "CompiledAction",
    condition: "BaseCondition | None",
    timing: Any,
    ctx: "BuildContext",
) -> "BaseAction":
    """Build a :class:`TrafficParametersAction`."""
    from ..actions import TrafficParametersAction  # noqa: PLC0415
    from ..traffic.authoring import authored_vehicle_parameters  # noqa: PLC0415

    assert compiled.actor_role is not None  # noqa: S101 -- required by the spec
    params = compiled.params
    vehicles = authored_vehicle_parameters(
        speed_factor_mean=params["speed_factor_mean"],
        speed_factor_std=params["speed_factor_std"],
        min_gap_m_mean=params["min_gap_m_mean"],
        min_gap_m_std=params["min_gap_m_std"],
        ignore_lights_mean=params["ignore_lights_mean"],
        ignore_lights_std=params["ignore_lights_std"],
        ignore_signs_mean=params["ignore_signs_mean"],
        ignore_signs_std=params["ignore_signs_std"],
        ignore_vehicles_mean=params["ignore_vehicles_mean"],
        ignore_vehicles_std=params["ignore_vehicles_std"],
        ignore_walkers_mean=params["ignore_walkers_mean"],
        ignore_walkers_std=params["ignore_walkers_std"],
        random_lane_changes_mean=params["random_lane_changes_mean"],
        random_lane_changes_std=params["random_lane_changes_std"],
        accel_mean=params["accel_mean"],
        accel_std=params["accel_std"],
        decel_mean=params["decel_mean"],
        decel_std=params["decel_std"],
        reaction_time_s_mean=params["reaction_time_s_mean"],
        reaction_time_s_std=params["reaction_time_s_std"],
        imperfection_mean=params["imperfection_mean"],
        imperfection_std=params["imperfection_std"],
        max_lateral_speed_mps_mean=params["max_lateral_speed_mps_mean"],
        max_lateral_speed_mps_std=params["max_lateral_speed_mps_std"],
        max_lateral_accel_mean=params["max_lateral_accel_mean"],
        max_lateral_accel_std=params["max_lateral_accel_std"],
        lane_changes=params["lane_changes"],
        follow_visibility=params["follow_visibility"],
    )
    return TrafficParametersAction(
        vehicles=vehicles,
        entity_name=compiled.actor_role,
        condition=condition,
        timing=timing,
        label=compiled.label,
        once=compiled.node.once,
    )
