"""Example scenario: Ego vehicle passes through an intersection on expected roads.

This scenario spawns the ego vehicle on a configurable lanelet, sets all
traffic lights to green, and verifies that the vehicle traverses the expected
OpenDRIVE roads.  Each road check uses a
:class:`~autoware_carla_scenario.StickyCondition` so that the condition latches
once the vehicle visits the road, and the conditions are combined with
:class:`~autoware_carla_scenario.AndCondition` to assert the full route.

The scenario supports an optional ``turn_direction`` (``"left"`` / ``"right"`` /
``"straight"``) that registers a :class:`~autoware_carla_scenario.TurnAction` to steer the ego
through the junction, and an optional ``min_speed_kmh`` that adds a
:class:`~autoware_carla_scenario.SpeedCondition` fail condition.

More fail conditions judge how the ego turns, from the start until it is
through the junction: ``turn_indicator_before_m`` fails a run whose ego, that
close to the junction and not yet in it, does not show the indicator of its
turn;
``max_steering_reversals`` one whose steering turned back that many times (the
``steering_reversals`` metric); and ``max_turn_indicator_changes``, counted
apart from the steering, one whose indicators changed that many times (the
``turn_indicator_changes`` metric).

Typical usage
-------------
Straight-through::

    uv run scenario scenario=intersection_passing

Left turn::

    uv run scenario scenario=left_turn

With pytest — see ``test/carla_scenario/test_examples.py``.
"""

from __future__ import annotations

import logging

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AndCondition,
    BaseScenario,
    ComparisonRule,
    EgoConfig,
    ElapsedTimeCondition,
    EntityLanePositionCondition,
    EntityPositionDistanceCondition,
    EntityRole,
    GroundProjectionConfig,
    Lanelet2Pose,
    MetricCondition,
    NotCondition,
    PersistentCondition,
    SpawnTransform,
    SpeedCondition,
    StickyCondition,
    TickTiming,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    TurnAction,
    TurnDirection,
    TurnIndicatorCondition,
    TurnIndicatorState,
    VehicleEntity,
    VehicleEntityConfig,
    snap_to_carla_road,
    to_opendrive,
)

from autoware_carla_scenario.coordinate import MapManager, route_road_ids
from autoware_carla_scenario.metrics import STEERING_REVERSALS, TURN_INDICATOR_CHANGES
from autoware_carla_scenario.odd import lanelet_in_junction
from .configs import IntersectionPassingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lanelet IDs that define the expected route
# ---------------------------------------------------------------------------

#: Ordered lanelets the ego is expected to traverse through the intersection.
EXPECTED_ROUTE_LANELET_IDS: list[int] = [460, 265]

#: Timeout in seconds — if the ego has not completed the route by this time the
#: scenario fails.
SCENARIO_TIMEOUT_SECONDS: float = 5.0

#: Minimum speed (km/h) — the scenario fails if the ego drops below this speed.
MIN_SPEED_KMH: float = 5.0


def _is_junction_lanelet(lanelet_id: int) -> bool:
    """Whether the lanelet is in a junction, as the ODD reads it
    (:func:`~autoware_carla_scenario.odd.probes.lanelet_in_junction`)."""
    return lanelet_in_junction(
        MapManager.get_instance().lanelet_map.laneletLayer[lanelet_id]
    )


_TURN_DIRECTION_MAP: dict[str, TurnDirection] = {
    "left": TurnDirection.LEFT,
    "right": TurnDirection.RIGHT,
    "straight": TurnDirection.STRAIGHT,
}

#: The indicator a turn is signalled with.
_TURN_INDICATOR: dict[str, TurnIndicatorState] = {
    "left": TurnIndicatorState.LEFT,
    "right": TurnIndicatorState.RIGHT,
}


class IntersectionPassingScenario(BaseScenario):
    """Spawn the ego and verify it crosses expected roads through an intersection.

    The scenario:

    1. Snaps a Lanelet2 pose to the CARLA road surface for the ego spawn, and
       routes an ego that plans for itself (an Autoware one) to the last
       lanelet of ``expected_route_lanelet_ids`` unless a goal is configured.
    2. Sets every traffic light in the world to green so the ego proceeds
       without stopping.
    3. If ``turn_direction`` is set, registers a :class:`TurnAction` to steer
       the ego through the junction.
    4. Registers a pass condition: an :class:`AndCondition` of sticky
       :class:`EntityLanePositionCondition` instances, one per expected road.
    5. If ``min_speed_kmh`` is set, registers a :class:`SpeedCondition` that
       fails the scenario when the ego drops below the threshold.
    6. If ``turn_indicator_before_m`` is set, fails the scenario once the ego,
       having come that close to the junction and not yet in it, has not shown
       the indicator of its turn for ``turn_indicator_grace_seconds``.
    7. If ``max_steering_reversals`` is set, fails it once the steering has
       turned back that many times; if ``max_turn_indicator_changes`` is, once
       the indicators have changed that many times.  Each is counted on its own.
    8. Registers a :class:`TimeoutCondition` as a fail-safe.

    Once all expected roads have been visited the scenario passes.
    """

    _config: IntersectionPassingConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: IntersectionPassingConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or IntersectionPassingConfig()

    def setup(self) -> None:
        """Snap ego spawn to CARLA road, set lights to green, register conditions."""
        world = self.world
        cfg = self._config

        # --- Route the ego to the end of the route this scenario asserts ---
        # The scenario already declares where the ego is meant to end up, and
        # the last lanelet of that route is where the pass condition latches --
        # the finish line.  A goal given in the config wins.
        self.derive_goal_from_route(cfg.expected_route_lanelet_ids)

        # --- Compute ego spawn from Lanelet2Pose via OpenDrivePose ---
        self._setup_ego_spawn()

        # --- Spawn NPC vehicles ---
        for i, npc_cfg in enumerate(cfg.npc_vehicles, start=1):
            npc_pose = Lanelet2Pose(
                lanelet_id=npc_cfg.spawn_lanelet_id, s=npc_cfg.spawn_s
            )
            # Snapped as the Lanelet2 pose it was authored as; the OpenDRIVE
            # pose is carried on to the entity only to enable the spawn retries
            # (which offset the snapped transform, not this pose).
            npc_od_pose = to_opendrive(npc_pose)
            npc_snapped = snap_to_carla_road(
                npc_pose, world, ground_projection=self._ground_projection
            )
            npc_entity = VehicleEntity(
                VehicleEntityConfig(
                    role_name=str(EntityRole.npc(i)),
                    spawn_location=SpawnTransform(npc_snapped.to_carla_transform()),
                    vehicle_type=npc_cfg.vehicle_type,
                    initial_speed_kmh=npc_cfg.initial_speed_kmh,
                    spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                    spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                    spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                    od_pose=npc_od_pose,
                    ground_projection=self._ground_projection,
                )
            )
            npc_entity.spawn(world)
            self.register_entity(npc_entity)
            logger.info(
                "Spawned NPC %s on lanelet %d (s=%.1f)",
                EntityRole.npc(i),
                npc_cfg.spawn_lanelet_id,
                npc_cfg.spawn_s,
            )

        # --- Set all traffic lights to green ---
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        # --- Optional turn action ---
        if cfg.turn_direction is not None:
            direction = _TURN_DIRECTION_MAP[cfg.turn_direction]
            turn_action = TurnAction(
                entity_name=EGO_ROLE_NAME,
                direction=direction,
                timing=TickTiming.PRE_TICK,
            )
            self.register_pre_tick(turn_action)
            logger.info("Registered TurnAction: %s", cfg.turn_direction)

        # --- Build unique route road IDs from lanelet IDs ---
        road_ids = route_road_ids(list(cfg.expected_route_lanelet_ids))

        # --- Register sticky lane-position conditions ---
        sticky_conditions = [
            StickyCondition(
                EntityLanePositionCondition.anywhere_on_road(
                    EGO_ROLE_NAME,
                    rid,
                    label=f"ego_on_road_{rid}",
                )
            )
            for rid in road_ids
        ]

        if len(sticky_conditions) == 1:
            # A turn's route is often its junction and the road after it alone.
            self.register_pass_condition(sticky_conditions[0])
        else:
            self.register_pass_condition(AndCondition(sticky_conditions))

        # --- Fail: speed drops below minimum (if configured) ---
        # Wrapped with ElapsedTimeCondition so the check only activates after
        # a grace period, giving the ego time to reach cruising speed.
        if cfg.min_speed_kmh is not None:
            self.register_fail_condition(
                AndCondition(
                    [
                        ElapsedTimeCondition(
                            cfg.speed_check_delay_seconds,
                            label="speed_check_delay",
                        ),
                        SpeedCondition(
                            entity_name=EGO_ROLE_NAME,
                            value=cfg.min_speed_kmh / 3.6,
                            rule=ComparisonRule.LESS_THAN,
                            label="ego_min_speed",
                        ),
                    ]
                )
            )

        # --- Fail: no indicator for the turn, close to the junction ---
        # From that close to the junction's mouth (the start of the route's
        # first junction lanelet, at its stop line; the route's first lanelet
        # when none is tagged) until the ego is on that lanelet: the
        # approach.  Each is latched, as the distance grows again inside the
        # junction.  In it, a driver cancels the indicator as the turn
        # straightens out; one that flickers there is turn_indicator_changes'.
        if cfg.turn_indicator_before_m is not None and cfg.turn_direction:
            route = list(cfg.expected_route_lanelet_ids)
            entry = next((i for i in route if _is_junction_lanelet(i)), route[0])
            mouth = Lanelet2Pose(lanelet_id=entry, s=0.0)
            near_junction = StickyCondition(
                EntityPositionDistanceCondition(
                    EGO_ROLE_NAME,
                    mouth,
                    cfg.turn_indicator_before_m,
                    ComparisonRule.LESS_THAN_OR_EQUAL,
                    label="ego_near_junction",
                )
            )
            in_junction = StickyCondition(
                EntityLanePositionCondition(
                    EGO_ROLE_NAME, mouth, label="ego_in_junction"
                )
            )
            signalling = TurnIndicatorCondition(
                EGO_ROLE_NAME,
                _TURN_INDICATOR[cfg.turn_direction],
                label=f"ego_signals_{cfg.turn_direction}",
            )
            self.register_fail_condition(
                PersistentCondition(
                    AndCondition(
                        [
                            near_junction,
                            NotCondition(in_junction, holds_when_pending=True),
                            NotCondition(signalling, holds_when_pending=True),
                        ]
                    ),
                    cfg.turn_indicator_grace_seconds,
                    label=f"no_{cfg.turn_direction}_indicator",
                )
            )

        # --- Fail: the steering turned back, or the indicators changed, too
        # often (each counted apart from the other) ---
        for metric, limit in (
            (STEERING_REVERSALS, cfg.max_steering_reversals),
            (TURN_INDICATOR_CHANGES, cfg.max_turn_indicator_changes),
        ):
            if limit is not None:
                self.register_fail_condition(
                    MetricCondition(
                        metric,
                        ComparisonRule.GREATER_THAN_OR_EQUAL,
                        float(limit),
                        label=metric,
                    )
                )

        # --- Fail-safe timeout ---
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
