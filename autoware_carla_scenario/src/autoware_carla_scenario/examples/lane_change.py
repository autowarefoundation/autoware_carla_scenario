"""Example scenario: Ego vehicle performs a lane change on a straight road.

This scenario spawns the ego vehicle on a configurable lanelet, sets all
traffic lights to green, and registers a
:class:`~autoware_carla_scenario.LaneChangeAction` to force a lane change
via TrafficManager.  It has no trigger condition: it fires at once, on the
first tick.

The target lane is tested in the Lanelet2 frame
(:func:`lane_change_target`): the lanelet beside the spawn lanelet on the
requested side -- the routing graph's ``left`` / ``right``, the lane the
sweep's ``has_adjacent`` asks about, failing that its ``adjacentLeft`` /
``adjacentRight`` (:func:`lane_beside`) -- and the lanelets ahead of it before
the next junction.  The ego has changed lanes once its centre is on one of
them.  An OpenDRIVE road and a lane ID shifted by one is not used: matched
through the nearest road, part of a lanelet's centreline can read as the lane
beside it (#1081).

Typical usage
-------------
Left lane change::

    uv run scenario scenario=lane_change/left

Right lane change::

    uv run scenario scenario=lane_change/right

Its shipped configs and target-lane conditions are tested without CARLA, in
``test/carla_scenario/test_lane_change_example.py``.
"""

from __future__ import annotations

import logging

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AlwaysTrueCondition,
    AndCondition,
    BaseCondition,
    BaseScenario,
    EgoConfig,
    ElapsedTimeCondition,
    EntityLanePositionCondition,
    GroundProjectionConfig,
    LaneChangeAction,
    LaneChangeDirection,
    Lanelet2Pose,
    NotCondition,
    OrCondition,
    StickyCondition,
    TickTiming,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
)
from autoware_carla_scenario.failure import FailureCategory

from autoware_carla_scenario.coordinate import (
    lane_beside as lane_beside,
    lane_change_target_lanelets,
    lanes_ahead_before_junction as lanes_ahead_before_junction,
)

from .configs import LaneChangeConfig
from .cut_in import ego_lanes_ahead, left_lanes_ahead

logger = logging.getLogger(__name__)

_DIRECTION_MAP: dict[str, LaneChangeDirection] = {
    "left": LaneChangeDirection.LEFT,
    "right": LaneChangeDirection.RIGHT,
}

#: How far ahead of the target lanelet the lanelets that also count as the
#: target lane are looked for; :func:`lanes_ahead_before_junction` stops at
#: the next junction well before it.
TARGET_LANES_AHEAD_M = 1000.0

#: ``expect=fail``: how far ahead of the spawn lanelet the ego's own lanes are
#: followed (m): the spawn lanelet and every lanelet that continues it,
#: through junctions (:func:`~.cut_in.ego_lanes_ahead`), well past what the
#: ego drives before its timeout.
OWN_LANES_AHEAD_M = 1000.0

#: ``expect=fail``: how long the ego has to be off its own lanes before the
#: run fails (s), as ``CutInConfig.leaving_lane_seconds``: a tick or two off
#: at a seam between lanelets is not leaving them.
LEAVING_LANE_SECONDS = 0.2


def lane_change_target(lanelet_id: int, direction: str) -> BaseCondition:
    """The ego is on the lane beside *lanelet_id* on *direction*'s side, latched.

    The lanelet :func:`lane_beside` gives, and the lanelets ahead of it before
    the next junction (:func:`lanes_ahead_before_junction`), less any that
    the ego's own lane also reaches -- a lane beside that merges into the
    ego's.  An ``EntityLanePositionCondition`` per lanelet, under an
    ``OrCondition`` when there is more than one, latched by a
    ``StickyCondition``.  With no lane beside (a ``lane_change_fail`` case
    with no lane on that side in its direction), a lane change has nowhere to
    end: the condition never holds.
    """
    # Appended one by one, and chosen by if/else: Codon gives neither a list
    # comprehension nor a conditional expression the base class's type.
    on_lanes: list[BaseCondition] = []
    for lanelet in lane_change_target_lanelets(
        lanelet_id, direction, TARGET_LANES_AHEAD_M
    ):
        on_lanes.append(
            EntityLanePositionCondition(
                EGO_ROLE_NAME,
                Lanelet2Pose(lanelet_id=lanelet, s=0.0),
                label=f"ego_on_lanelet_{lanelet}",
            )
        )
    on_any: BaseCondition
    if not on_lanes:
        # Pending on every check: a logical NOT of a condition that always holds.
        on_any = NotCondition(
            AlwaysTrueCondition(label="no_lane_beside"),
            label="ego_target_lane",
            holds_when_pending=True,
        )
    elif len(on_lanes) == 1:
        on_any = on_lanes[0]
    else:
        on_any = OrCondition(on_lanes, label="ego_target_lane")
    return StickyCondition(on_any, label="ego_target_lane_sticky")


def on_own_lanes(lanelet_id: int, distance_m: float) -> BaseCondition:
    """The ego is on its own lanes: *lanelet_id* or a lanelet that continues
    it, through junctions, within *distance_m* (:func:`~.cut_in.ego_lanes_ahead`).

    An ``EntityLanePositionCondition`` per lanelet, under an ``OrCondition``
    when there is more than one; not latched, so it holds only while the ego
    is on one of them.
    """
    # Appended one by one, and chosen by if/else, as in lane_change_target.
    on_lanes: list[BaseCondition] = []
    for lanelet in ego_lanes_ahead(lanelet_id, distance_m):
        on_lanes.append(
            EntityLanePositionCondition(
                EGO_ROLE_NAME,
                Lanelet2Pose(lanelet_id=lanelet, s=0.0),
                label=f"ego_on_own_lanelet_{lanelet}",
            )
        )
    on_any: BaseCondition
    if len(on_lanes) == 1:
        on_any = on_lanes[0]
    else:
        on_any = OrCondition(on_lanes, label="ego_on_own_lanes")
    return on_any


class LaneChangeScenario(BaseScenario):
    """Spawn the ego and verify it completes a lane change.

    The scenario:

    1. Snaps a Lanelet2 pose to the CARLA road surface for the ego spawn.
    2. Sets every traffic light in the world to green.
    3. Registers a :class:`LaneChangeAction` that fires immediately.
    4. Finds the lanelet beside the spawn lanelet on the requested side, and
       registers a condition on the ego reaching it (:func:`lane_change_target`).
    5. Registers a :class:`TimeoutCondition` as a fail-safe -- or, for
       ``expect=fail``, passes once it has run out with the ego still in its
       own lane (:func:`on_own_lanes`), and fails as soon as the ego reaches
       the lane beside or leaves its own lanes for
       :data:`LEAVING_LANE_SECONDS` (#2253).
    """

    _config: LaneChangeConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: LaneChangeConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or LaneChangeConfig()

    def setup(self) -> None:
        """Snap ego spawn to CARLA road, register lane-change action and conditions."""
        cfg = self._config

        # --- Compute ego spawn from Lanelet2Pose via OpenDrivePose ---
        od_pose = self._setup_ego_spawn()

        # --- Set all traffic lights to green ---
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        # --- Lane-change action (fires immediately) ---
        direction = _DIRECTION_MAP[cfg.direction]
        lane_change_action = LaneChangeAction(
            entity_name=EGO_ROLE_NAME,
            direction=direction,
            timing=TickTiming.PRE_TICK,
        )
        self.register_pre_tick(lane_change_action)
        logger.info("Registered LaneChangeAction: %s", cfg.direction)

        # --- Conditions based on expected outcome ---
        # The lane beside the spawn lanelet, in the Lanelet2 frame.
        spawn = self._spawn_pose
        if spawn is None:
            raise ValueError("lane_change needs an ego spawn pose")
        spawn_pose: Lanelet2Pose = spawn
        expect_fail = cfg.expect == "fail"
        logger.info(
            "Expecting lane change: lanelet %d -> %s (%s) [expect=%s]; "
            "spawn on road '%s' lane %d",
            spawn_pose.lanelet_id,
            lane_beside(spawn_pose.lanelet_id, cfg.direction),
            cfg.direction,
            cfg.expect,
            od_pose.road_id,
            od_pose.lane_id,
        )

        lane_position_condition = lane_change_target(
            spawn_pose.lanelet_id, cfg.direction
        )
        timeout_condition = TimeoutCondition(
            cfg.timeout_seconds, label="scenario_timeout"
        )

        if expect_fail:
            # expect=fail: the lane change is refused, and the ego stays in
            # its own lane (#2253).
            # The timeout runs out with the ego on its own lanes → pass.  Pass
            # conditions are checked first on a tick, so the timeout below
            # fails the run only when the ego is not on them then.  An
            # ElapsedTimeCondition, which fires on the same tick as the
            # timeout: a TimeoutCondition's failure would be the AND's result,
            # and the passed run would read "Timeout after ...".
            self.register_pass_condition(
                AndCondition(
                    [
                        ElapsedTimeCondition(
                            cfg.timeout_seconds, label="lane_change_window"
                        ),
                        on_own_lanes(spawn_pose.lanelet_id, OWN_LANES_AHEAD_M),
                    ],
                    label="ego_kept_its_lane",
                )
            )
            # The ego reaches the lane beside → fail.
            self.register_fail_condition(
                NotCondition(lane_position_condition),
                category=FailureCategory.LANE_KEEPING,
            )
            # The ego leaves its own lanes some other way (onto a lane beside
            # across a line, off the road) → fail.
            self.register_fail_condition(
                left_lanes_ahead(
                    spawn_pose.lanelet_id, OWN_LANES_AHEAD_M, LEAVING_LANE_SECONDS
                ),
                category=FailureCategory.LANE_KEEPING,
            )
            self.register_fail_condition(timeout_condition)
        else:
            # expect=success: lane change should happen.
            # If ego reaches the target lane → pass.
            # If timeout expires → fail.
            self.register_pass_condition(lane_position_condition)
            self.register_fail_condition(timeout_condition)

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
