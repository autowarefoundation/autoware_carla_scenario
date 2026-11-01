"""Example scenario: a pedestrian darts out into the road in front of the ego.

The ego drives along a lane at the edge of the road. A pedestrian waits at the
kerb some way ahead and, once the ego is close, runs out into the lane. The ego
must not hit them: it passes once it is past the point where they crossed, and
fails on a collision with them.

The pedestrian stands ``pedestrian_offset_m`` from the centre of the ego's
lanelet, on its kerb side -- the right of the direction of travel where the
road is right-hand traffic, the left where it is left-hand traffic (the
OpenDRIVE road's ``rule``) -- and runs across it, towards the lane's centre
and beyond.

Typical usage
-------------
On every lane at the kerb of a map, one case each (the sweep's kerb side is
set for the shipped left-hand-traffic map; see its YAML)::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=pedestrian_dart_out/pedestrian_dart_out map=<map>
"""

from __future__ import annotations

import logging
import math

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    CollisionCondition,
    EgoConfig,
    EntityDistanceCondition,
    EntityPositionDistanceCondition,
    EntityRole,
    GroundProjectionConfig,
    Lanelet2Pose,
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    WalkStraightAction,
    to_carla_world,
)
from autoware_carla_scenario.coordinate import road_traffic_rule

from .configs import PedestrianDartOutConfig

logger = logging.getLogger(__name__)

#: The role the pedestrian answers to.
PEDESTRIAN_ROLE = EntityRole("pedestrian1")

#: How close to the pass point the ego has to come (m): the width of a lane.
_PASS_RADIUS_M = 3.0

#: How far above the ground a pedestrian's spawn point is (m). CARLA places a
#: walker by its middle; a point at the road's height puts its feet in a kerb
#: that stands above the road, and the spawn is refused.
_WALKER_SPAWN_HEIGHT_M = 1.0


def kerb_offset_m(distance_m: float, traffic_rule: str) -> float:
    """The lateral offset ``t`` (m, positive to the left of the direction of
    travel) of a point *distance_m* from a lane's centre on its kerb side.

    The kerb is on the right in right-hand traffic and on the left in
    left-hand traffic.  The sign of *distance_m* is ignored, so the old
    ``-3.0`` still means 3 m towards the kerb.

    Raises:
        ValueError: If *traffic_rule* is not ``"RHT"`` or ``"LHT"``.
    """
    if traffic_rule == "LHT":
        return abs(distance_m)
    if traffic_rule == "RHT":
        return -abs(distance_m)
    raise ValueError(f"traffic_rule must be 'RHT' or 'LHT', got {traffic_rule!r}")


def kerb_pose(
    spawn: Lanelet2Pose, config: PedestrianDartOutConfig, traffic_rule: str
) -> Lanelet2Pose:
    """Where the pedestrian waits: ``pedestrian_ahead_m`` along the ego's
    lanelet from *spawn*, ``pedestrian_offset_m`` off its centre on the kerb
    side of *traffic_rule*, facing across the lane."""
    t = kerb_offset_m(config.pedestrian_offset_m, traffic_rule)
    # A quarter turn to the left of the lane's direction from the right-hand
    # kerb, to the right from the left one.
    across = math.pi / 2 if t < 0 else -math.pi / 2
    return Lanelet2Pose(
        lanelet_id=spawn.lanelet_id,
        s=spawn.s + config.pedestrian_ahead_m,
        t=t,
        heading=across,
    )


class PedestrianDartOutScenario(BaseScenario):
    """Spawn the ego and a pedestrian at the kerb ahead; it runs out as the ego nears."""

    _config: PedestrianDartOutConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: PedestrianDartOutConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or PedestrianDartOutConfig()

    def setup(self) -> None:
        """Spawn the pedestrian at the kerb, and register its run and the conditions."""
        world = self.world
        cfg = self._config
        spawn = self._spawn_pose
        if spawn is None:
            raise ValueError("pedestrian_dart_out needs an ego spawn pose")

        crossing_s = spawn.s + cfg.pedestrian_ahead_m
        beyond = Lanelet2Pose(
            lanelet_id=spawn.lanelet_id, s=crossing_s + cfg.pass_beyond_m
        )
        # An ego that plans its own route is sent past the crossing.
        if self.goal_pose is None:
            self.goal_pose = beyond
        od_spawn = self._setup_ego_spawn()

        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        # At the kerb on the side the road's traffic keeps to.
        kerb = kerb_pose(spawn, cfg, road_traffic_rule(od_spawn.road_id))
        # Not snapped to the road: snapping would put it in the carriageway.
        at_kerb = to_carla_world(kerb).to_carla_transform()
        lifted = carla.Transform(
            carla.Location(
                at_kerb.location.x,
                at_kerb.location.y,
                at_kerb.location.z + _WALKER_SPAWN_HEIGHT_M,
            ),
            at_kerb.rotation,
        )
        pedestrian = PedestrianEntity(
            PedestrianEntityConfig(
                role_name=str(PEDESTRIAN_ROLE),
                spawn_location=SpawnTransform(lifted),
                walker_type=cfg.walker_type,
            )
        )
        pedestrian.spawn(world)
        self.register_pedestrian(pedestrian)
        logger.info(
            "Pedestrian waits on lanelet %d at s=%.1f, t=%.1f m off its centre",
            spawn.lanelet_id,
            crossing_s,
            kerb.t,
        )

        self.register_pre_tick(
            WalkStraightAction(
                PEDESTRIAN_ROLE,
                speed_ms=cfg.walk_speed_ms,
                condition=EntityDistanceCondition(
                    EGO_ROLE_NAME,
                    PEDESTRIAN_ROLE,
                    cfg.trigger_distance_m,
                    label="ego_near_pedestrian",
                ),
                label="dart_out",
                turn_back_ttc_s=cfg.turn_back_ttc_s,
                turn_back_distance_m=cfg.turn_back_distance_m,
                turn_back_within_m=cfg.turn_back_within_m,
            )
        )

        self.register_pass_condition(
            StickyCondition(
                EntityPositionDistanceCondition(
                    EGO_ROLE_NAME, beyond, _PASS_RADIUS_M, label="ego_past_crossing"
                )
            )
        )
        self.register_fail_condition(
            CollisionCondition(target=PEDESTRIAN_ROLE, label="hit_pedestrian")
        )
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
