"""Example scenario: a pedestrian darts out into the road in front of the ego.

The ego drives along a lane at the edge of the road. A pedestrian waits at the
kerb some way ahead and, once the ego is close, runs out into the lane. The ego
must not hit them: it passes once it is past the point where they crossed, and
fails on a collision with them.

The pedestrian stands ``pedestrian_offset_m`` from the centre of the ego's
lanelet (negative is to the right of the direction of travel, the kerb of the
rightmost lane) and runs across it, towards the lane's centre and beyond.

Typical usage
-------------
On every lane at the road's edge of a map, one case each::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=pedestrian_dart_out/pedestrian_dart_out map=<map>
"""

from __future__ import annotations

import logging
import math

import carla

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
        self._setup_ego_spawn()

        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        # Facing across the lane: a quarter turn to the left of the lane's
        # direction from the right-hand kerb, to the right from the left one.
        across = math.pi / 2 if cfg.pedestrian_offset_m < 0 else -math.pi / 2
        kerb = Lanelet2Pose(
            lanelet_id=spawn.lanelet_id,
            s=crossing_s,
            t=cfg.pedestrian_offset_m,
            heading=across,
        )
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
                role_name=PEDESTRIAN_ROLE,
                spawn_location=SpawnTransform(lifted),
                walker_type=cfg.walker_type,
            )
        )
        pedestrian.spawn(world)
        self.register_pedestrian(pedestrian)
        logger.info(
            "Pedestrian waits on lanelet %d at s=%.1f, %.1f m off its centre",
            spawn.lanelet_id,
            crossing_s,
            cfg.pedestrian_offset_m,
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
