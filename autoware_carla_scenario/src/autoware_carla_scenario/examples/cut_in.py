"""Example scenario: a vehicle in the next lane cuts in just ahead of the ego.

An NPC starts in the lane beside the ego, a little ahead, and after a moment
changes into the ego's lane. The ego must not hit it: the run passes once the
NPC has been in the ego's lane for ``hold_seconds`` without a collision, and
fails on a collision with it.

``side`` is the side of the ego the NPC comes from, so it changes lanes the
other way. Its lanelet is the one beside the ego's, which a sweep binds with
the ``adjacent`` binding.

Typical usage
-------------
On every lane of a map with a lane beside it, one case each::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=cut_in/left map=<map>
"""

from __future__ import annotations

import logging

import carla

from autoware_carla_scenario import (
    BaseScenario,
    CollisionCondition,
    EgoConfig,
    ElapsedTimeCondition,
    EntityLanePositionCondition,
    EntityRole,
    GroundProjectionConfig,
    LaneChangeAction,
    LaneChangeDirection,
    Lanelet2Pose,
    PersistentCondition,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    VehicleEntity,
    VehicleEntityConfig,
    snap_to_carla_road,
    to_opendrive,
)

from .configs import CutInConfig

logger = logging.getLogger(__name__)

#: The role of the vehicle that cuts in.
CUT_IN_ROLE = EntityRole.npc(1)

#: The way the NPC changes lanes, by the side it comes from.
_CHANGE_TOWARDS_EGO: dict[str, LaneChangeDirection] = {
    "left": LaneChangeDirection.RIGHT,
    "right": LaneChangeDirection.LEFT,
}


class CutInScenario(BaseScenario):
    """Spawn the ego and an NPC beside it; the NPC changes into the ego's lane."""

    _config: CutInConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: CutInConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or CutInConfig()

    def setup(self) -> None:
        """Spawn the NPC beside the ego, and register its lane change and the conditions."""
        world = self.world
        cfg = self._config
        spawn = self._spawn_pose
        if spawn is None:
            raise ValueError("cut_in needs an ego spawn pose")
        if cfg.side not in _CHANGE_TOWARDS_EGO:
            raise ValueError(f"cut_in side must be 'left' or 'right', got {cfg.side!r}")

        ego_lane = self._setup_ego_spawn()
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        npc_pose = Lanelet2Pose(
            lanelet_id=cfg.npc_lanelet_id, s=spawn.s + cfg.npc_ahead_m
        )
        npc_snapped = snap_to_carla_road(
            npc_pose, world, ground_projection=self._ground_projection
        )
        npc = VehicleEntity(
            VehicleEntityConfig(
                role_name=CUT_IN_ROLE,
                spawn_location=SpawnTransform(npc_snapped.to_carla_transform()),
                vehicle_type=cfg.npc_vehicle_type,
                initial_speed_kmh=cfg.npc_initial_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                od_pose=to_opendrive(npc_pose),
                ground_projection=self._ground_projection,
            )
        )
        npc.spawn(world)
        self.register_entity(npc)
        logger.info(
            "Cut-in NPC on lanelet %d at s=%.1f, %s of the ego",
            cfg.npc_lanelet_id,
            npc_pose.s,
            cfg.side,
        )

        self.register_pre_tick(
            LaneChangeAction(
                CUT_IN_ROLE,
                _CHANGE_TOWARDS_EGO[cfg.side],
                condition=ElapsedTimeCondition(
                    cfg.cut_in_after_seconds, label="cut_in_time"
                ),
                label="cut_in",
            )
        )

        # Latched once the NPC is on the ego's OpenDRIVE lane, as lane_change
        # checks its ego; then held for hold_seconds with no collision.
        cut_in = StickyCondition(
            EntityLanePositionCondition(CUT_IN_ROLE, ego_lane, label="npc_in_ego_lane")
        )
        self.register_pass_condition(
            PersistentCondition(cut_in, cfg.hold_seconds, label="npc_cut_in_held")
        )
        self.register_fail_condition(
            CollisionCondition(target=CUT_IN_ROLE, label="hit_cut_in_vehicle")
        )
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
