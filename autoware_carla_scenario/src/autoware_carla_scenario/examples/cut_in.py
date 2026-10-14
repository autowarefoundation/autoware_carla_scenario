"""Example scenario: a vehicle in the next lane cuts in just ahead of the ego.

An NPC starts in the lane beside the ego, a little ahead, and after a moment
changes into the ego's lane. The ego must not hit it: the run passes once the
NPC has been in the ego's lane for ``hold_seconds`` without a collision. It
fails on any collision of the ego (``fail_on_any_collision``; off, only on one
with the NPC), and when the ego leaves the lanes ahead of it -- its spawn
lanelet and the lanelets that follow it -- for ``leaving_lane_seconds``: a lane
change, a turn onto a lanelet that does not follow it, or leaving the road
(``fail_on_leaving_lane``).

``side`` is the side of the ego the NPC comes from, so it changes lanes the
other way. Its lanelet is the one beside the ego's, which a sweep binds with
the ``adjacent`` binding.

Where it starts is given one of two ways. ``npc_ahead_m`` is a raw setting:
the NPC's ``s`` along its lanelet less the ego's along its own. ``npc_gap_m``
is the gap ``vehicle_ahead_gap_m`` should read at the start: the NPC's ``s``
is solved for, from the two spawn poses, so it does
(:func:`~autoware_carla_scenario.place_at_vehicle_ahead_gap`) -- the same
wherever the scenario is placed, however the two lanelets curve.  The
scenario's controls set ``npc_gap_m``, so a gap drawn from an ODD is the gap
the run starts at.

Typical usage
-------------
On every lane of a map with a lane beside it, one case each::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=cut_in/left map=<map>
"""

from __future__ import annotations

import logging

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseCondition,
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
    NotCondition,
    OrCondition,
    PersistentCondition,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    VehicleEntity,
    VehicleEntityConfig,
    lanelets_ahead,
    place_at_vehicle_ahead_gap,
    snap_to_carla_road,
    to_carla_world,
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


def left_lanes_ahead(
    lanelet_id: int, distance_m: float, seconds: float
) -> BaseCondition:
    """The ego has been off the lanes ahead of *lanelet_id* for *seconds*.

    A composition of the framework's conditions, no condition of its own: an
    ``EntityLanePositionCondition`` for each lanelet :func:`lanelets_ahead`
    returns, under an ``OrCondition`` (the ego is on one of them), under a
    logical ``NotCondition`` (``holds_when_pending=True``: on none of them),
    held for *seconds* (``PersistentCondition``) so that a tick off at a seam
    between lanelets does not count.  On none of them is in a lane beside
    them, on a lanelet that turns off them, or on no lanelet at all -- off the
    road.  The position tested is the vehicle's centre, its CARLA location.

    *seconds* of zero or less fails on the first tick off them.
    """
    # Appended one by one, and chosen by if/else: Codon gives neither a list
    # comprehension nor a conditional expression the base class's type.
    on_lanes: list[BaseCondition] = []
    for lanelet in lanelets_ahead(lanelet_id, distance_m):
        on_lanes.append(
            EntityLanePositionCondition(
                EGO_ROLE_NAME,
                Lanelet2Pose(lanelet_id=lanelet, s=0.0),
                label=f"ego_on_lanelet_{lanelet}",
            )
        )
    on_any: BaseCondition
    if len(on_lanes) == 1:
        on_any = on_lanes[0]
    else:
        on_any = OrCondition(on_lanes, label="ego_on_its_lanes")
    if seconds <= 0.0:
        return NotCondition(on_any, label="ego_left_its_lane", holds_when_pending=True)
    return PersistentCondition(
        NotCondition(on_any, label="ego_off_its_lanes", holds_when_pending=True),
        seconds,
        label="ego_left_its_lane",
    )


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

        npc_gap_m = cfg.npc_gap_m
        if npc_gap_m is None:
            npc_pose = Lanelet2Pose(
                lanelet_id=cfg.npc_lanelet_id, s=spawn.s + cfg.npc_ahead_m
            )
        else:
            # The gap as the measure reads it, from where the ego spawns: x, y
            # and heading come off its lanelet, as _setup_ego_spawn places it.
            ego_pose: Lanelet2Pose = spawn
            ego_at = to_carla_world(ego_pose).to_carla_transform()
            npc_pose = place_at_vehicle_ahead_gap(ego_at, cfg.npc_lanelet_id, npc_gap_m)
        npc_snapped = snap_to_carla_road(
            npc_pose, world, ground_projection=self._ground_projection
        )
        npc = VehicleEntity(
            VehicleEntityConfig(
                role_name=str(CUT_IN_ROLE),
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
            "Cut-in NPC on lanelet %d at s=%.2f, %s of the ego",
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
        if cfg.fail_on_any_collision:
            self.register_fail_condition(CollisionCondition(label="ego_collided"))
        else:
            self.register_fail_condition(
                CollisionCondition(target=CUT_IN_ROLE, label="hit_cut_in_vehicle")
            )
        if cfg.fail_on_leaving_lane:
            self.register_fail_condition(
                left_lanes_ahead(
                    spawn.lanelet_id, cfg.lanes_ahead_m, cfg.leaving_lane_seconds
                )
            )
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
