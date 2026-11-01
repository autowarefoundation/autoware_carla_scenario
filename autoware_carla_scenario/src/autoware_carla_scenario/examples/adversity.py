"""Example scenarios: TeraSim's adversities carried out in CARLA (#404).

One case per phase-1 adversity, each with a road user placed so that the
adversity's own trigger fires (its ``AdversityFeasibleCondition``, the
action's default), and one of NADE among background traffic:

* ``brake_check``: an NPC ahead of the ego, which follows it, brakes hard
  (``AdversarialBrakeAction``).
* ``ignore_leader``: an NPC behind a slow ego closes in on it as if it were
  not there (``IgnoreLeaderAction``).
* ``cut_in``: an NPC a little ahead in the lane beside changes in front of
  the ego, close behind there (``AdversarialCutInAction``).
* ``traffic_rule``: with every light red, an NPC ahead of the ego, stopping
  for its light, runs it (``RunTrafficRuleAction``).
* ``jaywalking``: a pedestrian at the kerb ahead walks across the road, off
  any crosswalk (``JaywalkAction``).
* ``nade``: NADE (``AdversarialTrafficAction``) over the five kinds among the
  TrafficManager's background vehicles, sampling far more often than
  TeraSim's numbers would, so that a short run shows several.

The ego is driven by the traffic manager. A single adversity's run passes
once it has been carried out and its road user handed back to traffic, and
fails if that has not happened by ``timeout_seconds``; NADE's passes after
``nade_seconds``, its weight and the adversities it sampled in the run's log.

Typical usage
-------------
The shipped cases are placed on Town10HD_Opt::

    uv run scenario scenario=adversity/cut_in map=town10hd_opt
"""

from __future__ import annotations

import logging
import math

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    ActionStateCondition,
    AdversarialBrakeAction,
    AdversarialCutInAction,
    AdversarialTrafficAction,
    AdversityKind,
    AdversityLocation,
    BaseAction,
    BaseScenario,
    EgoConfig,
    ElapsedTimeCondition,
    EntityRole,
    GroundProjectionConfig,
    IgnoreLeaderAction,
    JaywalkAction,
    Lanelet2Pose,
    NadeAdversity,
    NadeSettings,
    PedestrianEntity,
    PedestrianEntityConfig,
    RunTrafficRuleAction,
    SetSpeedAction,
    SpawnTransform,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    VehicleEntity,
    VehicleEntityConfig,
    snap_to_carla_road,
    to_carla_world,
    to_opendrive,
)

from .configs import AdversityConfig

logger = logging.getLogger(__name__)

#: The NPC that carries out a vehicle's adversity.
ADVERSARY_ROLE = EntityRole.npc(1)

#: The pedestrian that jaywalks.
PEDESTRIAN_ROLE = EntityRole("pedestrian1")

#: The kinds an NPC carries out; ``jaywalking`` is a pedestrian's, ``nade``
#: NADE's.
VEHICLE_KINDS = ("brake_check", "ignore_leader", "cut_in", "traffic_rule")

#: How far above the ground a pedestrian's spawn point is (m): CARLA places a
#: walker by its middle, so one at the road's height has its feet in the kerb.
_WALKER_SPAWN_HEIGHT_M = 1.0


class AdversityScenario(BaseScenario):
    """Spawn the ego, and a road user that carries out an adversity at it --
    or NADE among background traffic."""

    _config: AdversityConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: AdversityConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or AdversityConfig()

    def setup(self) -> None:
        """Spawn what carries the adversity out, and register it and the conditions."""
        cfg = self._config
        spawn = self._spawn_pose
        if spawn is None:
            raise ValueError("adversity needs an ego spawn pose")
        known = cfg.kind in VEHICLE_KINDS or cfg.kind in ("jaywalking", "nade")
        if not known:
            raise ValueError(
                f"adversity kind must be one of {', '.join(VEHICLE_KINDS)}, "
                f"jaywalking or nade, not {cfg.kind!r}"
            )
        self._setup_ego_spawn()
        light = carla.TrafficLightState.Green
        if cfg.red_light:
            light = carla.TrafficLightState.Red
        self.register_init(
            TrafficSignalAction(
                state=light,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_lights",
            )
        )
        ego_speed = cfg.ego_speed_kmh
        if ego_speed is not None:
            self.register_pre_tick(
                SetSpeedAction(EGO_ROLE_NAME, ego_speed, label="ego_speed")
            )
        if cfg.kind == "nade":
            self._setup_nade()
            return
        action = self._adversity(spawn)
        self.register_pre_tick(action)
        # Carried out, and the road user handed back to traffic.
        actions: dict[str, BaseAction] = {}
        actions[cfg.kind] = action
        self.register_pass_condition(
            ActionStateCondition(
                cfg.kind,
                "completeState",
                actions,
                label=f"{cfg.kind}_carried_out",
            )
        )
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )
        logger.info("Adversity %s by '%s'", cfg.kind, self._actor_of(cfg.kind))

    def _adversity(self, spawn: Lanelet2Pose) -> BaseAction:
        """Spawn the road user that carries the adversity out; its action, by
        kind.  An NPC's takes it over from the speed it was held at."""
        kind = self._config.kind
        if kind == "jaywalking":
            self._spawn_pedestrian(spawn)
            jaywalk: BaseAction = JaywalkAction(
                PEDESTRIAN_ROLE, AdversityLocation.ANY, label=kind
            )
            return jaywalk
        self._spawn_npc(spawn)
        if kind == "brake_check":
            brake: BaseAction = AdversarialBrakeAction(
                ADVERSARY_ROLE, AdversityLocation.ANY, label=kind, overrides=True
            )
            return brake
        if kind == "ignore_leader":
            ignore: BaseAction = IgnoreLeaderAction(
                ADVERSARY_ROLE, AdversityLocation.ANY, label=kind, overrides=True
            )
            return ignore
        if kind == "cut_in":
            cut_in: BaseAction = AdversarialCutInAction(
                ADVERSARY_ROLE, AdversityLocation.ANY, label=kind, overrides=True
            )
            return cut_in
        rule: BaseAction = RunTrafficRuleAction(
            ADVERSARY_ROLE, AdversityLocation.ANY, label=kind, overrides=True
        )
        return rule

    def _spawn_npc(self, spawn: Lanelet2Pose) -> None:
        """The NPC on its lanelet, ``npc_ahead_m`` on from the ego, held at
        its speed."""
        world = self.world
        cfg = self._config
        pose = Lanelet2Pose(lanelet_id=cfg.npc_lanelet_id, s=spawn.s + cfg.npc_ahead_m)
        snapped = snap_to_carla_road(
            pose, world, ground_projection=self._ground_projection
        )
        npc = VehicleEntity(
            VehicleEntityConfig(
                role_name=str(ADVERSARY_ROLE),
                spawn_location=SpawnTransform(snapped.to_carla_transform()),
                vehicle_type=cfg.npc_vehicle_type,
                initial_speed_kmh=cfg.npc_speed_kmh,
                spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
                spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
                spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
                od_pose=to_opendrive(pose),
                ground_projection=self._ground_projection,
            )
        )
        npc.spawn(world)
        self.register_entity(npc)
        self.register_pre_tick(
            SetSpeedAction(ADVERSARY_ROLE, cfg.npc_speed_kmh, label="npc_speed")
        )

    def _spawn_pedestrian(self, spawn: Lanelet2Pose) -> None:
        """The pedestrian at the kerb, ``npc_ahead_m`` on from the ego, facing
        across the road."""
        world = self.world
        cfg = self._config
        across = math.pi / 2
        if cfg.npc_offset_m > 0.0:
            across = -math.pi / 2
        kerb = to_carla_world(
            Lanelet2Pose(
                lanelet_id=cfg.npc_lanelet_id,
                s=spawn.s + cfg.npc_ahead_m,
                t=cfg.npc_offset_m,
                heading=across,
            )
        ).to_carla_transform()
        lifted = carla.Transform(
            carla.Location(
                kerb.location.x,
                kerb.location.y,
                kerb.location.z + _WALKER_SPAWN_HEIGHT_M,
            ),
            kerb.rotation,
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

    def _setup_nade(self) -> None:
        """NADE over the five kinds, anywhere, for ``nade_seconds``."""
        cfg = self._config
        adversities: list[NadeAdversity] = []
        for kind in (
            AdversityKind.CUT_IN,
            AdversityKind.BRAKE_CHECK,
            AdversityKind.IGNORE_LEADER,
            AdversityKind.TRAFFIC_RULE,
            AdversityKind.JAYWALKING,
        ):
            adversities.append(
                NadeAdversity(kind, AdversityLocation.ANY, cfg.nade_probability)
            )
        self.register_pre_tick(
            AdversarialTrafficAction(
                adversities,
                label="nade",
                seed=cfg.nade_seed,
                settings=NadeSettings(
                    fixed_is_probability=cfg.nade_is_probability,
                    cooldown=cfg.nade_cooldown,
                    challenge_npcs=cfg.nade_challenge_npcs,
                ),
            )
        )
        self.register_pass_condition(
            ElapsedTimeCondition(cfg.nade_seconds, label="nade_over")
        )
        logger.info(
            "NADE for %.0f s: p=%.3g, q=%.3g, cooldown %.1f s",
            cfg.nade_seconds,
            cfg.nade_probability,
            cfg.nade_is_probability,
            cfg.nade_cooldown,
        )

    def _actor_of(self, kind: str) -> str:
        if kind == "jaywalking":
            return str(PEDESTRIAN_ROLE)
        return str(ADVERSARY_ROLE)

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
