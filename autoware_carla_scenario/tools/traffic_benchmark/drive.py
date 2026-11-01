"""The benchmark's scenario: the ego driving on its own for a fixed time.

Nothing happens in it but the traffic: the ego is driven by the
TrafficManager from its spawn, and the run passes after ``seconds``, so
every configuration of the benchmark runs exactly as long.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    EntityDistanceCondition,
    EntityRole,
    SetSpeedAction,
    SpawnTransform,
    VehicleEntity,
    VehicleEntityConfig,
    snap_to_carla_road,
    to_opendrive,
    EgoConfig,
    ElapsedTimeCondition,
    GroundProjectionConfig,
    Lanelet2Pose,
    TimeoutCondition,
)


@dataclass
class DriveConfig:
    name: str = "traffic_benchmark"
    #: How long the run lasts (s).
    seconds: float = 120.0
    #: A scenario vehicle (npc1) too, this far ahead of the ego on its
    #: lanelet; None: none. A condition reads it every tick, so a run
    #: that lets it leave CARLA (mesoscopic.scenario_vehicles) shows the
    #: lookup warning.
    npc_ahead_m: float | None = None
    npc_speed_kmh: float = 40.0
    #: Hold the ego at this speed (km/h); None: the TrafficManager's.
    ego_speed_kmh: float | None = None


class DriveScenario(BaseScenario):
    """The ego, driving; the run passes after ``seconds``."""

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: DriveConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or DriveConfig()

    def setup(self) -> None:
        self._setup_ego_spawn()
        cfg = self._config
        spawn = self._spawn_pose
        if cfg.ego_speed_kmh is not None:
            self.register_pre_tick(
                SetSpeedAction(EGO_ROLE_NAME, cfg.ego_speed_kmh, label="ego_speed")
            )
        if cfg.npc_ahead_m is not None and spawn is not None:
            pose = Lanelet2Pose(
                lanelet_id=spawn.lanelet_id, s=spawn.s + cfg.npc_ahead_m
            )
            snapped = snap_to_carla_road(
                pose, self.world, ground_projection=self._ground_projection
            )
            npc = VehicleEntity(
                VehicleEntityConfig(
                    role_name="npc1",
                    spawn_location=SpawnTransform(snapped.to_carla_transform()),
                    initial_speed_kmh=cfg.npc_speed_kmh,
                    od_pose=to_opendrive(pose),
                    ground_projection=self._ground_projection,
                )
            )
            npc.spawn(self.world)
            self.register_entity(npc)
            # Never holds (a distance below zero); looks npc1 up every tick.
            self.register_fail_condition(
                EntityDistanceCondition(
                    EGO_ROLE_NAME, EntityRole("npc1"), -1.0, label="npc1_lookup"
                )
            )
        self.register_pass_condition(
            ElapsedTimeCondition(self._config.seconds, label="benchmark_over")
        )
        # Its own, past the end: the runner's default would stop it first.
        self.register_fail_condition(
            TimeoutCondition(self._config.seconds + 60.0, label="benchmark_timeout")
        )

    def is_done(self) -> bool:
        return False
