"""Ego vehicle driven by an external policy over the ``egodriver`` gRPC contract.

Where :class:`~autoware_carla_scenario.entity.ego.EgoVehicle` hands the vehicle to
CARLA's TrafficManager, this entity closes the loop against a driver policy served with
``carla-driver-interface`` or any other implementation of ``egodriver.EgodriverService``.

Each simulation tick the entity applies control; every
:attr:`~autoware_carla_scenario.driver.base.DriverClientConfig.policy_timestep_s` it also
submits fresh observations and asks the policy to re-plan.  Between policy steps the most
recent plan keeps being tracked, which is how the upstream runtime handles a policy that
runs slower than the simulation.  The turn indicators the policy asks for are lit on the
ego and held until it asks for others.

Given a ``bridge``, the entity also holds the splatsim-consistent
:class:`~autoware_carla_scenario.entity.bridged_ego.BridgeHandshake`, for a stack that
serves the policy's world -- one that renders what its cameras see, say -- and must come
up before the policy can drive: the stack pulls the scenario's mission over the bridge,
and the policy starts (its warm-up included) only once the stack reports ready.  Until
then the entity is not initialized, so the runner holds the scenario clock.
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..constants import FIXED_DELTA_SECONDS
from ..debug_markers import DebugMarkers
from ..coordinate.map_manager import run_carla_map
from ..driver.base import BaseEgoDriverClient, DriverClientConfig, EgoObservation
from ..driver.control import ControlConfig, TrajectoryFollower
from ..driver.egodriver_client import EgoDriverGrpcClient
from ..driver.geometry import Pose, Trajectory
from ..ego_plan import ego_plan_markers
from ..utils.powertrain import ChaosPowertrain
from ..utils.turn_indicator import (
    TurnIndicatorState,
    set_turn_indicators,
    turn_indicators_of,
)
from ..driver.renderer import RendererDataBuilder
from ..driver.observation import (
    AccelerationEstimate,
    ego_observation,
    encode_frame_jpeg,
    lidar_points_to_rig,
    rear_axle_offset,
    route_reference_trajectory,
    route_waypoints_in_rig,
    sensor_pose_in_rig,
)
from carla_driver_interface.driver import TurnIndicator
from carla_driver_interface.protocol import LidarSweep, pack_lidar_sweep
from ..rerun_lidar import CameraMount, active_recorder
from ..ego_base_link import clear_ego_base_link, set_ego_base_link
from ..t4 import active_recording, offer_driver_sensors, withdraw_driver_sensors
from ._policy_warmup import PolicyWarmup, _CarriedActor
from .bridged_ego import BridgeHandshake
from .externally_driven import ExternallyDrivenEgo

if TYPE_CHECKING:
    import numpy as np
    import typesafe_carla.carla as carla

    from ..autoware_bridge.base import AutowareBridge, AutowareBridgeConfig
    from ..autoware_stack.launcher import AutowareLauncher
    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose

    from ..sensor.carla_camera import CarlaCameraSensor
    from ..rerun_lidar import SensorRecorder
    from ..sensor.carla_lidar import CarlaLidarSensor
    from ..t4_scene.live import LiveT4Scene


logger = logging.getLogger(__name__)

#: Simulation step used to convert ticks to simulation time -- the same one
#: :class:`ScenarioRunner` applies to the world, read from where it is named.
_FIXED_DELTA_S: float = FIXED_DELTA_SECONDS

#: Microseconds per second, for the contract's ``fixed64`` timestamps.
_US_PER_S: int = 1_000_000

#: Log the policy's own diagnostics every N policy steps (~1 s at 10 Hz).
_DEBUG_LOG_INTERVAL: int = 10

#: What each turn-indicator command a policy can give shows on the ego.
_SHOWN: Dict[TurnIndicator, TurnIndicatorState] = {
    TurnIndicator.DISABLE: TurnIndicatorState.OFF,
    TurnIndicator.ENABLE_LEFT: TurnIndicatorState.LEFT,
    TurnIndicator.ENABLE_RIGHT: TurnIndicatorState.RIGHT,
}


class CarlaDriverEntity(ExternallyDrivenEgo):
    """Ego vehicle controlled by an external driver policy.

    Args:
        config: Driver connection settings.  ``None`` uses the defaults, which dial
            ``localhost:50051``.
        control_config: Gains for the trajectory follower.  ``None`` uses the defaults.
        client: Pre-built driver client, used by tests to substitute a fake or an
            in-process gRPC channel.  ``None`` builds an
            :class:`~autoware_carla_scenario.driver.egodriver_client.EgoDriverGrpcClient`
            from *config*.
        bridge: The bridge a stack serving the policy's world dials, for the
            mission and readiness handshake.  ``None`` (the default) runs the
            policy alone, from the first tick.
        bridge_config: Bridge connection settings.  ``None`` uses the defaults.
        launcher: Starts that stack when the scenario starts and stops it when it
            ends.  Needs a *bridge*.
        episode_name: The scenario's name, for the launcher's logs.

    Raises:
        ValueError: If a *launcher* is given without a *bridge*.
    """

    stack_name: str = "driver policy"

    def __init__(
        self,
        config: Optional[DriverClientConfig] = None,
        control_config: Optional[ControlConfig] = None,
        client: Optional[BaseEgoDriverClient] = None,
        *,
        bridge: Optional["AutowareBridge"] = None,
        bridge_config: Optional["AutowareBridgeConfig"] = None,
        launcher: Optional["AutowareLauncher"] = None,
        episode_name: str = "",
    ) -> None:
        super().__init__()
        if launcher is not None and bridge is None:
            raise ValueError(
                "CarlaDriverEntity: a launcher starts a stack that dials the bridge, "
                "so it needs a bridge"
            )
        #: The handshake with the stack serving the policy's world, if any.
        self._handshake: Optional[BridgeHandshake] = None
        if bridge is not None:
            self._handshake = BridgeHandshake(
                bridge_config,
                bridge,
                self.stack_name,
                launcher=launcher,
                episode_name=episode_name,
            )
            # That stack plans from a mission, and the mission needs a goal.
            self.requires_goal = True
        self._config = config or DriverClientConfig()
        self._client = (
            client if client is not None else EgoDriverGrpcClient(self._config)
        )
        self._follower = TrajectoryFollower(control_config)

        self._cameras: List[Tuple[str, "CarlaCameraSensor"]] = []
        #: ``(logical_id, sensor, mount in the rig frame)`` per LiDAR.
        self._lidars: List[Tuple[str, "CarlaLidarSensor", Pose]] = []
        #: The frame of the sweep last written into the LiDAR recording, by LiDAR.
        self._recorded_lidar: Dict[str, int] = {}
        #: The ego's pose (the rig's, in the local frame) on the last ticks, by
        #: simulator frame, kept while there are LiDARs and only as far back as
        #: their longest sweep interval: a sweep is recorded on the pose of the
        #: tick it was taken on, which can be before the policy step (#950).
        self._pose_history: Dict[int, Pose] = {}
        #: Each camera's mount, for the run's recording (``rerun.cameras``).
        self._camera_mounts: Dict[str, CameraMount] = {}
        self._world: Optional["carla.World"] = None
        self._plan: Trajectory = Trajectory.empty()
        #: The plan as debug markers, built once per plan.
        self._plan_markers: DebugMarkers = DebugMarkers()
        self._sim_time_us: int = 0
        self._last_policy_time_us: Optional[int] = None
        #: Every tick's ego state since the last policy step, sent as one egomotion
        #: history -- as the upstream runtime does -- rather than only the newest.
        self._pending_egomotion: List[EgoObservation] = []
        self._rear_axle_offset_m: float = 0.0
        self._session_open: bool = False
        self._drive_count: int = 0
        self._map: Optional["carla.Map"] = None
        self._renderer: Optional[RendererDataBuilder] = None
        #: The speed each vehicle starts the scenario at, by actor id.
        self._initial_speeds: Dict[int, float] = {}
        #: The run-up onto the spawn pose while it lasts (``warmup_s``).
        self._warmup: Optional[PolicyWarmup] = None
        #: What the policy is told the ego accelerates at: Autoware's estimate,
        #: not CARLA's per-step one.
        self._acceleration = AccelerationEstimate()
        #: The turn indicators last lit at the policy's request; ``NO_COMMAND``
        #: until it asks for any.
        self._turn_indicator: TurnIndicator = TurnIndicator.NO_COMMAND
        #: The OpenDRIVE roads of the route to the scenario's goal, which pick the
        #: way the policy's road ahead takes at a fork; ``None`` without a goal.
        self._route_roads: Optional[Set[str]] = None
        #: The road ahead last sent: the ego's pose then and the rig-frame points.
        self._road_ahead: Optional[Tuple[Pose, Any]] = None
        #: The goal as ``(x, y, yaw)`` in Autoware's ``map`` frame, a T4Scene's
        #: (``t4_scene_goal: scenario``).
        self._goal_in_map: Optional[Tuple[float, float, float]] = None
        #: The run so far as T4Scenes (``driver.t4_scene``), while one runs.
        self._t4: Optional["LiveT4Scene"] = None
        self._scene_id: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def turn_indicator(self) -> TurnIndicator:
        """The turn indicators lit at the policy's request (``NO_COMMAND``: none yet)."""
        return self._turn_indicator

    @property
    def config(self) -> DriverClientConfig:
        """Return the driver connection settings."""
        return self._config

    @property
    def client(self) -> BaseEgoDriverClient:
        """Return the driver client."""
        return self._client

    @property
    def handshake(self) -> Optional[BridgeHandshake]:
        """The handshake with the stack serving the policy's world, if there is one."""
        return self._handshake

    @property
    def is_initialized(self) -> bool:
        """``False`` until the bridged stack is ready and the run-up is over."""
        if self._handshake is not None and not self._handshake.ready:
            return False
        return self._warmup is None

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: Optional["CarlaWorldPose"] = None,
        ground_projection: Optional["GroundProjectionConfig"] = None,
        waypoints: Sequence["Lanelet2Pose"] = (),
    ) -> None:
        """Hand *goal* to the bridged stack's mission, and route the road ahead.

        The policy itself is given the road ahead rather than a destination, and
        the road ahead turns at a fork the way the route to *goal* (through
        *waypoints*) does: a turn the scenario asserts is the one the policy is
        shown.  Without a route there -- no lane route reaches the goal -- it
        takes the first way, as it does without a goal.
        """
        start = initial_pose
        if start is None and self.actor is not None:
            from ..coordinate import CarlaWorldPose  # noqa: PLC0415

            start = CarlaWorldPose.from_carla_transform(self.actor.get_transform())
        if start is not None:
            lanelet_ids = _route_lanelets(start, [*waypoints, goal])
            self._route_roads = None if lanelet_ids is None else _roads_of(lanelet_ids)
        self._goal_in_map = _in_map_frame(goal)
        if self._handshake is not None:
            self._handshake.route_to(
                world, goal, initial_pose, ground_projection, waypoints
            )

    @property
    def carried_actor_ids(self) -> FrozenSet[int]:
        """What the run-up is moving, which the init phase's brakes must spare."""
        return (
            self._warmup.carried_actor_ids if self._warmup is not None else frozenset()
        )

    def set_initial_speeds(self, speeds_mps: Mapping[int, float]) -> None:
        """Keep each vehicle's initial speed, for the run-up to arrive at."""
        self._initial_speeds = dict(speeds_mps)

    @property
    def drive_count(self) -> int:
        """Number of ``drive`` calls made so far."""
        return self._drive_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Attach the cameras, open the driver session, and send the initial route.

        Raises:
            RuntimeError: If the ego actor has not been spawned yet.
        """
        actor = self.actor
        if actor is None:
            raise RuntimeError(
                "CarlaDriverEntity.on_scenario_start called before spawn()"
            )
        # A retried scenario runs this entity again: the policy's request to end
        # the attempt before does not end this one.
        self._reset_termination()
        # Nor does its clock or bookkeeping carry over: the new session starts
        # at simulation time 0 and asks the policy on its first tick, as the
        # first attempt did.
        self._sim_time_us = 0
        self._last_policy_time_us = None
        self._drive_count = 0
        self._pending_egomotion = []
        self._recorded_lidar = {}
        self._pose_history = {}
        self._camera_mounts = {}
        self._plan_markers = DebugMarkers()
        if self._handshake is not None:
            # Before anything moves the ego: the mission is checked against
            # where it stands.
            self._handshake.start(actor, type(self).__name__)

        self._rear_axle_offset_m = rear_axle_offset(
            actor, self._config.rear_axle_offset_m
        )
        logger.info(
            "Rear-axle offset for the rig origin: %.2f m", self._rear_axle_offset_m
        )
        set_ego_base_link(self._rear_axle_offset_m)

        self._attach_cameras(world, actor)
        self._world = world
        self._acceleration.reset()
        self._turn_indicator = TurnIndicator.NO_COMMAND
        # The ego drives on CARLA's own (Chaos) vehicle physics: this entity puts it
        # on no other, so its pedals are worked out from that model.
        self._follower.powertrain = ChaosPowertrain.from_physics_control(
            actor.get_physics_control()
        )
        # A new session: nothing of a previous run's plans or control state.
        self._follower.reset()
        self._plan = Trajectory.empty()

        # CARLA rebuilds the map object on every ``get_map()`` call, so the run's
        # one is taken (the MapManager's, when it holds one) and reused for the
        # rolling route walk.
        self._map = run_carla_map(world)

        if self._config.t4_scene:
            self._t4 = _live_t4_scene(self._config, self._rear_axle_offset_m)
            self._t4.start(world)
        if self._config.send_renderer_data:
            self._attach_lidars(world, actor)
            self._renderer = RendererDataBuilder(
                world, actor, self._config, map_id=self._write_map()
            )
        elif self._config.lidars or self._config.map_dir:
            logger.warning(
                "driver.lidars / driver.map_dir need send_renderer_data, which is off; "
                "the policy gets neither sweeps nor traffic lights"
            )
        # A T4 dataset of the run (t4.enabled) takes what the policy is sent
        # rather than rendering it again.
        offer_driver_sensors(self._rear_axle_offset_m)

        session_uuid = str(uuid.uuid4())
        #: What the run's T4Scenes are named: the session's.
        self._scene_id = session_uuid
        try:
            self._client.start_session(
                session_uuid,
                str(self._map.name),
                rear_axle_offset_m=self._rear_axle_offset_m,
            )
        except BaseException:
            # No session, so nothing feeds a dataset: do not leave the offer
            # standing for the next recording in this process.
            withdraw_driver_sensors()
            raise
        self._session_open = True

        if self._handshake is None:
            self._begin_driving(world, actor)

    def _begin_driving(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Start the run-up, or send the first route: the policy's first tick."""
        assert self._map is not None  # noqa: S101 - fetched when the session opened
        if self._config.warmup_s > 0.0:
            # The ego leaves for its run-up before the first tick: its spawn
            # pose is where it arrives, not where its history begins.  Built on a
            # world ticked since anything froze a light: it reads is_frozen()
            # from the last tick, and would take over and thaw a newer freeze.
            self._warmup = PolicyWarmup(
                world,
                actor,
                self._map,
                self._config.warmup_s,
                self._initial_speeds,
                _FIXED_DELTA_S,
            )
            self._pending_egomotion = []
            return
        observation = self._ego_observation(actor)
        self._pending_egomotion = [observation]
        self._submit_route(actor, observation)

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Advance the driver loop by one simulation tick.

        Observations and a new plan are requested every ``policy_timestep_s``; the
        current plan is tracked and applied on every tick.
        """
        actor = self.actor
        if actor is None or not self._session_open:
            return
        if self._handshake is not None and not self._await_stack(world, actor):
            return

        self._sim_time_us += int(round(_FIXED_DELTA_S * _US_PER_S))

        # The actor does not move within a tick, so one observation serves both the
        # policy step and the controller.
        observation = self._ego_observation(actor)
        self._pending_egomotion.append(observation)
        self._remember_pose(world, observation.pose)
        if self._t4 is not None:
            self._record_t4(world, elapsed, actor, observation)

        if self._is_policy_step():
            self._run_policy_step(actor, observation)

        if self._warmup is not None:
            # The policy plans, but the run-up carries the ego.
            self._warmup.advance()
            if self._warmup.done:
                self._warmup = None
                logger.info(
                    "Policy warm-up over after %d policy step(s)", self._drive_count
                )
            return
        self._apply_control(actor, observation)

    def _await_stack(self, world: "carla.World", actor: "carla.Actor") -> bool:
        """Poll the bridged stack; whether the policy may drive on this tick.

        The tick the stack turns ready starts the policy (its run-up, or its first
        route), which then drives from the next one.  A stack that fails ends the run.
        """
        assert self._handshake is not None  # noqa: S101 - checked by the caller
        was_ready = self._handshake.ready
        reason = self._handshake.poll()
        if reason is not None:
            self._terminate(reason)
            return False
        if not self._handshake.ready:
            return False
        if not was_ready:
            self._begin_driving(world, actor)
            return False
        return True

    def on_scenario_end(self, world: "carla.World") -> None:
        """Close the driver session, the bridged stack and the cameras.

        Teardown failures are logged rather than raised so that one unreachable policy
        cannot abort the rest of the scenario cleanup.
        """
        withdraw_driver_sensors()
        clear_ego_base_link()
        if self._warmup is not None:
            self._warmup.restore()
            self._warmup = None
        if self._session_open:
            try:
                self._client.close_session()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("Failed to close the driver session", exc_info=True)
            self._session_open = False
        if self._handshake is not None:
            self._handshake.close(type(self).__name__)

        for logical_id, camera in self._cameras:
            try:
                camera.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("Failed to destroy camera %s", logical_id, exc_info=True)
        self._cameras = []
        for logical_id, lidar, _ in self._lidars:
            try:
                lidar.destroy()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.warning("Failed to destroy LiDAR %s", logical_id, exc_info=True)
        self._lidars = []
        self._world = None
        self._pending_egomotion = []
        self._map = None
        self._renderer = None
        self._t4 = None
        self._road_ahead = None

        reason = self.termination_reason
        logger.info(
            "Driver session finished after %d policy step(s)%s",
            self._drive_count,
            f" ({reason})" if reason is not None else "",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _attach_cameras(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Spawn and attach every configured camera to the ego actor."""
        from ..sensor.carla_camera import CarlaCameraSensor  # noqa: PLC0415

        for logical_id, sensor_config in self._client.camera_sensor_configs():
            camera = CarlaCameraSensor(sensor_config)
            camera.attach(world, actor)
            self._cameras.append((logical_id, camera))
            self._camera_mounts[logical_id] = CameraMount(
                sensor_config.image_width,
                sensor_config.image_height,
                sensor_config.fov,
                (
                    sensor_config.position_x,
                    sensor_config.position_y,
                    sensor_config.position_z,
                ),
                (sensor_config.roll, sensor_config.pitch, sensor_config.yaw),
            )
        logger.info("Attached %d driver camera(s) to the ego", len(self._cameras))

    def _attach_lidars(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Spawn and attach every configured LiDAR to the ego actor."""
        from ..sensor.carla_lidar import CarlaLidarSensor  # noqa: PLC0415

        for config in self._config.lidars:
            # Swept only on the ticks the policy reads it: the sweeps between
            # would be ray-cast for nobody.
            lidar = CarlaLidarSensor(
                config, sweep_interval_s=self._config.policy_timestep_s
            )
            lidar.attach(world, actor)
            mount = sensor_pose_in_rig(
                config.position_x,
                config.position_y,
                config.position_z,
                config.roll,
                config.pitch,
                config.yaw,
                self._rear_axle_offset_m,
            )
            self._lidars.append((config.logical_id, lidar, mount))
        if self._lidars:
            logger.info("Attached %d driver LiDAR(s) to the ego", len(self._lidars))

    def _write_map(self) -> str:
        """Write the world's map for the policy (``map_dir``); its id, or ``""``.

        With T4Scenes, the run's Lanelet2 map they reference goes into the set
        too, as its ``t4`` format.
        """
        if self._config.map_dir is None or self._map is None:
            return ""
        from ..driver.hdmap_export import MAP_FORMATS, export_map  # noqa: PLC0415

        if self._t4 is not None and self._config.t4_scene_map == "map_set":
            # The scenes go on the set's Lanelet2 map: write the set without
            # them first, then again with the map they reference.
            plain = export_map(
                str(self._map.to_opendrive()),
                str(self._map.name),
                self._config.map_dir,
                ("lanelet2",),
                lanelet2_path=self._config.lanelet2_path,
            )
            self._t4.use_map(
                Path(self._config.map_dir) / plain / MAP_FORMATS["lanelet2"]
            )
        resource = None if self._t4 is None else self._t4.map_resource
        return export_map(
            str(self._map.to_opendrive()),
            str(self._map.name),
            self._config.map_dir,
            self._config.map_formats,
            lanelet2_path=self._config.lanelet2_path,
            t4_map=(
                None
                if resource is None
                else resource.SerializeToString(deterministic=True)
            ),
        )

    def _remember_pose(self, world: "carla.World", pose: Pose) -> None:
        """Keep the ego's *pose* on this tick, for the sweeps taken on it."""
        if not self._lidars:
            return
        frame = int(world.get_snapshot().frame)
        self._pose_history[frame] = pose
        # A sweep is at most an interval old when it is read (latest_sweep).
        oldest = frame - max(lidar.sweep_interval_ticks for _, lidar, _ in self._lidars)
        for old in [f for f in self._pose_history if f < oldest]:
            del self._pose_history[old]

    def _lidar_sweeps(self, pose: Optional[Pose] = None) -> List[LidarSweep]:
        """Every LiDAR's newest sweep, in the rig frame, stamped with when it was taken.

        A LiDAR sweeps once a policy step, which need not be this very tick: a
        sweep from an earlier tick carries that tick's time.  With *pose*, the
        ego's in the local frame, a run recording its LiDARs
        (``rerun.lidar``) is handed each new sweep too, on the ego's pose of the
        tick it was taken on, and a T4 dataset of the run is fed it as taken
        on that tick (#950).  A sweep older than the poses kept is not
        recorded (logged at debug level); the policy still gets it.
        """
        if not self._lidars or self._world is None:
            return []
        frame = int(self._world.get_snapshot().frame)
        tick_us = int(round(_FIXED_DELTA_S * _US_PER_S))
        recorder = active_recorder() if pose is not None else None
        if recorder is not None and not recorder.records_lidar:
            recorder = None
        dataset = active_recording() if pose is not None else None
        sweeps = []
        for logical_id, lidar, mount in self._lidars:
            sweep = lidar.latest_sweep(frame)
            if sweep is None:
                continue
            age_ticks = max(frame - sweep[0], 0)
            points = lidar_points_to_rig(sweep[1], mount)
            sweeps.append(
                pack_lidar_sweep(
                    logical_id,
                    self._sim_time_us - age_ticks * tick_us,
                    points,
                    mount.to_proto(),
                )
            )
            if recorder is not None and pose is not None:
                taken_pose = (
                    pose if sweep[0] == frame else self._pose_history.get(sweep[0])
                )
                if taken_pose is None:
                    logger.debug(
                        "no ego pose of frame %d for LiDAR %s's sweep; not recorded",
                        sweep[0],
                        logical_id,
                    )
                else:
                    self._record_lidar(
                        recorder, logical_id, sweep[0], points, taken_pose
                    )
            if dataset is not None:
                dataset.feed_lidar(frame, points, taken_on=sweep[0])
        return sweeps

    def _record_lidar(
        self,
        recorder: "SensorRecorder",
        logical_id: str,
        frame: int,
        points: "np.ndarray[tuple[int, int], np.dtype[np.float32]]",
        pose: Pose,
    ) -> None:
        """Write a sweep into the run's LiDAR recording, once (a sweep is read
        again on the ticks before the next)."""
        if self._recorded_lidar.get(logical_id) == frame:
            return
        self._recorded_lidar[logical_id] = frame
        try:
            recorder.log_sweep(
                logical_id, frame, points, pose.rotation_matrix, pose.position
            )
        except Exception:  # noqa: BLE001 - a lost sweep must not stop the run
            logger.exception("recording LiDAR %s's sweep failed", logical_id)

    def _ego_observation(self, actor: "carla.Actor") -> EgoObservation:
        """Return the ego's state at the current simulation time."""
        if self._warmup is not None:
            actor = _CarriedActor(actor, self._warmup, self._rear_axle_offset_m)
        return self._acceleration(
            ego_observation(actor, self._sim_time_us, self._rear_axle_offset_m)
        )

    def _submit_route(self, actor: "carla.Actor", observation: EgoObservation) -> None:
        """Send the road ahead of the ego to the policy.

        The route is a rolling window: it is re-sent on every policy step so the policy
        always sees ``route_horizon_m`` of road ahead, rather than running off the end of
        a route computed once at spawn.  With ``send_ground_truth`` it also goes out as
        the reference trajectory, CARLA having no recorded drive to offer instead.
        """
        if self._map is None:
            return
        waypoints = route_waypoints_in_rig(
            self._map,
            actor,
            observation.pose,
            horizon_m=self._config.route_horizon_m,
            resolution_m=self._config.route_resolution_m,
            route_roads=self._route_roads,
        )
        self._client.submit_route(self._sim_time_us, waypoints)
        if self._t4 is not None:
            # The T4Scene's route is this road ahead, on the map's lanelets.
            self._road_ahead = (observation.pose, waypoints)
        if self._config.send_ground_truth and len(waypoints):
            self._client.submit_recording_ground_truth(
                self._sim_time_us,
                route_reference_trajectory(
                    waypoints,
                    self._sim_time_us,
                    int(round(self._config.policy_timestep_s * _US_PER_S)),
                ),
            )

    def _is_policy_step(self) -> bool:
        """Whether this tick should query the policy."""
        if self._last_policy_time_us is None:
            return True
        interval_us = int(round(self._config.policy_timestep_s * _US_PER_S))
        return self._sim_time_us - self._last_policy_time_us >= interval_us

    def _run_policy_step(
        self, actor: "carla.Actor", observation: EgoObservation
    ) -> None:
        """Submit observations, ask the policy to plan, and cache the result."""
        self._submit_camera_frames()
        self._submit_route(actor, observation)
        self._client.submit_egomotion_observation(self._pending_egomotion)
        self._pending_egomotion = []

        query_us = self._sim_time_us + int(
            round(self._config.policy_timestep_s * _US_PER_S)
        )
        outcome = self._client.drive(
            self._sim_time_us,
            query_us,
            self._renderer_data(observation),
        )
        self._plan = outcome.trajectory
        self._drive_count += 1
        self._plan_markers = ego_plan_markers(
            outcome.trajectory,
            self._sim_time_us,
            self._drive_count,
            policy_name=outcome.policy_name,
            inference_seconds=outcome.inference_seconds,
            turn_indicator=outcome.turn_indicator,
            debug_scalars=outcome.debug_scalars,
        )
        self._last_policy_time_us = self._sim_time_us
        if outcome.debug_scalars and self._drive_count % _DEBUG_LOG_INTERVAL == 0:
            logger.info(
                "[%s] step %d: %s",
                outcome.policy_name or "driver",
                self._drive_count,
                " ".join(
                    f"{key}={value:.3f}"
                    for key, value in sorted(outcome.debug_scalars.items())
                ),
            )

        if outcome.terminate_session and not self.termination_requested:
            self._terminate("The driver policy requested session termination")

        self._show_turn_indicator(actor, outcome.turn_indicator, outcome.policy_name)

    def _show_turn_indicator(
        self, actor: "carla.Actor", requested: int, policy_name: str
    ) -> None:
        """Light the ego's turn indicators as the policy asks.

        Only the blinker bits change, so the rest of the ego's lights stay as they
        are; ``NO_COMMAND`` changes nothing.  Compared with what the ego shows,
        not with the last command: indicators a scenario set since are put back
        on the policy's next command, as ``docs/driver_interface.md`` says.
        """
        if requested == TurnIndicator.NO_COMMAND:
            return
        try:
            indicator = TurnIndicator(requested)
        except ValueError:
            logger.warning(
                "[%s] asked for turn indicator %d, which this runtime does not know; "
                "ignored",
                policy_name or "driver",
                requested,
            )
            return
        changed = indicator != self._turn_indicator
        self._turn_indicator = indicator
        if turn_indicators_of(actor) == _SHOWN[indicator]:
            return
        set_turn_indicators(actor, _SHOWN[indicator])
        if not changed:
            return
        logger.info(
            "[%s] step %d: turn indicator %s",
            policy_name or "driver",
            self._drive_count,
            indicator.name,
        )

    def _renderer_data(self, observation: EgoObservation) -> bytes:
        """Return the CARLA ground-truth payload for this policy step.

        Empty when ``send_renderer_data`` is off.  Note that a policy cannot tell
        that apart from "nothing to report": both read as no traffic light and no
        other vehicles, so turning this off quietly weakens any policy that
        reasons about them.
        """
        if self._renderer is None:
            return b""
        sweeps = self._lidar_sweeps(observation.pose)
        return self._renderer.build(
            self._sim_time_us,
            observation.pose,
            lidar=sweeps,
            t4_scene=self._t4_scene(sweeps),
        )

    def _record_t4(
        self,
        world: "carla.World",
        elapsed: float,
        actor: "carla.Actor",
        observation: EgoObservation,
    ) -> None:
        """Add the tick to the run's T4Scenes, with what the policy is told of
        the ego then."""
        from ..t4_scene.live import EgoExtras  # noqa: PLC0415

        assert self._t4 is not None  # noqa: S101 - checked by the caller
        steering: Optional[float]
        try:
            steering = _tire_steering_angle(actor)
        except Exception:  # noqa: BLE001 - the scene goes on without it
            steering = None
        self._t4.record(
            world,
            elapsed,
            EgoExtras(
                acceleration_rig=observation.linear_acceleration[:2],
                angular_velocity=observation.angular_velocity[:3],
                tire_steering_angle=steering,
                velocity_rig=observation.linear_velocity[:2],
            ),
        )

    def _t4_scene(self, sweeps: Sequence[LidarSweep]) -> bytes:
        """The run so far as a serialized T4Scene, the current sweep (the first
        LiDAR's) on its last row; empty without ``driver.t4_scene``."""
        if self._t4 is None:
            return b""
        points = None
        age_ns = 0
        if sweeps:
            import numpy as np  # noqa: PLC0415

            sweep = sweeps[0]
            points = np.frombuffer(sweep.points_xyzi, dtype="<f4").reshape(-1, 4)
            age_ns = (self._sim_time_us - int(sweep.timestamp_us)) * 1000
        sent = self._road_ahead_in_map()
        road_ahead = None if sent is None else sent[0]
        scene = self._t4.scene(
            self._scene_id,
            lidar_points=points,
            lidar_age_ns=age_ns,
            road_ahead=road_ahead,
            road_ahead_origin=None if sent is None else sent[1],
            goal_pose=(
                _far_end(road_ahead)
                if self._config.t4_scene_goal == "road_ahead"
                else self._goal_in_map
            ),
        )
        return b"" if scene is None else scene.SerializeToString()

    def _road_ahead_in_map(self) -> Any:
        """The road ahead last sent and where the ego then was, ``(N, 2)`` and
        ``(2,)`` in Autoware's ``map`` frame; ``None`` before one was.

        Both as the egodriver observations carry them -- in single precision
        -- which is what a policy reading those walked the lanelets with: two
        lanelets meeting at a join are a micrometre apart there, and which the
        walk takes turns on the rounding.
        """
        if self._road_ahead is None or not len(self._road_ahead[1]):
            return None
        import numpy as np  # noqa: PLC0415

        from ..coordinate.map_manager import MapManager  # noqa: PLC0415

        pose, waypoints = self._road_ahead
        told = Pose.from_proto(pose.to_proto())
        rig = np.asarray(waypoints, dtype=np.float32).astype(np.float64)
        # The local frame is CARLA's with y flipped; the map's adds its offset.
        offset = np.asarray(MapManager.get_instance().mgrs_offset, dtype=np.float64)
        points = told.transform_points(rig)[:, :2] + offset
        return points, told.position[:2] + offset

    def _submit_camera_frames(self) -> None:
        """Encode and send the latest frame from each camera."""
        for logical_id, camera in self._cameras:
            image = camera.get_image()
            if image is None:
                logger.debug("No frame available from camera %s", logical_id)
                continue
            encoded = encode_frame_jpeg(image, self._config.image_quality)
            if encoded is None:
                continue
            self._client.submit_image_observation(
                logical_id,
                self._sim_time_us,
                self._sim_time_us,
                encoded,
            )
            self._record_camera(logical_id, encoded)
            dataset = active_recording()
            if dataset is not None and self._world is not None:
                dataset.feed_camera_jpeg(
                    int(self._world.get_snapshot().frame), logical_id, encoded
                )

    def _record_camera(self, logical_id: str, encoded: bytes) -> None:
        """Write a frame the policy got into the run's recording, when it
        records its cameras (``rerun.cameras``)."""
        recorder = active_recorder()
        mount = self._camera_mounts.get(logical_id)
        if recorder is None or not recorder.records_cameras or mount is None:
            return
        if self._world is None:
            return
        try:
            frame = int(self._world.get_snapshot().frame)
            recorder.log_camera_frame(logical_id, frame, encoded, mount)
        except Exception:  # noqa: BLE001 - a lost frame must not stop the run
            logger.exception("recording camera %s's frame failed", logical_id)

    def _apply_control(self, actor: "carla.Actor", observation: EgoObservation) -> None:
        """Track the cached plan and push the resulting control to CARLA."""
        command = self._follower.step(
            self._plan,
            observation.pose,
            observation.speed_mps,
            _FIXED_DELTA_S,
            yaw_rate_rps=float(observation.angular_velocity[2]),
            now_us=observation.timestamp_us,
            gear=int(actor.get_control().gear),
        )
        actor.apply_control(command.to_carla_control())

    def debug_markers(self) -> DebugMarkers:
        """The policy's current plan, as a line along the road, carrying the
        plan itself (:class:`~autoware_carla_scenario.ego_plan.PlanMarkers`)."""
        return self._plan_markers

    def debug_state(self) -> Dict[str, float]:
        """Return a snapshot of the loop's state, for logging and tests."""
        return {
            "sim_time_us": float(self._sim_time_us),
            "drive_count": float(self._drive_count),
            "plan_length": float(len(self._plan)),
            "rear_axle_offset_m": self._rear_axle_offset_m,
            "turn_indicator": float(self._turn_indicator),
        }


def _route_lanelets(
    start: "CarlaWorldPose", stops: Sequence["Lanelet2Pose"]
) -> Optional[List[int]]:
    """The lanelets of the lane route from *start* through *stops*, in order.

    They are :func:`~autoware_carla_scenario.odd.route_plan.route_lanelets`'s,
    the route the ODD plans for the scenario.  ``None``, said once, when a stop
    is on no lane route from the one before or the map cannot answer.
    """
    from ..coordinate import to_lanelet2  # noqa: PLC0415
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..odd.route_plan import PlannedRoute, RouteError, route_lanelets  # noqa: PLC0415

    *via, goal = stops
    try:
        mm = MapManager.get_instance()
        route = PlannedRoute(start=to_lanelet2(start), goal=goal, via=via)
        lanelets = route_lanelets(route, mm.lanelet_map, mm.routing_graph)
        return [int(lanelet.id) for lanelet in lanelets]
    except RouteError as error:
        logger.warning(
            "No lane route to the goal (%s): the policy's road ahead takes the "
            "first way at a fork",
            error,
        )
    except Exception:
        logger.warning(
            "Could not route the policy's road ahead to the goal: it takes the "
            "first way at a fork",
            exc_info=True,
        )
    return None


def _roads_of(lanelet_ids: List[int]) -> Optional[Set[str]]:
    """The OpenDRIVE roads of a lanelet route,
    :func:`~autoware_carla_scenario.coordinate.route_road_ids`': those outside
    the junctions, which say which way each is crossed; ``None`` when the map
    cannot answer."""
    from ..coordinate import route_road_ids  # noqa: PLC0415

    try:
        return set(route_road_ids(lanelet_ids))
    except Exception:
        logger.warning(
            "Could not read the roads of the route to the goal: the policy's road "
            "ahead takes the first way at a fork",
            exc_info=True,
        )
        return None


def _in_map_frame(pose: "Lanelet2Pose") -> Optional[Tuple[float, float, float]]:
    """*pose* as ``(x, y, yaw)`` in Autoware's ``map`` frame, a T4Scene's;
    ``None`` when the map cannot place it."""
    from ..coordinate import to_map_frame  # noqa: PLC0415

    try:
        placed = to_map_frame(pose)
    except Exception:  # noqa: BLE001 - a scene goes on without its goal
        logger.debug("Could not place the goal in the map frame", exc_info=True)
        return None
    q = placed.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return (float(placed.position.x), float(placed.position.y), yaw)


#: A goal's heading is taken over at least this much road, so a last point
#: duplicated (or nearly) by the walk's sampling cannot spin it.
_GOAL_HEADING_SPAN_M = 0.5


def _far_end(points: Any) -> Optional[Tuple[float, float, float]]:
    """``(x, y, yaw)`` at the far end of a road's ``(N, 2)`` points, heading
    along its last ``_GOAL_HEADING_SPAN_M``; ``None`` for a road shorter."""
    if points is None or not len(points):
        return None
    end = points[-1]
    for before in points[-2::-1]:
        if math.hypot(end[0] - before[0], end[1] - before[1]) >= _GOAL_HEADING_SPAN_M:
            yaw = math.atan2(end[1] - before[1], end[0] - before[0])
            return (float(end[0]), float(end[1]), yaw)
    return None


def _tire_steering_angle(actor: "carla.Actor") -> float:
    """The front tyres' angle, radians, positive to the left (Autoware's)."""
    import typesafe_carla.carla as _carla  # noqa: PLC0415

    wheels = _carla.VehicleWheelLocation
    left = float(actor.get_wheel_steer_angle(wheels.FL_Wheel))
    right = float(actor.get_wheel_steer_angle(wheels.FR_Wheel))
    # CARLA reports degrees, positive to the right.
    return -math.radians(0.5 * (left + right))


def _live_t4_scene(
    config: DriverClientConfig, rear_axle_offset_m: float
) -> "LiveT4Scene":
    """The run's T4Scenes, as ``driver.t4_scene*`` asks for them.

    Raises:
        RuntimeError: If carla-driver-interface is too old to carry a scene.
    """
    from carla_driver_interface.protocol import carla_driver_pb2  # noqa: PLC0415

    from ..t4_scene.live import LiveT4Scene  # noqa: PLC0415

    if "t4_scene" not in carla_driver_pb2.CarlaRendererData.DESCRIPTOR.fields_by_name:
        raise RuntimeError(
            "driver.t4_scene needs carla-driver-interface 2.4 or later, which "
            "carries CarlaRendererData.t4_scene"
        )
    return LiveT4Scene(
        round(config.t4_scene_history_s / 0.1) + 1,
        rear_axle_offset_m=rear_axle_offset_m,
        wheel_base_m=config.t4_scene_wheel_base_m,
    )
