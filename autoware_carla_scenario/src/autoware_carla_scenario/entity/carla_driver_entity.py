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
import uuid
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ..constants import FIXED_DELTA_SECONDS
from ..debug_markers import DebugMarkers, plan_polyline
from ..coordinate.map_manager import run_carla_map
from ..driver.base import BaseEgoDriverClient, DriverClientConfig, EgoObservation
from ..driver.control import ControlConfig, TrajectoryFollower
from ..driver.egodriver_client import EgoDriverGrpcClient
from ..driver.geometry import Pose, Trajectory
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
from ._policy_warmup import PolicyWarmup
from .bridged_ego import BridgeHandshake
from .externally_driven import ExternallyDrivenEgo

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..autoware_bridge.base import AutowareBridge, AutowareBridgeConfig
    from ..autoware_stack.launcher import AutowareLauncher
    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose

    from ..sensor.carla_camera import CarlaCameraSensor
    from ..sensor.carla_lidar import CarlaLidarSensor


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
        self._world: Optional["carla.World"] = None
        self._plan: Trajectory = Trajectory.empty()
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
        """Hand *goal* to the bridged stack's mission; a no-op without a bridge.

        The policy itself is given the road ahead rather than a destination.
        """
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

        self._attach_cameras(world, actor)
        self._world = world
        self._acceleration.reset()
        self._turn_indicator = TurnIndicator.NO_COMMAND
        # The ego drives on CARLA's own (Chaos) vehicle physics: this entity puts it
        # on no other, so its pedals are worked out from that model.
        self._follower.powertrain = ChaosPowertrain.from_physics_control(
            actor.get_physics_control()
        )

        # CARLA rebuilds the map object on every ``get_map()`` call, so the run's
        # one is taken (the MapManager's, when it holds one) and reused for the
        # rolling route walk.
        self._map = run_carla_map(world)

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

        session_uuid = str(uuid.uuid4())
        self._client.start_session(session_uuid, str(self._map.name))
        self._session_open = True

        if self._handshake is None:
            self._begin_driving(world, actor)

    def _begin_driving(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Start the run-up, or send the first route: the policy's first tick."""
        assert self._map is not None  # noqa: S101 - fetched when the session opened
        if self._config.warmup_s > 0.0:
            # The ego leaves for its run-up before the first tick: its spawn
            # pose is where it arrives, not where its history begins.
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
        """Write the world's map for the policy (``map_dir``); its id, or ``""``."""
        if self._config.map_dir is None or self._map is None:
            return ""
        from ..driver.hdmap_export import export_map  # noqa: PLC0415

        return export_map(
            str(self._map.to_opendrive()),
            str(self._map.name),
            self._config.map_dir,
            self._config.map_formats,
            lanelet2_path=self._config.lanelet2_path,
        )

    def _lidar_sweeps(self) -> List[LidarSweep]:
        """Every LiDAR's newest sweep, in the rig frame, stamped with when it was taken.

        A LiDAR sweeps once a policy step, which need not be this very tick: a
        sweep from an earlier tick carries that tick's time.
        """
        if not self._lidars or self._world is None:
            return []
        frame = int(self._world.get_snapshot().frame)
        tick_us = int(round(_FIXED_DELTA_S * _US_PER_S))
        sweeps = []
        for logical_id, lidar, mount in self._lidars:
            sweep = lidar.latest_sweep(frame)
            if sweep is None:
                continue
            age_ticks = max(frame - sweep[0], 0)
            sweeps.append(
                pack_lidar_sweep(
                    logical_id,
                    self._sim_time_us - age_ticks * tick_us,
                    lidar_points_to_rig(sweep[1], mount),
                    mount.to_proto(),
                )
            )
        return sweeps

    def _ego_observation(self, actor: "carla.Actor") -> EgoObservation:
        """Return the ego's state at the current simulation time."""
        if self._warmup is not None:
            actor = _CarriedActor(actor, self._warmup)
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
        )
        self._client.submit_route(self._sim_time_us, waypoints)
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
        return self._renderer.build(
            self._sim_time_us, observation.pose, lidar=self._lidar_sweeps()
        )

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
        """The policy's current plan, as a line along the road."""
        markers = DebugMarkers()
        markers.add_polyline(plan_polyline(self._plan))
        return markers

    def debug_state(self) -> Dict[str, float]:
        """Return a snapshot of the loop's state, for logging and tests."""
        return {
            "sim_time_us": float(self._sim_time_us),
            "drive_count": float(self._drive_count),
            "plan_length": float(len(self._plan)),
            "rear_axle_offset_m": self._rear_axle_offset_m,
            "turn_indicator": float(self._turn_indicator),
        }


class _CarriedActor:
    """The ego as the run-up carries it: its pose, and the motion it is carried at.

    Placed without physics, the actor itself reports no motion.
    """

    def __init__(self, actor: "carla.Actor", warmup: PolicyWarmup) -> None:
        self._actor = actor
        self._warmup = warmup

    def get_transform(self) -> "carla.Transform":
        return self._actor.get_transform()

    def get_velocity(self) -> "carla.Vector3D":
        return self._warmup.velocity()

    def get_acceleration(self) -> "carla.Vector3D":
        return self._warmup.acceleration()

    def get_angular_velocity(self) -> "carla.Vector3D":
        return self._warmup.angular_velocity()
