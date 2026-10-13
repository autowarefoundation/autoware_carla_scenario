"""Ego vehicle driven by ROS 2 nodes, over DDS, with the framework in the loop.

The framework stays the tick master and the one that applies control.  Every tick
the entity publishes the ego's state on ROS 2 topics, takes the commands that
arrived since the last tick, picks the source that drives (highest priority among
the fresh ones, :class:`~autoware_carla_scenario.ros2.commands.CommandArbiter`) and
applies its command:

* a planned trajectory is tracked by the framework's own
  :class:`~autoware_carla_scenario.driver.control.TrajectoryFollower`, as
  :class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`
  tracks a policy's plan;
* a steering/acceleration pair or an Autoware control command goes to CARLA's
  Ackermann controller, as VisionPilot's ``carla_control_publisher`` hands it over
  ``/carla/hero/ackermann_control_cmd``.

Nothing waits for a command: the loop runs at the pace
``ScenarioRunner`` sets (``max_tick_rate_hz``), and a source that falls silent
goes stale after its ``stale_s``.  That is how a ROS 2 bridge that relays to CARLA
behaves, and what a node measuring wall-clock deadlines -- a safety island's
heartbeat watch -- needs; a lockstep would stall the simulation under it.

Cameras and ``/clock`` are not this entity's: CARLA publishes them itself when
built with ROS 2.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

from ..constants import FIXED_DELTA_SECONDS, US_PER_S
from ..driver.base import EgoObservation
from ..driver.control import ControlConfig, TrajectoryFollower, VehicleCommand
from ..driver.geometry import Trajectory
from ..driver.observation import ego_observation, rear_axle_offset
from ..ros2.commands import (
    CommandArbiter,
    ControlCommand,
    EgoState,
    Selection,
    SteerAccelCommand,
    TrajectoryCommand,
)
from ..ros2.config import Ros2EgoConfig
from ..ros2.transport import Ros2EgoTransport
from ..utils.powertrain import ChaosPowertrain
from .ego import EgoVehicle

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

__all__ = ["Ros2EgoEntity", "plan_from_trajectory"]

#: Below this speed (m/s) a trajectory without timing is not timed from it.
_MIN_TIMING_SPEED_MPS: float = 0.1


def plan_from_trajectory(command: TrajectoryCommand, anchor_us: int) -> Trajectory:
    """Return *command* as a timed plan in the local frame.

    Each point is stamped ``anchor_us + time_from_start``.  A trajectory whose
    ``time_from_start`` does not increase -- many planners leave it zero -- is
    timed from its speeds instead: each segment takes its length over the mean of
    its end speeds.

    Args:
        command: The received trajectory.
        anchor_us: The simulation time its first point belongs to.
    """
    points = command.points
    offsets = [point.time_from_start_us for point in points]
    if any(later <= earlier for earlier, later in zip(offsets, offsets[1:])):
        offsets = [0]
        for previous, point in zip(points, points[1:]):
            # math.hypot of the differences, which is math.dist's own
            # computation: Codon's math has no dist().
            start = previous.pose.position
            end = point.pose.position
            length = math.hypot(
                float(start[0]) - float(end[0]), float(start[1]) - float(end[1])
            )
            speed = 0.5 * (previous.velocity_mps + point.velocity_mps)
            if speed < _MIN_TIMING_SPEED_MPS:
                # The plan stops here: whatever follows is not reached.
                break
            offsets.append(offsets[-1] + int(round(length / speed * US_PER_S)))
    plan = Trajectory.empty()
    for point, offset in zip(points, offsets):
        plan.append(anchor_us + offset, point.pose)
    return plan


class Ros2EgoEntity(EgoVehicle):
    """Ego vehicle whose commands come from ROS 2 nodes over DDS.

    Args:
        config: Topics, domain and command sources.  ``None`` uses the defaults.
        control_config: Gains for tracking a trajectory.  ``None`` uses the
            defaults.
        transport: Pre-built transport, used by tests to substitute a fake.
            ``None`` builds a
            :class:`~autoware_carla_scenario.ros2.cyclone.CycloneRos2Transport`
            when the scenario starts, which needs the ``ros2`` extra.
    """

    #: The ROS 2 nodes drive; TrafficManager must keep its hands off this actor.
    use_autopilot: bool = False

    _config: Ros2EgoConfig
    _transport: Optional[Ros2EgoTransport]
    _arbiter: CommandArbiter
    _follower: TrajectoryFollower
    _rear_axle_offset_m: float
    _open: bool
    _active: Optional[str]
    _active_fresh: bool
    #: The trajectory command last tracked and its timed plan.
    _planned: Optional[Tuple[TrajectoryCommand, Trajectory]]

    def __init__(
        self,
        config: Optional[Ros2EgoConfig] = None,
        control_config: Optional[ControlConfig] = None,
        transport: Optional[Ros2EgoTransport] = None,
    ) -> None:
        super().__init__()
        self._config = config if config is not None else Ros2EgoConfig()
        self._transport = transport
        self._arbiter = CommandArbiter(self._config.sources)
        self._follower = TrajectoryFollower(control_config)
        self._rear_axle_offset_m = 0.0
        self._open = False
        self._active = None
        self._active_fresh = False
        self._planned = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Ros2EgoConfig:
        """Return the settings."""
        return self._config

    @property
    def transport(self) -> Optional[Ros2EgoTransport]:
        """Return the transport, or ``None`` before the scenario starts."""
        return self._transport

    @property
    def active_source(self) -> Optional[str]:
        """The source that drove on the last tick, or ``None`` before any."""
        return self._active

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_scenario_start(self, world: "carla.World") -> None:
        """Join the DDS domain and publish the ego's first state.

        Raises:
            RuntimeError: If the ego actor has not been spawned yet.
            ImportError: If no transport was given and ``cyclonedds`` is not
                installed.
        """
        actor = self.actor
        if actor is None:
            raise RuntimeError("Ros2EgoEntity.on_scenario_start called before spawn()")

        self._rear_axle_offset_m = rear_axle_offset(
            actor, self._config.rear_axle_offset_m
        )
        self._follower.powertrain = ChaosPowertrain.from_physics_control(
            actor.get_physics_control()
        )
        if self._transport is None:
            if TYPE_CHECKING:
                # ros2.cyclone imports cyclonedds, which Codon cannot compile:
                # the check builds no transport (docs/typecheck.md).
                raise ImportError("the static check builds no Cyclone DDS transport")
            else:
                self._transport = _cyclone_transport(self._config)
        self._transport.open()
        self._open = True
        self._transport.publish_state(self._state(world, actor)[0])

    def on_tick(self, world: "carla.World", elapsed: float) -> None:
        """Publish the ego's state, take the commands, and apply the one that drives."""
        actor = self.actor
        transport = self._transport
        if actor is None or transport is None or not self._open:
            return
        state, observation = self._state(world, actor)
        transport.publish_state(state)
        for source_name, command in transport.poll():
            self._arbiter.observe(source_name, command, state.stamp_us)
        selection = self._arbiter.select(state.stamp_us)
        self._note_selection(selection)
        self._apply(actor, selection, observation)

    def on_scenario_end(self, world: "carla.World") -> None:
        """Leave the DDS domain.

        A failure is logged rather than raised, so it cannot abort the rest of
        the scenario's cleanup.
        """
        if self._transport is not None and self._open:
            try:
                self._transport.close()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.exception("Failed to close the ROS 2 transport")
        self._open = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _state(
        self, world: "carla.World", actor: "carla.Actor"
    ) -> tuple[EgoState, EgoObservation]:
        stamp_us = int(
            round(float(world.get_snapshot().timestamp.elapsed_seconds) * US_PER_S)
        )
        observation = ego_observation(actor, stamp_us, self._rear_axle_offset_m)
        state = EgoState(
            stamp_us=stamp_us,
            pose=observation.pose,
            linear_velocity=observation.linear_velocity,
            angular_velocity=observation.angular_velocity,
            speed_mps=observation.speed_mps,
            # Two server round trips a tick: only when something reads it.
            steering_tire_angle_rad=(
                _steering_tire_angle(actor)
                if self._config.steering_status_topic
                else 0.0
            ),
        )
        return state, observation

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _note_selection(self, selection: Optional[Selection]) -> None:
        name: Optional[str] = None
        if selection is not None:
            name = selection.source.name
        fresh = selection is not None and selection.fresh
        if name != self._active:
            logger.info("Ego driven by command source %r", name)
            self._follower.reset()
        elif self._active_fresh and not fresh:
            logger.warning(
                "Command source %r went stale; %s",
                name,
                "holding its last command"
                if self._config.on_stale == "hold"
                else "braking",
            )
        elif fresh and not self._active_fresh:
            logger.info("Command source %r is fresh again", name)
        self._active = name
        self._active_fresh = fresh

    def _apply(
        self,
        actor: "carla.Actor",
        selection: Optional[Selection],
        observation: EgoObservation,
    ) -> None:
        if selection is None or (
            not selection.fresh and self._config.on_stale == "brake"
        ):
            actor.apply_control(VehicleCommand(brake=1.0).to_carla_control())
            return
        command = selection.command
        source = selection.source
        # Each branch names the command as its class: Codon 0.19 reads no
        # attribute of a union (docs/typecheck.md).
        if isinstance(command, TrajectoryCommand):
            trajectory: TrajectoryCommand = command
            vehicle_command = self._follower.step(
                self._plan(trajectory, selection.received_us),
                observation.pose,
                observation.speed_mps,
                FIXED_DELTA_SECONDS,
                yaw_rate_rps=float(observation.angular_velocity[2]),
                now_us=observation.timestamp_us,
                gear=int(actor.get_control().gear),
            )
            actor.apply_control(vehicle_command.to_carla_control())
            return
        if isinstance(command, SteerAccelCommand):
            steer_accel: SteerAccelCommand = command
            steer_rad = source.steering_sign * steer_accel.steering_rad
            speed_mps = (
                observation.speed_mps
                + steer_accel.acceleration_mps2 * source.speed_horizon_s
            )
            acceleration_mps2 = steer_accel.acceleration_mps2
            jerk_mps3 = source.jerk_mps3
        else:
            control: ControlCommand = command
            # Autoware steers positive to the left; CARLA's Ackermann
            # controller, positive to the right.
            steer_rad = -control.steering_tire_angle_rad
            speed_mps = control.velocity_mps
            acceleration_mps2 = control.acceleration_mps2
            jerk_mps3 = source.jerk_mps3
            control_jerk = control.jerk_mps3
            if control_jerk is not None:
                jerk_mps3 = control_jerk
        import typesafe_carla.carla as _carla  # noqa: PLC0415

        actor.apply_ackermann_control(
            _carla.VehicleAckermannControl(
                steer=float(steer_rad),
                steer_speed=0.0,
                speed=float(max(0.0, speed_mps)),
                acceleration=float(acceleration_mps2),
                jerk=float(jerk_mps3),
            )
        )

    def _plan(self, command: TrajectoryCommand, received_us: int) -> Trajectory:
        """The timed plan of *command*, built once while it is the one tracked."""
        if self._planned is None or self._planned[0] is not command:
            anchor_us = command.stamp_us if command.stamp_us > 0 else received_us
            self._planned = (command, plan_from_trajectory(command, anchor_us))
        return self._planned[1]


def _steering_tire_angle(actor: "carla.Actor") -> float:
    """The front tyres' angle, radians, positive to the left (Autoware's)."""
    import typesafe_carla.carla as _carla  # noqa: PLC0415

    wheels = _carla.VehicleWheelLocation
    left = float(actor.get_wheel_steer_angle(wheels.FL_Wheel))
    right = float(actor.get_wheel_steer_angle(wheels.FR_Wheel))
    # CARLA reports degrees, positive to the right.
    return -math.radians(0.5 * (left + right))


def _cyclone_transport(config: Ros2EgoConfig) -> Ros2EgoTransport:
    try:
        from ..ros2.cyclone import CycloneRos2Transport  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ego.entity=ros2 needs Cyclone DDS's Python binding: install the "
            "'ros2' extra (pip install 'autoware-carla-scenario[ros2]'). "
            f"({exc})"
        ) from exc
    return CycloneRos2Transport(config)
