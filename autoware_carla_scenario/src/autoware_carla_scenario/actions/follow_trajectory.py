"""Follow-trajectory action: move an entity along a given trajectory.

The OpenSCENARIO ``FollowTrajectoryAction``, with the same four parts -- the
trajectory, its time reference, the following mode and an initial distance
offset (see :mod:`autoware_carla_scenario.trajectory.model`).  It is what lets a
recorded drive be written down as a scenario: every vehicle and pedestrian of a
T4 scene becomes an entity following the trajectory it was recorded on
(:mod:`autoware_carla_scenario.trajectory.t4`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

from ..action_state import ActionState
from ..conditions import BaseCondition
from ..conditions.base import ScenarioResult
from ..entity.registry import find_entity_by_role_name
from ..entity_role import EntityRole
from ..trajectory.model import (
    ReferenceContext,
    ResolvedTrajectory,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectorySample,
    TrajectoryTiming,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..driver.control import ControlConfig, TrajectoryFollower

logger = logging.getLogger(__name__)

__all__ = ["ARRIVAL_TOLERANCE_M", "FollowTrajectoryAction", "HIDDEN_DEPTH_M"]

#: How near the end of the path (m) a followed trajectory counts as finished.
#: A controller tracks a target rather than sitting on it, so ``follow`` mode
#: needs a band; ``position`` mode is on the path and needs none.
ARRIVAL_TOLERANCE_M: float = 1.0

#: How far under the road (m) a hidden entity is parked: out of every sensor's
#: view and of every other actor's way.
HIDDEN_DEPTH_M: float = 500.0

#: How far ahead of the entity (m) the controller is shown the path.  Past the
#: lookahead the controller never looks, and a window keeps a path that passes
#: the same place twice from being steered towards its other pass.
_PLAN_HORIZON_M = 60.0
#: How far behind the entity (m) the plan starts, so the controller's own
#: projection onto it has a segment to land on.
_PLAN_BEHIND_M = 2.0
#: Below this speed (m/s) an untimed trajectory is not moving at all.
_STANDSTILL_MPS = 0.05
#: How far ahead (m) a walker following the path is aimed.
_WALKER_LOOKAHEAD_M = 1.5


class FollowTrajectoryAction(BaseAction):
    """Make an entity follow a trajectory (OpenSCENARIO ``FollowTrajectoryAction``).

    **Time reference.**  With a :class:`~autoware_carla_scenario.TrajectoryTiming`
    the entity is where the trajectory says at each moment: a vertex at time
    ``τ`` is reached at ``τ * scale + offset`` seconds after the scenario
    (``ReferenceContext.ABSOLUTE``) or the action (``RELATIVE``) started.  With
    ``None`` -- OpenSCENARIO's ``<None/>`` -- the vertex times are ignored and
    the entity goes along the path at the speed it had when the action
    started, so give it an initial speed or use a timing.

    **Following mode.**  ``POSITION`` places the entity on the trajectory every
    tick, with the velocity the trajectory implies: a kinematic replay that
    reproduces a recording exactly, whatever the vehicle could actually do.
    ``FOLLOW`` drives it there with the pure-pursuit and speed controller the
    driver interface uses (:class:`~autoware_carla_scenario.driver.control.TrajectoryFollower`),
    so the motion is the vehicle's own.  A pedestrian is walked towards the
    path in ``FOLLOW`` mode.

    **Who drives.**  For the length of the run the action is the entity's only
    driver: a vehicle is taken back from the traffic backend first (see
    :meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.release`), and
    an ego driven by an external stack (Autoware, a driver policy) is refused,
    because two authorities on one vehicle is not a scenario anyone wrote.
    When the run ends a vehicle is braked to a stop and a walker stood still,
    where the trajectory ended.

    **End.**  The run is
    :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING` until the
    entity reaches the end of the trajectory: the last vertex's time with a
    timing, the end of the path without one (never, for a closed trajectory),
    within :data:`ARRIVAL_TOLERANCE_M` of it in ``FOLLOW`` mode.

    Args:
        entity_name: ``role_name`` of the vehicle or pedestrian to move.
        trajectory: What to follow.
        time_reference: How the vertex times map onto the scenario clock;
            ``None`` ignores them.  A timing needs a timed trajectory.
        following_mode: ``POSITION`` (default) or ``FOLLOW``.
        initial_distance_offset: Start this many metres along the trajectory
            instead of at its first vertex.  With a timing, the clock starts at
            the time the trajectory reaches that point.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.  Pre-tick (default), so the pose set is the one the
            world then advances from.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        until: Overrides what ends the run; defaults to reaching the end of
            the trajectory as above.
        control_config: Gains for ``FOLLOW`` mode; ``None`` uses the
            controller's defaults.
        hidden_outside_trajectory: Keep the entity out of the world while the
            trajectory's clock is before its first vertex or past its last:
            parked under the map with its physics off, so it neither shows nor
            collides.  What a recording needs for a road user that enters it
            late or leaves it early, since an entity cannot be spawned once a
            run has started.  Not part of OpenSCENARIO; needs ``POSITION`` mode
            and a time reference.

    Raises:
        ValueError: If a timing is given for an untimed trajectory, the
            initial distance offset is negative, or
            *hidden_outside_trajectory* is asked for without ``POSITION`` mode
            and a time reference.
    """

    #: The command is a pose (or a control) for this tick only: the action
    #: moves the entity by being executed again on every tick of the run.
    REISSUES_BY_DEFAULT = True

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        trajectory: Trajectory,
        time_reference: Optional[TrajectoryTiming] = None,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.POSITION,
        initial_distance_offset: float = 0.0,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "follow_trajectory",
        once: bool = True,
        until: Optional[BaseCondition] = None,
        control_config: Optional["ControlConfig"] = None,
        hidden_outside_trajectory: bool = False,
    ) -> None:
        if hidden_outside_trajectory and (
            time_reference is None
            or following_mode is not TrajectoryFollowingMode.POSITION
        ):
            raise ValueError(
                "hidden_outside_trajectory needs POSITION mode and a time "
                "reference: only a timed replay knows when the entity is absent"
            )
        if time_reference is not None and not trajectory.is_timed:
            raise ValueError(
                f"trajectory {trajectory.name!r} has no vertex times, so a "
                "time reference has nothing to apply to; pass time_reference=None"
            )
        if initial_distance_offset < 0.0:
            raise ValueError("initial_distance_offset must not be negative")
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            until=until if until is not None else _TrajectoryEnd(self, label),
        )
        self._entity_name = entity_name
        self._trajectory = trajectory
        self._time_reference = time_reference
        self._following_mode = following_mode
        self._initial_distance_offset = float(initial_distance_offset)
        self._control_config = control_config
        self._hidden_outside = hidden_outside_trajectory

        self._resolved: Optional[ResolvedTrajectory] = None
        self._elapsed = 0.0
        # Per run.
        self._finished = False
        self._start_time = 0.0
        self._clock_shift = 0.0
        self._distance = 0.0
        self._speed = 0.0
        self._follower: Optional["TrajectoryFollower"] = None
        self._released = False
        self._hidden = False
        self._begun = False

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> Trajectory:
        """The trajectory followed."""
        return self._trajectory

    @property
    def finished(self) -> bool:
        """Whether the current run has reached the end of the trajectory."""
        return self._finished

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Move the entity one tick along the trajectory."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning(
                "FollowTrajectoryAction: entity '%s' not found", str(self._entity_name)
            )
            return
        actor = getattr(entity, "actor", None)
        if actor is None:
            logger.warning(
                "FollowTrajectoryAction: '%s' has no actor", str(self._entity_name)
            )
            return
        if getattr(entity, "use_autopilot", True) is False:
            logger.warning(
                "FollowTrajectoryAction: '%s' is driven by an external stack and "
                "cannot also follow a trajectory",
                str(self._entity_name),
            )
            return
        if self._resolved is None:
            self._resolved = self._resolve(world)
        if self.state is ActionState.STANDBY or not self._begun:
            # The trigger's call begins a run -- or, when the entity was not
            # there to begin it with, the first call that finds it.
            self._begin(world, entity, actor)
        if self._finished:
            return

        walker = _is_walker(actor)
        if self._following_mode is TrajectoryFollowingMode.POSITION:
            sample = self._position_sample(world)
            if self._hidden_outside and self._absent():
                self._hide(actor, sample)
                return
            self._show(actor)
            _place(actor, sample, walker)
        elif walker:
            self._walk(actor)
        else:
            self._drive(world, actor)
        if self._finished:
            if not self._hidden:
                _stop(actor, walker)
            logger.info(
                "FollowTrajectoryAction: '%s' reached the end of %r",
                self._entity_name,
                self._trajectory.name,
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _resolve(self, world: "carla.World") -> ResolvedTrajectory:
        from ..trajectory.resolve import resolve_trajectory, road_height  # noqa: PLC0415

        return resolve_trajectory(self._trajectory, road_height(world.get_map()))

    def _begin(self, world: "carla.World", entity: Any, actor: Any) -> None:
        """Start a run: take the entity over and set the clock or the distance."""
        assert self._resolved is not None
        self._begun = True
        self._finished = False
        self._start_time = self._elapsed
        self._distance = self._initial_distance_offset
        self._clock_shift = 0.0
        if self._time_reference is not None and self._initial_distance_offset > 0.0:
            # Start that far along: the trajectory's clock begins where it is
            # there, not at its first vertex.
            self._clock_shift = (
                self._resolved.time_at_distance(self._initial_distance_offset)
                - self._resolved.start_time
            )
        velocity = actor.get_velocity()
        self._speed = math.hypot(velocity.x, velocity.y)
        if self._time_reference is None and self._speed < _STANDSTILL_MPS:
            logger.warning(
                "FollowTrajectoryAction: '%s' is standing still and %r has no "
                "time reference, so it will not move along it",
                self._entity_name,
                self._trajectory.name,
            )
        if not self._released:
            release = getattr(entity, "release_from_traffic", None)
            if release is not None:
                release(world)
            self._released = True
        if self._following_mode is TrajectoryFollowingMode.FOLLOW and not _is_walker(
            actor
        ):
            from ..driver.control import TrajectoryFollower  # noqa: PLC0415
            from ..utils.powertrain import ChaosPowertrain  # noqa: PLC0415

            self._follower = TrajectoryFollower(
                self._control_config,
                ChaosPowertrain.from_physics_control(actor.get_physics_control()),
            )

    def _absent(self) -> bool:
        """Whether the trajectory's clock is outside its vertices' times."""
        assert self._resolved is not None
        now = self._trajectory_time()
        return now < self._resolved.start_time or now >= self._resolved.end_time

    def _hide(self, actor: Any, sample: TrajectorySample) -> None:
        """Park *actor* under where *sample* is, out of the simulation."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        if not self._hidden:
            actor.set_simulate_physics(False)
            self._hidden = True
        actor.set_transform(
            carla.Transform(
                carla.Location(x=sample.x, y=sample.y, z=sample.z - HIDDEN_DEPTH_M),
                carla.Rotation(yaw=sample.yaw),
            )
        )

    def _show(self, actor: Any) -> None:
        """Bring a hidden *actor* back into the simulation."""
        if self._hidden:
            actor.set_simulate_physics(True)
            self._hidden = False

    def _trajectory_time(self) -> float:
        """The vertex time the trajectory is at, now."""
        assert self._time_reference is not None and self._resolved is not None
        return (
            self._time_reference.trajectory_time(self._elapsed, self._start_time)
            + self._clock_shift
        )

    def _position_sample(self, world: "carla.World") -> TrajectorySample:
        """Where ``POSITION`` mode puts the entity this tick."""
        resolved = self._resolved
        assert resolved is not None
        if self._time_reference is not None:
            now = self._trajectory_time()
            if now >= resolved.end_time:
                self._finished = True
            sample = resolved.at_time(now)
            # The sample moves in vertex time; the world, in scenario time,
            # which runs `scale` times slower.
            scale = self._time_reference.scale
            return TrajectorySample(
                sample.x,
                sample.y,
                sample.z,
                sample.yaw,
                vx=sample.vx / scale,
                vy=sample.vy / scale,
                yaw_rate=sample.yaw_rate / scale,
            )
        sample = resolved.at_distance(self._distance, self._speed)
        if not resolved.closed and self._distance >= resolved.length:
            self._finished = True
        self._distance += self._speed * _tick_seconds(world)
        return sample

    def _drive(self, world: "carla.World", actor: Any) -> None:
        """One tick of ``FOLLOW`` mode for a vehicle."""
        from carla_driver_interface.geometry import Pose  # noqa: PLC0415

        resolved = self._resolved
        assert resolved is not None and self._follower is not None
        transform = actor.get_transform()
        here = resolved.project(
            transform.location.x, transform.location.y, near=self._distance
        )
        self._distance = here
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        plan = self._plan(here)
        velocity = actor.get_velocity()
        angular = actor.get_angular_velocity()
        control = actor.get_control()
        command = self._follower.step(
            plan,
            # The plan's frame is CARLA's with y flipped: right-handed, as the
            # controller expects.
            Pose.from_xyz_yaw(
                transform.location.x,
                -transform.location.y,
                transform.location.z,
                -math.radians(transform.rotation.yaw),
            ),
            math.hypot(velocity.x, velocity.y),
            _tick_seconds(world),
            yaw_rate_rps=-math.radians(angular.z),
            now_us=int(round(self._elapsed * 1e6)),
            gear=int(getattr(control, "gear", 0)),
        )
        actor.apply_control(command.to_carla_control())

    def _plan(self, here: float) -> Any:
        """The path ahead of *here*, timed on the scenario clock, for the controller."""
        from carla_driver_interface.geometry import Pose  # noqa: PLC0415
        from carla_driver_interface.geometry import Trajectory as Plan  # noqa: PLC0415

        resolved = self._resolved
        assert resolved is not None
        plan = Plan.empty()
        step = 1.0
        distance = max(0.0, here - _PLAN_BEHIND_M)
        last = here + _PLAN_HORIZON_M
        if not resolved.closed:
            last = min(last, resolved.length)
        previous_us: Optional[int] = None
        while distance <= last + 1e-9:
            sample = resolved.at_distance(distance)
            if self._time_reference is not None:
                stamp = self._scenario_time_at(distance)
            else:
                speed = max(self._speed, _STANDSTILL_MPS)
                stamp = self._elapsed + (distance - here) / speed
            stamp_us = int(round(stamp * 1e6))
            if previous_us is not None and stamp_us <= previous_us:
                # Standing still on the recording: one microsecond on, so the
                # plan's clock keeps running forward.
                stamp_us = previous_us + 1
            plan.append(
                stamp_us,
                Pose.from_xyz_yaw(
                    sample.x, -sample.y, sample.z, -math.radians(sample.yaw)
                ),
            )
            previous_us = stamp_us
            distance += step
        return plan

    def _scenario_time_at(self, distance: float) -> float:
        """The scenario time at which a timed trajectory is *distance* along."""
        assert self._time_reference is not None and self._resolved is not None
        reference = self._time_reference
        vertex_time = self._resolved.time_at_distance(distance) - self._clock_shift
        origin = (
            0.0 if reference.domain is ReferenceContext.ABSOLUTE else self._start_time
        )
        return origin + reference.offset + vertex_time * reference.scale

    def _walk(self, actor: Any) -> None:
        """One tick of ``FOLLOW`` mode for a pedestrian."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        resolved = self._resolved
        assert resolved is not None
        location = actor.get_transform().location
        here = resolved.project(location.x, location.y, near=self._distance)
        self._distance = here
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        if self._time_reference is not None:
            now = self._trajectory_time()
            wanted = resolved.distance_at_time(now)
            # The recorded speed in scenario time, plus a catch-up of the
            # distance it is behind, closed over about a second.
            speed = resolved.at_time(now).speed / self._time_reference.scale + max(
                0.0, wanted - here
            )
        else:
            speed = self._speed
        target = resolved.at_distance(here + _WALKER_LOOKAHEAD_M)
        dx, dy = target.x - location.x, target.y - location.y
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            speed, dx, dy, norm = 0.0, 1.0, 0.0, 1.0
        actor.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(dx / norm, dy / norm, 0.0),
                speed=_walker_speed(speed),
            )
        )


class _TrajectoryEnd(BaseCondition):
    """Satisfied once the action it watches has reached the end of its trajectory."""

    def __init__(self, action: FollowTrajectoryAction, label: str) -> None:
        super().__init__(label=f"{label}_end")
        self._action = action

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        if not self._action.finished:
            return None
        return ScenarioResult(
            passed=True,
            message=f"{self._action.trajectory.name!r} followed to its end",
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------


def _is_walker(actor: Any) -> bool:
    return str(getattr(actor, "type_id", "")).startswith("walker.")


def _tick_seconds(world: "carla.World") -> float:
    """How much simulated time the last tick covered."""
    return float(world.get_snapshot().timestamp.delta_seconds)


def _walker_speed(speed_mps: float) -> float:
    """The ``WalkerControl`` speed for *speed_mps*, capped at what CARLA can walk.

    Capped here rather than left to warn: the command is re-sent every tick,
    and a recording of someone jogging would otherwise log on every one.
    """
    from ..entity.pedestrian_entity import (  # noqa: PLC0415
        _UE5_WALKER_MAX_SPEED_MS,
        walker_speed_command,
    )

    return walker_speed_command(min(max(speed_mps, 0.0), _UE5_WALKER_MAX_SPEED_MS))


def _place(actor: Any, sample: TrajectorySample, walker: bool) -> None:
    """Put *actor* at *sample*, moving with the sample's velocity.

    The velocity goes with the pose so that what reads the actor's speed -- a
    speed condition, the recorder, a TrafficManager car behind it -- sees the
    trajectory's, and so that the tick the world then takes carries the actor
    towards where the next sample will put it rather than leaving it parked.
    A walker's origin is at its middle, so it is raised by half its height.
    """
    import typesafe_carla.carla as carla  # noqa: PLC0415

    lift = float(actor.bounding_box.extent.z) if walker else 0.0
    actor.set_transform(
        carla.Transform(
            carla.Location(x=sample.x, y=sample.y, z=sample.z + lift),
            carla.Rotation(yaw=sample.yaw),
        )
    )
    if walker:
        speed = sample.speed
        direction = (
            carla.Vector3D(sample.vx / speed, sample.vy / speed, 0.0)
            if speed > 1e-6
            else carla.Vector3D(1.0, 0.0, 0.0)
        )
        actor.apply_control(
            carla.WalkerControl(direction=direction, speed=_walker_speed(speed))
        )
        return
    actor.set_target_velocity(carla.Vector3D(sample.vx, sample.vy, 0.0))
    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, sample.yaw_rate))


def _stop(actor: Any, walker: bool) -> None:
    """Leave *actor* standing where the trajectory ended."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    if walker:
        actor.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0)
        )
        return
    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
