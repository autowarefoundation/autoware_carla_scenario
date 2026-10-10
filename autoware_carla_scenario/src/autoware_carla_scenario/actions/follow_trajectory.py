"""Follow-trajectory action: move an entity along a given trajectory.

The OpenSCENARIO ``FollowTrajectoryAction``, with the same four parts -- the
trajectory, its time reference, the following mode and an initial distance
offset (see :mod:`autoware_carla_scenario.trajectory.model`).  It is what lets a
recorded drive be written down as a scenario: every vehicle and pedestrian of a
recorded scene becomes an entity following the trajectory it was recorded on.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

from ..action_state import ActionState
from ..conditions import BaseCondition
from ..conditions.base import ScenarioResult
from ..coordinate.poses import CarlaWorldPose
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
from ._departures import DepartureTimeline, TimelineVertex
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
#: How near (m) a walker has to come to a waiting vertex to have reached it.
#: A walker stops where it is told to, so it needs less of a band than a car.
_WALKER_GATE_TOLERANCE_M = 0.3
#: The slowest (m/s) a walker is sent towards a waiting vertex it has not
#: reached yet, so that it gets there rather than creeping up on it for ever.
_WALKER_GATE_CREEP_MPS = 0.5
#: The deceleration (m/s²) a ``FOLLOW`` plan brakes at to stop at a vertex
#: whose condition has not held yet.
_GATE_DECELERATION_MPS2 = 2.0
#: How far past now (s) a plan stamp may lie: a segment that is never
#: finished (no speed to go at) is stamped this far off rather than at
#: infinity.
_PLAN_FAR_S = 3600.0


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

    **Waypoint conditions.**  Each vertex is departed on its condition
    (:attr:`~autoware_carla_scenario.TrajectoryVertex.advance`): a time
    (``TrajectoryTimeCondition``) -- on the trajectory's clock, so only with a
    time reference, and ignored without one -- any other condition, or
    nothing (on arrival).  A trajectory whose vertices carry
    times alone, or nothing at all, moves exactly as described above.  Once a
    vertex carries another condition (:attr:`Trajectory.is_gated`), the
    segment from a vertex departed at scenario time ``T`` to the next one
    arrives at the next vertex's time if it has one not before ``T`` (the
    timed interpolation), and otherwise goes at the action's *speed* -- so
    after a late departure, a next vertex whose time is at or just after it
    is reached at once (a jump in ``POSITION``; ``FOLLOW`` cannot keep up and
    drives as fast as it can).  In ``POSITION`` mode a vertex's condition is
    checked from the tick the entity reaches it (``FOLLOW``: see below), then
    every tick while it waits there -- standing still, at zero velocity -- with
    the scenario's elapsed time, as a trigger is; once it holds it is passed
    for that run (and, on a closed path, that lap).  If it holds on arrival
    the entity does not stop.  A condition that never holds keeps the entity
    there until ``until=`` ends the run.

    In ``FOLLOW`` mode a vehicle cannot stop in a tick, so a vertex with a
    condition still to hold is a stop line: the controller's plan ends on it,
    braking at 2 m/s².  So the condition is checked *before* arrival, from
    as far off as stopping there takes, and one that holds then is passed --
    the vehicle goes through at speed even if it would no longer hold on
    arrival, and a ``PersistentCondition`` starts counting from there; one
    that does not is looked at every tick after.  The last vertex of an open
    path is the exception: it is checked only on arrival.  The
    vehicle has reached the vertex within :data:`ARRIVAL_TOLERANCE_M` of it
    along the path (a walker within 0.3 m), and waits there on the brake (a
    walker standing).  On a closed path ``FOLLOW`` finds where the entity is
    by projecting it near where it was, which does not wrap from the end of
    the path to its start, so its gates are reliable on the first lap only;
    ``POSITION`` re-arms them on every lap.

    **End.**  The run is
    :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING` until the
    entity reaches the end of the trajectory: the last vertex's time with a
    timing, the end of the path without one (never, for a closed trajectory),
    within :data:`ARRIVAL_TOLERANCE_M` of it in ``FOLLOW`` mode.  A last vertex
    with another condition is waited at, and the run ends when it holds.

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
        speed: The speed (m/s) the entity goes at where no vertex time paces
            it: without a time reference, and on a gated trajectory's segments
            that end at no time (or a time already past).  ``None`` (default)
            takes the speed the entity had when the run started.
        hidden_outside_trajectory: Keep the entity out of the world while the
            trajectory's clock is before its first vertex or past its last:
            parked under the map with its physics off, so it neither shows nor
            collides.  What a recording needs for a road user that enters it
            late or leaves it early, since an entity cannot be spawned once a
            run has started.  Not part of OpenSCENARIO; needs ``POSITION`` mode
            and a time reference.  On a gated trajectory: until the first
            vertex is departed on its time, and after the run ends.
        appear_on_start: Bring the entity into the world when a run starts:
            put it on the first vertex (on the point *initial_distance_offset*
            names), facing along the path, physics on, moving at *speed*
            (``None``: standing) -- then follow the trajectory as usual, in
            either mode, a vehicle or a pedestrian.  What an entity spawned out
            of the world (``spawn.hidden``) needs to enter a recording when
            its action is triggered, e.g. once the ego has progressed to where
            it was first seen.  Not part of OpenSCENARIO; not combined with
            *hidden_outside_trajectory*.

    Raises:
        ValueError: If a timing is given for an untimed trajectory, the
            initial distance offset or the speed is negative,
            *hidden_outside_trajectory* is asked for without ``POSITION`` mode
            and a time reference, or together with *appear_on_start*.
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
        speed: Optional[float] = None,
        appear_on_start: bool = False,
    ) -> None:
        if appear_on_start and hidden_outside_trajectory:
            raise ValueError(
                "appear_on_start and hidden_outside_trajectory both decide when "
                "the entity enters the world; choose one"
            )
        if hidden_outside_trajectory and (
            time_reference is None
            or following_mode is not TrajectoryFollowingMode.POSITION
        ):
            raise ValueError(
                "hidden_outside_trajectory needs POSITION mode and a time "
                "reference: only a timed replay knows when the entity is absent"
            )
        if time_reference is not None and not trajectory.has_times:
            raise ValueError(
                f"trajectory {trajectory.name!r} has no vertex times, so a "
                "time reference has nothing to apply to; pass time_reference=None"
            )
        if initial_distance_offset < 0.0:
            raise ValueError("initial_distance_offset must not be negative")
        if speed is not None and not speed >= 0.0:
            raise ValueError("speed must not be negative")
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
        self._appear_on_start = appear_on_start
        self._given_speed = None if speed is None else float(speed)
        # Departure conditions other than times: the timeline engine.
        self._gated = trajectory.is_gated
        self._timeline: Optional[DepartureTimeline] = None
        self._waiting = False
        self._held_at: Optional[int] = None
        self._progress = 0.0
        self._last_here = 0.0

        self._resolved: Optional[ResolvedTrajectory] = None
        self._elapsed = 0.0
        # Per run.
        self._finished = False
        self._warned_missing = False
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

    @property
    def held_vertex(self) -> Optional[int]:
        """The vertex (0-based) the entity waits at for its ``advance`` condition.

        ``None`` while it moves, waits on a time, and before and after a run
        -- however the run ended, an ``until=`` condition included.
        """
        if self._finished or self.state is not ActionState.RUNNING:
            return None
        return self._held_at

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Move the entity one tick along the trajectory."""
        if self.state is ActionState.STANDBY:
            # The trigger's call: a new run begins, on the first call that
            # finds the entity (and, for a relative trajectory, its references).
            self._begun = False
            # Not until it has run: a run waiting for its entity or its
            # references must not end on the last run's arrival.
            self._finished = False
            self._warned_missing = False
            if self._trajectory.is_relative:
                # Placed against where the reference entities are now, at
                # every start of a run rather than once for all of them.
                self._resolved = None
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
            if self._resolved is None:
                return
        if not self._begun:
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
            self._walk(world, actor)
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

    def _resolve(self, world: "carla.World") -> Optional[ResolvedTrajectory]:
        """The trajectory in CARLA world coordinates, or ``None`` to retry.

        A :class:`~autoware_carla_scenario.trajectory.RelativeLanePose` is
        placed against where its reference entity is now; one not (yet) in the
        world puts the start off to a later tick.
        """
        from ..trajectory.resolve import resolve_trajectory, road_height  # noqa: PLC0415

        try:
            return resolve_trajectory(
                self._trajectory,
                road_height(world.get_map()),
                reference=self._reference_pose,
            )
        except _ReferenceMissing as missing:
            # Once a run: the action waits for it every tick, indefinitely.
            if not self._warned_missing:
                self._warned_missing = True
                logger.warning(
                    "FollowTrajectoryAction: %r is placed relative to '%s', "
                    "which is not in the world yet; waiting for it",
                    self._trajectory.name,
                    missing.name,
                )
            return None

    def _reference_pose(
        self, entity_ref: Optional[Union[EntityRole, str]]
    ) -> CarlaWorldPose:
        """Where a relative vertex's reference entity is now."""
        name = self._entity_name if entity_ref is None else entity_ref
        entity = find_entity_by_role_name(name)
        actor = getattr(entity, "actor", None) if entity is not None else None
        if actor is None:
            raise _ReferenceMissing(str(name))
        return CarlaWorldPose.from_carla_transform(actor.get_transform())

    def _begin(self, world: "carla.World", entity: Any, actor: Any) -> None:
        """Start a run: take the entity over and set the clock or the distance."""
        assert self._resolved is not None
        self._begun = True
        self._finished = False
        self._start_time = self._elapsed
        self._distance = self._initial_distance_offset
        self._clock_shift = 0.0
        if (
            self._time_reference is not None
            and self._initial_distance_offset > 0.0
            and not self._gated
        ):
            # Start that far along: the trajectory's clock begins where it is
            # there, not at its first vertex.
            self._clock_shift = (
                self._resolved.time_at_distance(self._initial_distance_offset)
                - self._resolved.start_time
            )
        if self._appear_on_start:
            self._appear(actor)
        velocity = actor.get_velocity()
        self._speed = (
            math.hypot(velocity.x, velocity.y)
            if self._given_speed is None
            else self._given_speed
        )
        if self._time_reference is None and self._speed < _STANDSTILL_MPS:
            logger.warning(
                "FollowTrajectoryAction: '%s' is standing still and %r has no "
                "time reference, so it will not move along it",
                self._entity_name,
                self._trajectory.name,
            )
        elif self._gated and self._speed < _STANDSTILL_MPS:
            logger.warning(
                "FollowTrajectoryAction: '%s' is standing still and has no "
                "speed, so it will not move along a segment of %r that no "
                "vertex time paces",
                self._entity_name,
                self._trajectory.name,
            )
        self._held_at = None
        self._waiting = False
        if self._gated:
            self._begin_timeline()
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

    def _appear(self, actor: Any) -> None:
        """Put *actor* where the run starts, facing along the path, at speed."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        resolved = self._resolved
        assert resolved is not None
        speed = 0.0 if self._given_speed is None else self._given_speed
        sample = resolved.at_distance(self._initial_distance_offset)
        walker = _is_walker(actor)
        actor.set_simulate_physics(True)
        self._hidden = False
        lift = float(actor.bounding_box.extent.z) if walker else 0.0
        actor.set_transform(
            carla.Transform(
                carla.Location(x=sample.x, y=sample.y, z=sample.z + lift),
                carla.Rotation(yaw=sample.yaw),
            )
        )
        heading = math.radians(sample.yaw)
        direction = carla.Vector3D(math.cos(heading), math.sin(heading), 0.0)
        if walker:
            actor.apply_control(
                carla.WalkerControl(direction=direction, speed=_walker_speed(speed))
            )
        else:
            actor.set_target_velocity(
                carla.Vector3D(speed * direction.x, speed * direction.y, 0.0)
            )
        logger.info(
            "FollowTrajectoryAction: '%s' appears at the start of %r at %.1f m/s",
            self._entity_name,
            self._trajectory.name,
            speed,
        )

    def _absent(self) -> bool:
        """Whether the trajectory's clock is outside its vertices' times."""
        assert self._resolved is not None
        if self._timeline is not None:
            # Before the first departure (on its time), and once ended.
            return self._finished or self._elapsed < self._timeline.times[0]
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
        if self._timeline is not None:
            return self._timeline_sample(world)
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
        stop: Optional[float] = None
        if self._timeline is not None:
            velocity = actor.get_velocity()
            stop = self._follow_gates(
                world, here, ARRIVAL_TOLERANCE_M, math.hypot(velocity.x, velocity.y)
            )
            if self._held_at is not None:
                # Waiting at the vertex: on the brake, and the controller's
                # memory of the run up to it cleared, so it pulls away cleanly.
                import typesafe_carla.carla as carla  # noqa: PLC0415

                self._follower.reset()
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                return
            if self._finished:
                return
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        plan = (
            self._plan(here)
            if self._timeline is None
            else self._timeline_plan(here, stop)
        )
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

    def _walk(self, world: "carla.World", actor: Any) -> None:
        """One tick of ``FOLLOW`` mode for a pedestrian."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        resolved = self._resolved
        assert resolved is not None
        location = actor.get_transform().location
        here = resolved.project(location.x, location.y, near=self._distance)
        self._distance = here
        if self._timeline is not None:
            self._walk_timeline(world, actor, here)
            return
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

    # ------------------------------------------------------------------
    # Waypoint conditions: the departure timeline
    # ------------------------------------------------------------------

    def _scenario_time_of(self, vertex_time: float) -> float:
        """The scenario time the trajectory's clock reads *vertex_time* at."""
        assert self._time_reference is not None
        reference = self._time_reference
        origin = (
            0.0 if reference.domain is ReferenceContext.ABSOLUTE else self._start_time
        )
        return (
            origin
            + reference.offset
            + (vertex_time - self._clock_shift) * reference.scale
        )

    def _begin_timeline(self) -> None:
        """Lay out the run's timeline from where it starts."""
        resolved = self._resolved
        assert resolved is not None
        vertices = self._trajectory.vertices
        timed = self._time_reference is not None
        offset = self._initial_distance_offset
        closed = resolved.closed and resolved.length > 1e-6
        count = len(vertices)

        def distance(k: int) -> float:
            if not closed:
                return resolved.vertex_distance(min(k, count - 1))
            lap, index = divmod(k, count)
            return lap * resolved.length + resolved.vertex_distance(index)

        # The vertex the run starts at, or the one before the point it does.
        k = 0
        while (closed or k + 1 < count) and distance(k + 1) <= offset + 1e-9:
            k += 1
        at_vertex = abs(distance(k) - offset) <= 1e-9
        start_point = distance(k) if at_vertex else offset
        point_time: Optional[float] = None
        if timed:
            index = k % count
            here = vertices[index].time
            if at_vertex:
                point_time = here
            elif not closed and k + 1 < count:
                there = vertices[k + 1].time
                if here is not None and there is not None:
                    span = distance(k + 1) - distance(k)
                    fraction = 0.0 if span <= 1e-9 else (offset - distance(k)) / span
                    point_time = here + fraction * (there - here)
            if offset > 0.0 and point_time is not None:
                # As for a timed trajectory: the clock starts where it is there.
                first = vertices[0].time
                self._clock_shift = point_time - (first if first is not None else 0.0)
        self._timeline = DepartureTimeline(
            [
                TimelineVertex(
                    distance=resolved.vertex_distance(index),
                    time=(
                        None
                        if not timed or (time := vertex.time) is None
                        else self._scenario_time_of(time)
                    ),
                    gate=vertex.gate,
                )
                for index, vertex in enumerate(vertices)
            ],
            resolved.length,
            closed,
            self._speed,
        )
        self._progress = start_point
        self._last_here = start_point % resolved.length if closed else start_point
        if at_vertex and vertices[k % count].gate is not None:
            self._timeline.start_waiting(k, self._elapsed)
            return
        departure = (
            self._scenario_time_of(point_time)
            if point_time is not None
            else self._elapsed
        )
        self._timeline.start(k, start_point, departure)

    def _gate_opens(
        self,
        world: "carla.World",
        condition: BaseCondition,
        index: int,
        *,
        early: bool = False,
    ) -> bool:
        """Whether vertex *index*'s *condition* holds now; logs waits and departures.

        *early* is a ``FOLLOW`` entity looking ahead before it has reached the
        vertex: a ``None`` then neither waits nor logs.
        """
        opened = condition.check(world, self._elapsed) is not None
        if early and not opened:
            return False
        if opened and self._waiting:
            logger.info(
                "FollowTrajectoryAction: '%s' departs vertex %d of %r: %s holds",
                self._entity_name,
                index,
                self._trajectory.name,
                condition.label,
            )
        elif not opened and not self._waiting:
            logger.info(
                "FollowTrajectoryAction: '%s' waits at vertex %d of %r until %s "
                "holds",
                self._entity_name,
                index,
                self._trajectory.name,
                condition.label,
            )
        return opened

    def _timeline_sample(self, world: "carla.World") -> TrajectorySample:
        """``POSITION`` mode on a gated trajectory: where the timeline has it now."""
        resolved = self._resolved
        timeline = self._timeline
        assert resolved is not None and timeline is not None
        elapsed = self._elapsed
        self._held_at = None
        last = timeline.count - 1
        for _ in range(timeline.count + 2):
            if elapsed < timeline.end_arrival:
                break
            index = timeline.index(timeline.end_vertex)
            if timeline.end_kind == "end":
                self._finished = True
                return resolved.at_vertex(index)
            condition = timeline.vertex(timeline.end_vertex).gate
            assert condition is not None
            if not self._gate_opens(world, condition, index):
                self._waiting = True
                self._held_at = index
                return resolved.at_vertex(index)
            # Departed on the tick it held: on arrival, it does not stop.
            departure = elapsed if self._waiting else timeline.end_arrival
            self._waiting = False
            if not resolved.closed and index == last:
                self._finished = True
                return resolved.at_vertex(index)
            timeline.depart(departure)
        distance, speed = timeline.at(elapsed)
        return resolved.at_distance(distance, speed)

    def _track(self, here: float) -> float:
        """*here* unwrapped across the laps of a closed path."""
        resolved = self._resolved
        assert resolved is not None
        if not resolved.closed or resolved.length <= 1e-6:
            self._progress = here
            return here
        delta = here - self._last_here
        half = resolved.length / 2.0
        if delta < -half:
            delta += resolved.length
        elif delta > half:
            delta -= resolved.length
        self._last_here = here
        self._progress += delta
        return self._progress

    def _follow_gates(
        self, world: "carla.World", here: float, tolerance: float, speed: float
    ) -> Optional[float]:
        """Check the vertices a ``FOLLOW`` entity is at or nearing; the stop ahead.

        A vertex waiting for its condition is looked at from as far off as it
        would take to stop there (:func:`_approach_m`): a condition that holds
        then departs it at the time the timeline had the entity arrive (or
        now, if later), so it costs the schedule nothing.  One that does not
        is the stop line, and is looked at again every tick; reached and still
        not holding, the entity waits (:attr:`_held_at`).  The last vertex of
        an open path is not looked at early: departing it ends the run
        (:attr:`_finished`), so it is checked only once the entity is there.

        Returns:
            The stop line, as a distance in the frame of *here*, or ``None``.
        """
        resolved = self._resolved
        timeline = self._timeline
        assert resolved is not None and timeline is not None
        self._held_at = None
        progress = self._track(here)
        last = timeline.count - 1
        for _ in range(timeline.count + 2):
            if timeline.end_kind != "gate":
                return None
            k = timeline.end_vertex
            index = timeline.index(k)
            at = timeline.distance(k)
            stop = here + (at - progress)
            condition = timeline.vertex(k).gate
            if condition is None:
                return stop  # a segment it has no speed for: never reached
            if progress >= at - tolerance:
                if not self._gate_opens(world, condition, index):
                    self._waiting = True
                    self._held_at = index
                    return stop
                departure = (
                    self._elapsed
                    if self._waiting
                    else max(timeline.end_arrival, self._elapsed)
                )
            elif (
                # The end is not a vertex to pass: departing it ends the run,
                # so it is only ever checked once the entity is there.
                (resolved.closed or index != last)
                and progress >= at - _approach_m(speed, tolerance)
                and self._gate_opens(world, condition, index, early=True)
            ):
                departure = max(timeline.end_arrival, self._elapsed)
            else:
                return stop
            self._waiting = False
            if not resolved.closed and index == last:
                self._finished = True
                return None
            timeline.depart(departure)
        return None

    def _timeline_plan(self, here: float, stop: Optional[float]) -> Any:
        """The path ahead of *here*, stamped from the timeline, for the controller."""
        from carla_driver_interface.geometry import Pose  # noqa: PLC0415
        from carla_driver_interface.geometry import Trajectory as Plan  # noqa: PLC0415

        resolved = self._resolved
        timeline = self._timeline
        assert resolved is not None and timeline is not None
        distances: list[float] = []
        distance = max(0.0, here - _PLAN_BEHIND_M)
        last = here + _PLAN_HORIZON_M
        if not resolved.closed:
            last = min(last, resolved.length)
        if stop is not None:
            last = min(last, stop)
        while distance <= last + 1e-9:
            distances.append(distance)
            distance += 1.0
        if stop is not None and stop <= last + 1e-9:
            if not distances or distances[-1] < stop - 1e-6:
                distances.append(stop)
        shift = self._progress - here
        far = self._elapsed + _PLAN_FAR_S
        stamps = [
            min(timeline.stamp(distance + shift, self._speed), far)
            for distance in distances
        ]
        if stop is not None:
            stamps = _brake_to(stop, distances, stamps)
        plan = Plan.empty()
        previous_us: Optional[int] = None
        for distance, stamp in zip(distances, stamps):
            sample = resolved.at_distance(distance)
            stamp_us = int(round(stamp * 1e6))
            if previous_us is not None and stamp_us <= previous_us:
                stamp_us = previous_us + 1
            plan.append(
                stamp_us,
                Pose.from_xyz_yaw(
                    sample.x, -sample.y, sample.z, -math.radians(sample.yaw)
                ),
            )
            previous_us = stamp_us
        return plan

    def _walk_timeline(self, world: "carla.World", actor: Any, here: float) -> None:
        """One ``FOLLOW`` tick for a pedestrian on a gated trajectory."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        resolved = self._resolved
        timeline = self._timeline
        assert resolved is not None and timeline is not None
        velocity = actor.get_velocity()
        stop = self._follow_gates(
            world, here, _WALKER_GATE_TOLERANCE_M, math.hypot(velocity.x, velocity.y)
        )
        if self._held_at is not None or self._finished:
            actor.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0)
            )
            return
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        wanted, pace = timeline.at(self._elapsed)
        # The timeline's pace, plus a catch-up of the distance it is behind,
        # closed over about a second.
        speed = pace + max(0.0, wanted - self._progress)
        aim = here + _WALKER_LOOKAHEAD_M
        if stop is not None:
            # Slow down onto the vertex rather than walk past it.
            speed = min(speed, max(stop - here, _WALKER_GATE_CREEP_MPS))
            aim = min(aim, stop)
        target = resolved.at_distance(aim)
        dx, dy = (
            target.x - actor.get_transform().location.x,
            (target.y - actor.get_transform().location.y),
        )
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            speed, dx, dy, norm = 0.0, 1.0, 0.0, 1.0
        actor.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(dx / norm, dy / norm, 0.0),
                speed=_walker_speed(speed),
            )
        )


def _approach_m(speed: float, tolerance: float) -> float:
    """From how far off (m) a ``FOLLOW`` entity looks at the vertex it may wait at.

    What stopping from *speed* at :data:`_GATE_DECELERATION_MPS2` takes, plus
    a second at that speed (the controller reads its plan a second ahead) and
    the arrival band.
    """
    return speed * speed / (2.0 * _GATE_DECELERATION_MPS2) + speed + tolerance


def _brake_to(stop: float, distances: list[float], stamps: list[float]) -> list[float]:
    """*stamps* slowed so the plan comes to rest at *stop*.

    Each step takes at least as long as braking at
    :data:`_GATE_DECELERATION_MPS2` towards *stop* would: the speed cap at a
    point *d* short of the stop is ``sqrt(2 * a * d)``, and a step between two
    caps takes ``2 * length / (v0 + v1)``.  The first stamp stays where it was.
    """
    caps = [
        math.sqrt(2.0 * _GATE_DECELERATION_MPS2 * max(0.0, stop - distance))
        for distance in distances
    ]
    out = stamps[:1]
    for index in range(1, len(stamps)):
        length = distances[index] - distances[index - 1]
        given = stamps[index] - stamps[index - 1]
        both = caps[index - 1] + caps[index]
        braking = 2.0 * length / both if both > 1e-9 else given
        out.append(out[-1] + max(given, braking))
    return out


class _ReferenceMissing(Exception):
    """A relative vertex's reference entity is not in the world."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


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
