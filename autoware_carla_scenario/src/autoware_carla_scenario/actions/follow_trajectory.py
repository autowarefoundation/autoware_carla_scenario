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
from dataclasses import dataclass
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
#: How near (m) a walker has to come to a gated vertex to have reached it.  A
#: walker stops where it is told to, so it needs less of a band than a car.
_WALKER_GATE_TOLERANCE_M = 0.3
#: The slowest (m/s) a walker is sent towards a gated vertex it has not yet
#: reached, so that it gets there rather than creeping up on it for ever.
_WALKER_GATE_CREEP_MPS = 0.5
#: The deceleration (m/s²) a ``FOLLOW`` plan brakes at to stop at a gated
#: vertex: firm, and well within what a car does on a dry road.
_GATE_DECELERATION_MPS2 = 2.0


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

    **Waypoint conditions.**  A vertex may carry an ``advance`` condition
    (:attr:`~autoware_carla_scenario.TrajectoryVertex.advance`): the entity is
    held at that vertex until the condition holds, and then goes on.  An
    ungated vertex has the implicit condition the trajectory always had -- the
    clock with a timing, the speed without one -- so a trajectory with no
    gates follows exactly as before.  A gate's condition is first checked on
    the tick the entity reaches the vertex, then on every tick while it is
    held, with the scenario's elapsed time (as a trigger is); once it holds it
    stays passed for the rest of the run.  If it already holds on arrival the
    entity does not stop.  While held:

    * ``POSITION`` with a timing: the trajectory's clock is paused at the
      vertex's time, so every later vertex is reached that much later and
      each segment after it keeps its recorded duration; the entity stands
      at the vertex with zero velocity;
    * ``POSITION`` without one: the distance stops at the vertex, at zero
      velocity, and goes on at the action's speed afterwards;
    * ``FOLLOW``: a gate not yet passed is a stop line -- the controller's plan
      ends there, braking to a stop -- which the vehicle has reached within
      :data:`ARRIVAL_TOLERANCE_M` of it along the path (a walker, which is
      walked onto it, within 0.3 m); held, a vehicle stands on the brake and
      a walker stands still.  The clock of a timed plan does not run past a
      gate not passed, and every stamp after it is shifted by the time held.

    A closed trajectory's gates apply on every lap; a repeated run
    (``once=False``) re-arms every gate.  Gates before the point a run starts
    from (*initial_distance_offset*) are not checked.  The condition objects
    are the caller's and are not reset between laps or runs, so a latching
    wrapper (``StickyCondition``) stays latched.

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
        self._warned_missing = False
        self._start_time = 0.0
        self._clock_shift = 0.0
        self._distance = 0.0
        self._speed = 0.0
        self._follower: Optional["TrajectoryFollower"] = None
        self._released = False
        self._hidden = False
        self._begun = False
        # Waypoint conditions, per run.
        self._gates: list[_Gate] = []
        self._gate_next = 0
        self._gate_lap = 0
        self._waiting = False
        self._held = False
        self._hold = 0.0
        self._now = 0.0
        self._progress = 0.0
        self._last_here = 0.0

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
        """The vertex whose ``advance`` condition is holding the entity, if any.

        ``None`` while the entity moves, including before a run and after it.
        """
        if not self._held or self._finished:
            return None
        gate = self._gate_ahead()
        return None if gate is None else gate.vertex

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
        if self._time_reference is not None and self._initial_distance_offset > 0.0:
            # Start that far along: the trajectory's clock begins where it is
            # there, not at its first vertex.
            self._clock_shift = (
                self._resolved.time_at_distance(self._initial_distance_offset)
                - self._resolved.start_time
            )
        self._begin_gates()
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
        now = self._now
        if self._held:
            # Held at a gate, the entity is on the trajectory however its
            # vertex's time compares with the last one's.
            return False
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
        """The vertex time the trajectory's clock is at now, before any hold."""
        assert self._time_reference is not None and self._resolved is not None
        return (
            self._time_reference.trajectory_time(self._elapsed, self._start_time)
            + self._clock_shift
        )

    # ------------------------------------------------------------------
    # Waypoint conditions
    # ------------------------------------------------------------------

    def _begin_gates(self) -> None:
        """Arm every gate for a new run, past the ones it starts beyond."""
        resolved = self._resolved
        assert resolved is not None
        self._gates = [
            _Gate(
                vertex=index,
                distance=resolved.vertex_distance(index),
                time=vertex.time,
                condition=vertex.advance,
            )
            for index, vertex in enumerate(self._trajectory.vertices)
            if vertex.advance is not None
        ]
        self._gate_next = 0
        self._gate_lap = 0
        self._waiting = False
        self._held = False
        self._hold = 0.0
        self._now = 0.0
        offset = self._initial_distance_offset
        self._progress = offset
        closed = resolved.closed and resolved.length > 1e-6
        self._last_here = offset % resolved.length if closed else offset
        if not self._gates:
            return
        if self._time_reference is not None:
            start = resolved.start_time + self._clock_shift
            while self._gate_next < len(self._gates) and (
                float(self._gates[self._gate_next].time or 0.0) < start
            ):
                self._gate_next += 1
            return
        if closed:
            self._gate_lap = int(offset // resolved.length)
        for _ in range(len(self._gates)):
            gate = self._gate_ahead()
            if gate is None or self._gate_distance(gate) >= offset:
                break
            self._pass_gate()

    def _gate_ahead(self) -> Optional["_Gate"]:
        """The next gate the run has not passed, or ``None``."""
        if self._gate_next >= len(self._gates):
            return None
        return self._gates[self._gate_next]

    def _gate_distance(self, gate: "_Gate") -> float:
        """How far along (m) *gate* is on the current lap."""
        assert self._resolved is not None
        return gate.distance + self._gate_lap * self._resolved.length

    def _pass_gate(self) -> None:
        """Latch the gate ahead; on a closed path, wrap to the next lap's first."""
        assert self._resolved is not None
        self._gate_next += 1
        self._waiting = False
        if self._resolved.closed and self._gate_next >= len(self._gates):
            self._gate_next = 0
            self._gate_lap += 1

    def _gate_opens(self, world: "carla.World", gate: "_Gate") -> bool:
        """Whether *gate*'s condition holds now; logs the hold and the release."""
        opened = gate.condition.check(world, self._elapsed) is not None
        if opened and self._waiting:
            logger.info(
                "FollowTrajectoryAction: '%s' leaves vertex %d of %r: %s holds",
                self._entity_name,
                gate.vertex,
                self._trajectory.name,
                gate.condition.label,
            )
        elif not opened and not self._waiting:
            logger.info(
                "FollowTrajectoryAction: '%s' is held at vertex %d of %r until "
                "%s holds",
                self._entity_name,
                gate.vertex,
                self._trajectory.name,
                gate.condition.label,
            )
        return opened

    def _gated_clock(self, world: "carla.World") -> float:
        """``POSITION`` mode's clock: the trajectory time, paused at a held gate.

        Every gate whose time the clock has reached is checked in order; the
        first that does not hold stops the clock at its time, and the time
        it stands there is added to the hold, which every later vertex is
        reached after.  On the tick a gate that held the entity opens, the
        entity is at the vertex, moving off.
        """
        raw = self._trajectory_time()
        self._held = False
        if not self._gates:
            return raw
        now = raw - self._hold
        for _ in range(len(self._gates)):
            gate = self._gate_ahead()
            if gate is None or gate.time is None or now < gate.time:
                break
            waited = self._waiting
            if not self._gate_opens(world, gate):
                self._waiting = True
                self._held = True
                self._hold = raw - gate.time
                return gate.time
            if waited:
                # Paused up to and including this tick: it leaves from the
                # vertex now, rather than a tick's travel past it.
                self._hold = raw - gate.time
                now = gate.time
            self._pass_gate()
        return now

    def _follow_clock(self) -> float:
        """``FOLLOW`` mode's clock: the trajectory time, paused at a gate not passed.

        Whether a gate is passed is decided by where the entity is
        (:meth:`_follow_gates`), so here the clock only waits for it: it does
        not run past the time of a gate the entity has not passed.
        """
        raw = self._trajectory_time()
        if not self._gates:
            return raw
        now = raw - self._hold
        gate = self._gate_ahead()
        if gate is not None and gate.time is not None and now >= gate.time:
            self._hold = raw - gate.time
            now = gate.time
        return now

    def _track(self, here: float) -> float:
        """*here* unwrapped across the laps of a closed path, for the gates."""
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
        self, world: "carla.World", here: float, tolerance: float
    ) -> Optional[float]:
        """Check the gates a ``FOLLOW`` entity has reached; the stop ahead, if any.

        Returns:
            The distance along the path, in the frame of *here*, of the next
            gate not passed -- the stop line the entity has to stop at -- or
            ``None`` when there is none.  :attr:`_held` says whether the
            entity has reached it and is being held there.
        """
        self._held = False
        if not self._gates:
            return None
        progress = self._track(here)
        for _ in range(len(self._gates)):
            gate = self._gate_ahead()
            if gate is None:
                return None
            at = self._gate_distance(gate)
            if progress < at - tolerance:
                return here + (at - progress)
            if not self._gate_opens(world, gate):
                self._waiting = True
                self._held = True
                return here + (at - progress)
            self._pass_gate()
        gate = self._gate_ahead()
        return None if gate is None else here + (self._gate_distance(gate) - progress)

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def _position_sample(self, world: "carla.World") -> TrajectorySample:
        """Where ``POSITION`` mode puts the entity this tick."""
        resolved = self._resolved
        assert resolved is not None
        if self._time_reference is not None:
            now = self._now = self._gated_clock(world)
            if now >= resolved.end_time and not self._held:
                self._finished = True
            sample = resolved.at_time(now)
            if self._held:
                return TrajectorySample(sample.x, sample.y, sample.z, sample.yaw)
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
        if self._gates:
            return self._untimed_gated_sample(world)
        sample = resolved.at_distance(self._distance, self._speed)
        if not resolved.closed and self._distance >= resolved.length:
            self._finished = True
        self._distance += self._speed * _tick_seconds(world)
        return sample

    def _untimed_gated_sample(self, world: "carla.World") -> TrajectorySample:
        """``POSITION`` mode without a timing, on a trajectory with gates.

        The distance runs on at the action's speed up to a gate, stops there
        while it is held, and runs on from it once it opens.
        """
        resolved = self._resolved
        assert resolved is not None
        self._held = False
        distance = self._distance
        for _ in range(len(self._gates)):
            gate = self._gate_ahead()
            if gate is None:
                break
            at = self._gate_distance(gate)
            if distance < at:
                break
            if not self._gate_opens(world, gate):
                self._waiting = True
                self._held = True
                distance = at
                break
            self._pass_gate()
        self._distance = distance
        if self._held:
            return resolved.at_distance(distance, 0.0)
        sample = resolved.at_distance(distance, self._speed)
        if not resolved.closed and distance >= resolved.length:
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
        if self._time_reference is not None:
            self._now = self._follow_clock()
        stop = self._follow_gates(world, here, ARRIVAL_TOLERANCE_M)
        if self._held:
            # Standing at the gate: on the brake, and the controller's memory
            # of the run up to it cleared, so it pulls away cleanly.
            import typesafe_carla.carla as carla  # noqa: PLC0415

            self._follower.reset()
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            return
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        plan = self._plan(here, stop)
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

    def _plan(self, here: float, stop: Optional[float] = None) -> Any:
        """The path ahead of *here*, timed on the scenario clock, for the controller.

        With a *stop* -- a gate not passed, as a distance in the frame of
        *here* -- the plan ends there, and is slowed so that it comes to rest
        there at :data:`_GATE_DECELERATION_MPS2`: no faster anywhere than a
        car braking at that rate would be, that far from the stop.
        """
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
        if stop is not None:
            last = min(last, stop)
        distances: list[float] = []
        while distance <= last + 1e-9:
            distances.append(distance)
            distance += step
        if stop is not None and stop <= last + 1e-9:
            if not distances or distances[-1] < stop - 1e-6:
                # The plan has to end on the stop itself, not short of it.
                distances.append(stop)
        stamps: list[float] = []
        for distance in distances:
            if self._time_reference is not None:
                stamps.append(self._scenario_time_at(distance))
            else:
                speed = max(self._speed, _STANDSTILL_MPS)
                stamps.append(self._elapsed + (distance - here) / speed)
        if stop is not None:
            stamps = _brake_to(stop, distances, stamps)
        previous_us: Optional[int] = None
        for distance, stamp in zip(distances, stamps):
            sample = resolved.at_distance(distance)
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
        return plan

    def _scenario_time_at(self, distance: float) -> float:
        """The scenario time at which a timed trajectory is *distance* along."""
        assert self._time_reference is not None and self._resolved is not None
        reference = self._time_reference
        vertex_time = self._resolved.time_at_distance(distance) - self._clock_shift
        if self._hold:
            # Every vertex past the gates passed is reached that much later.
            vertex_time += self._hold
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
        if self._time_reference is not None:
            self._now = self._follow_clock()
        stop = self._follow_gates(world, here, _WALKER_GATE_TOLERANCE_M)
        if self._held:
            actor.apply_control(
                carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0)
            )
            return
        if not resolved.closed and here >= resolved.length - ARRIVAL_TOLERANCE_M:
            self._finished = True
            return
        if self._time_reference is not None:
            now = self._now
            wanted = resolved.distance_at_time(now)
            # The recorded speed in scenario time, plus a catch-up of the
            # distance it is behind, closed over about a second.
            speed = resolved.at_time(now).speed / self._time_reference.scale + max(
                0.0, wanted - here
            )
        else:
            speed = self._speed
        aim = here + _WALKER_LOOKAHEAD_M
        if stop is not None:
            # Slow down onto the gate rather than walk through it.
            speed = min(speed, max(stop - here, _WALKER_GATE_CREEP_MPS))
            aim = min(aim, stop)
        target = resolved.at_distance(aim)
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


@dataclass(frozen=True)
class _Gate:
    """A vertex with an ``advance`` condition, placed on the resolved path."""

    vertex: int
    distance: float
    time: Optional[float]
    condition: BaseCondition


def _brake_to(stop: float, distances: list[float], stamps: list[float]) -> list[float]:
    """*stamps* slowed so the plan comes to rest at *stop*.

    Each step takes at least as long as braking at
    :data:`_GATE_DECELERATION_MPS2` towards *stop* would: the speed cap at a
    point *d* short of the stop is ``sqrt(2 * a * d)``, and a step between two
    caps takes ``2 * length / (v0 + v1)``.  The first stamp stays where it was,
    so the plan's clock still starts where the timing has it.
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
