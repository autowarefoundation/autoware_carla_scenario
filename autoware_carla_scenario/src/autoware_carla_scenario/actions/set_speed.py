"""Set-speed action: command a vehicle's target speed during a run."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Union

from ..action_state import ActionState
from ..conditions.always_true import NeverCondition
from ..conditions.base import BaseCondition, ScenarioResult, find_actor_by_role_name
from ..constants import tick_seconds
from ..conditions.comparison import ComparisonRule
from ..conditions.composition.base import CompositionCondition
from ..conditions.composition.speed import SpeedCondition
from ..conditions.entity_existence import find_present_actor_by_role_name
from ..entity.registry import fail_if_pedestrian, find_vehicle_entity
from ..entity_role import EntityRole
from ..kinematics import Vector3
from ..kinematics.walker_motion import WalkerVelocities, simulated_seconds
from ..traffic.driven import BackendDriven
from ..control_domain import ControlDomain
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["SetSpeedAction", "ARRIVAL_TOLERANCE_KMH"]

#: How close to the target counts as having arrived, in km/h.
#:
#: The run ends on the *vehicle's* speed rather than on a stopwatch, so it needs
#: a band: whatever drives the vehicle is chasing a target under its own control
#: law and will sit near it rather than exactly on it.
ARRIVAL_TOLERANCE_KMH: float = 0.5

_KMH_PER_MS: float = 3.6


class _ResolvedTarget:
    """The target read from the reference for this run (``None`` until
    read), shared with the condition that says it arrived."""

    kmh: Optional[float]

    def __init__(self) -> None:
        self.kmh = None


class _ReachedResolvedTarget(CompositionCondition):
    """A rate-limited relative change's arrival: the entity within
    :data:`ARRIVAL_TOLERANCE_KMH` of :class:`_ResolvedTarget`."""

    _resolved: _ResolvedTarget

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        resolved: _ResolvedTarget,
        *,
        label: str,
    ) -> None:
        super().__init__(entity_name=entity_name, label=label)
        self._resolved = resolved

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        target = self._resolved.kmh
        if target is None:
            return None
        assert self._entity_name is not None  # noqa: S101
        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            return None
        speed = _actor_speed_kmh(actor)
        if abs(speed - target) > ARRIVAL_TOLERANCE_KMH:
            return None
        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' reached {speed:.1f} km/h, "
                f"its relative target {target:.1f} km/h"
            ),
            elapsed_seconds=elapsed,
        )


def _actor_speed_kmh(actor: "carla.Actor") -> float:
    """*actor*'s speed in km/h."""
    velocity = actor.get_velocity()
    return (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5 * _KMH_PER_MS


class SetSpeedAction(BaseAction):
    """Command a vehicle to drive at a new speed.

    The framework could say what speed a vehicle *started* at and nothing
    else, so "the car ahead brakes to 3 m/s once the ego is within 30 m" had no
    way of being written.  This is that missing half.

    What the command reaches depends on what drives the vehicle: the intent
    goes to the entity, which carries it to its traffic backend.  For a
    TrafficManager-driven vehicle that is ``set_desired_speed``.

    **The target is a target.** Whatever drives the vehicle gets it there under
    its own acceleration limits, so commanding a speed is not commanding a
    velocity, and an immediate change of target is not an immediate change of
    speed.

    Giving *rate_kmh_s* asks for the change to be made no faster than that.  It
    is honoured by walking the commanded target up or down from the vehicle's
    **actual** speed, one tick at a time, which is why the number means
    something: the command can never run ahead of what the vehicle achieved, so
    a vehicle held up by traffic or a speed limit simply takes longer rather
    than being chased by a target it never reached.  OpenSCENARIO's
    ``dynamicsShape: linear`` with ``dynamicsDimension: rate`` is this.

    **A target relative to another entity.**  With *relative_to*, the target
    is that entity's speed times *factor* plus *delta_kmh* (OpenSCENARIO's
    ``RelativeTargetSpeed``; a result below 0 is 0): ``relative_to="ego",
    delta_kmh=10`` drives 10 km/h faster than the ego.  A walker's speed is
    measured from where it moved (CARLA reports a walker on its controller
    standing).  Without *continuous* the target is read once, when the action
    fires (each time, for a repeating one), and held for that run -- a
    rate-limited change then completes when the vehicle reaches it, as an
    absolute one does.  With *continuous* it is read again every
    tick, so the vehicle keeps following the other's speed; such an action
    never arrives, and runs until the scenario ends (or its *until* holds).
    A reference that is not in the scene (not spawned yet, or parked out of
    sight) gives no target: nothing is commanded until it is.

    ``duration`` is deliberately not offered.  A transition stated as a time
    would have to be interpolated open-loop, and a TrafficManager that could
    not keep up would leave the commanded value describing a manoeuvre that
    never happened -- the scenario's number kept while its meaning changed.

    A rate-limited change holds the action
    :attr:`~autoware_carla_scenario.action_state.ActionState.RUNNING` until the
    vehicle is within :data:`ARRIVAL_TOLERANCE_KMH` of the target, so
    ``completeState`` means the vehicle got there rather than that the commands
    stopped.  A vehicle that never gets there never completes, which is the
    same answer a lane change that does not happen gives.

    Args:
        entity_name: ``role_name`` of the vehicle to command.
        target_speed_kmh: The speed to hold, in km/h.  Given unless
            *relative_to* is, never with it.
        rate_kmh_s: Most the commanded speed may change per second.  ``None``
            (default) commands the target at once.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        until: Overrides what counts as having arrived.  Defaults to the
            vehicle being within :data:`ARRIVAL_TOLERANCE_KMH` of the target
            for a rate-limited change, and to nothing at all for an immediate
            one -- there is no manoeuvre to wait for when the target is simply
            handed over.
        reissue: Overrides whether the command is re-sent every tick.  It is
            the injection point for an *immediate* change, whose single call a
            backend that does not hold its target would need repeated.  A
            rate-limited change has no such choice: the seam carries a target
            and not a rate, so the walk advances only by being re-sent, and
            ``reissue=False`` beside a rate is refused rather than left to
            stall one step short.
        relative_to: ``role_name`` of the entity whose speed the target is
            relative to; ``None`` (default) for an absolute target.
        delta_kmh: Added to the reference's speed (times *factor*), in
            km/h; negative for slower.  Only with *relative_to*.
        factor: The reference's speed is multiplied by it before
            *delta_kmh* is added.  Only with *relative_to*.
        continuous: Follow the reference's speed every tick rather than
            take it once when the action starts.  Only with *relative_to*.

    Raises:
        ValueError: If the target is given neither or both ways, or a
            relative-only argument without *relative_to*; if a number is out
            of range (*target_speed_kmh* and *factor* negative, *rate_kmh_s*
            not positive), not finite or a boolean; or if *reissue* is
            ``False`` where the command must be re-sent (a rate, or
            *continuous*).  A rate of zero is a change that never arrives,
            and a rate never re-sent is a walk that never takes its second
            step; both are worth reporting rather than running.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _target_speed_kmh: Optional[float]
    _rate_kmh_s: Optional[float]
    _relative_to: Optional[str]
    _delta_kmh: float
    _factor: float
    _continuous: bool
    _resolved: _ResolvedTarget
    _velocities: WalkerVelocities

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        target_speed_kmh: Optional[float] = None,
        rate_kmh_s: Optional[float] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "set_speed",
        once: bool = True,
        overrides: bool = False,
        until: Optional[BaseCondition] = None,
        reissue: Optional[bool] = None,
        relative_to: Union[EntityRole, str, None] = None,
        delta_kmh: Optional[float] = None,
        factor: Optional[float] = None,
        continuous: Optional[bool] = None,
    ) -> None:
        if relative_to is None:
            if target_speed_kmh is None:
                raise ValueError(
                    "SetSpeedAction needs a target: target_speed_kmh, or "
                    "relative_to (with delta_kmh and/or factor)"
                )
            for name, given in (
                ("delta_kmh", delta_kmh is not None),
                ("factor", factor is not None),
                # False is the default an editor's checkbox sends.
                ("continuous", continuous is True),
            ):
                if given:
                    raise ValueError(
                        f"{name} is relative to another entity's speed: "
                        "give relative_to too"
                    )
            # Finite as well as in range: NaN fails every comparison, so a
            # NaN target would never arrive and be sent to the backend every
            # tick.  ``inf`` is refused as well: no backend can hold an
            # infinite speed, and an unbounded rate is already said by
            # ``rate_kmh_s=None``.  A boolean is refused too: True would be
            # read as 1 km/h (#2191).
            absolute: float = target_speed_kmh
            if isinstance(absolute, bool) or not (
                math.isfinite(absolute) and absolute >= 0
            ):
                raise ValueError(
                    "target_speed_kmh must be finite and not negative, "
                    f"not {target_speed_kmh!r}"
                )
        else:
            if target_speed_kmh is not None:
                raise ValueError(
                    "target_speed_kmh and relative_to are two ways to say the "
                    "target: give one (delta_kmh and factor say the relative "
                    "one)"
                )
            if delta_kmh is not None:
                delta: float = delta_kmh
                if isinstance(delta, bool) or not math.isfinite(delta):
                    raise ValueError(f"delta_kmh must be finite, not {delta_kmh!r}")
            if factor is not None:
                scale: float = factor
                if isinstance(scale, bool) or not (math.isfinite(scale) and scale >= 0):
                    raise ValueError(
                        f"factor must be finite and not negative, not {factor!r}"
                    )
        if rate_kmh_s is not None and (
            isinstance(rate_kmh_s, bool)
            or not (math.isfinite(rate_kmh_s) and rate_kmh_s > 0)
        ):
            raise ValueError(
                f"rate_kmh_s must be positive and finite, not {rate_kmh_s!r}"
            )
        if rate_kmh_s is not None and reissue is False:
            raise ValueError(
                "a rate-limited speed change must reissue: the traffic seam "
                "carries a target and not a rate, so the walk towards it only "
                "advances by being re-sent. A backend that honours a rate "
                "natively cannot be told one through set_desired_speed, so "
                "drop rate_kmh_s to command the target at once."
            )

        # Refused above without relative_to.
        follows = continuous is True
        if follows and reissue is False:
            raise ValueError(
                "a continuous relative speed follows the reference by being "
                "read and re-sent every tick: reissue=False would send it once"
            )
        resolved = _ResolvedTarget()
        if until is None and follows:
            # It follows the reference for the rest of the run: an action with
            # no *until* would be over on the tick after it fired.
            forever: BaseCondition = NeverCondition()
            until = forever
        elif until is None and rate_kmh_s is not None:
            if target_speed_kmh is None:
                # Measured against the target read when the action fired.
                reached: BaseCondition = _ReachedResolvedTarget(
                    entity_name, resolved, label=f"{label}_arrived"
                )
                until = reached
            else:
                arrived: BaseCondition = SpeedCondition(
                    entity_name=entity_name,
                    value=target_speed_kmh / _KMH_PER_MS,
                    rule=ComparisonRule.EQUAL_TO,
                    tolerance=ARRIVAL_TOLERANCE_KMH / _KMH_PER_MS,
                    label=f"{label}_arrived",
                )
                until = arrived

        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            overrides=overrides,
            until=until,
            reissue=reissue,
        )
        self._entity_name = str(entity_name)
        self._target_speed_kmh = target_speed_kmh
        self._rate_kmh_s = rate_kmh_s
        self._relative_to = str(relative_to) if relative_to is not None else None
        self._delta_kmh = delta_kmh if delta_kmh is not None else 0.0
        self._factor = factor if factor is not None else 1.0
        self._continuous = follows
        self._resolved = resolved
        # A walker reference's speed, measured from where it moved.
        self._velocities = WalkerVelocities()

    def reset(self) -> None:
        """Back to standby (:meth:`BaseAction.reset`), forgetting the walkers
        measured: the next scenario run measures its own."""
        super().reset()
        self._velocities.clear()

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    @property
    def needs_tick_loop(self) -> bool:
        """With an *until* -- a rate's arrival, or one given: the target is
        walked to (or waited for) a tick at a time -- or reissued: the
        command is sent again every tick.  An immediate change with neither is
        one call, which the initialization phase can make."""
        return self._until is not None or self.reissues_while_running

    def _reissues_by_default(self) -> bool:
        """A rate is walked towards, so its target has to be moved every tick;
        a continuous relative target follows the reference, so it is read and
        sent again every tick.

        An immediate change is one call: ``set_desired_speed`` holds what it was
        given, and repeating it would say the same thing again.
        """
        return self._rate_kmh_s is not None or self._continuous

    def _target_kmh(self, world: "carla.World") -> Optional[float]:
        """The target this tick: the absolute one, or the one relative to the
        reference -- read now when continuous, else the one read when this
        run of the action fired.  ``None`` while the reference is not in the
        scene, or is a walker not measured yet."""
        if self._relative_to is None:
            return self._target_speed_kmh
        if not self._continuous and self._resolved.kmh is not None:
            return self._resolved.kmh
        reference = find_present_actor_by_role_name(world, self._relative_to)
        if reference is None:
            return None
        read = self._velocities.velocity(reference, simulated_seconds(world))
        if read is None:
            return None
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        velocity: Vector3 = read
        speed_kmh = velocity.magnitude() * _KMH_PER_MS
        target = max(0.0, speed_kmh * self._factor + self._delta_kmh)
        if not self._continuous:
            self._resolved.kmh = target
        return target

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The vehicle's speed: its longitudinal domain."""
        return [(str(self._entity_name), ControlDomain.LONGITUDINAL)]

    def execute(self, world: "carla.World") -> None:
        """Command the target, or this tick's step towards it."""
        if self._lifecycle is not ActionState.RUNNING:
            # The trigger fired: a run of its own, which reads the reference
            # anew (a repeating action fires again after it completed).
            self._resolved.kmh = None
        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            fail_if_pedestrian(self._entity_name, "set_desired_speed")
            if self._report_missing():
                logger.warning(
                    "SetSpeedAction: entity '%s' not found", str(self._entity_name)
                )
            return

        found = self._target_kmh(world)
        if found is None:
            if self._report_missing():
                logger.warning(
                    "SetSpeedAction: reference '%s' of '%s' not in the scene",
                    str(self._relative_to),
                    str(self._entity_name),
                )
            return
        target_kmh: float = found

        if self._rate_kmh_s is None:
            entity.set_desired_speed(world, target_kmh)
            if not self._continuous:
                logger.info(
                    "SetSpeedAction: '%s' commanded to %.1f km/h%s",
                    self._entity_name,
                    target_kmh,
                    ""
                    if self._relative_to is None
                    else f", relative to '{self._relative_to}'",
                )
            return

        now_kmh = _current_speed_kmh(entity)
        if abs(now_kmh - target_kmh) <= ARRIVAL_TOLERANCE_KMH:
            # Inside the arrival band the walk is over, so send the target
            # itself rather than one more step towards it.
            #
            # This is what the run ends on, and `execute` runs before `until`
            # is asked within the same tick, so the value the backend is left
            # holding is the one the scenario asked for.  Stepping here instead
            # would leave an intermediate target in force forever whenever
            # `rate_kmh_s * delta_seconds` is smaller than the band -- a brake
            # to a stop would complete while still commanding a crawl.
            entity.set_desired_speed(world, target_kmh)
            return

        rate: float = self._rate_kmh_s
        step = rate * tick_seconds(world)
        entity.set_desired_speed(world, _toward(now_kmh, target_kmh, step))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _toward(now_kmh: float, target_kmh: float, step_kmh: float) -> float:
    """Return *now_kmh* moved at most *step_kmh* towards *target_kmh*."""
    if target_kmh > now_kmh:
        return min(now_kmh + step_kmh, target_kmh)
    return max(now_kmh - step_kmh, target_kmh)


def _current_speed_kmh(entity: BackendDriven) -> float:
    """Return *entity*'s present speed in km/h, or 0.0 when unknowable.

    The step is taken from the vehicle's own speed, which is what keeps the
    commanded target from running ahead of it.  An entity with no actor yet --
    or one whose actor has gone -- reads as stopped, which is the only honest
    answer available and never worse than refusing to command anything.
    """
    # `hasattr`, which Codon decides when it compiles: BackendDriven, the
    # mixin, declares no actor; the entities it is mixed into do.
    if not hasattr(entity, "actor"):
        return 0.0
    actor = entity.actor
    if actor is None:
        return 0.0
    return _actor_speed_kmh(actor)
