"""TeraSim's adversities as actions (#393, #398, #399, #400).

Each action carries out one adversity of :mod:`~..adversity.triggers` on a
named vehicle (or, for the pedestrian kinds, a pedestrian):

* :class:`AdversarialBrakeAction` -- brakes hard in front of a follower that
  keeps coming (:attr:`~..adversity.triggers.AdversityKind.BRAKE_CHECK`).
* :class:`IgnoreLeaderAction` -- drives on as if its leader were not there
  (``IGNORE_LEADER``).
* :class:`AdversarialCutInAction` -- changes lane in front of a vehicle too
  close behind there (``CUT_IN``).
* :class:`LaneChangeAbortAction` -- starts into the lane beside, in front of
  a vehicle too close behind there, and swerves back (``LANE_CHANGE_ABORT``).
* :class:`HeadOnAction` -- drives on at a leader coming the other way
  (``HEAD_ON``).
* :class:`AdversarialMergeAction` -- beside a merging lane, speeds up or
  changes in to cut a merging vehicle off (``MERGE``).
* :class:`RunTrafficRuleAction` -- runs the red or yellow light it is
  stopping for (``TRAFFIC_RULE``).
* :class:`JaywalkAction` -- a pedestrian walks straight across the road, off
  any crosswalk (``JAYWALKING``).
* :class:`PedestrianRunRedLightAction` -- a pedestrian waiting at a crosswalk
  crosses on red (``RUNNING_RED_LIGHT``).
* :class:`StopCrossingAction` -- a pedestrian stops on the crosswalk, then
  hurries across (``STOP_CROSSING``).

The trigger is an :class:`~..conditions.adversity_feasible.AdversityFeasibleCondition`
of the action's kind at its *location* by default -- which is why *location*
is required: :attr:`~..adversity.location.AdversityLocation.ANY` says
anywhere, and has to be said.  When it holds, the command it found is
planned on the map (:func:`~..adversity.command.plan_command`) and driven by
an :class:`~._adversarial_execution.AdversarialExecution`; the run ends when
the path is driven and the vehicle is handed back to its traffic backend.

A *condition* given in its place decides when the action fires instead: the
adversity's trigger is then asked on that tick, and when it does not hold
the run ends with nothing done (and a warning).  To require both, give
``AndCondition`` of the two.

Not re-exported from :mod:`autoware_carla_scenario.actions`, which every
scenario imports, because it pulls in the Lanelet2 map layer: import it from
here or from the package's top level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

from ..action_state import ActionState
from ..adversity.location import AdversityLocation
from ..adversity.triggers import DEFAULT_THRESHOLDS, AdversityKind, TriggerThresholds
from ..conditions.adversity_feasible import AdversityFeasibleCondition
from ..conditions.base import BaseCondition, ScenarioResult
from ..entity_role import EntityRole
from ..trajectory.model import TrajectoryFollowingMode
from ._adversarial_execution import AdversarialExecution, start_execution
from ..control_domain import ControlDomain, movement_of
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "AdversarialAction",
    "AdversarialBrakeAction",
    "AdversarialCutInAction",
    "AdversarialMergeAction",
    "HeadOnAction",
    "IgnoreLeaderAction",
    "JaywalkAction",
    "LaneChangeAbortAction",
    "PedestrianRunRedLightAction",
    "RunTrafficRuleAction",
    "StopCrossingAction",
]

logger = logging.getLogger(__name__)


class AdversarialAction(BaseAction):
    """An adversity of *kind* carried out by *entity_name* at *location*.

    The shared machinery of the phase-1 actions, which fix *kind*; see the
    module for how a run goes.

    Args:
        entity_name: ``role_name`` of the vehicle (or pedestrian).
        kind: The adversity.
        location: Where it may fire; :attr:`AdversityLocation.ANY` anywhere.
        condition: Fires the action in place of the adversity's own trigger.
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action runs at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        seed: Seeds the trigger's random draws (a jaywalker's
            speed, a stopped crosser's wait); ``None`` (default) takes the
            run's seed (:mod:`~autoware_carla_scenario.seed`).
        thresholds: The triggers' numbers; TeraSim's by default.
        following_mode: How the path is driven: ``FOLLOW`` (the vehicle's own
            motion, default) or ``POSITION`` (placed on it, as TeraSim's
            ``moveToXY``).
    """

    REISSUES_BY_DEFAULT = True

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _feasible: AdversityFeasibleCondition
    _gated: bool
    _following_mode: TrajectoryFollowingMode
    _execution: Optional[AdversarialExecution]
    _elapsed: float

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        kind: AdversityKind,
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str,
        once: bool = True,
        overrides: bool = False,
        seed: Optional[int] = None,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        feasible = AdversityFeasibleCondition(
            entity_name,
            kind,
            location,
            label=f"{label}_feasible",
            seed=seed,
            thresholds=thresholds,
        )
        # If statements, not conditional expressions: Codon does not take a
        # subclass for an Optional of its base there (docs/typecheck.md).
        trigger: Optional[BaseCondition] = condition
        if trigger is None:
            gate: BaseCondition = feasible
            trigger = gate
        end: BaseCondition = _AdversityDone(self, label)
        super().__init__(
            label=label,
            condition=trigger,
            timing=timing,
            once=once,
            overrides=overrides,
            until=end,
        )
        self._entity_name = str(entity_name)
        self._feasible = feasible
        self._gated = condition is None
        self._following_mode = following_mode
        self._execution = None
        self._elapsed = 0.0

    @property
    def kind(self) -> AdversityKind:
        """The adversity carried out."""
        return self._feasible.kind

    @property
    def location(self) -> AdversityLocation:
        """Where it may fire."""
        return self._feasible.location

    @property
    def finished(self) -> bool:
        """Whether the current run is over: its path driven and the vehicle
        handed back, or nothing to drive."""
        if self._execution is None:
            return True
        execution: AdversarialExecution = self._execution
        return execution.done

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The entity's movement: the adversity drives it along its path."""
        return movement_of(str(self._entity_name))

    def execute(self, world: "carla.World") -> None:
        """Plan the command when triggered, then drive it a tick at a time."""
        if self.state is ActionState.STANDBY:
            self._execution = self._plan(world)
        if self._execution is None:
            return
        execution: AdversarialExecution = self._execution
        execution.tick(world, self._elapsed)

    def _stopped(self, world: "carla.World") -> None:
        """Overridden mid-path: hand the vehicle back, as at the path's end."""
        if self._execution is not None:
            execution: AdversarialExecution = self._execution
            execution.hand_back(world)

    def close(self) -> None:
        """Close the conditions, the trigger's and the path's."""
        super().close()
        if not self._gated:
            self._feasible.close()
        if self._execution is not None:
            execution: AdversarialExecution = self._execution
            execution.close()

    def _plan(self, world: "carla.World") -> Optional[AdversarialExecution]:
        """The run's execution, or None when there is nothing to carry out."""
        if not self._gated and self._feasible.check(world, self._elapsed) is None:
            logger.warning(
                "%s: '%s' cannot %s now; nothing done",
                self.label,
                self._entity_name,
                self.kind.value.replace("_", " "),
            )
            return None
        command = self._feasible.command
        scene = self._feasible.scene
        if command is None or scene is None:
            return None
        return start_execution(
            scene, self._entity_name, command, self._following_mode, self.label
        )


class _AdversityDone(BaseCondition):
    """Holds once the action it watches has finished its run."""

    _action: AdversarialAction

    def __init__(self, action: AdversarialAction, label: str) -> None:
        super().__init__(label=f"{label}_done")
        self._action = action

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        if not self._action.finished:
            return None
        return ScenarioResult(
            passed=True,
            message=f"{self._action.label} is over",
            elapsed_seconds=elapsed,
        )


class AdversarialBrakeAction(AdversarialAction):
    """Brake hard in front of a follower that keeps coming.

    TeraSim's Follower adversity: fires with a follower close behind, the
    vehicle not speeding up and the follower not slowing, and brakes at the
    IDM's emergency deceleration for ``thresholds.duration``.  See
    :class:`AdversarialAction` for the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "adversarial_brake",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.BRAKE_CHECK,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class IgnoreLeaderAction(AdversarialAction):
    """Drive on as if the close, slower leader were not there.

    TeraSim's Leader adversity: fires when the IDM's free-flow acceleration
    beats its car-following one by more than
    ``thresholds.acceleration_gain``, and accelerates at the free-flow rate
    for ``thresholds.duration``.  See :class:`AdversarialAction` for the
    arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "ignore_leader",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.IGNORE_LEADER,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class AdversarialCutInAction(AdversarialAction):
    """Change lane into a gap too short for the vehicle behind there.

    TeraSim's LaneChange adversity: fires when a lane beside is blocked only
    by the vehicle behind there, closer than SUMO's secure gap, and changes
    into it in ``thresholds.lane_change_duration``.  See
    :class:`AdversarialAction` for the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "adversarial_cut_in",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.CUT_IN,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class AdversarialMergeAction(AdversarialAction):
    """Make a merging vehicle's way in harder.

    TeraSim's Merge adversity, beside a merging lane with traffic on it
    (:meth:`~..route.geometry.MapFeatures.merges`): one lane over, it speeds
    up as if the vehicle merging ahead were not there, when that beats
    giving way by more than ``thresholds.acceleration_gain``, for
    ``thresholds.duration``; two lanes over, it changes into the lane
    between in ``thresholds.lane_change_duration`` when nobody there is
    within ``thresholds.merge_headway``.  See :class:`AdversarialAction` for
    the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "adversarial_merge",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.MERGE,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class LaneChangeAbortAction(AdversarialAction):
    """Start into the lane beside, then swerve back.

    TeraSim's LanechangeAbort adversity: fires as the cut-in does, the
    vehicle moving and its lanelet long enough for the manoeuvre, and drifts
    half a lane toward that side and back in ``thresholds.lane_change_duration``.
    See :class:`AdversarialAction` for the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "lane_change_abort",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.LANE_CHANGE_ABORT,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class HeadOnAction(AdversarialAction):
    """Drive on at a leader coming the other way.

    TeraSim's Headon adversity: the Leader adversity's command, kept only
    where the lanelets the vehicle and its leader are on start more than
    ``thresholds.head_on_angle`` apart.  See :class:`AdversarialAction` for
    the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "head_on",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.HEAD_ON,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class RunTrafficRuleAction(AdversarialAction):
    """Run the red (or yellow) light the vehicle is braking for.

    TeraSim's TrafficRule adversity: fires when the vehicle is stopping for
    the light rather than for its leader, and accelerates at the free-flow
    rate for ``thresholds.duration``.  See
    :class:`AdversarialAction` for the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "run_traffic_rule",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.TRAFFIC_RULE,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class JaywalkAction(AdversarialAction):
    """Walk a pedestrian straight across the road, off a crosswalk.

    TeraSim's Jaywalking adversity: fires for a pedestrian off any
    crosswalk with a road nearby, and walks it across at its speed times a
    factor drawn from ``thresholds.pedestrian_speed_factor`` (seeded by
    *seed*), for up to ``thresholds.jaywalk_duration``.  See :class:`AdversarialAction` for the
    arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "jaywalk",
        once: bool = True,
        overrides: bool = False,
        seed: Optional[int] = None,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.JAYWALKING,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            seed=seed,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class PedestrianRunRedLightAction(AdversarialAction):
    """Walk a pedestrian waiting at a crosswalk across it on red.

    TeraSim's RunningRedLight adversity: fires for a pedestrian at most
    ``thresholds.waiting_speed`` fast within ``thresholds.crosswalk_reach``
    of a crosswalk whose Lanelet2 traffic light is red, and walks it from
    the crosswalk's near end to its far end at
    ``thresholds.pedestrian_max_speed``.  See :class:`AdversarialAction` for
    the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "pedestrian_run_red_light",
        once: bool = True,
        overrides: bool = False,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.RUNNING_RED_LIGHT,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            thresholds=thresholds,
            following_mode=following_mode,
        )


class StopCrossingAction(AdversarialAction):
    """Stop a pedestrian on the crosswalk a moment, then hurry it across.

    TeraSim's StopCrossing adversity: fires for a pedestrian on a crosswalk,
    stands it still for a time drawn from ``thresholds.stop_crossing_wait``
    (seeded by *seed*), then walks it a crosswalk's length on along its
    heading at ``thresholds.pedestrian_max_speed``.  See
    :class:`AdversarialAction` for the arguments.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        location: AdversityLocation,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "stop_crossing",
        once: bool = True,
        overrides: bool = False,
        seed: Optional[int] = None,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        super().__init__(
            entity_name,
            AdversityKind.STOP_CROSSING,
            location,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            seed=seed,
            thresholds=thresholds,
            following_mode=following_mode,
        )
