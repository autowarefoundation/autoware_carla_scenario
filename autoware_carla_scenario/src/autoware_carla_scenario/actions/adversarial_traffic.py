"""NADE as an action: adversarial traffic around the ego, importance sampled (#395).

:class:`AdversarialTrafficAction` runs TeraSim's NADE loop
(:mod:`~..adversity.nade`) every tick of the run: of the NPCs and
pedestrians around the ego, the one whose adversity would challenge it is
made to carry it out -- with the importance-sampling probability rather
than the naturalistic one -- and :attr:`~AdversarialTrafficAction.weight`
keeps the likelihood ratio that makes the run's statistics unbiased.

What it drives is what a scenario registers: named NPCs and pedestrians.
A traffic backend's background vehicles have no entity to drive them
through yet (#417).  The ego is never driven; its turn indicators say
whether it is predicted to change lane.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ..action_state import ActionState
from ..adversity.command import AdversarialCommand
from ..adversity.nade import (
    DEFAULT_NADE,
    ImportanceSampler,
    NadeAdversity,
    NadeChoice,
    NadeSettings,
)
from ..adversity.observation import Scene
from ..adversity.snapshot import scene_at
from ..adversity.triggers import DEFAULT_THRESHOLDS, TriggerThresholds
from ..conditions.base import BaseCondition, find_actor_by_role_name
from ..constants import EGO_ROLE_NAME
from ..trajectory.model import TrajectoryFollowingMode
from ..utils.turn_indicator import TurnIndicatorState, is_vehicle, turn_indicators_of
from ._adversarial_execution import AdversarialExecution, drivable, start_execution
from .background_traffic import _Never
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversarialTrafficAction"]

logger = logging.getLogger(__name__)


class AdversarialTrafficAction(BaseAction):
    """NADE: sample adversities around the ego, keeping the IS weight.

    Runs for the rest of the run once triggered (every tick, an environment
    action), and carries out at most one adversity at a time -- with its
    NPC victims' response to it (:mod:`~..adversity.avoidance`), when the
    settings challenge NPCs.

    Args:
        adversities: The adversities to sample, in order: an agent's is the
            first whose trigger fires.
        condition: When it starts; at once by default.
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        seed: Seeds the sampling and the triggers' draws.
        settings: NADE's numbers; TeraSim's by default.
        thresholds: The triggers' numbers; TeraSim's by default.
        following_mode: How a sampled adversity is driven: ``FOLLOW`` (the
            vehicle's own motion, default) or ``POSITION``.
    """

    REISSUES_BY_DEFAULT = True

    # Declared for the static check (docs/typecheck.md).
    _adversities: list[NadeAdversity]
    _sampler: ImportanceSampler
    _following_mode: TrajectoryFollowingMode
    _executions: list[AdversarialExecution]
    _choices: list[NadeChoice]
    _elapsed: float

    def __init__(
        self,
        adversities: list[NadeAdversity],
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "adversarial_traffic",
        seed: int = 0,
        settings: NadeSettings = DEFAULT_NADE,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
    ) -> None:
        for adversity in adversities:
            if not 0.0 <= adversity.probability <= 1.0:
                raise ValueError(
                    f"{adversity.kind.value}: a probability is in [0, 1], "
                    f"not {adversity.probability}"
                )
        forever: BaseCondition = _Never()
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=True,
            until=forever,
        )
        self._adversities = list(adversities)
        self._sampler = ImportanceSampler(seed, settings, thresholds)
        self._following_mode = following_mode
        self._executions = []
        self._choices = []
        self._elapsed = 0.0

    @property
    def weight(self) -> float:
        """The run's importance-sampling weight so far: 1 until a challenger."""
        return self._sampler.weight

    @property
    def choices(self) -> list[NadeChoice]:
        """Every adversity sampled so far, in order."""
        return list(self._choices)

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Remember the scenario time, which :meth:`execute` is not given."""
        self._elapsed = elapsed
        super().tick(world, elapsed)

    def execute(self, world: "carla.World") -> None:
        """Drive the adversity under way, or sample the next one."""
        if self.state is ActionState.STANDBY:
            self._executions = []
        for running in self._executions:
            running.tick(world, self._elapsed)
        self._executions = [run for run in self._executions if not run.done]
        if self._executions:
            return
        if self._sampler.cooling(self._elapsed):
            return  # nothing to sample: no need to read the world
        scene = scene_at(world)
        choice = self._sampler.step(
            scene,
            str(EGO_ROLE_NAME),
            _indicator(world),
            {name for name in scene.agents if drivable(name)},
            self._adversities,
            self._elapsed,
        )
        if choice is None:
            return
        sampled: NadeChoice = choice
        self._choices.append(sampled)
        logger.info(
            "%s: '%s' carries out %s (q=%.3g, p=%.3g), weight %.6g",
            self.label,
            sampled.name,
            sampled.adversity.kind.value,
            sampled.is_probability,
            sampled.adversity.probability,
            self.weight,
        )
        self._start(world, scene, sampled.name, sampled.command)
        for victim, response in sampled.responses:
            logger.info("%s: '%s' responds: %s", self.label, victim, response)
            self._start(world, scene, victim, response)

    def _start(
        self,
        world: "carla.World",
        scene: Scene,
        name: str,
        command: AdversarialCommand,
    ) -> None:
        """Start driving *command* on *name*, from this tick."""
        started = start_execution(
            scene, name, command, self._following_mode, f"{self.label}:{name}"
        )
        if started is None:
            return
        execution: AdversarialExecution = started
        self._executions.append(execution)
        execution.tick(world, self._elapsed)

    def close(self) -> None:
        """Close the conditions, the action's and those of what it drives."""
        super().close()
        for running in self._executions:
            running.close()


def _indicator(world: "carla.World") -> str:
    """The side the ego's turn indicator shows: ``"left"``, ``"right"`` or ``""``."""
    found = find_actor_by_role_name(world, EGO_ROLE_NAME)
    if found is None:
        return ""
    actor: "carla.Actor" = found
    if not is_vehicle(actor):
        return ""
    state = turn_indicators_of(actor)
    if state == TurnIndicatorState.LEFT:
        return "left"
    if state == TurnIndicatorState.RIGHT:
        return "right"
    return ""
