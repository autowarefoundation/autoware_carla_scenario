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
from ..adversity.command import AdversarialCommand, plan_command
from ..adversity.nade import (
    DEFAULT_NADE,
    ImportanceSampler,
    NadeAdversity,
    NadeChoice,
    NadeSettings,
    candidates,
)
from ..adversity.observation import Scene
from ..adversity.snapshot import scene_at
from ..adversity.triggers import DEFAULT_THRESHOLDS, TriggerThresholds
from ..conditions.base import BaseCondition, find_actor_by_role_name
from ..constants import EGO_ROLE_NAME
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..trajectory.model import TrajectoryFollowingMode
from ..utils.turn_indicator import TurnIndicatorState, is_vehicle, turn_indicators_of
from ._adversarial_execution import AdversarialExecution
from .background_traffic import _Never
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversarialTrafficAction"]

logger = logging.getLogger(__name__)


class AdversarialTrafficAction(BaseAction):
    """NADE: sample adversities around the ego, keeping the IS weight.

    Runs for the rest of the run once triggered (every tick, an environment
    action), and carries out at most one adversity at a time.

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
    _execution: Optional[AdversarialExecution]
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
        self._execution = None
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
            self._execution = None
        if self._execution is not None:
            running: AdversarialExecution = self._execution
            running.tick(world, self._elapsed)
            if not running.done:
                return
            self._execution = None
        scene = scene_at(world)
        ego = str(EGO_ROLE_NAME)
        if ego not in scene.agents:
            return
        choice = self._sampler.step(
            scene,
            ego,
            _ego_command(world, scene, ego, self._sampler.settings),
            candidates(scene, ego, _controllable(scene, ego), self._sampler.settings),
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
        try:
            path = plan_command(
                scene.features, scene.vehicle(sampled.name), sampled.command
            )
        except ValueError as error:
            logger.warning("%s: %s; nothing done", self.label, error)
            return
        execution = AdversarialExecution(
            sampled.name,
            path,
            self._following_mode,
            f"{self.label}:{sampled.name}",
        )
        self._execution = execution
        execution.tick(world, self._elapsed)

    def close(self) -> None:
        """Close the conditions, the action's and the adversity's."""
        super().close()
        if self._execution is not None:
            running: AdversarialExecution = self._execution
            running.close()


def _controllable(scene: Scene, ego: str) -> set[str]:
    """The agents of *scene* an action can drive: the registered entities."""
    names: set[str] = set()
    for name in scene.agents:
        if name == ego:
            continue
        if find_vehicle_entity(name) is not None:
            names.add(name)
        elif find_pedestrian_entity(name) is not None:
            names.add(name)
    return names


def _ego_command(
    world: "carla.World", scene: Scene, ego: str, settings: NadeSettings
) -> AdversarialCommand:
    """The ego's predicted normal behaviour: a lane change its indicator
    announces (where there is a lane), else its acceleration held."""
    seen = scene.vehicle(ego)
    held = AdversarialCommand.accelerate(seen.agent.acceleration, settings.horizon)
    found = find_actor_by_role_name(world, ego)
    if found is None:
        return held
    actor: "carla.Actor" = found
    if not is_vehicle(actor):
        return held
    state = turn_indicators_of(actor)
    if state == TurnIndicatorState.LEFT and seen.left is not None:
        return AdversarialCommand.change_lane("left")
    if state == TurnIndicatorState.RIGHT and seen.right is not None:
        return AdversarialCommand.change_lane("right")
    return held
