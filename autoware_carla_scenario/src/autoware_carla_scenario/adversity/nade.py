"""TeraSim's NADE: which adversity to sample, with an unbiased weight (#395).

Naturalistic and Adversarial Driving Environment.  Each tick, every agent
around the ego that may carry out an adversity is a choice between its
normal behaviour (probability ``1 - p``) and the adversity (``p``).
Naturalistically ``p`` is tiny, so NADE samples the adversity more often
-- with the importance-sampling probability ``q`` -- whenever it would
challenge the ego, and keeps the weight that undoes the bias: ``p / q``
when the adversity is sampled, ``(1 - p) / (1 - q)`` when it is not.  The
weight's product over a run is the run's likelihood ratio, so a statistic
weighted by it is the naturalistic one.

As TeraSim's ``NADE`` (``envs/nade.py``, ``envs/nade_with_av.py``):

* **Candidates** are the agents within ``control_radius`` of the ego: the
  ``max_controlled`` nearest vehicles, and every pedestrian.
* **Adversity**: each candidate's is the first of the list whose trigger
  fires (:func:`~.triggers.adversity_command`, the location gate
  included), giving ``{normal: 1 - p, adversarial: p}``.
* **Prediction**: ``horizon`` seconds every ``resolution``
  (:func:`~.command.predict_command`).  The ego is predicted
  normal -- its acceleration held along its lane, or a lane change when
  its turn indicator says so.
* **Challenge**: the candidate's adversarial future conflicts with the
  ego's normal future (:func:`~.collision.trajectories_intersect`); a
  pedestrian's only while the ego moves.
* **Importance sampling**: ``q = clip(p * magnitude, 0,
  max_is_probability)``, or ``fixed_is_probability`` when set.  At most one
  adversity is sampled a tick, the nearest challenger first, and none for
  ``cooldown`` seconds after one.

NADE discards the command TeraSim's ``AdversityManager`` samples and uses
only the distribution; so does this.  Avoidability -- which multiplies the
magnitude for an unavoidable conflict -- is #396.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .collision import trajectories_intersect
from .command import AdversarialCommand, predict_command
from .geometry import PredictedState
from .location import AdversityLocation
from .observation import AgentSnapshot, Scene
from .triggers import (
    DEFAULT_THRESHOLDS,
    AdversityKind,
    TriggerThresholds,
    adversity_command,
)

__all__ = [
    "DEFAULT_NADE",
    "ImportanceSampler",
    "NadeAdversity",
    "NadeChoice",
    "NadeSettings",
    "candidates",
    "ego_command",
    "is_probability",
]


@dataclass(frozen=True)
class NadeAdversity:
    """One entry of NADE's list: an adversity, where, and how likely.

    Attributes:
        kind: The adversity.
        location: Where it may fire.
        probability: Its naturalistic probability ``p`` a tick, when its
            trigger holds.
    """

    kind: AdversityKind
    location: AdversityLocation
    probability: float


@dataclass(frozen=True)
class NadeSettings:
    """NADE's numbers; TeraSim's by default.

    Attributes:
        max_controlled: How many of the nearest vehicles are candidates.
        control_radius: How near the ego (m) a candidate is.
        horizon: How far ahead futures are predicted (s).
        resolution: How far apart their states are (s).
        magnitude: ``M`` of ``q = clip(p * M, 0, max_is_probability)``.
        max_is_probability: The most ``q`` is.
        fixed_is_probability: ``q`` for every adversity when above 0, in
            place of the clip (TeraSim's ``adversity_sampling_probability``).
        cooldown: How long after a sampled adversity none is (s).
    """

    max_controlled: int = 5
    control_radius: float = 50.0
    horizon: float = 5.0
    resolution: float = 0.5
    magnitude: float = 20.0
    max_is_probability: float = 0.05
    fixed_is_probability: float = 0.0
    cooldown: float = 5.5


DEFAULT_NADE = NadeSettings()


@dataclass(frozen=True)
class NadeChoice:
    """The adversity NADE sampled this tick.

    Attributes:
        name: Who carries it out.
        adversity: The list's entry it is.
        command: The command its trigger gave.
        is_probability: ``q`` it was sampled with.
    """

    name: str
    adversity: NadeAdversity
    command: AdversarialCommand
    is_probability: float


def is_probability(probability: float, settings: NadeSettings) -> float:
    """``q`` for an adversity of naturalistic *probability* ``p``."""
    if settings.fixed_is_probability > 0.0:
        return settings.fixed_is_probability
    return min(max(probability * settings.magnitude, 0.0), settings.max_is_probability)


def candidates(
    scene: Scene,
    ego: str,
    controllable: set[str],
    settings: NadeSettings = DEFAULT_NADE,
) -> list[str]:
    """The agents NADE considers, nearest first.

    The ``max_controlled`` nearest vehicles within ``control_radius`` of
    *ego*, then every pedestrian there; only those named in
    *controllable* (what an action can drive), never *ego*.
    """
    me = scene.agents[ego]
    vehicles: list[tuple[float, str]] = []
    pedestrians: list[tuple[float, str]] = []
    for name, agent in scene.agents.items():
        if name == ego or name not in controllable:
            continue
        distance = math.hypot(agent.x - me.x, agent.y - me.y)
        if distance > settings.control_radius:
            continue
        if agent.footprint.is_vehicle:
            vehicles.append((distance, name))
        else:
            pedestrians.append((distance, name))
    vehicles.sort()
    pedestrians.sort()
    near = [name for _, name in vehicles[: max(settings.max_controlled, 0)]]
    return near + [name for _, name in pedestrians]


def ego_command(
    scene: Scene, ego: str, indicator: str, settings: NadeSettings = DEFAULT_NADE
) -> AdversarialCommand:
    """The ego's predicted normal behaviour.

    A lane change its turn *indicator* (``"left"``, ``"right"`` or ``""``)
    announces, where there is a lane to change into; else its acceleration
    held -- as TeraSim predicts its AV.
    """
    seen = scene.vehicle(ego)
    if indicator == "left" and seen.left is not None:
        return AdversarialCommand.change_lane("left")
    if indicator == "right" and seen.right is not None:
        return AdversarialCommand.change_lane("right")
    return AdversarialCommand.accelerate(seen.agent.acceleration, settings.horizon)


class ImportanceSampler:
    """NADE's sampling over a run: the weight, the cooldown, the draws.

    Args:
        seed: Seeds the draws (and a jaywalker's speed).
        settings: NADE's numbers.
        thresholds: The triggers' numbers.
    """

    settings: NadeSettings
    thresholds: TriggerThresholds
    weight: float
    _draws: random.Random
    _triggers: random.Random
    _sampled_at: float

    def __init__(
        self,
        seed: int = 0,
        settings: NadeSettings = DEFAULT_NADE,
        thresholds: TriggerThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self.settings = settings
        self.thresholds = thresholds
        self.weight = 1.0
        # Apart, so a trigger's draw (a jaywalker's speed) does not shift
        # the sampling's sequence.
        self._draws = random.Random(seed)
        self._triggers = random.Random(seed + 1)
        self._sampled_at = -math.inf

    def cooling(self, elapsed: float) -> bool:
        """Whether *elapsed* is within the cooldown of the last sample."""
        return elapsed - self._sampled_at < self.settings.cooldown

    def step(
        self,
        scene: Scene,
        ego: str,
        indicator: str,
        controllable: set[str],
        adversities: list[NadeAdversity],
        elapsed: float,
    ) -> Optional[NadeChoice]:
        """Sample this tick: the adversity to carry out, or None.

        The candidates are those of :func:`candidates` among *controllable*,
        and the ego is predicted from its turn *indicator*
        (:func:`ego_command`).  Updates :attr:`weight` for every challenger
        the tick decides on.
        """
        if self.cooling(elapsed) or ego not in scene.agents:
            return None
        settings = self.settings
        names = candidates(scene, ego, controllable, settings)
        if not names:
            return None
        me = scene.agents[ego]
        # Predicted once a trigger fires: most ticks none does.
        ego_future: list[PredictedState] = []
        for name in names:
            found = self._adversity(scene, name, adversities)
            if found is None:
                continue
            adversity, command = found
            if not ego_future:
                try:
                    ego_future = predict_command(
                        scene.features,
                        scene.vehicle(ego),
                        ego_command(scene, ego, indicator, settings),
                        settings.resolution,
                        settings.horizon,
                    )
                except ValueError:
                    return None  # off the lanes: no future to challenge
            if not self._challenges(scene, name, command, me, ego_future):
                continue
            p = adversity.probability
            q = is_probability(p, settings)
            if q <= 0.0:
                continue  # never sampled: nothing to decide
            if self._draws.random() < q:
                self.weight *= p / q
                self._sampled_at = elapsed
                return NadeChoice(name, adversity, command, q)
            self.weight *= (1.0 - p) / (1.0 - q)
        return None

    def _adversity(
        self, scene: Scene, name: str, adversities: list[NadeAdversity]
    ) -> Optional[tuple[NadeAdversity, AdversarialCommand]]:
        """The first of *adversities* agent *name* may carry out now."""
        for adversity in adversities:
            command = adversity_command(
                adversity.kind,
                scene,
                name,
                adversity.location,
                self._triggers,
                self.thresholds,
            )
            if command is not None:
                # A typed local: Codon unwraps the Optional here, not in the
                # tuple (docs/typecheck.md).
                fires: AdversarialCommand = command
                return adversity, fires
        return None

    def _challenges(
        self,
        scene: Scene,
        name: str,
        command: AdversarialCommand,
        me: AgentSnapshot,
        ego_future: list[PredictedState],
    ) -> bool:
        """Whether *name* carrying out *command* conflicts with the ego."""
        agent = scene.agents[name]
        if not agent.footprint.is_vehicle and me.speed < 1.0:
            return False  # a pedestrian challenges a moving ego only
        try:
            future = predict_command(
                scene.features,
                scene.vehicle(name),
                command,
                self.settings.resolution,
                self.settings.horizon,
            )
        except ValueError:
            return False
        return trajectories_intersect(future, agent.footprint, ego_future, me.footprint)
