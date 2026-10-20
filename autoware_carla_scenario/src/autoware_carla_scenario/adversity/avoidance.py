"""Who an adversity challenges, whether they can avoid it, and how they respond (#396).

TeraSim's ``utils/nade/avoidance.py``, on predicted futures:

* **Victims** are the agents whose normal future the adversarial one
  conflicts with (:func:`victims`): the ego, as TeraSim's AV-centred NADE
  checks, and with ``challenge_npcs`` the other candidates too.
* **Avoidability** (:func:`avoidable`): each victim's *avoid* future -- an
  emergency brake held to the horizon -- is checked against the adversarial
  future again, with a 0.5 m buffer.  A conflict no victim can brake out of
  is unavoidable: an unreasonable adversity, whose naturalistic probability
  is cut by ``unavoidable_factor`` (3e-4) and whose IS magnitude is
  multiplied by ``unavoidable_magnitude`` (40).
* **Response** (:func:`respond`): once an adversity is sampled, NPC victims
  avoid it -- brake at the emergency deceleration for 3 s -- with the IS
  probability ``avoid_is_probability`` (0.2) against the naturalistic 0.99,
  else accept it (hold their speed for 2 s).  An ego victim is never
  driven (TeraSim's ``AV_neglected``), and nothing is drawn for an
  unavoidable conflict: the victims accept it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .collision import trajectories_intersect
from .command import AdversarialCommand
from .geometry import AgentFootprint, PredictedState

__all__ = [
    "AVOID",
    "ACCEPT",
    "DEFAULT_AVOIDANCE",
    "AvoidanceSettings",
    "Response",
    "accept_command",
    "avoid_command",
    "avoidable",
    "respond",
    "victims",
]

#: A victim's responses.
AVOID = "avoid"
ACCEPT = "accept"


@dataclass(frozen=True)
class AvoidanceSettings:
    """TeraSim's avoidance numbers.

    Attributes:
        buffer: How much further apart (m) an avoid future must stay.
        unavoidable_factor: What an unavoidable adversity's probability is
            multiplied by.
        unavoidable_magnitude: What its IS magnitude is multiplied by.
        avoid_probability: The naturalistic probability that a victim avoids.
        avoid_is_probability: The IS probability that it does.
        avoid_duration: How long an avoiding victim brakes (s).
        accept_duration: How long an accepting victim holds its speed (s).
    """

    buffer: float = 0.5
    unavoidable_factor: float = 3e-4
    unavoidable_magnitude: float = 40.0
    avoid_probability: float = 0.99
    avoid_is_probability: float = 0.2
    avoid_duration: float = 3.0
    accept_duration: float = 2.0


DEFAULT_AVOIDANCE = AvoidanceSettings()


@dataclass(frozen=True)
class Response:
    """How the victims respond to a sampled adversity.

    Attributes:
        mode: :data:`AVOID`, :data:`ACCEPT`, or ``""`` when no victim is
            driven (the ego is one).
        weight: The factor it puts on the run's weight.
    """

    mode: str
    weight: float


def avoid_command(
    emergency_deceleration: float, settings: AvoidanceSettings = DEFAULT_AVOIDANCE
) -> AdversarialCommand:
    """A victim's emergency brake."""
    return AdversarialCommand.accelerate(
        -emergency_deceleration, settings.avoid_duration
    )


def accept_command(
    settings: AvoidanceSettings = DEFAULT_AVOIDANCE,
) -> AdversarialCommand:
    """A victim holding its speed, as if the adversity were not there."""
    return AdversarialCommand.accelerate(0.0, settings.accept_duration)


def victims(
    future: list[PredictedState],
    footprint: AgentFootprint,
    futures: dict[str, list[PredictedState]],
    footprints: dict[str, AgentFootprint],
    buffer: float = 0.0,
) -> list[str]:
    """The agents of *futures* the adversarial *future* conflicts with, in
    the order given; each further apart by *buffer* (m)."""
    return [
        name
        for name, other in futures.items()
        if trajectories_intersect(future, footprint, other, footprints[name], buffer)
    ]


def avoidable(
    future: list[PredictedState],
    footprint: AgentFootprint,
    avoid_futures: dict[str, list[PredictedState]],
    footprints: dict[str, AgentFootprint],
    settings: AvoidanceSettings = DEFAULT_AVOIDANCE,
) -> bool:
    """Whether every victim can brake out of the adversarial *future*."""
    return not victims(future, footprint, avoid_futures, footprints, settings.buffer)


def respond(
    rng: random.Random,
    ego_is_victim: bool,
    can_avoid: bool,
    settings: AvoidanceSettings = DEFAULT_AVOIDANCE,
) -> Response:
    """Draw the NPC victims' response to a sampled adversity."""
    if ego_is_victim:
        return Response("", 1.0)  # TeraSim's AV_neglected: the ego drives itself
    if not can_avoid:
        return Response(ACCEPT, 1.0)  # nothing to choose: they cannot avoid it
    q = settings.avoid_is_probability
    p = settings.avoid_probability
    if rng.random() < q:
        return Response(AVOID, p / q)
    return Response(ACCEPT, (1.0 - p) / (1.0 - q))
