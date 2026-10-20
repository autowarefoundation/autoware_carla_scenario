"""What an action reports for the run's log, beside its lifecycle (#397).

A module of its own at the package's top, importing nothing, so the
trajectory recorder, the runner and every action read it without pulling
in the actions or the adversity package.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AdversityEvent"]


@dataclass(frozen=True)
class AdversityEvent:
    """An adversity NADE sampled, as the trajectory log records it.

    Attributes:
        agent: Who carries it out.
        kind: Its kind (an ``AdversityKind`` value).
        location: Where it may fire (an ``AdversityLocation`` value).
        probability: Its naturalistic probability ``p``, as sampled (cut for
            an unavoidable one).
        is_probability: The importance-sampling probability ``q``.
        weight: The run's importance-sampling weight once it was sampled.
        avoidable: Whether its victims could brake out of it.
        victims: The NPCs that respond to it.
        response: Theirs: ``"avoid"``, ``"accept"``, or ``""`` (none).
    """

    agent: str
    kind: str
    location: str
    probability: float
    is_probability: float
    weight: float
    avoidable: bool
    victims: list[str]
    response: str
