"""What an action reports for the run's log, beside its lifecycle (#397).

Kept apart from :mod:`.base`, with nothing to import, so the trajectory
recorder and the runner can read it without pulling in what an action
needs to act.
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
