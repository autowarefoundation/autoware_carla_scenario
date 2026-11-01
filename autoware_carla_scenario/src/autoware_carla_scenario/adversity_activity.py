"""Which adversities are under way now, for the conditions that ask.

An adversarial action and NADE both carry an adversity out by driving its
path (:class:`~.actions._adversarial_execution.AdversarialExecution`): the
execution says here when it starts driving and when it hands the road user
back, so a condition can ask who is carrying out what, whichever action
started it -- a named NPC's own action, or NADE on a background vehicle.

A module of its own at the package's top, importing nothing, as
:mod:`.adversity_event` is: the conditions read it without pulling in the
actions.  The runner forgets what is under way before each scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ActiveAdversity",
    "active_adversities",
    "begin_adversity",
    "clear_adversities",
    "end_adversity",
]


@dataclass(frozen=True)
class ActiveAdversity:
    """An adversity a road user is carrying out now.

    Attributes:
        agent: Who carries it out (its role name).
        kind: Its kind (an ``AdversityKind`` value).
        source: The label of what drives it (the action, or NADE's).
        responding_to: For a victim's response to another's adversity (NADE's
            avoid or accept), whose it is; empty for the adversity itself.
        since: The scenario time it started (s).
    """

    agent: str
    kind: str
    source: str
    responding_to: str
    since: float


_ACTIVE: dict[str, ActiveAdversity] = {}


def begin_adversity(adversity: ActiveAdversity) -> None:
    """Note that *adversity* is under way (one per road user)."""
    _ACTIVE[adversity.agent] = adversity


def end_adversity(agent: str) -> None:
    """Note that *agent* carries out nothing any more."""
    if agent in _ACTIVE:
        del _ACTIVE[agent]


def active_adversities() -> list[ActiveAdversity]:
    """The adversities under way, in the order they started."""
    return list(_ACTIVE.values())


def clear_adversities() -> None:
    """Forget every adversity: a new scenario starts with none."""
    _ACTIVE.clear()
