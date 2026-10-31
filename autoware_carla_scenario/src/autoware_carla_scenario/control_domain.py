"""Which actions control an entity: its control domains and who holds each.

ASAM OpenSCENARIO gives an entity four **control domains**: *longitudinal*
(how fast it goes along its path), *lateral* (where across it goes: lane
changes, turns, a route), *lighting* (its lights) and *animation* (a
pedestrian's gestures, a vehicle's moving parts) -- the first two its
*movement*, the last two its *appearance*.  An action takes control of the
domains of its entity that it commands: a ``SpeedAction`` the longitudinal
one, a ``LaneChangeAction`` the lateral one, a ``FollowTrajectoryAction`` both.

Two actions that command the same domain of the same entity conflict.  What
happens then is the newer action's to say:

* by default it is an error (:class:`ControlConflictError`), raised as the
  newer one fires, before it has commanded anything: a scenario whose actions
  fight over an entity says two things at once, and which one CARLA ends up
  doing is an accident of the tick order, not what the author meant;
* an action that overrides takes the domain over, as OpenSCENARIO has every
  action do (*Actions at runtime*): the action holding it is stopped -- it
  goes to ``completeState`` -- and gives back every domain it held.

A domain is held by the action that took it (:class:`ControlSlots`):

* an action that runs on -- one with an end condition, or one that reissues
  its command every tick -- holds it until its run ends;
* one that is over the moment it is commanded holds it for that tick alone,
  so two such commands in the same tick conflict and one on the next tick
  does not.

"The same tick" is the scenario time an action fires at.  A post-tick action
and the pre-tick action of the next loop see the same time, and they do act on
the same simulation step.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = [
    "MOVEMENT",
    "ControlConflictError",
    "ControlDomain",
    "ControlSlots",
    "control_slots",
    "reset_control_slots",
]


class ControlDomain(enum.Enum):
    """A domain of an entity an action may control (OpenSCENARIO's
    ``ControllerType`` domains)."""

    LONGITUDINAL = "longitudinal"
    LATERAL = "lateral"
    LIGHTING = "lighting"
    ANIMATION = "animation"


#: Both domains of an entity's movement (OpenSCENARIO's ``movement``).
MOVEMENT: tuple[ControlDomain, ...] = (
    ControlDomain.LONGITUDINAL,
    ControlDomain.LATERAL,
)


class ControlConflictError(RuntimeError):
    """An action fired to command a domain of an entity another action holds.

    Attributes:
        entity: The entity's role name.
        domain: The domain both command.
        holder: The label of the action holding it.
        claimant: The label of the action that fired.
    """

    entity: str
    domain: ControlDomain
    holder: str
    claimant: str

    def __init__(
        self, entity: str, domain: ControlDomain, holder: str, claimant: str
    ) -> None:
        super().__init__(
            f"{claimant}: cannot take {domain.value} control of '{entity}': "
            f"{holder} holds it"
        )
        self.entity = entity
        self.domain = domain
        self.holder = holder
        self.claimant = claimant


@dataclass
class _Hold:
    """A domain taken by an action."""

    owner: int
    label: str
    at: float
    runs: bool


class ControlSlots:
    """Who holds each control domain of each entity."""

    _holds: dict[tuple[str, str], _Hold]
    _next: int

    def __init__(self) -> None:
        self._holds = {}
        self._next = 0

    def take(
        self,
        label: str,
        claims: Sequence[tuple[str, ControlDomain]],
        at: float,
        runs: bool,
        overrides: bool = False,
    ) -> tuple[int, list[int]]:
        """Take each ``(entity, domain)`` of *claims* for the action *label*,
        fired at scenario time *at*: until it is :meth:`released <release>`
        if it *runs* on, else for that tick alone.

        Args:
            overrides: Take a domain another action holds, releasing
                everything that action holds, rather than refuse.

        Returns:
            The token :meth:`release` gives them back with, and the tokens of
            the actions it overrode, to be stopped.

        Raises:
            ControlConflictError: If another action holds one of them and
                *overrides* is false; none is taken then.
        """
        overridden: list[int] = []
        for entity, domain in claims:
            held = self._holder(entity, domain, at)
            if held is None:
                continue
            hold: _Hold = held
            if not overrides:
                raise ControlConflictError(entity, domain, hold.label, label)
            if hold.owner not in overridden:
                overridden.append(hold.owner)
        for owner in overridden:
            self.release(owner)
        self._next += 1
        owner = self._next
        for entity, domain in claims:
            self._holds[(entity, domain.value)] = _Hold(owner, label, at, runs)
        return owner, overridden

    def release(self, owner: int) -> None:
        """Give back what the action holding *owner* took."""
        for key in [key for key, hold in self._holds.items() if hold.owner == owner]:
            del self._holds[key]

    def holder(self, entity: str, domain: ControlDomain, at: float) -> Optional[str]:
        """The label of the action holding *domain* of *entity* at scenario
        time *at*, or ``None``."""
        held = self._holder(entity, domain, at)
        if held is None:
            return None
        hold: _Hold = held
        return hold.label

    def clear(self) -> None:
        """Forget every hold."""
        self._holds = {}

    def _holder(self, entity: str, domain: ControlDomain, at: float) -> Optional[_Hold]:
        key = (entity, domain.value)
        if key not in self._holds:
            return None
        hold = self._holds[key]
        if hold.runs or hold.at == at:
            return hold
        return None


_SLOTS = ControlSlots()


def control_slots() -> ControlSlots:
    """The run's holds: process-wide, as the entity registry is."""
    return _SLOTS


def reset_control_slots() -> None:
    """Forget every hold, as a run begins: one a previous run left (an action
    that never finished) is not this run's."""
    _SLOTS.clear()
