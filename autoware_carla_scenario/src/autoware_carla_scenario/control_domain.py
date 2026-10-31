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

A domain is held by the action that took it
(:func:`~autoware_carla_scenario.actions.base.reset_control_slots` forgets
every hold, as a run begins):

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

__all__ = [
    "MOVEMENT",
    "ControlConflictError",
    "ControlDomain",
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
