"""A traffic backend's background vehicle, as an entity an action can drive (#417).

Background traffic has no entity: the backend spawned it and drives it, and
nothing in a scenario names it.  An adversity that takes one over for a while
(:class:`~autoware_carla_scenario.actions.adversarial_traffic.AdversarialTrafficAction`)
needs what the trajectory follower drives -- an ``actor``, a ``role_name``,
and :meth:`~..traffic.driven.BackendDriven.release_from_traffic` /
:meth:`~..traffic.driven.BackendDriven.resume_traffic` -- so it wraps the
vehicle in a :class:`BackgroundVehicle`, registers it under the vehicle's
role name for as long as it drives it, and forgets it afterwards.  The
backend still owns the actor: the wrapper never spawns or destroys one, and
never lets the backend take it out of the world
(:meth:`~..traffic.base.DrivenEntity._replace_actor`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..traffic.base import TrafficBackend
from ..traffic.driven import BackendDriven

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["BackgroundVehicle"]


class BackgroundVehicle(BackendDriven):
    """The background vehicle *actor*, known as *role_name*, of *backend*.

    Args:
        role_name: The name the vehicle answers to (its ``role_name``).
        actor: Its CARLA actor.
        backend: The backend that drives it.
    """

    #: Driven by a traffic model, which the follower may take it from.
    use_autopilot = True

    _role_name: str
    _actor: "carla.Actor"

    def __init__(
        self, role_name: str, actor: "carla.Actor", backend: TrafficBackend
    ) -> None:
        self._role_name = str(role_name)
        self._actor = actor
        self.set_traffic_backend(backend)

    @property
    def role_name(self) -> str:
        """The name it answers to."""
        return self._role_name

    @property
    def actor(self) -> "carla.Actor":
        """Its CARLA actor, the backend's."""
        return self._actor
