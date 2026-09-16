"""Driving a vehicle through whatever backend owns the run's traffic.

A manoeuvre asked of a vehicle is an *intent*: "change lane", "turn at the next
junction".  What the intent becomes depends entirely on what is driving --
``tm.set_path`` is TrafficManager vocabulary, a SUMO vehicle takes a new route,
an Autoware ego takes a goal and a policy-driven ego takes none of the three.

So the intent stops here.  The entity carries it to the
:class:`~autoware_carla_scenario.traffic.base.TrafficBackend` that drives it, and
the backend is the only place that knows a simulator API.  An action that reached
for the TrafficManager itself would work for exactly one kind of vehicle while
looking as though it worked for all of them.

Entities that are *not* backend-driven
(:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`,
:class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`)
override these and say so rather than inheriting a call that would be sent to
something which is not driving them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..constants import DEFAULT_TM_PORT
from ..traffic.base import LaneChangeDirection, TrafficBackend, TurnDirection

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

__all__ = ["BackendDriven"]


class BackendDriven:
    """Manoeuvres for a vehicle a traffic backend steers.

    Mixed into :class:`~autoware_carla_scenario.entity.ego.EgoVehicle` and
    :class:`~autoware_carla_scenario.entity.vehicle_entity.VehicleEntity`, the
    two entities the run's traffic model drives.  Both already own an ``actor``;
    this adds the backend the manoeuvres are delegated to.

    The backend is injected rather than passed to each call:
    :class:`~autoware_carla_scenario.ScenarioRunner` hands it to the ego and
    :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`
    to each NPC -- the same two places that already inject the CARLA client.

    An entity built outside a run has neither, and :meth:`set_client` is the
    path left for it: it names a TrafficManager, which is what such an entity
    always meant, and the manoeuvres resolve to a TrafficManager backend built
    on the spot.  An entity with neither a backend nor a client says so rather
    than failing inside CARLA.
    """

    #: Set by :meth:`set_traffic_backend`; ``None`` until then.
    _traffic_backend: Optional[TrafficBackend] = None
    #: Set by :meth:`set_client`; ``None`` until then.
    _tm_client: Optional["carla.Client"] = None
    _tm_port: int = DEFAULT_TM_PORT

    #: The lane a backend aimed this vehicle at, and the map it was read from.
    #: Written by whichever backend performs the lane change and read by
    #: whichever one judges it finished -- state of *this vehicle's* manoeuvre,
    #: so it lives on the vehicle: a backend is shared by the whole run, and the
    #: entity is the one thing there is exactly one of per manoeuvre.
    _lane_change_target: Optional[Tuple[int, int]] = None
    _lane_change_map: Optional[Any] = None

    def set_traffic_backend(self, backend: TrafficBackend) -> None:
        """Inject the backend that drives this entity.

        Takes precedence over :meth:`set_client` whichever order the two arrive
        in: a run that selected a traffic model means that model to drive, and
        the client is only ever the fallback's ingredient.
        """
        self._traffic_backend = backend

    def set_client(
        self, client: "carla.Client", tm_port: int = DEFAULT_TM_PORT
    ) -> None:
        """Inject the CARLA client a TrafficManager would be reached through.

        Kept because it is what external scenario packages call, and because it
        is still the whole truth for an entity built outside a run: no backend
        was selected, so the TrafficManager is what drives.
        """
        self._tm_client = client
        self._tm_port = tm_port

    # ------------------------------------------------------------------
    # Manoeuvres -- delegated, every one of them
    # ------------------------------------------------------------------

    def change_lane(self, world: "carla.World", direction: LaneChangeDirection) -> None:
        """Move one lane in *direction*."""
        backend = self._resolve_backend("change_lane")
        if backend is None:
            return
        backend.change_lane(self, world, direction)

    def lane_change_finished(self, world: "carla.World") -> bool:
        """Whether the manoeuvre has settled."""
        backend = self._resolve_backend("lane_change_finished")
        if backend is None:
            return False
        return backend.lane_change_finished(self, world)

    def turn_at_junction(
        self, world: "carla.World", direction: TurnDirection, **kwargs: Any
    ) -> None:
        """Go *direction* at the next junction ahead."""
        backend = self._resolve_backend("turn_at_junction")
        if backend is None:
            return
        backend.turn_at_junction(self, world, direction, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_backend(self, what: str) -> Optional[TrafficBackend]:
        """Return the backend driving this entity, or ``None`` with a warning.

        The fallback is built fresh rather than cached because a client can
        arrive after the first manoeuvre was attempted, and a cached backend
        holding the ``None`` from before would keep refusing afterwards.  It
        costs one small object per manoeuvre, which is a handful per run.
        """
        if self._traffic_backend is not None:
            return self._traffic_backend

        from ..traffic.config import TrafficManagerBackendConfig  # noqa: PLC0415
        from ..traffic.traffic_manager import TrafficManagerBackend  # noqa: PLC0415

        if self._tm_client is None:
            logger.debug(
                "%s: %s falls back to a TrafficManager with no client injected",
                type(self).__name__,
                what,
            )
        return TrafficManagerBackend(
            TrafficManagerBackendConfig(port=self._tm_port), client=self._tm_client
        )
