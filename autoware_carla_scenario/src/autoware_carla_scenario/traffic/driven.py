"""The vehicle side of the traffic seam.

A manoeuvre asked of a vehicle is an *intent*: "change lane", "turn at the next
junction".  What the intent becomes depends entirely on what is driving --
``tm.set_path`` is TrafficManager vocabulary, a SUMO vehicle takes a new route,
an Autoware ego takes a goal and a policy-driven ego takes none of the three.

So the intent stops here.  The entity carries it to the
:class:`~autoware_carla_scenario.traffic.base.TrafficBackend` that drives it, and
the backend is the only place that knows a simulator API.  An action that reached
for the TrafficManager itself would work for exactly one kind of vehicle while
looking as though it worked for all of them.

This module lives beside :mod:`~autoware_carla_scenario.traffic.base` rather than
in :mod:`~autoware_carla_scenario.entity` because a seam has two halves and they
are one design: ``TrafficBackend`` is what a traffic model must provide,
:class:`BackendDriven` is what a vehicle must offer for one to drive it, and the
two are only meaningful together.  It is mixed into entities, but it is not an
entity -- it holds no actor, spawns nothing, and names no CARLA type.

Entities that are *not* backend-driven
(:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`,
:class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`)
override these and say so rather than inheriting a call that would be sent to
something which is not driving them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from ..constants import DEFAULT_TM_PORT
from .base import DrivenEntity, LaneChangeDirection, TrafficBackend, TurnDirection

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose

logger = logging.getLogger(__name__)

__all__ = ["BackendDriven", "ROUTE_POINT_STEP_M"]

#: Metres between the points of the route a backend-driven vehicle is handed
#: (:meth:`BackendDriven.route_to`).
ROUTE_POINT_STEP_M: float = 5.0

#: The TrafficManager an entity with no client at all is driven by: it can
#: answer the questions that need no CARLA call and reports the missing client
#: for the rest.  Shared because it holds no per-vehicle state -- a manoeuvre's
#: bookkeeping lives on the entity -- and built once, on first use, so importing
#: this module stays free.
_CLIENTLESS_BACKEND: Optional[TrafficBackend] = None


def _clientless() -> TrafficBackend:
    """Return the shared client-less TrafficManager backend."""
    global _CLIENTLESS_BACKEND
    if _CLIENTLESS_BACKEND is None:
        from .traffic_manager import TrafficManagerBackend  # noqa: PLC0415

        backend: TrafficBackend = TrafficManagerBackend()
        _CLIENTLESS_BACKEND = backend
    return _CLIENTLESS_BACKEND


class BackendDriven(DrivenEntity):
    """Manoeuvres for a vehicle a traffic backend steers.

    Mixed into :class:`~autoware_carla_scenario.entity.ego.EgoVehicle` and
    :class:`~autoware_carla_scenario.entity.vehicle_entity.VehicleEntity`, the
    two entities the run's traffic model drives.  Both already own an ``actor``;
    this adds the backend the manoeuvres are delegated to.

    :meth:`set_traffic_backend` is the whole of the live injection:
    :class:`~autoware_carla_scenario.ScenarioRunner` calls it on the ego and
    :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`
    on each NPC, and nothing inside a run names a traffic model here.

    :meth:`set_client` is the compatibility entry point beside it, and the one
    thing on this class that names a particular traffic model.  It is kept
    because external scenario packages call it and because it is still the whole
    truth for an entity built outside a run -- "a TrafficManager on this client
    drives me" is what such a call always meant.  No code in this package calls
    it on the live path any more; a scenario that has no backend is the only
    caller left (see
    :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`).
    """

    #: Set by :meth:`set_traffic_backend`; ``None`` until then.
    _traffic_backend: Optional[TrafficBackend] = None
    #: Set by :meth:`set_client`; ``None`` until then.  Kept because external
    #: scenario packages and the tests read them.
    _tm_client: Optional["carla.Client"] = None
    _tm_port: int = DEFAULT_TM_PORT
    #: The TrafficManager backend :meth:`set_client` stands for, built there
    #: because that call is the only thing that ever supplies its ingredient.
    _fallback_backend: Optional[TrafficBackend] = None

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
        """Say that a TrafficManager on *client* drives this entity.

        Deprecated sugar, kept for the callers that predate the traffic seam.
        It is exactly::

            entity.set_traffic_backend(
                TrafficManagerBackend(
                    TrafficManagerBackendConfig(port=tm_port), client=client
                )
            )

        except that it fills the *fallback* slot rather than the injected one,
        so a backend the run selected is never displaced by a late call to this
        -- which would put a vehicle back under the TrafficManager without
        saying so.  New code names the backend it means.

        The backend is built here rather than per manoeuvre, because this call
        is the only thing that ever supplies its ingredient.
        """
        from .config import TrafficManagerBackendConfig  # noqa: PLC0415
        from .traffic_manager import TrafficManagerBackend  # noqa: PLC0415

        self._tm_client = client
        self._tm_port = tm_port
        # Through a local of the base type: Codon does not upcast and wrap in
        # an Optional in one step (docs/typecheck.md).
        fallback: TrafficBackend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=tm_port), client=client
        )
        self._fallback_backend = fallback

    # ------------------------------------------------------------------
    # Manoeuvres -- delegated, every one of them
    # ------------------------------------------------------------------

    # Each manoeuvre hands the backend ``self`` through a local typed as the
    # DrivenEntity the backend takes: Codon passes a subclass to a method a
    # subclass overrides only once it is that type (docs/typecheck.md).

    def change_lane(self, world: "carla.World", direction: LaneChangeDirection) -> None:
        """Move one lane in *direction*."""
        entity: DrivenEntity = self
        self._resolve_backend().change_lane(entity, world, direction)

    def lane_change_finished(self, world: "carla.World") -> bool:
        """Whether the manoeuvre has settled."""
        entity: DrivenEntity = self
        return self._resolve_backend().lane_change_finished(entity, world)

    def set_desired_speed(self, world: "carla.World", speed_kmh: float) -> None:
        """Aim for *speed_kmh* from now on."""
        entity: DrivenEntity = self
        self._resolve_backend().set_desired_speed(entity, world, speed_kmh)

    def turn_at_junction(
        self, world: "carla.World", direction: TurnDirection, **kwargs: Any
    ) -> None:
        """Go *direction* at the next junction ahead."""
        if TYPE_CHECKING:
            # Codon calls a method a subclass overrides through a dispatch
            # thunk, which cannot call one that takes ``**kwargs``
            # (docs/typecheck.md): the check resolves the backend and stops
            # there, and checks each backend's turn_at_junction() on its own.
            self._resolve_backend()
        else:
            self._resolve_backend().turn_at_junction(self, world, direction, **kwargs)

    def follow_route(self, world: "carla.World", path: List["carla.Location"]) -> None:
        """Drive along *path*, through every junction on it."""
        entity: DrivenEntity = self
        self._resolve_backend().follow_route(entity, world, path)

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: "Optional[CarlaWorldPose]" = None,
        ground_projection: "Optional[GroundProjectionConfig]" = None,
        waypoints: "Sequence[Lanelet2Pose]" = (),
    ) -> None:
        """Send this vehicle to *goal* through *waypoints*, by the lanelets between.

        The backend driving it picks its own way at every junction when left to
        itself.  So the way is worked out here: the Lanelet2 map's shortest path
        from where the vehicle is through each waypoint in turn to the goal
        (:func:`~autoware_carla_scenario.odd.route_plan.route_poses`), its
        centrelines a point every :data:`ROUTE_POINT_STEP_M` metres, handed to
        the backend as one path (:meth:`follow_route`).  A route that cannot be
        planned is logged, and the vehicle left to the backend.

        What :class:`~autoware_carla_scenario.actions.RoutingAction` calls, on
        the ego and on an NPC vehicle alike.  Routing lives on the entity
        because *how* a destination is delivered is the entity's business: an
        ego with a stack of its own takes a map-frame pose over a bridge (see
        :class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`),
        one an external policy drives takes nothing (see
        :class:`~autoware_carla_scenario.entity.externally_driven.ExternallyDrivenEgo`),
        and both override this.  The action only says where and when.

        Args:
            world: The CARLA world.
            goal: Lanelet2 pose to route to.
            initial_pose: Where the vehicle starts, which the route starts from.
                ``None`` reads its actor.
            ground_projection: Not used: the route is laid on the lanelets'
                centrelines.
            waypoints: Lanelet2 poses the route passes through, in order.
        """
        path = self._route_path(goal, initial_pose, waypoints)
        if path:
            # Through a local of the type that declares it: Codon calls a
            # method a subclass overrides through a dispatch thunk, which does
            # not upcast (docs/typecheck.md).
            driven: BackendDriven = self
            driven.follow_route(world, path)

    def _route_path(
        self,
        goal: "Lanelet2Pose",
        initial_pose: "Optional[CarlaWorldPose]",
        waypoints: "Sequence[Lanelet2Pose]",
    ) -> List["carla.Location"]:
        """The CARLA locations :meth:`route_to` hands the backend, or none."""
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.poses import CarlaWorldPose  # noqa: PLC0415
        from ..coordinate.transform import (  # noqa: PLC0415
            project_onto_lanelet,
            to_carla_location,
            to_lanelet2,
        )
        from ..odd.route_plan import PlannedRoute, RouteError, route_poses  # noqa: PLC0415

        located: List["carla.Location"] = []
        start_pose = initial_pose
        if start_pose is None:
            # hasattr rather than getattr with a default (docs/typecheck.md):
            # the mixin itself declares no actor; the entities do.
            actor: Optional["carla.Actor"] = None
            if hasattr(self, "actor"):
                actor = self.actor
            if actor is None:
                logger.warning(
                    "%s: a route was asked for before the actor exists",
                    type(self).__name__,
                )
                return located
            # A typed local: Codon unwraps the Optional here.
            vehicle: "carla.Actor" = actor
            transform = vehicle.get_transform()
            start_pose = CarlaWorldPose(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
                yaw=transform.rotation.yaw,
            )
        start: CarlaWorldPose = start_pose
        # The nearest lanelet, then where along it: to_lanelet2() measures to
        # the nearest vertex of the centreline, which on a straight lanelet of
        # two vertices is one end or the other.
        here = project_onto_lanelet(start, to_lanelet2(start).lanelet_id)
        mm = MapManager.get_instance()
        try:
            poses = route_poses(
                PlannedRoute(start=here, goal=goal, via=tuple(waypoints)),
                mm.lanelet_map,
                mm.routing_graph,
                ROUTE_POINT_STEP_M,
            )
        except RouteError as error:
            logger.warning("%s: no route to follow: %s", type(self).__name__, error)
            return located
        # The pose the vehicle stands on is left out, so the path starts a
        # step ahead of it.  The TrafficManager takes the points in order and
        # waits for each to come within a few metres of its next waypoint: a
        # point behind the vehicle never does, and with it outstanding the
        # TrafficManager chooses its own way at every junction.
        ahead = poses[1:] if len(poses) > 1 else poses
        for pose in ahead:
            located.append(to_carla_location(pose))
        return located

    def release_from_traffic(self, world: "carla.World") -> None:
        """Stop the backend driving this vehicle; the caller drives it now."""
        entity: DrivenEntity = self
        self._resolve_backend().release(entity, world)

    def resume_traffic(self, world: "carla.World") -> None:
        """Hand this vehicle back to the backend (:meth:`release_from_traffic`'s inverse)."""
        entity: DrivenEntity = self
        self._resolve_backend().resume(entity, world)

    def require_microscopic(self, world: "carla.World") -> bool:
        """Have the backend put this vehicle back in the world if it took it
        out (:class:`~autoware_carla_scenario.traffic.SimulationFidelity`);
        whether it is in the world afterwards."""
        entity: DrivenEntity = self
        return self._resolve_backend().require_microscopic(entity, world)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> TrafficBackend:
        """Return the backend this entity's manoeuvres go to.

        An injected backend wins whichever order it and :meth:`set_client`
        arrive in.  Next is the TrafficManager that call stands for.  An entity
        given neither still resolves -- to a TrafficManager with no client,
        shared and stateless, which answers what needs no CARLA call (whether a
        lane change has settled is arithmetic on this entity's own state) and
        reports the missing client for what does.  That is what such an entity
        did before the backend seam existed, and there is nothing else it could
        honestly mean.

        Each step tests for ``None`` rather than truthiness: a backend is a
        third party's object and may define ``__bool__`` or ``__len__``, and one
        that was injected must receive the intent whatever it reports.
        """
        if self._traffic_backend is not None:
            return self._traffic_backend
        if self._fallback_backend is not None:
            return self._fallback_backend
        return _clientless()
