"""CARLA's TrafficManager as a traffic backend.

This is what drove every run before the backend seam existed, moved rather than
rewritten: the synchronous-mode and seeding block, the autopilot loop, and the
manoeuvres that used to live on the entity mixin are all here, doing exactly
what they did.

What *did* change is where the knowledge sits.  ``tm.set_path`` is TrafficManager
vocabulary -- an Autoware ego would not take a list of waypoints for "turn left",
it would take a goal, and a SUMO-driven vehicle would take neither -- so the
translation from intent to mechanism belongs to the thing that drives, not to the
action that asks or to the entity that is asked.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, ClassVar, Collection, List, Optional, Tuple

from ..constants import (
    LANE_CHANGE_CENTER_TOLERANCE_M,
    LANE_CHANGE_HEADING_TOLERANCE_DEG,
)
from ..coordinate.map_manager import run_carla_map
from ..kinematics.angle import normalize_angle_deg
from .base import (
    DrivenEntity,
    LaneChangeDirection,
    TrafficBackend,
    TrafficContext,
    TurnDirection,
    _entity_name,
)
from .config import TrafficManagerBackendConfig
from .parameters import VEHICLE_BOUNDS, Spread, VehicleParameters
from ..world_reset import episode_of
from ..visibility import (
    DEFAULT_DECEL,
    DEFAULT_REACTION_S,
    sight_speed,
    world_visibility,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "TrafficManagerBackend",
    "compute_turn_route",
    "TURN_SEARCH_DISTANCE_M",
    "TURN_WAYPOINT_STEP_M",
    "TURN_POST_JUNCTION_DISTANCE_M",
]

#: How far ahead to look for the next junction, in metres.
TURN_SEARCH_DISTANCE_M: float = 200.0
#: Spacing between the waypoints a route is traced with, in metres.
TURN_WAYPOINT_STEP_M: float = 2.0
#: How far past the junction exit to keep tracing, in metres.  The exit heading
#: is what the branches are compared by, and it is only stable once the road
#: has straightened out.
TURN_POST_JUNCTION_DISTANCE_M: float = 20.0

#: Heading change, in degrees, that each turn amounts to.  CARLA's yaw is
#: left-handed and clockwise-positive seen from above.
_LEFT_TARGET_DEG: float = -90.0
_RIGHT_TARGET_DEG: float = 90.0
#: The heading change each direction aims for through a junction.
_TARGET_DEG: dict[TurnDirection, float] = {
    TurnDirection.LEFT: _LEFT_TARGET_DEG,
    TurnDirection.RIGHT: _RIGHT_TARGET_DEG,
    TurnDirection.STRAIGHT: 0.0,
}


#: How often the vehicles held to what their drivers see are worked out again,
#: seconds of simulated time: each is held as a percentage under its road's
#: speed limit, which changes along its route.
SIGHT_UPDATE_S: float = 1.0
#: How much a percentage held to sight may differ from the one sent last, and
#: still not be sent again: each is an RPC.
_SIGHT_TOLERANCE_PCT: float = 0.5
#: The TrafficManager's own percentage under the speed limit, which a vehicle
#: without a speed factor of its own drives at (and is given back).
TM_DEFAULT_SPEED_DIFFERENCE_PCT: float = 30.0

#: The blueprint a background vehicle gets when none is asked for: one CARLA
#: 0.9 and 0.10 both ship.
_BACKGROUND_BLUEPRINT = "vehicle.lincoln.mkz*"


class TrafficManagerBackend(TrafficBackend):
    """Traffic driven by CARLA's own TrafficManager.

    The default backend, and the one every existing scenario gets: vehicles the
    scenario spawned are handed to the TrafficManager after warm-up, and the
    manoeuvres a scenario asks for become TrafficManager calls.

    How its vehicles drive is :class:`~autoware_carla_scenario.traffic.parameters.VehicleParameters`
    (those :data:`~autoware_carla_scenario.traffic.parameters.VEHICLE_PARAMETER_BACKENDS`
    gives ``traffic_manager``; it has no reaction time, braking or
    acceleration of a driver to set): each vehicle draws its own value of
    each from the parameter's spread as it is first put on autopilot, and the
    TrafficManager is told it for that vehicle.  Every vehicle it drives gets
    them, the scenario's own as well as its background traffic.

    The TrafficManager has no notion of sight either, so with
    ``follow_visibility: true`` this backend gives it one (:meth:`tick`): each
    vehicle it drives is held to the speed a driver stops from within what
    they see.  Off unless asked for, so a scenario's vehicles drive as they
    always did.

    The client is injected rather than passed to each call, from either of the
    two places that already hold one: :meth:`prepare`, when
    :class:`~autoware_carla_scenario.ScenarioRunner` builds the run's context,
    or the constructor, for an entity that was only ever given a client (see
    :meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.set_client`).
    A backend that was never given one says so rather than failing inside CARLA.
    """

    name: ClassVar[str] = "traffic_manager"

    _config: TrafficManagerBackendConfig
    _client: Optional["carla.Client"]
    _random_seed: Optional[int]
    _closed: bool
    _background: dict[str, "carla.Actor"]
    _background_count: int
    _started: bool
    _released: set[int]
    _skipped: set[int]
    _lane_changed: List[DrivenEntity]
    _visibility: Optional[float]
    _draws: Optional[random.Random]
    _drawn: set[int]
    _autopiloted: dict[int, "carla.Actor"]
    _speed_differences: dict[int, float]
    _held: dict[int, float]
    _sight_due: float

    def __init__(
        self,
        config: Optional[TrafficManagerBackendConfig] = None,
        *,
        client: Optional["carla.Client"] = None,
        vehicles: Optional[VehicleParameters] = None,
    ) -> None:
        """Create the backend.

        Args:
            config: Backend options; defaults are used when omitted.
            client: The CARLA client the TrafficManager is reached through, for
                a caller that already holds one.  :meth:`prepare` supplies it
                otherwise.
            vehicles: How its vehicles drive; the TrafficManager's defaults
                when omitted.  It drives no pedestrians, so has no pedestrian
                parameters.

        Raises:
            TrafficParameterError: If *vehicles* sets a field the
                TrafficManager does not support.
        """
        self._config = config or TrafficManagerBackendConfig()
        self._client = client
        self._random_seed = None
        self._closed = False
        #: Handle -> actor of every background vehicle spawned this run.
        self._background = {}
        self._background_count = 0
        self._started = False
        #: Actors taken back with :meth:`release`, never handed to autopilot.
        self._released = set()
        #: Actors :meth:`start` was told are driven elsewhere (an ego).
        self._skipped = set()
        #: The vehicles this backend left lane change state on (a map with
        #: it), forgotten when the run closes: the queue keeps a finished
        #: run's scenarios, and their entities with them.
        self._lane_changed = []
        #: How far drivers see, m, as last reported; ``None``: nothing set.
        self._visibility = None
        #: Every vehicle put on autopilot here, by actor id.
        self._autopiloted = {}
        #: Actor id -> the percentage under the speed limit its own speed
        #: factor drew; a vehicle without one drives at the TrafficManager's.
        self._speed_differences = {}
        #: Actor id -> the percentage it is held to by what its driver sees.
        self._held = {}
        #: The simulated time the vehicles held to sight are worked out again.
        self._sight_due = 0.0
        #: Each vehicle's parameter draws, reseeded by prepare(); the actors
        #: that have theirs.
        self._draws = None
        self._drawn = set()
        if vehicles is not None:
            self.set_vehicle_parameters(vehicles)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        """Port the TrafficManager is reached on."""
        return self._config.port

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def prepare(self, context: TrafficContext) -> None:
        """Put the TrafficManager in step with the world and seed it.

        Synchronous mode and the seed are set here, before anything is spawned,
        because the seed only decides anything if it is set before the first
        ``set_autopilot`` call -- which is what makes a run reproducible.
        """
        self._closed = False
        if context.client is not None:
            self._client = context.client
        self._random_seed = context.random_seed

        tm = self._require_tm("prepare")
        if tm is None:
            return
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(context.random_seed)
        # Each vehicle's draws, from the run's seed: the same seed, the same
        # drivers, in the order the vehicles are put on autopilot.
        self._draws = None

    def start(
        self, world: "carla.World", *, skip_actor_ids: Collection[int] = ()
    ) -> None:
        """Hand every vehicle that is not driven elsewhere to the TrafficManager."""
        self._started = True
        skip = set(skip_actor_ids)
        self._skipped = skip
        enabled = 0
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in skip or actor.id in self._released:
                continue
            self._autopilot(actor)
            enabled += 1
        if enabled:
            logger.info("Autopilot enabled on %d vehicle(s)", enabled)
        if skip:
            logger.info(
                "Autopilot skipped for %s — external control expected",
                ", ".join(str(actor_id) for actor_id in sorted(skip)),
            )

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Hold the traffic to what its drivers see, with ``follow_visibility``.

        The TrafficManager rides the CARLA tick, and has no notion of sight:
        its vehicles know where every actor is, fog or not
        (:mod:`autoware_carla_scenario.visibility`).  With
        ``follow_visibility: true`` each vehicle it drives is held to the
        speed it stops from within the visibility the run's weather set
        (:func:`~autoware_carla_scenario.visibility.sight_speed`, for SUMO's
        default car's reaction and braking): as a percentage under its road's speed limit, never less than its own
        speed factor's, worked out again every :data:`SIGHT_UPDATE_S` and
        whenever the visibility changes.  A vehicle a scenario
        gave a speed of its own (``set_desired_speed``) keeps that.  Without
        it, the visibility is only logged, with the speed it would hold a car
        to.
        """
        seen = world_visibility(episode_of(world))
        follows = self.vehicle_parameters.follow_visibility is True
        changed = seen != self._visibility
        if changed:
            self._visibility = seen
            self._log_sight(seen, follows)
        # Negative: no hold, so every vehicle held is given its own back (no
        # visibility any more, or follow_visibility turned off during the run).
        limit_kmh = -1.0
        if follows and seen is not None:
            limit_kmh = sight_speed(seen, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        if limit_kmh < 0.0 and len(self._held) == 0:
            return
        if follows and not changed and elapsed < self._sight_due:
            return
        self._sight_due = elapsed + SIGHT_UPDATE_S
        tm = self._require_tm("follow_visibility")
        if tm is None:
            return
        for actor_id, actor in list(self._autopiloted.items()):
            if not actor.is_alive:
                # Destroyed since it was put on autopilot: forgotten.
                del self._autopiloted[actor_id]
                if actor_id in self._held:
                    del self._held[actor_id]
                continue
            if actor_id in self._released:
                continue
            try:
                self._hold_to_sight(tm, actor, limit_kmh)
            except RuntimeError:
                continue  # destroyed under us: forgotten at the next update

    def _hold_to_sight(
        self, tm: "carla.TrafficManager", actor: "carla.Actor", limit_kmh: float
    ) -> None:
        """Hold *actor* to *limit_kmh*, as a percentage under its road's
        speed limit; a negative *limit_kmh* gives it its own back."""
        own = TM_DEFAULT_SPEED_DIFFERENCE_PCT
        if actor.id in self._speed_differences:
            own = self._speed_differences[actor.id]
        vehicle = _vehicle(actor)
        wanted = own
        if limit_kmh >= 0.0:
            road_kmh = float(vehicle.get_speed_limit())
            if road_kmh > 0.0:
                wanted = max(own, 100.0 * (1.0 - limit_kmh / road_kmh))
        if actor.id in self._held:
            if abs(self._held[actor.id] - wanted) < _SIGHT_TOLERANCE_PCT:
                return
        elif wanted == own:
            return  # never held, and not to be
        tm.vehicle_percentage_speed_difference(vehicle, wanted)
        if wanted == own:
            del self._held[actor.id]
        else:
            self._held[actor.id] = wanted

    def _log_sight(self, seen: Optional[float], follows: bool) -> None:
        """Say how far drivers see now, and what that does to the traffic."""
        if seen is None:
            logger.info("TrafficManager: no visibility set any more")
            return
        logger.info(
            "TrafficManager: drivers see %.0f m, so traffic %s at most %.0f km/h "
            "(braking at %.1f m/s² after %.1f s)%s",
            seen,
            "drives" if follows else "that follows it would drive",
            sight_speed(seen, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6,
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            ""
            if follows
            else "; traffic.vehicles.follow_visibility is not on, and its "
            "vehicles drive as before",
        )

    def close(self) -> None:
        """Shut the TrafficManager down so the next run gets a fresh one.

        Resets its ``InMemoryMap`` cache and internal state.  Safe to call twice
        and safe to call after a failed :meth:`prepare`, because teardown runs
        whatever happened during the run.
        """
        # First: nothing below may keep a finished run's map on its vehicles.
        for entity in self._lane_changed:
            entity.forget_lane_change()
        self._lane_changed = []
        for actor in self._background.values():
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self._background = {}
        self._background_count = 0
        self._started = False
        self._released = set()
        self._skipped = set()
        self._visibility = None
        self._autopiloted = {}
        self._speed_differences = {}
        self._held = {}
        self._sight_due = 0.0
        self._drawn = set()
        self._draws = None
        self._restore_parameters()
        if self._closed or self._client is None:
            return
        self._closed = True
        try:
            self._client.get_trafficmanager(self.port).shut_down()
        except Exception:
            logger.warning("TrafficManager shut down failed", exc_info=True)
        else:
            logger.info("TrafficManager shut down")

    def describe(self) -> dict[str, Any]:
        """Return the backend name, its port and the seed it was given."""
        out = super().describe()
        out["port"] = self.port
        out["random_seed"] = self._random_seed
        out["follow_visibility"] = self.vehicle_parameters.follow_visibility is True
        out["visibility_m"] = self._visibility
        return out

    # ------------------------------------------------------------------
    # Background traffic
    # ------------------------------------------------------------------

    def spawn_background(
        self,
        world: "carla.World",
        transform: "carla.Transform",
        *,
        speed_kmh: float,
        blueprint: Optional[str] = None,
    ) -> Optional[str]:
        """Spawn a CARLA vehicle at *transform* and give it to the TrafficManager.

        One spawned before :meth:`start` is put on autopilot there, with every
        other vehicle; one spawned during the run is put on it at once.
        """
        import typesafe_carla.carla as carla  # noqa: PLC0415

        library = world.get_blueprint_library()
        found = library.filter(blueprint or _BACKGROUND_BLUEPRINT)
        if len(found) == 0:  # A BlueprintLibrary has no truth value in Codon.
            logger.warning("No vehicle blueprint %r for background traffic", blueprint)
            return None
        self._background_count += 1
        handle = f"background{self._background_count}"
        chosen = found[0]
        if chosen.has_attribute("role_name"):
            chosen.set_attribute("role_name", handle)
        lifted = carla.Transform(
            carla.Location(
                transform.location.x, transform.location.y, transform.location.z + 0.5
            ),
            transform.rotation,
        )
        actor = world.try_spawn_actor(chosen, lifted)
        if actor is None:
            return None  # The spot is taken; the caller tries elsewhere.
        self._background[handle] = actor
        speed = max(speed_kmh, 0.0) / 3.6
        forward = lifted.get_forward_vector()
        actor.set_target_velocity(
            carla.Vector3D(forward.x * speed, forward.y * speed, 0.0)
        )
        if self._started:
            self._autopilot(actor)
        tm = self._require_tm("spawn_background")
        if tm is not None:
            tm.set_desired_speed(_vehicle(actor), max(speed_kmh, 1.0))
        return handle

    def background_vehicles(
        self, world: "carla.World"
    ) -> dict[str, tuple[float, float]]:
        """Every background vehicle still in the world, at its ``(x, y)``."""
        out: dict[str, tuple[float, float]] = {}
        for handle, actor in list(self._background.items()):
            try:
                location = actor.get_location()
            except RuntimeError:
                self._background.pop(handle)  # Destroyed by something else.
                continue
            out[handle] = (location.x, location.y)
        return out

    def background_actors(self, world: "carla.World") -> dict[str, "carla.Actor"]:
        """Every background vehicle, by its handle, which is its ``role_name``;
        a destroyed one is no agent of the scene, so never chosen."""
        return dict(self._background)

    def remove_background(self, world: "carla.World", handle: str) -> None:
        """Destroy the background vehicle *handle*."""
        if handle not in self._background:
            return
        actor = self._background.pop(handle)
        try:
            actor.destroy()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(
        self, entity: DrivenEntity, world: "carla.World", direction: LaneChangeDirection
    ) -> None:
        """Move *entity* one lane in *direction*.

        Records the lane aimed at on the entity so :meth:`lane_change_finished`
        can tell a completed manoeuvre from a vehicle that merely drove onto the
        next road: lane ids are scoped to a road, so ``(road_id, lane_id)``
        changes without the vehicle having moved sideways at all.

        The bookkeeping lives on the entity rather than here because it is the
        state of *that vehicle's* manoeuvre: a backend is shared by every
        vehicle in the run, and the entity is the one thing there is exactly one
        of per manoeuvre.
        """
        actor = _require_actor(entity, "change_lane")
        if actor is None:
            return
        tm = self._require_tm("change_lane")
        if tm is None:
            return

        carla_map = run_carla_map(world)
        entity._lane_change_target = _adjacent_lane(
            carla_map, actor.get_location(), direction
        )
        entity._lane_change_map = carla_map
        self._lane_changed.append(entity)
        if entity._lane_change_target is None:
            logger.warning(
                "%s: no lane %s to change into",
                _entity_name(entity),
                direction.value,
            )

        tm.force_lane_change(_vehicle(actor), direction.to_carla_bool())
        logger.info(
            "%s: forced a %s lane change", _entity_name(entity), direction.value
        )

    def lane_change_finished(self, entity: DrivenEntity, world: "carla.World") -> bool:
        """Whether *entity* has settled onto the lane it was sent to.

        Three things have to be true, and a lane id change on its own is not
        enough: a vehicle whose centre has just crossed the boundary is still
        diagonal across two lanes, and calling that finished would let a
        reaction fire mid-manoeuvre.

        A manoeuvre the TrafficManager never makes simply never finishes; see
        :meth:`TrafficBackend.lane_change_finished` for why that is the honest
        answer rather than a timeout of its own.
        """
        # hasattr rather than getattr with a default (docs/typecheck.md): an
        # entity that is not a DrivenEntity (a test's stand-in) may lack them.
        if not hasattr(entity, "_lane_change_target") or not hasattr(
            entity, "_lane_change_map"
        ):
            return False
        target = entity._lane_change_target
        carla_map = entity._lane_change_map
        actor = _actor_of(entity)
        if target is None or carla_map is None or actor is None:
            return False
        # Codon does not narrow an Optional (docs/typecheck.md).
        aimed: Tuple[int, int] = target
        lane_map: "carla.Map" = carla_map

        # One RPC per tick: the transform carries both the location the map is
        # queried with and the heading the check needs.
        transform = actor.get_transform()
        found = lane_map.get_waypoint(transform.location, project_to_road=True)
        if found is None:
            return False
        waypoint: "carla.Waypoint" = found
        if _lane_key_of(waypoint) != aimed:
            return False

        # ``get_waypoint`` projects onto the lane centre, so the distance to it
        # is the lateral offset.
        if (
            transform.location.distance(waypoint.transform.location)
            > LANE_CHANGE_CENTER_TOLERANCE_M
        ):
            return False

        return (
            _heading_error_deg(transform.rotation.yaw, waypoint.transform.rotation.yaw)
            <= LANE_CHANGE_HEADING_TOLERANCE_DEG
        )

    def set_desired_speed(
        self, entity: DrivenEntity, world: "carla.World", speed_kmh: float
    ) -> None:
        """Hold *entity* at *speed_kmh*.

        ``set_desired_speed`` is a target, not a jump: the TrafficManager gets
        the vehicle there under its own acceleration limits.  A rate asked for
        by the scenario is therefore a walk of *targets*, stepped by the action
        once per tick from the vehicle's own speed, and this call is one step
        of it.  The value given here stays in force until it is changed, which
        is why the action reissues only when it has a rate to walk.
        """
        actor = _require_actor(entity, "set_desired_speed")
        if actor is None:
            return
        tm = self._require_tm("set_desired_speed")
        if tm is None:
            return

        tm.set_desired_speed(_vehicle(actor), speed_kmh)
        logger.debug(
            "%s: desired speed set to %.1f km/h", _entity_name(entity), speed_kmh
        )

    def turn_at_junction(
        self,
        entity: DrivenEntity,
        world: "carla.World",
        direction: TurnDirection,
        *,
        search_distance: float = TURN_SEARCH_DISTANCE_M,
        waypoint_step: float = TURN_WAYPOINT_STEP_M,
        post_junction_distance: float = TURN_POST_JUNCTION_DISTANCE_M,
        **kwargs: Any,
    ) -> None:
        """Send *entity* *direction* at the next junction ahead.

        The route through the junction is worked out from the map and handed to
        the TrafficManager, which is what steering this vehicle means.  Another
        backend answers the same intent differently -- by giving a SUMO vehicle
        a new route, or an Autoware ego a goal beyond the junction -- which is
        why the intent stops at this boundary.
        """
        actor = _require_actor(entity, "turn_at_junction")
        if actor is None:
            return
        tm = self._require_tm("turn_at_junction")
        if tm is None:
            return

        current = run_carla_map(world).get_waypoint(actor.get_location())
        path = compute_turn_route(
            current,
            direction,
            search_distance=search_distance,
            waypoint_step=waypoint_step,
            post_junction_distance=post_junction_distance,
        )
        if not path:
            logger.warning(
                "%s: no %s turn route found", _entity_name(entity), direction.value
            )
            return

        tm.set_path(_vehicle(actor), path)
        logger.info(
            "%s: set a %s turn route (%d points)",
            _entity_name(entity),
            direction.value,
            len(path),
        )

    def release(self, entity: DrivenEntity, world: "carla.World") -> None:
        """Take *entity*'s vehicle off autopilot, and keep it off at :meth:`start`."""
        actor = _require_actor(entity, "release")
        if actor is None:
            return
        # A typed local: Codon unwraps the Optional here.
        vehicle: "carla.Actor" = actor
        self.release_actor(world, vehicle)

    def release_actor(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Take *actor* off autopilot, and keep it off at :meth:`start`."""
        self._released.add(actor.id)
        if self._started:
            actor.set_autopilot(False, self.port)
            logger.info("%s: autopilot released", _actor_name(actor))

    def resume(self, entity: DrivenEntity, world: "carla.World") -> None:
        """Put *entity*'s vehicle back on autopilot, and on it at :meth:`start`.

        The TrafficManager keeps a vehicle's settings (its desired speed, its
        lane-change rules) across autopilot switches, so it drives on as it
        did before it was released.
        """
        actor = _require_actor(entity, "resume")
        if actor is None:
            return
        # A typed local: Codon unwraps the Optional here.
        vehicle: "carla.Actor" = actor
        self.resume_actor(world, vehicle)

    def resume_actor(self, world: "carla.World", actor: "carla.Actor") -> None:
        """Put *actor* back on autopilot, and on it at :meth:`start`."""
        if actor.id not in self._released:
            return
        if actor.id in self._skipped:
            # Driven elsewhere from the start (an ego's stack): never ours.
            return
        self._released.discard(actor.id)
        if self._started:
            self._autopilot(actor)
            logger.info("%s: autopilot resumed", _actor_name(actor))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _autopilot(self, actor: "carla.Actor") -> None:
        """Put *actor* on autopilot, with its own draw of each parameter.

        Drawn once per vehicle: one taken back (:meth:`release_actor`) and
        resumed keeps the draws the TrafficManager holds for it.
        """
        actor.set_autopilot(True, self.port)
        self._autopiloted[actor.id] = actor
        if actor.id in self._drawn:
            return
        self._drawn.add(actor.id)
        self._draw_for(actor, self.vehicle_parameters)

    def _draw_for(self, actor: "carla.Actor", parameters: VehicleParameters) -> None:
        """Give *actor* its own draw of each field *parameters* sets."""
        if not parameters.set_fields():
            return
        tm = self._require_tm("traffic parameters")
        if tm is None:
            return
        if self._draws is None:
            self._draws = random.Random(self._random_seed)
        rng: random.Random = self._draws
        vehicle = _vehicle(actor)
        speed_factor = parameters.speed_factor
        if speed_factor is not None:
            # The TrafficManager's is a percentage *under* the limit.
            factor = speed_factor.draw(rng, VEHICLE_BOUNDS["speed_factor"])
            difference = (1.0 - factor) * 100.0
            tm.vehicle_percentage_speed_difference(vehicle, difference)
            # Its own, which a hold to sight never drives it faster than;
            # held, it is held again from this at the next update.
            self._speed_differences[actor.id] = difference
            if actor.id in self._held:
                del self._held[actor.id]
        min_gap = parameters.min_gap_m
        if min_gap is not None:
            tm.distance_to_leading_vehicle(
                vehicle, min_gap.draw(rng, VEHICLE_BOUNDS["min_gap_m"])
            )
        if parameters.lane_changes is not None:
            tm.auto_lane_change(vehicle, parameters.lane_changes)
        # The TrafficManager's are percentages, the parameters' probabilities.
        ignore_lights = parameters.ignore_lights
        if ignore_lights is not None:
            tm.ignore_lights_percentage(
                vehicle, _percent(ignore_lights, rng, "ignore_lights")
            )
        ignore_signs = parameters.ignore_signs
        if ignore_signs is not None:
            tm.ignore_signs_percentage(
                vehicle, _percent(ignore_signs, rng, "ignore_signs")
            )
        ignore_vehicles = parameters.ignore_vehicles
        if ignore_vehicles is not None:
            tm.ignore_vehicles_percentage(
                vehicle, _percent(ignore_vehicles, rng, "ignore_vehicles")
            )
        ignore_walkers = parameters.ignore_walkers
        if ignore_walkers is not None:
            tm.ignore_walkers_percentage(
                vehicle, _percent(ignore_walkers, rng, "ignore_walkers")
            )
        random_lane_changes = parameters.random_lane_changes
        if random_lane_changes is not None:
            chance = _percent(random_lane_changes, rng, "random_lane_changes")
            tm.random_left_lanechange_percentage(vehicle, chance)
            tm.random_right_lanechange_percentage(vehicle, chance)

    # ------------------------------------------------------------------
    # Parameters changed during a run
    # ------------------------------------------------------------------

    def supported_vehicle_overrides(self) -> set[str]:
        """Every vehicle parameter it supports: each is set per vehicle."""
        return self.supported_vehicle_parameters()

    def override_vehicle_parameters(
        self, world: "carla.World", overrides: VehicleParameters
    ) -> None:
        """Give every vehicle it drives a new draw of the fields *overrides*
        sets; the vehicles put on autopilot later draw from the new
        parameters too."""
        self._remember_parameters()
        self._vehicle_parameters = self.vehicle_parameters.overridden_by(overrides)
        for driven in world.get_actors().filter("vehicle.*"):
            # The ones that drew already: a vehicle not yet on autopilot draws
            # from the new parameters when it is put on it.
            if driven.id in self._drawn:
                self._draw_for(driven, overrides)

    def override_entity_parameters(
        self, world: "carla.World", overrides: VehicleParameters, entity: DrivenEntity
    ) -> None:
        """Give *entity*'s vehicle alone a new draw of the fields *overrides* sets."""
        actor = _require_actor(entity, "override_entity_parameters")
        if actor is not None:
            # A typed local: Codon unwraps the Optional here.
            vehicle: "carla.Actor" = actor
            self._draw_for(vehicle, overrides)

    def _require_tm(self, what: str) -> Optional["carla.TrafficManager"]:
        """Return the TrafficManager handle, or ``None`` with a warning."""
        if self._client is None:
            logger.warning(
                "%s: %s needs a CARLA client; none was injected. "
                "ScenarioRunner gives one to the backend it builds, and "
                "set_client() to an entity built outside a run.",
                type(self).__name__,
                what,
            )
            return None
        return self._client.get_trafficmanager(self.port)


def _percent(spread: Spread, rng: random.Random, name: str) -> float:
    """One vehicle's draw of the probability *name*, as a percentage."""
    return spread.draw(rng, VEHICLE_BOUNDS[name]) * 100.0


def _actor_of(entity: DrivenEntity) -> Optional["carla.Actor"]:
    """Return *entity*'s actor, or ``None`` when it has none (yet).

    ``hasattr`` rather than ``getattr`` with a default: Codon decides it when it
    compiles (docs/typecheck.md), and the mixin itself declares no ``actor``.
    """
    actor: Optional["carla.Actor"] = None
    if hasattr(entity, "actor"):
        actor = entity.actor
    return actor


def _vehicle(actor: "carla.Actor") -> "carla.Vehicle":
    """Return *actor* as the vehicle a TrafficManager call takes.

    At run time that is *actor* itself, which typesafe_carla's Python API takes
    as it is.  Its Codon library takes a ``carla.Vehicle``, which the library
    check gets with the checked conversion: Codon reads ``TYPE_CHECKING`` as a
    compile-time true (docs/typecheck.md), and Python as false.
    """
    if TYPE_CHECKING:
        return actor.as_vehicle()
    else:
        return actor


def _actor_name(actor: "carla.Actor") -> str:
    """*actor*'s role name, for the logs; its id when it has none."""
    attributes = actor.attributes
    if "role_name" in attributes:
        return str(attributes["role_name"])
    return f"actor {actor.id}"


def _require_actor(entity: DrivenEntity, what: str) -> Optional["carla.Actor"]:
    """Return *entity*'s actor, or ``None`` with a warning."""
    actor = _actor_of(entity)
    if actor is None:
        logger.warning(
            "%s: %s asked for before the actor exists", _entity_name(entity), what
        )
    return actor


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _adjacent_lane(
    carla_map: "carla.Map",
    location: "carla.Location",
    direction: LaneChangeDirection,
) -> Optional[Tuple[int, int]]:
    """Return the ``(road_id, lane_id)`` beside *location* in *direction*.

    ``None`` when there is no lane that way, which is the honest answer to a
    lane change that cannot happen.
    """
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    if waypoint is None:
        return None
    neighbour: Optional["carla.Waypoint"] = None
    if direction is LaneChangeDirection.RIGHT:
        neighbour = waypoint.get_right_lane()
    else:
        neighbour = waypoint.get_left_lane()
    if neighbour is None:
        return None
    # Codon does not narrow an Optional (docs/typecheck.md).
    lane: "carla.Waypoint" = neighbour
    return _lane_key_of(lane)


def _lane_key_of(waypoint: "carla.Waypoint") -> Tuple[int, int]:
    """Return the ``(road_id, lane_id)`` a waypoint sits on."""
    return (waypoint.road_id, waypoint.lane_id)


def _heading_error_deg(yaw: float, reference_yaw: float) -> float:
    """Return the absolute heading difference in degrees, wrapped to 180."""
    return abs(normalize_angle_deg(yaw - reference_yaw))


# ---------------------------------------------------------------------------
# Turn route geometry
# ---------------------------------------------------------------------------


def compute_turn_route(
    current_wp: "carla.Waypoint",
    direction: TurnDirection,
    *,
    search_distance: float = TURN_SEARCH_DISTANCE_M,
    waypoint_step: float = TURN_WAYPOINT_STEP_M,
    post_junction_distance: float = TURN_POST_JUNCTION_DISTANCE_M,
) -> List["carla.Location"]:
    """Build a waypoint path through the next junction in the desired direction."""
    pre_junction_wp, junction_entries = _walk_to_junction(
        current_wp, search_distance, waypoint_step
    )
    if pre_junction_wp is None or not junction_entries:
        return []

    branches: List[List["carla.Waypoint"]] = []
    for entry_wp in junction_entries:
        branch = _trace_through_junction(
            entry_wp, waypoint_step, post_junction_distance
        )
        if branch:
            branches.append(branch)

    if not branches:
        return []

    best = _pick_branch(pre_junction_wp, branches, direction)
    if best is None:
        return []

    return [wp.transform.location for wp in best]


def _walk_to_junction(
    start_wp: "carla.Waypoint", search_distance: float, waypoint_step: float
) -> tuple[Optional["carla.Waypoint"], List["carla.Waypoint"]]:
    """Walk forward from *start_wp* until the next OpenDRIVE junction.

    If *start_wp* is already inside a junction it is first skipped so that
    the *next* junction ahead is found.

    Returns:
        ``(pre_junction_waypoint, junction_entry_waypoints)`` where the
        entries are the first waypoints on connecting roads inside the
        junction, or ``(None, [])`` when no junction is found within
        *search_distance*.
    """
    wp = start_wp
    distance = 0.0

    # Skip past current junction (if any)
    while wp.is_junction and distance < search_distance:
        nxt = wp.next(waypoint_step)
        if not nxt:
            return None, []
        wp = nxt[0]
        distance += waypoint_step

    # Walk forward to the next junction boundary
    while distance < search_distance:
        next_wps = wp.next(waypoint_step)
        if not next_wps:
            return None, []

        entries = [w for w in next_wps if w.is_junction]
        if entries:
            # An Optional, as the other returns' None is (docs/typecheck.md).
            pre_junction: Optional["carla.Waypoint"] = wp
            return pre_junction, entries

        wp = next_wps[0]
        distance += waypoint_step

    return None, []


def _trace_through_junction(
    entry_wp: "carla.Waypoint", waypoint_step: float, post_junction_distance: float
) -> List["carla.Waypoint"]:
    """Follow waypoints from *entry_wp* through the junction and a bit beyond.

    The extra post-junction distance provides a stable exit heading for
    direction comparison.
    """
    path: List["carla.Waypoint"] = [entry_wp]
    wp = entry_wp

    # Walk through junction connecting road
    safety_limit = 500
    while wp.is_junction and safety_limit > 0:
        nxt = wp.next(waypoint_step)
        if not nxt:
            break
        wp = nxt[0]
        path.append(wp)
        safety_limit -= 1

    # Continue past junction exit for a reliable heading measurement
    post = 0.0
    while post < post_junction_distance:
        nxt = wp.next(waypoint_step)
        if not nxt:
            break
        wp = nxt[0]
        path.append(wp)
        post += waypoint_step

    return path


def _pick_branch(
    pre_junction_wp: "carla.Waypoint",
    branches: List[List["carla.Waypoint"]],
    direction: TurnDirection,
) -> Optional[List["carla.Waypoint"]]:
    """Select the branch whose exit heading change is closest to the target.

    CARLA yaw convention (left-hand, clockwise-positive when viewed from
    above):

    - Left turn  ≈ −90° heading change
    - Right turn ≈ +90° heading change
    - Straight on ≈ 0°
    """
    entry_yaw = pre_junction_wp.transform.rotation.yaw
    target = _TARGET_DEG[direction]

    best: Optional[List["carla.Waypoint"]] = None
    best_score = float("inf")

    for branch in branches:
        if not branch:
            continue
        exit_yaw = branch[-1].transform.rotation.yaw
        diff = normalize_angle_deg(exit_yaw - entry_yaw)
        score = abs(diff - target)
        logger.debug(
            "turn route: branch exit_yaw=%.1f, diff=%.1f, score=%.1f",
            exit_yaw,
            diff,
            score,
        )
        if score < best_score:
            best_score = score
            best = branch

    if best is not None:
        exit_yaw = best[-1].transform.rotation.yaw
        diff = normalize_angle_deg(exit_yaw - entry_yaw)
        logger.info(
            "turn route: selected branch with heading change %.1f deg "
            "(target: %.1f deg %s)",
            diff,
            target,
            direction.value,
        )

    return best
