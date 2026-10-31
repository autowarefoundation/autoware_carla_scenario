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
import math
import random
from typing import TYPE_CHECKING, Any, ClassVar, Collection, List, Optional, Tuple

from ..constants import (
    FIXED_DELTA_SECONDS,
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
from .final_approach import APPROACH_SPEED_MPS, FinalApproach, lane_changes
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

    from ..coordinate.poses import Lanelet2Pose

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTE_POINT_STEP_M",
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
#: Metres between the points of a route :meth:`TrafficManagerBackend.follow_route`
#: sets as a vehicle's path.
ROUTE_POINT_STEP_M: float = 5.0
#: An actor this far down (m) is parked under the map, out of the scene.
_HIDDEN_BELOW_M: float = 100.0
#: How near (m) ahead of a junction a vehicle may start for the
#: TrafficManager to take its route through it (from its route's start, which
#: may lie a few metres short of where it appears).  It places a vehicle on its
#: map's waypoint nearest it, and just past a junction's entry the nearest is
#: on one of the junction's ways through, all starting there: whichever it is,
#: the vehicle keeps to it, whatever its path (a car meant to turn left went
#: right and off the map).  Such a vehicle is driven out of the junction
#: along its route first (:class:`FinalApproach`, not stopping), then handed
#: to the TrafficManager.
_JUNCTION_MOUTH_M: float = 4.0

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

# What the TrafficManager destroys a vehicle it drives for, and how a
# scenario's own vehicles are kept from it: "Vehicles the TrafficManager must
# not destroy" in docs/traffic_backends.md.  The TrafficManager's figures are
# LibCarla's ``trafficmanager/Constants.h``.

#: Seconds standing before the TrafficManager destroys a vehicle
#: (``BLOCKED_TIME_THRESHOLD``; twice that at a red light).
_TM_BLOCKED_TIME_THRESHOLD_S: float = 90.0
#: How long before that a standing kept vehicle is re-registered, s.
_TM_BLOCKED_MARGIN_S: float = 30.0
#: Below this the TrafficManager counts a vehicle as standing, m/s
#: (``STOPPED_VELOCITY_THRESHOLD``).
_TM_STOPPED_SPEED_MPS: float = 0.8
#: The least it plans a vehicle's path ahead, m (``MINIMUM_HORIZON_LENGTH``).
_TM_MIN_HORIZON_M: float = 15.0
#: The most driving it plans ahead, s (``HIGH_SPEED_HORIZON_RATE``).
_TM_HORIZON_S: float = 4.0
#: How often each kept vehicle is looked at, s.
_CHECK_PERIOD_S: float = 0.5
#: How many slices of a period the vehicles' first looks are spread over.
_CHECK_STAGGER: int = 8
#: Metres searched beyond the TrafficManager's horizon, for its rounding.
_DEAD_END_MARGIN_M: float = 10.0
#: Step the road ahead is walked in, m: a lane end is found within one step.
_DEAD_END_STEP_M: float = 5.0
#: Waypoints a search walks at the most; a road that branches more than that
#: is taken to go on.
_DEAD_END_MAX_WAYPOINTS: int = 256
#: The road a lane change's target lane is followed along, beyond twice the
#: straight-line distance the vehicle has come from where it was asked to
#: change lane, m: the vehicle has driven at least that far along the lane,
#: and on a bend more (:meth:`TrafficManagerBackend.lane_change_finished`).
_LANE_CHANGE_AHEAD_MARGIN_M: float = 50.0
#: Step the target lane is followed in, m: shorter than a road in a junction.
_LANE_CHANGE_AHEAD_STEP_M: float = 2.0
#: Waypoints that walk visits at the most.
_LANE_CHANGE_AHEAD_MAX_WAYPOINTS: int = 4096


#: How near (m) a point where its path crosses to the lane beside a vehicle is
#: told to change lanes.
_LANE_CHANGE_AHEAD_M = 8.0


def _lane_change_marks(
    path: List["carla.Location"],
) -> List[tuple["carla.Location", bool]]:
    """Where *path* crosses to the lane beside
    (:func:`~.final_approach.lane_changes`), and whether to the right
    (CARLA's world is left-handed: y points south)."""
    xs = [float(p.x) for p in path]
    ys = [float(p.y) for p in path]
    return [(path[i], across > 0.0) for i, across in lane_changes(xs, ys)]


class _Kept:
    """One vehicle the backend keeps from the TrafficManager's removal
    (:meth:`TrafficManagerBackend._keep_scenario_vehicles`)."""

    actor: "carla.Actor"
    moved_at: float
    check_due: float
    path: List["carla.Location"]
    near_route_end: bool
    checked_x: float
    checked_y: float
    checked_lookahead: float
    held: bool
    refreshed: bool
    approach: Optional[FinalApproach]
    leading_out: bool
    lane_changes: List[tuple["carla.Location", bool]]

    def __init__(self, actor: "carla.Actor") -> None:
        self.actor = actor
        #: When it was last seen moving, s of the run.
        self.moved_at = 0.0
        #: When it is next looked at, s of the run; negative: not yet.
        self.check_due = -1.0
        #: The rest of the path it was last given, to give again.
        self.path = []
        #: Whether its path's end has come near: its road is searched for a
        #: dead end only then (or without a path).
        self.near_route_end = False
        #: Where the last walk found the road going on, and how far it looked;
        #: a negative lookahead: no walk to go by.
        self.checked_x = 0.0
        self.checked_y = 0.0
        self.checked_lookahead = -1.0
        #: Taken off autopilot before a dead end, braking where it is.
        self.held = False
        #: Re-registered at least once (logged at info once).
        self.refreshed = False
        #: Driving the last of its path itself, off autopilot, before a dead
        #: end the TrafficManager would plan into (:class:`FinalApproach`).
        self.approach = None
        #: The approach drives it out of the junction it started in, to be
        #: handed back to the TrafficManager at its end.
        self.leading_out = False
        #: Where its path crosses to the lane beside, and whether to the
        #: right, in order (:func:`_lane_change_marks`): it is told to change
        #: lanes there, as its own lane changes are off.
        self.lane_changes = []


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

    Only background vehicles may be destroyed by the TrafficManager (standing
    90 s, a dead end); see "Vehicles the TrafficManager must not destroy" in
    ``docs/traffic_backends.md``.
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
    _kept: dict[int, _Kept]
    _desired_mps: dict[int, float]
    _given: dict[int, List["carla.Location"]]

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
        #: Every vehicle on autopilot but a background one, by actor id.
        self._kept = {}
        #: The speed (m/s) each vehicle was last asked to hold, by actor id.
        self._desired_mps = {}
        #: The path each vehicle was last given, by actor id, kept or not yet.
        self._given = {}
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
        background = {actor.id for actor in self._background.values()}
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in skip or actor.id in self._released:
                continue
            self._autopilot(actor, keep=actor.id not in background)
            enabled += 1
        if enabled:
            logger.info("Autopilot enabled on %d vehicle(s)", enabled)
        if skip:
            logger.info(
                "Autopilot skipped for %s — external control expected",
                ", ".join(str(actor_id) for actor_id in sorted(skip)),
            )

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Keep the scenario's vehicles from the TrafficManager's removal
        (:meth:`_keep_scenario_vehicles`), and hold the traffic to what its
        drivers see, with ``follow_visibility``.

        The TrafficManager has no notion of sight:
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
        self._keep_scenario_vehicles(world, elapsed)
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
        self._kept = {}
        self._desired_mps = {}
        self._given = {}
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
            self._autopilot(actor, keep=False)
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
        changes without the vehicle having moved sideways at all.  Its
        waypoint is kept too, so the lane can be followed onto the roads after
        it should the vehicle settle there.

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
        here = carla_map.get_waypoint(actor.get_location(), project_to_road=True)
        neighbour: Optional["carla.Waypoint"] = None
        if here is not None:
            # Codon does not narrow an Optional (docs/typecheck.md).
            origin: "carla.Waypoint" = here
            neighbour = _adjacent_lane(origin, direction)
        entity._lane_change_target = None
        entity._lane_change_waypoint = neighbour
        entity._lane_change_ahead = None
        entity._lane_change_horizon_m = 0.0
        entity._lane_change_origin = here
        entity._lane_change_origin_ahead = None
        entity._lane_change_crossed = False
        if neighbour is not None:
            # Codon does not narrow an Optional (docs/typecheck.md).
            lane: "carla.Waypoint" = neighbour
            entity._lane_change_target = _lane_key_of(lane)
        entity._lane_change_map = carla_map
        self._lane_changed.append(entity)
        if neighbour is None:
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

        The target lane is the one aimed at or where its successors lead: a
        lane change often takes longer than a road is long, and lane ids are
        scoped to a road, so the vehicle settles on the next road's lane.  A
        lane the original lane's successors lead to as well (the two merge,
        or meet again past a junction) counts only once the vehicle has been
        seen on one only the target leads to: a vehicle that never left its
        lane reaches it too.

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

        # The side first, every tick: the vehicle crosses onto the target
        # lane before it is centred, and that is where it is seen to cross.
        key = _lane_key_of(waypoint)
        if key == aimed:
            entity._lane_change_crossed = True
        elif not _on_target_side(entity, key, transform.location):
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
        # Driven out of its junction off autopilot: at this speed too.
        self._desired_mps[int(actor.id)] = speed_kmh / 3.6
        found = self._kept.get(int(actor.id))
        if found is not None:
            kept: _Kept = found
            approach = kept.approach
            if kept.leading_out and approach is not None:
                leading: FinalApproach = approach
                leading.set_speed(speed_kmh / 3.6)

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
        self._remember_path(actor, path)
        self._drive_on(actor)
        logger.info(
            "%s: set a %s turn route (%d points)",
            _entity_name(entity),
            direction.value,
            len(path),
        )

    def follow_route(
        self, entity: DrivenEntity, world: "carla.World", stops: List["Lanelet2Pose"]
    ) -> None:
        """Set the route through *stops* as *entity*'s TrafficManager path: a point every
        :data:`ROUTE_POINT_STEP_M` metres, from the one after where it stands,
        junctions left out (:func:`_clear_of_junctions`); see
        ``docs/traffic_backends.md``."""
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.transform import to_carla_location  # noqa: PLC0415
        from ..odd.route_plan import PlannedRoute, RouteError, route_poses  # noqa: PLC0415

        actor = _require_actor(entity, "follow_route")
        if actor is None:
            return
        tm = self._require_tm("follow_route")
        if tm is None:
            return
        mm = MapManager.get_instance()
        route = PlannedRoute(start=stops[0], goal=stops[-1], via=tuple(stops[1:-1]))
        try:
            poses = route_poses(
                route, mm.lanelet_map, mm.routing_graph, ROUTE_POINT_STEP_M
            )
        except RouteError as error:
            logger.warning("%s: no route to follow: %s", _entity_name(entity), error)
            return
        # Not the point it stands on: the TrafficManager waits for each point
        # to come within a few metres, and one behind the vehicle never does.
        first = to_carla_location(poses[0])
        ahead = [to_carla_location(pose) for pose in poses[1:] or poses]
        carla_map = run_carla_map(world)
        path = _clear_of_junctions(carla_map, ahead)
        if not path or _in_junction(carla_map, ahead[-1]):
            # A route that ends in a junction (or is all junction) has no
            # point on the road out to say which way: carry it on along the
            # lanes after its goal until it is clear of the junction.
            beyond = _beyond_goal(stops[-1], ROUTE_POINT_STEP_M)
            path = _clear_of_junctions(
                carla_map, ahead + [to_carla_location(pose) for pose in beyond]
            )
        if not path:
            # Still nothing clear of a junction: the points themselves, which
            # the TrafficManager follows less surely than none at all.
            logger.warning(
                "%s: its route runs through a junction only and leads nowhere "
                "past it; given as it is (%d points)",
                _entity_name(entity),
                len(ahead),
            )
            path = ahead
        # The route says where to change lane; the TrafficManager's own lane
        # changes would take the vehicle off it.
        tm.auto_lane_change(_vehicle(actor), False)
        tm.set_path(_vehicle(actor), path)
        self._remember_path(actor, path)
        self._drive_on(actor)
        logger.info("%s: set a route of %d points", _entity_name(entity), len(path))
        if _at_junction_mouth(carla_map, first) and path and actor.id in self._kept:
            self._lead_out(actor, first, ahead, path)

    def _lead_out(
        self,
        actor: "carla.Actor",
        start: "carla.Location",
        route: List["carla.Location"],
        path: List["carla.Location"],
    ) -> None:
        """Drive *actor*, starting at *start* in a junction's mouth, along
        *route* out of the junction to its *path*'s first point past it, off
        autopilot; :meth:`_keep_scenario_vehicles` hands it to the
        TrafficManager there (:data:`_JUNCTION_MOUTH_M`)."""
        # To the first point of its path past the junction: the route's points
        # up to its first one in a junction lie on the road it comes from.
        carla_map = run_carla_map(actor.get_world())
        out: List["carla.Location"] = []
        through = False
        for point in route:
            out.append(point)
            if not through:
                through = _in_junction(carla_map, point)
            elif any(point.distance(kept_point) < 0.1 for kept_point in path):
                break
        # From where it stands, if it is in the world already (brought in by
        # a spawn, which may have put it on along its lane): the route's
        # points behind it are passed, and a path starting behind it would
        # steer it back (vehicle_129, wedged on the kerb).
        standing = actor.get_location()
        if standing.z > -_HIDDEN_BELOW_M:
            start = standing
            out = _from_here(out, standing)
        kept = self._kept[actor.id]
        actor.set_autopilot(False, self.port)
        kept.held = True
        approach = FinalApproach(
            actor,
            start,
            out,
            stop=False,
            speed=self._desired_mps.get(int(actor.id), APPROACH_SPEED_MPS),
        )
        kept.approach = approach
        kept.leading_out = True
        logger.info(
            "%s: starts in a junction's mouth, where the TrafficManager would "
            "take any way through it; off autopilot, it drives the %.0f m of its "
            "route out of the junction itself, then the TrafficManager takes it on",
            _actor_name(actor),
            approach.length,
        )

    def finish_at(
        self,
        entity: DrivenEntity,
        world: "carla.World",
        points: List["carla.Location"],
    ) -> bool:
        """Take *entity*'s vehicle off autopilot and drive it along *points* to
        the last and stop it there (:class:`FinalApproach`): the TrafficManager
        keeps a vehicle to the middle of its lane, and one that stopped beside
        a cyclist in the recording would stop in the cyclist's way."""
        actor = _require_actor(entity, "finish_at")
        if actor is None or len(points) == 0:
            return False
        kept = self._kept.get(int(actor.id))
        if kept is None:
            return False
        actor.set_autopilot(False, self.port)
        kept.held = True
        kept.leading_out = False
        approach = FinalApproach(actor, actor.get_location(), list(points), stop=True)
        kept.approach = approach
        logger.info(
            "%s: off autopilot, it drives the last %.0f m of its route to where "
            "it is to stop, itself, and stops there",
            _actor_name(actor),
            approach.length,
        )
        return True

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
        # The scenario drives it now: neither kept nor held any more.  Not
        # pop(key, default): Codon has no such overload.
        if actor.id in self._kept:
            del self._kept[actor.id]
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
            background = any(b.id == actor.id for b in self._background.values())
            self._autopilot(actor, keep=not background)
            logger.info("%s: autopilot resumed", _actor_name(actor))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _autopilot(self, actor: "carla.Actor", keep: bool = True) -> None:
        """Put *actor* on autopilot, with its own draw of each parameter, and
        keep it from the TrafficManager's removal unless *keep* is false (a
        background vehicle).

        Drawn once per vehicle: one taken back (:meth:`release_actor`) and
        resumed keeps the draws the TrafficManager holds for it.
        """
        actor.set_autopilot(True, self.port)
        self._autopiloted[actor.id] = actor
        if keep and actor.id not in self._kept:
            kept = _Kept(actor)
            if actor.id in self._given:
                # A path given before it was kept (the ego's, at its init).
                kept.path = list(self._given[actor.id])
            self._kept[actor.id] = kept
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
    # Vehicles the TrafficManager must not destroy
    # ------------------------------------------------------------------

    def _keep_scenario_vehicles(self, world: "carla.World", elapsed: float) -> None:
        """Keep every vehicle on autopilot but a background one from the
        TrafficManager's removal, each looked at every :data:`_CHECK_PERIOD_S`
        (``docs/traffic_backends.md``)."""
        for actor_id, kept in list(self._kept.items()):
            approach = kept.approach
            if approach is not None:
                driving: FinalApproach = approach
                try:
                    driving.step(_tick_seconds(world), elapsed)
                except RuntimeError as error:
                    if not kept.actor.is_alive:
                        del self._kept[actor_id]  # Destroyed under us.
                        continue
                    # Alive: not left standing off autopilot for the run.
                    logger.warning(
                        "%s: driving its path itself failed (%s); the "
                        "TrafficManager takes it on",
                        _actor_name(kept.actor),
                        error,
                    )
                    self._hand_on(kept)
                    continue
                if kept.leading_out and driving.done:
                    self._hand_on(kept)
                continue
            if not kept.held and len(kept.lane_changes) > 0:
                self._change_lanes_on_path(kept)
            if kept.held or elapsed < kept.check_due:
                continue
            actor = kept.actor
            if not actor.is_alive:
                del self._kept[actor_id]  # Destroyed by something else.
                continue
            if kept.check_due < 0.0:
                # First looked at: staggered, so the walks spread over a period.
                kept.moved_at = elapsed
                kept.check_due = elapsed + (actor_id % _CHECK_STAGGER) * (
                    _CHECK_PERIOD_S / _CHECK_STAGGER
                )
                continue
            kept.check_due = elapsed + _CHECK_PERIOD_S
            try:
                velocity = actor.get_velocity()
                # The TrafficManager's own measure: the 3-D speed, squared.
                speed_sq = (
                    velocity.x * velocity.x
                    + velocity.y * velocity.y
                    + velocity.z * velocity.z
                )
                self._keep_standing(kept, speed_sq, elapsed)
                self._keep_off_dead_ends(world, kept, math.sqrt(speed_sq))
            except RuntimeError:
                del self._kept[actor_id]  # Destroyed under us.

    def _keep_standing(self, kept: _Kept, speed_sq: float, elapsed: float) -> None:
        """Take a vehicle that has stood :data:`_TM_BLOCKED_TIME_THRESHOLD_S`
        less :data:`_TM_BLOCKED_MARGIN_S` off autopilot and straight back on,
        which restarts the TrafficManager's count."""
        if speed_sq > _TM_STOPPED_SPEED_MPS * _TM_STOPPED_SPEED_MPS:
            kept.moved_at = elapsed
            return
        stood = elapsed - kept.moved_at
        if stood < _TM_BLOCKED_TIME_THRESHOLD_S - _TM_BLOCKED_MARGIN_S:
            return
        actor = kept.actor
        actor.set_autopilot(False, self.port)
        actor.set_autopilot(True, self.port)
        kept.moved_at = elapsed
        self._give_path_again(kept)
        if kept.refreshed:
            logger.debug("%s: re-registered after %.0f s", _actor_name(actor), stood)
            return
        kept.refreshed = True
        logger.info(
            "%s: stood %.0f s; taken off autopilot and put back on so the "
            "TrafficManager, which destroys a vehicle standing %.0f s, keeps it "
            "(again each time it stands that long)",
            _actor_name(actor),
            stood,
            _TM_BLOCKED_TIME_THRESHOLD_S,
        )

    def _give_path_again(self, kept: _Kept) -> None:
        """Give a re-registered vehicle the rest of its path again.

        The TrafficManager moves a path's points into the vehicle's planned
        waypoints as it goes and drops them from the path it holds
        (``LocalizationStage::ImportPath``: ``UpdateUploadPath``, and
        ``RemoveUploadPath`` once all are in); unregistering drops the planned
        waypoints (``ALSM::RemoveActor``: ``buffer_map.erase``).  So the points
        it had planned would be lost.
        """
        if not kept.path:
            return
        kept.path = _ahead_of(kept.path, kept.actor.get_location())
        if not kept.path:
            return
        tm = self._require_tm("re-registering")
        if tm is not None:
            tm.set_path(_vehicle(kept.actor), kept.path)

    def _hand_on(self, kept: _Kept) -> None:
        """Give a vehicle driven out of its junction back to the
        TrafficManager, with the rest of its path."""
        kept.approach = None
        kept.leading_out = False
        kept.held = False
        kept.check_due = -1.0
        self._autopilot(kept.actor)
        self._give_path_again(kept)
        logger.info(
            "%s: out of its junction, back on autopilot along its path",
            _actor_name(kept.actor),
        )

    def _remember_path(
        self, actor: "carla.Actor", path: List["carla.Location"]
    ) -> None:
        """Remember the path a vehicle was just given: for the keeping, now
        if it is kept, else from when it is put on autopilot."""
        self._given[actor.id] = list(path)
        if actor.id not in self._kept:
            return
        kept = self._kept[actor.id]
        kept.path = list(path)
        kept.near_route_end = False
        kept.checked_lookahead = -1.0
        kept.approach = None
        kept.lane_changes = _lane_change_marks(path)

    def _drive_on(self, actor: "carla.Actor") -> None:
        """Put a vehicle held at a dead end back on autopilot: it was given a
        new route, the way on it was waiting for."""
        if actor.id not in self._kept or not self._kept[actor.id].held:
            return
        kept = self._kept[actor.id]
        kept.held = False
        kept.check_due = -1.0
        self._autopilot(actor)
        logger.info(
            "%s: given a new route, back on autopilot from where it was held",
            _actor_name(actor),
        )

    def _change_lanes_on_path(self, kept: _Kept) -> None:
        """Tell a vehicle to change lanes where its path crosses to the lane
        beside, once it is within :data:`_LANE_CHANGE_AHEAD_M` of that point
        (or past it): the TrafficManager keeps to its lane on a path that
        jumps across, and the vehicle would cross only at the path's end."""
        tm = self._require_tm("force_lane_change")
        if tm is None:
            return
        try:
            location = kept.actor.get_location()
        except RuntimeError:
            return
        mark, to_the_right = kept.lane_changes[0]
        if location.distance(mark) > _LANE_CHANGE_AHEAD_M:
            return
        kept.lane_changes = kept.lane_changes[1:]
        # CARLA's force_lane_change: True to the left.
        tm.force_lane_change(_vehicle(kept.actor), not to_the_right)
        logger.info(
            "%s: changes lanes to the %s where its path crosses",
            _actor_name(kept.actor),
            "right" if to_the_right else "left",
        )

    def _keep_off_dead_ends(
        self, world: "carla.World", kept: _Kept, speed: float
    ) -> None:
        """Take a vehicle off autopilot, braking, before the TrafficManager's
        plan for it reaches a lane with no successor; on a path it was given,
        only once the path's end is near."""
        # Before any annotation names it: Codon makes it a local (docs/typecheck.md).
        import typesafe_carla.carla as carla  # noqa: PLC0415

        lookahead = (
            max(_TM_MIN_HORIZON_M, speed * _TM_HORIZON_S)
            + speed * _CHECK_PERIOD_S
            + _DEAD_END_MARGIN_M
        )
        actor = kept.actor
        location = actor.get_location()
        if kept.path and not kept.near_route_end:
            if location.distance(kept.path[-1]) > lookahead:
                return
            kept.near_route_end = True
        # Where the last walk found the road going on, no further ahead.
        if lookahead <= kept.checked_lookahead and (
            math.hypot(location.x - kept.checked_x, location.y - kept.checked_y)
            < _DEAD_END_STEP_M
        ):
            return
        found = run_carla_map(world).get_waypoint(location)
        if found is None:
            return
        # Codon does not narrow an Optional (docs/typecheck.md).
        start: "carla.Waypoint" = found
        if not _dead_end_within(start, lookahead):
            kept.checked_x = location.x
            kept.checked_y = location.y
            kept.checked_lookahead = lookahead
            return
        actor.set_autopilot(False, self.port)
        kept.held = True
        ahead = _ahead_of(kept.path, location)
        if ahead:
            # On a path it was given, which ends before the dead end: the
            # TrafficManager would plan past its end into it.  The rest of the
            # path is driven here instead, and the vehicle stopped on its end.
            approach = FinalApproach(actor, location, ahead)
            kept.approach = approach
            logger.info(
                "%s: its road ends within %.0f m, past the end of its path, "
                "where the TrafficManager would destroy it; off autopilot, "
                "it drives the last %.0f m of its path itself and stops there",
                _actor_name(actor),
                lookahead,
                approach.length,
            )
            return
        _vehicle(actor).apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        logger.info(
            "%s: its road ends within %.0f m, where the TrafficManager would "
            "destroy it; taken off autopilot and held braking until it is "
            "given a new route",
            _actor_name(actor),
            lookahead,
        )

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


def _in_junction(carla_map: "carla.Map", location: "carla.Location") -> bool:
    """Whether *location* is on a junction's road."""
    found = carla_map.get_waypoint(location)
    return found is not None and found.is_junction


def _at_junction_mouth(carla_map: "carla.Map", location: "carla.Location") -> bool:
    """Whether *location* is in a junction or within :data:`_JUNCTION_MOUTH_M`
    of one ahead along its lane."""
    found = carla_map.get_waypoint(location)
    if found is None:
        return False
    if found.is_junction:
        return True
    for following in found.next(_JUNCTION_MOUTH_M):
        if following.is_junction:
            return True
    return False


def _beyond_goal(
    goal: "Lanelet2Pose", step: float, distance: float = 40.0
) -> List["Lanelet2Pose"]:
    """Poses every *step* metres along the lanes after *goal*, up to
    *distance*: the first lanelet following each one (the routing graph's)."""
    from lanelet2.core import ConstLanelet  # noqa: PLC0415

    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415
    from ..trajectory.relative_lane import _length  # noqa: PLC0415

    mm = MapManager.get_instance()
    layer = mm.lanelet_map.laneletLayer
    if not layer.exists(int(goal.lanelet_id)):
        return []
    lanelet: ConstLanelet = layer[int(goal.lanelet_id)]
    s = float(goal.s) + step
    covered = 0.0
    out: List[Lanelet2Pose] = []
    while covered < distance:
        length = _length(lanelet)
        while s <= length and covered < distance:
            out.append(Lanelet2Pose(lanelet_id=int(lanelet.id), s=s))
            s += step
            covered += step
        following = mm.routing_graph.following(lanelet)
        if not following:
            break
        lanelet = following[0]
        s -= length
    return out


def _clear_of_junctions(
    carla_map: "carla.Map", path: List["carla.Location"]
) -> List["carla.Location"]:
    """*path* without its points in a junction or next to one.

    The TrafficManager takes a junction's way out whose road is the road of
    the next point, failing that the one nearest it.  A point in a junction is
    on a road of the junction's own -- or on several, at its edge -- and the
    nearest way out of a junction's middle is any of them; so the next point
    at a junction has to be on the road out.
    """
    inside: List[bool] = []
    for location in path:
        found = carla_map.get_waypoint(location)
        inside.append(found is not None and found.is_junction)
    clear: List["carla.Location"] = []
    for i, location in enumerate(path):
        if not any(inside[max(0, i - 1) : i + 2]):
            clear.append(location)
    return clear


def _tick_seconds(world: "carla.World") -> float:
    """How much simulated time the last tick covered (the run's fixed step
    when the world cannot say)."""
    try:
        return float(world.get_snapshot().timestamp.delta_seconds)
    except (AttributeError, RuntimeError):
        return FIXED_DELTA_SECONDS


def _from_here(
    path: List["carla.Location"], location: "carla.Location"
) -> List["carla.Location"]:
    """The points of *path* from *location* on: the one nearest it and those
    after, less that nearest one where *location* is already past it (nearer
    the next point than it is)."""
    nearest = 0
    best = float("inf")
    for i, point in enumerate(path):
        gap = location.distance(point)
        if gap < best:
            best = gap
            nearest = i
    if nearest + 1 < len(path) and location.distance(path[nearest + 1]) < path[
        nearest
    ].distance(path[nearest + 1]):
        nearest += 1
    return path[nearest:]


def _ahead_of(
    path: List["carla.Location"], location: "carla.Location"
) -> List["carla.Location"]:
    """The points of *path* after the one nearest *location*.

    Not that nearest one: the TrafficManager waits for each point to come
    within a few metres, and one beside or behind the vehicle may never.
    """
    nearest = 0
    best = float("inf")
    for i, point in enumerate(path):
        gap = location.distance(point)
        if gap < best:
            best = gap
            nearest = i
    return path[nearest + 1 :]


def _dead_end_within(start: "carla.Waypoint", distance: float) -> bool:
    """Whether some way on from *start* ends within *distance* metres.

    Walked in :data:`_DEAD_END_STEP_M` steps down every branch: ``next()`` is
    empty where a lane ends with no successor.  A road that branches into
    more than :data:`_DEAD_END_MAX_WAYPOINTS` waypoints is taken to go on.
    """
    frontier: List[Tuple["carla.Waypoint", float]] = [(start, 0.0)]
    # Where branches meet again, the road on is walked once.
    seen: set[Tuple[int, int, int, int]] = set()
    while frontier and len(seen) < _DEAD_END_MAX_WAYPOINTS:
        waypoint, travelled = frontier.pop()
        key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id, int(waypoint.s))
        if travelled >= distance or key in seen:
            continue
        seen.add(key)
        ahead = waypoint.next(_DEAD_END_STEP_M)
        if not ahead:
            return True
        for following in ahead:
            frontier.append((following, travelled + _DEAD_END_STEP_M))
    return False


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
    waypoint: "carla.Waypoint", direction: LaneChangeDirection
) -> Optional["carla.Waypoint"]:
    """Return the waypoint of the lane beside *waypoint* in *direction*.

    ``None`` when there is no lane that way, which is the honest answer to a
    lane change that cannot happen.
    """
    neighbour: Optional["carla.Waypoint"] = None
    if direction is LaneChangeDirection.RIGHT:
        neighbour = waypoint.get_right_lane()
    else:
        neighbour = waypoint.get_left_lane()
    return neighbour


def _lane_key_of(waypoint: "carla.Waypoint") -> Tuple[int, int]:
    """Return the ``(road_id, lane_id)`` a waypoint sits on."""
    return (waypoint.road_id, waypoint.lane_id)


def _on_target_side(
    entity: DrivenEntity, key: Tuple[int, int], location: "carla.Location"
) -> bool:
    """Whether the lane *key*, which is not the one aimed at, is where
    *entity*'s target lane leads rather than where its original lane does.

    A lane both lead to is the target side only once the vehicle has been
    there (``_lane_change_crossed``); being on one only the target leads to
    records that it has.
    """
    if not hasattr(entity, "_lane_change_waypoint") or not hasattr(
        entity, "_lane_change_origin"
    ):
        return False
    ahead = _lane_keys_reached(entity, location)
    if key not in ahead:
        return False
    behind = entity._lane_change_origin_ahead
    if behind is not None:
        left: set[Tuple[int, int]] = behind
        if key in left:
            return entity._lane_change_crossed
    entity._lane_change_crossed = True
    return True


def _lane_keys_reached(
    entity: DrivenEntity, location: "carla.Location"
) -> set[Tuple[int, int]]:
    """The lanes *entity*'s lane change target leads to, for one at *location*.

    Walked from the target's waypoint as far as the vehicle can have driven
    (:data:`_LANE_CHANGE_AHEAD_MARGIN_M`), and kept on the entity: walked
    again only when the vehicle has come further than the last walk reached,
    then at least twice as far.  The original lane is walked as far beside
    it, into ``_lane_change_origin_ahead``.  Empty without a target waypoint.
    """
    target = entity._lane_change_waypoint
    if target is None:
        return set[Tuple[int, int]]()
    start: "carla.Waypoint" = target
    needed = (
        2.0 * location.distance(start.transform.location) + _LANE_CHANGE_AHEAD_MARGIN_M
    )
    cached = entity._lane_change_ahead
    if cached is not None and needed <= entity._lane_change_horizon_m:
        known: set[Tuple[int, int]] = cached
        return known
    horizon = max(needed, 2.0 * entity._lane_change_horizon_m)
    keys = _lane_keys_ahead(start, horizon)
    entity._lane_change_ahead = keys
    entity._lane_change_horizon_m = horizon
    origin = entity._lane_change_origin
    if origin is not None:
        beside: "carla.Waypoint" = origin
        entity._lane_change_origin_ahead = _lane_keys_ahead(beside, horizon)
    return keys


def _lane_keys_ahead(start: "carla.Waypoint", distance: float) -> set[Tuple[int, int]]:
    """The ``(road_id, lane_id)`` of every lane *start*'s lane leads to within
    *distance* metres, its own included.

    Walked in :data:`_LANE_CHANGE_AHEAD_STEP_M` steps down every branch,
    nearest first, and no further than
    :data:`_LANE_CHANGE_AHEAD_MAX_WAYPOINTS` waypoints.
    """
    keys: set[Tuple[int, int]] = set()
    keys.add(_lane_key_of(start))
    # Breadth first: every step is as long, so a waypoint is first reached
    # the shortest way, and a branch that meets another again is not cut
    # short by having been reached the long way round first.
    frontier: List[Tuple["carla.Waypoint", float]] = [(start, 0.0)]
    index = 0
    # Where branches meet again, the road on is walked once.
    seen: set[Tuple[int, int, int, int]] = set()
    while index < len(frontier) and len(seen) < _LANE_CHANGE_AHEAD_MAX_WAYPOINTS:
        waypoint, travelled = frontier[index]
        index += 1
        key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id, int(waypoint.s))
        if travelled >= distance or key in seen:
            continue
        seen.add(key)
        for following in waypoint.next(_LANE_CHANGE_AHEAD_STEP_M):
            keys.add(_lane_key_of(following))
            frontier.append((following, travelled + _LANE_CHANGE_AHEAD_STEP_M))
    return keys


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
