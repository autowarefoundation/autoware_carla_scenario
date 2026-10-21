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
from .parameters import TrafficParameters
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


#: The blueprint a background vehicle gets when none is asked for: one CARLA
#: 0.9 and 0.10 both ship.
_BACKGROUND_BLUEPRINT = "vehicle.lincoln.mkz*"


class TrafficManagerBackend(TrafficBackend):
    """Traffic driven by CARLA's own TrafficManager.

    The default backend, and the one every existing scenario gets: vehicles the
    scenario spawned are handed to the TrafficManager after warm-up, and the
    manoeuvres a scenario asks for become TrafficManager calls.

    How its vehicles drive is :class:`~autoware_carla_scenario.traffic.parameters.TrafficParameters`
    (those :data:`~autoware_carla_scenario.traffic.parameters.PARAMETER_BACKENDS`
    gives ``traffic_manager``; it has no reaction time, braking or
    acceleration of a driver to set, nor any notion of sight): the speed and the gap are the
    TrafficManager's global settings, set in :meth:`prepare`; the rest are set
    on each vehicle as it is put on autopilot.  Every vehicle it drives gets
    them, the scenario's own as well as its background traffic.

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

    def __init__(
        self,
        config: Optional[TrafficManagerBackendConfig] = None,
        *,
        client: Optional["carla.Client"] = None,
        parameters: Optional[TrafficParameters] = None,
    ) -> None:
        """Create the backend.

        Args:
            config: Backend options; defaults are used when omitted.
            client: The CARLA client the TrafficManager is reached through, for
                a caller that already holds one.  :meth:`prepare` supplies it
                otherwise.
            parameters: How its vehicles drive; the TrafficManager's defaults
                when omitted.

        Raises:
            TrafficParameterError: If *parameters* sets a field the
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
        if parameters is not None:
            self.set_parameters(parameters)

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
        parameters = self.parameters
        if parameters.speed_factor is not None:
            # The TrafficManager's is a percentage *under* the limit.
            tm.global_percentage_speed_difference(
                (1.0 - parameters.speed_factor) * 100.0
            )
        if parameters.min_gap_m is not None:
            tm.set_global_distance_to_leading_vehicle(parameters.min_gap_m)

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
        """Report how far drivers see, when it changes, and do nothing about it.

        The TrafficManager rides the CARLA tick, and has no notion of sight:
        its vehicles know where every actor is, fog or not
        (:mod:`autoware_carla_scenario.visibility`).  So a visibility the
        run's weather set is logged with the speed a backend that models
        sight would hold a car to (:func:`~autoware_carla_scenario.visibility.sight_speed`,
        for SUMO's default car), and the traffic drives as it did --
        which is why ``follow_visibility`` is refused here.
        """
        seen = world_visibility(episode_of(world))
        if seen == self._visibility:
            return
        self._visibility = seen
        if seen is None:
            logger.info("TrafficManager: no visibility set any more; nothing to undo")
            return
        limit_kmh = sight_speed(seen, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        logger.info(
            "TrafficManager: drivers see %.0f m, so traffic that models sight "
            "would drive at most %.0f km/h (braking at %.1f m/s² after %.1f s); "
            "the TrafficManager models none, and its vehicles drive as before",
            seen,
            limit_kmh,
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
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
        """Put *actor* on autopilot, with the per-vehicle parameters set."""
        actor.set_autopilot(True, self.port)
        parameters = self.parameters
        if not _sets_per_vehicle(parameters):
            return  # the speed and the gap are global, set in prepare()
        tm = self._require_tm("autopilot")
        if tm is None:
            return
        vehicle = _vehicle(actor)
        if parameters.lane_changes is not None:
            tm.auto_lane_change(vehicle, parameters.lane_changes)
        # The TrafficManager's are percentages, the parameters' probabilities.
        if parameters.ignore_lights is not None:
            tm.ignore_lights_percentage(vehicle, parameters.ignore_lights * 100.0)
        if parameters.ignore_signs is not None:
            tm.ignore_signs_percentage(vehicle, parameters.ignore_signs * 100.0)
        if parameters.ignore_vehicles is not None:
            tm.ignore_vehicles_percentage(vehicle, parameters.ignore_vehicles * 100.0)
        if parameters.ignore_walkers is not None:
            tm.ignore_walkers_percentage(vehicle, parameters.ignore_walkers * 100.0)
        if parameters.random_lane_changes is not None:
            chance = parameters.random_lane_changes * 100.0
            tm.random_left_lanechange_percentage(vehicle, chance)
            tm.random_right_lanechange_percentage(vehicle, chance)

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


def _sets_per_vehicle(parameters: TrafficParameters) -> bool:
    """Whether *parameters* set a field the TrafficManager sets on each
    vehicle, rather than globally."""
    return (
        parameters.lane_changes is not None
        or parameters.ignore_lights is not None
        or parameters.ignore_signs is not None
        or parameters.ignore_vehicles is not None
        or parameters.ignore_walkers is not None
        or parameters.random_lane_changes is not None
    )


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
