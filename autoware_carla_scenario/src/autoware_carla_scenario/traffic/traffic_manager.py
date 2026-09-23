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
from typing import Any, ClassVar, Collection, List, Optional, Tuple

from ..constants import (
    LANE_CHANGE_CENTER_TOLERANCE_M,
    LANE_CHANGE_HEADING_TOLERANCE_DEG,
)
from ..kinematics.angle import normalize_angle_deg
from .base import (
    LaneChangeDirection,
    TrafficBackend,
    TrafficContext,
    TurnDirection,
    _entity_name,
)
from .config import TrafficManagerBackendConfig

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


class TrafficManagerBackend(TrafficBackend):
    """Traffic driven by CARLA's own TrafficManager.

    The default backend, and the one every existing scenario gets: vehicles the
    scenario spawned are handed to the TrafficManager after warm-up, and the
    manoeuvres a scenario asks for become TrafficManager calls.

    The client is injected rather than passed to each call, from either of the
    two places that already hold one: :meth:`prepare`, when
    :class:`~autoware_carla_scenario.ScenarioRunner` builds the run's context,
    or the constructor, for an entity that was only ever given a client (see
    :meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.set_client`).
    A backend that was never given one says so rather than failing inside CARLA.
    """

    name: ClassVar[str] = "traffic_manager"

    def __init__(
        self,
        config: Optional[TrafficManagerBackendConfig] = None,
        *,
        client: Any = None,
    ) -> None:
        """Create the backend.

        Args:
            config: Backend options; defaults are used when omitted.
            client: The CARLA client the TrafficManager is reached through, for
                a caller that already holds one.  :meth:`prepare` supplies it
                otherwise.
        """
        self._config = config or TrafficManagerBackendConfig()
        self._client = client
        self._random_seed: Optional[int] = None
        self._closed = False

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

    def start(self, world: Any, *, skip_actor_ids: Collection[int] = ()) -> None:
        """Hand every vehicle that is not driven elsewhere to the TrafficManager."""
        skip = set(skip_actor_ids)
        enabled = 0
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id in skip:
                continue
            actor.set_autopilot(True, self.port)
            enabled += 1
        if enabled:
            logger.info("Autopilot enabled on %d vehicle(s)", enabled)
        if skip:
            logger.info(
                "Autopilot skipped for %s — external control expected",
                ", ".join(str(actor_id) for actor_id in sorted(skip)),
            )

    def close(self) -> None:
        """Shut the TrafficManager down so the next run gets a fresh one.

        Resets its ``InMemoryMap`` cache and internal state.  Safe to call twice
        and safe to call after a failed :meth:`prepare`, because teardown runs
        whatever happened during the run.
        """
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
        return {
            "backend": self.name,
            "port": self.port,
            "random_seed": self._random_seed,
        }

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(
        self, entity: Any, world: Any, direction: LaneChangeDirection
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

        carla_map = world.get_map()
        entity._lane_change_target = _adjacent_lane(
            carla_map, actor.get_location(), direction
        )
        entity._lane_change_map = carla_map
        if entity._lane_change_target is None:
            logger.warning(
                "%s: no lane %s to change into",
                _entity_name(entity),
                direction.value,
            )

        tm.force_lane_change(actor, direction.to_carla_bool())
        logger.info(
            "%s: forced a %s lane change", _entity_name(entity), direction.value
        )

    def lane_change_finished(self, entity: Any, world: Any) -> bool:
        """Whether *entity* has settled onto the lane it was sent to.

        Three things have to be true, and a lane id change on its own is not
        enough: a vehicle whose centre has just crossed the boundary is still
        diagonal across two lanes, and calling that finished would let a
        reaction fire mid-manoeuvre.

        A manoeuvre the TrafficManager never makes simply never finishes; see
        :meth:`TrafficBackend.lane_change_finished` for why that is the honest
        answer rather than a timeout of its own.
        """
        del world
        target = getattr(entity, "_lane_change_target", None)
        carla_map = getattr(entity, "_lane_change_map", None)
        actor = getattr(entity, "actor", None)
        if target is None or carla_map is None or actor is None:
            return False

        # One RPC per tick: the transform carries both the location the map is
        # queried with and the heading the check needs.
        transform = actor.get_transform()
        waypoint = carla_map.get_waypoint(transform.location, project_to_road=True)
        if waypoint is None or _lane_key_of(waypoint) != target:
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

    def set_desired_speed(self, entity: Any, world: Any, speed_kmh: float) -> None:
        """Hold *entity* at *speed_kmh*.

        ``set_desired_speed`` is a target, not a jump: the TrafficManager gets
        the vehicle there under its own acceleration limits.  A rate asked for
        by the scenario is therefore a walk of *targets*, stepped by the action
        once per tick from the vehicle's own speed, and this call is one step
        of it.  The value given here stays in force until it is changed, which
        is why the action reissues only when it has a rate to walk.
        """
        del world
        actor = _require_actor(entity, "set_desired_speed")
        if actor is None:
            return
        tm = self._require_tm("set_desired_speed")
        if tm is None:
            return

        tm.set_desired_speed(actor, speed_kmh)
        logger.debug(
            "%s: desired speed set to %.1f km/h", _entity_name(entity), speed_kmh
        )

    def turn_at_junction(
        self,
        entity: Any,
        world: Any,
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
        del kwargs
        actor = _require_actor(entity, "turn_at_junction")
        if actor is None:
            return
        tm = self._require_tm("turn_at_junction")
        if tm is None:
            return

        current = world.get_map().get_waypoint(actor.get_location())
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

        tm.set_path(actor, path)
        logger.info(
            "%s: set a %s turn route (%d points)",
            _entity_name(entity),
            direction.value,
            len(path),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_tm(self, what: str) -> Any:
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


def _require_actor(entity: Any, what: str) -> Any:
    """Return *entity*'s actor, or ``None`` with a warning."""
    actor = getattr(entity, "actor", None)
    if actor is None:
        logger.warning(
            "%s: %s asked for before the actor exists", _entity_name(entity), what
        )
    return actor


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _adjacent_lane(
    carla_map: Any,
    location: Any,
    direction: LaneChangeDirection,
) -> Optional[Tuple[int, int]]:
    """Return the ``(road_id, lane_id)`` beside *location* in *direction*.

    ``None`` when there is no lane that way, which is the honest answer to a
    lane change that cannot happen.
    """
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    if waypoint is None:
        return None
    neighbour = (
        waypoint.get_right_lane()
        if direction is LaneChangeDirection.RIGHT
        else waypoint.get_left_lane()
    )
    return None if neighbour is None else _lane_key_of(neighbour)


def _lane_key_of(waypoint: Any) -> Tuple[int, int]:
    """Return the ``(road_id, lane_id)`` a waypoint sits on."""
    return (waypoint.road_id, waypoint.lane_id)


def _heading_error_deg(yaw: float, reference_yaw: float) -> float:
    """Return the absolute heading difference in degrees, wrapped to 180."""
    return abs(normalize_angle_deg(yaw - reference_yaw))


# ---------------------------------------------------------------------------
# Turn route geometry
# ---------------------------------------------------------------------------


def compute_turn_route(
    current_wp: Any,
    direction: TurnDirection,
    *,
    search_distance: float = TURN_SEARCH_DISTANCE_M,
    waypoint_step: float = TURN_WAYPOINT_STEP_M,
    post_junction_distance: float = TURN_POST_JUNCTION_DISTANCE_M,
) -> List[Any]:
    """Build a waypoint path through the next junction in the desired direction."""
    pre_junction_wp, junction_entries = _walk_to_junction(
        current_wp, search_distance, waypoint_step
    )
    if pre_junction_wp is None or not junction_entries:
        return []

    branches: List[List[Any]] = []
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
    start_wp: Any, search_distance: float, waypoint_step: float
) -> tuple[Optional[Any], List[Any]]:
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
            return wp, entries

        wp = next_wps[0]
        distance += waypoint_step

    return None, []


def _trace_through_junction(
    entry_wp: Any, waypoint_step: float, post_junction_distance: float
) -> List[Any]:
    """Follow waypoints from *entry_wp* through the junction and a bit beyond.

    The extra post-junction distance provides a stable exit heading for
    direction comparison.
    """
    path: List[Any] = [entry_wp]
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
    pre_junction_wp: Any,
    branches: List[List[Any]],
    direction: TurnDirection,
) -> Optional[List[Any]]:
    """Select the branch whose exit heading change is closest to the target.

    CARLA yaw convention (left-hand, clockwise-positive when viewed from
    above):

    - Left turn  ≈ −90° heading change
    - Right turn ≈ +90° heading change
    """
    entry_yaw = pre_junction_wp.transform.rotation.yaw
    target = _LEFT_TARGET_DEG if direction is TurnDirection.LEFT else _RIGHT_TARGET_DEG

    best: Optional[List[Any]] = None
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
