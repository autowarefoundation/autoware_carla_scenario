"""Driving a vehicle through the CARLA TrafficManager.

This is the default mechanism: a vehicle the scenario spawns and leaves to
CARLA is steered by the TrafficManager, and a manoeuvre asked of it becomes a
TrafficManager call.

It lives on the entity rather than in the actions that ask for manoeuvres,
because "change lane" and "turn at the next junction" are *intents*, and how an
intent becomes motion depends entirely on what is driving.  ``tm.set_path`` is
TrafficManager vocabulary: an Autoware ego would not take a list of waypoints
for "turn left", it would take a different goal, and a policy-driven ego would
take neither.  An action that reached for the TrafficManager itself would work
for exactly one kind of vehicle while looking as though it worked for all of
them.

Entities that are *not* TrafficManager-driven
(:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`,
:class:`~autoware_carla_scenario.entity.carla_driver_entity.CarlaDriverEntity`)
override these and say so rather than inheriting a call that would be sent to a
TrafficManager which is not driving them.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, runtime_checkable

from ..kinematics.angle import normalize_angle_deg
from ..constants import (
    DEFAULT_TM_PORT,
    LANE_CHANGE_CENTER_TOLERANCE_M,
    LANE_CHANGE_HEADING_TOLERANCE_DEG,
)

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

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


class LaneChangeDirection(enum.Enum):
    """Direction of a lane change.

    Defined here, with the mechanism, rather than in the action that asks for
    one: the action names an intent, and this is the side that knows what the
    intent means to a driver.  ``actions.lane_change`` re-exports it, which is
    where scenario authors and the editor reach it.
    """

    LEFT = "left"
    RIGHT = "right"

    def to_carla_bool(self) -> bool:
        """Convert to the boolean expected by ``TrafficManager.force_lane_change``.

        CARLA convention: ``True`` -> right, ``False`` -> left.
        """
        return self is LaneChangeDirection.RIGHT


class TurnDirection(enum.Enum):
    """Direction of a turn at a junction."""

    LEFT = "left"
    RIGHT = "right"


@runtime_checkable
class LaneChanging(Protocol):
    """What :class:`~autoware_carla_scenario.actions.lane_change.LaneChangeAction`
    needs of an entity.

    Stated as a protocol rather than a base class because the action does not
    care what performs the manoeuvre -- only that something can be asked for one
    and asked whether it is done.
    """

    def change_lane(
        self, world: "carla.World", direction: "LaneChangeDirection"
    ) -> None:
        """Move one lane in *direction*."""
        ...

    def lane_change_finished(self, world: "carla.World") -> bool:
        """Whether the manoeuvre has settled."""
        ...


@runtime_checkable
class TurningAtJunctions(Protocol):
    """What :class:`~autoware_carla_scenario.actions.turn.TurnAction` needs."""

    def turn_at_junction(
        self, world: "carla.World", direction: "TurnDirection", **kwargs: object
    ) -> None:
        """Go *direction* at the next junction ahead."""
        ...


class TrafficManagerDriven:
    """Manoeuvres for a vehicle the TrafficManager steers.

    Mixed into :class:`~autoware_carla_scenario.entity.ego.EgoVehicle` and
    :class:`~autoware_carla_scenario.entity.vehicle_entity.VehicleEntity`, the
    two entities CARLA drives.  Both already own an ``actor``; this adds the
    client the TrafficManager is reached through and the manoeuvres themselves.

    The client is injected rather than passed to each call:
    :class:`~autoware_carla_scenario.ScenarioRunner` hands it to the ego and
    :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_entity`
    to each NPC, the same two places that already know it.  An entity that was
    never given one says so rather than failing inside CARLA.
    """

    #: Set by :meth:`set_client`; ``None`` until then.
    _tm_client: Optional["carla.Client"] = None
    _tm_port: int = DEFAULT_TM_PORT

    def set_client(
        self, client: "carla.Client", tm_port: int = DEFAULT_TM_PORT
    ) -> None:
        """Inject the CARLA client the TrafficManager is reached through."""
        self._tm_client = client
        self._tm_port = tm_port

    # ------------------------------------------------------------------
    # Manoeuvres
    # ------------------------------------------------------------------

    def change_lane(self, world: "carla.World", direction: LaneChangeDirection) -> None:
        """Move one lane in *direction*.

        Records the lane aimed at so :meth:`lane_change_finished` can tell a
        completed manoeuvre from a vehicle that merely drove onto the next
        road: lane ids are scoped to a road, so ``(road_id, lane_id)`` changes
        without the vehicle having moved sideways at all.
        """
        actor = self._require_actor("change_lane")
        if actor is None:
            return
        tm = self._require_tm("change_lane")
        if tm is None:
            return

        carla_map = world.get_map()
        self._lane_change_target = _adjacent_lane(
            carla_map, actor.get_location(), direction
        )
        self._lane_change_map = carla_map
        if self._lane_change_target is None:
            logger.warning(
                "%s: no lane %s to change into", type(self).__name__, direction.value
            )

        tm.force_lane_change(actor, direction.to_carla_bool())
        logger.info("%s: forced a %s lane change", type(self).__name__, direction.value)

    def lane_change_finished(self, world: "carla.World") -> bool:
        """Whether the vehicle has settled onto the lane it was sent to.

        Three things have to be true, and a lane id change on its own is not
        enough: a vehicle whose centre has just crossed the boundary is still
        diagonal across two lanes, and calling that finished would let a
        reaction fire mid-manoeuvre.

        A manoeuvre the TrafficManager never makes simply never finishes, which
        is OpenSCENARIO's behaviour: ending the run on a timer is the scenario
        timeout's job.
        """
        del world
        target = getattr(self, "_lane_change_target", None)
        carla_map = getattr(self, "_lane_change_map", None)
        actor = getattr(self, "actor", None)
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

    def turn_at_junction(
        self,
        world: "carla.World",
        direction: TurnDirection,
        *,
        search_distance: float = TURN_SEARCH_DISTANCE_M,
        waypoint_step: float = TURN_WAYPOINT_STEP_M,
        post_junction_distance: float = TURN_POST_JUNCTION_DISTANCE_M,
    ) -> None:
        """Go *direction* at the next junction ahead.

        The route through the junction is worked out from the map and handed to
        the TrafficManager, which is what steering this vehicle means.  Another
        entity would answer the same intent differently -- an Autoware ego by
        being sent to a goal beyond the junction, not by being given a list of
        waypoints -- which is why the intent stops at the entity boundary and
        the mechanism does not leak back into the action.
        """
        actor = self._require_actor("turn_at_junction")
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
                "%s: no %s turn route found", type(self).__name__, direction.value
            )
            return

        tm.set_path(actor, path)
        logger.info(
            "%s: set a %s turn route (%d points)",
            type(self).__name__,
            direction.value,
            len(path),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_actor(self, what: str) -> Optional["carla.Actor"]:
        actor = getattr(self, "actor", None)
        if actor is None:
            logger.warning(
                "%s: %s asked for before the actor exists",
                type(self).__name__,
                what,
            )
        return actor

    def _require_tm(self, what: str) -> Optional["carla.TrafficManager"]:
        if self._tm_client is None:
            logger.warning(
                "%s: %s needs a CARLA client; none was injected. "
                "ScenarioRunner gives one to the ego and register_entity() to "
                "each NPC.",
                type(self).__name__,
                what,
            )
            return None
        return self._tm_client.get_trafficmanager(self._tm_port)


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
    neighbour = (
        waypoint.get_right_lane()
        if direction is LaneChangeDirection.RIGHT
        else waypoint.get_left_lane()
    )
    return None if neighbour is None else _lane_key_of(neighbour)


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
            return wp, entries

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
    """
    entry_yaw = pre_junction_wp.transform.rotation.yaw
    target = _LEFT_TARGET_DEG if direction is TurnDirection.LEFT else _RIGHT_TARGET_DEG

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
