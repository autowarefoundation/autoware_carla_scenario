"""Lane-change action: force a lane change via TrafficManager."""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Union

from typing import Optional as _Optional

from ..conditions import BaseCondition
from ..conditions.base import find_actor_by_role_name
from ..constants import (
    DEFAULT_TM_PORT,
    LANE_CHANGE_CENTER_TOLERANCE_M,
    LANE_CHANGE_HEADING_TOLERANCE_DEG,
)
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)


class LaneChangeDirection(enum.Enum):
    """Direction of a lane change."""

    LEFT = "left"
    RIGHT = "right"

    def to_carla_bool(self) -> bool:
        """Convert to the boolean expected by ``TrafficManager.force_lane_change``.

        CARLA convention: ``True`` → right, ``False`` → left.
        """
        return self is LaneChangeDirection.RIGHT


class LaneChangeAction(BaseAction):
    """Force a lane change via TrafficManager.

    When the associated condition is satisfied, this action:

    1. Locates the target vehicle by its ``role_name``
    2. Calls ``TrafficManager.force_lane_change(actor, direction)`` to
       command an immediate lane change

    ``force_lane_change`` only queues the manoeuvre, so the action keeps
    watching the vehicle afterwards (see :meth:`is_finished`) and stays
    :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING` until the
    vehicle has actually settled onto the next lane.  That is the signal another
    actor can react to; ``done`` would fire while the car is still straddling
    the line.

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`LaneChangeDirection` — ``LEFT`` or ``RIGHT``.
        client: A ``carla.Client`` used to obtain the TrafficManager.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        once: If ``True`` (default) the action fires at most once.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: LaneChangeDirection,
        client: "carla.Client",
        condition: _Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "lane_change",
        once: bool = True,
        tm_port: int = DEFAULT_TM_PORT,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._direction = direction
        self._client = client
        self._tm_port = tm_port
        #: ``(road_id, lane_id)`` of the lane the command aims at, or ``None``
        #: when the manoeuvre could not be started.
        self._target_lane: _Optional[tuple[int, int]] = None
        #: The actor and the map, kept from :meth:`execute`.  ``is_finished``
        #: runs on every frame until the manoeuvre completes -- which, by
        #: design, may be never -- and CARLA rebuilds the map object on each
        #: ``get_map()`` call while ``find_actor_by_role_name`` fetches the
        #: whole actor list, so looking either up per tick is a server round
        #: trip for something that cannot change.
        self._actor: _Optional["carla.Actor"] = None
        self._map: _Optional["carla.Map"] = None

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Command a lane change via TrafficManager."""
        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            logger.warning("LaneChangeAction: actor '%s' not found", self._entity_name)
            self._target_lane = None
            return

        self._actor = actor
        self._map = world.get_map()
        # The lane aimed at, not merely the one left behind.  "Not the lane it
        # started on" is satisfied by simply driving onto the next road, whose
        # ids differ without the vehicle having moved sideways at all -- that
        # would report a lane change that never happened, and anything waiting
        # on `completeState` would fire early.
        self._target_lane = _adjacent_lane(
            self._map, actor.get_location(), self._direction
        )
        if self._target_lane is None:
            logger.warning(
                "LaneChangeAction: no lane %s of '%s' to change into",
                self._direction.value,
                self._entity_name,
            )

        tm = self._client.get_trafficmanager(self._tm_port)
        tm.force_lane_change(actor, self._direction.to_carla_bool())
        logger.info(
            "LaneChangeAction: forced %s lane change for '%s'",
            self._direction.value,
            self._entity_name,
        )

    def is_finished(self, world: "carla.World", running_for: float) -> bool:
        """Whether the vehicle has settled onto a different lane.

        Three things have to be true, and a lane id change on its own is not
        enough: a vehicle whose centre has just crossed the boundary is still
        diagonal across two lanes, and calling that finished would let a
        reaction fire mid-manoeuvre.

        Completion is the *target* lane, resolved when the command went out --
        not merely "a different lane id from before".  Lane ids are scoped to a
        road, so a vehicle that reaches a continuation road before moving
        sideways gets a different ``(road_id, lane_id)`` without having changed
        lane at all.

        A manoeuvre the TrafficManager never makes simply never finishes, and
        the action stays
        :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING`.  That
        is OpenSCENARIO's behaviour, and it is what keeps ``completeState`` from
        being reachable by a lane change that did not happen; ending the run on
        a timer is the scenario timeout's job, not this action's.  The same
        applies when the vehicle crosses onto a continuation road mid-manoeuvre
        and the recorded target no longer names its lane: the action reports
        nothing rather than reporting the wrong thing.

        Args:
            world: The CARLA world instance.
            running_for: Seconds since the command was issued.

        Returns:
            ``True`` once the vehicle is on another lane, within
            :data:`~autoware_carla_scenario.constants.LANE_CHANGE_CENTER_TOLERANCE_M`
            of its centre and within
            :data:`~autoware_carla_scenario.constants.LANE_CHANGE_HEADING_TOLERANCE_DEG`
            of its heading.
        """
        if self._target_lane is None or self._actor is None or self._map is None:
            return False

        # One RPC per tick: the transform carries both the location the map is
        # queried with and the heading the check needs.
        transform = self._actor.get_transform()
        waypoint = self._map.get_waypoint(transform.location, project_to_road=True)
        if waypoint is None or _lane_key_of(waypoint) != self._target_lane:
            return False

        # ``get_waypoint`` projects onto the lane centre, so the distance to it
        # is the lateral offset.
        offset = transform.location.distance(waypoint.transform.location)
        if offset > LANE_CHANGE_CENTER_TOLERANCE_M:
            return False

        heading_error = _heading_error_deg(
            transform.rotation.yaw, waypoint.transform.rotation.yaw
        )
        if heading_error > LANE_CHANGE_HEADING_TOLERANCE_DEG:
            return False

        logger.info(
            "LaneChangeAction: '%s' settled onto its new lane after %.1fs",
            self._entity_name,
            running_for,
        )
        return True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _adjacent_lane(
    carla_map: "carla.Map",
    location: "carla.Location",
    direction: LaneChangeDirection,
) -> _Optional[tuple[int, int]]:
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


def _lane_key_of(waypoint: "carla.Waypoint") -> tuple[int, int]:
    """Return the ``(road_id, lane_id)`` a waypoint identifies."""
    return (int(waypoint.road_id), int(waypoint.lane_id))


def _heading_error_deg(yaw: float, reference_yaw: float) -> float:
    """Return the absolute angle between two headings, in degrees, 0-180."""
    return abs((yaw - reference_yaw + 180.0) % 360.0 - 180.0)
