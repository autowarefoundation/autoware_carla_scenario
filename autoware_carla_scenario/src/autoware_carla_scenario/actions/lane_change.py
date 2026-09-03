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
        #: ``(road_id, lane_id)`` the vehicle was on when the command went out,
        #: or ``None`` when the manoeuvre could not be started.
        self._start_lane: _Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Command a lane change via TrafficManager."""
        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            logger.warning("LaneChangeAction: actor '%s' not found", self._entity_name)
            self._start_lane = None
            return

        # Recorded before the command, because "which lane did it leave" is the
        # only way to tell afterwards that it went anywhere.
        self._start_lane = _lane_key(world, actor)

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

        A manoeuvre the TrafficManager never makes simply never finishes, and
        the action stays
        :attr:`~autoware_carla_scenario.actions.base.ActionState.RUNNING`.  That
        is OpenSCENARIO's behaviour, and it is what keeps ``completeState`` from
        being reachable by a lane change that did not happen; ending the run on
        a timer is the scenario timeout's job, not this action's.

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
        if self._start_lane is None:
            return False

        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            return False

        waypoint = world.get_map().get_waypoint(
            actor.get_location(), project_to_road=True
        )
        if waypoint is None or _lane_key_of(waypoint) == self._start_lane:
            return False

        # ``get_waypoint`` projects onto the lane centre, so the distance to it
        # is the lateral offset.
        offset = actor.get_location().distance(waypoint.transform.location)
        if offset > LANE_CHANGE_CENTER_TOLERANCE_M:
            return False

        heading_error = _heading_error_deg(
            actor.get_transform().rotation.yaw, waypoint.transform.rotation.yaw
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


def _lane_key(world: "carla.World", actor: "carla.Actor") -> _Optional[tuple[int, int]]:
    """Return the ``(road_id, lane_id)`` under *actor*, or ``None``.

    The road id is carried too: lane ids restart per road, so comparing lane ids
    alone would read a road change as a lane change.
    """
    waypoint = world.get_map().get_waypoint(actor.get_location(), project_to_road=True)
    return None if waypoint is None else _lane_key_of(waypoint)


def _lane_key_of(waypoint: "carla.Waypoint") -> tuple[int, int]:
    """Return the ``(road_id, lane_id)`` a waypoint identifies."""
    return (int(waypoint.road_id), int(waypoint.lane_id))


def _heading_error_deg(yaw: float, reference_yaw: float) -> float:
    """Return the absolute angle between two headings, in degrees, 0-180."""
    return abs((yaw - reference_yaw + 180.0) % 360.0 - 180.0)
