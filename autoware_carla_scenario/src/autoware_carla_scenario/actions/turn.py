"""Turn action: analyse OpenDRIVE junctions ahead and set a turn route via TrafficManager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from typing import Optional as _Optional

from ..conditions import BaseCondition
from ..entity.registry import find_entity_by_role_name
from ..entity.tm_driving import TurnDirection
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming, warn_ignored_arguments

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)


class TurnAction(BaseAction):
    """Analyse OpenDRIVE junctions ahead and set a turn route via TrafficManager.

    When the associated condition is satisfied, this action:

    1. Locates the target vehicle by its ``role_name``
    2. Walks forward along CARLA waypoints (which reflect the underlying
       OpenDRIVE road network) to find the next junction
    3. Enumerates all possible paths through the junction
    4. Selects the path whose heading change best matches *direction*
       (approximately −90° for left, +90° for right in CARLA's yaw convention)
    5. Calls ``TrafficManager.set_path`` to apply the route

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`TurnDirection` — ``LEFT`` or ``RIGHT``.
        condition: Trigger condition (see :class:`BaseCondition`).
        client: Accepted and ignored.  It named the ``carla.Client`` this action
            used to reach the TrafficManager through, before the manoeuvre moved
            onto the entity, which is given a client of its own.  Passing it
            warns; new code omits it.
        tm_port: Accepted and ignored, for the same reason as *client*: the
            entity reaches the TrafficManager on the port the runner gave it.
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        once: If ``True`` (default) the action fires at most once.
        search_distance: Maximum distance (m) to look ahead for a junction.
        waypoint_step: Sampling distance (m) between waypoints.
        post_junction_distance: How far (m) past the junction exit to extend
            the route for a stable heading measurement.
    """

    _LEFT_TARGET_DEG: float = -90.0
    _RIGHT_TARGET_DEG: float = 90.0

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: TurnDirection,
        client: _Optional["carla.Client"] = None,
        condition: _Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "turn_signal",
        once: bool = True,
        search_distance: float = 200.0,
        waypoint_step: float = 2.0,
        post_junction_distance: float = 20.0,
        tm_port: _Optional[int] = None,
    ) -> None:
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = entity_name
        self._direction = direction
        self._search_distance = search_distance
        self._waypoint_step = waypoint_step
        self._post_junction_distance = post_junction_distance
        warn_ignored_arguments(type(self).__name__, client=client, tm_port=tm_port)

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Ask the named entity to turn at the next junction."""
        entity = find_entity_by_role_name(self._entity_name)
        if entity is None:
            logger.warning("TurnAction: entity '%s' not found", str(self._entity_name))
            return

        entity.turn_at_junction(
            world,
            self._direction,
            search_distance=self._search_distance,
            waypoint_step=self._waypoint_step,
            post_junction_distance=self._post_junction_distance,
        )
