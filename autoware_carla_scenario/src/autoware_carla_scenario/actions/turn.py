"""Turn action: have a vehicle take a left, right or straight path through the next junction."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Union

from ..conditions.base import BaseCondition
from ..entity.registry import fail_if_pedestrian, find_vehicle_entity
from ..traffic import TurnDirection
from ..entity_role import EntityRole
from ..control_domain import ControlDomain
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..traffic.driven import BackendDriven

logger = logging.getLogger(__name__)


class TurnAction(BaseAction):
    """Have a vehicle take a left, right or straight path through the next junction.

    When the associated condition is satisfied, this action locates the
    vehicle by its ``role_name`` and hands the turn to it
    (:meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.turn_at_junction`),
    which carries it to the run's traffic backend.  The TrafficManager backend,
    for one, walks forward along CARLA waypoints (which reflect the underlying
    OpenDRIVE road network) to the next junction, picks the path through it
    whose heading change best matches *direction* (approximately -90 degrees
    for left, +90 for right and 0 for straight on, in CARLA's yaw convention)
    and sets it as the vehicle's path.

    The default *label* is ``"turn"`` (#2191): ``"turn_signal"``, its old
    default, read as :class:`~.turn_indicator.TurnIndicatorAction`'s.

    Args:
        entity_name: ``role_name`` of the vehicle actor to control.
        direction: :class:`TurnDirection` — ``LEFT``, ``RIGHT`` or ``STRAIGHT``.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        search_distance: Maximum distance (m) to look ahead for a junction.
        waypoint_step: Sampling distance (m) between waypoints.
        post_junction_distance: How far (m) past the junction exit to extend
            the route for a stable heading measurement.

    Raises:
        ValueError: *search_distance*, *waypoint_step* or
            *post_junction_distance* is a boolean, not positive or not
            finite.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _direction: TurnDirection
    _search_distance: float
    _waypoint_step: float
    _post_junction_distance: float

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        direction: TurnDirection,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "turn",
        once: bool = True,
        overrides: bool = False,
        search_distance: float = 200.0,
        waypoint_step: float = 2.0,
        post_junction_distance: float = 20.0,
    ) -> None:
        # #2077: a step of 0 (or less) failed mid-run in the waypoint walk,
        # and a NaN search distance silently set no turn.  A boolean would
        # read as 0 or 1 m (#2273).
        if (
            isinstance(search_distance, bool)
            or not math.isfinite(search_distance)
            or search_distance <= 0.0
        ):
            raise ValueError(
                f"search_distance must be finite and positive, not {search_distance!r}"
            )
        if (
            isinstance(waypoint_step, bool)
            or not math.isfinite(waypoint_step)
            or waypoint_step <= 0.0
        ):
            raise ValueError(
                f"waypoint_step must be finite and positive, not {waypoint_step!r}"
            )
        if (
            isinstance(post_junction_distance, bool)
            or not math.isfinite(post_junction_distance)
            or post_junction_distance <= 0.0
        ):
            raise ValueError(
                "post_junction_distance must be finite and positive, "
                f"not {post_junction_distance!r}"
            )
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            overrides=overrides,
        )
        self._entity_name = str(entity_name)
        self._direction = direction
        self._search_distance = search_distance
        self._waypoint_step = waypoint_step
        self._post_junction_distance = post_junction_distance

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The way the vehicle turns: its lateral domain."""
        return [(str(self._entity_name), ControlDomain.LATERAL)]

    def execute(self, world: "carla.World") -> None:
        """Ask the named entity to turn at the next junction."""
        entity = find_vehicle_entity(self._entity_name)
        if entity is None:
            fail_if_pedestrian(self._entity_name, "turn_at_junction")
            logger.warning("TurnAction: entity '%s' not found", str(self._entity_name))
            return

        if TYPE_CHECKING:
            # Codon calls a method a subclass overrides (AutowareEgoEntity's)
            # through a dispatch thunk, which takes no keyword argument
            # (docs/typecheck.md): the check calls BackendDriven's own, and
            # checks each override on its own, through its class.
            BackendDriven.turn_at_junction(
                entity,
                world,
                self._direction,
                search_distance=self._search_distance,
                waypoint_step=self._waypoint_step,
                post_junction_distance=self._post_junction_distance,
            )
        else:
            entity.turn_at_junction(
                world,
                self._direction,
                search_distance=self._search_distance,
                waypoint_step=self._waypoint_step,
                post_junction_distance=self._post_junction_distance,
            )
