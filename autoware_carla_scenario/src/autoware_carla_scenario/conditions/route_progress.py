"""How far an entity has come along the scenario's route.

The condition a logical scenario times its events by: not "12.4 s into the
run" or "at x = 81234", which mean nothing on another map, but "once the ego
is 15 m short of the junction" -- a distance along the route the scenario's
route search found (:mod:`autoware_carla_scenario.route`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ..constants import EGO_ROLE_NAME
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult, find_actor_by_role_name
from .comparison import ComparisonRule, ScalarComparisonRule

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "HIDDEN_BELOW_ROUTE_M",
    "RouteProgressCondition",
    "route_s_of_carla_point",
]

#: An entity this far (m) or more below the route where it projects is out of
#: the world -- spawned hidden, parked under the map -- and has no progress.
HIDDEN_BELOW_ROUTE_M = 50.0


def route_below(s: float, z: float) -> float:
    """How far (m) CARLA height *z* is below the scenario route at route s *s*."""
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..route.active import scenario_route_frame  # noqa: PLC0415

    frame = scenario_route_frame()
    return frame.height(s) - MapManager.get_instance().z_offset - z


def route_s_of_carla_point(
    x: float, y: float, near: Optional[float] = None
) -> tuple[float, float]:
    """``(route s, distance)`` of a CARLA world point on the scenario's route.

    The point is projected onto the route's reference line (see
    :class:`~autoware_carla_scenario.route.frame.RouteFrame.project`), so an
    entity on a lane beside the route -- after a lane change, say -- is at the
    route s it is abreast of.

    Raises:
        ValueError: If the scenario has no route.
    """
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..route.active import scenario_route_frame  # noqa: PLC0415

    frame = scenario_route_frame()
    offset_x, offset_y = MapManager.get_instance().mgrs_offset
    return frame.project(x + offset_x, -y + offset_y, near)


class RouteProgressCondition(BaseCondition):
    """Satisfied when an entity's distance along the route meets a rule.

    The entity's progress is its route s -- metres along the scenario's route
    from where the route starts -- found by projecting its position onto the
    route's reference line, so a lane change onto a parallel lane does not
    change it.  It is compared with ``anchor_s + value``: *value* metres past
    the *anchor* (``start``, ``end``, ``segment:K:start``, ``segment:K:end``,
    ``junction:K:entry``, ``junction:K:exit``; see
    :func:`~autoware_carla_scenario.route.model.parse_anchor`), negative before
    it.  So ``value=-15, anchor="junction:0:entry"`` with
    ``GREATER_THAN_OR_EQUAL`` holds from 15 m before the route's first
    junction on, on whatever map the route was found.

    Successive checks look for the entity near where it was last found, so a
    route that passes the same place twice is followed pass by pass -- and
    over the whole route again when it is not near there any more (it was
    teleported, or the condition was not checked for a while).  An entity
    :data:`HIDDEN_BELOW_ROUTE_M` or more below the route (spawned hidden,
    parked under the map) or not in the world has no progress: the condition
    does not hold and :attr:`progress` is ``None``.

    Usable as an action's trigger and as a waypoint condition (a vertex of a
    *Follow Trajectory* departs once it holds).

    Args:
        entity_name: ``role_name`` of the entity measured; ``None`` is the ego.
        value: Metres past the anchor to compare with.
        rule: Comparison operator; ``GREATER_THAN_OR_EQUAL`` (default) reads
            "has come at least this far".
        anchor: The named point *value* counts from; ``None`` is ``start``.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.

    Raises:
        ValueError: On an anchor that is not one, or a negative tolerance.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _value: float
    _rule: ComparisonRule
    _anchor: Optional[str]
    _tolerance: float
    _last: Optional[float]

    def __init__(
        self,
        entity_name: Union[EntityRole, str, None] = None,
        value: float = 0.0,
        rule: ComparisonRule = ComparisonRule.GREATER_THAN_OR_EQUAL,
        anchor: Optional[str] = None,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        from ..route.model import parse_anchor  # noqa: PLC0415

        super().__init__(label=label)
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if anchor is not None and str(anchor).strip():
            parse_anchor(anchor)
        else:
            anchor = None
        # Kept as its string, the one thing every use of it reads.
        self._entity_name = str(EGO_ROLE_NAME if entity_name is None else entity_name)
        self._value = float(value)
        self._rule = rule
        self._anchor = anchor
        self._tolerance = float(tolerance)
        self._last: Optional[float] = None

    @property
    def progress(self) -> Optional[float]:
        """The entity's route s when last checked, or ``None``."""
        return self._last

    def get_details(self) -> dict[str, Any]:
        return {
            "entity_name": str(self._entity_name),
            "value": self._value,
            "anchor": self._anchor or "start",
            "rule": self._rule.name,
            "progress": self._last,
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a success result when the entity's progress meets the rule."""
        from ..route.active import scenario_route  # noqa: PLC0415

        route = scenario_route()
        if route is None:
            raise ValueError(
                f"{self.label}: a route_progress condition needs the scenario's "
                "route, and this scenario has none"
            )
        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            self._last = None
            return None
        location = actor.get_transform().location
        s, _gap = route_s_of_carla_point(location.x, location.y, self._last)
        if route_below(s, location.z) >= HIDDEN_BELOW_ROUTE_M:
            # Parked out of the world: wherever it projects is not progress.
            self._last = None
            return None
        self._last = s
        target = route.anchor_s(self._anchor) + self._value
        comparison = ScalarComparisonRule(
            field="route_s", rule=self._rule, value=target, tolerance=self._tolerance
        )
        if not comparison.satisfied(s):
            return None
        # Codon's f-string parser rejects a sign in a format spec (`:+.1f`).
        offset = self._value.__format__("+.1f")
        return ScenarioResult(
            passed=True,
            message=(
                f"{self._entity_name} at route s {s:.1f} m {self._rule.text} "
                f"{target:.1f} m ({self._anchor or 'start'} {offset} m)"
            ),
            elapsed_seconds=elapsed,
        )
