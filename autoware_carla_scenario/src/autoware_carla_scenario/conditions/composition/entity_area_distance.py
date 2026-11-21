"""Distance from an entity's box to an area of the map, and the closest it came."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...utils.polygon import Point, bounding_circle, ground_box, polygon_distance
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ComparisonRule, ScalarComparisonRule
from ..map_area import MapArea, area_shapes
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class EntityAreaDistanceCondition(CompositionCondition):
    """Pass condition on how near an entity's body comes to an area of the map,
    keeping the closest it came.

    The area is ground chosen by what it is (:class:`~..map_area.MapArea`):
    the crosswalks, one crosswalk picked on the map, the lanes through a
    junction, Autoware's intersection areas.  The distance is on the ground,
    from the entity's bounding box -- its body, not its origin -- to the
    nearest outline of the area: ``0`` once the body touches or is over it.
    So "the ego stops short of the crosswalk" is ``GREATER_THAN 0`` held to
    the end, and "the ego's nose reaches the stop area" is ``LESS_THAN_OR_EQUAL
    0``.

    Whether or not it fires, the condition keeps the closest the entity came
    (:attr:`closest_m`, with when, and to which element of the area), and its
    details carry it into the run's result: a run that never came within the
    threshold still says how near it got.

    .. note::
        The area is resolved on the run's Lanelet2 map **in this
        constructor**, so
        :class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
        must be initialised first -- as it is in a scenario's ``setup()``.
        An area the map cannot hold -- a picked lanelet it lacks or of
        another ground, a ground it has none of -- is said there, once, as a
        warning; the condition then never holds and its details carry the
        problem (``problem``).  It is not refused, because only the map can
        tell: the scenario editor outlines the area on its map, and says so,
        before the run.

        The measurement is on the ground plane, as a plan of the map would
        draw it: where roads are stacked, an area on the road above or below
        the entity is as near as it is in the plan.

    Args:
        entity_name: The ``role_name`` of the actor whose body is measured.
        area: The ground to measure to.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative, or the area is not one any
            map could hold (:meth:`~..map_area.MapArea.validate`).
    """

    # Declared for the static check (docs/typecheck.md).
    _area: MapArea
    _comparison: ScalarComparisonRule
    # The area's outlines in CARLA's world frame, which layer and id each is,
    # and the circle round each (x, y, radius) that skips the far ones.
    _outlines: list[list[Point]]
    _layers: list[str]
    _element_ids: list[int]
    _circles: list[tuple[float, float, float]]
    _closest_m: Optional[float]
    _closest_at_s: Optional[float]
    _closest_index: int
    # Why the run's map holds none of the area, or "".
    _problem: str

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        area: MapArea,
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        super().__init__(entity_name=entity_name, label=label)
        # First, so a bad tolerance or area is refused before the map is read.
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )
        area.validate()
        self._area = area
        self._outlines = []
        self._layers = []
        self._element_ids = []
        self._circles = []
        self._problem = ""
        self._resolve()
        self._closest_m = None
        self._closest_at_s = None
        self._closest_index = -1

    def _resolve(self) -> None:
        """Read the area's outlines off the run's map, into CARLA's frame."""
        from ...coordinate.map_manager import MapManager  # noqa: PLC0415

        manager = MapManager.get_instance()
        offset_x, offset_y = manager.mgrs_offset
        try:
            shapes = area_shapes(manager.lanelet_map, self._area)
        except ValueError as exc:
            self._unheld(str(exc))
            return
        for shape in shapes:
            # Lanelet2's frame is MGRS and right-handed; CARLA's is the
            # OpenDRIVE origin's and left-handed (coordinate.transform).
            outline: list[Point] = []
            for x, y in shape.points:
                outline.append((x - offset_x, -(y - offset_y)))
            if not outline:
                continue
            self._outlines.append(outline)
            self._layers.append(shape.layer)
            self._element_ids.append(shape.element_id)
            self._circles.append(bounding_circle(outline))
        if not self._outlines:
            self._unheld(f"no {self._area.describe()} on the map")

    def _unheld(self, problem: str) -> None:
        """Say once that the run's map holds none of the area."""
        self._problem = problem
        logger.warning(
            "%s: nothing to measure to, so it never holds: %s", self.label, problem
        )

    @property
    def area(self) -> MapArea:
        """The ground measured to."""
        return self._area

    @property
    def closest_m(self) -> Optional[float]:
        """The closest (m) the entity's body has come to the area since the run
        (or the last :meth:`reset`) began; ``None`` before it was measured."""
        return self._closest_m

    @property
    def problem(self) -> str:
        """Why the run's map holds none of the area (it then never holds), or
        ``""``."""
        return self._problem

    @property
    def closest_at_s(self) -> Optional[float]:
        """The elapsed time (s) the entity first came that close; ``None``
        before it was measured."""
        return self._closest_at_s

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "area": self._area.describe(),
                "elements": len(self._outlines),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
            }
        )
        if self._problem:
            details["problem"] = self._problem
        closest = self._closest_m
        if closest is not None:
            details["closest_m"] = closest
            details["closest_at_s"] = self._closest_at_s
            index = self._closest_index
            details[f"closest_{self._layers[index]}"] = self._element_ids[index]
        return details

    def reset(self) -> None:
        """Forget the closest the entity came (:meth:`CompositionCondition.reset`)."""
        super().reset()
        self._closest_m = None
        self._closest_at_s = None
        self._closest_index = -1

    def _measure(self, world: "carla.World") -> Optional[tuple[float, int]]:
        """Return the distance from the entity's body to the area and the
        index of the nearest outline, or ``None`` while it is not there."""
        assert self._entity_name is not None  # noqa: S101
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None
        box = ground_box(entity, entity.get_transform())
        box_x, box_y, box_reach = bounding_circle(box)
        best = math.inf
        nearest = -1
        for i in range(len(self._outlines)):
            cx, cy, radius = self._circles[i]
            # No point of this outline is nearer than this: skip it once a
            # nearer one is known, which is most of a city's crosswalks.
            if math.hypot(cx - box_x, cy - box_y) - radius - box_reach >= best:
                continue
            distance = polygon_distance(box, self._outlines[i])
            if distance < best:
                best = distance
                nearest = i
        if nearest < 0:
            return None
        return (best, nearest)

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Keep the closest the entity came, and return a pass result once the
        distance satisfies the rule."""
        measured = self._measure(world)
        if measured is None:
            return None
        distance, nearest = measured
        closest = self._closest_m
        if closest is None or distance < closest:
            self._closest_m = distance
            self._closest_at_s = elapsed
            self._closest_index = nearest
            closest = distance
        if not self._comparison.satisfied(distance):
            return None
        # Which element was nearest, where the area has more than one.
        nearest_text = ""
        if len(self._outlines) > 1:
            nearest_text = (
                f" (nearest: {self._layers[nearest]} {self._element_ids[nearest]})"
            )
        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' is {distance:.2f} m from"
                f" {self._area.describe()}{nearest_text}, which is"
                f" {self._comparison.rule.text} {self._comparison.value:.2f} m;"
                f" closest so far {closest:.2f} m"
            ),
            elapsed_seconds=elapsed,
        )
