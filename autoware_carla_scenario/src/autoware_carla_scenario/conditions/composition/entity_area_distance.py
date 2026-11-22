"""Distance from an entity's box to an area of the map, and the closest it came."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...utils.polygon import (
    Point,
    bounding_circle,
    ground_box,
    polygon_distance,
    polygon_gap,
)
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ComparisonRule, ScalarComparisonRule
from ..map_area import MapArea, area_shapes, surface_regions
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class _Outline:
    """One element of an area, in CARLA's world frame: its outline, which
    layer it is from (``lanelet``, ``polygon``, ``surface``), its id there (a
    surface region's: its index), what to call it, and the circle round it
    ``(x, y, radius)`` that skips it when it is far."""

    points: list[Point]
    layer: str
    element_id: int
    name: str
    circle: tuple[float, float, float]

    def __init__(
        self, points: list[Point], layer: str, element_id: int, name: str = ""
    ) -> None:
        self.points = points
        self.layer = layer
        self.element_id = element_id
        self.name = name if name else f"{layer} {element_id}"
        self.circle = bounding_circle(points)


class EntityAreaDistanceCondition(CompositionCondition):
    """Pass condition on how near an entity's body comes to an area of the map,
    keeping the closest it came.

    The area is ground chosen by what it is (:class:`~..map_area.MapArea`):
    the crosswalks, one crosswalk picked on the map, the lanes through a
    junction, Autoware's intersection areas -- or the parts of the roads'
    surface raised or grooved past a threshold, read from the OpenCRG grids
    the OpenDRIVE map refers to and placed through each road's reference
    line.  The distance is on the ground,
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
    # The area's outlines in CARLA's world frame.
    _outlines: list[_Outline]
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
        self._problem = ""
        self._resolve()
        self._closest_m = None
        self._closest_at_s = None
        self._closest_index = -1

    def _resolve(self) -> None:
        """Read the area's outlines off the run's map, into CARLA's frame."""
        from ...coordinate.map_manager import MapManager  # noqa: PLC0415

        manager = MapManager.get_instance()
        if self._area.ground.on_surface:
            self._resolve_surface(manager.xodr_path)
            return
        offset_x, offset_y = manager.mgrs_offset
        try:
            shapes = area_shapes(manager.lanelet_map, self._area)
        except ValueError as exc:
            self._unheld(str(exc))
            return
        for shape in shapes:
            # Lanelet2's frame is MGRS and right-handed; CARLA's is the
            # OpenDRIVE origin's and left-handed (coordinate.transform).
            points: list[Point] = []
            for x, y in shape.points:
                points.append((x - offset_x, -(y - offset_y)))
            if points:
                self._outlines.append(_Outline(points, shape.layer, shape.element_id))
        if not self._outlines:
            self._unheld(f"no {self._area.describe()} on the map")

    def _resolve_surface(self, xodr_path: Path) -> None:
        """Read the raised or grooved parts of the roads' surface off the
        OpenDRIVE map's OpenCRG grids, into CARLA's frame through each road's
        reference line."""
        from ...coordinate.poses import OpenDrivePose  # noqa: PLC0415
        from ...coordinate.transform import to_carla_world  # noqa: PLC0415

        try:
            regions, skipped = surface_regions(xodr_path, self._area)
        except OSError as exc:
            self._unheld(str(exc))
            return
        for reason in skipped:
            logger.warning("%s: an OpenCRG grid is passed over: %s", self.label, reason)
        for index, region in enumerate(regions):
            points: list[Point] = []
            try:
                for s, t in region.outline:
                    placed = to_carla_world(OpenDrivePose(region.road_id, 0, s, t))
                    points.append((float(placed.x), float(placed.y)))
            except KeyError:
                logger.warning(
                    "%s: road %s of an OpenCRG grid is not on the OpenDRIVE map",
                    self.label,
                    region.road_id,
                )
                continue
            if points:
                self._outlines.append(
                    _Outline(points, "surface", index, region.describe())
                )
        if not self._outlines:
            problem = f"no {self._area.describe()} on the map"
            if skipped:
                problem += f" ({skipped[0]})"
            self._unheld(problem)

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
            nearest = self._outlines[self._closest_index]
            details["closest_m"] = closest
            details["closest_at_s"] = self._closest_at_s
            details["closest"] = nearest.name
            if nearest.layer != "surface":
                details[f"closest_{nearest.layer}"] = nearest.element_id
        return details

    def reset(self) -> None:
        """Forget the closest the entity came (:meth:`CompositionCondition.reset`)."""
        super().reset()
        self._closest_m = None
        self._closest_at_s = None
        self._closest_index = -1

    def _measure(self, world: "carla.World") -> Optional[tuple[float, int]]:
        """Return the distance from the entity's body to the area and the
        index of the nearest outline, or ``None`` while it is not there.

        Outlines are visited nearest first by their bounding circles, so the
        loop stops at the first whose circle is no nearer than the best
        found -- most of a city's crosswalks are never measured; one whose
        circle is clear of the body's cannot overlap it, so only its gap is
        measured."""
        assert self._entity_name is not None  # noqa: S101
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None
        box = ground_box(entity, entity.get_transform())
        box_x, box_y, box_reach = bounding_circle(box)
        bounds: list[tuple[float, int]] = []
        for i in range(len(self._outlines)):
            cx, cy, radius = self._outlines[i].circle
            bounds.append((math.hypot(cx - box_x, cy - box_y) - radius - box_reach, i))
        bounds.sort()
        best = math.inf
        nearest = -1
        for bound, i in bounds:
            if bound >= best:
                break
            points = self._outlines[i].points
            distance = 0.0
            if bound > 0.0:
                distance = polygon_gap(box, points)
            else:
                distance = polygon_distance(box, points)
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
            nearest_text = f" (nearest: {self._outlines[nearest].name})"
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
