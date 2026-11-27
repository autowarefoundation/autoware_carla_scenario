"""Distance from an entity's box to an area of the map, and the closest it came."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...utils.clearance import (
    _MAX_GRADE,
    actor_vertical_span,
    level_gap,
    on_another_level,
)
from ...utils.polygon import (
    Point,
    bounding_circle,
    edge_value,
    ground_box,
    polygon_distance_at,
    polygon_gap_at,
)
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ComparisonRule, ScalarComparisonRule
from ..map_area import MapArea, area_shapes, surface_regions
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: How an area on another level of road than the entity counts
#: (:class:`EntityAreaDistanceCondition`'s *levels*): measured through the
#: height between them, left out, or measured in plan as if it were not there.
LEVELS_MEASURE = "measure"
LEVELS_IGNORE = "ignore"
LEVELS_PLAN = "plan"
_LEVELS = (LEVELS_MEASURE, LEVELS_IGNORE, LEVELS_PLAN)


class _Outline:
    """One element of an area, in CARLA's world frame: its outline, the
    height of each of its points, which layer it is from (``lanelet``,
    ``polygon``, ``surface``), its id there (a surface region's: its index),
    what to call it, the circle round it ``(x, y, radius)`` that skips it
    when it is far, and the span of its heights ``(low, high)`` that skips it
    when it is a level away."""

    points: list[Point]
    heights: list[float]
    layer: str
    element_id: int
    name: str
    circle: tuple[float, float, float]
    span: tuple[float, float]

    def __init__(
        self,
        points: list[Point],
        heights: list[float],
        layer: str,
        element_id: int,
        name: str = "",
    ) -> None:
        self.points = points
        self.heights = heights
        self.layer = layer
        self.element_id = element_id
        self.name = name if name else f"{layer} {element_id}"
        self.circle = bounding_circle(points)
        self.span = (min(heights), max(heights))


class EntityAreaDistanceCondition(CompositionCondition):
    """Pass condition on how near an entity's body comes to an area of the map,
    keeping the closest it came.

    The area is ground chosen by what it is (:class:`~..map_area.MapArea`):
    the crosswalks, one crosswalk picked on the map, the lanes through a
    junction, Autoware's intersection areas -- or the parts of the roads'
    surface raised or grooved past a threshold, read from the OpenCRG grids
    the OpenDRIVE map refers to and placed through each road's reference
    line.  The distance is from the entity's bounding box -- its body, not
    its origin -- to the nearest outline of the area: ``0`` once the body
    touches or is over it.  So "the ego stops short of the crosswalk" is ``GREATER_THAN 0`` held to
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

    Where roads are stacked -- an elevated road over a street, a ramp over an
    underpass -- the measurement is level-aware (#2314), as the
    ``min_distance_to_obstacle_m`` metric and
    :class:`~..place_clear.PlaceClearCondition` are.  Every point of an
    outline has its height in CARLA's frame (a lanelet's or polygon's ``z``
    less :attr:`~...coordinate.map_manager.MapManager.z_offset`; a surface
    region's, the road's elevation there plus the grid's value), and the
    area's height where the distance is read -- at the nearest point of the
    outline, or under the middle of the body when it is over the area -- is
    interpolated along the edge it is on.  The area there is on another level
    than the entity (:func:`~...utils.clearance.on_another_level`) when it is
    clear of the entity's box by more than a metre in height
    (:func:`~...utils.clearance.level_gap`) and further up or down from where
    the entity stands than a road's grade (10 %) climbs between the middle of
    its body and that point -- so a crosswalk on the same street further
    down a hill, or the far edge of an intersection on a slope the entity is
    in, is on the entity's level.  Such an area counts as *levels* says:

    * ``"measure"`` (the default): its distance is ``hypot(plan, height)``,
      the height being how far it is clear of the box.  An area on the
      entity's level is as near as in the plan, so on a map without stacked
      roads the distance is the plan's; a crosswalk 6 m under the ego on a
      bridge is some 6 m away, not ``0``.
    * ``"ignore"``: it is left out, and the rest measured in plan.  With all
      of the area on another level, nothing is measured and the condition
      does not hold.
    * ``"plan"``: in plan, as a map's plan would draw it: an area on the road
      above or below is as near as it is in the plan.

    Args:
        entity_name: The ``role_name`` of the actor whose body is measured.
        area: The ground to measure to.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.
        levels: How an area on another level of road counts: ``"measure"``,
            ``"ignore"`` or ``"plan"`` (above).

    Raises:
        ValueError: If *tolerance* is negative, *levels* is none of the
            three, or the area is not one any map could hold
            (:meth:`~..map_area.MapArea.validate`).
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
    # How an area on another level counts (_LEVELS).
    _levels: str

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        area: MapArea,
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        tolerance: float = 1e-6,
        *,
        label: str,
        levels: str = LEVELS_MEASURE,
    ) -> None:
        super().__init__(entity_name=entity_name, label=label)
        # First, so a bad tolerance, area or levels is refused before the map
        # is read.
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )
        area.validate()
        if levels not in _LEVELS:
            raise ValueError(
                f"levels must be one of {', '.join(_LEVELS)}, not {levels!r}"
            )
        self._levels = levels
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
        from ...coordinate.transform import map_xy_to_carla  # noqa: PLC0415

        manager = MapManager.get_instance()
        if self._area.ground.on_surface:
            self._resolve_surface(manager.xodr_path)
            return
        offset = manager.mgrs_offset
        # Lanelet2 stores absolute elevation; CARLA's is the OpenDRIVE map's.
        z_offset = manager.z_offset
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
                points.append(map_xy_to_carla(x, y, offset))
            heights: list[float] = []
            for z in shape.heights:
                heights.append(z - z_offset)
            if points:
                self._outlines.append(
                    _Outline(points, heights, shape.layer, shape.element_id)
                )
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
        except (OSError, ValueError) as exc:
            # What relief_regions documents raising: said, as a map that
            # holds none of the area is, rather than raised out of setup()
            # (#2598).  A <CRG> it cannot read is in *skipped*.
            self._unheld(str(exc))
            return
        for reason in skipped:
            logger.warning("%s: an OpenCRG grid is passed over: %s", self.label, reason)
        for index, region in enumerate(regions):
            points: list[Point] = []
            heights: list[float] = []
            try:
                for k in range(len(region.outline)):
                    s, t = region.outline[k]
                    placed = to_carla_world(OpenDrivePose(region.road_id, 0, s, t))
                    points.append((float(placed.x), float(placed.y)))
                    # The road's height there, and how far the grid stands
                    # off it.
                    heights.append(float(placed.z) + region.relief[k])
            except KeyError:
                logger.warning(
                    "%s: road %s of an OpenCRG grid is not on the OpenDRIVE map",
                    self.label,
                    region.road_id,
                )
                continue
            if points:
                self._outlines.append(
                    _Outline(points, heights, "surface", index, region.describe())
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
    def levels(self) -> str:
        """How an area on another level of road counts: ``"measure"``,
        ``"ignore"`` or ``"plan"``."""
        return self._levels

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
        index of the nearest outline, or ``None`` while it is not there (or,
        with *levels* ``"ignore"``, while all of the area is on another
        level).

        Outlines are visited nearest first by their bounding circles, so the
        loop stops at the first whose circle is no nearer than the best
        found -- most of a city's crosswalks are never measured; one whose
        circle is clear of the body's cannot overlap it, so only its gap is
        measured.  Measuring through the levels, the bound of an outline
        whose circle is clear of the body's is the circles' combined with how
        far the outline's span of heights is from the body's, so an area a
        level away is skipped as one far in plan is."""
        assert self._entity_name is not None  # noqa: S101
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None
        transform = entity.get_transform()
        box = ground_box(entity, transform)
        box_x, box_y, box_reach = bounding_circle(box)
        body = actor_vertical_span(entity, transform)
        measure = self._levels == LEVELS_MEASURE
        planar_bounds: list[float] = []
        bounds: list[tuple[float, int]] = []
        for i in range(len(self._outlines)):
            outline = self._outlines[i]
            cx, cy, radius = outline.circle
            planar_bound = math.hypot(cx - box_x, cy - box_y) - radius - box_reach
            planar_bounds.append(planar_bound)
            bound = planar_bound
            # Only for an outline that cannot overlap the body: one that can
            # may be on the entity's level however far its edges climb from
            # where the body stands (_outline_distance), so it keeps its plan
            # bound.
            if measure and planar_bound > 0.0:
                apart = level_gap(body, outline.span)
                if apart > 0.0:
                    # Through the height, at least the span's gap; in plan --
                    # the height within a road's grade of how far the point
                    # read is from the middle of the body, which is at most
                    # the plan distance and the body's reach -- at least as
                    # far as that grade climbs the gap, less the reach.
                    bound = min(
                        math.hypot(planar_bound, apart),
                        max(planar_bound, apart / _MAX_GRADE - box_reach),
                    )
            bounds.append((bound, i))
        bounds.sort()
        best = math.inf
        nearest = -1
        for bound, i in bounds:
            if bound >= best:
                break
            distance = self._outline_distance(
                self._outlines[i], box, body, planar_bounds[i] > 0.0
            )
            if distance < best:
                best = distance
                nearest = i
        if nearest < 0:
            return None
        return (best, nearest)

    def _outline_distance(
        self,
        outline: _Outline,
        box: list[Point],
        body: tuple[float, float],
        apart: bool,
    ) -> float:
        """The distance from the body's outline *box* (its vertical span
        *body*) to *outline*, as *levels* counts another level: ``inf`` for
        one left out.  *apart*: the two cannot overlap (their circles are
        clear), so only the gap is measured.

        Whether the area is on the entity's level is reckoned from the middle
        of the body, where it stands, to the point of the outline its height
        is read at: inside a wide area -- an intersection on a hill -- that
        point is the edge nearest the middle, and the road climbs or falls
        between them as it does between the body and an area beside it."""
        box_x, box_y, _reach = bounding_circle(box)
        planar = 0.0
        edge = -1
        along = 0.0
        if apart:
            planar, edge, along = polygon_gap_at(box, outline.points)
        else:
            planar, edge, along = polygon_distance_at(
                box, outline.points, (box_x, box_y)
            )
        if self._levels == LEVELS_PLAN or edge < 0:
            return planar
        height = edge_value(outline.heights, edge, along)
        points = outline.points
        start = points[edge]
        end = points[(edge + 1) % len(points)]
        x = start[0] + (end[0] - start[0]) * along
        y = start[1] + (end[1] - start[1]) * along
        if not on_another_level(
            body, (height, height), math.hypot(x - box_x, y - box_y)
        ):
            return planar
        if self._levels == LEVELS_IGNORE:
            return math.inf
        return math.hypot(planar, level_gap(body, (height, height)))

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
