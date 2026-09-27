"""Relative-distance condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...coordinate.lane_distance import lane_separation
from ...coordinate.poses import CarlaWorldPose
from ...entity_role import EntityRole
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, DistanceCoordinateSystem
from .distance_measure import RelativeDistanceType, separation

if TYPE_CHECKING:
    import carla


class EntityDistanceCondition(CompositionCondition):
    """Pass condition on the distance from a *source* entity to a *target* entity.

    By default the measured value is the Euclidean distance between the two
    actors' centres, ignoring the ``z`` component -- which is what a scenario
    author means by "how far apart are these two cars".

    Three things change what is being asked, and all of them change the answer
    at the ranges scenarios care about:

    * *coordinate_system* picks the frame.  The entity frame measures the
      straight line between the two; the lane frame measures along the road
      they share, which on a curve is the longer and more useful number.
    * *distance_type* picks the axis within the entity frame.  A car in the
      next lane is 20 m away in a straight line and 2 m away longitudinally,
      and a following-distance or cut-in scenario means the second.
    * *edge_to_edge* measures between the bounding boxes rather than between
      the centres.  The difference is about a vehicle length, which at a 5 m
      threshold is most of the threshold.

    This is the relational counterpart of
    :class:`~autoware_carla_scenario.conditions.composition.speed.SpeedCondition`:
    it reads as ``source -> target | Distance | <rule> <value> m``.

    Args:
        source: ``role_name`` of the entity the distance is measured *from*.
        target: ``role_name`` of the entity the distance is measured *to*.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        vertical: Include the ``z`` component in the distance when ``True``.
            Only meaningful for :attr:`RelativeDistanceType.EUCLIDEAN` in the
            entity frame.
        distance_type: Which component of the separation to measure.  Defaults
            to :attr:`RelativeDistanceType.EUCLIDEAN`, the previous behaviour.
            A lane-frame distance is along the road by construction, so only
            the euclidean and longitudinal values mean anything there.
        edge_to_edge: Measure between the bounding boxes rather than the
            centres, clamped at zero once they overlap.  Off by default,
            OpenSCENARIO's ``freespace``, and not available in the lane frame.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        coordinate_system: :attr:`~DistanceCoordinateSystem.ENTITY` (default)
            measures the straight line between the two.
            :attr:`~DistanceCoordinateSystem.LANE` measures along the roads
            that connect them, which on a curve is the longer and more useful
            number, and which has no answer once a junction stands between
            them.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative, or if a combination asks for
            two incompatible measurements at once:

            * *vertical* with a directional *distance_type*, or in the lane
              frame.  Longitudinal, lateral and along-the-road are all
              ground-plane, so asking for height as well is a contradiction
              rather than a refinement.
            * a lateral *distance_type* in the lane frame.  A length along the
              road is one-dimensional; there is no across-it to report.
            * *edge_to_edge* in the lane frame.  Bumper-to-bumper along a curve
              needs the boxes projected onto the road, which this does not do,
              and quietly measuring centre to centre instead would make a
              close-quarters scenario pass for the wrong reason.

            Each is refused rather than resolved, because resolving it silently
            would leave the document saying something the run does not do.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        distance_type: RelativeDistanceType = RelativeDistanceType.EUCLIDEAN,
        edge_to_edge: bool = False,
        tolerance: float = 1e-6,
        coordinate_system: DistanceCoordinateSystem = (DistanceCoordinateSystem.ENTITY),
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        in_lane = coordinate_system is DistanceCoordinateSystem.LANE
        if vertical and in_lane:
            raise ValueError(
                "a lane-frame distance is measured along the road and has no "
                "vertical component; drop vertical=True, or measure in the "
                "entity frame"
            )
        if vertical and distance_type is not RelativeDistanceType.EUCLIDEAN:
            raise ValueError(
                "vertical applies to a euclidean distance; "
                f"{distance_type.value} is a ground-plane component"
            )
        if in_lane and distance_type is RelativeDistanceType.LATERAL:
            raise ValueError(
                "a lane-frame distance is measured along the road, so there is "
                "no lateral component to report; measure the lateral offset in "
                "the entity frame"
            )
        if in_lane and edge_to_edge:
            raise ValueError(
                "edge_to_edge is not available in the lane frame: bumper to "
                "bumper along a curve needs the bounding boxes projected onto "
                "the road"
            )
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._vertical = vertical
        self._distance_type = distance_type
        self._edge_to_edge = edge_to_edge
        self._coordinate_system = coordinate_system
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "source": str(self._entity_name),
                "target": str(self._target),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "vertical": self._vertical,
                "distance_type": self._distance_type.name,
                "edge_to_edge": self._edge_to_edge,
                "coordinate_system": self._coordinate_system.name,
            }
        )
        return details

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the source-to-target distance, or ``None`` if unavailable."""
        assert self._entity_name is not None  # noqa: S101
        source, target = find_actor_pair(actors, self._entity_name, self._target)
        if source is None or target is None:
            return None

        if self._coordinate_system is DistanceCoordinateSystem.LANE:
            # Unsigned, so neither entity has to be moving for this to mean
            # something -- unlike a headway, a separation has no direction.
            src_loc = source.get_location()
            tgt_loc = target.get_location()
            return lane_separation(
                CarlaWorldPose(x=src_loc.x, y=src_loc.y, z=src_loc.z, yaw=0.0),
                CarlaWorldPose(x=tgt_loc.x, y=tgt_loc.y, z=tgt_loc.z, yaw=0.0),
            )

        return separation(
            source,
            target,
            distance_type=self._distance_type,
            edge_to_edge=self._edge_to_edge,
            vertical=self._vertical,
        )

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the distance satisfies the comparison rule."""
        actors: list[carla.Actor] = world.get_actors()
        distance = self._measure(actors)
        if distance is None:
            return None

        if not self._comparison.satisfied(distance):
            return None

        measure = (
            "along lane"
            if self._coordinate_system is DistanceCoordinateSystem.LANE
            else self._distance_type.value
        )
        rule_text = self._comparison.rule.text
        return ScenarioResult(
            passed=True,
            message=(
                f"Distance '{self._entity_name}' -> '{self._target}'"
                f" [{measure}]"
                f" ({distance:.2f} m) {rule_text}"
                f" {self._comparison.value:.2f} m at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
