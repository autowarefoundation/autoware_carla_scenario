"""Time-to-collision (TTC) condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...coordinate.poses import AnyPose
from ...coordinate.transform import to_carla_location
from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_in_list, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition
from .distance_measure import half_extent_along

if TYPE_CHECKING:
    import carla

_CLOSING_SPEED_EPSILON = 1e-6
"""Closing speeds below this are treated as "not closing" (infinite TTC)."""


class TimeToCollisionCondition(CompositionCondition):
    """Pass condition on the time-to-collision from a *source* to a *target* entity.

    TTC is the range divided by the closing speed, where the closing speed is
    the component of the relative velocity along the line joining the two
    actors::

        range        = |p_target - p_source|
        closing      = (v_source - v_target) . unit(p_target - p_source)
        ttc          = range / closing

    When the pair is not closing (``closing <= 0``) the TTC is unbounded and
    the condition never fires -- a receding vehicle has no time-to-collision.
    Both positions and velocities are evaluated in the horizontal plane, the
    same convention :class:`EntityDistanceCondition` uses.

    The target may instead be a **place on the map**, which is what an
    intersection scenario is usually about: time to the stop line, time to the
    conflict point.  A place does not move, so the closing speed is simply the
    source's own speed along the line of sight -- the same expression with the
    target's velocity zero, rather than a second code path.

    *edge_to_edge* measures from the source's bounding box rather than from its
    centre.  It matters more here than anywhere else: TTC is compared against a
    handful of seconds, so a vehicle length in the numerator is a large
    fraction of the answer.

    It reads as ``source -> target | TTC | <rule> <value> s``.

    Args:
        source: ``role_name`` of the entity the TTC is measured *from*.
        target: ``role_name`` of the entity the TTC is measured *to*.  Give
            this or *position*, never both.
        position: A place the TTC is measured to, as a Lanelet2, OpenDRIVE or
            CARLA world pose.  Resolved once, in this constructor.
        value: Threshold time in seconds.
        rule: Comparison operator applied to ``ttc`` vs *value*.
        edge_to_edge: Measure from the bounding boxes rather than from the
            centres.  This is OpenSCENARIO's ``freespace``.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative, or if the target is given
            neither way or both ways.  Both would be two answers to one
            question, and neither leaves the condition with nothing to measure
            to -- both are mistakes worth reporting while the scenario is being
            built rather than a condition that never fires during the run.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str, None] = None,
        value: float = 4.0,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        position: Optional[AnyPose] = None,
        edge_to_edge: bool = False,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if (target is None) == (position is None):
            raise ValueError(
                "a time-to-collision condition measures to an entity or to a "
                "position; give exactly one"
            )
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._edge_to_edge = edge_to_edge
        self._place: Optional[Vector3] = None
        if position is not None:
            location = to_carla_location(position)
            self._place = Vector3(location.x, location.y, 0.0)
        self._comparison = ScalarComparisonRule(
            field="ttc", rule=rule, value=value, tolerance=tolerance
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "source": str(self._entity_name),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "edge_to_edge": self._edge_to_edge,
            }
        )
        if self._target is not None:
            details["target"] = str(self._target)
        if self._place is not None:
            details["target_x"] = self._place.x
            details["target_y"] = self._place.y
        return details

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the source-to-target TTC in seconds.

        Returns ``None`` when either actor is missing, when the two are
        co-located (no direction to close along), or when the pair is not
        closing at all.
        """
        assert self._entity_name is not None  # noqa: S101
        if self._target is not None:
            source, target = find_actor_pair(actors, self._entity_name, self._target)
            if target is None:
                return None
        else:
            source, target = find_actor_in_list(actors, self._entity_name), None
        if source is None:
            return None

        src_loc = source.get_location()
        if target is not None:
            tgt = Vector3(target.get_location().x, target.get_location().y, 0.0)
        else:
            assert self._place is not None  # noqa: S101
            tgt = self._place
        offset = Vector3(tgt.x - src_loc.x, tgt.y - src_loc.y, 0.0)
        distance = offset.magnitude()
        if distance < _CLOSING_SPEED_EPSILON:
            return None
        direction = offset / distance

        if self._edge_to_edge:
            distance = max(0.0, distance - half_extent_along(source, direction))
            if target is not None:
                distance = max(0.0, distance - half_extent_along(target, direction))

        src_vel = Vector3.from_carla_vector3d(source.get_velocity())
        if target is not None:
            tgt_vel = Vector3.from_carla_vector3d(target.get_velocity())
        else:
            # A place does not move, so the closing speed is the source's own
            # speed along the line of sight.  Written as a zero velocity rather
            # than as a separate branch: the two are the same expression, and a
            # second one would be a second place for the sign to go wrong.
            tgt_vel = Vector3.zero()
        relative = Vector3(src_vel.x - tgt_vel.x, src_vel.y - tgt_vel.y, 0.0)
        closing_speed = relative.dot(direction)
        if closing_speed <= _CLOSING_SPEED_EPSILON:
            # Receding (or holding station): the TTC is unbounded.
            return None

        return distance / closing_speed

    def _target_name(self) -> str:
        """Return how the target reads in a result message."""
        if self._target is not None:
            return f"'{self._target}'"
        assert self._place is not None  # noqa: S101
        return f"({self._place.x:.1f}, {self._place.y:.1f})"

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the TTC satisfies the comparison rule."""
        actors: list[carla.Actor] = world.get_actors()
        ttc = self._measure(actors)
        if ttc is None:
            return None

        if not self._comparison.satisfied(ttc):
            return None

        rule_text = self._comparison.rule.text
        return ScenarioResult(
            passed=True,
            message=(
                f"TTC '{self._entity_name}' -> {self._target_name()}"
                f" ({ttc:.2f} s) {rule_text}"
                f" {self._comparison.value:.2f} s at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
