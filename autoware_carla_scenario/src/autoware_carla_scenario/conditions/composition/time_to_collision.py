"""Time-to-collision (TTC) condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ...coordinate.lane_distance import lane_closing_speed, lane_gap
from ...coordinate.poses import CarlaWorldPose
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition, DistanceCoordinateSystem

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

    It reads as ``source -> target | TTC | <rule> <value> s``.

    Args:
        source: ``role_name`` of the entity the TTC is measured *from*.
        target: ``role_name`` of the entity the TTC is measured *to*.
        value: Threshold time in seconds.
        rule: Comparison operator applied to ``ttc`` vs *value*.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative.
    """

    def __init__(
        self,
        source: Union[EntityRole, str],
        target: Union[EntityRole, str],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        tolerance: float = 1e-6,
        coordinate_system: DistanceCoordinateSystem = (DistanceCoordinateSystem.ENTITY),
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=source, label=label)
        self._target = target
        self._coordinate_system = coordinate_system
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
                "target": str(self._target),
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
                "coordinate_system": self._coordinate_system.name,
            }
        )
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
        source, target = find_actor_pair(actors, self._entity_name, self._target)
        if source is None or target is None:
            return None

        src_loc = source.get_location()
        tgt_loc = target.get_location()
        src_vel = Vector3.from_carla_vector3d(source.get_velocity())
        tgt_vel = Vector3.from_carla_vector3d(target.get_velocity())

        if self._coordinate_system is DistanceCoordinateSystem.LANE:
            return self._lane_ttc(src_loc, tgt_loc, src_vel, tgt_vel)

        offset = Vector3(tgt_loc.x - src_loc.x, tgt_loc.y - src_loc.y, 0.0)
        distance = offset.magnitude()
        if distance < _CLOSING_SPEED_EPSILON:
            return None
        direction = offset / distance

        relative = Vector3(src_vel.x - tgt_vel.x, src_vel.y - tgt_vel.y, 0.0)
        closing_speed = relative.dot(direction)
        if closing_speed <= _CLOSING_SPEED_EPSILON:
            # Receding (or holding station): the TTC is unbounded.
            return None

        return distance / closing_speed

    def _lane_ttc(
        self,
        src_loc: Any,
        tgt_loc: Any,
        src_vel: Vector3,
        tgt_vel: Vector3,
    ) -> Optional[float]:
        """Return the TTC measured along the road, or ``None``.

        Both halves change frame together, and that is the point.  Reusing the
        straight-line closing speed under a lane-measured gap would count a
        vehicle's cornering as approach and make the TTC short for a reason
        that has nothing to do with the road -- the same silent swap of one
        measure for another that the lane frame exists to remove.

        The target must be *ahead* along the road.  In the entity frame a
        receding pair is excluded by the closing speed alone; here a target
        behind the source is excluded outright, because a positive closing
        speed towards something behind is a vehicle reversing into it, not a
        collision this condition is asked about.
        """
        source_pose = CarlaWorldPose(x=src_loc.x, y=src_loc.y, z=src_loc.z, yaw=0.0)
        target_pose = CarlaWorldPose(x=tgt_loc.x, y=tgt_loc.y, z=tgt_loc.z, yaw=0.0)

        gap = lane_gap(source_pose, src_vel.x, src_vel.y, target_pose)
        if gap is None or gap <= _CLOSING_SPEED_EPSILON:
            return None

        closing_speed = lane_closing_speed(
            source_pose,
            src_vel.x,
            src_vel.y,
            target_pose,
            tgt_vel.x,
            tgt_vel.y,
        )
        if closing_speed is None or closing_speed <= _CLOSING_SPEED_EPSILON:
            return None

        return gap / closing_speed

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
                f"TTC '{self._entity_name}' -> '{self._target}'"
                f" ({ttc:.2f} s) {rule_text}"
                f" {self._comparison.value:.2f} s at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
