"""Time-to-collision (TTC) condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_in_list
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition

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
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=source, label=label)
        self._target = target
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
        source = find_actor_in_list(actors, self._entity_name)
        target = find_actor_in_list(actors, self._target)
        if source is None or target is None:
            return None

        src_loc = source.get_location()
        tgt_loc = target.get_location()
        offset = Vector3(tgt_loc.x - src_loc.x, tgt_loc.y - src_loc.y, 0.0)
        distance = offset.magnitude()
        if distance < _CLOSING_SPEED_EPSILON:
            return None
        direction = offset / distance

        src_vel = Vector3.from_carla_vector3d(source.get_velocity())
        tgt_vel = Vector3.from_carla_vector3d(target.get_velocity())
        relative = Vector3(src_vel.x - tgt_vel.x, src_vel.y - tgt_vel.y, 0.0)
        closing_speed = relative.dot(direction)
        if closing_speed <= _CLOSING_SPEED_EPSILON:
            # Receding (or holding station): the TTC is unbounded.
            return None

        return distance / closing_speed

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the TTC satisfies the comparison rule."""
        actors: list[carla.Actor] = world.get_actors()
        ttc = self._measure(actors)
        if ttc is None:
            return None

        if not self._comparison.satisfied(ttc):
            return None

        rule_text = self._comparison.rule.name.lower().replace("_", " ")
        return ScenarioResult(
            passed=True,
            message=(
                f"TTC '{self._entity_name}' -> '{self._target}'"
                f" ({ttc:.2f} s) {rule_text}"
                f" {self._comparison.value:.2f} s at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
