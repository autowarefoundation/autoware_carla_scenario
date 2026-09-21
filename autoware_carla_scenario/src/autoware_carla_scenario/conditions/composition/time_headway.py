"""Time-headway condition between two entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_pair
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition

if TYPE_CHECKING:
    import carla

_SPEED_EPSILON = 1e-6
"""Below this speed the headway is undefined rather than very large."""


class TimeHeadwayCondition(CompositionCondition):
    """Pass condition on the time headway from a *source* to a *target* entity.

    Headway is the range divided by the **source's own** speed::

        range   = |p_target - p_source|
        headway = range / |v_source|

    That is the difference from :class:`TimeToCollisionCondition`, and it is
    not a detail.  TTC divides by the *closing* speed, so it is undefined
    whenever the pair is not closing -- two cars holding a steady gap have no
    time to collision at all.  Headway asks a different question: how long
    until the follower reaches where the leader is *now*.  It is the standard
    following-distance measure, and it is defined for any moving follower.

    Both positions are evaluated in the horizontal plane, the convention
    :class:`EntityDistanceCondition` and :class:`TimeToCollisionCondition`
    share.

    Args:
        source: ``role_name`` of the following entity.
        target: ``role_name`` of the entity ahead.
        value: Threshold time in seconds.
        rule: Comparison operator applied to ``headway`` vs *value*.
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
            field="headway", rule=rule, value=value, tolerance=tolerance
        )

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

    def _measure(self, actors: "list[carla.Actor]") -> Optional[float]:
        """Return the headway in seconds, or ``None`` when it is undefined.

        A stationary follower has no headway.  Reporting an infinite value
        instead would make a ``less than`` rule quietly false, which reads in a
        report as "the ego is keeping its distance" when the truth is that the
        question does not apply.
        """
        assert self._entity_name is not None  # noqa: S101
        source, target = find_actor_pair(actors, self._entity_name, self._target)
        if source is None or target is None:
            return None

        src_loc = source.get_location()
        tgt_loc = target.get_location()
        offset = Vector3(tgt_loc.x - src_loc.x, tgt_loc.y - src_loc.y, 0.0)

        velocity = Vector3.from_carla_vector3d(source.get_velocity())
        speed = Vector3(velocity.x, velocity.y, 0.0).magnitude()
        if speed < _SPEED_EPSILON:
            return None

        return offset.magnitude() / speed

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result when the headway satisfies the comparison rule."""
        headway = self._measure(world.get_actors())
        if headway is None:
            return None
        if not self._comparison.satisfied(headway):
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Headway from '{self._entity_name}' to '{self._target}'"
                f" ({headway:.2f} s)"
                f" {self._comparison.rule.text} {self._comparison.value:.2f} s"
            ),
            elapsed_seconds=elapsed,
        )
