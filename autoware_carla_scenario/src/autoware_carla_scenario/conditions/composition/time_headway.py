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

    Headway is the distance *ahead* divided by the source's own speed::

        ahead   = (p_target - p_source) . unit(v_source)
        headway = ahead / |v_source|

    The distance is the component along the direction the follower is
    travelling, not the straight-line range, and the difference is the whole
    correctness of the measure: an unsigned range gives a car 10 m *behind* a
    follower doing 10 m/s a headway of 1 s, so a "less than 2 s" tailgating
    rule fires the moment the follower passes the vehicle it was following.
    A target that is not ahead has no headway at all, and the condition says
    so by returning nothing.

    Projecting onto the velocity rather than onto the heading is deliberate:
    the follower is known to be moving -- a standstill has no headway either --
    so the direction of travel is always defined, and it is the direction the
    gap is actually closing along.

    .. warning::
        The measurement is in the **entity** coordinate system: a straight-line
        offset projected onto the direction of travel.  OpenSCENARIO's
        ``coordinateSystem: lane`` -- the distance *along the road*, which is
        what `scenario_simulator_v2` measures -- is not implemented; see
        the issue linked from ``docs/architecture.md``.

        On a straight road the two agree.  On a curve the projection is short,
        and the error grows with the curvature: for a leader 20 m ahead along
        the lane it reads 19.5 m on a 50 m radius, 17.9 m on 25 m, and 14.6 m
        on 15 m.

        Past a quarter turn it does worse than under-read: the projection goes
        negative, the target is taken to be *not ahead*, and the condition
        stops firing altogether.  On a roundabout or a tight corner a leader
        directly in front in-lane is invisible to it, silently, for the whole
        manoeuvre.  Use this condition where the road is straight enough for
        the difference not to matter, and read a negative result as "cannot
        tell" rather than as "nothing in front".

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

        Two cases have no answer rather than a large one.  A stationary
        follower has no headway: reporting an unbounded value would make a
        ``less than`` rule quietly false, which reads in a report as "the ego
        is keeping its distance" when the question does not apply.  A target
        that is not ahead has none either, and there the failure is the
        opposite way round -- an unsigned range would report a small number
        and fire a tailgating rule for a vehicle the follower has already
        overtaken.
        """
        assert self._entity_name is not None  # noqa: S101
        source, target = find_actor_pair(actors, self._entity_name, self._target)
        if source is None or target is None:
            return None

        src_loc = source.get_location()
        tgt_loc = target.get_location()
        offset = Vector3(tgt_loc.x - src_loc.x, tgt_loc.y - src_loc.y, 0.0)

        velocity = Vector3.from_carla_vector3d(source.get_velocity())
        travel = Vector3(velocity.x, velocity.y, 0.0)
        speed = travel.magnitude()
        if speed < _SPEED_EPSILON:
            return None

        ahead = offset.dot(travel / speed)
        if ahead <= _SPEED_EPSILON:
            # Behind, or exactly abeam: there is no gap in front to close.
            #
            # On a curve sharper than a quarter turn this is also reached by a
            # leader that *is* ahead along the lane, because the straight-line
            # projection has gone negative by then.  That is the cost of
            # measuring in the entity frame; see the class docstring.
            return None

        return ahead / speed

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
