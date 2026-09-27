"""Base class for composition conditions."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Union

from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import BaseCondition, ScenarioResult
from ..entity_existence import EntityExistenceCondition

if TYPE_CHECKING:
    import carla

_NEAR_ZERO_THRESHOLD = 1e-12
"""Magnitude below which a forward vector is considered degenerate."""


class DistanceCoordinateSystem(Enum):
    """Which frame a distance between two entities is measured in.

    OpenSCENARIO carries this on ``RelativeDistanceCondition``,
    ``TimeHeadwayCondition`` and ``TimeToCollisionCondition`` alike, and the
    two values are genuinely different measurements rather than two spellings
    of one.

    Attributes:
        ENTITY: A straight-line offset in the world frame, projected onto a
            direction belonging to the subject.  The default, so a document
            written before this existed keeps its meaning.
        LANE: The distance *along the road*, which is the gap a driver would
            describe.  Follows the chain of connected roads between the two,
            and has no answer once a junction stands between them -- see
            :mod:`autoware_carla_scenario.coordinate.lane_distance`.  Needs a
            loaded map.

    The two agree on a straight road and part company on a curve, where the
    entity frame reads short: for a leader 20 m ahead along the lane it gives
    19.5 m at a 50 m radius and 14.6 m at 15 m.  Past a quarter turn the
    projection changes sign, and a condition that only looks ahead stops firing
    for a leader that is directly in front.
    """

    ENTITY = auto()
    LANE = auto()


def entity_axes(actor: "carla.Actor") -> Optional[tuple[Vector3, Vector3]]:
    """Return *actor*'s ``(forward, left)`` unit vectors in the ground plane.

    Both are flattened to z = 0: a condition that decomposes a velocity or an
    acceleration into longitudinal and lateral parts is asking about the road,
    and a pitched vehicle on a slope should not leak that pitch into either
    component.

    Left is derived from forward rather than read off the transform, so the two
    axes cannot disagree.  CARLA's world frame is left-handed, so rotating
    ``(fx, fy)`` by 90 degrees gives ``(fy, -fx)``.

    Returns:
        The pair, or ``None`` when the forward vector is degenerate -- which a
        caller must report as "cannot evaluate" rather than as a zero
        component, because the two mean different things to an assertion.
    """
    carla_forward: carla.Vector3D = actor.get_transform().get_forward_vector()
    forward = Vector3(carla_forward.x, carla_forward.y, 0.0)
    magnitude = forward.magnitude()
    if magnitude < _NEAR_ZERO_THRESHOLD:
        return None
    forward_unit = forward / magnitude
    return forward_unit, Vector3(forward_unit.y, -forward_unit.x, 0.0)


class CompositionCondition(BaseCondition):
    """Base class for conditions composed from other conditions.

    :meth:`check` evaluates guards and prerequisites in order:

    1. **Entity existence guard** — when *entity_existence* is provided, the
       entity must be present in the world.  If the entity is absent the
       condition short-circuits and returns ``None``.
    2. **Child condition** — when *child* is provided it must have fired
       (returned ``ScenarioResult(passed=True)``).  If the child has not yet
       fired the condition short-circuits and returns ``None``.
    3. **Subclass logic** — :meth:`_check` is called only after both guards
       pass.

    Leaf composition conditions (e.g. :class:`SpeedCondition`) that have no
    child or entity guard pass ``None`` for both and implement all logic in
    :meth:`_check`.

    Args:
        child: An optional child condition that must be satisfied before
            :meth:`_check` is evaluated.  Typically an
            :class:`AndCondition` or :class:`OrCondition` combining
            multiple sub-conditions.
        entity_name: When provided, an :class:`EntityExistenceCondition`
            is constructed internally and used as a guard.  :meth:`check`
            returns ``None`` immediately if the entity is absent.
            Accepts both :class:`EntityRole` and plain ``str``.
    """

    def __init__(
        self,
        child: BaseCondition | None = None,
        entity_name: Union[EntityRole, str] | None = None,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        self._child = child
        self._entity_name = entity_name
        self._entity_existence: EntityExistenceCondition | None = (
            EntityExistenceCondition(entity_name, label=f"{label}_entity_exists")
            if entity_name is not None
            else None
        )

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {}
        if self._entity_name is not None:
            details["entity_name"] = str(self._entity_name)
        if self._child is not None:
            details["child"] = self._child.to_summary_dict()
        return details

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Evaluate entity existence, then child, then delegate to :meth:`_check`.

        Returns ``None`` immediately when the entity is absent or the
        child exists but has not yet fired.
        """
        if self._entity_existence is not None:
            existence_result = self._entity_existence.check(world, elapsed)
            if existence_result is not None:
                # Entity is absent — cannot evaluate further.
                return None

        if self._child is not None:
            result = self._child.check(world, elapsed)
            if result is None or not result.passed:
                return None

        return self._check(world, elapsed)

    @abstractmethod
    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Subclass-specific check logic, called after all guards pass."""
        ...
