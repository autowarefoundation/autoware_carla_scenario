"""Distance from an entity to a fixed place on the map."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from ...coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from ...coordinate.transform import to_carla_location
from ...trajectory.model import (
    MapPose,
    RelativeLanePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    TrajectoryVertex,
)
from ...entity_role import EntityRole
from ...kinematics import Vector3
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ComparisonRule, ScalarComparisonRule
from .base import CompositionCondition

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


class EntityPositionDistanceCondition(CompositionCondition):
    """Pass condition on the distance from an entity to a place on the map.

    The sibling of :class:`EntityDistanceCondition`, which measures between two
    *entities*.  This one measures to somewhere that does not move: a stop
    line, a conflict point, the mouth of a junction.  Both readings are wanted
    and neither substitutes for the other, which is why this is a second
    condition rather than a target that may be either.

    Like the other distance conditions the measurement is horizontal by
    default: two points on a slope are not far apart because of the height
    between them.

    .. note::
        The *position* is resolved to a CARLA world location **in this
        constructor**, and resolving a Lanelet2 pose reads the map, so
        :class:`~autoware_carla_scenario.coordinate.map_manager.MapManager`
        must be initialised first.  Conditions are built inside a scenario's
        ``setup()``, against a live world, so that holds there; a
        :class:`CarlaWorldPose` needs no map and can be built anywhere.

        Resolving once rather than on every tick is not only cheaper: it means
        a lanelet that does not exist on the map fails while the scenario is
        being built, rather than silently never firing during the run.

        Any other position a trajectory vertex takes -- a route pose, placed on
        the scenario's route -- is resolved when it is first measured to; one
        that cannot be placed holds nothing, said once.

    Args:
        entity_name: The ``role_name`` attribute of the actor to measure from.
        position: Where to measure to, as a Lanelet2, OpenDRIVE or CARLA world
            pose, or any position a trajectory vertex takes (a route pose,
            say).  Unlike :class:`EntityLanePositionCondition`, which uses only
            the lane a pose names, this uses the point: ``s`` along the lanelet
            is part of the answer.
        value: Threshold distance in metres.
        rule: Comparison operator applied to ``distance`` vs *value*.
        vertical: Include the height difference in the measurement.  Off by
            default.
        tolerance: Tolerance for :attr:`ComparisonRule.EQUAL_TO`.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *tolerance* is negative.
    """

    # Declared for the static check (docs/typecheck.md).
    # The position, held as a vertex: Codon types an attribute of a union
    # through it (docs/typecheck.md).
    _at: TrajectoryVertex
    _target: Optional[Vector3]
    _unplaced: bool
    _vertical: bool
    _comparison: ScalarComparisonRule

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        position: Union[
            CarlaWorldPose,
            Lanelet2Pose,
            OpenDrivePose,
            MapPose,
            RelativeLanePose,
            RouteLanePose,
            RouteOppositePose,
            RouteCrossingPose,
            RouteCrosswalkPose,
            RouteRoadsidePose,
        ],
        value: float,
        rule: ComparisonRule = ComparisonRule.LESS_THAN,
        vertical: bool = False,
        tolerance: float = 1e-6,
        *,
        label: str,
    ) -> None:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        super().__init__(entity_name=entity_name, label=label)
        self._at = TrajectoryVertex(position)
        self._target = None
        self._unplaced = False
        # A map pose now (a lanelet the map lacks fails here); any other
        # when it is first measured to.  Named as the class: Codon does not
        # narrow a union by isinstance() (docs/typecheck.md).
        held = self._at.position
        if isinstance(held, Lanelet2Pose):
            lane: Lanelet2Pose = held
            self._target = self._located(to_carla_location(lane))
        elif isinstance(held, OpenDrivePose):
            road: OpenDrivePose = held
            self._target = self._located(to_carla_location(road))
        elif isinstance(held, CarlaWorldPose):
            world_pose: CarlaWorldPose = held
            self._target = self._located(to_carla_location(world_pose))
        self._vertical = vertical
        self._comparison = ScalarComparisonRule(
            field="distance", rule=rule, value=value, tolerance=tolerance
        )

    @staticmethod
    def _located(location: "carla.Location") -> Vector3:
        return Vector3(float(location.x), float(location.y), float(location.z))

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        target = self._target
        if target is not None:
            details.update(
                {"target_x": target.x, "target_y": target.y, "target_z": target.z}
            )
        else:
            details["position"] = type(self._at.position).__name__
        details.update(
            {
                "vertical": self._vertical,
                "value": self._comparison.value,
                "rule": self._comparison.rule.name,
            }
        )
        return details

    def _place(self) -> Optional[Vector3]:
        """The place in CARLA's world, placed the first time it is asked for;
        ``None`` (said once) when it cannot be."""
        target = self._target
        if target is not None or self._unplaced:
            return target
        from ...trajectory.resolve import resolve_position  # noqa: PLC0415

        try:
            x, y, z, _yaw = resolve_position(self._at.position)
        except ValueError as exc:
            logger.warning("%s: no place to measure to: %s", self.label, exc)
            self._unplaced = True
            return None
        # Branched: Codon types no conditional expression of a None and a float.
        height = 0.0
        if z is not None:
            height = float(z)
        placed = Vector3(float(x), float(y), height)
        self._target = placed
        return placed

    def _measure(self, world: "carla.World") -> Optional[float]:
        """Return the distance from the entity to the place, or ``None``."""
        assert self._entity_name is not None  # noqa: S101
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None
        target = self._place()
        if target is None:
            return None

        location = entity.get_location()
        dz = (target.z - location.z) if self._vertical else 0.0
        offset = Vector3(
            target.x - location.x,
            target.y - location.y,
            dz,
        )
        return offset.magnitude()

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result once the distance satisfies the rule."""
        distance = self._measure(world)
        if distance is None:
            return None
        if not self._comparison.satisfied(distance):
            return None

        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' is {distance:.2f} m from the"
                f" position, which is {self._comparison.rule.text}"
                f" {self._comparison.value:.2f} m"
            ),
            elapsed_seconds=elapsed,
        )
