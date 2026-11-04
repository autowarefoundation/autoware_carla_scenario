"""Pass condition that holds while a place on the map is clear of road users."""

from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from ..trajectory.model import (
    MapPose,
    RelativeLanePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    TrajectoryVertex,
    placed_against_entity,
)
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["PlaceClearCondition"]


class PlaceClearCondition(BaseCondition):
    """Pass condition that holds while no vehicle's or walker's bounding box
    comes within *margin* of a rectangle *length* by *width* at *position*.

    The rectangle faces along the lane at a Lanelet2 *position*, turned by
    its ``heading``; any other position faces the way it resolves to (a
    route pose is placed on the scenario's route when the place is first
    looked at).  A place written against an entity -- a
    :class:`~..trajectory.model.RelativeLanePose`, against its
    ``entity_ref``; a route pose without an anchor, against the ego -- moves
    with it, so it is placed afresh on every tick, facing along the lane it
    is placed on where it states no ``yaw``; while that entity is not in
    the scene (not spawned yet, gone, or parked under the map) the place is
    nowhere and the condition does not hold.  An actor parked under the map
    (spawned out of sight) does not count; one put on the place this very
    tick, which the world's snapshot does not show yet, does.  Typically an
    entity's appearance spot, so that it is not brought in on top of another
    road user.  Other road users'
    outlines are their bounding boxes where they stand, ahead of a car's
    origin; the rectangle itself is centred on *position*, no actor's box
    being known to it (a vehicle spawned on *position* puts its origin there,
    its body a metre or so further on).

    Args:
        position: The place's centre: a Lanelet2 pose, or any position a
            trajectory vertex takes (a route pose, or a pose relative to an
            entity, say).
        length: Its length (m), along its heading.
        width: Its width (m).
        margin: How near (m) another's outline may come.
    """

    # Declared for the static check (docs/typecheck.md).
    # The position, held as a vertex: Codon types an attribute of a union
    # through it (docs/typecheck.md).
    _at: TrajectoryVertex
    _length: float
    _width: float
    _margin: float
    _spot: Optional[tuple[float, float, float]]
    # Placed against an entity, afresh on every tick (placed_against_entity).
    _follows: bool
    _unplaced: bool
    _said_unplaced: bool
    _said_blocker: int

    def __init__(
        self,
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
        length: float,
        width: float,
        margin: float = 0.3,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("width must be positive and finite")
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("margin must be non-negative and finite")
        self._at = TrajectoryVertex(position)
        self._length = float(length)
        self._width = float(width)
        self._margin = float(margin)
        self._spot = None
        self._follows = placed_against_entity(position)
        self._unplaced = False
        self._said_unplaced = False
        self._said_blocker = -1

    def reset(self) -> None:
        """Place it afresh next time it is looked at (:meth:`BaseCondition.reset`):
        a route pose is placed on the run's route, which may be another."""
        super().reset()
        self._spot = None
        self._unplaced = False
        self._said_unplaced = False
        self._said_blocker = -1

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "length": self._length,
            "width": self._width,
            "margin": self._margin,
        }
        position = self._at.position
        if isinstance(position, Lanelet2Pose):
            # Named as the class: Codon does not narrow a union by isinstance()
            # (docs/typecheck.md).
            lane: Lanelet2Pose = position
            details["lanelet_id"] = int(lane.lanelet_id)
            details["s"] = float(lane.s)
            details["t"] = float(lane.t)
        else:
            details["position"] = type(position).__name__
        spot = self._spot
        if spot is not None:
            details["x"], details["y"] = spot[0], spot[1]
        return details

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result while the place is clear.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.

        Returns:
            ScenarioResult with passed=True while nothing stands there,
            ``None`` while something does (its actor said once), or while
            the place is nowhere.
        """
        spot = self._spot
        if spot is None or self._follows:
            if self._unplaced:
                return None
            spot = self._placed(world)
            self._spot = spot
            if spot is None:
                return None
        x, y, yaw = spot
        blocker = _occupant(
            world,
            x,
            y,
            yaw,
            self._length / 2.0,
            self._width / 2.0,
            self._margin,
        )
        if blocker >= 0:
            if blocker != self._said_blocker:
                self._said_blocker = blocker
                logger.info(
                    "%s: actor %d stands on the place (%.1f, %.1f) at %.2fs",
                    self.label,
                    blocker,
                    x,
                    y,
                    elapsed,
                )
            return None
        self._said_blocker = -1
        return ScenarioResult(
            passed=True,
            message=f"The place ({x:.1f}, {y:.1f}) is clear at {elapsed:.2f}s",
            elapsed_seconds=elapsed,
        )

    def _placed(self, world: "carla.World") -> Optional[tuple[float, float, float]]:
        """CARLA ``(x, y, yaw)`` of the place now; ``None`` while the entity it
        is written against is not in the scene, or where it cannot be placed
        (said once; a fixed place is not tried again)."""
        from ..route.active import lane_place  # noqa: PLC0415
        from ..trajectory.resolve import (  # noqa: PLC0415
            resolve_along_lane,
            resolve_position,
        )
        from .entity_existence import ReferenceAbsentError, ReferencePoses  # noqa: PLC0415

        reference = ReferencePoses(world, present_only=True)
        position = self._at.position
        try:
            if self._follows:
                # Without a heading of its own, along the lane it is placed
                # on: no lane is looked up for it afterwards (lane_place).
                x0, y0, _z, yaw0 = resolve_along_lane(position, reference.pose)
            else:
                x0, y0, _z, yaw0 = resolve_position(position, reference.pose)
        except ReferenceAbsentError:
            # Waited for, quietly: the place is nowhere until it is there.
            return None
        except ValueError as exc:
            # A fixed place will not resolve on a later tick; one that moves
            # with an entity may, so it is tried again, said once.
            if not self._follows:
                self._unplaced = True
            if not self._said_unplaced:
                self._said_unplaced = True
                logger.warning("%s: no place to look at: %s", self.label, exc)
            return None
        # Branched: Codon types no conditional expression of a None and a
        # float.
        yaw_deg = 0.0
        if yaw0 is not None:
            yaw_deg = float(yaw0)
        else:
            # No heading of its own (a route pose without a yaw): along the
            # lane it is on, where it is on one.
            along = lane_place(position)
            if along is not None:
                _lx, _ly, _lz, lane_yaw = resolve_position(along, reference.pose)
                if lane_yaw is not None:
                    yaw_deg = float(lane_yaw)
        return (x0, y0, yaw_deg)


def _occupant(
    world: "carla.World",
    x: float,
    y: float,
    yaw: float,
    half_length: float,
    half_width: float,
    margin: float,
) -> int:
    """The id of a road user whose outline comes within *margin* of the
    rectangle at CARLA ``(x, y)`` facing *yaw*, or ``-1``."""
    from ..utils.clearance import overlaps  # noqa: PLC0415

    try:
        found = overlaps(world, -1, x, y, yaw, half_length, half_width, margin)
    except (AttributeError, RuntimeError):
        return -1
    return -1 if found is None else int(found)
