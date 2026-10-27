"""Pass condition that holds while a place on the map is clear of road users."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..coordinate.poses import Lanelet2Pose
from .base import BaseCondition, ScenarioResult

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = ["PlaceClearCondition"]


class PlaceClearCondition(BaseCondition):
    """Pass condition that holds while no vehicle's or walker's bounding box
    comes within *margin* of a rectangle *length* by *width* at *position*.

    The rectangle faces along the lane at *position*, turned by its
    ``heading``.  An actor parked under the map (spawned out of sight) does
    not count; one put on the place this very tick, which the world's snapshot
    does not show yet, does.  Typically an entity's appearance spot, so that it
    is not brought in on top of another road user.

    Args:
        position: The place's centre, on a lanelet.
        length: Its length (m), along its heading.
        width: Its width (m).
        margin: How near (m) another's outline may come.
    """

    # Declared for the static check (docs/typecheck.md).
    _position: Lanelet2Pose
    _length: float
    _width: float
    _margin: float
    _spot: Optional[tuple[float, float, float]]
    _said_blocker: int

    def __init__(
        self,
        position: Lanelet2Pose,
        length: float,
        width: float,
        margin: float = 0.3,
        *,
        label: str,
    ) -> None:
        super().__init__(label=label)
        if length <= 0.0 or width <= 0.0:
            raise ValueError("length and width must be positive")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        self._position = position
        self._length = float(length)
        self._width = float(width)
        self._margin = float(margin)
        self._spot = None
        self._said_blocker = -1

    def get_details(self) -> dict[str, Any]:
        return {
            "lanelet_id": int(self._position.lanelet_id),
            "s": float(self._position.s),
            "t": float(self._position.t),
            "length": self._length,
            "width": self._width,
            "margin": self._margin,
        }

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result while the place is clear.

        Args:
            world: The CARLA world instance.
            elapsed: Simulated seconds since the run began.

        Returns:
            ScenarioResult with passed=True while nothing stands there,
            ``None`` while something does (its actor said once).
        """
        spot = self._spot
        if spot is None:
            from ..coordinate.transform import to_carla_world  # noqa: PLC0415

            pose = to_carla_world(self._position)
            spot = (pose.x, pose.y, pose.yaw)
            self._spot = spot
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
