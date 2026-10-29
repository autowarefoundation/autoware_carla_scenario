"""Debug markers: what an ego entity knows that the world does not show.

The trajectory its planner is following, for one.  :meth:`EgoVehicle.debug_markers
<autoware_carla_scenario.entity.ego.EgoVehicle.debug_markers>` hands it out
as :class:`DebugMarkers` -- shapes in CARLA world coordinates -- and
:class:`~autoware_carla_scenario.scenario_runner.ScenarioRunner` writes them
into the run's trajectory log (a ``markers`` record whenever they change,
docs/trajectory_log.md), where a viewer of the run (a Rerun recording made from
the log) draws them beside the path the ego drove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from .driver.geometry import Pose, Trajectory

__all__ = [
    "DRIVEN_PATH_COLOR",
    "TRAJECTORY_COLOR",
    "DebugMarkers",
    "DebugPolyline",
    "plan_markers",
    "plan_polyline",
    "world_point",
]

#: An RGB colour, each channel 0-255.
RGB = Tuple[int, int, int]
#: A point in CARLA world coordinates, metres.
WorldPoint = Tuple[float, float, float]

#: The colour a planned trajectory is drawn in.
TRAJECTORY_COLOR: RGB = (0, 255, 0)

#: The colour a viewer draws the path the ego drove in.
DRIVEN_PATH_COLOR: RGB = (255, 128, 0)

#: How far above its points a trajectory is drawn, metres: a plan lies on the
#: road surface, where a line would sink into the asphalt.
_TRAJECTORY_LIFT_M: float = 0.3


@dataclass(frozen=True)
class DebugPolyline:
    """A line through *points*, in CARLA world coordinates.

    Attributes:
        points: The vertices, in order (CARLA world frame, metres).
        color: The line's colour.
        thickness: The line's thickness, metres.
        point_size: The size of a dot drawn at each vertex, metres; none when
            ``0``.
    """

    points: List[WorldPoint]
    color: RGB = TRAJECTORY_COLOR
    thickness: float = 0.1
    point_size: float = 0.0


class DebugMarkers:
    """The shapes an entity wants drawn on one frame."""

    polylines: List[DebugPolyline]

    def __init__(self) -> None:
        self.polylines = []

    def add_polyline(self, polyline: DebugPolyline) -> None:
        """Draw *polyline*, unless it has no points."""
        if polyline.points:
            self.polylines.append(polyline)

    def __bool__(self) -> bool:
        return len(self.polylines) > 0


def world_point(pose: "Pose") -> WorldPoint:
    """Where *pose*, in the right-handed local frame a driver plans in, is in
    CARLA's world: the local frame is the world with ``y`` flipped
    (:mod:`autoware_carla_scenario.driver.observation`)."""
    position = pose.position
    return (float(position[0]), -float(position[1]), float(position[2]))


def plan_polyline(points: List[WorldPoint]) -> DebugPolyline:
    """A plan through *points* (CARLA world) as it is drawn: lifted a little
    above the road, with a dot at each point."""
    return DebugPolyline(
        [(x, y, z + _TRAJECTORY_LIFT_M) for x, y, z in points],
        TRAJECTORY_COLOR,
        0.1,
        0.1,
    )


def plan_markers(plan: "Trajectory") -> DebugMarkers:
    """A driver's *plan*, given in the right-handed local frame, as markers
    (:func:`plan_polyline` through its :func:`world_point` positions).

    An entity builds these once per plan and hands out the same object until
    the plan changes: the trajectory log then writes it once per plan rather
    than once per tick.
    """
    markers = DebugMarkers()
    markers.add_polyline(plan_polyline([world_point(pose) for pose in plan.poses]))
    return markers
