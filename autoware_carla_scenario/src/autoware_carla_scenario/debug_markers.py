"""Debug markers an ego entity shows in the scenario's video.

An ego entity knows things the world does not show: the trajectory its
planner is following, for one.  :meth:`EgoVehicle.debug_markers
<autoware_carla_scenario.entity.ego.EgoVehicle.debug_markers>` hands them out
as :class:`DebugMarkers` -- shapes in CARLA world coordinates -- and
:class:`~autoware_carla_scenario.scenario_runner.ScenarioRunner` draws them into
the video with CARLA's debug helper (``world.debug``).

The video is rendered after the run, by replaying the CARLA recording, and the
recording keeps no debug drawings: the entity is gone by then, too.  So the
runner keeps each tick's markers in a :class:`DebugMarkerLog`, by the frame of
the recording they belong to, and the replay draws a frame's markers just
before the tick that shows it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import typesafe_carla.carla as carla

if TYPE_CHECKING:
    from .driver.geometry import Trajectory

__all__ = [
    "TRAJECTORY_COLOR",
    "DebugMarkerLog",
    "DebugMarkers",
    "DebugPolyline",
    "draw_debug_markers",
    "plan_polyline",
]

#: An RGB colour, each channel 0-255.
RGB = Tuple[int, int, int]
#: A point in CARLA world coordinates, metres.
WorldPoint = Tuple[float, float, float]

#: The colour a planned trajectory is drawn in.
TRAJECTORY_COLOR: RGB = (0, 255, 0)

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

    def __len__(self) -> int:
        return len(self.polylines)

    def __bool__(self) -> bool:
        return len(self.polylines) > 0


def plan_polyline(
    plan: "Trajectory",
    *,
    color: RGB = TRAJECTORY_COLOR,
    thickness: float = 0.1,
    point_size: float = 0.1,
) -> DebugPolyline:
    """The polyline of a driver's *plan*, given in the right-handed local frame.

    The local frame is CARLA's world with ``y`` flipped
    (:mod:`autoware_carla_scenario.driver.observation`), so flipping it back
    puts the plan in the world.  It is drawn a little above the road.
    """
    points: List[WorldPoint] = []
    for pose in plan.poses:
        position = pose.position
        points.append(
            (
                float(position[0]),
                -float(position[1]),
                float(position[2]) + _TRAJECTORY_LIFT_M,
            )
        )
    return DebugPolyline(points, color, thickness, point_size)


def _location(point: WorldPoint) -> "carla.Location":
    return carla.Location(point[0], point[1], point[2])


def _color(rgb: RGB) -> "carla.Color":
    return carla.Color(rgb[0], rgb[1], rgb[2])


def draw_debug_markers(
    world: "carla.World", markers: DebugMarkers, life_time: float
) -> None:
    """Draw *markers* with the world's debug helper, for *life_time* seconds.

    Args:
        world: The CARLA world.
        markers: What to draw.
        life_time: How long the shapes last, simulated seconds; one tick's
            worth draws them on the next frame alone.
    """
    debug = world.debug
    for polyline in markers.polylines:
        color = _color(polyline.color)
        points = polyline.points
        for begin, end in zip(points, points[1:]):
            debug.draw_line(
                _location(begin),
                _location(end),
                thickness=polyline.thickness,
                color=color,
                life_time=life_time,
            )
        if polyline.point_size > 0.0:
            for point in points:
                debug.draw_point(
                    _location(point),
                    size=polyline.point_size,
                    color=color,
                    life_time=life_time,
                )


class DebugMarkerLog:
    """Each frame's markers, by the frame of the CARLA recording they belong to.

    Frame ``k`` is the ``k``-th tick after :meth:`start`, which is called when
    the recorder starts: the frame the replay's ``k``-th tick shows.  Frames
    without markers are not kept, and nothing is before :meth:`start`.
    """

    _start_frame: int
    _active: bool
    _frames: Dict[int, DebugMarkers]

    def __init__(self) -> None:
        self._start_frame = 0
        self._active = False
        self._frames = {}

    @property
    def active(self) -> bool:
        """Whether :meth:`record` keeps what it is given."""
        return self._active

    def start(self, frame: int) -> None:
        """Start a log whose frame ``0`` is the world's frame *frame*."""
        self._start_frame = frame
        self._active = True
        self._frames = {}

    def record(self, frame: int, markers: DebugMarkers) -> None:
        """Keep *markers* as the world's frame *frame*'s.

        Nothing is kept before :meth:`start`, or when there are no markers.
        """
        if self._active and markers:
            self._frames[frame - self._start_frame] = markers

    def at(self, frame: int) -> Optional[DebugMarkers]:
        """Recorded frame *frame*'s markers, or ``None`` when it has none."""
        return self._frames.get(frame)

    def __len__(self) -> int:
        return len(self._frames)
