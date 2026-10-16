"""Debug markers in the scenario's video: the ego's plan and the path it drove.

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

The path the ego actually drove needs no entity: the replay moves the ego as
it moved, and a :class:`DrivenPathDrawer` traces it behind the car as the
replay goes, beside the plan it was following.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import typesafe_carla.carla as carla

if TYPE_CHECKING:
    from .driver.geometry import Trajectory

__all__ = [
    "DRIVEN_PATH_COLOR",
    "TRAJECTORY_COLOR",
    "DebugMarkerLog",
    "DebugMarkers",
    "DebugPolyline",
    "DrivenPathDrawer",
    "draw_debug_markers",
    "plan_markers",
]

#: An RGB colour, each channel 0-255.
RGB = Tuple[int, int, int]
#: A point in CARLA world coordinates, metres.
WorldPoint = Tuple[float, float, float]

#: The colour a planned trajectory is drawn in.
TRAJECTORY_COLOR: RGB = (0, 255, 0)

#: The colour the path the ego drove is drawn in.
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


def plan_markers(plan: "Trajectory") -> DebugMarkers:
    """A driver's *plan*, given in the right-handed local frame, as markers.

    The local frame is CARLA's world with ``y`` flipped
    (:mod:`autoware_carla_scenario.driver.observation`), so flipping it back
    puts the plan in the world.  It is drawn a little above the road, with a
    dot at each point.  An entity builds these once per plan and hands out the
    same object until the plan changes: :class:`DebugMarkerLog` then holds one
    per plan rather than one per tick, and the replay draws it once.
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
    markers = DebugMarkers()
    markers.add_polyline(DebugPolyline(points, TRAJECTORY_COLOR, 0.1, 0.1))
    return markers


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
    without markers are not kept, and nothing is outside :meth:`start` and
    :meth:`stop`.
    """

    #: The world's frame the log's frame ``0`` is; ``-1`` while not recording.
    _start_frame: int
    _frames: Dict[int, DebugMarkers]

    def __init__(self) -> None:
        self._start_frame = -1
        self._frames = {}

    @property
    def active(self) -> bool:
        """Whether :meth:`record` keeps what it is given."""
        return self._start_frame >= 0

    def start(self, frame: int) -> None:
        """Start a log whose frame ``0`` is the world's frame *frame*."""
        self._start_frame = frame
        self._frames = {}

    def stop(self) -> None:
        """Keep nothing more; what is kept stays."""
        self._start_frame = -1

    def record(self, frame: int, markers: DebugMarkers) -> None:
        """Keep *markers* as the world's frame *frame*'s.

        Nothing is kept while the log is not active, or when there are no
        markers.
        """
        if self.active and markers:
            self._frames[frame - self._start_frame] = markers

    def at(self, frame: int) -> Optional[DebugMarkers]:
        """Recorded frame *frame*'s markers, or ``None`` when it has none."""
        return self._frames.get(frame)

    def span(self, frame: int) -> int:
        """How many frames from *frame* on hold the very markers it holds.

        An entity hands out one object per plan (:func:`plan_markers`), so a
        plan the policy keeps for several ticks is drawn once, to last them
        all.  ``0`` when *frame* has none.
        """
        markers = self._frames.get(frame)
        if markers is None:
            return 0
        count = 1
        while self._frames.get(frame + count) is markers:
            count += 1
        return count


class DrivenPathDrawer:
    """Traces the path a vehicle drives, one segment at a time.

    Fed the vehicle's location after every tick, it draws a segment from the
    last point it drew to there once the vehicle has moved *step_m*, so a
    vehicle standing still adds nothing, and a long run stays a few segments
    per metre.  A segment is drawn once, to last as long as it is given: the
    rest of the video.  A jump longer than *max_jump_m* in one tick -- the
    vehicle placed somewhere else -- starts a new piece of path rather than
    drawing a line across the map.

    Args:
        step_m: How far the vehicle moves between two points of the path.
        max_jump_m: The longest a vehicle is taken to drive in one tick.
    """

    _step_m: float
    _max_jump_m: float
    _color: "carla.Color"
    _last: Optional["carla.Location"]

    def __init__(self, step_m: float = 0.5, max_jump_m: float = 10.0) -> None:
        self._step_m = step_m
        self._max_jump_m = max_jump_m
        self._color = _color(DRIVEN_PATH_COLOR)
        self._last = None

    def update(
        self, world: "carla.World", location: "carla.Location", life_time: float
    ) -> None:
        """Extend the path to *location*, drawing for *life_time* seconds."""
        here = carla.Location(location.x, location.y, location.z + _TRAJECTORY_LIFT_M)
        last = self._last
        if last is None:
            self._last = here
            return
        distance = here.distance(last)
        if distance > self._max_jump_m:
            self._last = here
            return
        if distance < self._step_m:
            return
        world.debug.draw_line(
            last, here, thickness=0.1, color=self._color, life_time=life_time
        )
        self._last = here
