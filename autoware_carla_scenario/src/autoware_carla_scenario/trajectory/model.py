"""The trajectory an entity is told to follow, after OpenSCENARIO's.

OpenSCENARIO 1.x describes ``FollowTrajectoryAction`` with four parts, and this
module models each one under the same name:

* a :class:`Trajectory` -- a named, possibly closed, shape.  Only the
  ``Polyline`` shape is offered: a recorded drive is a sequence of samples, and
  a clothoid or NURBS fitted to one would only add an approximation between the
  data and the vehicle;
* its :class:`TrajectoryVertex` list -- a position, and the condition the
  entity departs it on (:attr:`TrajectoryVertex.advance`): the time it is
  there (OpenSCENARIO's ``Vertex.time``, here a ``TrajectoryTimeCondition``)
  or -- an extension, a *waypoint condition* -- any other condition;
* a time reference -- ``None`` (OpenSCENARIO's ``<None/>``: the vertex times are
  ignored and the entity keeps its own speed) or a :class:`TrajectoryTiming`;
* a :class:`TrajectoryFollowingMode` -- ``position`` (be exactly on the
  trajectory) or ``follow`` (a controller tracks it).

A vertex's position may be written in any frame the framework addresses: a
:class:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose`,
:class:`~autoware_carla_scenario.coordinate.poses.OpenDrivePose` or
:class:`~autoware_carla_scenario.coordinate.poses.CarlaWorldPose`, a
:class:`MapPose` -- an absolute pose in Autoware's ``map`` frame, which is what a
recording (a T4 scene, a rosbag) states -- or a :class:`RelativeLanePose`, an
offset in lanes and metres from where an entity is when the action starts
(OpenSCENARIO's ``RelativeLanePosition``).  Positions are resolved to CARLA world
coordinates only when the action runs, because that needs the loaded map; the
geometry here (:class:`ResolvedTrajectory`) works on the resolved points and
imports nothing from CARLA, so it can be tested without a simulator.
"""

from __future__ import annotations

import bisect
import enum
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from ..conditions.base import BaseCondition
    from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
    from ..entity_role import EntityRole

__all__ = [
    "MapPose",
    "ReferenceContext",
    "RelativeLanePose",
    "ResolvedTrajectory",
    "Trajectory",
    "TrajectoryFollowingMode",
    "TrajectoryPosition",
    "TrajectorySample",
    "TrajectoryTiming",
    "TrajectoryVertex",
]

#: Segments shorter than this (m) have no direction of their own.
_EPSILON_M = 1e-6
#: Durations shorter than this (s) are treated as an instant.
_EPSILON_S = 1e-9


@dataclass(frozen=True)
class MapPose:
    """An absolute pose in Autoware's ``map`` frame.

    The frame Autoware's ``map_loader`` produces from the Lanelet2 file: the
    projected (MGRS or local) frame this package calls "Lanelet2 absolute",
    right-handed, x East, y North, z Up.  It is the frame a recording of
    Autoware states its poses in, so a recorded trajectory can be written down
    as it was recorded and placed in CARLA through the map's own offset.

    Args:
        x: East, in metres.
        y: North, in metres.
        yaw: Heading in radians, counter-clockwise from East.  ``None`` takes
            the direction of the trajectory at this vertex instead.
        z: Absolute elevation in metres.  ``None`` places the entity on the road
            surface under it, which is what a 2-D recording needs.
    """

    x: float
    y: float
    yaw: Optional[float] = None
    z: Optional[float] = None


@dataclass(frozen=True)
class RelativeLanePose:
    """A pose relative to an entity, in lane coordinates.

    OpenSCENARIO's ``RelativeLanePosition`` (``entityRef``, ``dLane``, ``ds``,
    ``offset``): start from the lanelet the reference entity is on and the
    distance it has come along it, go *ds* metres along the lane, then
    *d_lane* lanes across, and stand *offset* metres from that lane's
    centreline.  It is resolved against where the reference entity is when the
    action starts, and stays where it was resolved to afterwards: the vertex
    does not move with the entity (see ``docs/trajectory.md``).

    * Along the lane, a lanelet's end continues into the lanelet following it
      in the routing graph (a negative *ds* into the one before it).  Where
      there are several, the one that turns least is taken, then the lowest
      id: deterministic, and the "straight on" a route would most often take.
    * Across, *d_lane* counts lanes of the same direction of travel, through
      the routing graph's left and right neighbours, lane-changeable or not.
      The lane change is made at the point *ds* reached, onto the neighbour of
      the lanelet found there.

    Args:
        ds: Metres along the reference entity's lane, from where it is;
            negative goes back.
        offset: Metres from the target lane's centreline, positive to the left
            of its direction of travel -- the sign of
            :attr:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose.t`.
            Measured from the centreline, not from the reference entity's own
            lateral position, as OpenSCENARIO does.
        d_lane: Lanes across: ``+1`` one lane to the left, ``-1`` one to the
            right, ``0`` the reference entity's own lane.
        yaw: Heading in radians relative to the target lane's direction,
            counter-clockwise (left) positive -- the sense of
            :attr:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose.heading`.
            ``None`` takes the direction of the trajectory at this vertex.
        entity_ref: ``role_name`` of the entity the pose is relative to.
            ``None`` means the entity the action moves.
    """

    ds: float = 0.0
    offset: float = 0.0
    d_lane: int = 0
    yaw: Optional[float] = None
    entity_ref: Optional[Union["EntityRole", str]] = None

    def __post_init__(self) -> None:
        if isinstance(self.d_lane, bool) or int(self.d_lane) != self.d_lane:
            raise ValueError(
                f"RelativeLanePose.d_lane must be a whole number, got {self.d_lane!r}"
            )
        object.__setattr__(self, "d_lane", int(self.d_lane))
        for name in ("ds", "offset"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"RelativeLanePose.{name} must be finite")


#: Any position a vertex may be written in.
TrajectoryPosition = Union[
    "CarlaWorldPose", "Lanelet2Pose", "OpenDrivePose", MapPose, RelativeLanePose
]


@dataclass(frozen=True)
class TrajectoryVertex:
    """One vertex of a polyline: where, and what the entity departs it on.

    **Departing.**  Each vertex has at most one departure condition,
    :attr:`advance`: when it holds, the entity leaves the vertex for the next
    one.  It is

    * a **time** -- ``advance=TrajectoryTimeCondition(t)``: depart when the
      trajectory's clock reaches ``t`` (see :class:`TrajectoryTiming` for
      that clock).  This is what a vertex time has always meant; :attr:`time`
      reads it back;
    * any other **condition**: depart once it holds.  Any
      :class:`~autoware_carla_scenario.conditions.BaseCondition` will do --
      an elapsed time, a distance to another entity, a speed, a traffic
      signal, an ``AndCondition`` / ``OrCondition`` of them, or one written
      later -- because the action only asks what every condition answers,
      ``check(world, elapsed)``: anything but ``None`` is satisfied.
      ``elapsed`` is the scenario's elapsed time, as for an action's trigger;
    * ``None`` -- the vertex is passed through on arrival.

    See ``docs/trajectory.md`` (*Waypoint conditions*) for how the entity
    moves between vertices departed in these different ways.

    Args:
        position: Where the entity is at this vertex.
        advance: What the entity departs this vertex on.

    Raises:
        TypeError: If *advance* is not a condition -- in particular a number,
            or a ``time=`` keyword: a vertex no longer takes a bare time.
    """

    position: TrajectoryPosition
    advance: Optional["BaseCondition"] = None

    def __init__(
        self,
        position: TrajectoryPosition,
        advance: Optional["BaseCondition"] = None,
        **removed: Any,
    ) -> None:
        if "time" in removed or isinstance(advance, (int, float)):
            time = removed.get("time", advance)
            raise TypeError(
                "TrajectoryVertex takes no time any more: a vertex's time is its "
                "departure condition. Write TrajectoryVertex(position, "
                f"advance=TrajectoryTimeCondition({time!r}))."
            )
        if removed:
            raise TypeError(
                f"TrajectoryVertex got unexpected arguments {sorted(removed)}"
            )
        # Duck-typed rather than an isinstance check: importing the condition
        # package here would pull its CARLA-facing modules into the editor
        # process, which builds trajectories without a simulator.
        if advance is not None and not callable(getattr(advance, "check", None)):
            raise TypeError(
                "TrajectoryVertex.advance must be a condition (a BaseCondition), "
                f"got {type(advance).__name__}"
            )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "advance", advance)

    @property
    def time(self) -> Optional[float]:
        """The trajectory-clock time this vertex departs at, if that is its condition."""
        if self.advance is None:
            return None
        return _trajectory_time(self.advance)

    @property
    def gate(self) -> Optional["BaseCondition"]:
        """Its departure condition when that is not a time, else ``None``."""
        if self.advance is None or self.time is not None:
            return None
        return self.advance


def _trajectory_time(condition: Any) -> Optional[float]:
    """*condition*'s time if it is a ``TrajectoryTimeCondition``, else ``None``."""
    from ..conditions.trajectory_time import (  # noqa: PLC0415
        TrajectoryTimeCondition,
    )

    if isinstance(condition, TrajectoryTimeCondition):
        return condition.time
    return None


@dataclass(frozen=True)
class Trajectory:
    """A named polyline an entity can be told to follow.

    Args:
        name: What the trajectory is called, for logs.
        vertices: At least two vertices, in the order they are driven.  If
            any vertex departs at a time (a ``TrajectoryTimeCondition``),
            every vertex has to depart on some condition -- either every
            vertex has a time or none has, counting a vertex with another
            condition as having one -- and the times must not decrease along
            the path.
        closed: Whether the last vertex joins back to the first.  Only
            meaningful without timing -- a timed trajectory ends at its last
            vertex's time -- and refused with it, as OpenSCENARIO does.  Its
            other conditions apply on every lap.

    Raises:
        ValueError: On fewer than two vertices, on times given to some
            vertices and no condition to others, on decreasing times, or on a
            closed trajectory that carries times.
    """

    name: str
    vertices: Tuple[TrajectoryVertex, ...]
    closed: bool = False

    def __init__(
        self,
        name: str,
        vertices: Sequence[TrajectoryVertex],
        closed: bool = False,
    ) -> None:
        vertices = tuple(vertices)
        if len(vertices) < 2:
            raise ValueError(
                f"trajectory {name!r} needs at least two vertices, got {len(vertices)}"
            )
        times = [float(t) for t in (v.time for v in vertices) if t is not None]
        if times:
            if any(v.advance is None for v in vertices):
                raise ValueError(
                    f"trajectory {name!r}: either every vertex has a time or none "
                    "has (a vertex departed by another condition counts as "
                    "having one)"
                )
            if any(later < earlier for earlier, later in zip(times, times[1:])):
                raise ValueError(f"trajectory {name!r}: vertex times decrease")
            if closed:
                raise ValueError(
                    f"trajectory {name!r}: a closed trajectory cannot carry times"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "closed", closed)

    @property
    def is_timed(self) -> bool:
        """Whether every vertex carries a time."""
        return all(vertex.time is not None for vertex in self.vertices)

    @property
    def has_times(self) -> bool:
        """Whether any vertex carries a time (a time reference applies to)."""
        return any(vertex.time is not None for vertex in self.vertices)

    @property
    def is_relative(self) -> bool:
        """Whether any vertex is a :class:`RelativeLanePose`.

        Such a trajectory is placed anew each time the action starts, since
        where it lies depends on where the reference entity then is.
        """
        return any(
            isinstance(vertex.position, RelativeLanePose) for vertex in self.vertices
        )

    @property
    def is_gated(self) -> bool:
        """Whether any vertex is departed by a condition other than a time.

        A trajectory that is not gated moves by its times (or speed) alone,
        exactly as a trajectory always has.
        """
        return any(vertex.gate is not None for vertex in self.vertices)

    def gated(self, advance: Mapping[int, "BaseCondition"]) -> "Trajectory":
        """This trajectory with the vertices named departing on new conditions.

        A convenience for a trajectory built from somewhere else -- a
        recording, a lanelet route -- whose vertices are not written one by
        one: ``path.gated({3: ElapsedTimeCondition(20.0, label="go")})`` makes
        the fourth vertex wait until the scenario is 20 s old.  Each vertex
        named has its departure condition **replaced** -- a time included,
        since a vertex departs on one condition; the others keep theirs.

        Args:
            advance: Vertex index (0-based, negative counting from the end as
                for a list) to its new departure condition.

        Raises:
            IndexError: On an index the trajectory has no vertex for.
            ValueError: On a result the trajectory rules refuse.
        """
        vertices = list(self.vertices)
        count = len(vertices)
        for index, condition in advance.items():
            if not -count <= int(index) < count:
                raise IndexError(
                    f"trajectory {self.name!r} has no vertex {index} "
                    f"(it has {count})"
                )
            vertices[int(index)] = TrajectoryVertex(
                vertices[int(index)].position, condition
            )
        return Trajectory(self.name, vertices, self.closed)


class ReferenceContext(enum.Enum):
    """What a vertex time is measured from (OpenSCENARIO ``ReferenceContext``)."""

    #: From the start of the scenario: the clock pass/fail conditions report.
    ABSOLUTE = "absolute"
    #: From the moment the action starts.
    RELATIVE = "relative"


@dataclass(frozen=True)
class TrajectoryTiming:
    """How vertex times map onto the scenario's clock (OpenSCENARIO ``Timing``).

    A vertex at time ``τ`` is reached at scenario time ``τ * scale + offset``,
    measured from the start of the scenario (:attr:`ReferenceContext.ABSOLUTE`)
    or of the action (:attr:`ReferenceContext.RELATIVE`).

    Args:
        domain: What the times are measured from.
        scale: Factor applied to every vertex time; ``2.0`` plays a recording at
            half speed.  Must be positive.
        offset: Seconds added after scaling.

    Raises:
        ValueError: If *scale* is not positive.
    """

    domain: ReferenceContext = ReferenceContext.RELATIVE
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ValueError("TrajectoryTiming.scale must be positive")

    def trajectory_time(self, scenario_time: float, action_start: float) -> float:
        """The vertex time that corresponds to *scenario_time*."""
        origin = 0.0 if self.domain is ReferenceContext.ABSOLUTE else action_start
        return (scenario_time - origin - self.offset) / self.scale


class TrajectoryFollowingMode(enum.Enum):
    """How closely the entity is held to the trajectory (OpenSCENARIO)."""

    #: The entity is placed on the trajectory every tick: a kinematic replay
    #: that reproduces the recording exactly and ignores vehicle dynamics.
    POSITION = "position"
    #: A controller steers and drives the entity along it, so the motion is
    #: the vehicle's own and may lag or cut a corner.
    FOLLOW = "follow"


@dataclass(frozen=True)
class TrajectorySample:
    """Where a resolved trajectory has the entity, in CARLA world coordinates.

    ``yaw`` is in degrees (CARLA's convention), the velocity in m/s along the
    CARLA world axes.
    """

    x: float
    y: float
    z: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    @property
    def speed(self) -> float:
        """Ground speed in m/s."""
        return math.hypot(self.vx, self.vy)


@dataclass
class ResolvedTrajectory:
    """A trajectory whose vertices are CARLA world points.

    Built by the action from a :class:`Trajectory` once the map is loaded.
    Every query is plain arithmetic on the vertices, so this is where the
    interpolation is tested.

    Args:
        xs, ys, zs: Vertex positions in CARLA world coordinates (m).
        yaws: Vertex headings in CARLA degrees; ``None`` for a vertex whose
            heading is taken from the path.
        times: Vertex times, or ``None`` for an untimed trajectory.
        closed: Whether the last vertex joins back to the first.
    """

    xs: Sequence[float]
    ys: Sequence[float]
    zs: Sequence[float]
    yaws: Sequence[Optional[float]]
    times: Optional[Sequence[float]] = None
    closed: bool = False
    _arc: list[float] = field(init=False, repr=False)
    _headings: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        count = len(self.xs)
        if not count == len(self.ys) == len(self.zs) == len(self.yaws):
            raise ValueError("ResolvedTrajectory: vertex columns differ in length")
        if count < 2:
            raise ValueError("ResolvedTrajectory needs at least two vertices")
        if self.times is not None and len(self.times) != count:
            raise ValueError("ResolvedTrajectory: times and vertices differ in length")
        self.xs = [float(value) for value in self.xs]
        self.ys = [float(value) for value in self.ys]
        self.zs = [float(value) for value in self.zs]
        if self.closed:
            # The closing segment is an ordinary one, so the distance queries
            # need no special case for it.
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.zs.append(self.zs[0])
            self.yaws = [*self.yaws, self.yaws[0]]
        arc = [0.0]
        for index in range(1, len(self.xs)):
            arc.append(
                arc[-1]
                + math.hypot(
                    self.xs[index] - self.xs[index - 1],
                    self.ys[index] - self.ys[index - 1],
                )
            )
        self._arc = arc
        self._headings = self._fill_headings()

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Path length in metres, the closing segment included."""
        return self._arc[-1]

    def at_vertex(self, index: int) -> TrajectorySample:
        """Vertex *index* itself, at rest, with the heading it resolved to."""
        return self._vertex(index)

    def vertex_distance(self, index: int) -> float:
        """How far along the path (m) vertex *index* is.

        The index is the vertex's in the trajectory as written; on a closed
        trajectory the first vertex is at ``0`` (and again at :attr:`length`,
        a lap later).
        """
        return self._arc[index]

    @property
    def start_time(self) -> float:
        """The first vertex's time (``0.0`` for an untimed trajectory)."""
        return float(self.times[0]) if self.times is not None else 0.0

    @property
    def end_time(self) -> float:
        """The last vertex's time (``0.0`` for an untimed trajectory)."""
        return float(self.times[-1]) if self.times is not None else 0.0

    def start(self) -> TrajectorySample:
        """The first vertex, at rest."""
        return TrajectorySample(self.xs[0], self.ys[0], self.zs[0], self._headings[0])

    def end(self) -> TrajectorySample:
        """The last vertex, at rest."""
        return TrajectorySample(
            self.xs[-1], self.ys[-1], self.zs[-1], self._headings[-1]
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def at_time(self, time: float) -> TrajectorySample:
        """Where the entity is at vertex time *time*.

        Before the first vertex it waits there, after the last it stays there,
        at rest in both cases.  The velocity is the segment's own, so it is
        what moving from one vertex to the next at the recorded times takes.

        Raises:
            ValueError: On an untimed trajectory.
        """
        if self.times is None:
            raise ValueError("at_time needs a timed trajectory")
        times = self.times
        if time <= times[0]:
            return self.start()
        if time >= times[-1]:
            return self.end()
        index = bisect.bisect_right(times, time) - 1
        duration = float(times[index + 1]) - float(times[index])
        if duration < _EPSILON_S:
            return self._vertex(index + 1)
        fraction = (time - float(times[index])) / duration
        return self._between(index, fraction, duration)

    def at_distance(self, distance: float, speed: float = 0.0) -> TrajectorySample:
        """Where the entity is *distance* metres along the path, moving at *speed*.

        A closed trajectory wraps around; an open one clamps to its ends.
        """
        if self.closed and self.length > _EPSILON_M:
            distance = distance % self.length
        if distance <= 0.0:
            index, fraction = 0, 0.0
        elif distance >= self.length:
            index, fraction = len(self._arc) - 2, 1.0
        else:
            index = bisect.bisect_right(self._arc, distance) - 1
            span = self._arc[index + 1] - self._arc[index]
            fraction = (
                0.0 if span < _EPSILON_M else (distance - self._arc[index]) / span
            )
        sample = self._between(index, fraction, None)
        heading = math.radians(sample.yaw)
        direction = self._segment_heading(index)
        if direction is not None:
            heading = direction
        return TrajectorySample(
            sample.x,
            sample.y,
            sample.z,
            sample.yaw,
            vx=speed * math.cos(heading),
            vy=speed * math.sin(heading),
        )

    def distance_at_time(self, time: float) -> float:
        """How far along the path the entity is at vertex time *time*."""
        if self.times is None:
            raise ValueError("distance_at_time needs a timed trajectory")
        times = [float(value) for value in self.times]
        if time <= times[0]:
            return 0.0
        if time >= times[-1]:
            return self.length
        index = bisect.bisect_right(times, time) - 1
        duration = times[index + 1] - times[index]
        fraction = 0.0 if duration < _EPSILON_S else (time - times[index]) / duration
        return self._arc[index] + fraction * (self._arc[index + 1] - self._arc[index])

    def time_at_distance(self, distance: float) -> float:
        """The vertex time at which the entity is *distance* metres along."""
        if self.times is None:
            raise ValueError("time_at_distance needs a timed trajectory")
        if distance <= 0.0:
            return float(self.times[0])
        if distance >= self.length:
            return float(self.times[-1])
        index = bisect.bisect_right(self._arc, distance) - 1
        span = self._arc[index + 1] - self._arc[index]
        fraction = 0.0 if span < _EPSILON_M else (distance - self._arc[index]) / span
        return float(self.times[index]) + fraction * (
            float(self.times[index + 1]) - float(self.times[index])
        )

    def project(
        self, x: float, y: float, near: Optional[float] = None, window: float = 30.0
    ) -> float:
        """The distance along the path of the point nearest ``(x, y)``.

        Args:
            x, y: The point, in CARLA world coordinates.
            near: Where along the path to look, so that a path which passes the
                same place twice (a loop, a U-turn) answers with the pass the
                entity is on.  ``None`` searches the whole path.
            window: How far either side of *near* to look (m).
        """
        best_distance = math.inf
        best_along = 0.0
        for index in range(len(self._arc) - 1):
            if near is not None and (
                self._arc[index + 1] < near - window or self._arc[index] > near + window
            ):
                continue
            x0, y0 = self.xs[index], self.ys[index]
            dx, dy = self.xs[index + 1] - x0, self.ys[index + 1] - y0
            squared = dx * dx + dy * dy
            fraction = (
                0.0
                if squared < _EPSILON_M
                else min(max(((x - x0) * dx + (y - y0) * dy) / squared, 0.0), 1.0)
            )
            gap = math.hypot(x0 + fraction * dx - x, y0 + fraction * dy - y)
            if gap < best_distance:
                best_distance = gap
                best_along = self._arc[index] + fraction * math.sqrt(squared)
        return best_along

    def vertices(self) -> list[TrajectorySample]:
        """Every vertex, with the heading it resolved to and its times' velocity."""
        samples: list[TrajectorySample] = []
        for index in range(len(self.xs)):
            if self.times is not None and index + 1 < len(self.xs):
                duration = float(self.times[index + 1]) - float(self.times[index])
                if duration >= _EPSILON_S:
                    samples.append(self._between(index, 0.0, duration))
                    continue
            samples.append(self._vertex(index))
        return samples

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vertex(self, index: int) -> TrajectorySample:
        return TrajectorySample(
            self.xs[index], self.ys[index], self.zs[index], self._headings[index]
        )

    def _between(
        self, index: int, fraction: float, duration: Optional[float]
    ) -> TrajectorySample:
        """The point *fraction* of the way along segment *index*.

        With a *duration* the velocity is the segment's, and the yaw rate the
        turn between its vertex headings over that time.
        """
        x0, y0, z0 = self.xs[index], self.ys[index], self.zs[index]
        x1, y1, z1 = self.xs[index + 1], self.ys[index + 1], self.zs[index + 1]
        turn = _wrap_degrees(self._headings[index + 1] - self._headings[index])
        yaw = _wrap_degrees(self._headings[index] + fraction * turn)
        if duration is None:
            vx = vy = yaw_rate = 0.0
        else:
            vx, vy = (x1 - x0) / duration, (y1 - y0) / duration
            yaw_rate = turn / duration
        return TrajectorySample(
            x0 + fraction * (x1 - x0),
            y0 + fraction * (y1 - y0),
            z0 + fraction * (z1 - z0),
            yaw,
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
        )

    def _segment_heading(self, index: int) -> Optional[float]:
        """Segment *index*'s direction in radians, or ``None`` if it has none."""
        dx = self.xs[index + 1] - self.xs[index]
        dy = self.ys[index + 1] - self.ys[index]
        if math.hypot(dx, dy) < _EPSILON_M:
            return None
        return math.atan2(dy, dx)

    def _fill_headings(self) -> list[float]:
        """Every vertex's heading in CARLA degrees.

        A vertex's own heading wins.  One without takes the direction of the
        path there: the segment leaving it, or the one arriving at the last
        vertex.  A vertex where the path does not move (a stop) takes the
        heading of the nearest vertex before it that has one, so a vehicle that
        stopped does not spin to face East.
        """
        count = len(self.xs)
        headings: list[Optional[float]] = []
        for index in range(count):
            given = self.yaws[index]
            if given is not None:
                headings.append(float(given))
                continue
            direction = self._segment_heading(index) if index + 1 < count else None
            if direction is None and index > 0:
                direction = self._segment_heading(index - 1)
            headings.append(None if direction is None else math.degrees(direction))
        filled: list[float] = []
        previous: Optional[float] = None
        for heading in headings:
            if heading is None:
                heading = previous
            filled.append(heading if heading is not None else math.nan)
            previous = heading if heading is not None else previous
        # A path that does not move at all before its first heading: backfill.
        first = next((value for value in filled if not math.isnan(value)), 0.0)
        return [first if math.isnan(value) else value for value in filled]


def _wrap_degrees(angle: float) -> float:
    """*angle* wrapped into ``[-180, 180)``."""
    return (angle + 180.0) % 360.0 - 180.0
