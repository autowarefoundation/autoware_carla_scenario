"""When an entity following a trajectory is where: the departures it knows of.

A trajectory whose vertices are departed by conditions (waypoint conditions)
cannot be read off the vertex times alone: how long the entity waits at a
vertex is only known once that vertex's condition holds.  What *is* known at
any moment is a piecewise-linear timeline -- distance along the path against
scenario time -- from the last departure up to the next vertex whose
condition has not held yet (or the end).  :class:`DepartureTimeline` keeps
that timeline and extends it each time a waiting vertex is departed.

The rule for the segment from vertex *i*, departed at scenario time ``T``,
to vertex *i + 1*:

* if vertex *i + 1* has a time whose scenario time ``s`` is not before ``T``,
  the entity arrives there at ``s`` -- linear in between.  When every vertex
  is departed on its time this is exactly the timed interpolation a recording
  is played with (a zero-duration segment is a jump, as there);
* otherwise -- no time, or a time already past -- it goes at the action's
  speed.

A vertex with a time is departed on arrival, which is that time; one with no
condition on arrival too; one with any other condition ends the timeline
until :meth:`DepartureTimeline.depart` says when it held.

Pure arithmetic, so it is tested without a simulator.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..conditions.base import BaseCondition

__all__ = ["DepartureTimeline", "TimelineVertex"]

#: Lengths (m) and durations (s) below these are nothing.
_EPSILON_M = 1e-9
_EPSILON_S = 1e-9


@dataclass(frozen=True)
class TimelineVertex:
    """One vertex as the timeline needs it.

    Attributes:
        distance: How far along the path it is (m), on the first lap.
        time: The scenario time its time condition holds at; ``None`` when it
            has no time (or the action has no time reference).
        gate: Its other departure condition, if any -- opaque here: only
            whether there is one is read.
    """

    distance: float
    time: Optional[float] = None
    gate: Optional["BaseCondition"] = None


class DepartureTimeline:
    """Distance along the path against scenario time, as far as it is known.

    Vertices are numbered absolutely: on a closed path vertex ``k`` is vertex
    ``k % n`` on lap ``k // n``, so a lap later is a different number, and
    every lap's gates are new.

    Args:
        vertices: The trajectory's vertices, in order.
        length: The path's length (m), the closing segment included.
        closed: Whether the path joins back to its first vertex.
        speed: The action's speed (m/s), for segments no time says how long
            to take over.  At (near) zero such a segment is never finished.
    """

    _vertices: list[TimelineVertex]
    _length: float
    _closed: bool
    _speed: float
    distances: list[float]
    times: list[float]
    end_vertex: int
    end_kind: str

    def __init__(
        self,
        vertices: list[TimelineVertex],
        length: float,
        closed: bool,
        speed: float,
    ) -> None:
        self._vertices = vertices
        self._length = length
        self._closed = closed
        self._speed = speed
        self.distances: list[float] = []
        self.times: list[float] = []
        #: The vertex (absolute number) the timeline ends at, and why:
        #: ``"gate"`` -- it waits there for its condition -- or ``"end"``,
        #: the last vertex of an open path, departed by nothing.
        self.end_vertex = 0
        self.end_kind = "end"

    # ------------------------------------------------------------------
    # Numbering
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """How many vertices one lap has."""
        return len(self._vertices)

    def vertex(self, k: int) -> TimelineVertex:
        """Vertex number *k*'s data."""
        return self._vertices[k % self.count if self._closed else k]

    def index(self, k: int) -> int:
        """Vertex number *k*'s index in the trajectory."""
        return k % self.count if self._closed else k

    def distance(self, k: int) -> float:
        """How far along (m) vertex number *k* is, laps included."""
        if not self._closed:
            return self._vertices[k].distance
        lap, index = divmod(k, self.count)
        return lap * self._length + self._vertices[index].distance

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def start(self, k: int, distance: float, departure: float) -> None:
        """Begin at *distance*, departed at *departure*, after vertex *k*.

        *k* is the vertex the start is at (``distance`` is its distance) or,
        for a start between two vertices, the vertex before it.
        """
        self.distances = []
        self.times = []
        self._extend(k, distance, departure)

    def start_waiting(self, k: int, arrival: float) -> None:
        """Begin at vertex *k*, waiting there for its condition since *arrival*."""
        self.distances = [self.distance(k)]
        self.times = [arrival]
        self.end_vertex = k
        self.end_kind = "gate"

    def depart(self, departure: float) -> None:
        """The vertex the timeline ends at is departed at *departure*."""
        departure = max(departure, self.times[-1])
        self._extend(self.end_vertex, self.distances[-1], departure)

    def _extend(self, k: int, distance: float, departure: float) -> None:
        self.distances.append(distance)
        self.times.append(departure)
        previous_distance, previous_time = distance, departure
        last = None if self._closed else self.count - 1
        while True:
            k += 1
            if last is not None and k > last:
                # Started on the last vertex: nothing further to go.
                self.end_vertex, self.end_kind = last, "end"
                return
            vertex = self.vertex(k)
            here = self.distance(k)
            span = here - previous_distance
            if vertex.time is not None and vertex.time >= previous_time:
                arrival = vertex.time
            elif span <= _EPSILON_M:
                arrival = previous_time
            elif self._speed > _EPSILON_M:
                arrival = previous_time + span / self._speed
            else:
                arrival = math.inf
            self.distances.append(here)
            self.times.append(arrival)
            if vertex.gate is not None or math.isinf(arrival):
                self.end_vertex, self.end_kind = k, "gate"
                return
            if last is not None and k == last:
                self.end_vertex, self.end_kind = k, "end"
                return
            previous_distance, previous_time = here, arrival

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def end_arrival(self) -> float:
        """When the entity reaches the vertex the timeline ends at."""
        return self.times[-1]

    def at(self, elapsed: float) -> tuple[float, float]:
        """``(distance along the path, speed along it)`` at scenario time *elapsed*.

        Before the first point the entity stands there; past the last, at the
        last.  Between two points of the same distance it stands; between two
        of the same time it has jumped.
        """
        times = self.times
        index = bisect.bisect_right(times, elapsed) - 1
        if index < 0:
            return self.distances[0], 0.0
        if index >= len(times) - 1:
            return self.distances[-1], 0.0
        t0, t1 = times[index], times[index + 1]
        d0, d1 = self.distances[index], self.distances[index + 1]
        if math.isinf(t1) or d1 - d0 <= _EPSILON_M or t1 - t0 <= _EPSILON_S:
            return d0, 0.0
        speed = (d1 - d0) / (t1 - t0)
        return d0 + (elapsed - t0) * speed, speed

    def stamp(self, distance: float, fallback_speed: float) -> float:
        """When the entity is at *distance* -- for points on the timeline.

        Before its first point it is extrapolated back at *fallback_speed*;
        past its last it is the last point's time.  At a distance it stands
        at, the time it departs.
        """
        distances = self.distances
        index = bisect.bisect_right(distances, distance) - 1
        if index < 0:
            speed = max(fallback_speed, 1e-3)
            return self.times[0] - (distances[0] - distance) / speed
        if index >= len(distances) - 1:
            return self.times[-1]
        d0, d1 = distances[index], distances[index + 1]
        t0, t1 = self.times[index], self.times[index + 1]
        if d1 - d0 <= _EPSILON_M:
            return t1
        if math.isinf(t1):
            return math.inf
        return t0 + (distance - d0) / (d1 - d0) * (t1 - t0)
