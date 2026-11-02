"""A LiDAR's per-tick arcs, gathered into the whole revolutions a sensor sends.

hilspy's ``lidar:MODEL`` codec encodes one *revolution* into a sensor's
packets, while CARLA measures the arc a LiDAR sweeps in one tick;
:class:`LidarRevolution` adds the arcs up and hands back the points of each
revolution.  It needs only numpy: :mod:`.measurements` re-exports it beside
the conversions that do not compile.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["LidarRevolution"]

#: An arc this close to a whole revolution completes it (float sums of tick
#: lengths fall a hair short).
_REVOLUTION_TOLERANCE_DEG = 1e-3

#: A point this far short of a revolution's start, by its azimuth, is one at
#: the start: float rounding in CARLA's angle and in ``atan2`` must not put
#: the ray that starts a revolution at the end of the one before (a ray is a
#: hundredth of a degree or more from the next).
_START_TOLERANCE_DEG = 1e-3


def _angles(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]], fov: float
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """The horizontal angle each of *points* was measured at, in [0, *fov*).

    CARLA's ray-cast LiDAR turns its horizontal angle ``h`` through
    [0, horizontal_fov), from 0 when spawned, and shoots its rays at yaw
    ``h - horizontal_fov / 2``: a point's azimuth (``atan2(y, x)`` in CARLA's
    sensor frame, y right) plus half the field of view is the angle ``h`` it
    was measured at.
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return (np.degrees(np.arctan2(y, x)) + 0.5 * fov) % fov


def _arc_start(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]], arc: float, fov: float
) -> float:
    """Where an arc of *arc* degrees (less than *fov*) started, from its points.

    The arc's points leave out the angle it did not sweep, ``fov - arc``
    degrees or more: the point after the widest gap between them is the
    first of the arc.  -1 when the arc has no points, or no gap that wide
    (missing hits can only widen gaps).
    """
    if points.shape[0] == 0:
        return -1.0
    angles = np.sort(_angles(points, fov))
    gaps = np.diff(np.concatenate((angles, angles[:1] + fov)))
    widest = int(np.argmax(gaps))
    if float(gaps[widest]) + _REVOLUTION_TOLERANCE_DEG < fov - arc:
        return -1.0
    return float(angles[(widest + 1) % angles.shape[0]])


class LidarRevolution:
    """Gathers a LiDAR's per-tick arcs into whole revolutions.

    A revolution is one sweep of the field of view, ``horizontal_fov``
    degrees (CARLA turns the LiDAR's horizontal angle through [0, fov), its
    rays covering azimuths [-fov/2, fov/2)).  A measurement sweeps
    ``rotation_frequency * horizontal_fov * dt`` degrees, ``dt`` being the
    simulated time since the previous one: CARLA sweeps one world tick per
    measurement and measures every tick, so that time is the tick's, or the
    ticks' whose measurements were lost.  A ``sensor_tick`` longer than the
    world's tick breaks that (CARLA still sweeps one tick per measurement,
    so the LiDAR no longer turns at ``rotation_frequency``):
    :mod:`.config` refuses a LiDAR with one.  :meth:`add` returns the points
    of a revolution once the arcs add up to one, and keeps the rest for the
    next: the arc that completes a revolution is split at the azimuth where
    the revolution started, so each one covers the field of view once.  A
    LiDAR that turns once a tick yields each measurement as it comes, and
    so does an arc of a whole revolution or more (a slow tick): it covers
    every azimuth already, and the partial revolution gathered before it,
    measured earlier at the same azimuths, is dropped.

    Revolutions start where the first arc with points starts, read off its
    points, and again after a new episode or an arc of a whole revolution
    or more; the angles swept in between are the simulated time's.  The
    first measurements a sensor takes need not arrive (CARLA turns a LiDAR
    from the tick it is spawned on, listened to or not), so where it
    started, at horizontal angle 0, is not where the first arc that arrives
    does.

    Args:
        rotation_frequency: The blueprint's ``rotation_frequency``, Hz.
        first_dt: The simulated time the first measurement covers (the
            world's tick length).
        horizontal_fov: The blueprint's ``horizontal_fov``, degrees, in
            (0, 360].
    """

    _fov: float
    _per_second: float
    _first_dt: float
    _time: Optional[float]
    #: Degrees swept since the current revolution started.
    _swept: float
    #: The sensor's horizontal angle at which the current revolution
    #: started, in degrees, modulo the field of view.
    _start: float
    #: Whether :attr:`_start` was read off the points since the revolutions
    #: last started over.
    _anchored: bool
    _parts: List[np.ndarray[tuple[int, int], np.dtype[np.float32]]]

    def __init__(
        self, rotation_frequency: float, first_dt: float, horizontal_fov: float = 360.0
    ) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        if not 0.0 < horizontal_fov <= 360.0:
            raise ValueError(
                f"horizontal_fov must be in (0, 360] degrees, not {horizontal_fov}"
            )
        self._fov = horizontal_fov
        self._per_second = horizontal_fov * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
        self._start = 0.0
        self._anchored = False
        self._parts = []

    def add(
        self, time: float, points: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    ) -> Optional[np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """Add the measurement taken at *time*; a whole revolution, or ``None``."""
        previous = self._time
        if previous is None:
            dt = self._first_dt
        else:
            dt = time - previous
        if dt <= 0.0:  # a new episode, its clock restarted: start over
            self._parts.clear()
            self._swept = 0.0
            self._start = 0.0
            self._anchored = False
            dt = self._first_dt
        self._time = time
        arc = self._per_second * dt
        before = self._swept
        swept = before + arc
        fov = self._fov
        if not self._anchored and arc + _REVOLUTION_TOLERANCE_DEG < fov:
            start = _arc_start(points, arc, fov)
            if start >= 0.0:
                self._start = (start - before) % fov
                self._anchored = True
        if swept + _REVOLUTION_TOLERANCE_DEG < fov:
            self._parts.append(points)
            self._swept = swept
            return None
        if arc + _REVOLUTION_TOLERANCE_DEG >= fov:
            # The arc alone covers every azimuth: it is the revolution, and
            # the next one starts where it ended.
            self._parts = []
            self._swept = 0.0
            self._start = (self._start + swept) % fov
            # Not every renderer sweeps the whole arc (carla_canvas sweeps
            # one revolution at most): the next arc's points tell.
            self._anchored = False
            return points
        leftover = swept - fov
        if leftover < _REVOLUTION_TOLERANCE_DEG:
            # The arc ends on the revolution's end (a sweep dividing fov).
            # The next one starts where it did end, a hair either side of it
            # (float sums of tick lengths), so that hairs do not add up.
            self._parts.append(points)
            self._swept = 0.0
            self._start = (self._start + leftover) % fov
            if len(self._parts) == 1:
                revolution = self._parts[0]
            else:
                revolution = np.concatenate(self._parts)
            self._parts = []
            return revolution
        # The arc runs past the revolution's end: split it by each point's
        # azimuth, not by its place in the buffer.  CARLA lays the points out
        # channel by channel, each channel's in the order it turned, so the
        # first part of the buffer is the first channels' whole arc, not the
        # first part of the arc.  Relative to the revolution's start, the arc
        # covers [before, fov) then [0, leftover); half way through the angle
        # it did not sweep, [leftover, before), is the cut furthest from both.
        relative = (
            _angles(points, fov) + (_START_TOLERANCE_DEG - self._start)
        ) % fov - _START_TOLERANCE_DEG
        ends = relative >= 0.5 * (leftover + before)
        self._parts.append(points[ends])
        self._swept = leftover
        revolution = np.concatenate(self._parts)
        self._parts = [points[~ends]]
        return revolution
