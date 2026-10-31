"""hils.lidar_revolution's LidarRevolution against the class it was moved from.

``_OriginalLidarRevolution`` is ``hils.measurements.LidarRevolution`` as it was
before the move (#838), copied verbatim; the moved class must gather every
stream of arcs into the same revolutions, keeping the same state between.
"""

from __future__ import annotations

import itertools
import random
from typing import List, Optional

import numpy as np
import pytest

from autoware_carla_scenario.hils import measurements
from autoware_carla_scenario.hils.lidar_revolution import LidarRevolution

FloatPoints = np.ndarray[tuple[int, int], np.dtype[np.float32]]

_REVOLUTION_TOLERANCE_DEG = 1e-3


class _OriginalLidarRevolution:
    """Gathers a LiDAR's per-tick arcs into whole revolutions.

    A measurement sweeps ``rotation_frequency * 360 * dt`` degrees, ``dt``
    being the simulated time since the previous one; :meth:`add` returns the
    points of a revolution once the arcs add up to one, and keeps the rest
    for the next.  A LiDAR that turns once a tick (or once a ``sensor_tick``)
    yields each measurement as it comes.

    Args:
        rotation_frequency: The blueprint's ``rotation_frequency``, Hz.
        first_dt: The simulated time the first measurement covers (the
            world's tick length).
    """

    _per_second: float
    _first_dt: float
    _time: Optional[float]
    _swept: float
    _parts: List[FloatPoints]

    def __init__(self, rotation_frequency: float, first_dt: float) -> None:
        if rotation_frequency <= 0.0 or first_dt <= 0.0:
            raise ValueError("rotation_frequency and first_dt must be positive")
        self._per_second = 360.0 * rotation_frequency
        self._first_dt = first_dt
        self._time = None
        self._swept = 0.0
        self._parts = []

    def add(self, time: float, points: FloatPoints) -> Optional[FloatPoints]:
        """Add the measurement taken at *time*; a whole revolution, or ``None``."""
        dt = self._first_dt if self._time is None else time - self._time
        if dt <= 0.0:  # a new episode, its clock restarted: start over
            self._parts.clear()
            self._swept = 0.0
            dt = self._first_dt
        self._time = time
        self._parts.append(points)
        self._swept += self._per_second * dt
        if self._swept + _REVOLUTION_TOLERANCE_DEG < 360.0:
            return None
        self._swept = max(0.0, self._swept - 360.0)
        if self._swept < _REVOLUTION_TOLERANCE_DEG:
            self._swept = 0.0
        revolution = (
            self._parts[0] if len(self._parts) == 1 else np.concatenate(self._parts)
        )
        self._parts = []
        return revolution


def _points(rng: random.Random, serial: int) -> FloatPoints:
    """A few points (possibly none), each row tagged with *serial*."""
    count = rng.choice([0, 1, 2, 5])
    points = np.full((count, 4), float(serial), np.float32)
    points[:, 3] = np.float32(rng.random())
    return points


def _times(rng: random.Random, dt: float, steps: int) -> List[float]:
    """Tick times: steady, jittered, skipped ticks, and restarted clocks."""
    times: List[float] = []
    now = 0.0
    for _ in range(steps):
        kind = rng.random()
        if kind < 0.08:
            now = rng.choice([0.0, dt, dt * rng.random()])  # a new episode
        elif kind < 0.12:
            pass  # the same time again
        elif kind < 0.25:
            now += dt * rng.randint(2, 30)  # skipped ticks, past 360 degrees
        elif kind < 0.40:
            now += dt * (1.0 + 0.5 * (rng.random() - 0.5))  # jitter
        else:
            now += dt
        times.append(now)
    return times


def _same(got: Optional[FloatPoints], want: Optional[FloatPoints]) -> None:
    if want is None:
        assert got is None
        return
    assert got is not None
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def _assert_same_state(
    moved: LidarRevolution, original: _OriginalLidarRevolution
) -> None:
    assert moved._per_second == original._per_second
    assert moved._first_dt == original._first_dt
    assert moved._time == original._time
    assert moved._swept == original._swept
    assert len(moved._parts) == len(original._parts)
    for a, b in zip(moved._parts, original._parts):
        assert a is b


def test_measurements_re_exports_the_moved_class() -> None:
    assert measurements.LidarRevolution is LidarRevolution
    assert "LidarRevolution" in measurements.__all__


@pytest.mark.parametrize(
    ("rotation_frequency", "first_dt"),
    [(0.0, 0.05), (-1.0, 0.05), (10.0, 0.0), (10.0, -0.1), (0.0, 0.0)],
)
def test_the_same_arguments_are_refused(
    rotation_frequency: float, first_dt: float
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _OriginalLidarRevolution(rotation_frequency, first_dt)
    with pytest.raises(ValueError, match="must be positive"):
        LidarRevolution(rotation_frequency, first_dt)


@pytest.mark.parametrize(
    ("seed", "rotation_frequency", "first_dt"),
    [
        (seed, frequency, dt)
        for seed, (frequency, dt) in enumerate(
            itertools.product(
                [0.5, 5.0, 10.0, 20.0, 33.3, 100.0],
                [0.01, 0.05, 1.0 / 30.0, 0.1],
            )
        )
    ],
)
def test_streams_gather_into_the_same_revolutions(
    seed: int, rotation_frequency: float, first_dt: float
) -> None:
    rng = random.Random(seed)
    for _ in range(5):
        moved = LidarRevolution(rotation_frequency, first_dt)
        original = _OriginalLidarRevolution(rotation_frequency, first_dt)
        for serial, time in enumerate(_times(rng, first_dt, 200)):
            points = _points(rng, serial)
            got = moved.add(time, points)
            want = original.add(time, points)
            _same(got, want)
            if want is not None and len(original._parts) == 0 and want is points:
                assert got is points  # a single arc is yielded as it came
            _assert_same_state(moved, original)


def test_a_stream_of_steady_ticks_matches() -> None:
    # 10 Hz at 20 Hz ticks, with float sums of tick lengths.
    moved = LidarRevolution(10.0, 0.05)
    original = _OriginalLidarRevolution(10.0, 0.05)
    rng = random.Random(7)
    time = 0.0
    yielded = 0
    for serial in range(1000):
        time += 0.05
        points = _points(rng, serial)
        want = original.add(time, points)
        _same(moved.add(time, points), want)
        yielded += want is not None
        _assert_same_state(moved, original)
    assert yielded == 500
