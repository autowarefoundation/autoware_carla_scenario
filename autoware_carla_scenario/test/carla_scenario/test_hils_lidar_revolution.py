"""hils.lidar_revolution's LidarRevolution.

``_OriginalLidarRevolution`` is ``hils.measurements.LidarRevolution`` as it was
before the move (#838), copied verbatim.  Where the arcs end on revolution
boundaries (a tick's arc divides 360 degrees), the moved class must gather
every stream of arcs into the same revolutions, keeping the same state
between.  Elsewhere the original emitted the arc past 360 degrees with the
revolution it completed (#1122); the moved class splits that arc by azimuth,
which the tests below check on arcs laid out as CARLA's ray-cast LiDAR lays
them out.
"""

from __future__ import annotations

import math
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
    """Tick times: steady, and restarted clocks (so arcs end on boundaries).

    Jittered and skipped ticks make arcs that cross a boundary, which the
    original emitted whole (#1122): :func:`test_a_carla_lidar_stream` covers
    them.
    """
    times: List[float] = []
    now = 0.0
    for _ in range(steps):
        kind = rng.random()
        if kind < 0.08:
            now = rng.choice([0.0, dt])  # a new episode
        elif kind < 0.12:
            pass  # the same time again
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
            # Every pair whose arc (frequency * 360 * dt) divides 360 degrees.
            [
                (0.5, 1.0),
                (5.0, 0.05),
                (5.0, 0.1),
                (10.0, 0.01),
                (10.0, 0.05),
                (10.0, 0.1),
                (20.0, 0.05),
                (30.0, 1.0 / 30.0),
                (100.0, 0.01),
            ]
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


class _CarlaLidar:
    """Arcs as CARLA's ray-cast LiDAR measures them.

    Its horizontal angle starts at 0 and grows by ``rotation_frequency * 360
    * dt`` a tick; a ray at angle ``h`` points at yaw ``h - 180`` (``atan2(y,
    x)`` in the sensor frame), and the buffer holds the points channel by
    channel, each channel's in the order it turned.  A ray is shot every
    ``step`` degrees, at the middle of its step; each point's intensity is
    the ray's index (its angle ``(index + 0.5) * step``), its z the channel.
    """

    def __init__(self, rotation_frequency: float, channels: int, step: float):
        self.per_second = 360.0 * rotation_frequency
        self.channels = channels
        self.step = step
        self.next_ray = 0

    def arc(self, dt: float) -> FloatPoints:
        end = int(round(self.per_second * dt / self.step)) + self.next_ray
        rays = np.arange(self.next_ray, end)
        self.next_ray = end
        yaw = np.radians((rays + 0.5) * self.step - 180.0)
        per_channel = [
            np.stack(
                [
                    10.0 * np.cos(yaw),
                    10.0 * np.sin(yaw),
                    np.full(len(rays), float(channel)),
                    rays.astype(np.float64),
                ],
                axis=1,
            )
            for channel in range(self.channels)
        ]
        return np.concatenate(per_channel).astype(np.float32)


def _rays(revolution: FloatPoints, channel: int) -> List[int]:
    return sorted(int(r) for r in revolution[revolution[:, 2] == channel, 3])


@pytest.mark.parametrize(
    ("rotation_frequency", "dt"),
    [
        (15.0, 0.05),  # 270 degrees a tick (#1122)
        (10.0, 0.05),  # 180, dividing 360
        (7.0, 0.05),  # 126
        (13.0, 0.05),  # 234
        (19.0, 0.05),  # 342
        (10.0, 1.0 / 30.0),  # 120, dividing 360
        (25.0, 1.0 / 30.0),  # 300
        (2.5, 0.1),  # 90, dividing 360
        (8.0, 0.1),  # 288
    ],
)
def test_each_revolution_covers_every_azimuth_once(
    rotation_frequency: float, dt: float
) -> None:
    channels, step = 4, 0.5
    per_revolution = int(round(360.0 / step))
    lidar = _CarlaLidar(rotation_frequency, channels, step)
    revolution = LidarRevolution(rotation_frequency, dt)
    emitted: List[FloatPoints] = []
    time = 0.0
    for _ in range(60):
        time += dt
        whole = revolution.add(time, lidar.arc(dt))
        if whole is not None:
            emitted.append(whole)
    assert len(emitted) == int(60 * rotation_frequency * dt)
    for index, whole in enumerate(emitted):
        assert whole.dtype == np.float32 and whole.shape[1] == 4
        assert len(whole) == channels * per_revolution
        first = index * per_revolution
        for channel in range(channels):
            assert _rays(whole, channel) == list(range(first, first + per_revolution))
    # What is left is the start of the next revolution, nothing more.
    pending = np.concatenate(revolution._parts) if revolution._parts else None
    rays_pending = [] if pending is None else _rays(pending, 0)
    assert rays_pending == list(range(len(emitted) * per_revolution, lidar.next_ray))


def test_15_hz_at_20_hz_ticks_emits_360_degree_revolutions() -> None:
    lidar = _CarlaLidar(15.0, 1, 1.0)
    revolution = LidarRevolution(15.0, 0.05)
    sizes = []
    for tick in range(1, 9):
        whole = revolution.add(0.05 * tick, lidar.arc(0.05))
        if whole is not None:
            sizes.append(len(whole))
    assert sizes == [360] * 6


def test_an_arc_of_a_whole_revolution_or_more_is_one() -> None:
    # 15 Hz: 270 degrees a tick, then a tick of 0.1 s (540 degrees).
    lidar = _CarlaLidar(15.0, 2, 1.0)
    revolution = LidarRevolution(15.0, 0.05)
    assert revolution.add(0.05, lidar.arc(0.05)) is None
    slow = lidar.arc(0.1)
    assert revolution.add(0.15, slow) is slow  # the arc as it came
    assert revolution._parts == []
    # The next revolutions start where that arc ended (at 810 degrees).
    emitted = []
    time = 0.15
    for _ in range(8):
        time += 0.05
        whole = revolution.add(time, lidar.arc(0.05))
        if whole is not None:
            emitted.append(whole)
    assert len(emitted) == 6
    for index, whole in enumerate(emitted):
        first = 810 + 360 * index
        for channel in range(2):
            assert _rays(whole, channel) == list(range(first, first + 360))


def test_a_restart_after_a_split_starts_at_the_sensors_start() -> None:
    revolution = LidarRevolution(15.0, 0.05)
    lidar = _CarlaLidar(15.0, 1, 1.0)
    revolution.add(0.05, lidar.arc(0.05))
    assert revolution.add(0.10, lidar.arc(0.05)) is not None
    # A new episode (a new sensor turning from 0).
    lidar = _CarlaLidar(15.0, 1, 1.0)
    assert revolution.add(0.05, lidar.arc(0.05)) is None
    whole = revolution.add(0.10, lidar.arc(0.05))
    assert whole is not None and _rays(whole, 0) == list(range(360))


class _Sensor:
    """CARLA's ray-cast LiDAR as RayCastSemanticLidar.cpp turns it.

    Its horizontal angle starts at 0 when it is spawned and grows by
    ``rotation_frequency * 360 * dt`` every tick, measured or not; a tick
    shoots the rays from where the angle was, first ray included, each at yaw
    ``angle - 180``, and lays the hits out channel by channel.  Rays are
    ``step`` degrees apart, on multiples of ``step`` (so some fall exactly on
    a revolution's start); each point's intensity is the ray's index, its z
    the channel.  A ray misses with probability ``miss``.
    """

    def __init__(
        self,
        rotation_frequency: float,
        rng: random.Random,
        channels: int = 3,
        step: float = 0.5,
        miss: float = 0.1,
    ) -> None:
        self.per_second = 360.0 * rotation_frequency
        self.rng = rng
        self.channels = channels
        self.step = step
        self.miss = miss
        self.angle = 0.0
        self.np_rng = np.random.default_rng(rng.randrange(2**32))

    def tick(self, dt: float) -> FloatPoints:
        arc = self.per_second * dt
        first = math.ceil(self.angle / self.step - 1e-9)
        end = math.ceil((self.angle + arc) / self.step - 1e-9)
        self.angle += arc
        rays = np.tile(np.arange(first, end, dtype=np.float64), self.channels)
        channel = np.repeat(np.arange(self.channels, dtype=np.float64), end - first)
        hit = self.np_rng.random(len(rays)) >= self.miss
        rays, channel = rays[hit], channel[hit]
        yaw = np.radians(rays * self.step - 180.0)
        distance = 5.0 + 20.0 * self.np_rng.random(len(rays))
        return np.stack(
            [distance * np.cos(yaw), distance * np.sin(yaw), channel, rays], axis=1
        ).astype(np.float32)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    ("rotation_frequency", "dt"),
    [
        (15.0, 0.05),  # 270 degrees a tick (#1122)
        (10.0, 0.05),  # 180, dividing 360
        (7.0, 0.05),  # 126
        (19.0, 0.05),  # 342
        (25.0, 1.0 / 30.0),  # 300
        (10.0, 1.0 / 30.0),  # 120, dividing 360
        (1.0, 0.05),  # 18
    ],
)
def test_a_carla_lidar_stream(seed: int, rotation_frequency: float, dt: float) -> None:
    """Jittered and long ticks, lost measurements and new sensors.

    Every revolution holds the rays measured from the first ray of the
    first arc that arrived (a sensor's first measurements may not), through
    360 degrees, each ray once.  An arc of a whole revolution or more, by
    the time since the measurement before (which a lost one lengthens), is
    one as it came, and the next revolution starts with the next arc.
    """
    rng = random.Random(seed)
    revolution = LidarRevolution(rotation_frequency, dt)
    per_second = 360.0 * rotation_frequency
    per_revolution = 720  # rays 0.5 degrees apart
    time = 0.0
    last = 0.0  # when the measurement before was taken
    sensor: Optional[_Sensor] = None
    start: Optional[int] = None  # the first ray of the current revolution
    pending: List[FloatPoints] = []
    emitted = 0
    for _ in range(400):
        kind = rng.random()
        if sensor is None or (kind < 0.02 and time > 5.0 * dt):
            # A new episode: a new sensor, the clock started over (so the
            # first measurement that arrives is not after the last one),
            # its first measurements lost or not.
            sensor = _Sensor(rotation_frequency, rng)
            time = 0.0
            start = None
            pending = []
            for _ in range(rng.choice([0, 0, 1, 3])):
                time += dt
                sensor.tick(dt)
            step = dt  # a whole tick, as first_dt says
            last = time
        elif kind < 0.06:
            step = dt * rng.uniform(1.0, 30.0)  # a long tick
        elif kind < 0.35:
            step = dt * rng.uniform(0.75, 1.25)  # jitter
        else:
            step = dt
        lost = start is not None and rng.random() < 0.1
        short = 360.0 - per_second * (time + step - last)
        if not lost and 0.0 < short < 2.0:
            # Within a few rays of a whole revolution, an arc's first ray
            # (where its revolution starts) and its last fall either side of
            # the cut: make it a whole revolution.
            step += (short + 1.0) / per_second
        time += step
        points = sensor.tick(step)
        if lost:
            continue  # lost on the way
        whole = revolution.add(time, points)
        swept, last = per_second * (time - last), time
        if swept + _REVOLUTION_TOLERANCE_DEG >= 360.0:
            assert whole is points
            pending = []
            start = None
            continue
        if start is None and len(points):
            start = int(points[:, 3].min())
        pending.append(points)
        assert start is not None  # three channels: no arc misses every ray
        if whole is None:
            assert sensor.angle < (start + per_revolution + 1) * 0.5
            continue
        emitted += 1
        assert sensor.angle >= (start + per_revolution - 1) * 0.5
        gathered = np.concatenate(pending)
        rays = gathered[:, 3].astype(np.int64)
        inside = (rays >= start) & (rays < start + per_revolution)
        want = gathered[inside]
        assert np.array_equal(
            whole[np.lexsort((whole[:, 3], whole[:, 2]))],
            want[np.lexsort((want[:, 3], want[:, 2]))],
        )
        pending = [gathered[~inside]]
        start += per_revolution
    assert emitted > 0


@pytest.mark.parametrize("lost", [1, 2, 3])
def test_revolutions_start_where_the_first_arc_that_arrives_does(lost: int) -> None:
    # CARLA turns a LiDAR from the tick it is spawned on; the measurements
    # taken before the bridge's listen() took hold never arrive.
    sensor = _Sensor(15.0, random.Random(lost), miss=0.0)
    for _ in range(lost):
        sensor.tick(0.05)
    revolution = LidarRevolution(15.0, 0.05)
    first = 540 * lost  # 270 degrees, 540 rays a tick
    emitted = []
    for tick in range(1, 9):
        whole = revolution.add(0.05 * tick, sensor.tick(0.05))
        if whole is not None:
            emitted.append(whole)
    assert len(emitted) == 6
    for index, whole in enumerate(emitted):
        for channel in range(3):
            rays = sorted(int(r) for r in whole[whole[:, 2] == channel, 3])
            begin = first + 720 * index
            assert rays == list(range(begin, begin + 720))


class _FovSensor:
    """A ray-cast LiDAR with a ``horizontal_fov``, as RayCastSemanticLidar.cpp
    turns it (#1125).

    Its horizontal angle grows by ``rotation_frequency * fov * dt`` a tick,
    modulo ``fov``; a ray at angle ``h`` points at yaw ``h - fov / 2``, so the
    rays cover azimuths [-fov/2, fov/2).  Rays are ``step`` degrees apart on
    multiples of ``step`` of the angle turned since the sensor was spawned;
    each point's intensity is that ray's index, its z the channel.
    """

    def __init__(
        self,
        rotation_frequency: float,
        fov: float,
        channels: int = 2,
        step: float = 0.5,
    ) -> None:
        self.per_second = fov * rotation_frequency
        self.fov = fov
        self.channels = channels
        self.step = step
        self.turned = 0.0  # unbounded: the angle is this modulo fov

    def tick(self, dt: float) -> FloatPoints:
        arc = self.per_second * dt
        first = math.ceil(self.turned / self.step - 1e-9)
        end = math.ceil((self.turned + arc) / self.step - 1e-9)
        self.turned += arc
        rays = np.tile(np.arange(first, end, dtype=np.float64), self.channels)
        channel = np.repeat(np.arange(self.channels, dtype=np.float64), end - first)
        yaw = np.radians(np.fmod(rays * self.step, self.fov) - 0.5 * self.fov)
        return np.stack(
            [10.0 * np.cos(yaw), 10.0 * np.sin(yaw), channel, rays], axis=1
        ).astype(np.float32)


@pytest.mark.parametrize("horizontal_fov", [-1.0, 0.0, 360.5, 720.0, math.nan])
def test_a_field_of_view_outside_a_revolution_is_refused(
    horizontal_fov: float,
) -> None:
    with pytest.raises(ValueError, match="horizontal_fov must be in"):
        LidarRevolution(10.0, 0.05, horizontal_fov)


@pytest.mark.parametrize("lost", [0, 1, 3])
@pytest.mark.parametrize(
    ("horizontal_fov", "rotation_frequency", "dt"),
    [
        (180.0, 15.0, 0.05),  # 135 degrees a tick
        (180.0, 10.0, 0.05),  # 90, dividing the field of view
        (120.0, 7.0, 0.05),  # 42
        (90.0, 19.0, 0.05),  # 85.5
        (270.0, 25.0, 1.0 / 30.0),  # 225
        (359.0, 15.0, 0.05),  # 269.25
    ],
)
def test_each_revolution_sweeps_the_field_of_view_once(
    horizontal_fov: float, rotation_frequency: float, dt: float, lost: int
) -> None:
    """A revolution is one sweep of ``horizontal_fov`` degrees, not 360."""
    sensor = _FovSensor(rotation_frequency, horizontal_fov)
    for _ in range(lost):  # measured before the bridge listened
        sensor.tick(dt)
    began = sensor.turned  # where the first arc to arrive starts
    first = math.ceil(began / sensor.step - 1e-9)  # its first ray
    per_revolution = int(round(horizontal_fov / sensor.step))
    revolution = LidarRevolution(rotation_frequency, dt, horizontal_fov)
    emitted: List[FloatPoints] = []
    for tick in range(1, 61):
        whole = revolution.add(dt * tick, sensor.tick(dt))
        if whole is not None:
            emitted.append(whole)
    assert len(emitted) == int((sensor.turned - began) / horizontal_fov + 1e-9)
    for index, whole in enumerate(emitted):
        begin = first + index * per_revolution
        for channel in range(sensor.channels):
            assert _rays(whole, channel) == list(range(begin, begin + per_revolution))


def test_an_arc_of_the_whole_field_of_view_is_one_revolution() -> None:
    # 180 degrees at 20 Hz, 20 Hz ticks: the field of view a tick.
    sensor = _FovSensor(20.0, 180.0)
    revolution = LidarRevolution(20.0, 0.05, 180.0)
    for tick in range(1, 5):
        points = sensor.tick(0.05)
        assert revolution.add(0.05 * tick, points) is points
