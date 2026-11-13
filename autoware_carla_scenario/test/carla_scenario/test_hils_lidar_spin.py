"""What hils.measurements makes of a revolution's ray times, start and spin.

``_old_ray_times``, ``_old_steady_revolution_start`` and
``_old_spin_lidar_points`` are ``hils.measurements``' functions as they were
before they moved into the library check (#2116), copied verbatim.  The
functions the module exports now must give the same results bit for bit
(the same dtype, the same shape and the same bytes, NaN included) on inputs
of the types their callers hand them: ``hils.bridge``'s ``[N, 5]`` float32
revolution (CARLA's ``[N, 4]`` points tagged with their arc) and float64
measuring times and motion, CARLA's ``[N, 4]`` float32 points, and the
float64 points the tests of ``test_hils_lidar_revolution.py`` hand them.
"""

from __future__ import annotations

import math
import struct
from typing import Callable, Optional

import numpy as np
import pytest

from autoware_carla_scenario.hils import measurements


def _old_ray_times(points: np.ndarray, period: float) -> np.ndarray:
    """Seconds after a revolution's start a turning sensor fires each point's ray.

    hilspy dates the ray of azimuth ``atan2(y, x)`` (CARLA's sensor frame,
    clockwise from forward) ``azimuth / 360`` of *period* after the scan's
    timestamp, as the sensor fires it.
    """
    y, x = points[:, 1].astype(np.float64), points[:, 0].astype(np.float64)
    return np.mod(np.arctan2(y, x), 2.0 * np.pi) * (period / (2.0 * np.pi))


def _old_steady_revolution_start(
    measured: float, previous: float | None, period: float, gain: float = 0.2
) -> float:
    """When a revolution measured to start at *measured* is dated, *previous*
    being the last one's start.

    A turning sensor starts each revolution one *period* after the last, and
    its packets' times never step back.  CARLA's ticks come a few ms early or
    late on the wall clock, so the measured starts jitter; dated as they are,
    a revolution can start before the last one's packets end, which a viewer
    such as PandarView shows as a missing wedge.  The start therefore follows
    the period and only drifts later toward the measured one (*gain* of the
    difference): CARLA ticks no faster than the wall clock, so the measured
    starts do not run ahead for long.  Where one is a whole period off (a
    revolution missed, a stall) the stream restarts on it.  The points are
    re-observed for the start chosen (:func:`spin_lidar_points`), so the
    cloud stays right.
    """
    if previous is None:
        return measured
    expected = previous + period
    error = measured - expected
    if abs(error) >= period:
        return measured
    return expected + max(0.0, gain * error)


def _old_spin_lidar_points(
    points: np.ndarray,
    measured: np.ndarray,
    motion: np.ndarray,
    timestamp: float,
    rays: np.ndarray,
) -> np.ndarray:
    """Re-observe a revolution's points as a spinning sensor would have seen them.

    CARLA casts each tick's arc at once, from where the sensor was then
    (*measured*, per point); a turning sensor fires each ray at
    ``timestamp + rays`` (:func:`ray_times`).  Each point moves, with the
    sensor's *motion* then held constant (per point ``vx, vy, vz`` in m/s and
    ``wx, wy, wz`` in rad/s, CARLA's sensor frame, whose angular velocity
    follows the left-hand rule: ``wz`` turns x toward y), to where the sensor
    would have seen it at its ray's time -- small angles, first order -- so a
    driver correcting the motion over each point's time gets back a cloud of
    one instant.  *points* holds CARLA's ``x, y, z, intensity`` in its first
    four columns, ``[N, 4]`` or wider (the bridge passes each point's arc as a
    fifth); the result is ``[N, 4]``, those four.
    """
    # Differences first: epoch seconds do not fit a float32 (CARLA's points are).
    dt = (timestamp - np.asarray(measured, np.float64)) + np.asarray(rays, np.float64)
    vx, vy, vz, wx, wy, wz = (motion[:, k] * dt for k in range(6))
    a = points[:, 0] - vx
    b = points[:, 1] - vy
    c = points[:, 2] - vz
    out = np.empty((len(points), 4), np.float32)
    out[:, 0] = a - wy * c + wz * b
    out[:, 1] = b - wz * a + wx * c
    out[:, 2] = c - wx * b + wy * a
    out[:, 3] = points[:, 3]
    return out


def _assert_same(new: np.ndarray, old: np.ndarray) -> None:
    """The same array, bit for bit: dtype, shape and bytes (NaN's too)."""
    assert new.dtype == old.dtype
    assert new.shape == old.shape
    assert np.ascontiguousarray(new).tobytes() == np.ascontiguousarray(old).tobytes()


def _special(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: type
) -> np.ndarray:
    """Values of *dtype* with NaN, infinities, signed zeros and extremes among them."""
    values: np.ndarray = rng.normal(0.0, 30.0, shape).astype(dtype)
    pool: np.ndarray = np.array(
        [
            np.nan,
            np.inf,
            -np.inf,
            0.0,
            -0.0,
            1e-30,
            -1e-30,
            np.finfo(dtype).max,
            np.finfo(dtype).tiny,
        ],
        dtype,
    )
    mask = rng.random(shape) < 0.15
    values[mask] = rng.choice(pool, int(mask.sum()))
    return values


def _revolution(
    rng: np.random.Generator, n: int, columns: int, dtype: type, special: bool
) -> np.ndarray:
    if special:
        return _special(rng, (n, columns), dtype)
    points: np.ndarray = rng.uniform(-80.0, 80.0, (n, columns)).astype(dtype)
    if columns > 4:
        # The arc each point was measured in, as hils.bridge tags it.
        points[:, 4] = rng.integers(0, 12, n).astype(dtype)
    return points


def _motion(rng: np.random.Generator, n: int, special: bool) -> np.ndarray:
    if special:
        return _special(rng, (n, 6), np.float64)
    return np.column_stack(
        [rng.normal(0.0, 15.0, (n, 3)), rng.normal(0.0, 0.8, (n, 3))]
    )


_CASES = [
    (seed, n, columns, dtype, special)
    for seed, n in enumerate([0, 1, 2, 7, 64, 1000, 4096])
    for columns, dtype in ((5, np.float32), (4, np.float32), (4, np.float64))
    for special in (False, True)
]


# NaN, infinities and extremes overflow and compare as numpy does.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(("seed", "n", "columns", "dtype", "special"), _CASES)
def test_ray_times_and_spin_are_bit_identical(
    seed: int, n: int, columns: int, dtype: type, special: bool
) -> None:
    rng = np.random.default_rng(1000 + seed)
    points = _revolution(rng, n, columns, dtype, special)
    now = 1794402883.08 + float(rng.uniform(0.0, 1e5))
    measured = now + rng.uniform(-0.2, 0.2, n)
    if special and n:
        measured[rng.random(n) < 0.1] = np.nan
    motion = _motion(rng, n, special)
    for period in (0.1, 0.05, 1.0 / 3.0):
        rays = measurements.ray_times(points, period)
        old_rays = _old_ray_times(points, period)
        _assert_same(rays, old_rays)
        start = now + float(rng.uniform(-0.1, 0.1))
        _assert_same(
            measurements.spin_lidar_points(points, measured, motion, start, rays),
            _old_spin_lidar_points(points, measured, motion, start, old_rays),
        )


def test_spin_with_more_motion_columns_reads_the_first_six() -> None:
    rng = np.random.default_rng(5)
    points = _revolution(rng, 9, 5, np.float32, False)
    measured = 1794402883.08 + rng.uniform(-0.2, 0.2, 9)
    motion = rng.normal(0.0, 5.0, (9, 8))
    rays = _old_ray_times(points, 0.1)
    _assert_same(
        measurements.spin_lidar_points(points, measured, motion, 1794402883.1, rays),
        _old_spin_lidar_points(points, measured, motion, 1794402883.1, rays),
    )


def _raised(call: Callable[[], object]) -> tuple[type, str]:
    with pytest.raises(Exception) as raised:
        call()
    return type(raised.value), str(raised.value)


@pytest.mark.parametrize(
    "motion",
    [
        np.zeros((3, 5)),  # too few columns
        np.zeros((3,)),  # one dimension
        np.zeros((0,)),  # hils.bridge's motion of no arcs
    ],
)
def test_spin_raises_as_before_on_a_motion_of_another_shape(
    motion: np.ndarray,
) -> None:
    points = np.ones((3, 5), np.float32)
    measured = np.zeros(3)
    rays = _old_ray_times(points, 0.1)
    assert _raised(
        lambda: measurements.spin_lidar_points(points, measured, motion, 0.0, rays)
    ) == _raised(lambda: _old_spin_lidar_points(points, measured, motion, 0.0, rays))


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def test_steady_revolution_start_is_bit_identical() -> None:
    rng = np.random.default_rng(11)
    specials = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1e-300, 1e300]
    for _ in range(20000):
        now = 1794402883.08 + float(rng.uniform(0.0, 1e5))
        period = float(rng.choice([0.1, 0.05, 1.0 / 3.0, 0.0, -0.1]))
        measured = now + float(rng.normal(0.0, abs(period) or 0.1))
        previous: Optional[float] = (
            None if rng.random() < 0.1 else now - period + float(rng.normal(0.0, 0.01))
        )
        gain = float(rng.choice([0.2, 0.0, 1.0, 0.5]))
        if rng.random() < 0.1:
            measured = float(rng.choice(specials))
        if previous is not None and rng.random() < 0.05:
            previous = float(rng.choice(specials))
        if rng.random() < 0.05:
            period = float(rng.choice(specials))
        new = measurements.steady_revolution_start(measured, previous, period, gain)
        old = _old_steady_revolution_start(measured, previous, period, gain)
        assert type(new) is type(old)
        assert _bits(new) == _bits(old)
        # The default gain.
        assert _bits(
            measurements.steady_revolution_start(measured, previous, period)
        ) == _bits(_old_steady_revolution_start(measured, previous, period))


def test_measurements_re_exports_lidar_spin_s_functions() -> None:
    from autoware_carla_scenario.hils import lidar_spin

    assert measurements.ray_times is lidar_spin.ray_times
    assert measurements.spin_lidar_points is lidar_spin.spin_lidar_points
    assert measurements.steady_revolution_start is lidar_spin.steady_revolution_start
    assert "ray_times" in measurements.__all__
    assert "spin_lidar_points" in measurements.__all__
