"""t4_settings, split out of t4 so the library check compiles it (#731): t4
still exports every name it did, and the F-theta lens fit, rewritten without
numpy.polynomial (which Codon's NumPy lacks), returns what it did."""

from __future__ import annotations

from typing import Any

import itertools

import numpy as np
import pytest

from autoware_carla_scenario import t4, t4_settings
from autoware_carla_scenario.driver.base import DriverCameraLens

from ._ftheta_lens import COSMOS3_FRONT_WIDE

#: What t4 exported before the split (its ``__all__``).
_T4_EXPORTS = {
    "DATASET_SUFFIX",
    "INSTALL_HINT",
    "LIDAR_CHANNEL",
    "T4Recorder",
    "T4Recording",
    "T4Settings",
    "T4UnavailableError",
    "active_recording",
    "camera_channel",
    "category_of",
    "dataset_path",
    "opencv_of_ftheta",
    "offer_driver_sensors",
    "require_writer",
    "withdraw_driver_sensors",
}


def test_t4_still_exports_every_name() -> None:
    assert set(t4.__all__) >= _T4_EXPORTS
    for name in _T4_EXPORTS:
        assert hasattr(t4, name), name


@pytest.mark.parametrize("name", t4_settings.__all__)
def test_t4_reexports_the_settings_module(name: str) -> None:
    assert name in t4.__all__
    assert getattr(t4, name) is getattr(t4_settings, name)


def _numpy_polynomial_fit(
    lens: DriverCameraLens, width: int, height: int, *, samples: int = 2000
) -> tuple[list[list[float]], list[float], float, float]:
    """t4.opencv_of_ftheta as it was before #731, on numpy.polynomial."""
    poly = np.polynomial.polynomial
    c, d, e = (float(v) for v in lens.linear_cde)
    if abs(e) > 1e-12:
        raise ValueError("an F-theta lens with linear_cde e != 0 has no OpenCV model")
    cx, cy = (float(v) for v in lens.principal_point)
    to_angle = np.asarray(lens.pixeldist_to_angle_poly, np.float64)
    to_radius = np.asarray(lens.angle_to_pixeldist_poly, np.float64)
    focal = float(poly.polyder(to_radius)[0])
    corners_x = np.array([-0.5, width - 0.5, -0.5, width - 0.5]) - cx
    corners_y = np.array([-0.5, -0.5, height - 0.5, height - 0.5]) - cy
    corner_radius = float(np.hypot((corners_x - d * corners_y) / c, corners_y).max())
    max_angle = min(float(lens.max_angle), np.radians(85.0))
    radius = min(corner_radius, float(poly.polyval(max_angle, to_radius)))
    reach = min(float(poly.polyval(radius, to_angle)), max_angle)
    theta = np.linspace(1e-4, reach, samples)
    r = np.tan(theta)
    r2 = r * r
    ratio = poly.polyval(theta, to_radius) / focal / r
    weight = (r * focal)[:, None]
    a = np.stack([r2, r2**2, r2**3, -ratio * r2, -ratio * r2**2, -ratio * r2**3], 1)
    k1, k2, k3, k4, k5, k6 = np.linalg.lstsq(
        a * weight, (ratio - 1.0) * weight[:, 0], rcond=None
    )[0]
    numerator = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    if not (denominator > 0.0).all():
        raise ValueError("no OpenCV rational model fits this F-theta lens")
    error = np.abs(focal * r * (numerator / denominator - ratio))
    intrinsic = [[c * focal, d * focal, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    distortion = [float(k) for k in (k1, k2, 0.0, 0.0, k3, k4, k5, k6)]
    return intrinsic, distortion, float(error.mean()), float(error.max())


#: Lenses, as changes to Cosmos3's front wide camera, and the image sizes they
#: are fitted at.
_LENSES: list[tuple[str, dict[str, Any], int, int, int]] = [
    ("cosmos3", {}, 832, 480, 2000),
    ("cosmos3-full-hd", {}, 1920, 1080, 2000),
    ("cosmos3-few-samples", {}, 832, 480, 50),
    ("narrow-max-angle", {"max_angle": 0.6}, 832, 480, 2000),
    (
        "column-skew",
        {"linear_cde": [0.998, 0.0125, 0.0]},
        832,
        480,
        2000,
    ),
    (
        "equidistant",
        {
            "pixeldist_to_angle_poly": [0.0, 1.0 / 500.0],
            "angle_to_pixeldist_poly": [0.0, 500.0],
            "max_angle": 1.4,
            "linear_cde": [1.0, 0.0, 0.0],
            "principal_point": [639.5, 359.5],
        },
        1280,
        720,
        2000,
    ),
    (
        "cubic",
        {
            "pixeldist_to_angle_poly": [0.0, 0.0031, 1.5e-7, -2.0e-10],
            "angle_to_pixeldist_poly": [0.0, 322.5, -4.1, -9.75],
            "max_angle": 1.05,
            "principal_point": [320.25, 241.75],
        },
        640,
        480,
        777,
    ),
    (
        "constant-term",
        {
            "pixeldist_to_angle_poly": [1e-6, 0.0024, -3.9e-8, 6.0e-10],
            "angle_to_pixeldist_poly": [0.02, 410.0, 2.5, -16.0, 10.0, -3.4],
        },
        832,
        480,
        2000,
    ),
]


@pytest.mark.parametrize(
    ("change", "width", "height", "samples"),
    [case[1:] for case in _LENSES],
    ids=[case[0] for case in _LENSES],
)
def test_the_lens_fit_is_the_numpy_polynomial_one(
    change: dict[str, Any], width: int, height: int, samples: int
) -> None:
    """The Horner helpers evaluate as numpy.polynomial.polynomial.polyval does
    (from the highest coefficient, starting at ``c[-1] + x * 0``) and
    ``polyder(c)[0]`` is ``1 * c[1]``, so every operation is the same and the
    fit is bit for bit the old one."""
    lens = DriverCameraLens.from_mapping({**COSMOS3_FRONT_WIDE, **change})
    try:
        expected = _numpy_polynomial_fit(lens, width, height, samples=samples)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc)):
            t4_settings.opencv_of_ftheta(lens, width, height, samples=samples)
        return
    # repr: bit for bit, the sign of a zero included.
    assert repr(t4_settings.opencv_of_ftheta(lens, width, height, samples=samples)) == (
        repr(expected)
    )


@pytest.mark.parametrize(
    "coef",
    [
        [3.0],
        [0.0, 411.286296],
        [0.5, -2.0, 0.25],
        COSMOS3_FRONT_WIDE["angle_to_pixeldist_poly"],
    ],
)
def test_the_horner_helpers_are_numpy_polyval(coef: list[float]) -> None:
    poly = np.polynomial.polynomial
    c = np.asarray(coef, np.float64)
    x = np.linspace(-1.5, 1.5, 101)
    for value in (*x[::10], 0.0, -0.0, 1.226783919):
        assert t4_settings._polyval_at(float(value), c) == float(poly.polyval(value, c))
    np.testing.assert_array_equal(t4_settings._polyval_on(x, c), poly.polyval(x, c))


#: Coefficients and arguments numpy answers with signed zeros, infinities and
#: NaNs.
_SPECIAL = [0.0, -0.0, 1.0, -2.5, 1e308, -1e-320, np.nan, np.inf, -np.inf]


def _bits(value: Any) -> bytes:
    """*value*'s bits, the sign of a zero included; a NaN's sign and payload,
    which the CPU picks, are not compared."""
    array = np.array(value, np.float64)
    array[np.isnan(array)] = np.nan
    return array.tobytes()


@pytest.mark.parametrize("length", [1, 2, 3])
def test_the_horner_helpers_are_numpy_polyval_bit_for_bit(length: int) -> None:
    poly = np.polynomial.polynomial
    x = np.asarray(_SPECIAL, np.float64)
    with np.errstate(all="ignore"):
        for coef in itertools.product(_SPECIAL, repeat=length):
            c = np.asarray(coef, np.float64)
            for value in (*_SPECIAL, np.float64(1.25)):
                assert _bits(t4_settings._polyval_at(value, c)) == _bits(
                    poly.polyval(value, c)
                ), (coef, value)
            on = t4_settings._polyval_on(x, c)
            assert on.dtype == np.float64
            assert _bits(on) == _bits(poly.polyval(x, c)), coef


@pytest.mark.parametrize("length", [1, 2, 3, 6])
def test_the_derivative_at_zero_is_numpy_polyder(length: int) -> None:
    poly = np.polynomial.polynomial
    with np.errstate(all="ignore"):
        for first in _SPECIAL:
            for second in _SPECIAL:
                c = np.asarray([first, second, 0.5, -1.0, 2.0, 3.0][:length])
                assert _bits(t4_settings._polyder_at_zero(c)) == _bits(
                    poly.polyder(c)[0]
                ), c
