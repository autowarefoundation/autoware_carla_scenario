"""What a run's T4 dataset holds: its settings, names and categories, and the
OpenCV model of an F-theta camera.

:mod:`autoware_carla_scenario.t4` writes a run's T4 dataset through a Rust
writer it imports at run time, on threads of its own; this is the part of it
that needs neither: the settings (:class:`T4Settings`), where the dataset
goes (:func:`dataset_path`), a camera's channel (:func:`camera_channel`), an
actor's category (:func:`category_of`) and the fit of OpenCV's rational
camera model to an F-theta lens (:func:`opencv_of_ftheta`).  ``t4``
re-exports every name, and the library check compiles this module
(docs/typecheck.md, #731).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .constants import tick_multiple
from .driver.base import DriverCameraLens, DriverClientConfig
from .sensor.carla_camera import CarlaCameraSensorConfig
from .sensor.carla_lidar import CarlaLidarSensorConfig

__all__ = [
    "DATASET_SUFFIX",
    "LIDAR_CHANNEL",
    "T4Settings",
    "camera_channel",
    "category_of",
    "dataset_path",
    "opencv_of_ftheta",
]

#: What a run's dataset is named: ``{stem}.t4``, its one version ``0`` inside.
DATASET_SUFFIX = ".t4"
#: The channel every LiDAR's sweep goes into, in the ego's frame (T4's own
#: name for a concatenated cloud).
LIDAR_CHANNEL = "LIDAR_CONCAT"
#: A vehicle blueprint's ``base_type`` -> its T4 category.
_VEHICLE_CATEGORIES = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


def dataset_path(directory: Path, stem: str) -> Path:
    """``{directory}/{stem}.t4``: the run's dataset (its version ``0`` inside)."""
    return Path(directory) / f"{stem}{DATASET_SUFFIX}"


def camera_channel(logical_id: str) -> str:
    """The T4 channel of the driver camera *logical_id*: upper-cased, as T4's
    channels are (``camera_front_wide_120fov`` -> ``CAMERA_FRONT_WIDE_120FOV``)."""
    return logical_id.upper()


@dataclass(frozen=True)
class T4Settings:
    """What a run's dataset holds, and how often: the driver policy's sensors
    and step, and the ``t4`` config."""

    #: The LiDARs, as ``driver.lidars`` (their ``logical_id`` is not used:
    #: every sweep goes into :data:`LIDAR_CHANNEL`).
    lidars: tuple[CarlaLidarSensorConfig, ...] = ()
    #: The cameras, ``(logical id, sensor)``, as ``driver.cameras``.
    cameras: tuple[tuple[str, CarlaCameraSensorConfig], ...] = ()
    #: The cameras whose frames are drawn through a lens of their own, not
    #: CARLA's pinhole: ``(logical id, lens)``, as ``driver.cameras[].lens``.
    lenses: tuple[tuple[str, DriverCameraLens], ...] = ()
    #: Seconds between key frames, ``driver.policy_timestep_s``.
    timestep_s: float = 0.1
    #: The JPEG quality of the frames the recorder encodes itself
    #: (``driver.image_quality``, which the policy's frames have too).
    jpeg_quality: int = 90
    #: The rig origin's offset from the actor, ``driver.rear_axle_offset_m``;
    #: None derives it from the vehicle's wheels.  Used only when the running
    #: ego set no base_link of its own (``ego_base_link``), as a ROS 2 or a
    #: bridged stack's ego and a driver entity do.
    rear_axle_offset_m: Optional[float] = None
    #: Annotate the road users within this distance of the ego; None for all.
    max_distance_m: Optional[float] = 100.0
    #: Threads writing the files and counting the boxes' points.
    workers: int = 4

    def __post_init__(self) -> None:
        if not self.lidars and not self.cameras:
            raise ValueError(
                "t4: a dataset needs a sensor: set driver.lidars or driver.cameras"
            )
        self.interval_ticks  # noqa: B018 - validates timestep_s
        if self.max_distance_m is not None and self.max_distance_m <= 0.0:
            raise ValueError("t4.max_distance_m must be positive (or null)")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("driver.image_quality must be 1-100")
        if self.workers < 1:
            raise ValueError("t4.workers must be positive")

    @property
    def interval_ticks(self) -> int:
        """World ticks from one key frame to the next."""
        return tick_multiple(self.timestep_s, "driver.policy_timestep_s")

    @classmethod
    def from_driver(
        cls, driver: DriverClientConfig, config: Mapping[str, Any]
    ) -> T4Settings:
        """The settings of the driver policy's *driver* config and a ``t4``
        config section (``max_distance_m``, ``workers``)."""
        values: dict[str, Any] = {}
        if "max_distance_m" in config:
            distance = config["max_distance_m"]
            values["max_distance_m"] = None if distance is None else float(distance)
        if config.get("workers") is not None:
            values["workers"] = int(config["workers"])
        return cls(
            lidars=tuple(driver.lidars),
            cameras=tuple(
                (camera.logical_id, camera.to_sensor_config())
                for camera in driver.cameras
            ),
            lenses=tuple(
                (camera.logical_id, camera.lens)
                for camera in driver.cameras
                if camera.lens is not None
            ),
            timestep_s=float(driver.policy_timestep_s),
            jpeg_quality=int(driver.image_quality),
            rear_axle_offset_m=driver.rear_axle_offset_m,
            **values,
        )


#: The largest angle from the optical axis a pinhole model, OpenCV's rational
#: one included, can hold: it projects through tan(theta).
_PINHOLE_MAX_ANGLE_RAD = np.radians(85.0)


# numpy.polynomial, which Codon's NumPy lacks, evaluated as its polyval()
# does (Horner's scheme from the highest coefficient, starting at
# ``c[-1] + x * 0``), so the fit's every operation, and so its result, is
# unchanged (test_t4_settings.py compares the two).


def _polyval_at(x: float, coef: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    """``numpy.polynomial.polynomial.polyval(x, coef)`` at a scalar *x*."""
    value = 0.0
    for k in range(len(coef) - 1, -1, -1):
        value = float(coef[k]) + value * x
    return value


def _polyval_on(
    x: np.ndarray[tuple[int], np.dtype[np.float64]],
    coef: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """``numpy.polynomial.polynomial.polyval(x, coef)`` on an array *x*."""
    value = np.zeros(len(x))
    for k in range(len(coef) - 1, -1, -1):
        value = float(coef[k]) + value * x
    return value


def _polyder_at_zero(coef: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
    """``numpy.polynomial.polynomial.polyder(coef)[0]``, the derivative at 0:
    ``1 * coef[1]``, or ``coef[0] * 0`` of a constant (``-0.0`` of a negative
    one, NaN of an infinite one), as numpy answers."""
    if len(coef) > 1:
        return float(coef[1])
    return float(coef[0]) * 0.0


def opencv_of_ftheta(
    lens: DriverCameraLens, width: int, height: int, *, samples: int = 2000
) -> tuple[list[list[float]], list[float], float, float]:
    """OpenCV's rational camera model fitted to the F-theta *lens* of a
    *width* x *height* image: T4's cameras are OpenCV's (``camera_intrinsic``
    and the ``camera_distortion`` ``[k1, k2, p1, p2, k3, k4, k5, k6]``), and
    have no F-theta model.

    In normalised coordinates both models are radial, so the fit is of one
    curve: the distorted radius ``angle_to_pixeldist_poly(theta) / f`` against
    the pinhole's ``tan(theta)``, ``f`` being the lens' pixels per radian at
    its centre (the focal lengths: ``linear_cde``'s ``c`` and ``d`` scale and
    skew the columns). The radial terms are fitted by least squares, linear
    in them, weighted to pixels, at angles out to the image's farthest corner
    within the lens (``max_angle``, and 85 degrees, short of a pinhole's 90).

    Returns:
        ``(intrinsic, distortion, mean error, max error)``: the errors are the
        pixels between the fitted model and the lens along that curve.

    Raises:
        ValueError: If the lens is skewed in its rows (``linear_cde``'s ``e``
            is not 0), which OpenCV's intrinsic matrix cannot hold, or no
            rational model fits it (its denominator would pass through 0).
    """
    # DriverCameraLens holds three of linear_cde and two of principal_point.
    c, d, e = (
        float(lens.linear_cde[0]),
        float(lens.linear_cde[1]),
        float(lens.linear_cde[2]),
    )
    if abs(e) > 1e-12:
        raise ValueError("an F-theta lens with linear_cde e != 0 has no OpenCV model")
    cx, cy = float(lens.principal_point[0]), float(lens.principal_point[1])
    to_angle = np.asarray(lens.pixeldist_to_angle_poly, np.float64)
    to_radius = np.asarray(lens.angle_to_pixeldist_poly, np.float64)
    focal = _polyder_at_zero(to_radius)

    # The farthest the image reaches from the principal point, as a radius in
    # the lens (linear_cde undone), and its angle, within the lens.
    corners_x = np.array([-0.5, width - 0.5, -0.5, width - 0.5]) - cx
    corners_y = np.array([-0.5, -0.5, height - 0.5, height - 0.5]) - cy
    corner_radius = float(np.hypot((corners_x - d * corners_y) / c, corners_y).max())
    max_angle = min(float(lens.max_angle), _PINHOLE_MAX_ANGLE_RAD)
    radius = min(corner_radius, _polyval_at(max_angle, to_radius))
    reach = min(_polyval_at(radius, to_angle), max_angle)

    theta = np.linspace(1e-4, reach, samples)
    r = np.tan(theta)
    r2 = r * r
    ratio = _polyval_on(theta, to_radius) / focal / r
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


def category_of(type_id: str, base_type: str = "", wheels: int = 0) -> str:
    """The T4 category of a CARLA blueprint: ``pedestrian`` for a walker; a
    vehicle's from its ``base_type`` (``car``, ``truck``, ``bus``,
    ``motorcycle``, ``bicycle``), else ``motorcycle`` on two wheels, else
    ``car``."""
    if type_id.startswith("walker."):
        return "pedestrian"
    category = _VEHICLE_CATEGORIES.get(base_type.lower())
    if category is not None:
        return category
    return "motorcycle" if wheels == 2 else "car"
