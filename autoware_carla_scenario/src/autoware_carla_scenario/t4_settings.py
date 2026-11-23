"""What a run's T4 dataset holds: its settings, names and categories, and the
OpenCV model of an F-theta camera.

:mod:`autoware_carla_scenario.t4` writes a run's T4 dataset through a Rust
writer it imports at run time, on threads of its own; this is the part of it
that needs neither: the settings (:class:`T4Settings`), where the dataset
goes (:func:`dataset_path`), a camera's channel (:func:`camera_channel`), an
actor's category (:func:`category_of`), a sweep's points on T4's intensity
scale (:func:`t4_lidar_points`) and the fit of OpenCV's rational camera model
to an F-theta lens (:func:`opencv_of_ftheta`), and which of what a driver
entity fed is written, on which key frame (:func:`fed_key_frame`).  ``t4``
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
    "FED_KEPT",
    "FED_LATER",
    "FED_NOT_NEW",
    "FED_NO_SNAPSHOT",
    "FED_REPEATED",
    "INTENSITY_SCALE",
    "LIDAR_CHANNEL",
    "T4Settings",
    "camera_channel",
    "category_of",
    "dataset_path",
    "fed_key_frame",
    "opencv_of_ftheta",
    "t4_lidar_points",
]

#: What a run's dataset is named: ``{stem}.t4``, its one version ``0`` inside.
DATASET_SUFFIX = ".t4"
#: The channel every LiDAR's sweep goes into, in the ego's frame (T4's own
#: name for a concatenated cloud).
LIDAR_CHANNEL = "LIDAR_CONCAT"
#: A T4 sweep's intensity is on the 0-255 scale (as Autoware's point clouds'
#: and the T4Scenes' are); CARLA's ray-cast LiDAR reports it in ``[0, 1]``.
INTENSITY_SCALE = 255.0
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


def _checked_max_distance(value: Optional[float]) -> Optional[float]:
    """``t4.max_distance_m`` = *value* as a float, or ``None`` (every road
    user): 0, a negative distance or NaN would annotate no road user,
    infinity is what null says, and a bool or a string is not a distance
    (:func:`~autoware_carla_scenario.t4_scene.grid.checked_max_distance`,
    the T4Scenes' rule, naming the key)."""
    if value is None:
        return None
    # Inside: importing t4_scene.grid runs the t4_scene package, which loads
    # its converter (protobuf, Lanelet2), more than the settings need, and
    # this module is imported by every run that may write a dataset (t4).
    from .t4_scene.grid import checked_max_distance  # noqa: PLC0415

    return checked_max_distance(value, "t4.max_distance_m", "null")


def t4_lidar_points(
    points: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """``[N, 4]`` x, y, z and CARLA's intensity in ``[0, 1]`` as a T4 sweep's
    first four columns: float32, intensity on T4's 0-255 scale
    (:data:`INTENSITY_SCALE`).  A new array: *points* (the sweep a driver
    policy is sent too) is left as it is.

    The T4 dataset (``t4``) writes its sweeps with it and the live T4Scenes
    (``t4_scene.live``) theirs, so the two scale alike, and once.
    """
    out = np.asarray(points, dtype=np.float32).reshape(-1, 4).copy()
    out[:, 3] = out[:, 3] * np.float32(INTENSITY_SCALE)
    return out


@dataclass(frozen=True)
class T4Settings:
    """What a run's dataset holds, and how often: the driver policy's sensors
    and step, and the ``t4`` config."""

    #: The LiDARs, as ``driver.lidars`` (their ``logical_id`` only keeps their
    #: weather noise apart: every sweep goes into :data:`LIDAR_CHANNEL`).
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
    #: None derives it from the vehicle's wheels, else its box.  Used only when the running
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
        _checked_max_distance(self.max_distance_m)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("driver.image_quality must be 1-100")
        # A bool is an int to Python and a float would be cut to one
        # (``true`` -> 1, ``2.9`` -> 2): neither is a thread count.
        workers = self.workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"t4.workers must be a positive integer, not {workers!r}")
        self._check_lenses()

    def _check_lenses(self) -> None:
        """Refuse a camera's F-theta lens the dataset cannot hold: one with no
        OpenCV model (:func:`opencv_of_ftheta`), which the writer would
        otherwise refuse only as each run's dataset starts (#1686).  A lens
        whose logical id names no camera is not written, so not checked."""
        for logical_id, lens in self.lenses:
            for camera_id, cam in self.cameras:
                if camera_id != logical_id:
                    continue
                try:
                    opencv_of_ftheta(lens, cam.image_width, cam.image_height)
                except ValueError as error:
                    raise ValueError(
                        f"t4: the lens of camera '{logical_id}' cannot be written "
                        f"to a T4 dataset: {error}"
                    ) from None

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
            # float() would read true as 1 m and parse a string; neither is a
            # distance, and anything else it cannot read is refused naming
            # the key rather than with its own message.
            values["max_distance_m"] = _checked_max_distance(config["max_distance_m"])
        if config.get("workers") is not None:
            # As given: __post_init__ refuses a bool or a float, which int()
            # would turn into a count nobody asked for.
            values["workers"] = config["workers"]
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


#: What :func:`fed_key_frame` makes of a fed sweep or camera frame: written on
#: the key frame.
FED_KEPT = 0
#: A sweep, new by :data:`FED_NOT_NEW`, whose frame has no snapshot: dropped.
FED_NO_SNAPSHOT = 1
#: A sweep or camera frame from before the recording, or not after the key
#: frame written last -- a sweep a driver entity feeds again on every tick
#: until the LiDAR's next one (#2362): dropped.
FED_NOT_NEW = 2
#: A camera frame not after that camera's frame written last, which a camera
#: slower than the policy hands it again (#2352): dropped.
FED_REPEATED = 3
#: A sweep taken after the key frame: it joins nothing, as a LiDAR's on another
#: tick of the interval does not, and is dropped.
FED_LATER = 4


def fed_key_frame(
    frame: int,
    sweep_frames: list[int],
    image_frames: list[tuple[int, str]],
    snapshots: set[int],
    start_frame: int,
    last_frame: int,
    camera_written: dict[str, int],
) -> tuple[list[int], list[int], int]:
    """Which of what a driver entity fed on *frame* is written, and on which
    key frame (``t4``'s ``_record_fed``, #951, #2352, #2362).

    *sweep_frames* are the frames the fed sweeps were taken on,
    *image_frames* the fed camera frames' ``(frame rendered on, logical
    id)``, *snapshots* the frames the recording holds a world snapshot of,
    *start_frame* the recording's first frame, *last_frame* the key frame
    written last (``-1`` for none) and *camera_written* the frame each
    camera's frame written last was rendered on, by logical id.

    Returns each sweep's and each camera frame's fate (``FED_*``, in the
    order fed) and the key frame, ``-1`` when nothing is written.  The key
    frame is the oldest kept sweep's; without one, the oldest kept camera
    frame's that has a snapshot (*frame* when none has).  Every kept camera
    frame goes on it, a sweep only when taken on it.  Nothing is written when
    nothing is kept (no fate is ``FED_KEPT``), nor when the key frame is not
    after *last_frame*.
    """
    sweep_fates: list[int] = []
    oldest_sweep = -1
    for taken_on in sweep_frames:
        if taken_on < start_frame or taken_on <= last_frame:
            sweep_fates.append(FED_NOT_NEW)
            continue
        if taken_on not in snapshots:
            sweep_fates.append(FED_NO_SNAPSHOT)
            continue
        sweep_fates.append(FED_KEPT)
        if oldest_sweep < 0 or taken_on < oldest_sweep:
            oldest_sweep = taken_on
    image_fates: list[int] = []
    any_image = False
    oldest_image = -1
    for taken_on, logical_id in image_frames:
        if taken_on < start_frame or taken_on <= last_frame:
            image_fates.append(FED_NOT_NEW)
        elif logical_id in camera_written and taken_on <= camera_written[logical_id]:
            image_fates.append(FED_REPEATED)
        else:
            image_fates.append(FED_KEPT)
            any_image = True
            if taken_on in snapshots and (oldest_image < 0 or taken_on < oldest_image):
                oldest_image = taken_on
    if oldest_sweep < 0 and not any_image:
        return sweep_fates, image_fates, -1
    if oldest_sweep >= 0:
        key_frame = oldest_sweep
        for index, taken_on in enumerate(sweep_frames):
            if sweep_fates[index] == FED_KEPT and taken_on != key_frame:
                sweep_fates[index] = FED_LATER
    else:
        key_frame = oldest_image if oldest_image >= 0 else frame
    if key_frame <= last_frame:
        return sweep_fates, image_fates, -1
    return sweep_fates, image_fates, key_frame
