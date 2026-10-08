"""Build driver observations from CARLA state.

CARLA's world is **left-handed** (x=East, y=South, z=Up, yaw clockwise) while the alpasim
contract is **right-handed** (x forward, y left, z up).  Everything that crosses that
boundary goes through this module, using the same convention the rest of this package
already documents in :mod:`autoware_carla_scenario.coordinate.transform`: flip ``y`` and
negate the yaw.

There is a second offset to reconcile: alpasim puts the rig origin at the rear-axle centre
projected to the ground, whereas a CARLA actor's origin sits at the vehicle centre.
:func:`rear_axle_offset` derives that shift, and
:attr:`~autoware_carla_scenario.driver.base.DriverClientConfig.rear_axle_offset_m`
overrides it when the derived value is wrong.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from numpy.typing import NDArray

from .base import EgoObservation
from .geometry import Pose, Trajectory

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

__all__ = [
    "AccelerationEstimate",
    "camera_extrinsics_to_rig",
    "encode_frame_jpeg",
    "ego_observation",
    "lidar_points_to_rig",
    "rear_axle_offset",
    "route_reference_trajectory",
    "route_waypoints_in_rig",
    "sensor_pose_in_rig",
    "to_local_pose",
    "to_local_vector",
]

#: Rear-axle offsets outside this magnitude (metres) are considered implausible and are
#: replaced by a bounding-box estimate.  CARLA 0.9.x reports wheel positions in world
#: coordinates and centimetres, which lands far outside this range.
_MAX_PLAUSIBLE_AXLE_OFFSET_M: float = 5.0

#: Fallback when a fork in the road offers no continuation within the horizon.
_MIN_ROUTE_POINTS: int = 2


def to_local_vector(x: float, y: float, z: float) -> NDArray[np.float64]:
    """Return a CARLA world *polar* vector in the right-handed local frame."""
    return np.array([x, -y, z], dtype=np.float64)


def _to_local_axial(x: float, y: float, z: float) -> NDArray[np.float64]:
    """Return a CARLA world *axial* vector (e.g. angular velocity) in the local frame.

    The handedness flip is a reflection, so pseudovectors pick up an extra sign change
    relative to :func:`to_local_vector`.
    """
    return np.array([-x, y, -z], dtype=np.float64)


def to_local_pose(
    transform: "carla.Transform", rear_axle_offset_m: float = 0.0
) -> Pose:
    """Return the right-handed local pose of a CARLA transform.

    Args:
        transform: CARLA actor transform.
        rear_axle_offset_m: Signed offset along the vehicle's forward axis from the
            actor origin to the rig origin.  Negative values move the origin backwards.
    """
    yaw = -math.radians(transform.rotation.yaw)
    pose = Pose.from_xyz_yaw(
        transform.location.x, -transform.location.y, transform.location.z, yaw
    )
    if rear_axle_offset_m == 0.0:
        return pose
    shift = Pose.from_xyz_yaw(rear_axle_offset_m, 0.0, 0.0, 0.0)
    return pose @ shift


def camera_extrinsics_to_rig(
    position_x: float,
    position_y: float,
    position_z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> Pose:
    """Return a camera's ``base_link -> camera`` extrinsics as a rig-frame pose.

    Camera extrinsics are configured in CARLA's convention (x forward, y right, z
    up, degrees), whereas the pose the contract advertises as ``rig_to_camera`` is
    right-handed (x forward, y left, z up).  Crossing that boundary is the same
    ``y`` reflection this module applies elsewhere: the translation's ``y`` flips,
    and CARLA's ``yaw * pitch * roll`` rotation, rebuilt in the right-handed frame,
    is the same rotation with the yaw and pitch angles negated and the roll kept.
    The yaw-only case therefore reduces to the "negate the yaw" rule
    :func:`to_local_pose` already uses.
    """
    rotation = (
        Pose.from_axis_angle(2, -math.radians(yaw_deg))
        @ Pose.from_axis_angle(1, -math.radians(pitch_deg))
        @ Pose.from_axis_angle(0, math.radians(roll_deg))
    )
    position = np.array([position_x, -position_y, position_z], dtype=np.float64)
    return Pose(position, rotation.quat_xyzw)


def sensor_pose_in_rig(
    position_x: float,
    position_y: float,
    position_z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    rear_axle_offset_m: float,
) -> Pose:
    """Return a sensor mount, relative to the vehicle actor, as a pose in the rig frame.

    :func:`camera_extrinsics_to_rig` reads the mount as if it were relative to the rig
    origin; this also moves it from the actor origin to the rig origin (the rear axle),
    which is what points carried into the rig frame need.
    """
    mount = camera_extrinsics_to_rig(
        position_x, position_y, position_z, roll_deg, pitch_deg, yaw_deg
    )
    return Pose.from_xyz_yaw(-rear_axle_offset_m, 0.0, 0.0, 0.0) @ mount


def lidar_points_to_rig(
    points_xyzi_in_sensor: NDArray[np.float32], pose_in_rig: Pose
) -> NDArray[np.float32]:
    """Return a raw ``sensor.lidar.ray_cast`` buffer as ``[N, 4]`` rig-frame points.

    CARLA reports the points in the sensor's own frame, which is left-handed like the
    rest of CARLA: mirroring y makes them right-handed in the sensor frame, and the
    mount (*pose_in_rig*, from :func:`sensor_pose_in_rig`) then carries them into the
    rig.  Intensity passes through untouched.
    """
    raw = np.asarray(points_xyzi_in_sensor, dtype=np.float32).reshape(-1, 4)
    # The mirror folded into the rotation, and float32 throughout: a sweep is up to
    # ~10^5-10^6 points, and float32 is centimetre-exact at LiDAR range.
    rotation = pose_in_rig.rotation_matrix * np.array([1.0, -1.0, 1.0])
    out = np.empty_like(raw)
    np.matmul(raw[:, :3], rotation.T.astype(np.float32), out=out[:, :3])
    out[:, :3] += pose_in_rig.position.astype(np.float32)
    out[:, 3] = raw[:, 3]
    return out


def rear_axle_offset(actor: "carla.Actor", override: Optional[float] = None) -> float:
    """Return the offset from the actor origin back to the rear axle, in metres.

    Args:
        actor: The vehicle actor.
        override: When not ``None``, returned as-is.

    Returns:
        A negative value (the rear axle is behind the vehicle centre), or ``0.0`` when
        neither the wheel physics nor the bounding box can be read.
    """
    if override is not None:
        return override

    try:
        physics = actor.get_physics_control()
        wheels = list(physics.wheels)
    except (AttributeError, RuntimeError, TypeError):
        # CARLA versions differ in what physics control exposes, and a vehicle
        # blueprint without wheel data is not an error worth aborting a scenario for.
        wheels = []

    # CARLA 0.9.x names a wheel's position ``position`` and reports it in world
    # coordinates and centimetres. CARLA 0.10 names it ``location`` and leaves it at
    # zero, which says nothing about where the axle is: the bounding box below is
    # all there is then.
    rear_points = [
        point
        for w in wheels[2:4]
        if (point := getattr(w, "position", None) or getattr(w, "location", None))
        is not None
    ]
    if (
        len(wheels) >= 4
        and all((point.x, point.y, point.z) != (0.0, 0.0, 0.0) for point in rear_points)
        and len(rear_points) == 2
    ):
        transform = actor.get_transform()
        world_cm = np.array(
            [[point.x, point.y, point.z] for point in rear_points], dtype=np.float64
        )
        local = (
            to_local_pose(transform)
            .inverse()
            .transform_points(
                np.stack([to_local_vector(*(point / 100.0)) for point in world_cm])
            )
        )
        offset = float(np.mean(local[:, 0]))
        if abs(offset) <= _MAX_PLAUSIBLE_AXLE_OFFSET_M:
            return offset
        logger.warning(
            "Derived rear-axle offset %.2f m is implausible; falling back to the "
            "bounding box. Set driver.rear_axle_offset_m to override.",
            offset,
        )

    try:
        return -0.5 * float(actor.bounding_box.extent.x)
    except (AttributeError, RuntimeError, TypeError):
        logger.warning("Could not determine the rear-axle offset; using 0.0")
        return 0.0


def ego_observation(
    actor: "carla.Actor", timestamp_us: int, rear_axle_offset_m: float = 0.0
) -> EgoObservation:
    """Return the ego's pose and dynamic state for one instant.

    The pose is in the local frame; velocities and accelerations are rotated into the
    rig frame, as the contract requires.

    Args:
        actor: The ego vehicle actor.
        timestamp_us: Simulation time of this observation.
        rear_axle_offset_m: Offset from the actor origin to the rig origin.
    """
    pose = to_local_pose(actor.get_transform(), rear_axle_offset_m)
    rotation = pose.rotation_matrix

    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    angular = actor.get_angular_velocity()

    linear_local = to_local_vector(velocity.x, velocity.y, velocity.z)
    accel_local = to_local_vector(acceleration.x, acceleration.y, acceleration.z)
    angular_local = np.radians(_to_local_axial(angular.x, angular.y, angular.z))

    return EgoObservation(
        timestamp_us=timestamp_us,
        pose=pose,
        linear_velocity=rotation.T @ linear_local,
        angular_velocity=rotation.T @ angular_local,
        linear_acceleration=rotation.T @ accel_local,
        speed_mps=float(np.linalg.norm(linear_local)),
    )


class AccelerationEstimate:
    """The ego's acceleration as Autoware's localization reports it.

    CARLA's ``get_acceleration`` is the physics step's own: a seam in the road
    reads as -2 to -6 m/s² for a tick.  A policy trained on Autoware's logs has
    seen ``twist2accel``'s instead -- the twist differentiated in the vehicle
    frame and low-passed, ``a = g * a + (1 - g) * dv / dt`` with ``g = 0.9`` on
    a 50 Hz twist -- and it plans the acceleration it is given onward, so a raw
    spike becomes a hard stop.  This applies that filter, its gain rescaled to
    the interval between observations so its time constant stays Autoware's.

    Args:
        gain: ``twist2accel``'s ``accel_lowpass_gain``.
        rate_hz: The rate that gain applies at.
    """

    def __init__(self, gain: float = 0.9, rate_hz: float = 50.0) -> None:
        self._gain = gain
        self._rate_hz = rate_hz
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next observation starts at zero acceleration."""
        self._last: Optional[EgoObservation] = None
        self._acceleration = np.zeros(3)

    def __call__(self, observation: EgoObservation) -> EgoObservation:
        """Return *observation* with its acceleration replaced by the estimate.

        Idempotent per instant: a second observation at the same time gets the
        same estimate.
        """
        last = self._last
        if last is not None and observation.timestamp_us > last.timestamp_us:
            dt_s = (observation.timestamp_us - last.timestamp_us) * 1e-6
            gain = self._gain ** (dt_s * self._rate_hz)
            rate = (observation.linear_velocity - last.linear_velocity) / dt_s
            self._acceleration = gain * self._acceleration + (1.0 - gain) * rate
        if last is None or observation.timestamp_us > last.timestamp_us:
            self._last = observation
        return dataclasses.replace(
            observation, linear_acceleration=self._acceleration.copy()
        )


def encode_frame_jpeg(
    image_bgr: NDArray[np.uint8], quality: int = 90
) -> Optional[bytes]:
    """Return *image_bgr* encoded as JPEG bytes.

    Args:
        image_bgr: ``HxWx3`` BGR array, as produced by
            :meth:`~autoware_carla_scenario.sensor.carla_camera.CarlaCameraSensor.get_image`.
        quality: JPEG quality, 1-100.

    Returns:
        The encoded bytes, or ``None`` if the encoder failed.
    """
    import cv2  # noqa: PLC0415

    success, buffer = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not success:
        logger.warning("JPEG encoding failed for a %s frame", image_bgr.shape)
        return None
    return buffer.tobytes()


def route_waypoints_in_rig(
    carla_map: "carla.Map",
    actor: "carla.Actor",
    ego_pose: Pose,
    horizon_m: float = 80.0,
    resolution_m: float = 2.0,
) -> NDArray[np.float64]:
    """Return the road ahead of *actor* as waypoints in the rig frame.

    Walks CARLA's road graph forward from the ego's current waypoint, taking the first
    continuation at each fork.  This gives the policy a lane-following route without
    requiring a global planner; scenarios that need a specific route can build their own
    waypoint array and submit it directly.

    Args:
        carla_map: The map to walk.  Pass a cached ``world.get_map()`` -- CARLA rebuilds
            the map object on every call, which is far too costly at policy rate.
        actor: The ego vehicle actor.
        ego_pose: The ego's pose in the local frame, used to convert into the rig frame.
        horizon_m: How far ahead to walk.
        resolution_m: Spacing between waypoints.

    Returns:
        An ``(N, 3)`` array in the rig frame.  Empty when the ego is off-road.
    """
    waypoint = carla_map.get_waypoint(actor.get_location())
    if waypoint is None:
        logger.warning("Ego is not on a drivable waypoint; sending an empty route")
        return np.zeros((0, 3))

    step = max(resolution_m, 0.1)
    points: List[NDArray[np.float64]] = []
    current = waypoint
    travelled = 0.0
    while travelled < horizon_m:
        location = current.transform.location
        points.append(to_local_vector(location.x, location.y, location.z))
        following = current.next(step)
        if not following:
            break
        current = following[0]
        travelled += step

    if len(points) < _MIN_ROUTE_POINTS:
        logger.warning("Route walk produced %d point(s) only", len(points))
    if not points:
        return np.zeros((0, 3))
    return ego_pose.inverse().transform_points(np.stack(points))


def route_reference_trajectory(
    waypoints_in_rig: NDArray[np.float64], start_us: int, step_us: int
) -> Trajectory:
    """Return the route as a timed reference trajectory, for ``GroundTruth``.

    CARLA has no recorded drive to replay, so the route stands in for one: each
    waypoint becomes a pose headed along the path (the last one keeps the heading of
    the segment before it), and consecutive poses are *step_us* apart.

    Args:
        waypoints_in_rig: ``(N, 3)`` route waypoints in the rig frame.
        start_us: Timestamp of the first pose.
        step_us: Time between consecutive poses; must be positive.

    Returns:
        A trajectory in the rig frame, empty when *waypoints_in_rig* is.
    """
    points = np.asarray(waypoints_in_rig, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return Trajectory.empty()
    deltas = np.diff(points[:, :2], axis=0)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0]) if len(deltas) else np.zeros(1)
    yaws = np.append(yaws, yaws[-1])[: len(points)]
    return Trajectory(
        [start_us + index * step_us for index in range(len(points))],
        [
            Pose.from_xyz_yaw(point[0], point[1], point[2], float(yaw))
            for point, yaw in zip(points, yaws)
        ],
    )
