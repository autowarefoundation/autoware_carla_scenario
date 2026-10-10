"""Place a :class:`~.model.Trajectory` in CARLA world coordinates.

Kept apart from :mod:`.model` because this is the half that needs the loaded
map: a :class:`~.model.MapPose` goes through the map's projector offset, a
Lanelet2 or OpenDRIVE pose through :func:`~autoware_carla_scenario.coordinate.to_carla_world`,
and a vertex that states no height is put on the road surface under it.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

from ..coordinate.map_manager import MapManager
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from .model import MapPose, ResolvedTrajectory, Trajectory, TrajectoryPosition

__all__ = ["map_pose_to_carla", "resolve_position", "resolve_trajectory"]

#: ``(x, y) -> z`` of the ground at a CARLA world point.
GroundHeight = Callable[[float, float], Optional[float]]


def map_pose_to_carla(
    pose: MapPose,
    mgrs_offset: Tuple[float, float],
    z_offset: float,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """``(x, y, z, yaw_deg)`` of a map-frame pose in CARLA world coordinates.

    The inverse of :func:`~autoware_carla_scenario.coordinate.to_map_frame`:
    subtract the projector offset, flip y (North to South) and the heading's
    sense (counter-clockwise radians to clockwise degrees).  ``z`` and the yaw
    stay ``None`` when the pose does not state them.
    """
    offset_x, offset_y = mgrs_offset
    return (
        pose.x - offset_x,
        -(pose.y - offset_y),
        None if pose.z is None else pose.z - z_offset,
        None if pose.yaw is None else -math.degrees(pose.yaw),
    )


def resolve_position(
    position: TrajectoryPosition,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """``(x, y, z, yaw_deg)`` of any vertex position in CARLA world coordinates.

    Uses the initialized :class:`~autoware_carla_scenario.coordinate.MapManager`
    for every frame but CARLA's own.
    """
    if isinstance(position, CarlaWorldPose):
        return position.x, position.y, position.z, position.yaw
    if isinstance(position, MapPose):
        manager = MapManager.get_instance()
        return map_pose_to_carla(position, manager.mgrs_offset, manager.z_offset)
    if isinstance(position, (Lanelet2Pose, OpenDrivePose)):
        from ..coordinate.transform import to_carla_world  # noqa: PLC0415

        pose = to_carla_world(position)
        return pose.x, pose.y, pose.z, pose.yaw
    raise TypeError(f"unsupported trajectory position: {type(position).__name__}")


def road_height(carla_map: Any) -> GroundHeight:
    """The road surface's height under a point, read from *carla_map*."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    def height(x: float, y: float) -> Optional[float]:
        try:
            waypoint = carla_map.get_waypoint(
                carla.Location(x=x, y=y, z=0.0), project_to_road=True
            )
        except (AttributeError, RuntimeError, TypeError):
            return None
        if waypoint is None:
            return None
        return float(waypoint.transform.location.z)

    return height


def resolve_trajectory(
    trajectory: Trajectory,
    ground: Optional[GroundHeight] = None,
    *,
    resolve: Callable[
        [TrajectoryPosition], Tuple[float, float, Optional[float], Optional[float]]
    ] = resolve_position,
) -> ResolvedTrajectory:
    """Every vertex of *trajectory* in CARLA world coordinates.

    Args:
        trajectory: The trajectory as written.
        ground: Height of the ground at a CARLA point, for the vertices that
            state none.  ``None`` -- or a point it cannot answer for -- puts
            such a vertex at the last height known along the path, or at 0.
        resolve: Turns one position into ``(x, y, z, yaw_deg)``; the default
            uses the loaded map.  Injected by the tests.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[Optional[float]] = []
    yaws: list[Optional[float]] = []
    for vertex in trajectory.vertices:
        x, y, z, yaw = resolve(vertex.position)
        if z is None and ground is not None:
            z = ground(x, y)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        yaws.append(yaw)
    known = next((value for value in zs if value is not None), 0.0)
    heights: list[float] = []
    for value in zs:
        known = value if value is not None else known
        heights.append(known)
    times = (
        [float(vertex.time) for vertex in trajectory.vertices]  # type: ignore[arg-type]
        if trajectory.is_timed
        else None
    )
    return ResolvedTrajectory(
        xs=xs, ys=ys, zs=heights, yaws=yaws, times=times, closed=trajectory.closed
    )
