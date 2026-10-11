"""Place a :class:`~.model.Trajectory` in CARLA world coordinates.

Kept apart from :mod:`.model` because this is the half that needs the loaded
map: a :class:`~.model.MapPose` goes through the map's projector offset, a
Lanelet2 or OpenDRIVE pose through :func:`~autoware_carla_scenario.coordinate.to_carla_world`,
a :class:`~.model.RelativeLanePose` through the lanelet pose it names from where
its reference entity is (:mod:`.relative_lane`), and a vertex that states no
height is put on the road surface under it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Union

from ..coordinate.map_manager import MapManager
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from .model import (
    ROUTE_POSITIONS,
    MapPose,
    RelativeLanePose,
    ResolvedTrajectory,
    Trajectory,
    TrajectoryPosition,
)

if TYPE_CHECKING:
    from ..entity_role import EntityRole

__all__ = [
    "PositionResolver",
    "ReferencePose",
    "map_pose_to_carla",
    "position_resolver",
    "resolve_position",
    "resolve_trajectory",
]

#: ``(x, y) -> z`` of the ground at a CARLA world point.
GroundHeight = Callable[[float, float], Optional[float]]

#: ``(x, y, z, yaw_deg)`` of a vertex in CARLA world coordinates.
ResolvedPosition = Tuple[float, float, Optional[float], Optional[float]]

#: Turns one vertex position into a :data:`ResolvedPosition`.
PositionResolver = Callable[[TrajectoryPosition], ResolvedPosition]

#: Where a :class:`~.model.RelativeLanePose`'s reference entity is now, by its
#: ``entity_ref`` (``None`` for the entity the action moves).
ReferencePose = Callable[[Optional[Union["EntityRole", str]]], CarlaWorldPose]


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


def position_resolver(reference: Optional[ReferencePose] = None) -> PositionResolver:
    """A :func:`resolve_position` bound to *reference*, for a whole trajectory.

    The reference entities' lanelet poses are looked up once each and reused
    for every vertex relative to them, so every vertex of one resolution is
    placed against the same instant.
    """
    lanes: dict[str, Lanelet2Pose] = {}
    ego_on_route: list[float] = []

    def ego_route_s() -> float:
        """The ego's route s, looked up once for the whole trajectory."""
        if not ego_on_route:
            if reference is None:
                raise ValueError(
                    "a route pose without an anchor is placed against the ego, "
                    "and no reference pose was given to resolve it with"
                )
            from ..conditions.route_progress import (  # noqa: PLC0415
                route_s_of_carla_point,
            )
            from ..constants import EGO_ROLE_NAME  # noqa: PLC0415

            ego = reference(EGO_ROLE_NAME)
            ego_on_route.append(route_s_of_carla_point(ego.x, ego.y)[0])
        return ego_on_route[0]

    def resolve(position: TrajectoryPosition) -> ResolvedPosition:
        if isinstance(position, ROUTE_POSITIONS):
            return _resolve_route_position(position, ego_route_s)
        if not isinstance(position, RelativeLanePose):
            return resolve_position(position)
        if reference is None:
            raise ValueError(
                "a RelativeLanePose is placed against its reference entity, "
                "and no reference pose was given to resolve it with"
            )
        from ..coordinate.transform import to_carla_world  # noqa: PLC0415
        from .relative_lane import (  # noqa: PLC0415
            reference_lane_pose_of,
            relative_lane_pose,
        )

        key = "" if position.entity_ref is None else str(position.entity_ref)
        if key not in lanes:
            lanes[key] = reference_lane_pose_of(reference(position.entity_ref))
        manager = MapManager.get_instance()
        lane = relative_lane_pose(
            manager.lanelet_map, manager.routing_graph, lanes[key], position
        )
        pose = to_carla_world(lane)
        return pose.x, pose.y, pose.z, None if position.yaw is None else pose.yaw

    return resolve


def _resolve_route_position(
    position: Any, ego_route_s: Callable[[], float]
) -> ResolvedPosition:
    """``(x, y, z, yaw_deg)`` of a route pose, on the scenario's route."""
    from ..coordinate.transform import to_carla_world  # noqa: PLC0415
    from ..route.active import scenario_route_frame  # noqa: PLC0415
    from ..route.positions import resolve_route_pose  # noqa: PLC0415

    frame = scenario_route_frame()
    needs_ego = getattr(position, "anchor", "") is None
    placed = resolve_route_pose(position, frame, ego_route_s() if needs_ego else None)
    if isinstance(placed, MapPose):
        manager = MapManager.get_instance()
        return map_pose_to_carla(placed, manager.mgrs_offset, manager.z_offset)
    pose = to_carla_world(placed)
    return pose.x, pose.y, pose.z, None if position.yaw is None else pose.yaw


def resolve_position(
    position: TrajectoryPosition,
    reference: Optional[ReferencePose] = None,
) -> ResolvedPosition:
    """``(x, y, z, yaw_deg)`` of any vertex position in CARLA world coordinates.

    Uses the initialized :class:`~autoware_carla_scenario.coordinate.MapManager`
    for every frame but CARLA's own.

    Args:
        position: The vertex position.
        reference: Where a :class:`~.model.RelativeLanePose`'s reference entity
            is; needed only for one.

    Raises:
        ValueError: For a relative pose without a *reference*, or one that
            names a lane the map does not have.
    """
    if isinstance(position, (RelativeLanePose, *ROUTE_POSITIONS)):
        return position_resolver(reference)(position)
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
    resolve: Optional[PositionResolver] = None,
    reference: Optional[ReferencePose] = None,
) -> ResolvedTrajectory:
    """Every vertex of *trajectory* in CARLA world coordinates.

    Args:
        trajectory: The trajectory as written.
        ground: Height of the ground at a CARLA point, for the vertices that
            state none.  ``None`` -- or a point it cannot answer for -- puts
            such a vertex at the last height known along the path, or at 0.
        resolve: Turns one position into ``(x, y, z, yaw_deg)``; the default
            uses the loaded map (:func:`position_resolver`).  Injected by the
            tests.
        reference: Where the reference entities of its
            :class:`~.model.RelativeLanePose` vertices are, for the default
            *resolve*.

    Raises:
        ValueError: If a vertex cannot be placed, naming the vertex.
    """
    if resolve is None:
        resolve = position_resolver(reference)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[Optional[float]] = []
    yaws: list[Optional[float]] = []
    for index, vertex in enumerate(trajectory.vertices):
        try:
            x, y, z, yaw = resolve(vertex.position)
        except ValueError as exc:
            raise ValueError(
                f"trajectory {trajectory.name!r}, vertex {index}: {exc}"
            ) from exc
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
