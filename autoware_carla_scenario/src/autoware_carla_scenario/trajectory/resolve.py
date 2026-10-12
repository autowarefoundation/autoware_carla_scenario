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
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

from ..coordinate.map_manager import MapManager
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from .model import (
    MapPose,
    RelativeLanePose,
    ResolvedTrajectory,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    Trajectory,
    TrajectoryPosition,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..entity_role import EntityRole

__all__ = [
    "PositionResolver",
    "ReferencePose",
    "map_pose_to_carla",
    "position_resolver",
    "resolve_position",
    "resolve_trajectory",
]

# The signatures below spell these aliases out: the static check
# (docs/typecheck.md) reads a callable or a union through an alias as no type.

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


def _resolved(
    x: float, y: float, z: Optional[float], yaw: Optional[float]
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """``(x, y, z, yaw_deg)``: a function, so Codon 0.19 makes a float of *z*
    and *yaw* the Optional the tuple holds, which it does not in a tuple
    display (docs/typecheck.md)."""
    return x, y, z, yaw


def position_resolver(
    reference: Optional[
        Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]
    ] = None,
) -> Callable[
    [
        Union[
            CarlaWorldPose,
            Lanelet2Pose,
            OpenDrivePose,
            MapPose,
            RelativeLanePose,
            RouteLanePose,
            RouteOppositePose,
            RouteCrossingPose,
            RouteCrosswalkPose,
            RouteRoadsidePose,
        ]
    ],
    Tuple[float, float, Optional[float], Optional[float]],
]:
    """A :func:`resolve_position` bound to *reference*, for a whole trajectory.

    The reference entities' lanelet poses are looked up once each and reused
    for every vertex relative to them, so every vertex of one resolution is
    placed against the same instant.
    """
    lanes: dict[str, Lanelet2Pose] = {}
    ego_on_route: list[float] = []

    def resolve(
        position: Union[
            CarlaWorldPose,
            Lanelet2Pose,
            OpenDrivePose,
            MapPose,
            RelativeLanePose,
            RouteLanePose,
            RouteOppositePose,
            RouteCrossingPose,
            RouteCrosswalkPose,
            RouteRoadsidePose,
        ],
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        # The route classes spelled out, rather than unpacked from
        # ROUTE_POSITIONS, which the static check cannot unpack.
        if isinstance(
            position,
            (
                RouteLanePose,
                RouteOppositePose,
                RouteCrossingPose,
                RouteCrosswalkPose,
                RouteRoadsidePose,
            ),
        ):
            return _resolve_route_position(position, reference, ego_on_route)
        if not isinstance(position, RelativeLanePose):
            # With the reference, which a pose of any other class does not
            # read, so Codon types it as this one (docs/typecheck.md).
            return resolve_position(position, reference)
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

        # Named as the class: Codon does not narrow a union by isinstance().
        relative: RelativeLanePose = position
        key = "" if relative.entity_ref is None else str(relative.entity_ref)
        if key not in lanes:
            lanes[key] = reference_lane_pose_of(reference(relative.entity_ref))
        manager = MapManager.get_instance()
        lane = relative_lane_pose(
            manager.lanelet_map, manager.routing_graph, lanes[key], relative
        )
        pose = to_carla_world(lane)
        return _resolved(
            pose.x, pose.y, pose.z, None if relative.yaw is None else pose.yaw
        )

    return resolve


def _ego_route_s(
    reference: Optional[Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]],
    ego_on_route: list[float],
) -> float:
    """The ego's route s, looked up once for the whole trajectory.

    A function of its own rather than a closure in :func:`position_resolver`:
    Codon 0.19 does not pass a closure's captures on to another closure it
    calls (docs/typecheck.md).
    """
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


def _resolve_route_position(
    position: Union[
        RouteLanePose,
        RouteOppositePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteRoadsidePose,
    ],
    reference: Optional[Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]],
    ego_on_route: list[float],
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """``(x, y, z, yaw_deg)`` of a route pose, on the scenario's route.

    The ego's route s is looked up (:func:`_ego_route_s`) only for a pose
    without an anchor; a crossing or a crosswalk has none.
    """
    # Each class on its own, named as the class: Codon reads no attribute of
    # a union, and places one class at a time (docs/typecheck.md).
    if isinstance(position, RouteLanePose):
        on_lane: RouteLanePose = position
        return _place_on_route(
            on_lane, on_lane.anchor is None, on_lane.yaw, reference, ego_on_route
        )
    if isinstance(position, RouteOppositePose):
        opposite: RouteOppositePose = position
        return _place_on_route(
            opposite, opposite.anchor is None, opposite.yaw, reference, ego_on_route
        )
    if isinstance(position, RouteRoadsidePose):
        roadside: RouteRoadsidePose = position
        return _place_on_route(
            roadside, roadside.anchor is None, roadside.yaw, reference, ego_on_route
        )
    if isinstance(position, RouteCrossingPose):
        crossing: RouteCrossingPose = position
        return _place_on_route(crossing, False, crossing.yaw, reference, ego_on_route)
    if isinstance(position, RouteCrosswalkPose):
        crosswalk: RouteCrosswalkPose = position
        return _place_on_route(crosswalk, False, crosswalk.yaw, reference, ego_on_route)
    raise TypeError(f"not a route pose: {type(position).__name__}")


def _place_on_route(
    pose: Union[
        RouteLanePose,
        RouteOppositePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteRoadsidePose,
    ],
    needs_ego: bool,
    yaw: Optional[float],
    reference: Optional[Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]],
    ego_on_route: list[float],
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """:func:`_resolve_route_position` of one route pose, its anchor read."""
    from ..coordinate.transform import to_carla_world  # noqa: PLC0415
    from ..route.active import scenario_route_frame  # noqa: PLC0415
    from ..route.positions import resolve_route_pose  # noqa: PLC0415

    frame = scenario_route_frame()
    ego_s: Optional[float] = None
    if needs_ego:
        ego_s = _ego_route_s(reference, ego_on_route)
    placed = resolve_route_pose(pose, frame, ego_s)
    if isinstance(placed, MapPose):
        manager = MapManager.get_instance()
        return map_pose_to_carla(placed, manager.mgrs_offset, manager.z_offset)
    world = to_carla_world(placed)
    return _resolved(world.x, world.y, world.z, None if yaw is None else world.yaw)


def resolve_position(
    position: Union[
        CarlaWorldPose,
        Lanelet2Pose,
        OpenDrivePose,
        MapPose,
        RelativeLanePose,
        RouteLanePose,
        RouteOppositePose,
        RouteCrossingPose,
        RouteCrosswalkPose,
        RouteRoadsidePose,
    ],
    reference: Optional[
        Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]
    ] = None,
) -> Tuple[float, float, Optional[float], Optional[float]]:
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
    if isinstance(
        position,
        (
            RelativeLanePose,
            RouteLanePose,
            RouteOppositePose,
            RouteCrossingPose,
            RouteCrosswalkPose,
            RouteRoadsidePose,
        ),
    ):
        return position_resolver(reference)(position)
    if isinstance(position, CarlaWorldPose):
        # Named as the class: Codon reads no attribute of a union.
        world: CarlaWorldPose = position
        return _resolved(world.x, world.y, world.z, world.yaw)
    if isinstance(position, MapPose):
        manager = MapManager.get_instance()
        return map_pose_to_carla(position, manager.mgrs_offset, manager.z_offset)
    if isinstance(position, (Lanelet2Pose, OpenDrivePose)):
        from ..coordinate.transform import to_carla_world  # noqa: PLC0415

        pose = to_carla_world(position)
        return _resolved(pose.x, pose.y, pose.z, pose.yaw)
    raise TypeError(f"unsupported trajectory position: {type(position).__name__}")


def road_height(carla_map: carla.Map) -> Callable[[float, float], Optional[float]]:
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
    ground: Optional[Callable[[float, float], Optional[float]]] = None,
    *,
    resolve: Optional[
        Callable[
            [
                Union[
                    CarlaWorldPose,
                    Lanelet2Pose,
                    OpenDrivePose,
                    MapPose,
                    RelativeLanePose,
                    RouteLanePose,
                    RouteOppositePose,
                    RouteCrossingPose,
                    RouteCrosswalkPose,
                    RouteRoadsidePose,
                ]
            ],
            Tuple[float, float, Optional[float], Optional[float]],
        ]
    ] = None,
    reference: Optional[
        Callable[[Optional[Union[EntityRole, str]]], CarlaWorldPose]
    ] = None,
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
    # The default resolver and the one given are two types to Codon, which
    # types a variable once (docs/typecheck.md): each goes to _resolve_vertices.
    if resolve is None:
        return _resolve_vertices(trajectory, ground, position_resolver(reference))
    return _resolve_vertices(trajectory, ground, resolve)


def _resolve_vertices(
    trajectory: Trajectory,
    ground: Optional[Callable[[float, float], Optional[float]]],
    resolve: Callable[
        [
            Union[
                CarlaWorldPose,
                Lanelet2Pose,
                OpenDrivePose,
                MapPose,
                RelativeLanePose,
                RouteLanePose,
                RouteOppositePose,
                RouteCrossingPose,
                RouteCrosswalkPose,
                RouteRoadsidePose,
            ]
        ],
        Tuple[float, float, Optional[float], Optional[float]],
    ],
) -> ResolvedTrajectory:
    """:func:`resolve_trajectory` with the resolver decided."""
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
    # The first height known, or 0; then each vertex's, or the last known.
    # Statements rather than next() and conditional expressions, each of
    # whose values Codon would have to type as one (docs/typecheck.md).
    known = 0.0
    for value in zs:
        if value is not None:
            known = value
            break
    heights: list[float] = []
    for value in zs:
        if value is not None:
            known = value
        heights.append(known)
    times: Optional[list[float]] = None
    if trajectory.is_timed:
        times = [float(vertex.time) for vertex in trajectory.vertices]  # type: ignore[arg-type]
    return ResolvedTrajectory(
        xs=xs, ys=ys, zs=heights, yaws=yaws, times=times, closed=trajectory.closed
    )
