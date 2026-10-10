"""Place a route pose (``RouteLanePose`` and its kin) on the map.

Each function takes the :class:`~.frame.RouteFrame` of the scenario's route
and, for a pose measured from the ego, the ego's route s when the action
starts; it returns a :class:`~autoware_carla_scenario.coordinate.poses.Lanelet2Pose`
for a pose on a lane, or a :class:`~autoware_carla_scenario.trajectory.MapPose`
for one off it (a crosswalk, the roadside).  A pose the map cannot give -- no
opposite lane there, no crosswalk on that leg -- raises a ``ValueError`` that
says which, so the action's error names the vertex and the reason.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from ..coordinate.poses import Lanelet2Pose
from ..trajectory.model import (
    MapPose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
)
from ..trajectory.relative_lane import (
    _arc,
    _centreline,
    _heading_at,
    _point_at,
    advance_along_lane,
    shift_lanes,
)
from .frame import RouteFrame

__all__ = ["RoutePose", "resolve_route_pose", "route_base_s"]

RoutePose = Union[
    RouteLanePose,
    RouteOppositePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteRoadsidePose,
]


def route_base_s(
    frame: RouteFrame, anchor: Optional[str], ego_s: Optional[float]
) -> float:
    """Route s a pose's ``ds`` counts from: its anchor's, else the ego's.

    Raises:
        ValueError: If there is no anchor and no ego position to count from.
    """
    if anchor:
        return frame.match.anchor_s(anchor)
    if ego_s is None:
        raise ValueError(
            "a route pose without an anchor is measured from the ego, and the "
            "ego's position on the route is not known"
        )
    return ego_s


def resolve_route_pose(
    pose: RoutePose, frame: RouteFrame, ego_s: Optional[float] = None
) -> Union[Lanelet2Pose, MapPose]:
    """Where *pose* is on *frame*'s map.

    Raises:
        ValueError: When the map lacks what the pose names, saying what.
    """
    if isinstance(pose, RouteLanePose):
        return _lane(pose, frame, ego_s)
    if isinstance(pose, RouteOppositePose):
        return _opposite(pose, frame, ego_s)
    if isinstance(pose, RouteCrossingPose):
        return _crossing(pose, frame)
    if isinstance(pose, RouteCrosswalkPose):
        return _crosswalk(pose, frame)
    if isinstance(pose, RouteRoadsidePose):
        return _roadside(pose, frame, ego_s)
    raise TypeError(f"not a route pose: {type(pose).__name__}")


def _yaw(yaw: Optional[float]) -> float:
    return 0.0 if yaw is None else float(yaw)


def _lane(
    pose: RouteLanePose, frame: RouteFrame, ego_s: Optional[float]
) -> Lanelet2Pose:
    s = route_base_s(frame, pose.anchor, ego_s) + pose.ds
    lanelet_id, along = frame.locate(s)
    if pose.d_lane:
        lanelet_id, along = shift_lanes(
            frame.map, frame.graph, lanelet_id, along, pose.d_lane
        )
    return Lanelet2Pose(
        lanelet_id=lanelet_id, s=along, t=float(pose.offset), heading=_yaw(pose.yaw)
    )


def _opposite(
    pose: RouteOppositePose, frame: RouteFrame, ego_s: Optional[float]
) -> Lanelet2Pose:
    s = route_base_s(frame, pose.anchor, ego_s) + pose.ds
    lanelet_id, along = frame.locate(s)
    lanelet = frame.map.laneletLayer[lanelet_id]
    x, y = _point_at(_centreline(lanelet), along)
    found = frame.features.opposite_abreast(lanelet, x, y)
    if found is None:
        raise ValueError(
            f"there is no opposite-direction lane abreast of route s={s:.1f} m "
            f"(lanelet {lanelet_id})"
        )
    current, _side = found
    for step in range(1, pose.lane):
        outward = _outward(frame, current, x, y)
        if outward is None:
            raise ValueError(
                f"the opposite road abreast of route s={s:.1f} m has {step} "
                f"lane(s); there is no lane {pose.lane}"
            )
        current = outward
    on, _t = _arc(current, x, y)
    return Lanelet2Pose(
        lanelet_id=int(current.id), s=on, t=float(pose.offset), heading=_yaw(pose.yaw)
    )


def _outward(frame: RouteFrame, lanelet: Any, x: float, y: float) -> Any:
    """*lanelet*'s same-direction neighbour farther from ``(x, y)``, or ``None``.

    The neighbour's lanelet abreast of the point, as *lanelet* is.
    """
    best: Optional[tuple[float, Any]] = None
    for side in ("left", "right"):
        beside = frame.features.neighbour(lanelet, side)
        if beside is None:
            continue
        beside = frame.features.abreast(beside, x, y)
        s, _t = _arc(beside, x, y)
        bx, by = _point_at(_centreline(beside), s)
        own_s, _ = _arc(lanelet, x, y)
        ox, oy = _point_at(_centreline(lanelet), own_s)
        if math.hypot(bx - x, by - y) > math.hypot(ox - x, oy - y):
            gap = math.hypot(bx - x, by - y)
            if best is None or gap < best[0]:
                best = (gap, beside)
    return None if best is None else best[1]


def _crossing(pose: RouteCrossingPose, frame: RouteFrame) -> Lanelet2Pose:
    way, _entry, _exit = frame.junction_way(pose.junction)
    members = [
        m
        for m in frame.features.junction_members(way)
        if m.approach == pose.approach and (pose.turn is None or m.turn == pose.turn)
    ]
    if not members:
        turning = "" if pose.turn is None else f" turning {pose.turn}"
        raise ValueError(
            f"junction {pose.junction} of the route has no lanelet entering it "
            f"from the {pose.approach}{turning}"
        )
    # Traffic that actually crosses the ego's way first, then by id.
    chosen = min(members, key=lambda m: (not m.conflicts, int(m.lanelet.id)))
    lanelet_id, along = advance_along_lane(
        frame.map, frame.graph, int(chosen.lanelet.id), 0.0, float(pose.distance)
    )
    return Lanelet2Pose(
        lanelet_id=lanelet_id, s=along, t=float(pose.offset), heading=_yaw(pose.yaw)
    )


def _crosswalk(pose: RouteCrosswalkPose, frame: RouteFrame) -> MapPose:
    way, entry, exit_ = frame.junction_way(pose.junction)
    crosswalks = [
        c
        for c in frame.features.junction_crosswalks(way, entry, exit_)
        if c.leg == pose.leg
    ]
    if not crosswalks:
        raise ValueError(
            f"junction {pose.junction} of the route has no crosswalk across its "
            f"{pose.leg} leg"
        )
    through = sum(frame.features.length(ll) for ll in way)
    # The one nearest the junction on that leg.
    boundary = 0.0 if pose.leg == "entry" else through
    crosswalk = min(crosswalks, key=lambda c: (abs(c.u - boundary), int(c.lanelet.id)))
    line = _centreline(crosswalk.lanelet)
    px, py = crosswalk.point
    hx, hy = math.cos(crosswalk.heading), math.sin(crosswalk.heading)

    def left_of_path(point: tuple[float, float]) -> bool:
        return hx * (point[1] - py) - hy * (point[0] - px) > 0.0

    first_left = left_of_path(line[0])
    if (pose.side == "left") != first_left:
        line = list(reversed(line))
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line, line[1:]))
    if pose.along >= 0.0:
        x, y = _point_at(line, min(pose.along, total))
        heading = _heading_at(line, min(pose.along, total))
        if pose.along > total:
            x += math.cos(heading) * (pose.along - total)
            y += math.sin(heading) * (pose.along - total)
    else:
        heading = _heading_at(line, 0.0)
        x = line[0][0] + math.cos(heading) * pose.along
        y = line[0][1] + math.sin(heading) * pose.along
    return MapPose(x, y, None if pose.yaw is None else heading + float(pose.yaw))


def _roadside(
    pose: RouteRoadsidePose, frame: RouteFrame, ego_s: Optional[float]
) -> MapPose:
    s = route_base_s(frame, pose.anchor, ego_s) + pose.ds
    lanelet_id, along = frame.locate(s)
    lanelet = frame.map.laneletLayer[lanelet_id]
    points = _centreline(lanelet)
    x, y = _point_at(points, along)
    heading = _heading_at(points, along)
    edge = frame.features.abreast(frame.features.outermost(lanelet, pose.side), x, y)
    opposite = frame.features.opposite_abreast(lanelet, x, y)
    if opposite is not None and opposite[1] == pose.side:
        # The opposite road is on this side: its outermost lane is the edge.
        edge = opposite[0]
        while True:
            outward = _outward(frame, edge, x, y)
            if outward is None:
                break
            edge = outward
    # The edge lanelet's bound on that side, the farther one from the route.
    hx, hy = math.cos(heading), math.sin(heading)
    sign = 1.0 if pose.side == "left" else -1.0
    best: Optional[tuple[float, tuple[float, float]]] = None
    for bound in (edge.leftBound, edge.rightBound):
        bx, by = frame.features.nearest_on(bound, x, y)
        if sign * (hx * (by - y) - hy * (bx - x)) <= 1e-6:
            continue
        gap = math.hypot(bx - x, by - y)
        if best is None or gap > best[0]:
            best = (gap, (bx, by))
    if best is None:
        raise ValueError(
            f"no road edge found on the {pose.side} abreast of route s={s:.1f} m"
        )
    gap, (kx, ky) = best
    nx, ny = (kx - x) / gap, (ky - y) / gap
    distance = float(pose.kerb_distance)
    return MapPose(
        kx + nx * distance,
        ky + ny * distance,
        None if pose.yaw is None else heading + float(pose.yaw),
    )
