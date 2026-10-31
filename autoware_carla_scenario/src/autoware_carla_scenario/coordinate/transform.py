"""6-direction mutual conversion between Lanelet2, OpenDRIVE, and CARLA world coordinates.

Lanelet2 and OpenDRIVE are treated as two independent coordinate systems that
share nothing but the world they describe: a pose moves between them through
CARLA world coordinates, so no lanelet-to-road correspondence -- and nothing
from the converter that produced the OpenDRIVE -- is needed.

Coordinate systems
------------------
Lanelet2 (MGRS absolute, right-hand):  x=East, y=North, z=Up
OpenDRIVE / CARLA (XODR-relative):     origin = geoReference (lat_0, lon_0)
CARLA world (left-hand):               x≈East, y≈South, z=Up  (y flipped from XODR)
Autoware ``map`` (right-hand):         the Lanelet2 frame -- Autoware's own
                                       map_loader reads the same file with the
                                       same projector (see :func:`to_map_frame`)

Key relationships
-----------------
  mgrs_xy  = xodr_xy + mgrs_offset
  carla_x  = xodr_x                     (same origin as XODR)
  carla_y  = −xodr_y                    (flip North→South)
  carla_z  = xodr_z

  carla_x  = ll2_x − mgrs_offset_x     (MGRS absolute → XODR relative)
  carla_y  = −(ll2_y − mgrs_offset_y)   (MGRS absolute → XODR relative, then flip)
  carla_yaw_deg = −degrees(heading_rad)  (right-hand → left-hand)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Union, overload

import typesafe_carla.carla as carla
import numpy as np

if TYPE_CHECKING:
    import lxml.etree
    from pyxodr.road_objects.lane import Lane
    from pyxodr.road_objects.lane_section import LaneSection
    from pyxodr.road_objects.network import RoadNetwork
    from pyxodr.road_objects.road import Road

    from ..autoware_bridge.base import BridgePose

# autoware_lanelet2_extension_python must be imported before lanelet2
from autoware_lanelet2_extension_python.projection import MGRSProjector as _  # noqa: F401
import lanelet2.core
import lanelet2.geometry

from .map_manager import MapManager
from .poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose

logger = logging.getLogger(__name__)

#: How much nearer than the best road so far (m) another one has to be to win.
#: Within it the two are the same distance away -- the seam between connected
#: roads -- and the tie is broken by which road the point starts.
_SEAM_TOLERANCE = 1e-3

#: How many lanelets nearest in plan view are weighed against the pose's
#: elevation, to tell stacked lanelets apart.
_STACKED_CANDIDATES = 8


# ---------------------------------------------------------------------------
# Public API (overloaded)
# ---------------------------------------------------------------------------


def project_onto_road(pose: CarlaWorldPose, road_id: str) -> OpenDrivePose:
    """Project a CARLA world position onto a specific OpenDRIVE road.

    Unlike :func:`to_opendrive` which finds the nearest road, this projects
    directly onto the specified road regardless of distance.

    Args:
        pose: The CARLA world pose to project.
        road_id: The OpenDRIVE road ID to project onto.

    Returns:
        An :class:`OpenDrivePose` with ``road_id`` set to *road_id*.
    """
    mm = MapManager.get_instance()
    road = mm.road_network.road_ids_to_object[road_id]
    ref_line: np.ndarray = road.reference_line

    od_heading = -math.radians(pose.yaw)
    _idx, _dist, s, heading_ref, t = _project_onto_ref_line(
        ref_line,
        np.array([pose.x, -pose.y]),
        _road_arc_lengths(mm.road_network, road_id),
    )

    return OpenDrivePose(
        road_id=road_id,
        lane_id=_find_lane_at_t(road, s, t),
        s=s,
        t=t,
        heading=_normalize_angle(od_heading - heading_ref),
    )


def road_traffic_rule(road_id: str) -> str:
    """The traffic rule of an OpenDRIVE road: ``"RHT"`` or ``"LHT"``.

    The road's ``rule`` attribute, ``"RHT"`` (right-hand traffic) where the
    road does not set one, as the OpenDRIVE standard defaults it.  It says
    which side of the reference line drives along it: the right (negative
    lane ids) in right-hand traffic, the left (positive ids) in left-hand.

    Raises:
        KeyError: If the loaded road network has no road *road_id*.
        RuntimeError: If :class:`MapManager` is not initialized.
    """
    road = MapManager.get_instance().road_network.road_ids_to_object[road_id]
    return road.road_xml.get("rule", "RHT")


def project_onto_lanelet(pose: CarlaWorldPose, lanelet_id: int) -> Lanelet2Pose:
    """Project a CARLA world position onto a specific lanelet's centerline.

    The Lanelet2 counterpart of :func:`project_onto_road`: where
    :func:`to_lanelet2` finds the *nearest* lanelet, this measures against the
    one named, so a condition that addresses a lanelet reads ``s`` and ``t`` in
    that lanelet's own frame -- ``s`` from its start along its direction of
    travel, ``t`` positive to the left of it -- whichever way the OpenDRIVE road
    it lies on is drawn.  ``s`` is exact along the centerline (not snapped to a
    vertex) and clamped to the lanelet.

    Args:
        pose: The CARLA world pose to project.
        lanelet_id: The lanelet to project onto.

    Returns:
        A :class:`Lanelet2Pose` on *lanelet_id*.
    """
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
    x, y, heading = _carla_to_lanelet2_frame(pose)
    centerline = lanelet2.geometry.to2D(lanelet.centerline)
    arc = lanelet2.geometry.toArcCoordinates(
        centerline, lanelet2.core.BasicPoint2d(x, y)
    )
    length = float(lanelet2.geometry.length(centerline))
    s = min(max(float(arc.length), 0.0), length)
    points = np.array([(p.x, p.y) for p in lanelet.centerline])
    heading_cl = _heading_at_s(points, _compute_arc_lengths_2d(points), s)
    return Lanelet2Pose(
        lanelet_id=lanelet_id,
        s=s,
        t=float(arc.distance),
        heading=_normalize_angle(heading - heading_cl),
    )


def on_lanelet(pose: CarlaWorldPose, lanelet_id: int) -> bool:
    """Whether a CARLA world position lies within a lanelet's bounds.

    The membership test for an address given as a lanelet.  It is asked of the
    lanelet itself rather than answered by "the nearest lanelet is this one",
    which a point in a junction -- where lanelets overlap -- would answer for
    only one of them.
    """
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
    x, y, _heading = _carla_to_lanelet2_frame(pose)
    return bool(lanelet2.geometry.inside(lanelet, lanelet2.core.BasicPoint2d(x, y)))


def lanelet_length(lanelet_id: int) -> float:
    """The length of a lanelet's centerline, in metres."""
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
    return float(lanelet2.geometry.length(lanelet2.geometry.to2D(lanelet.centerline)))


def route_road_ids(lanelet_ids: list[int]) -> list[str]:
    """The OpenDRIVE roads a route of lanelets runs on, each once, in order.

    Each lanelet's road is read halfway along it: a lanelet begins where the
    one before it ends, and the point there reads as the road before.  A
    junction's lanelets are left out: its roads overlap, and a point on one
    reads as another -- a left turn's as the road straight on.  The roads
    before and after it say which way it is crossed.  A route of junction
    lanelets alone is read as it is.
    """
    # Imported here: the ODD's probes import the coordinate package.
    from ..odd.probes import lanelet_in_junction  # noqa: PLC0415

    layer = MapManager.get_instance().lanelet_map.laneletLayer
    plain = [i for i in lanelet_ids if not lanelet_in_junction(layer[i])] or lanelet_ids
    roads: list[str] = []
    for lanelet_id in plain:
        pose = Lanelet2Pose(lanelet_id=lanelet_id, s=lanelet_length(lanelet_id) / 2.0)
        road = str(to_opendrive(pose).road_id)
        logger.debug("Lanelet %d -> OpenDRIVE road '%s'", lanelet_id, road)
        if road not in roads:
            roads.append(road)
    return roads


def lanelets_ahead(lanelet_id: int, distance_m: float) -> list[int]:
    """A lanelet and the lanelets that continue ahead of it, within a distance.

    The lanes a vehicle on *lanelet_id* can follow without changing lanes:
    the lanelet itself, then the routing graph's following lanelets
    (``following(lanelet, withLaneChanges=False)``), transitively -- every
    branch of a fork or a junction, no lane beside them.  A successor is
    included when it begins less than *distance_m* from the start of
    *lanelet_id*, along centerlines by the shortest chain.

    The set an "ego stays on its lanes ahead" fail condition tests the ego's
    position against: not on any of them is a lane change, a turn off the
    road ahead, or leaving the road (see :class:`NotCondition`'s
    ``holds_when_pending``).

    Requires :class:`MapManager` to be initialised.

    Args:
        lanelet_id: The lanelet to start from.
        distance_m: How far ahead to follow it (m), from its start.

    Returns:
        The lanelet ids, *lanelet_id* first, then by distance.
    """
    mm = MapManager.get_instance()
    graph = mm.routing_graph
    layer = mm.lanelet_map.laneletLayer
    # The shortest distance from the start of lanelet_id to each lanelet's
    # start: Dijkstra, on a frontier small enough to scan for its nearest.
    begins: dict[int, float] = {lanelet_id: 0.0}
    done: list[int] = []
    frontier: list[int] = [lanelet_id]
    while frontier:
        index = 0
        for i in range(1, len(frontier)):
            if begins[frontier[i]] < begins[frontier[index]]:
                index = i
        current = frontier.pop(index)
        if current in done:
            continue
        done.append(current)
        lanelet = layer[current]
        ends_at = begins[current] + float(
            lanelet2.geometry.length(lanelet2.geometry.to2D(lanelet.centerline))
        )
        if ends_at >= distance_m:
            continue
        for successor in graph.following(lanelet, False):
            successor_id = int(successor.id)
            if successor_id not in begins or ends_at < begins[successor_id]:
                begins[successor_id] = ends_at
                frontier.append(successor_id)
    return done


def _carla_to_lanelet2_frame(pose: CarlaWorldPose) -> tuple[float, float, float]:
    """A CARLA pose's ``(x, y, heading)`` in the Lanelet2 map's frame."""
    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    return pose.x + offset_x, -pose.y + offset_y, -math.radians(pose.yaw)


def to_carla_location(
    pose: Union[Lanelet2Pose, OpenDrivePose, CarlaWorldPose, "carla.Location"],
) -> "carla.Location":
    """Convert any pose type to a ``carla.Location``.

    Accepts ``Lanelet2Pose``, ``OpenDrivePose``, ``CarlaWorldPose``, or a raw
    ``carla.Location``.  Lanelet2/OpenDRIVE poses are first converted to CARLA
    world coordinates via :func:`to_carla_world`.

    Args:
        pose: The source pose or location.

    Returns:
        A ``carla.Location`` instance.
    """
    if isinstance(pose, (Lanelet2Pose, OpenDrivePose)):
        world = to_carla_world(pose)
        return carla.Location(x=world.x, y=world.y, z=world.z)
    elif isinstance(pose, CarlaWorldPose):
        return carla.Location(x=pose.x, y=pose.y, z=pose.z)
    else:
        # Assume carla.Location
        return pose


@overload
def to_carla_world(pose: Lanelet2Pose) -> CarlaWorldPose: ...


@overload
def to_carla_world(pose: OpenDrivePose) -> CarlaWorldPose: ...


def to_carla_world(pose: Union[Lanelet2Pose, OpenDrivePose]) -> CarlaWorldPose:
    """Convert a Lanelet2Pose or OpenDrivePose to a CARLA world pose."""
    if isinstance(pose, Lanelet2Pose):
        return _lanelet2_to_carla(pose)
    if isinstance(pose, OpenDrivePose):
        return _opendrive_to_carla(pose)
    raise TypeError(f"Unsupported pose type: {type(pose)}")


def to_opendrive(
    pose: Union[Lanelet2Pose, OpenDrivePose, CarlaWorldPose],
) -> OpenDrivePose:
    """Convert any pose to an OpenDrivePose.

    OpenDRIVE is the frame the runtime works in, so this is the normalisation
    every caller that accepts an author's address runs first.  An
    :class:`OpenDrivePose` is returned unchanged: it is already in frame, and
    saying so here means a caller does not have to special-case it and, more
    to the point, does not need a loaded map to pass one through.

    A :class:`Lanelet2Pose` goes through CARLA world coordinates.
    """
    if isinstance(pose, OpenDrivePose):
        return pose
    if isinstance(pose, CarlaWorldPose):
        return _carla_to_opendrive(pose)
    if isinstance(pose, Lanelet2Pose):
        return _carla_to_opendrive(_lanelet2_to_carla(pose))
    raise TypeError(f"Unsupported pose type: {type(pose)}")


@overload
def to_lanelet2(pose: OpenDrivePose) -> Lanelet2Pose: ...


@overload
def to_lanelet2(pose: CarlaWorldPose) -> Lanelet2Pose: ...


def to_lanelet2(pose: Union[OpenDrivePose, CarlaWorldPose]) -> Lanelet2Pose:
    """Convert an OpenDrivePose or CarlaWorldPose to a Lanelet2Pose.

    An :class:`OpenDrivePose` goes through CARLA world coordinates.  The pose
    is read on the nearest lanelet: ``s`` is the arc length to the foot of the
    perpendicular on its centerline (not the nearest centerline vertex), and
    ``t`` and ``heading`` are measured against the segment that foot lies on.
    """
    if isinstance(pose, CarlaWorldPose):
        return _carla_to_lanelet2(pose)
    if isinstance(pose, OpenDrivePose):
        return _carla_to_lanelet2(_opendrive_to_carla(pose))
    raise TypeError(f"Unsupported pose type: {type(pose)}")


def to_map_frame(
    pose: Union[Lanelet2Pose, OpenDrivePose, CarlaWorldPose],
) -> "BridgePose":
    """Convert any pose to a 6-DoF pose in Autoware's ``map`` frame.

    Autoware's ``map`` frame is the frame its own ``map_loader`` produces: the
    Lanelet2 file read with the projector named in ``map_projector_info.yaml``.
    That is exactly the frame this module calls "Lanelet2 absolute" -- so the
    conversion from CARLA world coordinates is the inverse of
    :func:`_lanelet2_to_carla`: undo the y-flip, add the projector offset back,
    and restore the absolute elevation.

    Orientation follows the CARLA (left-handed, degrees) to ROS (right-handed,
    radians) convention: ``roll`` keeps its sign, ``pitch`` and ``yaw`` flip.
    A pose that carries pitch/roll -- one snapped onto a ramp or a banked
    segment, say -- keeps them, so what Autoware initializes localization at is
    the pose the vehicle actually stands in.

    Args:
        pose: A Lanelet2, OpenDRIVE or CARLA world pose.  The first two are
            converted through :func:`to_carla_world` first, which leaves
            pitch/roll at zero.

    Returns:
        The map-frame :class:`~autoware_carla_scenario.autoware_bridge.base.BridgePose`
        to hand Autoware as an initial pose or a goal.
    """
    from ..autoware_bridge.base import BridgePose  # noqa: PLC0415

    if isinstance(pose, (Lanelet2Pose, OpenDrivePose)):
        carla_pose = to_carla_world(pose)
    else:
        carla_pose = pose

    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    return BridgePose.from_rpy(
        x=carla_pose.x + offset_x,
        y=-carla_pose.y + offset_y,
        z=carla_pose.z + mm.z_offset,
        roll=math.radians(carla_pose.roll),
        pitch=-math.radians(carla_pose.pitch),
        yaw=-math.radians(carla_pose.yaw),
    )


# ---------------------------------------------------------------------------
# Forward: Lanelet2Pose → CarlaWorldPose
# ---------------------------------------------------------------------------


def _lanelet2_to_carla(pose: Lanelet2Pose) -> CarlaWorldPose:
    """Convert a Lanelet2 Frenet pose to CARLA world coordinates."""
    mm = MapManager.get_instance()
    lanelet = mm.lanelet_map.laneletLayer[pose.lanelet_id]
    points = [(p.x, p.y, p.z) for p in lanelet.centerline]

    x_cl, y_cl, z_cl, heading_cl = _interpolate_at_s(points, pose.s)
    total_heading = heading_cl + pose.heading

    # Positive t is left of the CENTRELINE, not of the pose: see the note in
    # _opendrive_to_carla.
    x = x_cl + pose.t * (-math.sin(heading_cl))
    y = y_cl + pose.t * math.cos(heading_cl)

    # Lanelet2 centerline uses MGRS absolute coords; CARLA world uses
    # XODR-relative coords.  Subtract the MGRS offset for x/y, and the
    # vertical offset for z (Lanelet2 stores absolute elevation).
    offset_x, offset_y = mm.mgrs_offset
    carla_yaw_deg = -math.degrees(total_heading)
    return CarlaWorldPose(
        x=x - offset_x,
        y=-(y - offset_y),
        z=z_cl - mm.z_offset,
        yaw=carla_yaw_deg,
    )


# ---------------------------------------------------------------------------
# Forward: OpenDrivePose → CarlaWorldPose
# ---------------------------------------------------------------------------


def _opendrive_to_carla(pose: OpenDrivePose) -> CarlaWorldPose:
    """Convert an OpenDRIVE (s, t) pose to CARLA world coordinates."""
    mm = MapManager.get_instance()
    road = mm.road_network.road_ids_to_object[pose.road_id]
    ref_line: np.ndarray = road.reference_line  # shape (N, 2)

    z_coords: np.ndarray = road.z_coordinates  # shape (N,)
    arc_lengths = _road_arc_lengths(mm.road_network, pose.road_id)

    x_ref = float(np.interp(pose.s, arc_lengths, ref_line[:, 0]))
    y_ref = float(np.interp(pose.s, arc_lengths, ref_line[:, 1]))
    z_ref = float(np.interp(pose.s, arc_lengths, z_coords))

    heading_ref = _heading_at_s(ref_line, arc_lengths, pose.s)
    total_heading = heading_ref + pose.heading

    # OpenDRIVE defines t as the offset perpendicular to the REFERENCE LINE,
    # positive to its left, and that is how it was measured on the way in (see
    # _signed_perp_distance, which is given heading_ref). The pose's own heading
    # says which way the thing at (s, t) faces; it does not move it. Laying t off
    # `total_heading` instead is the same thing only while the pose happens to
    # point along the reference line -- and this map is full of lanelets that do
    # not, because it is left-hand traffic converted from Lanelet2, where the
    # reference line commonly runs against the lane. For those, pose.heading is
    # about pi, the (-sin, cos) pair flips, and the point lands on the wrong side
    # of the line by twice t. Measured on lanelet 176640: t = -1.55 m came back
    # 3.09 m away, t = -1.71 m came back 3.43 m away -- 2|t| every time, which for
    # a goal pose is a whole lane.
    x = x_ref + pose.t * (-math.sin(heading_ref))
    y = y_ref + pose.t * math.cos(heading_ref)

    # XODR coords are already in CARLA's coordinate frame (same origin);
    # just flip y for CARLA's left-hand system (South=+y).
    carla_yaw_deg = -math.degrees(total_heading)
    return CarlaWorldPose(x=x, y=-y, z=z_ref, yaw=carla_yaw_deg)


# ---------------------------------------------------------------------------
# Inverse: CarlaWorldPose → OpenDrivePose
# ---------------------------------------------------------------------------


#: The reference lines of the road network last read: the network itself
#: (held, so that no other network can take its ``id`` while it is here), each
#: road with two or more vertices in the network's order as ``(road_id, road,
#: ref_line, arc_lengths)``, all their vertices stacked, where each road's
#: begin in that stack, and each road's arc lengths by its id. A road's arc
#: lengths never change, and the brute-force search measures every vertex of
#: the map on every call.
_REF_LINES: "Optional[tuple[RoadNetwork, list[tuple[str, Road, np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]], np.ndarray[tuple[int, int], np.dtype[np.float64]], list[int], dict[str, np.ndarray[tuple[int], np.dtype[np.float64]]]]]" = None


def _ref_lines(
    network: "RoadNetwork",
) -> "tuple[RoadNetwork, list[tuple[str, Road, np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]], np.ndarray[tuple[int, int], np.dtype[np.float64]], list[int], dict[str, np.ndarray[tuple[int], np.dtype[np.float64]]]]":
    """What :data:`_REF_LINES` keeps of *network*, read on first use."""
    global _REF_LINES

    kept = _REF_LINES
    if kept is not None and kept[0] is network:
        return kept
    roads: "list[tuple[str, Road, np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]]]" = []
    starts: list[int] = []
    arcs: "dict[str, np.ndarray[tuple[int], np.dtype[np.float64]]]" = {}
    count = 0
    for road_id, road in network.road_ids_to_object.items():
        ref_line = road.reference_line
        if len(ref_line) < 2:
            continue
        arc = _compute_arc_lengths_2d(ref_line)
        roads.append((road_id, road, ref_line, arc))
        arcs[road_id] = arc
        starts.append(count)
        count += len(ref_line)
    stacked = np.concatenate([r[2] for r in roads]) if roads else np.zeros((0, 2))
    read = (network, roads, stacked, starts, arcs)
    _REF_LINES = read
    return read


def _road_arc_lengths(
    network: "RoadNetwork", road_id: str
) -> "np.ndarray[tuple[int], np.dtype[np.float64]]":
    """The arc lengths of road *road_id*'s reference line, kept with
    *network*'s reference lines (computed for one of fewer than two vertices,
    which they leave out)."""
    arcs = _ref_lines(network)[4]
    if road_id in arcs:
        return arcs[road_id]
    return _compute_arc_lengths_2d(network.road_ids_to_object[road_id].reference_line)


def _carla_to_opendrive(pose: CarlaWorldPose) -> OpenDrivePose:
    """Convert a CARLA world pose to the nearest OpenDRIVE (s, t) pose.

    Uses ``carla.Map.get_waypoint()`` when available for accurate road/lane
    identification; falls back to brute-force reference-line search otherwise.
    """
    mm = MapManager.get_instance()
    carla_map = mm.carla_map

    if carla_map is not None:
        result = _carla_to_opendrive_via_waypoint(pose, carla_map, mm)
        if result is not None:
            return result
        logger.warning(
            "Waypoint lookup failed for (%.2f, %.2f); "
            "falling back to brute-force search",
            pose.x,
            pose.y,
        )

    return _carla_to_opendrive_bruteforce(pose, mm)


def _carla_to_opendrive_via_waypoint(
    pose: CarlaWorldPose,
    carla_map: carla.Map,
    mm: MapManager,
) -> Optional[OpenDrivePose]:
    """Use ``carla.Map.get_waypoint()`` for accurate road/lane identification.

    Returns ``None`` when the waypoint cannot be obtained or the road is not
    present in the pyxodr ``RoadNetwork``.
    """
    try:
        wp = carla_map.get_waypoint(
            carla.Location(x=pose.x, y=pose.y, z=pose.z),
        )
    except Exception:
        return None

    if wp is None:
        return None

    road_id = str(wp.road_id)
    lane_id: int = wp.lane_id

    if road_id not in mm.road_network.road_ids_to_object:
        return None

    road = mm.road_network.road_ids_to_object[road_id]
    ref_line: np.ndarray = road.reference_line
    if len(ref_line) < 2:
        return None

    # CARLA world coords share the same origin as XODR; just flip y back.
    od_x = pose.x
    od_y = -pose.y
    od_heading = -math.radians(pose.yaw)

    arc_lengths = _road_arc_lengths(mm.road_network, road_id)
    diffs = ref_line - np.array([od_x, od_y])
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = int(np.argmin(dists))

    s = float(arc_lengths[nearest_idx])
    heading_ref = _heading_at_s(ref_line, arc_lengths, s)
    t = _signed_perp_distance(od_x, od_y, ref_line, nearest_idx, heading_ref)
    heading_diff = _normalize_angle(od_heading - heading_ref)

    return OpenDrivePose(
        road_id=road_id,
        lane_id=lane_id,
        s=s,
        t=t,
        heading=heading_diff,
    )


def _carla_to_opendrive_bruteforce(
    pose: CarlaWorldPose,
    mm: MapManager,
) -> OpenDrivePose:
    """Brute-force road search across all roads in the network.

    Used as a fallback when ``carla.Map`` is unavailable or waypoint lookup
    fails.  The road whose lanes hold the point wins over one whose reference
    line merely passes closer: a lane two lanes out from its own reference
    line is still nearer the end of the road before it than to that line.
    Among roads that hold it (or none that do), the one whose lanes it is
    nearest to wins, then the one whose reference line is nearest.
    """
    # CARLA world coords share the same origin as XODR; just flip y back.
    point = np.array([pose.x, -pose.y])

    # Every vertex of the map measured at once; each road then reads its own.
    _network, roads, stacked, starts, _arcs = _ref_lines(mm.road_network)
    all_dists = np.linalg.norm(stacked - point, axis=1)

    # (outside, dist, at_end, road_id, road, s, t, heading_ref) of the best so far
    best: Optional[tuple[float, float, bool, str, Road, float, float, float]] = None
    for (road_id, road, ref_line, arc_lengths), begin in zip(roads, starts):
        dists = all_dists[begin : begin + len(ref_line)]
        nearest_idx = int(np.argmin(dists))
        dist = float(dists[nearest_idx])
        overshoot = _overshoot_along(ref_line, nearest_idx, point)
        # The lane widths are the costly part, and overshooting the road's end
        # alone already puts a point further out than the best road so far.
        if best is not None and overshoot > best[0] + _SEAM_TOLERANCE:
            continue
        s = float(arc_lengths[nearest_idx])
        heading_ref = _heading_at_s(ref_line, arc_lengths, s)
        t = _signed_perp_distance(
            float(point[0]), float(point[1]), ref_line, nearest_idx, heading_ref
        )
        # The lanes stack from the lane offset line, not the reference line.
        t_lane = t - _lane_offset_at_s(road, s)
        outside = math.hypot(
            overshoot,
            max(0.0, abs(t_lane) - _side_width(road, s, t_lane >= 0.0)),
        )
        candidate = (
            outside,
            dist,
            nearest_idx == len(ref_line) - 1,
            road_id,
            road,
            s,
            t,
            heading_ref,
        )
        if best is None or _beats(candidate, best):
            best = candidate

    if best is None:
        return OpenDrivePose(road_id="", lane_id=0, s=0.0, t=0.0, heading=0.0)
    _outside, _dist, _at_end, road_id, road, s, t, heading_ref = best
    return OpenDrivePose(
        road_id=road_id,
        lane_id=_find_lane_at_t(road, s, t),
        s=s,
        t=t,
        heading=_normalize_angle(-math.radians(pose.yaw) - heading_ref),
    )


def _beats(
    candidate: tuple[float, float, bool, str, Road, float, float, float],
    best: tuple[float, float, bool, str, Road, float, float, float],
) -> bool:
    """Whether road *candidate* holds the point better than road *best*.

    Both are ``(outside, dist, at_end, ...)``.  A point on the seam between two
    connected roads is as near to the end of one as to the start of the next.
    A road covers [0, length), so the seam belongs to the road that starts
    there -- otherwise which one won would depend on the order the roads were
    read in.
    """
    outside, dist, at_end = candidate[:3]
    best_outside, best_dist, best_at_end = best[:3]
    if abs(outside - best_outside) > _SEAM_TOLERANCE:
        return outside < best_outside
    if abs(dist - best_dist) > _SEAM_TOLERANCE:
        return dist < best_dist
    return best_at_end and not at_end


def _project_onto_ref_line(
    ref_line: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    point: np.ndarray[tuple[int], np.dtype[np.float64]],
    arc_lengths: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> tuple[int, float, float, float, float]:
    """Project *point* onto *ref_line* at its nearest vertex.

    Returns ``(nearest_idx, dist, s, heading_ref, t)``: the vertex, the
    distance to it, the arc length there, the reference heading at that arc
    length and the signed lateral offset from it.
    """
    dists = np.linalg.norm(ref_line - point, axis=1)
    nearest_idx = int(np.argmin(dists))
    s = float(arc_lengths[nearest_idx])
    heading_ref = _heading_at_s(ref_line, arc_lengths, s)
    t = _signed_perp_distance(
        float(point[0]), float(point[1]), ref_line, nearest_idx, heading_ref
    )
    return nearest_idx, float(dists[nearest_idx]), s, heading_ref, t


def _overshoot_along(
    ref_line: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    nearest_idx: int,
    point: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    """How far (m) *point* lies past the start or end of *ref_line*.

    Zero unless the nearest vertex is an endpoint and the point projects
    beyond it, along the first or last segment.
    """
    if nearest_idx == 0:
        direction = ref_line[1] - ref_line[0]
        offset = point - ref_line[0]
        sign = -1.0
    elif nearest_idx == len(ref_line) - 1:
        direction = ref_line[-1] - ref_line[-2]
        offset = point - ref_line[-1]
        sign = 1.0
    else:
        return 0.0
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return 0.0
    return max(0.0, sign * float(np.dot(offset, direction)) / length)


# ---------------------------------------------------------------------------
# Inverse: CarlaWorldPose → Lanelet2Pose
# ---------------------------------------------------------------------------


def _carla_to_lanelet2(pose: CarlaWorldPose) -> Lanelet2Pose:
    """Convert a CARLA world pose to the nearest Lanelet2 Frenet pose."""
    mm = MapManager.get_instance()

    # CARLA world uses XODR-relative coords; Lanelet2 uses MGRS absolute.
    # Add the MGRS offset for x/y, and the vertical offset for z.
    offset_x, offset_y = mm.mgrs_offset
    ll2_x = pose.x + offset_x
    ll2_y = -pose.y + offset_y
    ll2_heading = -math.radians(pose.yaw)

    query = lanelet2.core.BasicPoint2d(ll2_x, ll2_y)
    results = lanelet2.geometry.findNearest(
        mm.lanelet_map.laneletLayer, query, _STACKED_CANDIDATES
    )
    # Lanelets stacked over one another -- an overpass and the road beneath
    # it -- are all as near in plan view, so the pose's elevation picks one.
    nearest = results[0][0]
    ll2_z = pose.z + mm.z_offset
    best: Optional[tuple[float, lanelet2.core.Lanelet, float, float, float]] = None
    for distance, candidate in results:
        if distance > nearest + _SEAM_TOLERANCE:
            break
        points = [(p.x, p.y, p.z) for p in candidate.centerline]
        s, t, heading_cl = _project_to_centerline(points, ll2_x, ll2_y)
        dz = abs(_interpolate_at_s(points, s)[2] - ll2_z)
        if best is None or dz < best[0]:
            best = (dz, candidate, s, t, heading_cl)
    assert best is not None
    _dz, lanelet, s, t, heading_cl = best
    heading_diff = _normalize_angle(ll2_heading - heading_cl)

    return Lanelet2Pose(lanelet_id=lanelet.id, s=s, t=t, heading=heading_diff)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _compute_arc_lengths_2d(
    points: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute cumulative arc lengths along a 2D polyline (shape N×2)."""
    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.zeros(len(points))
    arc_lengths[1:] = np.cumsum(seg_lengths)
    return arc_lengths


def _interpolate_at_s(
    points: list[tuple[float, float, float]], s: float
) -> tuple[float, float, float, float]:
    """Interpolate position and heading at arc length s along a 3D polyline.

    Returns
    -------
    (x, y, z, heading)  where heading is the tangent direction in radians.
    """
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    zs = np.array([p[2] for p in points])

    arc = _compute_arc_lengths_2d(np.column_stack([xs, ys]))
    s_clamped = float(np.clip(s, arc[0], arc[-1]))

    x = float(np.interp(s_clamped, arc, xs))
    y = float(np.interp(s_clamped, arc, ys))
    z = float(np.interp(s_clamped, arc, zs))

    # Heading via finite difference of nearest segment
    idx = int(np.searchsorted(arc, s_clamped, side="right")) - 1
    idx = max(0, min(idx, len(points) - 2))
    heading = math.atan2(ys[idx + 1] - ys[idx], xs[idx + 1] - xs[idx])

    return x, y, z, heading


def _heading_at_s(
    ref_line: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    arc_lengths: np.ndarray[tuple[int], np.dtype[np.float64]],
    s: float,
) -> float:
    """Return the tangent heading (radians) at arc length s along a 2D polyline."""
    s_clamped = float(np.clip(s, arc_lengths[0], arc_lengths[-1]))
    idx = int(np.searchsorted(arc_lengths, s_clamped, side="right")) - 1
    idx = max(0, min(idx, len(ref_line) - 2))
    dx = ref_line[idx + 1, 0] - ref_line[idx, 0]
    dy = ref_line[idx + 1, 1] - ref_line[idx, 1]
    return math.atan2(dy, dx)


def _signed_perp_distance(
    x: float,
    y: float,
    ref_line: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    nearest_idx: int,
    heading_ref: float,
) -> float:
    """Signed perpendicular distance from point (x, y) to the reference line.

    Positive = left of the reference line direction (OpenDRIVE convention).
    Uses the cross product of the heading vector and the displacement vector.
    """
    rx = ref_line[nearest_idx, 0]
    ry = ref_line[nearest_idx, 1]
    # 2D cross product: heading_vec × displacement_vec
    # heading_vec = (cos θ, sin θ), displacement = (x − rx, y − ry)
    # cross = cos θ * (y − ry) − sin θ * (x − rx)
    t = math.cos(heading_ref) * (y - ry) - math.sin(heading_ref) * (x - rx)
    return t


def _project_to_centerline(
    points: list[tuple[float, float, float]], x: float, y: float
) -> tuple[float, float, float]:
    """Project (x, y) onto a 3D polyline centerline, in plan view.

    The point is projected onto every segment at once (the foot of the
    perpendicular, clamped to the segment) and the nearest foot wins, so ``s``
    is exact along the centerline rather than snapped to a vertex.

    A foot that falls exactly on an interior vertex belongs to the segment
    that *starts* there -- the same segment :func:`_interpolate_at_s` and
    :func:`_heading_at_s` read at that arc length -- so a point projecting onto
    a vertex reads the same ``s``, ``t`` and heading as it did when this took
    the nearest vertex.  Zero-length segments (repeated points) are skipped;
    a centerline of one distinct point reads ``s = 0`` and heading 0.

    Returns
    -------
    (s, t, heading)
        s       – arc length to the foot of the perpendicular, in
                  [0, centerline length]
        t       – signed lateral offset from that segment's line
                  (positive=left); past either end, from the end segment's
                  line extended
        heading – direction of that segment (radians)
    """
    xy = np.array([(p[0], p[1]) for p in points], dtype=np.float64)
    point = np.array([x, y], dtype=np.float64)

    starts = xy[:-1]
    deltas = np.diff(xy, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    arc_starts = np.concatenate(([0.0], np.cumsum(lengths)))[:-1]
    keep = lengths > 0.0
    if not keep.any():
        # One distinct point: no direction to measure along.
        return 0.0, float(point[1] - xy[0, 1]), 0.0
    starts, deltas, lengths, arc_starts = (
        starts[keep],
        deltas[keep],
        lengths[keep],
        arc_starts[keep],
    )

    offsets = point - starts
    # Fraction of each segment at the foot of the perpendicular, clamped.
    # (Row-wise dot products written out: Codon's numpy has no einsum.)
    dot = offsets[:, 0] * deltas[:, 0] + offsets[:, 1] * deltas[:, 1]
    u = np.clip(dot / lengths**2, 0.0, 1.0)
    gap_x = offsets[:, 0] - u * deltas[:, 0]
    gap_y = offsets[:, 1] - u * deltas[:, 1]
    dists = gap_x * gap_x + gap_y * gap_y
    idx = int(np.argmin(dists))
    if u[idx] == 1.0 and idx + 1 < len(u):
        # The foot is the end vertex, which the next segment starts at: it is
        # at least as near there, and a vertex belongs to the segment it starts.
        idx += 1

    dx, dy = float(deltas[idx, 0]), float(deltas[idx, 1])
    length = float(lengths[idx])
    s = float(arc_starts[idx] + u[idx] * length)
    # 2D cross product of the unit direction with the offset: positive = left.
    ox, oy = float(offsets[idx, 0]), float(offsets[idx, 1])
    t = (dx * oy - dy * ox) / length
    return s, t, math.atan2(dy, dx)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to (−π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _lane_width_at_ds(lane: Lane, ds: float) -> float:
    """Return the width of *lane* at offset *ds* from the lane-section start.

    Lane widths in OpenDRIVE are cubic polynomials:
    ``width(ds) = a + b·ds + c·ds² + d·ds³``.
    Multiple ``<width>`` elements may define piece-wise polynomials; this
    function selects the active segment for the given *ds*.
    """
    width_elements = lane.lane_xml.findall("width")
    if not width_elements:
        return 0.0

    active = width_elements[0]
    for elem in width_elements:
        s_offset = float(elem.attrib.get("sOffset", "0.0"))
        if s_offset <= ds:
            active = elem
        else:
            break

    return _cubic(active, ds - float(active.attrib.get("sOffset", "0.0")))


def _cubic(elem: lxml.etree._Element, ds: float) -> float:
    """Evaluate an OpenDRIVE ``a + b·ds + c·ds² + d·ds³`` record at *ds*."""
    a = float(elem.attrib["a"])
    b = float(elem.attrib["b"])
    c = float(elem.attrib["c"])
    d = float(elem.attrib["d"])
    return a + b * ds + c * ds**2 + d * ds**3


def _lane_offset_at_s(road: Road, s: float) -> float:
    """The road's ``<laneOffset>`` at *s*: how far (m) its lanes sit left of
    the reference line.  Zero when the road declares none.
    """
    lanes_xml = road.road_xml.find("lanes")
    if lanes_xml is None:
        return 0.0
    # The last record starting at or before s, if any: counted rather than
    # held in a local that starts as None, which Codon cannot type.
    offsets = lanes_xml.findall("laneOffset")
    started = 0
    for elem in offsets:
        if float(elem.attrib.get("s", "0.0")) <= s:
            started += 1
        else:
            break
    if started == 0:
        return 0.0
    active = offsets[started - 1]
    return _cubic(active, s - float(active.attrib.get("s", "0.0")))


def _find_lane_section_at_s(road: Road, s: float) -> Optional[LaneSection]:
    """Return the active lane section covering arc length *s*.

    Returns ``None`` when the road has no lane sections.
    """
    lane_sections = road.lane_sections
    if not lane_sections:
        return None

    active_section = lane_sections[0]
    for section in lane_sections:
        s_start = float(section.lane_section_xml.attrib.get("s", "0.0"))
        if s_start <= s:
            active_section = section
        else:
            break
    return active_section


def _lane_center_t(road: Road, s: float, lane_id: int) -> float | None:
    """Return the *t* offset that places a point at the centre of *lane_id*.

    OpenDRIVE measures *t* from the road reference line.  For lane −1 the
    centre is at ``t = −(width_of_lane_−1 / 2)``; for lane −2 it is
    ``t = −(width_of_lane_−1 + width_of_lane_−2 / 2)``; and so on for the
    left side (positive lane IDs, positive *t*).

    Returns ``None`` when the requested lane cannot be found in the active
    lane section.
    """
    if lane_id == 0:
        return 0.0

    active_section = _find_lane_section_at_s(road, s)
    if active_section is None:
        return None

    section_s_start = float(active_section.lane_section_xml.attrib.get("s", "0.0"))
    ds = s - section_s_start

    lanes_by_id = {lane.id: lane for lane in active_section.lanes}

    sign = 1.0 if lane_id > 0 else -1.0
    target_abs = abs(lane_id)

    total = 0.0
    for abs_id in range(1, target_abs + 1):
        current_id = abs_id if lane_id > 0 else -abs_id
        lane = lanes_by_id.get(current_id)
        if lane is None:
            return None
        width = _lane_width_at_ds(lane, ds)
        if abs_id == target_abs:
            total += width / 2.0
        else:
            total += width

    return sign * total


def _find_lane_at_t(road: Road, s: float, t: float) -> int:
    """Find the lane ID at lateral offset t for a given s on a road.

    Lanes are stacked outwards from the reference line, each as wide as its
    ``<width>`` polynomial says at *s*: positive t walks the left lanes
    (positive IDs), negative t the right ones (negative IDs).  The lane whose
    band holds ``|t|`` is the answer; a point beyond the outermost lane is
    given that lane.  With Lanelet2 and OpenDRIVE kept as separate frames this
    is what tells two neighbouring lanes of one road apart.

    *t* is measured from the reference line; the lanes stack from the lane
    offset line, ``<laneOffset>`` to its left.

    Returns 0 if no lane sections are found.
    """
    t_lane = t - _lane_offset_at_s(road, s)
    edges = _lane_edges(road, s, t_lane >= 0.0)
    for lane_id, outer in edges:
        if abs(t_lane) < outer:
            return lane_id
    return edges[-1][0] if edges else 0


def _side_width(road: Road, s: float, is_left: bool) -> float:
    """Total width (m) of the lanes on one side of *road* at *s*."""
    edges = _lane_edges(road, s, is_left)
    return edges[-1][1] if edges else 0.0


def _lane_edges(road: Road, s: float, is_left: bool) -> list[tuple[int, float]]:
    """``(lane_id, outer |t|)`` of each lane on one side of *road* at *s*.

    Innermost first, as pyxodr orders a section's ``left_lanes`` and
    ``right_lanes``; empty when the road has no lane sections.
    """
    active_section = _find_lane_section_at_s(road, s)
    if active_section is None:
        return []
    ds = s - float(active_section.lane_section_xml.attrib.get("s", "0.0"))
    lanes = active_section.left_lanes if is_left else active_section.right_lanes
    edges: list[tuple[int, float]] = []
    outer = 0.0
    for lane in lanes:
        outer += _lane_width_at_ds(lane, ds)
        edges.append((lane.id, outer))
    return edges
