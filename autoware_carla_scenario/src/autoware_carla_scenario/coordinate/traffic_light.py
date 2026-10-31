"""Traffic lights, placed and looked up in the ambient run.

Two things live here that :mod:`..utils.traffic_light` cannot hold.

``find_nearest_traffic_light`` converts whatever pose it is given into a CARLA
location, which is this package's job; it was in ``utils`` only by accident of
naming, and dragged the whole of ``coordinate`` up behind it.

The rest supply the road network a run happens to have loaded, because
:class:`MapManager` is this package's singleton. The reading itself stays in
``utils``, where it needs nothing but an XML root.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple, Union

from ..utils.traffic_light import (
    get_signal_ids_for_controller as _signal_ids_of,
    lanelet2_traffic_light_id_to_opendrive_controller_id as _controller_id_of,
)
from .map_manager import MapManager
from .poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from .transform import to_carla_location

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


def find_nearest_traffic_light(
    world: "carla.World",
    location: Union[Lanelet2Pose, OpenDrivePose, CarlaWorldPose, "carla.Location"],
    max_distance: float = 150.0,
) -> Tuple[Optional["carla.Actor"], float]:
    """Return the nearest traffic light to *location* within *max_distance*.

    All ``traffic.traffic_light`` actors are retrieved from *world*
    automatically.

    Args:
        world: The CARLA world instance used to enumerate traffic lights.
        location: The reference position to measure distances from.
            Accepts any pose type (``Lanelet2Pose``, ``OpenDrivePose``,
            ``CarlaWorldPose``) or a raw ``carla.Location``.
        max_distance: Maximum search radius in metres.  Traffic lights
            farther than this are ignored.

    Returns:
        A ``(traffic_light, distance)`` tuple, the light being the
        ``carla.Actor`` ``world.get_actors()`` lists (its
        ``as_traffic_light()`` is the ``carla.TrafficLight``).  If no traffic
        light is found within *max_distance*, returns ``(None, float('inf'))``.
    """
    loc = to_carla_location(location)
    nearest: Optional["carla.Actor"] = None
    nearest_dist = float("inf")

    for actor in world.get_actors():
        if not actor.type_id.startswith("traffic.traffic_light"):
            continue
        dist: float = actor.get_transform().location.distance(loc)
        if dist < nearest_dist and dist < max_distance:
            nearest = actor
            nearest_dist = dist

    return nearest, nearest_dist


def lanelet2_traffic_light_id_to_opendrive_controller_id(
    lanelet2_tl_id: int,
) -> Optional[int]:
    """Return the OpenDRIVE controller ID for a Lanelet2 traffic light ID.

    Reads the road network :class:`MapManager` has loaded.  Pass one explicitly
    to :func:`..utils.traffic_light.lanelet2_traffic_light_id_to_opendrive_controller_id`
    to avoid requiring the singleton.
    """
    return _controller_id_of(MapManager.get_instance().road_network, lanelet2_tl_id)


def get_signal_ids_for_controller(controller_id: int) -> list[str]:
    """Return the signal IDs controlled by an OpenDRIVE controller.

    Reads the road network :class:`MapManager` has loaded.
    """
    return _signal_ids_of(MapManager.get_instance().road_network, controller_id)


#: A CARLA light stands for a Lanelet2 light no OpenDRIVE controller is named
#: after when one of its stop waypoints lies this near (m) to that light's stop
#: line, facing across it (:func:`lights_on_stop_line`).
STOP_LINE_MATCH_M = 2.0


def stop_line_in_carla(
    lanelet2_tl_id: int,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """The ends of Lanelet2 light *lanelet2_tl_id*'s stop line, in CARLA world
    ``(x, y)``, or ``None`` when the loaded map has no such light or line."""
    if TYPE_CHECKING:
        return None  # the Lanelet2 map's geometry is not compiled (docs/typecheck.md)
    from ..trajectory.model import MapPose  # noqa: PLC0415
    from ..trajectory.resolve import map_pose_to_carla  # noqa: PLC0415

    try:
        manager = MapManager.get_instance()
        element = manager.lanelet_map.regulatoryElementLayer[int(lanelet2_tl_id)]
        line = element.stopLine
        offset, z_offset = manager.mgrs_offset, manager.z_offset
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    if line is None or len(line) < 2:
        return None
    ends = []
    for point in (line[0], line[len(line) - 1]):
        x, y, _z, _yaw = map_pose_to_carla(
            MapPose(x=float(point.x), y=float(point.y)), offset, z_offset
        )
        ends.append((x, y))
    return ends[0], ends[1]


def lights_on_stop_line(
    world: "carla.World",
    line: tuple[tuple[float, float], tuple[float, float]],
) -> "list[carla.Actor]":
    """The CARLA lights with a stop waypoint on *line* (CARLA ``(x, y)``):
    within :data:`STOP_LINE_MATCH_M` of it, its lane crossing it -- not a
    light of the road beside, whose lanes run along it."""
    (ax, ay), (bx, by) = line
    span = math.hypot(bx - ax, by - ay)
    if span <= 0.0:
        return []
    ux, uy = (bx - ax) / span, (by - ay) / span
    found: "list[carla.Actor]" = []
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        for waypoint in actor.get_stop_waypoints():
            location = waypoint.transform.location
            forward = waypoint.transform.get_forward_vector()
            along = min(max((location.x - ax) * ux + (location.y - ay) * uy, 0.0), span)
            gap = math.hypot(
                location.x - (ax + ux * along), location.y - (ay + uy * along)
            )
            if gap <= STOP_LINE_MATCH_M and abs(forward.x * ux + forward.y * uy) < 0.5:
                found.append(actor)
                break
    return found


def find_traffic_lights_for_lanelet2_id(
    world: "carla.World", lanelet2_tl_id: int
) -> "list[carla.Actor]":
    """Return the CARLA traffic lights a Lanelet2 regulatory element names.

    The Lanelet2 id resolves to an OpenDRIVE controller and thence to the
    signal ids the controller drives.  A light no controller is named after
    (a roadgen map, #659 and #901) is found by its stop line instead
    (:func:`lights_on_stop_line`).  This is the one resolution
    :class:`~autoware_carla_scenario.actions.TrafficSignalAction`,
    :class:`~autoware_carla_scenario.conditions.TrafficSignalCondition` and
    the signal controllers all make, so that a light one of them sets is the
    light the others read and drive.

    Args:
        world: The CARLA world.
        lanelet2_tl_id: Lanelet2 regulatory element id.

    Returns:
        The matching light actors, empty when the id resolves to nothing.
    """
    controller_id = lanelet2_traffic_light_id_to_opendrive_controller_id(lanelet2_tl_id)
    if controller_id is None:
        line = stop_line_in_carla(lanelet2_tl_id)
        if line is None:
            return []
        return lights_on_stop_line(world, line)
    # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
    controller: int = controller_id
    signal_ids = set(get_signal_ids_for_controller(controller))
    if not signal_ids:
        return []
    return [
        actor
        for actor in world.get_actors().filter("traffic.traffic_light*")
        if actor.get_opendrive_id() in signal_ids
    ]
