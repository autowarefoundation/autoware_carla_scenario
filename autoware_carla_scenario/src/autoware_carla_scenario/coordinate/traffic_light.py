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

from typing import TYPE_CHECKING, Optional, Tuple, Union

from ..utils.traffic_light import (
    get_signal_ids_for_controller as _signal_ids_of,
    lanelet2_traffic_light_id_to_opendrive_controller_id as _controller_id_of,
)
from .map_manager import MapManager
from .poses import AnyPose
from .transform import to_carla_location

if TYPE_CHECKING:
    import carla


def find_nearest_traffic_light(
    world: "carla.World",
    location: Union[AnyPose, "carla.Location"],
    max_distance: float = 150.0,
) -> Tuple[Optional["carla.TrafficLight"], float]:
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
        A ``(traffic_light, distance)`` tuple.  If no traffic light is found
        within *max_distance*, returns ``(None, float('inf'))``.
    """
    loc = to_carla_location(location)
    nearest: Optional["carla.TrafficLight"] = None
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
