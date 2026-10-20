"""Read the agents and the signals of a CARLA world for a :class:`.observation.Scene`.

The only CARLA-facing part of the observation layer: an actor becomes an
:class:`.observation.AgentSnapshot` in the Lanelet2 map frame (the y axis
and the yaw flipped, the map's offset added), a traffic light's state a
:class:`.observation.SignalState` by the Lanelet2 regulatory element that
names it.  Requires :class:`~..coordinate.map_manager.MapManager`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..coordinate.poses import CarlaWorldPose
from ..coordinate.traffic_light import find_traffic_lights_for_lanelet2_id
from ..coordinate.transform import _carla_to_lanelet2_frame
from ..route.geometry import MapFeatures
from ..utils.traffic_light import traffic_light_state_name
from .geometry import AgentFootprint
from .observation import AgentSnapshot, SignalState, _is_traffic_light

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["read_signal_states", "snapshot_actor", "traffic_light_ids"]

_STATES = {
    "Red": SignalState.RED,
    "Yellow": SignalState.YELLOW,
    "Green": SignalState.GREEN,
}


def snapshot_actor(actor: carla.Actor, name: str) -> AgentSnapshot:
    """*actor*'s state now, as *name*, in the Lanelet2 map frame.

    The speed is the horizontal one; the acceleration is along the actor's
    heading.  A ``vehicle.*`` blueprint is a vehicle, anything else (a
    ``walker.*``) is approximated by one circle.
    """
    transform = actor.get_transform()
    location = transform.location
    x, y, yaw = _carla_to_lanelet2_frame(
        CarlaWorldPose(
            float(location.x),
            float(location.y),
            float(location.z),
            yaw=float(transform.rotation.yaw),
        )
    )
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    forward = transform.get_forward_vector()
    # The dot product of two CARLA-frame vectors is the same in any frame.
    along = (
        float(acceleration.x) * float(forward.x)
        + float(acceleration.y) * float(forward.y)
        + float(acceleration.z) * float(forward.z)
    )
    extent = actor.bounding_box.extent
    return AgentSnapshot(
        name,
        x,
        y,
        yaw,
        math.hypot(float(velocity.x), float(velocity.y)),
        along,
        AgentFootprint(
            2.0 * float(extent.x),
            2.0 * float(extent.y),
            str(actor.type_id).startswith("vehicle."),
        ),
    )


def traffic_light_ids(features: MapFeatures) -> list[int]:
    """The ids of the traffic-light regulatory elements the map's lanelets name."""
    ids: list[int] = []
    seen: set[int] = set()
    for lanelet in features.map.laneletLayer:
        for element in lanelet.regulatoryElements:
            key = int(element.id)
            if key in seen or not _is_traffic_light(element):
                continue
            seen.add(key)
            ids.append(key)
    return ids


def read_signal_states(
    world: carla.World, regulatory_ids: list[int]
) -> dict[int, SignalState]:
    """The state of each traffic light named by *regulatory_ids*.

    A Lanelet2 element no CARLA light answers for is left out (it reads as
    UNKNOWN in a scene); one that several lights answer for takes the first.
    """
    states: dict[int, SignalState] = {}
    for regulatory_id in regulatory_ids:
        lights = find_traffic_lights_for_lanelet2_id(world, regulatory_id)
        if not lights:
            continue
        name = traffic_light_state_name(lights[0].get_state())
        states[regulatory_id] = _STATES.get(name, SignalState.UNKNOWN)
    return states
