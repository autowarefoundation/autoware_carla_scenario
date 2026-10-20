"""Read the agents and the signals of a CARLA world for a :class:`.observation.Scene`.

The only CARLA-facing part of the observation layer: an actor becomes an
:class:`.observation.AgentSnapshot` in the Lanelet2 map frame (the y axis
and the yaw flipped, the map's offset added), a traffic light's state a
:class:`.observation.SignalState` by the Lanelet2 regulatory element that
names it.  Requires :class:`~..coordinate.map_manager.MapManager`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..coordinate.poses import CarlaWorldPose
from ..coordinate.traffic_light import find_traffic_lights_for_lanelet2_id
from ..coordinate.transform import _carla_to_lanelet2_frame
from ..utils.traffic_light import traffic_light_state_name
from .geometry import AgentFootprint
from .observation import AgentSnapshot, SignalState

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["SignalReader", "snapshot_actor"]

_STATES = {
    "Red": SignalState.RED,
    "Yellow": SignalState.YELLOW,
    "Green": SignalState.GREEN,
}


def _speed_limit(actor: carla.Actor) -> Optional[float]:
    """The simulator's speed limit for a vehicle (km/h), as the ODD's
    ``speed_limit_kph`` falls back to it; None where it reports none."""
    # The speed limit is a vehicle's: a plain carla.Actor does not have it.
    if TYPE_CHECKING:
        vehicle = actor.as_vehicle()
    else:
        vehicle = actor.as_vehicle() if hasattr(actor, "as_vehicle") else actor
    limit = float(vehicle.get_speed_limit() or 0.0)
    if limit > 0.0:
        return limit
    return None


def snapshot_actor(actor: carla.Actor, name: str) -> AgentSnapshot:
    """*actor*'s state now, as *name*, in the Lanelet2 map frame.

    The position is the centre of its bounding box, which a vehicle's origin
    is often behind; the speed is the horizontal one; the acceleration is
    along its heading; a vehicle's speed limit is CARLA's for it.  A ``vehicle.*`` blueprint is a vehicle, anything else
    (a ``walker.*``) is approximated by one circle.
    """
    transform = actor.get_transform()
    x, y, yaw = _carla_to_lanelet2_frame(CarlaWorldPose.from_carla_transform(transform))
    box = actor.bounding_box
    ahead = float(box.location.x)
    velocity = actor.get_velocity()
    acceleration = actor.get_acceleration()
    forward = transform.get_forward_vector()
    # The dot product of two CARLA-frame vectors is the same in any frame.
    along = (
        float(acceleration.x) * float(forward.x)
        + float(acceleration.y) * float(forward.y)
        + float(acceleration.z) * float(forward.z)
    )
    is_vehicle = str(actor.type_id).startswith("vehicle.")
    return AgentSnapshot(
        name,
        x + ahead * math.cos(yaw),
        y + ahead * math.sin(yaw),
        yaw,
        math.hypot(float(velocity.x), float(velocity.y)),
        along,
        AgentFootprint(
            2.0 * float(box.extent.x), 2.0 * float(box.extent.y), is_vehicle
        ),
        _speed_limit(actor) if is_vehicle else None,
    )


class SignalReader:
    """The states of the traffic lights a map names, read off a CARLA world.

    Which CARLA lights answer for each Lanelet2 traffic-light element is
    resolved once, when the reader is made (the lights do not move); each
    :meth:`read` then only asks them their state.  An element no light
    answers for is left out (it reads as UNKNOWN in a scene); one several
    lights answer for takes the first.

    Args:
        world: The CARLA world.
        regulatory_ids: The elements to read, e.g.
            :meth:`~..route.geometry.MapFeatures.traffic_light_ids`.
    """

    _lights: dict[int, carla.Actor]

    def __init__(self, world: carla.World, regulatory_ids: list[int]) -> None:
        self._lights = {}
        for regulatory_id in regulatory_ids:
            lights = find_traffic_lights_for_lanelet2_id(world, regulatory_id)
            if lights:
                self._lights[regulatory_id] = lights[0]

    def read(self) -> dict[int, SignalState]:
        """The state of each light now, by its element's id."""
        states: dict[int, SignalState] = {}
        for regulatory_id, light in self._lights.items():
            name = traffic_light_state_name(light.get_state())
            states[regulatory_id] = _STATES.get(name, SignalState.UNKNOWN)
        return states
