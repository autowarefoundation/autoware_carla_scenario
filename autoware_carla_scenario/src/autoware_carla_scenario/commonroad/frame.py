"""A frame of the world as the CommonRoad worker reads it.

Every road user -- the ego, the vehicles and the pedestrians near it -- as an
actor of ``commonroad_metrics.Actor``: where its box's centre is, where it
heads, its velocity and acceleration, its box and its kind, in the CommonRoad
frame of the map.  That frame is the OpenDRIVE's (roadgen writes the scenario
in it): right-handed, where CARLA's is left-handed, so ``y``, the yaw and the
yaw rate change sign (:func:`to_commonroad`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

from ..constants import EGO_ROLE_NAME
from ..entity_role import find_actor_in_world
from ..utils.clearance import _HIDDEN_BELOW_M

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["Frame", "read_frame", "to_commonroad"]

#: The CommonRoad kind of a CARLA vehicle's ``base_type``.
_KINDS = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}

#: One road user, as the worker's ``Actor`` reads it.
ActorState = dict[str, Any]


class Frame:
    """The ego and the road users near it, in the CommonRoad frame.

    Attributes:
        ego: The ego.
        others: The other vehicles and pedestrians within the radius.
    """

    def __init__(self, ego: ActorState, others: list[ActorState]) -> None:
        self.ego = ego
        self.others = others


def to_commonroad(x: float, y: float) -> tuple[float, float]:
    """CARLA's ``(x, y)`` in the CommonRoad (OpenDRIVE) frame."""
    return x, -y


def _kind(actor: "carla.Actor") -> str:
    type_id = str(actor.type_id)
    if type_id.startswith("walker."):
        return "pedestrian"
    try:
        base = str(actor.attributes.get("base_type", "")).lower()
    except Exception:
        base = ""
    return _KINDS.get(base, "car")


def _state(actor: "carla.Actor", commonroad_id: int) -> ActorState:
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    box = actor.bounding_box
    # The box's centre, which may sit off the actor's origin.
    off_x, off_y = float(box.location.x), float(box.location.y)
    cx = transform.location.x + off_x * math.cos(yaw) - off_y * math.sin(yaw)
    cy = transform.location.y + off_x * math.sin(yaw) + off_y * math.cos(yaw)
    x, y = to_commonroad(cx, cy)
    velocity = actor.get_velocity()
    try:
        acceleration = actor.get_acceleration()
        ax, ay = float(acceleration.x), -float(acceleration.y)
    except Exception:
        ax, ay = 0.0, 0.0
    try:
        yaw_rate = -math.radians(float(actor.get_angular_velocity().z))
    except Exception:
        yaw_rate = 0.0
    return {
        "id": commonroad_id,
        "kind": _kind(actor),
        "x": x,
        "y": y,
        "yaw": -yaw,
        "vx": float(velocity.x),
        "vy": -float(velocity.y),
        "ax": ax,
        "ay": ay,
        "yaw_rate": yaw_rate,
        "length": max(2.0 * float(box.extent.x), 0.1),
        "width": max(2.0 * float(box.extent.y), 0.1),
    }


def read_frame(world: "carla.World", radius_m: float) -> Optional[Frame]:
    """The ego and the road users within *radius_m* of it; ``None`` without an
    ego.  Ids are CARLA's, which are positive, as CommonRoad's must be."""
    ego = find_actor_in_world(world, EGO_ROLE_NAME)
    if ego is None:
        return None
    ego_state = _state(ego, int(ego.id))
    others: list[ActorState] = []
    centre = ego.get_transform().location
    for actor in world.get_actors():
        type_id = str(actor.type_id)
        if actor.id == ego.id or not (
            type_id.startswith("vehicle.") or type_id.startswith("walker.")
        ):
            continue
        location = actor.get_transform().location
        # A road user parked under the map (hidden) is out of the scene.
        if location.z < _HIDDEN_BELOW_M:
            continue
        if math.hypot(location.x - centre.x, location.y - centre.y) > radius_m:
            continue
        others.append(_state(actor, int(actor.id)))
    return Frame(ego_state, others)
