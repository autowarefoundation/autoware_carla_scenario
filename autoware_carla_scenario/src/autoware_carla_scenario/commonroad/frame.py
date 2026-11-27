"""A frame of the world as the CommonRoad worker reads it.

Every road user -- the ego, the vehicles and the pedestrians near it on its
level of road -- as an actor of ``commonroad_metrics.Actor``: where its box's
centre is, where it heads, its velocity and acceleration, its box and its
kind, in the CommonRoad frame of the map.  That frame is the OpenDRIVE's
(roadgen writes the scenario in it): right-handed, where CARLA's is
left-handed, so ``y``, the yaw and the yaw rate change sign
(:func:`to_commonroad`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ..constants import EGO_ROLE_NAME
from ..entity_role import find_actor_in_world
from ..t4_settings import category_of
from ..utils.clearance import (
    HIDDEN_BELOW_M,
    actor_box_centre,
    actor_vertical_span,
    on_another_level,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["Frame", "read_frame", "to_commonroad"]

#: One road user, as the worker's ``Actor`` reads it.
ActorState = dict[str, Any]


class Frame:
    """The ego and the road users near it, in the CommonRoad frame.

    Attributes:
        ego: The ego.
        others: The other vehicles and pedestrians within the radius.
    """

    ego: ActorState
    others: list[ActorState]

    def __init__(self, ego: ActorState, others: list[ActorState]) -> None:
        self.ego = ego
        self.others = others


def to_commonroad(x: float, y: float) -> tuple[float, float]:
    """CARLA's ``(x, y)`` in the CommonRoad (OpenDRIVE) frame."""
    return x, -y


def _kind(actor: "carla.Actor") -> str:
    """*actor*'s CommonRoad kind: its T4 category (:func:`category_of`),
    whose names are the worker's -- ``pedestrian`` for a walker, a vehicle's
    from its ``base_type``, else ``car`` (its wheels are not read)."""
    try:
        base = str(actor.attributes.get("base_type", ""))
    except Exception:
        base = ""
    return category_of(str(actor.type_id), base)


@dataclass(frozen=True)
class _Kinematics:
    """Where a road user's box is centred and how it moves, in the CommonRoad
    frame: the typed part of its state (:func:`_kinematics`)."""

    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    ax: float
    ay: float
    yaw_rate: float
    length: float
    width: float


def _kinematics(actor: "carla.Actor") -> _Kinematics:
    """*actor*'s box centre, heading, velocity, acceleration, yaw rate and box
    size in the CommonRoad frame.  An acceleration or an angular velocity that
    cannot be read is zero."""
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    box = actor.bounding_box
    # The box's centre, which may sit off the actor's origin.
    cx, cy = actor_box_centre(actor, transform)
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
    return _Kinematics(
        x=x,
        y=y,
        yaw=-yaw,
        vx=float(velocity.x),
        vy=-float(velocity.y),
        ax=ax,
        ay=ay,
        yaw_rate=yaw_rate,
        length=max(2.0 * float(box.extent.x), 0.1),
        width=max(2.0 * float(box.extent.y), 0.1),
    )


class _RoadUser:
    """One road user as the worker is sent it, typed: its CommonRoad id, its
    kind and its kinematics (:func:`_state` makes the worker's object of it)."""

    commonroad_id: int
    kind: str
    kinematics: _Kinematics

    def __init__(self, commonroad_id: int, kind: str, kinematics: _Kinematics) -> None:
        self.commonroad_id = commonroad_id
        self.kind = kind
        self.kinematics = kinematics


def _road_user(actor: "carla.Actor") -> _RoadUser:
    """*actor* as a road user; its id is CARLA's, which is positive, as
    CommonRoad's must be."""
    return _RoadUser(int(actor.id), _kind(actor), _kinematics(actor))


def _state(user: _RoadUser) -> ActorState:
    k = user.kinematics
    return {
        "id": user.commonroad_id,
        "kind": user.kind,
        "x": k.x,
        "y": k.y,
        "yaw": k.yaw,
        "vx": k.vx,
        "vy": k.vy,
        "ax": k.ax,
        "ay": k.ay,
        "yaw_rate": k.yaw_rate,
        "length": k.length,
        "width": k.width,
    }


def _road_users_near(
    world: "carla.World", ego: "carla.Actor", radius_m: float
) -> list["carla.Actor"]:
    """The vehicles and walkers of *world* other than *ego*, within *radius_m*
    of it (in the plane), not hidden under the map and on the ego's level of
    road, in the world's order.

    CommonRoad's scenario is planar: a road user on a bridge over the ego, or
    in an underpass beneath it, would be on its lane there, and a TTC or a
    headway to it would read a collision the two can never have (#1580).
    One up or down a slope from the ego, within the road's grade over the
    distance between them, is on its level (#1597).
    """
    near: list["carla.Actor"] = []
    placed = ego.get_transform()
    centre = placed.location
    ego_span = actor_vertical_span(ego, placed)
    for actor in world.get_actors():
        type_id = str(actor.type_id)
        if actor.id == ego.id or not (
            type_id.startswith("vehicle.") or type_id.startswith("walker.")
        ):
            continue
        transform = actor.get_transform()
        location = transform.location
        # A road user parked under the map (hidden) is out of the scene.
        if location.z < HIDDEN_BELOW_M:
            continue
        planar = math.hypot(location.x - centre.x, location.y - centre.y)
        if planar > radius_m:
            continue
        if on_another_level(ego_span, actor_vertical_span(actor, transform), planar):
            continue
        near.append(actor)
    return near


def _read_road_users(
    world: "carla.World", radius_m: float
) -> Optional[tuple[_RoadUser, list[_RoadUser]]]:
    """The ego and the road users within *radius_m* of it
    (:func:`_road_users_near`), typed; ``None`` without an ego."""
    ego = find_actor_in_world(world, EGO_ROLE_NAME)
    if ego is None:
        return None
    ego_user = _road_user(ego)
    others: list[_RoadUser] = [
        _road_user(actor) for actor in _road_users_near(world, ego, radius_m)
    ]
    return ego_user, others


def read_frame(world: "carla.World", radius_m: float) -> Optional[Frame]:
    """The ego and the road users within *radius_m* of it; ``None`` without an
    ego.  Ids are CARLA's, which are positive, as CommonRoad's must be."""
    read = _read_road_users(world, radius_m)
    if read is None:
        return None
    ego, others = read
    return Frame(_state(ego), [_state(user) for user in others])
