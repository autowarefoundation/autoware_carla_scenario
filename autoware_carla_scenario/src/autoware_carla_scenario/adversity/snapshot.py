"""Read the agents and the signals of a CARLA world for a :class:`.observation.Scene`.

The only CARLA-facing part of the observation layer: an actor becomes an
:class:`.observation.AgentSnapshot` in the Lanelet2 map frame (the y axis
and the yaw flipped, the map's offset added), a traffic light's state a
:class:`.observation.SignalState` by the Lanelet2 regulatory element that
names it.  Requires :class:`~..coordinate.map_manager.MapManager`.

A walker's speed is measured from where it stood when last read
(:class:`~..kinematics.walker_motion.WalkerVelocities`): one walked by
CARLA's controller (``controller.ai.walker``) is placed, not driven, and
reports a velocity of 0 while it walks (#923).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from ..coordinate.map_manager import MapManager
from ..coordinate.poses import CarlaWorldPose
from ..coordinate.traffic_light import find_traffic_lights_for_lanelet2_id
from ..coordinate.transform import _carla_to_lanelet2_frame
from ..entity_role import _role_name
from ..kinematics.vector import Vector3
from ..kinematics.walker_motion import WalkerVelocities, is_walker, simulated_seconds
from ..route.geometry import MapFeatures
from ..utils.clearance import _HIDDEN_BELOW_M
from ..utils.traffic_light import traffic_light_state_name
from .geometry import AgentFootprint
from .observation import AgentSnapshot, Scene, SignalState

if TYPE_CHECKING:
    import typesafe_carla.carla as carla
    from lanelet2.core import LaneletMap

__all__ = ["SignalReader", "scene_at", "snapshot_actor", "snapshot_world"]

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
    is often behind; the offset is kept (``box_ahead``, ``box_left``) so a
    path planned for the centre can put the origin back on it (#889).  The
    speed is the horizontal one; the acceleration is along its heading; a
    vehicle's speed limit is CARLA's for it.  A ``vehicle.*`` blueprint is a
    vehicle, anything else (a ``walker.*``) is approximated by one circle.

    A walker's velocity is measured from where it stood when last read, on
    its world's clock (:func:`_velocity`); with no world to ask, it is what
    CARLA reports.
    """
    transform = actor.get_transform()
    elapsed: Optional[float] = None
    if is_walker(actor):
        try:
            elapsed = _walker_clock(actor.get_world())
        except (AttributeError, RuntimeError):
            elapsed = None  # no world to ask (a test double)
    return _snapshot(
        actor, transform, name, _velocity(actor, elapsed, transform.location)
    )


#: The walkers' velocities, measured from where they stood when last read;
#: cleared when the world changes (:func:`_walker_clock`).
_WALKERS = WalkerVelocities()


class _Walked:
    """The world :data:`_WALKERS` measures walkers in."""

    world: Optional[carla.World] = None


_WALKED = _Walked()


def _walker_clock(world: carla.World) -> Optional[float]:
    """*world*'s simulated time (s), the clock walkers are measured on, or
    ``None`` where it does not say; a world other than the last one asked
    about forgets every walker measured on that one."""
    held = _WALKED.world
    if held is None or held is not world:
        _WALKED.world = world
        _WALKERS.clear()
    return simulated_seconds(world)


def _velocity(
    actor: carla.Actor, elapsed: Optional[float], location: carla.Location
) -> Vector3:
    """*actor*'s velocity: a walker's measured from where it stood when last
    read to *location*, *elapsed* the time now
    (:meth:`~..kinematics.walker_motion.WalkerVelocities.walker`), any other
    actor's what CARLA reports."""
    if is_walker(actor):
        return _WALKERS.walker(actor, elapsed, location)
    return Vector3.from_carla_vector3d(actor.get_velocity())


def _snapshot(
    actor: carla.Actor, transform: carla.Transform, name: str, velocity: Vector3
) -> AgentSnapshot:
    """:func:`snapshot_actor`, with *actor*'s transform and velocity already
    read.

    The acceleration is what CARLA reports, a walker's too: a walker's
    measured one would be the difference of two speeds each measured from a
    tick's move, and so its noise divided by the tick twice.
    """
    x, y, yaw = _carla_to_lanelet2_frame(CarlaWorldPose.from_carla_transform(transform))
    box = actor.bounding_box
    # The box's offset from the origin, in the actor's own axes: CARLA's y is
    # to the right, the map frame's to the left.
    ahead = float(box.location.x)
    left = -float(box.location.y)
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
        x + ahead * math.cos(yaw) - left * math.sin(yaw),
        y + ahead * math.sin(yaw) + left * math.cos(yaw),
        yaw,
        math.hypot(velocity.x, velocity.y),
        along,
        AgentFootprint(
            2.0 * float(box.extent.x), 2.0 * float(box.extent.y), is_vehicle
        ),
        _speed_limit(actor) if is_vehicle else None,
        ahead,
        left,
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


def snapshot_world(world: carla.World) -> list[AgentSnapshot]:
    """Every vehicle and walker of *world*, each named by its ``role_name``
    (or, with none, its actor id).

    One parked under the map (``z`` below ``_HIDDEN_BELOW_M``), as an entity
    is until its spawn, is left out: it is not in the scene (#853).  A walker's speed is
    measured from where it stood when last read (:func:`snapshot_actor`);
    one parked is forgotten, so where it is brought in is not read as a
    walk from where it was before."""
    agents: list[AgentSnapshot] = []
    elapsed = _walker_clock(world)
    for pattern in ("vehicle.*", "walker.*"):
        for actor in world.get_actors().filter(pattern):
            # One RPC for the transform, shared with the snapshot itself.
            transform = actor.get_transform()
            location = transform.location
            if float(location.z) < _HIDDEN_BELOW_M:
                if is_walker(actor):
                    _WALKERS.forget(int(actor.id))
                continue
            role = _role_name(actor)
            name = str(actor.id) if role is None else str(role)
            velocity = _velocity(actor, elapsed, location)
            agents.append(_snapshot(actor, transform, name, velocity))
    return agents


class _Shared:
    """What every ``scene_at`` caller shares: the map's features and signal
    reader for one map, and the scene of one frame of one world.

    The world is held too: a world freed and another made may get the same
    ``id()``, and its first frames would be read off the old one's scene."""

    lanelet_map: Optional[LaneletMap] = None
    features: Optional[MapFeatures] = None
    signals: Optional[SignalReader] = None
    world: Optional[carla.World] = None
    frame: int = -1
    scene: Optional[Scene] = None


_SHARED = _Shared()


def scene_at(world: carla.World) -> Scene:
    """The :class:`~.observation.Scene` of *world*'s current frame.

    Built once a frame and shared by everything that asks on that frame --
    each adversarial action's condition, NADE -- so the world is read and
    its agents located on the map once a tick, however many ask.  The map's
    :class:`~..route.geometry.MapFeatures` (and its per-lanelet caches) and
    the traffic lights' :class:`SignalReader` are made once per map.
    """
    manager = MapManager.get_instance()
    if _SHARED.lanelet_map is not manager.lanelet_map:
        _SHARED.lanelet_map = manager.lanelet_map
        _SHARED.features = MapFeatures(manager.lanelet_map, manager.routing_graph)
        _SHARED.signals = None
        _SHARED.scene = None
    held = _SHARED.world
    if held is None or held is not world:
        _SHARED.world = world
        _SHARED.signals = None
        _SHARED.scene = None
    frame = int(world.get_snapshot().frame)
    cached = _SHARED.scene
    if cached is not None and frame == _SHARED.frame:
        return cached
    features = _SHARED.features
    assert features is not None
    reader = _SHARED.signals
    if reader is None:
        reader = SignalReader(world, features.traffic_light_ids())
        _SHARED.signals = reader
    scene = Scene(features, snapshot_world(world), reader.read())
    _SHARED.frame = frame
    _SHARED.scene = scene
    return scene
