"""Pedestrian entity for CARLA scenarios."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import typesafe_carla.carla as carla

from ..traffic.base import TrafficBackend, WalkedEntity

if TYPE_CHECKING:
    from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
    from ..trajectory.model import ResolvedTrajectory
    from ._spawn import SpawnTransform

logger = logging.getLogger(__name__)

__all__ = ["PedestrianEntity", "PedestrianEntityConfig"]

#: Metres the spawn point is raised before the walker is placed, tried in turn
#: until CARLA accepts one.  A pose taken off the map sits exactly on the
#: surface, and a walker spawned flush with it is rejected as colliding with
#: the ground; one at the roadside starts inside the kerb unless it is lifted
#: clear of it, by more than the ground needs.
_SPAWN_LIFTS_M = (0.3, 0.8, 1.3)

#: CARLA 0.10 (UE5) moves a walker at about 1/20.5 of the ``WalkerControl``
#: speed it is given, whatever the step length -- measured on 0.10.0: 2 -> 0.098
#: m/s, 20 -> 0.98 m/s, 40 -> 1.95 m/s -- and no faster than 2 m/s.
_UE5_WALKER_SPEED_SCALE = 20.5
_UE5_WALKER_MAX_SPEED_MS = 2.0

#: Metres between the poses of a walk :meth:`PedestrianEntity.route_to` lays out.
WALK_POINT_STEP_M: float = 1.5

#: How far ahead (m) along its path a walker is steered at (:func:`steer_walker`).
WALKER_LOOKAHEAD_M: float = 1.5

#: How near (m) the end of its walk a walker has arrived, as a
#: :class:`~autoware_carla_scenario.actions.RoutingAction` following a
#: trajectory takes it.
WALK_ARRIVAL_M: float = 1.0

#: A pedestrian's walking speed along a route, in m/s, unless it is told one.
DEFAULT_WALKING_SPEED_MPS: float = 1.4


def walker_speed_command(speed_ms: float) -> float:
    """The ``WalkerControl`` speed that moves a walker at *speed_ms* (CARLA UE5)."""
    if speed_ms > _UE5_WALKER_MAX_SPEED_MS:
        logger.warning(
            "CARLA 0.10 walks a pedestrian at %.1f m/s at most; %.1f m/s asked for",
            _UE5_WALKER_MAX_SPEED_MS,
            speed_ms,
        )
    return speed_ms * _UE5_WALKER_SPEED_SCALE


def walker_control_speed(speed_mps: float) -> float:
    """The ``WalkerControl`` speed for *speed_mps*, capped at what CARLA walks.

    Capped rather than left to warn: a walk re-sends it every tick.
    """
    return walker_speed_command(min(max(speed_mps, 0.0), _UE5_WALKER_MAX_SPEED_MS))


def steer_walker(
    actor: "carla.Actor",
    location: "carla.Location",
    path: "ResolvedTrajectory",
    here: float,
    speed_mps: float,
) -> None:
    """One tick of walking *actor*, at *location*, along *path*: towards the
    point :data:`WALKER_LOOKAHEAD_M` past *here* (m along it), at *speed_mps*.

    What a :class:`~autoware_carla_scenario.actions.RoutingAction` in
    ``follow`` mode and a routed pedestrian (:meth:`PedestrianEntity.walk_on`)
    both send.
    """
    target = path.at_distance(here + WALKER_LOOKAHEAD_M)
    dx, dy = target.x - location.x, target.y - location.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        speed_mps, dx, dy, norm = 0.0, 1.0, 0.0, 1.0
    actor.apply_control(
        carla.WalkerControl(
            direction=carla.Vector3D(x=dx / norm, y=dy / norm, z=0.0),
            speed=walker_control_speed(speed_mps),
        )
    )


def _walk_by_ai(
    walker: "carla.Actor",
    path: "ResolvedTrajectory",
    here: float,
    speed: float,
    location: "carla.Location",
    cross_factor: Optional[float],
) -> bool:
    """One tick of *walker* on CARLA's walker controller
    (:mod:`._walker_ai`); ``False`` on a map without a navigation mesh."""
    from ._walker_ai import drive  # noqa: PLC0415

    try:
        return drive(walker, path, here, speed, location, cross_factor)
    except AttributeError:
        return False  # a walker with no world to ask (a test double)


def _gives_way(
    walker: "carla.Actor",
    path: "ResolvedTrajectory",
    here: float,
    location: "carla.Location",
) -> bool:
    """Whether *walker*, stepped, stands this tick for a walker in its way
    (:func:`._walker_ai.give_way`)."""
    from ._walker_ai import give_way  # noqa: PLC0415

    try:
        return give_way(walker, path, here, location)
    except (AttributeError, RuntimeError):
        return False  # a walker with no world to ask (a test double)


def _release_walker_ai(walker: "carla.Actor") -> None:
    """Stop and destroy *walker*'s controller, if it has one."""
    from ._walker_ai import release  # noqa: PLC0415

    try:
        release(int(walker.id))
    except (AttributeError, RuntimeError):
        pass


@dataclass
class PedestrianEntityConfig:
    """Configuration for spawning a pedestrian.

    Attributes:
        role_name: The role the pedestrian answers to, as CARLA's
            ``role_name`` string.  An :class:`EntityRole` is accepted and
            stored as its string (:meth:`__post_init__`).
        spawn_location: Where to place it, as a transform.
        walker_type: CARLA blueprint id; must be a ``walker.*``.
    """

    role_name: str
    spawn_location: "SpawnTransform"
    walker_type: str = "walker.pedestrian.0001"

    def __post_init__(self) -> None:
        # Stored as a string: CARLA's role_name is one, every reader converts
        # it to one anyway, and the static check (docs/typecheck.md) needs an
        # attribute of one type.
        self.role_name = str(self.role_name)


class PedestrianEntity(WalkedEntity):
    """Manages the lifecycle of a pedestrian actor in CARLA.

    Deliberately **not** a :class:`~autoware_carla_scenario.traffic.driven.BackendDriven`
    entity: inheriting the vehicles' manoeuvres would offer ``change_lane`` on
    something that has no lane.  A pedestrian moves because a
    :class:`~autoware_carla_scenario.actions.WalkStraightAction` tells it to,
    and stands still otherwise.  The walk goes to the run's traffic backend
    first (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.walk_straight`):
    one that walks the scenario's pedestrians -- SUMO, with
    ``scenario_pedestrians=sumo`` -- takes it, and the walker follows that
    backend's pedestrian.  Otherwise the entity walks its walker itself.

    Walking it itself is commanded with :class:`carla.WalkerControl` rather than through
    a ``controller.ai.walker``.  The AI controller exists to send a walker *to
    a destination*, which is a different scenario primitive and one nothing
    here asks for yet; using it to walk straight would mean inventing a
    destination far enough away to look like "straight", and the walker would
    then route around obstacles the scenario put there on purpose.
    """

    # Declared for the static check (docs/typecheck.md): Codon types an
    # attribute from its declaration; to Python it is a bare annotation.
    _config: PedestrianEntityConfig
    _walker: Optional["carla.Actor"]
    _traffic_backend: Optional[TrafficBackend]
    _walk: Optional["ResolvedTrajectory"]
    _walk_distance: float
    _walk_speed: float
    _walk_cross: Optional[float]

    def __init__(self, config: PedestrianEntityConfig) -> None:
        self._config = config
        self._walker: Optional["carla.Actor"] = None
        self._traffic_backend: Optional[TrafficBackend] = None
        self._walk = None
        self._walk_distance = 0.0
        self._walk_speed = DEFAULT_WALKING_SPEED_MPS
        self._walk_cross = None

    def set_traffic_backend(self, backend: TrafficBackend) -> None:
        """Hand this pedestrian's walks to *backend*, the run's traffic backend
        (:meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_pedestrian`
        calls it)."""
        self._traffic_backend = backend

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def role_name(self) -> str:
        """Return the role name that identifies this entity."""
        return self._config.role_name

    @property
    def walker_type(self) -> str:
        """Return the CARLA blueprint id for this pedestrian."""
        return self._config.walker_type

    @property
    def actor(self) -> Optional["carla.Actor"]:
        """Return the underlying CARLA actor, or ``None`` if not spawned."""
        return self._walker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def spawn(self, world: "carla.World") -> "carla.Actor":
        """Spawn the pedestrian in the CARLA world.

        Raises:
            ValueError: If the blueprint is not available.
            RuntimeError: If CARLA refused the spawn point.
        """
        blueprint_library = world.get_blueprint_library()
        try:
            blueprint = blueprint_library.find(self._config.walker_type)
        # One clause per exception type, for the static check
        # (docs/typecheck.md): Codon 0.19 does not take a tuple.
        except IndexError as exc:
            raise ValueError(
                f"pedestrian blueprint {self._config.walker_type!r} is not "
                f"available in this CARLA build"
            ) from exc
        except RuntimeError as exc:
            raise ValueError(
                f"pedestrian blueprint {self._config.walker_type!r} is not "
                f"available in this CARLA build"
            ) from exc

        # Not guarded on the attribute existing.  Every entity-based condition
        # -- existence, speed, distance, TTC, waypoint -- finds its actor by
        # `role_name`, so a walker spawned without one is live in the world and
        # invisible to the scenario: each of those conditions would read it as
        # absent and never evaluate.  A blueprint that cannot carry the name is
        # therefore unusable here, and saying so beats spawning a pedestrian
        # nothing can assert about.
        if not blueprint.has_attribute("role_name"):
            raise ValueError(
                f"pedestrian blueprint {self._config.walker_type!r} has no "
                f"role_name attribute, so no condition could find it; pick a "
                f"walker blueprint that has one"
            )
        blueprint.set_attribute("role_name", self._config.role_name)
        # A walker that another actor can push is a walker the scenario no
        # longer controls; every pedestrian here is scripted.
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")

        walker: Optional["carla.Actor"] = None
        for lift in _SPAWN_LIFTS_M:
            transform = self._lifted_transform(lift)
            walker = world.try_spawn_actor(blueprint, transform)
            if walker is not None:
                break
        if walker is None:
            raise RuntimeError(
                f"could not spawn pedestrian {self._config.role_name!r} at "
                f"{transform.location}"
            )
        self._walker = walker
        logger.info(
            "Spawned pedestrian '%s' (%s) at %s",
            self._config.role_name,
            self._config.walker_type,
            transform.location,
        )
        return walker

    def destroy(self) -> None:
        """Destroy the pedestrian actor and release resources (its walker
        controller, first)."""
        if self._walker is not None:
            if TYPE_CHECKING:
                pass  # the walker controller is not compiled (docs/typecheck.md)
            else:
                _release_walker_ai(self._walker)
            self._walker.destroy()
            self._walker = None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def walk_straight(self, speed_ms: float, yaw_deg: Optional[float] = None) -> None:
        """Walk forward at *speed_ms*, holding the current heading -- or along
        *yaw_deg*, when given.

        The run's traffic backend walks it when it walks the scenario's
        pedestrians (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.walk_straight`);
        otherwise the walker is walked here.

        Args:
            speed_ms: Walking speed in metres per second.  Zero stops the
                walker where it stands.
            yaw_deg: The way to walk, as a CARLA yaw (degrees); ``None`` walks
                the way the walker faces.
        """
        if self._walker is None:
            logger.warning(
                "PedestrianEntity '%s': not spawned, cannot walk",
                self._config.role_name,
            )
            return
        backend = self._traffic_backend
        if backend is not None:
            walked: WalkedEntity = self
            if backend.walk_straight(walked, speed_ms, yaw_deg=yaw_deg):
                return
        if yaw_deg is None:
            direction = self._walker.get_transform().get_forward_vector()
        else:
            yaw = math.radians(yaw_deg)
            direction = carla.Vector3D(x=math.cos(yaw), y=math.sin(yaw), z=0.0)
        self._walker.apply_control(
            carla.WalkerControl(
                direction=direction, speed=walker_speed_command(speed_ms)
            )
        )

    def route_to(
        self,
        world: "carla.World",
        goal: "Lanelet2Pose",
        *,
        initial_pose: "Optional[CarlaWorldPose]" = None,
        ground_projection: "Optional[GroundProjectionConfig]" = None,
        waypoints: "Sequence[Lanelet2Pose]" = (),
        walking_speed: float = DEFAULT_WALKING_SPEED_MPS,
        cross_factor: Optional[float] = None,
    ) -> None:
        """Lay out a walk from where the walker is through *waypoints* to *goal*,
        which :meth:`walk_on` then walks a tick at a time; see
        ``docs/architecture.md``.  *ground_projection* is not used.
        *cross_factor* is the probability that CARLA's walker controller may
        take it across a road away from a crossing (``None``: CARLA's).

        Each leg is the shortest path on the walkable lanelets where there is
        one, and straight where there is none -- across a road whose crosswalks
        only overlap its walkways, unless a crosswalk is a waypoint.
        """
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.poses import CarlaWorldPose  # noqa: PLC0415
        from ..coordinate.transform import to_carla_world  # noqa: PLC0415
        from ..odd.route_plan import PlannedRoute, route_poses  # noqa: PLC0415
        from ..trajectory.model import ResolvedTrajectory  # noqa: PLC0415

        self._walk = None
        actor = self._walker
        if actor is None:
            logger.warning(
                "PedestrianEntity '%s': not spawned, no walk to plan",
                self._config.role_name,
            )
            return
        # A typed local: Codon unwraps the Optional here.
        walking: "carla.Actor" = actor
        start = initial_pose
        if start is None:
            start = CarlaWorldPose.from_carla_transform(walking.get_transform())
        here: CarlaWorldPose = start
        mm = MapManager.get_instance()
        on = _on_walkable_lanelet(here)
        poses: list[Lanelet2Pose] = list(waypoints)
        poses.append(goal)
        if on is not None:
            walk_from: Lanelet2Pose = on
            poses = route_poses(
                PlannedRoute(start=walk_from, goal=goal, via=tuple(waypoints)),
                mm.lanelet_map,
                mm.pedestrian_routing_graph,
                WALK_POINT_STEP_M,
                bidirectional=True,
                straight_legs=True,
            )
        xs, ys, zs = [here.x], [here.y], [here.z]
        yaws: list[Optional[float]] = [None]
        for pose in poses:
            point = to_carla_world(pose)
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
            yaws.append(None)
        self._walk = ResolvedTrajectory(xs, ys, zs, yaws)
        self._walk_distance = 0.0
        self._walk_speed = walking_speed
        self._walk_cross = cross_factor

    def walk_along(
        self,
        path: "ResolvedTrajectory",
        walking_speed: float,
        cross_factor: Optional[float] = None,
    ) -> None:
        """Walk *path* as it is, from its start, which :meth:`walk_on` then
        walks a tick at a time: the walk :meth:`route_to` would lay out, given
        whole (a recorded walk).  *cross_factor* as :meth:`route_to`'s."""
        self._walk = path
        self._walk_distance = 0.0
        self._walk_speed = walking_speed
        self._walk_cross = cross_factor

    @property
    def walking(self) -> bool:
        """Whether a walk :meth:`route_to` laid out is still being walked."""
        return self._walk is not None

    @property
    def walk_path(self) -> "Optional[ResolvedTrajectory]":
        """The walk :meth:`route_to` laid out, while it is being walked."""
        return self._walk

    def walk_on(self, speed_mps: Optional[float] = None) -> None:
        """Walk one tick along the walk :meth:`route_to` laid out, standing
        still once it has arrived: at *speed_mps*, or the walking speed
        :meth:`route_to` was given (``None``)."""
        walk = self._walk
        actor = self._walker
        if walk is None or actor is None:
            return
        # Typed locals: Codon unwraps the Optionals here.
        path: "ResolvedTrajectory" = walk
        walker: "carla.Actor" = actor
        location = walker.get_transform().location
        here = path.project(location.x, location.y, near=self._walk_distance)
        self._walk_distance = here
        if here >= path.length - WALK_ARRIVAL_M:
            self._walk = None
            if TYPE_CHECKING:
                pass  # the walker controller is not compiled (docs/typecheck.md)
            else:
                _release_walker_ai(walker)
            walker.apply_control(carla.WalkerControl(speed=0.0))
            return
        speed = self._walk_speed
        if speed_mps is not None:
            speed = speed_mps
        if TYPE_CHECKING:
            steer_walker(walker, location, path, here, speed)
        elif not _walk_by_ai(walker, path, here, speed, location, self._walk_cross):
            # Stepped along the route (over a road stretch, or on a map with
            # no navigation mesh), giving way to a walker in its way.
            if _gives_way(walker, path, here, location):
                speed = 0.0
            steer_walker(walker, location, path, here, speed)

    def release_from_traffic(self, world: "carla.World") -> None:
        """Take this pedestrian back from the traffic backend that walks it,
        for an action that moves the walker itself
        (:meth:`~autoware_carla_scenario.traffic.base.TrafficBackend.release_pedestrian`)."""
        backend = self._traffic_backend
        if backend is not None:
            walked: WalkedEntity = self
            backend.release_pedestrian(walked, world)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lifted_transform(self, lift: float) -> "carla.Transform":
        """Return the spawn transform raised *lift* metres."""
        transform = self._config.spawn_location.value
        return carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + lift,
            ),
            transform.rotation,
        )


def _on_walkable_lanelet(pose: "CarlaWorldPose") -> "Optional[Lanelet2Pose]":
    """*pose* on the nearest lanelet a pedestrian walks, or ``None`` far from any."""
    from ..route.geometry import MapFeatures  # noqa: PLC0415
    import lanelet2.core  # noqa: PLC0415
    import lanelet2.geometry  # noqa: PLC0415

    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..coordinate.transform import (  # noqa: PLC0415
        _carla_to_lanelet2_frame,
        project_onto_lanelet,
    )

    mm = MapManager.get_instance()
    x, y, _heading = _carla_to_lanelet2_frame(pose)
    nearest = lanelet2.geometry.findNearest(
        mm.lanelet_map.laneletLayer, lanelet2.core.BasicPoint2d(x, y), 16
    )
    for _distance, lanelet in nearest:
        if MapFeatures.is_walkable(lanelet):
            on: Optional[Lanelet2Pose] = project_onto_lanelet(pose, int(lanelet.id))
            return on
    none: Optional[Lanelet2Pose] = None
    return none
