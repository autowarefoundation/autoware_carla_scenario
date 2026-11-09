"""Put something into the scene mid-run -- the counterpart of :mod:`.despawn`.

**A declared road user.**  One transcribed from a recording enters the scene
somewhere -- turns in, comes into the sensors' range -- while a run is under
way, but CARLA's entities are all spawned when the run starts.  So such an
entity is spawned out of the world (``spawn.hidden``: parked under the map,
its physics off, its traffic backend kept off it) and :class:`SpawnAction`
brings that actor in: set on its place, physics on, facing its heading, at a
speed.  (Bringing in the parked actor rather than spawning one now keeps
everything that knows the entity by its actor -- its traffic backend, the
collision log, the run's recording -- as it was from the start.)  What then
drives it -- typically its :class:`~.routing.RoutingAction`, triggered on this
action completing (an ``action_state`` condition on it) -- takes it from
there.

**A prop** (``static.prop.*``: a cone, a barrier, a sign, a box) is part of
the environment, not an entity: nothing drives it and nothing asks where it
is, so it is kept by *name* in this module rather than in the entity registry,
spawned when the action fires, and removed by name
(:class:`~.despawn.DespawnAction` with *name*, or :func:`remove_prop`).  The
runner takes every prop away (:func:`clear_props`) when a run ends and before
the next one starts, as it clears the entities.  (PropAction until 1.0.)

**Its place** is written in any frame a trajectory vertex takes
(:data:`~..trajectory.model.TrajectoryPosition`: CARLA's, Lanelet2's,
OpenDRIVE's, Autoware's ``map``, relative to an entity's lane, or along the
ego's route).  Whether to wait for the place to be clear is the trigger's to
say (a ``place_clear`` condition on the same place).  If a road user stands
there all the same when it fires -- the trigger gave up waiting -- the thing is
put on the nearest clear place up to *shift_max* metres on along its lane (a
Lanelet2 place), or where it was due with none, saying what stood there.  Two
things put on one place on the same tick do not both land there: each claims
its place for the frame (:func:`~autoware_carla_scenario.utils.clearance.claim_spot`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

from ..conditions.base import BaseCondition
from ..conditions.entity_existence import ReferencePoses
from ..constants import EGO_ROLE_NAME
from ..coordinate.map_manager import run_carla_map
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose, OpenDrivePose
from ..entity_role import EntityRole
from ..trajectory.model import (
    MapPose,
    RelativeLanePose,
    RouteCrossingPose,
    RouteCrosswalkPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    TrajectoryVertex,
    placed_against_entity,
)
from ..route.active import lane_place
from ..trajectory.resolve import resolve_along_lane, resolve_position, road_height
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROP_BLUEPRINT",
    "SHIFT_STEP_M",
    "SpawnAction",
    "clear_props",
    "remove_prop",
    "standing_props",
]

#: The prop a :class:`SpawnAction` adds when no blueprint is given.
DEFAULT_PROP_BLUEPRINT = "static.prop.constructioncone"
#: Metres apart the places on along its lane a spawn whose place is taken
#: tries.
SHIFT_STEP_M: float = 2.0
#: Metres a prop on the road surface is raised, so CARLA does not refuse it
#: as colliding with the ground.
_LIFT_M = 0.05
#: Half the size (m) a prop is taken to have where its place is looked at,
#: centred on its origin: not spawned yet, it has no bounding box (nor its
#: offset) to read -- a CARLA blueprint carries none.
_PROP_HALF_M = 0.5
#: The braking (m/s²) a place entered at speed has to leave room ahead for.
_ENTRY_BRAKING_MPS2 = 4.0
#: The room (m) a vehicle entering at speed keeps beyond that.
_ENTRY_GAP_M = 2.0
#: What was brought in, by actor id: the frame and the speed (m/s) it
#: entered at (_speed_ahead).
_ENTERED: dict[int, tuple[int, float]] = {}
#: How deep (m) under the map a hidden entity is parked (``HIDDEN_DEPTH_M``):
#: an entity of a scenario that declares no spawn for it
#: (:meth:`SpawnAction.declare_spawn`) is brought in this far above it.
_PARKED_DEPTH_M: float = 500.0
#: Where an entity was brought in, by ``(world id, role name)``: the frame and
#: the CARLA pose it was put at.  A client sees an actor's new transform only
#: from the next tick on, so a place written against it on the same frame
#: (another spawn of the same phase) is placed against this
#: (:class:`_BroughtInReferences`).  Put again on that frame (a ``POSITION``
#: trajectory), it is replaced (:func:`note_brought_in`); hidden or despawned
#: again, it is forgotten (:func:`forget_brought_in`).
_BROUGHT_IN: dict[tuple[int, str], tuple[int, CarlaWorldPose]] = {}

#: The props standing now, by name.
_STANDING: dict[str, "carla.Actor"] = {}


def standing_props() -> list[str]:
    """The names of the props standing now."""
    return list(_STANDING.keys())


def remove_prop(name: str) -> bool:
    """Destroy the prop *name*; whether there was one."""
    # Looked up first: Codon's dict.pop takes no default.
    if name not in _STANDING:
        return False
    actor = _STANDING.pop(name)
    try:
        actor.destroy()
    except RuntimeError:
        pass  # gone with the world already
    return True


def clear_props() -> None:
    """Take every prop away.  Called by the runner when a run ends and before
    the next starts: a batch runs several scenarios against one world."""
    for name in list(_STANDING.keys()):
        remove_prop(name)
    _BROUGHT_IN.clear()
    # The entry speeds too: written on every spawn, by an actor id the next
    # run's world may reuse, and nothing else forgets them (#1599).
    _ENTERED.clear()


def note_brought_in(world: "carla.World", role: str, pose: CarlaWorldPose) -> None:
    """Record that *role* was put at the CARLA *pose* on this frame: a place
    written against it on the same frame reads it there
    (:class:`_BroughtInReferences`), the world showing the new pose only from
    the next tick on.  Called by every path that teleports an entity into the
    scene (a SpawnAction, a ``POSITION``-mode trajectory); nothing is
    recorded where the frame cannot be read."""
    frame = _frame(world)
    if frame >= 0:
        _BROUGHT_IN[(_world_id(world), role)] = (frame, pose)


def forget_brought_in(world: "carla.World", role: str) -> None:
    """Forget where a SpawnAction brought *role* in on this frame (if it did):
    hidden again or despawned since, it is no longer there, so a place written
    against it reads it as the world shows it (absent).  Called by every path
    that hides or despawns an entity."""
    key = (_world_id(world), role)
    # Looked up first: Codon's dict.pop takes no default.
    if key in _BROUGHT_IN:
        _BROUGHT_IN.pop(key)


def _frame(world: "carla.World") -> int:
    """The world's frame now, or ``-1`` where it cannot be read."""
    try:
        return int(world.get_snapshot().frame)
    except (AttributeError, RuntimeError):
        return -1


def _world_id(world: "carla.World") -> int:
    """The world's id, or ``-1`` where it has none (a test double)."""
    try:
        return int(world.id)
    except AttributeError:
        return -1


class _BroughtInReferences:
    """The reference poses a spawn's place is resolved against: an entity
    brought in on this very frame where it was put (:data:`_BROUGHT_IN`; the
    world still shows it parked under the map), any other where it is now if
    it is in the scene (:class:`~..conditions.entity_existence.ReferencePoses`
    with ``present_only``)."""

    # Declared for the static check (docs/typecheck.md).
    _world_id: int
    _frame: int
    _present: ReferencePoses

    def __init__(self, world: "carla.World") -> None:
        self._world_id = _world_id(world)
        self._frame = _frame(world)
        self._present = ReferencePoses(world, present_only=True)

    def pose(self, entity_ref: Optional[Union[EntityRole, str]]) -> CarlaWorldPose:
        """Where *entity_ref* is now (:meth:`ReferencePoses.pose`)."""
        if entity_ref is not None and self._frame >= 0:
            key = (self._world_id, str(entity_ref))
            if key in _BROUGHT_IN:
                frame, put = _BROUGHT_IN[key]
                if frame == self._frame:
                    return put
        return self._present.pose(entity_ref)


class SpawnAction(BaseAction):
    """Bring a vehicle or pedestrian spawned out of the world into the scene
    (not the ego), or add the prop *name*.

    Args:
        entity_name: ``role_name`` of the vehicle or pedestrian; or give
            *name*, for a prop.
        condition: When it enters; ``None`` at once.
        timing: Tick phase.
        label: Human-readable identifier.
        once: Whether it fires at most once (the default).
        position: Its place, in any frame a trajectory vertex takes; its
            heading is the position's (a Lanelet2 pose's is along its lane
            turned by its ``heading``; one with none faces along the lane it
            is on -- a :class:`~..trajectory.model.RelativeLanePose` or a
            route pose -- and 0 where it is on none).  A place written
            against an entity (a ``RelativeLanePose``'s ``entity_ref``; the
            ego, for a route pose without an ``anchor``) is placed against
            where that entity stands when the action fires -- or, brought in
            by a SpawnAction on that very frame (which the world shows only
            from the next tick on), where it was put; while it is not in the
            scene (not spawned yet, gone, or spawned hidden and parked under
            the map, and not brought in on this frame) the place cannot be
            resolved (said): an entity is brought in where it was declared,
            a prop is not added.  An entity's
            ``None`` (or a Lanelet2 pose on lanelet 0 or less, what a card
            leaves empty): where it was declared to spawn
            (:meth:`declare_spawn`), through the same clearance check --
            or, in a scenario that declares none, right above where it
            waits under the map.  A prop needs one.
        shift_max: How far (m) on along its lane (a Lanelet2 place, or a
            route pose on a lane counted from an anchor) it may be
            put instead, at the nearest place clear of road users, when its
            place is taken as it fires; 0 (the default) puts it there all the
            same, and a prop is not looked at then.
        length: The length (m) of the place it needs, along its heading, where
            its actor's outline does not say it -- a bicycle ridden by a
            walker mesh; 0 (the default): its actor's bounding box.
        width: Its width (m), likewise.
        lanes: The lanelets, in order, a shift runs along -- the way its
            route goes on from its place; it goes no further than their last.
            Empty (the default): on along the lane, the straightest way where
            it branches.
        speed: The speed (m/s) an entity enters at, along its heading.
        name: A prop's name; a :class:`~.despawn.DespawnAction` removes it by
            this name.  One standing already of that name is replaced.
        blueprint: The CARLA prop it is.
        on_ground: Stand a prop on the road surface under it, rather than at
            the position's own height.

    Raises:
        ValueError: For the ego; for neither or both of *entity_name* and
            *name* (a blank *name* is none); a prop with no *position*, or
            one on lanelet 0 or less; a negative or non-finite *shift_max*, *speed*, *length* or
            *width*.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _name: str
    # The position, held as a vertex: Codon types an attribute of a union
    # through it (docs/typecheck.md).
    _at: Optional[TrajectoryVertex]
    _shift_max: float
    _speed: float
    _given_speed: float
    _blueprint: str
    _on_ground: bool
    _ahead: float
    _length: float
    _width: float
    _lanes: list[int]
    _spawned: list[tuple[str, int]]
    _declared_given: bool
    _declared: Optional[TrajectoryVertex]
    _declared_lane: Optional[Lanelet2Pose]

    def __init__(
        self,
        entity_name: Optional[Union[EntityRole, str]] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "spawn",
        once: bool = True,
        position: Optional[
            Union[
                CarlaWorldPose,
                Lanelet2Pose,
                OpenDrivePose,
                MapPose,
                RelativeLanePose,
                RouteLanePose,
                RouteOppositePose,
                RouteCrossingPose,
                RouteCrosswalkPose,
                RouteRoadsidePose,
            ]
        ] = None,
        shift_max: float = 0.0,
        speed: float = 0.0,
        name: Optional[str] = None,
        blueprint: str = DEFAULT_PROP_BLUEPRINT,
        on_ground: bool = True,
        lanes: Optional[Sequence[int]] = (),
        length: float = 0.0,
        width: float = 0.0,
    ) -> None:
        # A blank name (what a card's empty text field may hold) is none.
        prop_name = ""
        if name is not None:
            given_name: str = name
            if given_name.strip() != "":
                prop_name = given_name
        if (entity_name is None) == (prop_name == ""):
            raise ValueError(
                f"{label}: spawn an entity or a prop (a name), one of the two"
            )
        if entity_name is not None and str(entity_name) == str(EGO_ROLE_NAME):
            raise ValueError(f"{label}: the ego is in the scene from the start")
        if entity_name is None and position is None:
            raise ValueError(f"{label}: adding prop '{prop_name}' needs a position")
        # NaN passes `< 0.0` and would reach the placement and the speed.
        if not math.isfinite(shift_max) or shift_max < 0.0:
            raise ValueError(
                f"{label}: shift_max must be finite and not negative, "
                f"not {shift_max!r}"
            )
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError(
                f"{label}: speed must be finite and not negative, not {speed!r}"
            )
        if not math.isfinite(length) or length < 0.0:
            raise ValueError(
                f"{label}: length must be finite and not negative, not {length!r}"
            )
        if not math.isfinite(width) or width < 0.0:
            raise ValueError(
                f"{label}: width must be finite and not negative, not {width!r}"
            )
        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._entity_name = "" if entity_name is None else str(entity_name)
        self._name = prop_name
        at: Optional[TrajectoryVertex] = None
        if position is not None:
            at = TrajectoryVertex(position)
            if entity_name is None:
                placed: TrajectoryVertex = at
                where = placed.position
                if isinstance(where, Lanelet2Pose):
                    # Named as the class: Codon does not narrow a union by
                    # isinstance() (docs/typecheck.md).
                    on_lane: Lanelet2Pose = where
                    if int(on_lane.lanelet_id) <= 0:
                        # An entity's "where it was declared to spawn": a
                        # prop has no such place, and would never be added.
                        raise ValueError(
                            f"{label}: adding prop '{prop_name}' needs a "
                            f"lanelet (got {int(on_lane.lanelet_id)}) or a "
                            "route pose"
                        )
        self._at = at
        self._shift_max = float(shift_max)
        self._speed = float(speed)
        # As given: a run may slow the entry behind a slower road user.
        self._given_speed = self._speed
        self._blueprint = blueprint
        self._on_ground = on_ground
        self._ahead = 0.0
        self._length = float(length)
        self._width = float(width)
        # A card leaves the list empty (None): no lanes.
        given_lanes: list[int] = []
        if lanes is not None:
            for lanelet in lanes:
                given_lanes.append(int(lanelet))
        self._lanes = given_lanes
        self._spawned = []
        self._declared_given = False
        self._declared = None
        self._declared_lane = None

    @property
    def entity_name(self) -> str:
        """``role_name`` of the entity brought in; ``""`` for a prop."""
        return self._entity_name

    def declare_spawn(
        self, pose: Optional[CarlaWorldPose], lane: Optional[Lanelet2Pose] = None
    ) -> None:
        """Say where the entity was declared to spawn: the CARLA *pose* it is
        brought in at with no place of its own (or one that cannot be
        resolved), through the clearance check its own place would go
        through, shifted along *lane* (its Lanelet2 spawn) within
        *shift_max* where that is taken.  ``None``: it has no such place (a
        spawn on lanelet 0 or less), and with no place of its own it is not
        brought in (said).  Not called: right above where it waits under
        the map, as a hand-written scenario hides it."""
        self._declared_given = True
        if pose is None:
            self._declared = None
        else:
            declared: CarlaWorldPose = pose
            self._declared = TrajectoryVertex(declared)
        self._declared_lane = lane

    def reset(self) -> None:
        """Enter at the speed given again (:meth:`BaseAction.reset`): an
        earlier run may have slowed it behind a road user of its own."""
        super().reset()
        self._speed = self._given_speed
        self._ahead = 0.0
        self._spawned = []

    def take_spawns(self) -> list[tuple[str, int]]:
        """The ``(role, actor id)`` brought in since the last call, for the log."""
        out = self._spawned
        self._spawned = []
        return out

    def execute(self, world: "carla.World") -> None:
        """Bring the entity in, or add the prop."""
        self.apply(world)

    def apply(
        self,
        world: "carla.World",
        library: Optional["carla.BlueprintLibrary"] = None,
        ground: Optional[Callable[[float, float, float], Optional[float]]] = None,
    ) -> bool:
        """Spawn now; whether it was done.

        For an action built on this one, which applies it itself rather than
        registering it, and may place many props in one tick: *library* is
        the world's blueprint library and *ground* the road height under a
        point, read near a hint height
        (:func:`~..trajectory.resolve.road_height`), fetched once for them
        all.
        """
        if self._name:
            return self._add_prop(world, library, ground)
        return self._bring_in(world)

    # -- a declared road user ------------------------------------------------

    def _bring_in(self, world: "carla.World") -> bool:
        from ._trajectory_run import _entity_actor  # noqa: PLC0415

        actor = _entity_actor(self._entity_name)
        if actor is None:
            logger.warning(
                "%s: '%s' is not in the world", self.label, self._entity_name
            )
            return False
        found: "carla.Actor" = actor
        extent = found.bounding_box.extent
        # Entering at speed, it needs the room to stop ahead of it clear too
        # (a car put 0.3 m behind a slower one ran into it).
        from ._trajectory_run import _is_walker  # noqa: PLC0415

        self._ahead = self._speed * self._speed / (2.0 * _ENTRY_BRAKING_MPS2)
        if self._speed > 0.0 and not _is_walker(found):
            self._ahead += _ENTRY_GAP_M
        half_length = float(extent.x)
        half_width = float(extent.y)
        if self._length > 0.0:
            half_length = self._length / 2.0
        if self._width > 0.0:
            half_width = self._width / 2.0
        place = self._place(world, int(found.id), half_length, half_width)
        if place is None and self._declared_given:
            # Where it was declared to spawn, not where it waits: one parked
            # under the ego's spawn (or deeper, under another) would come up
            # off its road (#2014).
            declared = self._declared
            if declared is None:
                # Branched: Codon types no conditional expression of two str.
                why = "it has no place of its own"
                if self._has_own_place():
                    why = "its own place could not be resolved"
                logger.warning(
                    "%s: '%s' is not brought in: %s, and no spawn was declared "
                    "for it here (lanelet 0 or less, or one this map does not "
                    "have)",
                    self.label,
                    self._entity_name,
                    why,
                )
                return False
            at: TrajectoryVertex = declared
            place = self._place_vertex(
                world,
                at,
                self._declared_lane,
                int(found.id),
                half_length,
                half_width,
            )
            if place is None:
                return False
        if place is None:
            standing = found.get_transform()
            place = (
                standing.location.x,
                standing.location.y,
                standing.location.z + _PARKED_DEPTH_M,
                standing.rotation.yaw,
            )
        x, y, z, yaw = place
        # Where it is put, as its next tick shows it: a walker stood on the
        # ground under the place and raised by half its height.
        put_z = _put(world, found, x, y, z, yaw, self._speed)
        _keep_entry_speed(found, self._speed)
        frame = _frame(world)
        if frame >= 0:
            _ENTERED[int(found.id)] = (frame, self._speed)
        # Read by a place written against it on this frame: the world shows
        # where it is put only from the next tick on.
        note_brought_in(world, self._entity_name, CarlaWorldPose(x, y, put_z, yaw=yaw))
        self._spawned.append((self._entity_name, int(found.id)))
        logger.info(
            "%s: '%s' spawned at (%.1f, %.1f) at %.1f m/s",
            self.label,
            self._entity_name,
            x,
            y,
            self._speed,
        )
        return True

    def _has_own_place(self) -> bool:
        """Whether it was given a place of its own: not none, nor a Lanelet2
        pose on lanelet 0 or less (an entity's declared spawn)."""
        vertex = self._at
        if vertex is None:
            return False
        at: TrajectoryVertex = vertex
        position = at.position
        if isinstance(position, Lanelet2Pose):
            # Named as the class: Codon does not narrow a union by isinstance()
            # (docs/typecheck.md).
            given: Lanelet2Pose = position
            return int(given.lanelet_id) > 0
        return True

    # -- a prop --------------------------------------------------------------

    def _add_prop(
        self,
        world: "carla.World",
        library: Optional["carla.BlueprintLibrary"],
        ground: Optional[Callable[[float, float, float], Optional[float]]],
    ) -> bool:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        place = self._place(world, -1, _PROP_HALF_M, _PROP_HALF_M)
        if place is None:
            return False
        x, y, z, yaw = place
        height = z
        if self._on_ground:
            # Branched: Codon types no conditional expression of the two.
            # Read near the place's own height, so that a prop on a road
            # under an overpass stays on that road (not the deck above).
            if ground is None:
                surface = road_height(run_carla_map(world))(x, y, z)
            else:
                given: Callable[[float, float, float], Optional[float]] = ground
                surface = given(x, y, z)
            if surface is not None:
                height = float(surface) + _LIFT_M
        blueprints = world.get_blueprint_library() if library is None else library
        found = blueprints.filter(self._blueprint)
        if len(found) == 0:  # A BlueprintLibrary has no truth value in Codon.
            logger.warning("%s: CARLA has no prop %r", self.label, self._blueprint)
            return False
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=height), carla.Rotation(yaw=yaw)
        )
        # A name already standing is replaced; taken away first, as it may
        # stand on the very spot.
        remove_prop(self._name)
        actor = world.try_spawn_actor(found[0], transform)
        if actor is None:
            logger.warning(
                "%s: CARLA refused prop '%s' at (%.1f, %.1f)",
                self.label,
                self._name,
                x,
                y,
            )
            return False
        spawned: "carla.Actor" = actor
        _STANDING[self._name] = spawned
        logger.info(
            "%s: prop '%s' (%s) added at (%.1f, %.1f)",
            self.label,
            self._name,
            self._blueprint,
            x,
            y,
        )
        return True

    # -- the place -----------------------------------------------------------

    def _place(
        self, world: "carla.World", actor_id: int, half_length: float, half_width: float
    ) -> Optional[tuple[float, float, float, float]]:
        """CARLA ``(x, y, z, yaw)`` to put the thing at: its place, or the
        nearest clear one on along its lane within *shift_max* if that is
        taken (said either way); ``None`` with no place of its own (an
        entity's declared spawn) or one that cannot be resolved (said) --
        among them one written against an entity that is not in the scene
        (not spawned yet, gone, or spawned hidden and parked under the map,
        and not brought in on this frame: :class:`_BroughtInReferences`).
        A place written against an entity
        (:func:`~..trajectory.model.placed_against_entity`) with no ``yaw``
        faces along the lane it is placed on."""
        vertex = self._at
        if vertex is None:
            return None
        at: TrajectoryVertex = vertex
        position = at.position
        if isinstance(position, Lanelet2Pose):
            # Named as the class: Codon does not narrow a union by isinstance()
            # (docs/typecheck.md).
            given: Lanelet2Pose = position
            if int(given.lanelet_id) <= 0:
                return None  # an entity's declared spawn
        return self._place_vertex(world, at, None, actor_id, half_length, half_width)

    def _place_vertex(
        self,
        world: "carla.World",
        at: TrajectoryVertex,
        lane_given: Optional[Lanelet2Pose],
        actor_id: int,
        half_length: float,
        half_width: float,
    ) -> Optional[tuple[float, float, float, float]]:
        """:meth:`_place` for the place *at*: shifted along *lane_given*
        where given, else along the lane *at* is on."""
        position = at.position
        # An entity parked under the map (spawned hidden, not brought in yet)
        # is not in the scene: nothing is placed against it, as against one
        # not in the world -- unless brought in on this very frame, which the
        # world does not show yet: then where it was put.
        reference = _BroughtInReferences(world)
        try:
            if placed_against_entity(position):
                # Without a heading of its own, along the lane it is placed
                # on: no lane is looked up for it afterwards (lane_place).
                x0, y0, z0, yaw0 = resolve_along_lane(position, reference.pose)
            else:
                x0, y0, z0, yaw0 = resolve_position(position, reference.pose)
        except ValueError as exc:
            logger.warning("%s: no place to spawn at: %s", self.label, exc)
            return None
        # Branched: Codon types no conditional expression of a None and a float.
        due_z = 0.0
        if z0 is not None:
            due_z = float(z0)
        due_yaw = 0.0
        if yaw0 is not None:
            due_yaw = float(yaw0)
        else:
            # No heading of its own (a route pose without a yaw): along the
            # lane it is on, where it is on one.
            along = lane_place(position)
            if along is not None:
                _lx, _ly, _lz, lane_yaw = resolve_position(along, reference.pose)
                if lane_yaw is not None:
                    due_yaw = float(lane_yaw)
        due = (x0, y0, due_z, due_yaw)
        if self._shift_max <= 0.0 and not self._entity_name:
            return due  # a prop not asked to look for a clear place
        x, y, z, yaw = due
        blocker = _taken(
            world, actor_id, x, y, yaw, half_length, half_width, self._ahead, z=z
        )
        if blocker >= 0 and self._ahead > 0.0:
            if _taken(world, actor_id, x, y, yaw, half_length, half_width, z=z) < 0:
                # Only the room to stop in is taken, by one ahead: shifted on
                # it would be put past that one (vehicle_135 past vehicle_117,
                # bicycle_89 past bicycle_87).  It keeps its place, entering
                # no faster than it can stop short of the one ahead as that
                # goes on.
                self._enter_behind(
                    world, actor_id, blocker, due, half_length, half_width
                )
                blocker = -1
        if blocker < 0:
            self._claim(world, actor_id, due, half_length, half_width)
            return due
        on_lane = lane_place(position)
        if lane_given is not None:
            on_lane = lane_given
        if on_lane is not None:
            lane: Lanelet2Pose = on_lane
            shift = SHIFT_STEP_M
            while shift <= self._shift_max + 1e-9:
                on = _along_lane(lane, shift, self._lanes)
                if on is None:
                    break
                further: Lanelet2Pose = on
                x1, y1, z1, yaw1 = resolve_position(further, reference.pose)
                there_z = 0.0
                if z1 is not None:
                    there_z = float(z1)
                there_yaw = 0.0
                if yaw1 is not None:
                    there_yaw = float(yaw1)
                there = (x1, y1, there_z, there_yaw)
                other = _taken(
                    world,
                    actor_id,
                    there[0],
                    there[1],
                    there[3],
                    half_length,
                    half_width,
                    self._ahead,
                    z=there[2],
                )
                if other < 0:
                    logger.warning(
                        "%s: spawns %.0f m on along its lane: actor %d stands "
                        "where it was due",
                        self.label,
                        shift,
                        blocker,
                    )
                    self._claim(world, actor_id, there, half_length, half_width)
                    return there
                shift += SHIFT_STEP_M
        logger.warning(
            "%s: spawns where it was due, on actor %d standing there (no clear "
            "place within %.0f m on along its lane)",
            self.label,
            blocker,
            self._shift_max,
        )
        self._claim(world, actor_id, due, half_length, half_width)
        return due

    def _enter_behind(
        self,
        world: "carla.World",
        actor_id: int,
        blocker: int,
        due: tuple[float, float, float, float],
        half_length: float,
        half_width: float,
    ) -> None:
        """Slow the entry speed until the room it needs behind *blocker* --
        to brake to *blocker*'s speed and the gap kept -- is clear (said),
        down to standing."""
        x, y, z, yaw = due
        braking = self._speed * self._speed / (2.0 * _ENTRY_BRAKING_MPS2)
        gap = max(self._ahead - braking, 0.0)
        ahead_speed = _speed_ahead(world, blocker, yaw)
        slower = 0.0
        for factor in (1.0, 0.75, 0.5, 0.25):
            trying = self._speed * factor
            room = max(trying * trying - ahead_speed * ahead_speed, 0.0)
            room = room / (2.0 * _ENTRY_BRAKING_MPS2) + gap
            taken = _taken(
                world, actor_id, x, y, yaw, half_length, half_width, room, z=z
            )
            if taken < 0:
                slower = trying
                break
        if slower < self._speed:
            logger.info(
                "%s: enters at %.1f m/s, not %.1f: actor %d is %s ahead",
                self.label,
                slower,
                self._speed,
                blocker,
                "close" if ahead_speed <= 0.0 else f"{ahead_speed:.1f} m/s",
            )
            self._speed = slower

    def _claim(
        self,
        world: "carla.World",
        actor_id: int,
        place: tuple[float, float, float, float],
        half_length: float,
        half_width: float,
    ) -> None:
        _hold(
            world,
            actor_id,
            place[0],
            place[1],
            place[3],
            half_length,
            half_width,
            z=place[2],
        )


def _along_lane(
    pose: Lanelet2Pose, distance: float, lanes: Optional[list[int]] = None
) -> Optional[Lanelet2Pose]:
    """*pose* moved *distance* m on along *lanes* (in order, from the one it
    is on) -- or, with none, along its lane -- or ``None`` where they end
    first."""
    from ..coordinate import lanelet_length  # noqa: PLC0415
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..trajectory.relative_lane import advance_along_lane  # noqa: PLC0415

    if lanes:
        given: list[int] = lanes
        if int(pose.lanelet_id) not in given:
            return None
        index = given.index(int(pose.lanelet_id))
        s = float(pose.s) + distance
        while index < len(given):
            length = lanelet_length(given[index])
            if s <= length:
                return Lanelet2Pose(
                    lanelet_id=given[index], s=s, t=pose.t, heading=pose.heading
                )
            s -= length
            index += 1
        return None
    mm = MapManager.get_instance()
    try:
        lanelet_id, along = advance_along_lane(
            mm.lanelet_map,
            mm.routing_graph,
            int(pose.lanelet_id),
            float(pose.s),
            distance,
        )
    except ValueError:
        return None
    return Lanelet2Pose(lanelet_id=lanelet_id, s=along, t=pose.t, heading=pose.heading)


def _keep_entry_speed(actor: "carla.Actor", speed: float) -> None:
    """A vehicle brought in before the clock starts (an ``init`` card) is
    held still until it does: have it start the run at *speed* then."""
    from ..entity.vehicle_entity import hold_entry_speed, warmup_done  # noqa: PLC0415
    from ._trajectory_run import _is_walker  # noqa: PLC0415

    if speed > 0.0 and not warmup_done() and not _is_walker(actor):
        hold_entry_speed(int(actor.id), speed)


def _put(
    world: "carla.World",
    actor: "carla.Actor",
    x: float,
    y: float,
    z: float,
    yaw: float,
    speed: float,
) -> float:
    """Put *actor* at CARLA ``(x, y, z)`` facing *yaw*, physics on, moving at
    *speed*: a walker stood on the ground under it (the lane's height is the
    road's, and a walker set into a raised kerb's mesh never moves).  The z
    its origin was put at."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from ..coordinate.snap import refine_z_with_ground_projection  # noqa: PLC0415
    from ._trajectory_run import _as_walker, _is_walker, _walker_speed  # noqa: PLC0415

    walker = _is_walker(actor)
    actor.set_simulate_physics(True)
    lift = 0.0
    if walker:
        try:
            z = refine_z_with_ground_projection(x, y, z, world)
        except RuntimeError:
            pass
        lift = float(actor.bounding_box.extent.z)
    put_z = z + lift
    actor.set_transform(
        carla.Transform(carla.Location(x=x, y=y, z=put_z), carla.Rotation(yaw=yaw))
    )
    heading = math.radians(yaw)
    direction = carla.Vector3D(math.cos(heading), math.sin(heading), 0.0)
    if walker:
        _as_walker(actor).apply_control(
            carla.WalkerControl(direction=direction, speed=_walker_speed(speed))
        )
    else:
        actor.set_target_velocity(
            carla.Vector3D(speed * direction.x, speed * direction.y, 0.0)
        )
    return put_z


def _taken(
    world: "carla.World",
    actor_id: int,
    x: float,
    y: float,
    yaw: float,
    half_length: float,
    half_width: float,
    ahead: float = 0.0,
    *,
    z: float,
) -> int:
    """The id of a road user whose outline overlaps the rectangle of
    *actor_id* put with its origin at CARLA ``(x, y)`` facing *yaw* -- centred
    on its bounding box, a car's ahead of its origin -- stretched *ahead* m
    further forward (the room to stop in), with a 0.3 m margin; or ``-1``.
    Only one on the level of road at height *z* there (#1588)."""
    from ..utils.clearance import box_centre, overlaps  # noqa: PLC0415

    heading = math.radians(yaw)
    try:
        bx, by = box_centre(world, actor_id, x, y, yaw)
    except (AttributeError, RuntimeError):
        bx, by = x, y
    cx = bx + math.cos(heading) * ahead / 2.0
    cy = by + math.sin(heading) * ahead / 2.0
    try:
        found = overlaps(
            world, actor_id, cx, cy, yaw, half_length + ahead / 2.0, half_width, z=z
        )
    except (AttributeError, RuntimeError):
        return -1
    return -1 if found is None else int(found)


def _speed_ahead(world: "carla.World", blocker: int, yaw: float) -> float:
    """*blocker*'s speed (m/s) the way *yaw* faces: brought in on this tick,
    the one it enters at (it has none yet, nor its new heading, and in the
    room ahead it is taken to go this way: vehicle_117, bicycle_87)."""
    from ..entity.vehicle_entity import entry_speeds, warmup_done  # noqa: PLC0415

    if not warmup_done():
        held = entry_speeds().get(int(blocker))
        if held is not None:
            return held
    try:
        frame = int(world.get_snapshot().frame)
    except (AttributeError, RuntimeError):
        frame = -1
    entered = _ENTERED.get(int(blocker))
    if entered is not None and entered[0] == frame:
        return entered[1]
    try:
        other = world.get_actor(blocker)
        if other is None:
            return 0.0
        velocity = other.get_velocity()
    except (AttributeError, RuntimeError):
        return 0.0
    heading = math.radians(yaw)
    return max(
        float(velocity.x) * math.cos(heading) + float(velocity.y) * math.sin(heading),
        0.0,
    )


def _hold(
    world: "carla.World",
    actor_id: int,
    x: float,
    y: float,
    yaw: float,
    half_length: float,
    half_width: float,
    *,
    z: float,
) -> None:
    """Hold the place -- *actor_id*'s box, its origin put at ``(x, y)`` on
    the road at height *z* -- for this frame (two put on one place on one
    tick do not both land there)."""
    from ..utils.clearance import box_centre, claim_spot  # noqa: PLC0415

    try:
        cx, cy = box_centre(world, actor_id, x, y, yaw)
        claim_spot(world, actor_id, cx, cy, yaw, half_length, half_width, z=z)
    except (AttributeError, RuntimeError):
        pass
