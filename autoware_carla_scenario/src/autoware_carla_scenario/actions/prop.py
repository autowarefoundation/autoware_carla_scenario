"""Prop action: add a CARLA prop to the world, or take one away (#447).

A prop (``static.prop.*``: a cone, a barrier, a sign, a box) is part of the
environment, not an entity: nothing drives it and nothing asks where it is,
so it is kept by *name* in this module rather than in the entity registry.
Unlike a vehicle, a prop may be spawned while a run is under way.

:class:`PropAction` adds a prop at a position written in any frame a
trajectory vertex takes (:data:`~..trajectory.model.TrajectoryPosition`:
CARLA's, Lanelet2's, OpenDRIVE's, Autoware's ``map``, relative to an entity's
lane, or along the ego's route), or removes the prop of a name.  The runner
takes every prop away (:func:`clear_props`) when a run ends and before the
next one starts, as it clears the entities.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Union

from ..conditions.base import BaseCondition, find_actor_by_role_name
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
)
from ..trajectory.resolve import resolve_position, road_height
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROP_BLUEPRINT",
    "PropAction",
    "PropOperation",
    "clear_props",
    "standing_props",
]

#: The prop :class:`PropAction` adds when no blueprint is given.
DEFAULT_PROP_BLUEPRINT = "static.prop.constructioncone"

#: Metres a prop on the road surface is raised, so CARLA does not refuse it
#: as colliding with the ground.
_LIFT_M = 0.05

#: The props standing now, by name.
_STANDING: dict[str, "carla.Actor"] = {}


def standing_props() -> list[str]:
    """The names of the props standing now."""
    return list(_STANDING.keys())


def _take_away(name: str) -> bool:
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
        _take_away(name)


class PropOperation(Enum):
    """What a :class:`PropAction` does."""

    ADD = "add"
    REMOVE = "remove"


class _ReferencePose:
    """Where an entity a relative position is written against is, in *world*."""

    _world: "carla.World"

    def __init__(self, world: "carla.World") -> None:
        self._world = world

    def pose(self, entity_ref: Optional[Union[EntityRole, str]]) -> CarlaWorldPose:
        if entity_ref is None:
            raise ValueError("a prop has no entity of its own to be relative to")
        actor = find_actor_by_role_name(self._world, str(entity_ref))
        if actor is None:
            raise ValueError(f"'{entity_ref}' is not in the world")
        found: "carla.Actor" = actor
        return CarlaWorldPose.from_carla_transform(found.get_transform())


class PropAction(BaseAction):
    """Add the prop *name* at *position*, or remove it.

    Args:
        name: The prop's name; a later action removes it by this name.
        operation: :attr:`PropOperation.ADD` or :attr:`PropOperation.REMOVE`.
        position: Where an added prop stands, in any frame a trajectory
            vertex takes; its heading is the position's (0 where it has
            none).  A :class:`~..trajectory.model.RelativeLanePose` names the
            entity it is relative to (``entity_ref``).
        blueprint: The CARLA prop an added one is.
        on_ground: Stand an added prop on the road surface under it, rather
            than at the position's own height.
        condition: Starts the action; at once by default.
        timing: Tick phase.
        label: Human-readable identifier.

    Raises:
        ValueError: :attr:`PropOperation.ADD` with no *position*.
    """

    # Declared for the static check (docs/typecheck.md).
    _name: str
    _operation: PropOperation
    # The position, held as a vertex: Codon types an attribute of a union
    # through it (docs/typecheck.md).
    _at: Optional[TrajectoryVertex]
    _blueprint: str
    _on_ground: bool

    def __init__(
        self,
        name: str,
        operation: PropOperation = PropOperation.ADD,
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
        *,
        blueprint: str = DEFAULT_PROP_BLUEPRINT,
        on_ground: bool = True,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        label: str = "prop",
    ) -> None:
        if operation is PropOperation.ADD and position is None:
            raise ValueError(f"{label}: adding prop '{name}' needs a position")
        super().__init__(label=label, condition=condition, timing=timing, once=True)
        self._name = name
        self._operation = operation
        at: Optional[TrajectoryVertex] = None
        if position is not None:
            at = TrajectoryVertex(position)
        self._at = at
        self._blueprint = blueprint
        self._on_ground = on_ground

    def execute(self, world: "carla.World") -> None:
        """Add or remove the prop."""
        self.apply(world)

    def apply(
        self,
        world: "carla.World",
        library: Optional["carla.BlueprintLibrary"] = None,
        ground: Optional[Callable[[float, float], Optional[float]]] = None,
    ) -> bool:
        """Add or remove the prop now; whether it was done.

        For an action built on this one, which applies it itself rather than
        registering it, and may place many props in one tick: *library* is
        the world's blueprint library and *ground* the road height under a
        point (:func:`~..trajectory.resolve.road_height`), fetched once for
        them all.
        """
        if self._operation is PropOperation.REMOVE:
            if not _take_away(self._name):
                logger.warning("%s: no prop '%s' to remove", self.label, self._name)
                return False
            logger.info("%s: prop '%s' removed", self.label, self._name)
            return True
        return self._add(world, library, ground)

    def _add(
        self,
        world: "carla.World",
        library: Optional["carla.BlueprintLibrary"],
        ground: Optional[Callable[[float, float], Optional[float]]],
    ) -> bool:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        assert self._at is not None
        vertex: TrajectoryVertex = self._at
        reference = _ReferencePose(world)
        try:
            x, y, z, yaw = resolve_position(vertex.position, reference.pose)
        except ValueError as exc:
            logger.warning(
                "%s: prop '%s' has no position: %s", self.label, self._name, exc
            )
            return False
        height = z if z is not None else 0.0
        if self._on_ground:
            # Branched: Codon types no conditional expression of the two.
            if ground is None:
                surface = road_height(run_carla_map(world))(x, y)
            else:
                given: Callable[[float, float], Optional[float]] = ground
                surface = given(x, y)
            if surface is not None:
                height = float(surface) + _LIFT_M
        blueprints = world.get_blueprint_library() if library is None else library
        found = blueprints.filter(self._blueprint)
        if len(found) == 0:  # A BlueprintLibrary has no truth value in Codon.
            logger.warning("%s: CARLA has no prop %r", self.label, self._blueprint)
            return False
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=height),
            carla.Rotation(yaw=0.0 if yaw is None else yaw),
        )
        # A name already standing is replaced; taken away first, as it may
        # stand on the very spot.
        _take_away(self._name)
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
