"""A pedestrian taken over by a motion: found, posed, placed, handed back.

:class:`WalkerPoser` is what the pedestrian-motion actions share: it finds
the pedestrian by role, takes its skeleton over through CARLA's custom pose
(``Walker.get_pose_from_animation()``, ``set_bones()``, ``blend_pose()``),
poses it one frame of bone directions at a time
(:class:`~autoware_carla_scenario.pedestrian.skeleton.WalkerRetarget`),
places it when the motion carries the walker too, and hands it back to its
animation.  Numpy-free, so Codon checks it.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import typesafe_carla.carla as carla

from ..entity.pedestrian_entity import walker_control_speed
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from .skeleton import Vector, WalkerRetarget

logger = logging.getLogger(__name__)

#: A change of the body's lift smaller than this does not move the walker.
_LIFT_TOLERANCE_M = 0.005
#: Slower than this (m/s), a placed walker stands still.
_STILL_MS = 0.05

__all__ = ["WalkerPoser"]


class WalkerPoser:
    """One pedestrian's skeleton, as a motion poses it.

    Args:
        entity_name: ``role_name`` of the pedestrian.
        owner: Who poses it, for the log (an action's class name).
    """

    _entity_name: str
    _owner: str
    _retarget: Optional[WalkerRetarget]
    _walker: Optional[carla.Actor]
    _blended: bool
    _lifted: float
    _placed: bool
    _unsimulated: bool

    def __init__(self, entity_name: str, owner: str) -> None:
        self._entity_name = entity_name
        self._owner = owner
        self._retarget = None
        self._walker = None
        self._blended = False
        # How far the walker stands moved up (down: negative) to keep its
        # posed body on the ground (WalkerRetarget.lift), and whether the
        # caller placed it this tick (then place() applies it).
        self._lifted = 0.0
        self._placed = False
        # Whether its physics is off: a lowered walker's capsule is in the
        # ground, which the physics would push it out of.
        self._unsimulated = False

    @property
    def actor(self) -> Optional[carla.Actor]:
        """The walker being posed, if one is."""
        return self._walker

    def find_actor(self) -> Optional[carla.Actor]:
        """The pedestrian's actor, or ``None`` (warned) if there is none."""
        entity = find_pedestrian_entity(self._entity_name)
        if entity is None:
            if find_vehicle_entity(self._entity_name) is None:
                logger.warning(
                    "%s: entity '%s' not found", self._owner, self._entity_name
                )
            else:
                logger.warning(
                    "%s: '%s' is not a pedestrian and has no skeleton to pose",
                    self._owner,
                    self._entity_name,
                )
            return None
        actor = entity.actor
        if actor is None:
            logger.warning(
                "%s: pedestrian '%s' is not spawned", self._owner, self._entity_name
            )
        return actor

    def begin(self, upper_body_only: bool) -> bool:
        """Take the walker's skeleton over, posed as its animation holds it.

        The retarget is built from the bones as they are now, so the bones a
        frame does not drive keep this frame of the animation.

        Returns:
            Whether the walker is now posed (``False``, warned: no walker, or
            none of the bones a motion drives).
        """
        self.end()
        actor = self.find_actor()
        if actor is None:
            return False
        walker = _as_walker(actor)
        retarget = WalkerRetarget(
            walker.get_bones().bone_transforms,
            walker.get_transform().rotation,
            upper_body_only=upper_body_only,
        )
        if retarget.missing:
            logger.warning(
                "%s: '%s' has no bones %s; they keep the animation",
                self._owner,
                self._entity_name,
                list(retarget.missing),
            )
        if not retarget.driven:
            logger.warning(
                "%s: '%s' has none of the bones a motion drives",
                self._owner,
                self._entity_name,
            )
            return False
        walker.get_pose_from_animation()
        self._retarget = retarget
        self._walker = actor
        return True

    def show(self, frame: dict[str, Vector]) -> bool:
        """Pose the walker after *frame*: each followed SOMA joint's bone
        direction in the body frame (forward, right, up) of the way the
        walker faces.  The first frame shown also blends the pose in.

        The walker is moved up or down so the posed body's lowest point stays
        where its feet were (:attr:`WalkerRetarget.lift`): lying down, it
        lies on the ground.  A walker :meth:`place` puts takes it with the
        next place; any other walker is moved here.

        Returns:
            Whether it is still posed (``False``: the walker was destroyed
            under the motion).
        """
        retarget = self._retarget
        actor = self._walker
        if retarget is None or actor is None:
            return False
        walker = _as_walker(actor)
        try:
            walker.set_bones(carla.WalkerBoneControlIn(retarget.pose(frame)))
            if not self._blended:
                walker.blend_pose(1.0)
                self._blended = True
            if self._placed:
                self._placed = False
            elif abs(retarget.lift - self._lifted) > _LIFT_TOLERANCE_M:
                self._simulate(actor, retarget.lift)
                self._raise(actor, retarget.lift - self._lifted)
                self._lifted = retarget.lift
        except RuntimeError:
            self._forget()
            return False
        return True

    def place(
        self, location: carla.Location, yaw_deg: float, velocity: carla.Vector3D
    ) -> bool:
        """Stand the walker at *location* facing *yaw_deg*, moving on the way
        it faces at *velocity*'s (m/s) speed that way -- so what reads its
        speed sees the motion's, and the tick the world takes carries it on
        towards the next frame, without turning it.

        Returns:
            Whether it is still posed (``False``: the walker is gone).
        """
        actor = self._walker
        retarget = self._retarget
        if actor is None or retarget is None:
            return False
        try:
            # On the ground as the last frame shown posed the body.
            self._simulate(actor, retarget.lift)
            lifted = carla.Location(
                x=location.x, y=location.y, z=location.z + retarget.lift
            )
            actor.set_transform(
                carla.Transform(
                    lifted, carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)
                )
            )
            self._lifted = retarget.lift
            self._placed = True
            # Told the way it faces, at the motion's speed that way: CARLA
            # turns a walker towards the way it is told to go, and a motion's
            # root drifts and steps sideways (sitting down, a replanned
            # frame), so any other way turns it from where the motion faces.
            yaw = math.radians(yaw_deg)
            direction = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
            speed = max(0.0, velocity.x * direction.x + velocity.y * direction.y)
            if speed < _STILL_MS:
                speed = 0.0
            _as_walker(actor).apply_control(
                carla.WalkerControl(
                    direction=direction, speed=walker_control_speed(speed)
                )
            )
        except RuntimeError:
            self._forget()
            return False
        return True

    def end(self) -> bool:
        """Hand the walker back to its animation.

        Returns:
            Whether a walker was posed and has been handed back.
        """
        actor = self._walker
        lifted = self._lifted
        unsimulated = self._unsimulated
        self._forget()
        if actor is None:
            return False
        try:
            if abs(lifted) > _LIFT_TOLERANCE_M:
                # Standing again, at the height it stood.
                self._raise(actor, -lifted)
            if unsimulated:
                actor.set_simulate_physics(True)
            _as_walker(actor).blend_pose(0.0)
        except RuntimeError:
            return False
        return True

    def _simulate(self, actor: carla.Actor, lift: float) -> None:
        """Physics off while the walker is lowered (*lift* below 0), on
        again once it stands."""
        lowered = lift < -_LIFT_TOLERANCE_M
        if lowered != self._unsimulated:
            actor.set_simulate_physics(not lowered)
            self._unsimulated = lowered

    def _raise(self, actor: carla.Actor, by_m: float) -> None:
        here = actor.get_location()
        actor.set_location(carla.Location(x=here.x, y=here.y, z=here.z + by_m))

    def _forget(self) -> None:
        self._walker = None
        self._retarget = None
        self._blended = False
        self._lifted = 0.0
        self._placed = False
        self._unsimulated = False


def _as_walker(actor: carla.Actor) -> carla.Walker:
    """*actor* as the walker whose bones are set (a spawned walker is one)."""
    if TYPE_CHECKING:
        return actor.as_walker()
    else:
        return actor
