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

    def __init__(self, entity_name: str, owner: str) -> None:
        self._entity_name = entity_name
        self._owner = owner
        self._retarget = None
        self._walker = None
        self._blended = False

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

        The body is lowered so its lowest point stays where its feet were
        (:attr:`WalkerRetarget.lift`): lying down, it lies on the ground.
        Only the mesh is: the walker, its capsule and its physics stay where
        CARLA has them, so it collides and moves as before.

        Returns:
            Whether it is still posed (``False``: the walker was destroyed
            under the motion).
        """
        actor = self._walker
        if actor is None:
            return False
        try:
            self._pose(actor, frame)
        except RuntimeError:
            self._forget()
            return False
        return True

    def place(
        self,
        frame: dict[str, Vector],
        location: carla.Location,
        yaw_deg: float,
        velocity: carla.Vector3D,
    ) -> bool:
        """:meth:`show` *frame* with the walker stood at *location* facing
        *yaw_deg*, moving on the way it faces
        at *velocity*'s (m/s) speed that way -- so what reads its speed sees
        the motion's, and the tick the world takes carries it on towards the
        next frame, without turning it.

        Returns:
            Whether it is still posed (``False``: the walker is gone).
        """
        actor = self._walker
        if actor is None:
            return False
        try:
            self._pose(actor, frame)
            actor.set_transform(
                carla.Transform(
                    location, carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)
                )
            )
            # Told the way it faces, at the motion's speed that way: CARLA
            # turns a walker towards the way it is told to go, and a motion's
            # root drifts and steps sideways (sitting down, a replanned
            # frame), so any other way turns it from where the motion faces.
            yaw = math.radians(yaw_deg)
            direction = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)
            ahead = velocity.x * direction.x + velocity.y * direction.y
            speed = ahead if ahead >= _STILL_MS else 0.0
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
        self._forget()
        if actor is None:
            return False
        try:
            _as_walker(actor).blend_pose(0.0)
        except RuntimeError:
            return False
        return True

    def _pose(self, actor: carla.Actor, frame: dict[str, Vector]) -> None:
        """Set the walker's bones for *frame*, blending the pose in the
        first time."""
        retarget = self._retarget
        if retarget is None:  # pragma: no cover - posed with a walker
            return
        walker = _as_walker(actor)
        walker.set_bones(carla.WalkerBoneControlIn(retarget.pose(frame)))
        if not self._blended:
            walker.blend_pose(1.0)
            self._blended = True

    def _forget(self) -> None:
        self._walker = None
        self._retarget = None
        self._blended = False


def _as_walker(actor: carla.Actor) -> carla.Walker:
    """*actor* as the walker whose bones are set (a spawned walker is one)."""
    if TYPE_CHECKING:
        return actor.as_walker()
    else:
        return actor
