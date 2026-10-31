"""Play-motion action: pose a pedestrian after a SOMA motion file."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import typesafe_carla.carla as carla

from ..action_state import ActionState
from ..conditions.base import BaseCondition, ScenarioResult
from ..entity._soma_motion import SomaMotion, WalkerRetarget, load_soma_motion
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

logger = logging.getLogger(__name__)

__all__ = ["PlayMotionAction"]


class PlayMotionAction(BaseAction):
    """Play a SOMA motion -- a wave, a beckon -- on a pedestrian's skeleton.

    *motion_path* names an ``.npz`` written by SOMA-X's
    ``soma.io.save_soma_npz`` (`NVlabs/SOMA-X
    <https://github.com/NVlabs/SOMA-X>`_).  Every identity model SOMA-X has
    (MHR, SOMA, Anny, SMPL) moves the same SOMA skeleton, so a motion is played
    the same whichever body it was made for.  The walker keeps its blueprint's
    mesh; the motion is retargeted onto its ``crl_*`` bones
    (:mod:`~autoware_carla_scenario.entity._soma_motion`) and shown through
    CARLA's custom pose (``Walker.set_bones()`` / ``blend_pose()``), one frame
    every tick, at the motion's own rate in simulated time.

    The custom pose is the whole skeleton: the bones the motion does not drive
    hold the frame of the walker's animation the motion began on.  So a walker
    playing one stands -- with *upper_body_only* its legs are frozen in that
    frame, and without, they follow the motion -- and a pedestrian should be
    stopped (``WalkStraightAction`` at speed 0) before it waves.  The hips are
    never driven: where the walker stands and faces stays CARLA's, and the
    motion is turned to face the way the walker does.

    The action runs until the motion has played (each loop of it, with
    *loop*, until *duration_s*), then hands the walker back to its animation.

    Args:
        entity_name: ``role_name`` of the pedestrian.
        motion_path: The SOMA ``.npz``.  A relative path is taken from the
            working directory.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        fps: The motion's frames per second.  ``None`` (default): the file's
            ``fps`` if it stores one, else 30.
        loop: Play the motion over again when it ends.
        duration_s: Stop after this many simulated seconds.  ``None``
            (default): when the motion ends; with *loop*, never -- it plays
            until the scenario ends.
        upper_body_only: Drive the spine, neck and arms only (default); the
            legs keep the animation's pose.

    Raises:
        FileNotFoundError: If there is no file at *motion_path*.
        ValueError: If the file is not a SOMA body motion, or *fps* or
            *duration_s* is not positive.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _motion_path: str
    _motion: SomaMotion
    _loop: bool
    _duration_s: Optional[float]
    _upper_body_only: bool
    _retarget: Optional[WalkerRetarget]
    _walker: Optional[carla.Actor]
    _started_at: float
    _playing: bool

    REISSUES_BY_DEFAULT = True

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        motion_path: Union[str, Path],
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "play_motion",
        once: bool = True,
        fps: Optional[float] = None,
        loop: bool = False,
        duration_s: Optional[float] = None,
        upper_body_only: bool = True,
    ) -> None:
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive")
        finished: BaseCondition = _MotionPlayed(self, label)
        super().__init__(
            label=label, condition=condition, timing=timing, once=once, until=finished
        )
        self._entity_name = str(entity_name)
        self._motion_path = str(motion_path)
        self._motion = load_soma_motion(motion_path, fps)
        self._loop = loop
        self._duration_s = duration_s
        self._upper_body_only = upper_body_only
        self._retarget = None
        self._walker = None
        self._started_at = 0.0
        self._playing = False

    @property
    def playing(self) -> bool:
        """Whether the motion is showing on the walker."""
        return self._playing

    def execute(self, world: carla.World) -> None:
        """Begin the motion on the trigger's call, then show its next frame on
        every tick of the run."""
        if self.state is ActionState.STANDBY:
            self._start(world)
            return
        self._step(world)

    def _start(self, world: carla.World) -> None:
        self._playing = False
        actor = self._find_actor()
        if actor is None:
            return
        walker = _as_walker(actor)
        retarget = WalkerRetarget(
            walker.get_bones().bone_transforms,
            walker.get_transform().rotation,
            upper_body_only=self._upper_body_only,
        )
        if retarget.missing:
            logger.warning(
                "PlayMotionAction: '%s' has no bones %s; they keep the animation",
                self._entity_name,
                list(retarget.missing),
            )
        if not retarget.driven:
            logger.warning(
                "PlayMotionAction: '%s' has none of the bones a motion drives; "
                "not playing %s",
                self._entity_name,
                self._motion_path,
            )
            return
        # The custom pose starts as the animation's frame, so the bones the
        # motion does not drive stay where the walker held them.
        walker.get_pose_from_animation()
        self._retarget = retarget
        self._walker = actor
        self._started_at = _now(world)
        self._playing = True
        logger.info(
            "PlayMotionAction: '%s' plays %s (%d frames at %.1f fps)",
            self._entity_name,
            self._motion_path,
            self._motion.frame_count,
            self._motion.fps,
        )
        self._show(0)
        walker.blend_pose(1.0)

    def _step(self, world: carla.World) -> None:
        if not self._playing:
            return
        elapsed = _now(world) - self._started_at
        if self._duration_s is not None and elapsed >= self._duration_s:
            self._stop()
            return
        frame = int(math.floor(elapsed * self._motion.fps + 1e-9))
        count = self._motion.frame_count
        if frame >= count:
            if not self._loop:
                self._stop()
                return
            frame %= count
        self._show(frame)

    def _show(self, frame: int) -> None:
        retarget = self._retarget
        actor = self._walker
        if retarget is None or actor is None:
            return
        bones = retarget.pose(self._motion.frame(frame))
        try:
            _as_walker(actor).set_bones(carla.WalkerBoneControlIn(bones))
        except RuntimeError:
            # The walker was destroyed under the motion.
            self._playing = False

    def _stop(self) -> None:
        """Hand the walker back to its animation."""
        self._playing = False
        actor = self._walker
        self._walker = None
        self._retarget = None
        if actor is None:
            return
        try:
            _as_walker(actor).blend_pose(0.0)
        except RuntimeError:
            return
        logger.info("PlayMotionAction: '%s' finished its motion", self._entity_name)

    def _find_actor(self) -> Optional[carla.Actor]:
        """The pedestrian's actor, or ``None`` (warned) if there is none."""
        entity = find_pedestrian_entity(self._entity_name)
        if entity is None:
            if find_vehicle_entity(self._entity_name) is None:
                logger.warning(
                    "PlayMotionAction: entity '%s' not found", self._entity_name
                )
            else:
                logger.warning(
                    "PlayMotionAction: '%s' is not a pedestrian and has no "
                    "skeleton to pose",
                    self._entity_name,
                )
            return None
        actor = entity.actor
        if actor is None:
            logger.warning(
                "PlayMotionAction: pedestrian '%s' is not spawned", self._entity_name
            )
        return actor


def _as_walker(actor: carla.Actor) -> carla.Walker:
    """*actor* as the walker whose bones are set (a spawned walker is one)."""
    if TYPE_CHECKING:
        return actor.as_walker()
    else:
        return actor


def _now(world: carla.World) -> float:
    """The simulated time, in seconds."""
    return float(world.get_snapshot().timestamp.elapsed_seconds)


class _MotionPlayed(BaseCondition):
    """Satisfied once the motion it watches is no longer showing."""

    _action: PlayMotionAction

    def __init__(self, action: PlayMotionAction, label: str) -> None:
        super().__init__(label=f"{label}_played")
        self._action = action

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        if self._action.playing:
            return None
        return ScenarioResult(
            passed=True, message="the motion has played", elapsed_seconds=elapsed
        )
