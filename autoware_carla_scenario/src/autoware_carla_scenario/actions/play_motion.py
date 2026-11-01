"""Play-motion action: pose a pedestrian after a SOMA motion file."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Union

import typesafe_carla.carla as carla

from ..action_state import ActionState
from ..conditions.and_condition import AndCondition
from ..conditions.base import BaseCondition, ScenarioResult
from ..control_domain import ControlDomain, movement_of
from ..entity.registry import find_pedestrian_entity
from ..entity_role import EntityRole
from ..kinematics.walker_motion import WalkerVelocities
from ..pedestrian.poser import WalkerPoser
from ..pedestrian.soma import SomaMotion, load_soma_motion
from .base import BaseAction, TickTiming

logger = logging.getLogger(__name__)

__all__ = ["STANDING_SPEED_MS", "PlayMotionAction"]

#: The speed (m/s) at or under which a pedestrian stands, and may play a motion.
STANDING_SPEED_MS: float = 0.1


class PlayMotionAction(BaseAction):
    """Play a SOMA motion -- a wave, a beckon -- on a pedestrian's skeleton.

    *motion_path* names an ``.npz`` written by SOMA-X's
    ``soma.io.save_soma_npz`` (`NVlabs/SOMA-X
    <https://github.com/NVlabs/SOMA-X>`_).  Every identity model SOMA-X has
    (MHR, SOMA, Anny, SMPL) moves the same SOMA skeleton, so a motion is played
    the same whichever body it was made for.  The walker keeps its blueprint's
    mesh; the motion is retargeted onto its ``crl_*`` bones
    (:mod:`~autoware_carla_scenario.pedestrian`) and shown through
    CARLA's custom pose (``Walker.set_bones()`` / ``blend_pose()``), one frame
    every tick, at the motion's own rate in simulated time.

    The custom pose is the whole skeleton: the bones the motion does not drive
    hold the frame of the walker's animation the motion began on.  So a walker
    plays a motion standing, and the action fires only while the pedestrian
    stands (no faster than :data:`STANDING_SPEED_MS`), as well as its
    *condition* holds: stop it first (``WalkStraightAction`` at speed 0).  The
    hips are never driven: where the walker stands and faces stays CARLA's,
    and the motion is turned to face the way the walker does.

    **Control domains** (:mod:`~autoware_carla_scenario.control_domain`).  The
    motion takes the pedestrian's *animation* domain; driving the whole body
    (``upper_body_only=False``), its legs too, it takes its movement as well,
    so nothing may walk it while it plays.  With *upper_body_only* its
    movement stays free -- but a pedestrian walked under a motion glides,
    its legs frozen.

    The action runs until the motion has played (each loop of it, with
    *loop*, until *duration_s*), then hands the walker back to its animation,
    as it does when another action overrides it or the run ends mid-motion.

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
        overrides: Take over a control domain of the pedestrian another
            action holds (:attr:`~BaseAction.overrides`).

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
    _poser: WalkerPoser
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
        overrides: bool = False,
    ) -> None:
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive")
        finished: BaseCondition = _MotionPlayed(self, label)
        standing: BaseCondition = _Standing(str(entity_name), label)
        trigger = standing
        if condition is not None:
            given: BaseCondition = condition
            both: BaseCondition = AndCondition([given, standing], label=given.label)
            trigger = both
        super().__init__(
            label=label,
            condition=trigger,
            timing=timing,
            once=once,
            until=finished,
            overrides=overrides,
        )
        self._entity_name = str(entity_name)
        self._motion_path = str(motion_path)
        self._motion = load_soma_motion(motion_path, fps)
        self._loop = loop
        self._duration_s = duration_s
        self._upper_body_only = upper_body_only
        self._poser = WalkerPoser(self._entity_name, "PlayMotionAction")
        self._started_at = 0.0
        self._playing = False

    @property
    def playing(self) -> bool:
        """Whether the motion is showing on the walker."""
        return self._playing

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The pedestrian's animation; its movement too for the whole body."""
        claims = [(self._entity_name, ControlDomain.ANIMATION)]
        if not self._upper_body_only:
            claims.extend(movement_of(self._entity_name))
        return claims

    def reset(self) -> None:
        """Forget the earlier runs' motion (:meth:`BaseAction.reset`)."""
        super().reset()
        self._started_at = 0.0
        self._playing = False

    def close(self) -> None:
        """Hand the walker back to its animation if the run ends mid-motion,
        and close the conditions (:meth:`BaseAction.close`)."""
        self._stop()
        super().close()

    def execute(self, world: carla.World) -> None:
        """Begin the motion on the trigger's call, then show its next frame on
        every tick of the run."""
        if self.state is ActionState.STANDBY:
            self._start(world)
            return
        self._step(world)

    def _start(self, world: carla.World) -> None:
        self._playing = False
        if not self._poser.begin(self._upper_body_only):
            logger.warning(
                "PlayMotionAction: not playing %s on '%s'",
                self._motion_path,
                self._entity_name,
            )
            return
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

    def _step(self, world: carla.World) -> None:
        if not self._playing:
            return
        elapsed = _now(world) - self._started_at
        # The tolerance absorbs the clock's summing error: 20 ticks of 0.05 s
        # can read 0.9999999999999998 s (ElapsedTimeCondition).
        if self._duration_s is not None and elapsed >= self._duration_s - 1e-9:
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
        if not self._poser.show(self._motion.frame(frame)):
            # The walker was destroyed under the motion.
            self._playing = False

    def _stopped(self, world: carla.World) -> None:
        """Overridden mid-motion: hand the walker back, as at the motion's end."""
        self._stop()

    def _stop(self) -> None:
        """Hand the walker back to its animation."""
        self._playing = False
        if self._poser.end():
            logger.info("PlayMotionAction: '%s' finished its motion", self._entity_name)


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


class _Standing(BaseCondition):
    """Satisfied while the pedestrian stands: no faster than
    :data:`STANDING_SPEED_MS`, measured as a walker moves
    (:class:`~..kinematics.walker_motion.WalkerVelocities`)."""

    _entity_name: str
    _velocities: WalkerVelocities

    def __init__(self, entity_name: str, label: str) -> None:
        super().__init__(label=f"{label}_standing")
        self._entity_name = entity_name
        # The pedestrian's velocity, measured from where it stood when last
        # checked.
        self._velocities = WalkerVelocities()

    def close(self) -> None:
        """Forget the pedestrian measured (the run is over)."""
        self._velocities.clear()
        super().close()

    def reset(self) -> None:
        """Forget the pedestrian measured (:meth:`BaseCondition.reset`)."""
        super().reset()
        self._velocities.clear()

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        entity = find_pedestrian_entity(self._entity_name)
        if entity is None:
            return None
        actor = entity.actor
        if actor is None:
            return None
        walker: carla.Actor = actor
        speed = self._velocities.walker(walker, elapsed).magnitude()
        if speed > STANDING_SPEED_MS:
            return None
        return ScenarioResult(
            passed=True,
            message=f"'{self._entity_name}' stands ({speed:.2f} m/s)",
            elapsed_seconds=elapsed,
        )
