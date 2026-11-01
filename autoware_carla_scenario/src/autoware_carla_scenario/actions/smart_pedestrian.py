"""Smart-pedestrian action: a pedestrian moved by a motion generator (ARDY).

The motion is generated as the run plays it, by a model told what the
pedestrian is to do: in words (``"a man walks to the kerb, looks left and
right, then crosses"``), along a path of waypoints, through full-body
keyframes.  The model runs in its own container (:class:`ArdyBackend`),
started before the run by the queue (``@need_prepare``).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import typesafe_carla.carla as carla

from ..action_state import ActionState
from ..conditions.base import BaseCondition, ScenarioResult
from ..control_domain import ControlDomain, movement_of
from ..entity.registry import find_pedestrian_entity
from ..entity_role import EntityRole
from ..pedestrian.ardy import ArdyBackend
from ..pedestrian.motion_client import (
    GeneratedFrame,
    PedestrianMotionClient,
    PoseKeyframe,
    RootWaypoint,
    TextSpan,
    to_carla,
    to_contract,
)
from ..pedestrian.poser import WalkerPoser
from ..pedestrian.skeleton import Vector, directions_from_joints
from ..prepare.declare import need_prepare
from ..prepare.pool import running_backend
from .base import BaseAction, TickTiming

logger = logging.getLogger(__name__)

__all__ = ["SmartPedestrianControllerAction", "TimedPrompt", "WalkWaypoint"]

#: Ask for more motion once no more than this many seconds of it are left.
_REPLAN_LEAD_S = 0.25

_SESSIONS = itertools.count(1)


@dataclass(frozen=True)
class TimedPrompt:
    """What a pedestrian does from *time_s* seconds into its motion on, in
    words (:class:`SmartPedestrianControllerAction`'s *prompts*)."""

    time_s: float
    text: str


@dataclass(frozen=True)
class WalkWaypoint:
    """A CARLA world point a pedestrian passes
    (:class:`SmartPedestrianControllerAction`'s *waypoints*)."""

    x: float
    y: float


@need_prepare(ArdyBackend)
class SmartPedestrianControllerAction(BaseAction):
    """Move a pedestrian -- its whole body and where it goes -- as a motion
    generator (`ARDY <https://github.com/nv-tlabs/ardy>`_) has it move.

    Conditioned by any of:

    - **text**: *prompt*, the whole run's, or *prompts*
      (:class:`TimedPrompt`), each taking over at its time into the motion;
    - **waypoints**: CARLA world points (:class:`WalkWaypoint`) the
      pedestrian passes, in order, at *speed_ms* along the path from where it
      stands;
    - **keyframes**: a JSON file of full-body poses the motion passes
      through (see ``docs/pedestrian_motion.md``).

    The generator runs in a container the queue starts before the run
    (``backends.ardy`` of the config), on a GPU of this host; a host that
    cannot run it fails the run before it starts.  Every tick the action
    places the walker at the motion's root and poses its skeleton
    (:mod:`~autoware_carla_scenario.pedestrian`); the walker stays on the
    ground it stood on when the motion began.  It asks the generator for
    more as the motion it has runs out, synchronously, so a run replays
    exactly from the same *seed*.

    **Control domains**: the pedestrian's movement and animation, both, for
    as long as the motion plays.

    The action runs for *duration_s*; without one, until the last waypoint's
    or keyframe's time, whichever is later, and without either, until the
    scenario ends. A prompt that starts after that time never plays.

    Args:
        entity_name: ``role_name`` of the pedestrian.
        prompt: What the pedestrian does, in words; ``""``: no text.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        prompts: Texts, each taking over at its time into the motion (after
            *prompt*, from 0).
        waypoints: CARLA world points to pass, in order.
        speed_ms: Walking speed along *waypoints*, m/s.
        keyframes: A JSON file of full-body poses.  A relative path is taken
            from the working directory.
        duration_s: Stop after this many simulated seconds.
        seed: The seed of the generator's draws.
        overrides: Take over a control domain of the pedestrian another
            action holds (:attr:`~BaseAction.overrides`).

    Raises:
        FileNotFoundError: If there is no file at *keyframes*.
        ValueError: If *speed_ms* or *duration_s* is not positive, a prompt's
            time is negative, or the keyframes file is malformed.
    """

    _entity_name: str
    _texts: tuple[tuple[float, str], ...]
    _waypoints: tuple[tuple[float, float], ...]
    _speed_ms: float
    _keyframes: tuple["_Keyframe", ...]
    _duration_s: Optional[float]
    _seed: int
    _poser: WalkerPoser
    _client: Optional[PedestrianMotionClient]
    _session: str
    _fps: float
    _frames: list[GeneratedFrame]
    _started_at: float
    _end_s: Optional[float]
    _ground_z: float
    _moving: bool

    REISSUES_BY_DEFAULT = True

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        prompt: str = "",
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "smart_pedestrian",
        once: bool = True,
        prompts: Sequence[TimedPrompt] = (),
        waypoints: Sequence[WalkWaypoint] = (),
        speed_ms: float = 1.3,
        keyframes: Optional[str] = None,
        duration_s: Optional[float] = None,
        seed: int = 0,
        overrides: bool = False,
    ) -> None:
        if speed_ms <= 0:
            raise ValueError("speed_ms must be positive")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration_s must be positive")
        texts = [(0.0, prompt)] if prompt else []
        for timed in prompts:
            if timed.time_s < 0:
                raise ValueError(
                    f"a prompt's time must not be negative, not {timed.time_s}"
                )
            texts.append((float(timed.time_s), str(timed.text)))
        super().__init__(
            label=label,
            condition=condition,
            timing=timing,
            once=once,
            until=_MotionEnded(self, label),
            overrides=overrides,
        )
        self._entity_name = str(entity_name)
        self._texts = tuple(sorted(texts, key=lambda span: span[0]))
        self._waypoints = tuple((float(p.x), float(p.y)) for p in waypoints)
        self._speed_ms = float(speed_ms)
        self._keyframes = () if keyframes is None else _read_keyframes(keyframes)
        self._duration_s = duration_s
        self._seed = int(seed)
        self._poser = WalkerPoser(self._entity_name, "SmartPedestrianControllerAction")
        self._client = None
        self._session = ""
        self._fps = 0.0
        self._frames = []
        self._started_at = 0.0
        self._end_s = None
        self._ground_z = 0.0
        self._moving = False

    @property
    def moving(self) -> bool:
        """Whether the generated motion moves the walker."""
        return self._moving

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """The pedestrian's movement and its animation."""
        return [
            *movement_of(self._entity_name),
            (self._entity_name, ControlDomain.ANIMATION),
        ]

    def execute(self, world: carla.World) -> None:
        """Open the session on the trigger's call; then, every tick, show the
        frame the run is at, asking for more motion as it runs out."""
        if self.state is ActionState.STANDBY:
            self._start(world)
        if self._moving:
            self._step(world)

    def _start(self, world: carla.World) -> None:
        self._end()
        entity = find_pedestrian_entity(self._entity_name)
        if entity is not None:
            entity.release_from_traffic(world)
        if not self._poser.begin(upper_body_only=False):
            return
        actor = self._poser.actor
        if actor is None:  # pragma: no cover - begin() succeeded
            return
        here = actor.get_transform()
        self._ground_z = float(here.location.z)
        backend = running_backend(ArdyBackend)
        timeout = backend.timeout_s if isinstance(backend, ArdyBackend) else 120.0
        # Kept at once: a call below that fails still has _end() close it
        # and hand the walker back.
        self._client = PedestrianMotionClient(backend.address(), timeout_s=timeout)
        try:
            self._open(world, self._client, here)
        except BaseException:
            self._end()
            raise

    def _open(
        self, world: carla.World, client: PedestrianMotionClient, here: carla.Transform
    ) -> None:
        info = client.negotiate()
        self._fps = info.fps
        x, y, heading = to_contract(here.location.x, here.location.y, here.rotation.yaw)
        waypoints = _timed_waypoints((x, y), self._waypoints, self._speed_ms, info.fps)
        self._session = f"{self._entity_name}/{self.label}/{next(_SESSIONS)}"
        client.open_session(
            self._session,
            (x, y),
            heading,
            seed=self._seed,
            texts=[
                TextSpan(start_frame=int(round(at * info.fps)), text=text)
                for at, text in self._texts
            ],
            waypoints=waypoints,
            keyframes=[
                key.contract(info.joint_names, info.fps, (x, y, heading))
                for key in self._keyframes
            ],
        )
        self._frames = client.step(self._session, -1)
        self._started_at = _now(world)
        ends = [waypoints[-1].frame / info.fps] if waypoints else []
        ends += [key.time_s for key in self._keyframes]
        if self._duration_s is not None:
            self._end_s = self._duration_s
        else:
            self._end_s = max(ends) if ends else None
        self._moving = True
        logger.info(
            "SmartPedestrianControllerAction: '%s' moves by %s %s (%s) at %.0f fps",
            self._entity_name,
            info.backend,
            info.backend_version,
            info.model,
            info.fps,
        )

    def _step(self, world: carla.World) -> None:
        elapsed = _now(world) - self._started_at
        if self._end_s is not None and elapsed >= self._end_s:
            self._end()
            return
        frame = int(math.floor(elapsed * self._fps + 1e-9))
        client = self._client
        if client is None:
            return
        last = self._frames[-1].frame if self._frames else -1
        if last - frame <= int(math.ceil(_REPLAN_LEAD_S * self._fps)):
            fresh = client.step(self._session, min(frame + 1, last))
            # The generator may keep more history than asked (whole tokens):
            # its frames take over from the first it sends.
            start = fresh[0].frame if fresh else last + 1
            # Frames already shown are dropped: a long run keeps only the plan ahead.
            self._frames = [f for f in self._frames if frame <= f.frame < start] + fresh
        self._show(frame)

    def _show(self, frame: int) -> None:
        frames = self._frames
        if not frames:
            return
        index = max(0, min(frame - frames[0].frame, len(frames) - 1))
        now = frames[index]
        after = frames[min(index + 1, len(frames) - 1)]
        x, y, yaw = to_carla(now.root[0], now.root[1], now.heading)
        nx, ny, _ = to_carla(after.root[0], after.root[1], after.heading)
        steps = max(1, after.frame - now.frame)
        velocity = carla.Vector3D(
            (nx - x) * self._fps / steps, (ny - y) * self._fps / steps, 0.0
        )
        placed = self._poser.place(
            carla.Location(x=x, y=y, z=self._ground_z), yaw, velocity
        )
        if not placed or not self._poser.show(
            directions_from_joints(now.joints, now.heading)
        ):
            # The walker was destroyed under the motion.
            self._end()

    def _end(self) -> None:
        """Hand the walker back and close the session."""
        was_moving = self._moving
        self._moving = False
        self._frames = []
        self._poser.end()
        client = self._client
        self._client = None
        if client is not None:
            if self._session:
                client.close_session(self._session)
            client.close()
        self._session = ""
        if was_moving:
            logger.info(
                "SmartPedestrianControllerAction: '%s' finished its motion",
                self._entity_name,
            )

    def _stopped(self, world: carla.World) -> None:
        self._end()

    def close(self) -> None:
        self._end()
        super().close()


class _Keyframe:
    """One full-body pose of a keyframes file, before the generator's joint
    order and rate are known."""

    def __init__(
        self,
        time_s: float,
        joints: dict[str, Vector],
        root: Optional[tuple[float, float]],
        yaw_deg: Optional[float],
    ) -> None:
        self.time_s = time_s
        self.joints = joints
        self.root = root
        self.yaw_deg = yaw_deg

    def contract(
        self, names: Sequence[str], fps: float, start: tuple[float, float, float]
    ) -> PoseKeyframe:
        """The pose in the contract frame, its joints in *names*' order, placed
        at its root (the motion's start pose where it gives none)."""
        missing = [name for name in names if name not in self.joints]
        if missing:
            raise ValueError(
                f"a keyframe at {self.time_s} s lacks joints the generator "
                f"places: {missing}"
            )
        if self.root is None:
            x, y = start[0], start[1]
        else:
            x, y, _ = to_contract(self.root[0], self.root[1], 0.0)
        heading = start[2] if self.yaw_deg is None else -math.radians(self.yaw_deg)
        cos, sin = math.cos(heading), math.sin(heading)
        joints = []
        for name in names:
            forward, left, up = self.joints[name]
            joints.append(
                (x + forward * cos - left * sin, y + forward * sin + left * cos, up)
            )
        return PoseKeyframe(frame=int(round(self.time_s * fps)), joints=tuple(joints))


def _read_keyframes(path: Union[str, Path]) -> tuple[_Keyframe, ...]:
    """The poses of a keyframes file (``docs/pedestrian_motion.md``)."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"no keyframes file at {file}")
    try:
        document = json.loads(file.read_text())
        poses = []
        for entry in document["keyframes"]:
            time_s = float(entry["time_s"])
            if time_s < 0:
                raise ValueError(f"time_s must not be negative, not {time_s}")
            joints = {
                str(name): (float(p[0]), float(p[1]), float(p[2]))
                for name, p in entry["joints"].items()
            }
            root = (
                (float(entry["x"]), float(entry["y"]))
                if "x" in entry and "y" in entry
                else None
            )
            yaw = float(entry["yaw_deg"]) if "yaw_deg" in entry else None
            poses.append(_Keyframe(time_s, joints, root, yaw))
    except (
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(f"{file}: not a keyframes file ({error})") from None
    return tuple(sorted(poses, key=lambda pose: pose.time_s))


def _timed_waypoints(
    start: tuple[float, float],
    points: Sequence[tuple[float, float]],
    speed_ms: float,
    fps: float,
) -> list[RootWaypoint]:
    """*points* (CARLA) in the contract frame, each at the frame a walk at
    *speed_ms* from *start* (contract) reaches it."""
    timed: list[RootWaypoint] = []
    x, y = start
    distance = 0.0
    for px, py in points:
        cx, cy, _ = to_contract(px, py, 0.0)
        distance += math.hypot(cx - x, cy - y)
        x, y = cx, cy
        timed.append(
            RootWaypoint(frame=int(round(distance / speed_ms * fps)), x=cx, y=cy)
        )
    return timed


def _now(world: carla.World) -> float:
    """The simulated time, in seconds."""
    return float(world.get_snapshot().timestamp.elapsed_seconds)


class _MotionEnded(BaseCondition):
    """Satisfied once the motion it watches has started and no longer moves
    the walker."""

    _action: SmartPedestrianControllerAction

    def __init__(self, action: SmartPedestrianControllerAction, label: str) -> None:
        super().__init__(label=f"{label}_ended")
        self._action = action

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        if self._action.moving:
            return None
        return ScenarioResult(
            passed=True, message="the motion has ended", elapsed_seconds=elapsed
        )
