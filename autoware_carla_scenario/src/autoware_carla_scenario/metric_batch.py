"""Metrics computed once the run is over, from the frames it recorded.

A run with a :class:`~autoware_carla_scenario.conditions.MetricCondition`
records its road users every tick (:class:`FrameRecorder`): each vehicle's and
walker's place, velocity, acceleration and box, the ego's steering and light
state, the run's importance-sampling weight.  When the run ends, while its
CARLA map is still loaded, :func:`evaluate_metrics` replays those frames:

* AlpaSim's metrics, and the run's own, by the very code that reads them off a
  live world (:func:`~autoware_carla_scenario.metrics.read_metric`), on a
  stand-in world (:class:`ReplayWorld`) that answers with the recorded actors
  and the run's map -- every tick, in order, so the weaving metrics see the
  run as it went;
* CriMe's and CommonRoad-Reach's (:data:`~autoware_carla_scenario.metrics.COMMONROAD_METRICS`)
  by the CommonRoad workers, every
  :attr:`~autoware_carla_scenario.commonroad.CommonRoadSettings.period_s` of
  simulated time, the frames shared among several workers that run at
  once (:func:`~autoware_carla_scenario.commonroad.session.evaluate_commonroad`).

The simulation never waits on a metric, and the frames, being independent of
one another, are evaluated in parallel.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

from .constants import EGO_ROLE_NAME
from .kinematics.walker_motion import WalkerVelocities
from .metrics import (
    BUILT_IN_METRICS,
    COMMONROAD_METRICS,
    read_metric,
    reset_ego_motion,
    set_importance_sampling_weight,
    track_ego_motion,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "ActorFrame",
    "Frame",
    "FrameRecorder",
    "MetricSeries",
    "ReplayWorld",
    "evaluate_metrics",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActorFrame:
    """A road user in a recorded frame, in CARLA's frame and units (degrees,
    deg/s)."""

    id: int
    type_id: str
    role: str
    base_type: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    #: The bounding box: half extent, centre offset (actor frame) and yaw.
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_yaw: float = 0.0
    #: The ego's steer (``get_control().steer``) and CARLA light state; None
    #: for the others.
    steer: Optional[float] = None
    light_state: Optional[int] = None


@dataclass(frozen=True)
class Frame:
    """One tick of a run: its CARLA frame, simulated time, the run's
    importance-sampling weight then, and its road users."""

    frame: int
    t: float
    weight: Optional[float]
    actors: tuple[ActorFrame, ...]


class FrameRecorder:
    """Records the frames a run's metrics are computed from, one
    :meth:`record` a tick."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._walkers = WalkerVelocities()
        #: What does not change about an actor, read once: type, role, kind,
        #: box.
        self._static: dict[
            int, tuple[str, str, str, tuple[float, ...], tuple[float, ...], float]
        ] = {}

    def _static_of(
        self, actor: Any
    ) -> tuple[str, str, str, tuple[float, ...], tuple[float, ...], float]:
        key = int(actor.id)
        if key not in self._static:
            attributes = actor.attributes
            box = actor.bounding_box
            self._static[key] = (
                str(actor.type_id),
                str(attributes.get("role_name", "")),
                str(attributes.get("base_type", "")),
                (float(box.extent.x), float(box.extent.y), float(box.extent.z)),
                (float(box.location.x), float(box.location.y), float(box.location.z)),
                float(box.rotation.yaw),
            )
        return self._static[key]

    def record(
        self, world: "carla.World", elapsed: float, weight: Optional[float]
    ) -> None:
        """Record the tick the world was just stepped to, at *elapsed*
        simulated seconds, with the run's importance-sampling *weight*."""
        actors: list[ActorFrame] = []
        for actor in world.get_actors():
            type_id = str(actor.type_id)
            walker = type_id.startswith("walker.")
            if not (walker or type_id.startswith("vehicle.")):
                continue
            transform = actor.get_transform()
            location, rotation = transform.location, transform.rotation
            if walker:
                velocity = self._walkers.walker(actor, elapsed, location)
                v = (velocity.x, velocity.y, velocity.z)
            else:
                reported = actor.get_velocity()
                v = (float(reported.x), float(reported.y), float(reported.z))
            acceleration = _read_vector(actor, "get_acceleration")
            angular = _read_vector(actor, "get_angular_velocity")
            kind, role, base, extent, centre, box_yaw = self._static_of(actor)
            steer: Optional[float] = None
            light: Optional[int] = None
            if role == str(EGO_ROLE_NAME) and not walker:
                try:
                    steer = float(actor.get_control().steer)
                    light = int(actor.get_light_state())
                except Exception:
                    pass
            actors.append(
                ActorFrame(
                    id=int(actor.id),
                    type_id=kind,
                    role=role,
                    base_type=base,
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                    roll=float(getattr(rotation, "roll", 0.0)),
                    pitch=float(getattr(rotation, "pitch", 0.0)),
                    yaw=float(rotation.yaw),
                    vx=v[0],
                    vy=v[1],
                    vz=v[2],
                    ax=acceleration[0],
                    ay=acceleration[1],
                    az=acceleration[2],
                    wx=angular[0],
                    wy=angular[1],
                    wz=angular[2],
                    extent=extent,  # type: ignore[arg-type]
                    centre=centre,  # type: ignore[arg-type]
                    box_yaw=box_yaw,
                    steer=steer,
                    light_state=light,
                )
            )
        try:
            frame = int(world.get_snapshot().frame)
        except Exception:
            frame = len(self.frames)
        self.frames.append(Frame(frame, float(elapsed), weight, tuple(actors)))


def _read_vector(actor: Any, method: str) -> tuple[float, float, float]:
    """``actor.<method>()`` as a tuple; zero when the actor cannot say."""
    try:
        v = getattr(actor, method)()
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Replaying a frame as a world
# ---------------------------------------------------------------------------


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _ReplayActor:
    """A recorded road user, answering as a CARLA actor does."""

    def __init__(self, state: ActorFrame) -> None:
        self._state = state
        self.id = state.id
        self.type_id = state.type_id
        self.is_alive = True
        self.attributes = {"role_name": state.role, "base_type": state.base_type}
        self.bounding_box = SimpleNamespace(
            extent=_vector(*state.extent),
            location=_vector(*state.centre),
            rotation=SimpleNamespace(yaw=state.box_yaw, pitch=0.0, roll=0.0),
        )

    def get_transform(self) -> Any:
        s = self._state
        return SimpleNamespace(
            location=_vector(s.x, s.y, s.z),
            rotation=SimpleNamespace(roll=s.roll, pitch=s.pitch, yaw=s.yaw),
        )

    def get_location(self) -> Any:
        s = self._state
        return _vector(s.x, s.y, s.z)

    def get_velocity(self) -> Any:
        s = self._state
        return _vector(s.vx, s.vy, s.vz)

    def get_acceleration(self) -> Any:
        s = self._state
        return _vector(s.ax, s.ay, s.az)

    def get_angular_velocity(self) -> Any:
        s = self._state
        return _vector(s.wx, s.wy, s.wz)

    def get_control(self) -> Any:
        return SimpleNamespace(
            steer=0.0 if self._state.steer is None else self._state.steer
        )

    def get_light_state(self) -> int:
        return 0 if self._state.light_state is None else self._state.light_state


class _ActorList(list):  # type: ignore[type-arg]
    """A list with ``ActorList.filter()``: a wildcard on ``type_id``."""

    def filter(self, pattern: str) -> _ActorList:
        return _ActorList(a for a in self if fnmatch.fnmatchcase(a.type_id, pattern))


class ReplayWorld:
    """A recorded frame, answering as a CARLA world does: its actors, its
    frame number, and the run's map (``get_map``), which is still loaded.

    One world replays every frame of a run (:meth:`show`), so what reads it
    keeps what it holds of the world -- the map above all -- across frames.
    """

    def __init__(self, carla_map: Any) -> None:
        self._map = carla_map
        self._frame = Frame(0, 0.0, None, ())
        self._actors = _ActorList()
        self._by_id: dict[int, _ReplayActor] = {}

    @property
    def id(self) -> int:
        return id(self)

    def show(self, frame: Frame) -> None:
        """Answer with *frame* from now on."""
        self._frame = frame
        self._actors = _ActorList(_ReplayActor(a) for a in frame.actors)
        self._by_id = {a.id: a for a in self._actors}

    def get_actors(self) -> _ActorList:
        return self._actors

    def get_actor(self, actor_id: int) -> Optional[_ReplayActor]:
        return self._by_id.get(int(actor_id))

    def get_snapshot(self) -> Any:
        return SimpleNamespace(
            frame=self._frame.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self._frame.t),
        )

    def get_map(self) -> Any:
        return self._map


# ---------------------------------------------------------------------------
# Evaluating
# ---------------------------------------------------------------------------


@dataclass
class MetricSeries:
    """A metric read over a run: its value at each time it was evaluated."""

    key: str
    times: list[float] = field(default_factory=list)
    values: list[Optional[float]] = field(default_factory=list)


def _finite(value: Optional[float]) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return float(value)


def sampled(frames: list[Frame], period_s: float) -> list[Frame]:
    """The frames at least *period_s* apart, from the first."""
    picked: list[Frame] = []
    for frame in frames:
        if not picked or frame.t - picked[-1].t >= period_s - 1e-9:
            picked.append(frame)
    return picked


def evaluate_metrics(
    frames: list[Frame],
    carla_map: Any,
    keys: list[str],
) -> dict[str, MetricSeries]:
    """The metrics *keys* over the recorded *frames*, on the run's
    *carla_map*.

    Raises:
        ValueError: A key is not a built-in metric's.
        ~autoware_carla_scenario.commonroad.CommonRoadWorkerError: A
            CommonRoad metric is asked for and its workers cannot be started.
    """
    unknown = [key for key in keys if key not in BUILT_IN_METRICS]
    if unknown:
        raise ValueError(f"not metrics: {unknown}")
    wanted = list(dict.fromkeys(keys))
    series = {key: MetricSeries(key) for key in wanted}
    world_keys = [key for key in wanted if key not in COMMONROAD_METRICS]
    commonroad_keys = [key for key in wanted if key in COMMONROAD_METRICS]
    world = ReplayWorld(carla_map)

    if world_keys:
        reset_ego_motion()
        try:
            for frame in frames:
                world.show(frame)
                set_importance_sampling_weight(frame.weight)
                track_ego_motion(world, frame.t)  # type: ignore[arg-type]
                for key in world_keys:
                    series[key].times.append(frame.t)
                    series[key].values.append(_finite(read_metric(key, world)))  # type: ignore[arg-type]
        finally:
            reset_ego_motion()
            set_importance_sampling_weight(None)

    if commonroad_keys:
        from .commonroad.session import (  # noqa: PLC0415
            commonroad_settings,
            evaluate_commonroad,
        )

        picked = sampled(frames, commonroad_settings().period_s)
        values = evaluate_commonroad(picked, carla_map, commonroad_keys)
        for key in commonroad_keys:
            series[key].times = [frame.t for frame in picked]
            series[key].values = [_finite(v) for v in values[key]]
    return series
