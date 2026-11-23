"""The frames a run records for its metrics, one a tick, and the series those
metrics are read into once the run is over.

:class:`FrameRecorder` records a run's road users every tick (each vehicle's
and walker's place, velocity, acceleration and box, the ego's steering and
light state, the run's importance-sampling weight) as :class:`Frame` values;
:mod:`~autoware_carla_scenario.metric_batch` replays them once the run is over
into one :class:`MetricSeries` a metric.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .constants import EGO_ROLE_NAME
from .entity_role import canonical_role_name
from .kinematics.walker_motion import WalkerVelocities

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = [
    "ActorFrame",
    "Frame",
    "FrameRecorder",
    "MetricSeries",
    "sampled",
]


@dataclass(frozen=True)
class ActorFrame:
    """A road user in a recorded frame, in CARLA's frame and units (degrees,
    deg/s).

    *role* is the actor's ``role_name`` as :func:`canonical_role_name` reads
    it: the ego's is ``"Ego"`` even when it was spawned as ``ego`` or ``hero``
    for CARLA's native Autoware interface.
    """

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


#: How many floats a recorded frame keeps an actor's motion in
#: (:class:`_Recorded`): place, rotation, velocity, acceleration and angular
#: velocity, three each.
_MOTION = 15


class _ActorStatic:
    """What does not change about an actor, read once and shared by every
    frame it is recorded in: its id, type, role, kind and box."""

    id: int
    type_id: str
    role: str
    base_type: str
    extent: tuple[float, float, float]
    centre: tuple[float, float, float]
    box_yaw: float

    def __init__(
        self,
        id: int,
        type_id: str,
        role: str,
        base_type: str,
        extent: tuple[float, float, float],
        centre: tuple[float, float, float],
        box_yaw: float,
    ) -> None:
        self.id = id
        self.type_id = type_id
        self.role = role
        self.base_type = base_type
        self.extent = extent
        self.centre = centre
        self.box_yaw = box_yaw


class _Recorded:
    """A frame's road users as :class:`FrameRecorder` keeps them (#2338): a
    reference to each one's :class:`_ActorStatic`, its motion as
    :data:`_MOTION` doubles in one ``array('d')`` (the floats it was read as,
    bit for bit), and the steer and light state of those that have them
    (the ego), by index.  :meth:`actors` makes the :class:`ActorFrame` values
    of it when the frame is read."""

    statics: list[_ActorStatic]
    motion: array
    controls: list[tuple[int, Optional[float], Optional[int]]]

    def __init__(self) -> None:
        self.statics = []
        self.motion = array("d")
        self.controls = []

    def actors(self) -> list[ActorFrame]:
        controls: dict[int, tuple[Optional[float], Optional[int]]] = {}
        for index, read_steer, read_light in self.controls:
            controls[index] = (read_steer, read_light)
        motion = self.motion
        actors: list[ActorFrame] = []
        for index, static in enumerate(self.statics):
            k = _MOTION * index
            steer: Optional[float] = None
            light: Optional[int] = None
            if index in controls:
                steer, light = controls[index]
            actors.append(
                ActorFrame(
                    id=static.id,
                    type_id=static.type_id,
                    role=static.role,
                    base_type=static.base_type,
                    x=motion[k],
                    y=motion[k + 1],
                    z=motion[k + 2],
                    roll=motion[k + 3],
                    pitch=motion[k + 4],
                    yaw=motion[k + 5],
                    vx=motion[k + 6],
                    vy=motion[k + 7],
                    vz=motion[k + 8],
                    ax=motion[k + 9],
                    ay=motion[k + 10],
                    az=motion[k + 11],
                    wx=motion[k + 12],
                    wy=motion[k + 13],
                    wz=motion[k + 14],
                    extent=static.extent,
                    centre=static.centre,
                    box_yaw=static.box_yaw,
                    steer=steer,
                    light_state=light,
                )
            )
        return actors


class Frame:
    """One tick of a run: its CARLA frame, simulated time, the run's
    importance-sampling weight then, and its road users.

    A frame :class:`FrameRecorder` records keeps its road users compactly
    and makes their :class:`ActorFrame` values each time :attr:`actors` is
    read (#2338); a frame made with its actors holds those.
    """

    frame: int
    t: float
    weight: Optional[float]
    _actors: tuple[ActorFrame, ...]
    _recorded: Optional[_Recorded]

    def __init__(
        self,
        frame: int,
        t: float,
        weight: Optional[float],
        actors: tuple[ActorFrame, ...],
    ) -> None:
        self.frame = frame
        self.t = t
        self.weight = weight
        self._actors = actors
        self._recorded = None

    @property
    def actors(self) -> tuple[ActorFrame, ...]:
        """The frame's road users."""
        recorded = self._recorded
        if recorded is None:
            return self._actors
        return tuple(recorded.actors())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        if self is other:
            # A frame equals itself, NaN readings and all, as the dataclass
            # did (its actors were one tuple; these are made afresh).
            return True
        return (self.frame, self.t, self.weight, self.actors) == (
            other.frame,
            other.t,
            other.weight,
            other.actors,
        )

    def __hash__(self) -> int:
        # Of what the frame says about the tick only: equal frames hash
        # alike all the same.
        return hash((self.frame, self.t, self.weight))


class FrameRecorder:
    """Records the frames a run's metrics are computed from, one
    :meth:`record` a tick."""

    frames: list[Frame]
    _walkers: WalkerVelocities
    #: What does not change about an actor, read once, by actor id.
    _static: dict[int, _ActorStatic]

    def __init__(self) -> None:
        self.frames = []
        self._walkers = WalkerVelocities()
        self._static = {}

    def _static_of(self, actor: "carla.Actor") -> _ActorStatic:
        key = int(actor.id)
        if key not in self._static:
            attributes = actor.attributes
            box = actor.bounding_box
            self._static[key] = _ActorStatic(
                key,
                str(actor.type_id),
                canonical_role_name(str(attributes.get("role_name", ""))),
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
        recorded = _Recorded()
        motion = recorded.motion
        for actor in world.get_actors():
            type_id = str(actor.type_id)
            walker = type_id.startswith("walker.")
            if not (walker or type_id.startswith("vehicle.")):
                continue
            transform = actor.get_transform()
            location, rotation = transform.location, transform.rotation
            v: tuple[float, float, float]
            if walker:
                velocity = self._walkers.walker(actor, elapsed, location)
                v = (float(velocity.x), float(velocity.y), float(velocity.z))
            else:
                reported = actor.get_velocity()
                v = (float(reported.x), float(reported.y), float(reported.z))
            acceleration = _acceleration(actor)
            angular = _angular_velocity(actor)
            static = self._static_of(actor)
            if static.role == str(EGO_ROLE_NAME) and not walker:
                steer: Optional[float] = None
                light: Optional[int] = None
                try:
                    # The steer and the light state are a vehicle's; the ego
                    # is one.  An actor without them (a stand-in) has neither
                    # read, as when the call raises.
                    if TYPE_CHECKING:
                        vehicle = actor.as_vehicle()
                    else:
                        vehicle = actor
                    steer = float(vehicle.get_control().steer)
                    light = int(vehicle.get_light_state())
                except Exception:
                    pass
                if steer is not None or light is not None:
                    recorded.controls.append((len(recorded.statics), steer, light))
            recorded.statics.append(static)
            motion.extend(
                (
                    float(location.x),
                    float(location.y),
                    float(location.z),
                    float(rotation.roll),
                    float(rotation.pitch),
                    float(rotation.yaw),
                    v[0],
                    v[1],
                    v[2],
                    acceleration[0],
                    acceleration[1],
                    acceleration[2],
                    angular[0],
                    angular[1],
                    angular[2],
                )
            )
        try:
            frame = int(world.get_snapshot().frame)
        except Exception:
            frame = len(self.frames)
        # Its actors are the recorded ones, not these.
        held: list[ActorFrame] = []
        recorded_frame = Frame(frame, float(elapsed), weight, tuple(held))
        recorded_frame._recorded = recorded
        self.frames.append(recorded_frame)


def _acceleration(actor: "carla.Actor") -> tuple[float, float, float]:
    """``actor.get_acceleration()`` as a tuple; zero when the actor cannot
    say."""
    try:
        v = actor.get_acceleration()
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return 0.0, 0.0, 0.0


def _angular_velocity(actor: "carla.Actor") -> tuple[float, float, float]:
    """``actor.get_angular_velocity()`` as a tuple; zero when the actor cannot
    say."""
    try:
        v = actor.get_angular_velocity()
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return 0.0, 0.0, 0.0


@dataclass
class MetricSeries:
    """A metric read over a run: its value at each time it was evaluated."""

    key: str
    times: list[float] = field(default_factory=list)
    values: list[Optional[float]] = field(default_factory=list)


def _not_nan(value: Optional[float]) -> Optional[float]:
    """*value*, NaN as ``None``; infinities are kept on purpose, for conditions."""
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
