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

    frames: list[Frame]
    _walkers: WalkerVelocities
    #: What does not change about an actor, read once: type, role, kind, box.
    _static: dict[
        int,
        tuple[
            str,
            str,
            str,
            tuple[float, float, float],
            tuple[float, float, float],
            float,
        ],
    ]

    def __init__(self) -> None:
        self.frames = []
        self._walkers = WalkerVelocities()
        self._static = {}

    def _static_of(
        self, actor: "carla.Actor"
    ) -> tuple[
        str, str, str, tuple[float, float, float], tuple[float, float, float], float
    ]:
        key = int(actor.id)
        if key not in self._static:
            attributes = actor.attributes
            box = actor.bounding_box
            self._static[key] = (
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
        actors: list[ActorFrame] = []
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
                v = (velocity.x, velocity.y, velocity.z)
            else:
                reported = actor.get_velocity()
                v = (float(reported.x), float(reported.y), float(reported.z))
            acceleration = _acceleration(actor)
            angular = _angular_velocity(actor)
            kind, role, base, extent, centre, box_yaw = self._static_of(actor)
            steer: Optional[float] = None
            light: Optional[int] = None
            if role == str(EGO_ROLE_NAME) and not walker:
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
            actors.append(
                ActorFrame(
                    id=int(actor.id),
                    type_id=kind,
                    role=role,
                    base_type=base,
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                    roll=float(rotation.roll),
                    pitch=float(rotation.pitch),
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
                    extent=extent,
                    centre=centre,
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
