"""What crosses between the ego entity and its DDS transport, and who drives.

The transport turns DDS samples into the plain commands here and the ego's state
into DDS samples, so the entity never touches a DDS type.  :class:`CommandArbiter`
picks, each tick, the command source that drives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..driver.geometry import Pose
from .config import CommandSourceConfig

__all__ = [
    "Command",
    "CommandArbiter",
    "ControlCommand",
    "EgoState",
    "Selection",
    "SteerAccelCommand",
    "SteerAccelPairer",
    "TrajectoryCommand",
    "TrajectoryCommandPoint",
    "yaw_from_quaternion",
]

#: Slack on the freshness test, so that a command exactly ``stale_s`` old --
#: as a sum of 0.05 s ticks lands, give or take a rounding -- is still fresh.
_FRESHNESS_EPSILON_S: float = 1e-9


@dataclass(frozen=True)
class EgoState:
    """The ego, as one tick publishes it.

    Attributes:
        stamp_s: Simulation time, as CARLA's ``/clock`` reports it.
        pose: The ``base_link`` pose in the right-handed local frame.
        linear_velocity: ``(3,)`` velocity in ``base_link``, m/s.
        angular_velocity: ``(3,)`` angular velocity in ``base_link``, rad/s.
        speed_mps: Ground speed.
        steering_tire_angle_rad: Front tyre angle, positive to the left.
    """

    stamp_s: float
    pose: Pose
    linear_velocity: np.ndarray[tuple[int], np.dtype[np.float64]]
    angular_velocity: np.ndarray[tuple[int], np.dtype[np.float64]]
    speed_mps: float
    steering_tire_angle_rad: float


@dataclass(frozen=True)
class SteerAccelCommand:
    """A steering angle and an acceleration, as VisionPilot publishes them.

    Attributes:
        steering_rad: Tyre angle in the source's own sign convention.
        acceleration_mps2: Desired acceleration.
    """

    steering_rad: float
    acceleration_mps2: float


@dataclass(frozen=True)
class ControlCommand:
    """An ``autoware_control_msgs/msg/Control``.

    Attributes:
        steering_tire_angle_rad: Tyre angle, positive to the left (Autoware's).
        velocity_mps: Target speed.
        acceleration_mps2: Desired acceleration.
        jerk_mps3: Jerk limit, ``None`` when the message leaves it undefined.
    """

    steering_tire_angle_rad: float
    velocity_mps: float
    acceleration_mps2: float
    jerk_mps3: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryCommandPoint:
    """One point of a planned trajectory, in the right-handed local frame.

    Attributes:
        time_from_start_s: When the ego should be here, after the plan's stamp.
        x: Position, metres.
        y: Position, metres.
        z: Position, metres.
        yaw: Heading, radians, counter-clockwise from ``+x``.
        velocity_mps: Longitudinal speed at this point.
    """

    time_from_start_s: float
    x: float
    y: float
    z: float
    yaw: float
    velocity_mps: float


@dataclass(frozen=True)
class TrajectoryCommand:
    """An ``autoware_planning_msgs/msg/Trajectory``.

    Attributes:
        stamp_s: ``header.stamp``, simulation seconds; ``0.0`` when unset.
        points: The plan.
    """

    stamp_s: float
    points: Tuple[TrajectoryCommandPoint, ...]


Command = Union[SteerAccelCommand, ControlCommand, TrajectoryCommand]


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return the heading of a quaternion, radians about ``+z``."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SteerAccelPairer:
    """Pairs a steering topic with an acceleration topic into one command.

    A command is produced once a fresh value has arrived on both topics, and
    each value is used once -- VisionPilot emits one of each per planning cycle,
    so this yields one command per cycle, exactly as its
    ``carla_control_publisher`` does.
    """

    _steering: Optional[float]
    _acceleration: Optional[float]

    def __init__(self) -> None:
        self._steering = None
        self._acceleration = None

    def steering(self, value: float) -> Optional[SteerAccelCommand]:
        """Take a steering sample; return the command it completes, if any."""
        self._steering = value
        return self._complete()

    def acceleration(self, value: float) -> Optional[SteerAccelCommand]:
        """Take an acceleration sample; return the command it completes, if any."""
        self._acceleration = value
        return self._complete()

    def _complete(self) -> Optional[SteerAccelCommand]:
        if self._steering is None or self._acceleration is None:
            return None
        command = SteerAccelCommand(self._steering, self._acceleration)
        self._steering = None
        self._acceleration = None
        return command


@dataclass(frozen=True)
class Selection:
    """The command that drives this tick.

    Attributes:
        source: The source it came from.
        command: The command.
        received_s: Simulation time it arrived.
        fresh: Whether its source was still fresh at the tick that chose it.
    """

    source: CommandSourceConfig
    command: Command
    received_s: float
    fresh: bool


class CommandArbiter:
    """Chooses, each tick, which source's command drives the ego.

    Of the sources whose last command arrived at most ``stale_s`` simulated
    seconds ago, the highest priority wins; between equal priorities, the most
    recent command.  When every source is stale, :meth:`select` returns the
    command that drove last, marked not fresh, so the caller can hold it or
    brake.

    Args:
        sources: The configured sources.
    """

    _sources: Dict[str, CommandSourceConfig]
    _latest: Dict[str, Tuple[Command, float]]
    _last: Optional[Selection]

    def __init__(self, sources: Sequence[CommandSourceConfig]) -> None:
        self._sources = {source.name: source for source in sources}
        self._latest = {}
        self._last = None

    def observe(self, source_name: str, command: Command, now_s: float) -> None:
        """Record *command* from *source_name*, arrived at *now_s*.

        Raises:
            KeyError: If *source_name* is not a configured source.
        """
        if source_name not in self._sources:
            raise KeyError(f"unknown command source {source_name!r}")
        self._latest[source_name] = (command, now_s)

    def select(self, now_s: float) -> Optional[Selection]:
        """Return the command that drives at *now_s*, or ``None`` before any."""
        best: Optional[Selection] = None
        for name, (command, received_s) in self._latest.items():
            source = self._sources[name]
            if now_s - received_s > source.stale_s + _FRESHNESS_EPSILON_S:
                continue
            if (
                best is None
                or source.priority > best.source.priority
                or (
                    source.priority == best.source.priority
                    and received_s > best.received_s
                )
            ):
                best = Selection(source, command, received_s, fresh=True)
        if best is not None:
            self._last = best
            return best
        if self._last is None:
            return None
        last = self._last
        return Selection(last.source, last.command, last.received_s, fresh=False)
