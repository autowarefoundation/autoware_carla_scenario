"""What crosses between the ego entity and its DDS transport, and who drives.

The transport turns DDS samples into the plain commands here and the ego's state
into DDS samples, so the entity never touches a DDS type.  :class:`CommandArbiter`
picks, each tick, the command source that drives.

Time is in integer microseconds of simulation time throughout, the clock the
trajectory follower and the ego observation already keep.
"""

from __future__ import annotations

import dataclasses
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
]


@dataclass(frozen=True)
class EgoState:
    """The ego, as one tick publishes it.

    Attributes:
        stamp_us: Simulation time, as CARLA's ``/clock`` reports it.
        pose: The ``base_link`` pose in the right-handed local frame.
        linear_velocity: ``(3,)`` velocity in ``base_link``, m/s.
        angular_velocity: ``(3,)`` angular velocity in ``base_link``, rad/s.
        speed_mps: Ground speed.
        steering_tire_angle_rad: Front tyre angle, positive to the left.
        turn_indicators: The indicators the ego shows, as a
            ``TurnIndicatorsReport`` value (1 off, 2 left, 3 right).
    """

    stamp_us: int
    pose: Pose
    linear_velocity: np.ndarray[tuple[int], np.dtype[np.float64]]
    angular_velocity: np.ndarray[tuple[int], np.dtype[np.float64]]
    speed_mps: float
    steering_tire_angle_rad: float
    turn_indicators: int = 1


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
    """One point of a planned trajectory.

    Attributes:
        time_from_start_us: When the ego should be here, after the plan's stamp.
        pose: Where, in the right-handed local frame.
        velocity_mps: Longitudinal speed at this point.
        acceleration_mps2: Longitudinal acceleration at this point.
    """

    time_from_start_us: int
    pose: Pose
    velocity_mps: float
    acceleration_mps2: float = 0.0


@dataclass(frozen=True)
class TrajectoryCommand:
    """An ``autoware_planning_msgs/msg/Trajectory``.

    Attributes:
        stamp_us: ``header.stamp``; ``0`` when unset.
        points: The plan.
    """

    stamp_us: int
    points: Tuple[TrajectoryCommandPoint, ...]


Command = Union[SteerAccelCommand, ControlCommand, TrajectoryCommand]


class SteerAccelPairer:
    """Pairs a steering topic with an acceleration topic into one command.

    VisionPilot emits one of each per planning cycle, back to back, so this
    yields one command per cycle, exactly as its ``carla_control_publisher``
    does.  The two halves of a cycle are matched by their source timestamps:
    the caller feeds every sample of both topics in the order they were written
    (by ``source_timestamp``), and a half pairs with the pending half of the
    other topic only if the two were written at most *window_ns* apart.

    At most one half per topic is pending, so a silent topic cannot grow
    memory, and an unmatched half is dropped rather than paired with a later
    cycle's, so a lost message costs that one cycle and never shifts the
    pairing:

    * a second half on the same topic supersedes the pending one, whose partner
      was lost (a cycle has one of each);
    * a half written more than *window_ns* after the pending half of the other
      topic drops that one, which is then stale, and waits for its own partner;
    * a half written more than *window_ns* before the pending half of the other
      topic (delivered late, across polls) is dropped itself.

    Each value is used once.

    Args:
        window_ns: How far apart, in source-clock nanoseconds, the two halves of
            one cycle may be written.  Keep it well under the publisher's cycle.
    """

    _window_ns: int
    _steering: Optional[Tuple[float, int]]
    _acceleration: Optional[Tuple[float, int]]

    def __init__(self, window_ns: int = 20_000_000) -> None:
        if window_ns < 0:
            raise ValueError(f"window_ns must not be negative, got {window_ns}")
        self._window_ns = window_ns
        self._steering = None
        self._acceleration = None

    def steering(self, value: float, stamp_ns: int) -> Optional[SteerAccelCommand]:
        """Take a steering sample written at *stamp_ns*; return the command it
        completes, if any."""
        partner = self._acceleration
        if partner is not None:
            if abs(stamp_ns - partner[1]) <= self._window_ns:
                self._acceleration = None
                self._steering = None
                return SteerAccelCommand(value, partner[0])
            if stamp_ns < partner[1]:
                return None
            self._acceleration = None
        self._steering = (value, stamp_ns)
        return None

    def acceleration(self, value: float, stamp_ns: int) -> Optional[SteerAccelCommand]:
        """Take an acceleration sample written at *stamp_ns*; return the command
        it completes, if any."""
        partner = self._steering
        if partner is not None:
            if abs(stamp_ns - partner[1]) <= self._window_ns:
                self._steering = None
                self._acceleration = None
                return SteerAccelCommand(partner[0], value)
            if stamp_ns < partner[1]:
                return None
            self._steering = None
        self._acceleration = (value, stamp_ns)
        return None


@dataclass(frozen=True)
class Selection:
    """The command that drives this tick.

    Attributes:
        source: The source it came from.
        command: The command.
        received_us: Simulation time it arrived.
        fresh: Whether its source was still fresh at the tick that chose it.
    """

    source: CommandSourceConfig
    command: Command
    received_us: int
    fresh: bool


class CommandArbiter:
    """Chooses, each tick, which source's command drives the ego.

    Of the sources whose last command arrived at most ``stale_s`` ago, the
    highest priority wins; between equal priorities, the most recent command.
    When every source is stale, :meth:`select` returns the command that drove
    last, marked not fresh, so the caller can hold it or brake.

    Args:
        sources: The configured sources.
    """

    _sources: Dict[str, CommandSourceConfig]
    _latest: Dict[str, Tuple[Command, int]]
    _last: Optional[Selection]

    def __init__(self, sources: Sequence[CommandSourceConfig]) -> None:
        self._sources = {source.name: source for source in sources}
        self._latest = {}
        self._last = None

    def observe(self, source_name: str, command: Command, now_us: int) -> None:
        """Record *command* from *source_name*, arrived at *now_us*.

        Raises:
            KeyError: If *source_name* is not a configured source.
        """
        if source_name not in self._sources:
            raise KeyError(f"unknown command source {source_name!r}")
        self._latest[source_name] = (command, now_us)

    def select(self, now_us: int) -> Optional[Selection]:
        """Return the command that drives at *now_us*, or ``None`` before any."""
        best: Optional[Selection] = None
        for name, (command, received_us) in self._latest.items():
            source = self._sources[name]
            if now_us - received_us > source.stale_us:
                continue
            if (
                best is None
                or source.priority > best.source.priority
                or (
                    source.priority == best.source.priority
                    and received_us > best.received_us
                )
            ):
                best = Selection(source, command, received_us, fresh=True)
        if best is not None:
            self._last = best
            return best
        if self._last is None:
            return None
        return dataclasses.replace(self._last, fresh=False)
