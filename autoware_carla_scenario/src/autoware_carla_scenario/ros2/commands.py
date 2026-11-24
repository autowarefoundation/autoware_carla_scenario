"""What crosses between the ego entity and its DDS transport, and who drives.

The transport turns DDS samples into the plain commands here and the ego's state
into DDS samples, so the entity never touches a DDS type.  :class:`CommandArbiter`
picks, each tick, the command source that drives.

Time is in integer microseconds of simulation time throughout, the clock the
trajectory follower and the ego observation already keep.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    "command_is_finite",
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


def command_is_finite(command: Command) -> bool:
    """Whether every number of *command* is finite.

    A steering angle, a speed, an acceleration, a jerk (when the command sets
    one), and each of a trajectory's poses, speeds and accelerations: a NaN or
    an infinity in any of them makes the command one no controller can apply.
    """
    # Each branch names the command as its class: Codon 0.19 reads no
    # attribute of a union (docs/typecheck.md).
    if isinstance(command, TrajectoryCommand):
        trajectory: TrajectoryCommand = command
        for point in trajectory.points:
            if not (
                math.isfinite(point.velocity_mps)
                and math.isfinite(point.acceleration_mps2)
            ):
                return False
            pose = point.pose
            for index in range(3):
                if not math.isfinite(float(pose.position[index])):
                    return False
            for index in range(4):
                if not math.isfinite(float(pose.quat_xyzw[index])):
                    return False
        return True
    if isinstance(command, SteerAccelCommand):
        steer_accel: SteerAccelCommand = command
        return math.isfinite(steer_accel.steering_rad) and math.isfinite(
            steer_accel.acceleration_mps2
        )
    control: ControlCommand = command
    jerk = control.jerk_mps3
    if jerk is not None:
        jerk_mps3: float = jerk
        if not math.isfinite(jerk_mps3):
            return False
    return (
        math.isfinite(control.steering_tire_angle_rad)
        and math.isfinite(control.velocity_mps)
        and math.isfinite(control.acceleration_mps2)
    )


#: How many recent write intervals :class:`SteerAccelPairer` estimates the
#: publisher's cycle from.  It takes their median, so a lost message (one
#: interval of two cycles) or a stall does not move the estimate.
_CYCLE_INTERVALS: int = 8

#: :meth:`SteerAccelPairer._match`'s verdicts.
_PAIR: int = 0
_DROP_NEW: int = 1
_DROP_PENDING: int = 2


class SteerAccelPairer:
    """Pairs a steering topic with an acceleration topic into one command.

    VisionPilot emits one of each per planning cycle, back to back, so this
    yields one command per cycle, exactly as its ``carla_control_publisher``
    does.  The two halves of a cycle are matched by their source timestamps:
    the caller feeds every sample of both topics in the order they were written
    (by ``source_timestamp``), and a half pairs with the pending half of the
    other topic only if the two were written at most :attr:`window_ns` apart.

    By default the window is half the publisher's cycle, measured from the
    samples' own timestamps (the median of the last few write intervals on
    either topic), so it assumes no particular rate or timing: the two halves
    of a cycle pair as long as they are written closer to each other than to
    the neighbouring cycles' halves, and a half whose partner was lost is too
    far from the next cycle's to be mistaken for it.  Until two samples of one
    topic have given an interval, any two halves pair.

    At most one half is pending, so a silent topic cannot grow memory, and an
    unmatched half is dropped rather than paired with a later cycle's, so a
    lost message costs that one cycle and never shifts the pairing:

    * a second half on the same topic supersedes the pending one, whose partner
      was lost (a cycle has one of each);
    * a half written more than the window after the pending half of the other
      topic drops that one, which is then stale, and waits for its own partner;
    * a half written more than the window before the pending half of the other
      topic (delivered late, across polls) is dropped itself.

    Each value is used once.  :attr:`dropped` counts the halves dropped, for
    the caller to report.

    Args:
        window_ns: A fixed window, in source-clock nanoseconds, in place of
            half the measured cycle.  ``None`` measures it.
    """

    _fixed_window_ns: Optional[int]
    _steering: Optional[Tuple[float, int]]
    _acceleration: Optional[Tuple[float, int]]
    _last_steering_ns: Optional[int]
    _last_acceleration_ns: Optional[int]
    _intervals: List[int]
    _dropped: int

    def __init__(self, window_ns: Optional[int] = None) -> None:
        if window_ns is not None and window_ns < 0:
            raise ValueError(f"window_ns must not be negative, got {window_ns}")
        self._fixed_window_ns = window_ns
        self._steering = None
        self._acceleration = None
        self._last_steering_ns = None
        self._last_acceleration_ns = None
        self._intervals = []
        self._dropped = 0

    @property
    def window_ns(self) -> Optional[int]:
        """How far apart two halves may be written to pair, in nanoseconds;
        ``None`` while the cycle has not been measured yet (any two pair)."""
        if self._fixed_window_ns is not None:
            return self._fixed_window_ns
        if not self._intervals:
            return None
        ordered = sorted(self._intervals)
        return ordered[len(ordered) // 2] // 2

    @property
    def dropped(self) -> int:
        """How many halves have been dropped unpaired so far."""
        return self._dropped

    def steering(self, value: float, stamp_ns: int) -> Optional[SteerAccelCommand]:
        """Take a steering sample written at *stamp_ns*; return the command it
        completes, if any."""
        self._last_steering_ns = self._observe(self._last_steering_ns, stamp_ns)
        partner = self._acceleration
        if partner is not None:
            verdict = self._match(stamp_ns, partner[1])
            if verdict == _PAIR:
                self._acceleration = None
                return SteerAccelCommand(value, partner[0])
            if verdict == _DROP_NEW:
                return None
            self._acceleration = None
        if self._steering is not None:
            self._dropped += 1
        self._steering = (value, stamp_ns)
        return None

    def acceleration(self, value: float, stamp_ns: int) -> Optional[SteerAccelCommand]:
        """Take an acceleration sample written at *stamp_ns*; return the command
        it completes, if any."""
        self._last_acceleration_ns = self._observe(self._last_acceleration_ns, stamp_ns)
        partner = self._steering
        if partner is not None:
            verdict = self._match(stamp_ns, partner[1])
            if verdict == _PAIR:
                self._steering = None
                return SteerAccelCommand(partner[0], value)
            if verdict == _DROP_NEW:
                return None
            self._steering = None
        if self._acceleration is not None:
            self._dropped += 1
        self._acceleration = (value, stamp_ns)
        return None

    def _observe(self, last_ns: Optional[int], stamp_ns: int) -> int:
        """Record the interval since *last_ns*, the topic's newest stamp so
        far; return its newest stamp now."""
        if last_ns is None:
            return stamp_ns
        if stamp_ns > last_ns:
            self._intervals.append(stamp_ns - last_ns)
            if len(self._intervals) > _CYCLE_INTERVALS:
                self._intervals.pop(0)
            return stamp_ns
        return last_ns

    def _match(self, stamp_ns: int, pending_ns: int) -> int:
        """Whether a half written at *stamp_ns* pairs with the other topic's
        pending half, written at *pending_ns*, and if not which one goes."""
        window = self.window_ns
        if window is None or abs(stamp_ns - pending_ns) <= window:
            return _PAIR
        self._dropped += 1
        return _DROP_NEW if stamp_ns < pending_ns else _DROP_PENDING


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
    "Most recent" is the command read last, in the order :meth:`observe` saw
    them, not their arrival time: every command polled on one tick shares that
    tick's time, so the time alone cannot tell which came later.  The
    transport drains the sources in their configured order, so between
    equal priorities that each sent a command within one tick, the source
    configured last wins.  When every source is stale, :meth:`select` returns the command that drove
    last, marked not fresh, so the caller can hold it or brake.

    A command that is not all finite (:func:`command_is_finite`) is refused by
    :meth:`observe`: it neither drives nor keeps its source fresh, so a source
    that sends nothing else goes stale after its ``stale_s``, and until then its
    last good command is the one that drives.

    Args:
        sources: The configured sources.
    """

    _sources: Dict[str, CommandSourceConfig]
    _latest: Dict[str, Tuple[Command, int, int]]
    _last: Optional[Selection]
    _sequence: int

    def __init__(self, sources: Sequence[CommandSourceConfig]) -> None:
        self._sources = {source.name: source for source in sources}
        # Per source: its last command, when it arrived, and its sequence
        # number, which orders the commands as observe() saw them.
        self._latest = {}
        self._last = None
        self._sequence = 0

    def observe(self, source_name: str, command: Command, now_us: int) -> bool:
        """Record *command* from *source_name*, arrived at *now_us*.

        Returns:
            Whether it was recorded: ``False`` for a command that is not all
            finite, which is refused and leaves the source as it was.

        Raises:
            KeyError: If *source_name* is not a configured source.
        """
        if source_name not in self._sources:
            raise KeyError(f"unknown command source {source_name!r}")
        if not command_is_finite(command):
            return False
        self._sequence += 1
        self._latest[source_name] = (command, now_us, self._sequence)
        return True

    def select(self, now_us: int) -> Optional[Selection]:
        """Return the command that drives at *now_us*, or ``None`` before any."""
        best: Optional[Selection] = None
        best_sequence = 0
        for name, (command, received_us, sequence) in self._latest.items():
            source = self._sources[name]
            if now_us - received_us > source.stale_us:
                continue
            if (
                best is None
                or source.priority > best.source.priority
                or (
                    source.priority == best.source.priority and sequence > best_sequence
                )
            ):
                best = Selection(source, command, received_us, fresh=True)
                best_sequence = sequence
        if best is not None:
            self._last = best
            return best
        if self._last is None:
            return None
        return dataclasses.replace(self._last, fresh=False)
