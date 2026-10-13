"""The ROS 2 ego transport on Cyclone DDS's Python binding.

Joins :attr:`~.config.Ros2EgoConfig.domain_id` as a plain DDS participant and
exchanges samples with ROS 2 nodes on ``rmw_cyclonedds_cpp`` through the types in
:mod:`.messages`.  Cyclone reads ``CYCLONEDDS_URI`` as a ROS 2 node does, so the
same configuration file (interfaces, peers, fragment size) applies to both.

Imports ``cyclonedds`` (the ``ros2`` extra).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.internal import InvalidSample
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from . import messages as msg
from .commands import (
    Command,
    ControlCommand,
    EgoState,
    SteerAccelPairer,
    TrajectoryCommand,
    TrajectoryCommandPoint,
    yaw_from_quaternion,
)
from .config import CommandSourceConfig, Ros2EgoConfig
from .transport import Received, Ros2EgoTransport

logger = logging.getLogger(__name__)

__all__ = [
    "CycloneRos2Transport",
    "control_from_message",
    "odometry_message",
    "ros_time",
    "trajectory_from_message",
]

#: Samples a reader keeps between polls.  Commands are taken every tick, so
#: only the newest matters; a little depth keeps a burst from being lost
#: before the pairing of a ``steer_accel`` source sees it.
_COMMAND_DEPTH: int = 4


def _take(reader: DataReader) -> List[Any]:
    """The reader's new samples, without the ones that carry no data.

    A writer that goes away -- a node shut down, a process killed -- leaves an
    instance-state change behind, which Cyclone hands over as an
    ``InvalidSample``.
    """
    return [
        sample
        for sample in reader.take(_COMMAND_DEPTH)
        if not isinstance(sample, InvalidSample)
    ]


def _qos(depth: int) -> Qos:
    # ROS 2's default profile (reliable, volatile, keep last), in XCDR1 -- the
    # representation rmw_cyclonedds_cpp writes, and the one a static-peer
    # embedded participant is most likely to read.
    return Qos(
        Policy.Reliability.Reliable(duration(milliseconds=100)),
        Policy.Durability.Volatile,
        Policy.History.KeepLast(depth),
        Policy.DataRepresentation(use_cdrv0_representation=True),
    )


def ros_time(seconds: float) -> msg.Time:
    """Return *seconds* as a ``builtin_interfaces/msg/Time``."""
    sec = math.floor(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return msg.Time(sec=int(sec), nanosec=nanosec)


def _seconds(time: Any) -> float:
    return float(time.sec) + float(time.nanosec) * 1e-9


def odometry_message(
    state: EgoState, frame_id: str, child_frame_id: str
) -> msg.Odometry:
    """Return the ``nav_msgs/msg/Odometry`` for *state*."""
    x, y, z = (float(v) for v in state.pose.position)
    qx, qy, qz, qw = (float(v) for v in state.pose.quat_xyzw)
    linear = state.linear_velocity
    angular = state.angular_velocity
    return msg.Odometry(
        header=msg.Header(stamp=ros_time(state.stamp_s), frame_id=frame_id),
        child_frame_id=child_frame_id,
        pose=msg.PoseWithCovariance(
            pose=msg.Pose(
                position=msg.Point(x=x, y=y, z=z),
                orientation=msg.Quaternion(x=qx, y=qy, z=qz, w=qw),
            )
        ),
        twist=msg.TwistWithCovariance(
            twist=msg.Twist(
                linear=msg.Vector3(
                    x=float(linear[0]), y=float(linear[1]), z=float(linear[2])
                ),
                angular=msg.Vector3(
                    x=float(angular[0]), y=float(angular[1]), z=float(angular[2])
                ),
            )
        ),
    )


def control_from_message(message: msg.Control) -> ControlCommand:
    """Return the command an ``autoware_control_msgs/msg/Control`` carries."""
    longitudinal = message.longitudinal
    return ControlCommand(
        steering_tire_angle_rad=float(message.lateral.steering_tire_angle),
        velocity_mps=float(longitudinal.velocity),
        acceleration_mps2=float(longitudinal.acceleration),
        jerk_mps3=float(longitudinal.jerk) if longitudinal.is_defined_jerk else None,
    )


def trajectory_from_message(message: msg.Trajectory) -> TrajectoryCommand:
    """Return the plan an ``autoware_planning_msgs/msg/Trajectory`` carries."""
    points = tuple(
        TrajectoryCommandPoint(
            time_from_start_s=_seconds(point.time_from_start),
            x=float(point.pose.position.x),
            y=float(point.pose.position.y),
            z=float(point.pose.position.z),
            yaw=yaw_from_quaternion(
                point.pose.orientation.x,
                point.pose.orientation.y,
                point.pose.orientation.z,
                point.pose.orientation.w,
            ),
            velocity_mps=float(point.longitudinal_velocity_mps),
        )
        for point in cast(List[msg.TrajectoryPoint], message.points)
    )
    return TrajectoryCommand(stamp_s=_seconds(message.header.stamp), points=points)


#: Reads one reader's new samples into commands for one source.
_Drain = Callable[[], List[Command]]


class CycloneRos2Transport(Ros2EgoTransport):
    """Publishes the ego's state and reads its commands over Cyclone DDS.

    Args:
        config: Topics, domain and command sources.
    """

    _participant: Optional[DomainParticipant]
    _writers: Dict[str, DataWriter]
    _drains: List[Tuple[str, _Drain]]

    def __init__(self, config: Ros2EgoConfig) -> None:
        super().__init__(config)
        self._participant = None
        self._writers = {}
        self._drains = []

    def open(self) -> None:
        if self._participant is not None:
            return
        config = self._config
        participant = DomainParticipant(config.domain_id)
        self._participant = participant
        state_types = (
            (config.kinematic_state_topic, msg.Odometry),
            (config.steering_status_topic, msg.SteeringReport),
            (config.speed_topic, msg.Float64),
            (config.clock_topic, msg.Clock),
        )
        for name, message_type in state_types:
            if name:
                self._writers[name] = DataWriter(
                    participant,
                    Topic(participant, msg.dds_topic(name), message_type),
                    qos=_qos(1),
                )
        for source in config.sources:
            self._drains.extend(self._source_drains(participant, source))
        logger.info(
            "Joined DDS domain %d: publishing %s; commands from %s",
            config.domain_id,
            ", ".join(sorted(self._writers)) or "nothing",
            ", ".join(
                f"{source.name} ({source.kind}: {', '.join(source.topics)})"
                for source in config.sources
            ),
        )

    def _reader(
        self, participant: DomainParticipant, name: str, message_type: Any
    ) -> DataReader:
        return DataReader(
            participant,
            Topic(participant, msg.dds_topic(name), message_type),
            qos=_qos(_COMMAND_DEPTH),
        )

    def _source_drains(
        self, participant: DomainParticipant, source: CommandSourceConfig
    ) -> List[Tuple[str, _Drain]]:
        if source.kind == "trajectory":
            reader = self._reader(participant, source.topic, msg.Trajectory)
            return [
                (
                    source.name,
                    lambda: [trajectory_from_message(s) for s in _take(reader)],
                )
            ]
        if source.kind == "control":
            reader = self._reader(participant, source.topic, msg.Control)
            return [
                (
                    source.name,
                    lambda: [control_from_message(s) for s in _take(reader)],
                )
            ]
        steering = self._reader(participant, source.steering_topic, msg.Float64)
        acceleration = self._reader(participant, source.acceleration_topic, msg.Float64)
        pairer = SteerAccelPairer()

        def drain() -> List[Command]:
            commands: List[Command] = []
            for sample in _take(steering):
                paired = pairer.steering(float(sample.data))
                if paired is not None:
                    commands.append(paired)
            for sample in _take(acceleration):
                paired = pairer.acceleration(float(sample.data))
                if paired is not None:
                    commands.append(paired)
            return commands

        return [(source.name, drain)]

    def publish_state(self, state: EgoState) -> None:
        config = self._config
        stamp = ros_time(state.stamp_s)
        if config.kinematic_state_topic:
            self._writers[config.kinematic_state_topic].write(
                odometry_message(state, config.frame_id, config.child_frame_id)
            )
        if config.steering_status_topic:
            self._writers[config.steering_status_topic].write(
                msg.SteeringReport(
                    stamp=stamp, steering_tire_angle=state.steering_tire_angle_rad
                )
            )
        if config.speed_topic:
            self._writers[config.speed_topic].write(msg.Float64(data=state.speed_mps))
        if config.clock_topic:
            self._writers[config.clock_topic].write(msg.Clock(clock=stamp))

    def poll(self) -> List[Received]:
        received: List[Received] = []
        for name, drain in self._drains:
            for command in drain():
                received.append((name, command))
        return received

    def close(self) -> None:
        # Dropping the participant deletes every entity it owns.
        self._drains = []
        self._writers = {}
        self._participant = None
