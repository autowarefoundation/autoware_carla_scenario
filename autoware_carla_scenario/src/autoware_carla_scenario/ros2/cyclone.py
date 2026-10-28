"""The ROS 2 ego transport on Cyclone DDS's Python binding.

Joins :attr:`~.config.Ros2EgoConfig.domain_id` as a plain DDS participant and
exchanges samples with ROS 2 nodes on ``rmw_cyclonedds_cpp`` through the types in
:mod:`.messages`.  Cyclone reads ``CYCLONEDDS_URI`` as a ROS 2 node does, so the
same configuration file (interfaces, peers, fragment size) applies to both.

Imports ``cyclonedds`` (the ``ros2`` extra).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.internal import InvalidSample
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from ..constants import US_PER_S
from ..driver.geometry import Pose
from . import messages as msg
from .commands import (
    Command,
    ControlCommand,
    EgoState,
    SteerAccelPairer,
    TrajectoryCommand,
    TrajectoryCommandPoint,
)
from .config import CommandSourceConfig, Ros2EgoConfig
from .transport import CameraFrame, Received, Ros2EgoTransport

logger = logging.getLogger(__name__)

__all__ = [
    "CycloneRos2Transport",
    "control_from_message",
    "odometry_message",
    "ros_time",
    "trajectory_from_message",
]

#: Samples a ``steer_accel`` reader keeps between polls: a little depth keeps
#: a burst (two cycles between two ticks) from being lost before the pairing
#: sees it; the pairing itself holds at most one unmatched sample.  The other kinds keep
#: one, the newest, which is all the arbiter uses.
_PAIRED_DEPTH: int = 4


def _take(reader: DataReader, depth: int) -> List[Any]:
    """The reader's new samples, without the ones that carry no data.

    A writer that goes away -- a node shut down, a process killed -- leaves an
    instance-state change behind, which Cyclone hands over as an
    ``InvalidSample``.
    """
    return [
        sample for sample in reader.take(depth) if not isinstance(sample, InvalidSample)
    ]


def _source_ns(sample: Any) -> int:
    """When the sample was written, in the writer's clock (ns)."""
    return int(sample.sample_info.source_timestamp)


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


def _reader(
    participant: DomainParticipant, name: str, message_type: Any, depth: int
) -> DataReader:
    return DataReader(
        participant,
        Topic(participant, msg.dds_topic(name), message_type),
        qos=_qos(depth),
    )


def ros_time(microseconds: int) -> msg.Time:
    """Return *microseconds* as a ``builtin_interfaces/msg/Time``."""
    sec, remainder = divmod(int(microseconds), US_PER_S)
    return msg.Time(sec=sec, nanosec=remainder * 1000)


def _microseconds(time: Any) -> int:
    return int(time.sec) * US_PER_S + int(time.nanosec) // 1000


def odometry_message(
    state: EgoState, stamp: msg.Time, frame_id: str, child_frame_id: str
) -> msg.Odometry:
    """Return the ``nav_msgs/msg/Odometry`` for *state*, stamped *stamp*."""
    x, y, z = (float(v) for v in state.pose.position)
    qx, qy, qz, qw = (float(v) for v in state.pose.quat_xyzw)
    lx, ly, lz = (float(v) for v in state.linear_velocity)
    ax, ay, az = (float(v) for v in state.angular_velocity)
    return msg.Odometry(
        header=msg.Header(stamp=stamp, frame_id=frame_id),
        child_frame_id=child_frame_id,
        pose=msg.PoseWithCovariance(
            pose=msg.Pose(
                position=msg.Point(x=x, y=y, z=z),
                orientation=msg.Quaternion(x=qx, y=qy, z=qz, w=qw),
            )
        ),
        twist=msg.TwistWithCovariance(
            twist=msg.Twist(
                linear=msg.Vector3(x=lx, y=ly, z=lz),
                angular=msg.Vector3(x=ax, y=ay, z=az),
            )
        ),
    )


def _pose(pose: msg.Pose) -> Pose:
    p, q = pose.position, pose.orientation
    return Pose(np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))


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
            time_from_start_us=_microseconds(point.time_from_start),
            pose=_pose(point.pose),
            velocity_mps=float(point.longitudinal_velocity_mps),
            acceleration_mps2=float(point.acceleration_mps2),
        )
        for point in cast(List[msg.TrajectoryPoint], message.points)
    )
    return TrajectoryCommand(
        stamp_us=_microseconds(message.header.stamp), points=points
    )


#: The single-topic command kinds: their message type and its conversion.
_SINGLE_TOPIC_KINDS: Dict[str, Tuple[Any, Callable[[Any], Command]]] = {
    "trajectory": (msg.Trajectory, trajectory_from_message),
    "control": (msg.Control, control_from_message),
}

#: Reads one source's new samples into commands.
_Drain = Callable[[], List[Command]]


def _source_drain(
    participant: DomainParticipant, source: CommandSourceConfig
) -> _Drain:
    if source.kind in _SINGLE_TOPIC_KINDS:
        message_type, convert = _SINGLE_TOPIC_KINDS[source.kind]
        reader = _reader(participant, source.topic, message_type, 1)

        def drain_newest() -> List[Command]:
            return [convert(sample) for sample in _take(reader, 1)]

        return drain_newest

    pairer = SteerAccelPairer(int(round(source.pair_window_s * 1_000_000_000)))
    steering = _reader(participant, source.steering_topic, msg.Float64, _PAIRED_DEPTH)
    acceleration = _reader(
        participant, source.acceleration_topic, msg.Float64, _PAIRED_DEPTH
    )

    def drain_pairs() -> List[Command]:
        # Both topics' samples in the order they were written, so a poll that
        # holds two cycles, or half of one, still pairs each cycle with itself
        # (#688).  Steering first on a tie.
        written: List[Tuple[int, int, float]] = [
            (_source_ns(sample), 0, float(sample.data))
            for sample in _take(steering, _PAIRED_DEPTH)
        ] + [
            (_source_ns(sample), 1, float(sample.data))
            for sample in _take(acceleration, _PAIRED_DEPTH)
        ]
        written.sort()
        commands: List[Command] = []
        for stamp_ns, topic, value in written:
            paired = (
                pairer.steering(value, stamp_ns)
                if topic == 0
                else pairer.acceleration(value, stamp_ns)
            )
            if paired is not None:
                commands.append(paired)
        return commands

    return drain_pairs


#: Builds one state topic's sample from the ego's state and its stamp.
_Build = Callable[[EgoState, msg.Time], Any]


class CycloneRos2Transport(Ros2EgoTransport):
    """Publishes the ego's state and reads its commands over Cyclone DDS.

    Args:
        config: Topics, domain and command sources.
    """

    _participant: Optional[DomainParticipant]
    _publishers: List[Tuple[DataWriter, _Build]]
    _image_writers: List[DataWriter]
    _drains: List[Tuple[str, _Drain]]
    _turn_indicators: Optional[DataReader]

    def __init__(self, config: Ros2EgoConfig) -> None:
        super().__init__(config)
        self._participant = None
        self._publishers = []
        self._image_writers = []
        self._drains = []
        self._turn_indicators = None

    def open(self) -> None:
        if self._participant is not None:
            return
        config = self._config
        participant = DomainParticipant(config.domain_id)
        self._participant = participant
        state_topics: Tuple[Tuple[Optional[str], Any, _Build], ...] = (
            (
                config.kinematic_state_topic,
                msg.Odometry,
                lambda state, stamp: odometry_message(
                    state, stamp, config.frame_id, config.child_frame_id
                ),
            ),
            (
                config.steering_status_topic,
                msg.SteeringReport,
                lambda state, stamp: msg.SteeringReport(
                    stamp=stamp, steering_tire_angle=state.steering_tire_angle_rad
                ),
            ),
            (
                config.speed_topic,
                msg.Float64,
                lambda state, stamp: msg.Float64(data=state.speed_mps),
            ),
            (
                config.clock_topic,
                msg.Clock,
                lambda state, stamp: msg.Clock(clock=stamp),
            ),
            (
                config.turn_indicators_report_topic,
                msg.TurnIndicatorsReport,
                lambda state, stamp: msg.TurnIndicatorsReport(
                    stamp=stamp, report=state.turn_indicators
                ),
            ),
        )
        published = [(name, t, build) for name, t, build in state_topics if name]
        self._publishers = [
            (
                DataWriter(
                    participant,
                    Topic(participant, msg.dds_topic(name), message_type),
                    qos=_qos(1),
                ),
                build,
            )
            for name, message_type, build in published
        ]
        # Reliable, as every other topic here: it matches a best-effort
        # subscriber (ROS 2's sensor-data profile) as well as a reliable one.
        self._image_writers = [
            DataWriter(
                participant,
                Topic(participant, msg.dds_topic(camera.topic), msg.Image),
                qos=_qos(1),
            )
            for camera in config.cameras
        ]
        self._drains = [
            (source.name, _source_drain(participant, source))
            for source in config.sources
        ]
        if config.turn_indicators_command_topic:
            self._turn_indicators = _reader(
                participant,
                config.turn_indicators_command_topic,
                msg.TurnIndicatorsCommand,
                1,
            )
        logger.info(
            "Joined DDS domain %d: publishing %s; commands from %s%s",
            config.domain_id,
            ", ".join(
                [name for name, _, _ in published]
                + [camera.topic for camera in config.cameras]
            )
            or "nothing",
            ", ".join(
                f"{source.name} ({source.kind}: {', '.join(source.topics)})"
                for source in config.sources
            ),
            (
                f"; turn indicators from {config.turn_indicators_command_topic}"
                if config.turn_indicators_command_topic
                else ""
            ),
        )

    def publish_state(self, state: EgoState) -> None:
        stamp = ros_time(state.stamp_us)
        for writer, build in self._publishers:
            writer.write(build(state, stamp))

    def publish_image(self, frame: CameraFrame) -> None:
        camera = self._config.cameras[frame.camera]
        self._image_writers[frame.camera].write(
            msg.Image(
                header=msg.Header(
                    stamp=ros_time(frame.stamp_us), frame_id=camera.header_frame_id
                ),
                height=frame.height,
                width=frame.width,
                encoding="bgra8",
                step=frame.width * 4,
                # Image.serialize writes bytes as one block.
                data=cast(Any, frame.bgra),
            )
        )

    def poll(self) -> List[Received]:
        return [(name, command) for name, drain in self._drains for command in drain()]

    def poll_turn_indicators(self) -> Optional[int]:
        if self._turn_indicators is None:
            return None
        samples = _take(self._turn_indicators, 1)
        return int(samples[-1].command) if samples else None

    def close(self) -> None:
        # Dropping the participant deletes every entity it owns.
        self._drains = []
        self._turn_indicators = None
        self._publishers = []
        self._image_writers = []
        self._participant = None
