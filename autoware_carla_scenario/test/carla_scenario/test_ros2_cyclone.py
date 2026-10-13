"""The Cyclone DDS transport against a second DDS participant, in one process.

The peer publishes and reads with the same ROS 2 message types a node on
``rmw_cyclonedds_cpp`` uses, on the ``rt/`` topics such a node maps its names to,
so this exercises the wire path short of an actual ROS 2 node.  Skipped where
the ``ros2`` extra (``cyclonedds``) is not installed.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, TypeVar

import numpy as np
import pytest

cyclonedds = pytest.importorskip("cyclonedds")

from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.pub import DataWriter  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402

from autoware_carla_scenario.driver.geometry import Pose  # noqa: E402
from autoware_carla_scenario.ros2 import (  # noqa: E402
    CommandSourceConfig,
    ControlCommand,
    EgoState,
    Ros2EgoConfig,
    SteerAccelCommand,
    TrajectoryCommand,
)
from autoware_carla_scenario.ros2 import messages as msg  # noqa: E402
from autoware_carla_scenario.ros2.commands import Command  # noqa: E402
from autoware_carla_scenario.ros2.cyclone import (  # noqa: E402
    CycloneRos2Transport,
    _qos,
    ros_time,
)

T = TypeVar("T")

#: How long discovery and delivery may take on loopback.
_TIMEOUT_S = 10.0


def _domain() -> int:
    # One domain per worker process, so parallel runs never see each other.
    return 100 + os.getpid() % 100


def _until(produce: Callable[[], List[T]], count: int = 1) -> List[T]:
    deadline = time.monotonic() + _TIMEOUT_S
    collected: List[T] = []
    while time.monotonic() < deadline:
        collected.extend(produce())
        if len(collected) >= count:
            return collected
        time.sleep(0.02)
    return collected


@pytest.fixture
def loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unicast on loopback: a CI runner's network may carry no multicast.
    monkeypatch.setenv(
        "CYCLONEDDS_URI",
        "<CycloneDDS><Domain id='any'><General><Interfaces>"
        "<NetworkInterface address='127.0.0.1'/></Interfaces>"
        "<AllowMulticast>false</AllowMulticast></General>"
        "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
        "<Peers><Peer address='127.0.0.1'/></Peers></Discovery>"
        "</Domain></CycloneDDS>",
    )


def _config(domain: int) -> Ros2EgoConfig:
    return Ros2EgoConfig(
        domain_id=domain,
        clock_topic="/clock",
        sources=(
            CommandSourceConfig(
                name="vp",
                kind="steer_accel",
                steering_topic="/vehicle/steering_cmd",
                acceleration_topic="/vehicle/throttle_cmd",
            ),
            CommandSourceConfig(
                name="si", kind="control", topic="/control/command/control_cmd"
            ),
            CommandSourceConfig(
                name="planner", kind="trajectory", topic="/planning/trajectory"
            ),
        ),
    )


def _state(stamp_us: int) -> EgoState:
    return EgoState(
        stamp_us=stamp_us,
        pose=Pose.from_xyz_yaw(1.0, 2.0, 0.0, 0.5),
        linear_velocity=np.array([3.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.1]),
        speed_mps=3.0,
        steering_tire_angle_rad=0.05,
    )


def test_ros_time_splits_microseconds() -> None:
    stamp = ros_time(12_250_000)
    assert (stamp.sec, stamp.nanosec) == (12, 250_000_000)
    stamp = ros_time(999_999)
    assert (stamp.sec, stamp.nanosec) == (0, 999_999_000)


def test_dds_topic_names_follow_rmw_cyclonedds() -> None:
    assert msg.dds_topic("/vehicle/speed") == "rt/vehicle/speed"


def test_state_reaches_a_ros_2_style_reader(loopback: None) -> None:
    domain = _domain()
    transport = CycloneRos2Transport(_config(domain))
    transport.open()
    try:
        peer = DomainParticipant(domain)
        odometry = DataReader(
            peer,
            Topic(peer, "rt/localization/kinematic_state", msg.Odometry),
            qos=_qos(4),
        )
        speed = DataReader(
            peer, Topic(peer, "rt/vehicle/speed", msg.Float64), qos=_qos(4)
        )
        clock = DataReader(peer, Topic(peer, "rt/clock", msg.Clock), qos=_qos(4))
        steering = DataReader(
            peer,
            Topic(peer, "rt/vehicle/status/steering_status", msg.SteeringReport),
            qos=_qos(4),
        )

        def publish_and_take() -> List[msg.Odometry]:
            transport.publish_state(_state(12_500_000))
            return list(odometry.take(4))

        (sample, *_) = _until(publish_and_take)
        assert (sample.header.stamp.sec, sample.header.stamp.nanosec) == (
            12,
            500_000_000,
        )
        assert sample.header.frame_id == "map"
        assert sample.child_frame_id == "base_link"
        assert sample.pose.pose.position.x == pytest.approx(1.0)
        assert sample.pose.pose.position.y == pytest.approx(2.0)
        assert sample.twist.twist.linear.x == pytest.approx(3.0)
        assert sample.twist.twist.angular.z == pytest.approx(0.1)

        assert _until(lambda: list(speed.take(4)))[0].data == pytest.approx(3.0)
        assert _until(lambda: list(clock.take(4)))[0].clock.sec == 12
        assert _until(lambda: list(steering.take(4)))[0].steering_tire_angle == (
            pytest.approx(0.05)
        )
    finally:
        transport.close()


def test_commands_from_ros_2_style_writers_are_taken(loopback: None) -> None:
    domain = _domain()
    transport = CycloneRos2Transport(_config(domain))
    transport.open()
    try:
        peer = DomainParticipant(domain)
        steer = DataWriter(
            peer, Topic(peer, "rt/vehicle/steering_cmd", msg.Float64), qos=_qos(1)
        )
        accel = DataWriter(
            peer, Topic(peer, "rt/vehicle/throttle_cmd", msg.Float64), qos=_qos(1)
        )
        control = DataWriter(
            peer,
            Topic(peer, "rt/control/command/control_cmd", msg.Control),
            qos=_qos(1),
        )
        trajectory = DataWriter(
            peer, Topic(peer, "rt/planning/trajectory", msg.Trajectory), qos=_qos(1)
        )

        def publish_and_poll():
            steer.write(msg.Float64(data=0.1))
            accel.write(msg.Float64(data=-0.5))
            control.write(
                msg.Control(
                    lateral=msg.Lateral(steering_tire_angle=0.2),
                    longitudinal=msg.Longitudinal(
                        velocity=4.0, acceleration=-3.0, jerk=1.5, is_defined_jerk=True
                    ),
                )
            )
            trajectory.write(
                msg.Trajectory(
                    header=msg.Header(stamp=msg.Time(sec=3, nanosec=0)),
                    points=[
                        msg.TrajectoryPoint(
                            time_from_start=msg.Duration(
                                sec=0, nanosec=i * 100_000_000
                            ),
                            pose=msg.Pose(position=msg.Point(x=float(i))),
                            longitudinal_velocity_mps=2.0,
                        )
                        for i in range(3)
                    ],
                )
            )
            return transport.poll()

        received = _until(publish_and_poll, count=3)
        by_source: Dict[str, Command] = {}
        for name, command in received:
            by_source.setdefault(name, command)
        assert set(by_source) == {"vp", "si", "planner"}

        assert by_source["vp"] == SteerAccelCommand(0.1, -0.5)
        control_command = by_source["si"]
        assert isinstance(control_command, ControlCommand)
        # float32 on the wire.
        assert control_command.steering_tire_angle_rad == pytest.approx(0.2)
        assert (
            control_command.velocity_mps,
            control_command.acceleration_mps2,
            control_command.jerk_mps3,
        ) == (4.0, -3.0, 1.5)
        plan = by_source["planner"]
        assert isinstance(plan, TrajectoryCommand)
        assert plan.stamp_us == 3_000_000
        assert [point.pose.position[0] for point in plan.points] == [0.0, 1.0, 2.0]
        assert [point.time_from_start_us for point in plan.points] == [
            0,
            100_000,
            200_000,
        ]
    finally:
        transport.close()


def test_close_is_idempotent_and_open_twice_joins_once(loopback: None) -> None:
    transport = CycloneRos2Transport(_config(_domain()))
    transport.open()
    transport.open()
    transport.close()
    transport.close()
    assert transport.poll() == []


def test_a_writer_going_away_is_not_a_command(loopback: None) -> None:
    """A node that shuts down leaves a sample with no data, which is skipped."""
    domain = _domain()
    transport = CycloneRos2Transport(_config(domain))
    transport.open()
    try:
        peer = DomainParticipant(domain)
        control = DataWriter(
            peer,
            Topic(peer, "rt/control/command/control_cmd", msg.Control),
            qos=_qos(1),
        )
        sample = msg.Control(longitudinal=msg.Longitudinal(velocity=4.0))

        def publish_and_poll():
            control.write(sample)
            return transport.poll()

        assert _until(publish_and_poll)
        control.dispose(sample)
        control.unregister_instance(sample)
        # Loopback delivers in milliseconds: long enough for the dataless
        # sample to reach the transport and be taken, and skipped.
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            assert transport.poll() == []
            time.sleep(0.02)
        # And the source still works after it.
        assert _until(publish_and_poll)
    finally:
        transport.close()
