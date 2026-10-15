"""ROS 2 messages as plain DDS types, for Cyclone DDS's Python binding.

A ROS 2 node on ``rmw_cyclonedds_cpp`` publishes a message as a DDS sample of a
type named ``<package>::msg::dds_::<Message>_`` on the topic ``rt/<name>``.
Declaring the same structs here lets this package exchange samples with such a
node directly, with no ROS 2 installation and no ``rclpy``.  Field order and
types follow the ``.msg`` definitions exactly; that is what keeps the samples
wire compatible.

Imports ``cyclonedds`` (the ``ros2`` extra).

No ``from __future__ import annotations`` here: Cyclone reads each field's type
from its annotation, and resolves a string annotation as a name in this module,
which a builtin (``str``, ``bool``) is not.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, cast

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import (
    array,
    float32,
    float64,
    int32,
    sequence,
    uint8,
    uint32,
)

__all__ = [
    "Clock",
    "Control",
    "Duration",
    "Float64",
    "Header",
    "Image",
    "Lateral",
    "Longitudinal",
    "Odometry",
    "Point",
    "Pose",
    "PoseWithCovariance",
    "Quaternion",
    "SteeringReport",
    "Time",
    "Trajectory",
    "TrajectoryPoint",
    "TurnIndicatorsCommand",
    "TurnIndicatorsReport",
    "Twist",
    "TwistWithCovariance",
    "Vector3",
    "dds_topic",
]


def dds_topic(ros_name: str) -> str:
    """Return the DDS topic a ROS 2 topic name maps to (``/a/b`` -> ``rt/a/b``)."""
    return "rt/" + ros_name.lstrip("/")


def _zeros36() -> List[float]:
    return [0.0] * 36


# builtin_interfaces -----------------------------------------------------------


@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Duration(IdlStruct, typename="builtin_interfaces::msg::dds_::Duration_"):
    sec: int32 = 0
    nanosec: uint32 = 0


# std_msgs / rosgraph_msgs ---------------------------------------------------


@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class Float64(IdlStruct, typename="std_msgs::msg::dds_::Float64_"):
    data: float64 = 0.0


@dataclass
class Clock(IdlStruct, typename="rosgraph_msgs::msg::dds_::Clock_"):
    clock: Time = field(default_factory=Time)


# sensor_msgs ------------------------------------------------------------------


@dataclass
class _ImageWire(IdlStruct, typename="autoware_carla_scenario::ImageWire_"):
    """:class:`Image` with its pixels as one ``bytes`` block, for serializing.

    Its CDR is byte for byte :class:`Image`'s.  Cyclone builds no type object for
    a ``bytes`` member, so it cannot name a topic's type; it only serializes.
    """

    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    encoding: str = ""
    is_bigendian: uint8 = 0
    step: uint32 = 0
    data: bytes = b""


@dataclass
class Image(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
    """``sensor_msgs/msg/Image``.  ``data`` may be ``bytes``; it is written as is."""

    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    encoding: str = ""
    is_bigendian: uint8 = 0
    step: uint32 = 0
    data: sequence[uint8] = field(default_factory=list)

    def serialize(
        self,
        buffer: Optional[Any] = None,
        endianness: Optional[Any] = None,
        use_version_2: Optional[bool] = None,
    ) -> bytes:
        # Cyclone writes a sequence[uint8] element by element: some 120 ms for a
        # 1280x720 BGRA frame, more than a 10 Hz camera's period.  The same
        # struct with a bytes member goes as one block, in about 2 ms.
        wire: Any = _ImageWire(
            header=self.header,
            height=self.height,
            width=self.width,
            encoding=self.encoding,
            is_bigendian=self.is_bigendian,
            step=self.step,
            data=bytes(cast(Any, self.data)),
        )
        return wire.serialize(
            buffer=buffer, endianness=endianness, use_version_2=use_version_2
        )


# geometry_msgs / nav_msgs ---------------------------------------------------


@dataclass
class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseWithCovariance(
    IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"
):
    pose: Pose = field(default_factory=Pose)
    covariance: array[float64, 36] = field(default_factory=_zeros36)


@dataclass
class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistWithCovariance(
    IdlStruct, typename="geometry_msgs::msg::dds_::TwistWithCovariance_"
):
    twist: Twist = field(default_factory=Twist)
    covariance: array[float64, 36] = field(default_factory=_zeros36)


@dataclass
class Odometry(IdlStruct, typename="nav_msgs::msg::dds_::Odometry_"):
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)


# autoware_vehicle_msgs / autoware_control_msgs / autoware_planning_msgs -----


@dataclass
class SteeringReport(
    IdlStruct, typename="autoware_vehicle_msgs::msg::dds_::SteeringReport_"
):
    stamp: Time = field(default_factory=Time)
    steering_tire_angle: float32 = 0.0


@dataclass
class TurnIndicatorsCommand(
    IdlStruct, typename="autoware_vehicle_msgs::msg::dds_::TurnIndicatorsCommand_"
):
    stamp: Time = field(default_factory=Time)
    command: uint8 = 0


@dataclass
class TurnIndicatorsReport(
    IdlStruct, typename="autoware_vehicle_msgs::msg::dds_::TurnIndicatorsReport_"
):
    stamp: Time = field(default_factory=Time)
    report: uint8 = 0


@dataclass
class Lateral(IdlStruct, typename="autoware_control_msgs::msg::dds_::Lateral_"):
    stamp: Time = field(default_factory=Time)
    control_time: Time = field(default_factory=Time)
    steering_tire_angle: float32 = 0.0
    steering_tire_rotation_rate: float32 = 0.0
    is_defined_steering_tire_rotation_rate: bool = False


@dataclass
class Longitudinal(
    IdlStruct, typename="autoware_control_msgs::msg::dds_::Longitudinal_"
):
    stamp: Time = field(default_factory=Time)
    control_time: Time = field(default_factory=Time)
    velocity: float32 = 0.0
    acceleration: float32 = 0.0
    jerk: float32 = 0.0
    is_defined_acceleration: bool = False
    is_defined_jerk: bool = False


@dataclass
class Control(IdlStruct, typename="autoware_control_msgs::msg::dds_::Control_"):
    stamp: Time = field(default_factory=Time)
    control_time: Time = field(default_factory=Time)
    lateral: Lateral = field(default_factory=Lateral)
    longitudinal: Longitudinal = field(default_factory=Longitudinal)


@dataclass
class TrajectoryPoint(
    IdlStruct, typename="autoware_planning_msgs::msg::dds_::TrajectoryPoint_"
):
    time_from_start: Duration = field(default_factory=Duration)
    pose: Pose = field(default_factory=Pose)
    longitudinal_velocity_mps: float32 = 0.0
    lateral_velocity_mps: float32 = 0.0
    acceleration_mps2: float32 = 0.0
    heading_rate_rps: float32 = 0.0
    front_wheel_angle_rad: float32 = 0.0
    rear_wheel_angle_rad: float32 = 0.0


@dataclass
class Trajectory(IdlStruct, typename="autoware_planning_msgs::msg::dds_::Trajectory_"):
    header: Header = field(default_factory=Header)
    points: sequence[TrajectoryPoint] = field(default_factory=list)
