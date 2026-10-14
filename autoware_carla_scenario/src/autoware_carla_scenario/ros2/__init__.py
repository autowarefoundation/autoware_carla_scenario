"""ros2 -- drive the ego from ROS 2 nodes, over DDS, with no bridge in between.

The framework joins the DDS domain itself (Cyclone DDS's Python binding, the
``ros2`` extra): it publishes the ego's state on ROS 2 topics and takes commands
from them -- a planned trajectory, or a steering/acceleration pair, or an
Autoware control command -- arbitrating between several sources by priority and
freshness.  No ``rclpy``, no ROS 2 installation, no gRPC.

Importing this package does not import ``cyclonedds``: only :mod:`.cyclone` and
:mod:`.messages` do.

Usage::

    from autoware_carla_scenario.entity.ros2_entity import Ros2EgoEntity
    from autoware_carla_scenario.ros2 import Ros2EgoConfig

    scenario.ego_entity = Ros2EgoEntity(Ros2EgoConfig(domain_id=1))

See ``autoware_carla_scenario/docs/ros2.md``.
"""

from .commands import (
    CommandArbiter,
    ControlCommand,
    EgoState,
    SteerAccelCommand,
    TrajectoryCommand,
    TrajectoryCommandPoint,
)
from .config import CommandSourceConfig, Ros2CameraConfig, Ros2EgoConfig
from .transport import CameraFrame, FakeRos2Transport, Ros2EgoTransport

__all__ = [
    "CameraFrame",
    "CommandArbiter",
    "CommandSourceConfig",
    "ControlCommand",
    "EgoState",
    "FakeRos2Transport",
    "Ros2CameraConfig",
    "Ros2EgoConfig",
    "Ros2EgoTransport",
    "SteerAccelCommand",
    "TrajectoryCommand",
    "TrajectoryCommandPoint",
]
