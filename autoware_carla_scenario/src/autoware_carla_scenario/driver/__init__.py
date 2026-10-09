"""driver -- connect the ego vehicle to an external driving policy.

The scenario framework plays the *runtime* role of the alpasim ``egodriver`` contract:
it owns the CARLA world, renders observations, and asks a policy what to do.  The policy
runs as a separate gRPC server, typically built on ``carla-driver-interface`` (the
policy-side package in this workspace)::

    carla-driver-interface serve --policy route_follower --port 50051

Usage from a scenario::

    from autoware_carla_scenario import CarlaDriverEntity, DriverClientConfig

    scenario.ego_entity = CarlaDriverEntity(
        DriverClientConfig(address="localhost:50051")
    )

See ``autoware_carla_scenario/docs/driver_interface.md`` for the full picture.
"""

from .base import (
    BaseEgoDriverClient,
    DriveOutcome,
    DriverCameraConfig,
    DriverClientConfig,
    DriverLidarConfig,
    EgoObservation,
)
from .control import ControlConfig, TrajectoryFollower, VehicleCommand
from .egodriver_client import EgoDriverGrpcClient
from .geometry import Pose, Trajectory

__all__ = [
    "BaseEgoDriverClient",
    "ControlConfig",
    "DriveOutcome",
    "DriverCameraConfig",
    "DriverClientConfig",
    "DriverLidarConfig",
    "EgoDriverGrpcClient",
    "EgoObservation",
    "Pose",
    "TrajectoryFollower",
    "Trajectory",
    "VehicleCommand",
]
