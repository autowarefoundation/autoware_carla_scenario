"""driver -- connect the ego vehicle to an external driving policy.

The scenario framework plays the *runtime* role of the alpasim ``egodriver`` contract:
it owns the CARLA world, renders observations, and asks a policy what to do.  The policy
is typically built on ``carla-driver-interface`` (the policy-side package).  It runs
as a separate gRPC server the run dials (:attr:`DriverClientConfig.address`)::

    carla-driver-interface serve --policy route_follower --port 50051

or, with :attr:`DriverClientConfig.policy` set, the run serves it itself, on a local
TCP port or Unix socket, or calls it in-process with no socket at all
(:attr:`DriverClientConfig.transport` ``direct``).

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
    DriverCameraLens,
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
    "DriverCameraLens",
    "DriverClientConfig",
    "DriverLidarConfig",
    "EgoDriverGrpcClient",
    "EgoObservation",
    "Pose",
    "TrajectoryFollower",
    "Trajectory",
    "VehicleCommand",
]
