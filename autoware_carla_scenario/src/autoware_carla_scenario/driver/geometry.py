"""Rigid-body primitives for the ``egodriver`` contract.

Re-exported from :mod:`carla_driver_interface.geometry`, which owns them now that
both ends of the contract live in that package; this path stays for existing code.
"""

from carla_driver_interface.geometry import Pose, Trajectory, waypoints_to_proto

__all__ = ["Pose", "Trajectory", "waypoints_to_proto"]
