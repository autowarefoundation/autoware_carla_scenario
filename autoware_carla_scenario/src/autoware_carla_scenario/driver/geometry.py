"""Rigid-body primitives for the ``egodriver`` contract.

Re-exported from :mod:`autoware_carla_egodriver.geometry`, which owns them now that
both ends of the contract live in that package; this path stays for existing code.
"""

from autoware_carla_egodriver.geometry import Pose, Trajectory, waypoints_to_proto

__all__ = ["Pose", "Trajectory", "waypoints_to_proto"]
