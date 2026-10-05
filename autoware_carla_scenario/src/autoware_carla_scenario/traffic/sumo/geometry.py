"""Poses between SUMO's network frame and CARLA's world frame.

The network comes from the OpenDRIVE the world runs (:mod:`.network`), built
without offset normalisation, so SUMO's x/y are OpenDRIVE's shifted by the
net's ``netOffset`` -- zero for the networks this backend builds, but read
from the net all the same.  CARLA mirrors OpenDRIVE's y axis.  Two more
conventions differ:

* SUMO places a vehicle at the centre of its *front bumper*; CARLA at the
  centre of its bounding box (for the vehicles this backend spawns).
* SUMO's angle is degrees clockwise from north (+y); CARLA's yaw is degrees
  clockwise from +x, in its left-handed frame.

Pure functions of numbers, so they are tested without either simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Pose2D", "carla_to_sumo", "sumo_to_carla"]


@dataclass(frozen=True)
class Pose2D:
    """A position and heading in one simulator's frame (degrees)."""

    x: float
    y: float
    z: float
    heading_deg: float


def sumo_to_carla(
    pose: Pose2D, length_m: float, net_offset: tuple[float, float] = (0.0, 0.0)
) -> Pose2D:
    """The CARLA pose (box centre, yaw) of a SUMO vehicle (front bumper, angle)."""
    yaw = pose.heading_deg - 90.0
    # Back from the front bumper to the middle of the vehicle, in SUMO's frame.
    math_heading = math.radians(90.0 - pose.heading_deg)
    cx = pose.x - math.cos(math_heading) * length_m / 2.0
    cy = pose.y - math.sin(math_heading) * length_m / 2.0
    return Pose2D(cx - net_offset[0], -(cy - net_offset[1]), pose.z, _wrap(yaw))


def carla_to_sumo(
    pose: Pose2D, length_m: float, net_offset: tuple[float, float] = (0.0, 0.0)
) -> Pose2D:
    """The SUMO pose (front bumper, angle) of a CARLA vehicle (box centre, yaw)."""
    angle = pose.heading_deg + 90.0
    sx, sy = pose.x + net_offset[0], -pose.y + net_offset[1]
    math_heading = math.radians(90.0 - angle)
    return Pose2D(
        sx + math.cos(math_heading) * length_m / 2.0,
        sy + math.sin(math_heading) * length_m / 2.0,
        pose.z,
        angle % 360.0,
    )


def _wrap(deg: float) -> float:
    """*deg* in [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0
