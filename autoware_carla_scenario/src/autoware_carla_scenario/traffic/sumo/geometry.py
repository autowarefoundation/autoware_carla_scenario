"""Poses between SUMO's network frame and CARLA's world frame.

The network comes from the OpenDRIVE the world runs (:mod:`.network`), built
without offset normalisation, so SUMO's x/y are OpenDRIVE's shifted by the
net's ``netOffset`` -- zero for the networks this backend builds, but read
from the net all the same.  CARLA mirrors OpenDRIVE's y axis.  Two more
conventions differ:

* SUMO places a vehicle at the centre of its *front bumper*.  CARLA places
  an actor at its blueprint's origin, which is not the centre of its
  bounding box: the box sits at ``bounding_box.location`` in the actor's own
  frame (a car's about a metre forward of its origin, between the axles).
  :func:`sumo_to_carla` and :func:`carla_to_sumo` relate the front bumper to
  the *box centre* -- half the vehicle's length behind it, along its heading
  -- and :func:`box_centre_to_origin` and :func:`origin_to_box_centre` the
  box centre to the origin a CARLA transform holds, by that offset (#887).
* SUMO's angle is degrees clockwise from north (+y); CARLA's yaw is degrees
  clockwise from +x, in its left-handed frame.

Pure functions of numbers, so they are tested without either simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Pose2D",
    "box_centre_to_origin",
    "carla_to_sumo",
    "carla_xy_to_sumo",
    "origin_to_box_centre",
    "sumo_to_carla",
    "sumo_xy_to_carla",
]


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
    """The CARLA pose (box centre, yaw) of a SUMO vehicle (front bumper, angle).

    The centre of the vehicle's bounding box, not the origin a CARLA
    transform holds: :func:`box_centre_to_origin` turns one into the other.
    """
    yaw = pose.heading_deg - 90.0
    # Back from the front bumper to the middle of the vehicle, in SUMO's frame.
    math_heading = math.radians(90.0 - pose.heading_deg)
    cx = pose.x - math.cos(math_heading) * length_m / 2.0
    cy = pose.y - math.sin(math_heading) * length_m / 2.0
    return Pose2D(cx - net_offset[0], -(cy - net_offset[1]), pose.z, _wrap(yaw))


def carla_to_sumo(
    pose: Pose2D, length_m: float, net_offset: tuple[float, float] = (0.0, 0.0)
) -> Pose2D:
    """The SUMO pose (front bumper, angle) of a CARLA vehicle (box centre, yaw).

    The centre of the vehicle's bounding box, not its transform's location:
    :func:`origin_to_box_centre` turns one into the other.
    """
    angle = pose.heading_deg + 90.0
    sx, sy = pose.x + net_offset[0], -pose.y + net_offset[1]
    math_heading = math.radians(90.0 - angle)
    return Pose2D(
        sx + math.cos(math_heading) * length_m / 2.0,
        sy + math.sin(math_heading) * length_m / 2.0,
        pose.z,
        angle % 360.0,
    )


def origin_to_box_centre(pose: Pose2D, offset: tuple[float, float]) -> Pose2D:
    """The centre of a CARLA actor's bounding box, the actor's origin at
    *pose*: moved by *offset* -- ``bounding_box.location``'s x and y, forward
    and to the right in the actor's own frame -- turned by its yaw (CARLA's
    frame is left-handed: +y is right of +x, yaw clockwise)."""
    forward, right = offset
    if forward == 0.0 and right == 0.0:
        return pose
    yaw = math.radians(pose.heading_deg)
    cos_h, sin_h = math.cos(yaw), math.sin(yaw)
    return Pose2D(
        pose.x + forward * cos_h - right * sin_h,
        pose.y + forward * sin_h + right * cos_h,
        pose.z,
        pose.heading_deg,
    )


def box_centre_to_origin(pose: Pose2D, offset: tuple[float, float]) -> Pose2D:
    """The origin -- what a CARLA transform holds -- of an actor whose
    bounding box is centred at *pose*: :func:`origin_to_box_centre` undone."""
    forward, right = offset
    return origin_to_box_centre(pose, (-forward, -right))


def carla_xy_to_sumo(
    x: float, y: float, net_offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """The SUMO point of CARLA's ``(x, y)``: y mirrored, the net offset added."""
    return x + net_offset[0], -y + net_offset[1]


def sumo_xy_to_carla(
    x: float, y: float, net_offset: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """The CARLA point of SUMO's ``(x, y)`` (:func:`carla_xy_to_sumo` undone)."""
    return x - net_offset[0], -(y - net_offset[1])


def _wrap(deg: float) -> float:
    """*deg* in [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0
