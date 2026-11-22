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
The backends' small helpers that import neither simulator live here too,
so that the library check compiles them (#2306): a driver's
speed in what it can see (:func:`_sight_cap`), a velocity in SUMO's frame
(:func:`_sumo_velocity`), the lane index of a lane change
(:func:`_lane_step`), the files beside a network
(:func:`_safe_weights_prefix`) and the world's OpenDRIVE
(:func:`world_opendrive`), :mod:`.backend`'s, and how the mesoscopic model
cuts a road (:func:`segment_count`), the height along a lane
(:func:`_height_at`) and the angle between two headings
(:func:`_angle_between`), :mod:`.mesosim`'s.  Both re-export them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ...coordinate.map_manager import run_carla_map
from ...kinematics.angle import normalize_angle_deg
from ...visibility import driver_quantile, sight_speed_at
from ..base import LaneChangeDirection, TrafficBackendUnavailable, TrafficContext

__all__ = [
    "Pose2D",
    "box_centre_to_origin",
    "carla_to_sumo",
    "carla_xy_to_sumo",
    "origin_to_box_centre",
    "segment_count",
    "sumo_to_carla",
    "sumo_xy_to_carla",
    "world_opendrive",
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


def _sight_cap(sumo_id: str, seen: float, held: tuple[float, float, float]) -> float:
    """The maximum speed (m/s) *sumo_id*'s driver keeps in *seen* metres,
    from *held*: its own maximum speed, deceleration and tau -- the weather
    rules' ``sight_speed`` at the driver's own quantile, scaled to the
    vehicle's braking and reaction."""
    own, decel, tau = held
    limit = sight_speed_at(seen, driver_quantile(sumo_id), decel, tau)
    # Never to a standstill: a car in thick fog still creeps on.  Nor above
    # its own: a background vehicle capped below the creep keeps its cap
    # (#967).
    return min(own, max(limit, 1.0))


def _safe_weights_prefix(net_file: Path) -> Path:
    """The prefix of the safe-weight files beside *net_file*, as roadgen names
    them: ``network.safe`` (``network.safe.src.xml`` ...) beside
    ``network.net.xml`` (:func:`.network.build_network`)."""
    name = net_file.name
    if name.endswith(".net.xml"):
        return net_file.with_name(name[: -len(".net.xml")] + ".safe")
    return net_file.with_suffix(".safe")


def _sumo_velocity(speed: float, angle_deg: float) -> tuple[float, float]:
    """A velocity in SUMO's frame from a speed and SUMO's angle (degrees
    clockwise from north)."""
    heading = math.radians(angle_deg)
    return speed * math.sin(heading), speed * math.cos(heading)


def _lane_step(direction: LaneChangeDirection) -> int:
    """The lane index one lane to *direction*: SUMO counts from the right kerb."""
    return 1 if direction is LaneChangeDirection.LEFT else -1


def world_opendrive(context: TrafficContext, who: str) -> str:
    """The OpenDRIVE of *context*'s world: its file, else the world's map.

    Raises:
        TrafficBackendUnavailable: The run has neither (*who* needs it).
    """
    if context.xodr_path is not None and Path(context.xodr_path).is_file():
        return Path(context.xodr_path).read_text()
    if context.world is None:
        raise TrafficBackendUnavailable(
            f"{who} needs the world's OpenDRIVE, and the run has neither a world "
            "nor an OpenDRIVE file"
        )
    # The run's map where one is held: get_map() builds a whole new one.
    return str(run_carla_map(context.world).to_opendrive())


def segment_count(length: float, segment_length: float) -> int:
    """How many segments SUMO's mesoscopic model cuts a road of *length* into.

    As ``MELoop::numSegmentsFor``: the road's length over the segment length,
    rounded half up, and at least one.  A length that is no finite number --
    the ``inf`` :meth:`SumoMesoscopicTraffic.position` takes for a road it has
    no :class:`MesoLink` of -- is one segment, not the ``OverflowError`` (or,
    for NaN, ``ValueError``) of flooring it (#2283).
    """
    if segment_length <= 0.0 or not math.isfinite(length):
        return 1
    return max(int(math.floor(length / segment_length + 0.5)), 1)


def _height_at(
    shape: Sequence[tuple[float, float]], heights: Sequence[float], offset: float
) -> float:
    """The height *offset* metres along *shape* (measured across the ground,
    as sumolib measures a shape offset), interpolated between the heights of
    the points either side."""
    if not heights:
        return 0.0
    walked = 0.0
    for i in range(1, len(shape)):
        (x0, y0), (x1, y1) = shape[i - 1], shape[i]
        length = math.hypot(x1 - x0, y1 - y0)
        if walked + length >= offset:
            if length <= 0.0:
                return heights[i]
            t = min(max((offset - walked) / length, 0.0), 1.0)
            return heights[i - 1] + (heights[i] - heights[i - 1]) * t
        walked += length
    return heights[-1]


def _angle_between(a: float, b: float) -> float:
    """How far apart two headings are, degrees, 0 to 180."""
    return abs(normalize_angle_deg(a - b))
