"""Footprints and predicted states (TeraSim's ``geometry_utils_cy``).

A vehicle is approximated by ``ceil(length / width) + 1`` circles along its
heading, anything else (a pedestrian, a cyclist) by one; two agents touch when
any of their circles do.  This is TeraSim's model, kept so a ported trigger
judges a conflict the way the original does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "AgentFootprint",
    "PredictedState",
    "circle_centers",
    "circle_offsets",
    "circle_radius",
]


@dataclass(frozen=True)
class PredictedState:
    """One sample of an agent's (predicted) motion, in the Lanelet2 map frame.

    Attributes:
        x: East, in metres, of the agent's centre.
        y: North, in metres, of the agent's centre.
        yaw: Heading in radians, counter-clockwise from +x.
        speed: Speed along the heading, in m/s.
        t: Time in seconds, on whatever clock the trajectory shares with the
            one it is compared against.
    """

    x: float
    y: float
    yaw: float
    speed: float
    t: float


@dataclass(frozen=True)
class AgentFootprint:
    """The size of an agent, for the circle approximation.

    Attributes:
        length: Bounding-box length in metres.
        width: Bounding-box width in metres.
        is_vehicle: True for a vehicle (several circles along the heading),
            False for a pedestrian or any other agent (one circle).
    """

    length: float
    width: float
    is_vehicle: bool = True


def _circle_count(footprint: AgentFootprint) -> int:
    if not footprint.is_vehicle:
        return 1
    return int(math.ceil(footprint.length / footprint.width)) + 1


def circle_radius(footprint: AgentFootprint) -> float:
    """Radius of each circle approximating *footprint*.

    A vehicle's circles cover its width and their share of its length; any
    other agent's one circle covers its larger dimension.
    """
    if not footprint.is_vehicle:
        return max(footprint.length, footprint.width) / 2.0
    count = _circle_count(footprint)
    return math.sqrt(
        (footprint.length / count / 2.0) ** 2 + (footprint.width / 2.0) ** 2
    )


def circle_offsets(footprint: AgentFootprint) -> list[float]:
    """Signed distances (metres) of the circle centres from the agent's centre.

    The circles are spaced ``length / count`` apart along the heading and
    centred on the agent: an even count straddles the centre, an odd one puts
    a circle on it.  They depend on the footprint alone, so a caller checking
    many samples computes them once.
    """
    count = _circle_count(footprint)
    spacing = footprint.length / count
    half = (count - 1) / 2.0
    return [(i - half) * spacing for i in range(count)]


def circle_centers(
    point: PredictedState, offsets: list[float]
) -> list[tuple[float, float]]:
    """Centres of the circles at *point*, from :func:`circle_offsets`."""
    cos_yaw = math.cos(point.yaw)
    sin_yaw = math.sin(point.yaw)
    return [(point.x + k * cos_yaw, point.y + k * sin_yaw) for k in offsets]
