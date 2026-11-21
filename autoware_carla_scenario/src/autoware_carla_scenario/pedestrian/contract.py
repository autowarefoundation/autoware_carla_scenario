"""The pedestrian motion contract's frame and values, without grpc.

``proto/pedestrian_motion/v0/pedestrian_motion.proto`` is the contract: the
generator places the joints of a body frame by frame, in its CONTRACT FRAME
(right-handed, z up, metres: CARLA's world with its y negated).
:func:`to_contract` / :func:`to_carla` move a ground point and a heading
between that frame and CARLA's, and the frozen dataclasses here are the
values :class:`~autoware_carla_scenario.pedestrian.motion_client.PedestrianMotionClient`
sends and answers.  They need no grpc, so this module is checked by Codon;
:mod:`~autoware_carla_scenario.pedestrian.motion_client`, which speaks the
contract over grpc, re-exports every name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .skeleton import Vector

__all__ = [
    "GeneratedFrame",
    "MotionGeneratorInfo",
    "PoseKeyframe",
    "RootWaypoint",
    "TextSpan",
    "to_carla",
    "to_contract",
]


def to_contract(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    """A CARLA ground point and yaw as the contract's ``(x, y, heading)``."""
    return x, -y, -math.radians(yaw_deg)


def to_carla(x: float, y: float, heading: float) -> tuple[float, float, float]:
    """A contract ground point and heading as CARLA's ``(x, y, yaw_deg)``."""
    return x, -y, -math.degrees(heading)


@dataclass(frozen=True)
class MotionGeneratorInfo:
    """What a generator answered
    :meth:`~autoware_carla_scenario.pedestrian.motion_client.PedestrianMotionClient.negotiate`."""

    backend: str
    backend_version: str
    model: str
    fps: float
    horizon_frames: int
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class TextSpan:
    """The text the motion follows from *start_frame* on."""

    start_frame: int
    text: str


@dataclass(frozen=True)
class RootWaypoint:
    """Where the root is at *frame* (contract frame), and how it faces."""

    frame: int
    x: float
    y: float
    heading: Optional[float] = None


@dataclass(frozen=True)
class PoseKeyframe:
    """The whole body at *frame*: each joint of the generator's
    :attr:`MotionGeneratorInfo.joint_names`, in order (contract frame)."""

    frame: int
    joints: tuple[Vector, ...]


@dataclass(frozen=True)
class GeneratedFrame:
    """One frame of a generated motion (contract frame).

    Attributes:
        frame: Its number from the session's start.
        root: The root's ground point.
        heading: The way the body faces.
        joints: Each joint's position, by name.
    """

    frame: int
    root: Vector
    heading: float
    joints: dict[str, Vector]
