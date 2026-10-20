"""Adversarial commands and the paths they drive (TeraSim's ``NDECommand``).

A trigger answers with an :class:`AdversarialCommand`: accelerate (or brake)
at a rate, change lanes, or follow a given path, for a duration.  TeraSim
turns every one into a predicted trajectory before it executes it
(``update_control_cmds_from_predicted_trajectory``); :func:`plan_command`
does the same, along the Lanelet2 lanes, so :mod:`.execution` has one kind
of thing to drive: a list of :class:`~.geometry.PredictedState`, timed from
the command's start, in the Lanelet2 map frame.

* **ACCELERATION** keeps to the lane, following the straightest successor
  past its end, at constant acceleration, the speed clipped to
  ``[0, max(desired speed, speed now)]``.
* **LANE_CHANGE** crosses to the lane beside within the duration (1 s in
  TeraSim), sideways at a constant rate, its forward speed reduced so the
  sideways part comes out of it (:func:`.prediction.lane_change_longitudinal_speed`).
* **TRAJECTORY** is the path the trigger gives, re-timed to start at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from ..route.geometry import MapFeatures
from ..trajectory.relative_lane import _centreline, _end_heading, _heading_at, _point_at
from .geometry import PredictedState
from .observation import SideLane, VehicleObservation
from .prediction import lane_change_longitudinal_speed, predict_distance_velocity

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = [
    "DEFAULT_RESOLUTION_S",
    "AdversarialCommand",
    "CommandKind",
    "plan_command",
]

#: How far apart in time (s) the states of a planned path are.
DEFAULT_RESOLUTION_S = 0.1


class CommandKind(Enum):
    """What an adversarial command does."""

    ACCELERATION = "acceleration"
    LANE_CHANGE = "lane_change"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class AdversarialCommand:
    """What an agent does instead of what traffic would have it do.

    Build one with :meth:`accelerate`, :meth:`change_lane` or :meth:`follow`.

    Attributes:
        kind: What it does.
        duration: For how long (s).
        acceleration: ACCELERATION's rate (m/s², negative to brake).
        side: LANE_CHANGE's side, ``"left"`` or ``"right"``.
        path: TRAJECTORY's path.
    """

    kind: CommandKind
    duration: float
    acceleration: float = 0.0
    side: str = ""
    path: tuple[PredictedState, ...] = ()

    @staticmethod
    def accelerate(acceleration: float, duration: float) -> AdversarialCommand:
        """Accelerate (brake, if negative) at *acceleration* for *duration*."""
        return AdversarialCommand(CommandKind.ACCELERATION, duration, acceleration)

    @staticmethod
    def change_lane(side: str, duration: float = 1.0) -> AdversarialCommand:
        """Cross to the lane on *side* (``"left"`` or ``"right"``) in *duration*."""
        if side not in ("left", "right"):
            raise ValueError(f"a lane change goes left or right, not {side!r}")
        return AdversarialCommand(CommandKind.LANE_CHANGE, duration, side=side)

    @staticmethod
    def follow(path: list[PredictedState]) -> AdversarialCommand:
        """Follow *path*, timed on any clock; it lasts as long as the path."""
        if len(path) < 2:
            raise ValueError("a path to follow needs at least two states")
        return AdversarialCommand(
            CommandKind.TRAJECTORY, path[-1].t - path[0].t, path=tuple(path)
        )


def _along(
    features: MapFeatures, lanelet: ConstLanelet, s: float, distance: float
) -> tuple[float, float, float]:
    """``(x, y, yaw)`` *distance* metres along the lane from *s* on *lanelet*.

    Past a lanelet's end the straightest successor carries on; with none,
    the line goes on straight along the end's heading.
    """
    here = lanelet
    at = s + distance
    while at > features.length(here):
        successor = features.straightest_following(here)
        if successor is None:
            heading = _end_heading(here)
            x, y = _centreline(here)[-1]
            beyond = at - features.length(here)
            return (
                x + beyond * math.cos(heading),
                y + beyond * math.sin(heading),
                heading,
            )
        at -= features.length(here)
        here = successor
    points = _centreline(here)
    x, y = _point_at(points, at)
    return x, y, _heading_at(points, at)


def _times(duration: float, resolution: float) -> list[float]:
    steps = max(int(math.ceil(duration / resolution - 1e-9)), 1)
    return [min(i * resolution, duration) for i in range(steps + 1)]


def _accelerate(
    features: MapFeatures,
    observation: VehicleObservation,
    command: AdversarialCommand,
    resolution: float,
) -> list[PredictedState]:
    lane = observation.lane
    assert lane is not None
    agent = observation.agent
    times = _times(command.duration, resolution)
    distances, speeds = predict_distance_velocity(
        agent.speed,
        command.acceleration,
        times,
        max(observation.desired_speed, agent.speed),
    )
    states: list[PredictedState] = []
    for t, d, v in zip(times, distances, speeds):
        x, y, yaw = _along(features, lane.lanelet, lane.s, d)
        states.append(PredictedState(x, y, yaw, v, t))
    return states


def _change_lane(
    features: MapFeatures,
    observation: VehicleObservation,
    command: AdversarialCommand,
    resolution: float,
) -> list[PredictedState]:
    lane = observation.lane
    assert lane is not None
    side: Optional[SideLane] = (
        observation.left if command.side == "left" else observation.right
    )
    if side is None:
        raise ValueError(f"there is no lane on the {command.side} to change into")
    agent = observation.agent
    x0, y0, _yaw = _along(features, lane.lanelet, lane.s, 0.0)
    x1, y1, _yaw1 = _along(features, side.lanelet, side.s, 0.0)
    speed = lane_change_longitudinal_speed(
        agent.speed, math.hypot(x1 - x0, y1 - y0), command.duration
    )
    times = _times(command.duration, resolution)
    points: list[tuple[float, float]] = []
    for t in times:
        f = t / command.duration
        ox, oy, _o = _along(features, lane.lanelet, lane.s, speed * t)
        nx, ny, _n = _along(features, side.lanelet, side.s, speed * t)
        points.append((ox + (nx - ox) * f, oy + (ny - oy) * f))
    states: list[PredictedState] = []
    for i, t in enumerate(times):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, len(points) - 1)]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        states.append(PredictedState(points[i][0], points[i][1], yaw, agent.speed, t))
    return states


def plan_command(
    features: MapFeatures,
    observation: VehicleObservation,
    command: AdversarialCommand,
    resolution: float = DEFAULT_RESOLUTION_S,
) -> list[PredictedState]:
    """The path *command* drives the observed vehicle along, timed from 0.

    Raises:
        ValueError: The vehicle is off the lanes (for ACCELERATION and
            LANE_CHANGE), or there is no lane on the side to change into.
    """
    if command.kind is CommandKind.TRAJECTORY:
        start = command.path[0].t
        return [
            PredictedState(p.x, p.y, p.yaw, p.speed, p.t - start) for p in command.path
        ]
    if observation.lane is None:
        raise ValueError(f"{observation.agent.name} is off the lanes")
    if command.kind is CommandKind.ACCELERATION:
        return _accelerate(features, observation, command, resolution)
    return _change_lane(features, observation, command, resolution)
