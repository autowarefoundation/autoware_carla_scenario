"""Adversarial commands and the paths they drive (TeraSim's ``NDECommand``).

A trigger answers with an :class:`AdversarialCommand`: accelerate (or brake)
at a rate, change lanes, or follow a given path, for a duration.  TeraSim
turns every one into a predicted trajectory before it executes it
(``update_control_cmds_from_predicted_trajectory``); :func:`plan_command`
does the same, along the Lanelet2 lanes, so the actions' executor
(``actions._adversarial_execution``) has one kind
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
from ..trajectory.relative_lane import _centreline, _heading_at, _point_at
from .geometry import PredictedState
from .observation import AgentSnapshot, LanePosition, SideLane, VehicleObservation
from .prediction import (
    interpolate_trajectory,
    lane_change_longitudinal_speed,
    predict_distance_velocity,
)

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = [
    "DEFAULT_RESOLUTION_S",
    "AdversarialCommand",
    "CommandKind",
    "drift_path",
    "plan_command",
    "predict_command",
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


class _LaneWalk:
    """Points along the lane from where an agent is, for growing distances.

    Past a lanelet's end the straightest successor carries on; with none, the
    line goes on straight along the end's heading.  Each lanelet's centreline
    is read once, however many points are asked of it.
    """

    features: MapFeatures
    lanelet: ConstLanelet
    points: list[tuple[float, float]]
    length: float
    start: float
    end_of_lane: bool

    def __init__(self, features: MapFeatures, lanelet: ConstLanelet, s: float) -> None:
        self.features = features
        self.lanelet = lanelet
        self.points = _centreline(lanelet)
        self.length = features.length(lanelet)
        # Where the current lanelet starts, in metres from the agent.
        self.start = -s
        self.end_of_lane = False

    def at(self, distance: float) -> tuple[float, float, float]:
        """``(x, y, yaw)`` *distance* metres on; not less than the last asked."""
        while not self.end_of_lane and distance - self.start > self.length:
            successor = self.features.straightest_following(self.lanelet)
            if successor is None:
                self.end_of_lane = True
                break
            # A typed local: Codon unwraps the Optional here, not at each use.
            following: ConstLanelet = successor
            self.start += self.length
            self.lanelet = following
            self.points = _centreline(following)
            self.length = self.features.length(following)
        along = distance - self.start
        if along > self.length:
            (x0, y0), (x1, y1) = self.points[-2], self.points[-1]
            heading = math.atan2(y1 - y0, x1 - x0)
            beyond = along - self.length
            return (
                x1 + beyond * math.cos(heading),
                y1 + beyond * math.sin(heading),
                heading,
            )
        x, y = _point_at(self.points, along)
        return x, y, _heading_at(self.points, along)


def drift_path(
    features: MapFeatures,
    lane: LanePosition,
    speed: float,
    drift: float,
    duration: float,
    resolution: float = DEFAULT_RESOLUTION_S,
) -> list[PredictedState]:
    """Drift *drift* metres sideways (negative: right) and back, in *duration*.

    TeraSim's lane-change abort path: along the lane at *speed*, the offset
    from the centreline growing at a constant rate from where the agent is
    to *drift* more at half the duration, then back; the heading is the
    lane's.  The lanelet's own frame and normal stand in for SUMO's
    ``convert2D``.
    """
    walk = _LaneWalk(features, lane.lanelet, lane.s)
    half = duration / 2.0
    states: list[PredictedState] = []
    for t in _times(duration, resolution):
        x, y, yaw = walk.at(speed * t)
        rise = t / half if t <= half else (duration - t) / half
        offset = lane.t + drift * rise
        states.append(
            PredictedState(
                x - offset * math.sin(yaw), y + offset * math.cos(yaw), yaw, speed, t
            )
        )
    return states


def _times(duration: float, resolution: float) -> list[float]:
    steps = max(int(math.ceil(duration / resolution - 1e-9)), 1)
    return [min(i * resolution, duration) for i in range(steps + 1)]


def _accelerate(
    features: MapFeatures,
    lane: LanePosition,
    agent: AgentSnapshot,
    desired_speed: float,
    command: AdversarialCommand,
    resolution: float,
) -> list[PredictedState]:
    times = _times(command.duration, resolution)
    distances, speeds = predict_distance_velocity(
        agent.speed, command.acceleration, times, max(desired_speed, agent.speed)
    )
    walk = _LaneWalk(features, lane.lanelet, lane.s)
    states: list[PredictedState] = []
    for t, d, v in zip(times, distances, speeds):
        x, y, yaw = walk.at(d)
        states.append(PredictedState(x, y, yaw, v, t))
    return states


def _change_lane(
    features: MapFeatures,
    lane: LanePosition,
    side: SideLane,
    agent: AgentSnapshot,
    command: AdversarialCommand,
    resolution: float,
    span: float,
) -> list[PredictedState]:
    """Cross in ``command.duration``, then (to *span*) keep to the new lane."""
    own = _LaneWalk(features, lane.lanelet, lane.s)
    beside = _LaneWalk(features, side.lanelet, side.s)
    times = _times(max(span, command.duration), resolution)
    xs: list[float] = []
    ys: list[float] = []
    speed = agent.speed
    for i, t in enumerate(times):
        ox, oy, _o = own.at(speed * t)
        nx, ny, _n = beside.at(speed * t)
        if i == 0:
            # The lanes' separation, sideways out of the speed (TeraSim's
            # lane-change longitudinal speed); from here on, at that speed.
            speed = lane_change_longitudinal_speed(
                agent.speed, math.hypot(nx - ox, ny - oy), command.duration
            )
        f = min(t / command.duration, 1.0)
        xs.append(ox + (nx - ox) * f)
        ys.append(oy + (ny - oy) * f)
    states: list[PredictedState] = []
    last = len(times) - 1
    for i, t in enumerate(times):
        a = max(i - 1, 0)
        b = min(i + 1, last)
        yaw = math.atan2(ys[b] - ys[a], xs[b] - xs[a])
        states.append(PredictedState(xs[i], ys[i], yaw, agent.speed, t))
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
    lane = observation.lane
    if lane is None:
        raise ValueError(f"{observation.agent.name} is off the lanes")
    if command.kind is CommandKind.ACCELERATION:
        return _accelerate(
            features,
            lane,
            observation.agent,
            observation.desired_speed,
            command,
            resolution,
        )
    side: Optional[SideLane] = observation.right
    if command.side == "left":
        side = observation.left
    if side is None:
        raise ValueError(f"there is no lane on the {command.side} to change into")
    return _change_lane(
        features, lane, side, observation.agent, command, resolution, command.duration
    )


def predict_command(
    features: MapFeatures,
    observation: VehicleObservation,
    command: AdversarialCommand,
    resolution: float,
    horizon: float,
) -> list[PredictedState]:
    """The observed vehicle's future under *command*, to *horizon* (s).

    As TeraSim's NADE predicts: an acceleration is held throughout, a lane
    change keeps to the new lane once across, and a given path is resampled
    every *resolution* and cut at the horizon.

    Raises:
        ValueError: As :func:`plan_command`.
    """
    if command.kind is CommandKind.ACCELERATION:
        held = AdversarialCommand.accelerate(command.acceleration, horizon)
        return plan_command(features, observation, held, resolution)
    if command.kind is CommandKind.TRAJECTORY:
        path = plan_command(features, observation, command, resolution)
        return [
            p for p in interpolate_trajectory(path, resolution) if p.t <= horizon + 1e-9
        ]
    lane = observation.lane
    if lane is None:
        raise ValueError(f"{observation.agent.name} is off the lanes")
    side: Optional[SideLane] = observation.right
    if command.side == "left":
        side = observation.left
    if side is None:
        raise ValueError(f"there is no lane on the {command.side} to change into")
    return _change_lane(
        features, lane, side, observation.agent, command, resolution, horizon
    )
