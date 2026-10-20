"""When each phase-1 adversity fires, and what it does then (#392).

TeraSim's triggers, on an :class:`~.observation.VehicleObservation` or
:class:`~.observation.PedestrianObservation` instead of traci.  Each answers
an :class:`~.command.AdversarialCommand` when the agent may do its
adversarial thing now, else None:

* :attr:`AdversityKind.BRAKE_CHECK` (TeraSim's Follower): a follower close
  behind, the agent not speeding up and the follower not slowing -- brake
  as hard as it can, for 2 s;
* :attr:`AdversityKind.IGNORE_LEADER` (Leader): held back by its leader by
  more than 1.5 m/s² (free flow against car following) -- drive as if the
  road were empty, for 2 s;
* :attr:`AdversityKind.CUT_IN` (LaneChange): a lane beside it may not change
  into because the vehicle behind there is too close, while the one ahead
  there is far enough -- change in anyway, in 1 s;
* :attr:`AdversityKind.TRAFFIC_RULE` (TrafficRule): stopping for a red or
  yellow light rather than for its leader, with free flow at least 1 m/s²
  better (or braking while free flow would not) -- drive on through, 2 s;
* :attr:`AdversityKind.LANE_CHANGE_ABORT` (LanechangeAbort): blocked beside
  as for CUT_IN, moving, and the lanelet long enough for 1 s more -- drift
  half a lane toward that side and back, in 1 s;
* :attr:`AdversityKind.HEAD_ON` (Headon): IGNORE_LEADER's command, where the
  leader's lanelet starts more than 120 degrees off the agent's;
* :attr:`AdversityKind.JAYWALKING`: a pedestrian off any crosswalk, a road
  within 10 m -- walk straight across it at its speed (1.4 m/s if it
  stands) times U(0.5, 2.0), for up to 5 s;
* :attr:`AdversityKind.RUNNING_RED_LIGHT` (RunningRedLight): a pedestrian
  waiting (at most 0.2 m/s) within 3 m of a crosswalk whose light is red --
  cross it anyway, from its near end to its far end, at full speed;
* :attr:`AdversityKind.STOP_CROSSING` (StopCrossing): a pedestrian on a
  crosswalk -- stand still for N(2 s, 0.5 s), then cross at full speed, a
  crosswalk's length on along its heading.

The pedestrian kinds are only a pedestrian's, the others only a vehicle's.

TeraSim's numbers are the defaults, every one overridable
(:class:`TriggerThresholds`).  The accelerations are the Intelligent Driver
Model's (:mod:`.idm`), which stands in for SUMO's car-following model.  Each
gate is a pure function of what it is given, so :func:`adversity_command`,
which adds the location gate (:mod:`.location`), is what both the
``AdversityFeasibleCondition`` and NADE ask.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from ..route.geometry import MapFeatures
from ..trajectory.relative_lane import _centreline, _point_at, _start_heading, _turn
from .command import AdversarialCommand, _times, drift_path
from .geometry import PredictedState
from .idm import (
    DEFAULT_IDM,
    IdmParameters,
    car_following_acceleration,
    free_flow_acceleration,
)
from .location import AdversityLocation, location_matches
from .observation import (
    AgentSnapshot,
    LanePosition,
    NearCrosswalk,
    Neighbour,
    PedestrianObservation,
    Scene,
    SideLane,
    SignalState,
    VehicleObservation,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "DEFAULT_THRESHOLDS",
    "PEDESTRIAN_KINDS",
    "AdversityKind",
    "TriggerThresholds",
    "adversity_command",
    "brake_check",
    "cut_in",
    "head_on",
    "ignore_leader",
    "jaywalking",
    "lane_change_abort",
    "running_red_light",
    "stop_crossing",
    "traffic_rule",
]


class AdversityKind(Enum):
    """The adversities ported from TeraSim."""

    BRAKE_CHECK = "brake_check"
    IGNORE_LEADER = "ignore_leader"
    CUT_IN = "cut_in"
    TRAFFIC_RULE = "traffic_rule"
    JAYWALKING = "jaywalking"
    LANE_CHANGE_ABORT = "lane_change_abort"
    HEAD_ON = "head_on"
    RUNNING_RED_LIGHT = "running_red_light"
    STOP_CROSSING = "stop_crossing"


#: The kinds a pedestrian carries out; every other kind is a vehicle's.
PEDESTRIAN_KINDS = (
    AdversityKind.JAYWALKING,
    AdversityKind.RUNNING_RED_LIGHT,
    AdversityKind.STOP_CROSSING,
)


@dataclass(frozen=True)
class TriggerThresholds:
    """TeraSim's numbers, as the defaults.

    Attributes:
        idm: The car-following model the accelerations come from.
        standstill_speed: Below this (m/s) a vehicle counts as stopped.
        acceleration_gain: How much (m/s²) free flow must beat car following
            for IGNORE_LEADER, and car following the agent's own
            acceleration for TRAFFIC_RULE to blame the rule, not the leader.
        rule_gain: How much (m/s²) free flow must beat the agent's own
            acceleration for TRAFFIC_RULE.
        rule_free_flow_floor: Or, for TRAFFIC_RULE, the least free flow
            (m/s²) of an agent that is braking.
        min_speed_difference: IGNORE_LEADER's least speed difference (m/s)
            from a leader it follows.
        overlap_limit: CUT_IN's most (m) the agent may already overlap the
            vehicle behind in the target lane.
        duration: How long (s) the acceleration commands last.
        lane_change_duration: How long (s) a cut-in, or a lane-change abort,
            takes.
        moving_speed: LANE_CHANGE_ABORT's least speed (m/s): TeraSim's
            "moving at all", not the ``standstill_speed`` the others use.
        abort_resolution: The time step (s) of a lane-change abort's path.
        head_on_angle: HEAD_ON's least difference (degrees) between the
            directions the agent's and its leader's lanelets start in.
        pedestrian_speed: A standing pedestrian's speed (m/s) when it goes.
        pedestrian_speed_factor: The random factor on a jaywalker's speed.
        jaywalk_duration: The most (s) a jaywalk lasts.
        jaywalk_resolution: The time step (s) of a pedestrian's path.
        waiting_speed: RUNNING_RED_LIGHT's most speed (m/s) of a pedestrian
            waiting at the crosswalk.
        crosswalk_reach: How near (m) the crosswalk it waits at must be.
        pedestrian_max_speed: The speed (m/s) a pedestrian crosses at in
            RUNNING_RED_LIGHT and STOP_CROSSING (TeraSim's "max speed"), as
            fast as the fastest jaywalker (1.4 m/s times 2).
        stop_crossing_wait: The mean and standard deviation (s) of how long
            a STOP_CROSSING pedestrian stands still first.
    """

    idm: IdmParameters = DEFAULT_IDM
    standstill_speed: float = 0.5
    acceleration_gain: float = 1.5
    rule_gain: float = 1.0
    rule_free_flow_floor: float = 0.2
    min_speed_difference: float = 2.0
    overlap_limit: float = 2.0
    duration: float = 2.0
    lane_change_duration: float = 1.0
    moving_speed: float = 0.01
    abort_resolution: float = 0.1
    head_on_angle: float = 120.0
    pedestrian_speed: float = 1.4
    pedestrian_speed_factor: tuple[float, float] = (0.5, 2.0)
    jaywalk_duration: float = 5.0
    jaywalk_resolution: float = 0.1
    waiting_speed: float = 0.2
    crosswalk_reach: float = 3.0
    pedestrian_max_speed: float = 2.8
    stop_crossing_wait: tuple[float, float] = (2.0, 0.5)


DEFAULT_THRESHOLDS = TriggerThresholds()


def brake_check(
    seen: VehicleObservation, limits: TriggerThresholds = DEFAULT_THRESHOLDS
) -> Optional[AdversarialCommand]:
    """BRAKE_CHECK: brake hard in front of a follower that keeps coming."""
    follower = seen.follower
    agent = seen.agent
    if follower is None or (
        agent.speed < limits.standstill_speed
        and follower.speed < limits.standstill_speed
    ):
        return None
    if agent.acceleration > 0.0 or follower.acceleration < 0.0:
        return None
    return AdversarialCommand.accelerate(
        -limits.idm.emergency_deceleration, limits.duration
    )


def _held_back_by(
    seen: VehicleObservation, leader: Neighbour, limits: TriggerThresholds
) -> float:
    """Car following's acceleration behind *leader*."""
    return car_following_acceleration(
        seen.agent.speed, seen.desired_speed, leader.gap, leader.speed, limits.idm
    )


def ignore_leader(
    seen: VehicleObservation, limits: TriggerThresholds = DEFAULT_THRESHOLDS
) -> Optional[AdversarialCommand]:
    """IGNORE_LEADER: drive on as if the leader were not there."""
    leader = seen.leader
    agent = seen.agent
    # Stopped itself, it has nothing to ignore its leader with.
    if leader is None or agent.speed < limits.standstill_speed:
        return None
    if abs(agent.speed - leader.speed) < limits.min_speed_difference:
        return None
    free = free_flow_acceleration(agent.speed, seen.desired_speed, limits.idm)
    if free - _held_back_by(seen, leader, limits) <= limits.acceleration_gain:
        return None
    return AdversarialCommand.accelerate(free, limits.duration)


def _secure_gap(
    behind_speed: float, ahead_speed: float, params: IdmParameters
) -> float:
    """The gap (m) a vehicle at *behind_speed* needs behind one at *ahead_speed*.

    SUMO's secure gap: the reaction distance plus what braking to the speed
    ahead takes at the comfortable deceleration.
    """
    braking = (behind_speed * behind_speed - ahead_speed * ahead_speed) / (
        2.0 * params.comfortable_deceleration
    )
    return max(behind_speed * params.time_headway + braking, 0.0)


def _blocked_by_follower_only(
    side: SideLane, speed: float, limits: TriggerThresholds
) -> bool:
    follower = side.follower
    if follower is None or follower.gap <= -limits.overlap_limit:
        return False
    if follower.gap >= _secure_gap(follower.speed, speed, limits.idm):
        return False  # not blocked behind: a lane change TeraSim leaves alone
    leader = side.leader
    return leader is None or leader.gap >= _secure_gap(speed, leader.speed, limits.idm)


def _blocked_side(
    seen: VehicleObservation, speed: float, limits: TriggerThresholds
) -> Optional[SideLane]:
    """The lane beside, left first, blocked only by the vehicle behind there."""
    left = seen.left
    if left is not None and _blocked_by_follower_only(left, speed, limits):
        return left
    right = seen.right
    if right is not None and _blocked_by_follower_only(right, speed, limits):
        return right
    return None


def cut_in(
    seen: VehicleObservation, limits: TriggerThresholds = DEFAULT_THRESHOLDS
) -> Optional[AdversarialCommand]:
    """CUT_IN: change into the lane beside in front of the vehicle there."""
    side = _blocked_side(seen, seen.agent.speed, limits)
    if side is None:
        return None
    if side is seen.left:
        return AdversarialCommand.change_lane("left", limits.lane_change_duration)
    return AdversarialCommand.change_lane("right", limits.lane_change_duration)


def lane_change_abort(
    seen: VehicleObservation,
    features: MapFeatures,
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """LANE_CHANGE_ABORT: start into the lane beside, then think better of it.

    Blocked as for CUT_IN, it drifts half the lanes' separation toward that
    side and back, within its lanelet.
    """
    lane = seen.lane
    if lane is None:
        return None
    speed = seen.agent.speed
    duration = limits.lane_change_duration
    if speed <= limits.moving_speed:
        return None
    if lane.s + speed * duration >= features.length(lane.lanelet):
        return None
    side = _blocked_side(seen, speed, limits)
    if side is None:
        return None
    # A typed local: Codon unwraps the Optional here, not at each use.
    beside: SideLane = side
    ox, oy = _point_at(_centreline(lane.lanelet), lane.s)
    nx, ny = _point_at(_centreline(beside.lanelet), beside.s)
    drift = math.hypot(nx - ox, ny - oy) / 2.0
    if beside is not seen.left:
        drift = -drift
    return AdversarialCommand.follow(
        drift_path(features, lane, speed, drift, duration, limits.abort_resolution)
    )


def head_on(
    seen: VehicleObservation,
    leader_lane: Optional[LanePosition],
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """HEAD_ON: IGNORE_LEADER at a leader coming the other way.

    TeraSim's ``is_head_on``: the lanelets the agent and its leader
    (*leader_lane*) are on start more than ``head_on_angle`` apart.
    """
    lane = seen.lane
    if lane is None or leader_lane is None:
        return None
    if lane.lanelet.id == leader_lane.lanelet.id:
        return None  # one lanelet, one direction
    command = ignore_leader(seen, limits)
    if command is None:
        return None
    apart = _turn(_start_heading(lane.lanelet), _start_heading(leader_lane.lanelet))
    if math.degrees(apart) <= limits.head_on_angle:
        return None
    return command


def traffic_rule(
    seen: VehicleObservation, limits: TriggerThresholds = DEFAULT_THRESHOLDS
) -> Optional[AdversarialCommand]:
    """TRAFFIC_RULE: run the red (or yellow) light it is stopping for."""
    signal = seen.signal
    if signal is None or signal.state not in (SignalState.RED, SignalState.YELLOW):
        return None
    agent = seen.agent
    current = agent.acceleration
    leader = seen.leader
    if leader is not None:
        if _held_back_by(seen, leader, limits) - current <= limits.acceleration_gain:
            return None  # the leader holds it back, not the rule
    free = free_flow_acceleration(agent.speed, seen.desired_speed, limits.idm)
    if free - current > limits.rule_gain or (
        current < 0.0 and free > limits.rule_free_flow_floor
    ):
        return AdversarialCommand.accelerate(free, limits.duration)
    return None


def jaywalking(
    seen: PedestrianObservation,
    rng: random.Random,
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """JAYWALKING: walk straight across the nearest road, off any crosswalk."""
    road = seen.nearest_road
    if seen.crosswalk is not None or road is None:
        return None
    agent = seen.agent
    base = agent.speed if agent.speed >= 1.0 else limits.pedestrian_speed
    low, high = limits.pedestrian_speed_factor
    speed = base * rng.uniform(low, high)
    heading = agent.yaw
    if road.distance > 0.0:
        heading = math.atan2(road.y - agent.y, road.x - agent.x)
    # TeraSim walks across, at least 10 s worth, cut to the command's 5 s.
    duration = min(max(road.distance / speed, 10.0), limits.jaywalk_duration)
    return AdversarialCommand.follow(
        _walk(agent, heading, speed, duration, limits.jaywalk_resolution)
    )


def _walk(
    agent: AgentSnapshot,
    heading: float,
    speed: float,
    duration: float,
    resolution: float,
    wait: float = 0.0,
) -> list[PredictedState]:
    """Stand still for *wait*, then walk straight on at *speed* for *duration*."""
    path: list[PredictedState] = []
    if wait > 0.0:
        for t in _times(wait, resolution):
            path.append(PredictedState(agent.x, agent.y, heading, 0.0, t))
    for t in _times(duration, resolution):
        if t > 0.0 or not path:
            path.append(
                PredictedState(
                    agent.x + speed * t * math.cos(heading),
                    agent.y + speed * t * math.sin(heading),
                    heading,
                    speed,
                    wait + t,
                )
            )
    return path


def running_red_light(
    seen: PedestrianObservation,
    features: MapFeatures,
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """RUNNING_RED_LIGHT: cross on red at the crosswalk it waits at.

    The crosswalk's light is the Lanelet2 traffic light the crosswalk
    lanelet carries (:attr:`~.observation.NearCrosswalk.signal`).
    """
    agent = seen.agent
    near = seen.nearest_crosswalk
    if agent.speed > limits.waiting_speed or near is None:
        return None
    # A typed local: Codon unwraps the Optional here, not at each use.
    crossing: NearCrosswalk = near
    if crossing.distance > limits.crosswalk_reach:
        return None
    if crossing.signal is not SignalState.RED:
        return None
    points = _centreline(crossing.lanelet)
    (ax, ay), (bx, by) = points[0], points[-1]
    if math.hypot(bx - agent.x, by - agent.y) < math.hypot(ax - agent.x, ay - agent.y):
        ax, ay, bx, by = bx, by, ax, ay  # from the end it waits at
    speed = limits.pedestrian_max_speed
    duration = features.length(crossing.lanelet) / speed
    return AdversarialCommand.follow(
        _walk(
            agent,
            math.atan2(by - ay, bx - ax),
            speed,
            duration,
            limits.jaywalk_resolution,
        )
    )


def stop_crossing(
    seen: PedestrianObservation,
    features: MapFeatures,
    rng: random.Random,
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """STOP_CROSSING: stop on the crosswalk a moment, then hurry across."""
    on = seen.crosswalk
    if on is None:
        return None
    mean, deviation = limits.stop_crossing_wait
    wait = max(rng.gauss(mean, deviation), 0.0)
    speed = limits.pedestrian_max_speed
    agent = seen.agent
    return AdversarialCommand.follow(
        _walk(
            agent,
            agent.yaw,
            speed,
            features.length(on) / speed,
            limits.jaywalk_resolution,
            wait,
        )
    )


def adversity_command(
    kind: AdversityKind,
    scene: Scene,
    name: str,
    location: AdversityLocation,
    rng: random.Random,
    limits: TriggerThresholds = DEFAULT_THRESHOLDS,
) -> Optional[AdversarialCommand]:
    """What agent *name* of *scene* does as *kind* at *location* now, or None.

    The location gate first -- a vehicle by its lane, a pedestrian by the road
    nearest it -- so a vehicle outside it costs no look around; then the
    kind's trigger.
    """
    anywhere = location is AdversityLocation.ANY
    if kind in PEDESTRIAN_KINDS:
        if scene.agents[name].footprint.is_vehicle:
            return None
        walker = scene.pedestrian(name)
        if not anywhere:
            road = walker.nearest_road
            if road is None or not location_matches(
                location, scene.features, road.lanelet, 0.0
            ):
                return None
        if kind is AdversityKind.JAYWALKING:
            return jaywalking(walker, rng, limits)
        if kind is AdversityKind.RUNNING_RED_LIGHT:
            return running_red_light(walker, scene.features, limits)
        return stop_crossing(walker, scene.features, rng, limits)
    lane = scene.lane_of(name)
    if lane is None:
        return None
    if not anywhere and not location_matches(
        location, scene.features, lane.lanelet, lane.s
    ):
        return None
    seen = scene.vehicle(name)
    if kind is AdversityKind.BRAKE_CHECK:
        return brake_check(seen, limits)
    if kind is AdversityKind.IGNORE_LEADER:
        return ignore_leader(seen, limits)
    if kind is AdversityKind.CUT_IN:
        return cut_in(seen, limits)
    if kind is AdversityKind.LANE_CHANGE_ABORT:
        return lane_change_abort(seen, scene.features, limits)
    if kind is AdversityKind.HEAD_ON:
        leader = seen.leader
        if leader is None:
            return None
        return head_on(seen, scene.lane_of(leader.name), limits)
    return traffic_rule(seen, limits)
