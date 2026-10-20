"""What an agent sees, for the adversity triggers (TeraSim's ``NDEEgoSensor``).

TeraSim's triggers read SUMO through traci: the leader and the follower
within 40 m (``getLeader``, ``getFollower``), the lanes beside
(``getLeftFollowers``, ``getLaneChangeStatePretty``), the next signal
(``getNextLinks``, ``getNextTLS``), a person's crossing and the lanes near
it.  :class:`Scene` answers the same questions from the Lanelet2 map and a
snapshot of the agents, so a trigger is a pure function of what it is given:

* every agent is an :class:`AgentSnapshot` -- position, heading, speed,
  acceleration and footprint in the Lanelet2 map frame -- located on its
  lane once per scene (:meth:`Scene.lane_of`);
* :meth:`Scene.vehicle` gives a :class:`VehicleObservation`: the leader and
  follower along the lane chain (:func:`.lanes.lane_chain`), the same for the
  lanes on either side, the desired speed, the next signal ahead;
* :meth:`Scene.pedestrian` gives a :class:`PedestrianObservation`: whether it
  is on a crosswalk, and the nearest point of a road within reach.

Gaps are bumper to bumper, along the lanes.  Signal states come in by the
id of the Lanelet2 traffic-light regulatory element (:mod:`.snapshot` reads
them off CARLA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import lanelet2.core
import lanelet2.geometry

from ..route.geometry import MapFeatures
from ..trajectory.relative_lane import (
    _arc,
    _centreline,
    _end_heading,
    _point_at,
    _start_heading,
    reference_lane_pose,
)
from ..utils.stop_line import regulatory_element_stop_lines
from .geometry import AgentFootprint
from .lanes import lane_chain

if TYPE_CHECKING:
    from lanelet2.core import ConstLanelet

__all__ = [
    "DEFAULT_DESIRED_SPEED",
    "DEFAULT_ROAD_RADIUS_M",
    "DEFAULT_SEARCH_M",
    "AgentSnapshot",
    "LanePosition",
    "Neighbour",
    "PedestrianObservation",
    "RoadPoint",
    "Scene",
    "SideLane",
    "SignalAhead",
    "SignalState",
    "VehicleObservation",
]

#: How far along the lanes (m) a leader, a follower or a signal is looked
#: for, as TeraSim's 40 m.
DEFAULT_SEARCH_M = 40.0

#: How far (m) from a pedestrian a road counts as near, as TeraSim's 10 m.
DEFAULT_ROAD_RADIUS_M = 10.0

#: The desired speed (m/s) on a lanelet without a speed limit: 50 km/h.
DEFAULT_DESIRED_SPEED = 50.0 / 3.6

#: An agent further off a lanelet's centreline than this (m), or turned more
#: than :data:`_MAX_HEADING_RAD` from it, is not on it.
_MAX_LATERAL_M = 2.5
_MAX_HEADING_RAD = math.pi / 3.0

#: A stop line further than this (m) beyond a lanelet's end is not on it.
_ON_LINE_M = 0.1


class SignalState(Enum):
    """The state of a traffic light, as the triggers read it."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AgentSnapshot:
    """An agent's state, in the Lanelet2 map frame.

    Attributes:
        name: Its role name (``npc1``, ``Ego``) or any unique key.
        x: East (m) of its centre.
        y: North (m) of its centre.
        yaw: Heading (rad), counter-clockwise from +x.
        speed: Speed (m/s).
        acceleration: Acceleration along its heading (m/s²).
        footprint: Its size; ``is_vehicle`` tells a vehicle from a pedestrian.
    """

    name: str
    x: float
    y: float
    yaw: float
    speed: float
    acceleration: float
    footprint: AgentFootprint


@dataclass(frozen=True)
class LanePosition:
    """Where on a lanelet an agent is."""

    lanelet: ConstLanelet
    s: float
    t: float


@dataclass(frozen=True)
class Neighbour:
    """A vehicle ahead of or behind an agent along a lane.

    Attributes:
        name: Its name.
        gap: Bumper-to-bumper distance along the lane (m), negative when they
            overlap.
        speed: Its speed (m/s).
        acceleration: Its acceleration (m/s²).
    """

    name: str
    gap: float
    speed: float
    acceleration: float


@dataclass(frozen=True)
class SideLane:
    """The lane on one side of an agent, and who is on it abreast.

    Attributes:
        lanelet: The neighbouring lanelet.
        s: The agent's position projected onto it (m along it).
        leader: The nearest vehicle ahead of that position, within the search.
        follower: The nearest vehicle behind it, within the search.
    """

    lanelet: ConstLanelet
    s: float
    leader: Optional[Neighbour]
    follower: Optional[Neighbour]


@dataclass(frozen=True)
class SignalAhead:
    """The next traffic light an agent has to stop for.

    Attributes:
        distance: From the agent's centre to the stop line, along the lanes
            (m): the stop line of the traffic-light regulatory element, else
            the end of the lanelet it regulates.
        state: The light's state.
        regulatory_element_id: The Lanelet2 traffic-light element's id.
    """

    distance: float
    state: SignalState
    regulatory_element_id: int


@dataclass(frozen=True)
class VehicleObservation:
    """What a vehicle sees.

    Attributes:
        agent: The vehicle.
        lane: Where it is, or None off every lane (nothing else is then
            observed).
        desired_speed: The lane's speed limit (m/s), else
            :data:`DEFAULT_DESIRED_SPEED`.
        leader: The nearest vehicle ahead along its lanes.
        follower: The nearest vehicle behind.
        left: The lane on its left, if any.
        right: The lane on its right, if any.
        signal: The next traffic light ahead within the search.
    """

    agent: AgentSnapshot
    lane: Optional[LanePosition]
    desired_speed: float
    leader: Optional[Neighbour]
    follower: Optional[Neighbour]
    left: Optional[SideLane]
    right: Optional[SideLane]
    signal: Optional[SignalAhead]


@dataclass(frozen=True)
class RoadPoint:
    """The nearest point of a road's centreline.

    Attributes:
        lanelet: The road lanelet.
        x: East (m).
        y: North (m).
        distance: From the pedestrian (m).
    """

    lanelet: ConstLanelet
    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class PedestrianObservation:
    """What a pedestrian sees.

    Attributes:
        agent: The pedestrian.
        crosswalk: The crosswalk lanelet it stands on, if any.
        nearest_road: The nearest road centreline point within the radius.
    """

    agent: AgentSnapshot
    crosswalk: Optional[ConstLanelet]
    nearest_road: Optional[RoadPoint]


def _is_crosswalk(lanelet: ConstLanelet) -> bool:
    return (
        "subtype" in lanelet.attributes
        and str(lanelet.attributes["subtype"]) == "crosswalk"
    )


def _is_traffic_light(element: lanelet2.core.RegulatoryElement) -> bool:
    return (
        "subtype" in element.attributes
        and str(element.attributes["subtype"]) == "traffic_light"
    )


class Scene:
    """The agents of one tick, located on the map once.

    Args:
        features: The map.
        agents: Every agent of the tick, vehicles and pedestrians.
        signals: The state of each traffic light, by its Lanelet2
            regulatory element's id; a light not in it is UNKNOWN.
        search: How far along the lanes to look (m).
    """

    features: MapFeatures
    agents: dict[str, AgentSnapshot]
    signals: dict[int, SignalState]
    search: float
    _lanes: dict[str, Optional[LanePosition]]

    def __init__(
        self,
        features: MapFeatures,
        agents: list[AgentSnapshot],
        signals: dict[int, SignalState],
        search: float = DEFAULT_SEARCH_M,
    ) -> None:
        self.features = features
        self.agents = {agent.name: agent for agent in agents}
        self.signals = signals
        self.search = search
        self._lanes = {}

    # -- where the agents are ------------------------------------------------

    def lane_of(self, name: str) -> Optional[LanePosition]:
        """Where vehicle *name* is on the lanes, or None (off them, or a pedestrian)."""
        if name in self._lanes:
            return self._lanes[name]
        agent = self.agents[name]
        found: Optional[LanePosition] = None
        if agent.footprint.is_vehicle:
            pose = reference_lane_pose(self.features.map, agent.x, agent.y, agent.yaw)
            lanelet: ConstLanelet = self.features.map.laneletLayer[pose.lanelet_id]
            if (
                abs(pose.t) <= _MAX_LATERAL_M
                and abs(pose.heading) <= _MAX_HEADING_RAD
                and MapFeatures.is_road(lanelet)
            ):
                found = LanePosition(lanelet, pose.s, pose.t)
        self._lanes[name] = found
        return found

    # -- vehicles ------------------------------------------------------------

    def _neighbours(
        self, agent: AgentSnapshot, lanelet: ConstLanelet, s: float
    ) -> tuple[Optional[Neighbour], Optional[Neighbour]]:
        """The nearest vehicle ahead and behind *s* on *lanelet*'s chain."""
        chain = lane_chain(self.features, lanelet, s, self.search, self.search)
        leader: Optional[Neighbour] = None
        follower: Optional[Neighbour] = None
        for name, other in self.agents.items():
            if name == agent.name or not other.footprint.is_vehicle:
                continue
            lane = self.lane_of(name)
            if lane is None:
                continue
            key = int(lane.lanelet.id)
            if key not in chain:
                continue
            along = chain[key].start + lane.s
            half = (agent.footprint.length + other.footprint.length) / 2.0
            if along >= 0.0:
                gap = along - half
                if gap <= self.search and (leader is None or gap < leader.gap):
                    leader = Neighbour(name, gap, other.speed, other.acceleration)
            else:
                gap = -along - half
                if gap <= self.search and (follower is None or gap < follower.gap):
                    follower = Neighbour(name, gap, other.speed, other.acceleration)
        return leader, follower

    def _side(
        self, agent: AgentSnapshot, lanelet: ConstLanelet, side: str
    ) -> Optional[SideLane]:
        neighbour = self.features.neighbour(lanelet, side)
        if neighbour is None:
            return None
        s, _t = _arc(neighbour, agent.x, agent.y)
        leader, follower = self._neighbours(agent, neighbour, s)
        return SideLane(neighbour, s, leader, follower)

    def _stop_line_s(
        self, lanelet: ConstLanelet, element: lanelet2.core.RegulatoryElement
    ) -> Optional[float]:
        """Where on *lanelet* the stop line of *element* is (m along it).

        None when the stop line lies before *lanelet* or past it: the
        junction lanelets after a stop line often carry the same light.  An element without a
        stop line stops at the lanelet's end.
        """
        lines = regulatory_element_stop_lines(element, set())
        if not lines:
            return self.features.length(lanelet)
        points = [(float(p.x), float(p.y)) for p in lines[0]]
        if not points:
            return self.features.length(lanelet)
        mx = sum(p[0] for p in points) / len(points)
        my = sum(p[1] for p in points) / len(points)
        s, _t = _arc(lanelet, mx, my)
        # The arc length is clamped to the lanelet: a stop line before its
        # start reads as 0, one past its end as its length.
        points = _centreline(lanelet)
        if s <= 0.0:
            heading = _start_heading(lanelet)
            x0, y0 = points[0]
            if (mx - x0) * math.cos(heading) + (my - y0) * math.sin(
                heading
            ) < -_ON_LINE_M:
                return None
        elif s >= self.features.length(lanelet):
            heading = _end_heading(lanelet)
            x1, y1 = points[-1]
            if (mx - x1) * math.cos(heading) + (my - y1) * math.sin(
                heading
            ) > _ON_LINE_M:
                return None
        return s

    def _signal(self, lane: LanePosition) -> Optional[SignalAhead]:
        chain = lane_chain(self.features, lane.lanelet, lane.s, self.search)
        best: Optional[SignalAhead] = None
        for span in chain.values():
            for element in span.lanelet.regulatoryElements:
                if not _is_traffic_light(element):
                    continue
                at = self._stop_line_s(span.lanelet, element)
                if at is None:
                    continue
                distance = span.start + at
                if distance < 0.0 or distance > self.search:
                    continue
                if best is None or distance < best.distance:
                    state = SignalState.UNKNOWN
                    if int(element.id) in self.signals:
                        state = self.signals[int(element.id)]
                    best = SignalAhead(distance, state, int(element.id))
        return best

    def vehicle(self, name: str) -> VehicleObservation:
        """What vehicle *name* sees."""
        agent = self.agents[name]
        lane = self.lane_of(name)
        if lane is None:
            return VehicleObservation(
                agent, None, DEFAULT_DESIRED_SPEED, None, None, None, None, None
            )
        limit = MapFeatures.speed_limit(lane.lanelet)
        desired = DEFAULT_DESIRED_SPEED if limit is None else limit / 3.6
        leader, follower = self._neighbours(agent, lane.lanelet, lane.s)
        return VehicleObservation(
            agent,
            lane,
            desired,
            leader,
            follower,
            self._side(agent, lane.lanelet, "left"),
            self._side(agent, lane.lanelet, "right"),
            self._signal(lane),
        )

    # -- pedestrians ---------------------------------------------------------

    def pedestrian(
        self, name: str, road_radius: float = DEFAULT_ROAD_RADIUS_M
    ) -> PedestrianObservation:
        """What pedestrian *name* sees."""
        agent = self.agents[name]
        point = lanelet2.core.BasicPoint2d(agent.x, agent.y)
        crosswalk: Optional[ConstLanelet] = None
        road: Optional[RoadPoint] = None
        for distance, lanelet in lanelet2.geometry.findNearest(
            self.features.map.laneletLayer, point, 20
        ):
            if float(distance) > road_radius:
                break
            if _is_crosswalk(lanelet):
                if crosswalk is None and float(distance) <= 0.0:
                    # A typed local: Codon upcasts a Lanelet to a ConstLanelet,
                    # not to an Optional of one.
                    on: ConstLanelet = lanelet
                    crosswalk = on
                continue
            if road is not None or not MapFeatures.is_road(lanelet):
                continue
            s, _t = _arc(lanelet, agent.x, agent.y)
            cx, cy = _point_at(_centreline(lanelet), s)
            road = RoadPoint(lanelet, cx, cy, math.hypot(cx - agent.x, cy - agent.y))
        return PedestrianObservation(agent, crosswalk, road)
