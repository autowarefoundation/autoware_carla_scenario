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
  follower along the lane chain (:meth:`~..route.geometry.MapFeatures.lane_chain`), the same for the
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

from ..odd.probes import lanelet_speed_limit
from ..route.geometry import LaneSpan, MapFeatures
from ..trajectory.relative_lane import _arc, _centreline, _point_at, reference_lane_pose
from .geometry import AgentFootprint

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
        speed_limit: The simulator's speed limit for it (km/h), if any: the
            desired speed where the lanelet has no ``speed_limit`` tag.
    """

    name: str
    x: float
    y: float
    yaw: float
    speed: float
    acceleration: float
    footprint: AgentFootprint
    speed_limit: Optional[float] = None


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
        desired_speed: The speed limit (m/s), read as the ODD's
            ``speed_limit_kph`` reads it -- the lanelet's tag, else the
            agent's own -- else :data:`DEFAULT_DESIRED_SPEED`.
        leader: The nearest vehicle ahead along its lanes.
        follower: The nearest vehicle behind.
        left: The lane on its left, if any.
        right: The lane on its right, if any.
        signal: The next traffic light ahead within the search.
    """

    agent: AgentSnapshot
    lane: Optional[LanePosition]
    desired_speed: float
    leader: Optional[Neighbour] = None
    follower: Optional[Neighbour] = None
    left: Optional[SideLane] = None
    right: Optional[SideLane] = None
    signal: Optional[SignalAhead] = None


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
    _vehicles: dict[str, VehicleObservation]
    _by_lanelet: Optional[dict[int, list[str]]]

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
        self._vehicles = {}
        self._by_lanelet = None

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

    def _on_lanelets(self) -> dict[int, list[str]]:
        """The located vehicles, by the id of the lanelet each is on."""
        if self._by_lanelet is None:
            by_lanelet: dict[int, list[str]] = {}
            for name in self.agents:
                lane = self.lane_of(name)
                if lane is not None:
                    by_lanelet.setdefault(int(lane.lanelet.id), []).append(name)
            self._by_lanelet = by_lanelet
        return self._by_lanelet

    # -- vehicles ------------------------------------------------------------

    def _neighbours(
        self, agent: AgentSnapshot, chain: dict[int, LaneSpan]
    ) -> tuple[Optional[Neighbour], Optional[Neighbour]]:
        """The nearest vehicle ahead and behind the agent along *chain*."""
        on_lanelets = self._on_lanelets()
        leader: Optional[Neighbour] = None
        follower: Optional[Neighbour] = None
        for key, span in chain.items():
            if key not in on_lanelets:
                continue
            for name in on_lanelets[key]:
                if name == agent.name:
                    continue
                other = self.agents[name]
                lane = self._lanes[name]
                assert lane is not None
                along = span.start + lane.s
                gap = (
                    abs(along) - (agent.footprint.length + other.footprint.length) / 2.0
                )
                if gap > self.search:
                    continue
                found = Neighbour(name, gap, other.speed, other.acceleration)
                if along >= 0.0:
                    if leader is None or gap < leader.gap:
                        leader = found
                elif follower is None or gap < follower.gap:
                    follower = found
        return leader, follower

    def _side(
        self, agent: AgentSnapshot, lanelet: ConstLanelet, side: str
    ) -> Optional[SideLane]:
        neighbour = self.features.neighbour(lanelet, side)
        if neighbour is None:
            return None
        s, _t = _arc(neighbour, agent.x, agent.y)
        chain = self.features.lane_chain(neighbour, s, self.search, self.search)
        leader, follower = self._neighbours(agent, chain)
        return SideLane(neighbour, s, leader, follower)

    def _signal(self, chain: dict[int, LaneSpan]) -> Optional[SignalAhead]:
        best: Optional[SignalAhead] = None
        for span in chain.values():
            for at, element_id in self.features.traffic_light_stops(span.lanelet):
                distance = span.start + at
                if distance < 0.0 or distance > self.search:
                    continue
                if best is None or distance < best.distance:
                    state = SignalState.UNKNOWN
                    if element_id in self.signals:
                        state = self.signals[element_id]
                    best = SignalAhead(distance, state, element_id)
        return best

    def vehicle(self, name: str) -> VehicleObservation:
        """What vehicle *name* sees, worked out once a scene."""
        if name in self._vehicles:
            return self._vehicles[name]
        seen = self._observe(name)
        self._vehicles[name] = seen
        return seen

    def _observe(self, name: str) -> VehicleObservation:
        agent = self.agents[name]
        lane = self.lane_of(name)
        if lane is None:
            return VehicleObservation(agent, None, DEFAULT_DESIRED_SPEED)
        # As the ODD's speed_limit_kph: the lanelet's tag, else the
        # simulator's limit for the vehicle, else a default.
        limit = lanelet_speed_limit(lane.lanelet)
        if limit is None:
            limit = agent.speed_limit
        desired = DEFAULT_DESIRED_SPEED if limit is None else limit / 3.6
        chain = self.features.lane_chain(lane.lanelet, lane.s, self.search, self.search)
        leader, follower = self._neighbours(agent, chain)
        return VehicleObservation(
            agent,
            lane,
            desired,
            leader,
            follower,
            self._side(agent, lane.lanelet, "left"),
            self._side(agent, lane.lanelet, "right"),
            self._signal(chain),
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
            if MapFeatures.is_crosswalk(lanelet):
                if crosswalk is None and float(distance) <= 0.0:
                    # A typed local: Codon upcasts a Lanelet to a ConstLanelet,
                    # not to an Optional of one.
                    on: ConstLanelet = lanelet
                    crosswalk = on
                continue
            if road is not None:
                if float(distance) > 0.0:
                    break  # sorted by distance: no crosswalk underfoot either
                continue
            if not MapFeatures.is_road(lanelet):
                continue
            s, _t = _arc(lanelet, agent.x, agent.y)
            cx, cy = _point_at(_centreline(lanelet), s)
            road = RoadPoint(lanelet, cx, cy, math.hypot(cx - agent.x, cy - agent.y))
        return PedestrianObservation(agent, crosswalk, road)
