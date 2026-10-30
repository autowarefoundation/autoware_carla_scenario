"""Which traffic light governs the ego, and where it stops for it.

The stop-line half of :class:`~autoware_carla_scenario.driver.renderer.RendererDataBuilder`:
the lane-graph walk that finds the light the driving policy must obey, the
index of which light governs which lane, and the distance along the ego's
heading to the point a vehicle stops at for a light.  The renderer keeps what
needs its protobuf messages (the light's state as the proto enum, the payload)
and the ``is_at_traffic_light()`` fallback, and delegates the rest here, which
the library check compiles (``docs/typecheck.md``).

Ported from ``carla_driver_interface``'s reference runtime
(``runtime/carla_world.py`` at ``af1dcd3``, Apache-2.0) so that a policy tuned
against that runtime sees the same numbers here.  The two subtleties worth
keeping are documented at :meth:`StopLines.governing_light` and
:meth:`StopLines.stop_line_points`; both were derived from measurements on
``Town10HD_Opt`` upstream, and re-deriving them by intuition produces a
policy that brakes too late and stops too early.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .base import DriverClientConfig
from .geometry import Pose
from .observation import to_local_pose

if TYPE_CHECKING:
    import typesafe_carla.carla as carla


logger = logging.getLogger(__name__)

__all__ = ["NO_STOP_LINE_M", "StopLines", "junction_mouth", "stop_waypoints"]

#: How far past a stop waypoint to look for the junction it governs, in metres.
_STOP_LINE_SEARCH_M: float = 8.0

#: Step size for that walk, in metres.
_STOP_LINE_STEP_M: float = 0.5

#: Distance reported when no stop line applies to the ego.  The proto documents
#: the field as negative in that case.
NO_STOP_LINE_M: float = -1.0


def stop_waypoints(light: "carla.TrafficLight") -> List["carla.Waypoint"]:
    """Return a light's stop waypoints, tolerating CARLA API differences."""
    try:
        return list(light.get_stop_waypoints())
    except (RuntimeError, AttributeError):
        logger.debug("get_stop_waypoints() unavailable", exc_info=True)
        return []


def junction_mouth(stop: "carla.Waypoint") -> "carla.Waypoint":
    """The first waypoint of the junction *stop* governs.

    Falls back to the stop waypoint itself when the walk finds no junction --
    a stop line on open road is unusual but not ours to second-guess.
    """
    waypoint, travelled = stop, 0.0
    while travelled < _STOP_LINE_SEARCH_M:
        options = waypoint.next(_STOP_LINE_STEP_M)
        if not options:
            return stop
        waypoint = options[0]
        travelled += _STOP_LINE_STEP_M
        if waypoint.is_junction:
            return waypoint
    return stop


class StopLines:
    """The stop lines of one ego's world, and the light that governs the ego.

    Traffic-light geometry is cached: lights and junctions do not move, and the
    lane-graph walks behind them are not cheap enough to repeat at policy rate.

    Args:
        world: The CARLA world.
        ego_actor: The ego vehicle actor.
        carla_map: The run's CARLA map, which the lane graph is walked on.
        config: Driver settings; supplies the sight distance and the lane-walk
            step, read at each call.
    """

    _world: "carla.World"
    _ego: "carla.Actor"
    _map: "carla.Map"
    _config: DriverClientConfig
    _stop_lines: Optional[Dict[Tuple[int, int], List["carla.TrafficLight"]]]
    _stop_line_points: Dict[int, List[np.ndarray[tuple[int], np.dtype[np.float64]]]]

    def __init__(
        self,
        world: "carla.World",
        ego_actor: "carla.Actor",
        carla_map: "carla.Map",
        config: DriverClientConfig,
    ) -> None:
        self._world = world
        self._ego = ego_actor
        self._map = carla_map
        self._config = config
        self._stop_lines = None
        self._stop_line_points = {}

    def governing_light(
        self, ego_pose: Pose
    ) -> Tuple[bool, Optional["carla.TrafficLight"]]:
        """The light the ego must obey, found by looking down its own lane.

        CARLA's ``is_at_traffic_light()`` answers a different question --
        whether the ego is *inside the light's trigger volume* -- and those
        volumes are about a metre thick along the road.  A policy asking CARLA
        therefore learns of a red light at the moment it arrives at it, when
        stopping from any ordinary speed is already impossible, and the overrun
        that follows says nothing about the policy.

        So the lane graph is walked forward instead, and any light whose stop
        line lies on one of those lanes governs us.  That is the question a
        driver answers by looking.

        Inside a junction nothing governs us: having crossed the line, the
        thing to do is clear the box.  Reading a light there is an active
        hazard, since a trigger volume reaches past its own line and a junction
        has four of them -- a vehicle in the middle can stand in the volume of
        a light governing traffic that crosses its path, and be told to stop
        where stopping is worst.

        Returns:
            ``(decided, light)``: ``decided`` is true when the walk answers --
            *light* is the light it found, or ``None`` inside a junction.  It
            is false when the sight distance is zero, or when the walk outside
            a junction found nothing; the caller then asks
            ``is_at_traffic_light()``.
        """
        if self._config.traffic_light_sight_distance_m > 0.0:
            waypoint = self.ego_waypoint()
            if waypoint is not None and waypoint.is_junction:
                return True, None
            lanes = self.lanes_ahead(self._config.traffic_light_sight_distance_m)
            light = self.nearest_light_ahead(lanes, ego_pose)
            if light is not None:
                return True, light
        return False, None

    def ego_waypoint(self) -> Optional["carla.Waypoint"]:
        """Where the ego sits on the lane graph, or ``None`` if nowhere."""
        return self._map.get_waypoint(self._ego.get_location(), project_to_road=True)

    def lanes_ahead(self, distance_m: float) -> List[Tuple[int, int]]:
        """``(road_id, lane_id)`` of the lanes up to the next junction.

        Walked rather than guessed, because a stop line sits on the lane it
        governs and the ego is often still on an earlier segment of road when
        it needs to know.  The walk stops at the junction it reaches -- the
        lane at the mouth may still carry the line, but nothing past it is ours
        to obey yet -- and reports nothing once the ego is already inside one.
        """
        waypoint = self.ego_waypoint()
        if waypoint is None or waypoint.is_junction:
            return []

        step = max(1.0, self._config.route_resolution_m)
        lanes = [(waypoint.road_id, waypoint.lane_id)]
        travelled = 0.0
        while travelled < distance_m:
            options = waypoint.next(step)
            if not options:
                break
            waypoint = options[0]
            travelled += step
            lane = (waypoint.road_id, waypoint.lane_id)
            if lane != lanes[-1]:
                lanes.append(lane)
            if waypoint.is_junction:
                break
        return lanes

    def nearest_light_ahead(
        self, lanes: List[Tuple[int, int]], ego_pose: Pose
    ) -> Optional["carla.TrafficLight"]:
        """The nearest light on *lanes* whose stop line is still ahead of the ego.

        :meth:`lights_on_lanes` keys its index only by ``(road_id, lane_id)``,
        so a light whose stop line sits behind the ego on the current lane --
        or beyond the sight distance -- lands in the same bucket as one
        genuinely ahead, and a bucket holds its lights in CARLA's arbitrary
        actor order.  Ranking the candidates by longitudinal distance and
        dropping any already passed or out of sight is what turns "a light on
        our lane" back into "the light we are about to reach".
        """
        sight = self._config.traffic_light_sight_distance_m
        nearest: Optional["carla.TrafficLight"] = None
        nearest_distance = float("inf")
        for light in self.lights_on_lanes(lanes):
            distance = self.light_distance(light, ego_pose)
            if distance < 0.0 or distance > sight:
                continue
            if distance < nearest_distance:
                nearest, nearest_distance = light, distance
        return nearest

    def lights_on_lanes(
        self, lanes: List[Tuple[int, int]]
    ) -> List["carla.TrafficLight"]:
        """Lights whose stop lines lie on *lanes*.

        The index is keyed by ``(road_id, lane_id)`` alone, so a lane's bucket
        can hold a light behind the ego or beyond the horizon, in CARLA's
        arbitrary actor order.  :meth:`nearest_light_ahead` is what filters and
        ranks these candidates by their stop-line position relative to the ego.
        """
        if not lanes:
            return []
        index = self.stop_lines_by_lane()
        found: List["carla.TrafficLight"] = []
        for lane in lanes:
            found.extend(index.get(lane, []))
        return found

    def stop_lines_by_lane(
        self,
    ) -> Dict[Tuple[int, int], List["carla.TrafficLight"]]:
        """Which light governs which lane, built once -- lights do not move."""
        if self._stop_lines is None:
            index: Dict[Tuple[int, int], List["carla.TrafficLight"]] = {}
            for actor in self._world.get_actors().filter("traffic.traffic_light*"):
                if TYPE_CHECKING:
                    light = actor.as_traffic_light()
                else:
                    light = actor
                for waypoint in stop_waypoints(light):
                    key = (waypoint.road_id, waypoint.lane_id)
                    index.setdefault(key, []).append(light)
            self._stop_lines = index
        return self._stop_lines

    def light_distance(
        self, light: Optional["carla.TrafficLight"], ego_pose: Pose
    ) -> float:
        """Distance the ego still has to travel before the stop line.

        Measured **along the ego's heading**, not as a straight line, and
        negative once the line is behind.  The difference matters: a Euclidean
        distance starts growing again the moment the ego crosses the line, so a
        policy reading it cannot tell "1 m to go" from "1 m past", and a policy
        that correctly stops just inside the trigger volume is told forever
        that there is a line ahead which it has in fact already crossed.
        """
        if light is None:
            return NO_STOP_LINE_M

        points = self.stop_line_points(light)
        if not points:
            return NO_STOP_LINE_M

        # +x in the rig frame is straight ahead.
        longitudinal = ego_pose.inverse().transform_points(np.stack(points))[:, 0]
        ahead = longitudinal[longitudinal >= 0.0]
        if len(ahead):
            return float(ahead.min())
        # All behind: report the nearest of those, so the value stays continuous
        # as the ego crosses the line.
        return float(longitudinal.max())

    def stop_line_points(
        self, light: "carla.TrafficLight"
    ) -> List[np.ndarray[tuple[int], np.dtype[np.float64]]]:
        """Where a vehicle should stop for *light*, in the local frame.

        Not the stop waypoints themselves, which sit further back than the line
        a driver aims at -- upstream measured them a median of 5.5 m upstream
        of the junction they govern on ``Town10HD_Opt``.  A policy told to stop
        there hesitates most of a car-and-a-half short of the mouth.  So each
        stop waypoint is walked forward to the mouth of its junction and that
        is the point reported.

        Cached by the light's actor id: lights and junctions do not move, and
        this walks the lane graph half a metre at a time.
        """
        key = int(light.id)
        cached = self._stop_line_points.get(key)
        if cached is None:
            cached = [
                to_local_pose(junction_mouth(waypoint).transform).position
                for waypoint in stop_waypoints(light)
            ]
            if not cached:
                # No stop waypoints: the light's own position is the best
                # available answer.
                cached = [to_local_pose(light.get_transform()).position]
            self._stop_line_points[key] = cached
        return cached
