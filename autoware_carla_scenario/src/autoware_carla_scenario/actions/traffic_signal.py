"""Traffic signal action: control traffic lights via Lanelet2 regulatory element IDs."""

from __future__ import annotations

import enum
import logging
import math
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ..conditions.base import BaseCondition
from ..coordinate.traffic_light import (
    get_signal_ids_for_controller,
    lanelet2_traffic_light_id_to_opendrive_controller_id,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)


#: A CARLA light stands for a Lanelet2 light no OpenDRIVE controller is named
#: after when one of its stop waypoints lies this near (m) to that light's stop
#: line, facing across it (:func:`lights_on_stop_line`).
STOP_LINE_MATCH_M = 2.0


def stop_line_in_carla(
    lanelet2_tl_id: int,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """The ends of Lanelet2 light *lanelet2_tl_id*'s stop line, in CARLA world
    ``(x, y)``, or ``None`` when the loaded map has no such light or line."""
    if TYPE_CHECKING:
        return None  # the Lanelet2 map's geometry is not compiled (docs/typecheck.md)
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..trajectory.model import MapPose  # noqa: PLC0415
    from ..trajectory.resolve import map_pose_to_carla  # noqa: PLC0415

    try:
        manager = MapManager.get_instance()
        element = manager.lanelet_map.regulatoryElementLayer[int(lanelet2_tl_id)]
        line = element.stopLine
        offset, z_offset = manager.mgrs_offset, manager.z_offset
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    if line is None or len(line) < 2:
        return None
    ends = []
    for point in (line[0], line[len(line) - 1]):
        x, y, _z, _yaw = map_pose_to_carla(
            MapPose(x=float(point.x), y=float(point.y)), offset, z_offset
        )
        ends.append((x, y))
    return ends[0], ends[1]


def lights_on_stop_line(
    world: "carla.World",
    line: tuple[tuple[float, float], tuple[float, float]],
) -> list[Any]:
    """The CARLA lights with a stop waypoint on *line* (CARLA ``(x, y)``):
    within :data:`STOP_LINE_MATCH_M` of it, its lane crossing it -- not a
    light of the road beside, whose lanes run along it."""
    (ax, ay), (bx, by) = line
    span = math.hypot(bx - ax, by - ay)
    if span <= 0.0:
        return []
    ux, uy = (bx - ax) / span, (by - ay) / span
    found: list[Any] = []
    for actor in world.get_actors().filter("traffic.traffic_light*"):
        for waypoint in actor.get_stop_waypoints():
            location = waypoint.transform.location
            forward = waypoint.transform.get_forward_vector()
            along = min(max((location.x - ax) * ux + (location.y - ay) * uy, 0.0), span)
            gap = math.hypot(
                location.x - (ax + ux * along), location.y - (ay + uy * along)
            )
            if gap <= STOP_LINE_MATCH_M and abs(forward.x * ux + forward.y * uy) < 0.5:
                found.append(actor)
                break
    return found


class TrafficLightTarget(enum.Enum):
    """Special target modes for traffic light selection."""

    ALL = "all"
    #: The lights of a path through a junction of the scenario's route:
    #: ``route_junction``, ``route_approach``, ``route_turn``.
    ROUTE = "route"


class TrafficSignalAction(BaseAction):
    """Control traffic light states using Lanelet2 regulatory element IDs.

    This action follows the :class:`BaseAction` pattern, enabling composable
    and condition-aware traffic light control.  For example, a signal change
    can be delayed using :class:`ElapsedTimeCondition` instead of hand-rolled
    closures with manual time tracking.

    Four target modes are supported:

    - ``None`` (default): NOP -- does nothing (safety default).
    - :attr:`TrafficLightTarget.ALL`: Sets all ``traffic.traffic_light*`` actors.
    - A sequence of Lanelet2 regulatory element IDs: Sets only the lights
      matching those IDs.
    - :attr:`TrafficLightTarget.ROUTE`: The lights of the paths through
      junction ``route_junction`` of the scenario's route that enter it from
      ``route_approach`` (``same`` -- the ego's own approach --, ``left``,
      ``right``, ``opposite``) and turn ``route_turn`` (``any``: any way) -- placed on the map the run is on when the action fires
      (:func:`~autoware_carla_scenario.route.positions.junction_traffic_lights`),
      so a logical scenario names a light without naming a map.

    Args:
        state: The desired :class:`carla.TrafficLightState`
            (e.g. ``carla.TrafficLightState.Green``).
        lanelet2_traffic_light_ids: Target specification -- see above.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase (``PRE_TICK`` or ``POST_TICK``).
        label: Human-readable label for logging.
        once: If ``True`` (default) the action fires at most once.
        freeze: If ``True`` (default), freeze every light so that the
            CARLA traffic manager does not override the state.
        route_junction / route_approach / route_turn: The route target.

    Raises:
        ValueError: For a route approach or turn that is not one.
    """

    _state: "carla.TrafficLightState"
    #: The target, split by kind: an attribute holds one type in the static
    #: check (Codon), and the target is one of three.
    _all_lights: bool
    _lanelet2_ids: Optional[Sequence[int]]
    _freeze: bool
    _route: bool
    _route_junction: int
    _route_approach: str
    _route_turn: Optional[str]

    def __init__(
        self,
        state: "carla.TrafficLightState",
        lanelet2_traffic_light_ids: Union[
            Sequence[int], TrafficLightTarget, None
        ] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "traffic_signal",
        once: bool = True,
        freeze: bool = True,
        route_junction: int = 0,
        route_approach: str = "same",
        route_turn: str = "any",
    ) -> None:
        from ..route.model import PATH_APPROACHES, PATH_TURNS  # noqa: PLC0415

        super().__init__(label=label, condition=condition, timing=timing, once=once)
        self._state = state
        self._all_lights = False
        self._route = False
        self._lanelet2_ids = None
        if isinstance(lanelet2_traffic_light_ids, TrafficLightTarget):
            if lanelet2_traffic_light_ids == TrafficLightTarget.ROUTE:
                self._route = True
            else:
                self._all_lights = True
        elif lanelet2_traffic_light_ids is not None:
            self._lanelet2_ids = lanelet2_traffic_light_ids
        self._freeze = freeze
        if route_approach not in PATH_APPROACHES:
            raise ValueError(
                f"{label}: route_approach must be one of {list(PATH_APPROACHES)}"
            )
        turn: Optional[str] = None
        if route_turn != "any":
            if route_turn not in PATH_TURNS:
                raise ValueError(
                    f"{label}: route_turn must be one of {list(PATH_TURNS)}"
                )
            turn = route_turn
        if route_junction < 0:
            raise ValueError(f"{label}: route_junction counts from 0")
        self._route_junction = int(route_junction)
        self._route_approach = route_approach
        self._route_turn = turn

    # ------------------------------------------------------------------
    # BaseAction interface
    # ------------------------------------------------------------------

    def execute(self, world: "carla.World") -> None:
        """Set the traffic light state according to the target specification."""
        if self._all_lights:
            self._set_all(world)
            return
        if self._route:
            ids = self._route_light_ids()
            if ids:
                self._set_by_lanelet2_ids(world, ids)
            return

        lanelet2_ids = self._lanelet2_ids
        if lanelet2_ids is None:
            logger.debug("TrafficSignalAction: target is None -- NOP")
            return
        self._set_by_lanelet2_ids(world, lanelet2_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route_light_ids(self) -> list[int]:
        """The route target's lights on the map the run is on (said if none)."""
        if TYPE_CHECKING:
            return []  # the route's frame is not compiled (docs/typecheck.md)
        from ..route.active import scenario_route_frame  # noqa: PLC0415
        from ..route.positions import junction_traffic_lights  # noqa: PLC0415

        turn = self._route_turn
        try:
            ids = junction_traffic_lights(
                scenario_route_frame(),
                self._route_junction,
                self._route_approach,
                turn,
            )
        except (ValueError, IndexError) as exc:
            logger.warning("TrafficSignalAction [%s]: %s", self.label, exc)
            return []
        if not ids:
            logger.warning(
                "TrafficSignalAction [%s]: no traffic light on the paths into route "
                "junction %d from %s turning %s",
                self.label,
                self._route_junction,
                self._route_approach,
                turn or "any way",
            )
        return ids

    def _set_all(self, world: "carla.World") -> None:
        """Set every traffic light actor in the world."""
        count = 0
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            actor.set_state(self._state)
            actor.freeze(self._freeze)
            count += 1
        logger.info(
            "TrafficSignalAction [%s]: set %d traffic lights to %s (freeze=%s)",
            self.label,
            count,
            self._state,
            self._freeze,
        )

    def _set_by_lanelet2_ids(
        self,
        world: "carla.World",
        lanelet2_ids: Sequence[int],
    ) -> None:
        """Set traffic lights matching the given Lanelet2 regulatory element IDs."""
        # Collect all OpenDRIVE signal IDs for the requested Lanelet2 IDs;
        # a light no controller is named after (a roadgen map), by its stop line.
        all_signal_ids: set[str] = set()
        by_stop_line: set[int] = set()  # actor ids: each listing wraps anew
        for ll2_id in lanelet2_ids:
            controller_id = lanelet2_traffic_light_id_to_opendrive_controller_id(ll2_id)
            if controller_id is None:
                line = stop_line_in_carla(ll2_id)
                found = [] if line is None else lights_on_stop_line(world, line)
                if found:
                    by_stop_line.update(int(light.id) for light in found)
                    continue
                logger.warning(
                    "TrafficSignalAction [%s]: no OpenDRIVE controller found "
                    "for Lanelet2 traffic light ID %d",
                    self.label,
                    ll2_id,
                )
                continue
            signal_ids = get_signal_ids_for_controller(controller_id)
            all_signal_ids.update(signal_ids)

        if not all_signal_ids and not by_stop_line:
            logger.warning(
                "TrafficSignalAction [%s]: no matching signals found for "
                "Lanelet2 IDs %s",
                self.label,
                list(lanelet2_ids),
            )
            return

        count = 0
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            if actor.get_opendrive_id() in all_signal_ids or actor.id in by_stop_line:
                actor.set_state(self._state)
                actor.freeze(self._freeze)
                count += 1

        logger.info(
            "TrafficSignalAction [%s]: set %d traffic lights to %s "
            "for Lanelet2 IDs %s (freeze=%s)",
            self.label,
            count,
            self._state,
            list(lanelet2_ids),
            self._freeze,
        )
