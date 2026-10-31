"""Traffic signal action: control traffic lights via Lanelet2 regulatory element IDs."""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ..conditions.base import BaseCondition
from ..coordinate.traffic_light import (
    STOP_LINE_MATCH_M,
    find_traffic_lights_for_lanelet2_id,
    lights_on_stop_line,
    stop_line_in_carla,
)
from .base import BaseAction, TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

logger = logging.getLogger(__name__)

#: Re-exported from :mod:`..coordinate.traffic_light`, where the stop-line
#: fallback now lives so the condition and the signal controllers share it.
__all__ = [
    "STOP_LINE_MATCH_M",
    "TrafficLightTarget",
    "TrafficSignalAction",
    "lights_on_stop_line",
    "stop_line_in_carla",
]


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
        """Set traffic lights matching the given Lanelet2 regulatory element IDs.

        Each id resolves as the condition and the signal controllers resolve
        it (:func:`~..coordinate.traffic_light.find_traffic_lights_for_lanelet2_id`):
        by its OpenDRIVE controller, or by its stop line on a map where no
        controller is named after it (a roadgen map).
        """
        seen: set[int] = set()  # actor ids: a light two ids name is set once
        count = 0
        for ll2_id in lanelet2_ids:
            lights = find_traffic_lights_for_lanelet2_id(world, ll2_id)
            if not lights:
                logger.warning(
                    "TrafficSignalAction [%s]: no traffic light found for "
                    "Lanelet2 traffic light ID %d",
                    self.label,
                    ll2_id,
                )
                continue
            for light in lights:
                light_id = int(light.id)
                if light_id in seen:
                    continue
                seen.add(light_id)
                light.set_state(self._state)
                light.freeze(self._freeze)
                count += 1

        if count == 0:
            logger.warning(
                "TrafficSignalAction [%s]: no matching signals found for "
                "Lanelet2 IDs %s",
                self.label,
                list(lanelet2_ids),
            )
            return

        logger.info(
            "TrafficSignalAction [%s]: set %d traffic lights to %s "
            "for Lanelet2 IDs %s (freeze=%s)",
            self.label,
            count,
            self._state,
            list(lanelet2_ids),
            self._freeze,
        )
