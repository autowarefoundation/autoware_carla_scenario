"""Walk-straight action: send a pedestrian forward with no destination."""

from __future__ import annotations

import logging
import math
from typing import Optional, Union

import typesafe_carla.carla as carla

from ..action_state import ActionState
from ..conditions.base import BaseCondition, ScenarioResult
from ..coordinate.lane_distance import lane_closing_speed, lane_separation
from ..coordinate.map_manager import run_carla_map
from ..coordinate.poses import CarlaWorldPose
from ..entity.pedestrian_entity import PedestrianEntity
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..entity_role import EntityRole
from .base import BaseAction, TickTiming

logger = logging.getLogger(__name__)

__all__ = ["WalkStraightAction"]

#: A brisk but ordinary walking pace.  `scenario_simulator_v2`'s own
#: `WalkStraightAction` takes no speed at all, so a scenario transpiled from it
#: needs a defensible default rather than a zero that would leave the
#: pedestrian standing in the road.
DEFAULT_WALKING_SPEED_MS = 1.4

#: How far onto the road a pedestrian may have walked and still turn back (m),
#: when the scenario does not say: about one stride and a half.
DEFAULT_TURN_BACK_WITHIN_M = 1.0

#: How far back from the kerb a pedestrian that turns back stands (m), at most
#: as far as where it set off.  Not on the kerb's edge itself: a CARLA 0.10
#: walker stopped with its feet on the edge tips over, slowly, into the road.
_STEP_BACK_M = 0.5

#: Closing speeds below this are "not closing", as in the TTC condition.
_CLOSING_SPEED_EPSILON = 1e-6

# Where a walk that may turn back is (``_phase``).
_WALKING = "walking"
_TURNING_BACK = "turning_back"
_SETTLED = "settled"


class WalkStraightAction(BaseAction):
    """Make a pedestrian walk forward, holding its current heading.

    `scenario_simulator_v2`'s built-in command of the same name.  There is no
    destination: the walker goes the way it is facing until something else
    stops it, which is what makes it usable for a crossing -- the pedestrian
    steps off the kerb on a trigger and keeps going, rather than routing
    around the vehicle the scenario is about.

    Setting *speed_ms* to zero stops the walker where it stands, so a scenario
    can start one walking and stop it later with a second card.

    **Turning back.**  Given *turn_back_ttc_s* or *turn_back_distance_m*, the
    pedestrian watches the vehicles while it walks, and turns back when one
    comes too close -- if it has not gone far onto the road yet:

    * *Too close* is measured in the lane frame, as
      :class:`~autoware_carla_scenario.conditions.TimeToCollisionCondition`
      and :class:`~autoware_carla_scenario.conditions.EntityDistanceCondition`
      measure with ``coordinate_system=LANE``: the separation along the road
      between the vehicle's centre and the pedestrian's, and its time to
      collision (that separation over the speed at which it closes).  Either
      at or under its threshold turns the pedestrian back; a threshold left
      ``None`` is not looked at.  Every vehicle in the world counts.
    * *On the road* means a driving lane is under the pedestrian, as
      :class:`~autoware_carla_scenario.conditions.OffRoadFractionCondition`
      reads it.  How far it has gone onto the road is the distance from the
      last place it stood off the road -- the kerb, where it stepped out.
      Within *turn_back_within_m* of the kerb it turns round and walks back
      at *speed_ms*, to half a metre behind the kerb (no further back than
      where it set off), and stands there.  Further out, it keeps going.
    * Not on the road yet, it steps back the same way rather than step out.

    Either way the walk ends there: the pedestrian stands until another
    action moves it.  The action runs, watching, until the pedestrian has
    turned back and stopped, or is too far out to (or started on the road,
    with no kerb behind it to go back to); a walk that does not turn back is
    over the tick after it was commanded, as before.

    Args:
        entity_name: ``role_name`` of the pedestrian to command.
        speed_ms: Walking speed in metres per second.
        condition: Trigger condition (see :class:`BaseCondition`).
        timing: Tick phase.
        label: Human-readable identifier.
        once: If ``True`` (default) the action fires at most once.
        turn_back_ttc_s: Turn back when a vehicle's time to collision with
            the pedestrian, along the road, is at most this (s).  ``None``
            (default): not by time to collision.
        turn_back_distance_m: Turn back when a vehicle is at most this far
            from the pedestrian along the road, centre to centre (m).  ``None``
            (default): not by distance.
        turn_back_within_m: How far onto the road, from the kerb, the
            pedestrian may be and still turn back (m).

    Raises:
        ValueError: If *speed_ms*, a turn-back threshold or
            *turn_back_within_m* is negative.  A negative speed would be read
            by CARLA as a direction rather than refused, and a pedestrian
            walking backwards through a crossing is not what anyone wrote.
    """

    # Declared for the static check (docs/typecheck.md).
    _entity_name: str
    _speed_ms: float
    _turn_back_ttc_s: Optional[float]
    _turn_back_distance_m: Optional[float]
    _turn_back_within_m: float
    _carla_map: Optional[carla.Map]
    _phase: str
    _yaw_deg: float
    _has_kerb: bool
    _kerb_x: float
    _kerb_y: float
    _start_x: float
    _start_y: float
    _target_x: float
    _target_y: float

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        speed_ms: float = DEFAULT_WALKING_SPEED_MS,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "walk_straight",
        once: bool = True,
        turn_back_ttc_s: Optional[float] = None,
        turn_back_distance_m: Optional[float] = None,
        turn_back_within_m: float = DEFAULT_TURN_BACK_WITHIN_M,
    ) -> None:
        if speed_ms < 0:
            raise ValueError("speed_ms must not be negative")
        if turn_back_ttc_s is not None and turn_back_ttc_s < 0:
            raise ValueError("turn_back_ttc_s must not be negative")
        if turn_back_distance_m is not None and turn_back_distance_m < 0:
            raise ValueError("turn_back_distance_m must not be negative")
        if turn_back_within_m < 0:
            raise ValueError("turn_back_within_m must not be negative")
        turns_back = turn_back_ttc_s is not None or turn_back_distance_m is not None
        # An if statement, not a conditional expression: Codon does not take
        # a subclass for an Optional of its base there (docs/typecheck.md).
        until: Optional[BaseCondition] = None
        if turns_back:
            settled: BaseCondition = _WalkSettled(self, label)
            until = settled
        super().__init__(
            label=label, condition=condition, timing=timing, once=once, until=until
        )
        self._entity_name = str(entity_name)
        self._speed_ms = speed_ms
        self._turn_back_ttc_s = turn_back_ttc_s
        self._turn_back_distance_m = turn_back_distance_m
        self._turn_back_within_m = turn_back_within_m
        self._carla_map: Optional[carla.Map] = None
        self._phase = _SETTLED
        self._yaw_deg = 0.0
        self._has_kerb = False
        self._kerb_x = 0.0
        self._kerb_y = 0.0
        self._start_x = 0.0
        self._start_y = 0.0
        self._target_x = 0.0
        self._target_y = 0.0

    @property
    def turns_back(self) -> bool:
        """Whether the pedestrian may turn back (a turn-back threshold is set)."""
        return (
            self._turn_back_ttc_s is not None or self._turn_back_distance_m is not None
        )

    @property
    def settled(self) -> bool:
        """Whether the walk has nothing left to watch: the pedestrian turned
        back and stands, or is past turning back."""
        return self._phase == _SETTLED

    def _reissues_by_default(self) -> bool:
        # A walk that may turn back is watched every tick of its run.
        return self.turns_back

    def execute(self, world: "carla.World") -> None:
        """Tell the named pedestrian to start walking, then, for a walk that
        may turn back, watch the vehicles every tick."""
        if self.state is ActionState.STANDBY:
            # The trigger's call: a new walk begins.
            self._start(world)
            return
        self._watch(world)

    def _start(self, world: "carla.World") -> None:
        self._phase = _SETTLED
        walker = self._find_walker()
        if walker is None:
            return
        walker.walk_straight(self._speed_ms)
        logger.info(
            "WalkStraightAction: '%s' walking at %.2f m/s",
            self._entity_name,
            self._speed_ms,
        )
        actor = walker.actor
        if not self.turns_back or self._speed_ms <= 0.0 or actor is None:
            return
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        walking: carla.Actor = actor
        transform = walking.get_transform()
        self._yaw_deg = transform.rotation.yaw
        self._start_x = transform.location.x
        self._start_y = transform.location.y
        self._has_kerb = False
        self._mark_kerb(world, transform.location)
        self._phase = _WALKING

    def _find_walker(self) -> Optional[PedestrianEntity]:
        """The pedestrian this action names, or ``None`` (warned) if there is none."""
        entity = find_pedestrian_entity(self._entity_name)
        if entity is None:
            if find_vehicle_entity(self._entity_name) is None:
                logger.warning(
                    "WalkStraightAction: entity '%s' not found", str(self._entity_name)
                )
            else:
                self._warn_not_a_pedestrian()
            return None
        # Codon types `hasattr` by the static type, so not the Optional.
        walker: PedestrianEntity = entity
        # An entity of no known kind is found here too; it may not walk.
        if not hasattr(walker, "walk_straight"):
            self._warn_not_a_pedestrian()
            return None
        return walker

    # ------------------------------------------------------------------
    # Turning back
    # ------------------------------------------------------------------

    def _watch(self, world: "carla.World") -> None:
        """One tick of a walk that may turn back."""
        if self._phase == _SETTLED:
            return
        entity = find_pedestrian_entity(self._entity_name)
        if entity is None:
            # Gone from the run: nothing left to walk.
            self._phase = _SETTLED
            return
        walker: PedestrianEntity = entity
        actor = walker.actor
        if actor is None:
            self._phase = _SETTLED
            return
        pedestrian: carla.Actor = actor
        location = pedestrian.get_transform().location
        if self._phase == _TURNING_BACK:
            self._walk_back(walker, location)
            return

        on_road = self._mark_kerb(world, location)
        if on_road and not self._has_kerb:
            logger.info(
                "WalkStraightAction: '%s' set off on the road, with no kerb to "
                "turn back to",
                self._entity_name,
            )
            self._phase = _SETTLED
            return
        out_m = 0.0
        if on_road:
            out_m = math.hypot(location.x - self._kerb_x, location.y - self._kerb_y)
        if out_m > self._turn_back_within_m:
            logger.info(
                "WalkStraightAction: '%s' is %.2f m onto the road, too far out "
                "to turn back; it keeps going",
                self._entity_name,
                out_m,
            )
            self._phase = _SETTLED
            return
        threat = self._threat(world, pedestrian, location)
        if threat is None:
            return
        # Where it stands once back: behind the kerb, on the pavement.
        back = math.radians(self._back_yaw_deg())
        step = min(
            _STEP_BACK_M,
            math.hypot(self._start_x - self._kerb_x, self._start_y - self._kerb_y),
        )
        self._target_x = self._kerb_x + step * math.cos(back)
        self._target_y = self._kerb_y + step * math.sin(back)
        walker.walk_straight(self._speed_ms, yaw_deg=self._back_yaw_deg())
        logger.info(
            "WalkStraightAction: '%s' turns back %.2f m onto the road (%s)",
            self._entity_name,
            out_m,
            threat,
        )
        self._phase = _TURNING_BACK
        self._walk_back(walker, location)

    def _walk_back(self, walker: PedestrianEntity, location: carla.Location) -> None:
        """Stand the pedestrian walking back once it has reached where it is
        to stand."""
        back = math.radians(self._back_yaw_deg())
        ahead = (self._target_x - location.x) * math.cos(back) + (
            self._target_y - location.y
        ) * math.sin(back)
        if ahead > 0.0:
            return
        walker.walk_straight(0.0)
        logger.info(
            "WalkStraightAction: '%s' is back behind the kerb", self._entity_name
        )
        self._phase = _SETTLED

    def _back_yaw_deg(self) -> float:
        return self._yaw_deg + 180.0

    def _mark_kerb(self, world: "carla.World", location: carla.Location) -> bool:
        """Whether the pedestrian at *location* is on the road; off it, the
        place is the kerb a walk back would return to."""
        if self._on_road(world, location):
            return True
        self._has_kerb = True
        self._kerb_x = location.x
        self._kerb_y = location.y
        return False

    def _on_road(self, world: "carla.World", location: carla.Location) -> bool:
        """Whether a driving lane is under *location*."""
        held = self._carla_map
        if held is None:
            # The run's: get_map() builds a whole new map on every call.
            held = run_carla_map(world)
            self._carla_map = held
        carla_map: carla.Map = held
        waypoint = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=int(carla.LaneType.Driving)
        )
        return waypoint is not None

    def _threat(
        self, world: "carla.World", pedestrian: carla.Actor, location: carla.Location
    ) -> Optional[str]:
        """What makes the pedestrian turn back -- a vehicle within a threshold,
        along the road -- or ``None``."""
        target = CarlaWorldPose(x=location.x, y=location.y, z=location.z, yaw=0.0)
        walking = pedestrian.get_velocity()
        for vehicle in world.get_actors().filter("vehicle.*"):
            at = vehicle.get_location()
            source = CarlaWorldPose(x=at.x, y=at.y, z=at.z, yaw=0.0)
            separation = lane_separation(source, target)
            if separation is None:
                continue
            distance_m = self._turn_back_distance_m
            if distance_m is not None and separation <= distance_m:
                return f"vehicle {vehicle.id} {separation:.1f} m away along the road"
            ttc_s = self._turn_back_ttc_s
            if ttc_s is None or separation < _CLOSING_SPEED_EPSILON:
                continue
            driving = vehicle.get_velocity()
            closing = lane_closing_speed(
                source, driving.x, driving.y, target, walking.x, walking.y
            )
            if closing is None or closing < _CLOSING_SPEED_EPSILON:
                continue
            ttc = separation / closing
            if ttc <= ttc_s:
                return f"vehicle {vehicle.id} {ttc:.1f} s away along the road"
        return None

    def _warn_not_a_pedestrian(self) -> None:
        logger.warning(
            "WalkStraightAction: '%s' is not a pedestrian and cannot walk",
            str(self._entity_name),
        )


class _WalkSettled(BaseCondition):
    """Satisfied once the walk it watches has nothing left to watch."""

    _action: WalkStraightAction

    def __init__(self, action: WalkStraightAction, label: str) -> None:
        super().__init__(label=f"{label}_settled")
        self._action = action

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        if not self._action.settled:
            return None
        return ScenarioResult(
            passed=True, message="the walk is settled", elapsed_seconds=elapsed
        )
