"""Example scenario: Ego vehicle stops at a red traffic light and proceeds when green.

This scenario verifies traffic light compliance by:

1. Spawning the ego vehicle on lanelet 242 with zero initial speed.
2. Setting all traffic lights to **red** immediately.
3. After ``light_switch_delay_seconds``, switching all traffic lights to
   **green** via :class:`~autoware_carla_scenario.TrafficSignalAction` with
   :class:`~autoware_carla_scenario.ElapsedTimeCondition`.
4. Judging the ego by where it stops, along its lane, against the stop line
   ahead (:data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M`):

   * **fail** the moment its front bumper crosses the stop line while the
     light is red (by more than ``stop_line_tolerance_m``);
   * **fail** at the green unless it has stood still
     (:class:`~autoware_carla_scenario.StandstillCondition`, for
     ``standstill_seconds``) with its front bumper in the **stop zone**,
     ``stop_zone_min_m`` to ``stop_zone_max_m`` before the line -- telling
     an ego that stopped short of the zone from one that had not stopped;
   * **pass** once it has, and after the green has pulled away, reaching
     ``departure_speed_kmh`` (``moving_speed_kmh`` when unset).

   Approaching the line on red is not a failure: an ego that starts far
   back may drive up to the zone and stop there.  One that starts inside
   the zone has to stay there.

The scenario's controls can set where the ego starts and how long the light
stays red: ``stop_line_distance_m`` places the ego that far before the stop
line ahead, as :data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M`
reads it, and the red phase is its own measure,
:data:`RED_LIGHT_DURATION_S`, ``light_switch_delay_seconds`` as the run holds
it.

Typical usage
-------------
Standalone (no pytest)::

    uv run scenario scenario=traffic_light_compliance

With pytest — see ``test/carla_scenario/test_examples.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AndCondition,
    BaseCondition,
    BaseScenario,
    ComparisonRule,
    EgoConfig,
    ElapsedTimeCondition,
    GroundProjectionConfig,
    Lanelet2Pose,
    ScenarioResult,
    SpeedCondition,
    StandstillCondition,
    StickyCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    find_actor_by_role_name,
    place_at_stop_line_distance,
    vehicle_bumpers,
)
from autoware_carla_scenario.failure import FailureCategory
from autoware_carla_scenario.measures import bumpers_of, stop_line_distance_between

from .configs import TrafficLightComplianceConfig

logger = logging.getLogger(__name__)

#: Seconds every light is held red from the run's start before it turns green:
#: this scenario's own measure (``light_switch_delay_seconds``).
RED_LIGHT_DURATION_S = "red_light_duration_s"

#: How far (metres) a reading of the stop-line distance may lie beyond where
#: the ego's own travel since the last reading puts the line and still be the
#: same line.  Past the line the measure reads the next one ahead, or nothing:
#: a jump far larger than a tick's travel and the lane's curve account for.
SAME_LINE_M = 1.0

#: How far (metres) before the line the ego's travel must still put it for a
#: reading that jumps beyond it to be taken as the ego's line instead: the
#: line it had was not on its lane after all (the other branch of a fork, the
#: lane it changed out of), since the measure reads the nearest line ahead.
#: Closer than this, the jump is taken as the line passed: the ego's travel
#: is off by millimetres a tick, not by this much.
PASSED_MARGIN_M = 0.5


class StopLineWatch:
    """Where the ego's front bumper is against the stop line ahead of it,
    tick by tick: signed metres before it, negative once past it.

    :data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M` reads only
    stop lines *ahead*, so past the line it reads the next one, or nothing.
    The watch keeps the line it reads and, each tick, moves it closer by how
    far the front bumper travelled along the ego's heading (*carried*).  It
    takes the measure's reading in its place when the reading is

    * no farther than the carried line (within :data:`SAME_LINE_M`): the same
      line, or a nearer one the ego's lane now leads to (a lane change onto a
      lane whose line is staggered nearer); or
    * farther, while the carried line is still :data:`PASSED_MARGIN_M` or
      more ahead: the ego has not reached the line it had, so that line is
      not on its lane (the other branch where the lane forks) and the reading
      is the ego's line.

    Otherwise -- the reading gone, or jumping beyond a line the ego has
    reached -- the line is carried on the ego's travel alone, and a crossing
    reads as a negative distance however the measure reads then.

    A tick counts as red when it began before the green: the light is switched
    before the world steps (a pre-tick action at ``elapsed >= red_until``) and
    conditions read the world after, so the tick that ends at ``red_until``
    still ran under red, and the first that ran under green ends one step
    later.

    The pass and fail conditions share one watch; it updates once per
    ``elapsed``, whichever of them asks first, so it has to be asked every
    tick (:class:`StoppedInStopZoneCondition` is the first child of the pass
    condition for that).

    Args:
        red_until: Seconds into the run the light turns green.
        stop_zone_min_m: The stop zone's near end, metres before the line.
        stop_zone_max_m: Its far end.
        tolerance_m: How far past the line the front bumper may be before it
            counts as crossed; that far past it counts as at the line (0 m).
        standstill_seconds: How long the ego has to stand still to count as
            stopped (:class:`~autoware_carla_scenario.StandstillCondition`).
        standstill_speed_mps: The speed (m/s) at or below which it stands
            still.
    """

    red_until: float
    stop_zone_min_m: float
    stop_zone_max_m: float
    tolerance_m: float
    standstill_seconds: float
    standstill_speed_mps: float
    #: Signed metres from the front bumper to the stop line; ``None`` until
    #: the ego is found with a stop line ahead of it.
    distance: Optional[float]
    #: When the front bumper first crossed the line, how far past it was, and
    #: whether that tick ran under red.
    crossed_at: Optional[float]
    crossed_by_m: float
    crossed_on_red: bool
    #: When the ego first counted as stopped in the stop zone on red, and
    #: where.  Latched: what it does after that, short of crossing the line on
    #: red, does not undo it.
    stopped_in_zone_at: Optional[float]
    stopped_at_m: float
    #: Whether the ego counted as standing still at the last update.
    standing: bool
    #: When the first tick that ran under green ended; ``None`` before it.
    green_seen_at: Optional[float]
    #: The last red tick: the distance, and for how long the ego had been at
    #: ``standstill_speed_mps`` or slower (``None``: moving).
    red_end_distance: Optional[float]
    red_end_still_for: Optional[float]
    _standstill: StandstillCondition
    _still_since: Optional[float]
    _at: float
    _has_front: bool
    _front_x: float
    _front_y: float

    def __init__(
        self,
        red_until: float,
        stop_zone_min_m: float,
        stop_zone_max_m: float,
        tolerance_m: float,
        standstill_seconds: float,
        standstill_speed_mps: float = 0.1,
    ) -> None:
        if stop_zone_min_m > stop_zone_max_m:
            raise ValueError(
                f"stop_zone_min_m ({stop_zone_min_m}) is past "
                f"stop_zone_max_m ({stop_zone_max_m})"
            )
        if tolerance_m < 0.0:
            raise ValueError("stop_line_tolerance_m must be non-negative")
        self.red_until = red_until
        self.stop_zone_min_m = stop_zone_min_m
        self.stop_zone_max_m = stop_zone_max_m
        self.tolerance_m = tolerance_m
        self.standstill_seconds = standstill_seconds
        self.standstill_speed_mps = standstill_speed_mps
        self.distance = None
        self.crossed_at = None
        self.crossed_by_m = 0.0
        self.crossed_on_red = False
        self.stopped_in_zone_at = None
        self.stopped_at_m = 0.0
        self.standing = False
        self.green_seen_at = None
        self.red_end_distance = None
        self.red_end_still_for = None
        self._standstill = StandstillCondition(
            entity_name=EGO_ROLE_NAME,
            duration=standstill_seconds,
            speed_threshold=standstill_speed_mps,
            label="ego_standstill",
        )
        self._still_since = None
        self._at = -math.inf
        self._has_front = False
        self._front_x = 0.0
        self._front_y = 0.0

    def in_zone(self, distance: float) -> bool:
        """Whether a front bumper *distance* metres before the line is in the
        stop zone; up to ``tolerance_m`` past the line counts as at it."""
        at = distance
        if -self.tolerance_m <= distance < 0.0:
            at = 0.0
        return self.stop_zone_min_m <= at <= self.stop_zone_max_m

    def update(self, world: carla.World, elapsed: float) -> None:
        """Read the ego at *elapsed*; once per ``elapsed``."""
        if elapsed <= self._at:
            return
        # The tick that ended now began at the last update: red if that was
        # before the green (TrafficSignalAction's ElapsedTimeCondition).
        on_red = self._at < self.red_until
        self._at = elapsed
        if not on_red and self.green_seen_at is None:
            self.green_seen_at = elapsed
        self.standing = self._standstill.check(world, elapsed) is not None
        ego = find_actor_by_role_name(world, EGO_ROLE_NAME)
        if ego is None:
            return
        velocity = ego.get_velocity()
        speed = math.sqrt(
            float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
        )
        if speed > self.standstill_speed_mps:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = elapsed
        transform = ego.get_transform()
        bumpers = bumpers_of(ego)
        yaw = math.radians(float(transform.rotation.yaw))
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        front_x = float(transform.location.x) + bumpers.front * forward_x
        front_y = float(transform.location.y) + bumpers.front * forward_y
        reading = stop_line_distance_between(transform, bumpers)
        self._track(reading, front_x, front_y, forward_x, forward_y)
        self._front_x, self._front_y, self._has_front = front_x, front_y, True
        distance = self.distance
        if distance is not None:
            if self.crossed_at is None and distance < -self.tolerance_m:
                self.crossed_at = elapsed
                self.crossed_by_m = -distance
                self.crossed_on_red = on_red
            if (
                self.stopped_in_zone_at is None
                and on_red
                and self.crossed_at is None
                and self.standing
                and self.in_zone(distance)
            ):
                self.stopped_in_zone_at = elapsed
                self.stopped_at_m = distance
        if on_red:
            self.red_end_distance = distance
            still_since = self._still_since
            if still_since is None:
                self.red_end_still_for = None
            else:
                self.red_end_still_for = elapsed - still_since

    def _track(
        self,
        reading: Optional[float],
        front_x: float,
        front_y: float,
        forward_x: float,
        forward_y: float,
    ) -> None:
        last = self.distance
        if last is None or not self._has_front:
            # The first line read is the one the ego is judged against.
            self.distance = reading
            return
        travelled = (front_x - self._front_x) * forward_x + (
            front_y - self._front_y
        ) * forward_y
        carried = last - travelled
        adopt = False
        if reading is not None and carried >= 0.0:
            if reading <= carried + SAME_LINE_M:
                # The same line, or a nearer one: the measure never reads a
                # line behind the front bumper.
                adopt = True
            elif carried >= PASSED_MARGIN_M:
                # Beyond a line the ego has not reached: that line was not on
                # its lane.
                adopt = True
        if adopt:
            self.distance = reading
        else:
            self.distance = carried

    def green_verdict(self) -> str:
        """Why the ego fails at the green, not having stood still in the stop
        zone, from the last tick that ran under red."""
        zone = f"{self.stop_zone_min_m:.1f}-{self.stop_zone_max_m:.1f} m"
        distance = self.red_end_distance
        still_for = self.red_end_still_for
        if distance is None:
            return "Ego had not stopped by the green: no stop line was read ahead of it"
        if still_for is None:
            return (
                f"Ego had not stopped by the green: still moving "
                f"{distance:.2f} m before the stop line (stop zone {zone})"
            )
        if self.in_zone(distance):
            return (
                f"Ego stopped in the stop zone, {distance:.2f} m before the "
                f"stop line, only {still_for:.2f}s before the green: less than "
                f"standstill_seconds ({self.standstill_seconds:.2f}s)"
            )
        if distance < 0.0:
            return (
                f"Ego stopped {-distance:.2f} m past the stop line (within "
                f"stop_line_tolerance_m), outside the stop zone ({zone} before "
                f"it), until the green"
            )
        return (
            f"Ego stopped {distance:.2f} m before the stop line, outside the "
            f"stop zone ({zone} before it), until the green"
        )

    def details(self) -> dict[str, Optional[float]]:
        """What the watch has seen, for the conditions' details: where the
        ego stopped (``stopped_at_stop_line_distance_m``), or when and how far
        it crossed the line."""
        # Item by item: a dict of Optional values in Codon takes a float so.
        found: dict[str, Optional[float]] = {}
        found["stop_line_distance_m"] = self.distance
        found["stop_zone_min_m"] = self.stop_zone_min_m
        found["stop_zone_max_m"] = self.stop_zone_max_m
        found["stop_line_tolerance_m"] = self.tolerance_m
        found["green_at_seconds"] = self.red_until
        found["stopped_in_zone_at_seconds"] = self.stopped_in_zone_at
        found["stopped_at_stop_line_distance_m"] = None
        if self.stopped_in_zone_at is not None:
            found["stopped_at_stop_line_distance_m"] = self.stopped_at_m
        found["crossed_at_seconds"] = self.crossed_at
        found["crossed_by_m"] = None
        if self.crossed_at is not None:
            found["crossed_by_m"] = self.crossed_by_m
        found["red_end_stop_line_distance_m"] = self.red_end_distance
        found["red_end_standing_still_seconds"] = self.red_end_still_for
        return found


class StoppedInStopZoneCondition(BaseCondition):
    """Pass once the ego has stood still in the stop zone before the green
    (:class:`StopLineWatch`); latched by the watch."""

    _watch: StopLineWatch

    def __init__(self, watch: StopLineWatch, *, label: str) -> None:
        super().__init__(label=label)
        self._watch = watch

    def get_details(self) -> dict[str, Optional[float]]:
        return self._watch.details()

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        watch = self._watch
        watch.update(world, elapsed)
        at = watch.stopped_in_zone_at
        if at is None:
            return None
        return ScenarioResult(
            passed=True,
            message=(
                f"Ego stood still {watch.stopped_at_m:.2f} m before the stop "
                f"line, in the stop zone, at {at:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )


class RedLightViolationCondition(BaseCondition):
    """Fail when the ego runs the red, or has not stopped in the stop zone by
    the green (:class:`StopLineWatch`).

    * On a tick that ran under red: the front bumper crossed the stop line.
    * On the first tick that ran under green, not having stood still in the
      stop zone: standing still short of it, stopped in it too briefly, or
      still on its way (or no stop line read at all), each with a message of
      its own (:meth:`StopLineWatch.green_verdict`).
    """

    _watch: StopLineWatch

    def __init__(self, watch: StopLineWatch, *, label: str) -> None:
        super().__init__(label=label)
        self._watch = watch

    def failure_category(self) -> Optional[FailureCategory]:
        """Running the red, or not stopping for it: :attr:`~FailureCategory.TRAFFIC_RULE`."""
        return FailureCategory.TRAFFIC_RULE

    def get_details(self) -> dict[str, Optional[float]]:
        return self._watch.details()

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        watch = self._watch
        watch.update(world, elapsed)
        crossed_at = watch.crossed_at
        if crossed_at is not None and watch.crossed_on_red:
            return ScenarioResult(
                passed=False,
                message=(
                    f"Ego ran the red: its front bumper crossed the stop line "
                    f"at {crossed_at:.2f}s, {watch.crossed_by_m:.2f} m past it, "
                    f"before the green at {watch.red_until:.2f}s"
                ),
                elapsed_seconds=elapsed,
            )
        if watch.green_seen_at is None or watch.stopped_in_zone_at is not None:
            return None
        return ScenarioResult(
            passed=False, message=watch.green_verdict(), elapsed_seconds=elapsed
        )


class TrafficLightComplianceScenario(BaseScenario):
    """Verify ego obeys traffic signals: stop at red, proceed at green.

    The scenario:

    1. Snaps a Lanelet2 pose to the CARLA road surface for the ego spawn.
    2. Enables autopilot so the ego drives via the traffic manager.
    3. Sets every traffic light to **red** (frozen) using
       :class:`TrafficSignalAction`.
    4. Registers a :class:`TrafficSignalAction` with
       :class:`ElapsedTimeCondition` to switch lights to **green** after
       the configured delay.
    5. Registers pass and fail conditions on where the ego stops against the
       stop line (:class:`StopLineWatch`): it must not cross it on red, must
       stand still in the stop zone before the green, and then pull away.
    6. Registers a :class:`TimeoutCondition` as a fail-safe.
    """

    _config: TrafficLightComplianceConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: TrafficLightComplianceConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or TrafficLightComplianceConfig()

    def setup(self) -> None:
        """Snap ego spawn, set lights to red, register conditions."""
        cfg = self._config

        # --- Compute ego spawn from Lanelet2Pose ---
        spawn = self._spawn_pose
        if cfg.stop_line_distance_m is not None and spawn is not None:
            self._spawn_pose = place_at_stop_line_distance(
                spawn,
                cfg.stop_line_distance_m,
                ego_bumpers=vehicle_bumpers(self.world, self.ego_config.vehicle_type),
            )
            logger.info(
                "Ego %.2f m before the stop line: lanelet %d at s=%.2f",
                cfg.stop_line_distance_m,
                self._spawn_pose.lanelet_id,
                self._spawn_pose.s,
            )
        self._setup_ego_spawn()

        delay = cfg.light_switch_delay_seconds
        self.register_measure(
            RED_LIGHT_DURATION_S,
            lambda world: delay,
            unit="s",
            text="Seconds every light is held red from the start before it turns green",
        )

        # --- Set all traffic lights to RED ---
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Red,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_red",
            )
        )

        # --- Register action to switch lights to green after delay ---
        green_action = TrafficSignalAction(
            state=carla.TrafficLightState.Green,
            lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
            condition=ElapsedTimeCondition(
                cfg.light_switch_delay_seconds, label="light_switch"
            ),
            label="switch_to_green",
        )
        self.register_pre_tick(green_action)

        # --- Pass and fail conditions: where the ego stops ---
        watch = StopLineWatch(
            red_until=cfg.light_switch_delay_seconds,
            stop_zone_min_m=cfg.stop_zone_min_m,
            stop_zone_max_m=cfg.stop_zone_max_m,
            tolerance_m=cfg.stop_line_tolerance_m,
            standstill_seconds=cfg.standstill_seconds,
            standstill_speed_mps=cfg.standstill_speed_mps,
        )

        # 1) The ego stood still in the stop zone before the green.  An AND
        #    checks every child on every tick, so the watch is asked every tick.
        stopped_in_zone = StoppedInStopZoneCondition(
            watch, label="ego_stopped_in_stop_zone"
        )

        # 2) Verify ego pulled away after green phase
        green_check_time = cfg.light_switch_delay_seconds + 0.5
        departure_speed_kmh = (
            cfg.moving_speed_kmh
            if cfg.departure_speed_kmh is None
            else cfg.departure_speed_kmh
        )
        moving_after_green = StickyCondition(
            AndCondition(
                [
                    ElapsedTimeCondition(
                        green_check_time,
                        ComparisonRule.GREATER_THAN_OR_EQUAL,
                        label="green_phase_elapsed",
                    ),
                    SpeedCondition(
                        entity_name=EGO_ROLE_NAME,
                        value=departure_speed_kmh / 3.6,
                        rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                        label="ego_moving_after_green",
                    ),
                ]
            )
        )

        self.register_pass_condition(
            AndCondition([stopped_in_zone, moving_after_green])
        )

        # --- Fail: ran the red, or not stopped in the stop zone by the green ---
        self.register_fail_condition(
            RedLightViolationCondition(watch, label="ego_red_light_violation")
        )

        # --- Fail-safe timeout ---
        timeout = (
            cfg.timeout_seconds
            if cfg.timeout_after_green_seconds is None
            else cfg.light_switch_delay_seconds + cfg.timeout_after_green_seconds
        )
        self.register_fail_condition(
            TimeoutCondition(timeout, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` — termination is driven by pass/fail conditions."""
        return False
