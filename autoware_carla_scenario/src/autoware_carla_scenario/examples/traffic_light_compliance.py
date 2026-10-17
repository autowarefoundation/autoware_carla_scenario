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
from autoware_carla_scenario.measures import bumpers_of, stop_line_distance_between

from .configs import TrafficLightComplianceConfig

logger = logging.getLogger(__name__)

#: Seconds every light is held red from the run's start before it turns green:
#: this scenario's own measure (``light_switch_delay_seconds``).
RED_LIGHT_DURATION_S = "red_light_duration_s"

#: How far (metres) a reading of the stop-line distance may stray from where
#: the ego's own travel since the last one puts the line and still be the same
#: line.  Past the line the measure reads the next one ahead, or nothing: a
#: jump far larger than a tick's travel and the lane's curve account for.
SAME_LINE_M = 1.0


class StopLineWatch:
    """Where the ego's front bumper is against the stop line it started
    before, tick by tick: signed metres before it, negative once past it.

    :data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M` reads only
    stop lines *ahead*, so past the line it reads the next one, or nothing.
    The watch keeps the line it first read: each tick it moves the line closer
    by how far the front bumper travelled along the ego's heading, and takes
    the measure's reading in its place only while the two agree
    (:data:`SAME_LINE_M`) and the travel has not taken it past the line.
    Otherwise -- the line passed, or the lane lost -- the line is carried on
    the ego's travel alone, so a crossing reads as a negative distance however
    the measure reads then.

    The pass and fail conditions share one watch; it updates once per
    ``elapsed``, whichever of them asks first, so it has to be asked every
    tick (:class:`StoppedInStopZoneCondition` is the first child of the pass
    condition for that).

    Args:
        red_until: Seconds into the run the light turns green.
        stop_zone_min_m: The stop zone's near end, metres before the line.
        stop_zone_max_m: Its far end.
        tolerance_m: How far past the line the front bumper may be before it
            counts as crossed.
        standstill_seconds: How long the ego has to stand still to count as
            stopped (:class:`~autoware_carla_scenario.StandstillCondition`).
    """

    red_until: float
    stop_zone_min_m: float
    stop_zone_max_m: float
    tolerance_m: float
    #: Signed metres from the front bumper to the stop line; ``None`` until
    #: the ego is found with a stop line ahead of it.
    distance: Optional[float]
    #: When the front bumper first crossed the line, and how far past it was.
    crossed_at: Optional[float]
    crossed_by_m: float
    #: When the ego first counted as stopped in the stop zone before the
    #: green, and where.
    stopped_in_zone_at: Optional[float]
    stopped_at_m: float
    #: Whether the ego counted as standing still at the last update.
    standing: bool
    _standstill: StandstillCondition
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
        self.distance = None
        self.crossed_at = None
        self.crossed_by_m = 0.0
        self.stopped_in_zone_at = None
        self.stopped_at_m = 0.0
        self.standing = False
        self._standstill = StandstillCondition(
            entity_name=EGO_ROLE_NAME,
            duration=standstill_seconds,
            label="ego_standstill",
        )
        self._at = -math.inf
        self._has_front = False
        self._front_x = 0.0
        self._front_y = 0.0

    def in_zone(self, distance: float) -> bool:
        """Whether a front bumper *distance* metres before the line is in the
        stop zone."""
        return self.stop_zone_min_m <= distance <= self.stop_zone_max_m

    def update(self, world: carla.World, elapsed: float) -> None:
        """Read the ego at *elapsed*; once per ``elapsed``."""
        if elapsed <= self._at:
            return
        self._at = elapsed
        self.standing = self._standstill.check(world, elapsed) is not None
        ego = find_actor_by_role_name(world, EGO_ROLE_NAME)
        if ego is None:
            return
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
        if distance is None:
            return
        if self.crossed_at is None and distance < -self.tolerance_m:
            self.crossed_at = elapsed
            self.crossed_by_m = -distance
        if (
            self.stopped_in_zone_at is None
            and elapsed < self.red_until
            and self.crossed_at is None
            and self.standing
            and self.in_zone(distance)
        ):
            self.stopped_in_zone_at = elapsed
            self.stopped_at_m = distance

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
        # Once the ego's travel has taken it past the line, a reading is of a
        # line further on, however close: the measure reads none behind.
        if (
            reading is not None
            and carried >= 0.0
            and abs(reading - carried) <= SAME_LINE_M
        ):
            self.distance = reading
        else:
            self.distance = carried

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

    * While the light is red: the front bumper crossed the stop line.
    * At the green, not having stood still in the stop zone: standing still
      short of it, or still on its way (or no stop line read at all), each
      with a message of its own.
    """

    _watch: StopLineWatch

    def __init__(self, watch: StopLineWatch, *, label: str) -> None:
        super().__init__(label=label)
        self._watch = watch

    def get_details(self) -> dict[str, Optional[float]]:
        return self._watch.details()

    def check(self, world: carla.World, elapsed: float) -> Optional[ScenarioResult]:
        watch = self._watch
        watch.update(world, elapsed)
        crossed_at = watch.crossed_at
        if crossed_at is not None and crossed_at < watch.red_until:
            return ScenarioResult(
                passed=False,
                message=(
                    f"Ego ran the red: its front bumper crossed the stop line "
                    f"at {crossed_at:.2f}s, {watch.crossed_by_m:.2f} m past it, "
                    f"before the green at {watch.red_until:.2f}s"
                ),
                elapsed_seconds=elapsed,
            )
        if elapsed < watch.red_until or watch.stopped_in_zone_at is not None:
            return None
        zone = f"{watch.stop_zone_min_m:.1f}-{watch.stop_zone_max_m:.1f} m"
        distance = watch.distance
        message = "Ego had not stopped by the green: no stop line was read ahead of it"
        if distance is not None:
            if watch.standing:
                message = (
                    f"Ego stopped {distance:.2f} m before the stop line, "
                    f"outside the stop zone ({zone} before it), until the green"
                )
            else:
                message = (
                    f"Ego had not stopped by the green: still moving "
                    f"{distance:.2f} m before the stop line (stop zone {zone})"
                )
        return ScenarioResult(passed=False, message=message, elapsed_seconds=elapsed)


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
        )

        # 1) The ego stood still in the stop zone before the green.  First in
        #    the AND, so the watch is asked every tick.
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
