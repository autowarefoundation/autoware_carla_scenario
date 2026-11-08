"""Hydra-compatible dataclass definitions for scenario parameters.

Each dataclass corresponds to a YAML config group and provides typed,
documented defaults.  Hydra's ``structured configs`` feature maps the
YAML values directly to these dataclasses via OmegaConf.

The **shared** configs (``ServerConfig``, ``MapConfig``, ``EntityConfig``,
``EgoVehicleConfig``, ``NpcVehicleConfig``, ``SweepConfig``) live in the public
:mod:`autoware_carla_scenario.scenario_config` module so that external scenario
packages can reuse them; import them from the top-level package::

    from autoware_carla_scenario import ServerConfig, MapConfig

This module defines only the *example-specific* scenario configs and the root
``ScenarioRunConfig`` (which composes the shared configs below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

# The shared configs are imported because the composite configs in this module
# (``ScenarioRunConfig`` and ``IntersectionPassingConfig``) reference them.
from ..scenario_config import (
    EgoVehicleConfig,
    EntityConfig,
    MapConfig,
    NpcVehicleConfig,
    ServerConfig,
    SweepConfig,
)

# ---------------------------------------------------------------------------
# Bounds shared by the scenario configs
# ---------------------------------------------------------------------------

#: The kinds of adversity an NPC carries out
#: (:class:`~autoware_carla_scenario.examples.adversity.AdversityScenario`).
ADVERSITY_VEHICLE_KINDS = ("brake_check", "ignore_leader", "cut_in", "traffic_rule")

#: Every ``AdversityConfig.kind``: the NPC's, ``jaywalking`` (a pedestrian's)
#: and ``nade``.  The config refuses another kind when it is built, and the
#: scenario's setup again, as a backstop.
ADVERSITY_KINDS = (
    "brake_check",
    "ignore_leader",
    "cut_in",
    "traffic_rule",
    "jaywalking",
    "nade",
)


#: Every ``CutInConfig.side``: the side of the ego the NPC comes from.  The
#: config refuses another side when it is built, and the scenario's setup
#: again, as a backstop.
CUT_IN_SIDES = ("left", "right")

#: Every ``IntersectionPassingConfig.turn_direction`` but ``None`` (straight
#: through, with no turn action): the ones the scenario's setup turns into a
#: ``TurnAction``.  The config refuses another when it is built, and the
#: setup again, as a backstop.
TURN_DIRECTIONS = ("left", "right", "straight")


def _require_number(key: str, value: Union[float, bool, str]) -> float:
    """*value* of ``scenario.<key>``, a number, as a ``float``.

    A config dataclass is not coerced, so a field typed ``float`` holds
    whatever its mapping did (#1818): Hydra reads ``=true`` as a bool, which
    would pass as 1, ``=null`` as None and a quoted ``='10'`` as a string,
    which ``math.isfinite()`` refuses with a ``TypeError`` naming no key.  An
    int (Hydra reads ``=10`` as one) or a float is a number; a bool, a string
    or anything ``float()`` refuses is not.

    Raises:
        ValueError: naming ``scenario.<key>``.
    """
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"scenario.{key} must be a number, got {value!r}")
    try:
        number = float(value)
    except TypeError:
        raise ValueError(f"scenario.{key} must be a number, got {value!r}") from None
    return number


def _require_finite(key: str, value: float) -> None:
    """Refuse *value* of ``scenario.<key>`` unless it is a finite number."""
    number = _require_number(key, value)
    if not math.isfinite(number):
        raise ValueError(f"scenario.{key} must be finite, not {value}")


def _require_positive(key: str, value: float) -> None:
    """Refuse *value* of ``scenario.<key>`` unless it is a number, finite and
    above 0.

    A threshold, distance or duration at 0 or below makes the check that
    reads it vacuous (#1815), or is refused only once CARLA is up (#1816);
    NaN fails every comparison, and so the ``<=`` too.
    """
    number = _require_number(key, value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"scenario.{key} must be finite and positive, not {value}")


def _require_not_negative(key: str, value: float) -> None:
    """Refuse *value* of ``scenario.<key>`` unless it is a number, finite and
    at least 0."""
    number = _require_number(key, value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"scenario.{key} must be finite and not negative, not {value}")


def _require_count(key: str, value: int) -> None:
    """Refuse *value* of ``scenario.<key>`` unless it is a whole number (an
    int, not a bool, which would pass as 1 or 0) of at least 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"scenario.{key} must be a whole number, got {value!r}")
    if value < 1:
        raise ValueError(f"scenario.{key} must be at least 1, not {value}")


# ---------------------------------------------------------------------------
# Scenario-specific configs
# ---------------------------------------------------------------------------


@dataclass
class IntersectionPassingConfig:
    """Parameters for the intersection-passing scenario.

    Supports straight-through, left-turn, and right-turn variants via the
    optional ``turn_direction`` field.
    """

    name: str = "intersection_passing"

    #: Ordered lanelets the ego is expected to traverse.
    expected_route_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])

    #: Pass only once the ego is on the route's last lanelet itself -- its
    #: lane, and so its direction -- not anywhere on that lanelet's road.  A
    #: turn onto the oncoming lane of the right road passes otherwise.
    pass_on_last_lanelet: bool = False

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 5.0

    #: Minimum speed (km/h) — scenario fails if ego drops below this.
    #: ``None`` disables the speed check.
    min_speed_kmh: float | None = None

    #: Grace period (seconds) before the minimum-speed check activates.
    #: Gives the ego time to reach cruising speed after spawn.
    speed_check_delay_seconds: float = 0.3

    #: Turn direction at the intersection (``"left"``, ``"right"``, or
    #: ``None`` for straight-through).
    turn_direction: str | None = None

    #: From this many metres before the junction (the start of the route's
    #: first lanelet, where its stop line is) until it is on that lanelet,
    #: the ego must show the turn indicator of ``turn_direction``; the run
    #: fails once it has not for ``turn_indicator_grace_seconds``, and as it
    #: enters the junction if it never showed it on that approach (one fast
    #: enough is through it before the grace runs out).  In the junction the
    #: indicator is ``max_turn_indicator_changes``' to judge.
    #: ``None`` checks nothing.  Needs a ``"left"`` or ``"right"`` turn.
    turn_indicator_before_m: float | None = None

    #: How long the indicator may be wrong before the run fails, in seconds:
    #: a blink of the command, not a turn without one.
    turn_indicator_grace_seconds: float = 0.5

    #: The run fails once the steering applied to the ego has turned back
    #: this many times (the ``steering_reversals`` metric, docs/metrics.md),
    #: counted from the start until the run passes.  ``None`` checks nothing.
    max_steering_reversals: int | None = None

    #: The run fails once the ego's turn indicators have changed this many
    #: times (the ``turn_indicator_changes`` metric), counted apart from the
    #: steering, from the start until the run passes.  ``None`` checks
    #: nothing.
    max_turn_indicator_changes: int | None = None

    #: Optional list of NPC vehicles to spawn in the scenario.
    #: Empty by default for backwards compatibility with existing YAML configs.
    npc_vehicles: list[NpcVehicleConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert raw dicts from OmegaConf into NpcVehicleConfig instances,
        and refuse a value the scenario would refuse or never act on, naming
        its key (#1815, #1816)."""
        npc_vehicles: list[NpcVehicleConfig] = []
        for i, v in enumerate(self.npc_vehicles):
            if not isinstance(v, dict) and not isinstance(v, NpcVehicleConfig):
                # A bare number or string would reach setup() as it is (#1984).
                raise ValueError(
                    f"scenario.npc_vehicles[{i}] must be a mapping of an NPC "
                    f"vehicle's settings, got {v!r}"
                )
            try:
                # The conditional expression, not an if statement: Codon
                # realises only the branch the static isinstance selects.
                npc_vehicles.append(NpcVehicleConfig(**v) if isinstance(v, dict) else v)
            except ValueError as exc:
                # Named by its key, as a run's other refusals are (#1809).
                raise ValueError(f"scenario.npc_vehicles[{i}].{exc}") from None
            except TypeError as exc:
                # A key NpcVehicleConfig does not take.
                raise ValueError(f"scenario.npc_vehicles[{i}]: {exc}") from None
        self.npc_vehicles = npc_vehicles
        _require_positive("timeout_seconds", self.timeout_seconds)
        min_speed_kmh = self.min_speed_kmh
        if min_speed_kmh is not None:
            # At 0 or below the ego is never slower: the check never fires.
            _require_positive("min_speed_kmh", min_speed_kmh)
        # ElapsedTimeCondition's bound, whether or not min_speed_kmh is set.
        _require_positive("speed_check_delay_seconds", self.speed_check_delay_seconds)
        turn_direction = self.turn_direction
        if turn_direction is not None and turn_direction not in TURN_DIRECTIONS:
            # The setup's TurnAction knows no other (a bare KeyError, #1818).
            raise ValueError(
                f"scenario.turn_direction must be one of "
                f"{', '.join(TURN_DIRECTIONS)} or null, not {turn_direction!r}"
            )
        turn_indicator_before_m = self.turn_indicator_before_m
        if turn_indicator_before_m is not None:
            _require_finite("turn_indicator_before_m", turn_indicator_before_m)
        if self.turn_indicator_before_m is not None and self.turn_direction not in (
            "left",
            "right",
        ):
            raise ValueError(
                "scenario.turn_indicator_before_m needs scenario.turn_direction "
                f"left or right, not {self.turn_direction!r}"
            )
        _require_positive(
            "turn_indicator_grace_seconds", self.turn_indicator_grace_seconds
        )
        for name, limit in (
            ("max_steering_reversals", self.max_steering_reversals),
            ("max_turn_indicator_changes", self.max_turn_indicator_changes),
        ):
            if limit is not None:
                _require_count(name, limit)
        if (
            self.turn_indicator_before_m is not None
            and self.max_turn_indicator_changes is not None
            and self.max_turn_indicator_changes < 3
        ):
            # Its own on and off are two changes of a turn that signals.
            raise ValueError(
                "scenario.max_turn_indicator_changes must be at least 3 with "
                "scenario.turn_indicator_before_m: the turn's own indicator on "
                "and off are two changes"
            )


@dataclass
class TrafficLightComplianceConfig:
    """Parameters for the traffic-light-compliance scenario."""

    name: str = "traffic_light_compliance"

    #: Delay (seconds) before switching traffic lights from red to green.
    light_switch_delay_seconds: float = 3.0

    #: Fail-safe timeout in seconds: the scenario's own timeout, so the run's
    #: default timeout is not added on top of it.  Ignored when
    #: ``timeout_after_green_seconds`` is set -- a ``scenario.timeout_seconds=``
    #: override then no longer caps the run.
    timeout_seconds: float = 5.0

    #: Speed (km/h) the ego must reach after green when
    #: ``departure_speed_kmh`` is unset.  Moving on red no longer fails a run
    #: by itself: crossing the stop line does (``stop_zone_min_m``).
    moving_speed_kmh: float = 1.0

    #: Speed (km/h) the ego must reach after the light turns green for the run
    #: to pass, before ``timeout_seconds``. ``None`` uses ``moving_speed_kmh``,
    #: which passes a mere creep; set it higher to require a real departure.
    departure_speed_kmh: float | None = None

    #: Deprecated and unused: the run no longer fails on the ego moving on red
    #: after this many seconds, but on its front bumper crossing the stop line.
    #: Kept so a config that sets it still loads.
    merging_time_seconds: float = 0.5

    #: The stop zone: where the ego's front bumper has to stand still before
    #: the light turns green, as metres before the stop line
    #: (:data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M`), from
    #: ``stop_zone_min_m`` to ``stop_zone_max_m``.  Standing still farther
    #: back until the green fails the run.
    stop_zone_min_m: float = 0.0

    #: The far end of the stop zone (``stop_zone_min_m``).
    stop_zone_max_m: float = 3.0

    #: How far (metres) the front bumper may pass the stop line while the
    #: light is red before the run fails for running the red.  Up to this far
    #: past it counts as at the line (0 m) for the stop zone.
    stop_line_tolerance_m: float = 0.0

    #: Seconds the ego has to stand still (at ``standstill_speed_mps`` or
    #: less, as :class:`~autoware_carla_scenario.StandstillCondition` counts
    #: it) in the stop zone before the green for it to count as stopped there.
    #: Once it has, it may move again (short of crossing the line on red).
    standstill_seconds: float = 0.5

    #: The speed (m/s) at or below which the ego stands still
    #: (:class:`~autoware_carla_scenario.StandstillCondition`'s
    #: ``speed_threshold``).
    standstill_speed_mps: float = 0.1

    #: Where the ego starts, as
    #: :data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M` reads it:
    #: metres from its front bumper to the stop line ahead, along its lane.
    #: The ego is moved along the lane its spawn pose is on to read it
    #: (:func:`~autoware_carla_scenario.place_at_stop_line_distance`).
    #: ``None`` leaves the spawn pose as it is.
    stop_line_distance_m: float | None = None

    #: The fail-safe timeout counted from the green light instead:
    #: ``light_switch_delay_seconds`` plus this, in seconds, in place of
    #: ``timeout_seconds``, so a sweep that draws the delay keeps the time
    #: the ego has to pull away.  ``None`` uses ``timeout_seconds``.
    #:
    #: When set, the run lasts until this limit however short
    #: ``timeout_seconds`` is: the scenario registers its own timeout, so
    #: ``ScenarioRunner`` adds no default, and neither ``timeout_seconds`` (a
    #: ``scenario.timeout_seconds=`` override included) nor a
    #: ``run_scenario``/``ScenarioQueue.add`` timeout caps it.  A sweep's
    #: ``sweep.job_timeout_seconds`` still guards each job on the wall clock.
    timeout_after_green_seconds: float | None = None

    def __post_init__(self) -> None:
        """Refuse a value the scenario would refuse once CARLA is up, or one
        that passes an ego standing still at the green, naming its key
        (#1815, #1816)."""
        _require_positive("timeout_seconds", self.timeout_seconds)
        # The green's ElapsedTimeCondition.
        _require_positive("light_switch_delay_seconds", self.light_switch_delay_seconds)
        # At 0 or below an ego standing still has "reached" it.
        _require_positive("moving_speed_kmh", self.moving_speed_kmh)
        departure_speed_kmh = self.departure_speed_kmh
        if departure_speed_kmh is not None:
            _require_positive("departure_speed_kmh", departure_speed_kmh)
        timeout_after_green_seconds = self.timeout_after_green_seconds
        if timeout_after_green_seconds is not None:
            # Below 0 the run times out before the light turns green.
            _require_positive(
                "timeout_after_green_seconds", timeout_after_green_seconds
            )
        # StopLineWatch's, which takes a NaN zone that nothing is ever in.
        for key, value in (
            ("stop_zone_min_m", self.stop_zone_min_m),
            ("stop_zone_max_m", self.stop_zone_max_m),
        ):
            _require_finite(key, value)
        if self.stop_zone_min_m > self.stop_zone_max_m:
            raise ValueError(
                f"scenario.stop_zone_min_m ({self.stop_zone_min_m}) is past "
                f"scenario.stop_zone_max_m ({self.stop_zone_max_m})"
            )
        _require_not_negative("stop_line_tolerance_m", self.stop_line_tolerance_m)
        # The StandstillCondition's.
        _require_positive("standstill_seconds", self.standstill_seconds)
        _require_not_negative("standstill_speed_mps", self.standstill_speed_mps)


@dataclass
class LaneChangeConfig:
    """Parameters for the lane-change scenario."""

    name: str = "lane_change"

    #: Direction of the lane change (``"left"`` or ``"right"``).
    direction: str = "left"

    #: Expected outcome: ``"success"`` expects the lane change to complete,
    #: ``"fail"`` expects it to NOT complete (scenario passes if lane change
    #: does not happen within the timeout).
    expect: str = "success"

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Refuse an unknown ``direction`` or ``expect``, or a timeout that is
        not positive, naming its key.

        ``lane_change.py`` checks the failure outcome only for ``"fail"``, so
        without this a typo (``"fial"``) would check the opposite outcome.
        """
        if self.direction not in ("left", "right"):
            raise ValueError(
                "scenario.direction must be 'left' or 'right', "
                f"not {self.direction!r}"
            )
        if self.expect not in ("success", "fail"):
            raise ValueError(
                f"scenario.expect must be 'success' or 'fail', not {self.expect!r}"
            )
        _require_positive("timeout_seconds", self.timeout_seconds)


@dataclass
class PedestrianDartOutConfig:
    """Parameters for the pedestrian dart-out scenario."""

    name: str = "pedestrian_dart_out"

    #: How far ahead of the ego's spawn, along its lanelet, the pedestrian
    #: waits at the kerb (m).
    pedestrian_ahead_m: float = 35.0

    #: The pedestrian's distance from the lane's centreline, on its kerb side
    #: (m): the right of the direction of travel in right-hand traffic, the
    #: left in left-hand traffic, by the OpenDRIVE road's ``rule``.  Half a
    #: lane and a little more puts it at the kerb of a lane at the road's edge.
    #: The sign is ignored (``-3.0``, the old right-hand value, is 3 m too).
    pedestrian_offset_m: float = 3.0

    #: The pedestrian steps out once the ego is this close to it (m).
    trigger_distance_m: float = 20.0

    #: Walking speed into the road (m/s); a run, not a stroll.
    walk_speed_ms: float = 2.0

    #: The pedestrian turns back when a vehicle's time to collision with it,
    #: along the road, is at most this (s); ``None``: not by TTC.  See
    #: :class:`~autoware_carla_scenario.actions.WalkStraightAction`.
    turn_back_ttc_s: Optional[float] = None

    #: The pedestrian turns back when a vehicle is at most this far from it
    #: along the road (m); ``None``: not by distance.
    turn_back_distance_m: Optional[float] = None

    #: How far onto the road, from the kerb, the pedestrian may be and still
    #: turn back (m).
    turn_back_within_m: float = 1.0

    #: CARLA blueprint of the pedestrian: one CARLA 0.9 and 0.10 both ship.
    walker_type: str = "walker.pedestrian.0015"

    #: The run passes once the ego is this far past the pedestrian's crossing
    #: line, on its own lanelet (m).
    pass_beyond_m: float = 10.0

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 25.0

    def __post_init__(self) -> None:
        """Refuse a value the scenario would refuse once CARLA is up, or one
        with which the pedestrian never darts out, naming its key (#1815,
        #1816)."""
        # A distance (unsigned) never below 0: the trigger would never fire.
        _require_positive("trigger_distance_m", self.trigger_distance_m)
        # At 0 the pedestrian stays at the kerb.
        _require_positive("walk_speed_ms", self.walk_speed_ms)
        # Below 0 the pass point lies before the crossing.
        _require_not_negative("pass_beyond_m", self.pass_beyond_m)
        # WalkStraightAction's.
        turn_back_ttc_s = self.turn_back_ttc_s
        if turn_back_ttc_s is not None:
            _require_not_negative("turn_back_ttc_s", turn_back_ttc_s)
        turn_back_distance_m = self.turn_back_distance_m
        if turn_back_distance_m is not None:
            _require_not_negative("turn_back_distance_m", turn_back_distance_m)
        _require_not_negative("turn_back_within_m", self.turn_back_within_m)
        _require_positive("timeout_seconds", self.timeout_seconds)


@dataclass
class CutInConfig:
    """Parameters for the cut-in scenario."""

    name: str = "cut_in"

    #: The side of the ego the NPC comes from (``"left"`` or ``"right"``).
    side: str = "left"

    #: The lanelet beside the ego's spawn lanelet the NPC starts on.  A sweep
    #: binds it to the lane beside the pick (``adjacent`` binding).
    npc_lanelet_id: int = 0

    #: How far ahead of the ego the NPC starts, along its lanelet (m): its
    #: ``s`` less the ego's.  A raw setting -- the lanelets beside each other
    #: need not be the same length, so on a curve this is not the gap measured.
    npc_ahead_m: float = 10.0

    #: The gap the NPC starts at, as ``vehicle_ahead_gap_m`` measures it (m):
    #: bumper to bumper along the ego's lane.  When given it replaces
    #: :attr:`npc_ahead_m`, and the NPC's ``s`` is solved for so the measure
    #: reads exactly this at the start (``place_at_vehicle_ahead_gap``).
    npc_gap_m: float | None = None

    #: NPC blueprint and starting speed.
    npc_vehicle_type: str = "vehicle.mini.cooper"
    npc_initial_speed_kmh: float = 30.0

    #: The NPC changes into the ego's lane this long after the start (s).
    cut_in_after_seconds: float = 2.0

    #: The run passes once the NPC has been on the ego's lane this long
    #: without a collision (s).
    hold_seconds: float = 3.0

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 20.0

    #: Fail when the ego leaves the lanes ahead of it: its spawn lanelet and
    #: the lanelets that follow it (:func:`lanelets_ahead`) -- a lane change,
    #: a turn onto a lanelet that does not follow it, or leaving the road.
    fail_on_leaving_lane: bool = True

    #: How far ahead of the start of the spawn lanelet the lanes ahead reach
    #: (m), the ones the ego is to stay on and the ones the NPC cuts in on
    #: (those only up to the first junction).  The ego's never include a
    #: lanelet beside its lane, however far they loop round a block.
    #: Further than the ego can drive before the timeout.
    lanes_ahead_m: float = 1000.0

    #: How long the ego has to be off those lanes before the run fails (s):
    #: a tick or two off at a seam between lanelets is not leaving them.
    leaving_lane_seconds: float = 0.2

    #: Fail on any collision of the ego, with an actor or a static object;
    #: ``False`` fails only on a collision with the cut-in vehicle.
    fail_on_any_collision: bool = True

    def __post_init__(self) -> None:
        """Refuse a value the scenario would refuse once CARLA is up, naming
        its key (#1816, #1818)."""
        if self.side not in CUT_IN_SIDES:
            raise ValueError(
                f"scenario.side must be one of {', '.join(CUT_IN_SIDES)}, "
                f"not {self.side!r}"
            )
        # VehicleEntityConfig's and SetSpeedAction's.
        _require_not_negative("npc_initial_speed_kmh", self.npc_initial_speed_kmh)
        # The cut-in's ElapsedTimeCondition, and the PersistentCondition held.
        _require_positive("cut_in_after_seconds", self.cut_in_after_seconds)
        _require_positive("hold_seconds", self.hold_seconds)
        _require_positive("timeout_seconds", self.timeout_seconds)


@dataclass
class TemporaryStopConfig:
    """Parameters for the temporary-stop scenario."""

    name: str = "temporary_stop"

    #: Arc-length margin (m) around the stop position.
    s_margin: float = 5.0

    #: Maximum speed (m/s) considered as stopped.
    speed_threshold: float = 0.1

    #: Minimum consecutive seconds the entity must remain stopped.
    stop_duration: float = 1.0

    #: Speed (km/h) above which the ego is considered to have restarted.
    restart_speed_kmh: float = 3.0

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Refuse a value the scenario would refuse once CARLA is up, or one
        that passes an ego which never pulls away, naming its key (#1815,
        #1816)."""
        # TemporaryStopCondition's.
        _require_positive("s_margin", self.s_margin)
        _require_not_negative("speed_threshold", self.speed_threshold)
        _require_positive("stop_duration", self.stop_duration)
        # At 0 or below an ego standing still has restarted.
        _require_positive("restart_speed_kmh", self.restart_speed_kmh)
        _require_positive("timeout_seconds", self.timeout_seconds)


# ---------------------------------------------------------------------------
# Top-level Hydra config
# ---------------------------------------------------------------------------


@dataclass
class AdversityConfig:
    """Parameters for the adversity examples: one of TeraSim's phase-1
    adversities carried out by an NPC (or a pedestrian) placed to make it
    fire, or NADE among background traffic (#404).

    The shipped cases (``conf/scenario/adversity``) are placed on Town10HD_Opt
    (``map=town10hd_opt``).
    """

    name: str = "adversity"

    #: What is carried out: ``brake_check``, ``ignore_leader``, ``cut_in``,
    #: ``traffic_rule``, ``jaywalking`` (by one NPC or pedestrian), or ``nade``.
    kind: str = "brake_check"

    #: The lanelet of the NPC (or the pedestrian's kerb).
    npc_lanelet_id: int = 324

    #: How far ahead of the ego's spawn ``s`` the NPC starts, along its
    #: lanelet (m; negative: behind).
    npc_ahead_m: float = 15.0

    #: The pedestrian's offset from its lanelet's centreline (m, negative to
    #: the right of the direction of travel): at the kerb.
    npc_offset_m: float = -3.0

    #: The NPC's speed, held until the adversity takes it over (km/h).
    npc_speed_kmh: float = 30.0

    #: The NPC's blueprint.
    npc_vehicle_type: str = "vehicle.mini.cooper"

    #: The pedestrian's blueprint.
    walker_type: str = "walker.pedestrian.0015"

    #: The ego's speed, held for the run (km/h); ``None``: the traffic
    #: manager's own.
    ego_speed_kmh: Optional[float] = None

    #: Every traffic light red (``traffic_rule``: the NPC stops for it, then
    #: runs it); green otherwise.
    red_light: bool = False

    #: The run fails when the adversity has not been carried out by then (s).
    timeout_seconds: float = 30.0

    #: ``nade``: how long the run lasts (s).
    nade_seconds: float = 45.0

    #: ``nade``: each kind's naturalistic probability a tick when its trigger
    #: holds.
    nade_probability: float = 0.05

    #: ``nade``: the importance-sampling probability every adversity is drawn
    #: with (TeraSim's is far smaller, so a short run shows several).
    nade_is_probability: float = 0.5

    #: ``nade``: how long after a sampled adversity none is (s).
    nade_cooldown: float = 2.0

    #: ``nade``: an adversity counts when it challenges another NPC, not only
    #: the ego.
    nade_challenge_npcs: bool = True

    #: ``nade``: the sampler's seed; ``None`` follows the run's ``seed``, as
    #: the scenario's other adversarial actions do.
    nade_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Refuse an unknown kind, or a number outside its bounds, naming its
        key (#1458, #1816).

        As :class:`~autoware_carla_scenario.NadeSettings`, the adversarial
        traffic action (which refuses a ``probability`` outside [0, 1]), the
        NPC's ``VehicleEntityConfig``, ``SetSpeedAction``,
        ``TimeoutCondition`` and the scenario's setup (an unknown ``kind``)
        would once the scenario is set up, but when the config is built: a
        run checks it before anything starts
        (``examples.run.check_batch_configs``).
        """
        if self.kind not in ADVERSITY_KINDS:
            raise ValueError(
                f"scenario.kind must be one of {', '.join(ADVERSITY_KINDS)}, "
                f"not {self.kind!r}"
            )
        _require_not_negative("npc_speed_kmh", self.npc_speed_kmh)
        ego_speed_kmh = self.ego_speed_kmh
        if ego_speed_kmh is not None:
            _require_not_negative("ego_speed_kmh", ego_speed_kmh)
        _require_positive("timeout_seconds", self.timeout_seconds)
        for key, value in (
            ("nade_probability", self.nade_probability),
            ("nade_is_probability", self.nade_is_probability),
        ):
            if not (0.0 <= _require_number(key, value) <= 1.0):  # NaN fails it too
                raise ValueError(f"scenario.{key} must be in [0, 1], not {value}")
        _require_not_negative("nade_cooldown", self.nade_cooldown)
        _require_positive("nade_seconds", self.nade_seconds)


@dataclass
class ScenarioRunConfig:
    """Root configuration assembled by Hydra.

    ``defaults`` in ``conf/config.yaml`` populate the ``server``, ``map``,
    ``ego``, and ``scenario`` sub-trees automatically.
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    map: MapConfig = field(default_factory=MapConfig)
    ego: EgoVehicleConfig = field(default_factory=EgoVehicleConfig)
    entity: EntityConfig = field(default_factory=EntityConfig)
    scenario: (
        IntersectionPassingConfig
        | LaneChangeConfig
        | TrafficLightComplianceConfig
        | TemporaryStopConfig
        | PedestrianDartOutConfig
        | CutInConfig
        | AdversityConfig
    ) = field(default_factory=IntersectionPassingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    #: The run's seed, which every random draw of the run follows
    #: (:mod:`autoware_carla_scenario.seed`).
    seed: int = 0
    #: Static check before the run: ``auto``, ``required`` or ``off``
    #: (:mod:`autoware_carla_scenario.typecheck.mode`).
    typecheck: str = "auto"
