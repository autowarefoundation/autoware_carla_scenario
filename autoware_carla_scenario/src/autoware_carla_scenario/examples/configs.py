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

from dataclasses import dataclass, field

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

    #: Optional list of NPC vehicles to spawn in the scenario.
    #: Empty by default for backwards compatibility with existing YAML configs.
    npc_vehicles: list[NpcVehicleConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert raw dicts from OmegaConf into NpcVehicleConfig instances."""
        self.npc_vehicles = [
            NpcVehicleConfig(**v) if isinstance(v, dict) else v
            for v in self.npc_vehicles
        ]


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
    #: light is red before the run fails for running the red.
    stop_line_tolerance_m: float = 0.0

    #: Seconds the ego has to stand still (at 0.1 m/s or less, as
    #: :class:`~autoware_carla_scenario.StandstillCondition` counts it) in the
    #: stop zone before the green for it to count as stopped there.
    standstill_seconds: float = 0.5

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


@dataclass
class PedestrianDartOutConfig:
    """Parameters for the pedestrian dart-out scenario."""

    name: str = "pedestrian_dart_out"

    #: How far ahead of the ego's spawn, along its lanelet, the pedestrian
    #: waits at the kerb (m).
    pedestrian_ahead_m: float = 35.0

    #: The pedestrian's offset from the lane's centreline (m, negative to the
    #: right of the direction of travel): half a lane and a little more puts
    #: it at the kerb of the rightmost lane.
    pedestrian_offset_m: float = -3.0

    #: The pedestrian steps out once the ego is this close to it (m).
    trigger_distance_m: float = 20.0

    #: Walking speed into the road (m/s); a run, not a stroll.
    walk_speed_ms: float = 2.0

    #: CARLA blueprint of the pedestrian: one CARLA 0.9 and 0.10 both ship.
    walker_type: str = "walker.pedestrian.0015"

    #: The run passes once the ego is this far past the pedestrian's crossing
    #: line, on its own lanelet (m).
    pass_beyond_m: float = 10.0

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 25.0


@dataclass
class CutInConfig:
    """Parameters for the cut-in scenario."""

    name: str = "cut_in"

    #: The side of the ego the NPC comes from (``"left"`` or ``"right"``).
    side: str = "left"

    #: The lanelet beside the ego's spawn lanelet the NPC starts on.  A sweep
    #: binds it to the pick's neighbour (``neighbour`` binding).
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
    #: (m).  Further than the ego can drive before the timeout.
    lanes_ahead_m: float = 1000.0

    #: How long the ego has to be off those lanes before the run fails (s):
    #: a tick or two off at a seam between lanelets is not leaving them.
    leaving_lane_seconds: float = 0.2

    #: Fail on any collision of the ego, with an actor or a static object;
    #: ``False`` fails only on a collision with the cut-in vehicle.
    fail_on_any_collision: bool = True


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


# ---------------------------------------------------------------------------
# Top-level Hydra config
# ---------------------------------------------------------------------------


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
    ) = field(default_factory=IntersectionPassingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    #: Static check before the run: ``auto``, ``required`` or ``off``
    #: (:mod:`autoware_carla_scenario.typecheck.mode`).
    typecheck: str = "auto"
