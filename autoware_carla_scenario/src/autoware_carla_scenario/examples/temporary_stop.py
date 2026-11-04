"""Example scenario: the ego stops at a stop line, then drives on.

The ego is spawned on ``ego.spawn_lanelet_id``. Without expanding, that is
lanelet 285 of the Nishi-Shinjuku map, the default in
``conf/scenario/temporary_stop/temporary_stop.yaml``. The YAML's ``sweep:``
expands one case per stop line that no traffic light governs (a stop sign's),
skipping the map's lanelets with no 3D model; each case replaces
``ego.spawn_lanelet_id``, ``ego.spawn_s`` (15 m short of the line, on the
lanelet before when the line is nearer its lanelet's start than that) and
``ego.goal_lanelet_id`` (the lanelet past the line).

The stop lines are not configured: they are read off the map with
:func:`~autoware_carla_scenario.get_stop_line_poses_with_following`, from the
spawn lanelet and up to three lanelets after it (``depth=3``), each projected
onto the lanelet that owns it. Setup raises :class:`ValueError` if none is
found.

The pass condition is a stop *followed by* a restart, both at once:

* a latched (:class:`~autoware_carla_scenario.StickyCondition`)
  :class:`~autoware_carla_scenario.TemporaryStopCondition`: the ego has been
  within ``s_margin`` metres of one of the stop lines, on its OpenDRIVE road,
  below ``speed_threshold`` m/s for at least ``stop_duration`` seconds; **and**
* a :class:`~autoware_carla_scenario.SpeedCondition`: the ego is now moving at
  ``restart_speed_kmh`` km/h or more.

The scenario fails once ``timeout_seconds`` pass.

Typical usage
-------------
The unexpanded case (lanelet 285)::

    uv run scenario scenario=temporary_stop/temporary_stop

Every case on a map, one run each::

    uv run scenario --multirun hydra/sweeper=lanelet_constraint \\
        scenario=temporary_stop/temporary_stop map=<map>
"""

from __future__ import annotations

import logging

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AndCondition,
    BaseScenario,
    ComparisonRule,
    EgoConfig,
    GroundProjectionConfig,
    Lanelet2Pose,
    SpeedCondition,
    StickyCondition,
    TemporaryStopCondition,
    TimeoutCondition,
    TrafficLightTarget,
    TrafficSignalAction,
    get_stop_line_poses_with_following,
    to_opendrive,
)

from .configs import TemporaryStopConfig

logger = logging.getLogger(__name__)


class TemporaryStopScenario(BaseScenario):
    """Spawn the ego and verify it stops at a stop line, then drives on.

    The scenario:

    1. Snaps the Lanelet2 spawn pose to the CARLA road surface.
    2. Sets every traffic light to green.
    3. Finds the stop lines on the spawn lanelet and up to three lanelets
       after it (:func:`get_stop_line_poses_with_following`, ``depth=3``),
       raising :class:`ValueError` if there are none.
    4. Registers the pass condition: a :class:`StickyCondition`-latched
       :class:`TemporaryStopCondition` -- within ``s_margin`` m of a stop line,
       below ``speed_threshold`` m/s for ``stop_duration`` s -- AND a
       :class:`SpeedCondition` at ``restart_speed_kmh`` km/h or more. A stop
       alone does not pass; the ego has to restart after it.
    5. Registers a :class:`TimeoutCondition` of ``timeout_seconds`` as the
       fail condition.

    The ego is expected to see the stop line and make the stop on its own.
    """

    _config: TemporaryStopConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: TemporaryStopConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or TemporaryStopConfig()

    def setup(self) -> None:
        """Snap ego spawn to CARLA road, register temporary stop condition."""
        cfg = self._config

        # --- Compute ego spawn from Lanelet2Pose via OpenDrivePose ---
        self._setup_ego_spawn()
        assert self._spawn_pose is not None  # narrow type for mypy

        # --- Set all traffic lights to green ---
        self.register_init(
            TrafficSignalAction(
                state=carla.TrafficLightState.Green,
                lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
                label="set_all_green",
            )
        )

        # --- Auto-detect stop line from spawn lanelet + following lanelets ---
        spawn_lanelet_id = self._spawn_pose.lanelet_id
        # A swept spawn is walked back from the stop line and can sit more
        # than one lanelet short of it.
        stop_poses = get_stop_line_poses_with_following(spawn_lanelet_id, depth=3)
        if not stop_poses:
            msg = (
                f"No stop lines found on lanelet {spawn_lanelet_id} "
                "or its following lanelets."
            )
            raise ValueError(msg)
        for pose in stop_poses:
            od = to_opendrive(pose)
            logger.info(
                "Stop line detected: lanelet %d s=%.1f -> OpenDRIVE road='%s' s=%.1f",
                pose.lanelet_id,
                pose.s,
                od.road_id,
                od.s,
            )

        # --- Pass condition: temporary stop + restart ---
        # 1) Sticky: latches once ego has temporarily stopped at the position
        stopped = StickyCondition(
            TemporaryStopCondition(
                entity_name=EGO_ROLE_NAME,
                stop_positions=stop_poses,
                s_margin=cfg.s_margin,
                speed_threshold=cfg.speed_threshold,
                stop_duration=cfg.stop_duration,
                label="temporary_stop",
            )
        )
        # 2) Ego has restarted (speed exceeds threshold)
        restarted = SpeedCondition(
            entity_name=EGO_ROLE_NAME,
            value=cfg.restart_speed_kmh / 3.6,
            rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
            label="ego_restart_speed",
        )
        self.register_pass_condition(AndCondition([stopped, restarted]))

        # --- Fail-safe timeout ---
        self.register_fail_condition(
            TimeoutCondition(cfg.timeout_seconds, label="scenario_timeout")
        )

    def is_done(self) -> bool:
        """Always ``False`` -- termination is driven by pass/fail conditions."""
        return False
