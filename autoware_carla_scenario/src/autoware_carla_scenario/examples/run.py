"""Hydra-based unified entry point for all example scenarios.

Usage examples::

    # Run intersection-passing scenario (straight-through)
    uv run scenario scenario=intersection_passing/straight

    # Run left-turn variant (uses intersection_passing with turn_direction=left)
    uv run scenario scenario=intersection_passing/left_turn

    # Run traffic-light-compliance scenario
    uv run scenario scenario=traffic_light_compliance/traffic_light_compliance

    # Run all intersection-passing variants in a single batch
    uv run scenario scenario='intersection_passing/*'

    # Glob patterns also work with ? and [

    # Run every case of their sweeps (lanelets, routes, ODD samples) in one batch
    uv run scenario --multirun 'scenario=cut_in/*' +sweep.odd_sample.count=10
    uv run scenario scenario='intersection_passing/left_*'

    # Select a different map
    uv run scenario scenario=intersection_passing/straight map=nishishinjuku

    # Override individual parameters (scenario.timeout_seconds is the run's
    # default timeout and whatever the scenario's own reads it; it does not
    # cap a scenario whose own timeout comes from another key, e.g.
    # traffic_light_compliance with timeout_after_green_seconds set)
    uv run scenario scenario=intersection_passing/straight scenario.timeout_seconds=15.0

    # Override server connection
    uv run scenario scenario=intersection_passing/straight server.host=192.168.1.100 server.port=3000
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import signal
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import typesafe_carla.carla as carla
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, ListConfig, OmegaConf

from autoware_carla_scenario import (
    BaseScenario,
    EgoConfig,
    EgoVehicle,
    ElapsedTimeCondition,
    GroundProjectionConfig,
    Lanelet2Pose,
    ScenarioQueue,
    ServerConfig,
    SpawnTransform,
    TrafficSinkAction,
    TrafficSourceAction,
)
import autoware_carla_scenario.hydra_compat  # noqa: F401  (Python 3.14 argparse)
from autoware_carla_scenario.actions.background_traffic import parse_constraints
from autoware_carla_scenario.autoware_stack.launcher import (
    DEFAULT_BRIDGE_PARAMETERS,
    AutowareLauncher,
)
from autoware_carla_scenario.batch_report import print_summary as _print_summary
from autoware_carla_scenario.batch_report import write_batch_result_json
from autoware_carla_scenario.batch_resume import (
    KeptResult,
    ResumedBatch,
    ResumeIntoError,
    read_resumed_batch,
)
from autoware_carla_scenario.canvas import use_canvas
from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.config_flags import check_flag
from autoware_carla_scenario import crash_report
from autoware_carla_scenario.constants import DEFAULT_TM_PORT
from autoware_carla_scenario.gate.errored import (
    remove_case_errored,
    write_case_errored,
    write_case_unfinished,
)
from autoware_carla_scenario.maps import resolve_map_paths
from autoware_carla_scenario.maps.pack import MapPack, packs_dir
from autoware_carla_scenario.maps.config import map_setting
from autoware_carla_scenario.odd.route import PlannedRoute, RouteError
from autoware_carla_scenario.odd.sampler import (
    ODD_CASE_KEY,
    OddCase,
    check_sample_settings,
    decode_case,
    seed_setting,
    whole_number_setting,
)
from autoware_carla_scenario.prepare.pool import BackendPool
from autoware_carla_scenario.rerun import (
    RerunConversion,
    RerunConverter,
    RerunUnavailableError,
)
from autoware_carla_scenario.rerun import report as rerun_report
from autoware_carla_scenario.rerun_report import rerun_errors as _rerun_errors
from autoware_carla_scenario.coordinate.snap import check_ray_distances
from autoware_carla_scenario.run_dirs import (
    _batch_output_dir_refusal as _batch_output_dir_refusal,
    _remove_if_empty as _remove_if_empty,
    asks_for_no_job,
    is_multirun_mode_override as _is_multirun_mode_override,
    lanelet_sweeper_requested_in_argv as _lanelet_sweeper_requested_in_argv,
    multirun_requested_in_argv as _multirun_requested_in_argv,
    reserve_timestamped_dir,
    sets_hydra_key as _sets_hydra_key,
)

# _lanelet_sweeper_requested_in_argv() and _result_json_line() are
# run_dirs' and run_outputs', which the library check compiles, as are
# _batch_output_dir_refusal() and _remove_if_empty() (#2306); RESULT_SUFFIX
# and those two stay importable from here, as they were.
from autoware_carla_scenario.run_outputs import (
    RESULT_SUFFIX,  # noqa: F401
    remove_run_outputs,
    result_json_line as _result_json_line,
)
from autoware_carla_scenario.run_timeout import run_timeout_seconds
from autoware_carla_scenario.scenario_config import EgoVehicleConfig, EntityConfig
from autoware_carla_scenario.scenario_hash import (
    SCENARIO_SECTIONS,
    SCENARIO_SETTINGS,
    scenario_hash,
)
from autoware_carla_scenario.seed import DEFAULT_SEED, MAX_SEED, set_run_seed
from autoware_carla_scenario.server import fog_density_scale_args
from autoware_carla_scenario.t4 import T4Recorder, T4Settings, T4UnavailableError
from autoware_carla_scenario.t4_scene import T4SceneExporter
from autoware_carla_scenario.typecheck import ScenarioTypeError
from autoware_carla_scenario.typecheck.mode import (
    check_odd,
    check_registered_scenario,
    typecheck_mode,
)
from autoware_carla_scenario.traffic.base import check_background_speed
from autoware_carla_scenario.traffic.config import traffic_backend_name
from autoware_carla_scenario.traffic.registry import takes_null_options, tm_port_option
from autoware_carla_scenario.traffic import (
    TrafficBackend,
    TrafficConfig,
    available_backends,
    build_backend,
    load_traffic_backend_plugins,
)
from autoware_carla_scenario.registry import (
    BuildScenarioFn,
    _register_builtin_scenario,
    get_conf_dirs,
    get_scenario_builder,
    get_scenario_classes,
    get_scenario_registry,
    load_scenario_plugins,
    register_conf_dir,
)
from autoware_carla_scenario.scenario_glob import resolve_scenario_glob, scenario_dirs
from autoware_carla_scenario.scenario_runner import _run_file_stem
from autoware_carla_scenario.slurm.stems import sequential_stems
from autoware_carla_scenario.value_text import shown

from .argv import (
    _RUN_DIR_KEY,
    _SWEEP_DIR_KEY,
    BatchCase,
    _arena_max_set_by_user,
    _batch_dir_text,
    _case_labels,
    _errored_case,
    _extract_resume_from,
    _extract_resume_into,
    _extract_scenario_override,
    _is_glob_pattern,
    _plain_overrides,
    _split_batch_output_dir,
    _sweep_failed_lines,
)
from .configs import (
    IntersectionPassingConfig,
    LaneChangeConfig,
    TemporaryStopConfig,
    PedestrianDartOutConfig,
    CutInConfig,
    AdversityConfig,
    TrafficLightComplianceConfig,
)
from .adversity import AdversityScenario
from .intersection_passing import IntersectionPassingScenario
from .lane_change import LaneChangeScenario
from .temporary_stop import TemporaryStopScenario
from .pedestrian_dart_out import PedestrianDartOutScenario
from .cut_in import CutInScenario
from .traffic_light_compliance import TrafficLightComplianceScenario

if TYPE_CHECKING:
    from autoware_carla_scenario import AutowareBridgeConfig
    from autoware_carla_scenario.driver import ControlConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------
#
# The registry itself (``register_scenario``, ``register_scenario_builder``,
# ``get_scenario_registry``, the conf-dir registry, and entry-point plugin
# discovery) lives in :mod:`autoware_carla_scenario.registry` (and is
# re-exported from the top-level package) so it can be imported without pulling
# in CARLA.  This runner only imports the pieces it actually uses.

# Directory containing the built-in Hydra config files (conf/ next to this
# module).  Registered so it becomes the primary entry in the search path.
_CONF_DIR = Path(__file__).resolve().parent / "conf"
register_conf_dir(_CONF_DIR)


# Register built-in scenarios.  As built-ins: a scenario plugin that replaced
# one before this module was imported (the viewer and ``scenario-expand`` load
# the plugins first) keeps it, and the plugin walk does not take a built-in an
# import of this module registers for the importing plugin's own (#2034).
_register_builtin_scenario(
    "intersection_passing", IntersectionPassingScenario, IntersectionPassingConfig
)
_register_builtin_scenario(
    "traffic_light_compliance",
    TrafficLightComplianceScenario,
    TrafficLightComplianceConfig,
)
_register_builtin_scenario("lane_change", LaneChangeScenario, LaneChangeConfig)
_register_builtin_scenario("temporary_stop", TemporaryStopScenario, TemporaryStopConfig)
_register_builtin_scenario(
    "pedestrian_dart_out", PedestrianDartOutScenario, PedestrianDartOutConfig
)
_register_builtin_scenario("cut_in", CutInScenario, CutInConfig)
_register_builtin_scenario("adversity", AdversityScenario, AdversityConfig)


# ---------------------------------------------------------------------------
# Process memory
# ---------------------------------------------------------------------------

#: glibc's ``mallopt`` parameter for the number of malloc arenas.
_M_ARENA_MAX = -8

#: The arenas a run keeps: few enough that freed memory is reused rather than
#: stranded in an arena no thread allocates from again.
_ARENA_MAX = 2


def _cap_malloc_arenas() -> bool:
    """Cap glibc's malloc arenas, so a long batch does not fragment its heap.

    Every ``world.get_map()`` builds a whole client-side ``carla.Map``: on a
    large map (Odaiba) some 270 MB -- a 72 MB OpenDRIVE string, the road
    geometry and an rtree -- and seconds of work.  The run reuses the one it
    holds, but whatever is built and dropped is allocated on whichever thread
    asks, and glibc gives each thread an arena of its own (up to eight per
    core): memory freed in one is not reused by the next allocation on
    another, so the resident size of a cut_in batch on Odaiba grew some 40 MB
    a case although nothing leaked.  With ``arena_max=2`` it stays flat.

    Before any worker thread starts, as arenas are made for threads.  Only on
    glibc Linux, and not when ``MALLOC_ARENA_MAX`` or ``GLIBC_TUNABLES``
    already set it: the user's choice stands.  Any failure is ignored -- this
    is a tuning, not something a run needs.

    Returns:
        Whether the cap was applied.
    """
    if not sys.platform.startswith("linux") or _arena_max_set_by_user(os.environ):
        return False
    try:
        import ctypes  # noqa: PLC0415
        import ctypes.util  # noqa: PLC0415
        import platform  # noqa: PLC0415

        if platform.libc_ver()[0] != "glibc":
            return False
        libc_name = ctypes.util.find_library("c")
        if not libc_name:
            return False
        libc = ctypes.CDLL(libc_name)
        # mallopt() returns 1 on success, 0 on error.
        return bool(libc.mallopt(_M_ARENA_MAX, _ARENA_MAX))
    except Exception:  # any libc oddity: a tuning is not worth a failed run
        logger.debug("Could not cap the malloc arenas", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------


def _config_float(key: str, value: Any) -> float:
    """``float(value)`` of the config's *key*; for a value ``float()`` cannot
    read -- not a number, or an int too big for a float (``10**400``) -- a
    ``ValueError`` naming the key (#1660).  What ``float()`` reads, a boolean
    or a numeric string included, is read as before.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {shown(value)}") from None
    except OverflowError:
        raise ValueError(f"{key} must be finite, got {shown(value)}") from None


def _config_finite(key: str, value: Any, *, minimum: float | None = None) -> float:
    """:func:`_config_float` of the config's *key*, refused unless finite and,
    with *minimum*, at least *minimum* -- a ``ValueError`` naming the key
    (#1809).  A NaN, an infinity or a negative speed would otherwise reach the
    run: an ego that starts still with no error, an actor whose physics an
    infinite velocity breaks, a pose that fails far from the key behind it.
    """
    number = _config_float(key, value)
    if minimum is None:
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite, got {shown(value)}")
    elif not (math.isfinite(number) and number >= minimum):
        raise ValueError(
            f"{key} must be finite and at least {minimum:g}, got {shown(value)}"
        )
    return number


#: The ego's and entities' number settings a run reads through
#: :func:`_config_finite`, with the least each may be (``None``: any finite
#: number), as :func:`_check_ego_numbers` refuses them before a run starts
#: (#1809).  A speed is never below zero (the initial speed is only applied
#: above it); a retry step is a distance the retries are shifted by on both
#: sides, so a negative one is meaningless and 0 keeps that axis where it is;
#: a pose is any finite offset or angle.
_EGO_NUMBER_MINIMA: tuple[tuple[str, float | None], ...] = (
    ("ego.initial_speed_kmh", 0.0),
    ("ego.spawn_s", None),
    ("ego.spawn_t", None),
    ("ego.spawn_heading", None),
    ("ego.goal_s", None),
    ("entity.spawn_retry_t_step", 0.0),
    ("entity.spawn_retry_z_step", 0.0),
)


def _ego_number(key: str, value: Any) -> float:
    """:func:`_config_finite` of *key*, one of :data:`_EGO_NUMBER_MINIMA`,
    with its minimum."""
    return _config_finite(key, value, minimum=dict(_EGO_NUMBER_MINIMA)[key])


def _check_ego_numbers(cfg: DictConfig) -> None:
    """Refuse the ego's and entities' number settings of
    :data:`_EGO_NUMBER_MINIMA` -- a NaN, an infinity, a negative speed or
    retry step -- and ``entity.spawn_retry_max_count`` unless a whole number,
    0 or more (:func:`_spawn_retry_max_count`), as building the ego would
    (:func:`build_ego_and_spawn`), so a batch refuses them before CARLA
    starts (#1809, #1984).  A key that is absent or null is left to the
    build, which reads it as its default (:func:`_given`): a pose's 0
    (#2143, #2161), ``EgoVehicleConfig``'s speed and ``EntityConfig``'s
    retry settings (#2294).

    Raises:
        ValueError: naming the key of the value refused.
    """
    for key, _ in _EGO_NUMBER_MINIMA:
        block_name, name = key.split(".", 1)
        block = cfg.get(block_name)
        if not isinstance(block, (DictConfig, Mapping)):
            continue
        value = block.get(name)
        if value is not None:
            _ego_number(key, value)
    # The retry count, a whole number rather than a finite one (#1984).
    entity = cfg.get("entity")
    if isinstance(entity, (DictConfig, Mapping)):
        count = entity.get("spawn_retry_max_count")
        if count is not None:
            _spawn_retry_max_count(count)


#: The config key of the ego's blueprint (:func:`_ego_vehicle_type`).
_EGO_VEHICLE_TYPE_KEY = "ego.vehicle_type"


def _ego_vehicle_type(ego_cfg: Any) -> str:
    """``ego.vehicle_type`` of the ``ego`` group *ego_cfg*: the CARLA
    blueprint the ego is spawned as, ``EgoVehicleConfig``'s when absent or
    null.

    Raises:
        ValueError: naming the key, for a value that is not text or is blank
            (``''``, ``false``), which would otherwise reach CARLA as is
            (#2294).
    """
    value = _given(ego_cfg, "vehicle_type", EgoVehicleConfig.vehicle_type)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{_EGO_VEHICLE_TYPE_KEY} must be a CARLA blueprint id "
            f"(non-empty text), got {value!r}"
        )
    return value


def _check_ego_vehicle_type(cfg: DictConfig) -> None:
    """Refuse ``ego.vehicle_type`` as building the ego would
    (:func:`_ego_vehicle_type`), so a batch refuses it before CARLA starts.

    Raises:
        ValueError: naming the key.
    """
    ego_cfg = cfg.get("ego")
    if isinstance(ego_cfg, (DictConfig, Mapping)):
        _ego_vehicle_type(ego_cfg)


#: The config key of the ego's spawn retries (:func:`_spawn_retry_max_count`).
_SPAWN_RETRY_MAX_COUNT_KEY = "entity.spawn_retry_max_count"


def _spawn_retry_max_count(value: Any) -> int:
    """``entity.spawn_retry_max_count``, checked as every whole-number setting
    is (:func:`~autoware_carla_scenario.odd.sampler.whole_number_setting`):
    ``4``, ``4.0`` or ``"4"``, 0 or more -- not ``2.5`` or ``true``, which
    ``int()`` alone read as 2 and 1, a negative count or null (#1984).

    Raises:
        ValueError: naming the key.
    """
    count = whole_number_setting(_SPAWN_RETRY_MAX_COUNT_KEY, value, minimum=0)
    if count is None:
        raise ValueError(
            f"{_SPAWN_RETRY_MAX_COUNT_KEY} must be a whole number, got None"
        )
    return count


def build_ego_and_spawn(
    cfg: DictConfig,
) -> tuple[EgoConfig, Lanelet2Pose, GroundProjectionConfig]:
    """Extract :class:`EgoConfig`, spawn pose, and ground-projection config.

    This is the common preamble shared by all built-in scenarios.  Downstream
    projects can call this helper and then instantiate their own scenario class
    without duplicating the boilerplate.

    Both ends of the run travel with the ego config: ``ego.spawn_lanelet_id``
    and ``ego.goal_lanelet_id``.  A config that names no goal is not refused
    here, because it is not yet wrong -- a scenario may know the destination the
    config does not, and derive it in ``setup()``, as
    :class:`~autoware_carla_scenario.examples.intersection_passing.IntersectionPassingScenario`
    does from the route it asserts.  The scenario is given its say first, and
    :meth:`~autoware_carla_scenario.BaseScenario.register_route_to_goal` refuses
    an ego that still has nowhere to go.
    """
    ground_projection = build_ground_projection(cfg)
    # Each absent or null is its default, EgoVehicleConfig's or
    # EntityConfig's (#2294), as _check_ego_numbers leaves it.
    ego_cfg = cfg.get("ego") or {}
    entity = cfg.get("entity") or {}
    ego = EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type=_ego_vehicle_type(ego_cfg),
        initial_speed_kmh=_ego_number(
            "ego.initial_speed_kmh",
            _given(ego_cfg, "initial_speed_kmh", EgoVehicleConfig.initial_speed_kmh),
        ),
        spawn_retry_max_count=_spawn_retry_max_count(
            _given(entity, "spawn_retry_max_count", EntityConfig.spawn_retry_max_count)
        ),
        spawn_retry_t_step=_ego_number(
            "entity.spawn_retry_t_step",
            _given(entity, "spawn_retry_t_step", EntityConfig.spawn_retry_t_step),
        ),
        spawn_retry_z_step=_ego_number(
            "entity.spawn_retry_z_step",
            _given(entity, "spawn_retry_z_step", EntityConfig.spawn_retry_z_step),
        ),
        goal_pose=build_goal_pose(cfg),
        waypoint_poses=build_waypoint_poses(cfg),
    )
    return ego, build_spawn_pose(cfg), ground_projection


#: The config keys of the ground-projection ray (:func:`build_ground_projection`).
_RAY_UPPER_KEY = "entity.ground_projection_ray_distance_upper"
_RAY_LOWER_KEY = "entity.ground_projection_ray_distance_lower"


def build_ground_projection(cfg: DictConfig) -> GroundProjectionConfig:
    """The :class:`GroundProjectionConfig` of *cfg*'s
    ``entity.ground_projection_ray_distance_upper`` and ``_lower``.

    Each absent or null is its default (``EntityConfig``'s; #2188).

    Raises:
        ValueError: naming the key, for a value that is not a number, is
            negative or not finite, or a ray of no length (both zero): a ray
            that cannot hit leaves every pose at its estimate (#1416).
    """
    entity = cfg.get("entity") or {}
    distances: list[float] = []
    for key, default in (
        (_RAY_UPPER_KEY, EntityConfig.ground_projection_ray_distance_upper),
        (_RAY_LOWER_KEY, EntityConfig.ground_projection_ray_distance_lower),
    ):
        value = _given(entity, key.split(".", 1)[1], default)
        try:
            distances.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
        except OverflowError:
            # An int too big for a float (10**400): not finite (#1652).
            raise ValueError(
                f"{key} must be non-negative and finite, got {shown(value)}"
            ) from None
    upper, lower = distances
    check_ray_distances(upper, lower, _RAY_UPPER_KEY, _RAY_LOWER_KEY)
    return GroundProjectionConfig(ray_distance_upper=upper, ray_distance_lower=lower)


def _lanelet_id(key: str, value: Any) -> int:
    """The lanelet id *value* of *key* names, checked as every whole-number
    setting is (:func:`~autoware_carla_scenario.odd.sampler.whole_number_setting`):
    ``242``, ``242.0`` or ``"242"`` -- not ``242.7``, which ``int()`` alone
    truncated to 242 and so spawned the ego on another lanelet, ``abc``, a
    boolean or null (#2089).

    Raises:
        ValueError: naming *key*.
    """
    lanelet_id = whole_number_setting(key, value)
    if lanelet_id is None:
        raise ValueError(f"{key} must be a lanelet id (a whole number), got None")
    return lanelet_id


def _waypoint_lanelet_ids(ego_cfg: Any) -> list[int]:
    """``ego.waypoint_lanelet_ids`` of the ``ego`` group *ego_cfg*, each
    checked (:func:`_lanelet_id`); empty when absent, null or empty.

    Raises:
        ValueError: naming the key (and the item), when it is not a list or
            an item is not a lanelet id.
    """
    value = ego_cfg.get("waypoint_lanelet_ids")
    if value is None:
        return []
    if not isinstance(value, (ListConfig, list, tuple)):
        raise ValueError(
            f"ego.waypoint_lanelet_ids must be a list of lanelet ids, got {value!r}"
        )
    return [
        _lanelet_id(f"ego.waypoint_lanelet_ids[{i}]", item)
        for i, item in enumerate(value)
    ]


def _check_ego_lanelet_ids(cfg: DictConfig) -> None:
    """Refuse the ego's lanelet ids -- ``ego.spawn_lanelet_id``,
    ``ego.goal_lanelet_id`` and ``ego.waypoint_lanelet_ids`` -- unless each
    is a whole number (:func:`_lanelet_id`), as building the ego would
    (:func:`build_spawn_pose`, :func:`build_goal_pose`,
    :func:`build_waypoint_poses`), so a batch or a sweep refuses them before
    CARLA starts (#2089).  A spawn or goal that is absent or null is left to
    the build, which reads it as none or refuses it.

    Raises:
        ValueError: naming the key of the value refused.
    """
    ego_cfg = cfg.get("ego")
    if not isinstance(ego_cfg, (DictConfig, Mapping)):
        return
    for key in ("spawn_lanelet_id", "goal_lanelet_id"):
        value = ego_cfg.get(key)
        if value is not None:
            _lanelet_id(f"ego.{key}", value)
    _waypoint_lanelet_ids(ego_cfg)


def _given(block: Any, key: str, default: Any) -> Any:
    """*block*'s *key*, or *default* only when it is absent or null -- as
    :func:`_check_ego_numbers` leaves to the build only an absent or null
    key, so the two read every other value alike (``""`` refused by both,
    ``0`` read as 0 by both; #2143)."""
    value = block.get(key)
    return default if value is None else value


def build_spawn_pose(cfg: DictConfig) -> Lanelet2Pose:
    """Extract the ego's spawn pose from the ``ego.spawn_*`` keys.

    ``ego.spawn_lanelet_id`` and ``ego.spawn_s`` place it along the lanelet,
    ``ego.spawn_t`` across it and ``ego.spawn_heading`` turns it against the
    lane's direction of travel -- the same pose an NPC's document spawn is
    (``spawn.t`` and ``spawn.heading``).  A config that predates the last two
    keys spawns on the centreline facing along the lane, as it always did.
    Each of ``ego.spawn_s``, ``ego.spawn_t`` and ``ego.spawn_heading`` that is
    absent or null is 0 (#2143, #2161).
    """
    ego_cfg = cfg.ego
    return Lanelet2Pose(
        lanelet_id=_lanelet_id("ego.spawn_lanelet_id", ego_cfg.spawn_lanelet_id),
        s=_ego_number("ego.spawn_s", _given(ego_cfg, "spawn_s", 0.0)),
        t=_ego_number("ego.spawn_t", _given(ego_cfg, "spawn_t", 0.0)),
        heading=_ego_number("ego.spawn_heading", _given(ego_cfg, "spawn_heading", 0.0)),
    )


def build_goal_pose(cfg: DictConfig) -> Lanelet2Pose | None:
    """Extract the ego's goal pose from ``ego.goal_lanelet_id`` / ``ego.goal_s``.

    Returns ``None`` when no goal is configured, which does not yet mean the run
    has none: a scenario may derive its own in ``setup()``.  An absent or null
    ``ego.goal_s`` is 0, the start of the lanelet (#2161).
    """
    ego_cfg = cfg.get("ego") or {}
    goal_lanelet_id = ego_cfg.get("goal_lanelet_id")
    if goal_lanelet_id is None:
        return None
    return Lanelet2Pose(
        lanelet_id=_lanelet_id("ego.goal_lanelet_id", goal_lanelet_id),
        s=_ego_number("ego.goal_s", _given(ego_cfg, "goal_s", 0.0)),
    )


def build_waypoint_poses(cfg: DictConfig) -> list[Lanelet2Pose]:
    """Extract the ego's waypoints from ``ego.waypoint_lanelet_ids``.

    Each is taken at the start of its lanelet: a waypoint says which road the
    route goes down, and where along it is the router's business. An absent or
    empty key means no particular way to the goal was asked for, which is not
    the same as none being intended -- a scenario may name its own in
    ``setup()``, e.g. from the route it already asserts.
    """
    ego_cfg = cfg.get("ego") or {}
    return [
        Lanelet2Pose(lanelet_id=lanelet_id, s=0.0)
        for lanelet_id in _waypoint_lanelet_ids(ego_cfg)
    ]


def build_planned_route(cfg: DictConfig, name: str = "") -> PlannedRoute:
    """The route the config sends the ego along, for checking it against an ODD.

    Raises :class:`~autoware_carla_scenario.odd.route.RouteError` when the
    config names no spawn lanelet.

    From ``ego.spawn_lanelet_id`` / ``ego.spawn_s``, through
    ``ego.waypoint_lanelet_ids``, to ``ego.goal_lanelet_id`` / ``ego.goal_s``:
    the poses a run hands the ego (``scenario-odd route``, docs/odd.md).

    A scenario may derive its goal and waypoints in ``setup()``, which needs
    CARLA.  The one way a built-in scenario does -- the goal from
    ``scenario.expected_route_lanelet_ids``, with
    :meth:`~autoware_carla_scenario.BaseScenario.derive_goal_from_route` --
    is followed here with the same rule (a configured goal wins).  Any other
    is not known before the run: the route then has no goal.
    """
    ego_cfg = cfg.get("ego") or {}
    if ego_cfg.get("spawn_lanelet_id") is None:
        raise RouteError("no ego.spawn_lanelet_id: the route has no start")
    start = build_spawn_pose(cfg)
    goal = build_goal_pose(cfg)
    scenario_cfg = cfg.get("scenario") or {}
    expected = scenario_cfg.get("expected_route_lanelet_ids") or []
    if goal is None and expected:
        # Checked as every other lanelet id is (#2089, #2188).
        goal = Lanelet2Pose(
            lanelet_id=_lanelet_id(
                f"scenario.expected_route_lanelet_ids[{len(expected) - 1}]",
                expected[-1],
            ),
            s=0.0,
        )
    via = build_waypoint_poses(cfg)
    return PlannedRoute(start=start, goal=goal, via=tuple(via), name=name)


#: One launcher per launcher configuration in this process: every Autoware ego
#: of a batch, and the queue running them, share the launcher that builds the
#: workspace once and starts one stack at a time.
_AUTOWARE_LAUNCHERS: dict[str, AutowareLauncher] = {}


def build_autoware_launcher(cfg: DictConfig) -> AutowareLauncher | None:
    """The launcher that starts Autoware for each scenario, per ``autoware.launcher``.

    ``autoware.launcher.type`` is ``none`` (Autoware is started by someone
    else -- the default), ``docker`` (a local Autoware workspace, built and run
    in its dev container) or ``command`` (``autoware.launcher.command`` run on
    this host).  Only an ``ego.entity=autoware`` run has one.  The same
    configuration yields the same launcher, so a batch's egos and its queue
    share it.

    The group holds the keys of both launcher types at once (the shipped
    config does), and each type takes its own; a key neither type reads
    (``+autoware.launcher.stop_timout_s=60``) is refused, whatever the type,
    rather than leaving the default in place.

    The settings the launcher of the type reads are checked too, naming the
    key: ``stop_timeout_s`` must be a finite number above 0, and the docker
    launcher's ``ros_domain_ids`` a list of ROS domain ids (ints from 0 to
    232).  The argument lists (``launch``, ``colcon_args``, ``docker_args``,
    ``command``) must be lists, not strings, of strings or numbers; ``env``
    and ``mounts`` mappings of strings or numbers; ``bridge_parameters`` a
    mapping; ``workspace`` a non-empty path; ``devcontainer``, ``docker`` and
    ``container_prefix`` non-empty text, and ``image``, ``workspace_mount``,
    ``gpus`` and ``user`` null or non-empty text (``gpus`` may also be a GPU
    count, 1 or more, as ``docker run --gpus 2`` takes).  A null setting is its
    default (#2158, #2161): ``type`` is ``none``, ``workspace``
    ``~/autoware``, and ``build``, ``rosdep``, ``stop_timeout_s``,
    ``devcontainer``, ``docker`` and ``container_prefix`` those of
    :class:`~autoware_carla_scenario.autoware_stack.DockerAutowareConfig`;
    a null argument list or mapping is empty, and a null ``gpus`` no GPU, as
    they always were.  :func:`check_batch_configs` calls this, so a batch or a
    sweep is refused for any of these before its first scenario starts, and
    before any container is started or workspace prepared.

    Raises:
        ValueError: If ``autoware.launcher.type`` names an unknown launcher,
            the group holds a key no launcher reads, or a setting the type
            reads is refused, naming it.
    """
    if _ego_entity_setting(cfg) != "autoware":
        return None
    autoware_cfg = cfg.get("autoware")
    launcher_cfg = None if autoware_cfg is None else autoware_cfg.get("launcher")
    if launcher_cfg is None:
        return None
    settings = _to_dict(launcher_cfg)

    from autoware_carla_scenario.autoware_stack import (  # noqa: PLC0415
        CommandAutowareLauncher,
        DockerAutowareConfig,
        DockerAutowareLauncher,
    )

    docker_fields = set(DockerAutowareConfig.__dataclass_fields__)
    known = (
        {"type"}
        | docker_fields
        | {"command", "env", "stop_timeout_s", "bridge_parameters"}
    )
    unknown = sorted(set(settings) - known)
    if unknown:
        msg = (
            "Unknown autoware.launcher key(s): "
            f"{[f'autoware.launcher.{name}' for name in unknown]}. "
            f"Known keys: {sorted(known)}"
        )
        raise ValueError(msg)

    kind = _launcher_kind(settings.pop("type", None))
    if kind == "none":
        return None

    # A GPU count is what docker run --gpus takes too (``gpus: 2``), as the
    # prepare backends read it (prepare.docker); a bool or a count below 1 is
    # left for DockerAutowareConfig to refuse.  Made text before the key, so
    # ``gpus: 2`` and ``gpus: '2'`` share one launcher.
    gpus = settings.get("gpus")
    if isinstance(gpus, int) and not isinstance(gpus, bool) and gpus >= 1:
        settings["gpus"] = str(gpus)
    key = json.dumps({"type": kind, **settings}, sort_keys=True, default=str)
    cached = _AUTOWARE_LAUNCHERS.get(key)
    if cached is not None:
        return cached

    launcher: AutowareLauncher
    if kind not in ("docker", "command"):
        msg = (
            f"Unknown autoware.launcher.type: {kind!r}. "
            "Expected one of: 'none', 'docker', 'command'."
        )
        raise ValueError(msg)
    stop_timeout_s = _launcher_stop_timeout_s(settings)
    if kind == "docker":
        # Null is the field's default (#2161), as it is the workspace's,
        # below: a null build or rosdep used to be refused, and a null
        # devcontainer, docker or container_prefix handed on as None.  A null
        # argument list or mapping stays none and a null gpus no GPU, as the
        # docs have them.
        options = {
            k: v
            for k, v in settings.items()
            if k in docker_fields and (v is not None or k not in _NULL_IS_DEFAULT)
        }
        # Null is the default; false, a number or '' is refused rather than
        # made Path("False") (#2158).
        workspace = options.get("workspace")
        if workspace is None:
            workspace = "~/autoware"
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError(
                "autoware.launcher.workspace must be a path to the Autoware "
                f"workspace, got {workspace!r}"
            )
        options["workspace"] = Path(workspace)
        # Absent is empty, refused naming the key.
        options.setdefault("launch", ())
        # The argument lists' and mappings' types are checked here, naming
        # the key (the _launcher_* helpers do); the text settings' types and
        # every value by DockerAutowareConfig, below, whose refusal is headed
        # with the group.
        for name in ("launch", "colcon_args", "docker_args"):
            if name in options:
                options[name] = _launcher_str_list(name, options[name])
        for name in ("mounts", "env"):
            if name in options:
                options[name] = _launcher_str_mapping(name, options[name])
        # Read by truthiness by the launcher (it runs rosdep and feeds the
        # image's hash): strictly here, so "false" is off and a typo refused
        # (#1761).
        if "rosdep" in options:
            options["rosdep"] = check_flag(
                "autoware.launcher.rosdep", options["rosdep"]
            )
        options["ros_domain_ids"] = _launcher_ros_domain_ids(settings)
        options["stop_timeout_s"] = stop_timeout_s
        options["bridge_parameters"] = _launcher_bridge_parameters(settings)
    else:
        argv = _launcher_str_list("command", settings.get("command"))
        env = _launcher_str_mapping("env", settings.get("env"))
        bridge_parameters = _launcher_bridge_parameters(settings)
    try:
        if kind == "docker":
            launcher = DockerAutowareLauncher(DockerAutowareConfig(**options))
        else:
            launcher = CommandAutowareLauncher(
                argv,
                env=env,
                stop_timeout_s=stop_timeout_s,
                bridge_parameters=bridge_parameters,
            )
    except ValueError as exc:
        # The configs' own refusals name the field; this is the group.
        raise ValueError(f"autoware.launcher.{exc}") from None
    _AUTOWARE_LAUNCHERS[key] = launcher
    return launcher


def _finite_number(value: Any) -> bool:
    """Whether *value* is a finite number: an int or a float, not a bool.

    An int too big for a float (``10**400``) is not: ``math.isfinite()``
    raises ``OverflowError`` on it (#1652).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


#: The ``autoware.launcher`` keys of :class:`DockerAutowareConfig` fields that
#: a null leaves at the field's default (:func:`build_autoware_launcher`,
#: #2161).  ``workspace`` and ``stop_timeout_s`` are read on their own, with
#: the same rule.
_NULL_IS_DEFAULT = frozenset(
    {
        "build",
        "rosdep",
        "devcontainer",
        "image",
        "workspace_mount",
        "user",
        "docker",
        "container_prefix",
    }
)


def _launcher_kind(value: Any) -> str:
    """The launcher type ``autoware.launcher.type`` *value* names, as a run
    and a Slurm task read it: ``none`` when it is absent or null (#2158), not
    the unknown type ``'None'``; any other value as its text, which
    :func:`build_autoware_launcher` refuses unless ``none``, ``docker`` or
    ``command``."""
    return "none" if value is None else str(value)


def _launcher_stop_timeout_s(settings: dict[str, Any]) -> float:
    """``autoware.launcher.stop_timeout_s`` of *settings*, checked: a finite
    number of seconds -- an int or a float, not a bool or a string -- above
    0, as the bridge's timeouts are; absent or null is 20 s, the default
    (#2161).  Each launcher type reads it.

    A NaN never times out (``Popen.wait(timeout=nan)``), so a stack that
    ignores ``SIGINT`` would never be escalated and the batch would hang; 0 or
    less skips the grace the stack is given to stop.

    Raises:
        ValueError: naming the key.
    """
    value = _given(settings, "stop_timeout_s", 20.0)
    if not _finite_number(value) or value <= 0:
        raise ValueError(
            "autoware.launcher.stop_timeout_s must be a finite number > 0 "
            f"(got {shown(value)})"
        )
    return float(value)


def _launcher_str_list(name: str, value: object) -> tuple[str, ...]:
    """``autoware.launcher.<name>``, a list of arguments, checked: a list (or a
    tuple) of strings or numbers, each made a ``str``; null or absent is empty.

    ``launch``, ``colcon_args``, ``docker_args`` and ``command`` are read
    through it.  A string is refused rather than taken as one argument or
    split: Python would iterate it into characters (``docker_args:
    '--shm-size=8g'`` became ``('-', '-', 's', ...)``), and neither reading is
    safe -- ``'ros2 launch x y'`` may be one argument (a ``bash -c`` script)
    or four, and only the author knows which.  An element that is a list, a
    mapping, null or a boolean is refused too: its ``str()`` (``"['a']"``,
    ``"None"``, ``"True"``) is never the argument that was meant.

    Raises:
        ValueError: naming the key.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"autoware.launcher.{name} must be a list of arguments, not "
            f"{type(value).__name__} {value!r}: write "
            f"'autoware.launcher.{name}=[a,b,c]' (a list item may hold spaces "
            "when quoted)"
        )
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(
                f"autoware.launcher.{name}[{index}] must be a string or a number, "
                f"not {type(item).__name__} {item!r}"
            )
    return tuple(str(item) for item in value)


def _launcher_str_mapping(name: str, value: object) -> dict[str, str]:
    """``autoware.launcher.<name>`` (``env``, ``mounts``), checked: a mapping
    whose values are strings or numbers, each made a ``str``; null or absent is
    empty.  A value that is null, a boolean, a list or a mapping is refused, as
    :func:`_launcher_str_list` refuses such an argument.  So is a key or a
    value holding a NUL, and, in ``env``, an empty variable name or one holding
    ``=``: no process can be handed those, and ``Popen`` would refuse them only
    once the stack starts (#2079).

    Raises:
        ValueError: naming the key.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"autoware.launcher.{name} must be a mapping, not "
            f"{type(value).__name__} {value!r}"
        )
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(
                f"autoware.launcher.{name}.{key} must be a string or a number, "
                f"not {type(item).__name__} {item!r}"
            )
    mapping = {str(key): str(item) for key, item in value.items()}
    for key, item in mapping.items():
        if "\0" in key or "\0" in item:
            raise ValueError(
                f"autoware.launcher.{name}: {key!r}: {item!r} holds a NUL "
                "character, which no process can be handed"
            )
        if name == "env" and (not key or "=" in key):
            raise ValueError(
                f"autoware.launcher.env: {key!r} is not an environment variable "
                "name: it must be non-empty and hold no '='"
            )
    return mapping


def _launcher_bridge_parameters(settings: dict[str, Any]) -> dict[str, Any]:
    """``autoware.launcher.bridge_parameters`` of *settings*: a mapping of ROS
    parameters (values of any type: ``auto_engage: false``) merged over the
    default, ``{use_sim_time: true}``
    (:data:`~autoware_carla_scenario.autoware_stack.launcher.DEFAULT_BRIDGE_PARAMETERS`),
    for both launcher types; null or absent is the default (#2217).  Merged,
    so a mapping that sets something else keeps the node on CARLA's clock: a
    parameter is turned off by setting it (``use_sim_time: false``), or left
    at the node's own default by a null.

    Raises:
        ValueError: naming the key, for a value that is no mapping, or a
            parameter the node would refuse or misbehave with as it starts (a
            fraction for an integer one, a string for a number, a timeout that
            is not positive, :func:`~autoware_carla_scenario.autoware_stack.bridge_node.bridge_ros_args`).
    """
    from autoware_carla_scenario.autoware_stack.bridge_node import (  # noqa: PLC0415
        bridge_ros_args,
    )

    value = settings.get("bridge_parameters")
    if value is None:
        return dict(DEFAULT_BRIDGE_PARAMETERS)
    if not isinstance(value, dict):
        raise ValueError(
            "autoware.launcher.bridge_parameters must be a mapping of ROS "
            f"parameters, not {type(value).__name__} {value!r}"
        )
    parameters = {**DEFAULT_BRIDGE_PARAMETERS, **value}
    try:
        bridge_ros_args("", parameters)
    except ValueError as exc:
        raise ValueError(f"autoware.launcher.bridge_parameters.{exc}") from None
    return parameters


def _launcher_ros_domain_ids(settings: dict[str, Any]) -> tuple[int, ...]:
    """``autoware.launcher.ros_domain_ids`` of *settings*, checked: a list of
    ints (not bools) from 0 to 232, the ``ROS_DOMAIN_ID`` range ROS 2 can use
    (:class:`~autoware_carla_scenario.autoware_stack.DockerAutowareConfig`
    checks the range); null or absent is none.

    Raises:
        ValueError: naming the key.
    """
    value = settings.get("ros_domain_ids")
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(
            "autoware.launcher.ros_domain_ids must be a list of ROS domain ids, "
            f"whole numbers from 0 to 232 (got {value!r})"
        )
    return tuple(value)


def _native_autoware_interface(cfg: DictConfig) -> bool:
    """``ego.native_autoware_interface``, a true/false switch read as every
    other is (:func:`~autoware_carla_scenario.config_flags.check_flag`:
    ``"false"`` and ``0`` are off; absent or null is off, #2188), and refused
    ``true`` unless ``ego.entity`` is ``autoware``.

    Only :class:`~autoware_carla_scenario.AutowareEgoEntity` spawns its ego
    for CARLA's native Autoware interface (``ros2_autoware_control=true``,
    role name ``ego``).  ``true`` with another entity is refused rather than
    ignored with a warning: the run it would start is not the one asked for --
    the ego would not be served by the server -- and a refusal before CARLA
    starts costs one edit of the config, where a warning is easily lost in a
    batch's logs.

    Raises:
        ValueError: naming ``ego.native_autoware_interface``.
    """
    native = _switch(cfg, "ego", "native_autoware_interface")
    entity = _ego_entity_setting(cfg)
    if native and entity != "autoware":
        raise ValueError(
            "ego.native_autoware_interface=true needs ego.entity=autoware "
            f"(got ego.entity={entity}): only the Autoware ego is spawned for "
            "CARLA's native Autoware interface"
        )
    return native


def _autoware_bridge_config(cfg: DictConfig) -> "AutowareBridgeConfig":
    """The ``autoware`` group's bridge settings (all of it but ``launcher``).

    Raises:
        ValueError: naming the key, for one the group does not define
            (``autoware.boot_timout_s``, a typo that would leave the default
            in place) or a value :class:`AutowareBridgeConfig` refuses.
    """
    from autoware_carla_scenario import AutowareBridgeConfig  # noqa: PLC0415
    from autoware_carla_scenario.utils.config import (  # noqa: PLC0415
        checked_options,
        null_as_default,
    )

    autoware_cfg = cfg.get("autoware")
    try:
        bridge_options = checked_options(
            AutowareBridgeConfig,
            _to_dict(autoware_cfg) if autoware_cfg is not None else {},
            ignore={"launcher"},
        )
    except ValueError as exc:
        raise ValueError(f"autoware: {exc}") from None
    try:
        # A null setting is its default (#2217), as an absent one.
        return AutowareBridgeConfig(
            **null_as_default(AutowareBridgeConfig, bridge_options)
        )
    except ValueError as exc:
        raise ValueError(f"autoware.{exc}") from None


def _check_autoware_liveness(
    bridge_cfg: "AutowareBridgeConfig", launcher: AutowareLauncher | None
) -> None:
    """Refuse an ``autoware.liveness_timeout_s`` the launcher's node would trip.

    The ``scenario_bridge`` node the launcher starts polls every
    ``route_watch_period_s`` of its ``autoware.launcher.bridge_parameters``
    (:func:`~autoware_carla_scenario.autoware_stack.bridge_node.liveness_timeout_problem`).
    The entity refuses it too; this says so naming the group's keys, and
    before anything starts.

    Raises:
        ValueError: headed ``autoware.``.
    """
    if launcher is None:
        return
    from autoware_carla_scenario.autoware_stack.bridge_node import (  # noqa: PLC0415
        launcher_bridge_parameters,
        liveness_timeout_problem,
    )

    problem = liveness_timeout_problem(
        bridge_cfg.liveness_timeout_s, launcher_bridge_parameters(launcher)
    )
    if problem is not None:
        raise ValueError(f"autoware.{problem}")


#: The ego entities ``ego.entity`` may name (:func:`build_ego_entity`).
EGO_ENTITIES: tuple[str, ...] = (
    "autopilot",
    "autoware",
    "carla_driver",
    "ros2",
    "hils",
)


def _ego_entity_setting(cfg: Any) -> Any:
    """``ego.entity`` of *cfg* as written, unchecked: ``autopilot`` when the
    config has no ``ego`` group or the key is absent or null (#2161), not the
    unknown entity ``'None'``.  Every reader of the key -- a run's, a batch's
    check and a Slurm task's -- reads it through this; :func:`ego_entity_name`
    checks it.

    Raises:
        ValueError: naming ``ego``, for a block that is not a mapping
            (:func:`_block`, #2298).
    """
    value = _block(cfg, "ego").get("entity")
    return "autopilot" if value is None else value


def ego_entity_name(cfg: DictConfig) -> str:
    """``ego.entity`` of *cfg*, checked: one of :data:`EGO_ENTITIES`;
    ``autopilot`` when the config has no ``ego`` group or does not set it, or
    sets it to null (:func:`_ego_entity_setting`).

    :func:`check_batch_configs` calls this, so a run, a batch or a lanelet
    sweep naming an unknown entity is refused before anything starts rather
    than once its scenario is built (#2089).

    Raises:
        ValueError: naming the key, when it names no entity.
    """
    entity = _ego_entity_setting(cfg)
    if not isinstance(entity, str) or entity not in EGO_ENTITIES:
        expected = ", ".join(repr(name) for name in EGO_ENTITIES)
        raise ValueError(
            f"Unknown ego.entity: {entity!r}. Expected one of: {expected}."
        )
    return entity


def build_ego_entity(cfg: DictConfig) -> EgoVehicle | None:
    """Build the ego entity selected by ``cfg.ego.entity``.

    Returns ``None`` for ``"autopilot"``, letting the scenario fall back to its default
    :class:`~autoware_carla_scenario.entity.ego.EgoVehicle`. A config with no ``ego``
    group at all selects ``"autopilot"`` too: an injected ``build_scenario_fn`` supplies
    its own :class:`EgoConfig`, so it has no reason to carry the group Hydra would.

    Raises:
        ValueError: If ``cfg.ego.entity`` names an unknown entity, or a setting
            of the entity's group is out of range (a NaN timeout, say) or not
            one the group defines (``autoware.boot_timout_s``), naming the key, or
            ``ego.native_autoware_interface`` is refused
            (:func:`_native_autoware_interface`).
    """
    entity = ego_entity_name(cfg)
    native = _native_autoware_interface(cfg)

    if entity == "autopilot":
        return None

    if entity == "autoware":
        # The closed-loop entity: it spawns the ego, hands Autoware the
        # scenario's mission over the bridge the framework hosts and -- given a
        # launcher (autoware.launcher) -- starts a fresh Autoware for the
        # scenario and removes it afterwards.  The mission itself comes from the scenario, whose
        # ``setup()`` registers a ``RoutingAction`` for the spawn and
        # the goal its ``EgoConfig`` carries, snapped onto the live map -- poses
        # that do not exist before then -- and the runner performs it in the init
        # phase.  The goal comes from the config (``ego.goal_lanelet_id``) or
        # from the scenario itself; an ego that ends ``setup()`` with neither is
        # refused there.
        from autoware_carla_scenario import (  # noqa: PLC0415
            AutowareEgoEntity,
            GrpcAutowareBridgeServer,
        )

        bridge_cfg = _autoware_bridge_config(cfg)
        launcher = build_autoware_launcher(cfg)
        _check_autoware_liveness(bridge_cfg, launcher)
        # autostart=False: in a batch every scenario is built before the first
        # one runs, and two bridges cannot hold the same address at once.  The
        # entity starts this one when its own scenario starts.
        scenario_cfg = cfg.get("scenario") or {}
        return AutowareEgoEntity(
            bridge_cfg,
            bridge=GrpcAutowareBridgeServer(bridge_cfg, autostart=False),
            launcher=launcher,
            episode_name=str(scenario_cfg.get("name", "")),
            native_autoware_interface=native,
        )

    if entity == "carla_driver":
        from autoware_carla_scenario import CarlaDriverEntity  # noqa: PLC0415
        from autoware_carla_scenario.driver import (  # noqa: PLC0415
            DriverClientConfig,
        )

        driver_cfg = cfg.get("driver")
        if driver_cfg is None:
            msg = (
                "ego.entity=carla_driver requires the 'driver' config group. "
                "Add 'driver: default' to the defaults list."
            )
            raise ValueError(msg)

        driver_dict = _with_fog_density_scale(_to_dict(driver_cfg), cfg)
        # ``driver.control: null`` means the default gains, as ``ros2.control`` does.
        control = _control_config(driver_dict, "driver")
        bridge_dict = driver_dict.get("bridge")
        if bridge_dict is None:
            return CarlaDriverEntity(
                DriverClientConfig.from_mapping(driver_dict), control
            )
        # A stack serving the policy's world (splatsim, say) dials the bridge the
        # framework hosts for the mission and reports when it is ready, as
        # autoware_carla_interface does for ego.entity=autoware.
        from autoware_carla_scenario import (  # noqa: PLC0415
            AutowareBridgeConfig,
            GrpcAutowareBridgeServer,
        )
        from autoware_carla_scenario.utils.config import (  # noqa: PLC0415
            checked_options,
            checked_section,
            null_as_default,
        )

        try:
            bridge_options = checked_options(
                AutowareBridgeConfig,
                checked_section("driver.bridge", bridge_dict) or {},
            )
        except ValueError as exc:
            # Headed as the autoware group's (_autoware_bridge_config).
            if str(exc).startswith("driver.bridge "):
                raise
            raise ValueError(f"driver.bridge: {exc}") from None
        try:
            bridge_cfg = AutowareBridgeConfig(
                **null_as_default(AutowareBridgeConfig, bridge_options)
            )
        except ValueError as exc:
            raise ValueError(f"driver.bridge.{exc}") from None
        scenario_cfg = cfg.get("scenario") or {}
        # autostart=False: the entity starts the bridge when its own scenario
        # starts, as AutowareEgoEntity does.
        return CarlaDriverEntity(
            DriverClientConfig.from_mapping(driver_dict),
            control,
            bridge=GrpcAutowareBridgeServer(bridge_cfg, autostart=False),
            bridge_config=bridge_cfg,
            episode_name=str(scenario_cfg.get("name", "")),
        )

    if entity == "ros2":
        # ROS 2 nodes drive the ego over DDS (docs/ros2.md). The DDS binding is
        # imported when the scenario starts, so a missing `ros2` extra fails
        # there, with the install hint, rather than here.
        from autoware_carla_scenario.entity.ros2_entity import (  # noqa: PLC0415
            Ros2EgoEntity,
        )
        from autoware_carla_scenario.ros2 import Ros2EgoConfig  # noqa: PLC0415

        ros2_cfg = cfg.get("ros2")
        if ros2_cfg is None:
            msg = (
                "ego.entity=ros2 requires the 'ros2' config group. "
                "Add 'ros2: default' to the defaults list."
            )
            raise ValueError(msg)

        if not _paced(cfg):
            # Freshness is judged in simulation time while the nodes publish on
            # the wall clock: unpaced, a 10 Hz node's commands land more than
            # stale_s apart and its source flaps between fresh and stale.
            # A run refuses a rate that is not a finite positive number
            # (check_run_config); built without that check, the runner reads
            # one as uncapped, so it is warned of as unset is.
            logger.warning(
                "ego.entity=ros2 without server.max_tick_rate_hz: the run is not "
                "paced to real time, so ROS 2 nodes may look stale; set "
                "server.max_tick_rate_hz=20 (docs/ros2.md, Timing)"
            )
        ros2_dict = _to_dict(ros2_cfg)
        control = _control_config(ros2_dict, "ros2")
        return Ros2EgoEntity(Ros2EgoConfig.from_mapping(ros2_dict), control)

    if entity == "hils":
        # ECUs under test drive the ego, through the HILS bridge resident on
        # the HILS PC (scenario-hils, docs/hils.md): the entity tells it when
        # the scenario starts and ends, and applies the ECUs' command.
        from autoware_carla_scenario.entity.hils_entity import (  # noqa: PLC0415
            HilsEgoConfig,
            HilsEgoEntity,
        )

        hils_cfg = cfg.get("hils")
        scenario_cfg = cfg.get("scenario") or {}
        return HilsEgoEntity(
            HilsEgoConfig.from_mapping(
                _to_dict(hils_cfg) if hils_cfg is not None else {}
            ),
            episode_name=str(scenario_cfg.get("name", "")),
        )

    # Unreachable: ego_entity_name() refused any other name.
    raise ValueError(f"Unknown ego.entity: {entity!r}")


def _legacy_traffic_manager_options(cfg: DictConfig) -> dict[str, Any]:
    """Return the TrafficManager options a config predating the group states.

    Only ``traffic_manager.port`` was ever settable that way, and a config that
    does not set it means the framework default.  The port is passed on as
    given, so :class:`TrafficManagerBackendConfig` refuses a bool, a float or
    a string rather than this turning ``true`` into port 1 (#1423).
    """
    legacy = cfg.get("traffic_manager")
    port = None if legacy is None else legacy.get("port")
    return {} if port is None else {"port": port}


def build_traffic_backend(cfg: DictConfig) -> TrafficBackend:
    """Build the traffic backend selected by ``cfg.traffic``.

    The traffic counterpart of :func:`build_ego_entity`: the config names a
    backend, the registry turns the name into one, and the runner never learns
    which it got.  A config with no ``traffic`` group at all selects CARLA's
    TrafficManager, which is what every scenario written before the group
    existed means.

    ``traffic_manager.port`` -- where the port lived before there was a traffic
    group, and what exported scenario packages still set -- keeps deciding the
    TrafficManager's port, by either of two routes.  A config that composes the
    ``traffic`` group gets it through the interpolation in
    ``conf/traffic/traffic_manager.yaml``, which is where the rest of the
    defaults live.  A config that predates the group and never composes it is
    read here instead: an interpolation in a file that config does not include
    cannot speak for it, and dropping the port would silently move such a run
    onto a different TrafficManager.

    Raises:
        ValueError: If ``traffic.backend`` names no registered backend.
    """
    # Third-party backends first, so a name from another package resolves.
    load_traffic_backend_plugins()

    traffic_cfg = cfg.get("traffic")
    config = (
        TrafficConfig.from_mapping(_to_dict(traffic_cfg))
        if traffic_cfg is not None
        else TrafficConfig(options=_legacy_traffic_manager_options(cfg))
    )

    # The groups declare their keys, null included, so that plain
    # `traffic.options.x=y` overrides work under Hydra's struct mode.  A
    # built-in backend reads a null itself (#2178): the default of an option
    # that does not take null -- so an interpolation that resolved to nothing
    # leaves the default alone -- and a value of its own for one that does
    # (sumo's `lateral_resolution_m: null`, its lane-based model).  Any other
    # backend, an installed package's, never sees a null: a key left at null
    # is a key the config did not set, and is dropped.
    options = dict(config.options)
    if not takes_null_options(config.backend):
        options = {key: value for key, value in options.items() if value is not None}

    backend = build_backend(
        config.backend, options, config.vehicles, config.pedestrians, config.bicycles
    )
    logger.info(
        "Traffic backend: %s (registered: %s)", config.backend, available_backends()
    )
    return backend


def _traffic_manager_port_text(cfg: DictConfig) -> str:
    """The TrafficManager port the backend *cfg* selects will use, as text.

    What :func:`build_traffic_backend` gives it, without building it: the
    backend's own option (``traffic.options.port``, sumo's
    ``traffic.options.tm_port``) when the ``traffic`` group is composed, and
    ``traffic_manager.port`` when it is not; the default when either is
    ``null``.  A backend with no TrafficManager port says so (#1863).
    """
    traffic = cfg.get("traffic")
    if traffic is None:
        port = _legacy_traffic_manager_options(cfg).get("port")
    else:
        backend = traffic_backend_name(traffic.get("backend"))
        option = tm_port_option(backend)
        if option is None:
            return f"- (traffic backend {backend})"
        options = traffic.get("options")
        port = None if options is None else options.get(option)
    return str(DEFAULT_TM_PORT if port is None else port)


def _apply_ego_config(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Give a scenario the registry did not build the ego the config selects.

    An injected ``build_scenario_fn`` brings its own :class:`EgoConfig`, so
    neither the entity nor the goal has reached the scenario yet.  On the
    registry path both arrive with the config :func:`build_ego_and_spawn`
    builds, which is why this is the only caller left: applying the goal twice
    would put two writers on one field.

    Each is applied only when the config names one: ``ego.entity=autopilot``
    (the default) yields no entity, and overwriting ``scenario.ego_entity`` with
    ``None`` would throw away an entity the scenario constructed in its own
    ``__init__``; a config that names no goal leaves the scenario's own, which
    it may have derived.  The goal lands on the scenario's ego config --
    ``scenario.goal_pose`` reads and writes
    :attr:`~autoware_carla_scenario.EgoConfig.goal_pose`.

    The way there travels with the goal, for the same reason and under the same
    rule.  Leaving it behind would drop ``ego.waypoint_lanelet_ids`` silently on
    this path, and a run that named its roads would be planned by whatever route
    reached the goal soonest -- the shortcut waypoints exist to prevent.  An
    empty list is "no particular way there was asked for", not "no way there",
    so like a missing goal it leaves the scenario's own.
    """
    entity = build_ego_entity(cfg)
    if entity is not None:
        scenario.ego_entity = entity

    goal_pose = build_goal_pose(cfg)
    if goal_pose is not None:
        scenario.goal_pose = goal_pose

    waypoint_poses = build_waypoint_poses(cfg)
    if waypoint_poses:
        scenario.waypoint_poses = waypoint_poses


def run_scenario_with_queue(
    scenario: BaseScenario,
    *,
    host: str = "localhost",
    port: int = 2000,
    tm_port: int = DEFAULT_TM_PORT,
    xodr_path: Path | None = None,
    overwrite_xodr: bool = False,
    opendrive_path: Path | None = None,
    lanelet2_path: Path | None = None,
    map_name: str | None = None,
    cooldown_seconds: float = 0.0,
    cooldown_max_retries: int = 0,
    output_dir: Path = Path("scenario_outputs"),
    timeout_seconds: float = 60.0,
    max_tick_rate_hz: float | None = None,
    projector_type: str | None = None,
    traffic_backend: TrafficBackend | None = None,
    odd: str | None = None,
    autoware_launcher: AutowareLauncher | None = None,
    case: OddCase | None = None,
    server_extra_args: list[str] | None = None,
    map_source: str | None = None,
    rerun: RerunConverter | None = None,
    t4: T4Recorder | None = None,
    world_reset: str = "reload",
    reload_every: int = 0,
    t4scene: T4SceneExporter | None = None,
    backends: BackendPool | None = None,
    file_stem: str | None = None,
    map_pack: MapPack | None = None,
) -> ScenarioResult:
    """Run a single pre-built scenario using :class:`ScenarioQueue`.

    *case* is the case drawn from the ODD that it runs, recorded with its
    outcome in its coverage file (:meth:`ScenarioRunner.run_scenario`).
    *map_source* is where the map
    came from, which the run's trajectory log names.  *server_extra_args* are passed to
    ``CarlaUnreal.sh`` when the run starts the server (:func:`server_extra_args`).
    *rerun* converts the run's trajectory log into a Rerun recording once it
    is closed (:func:`build_rerun_converter`); the caller waits for it.
    *t4* writes the run as a T4 dataset (:func:`build_t4_recorder`).
    *world_reset* and *reload_every* are how the queue resets the world
    (:func:`world_reset_settings`).  *t4scene* writes the run's trajectory log
    as a T4Scene once it is closed (:func:`build_t4scene_exporter`).
    *file_stem* is the stem the run's files are named by
    (``{file_stem}_result.json`` ...); ``None`` names them by the scenario's
    class, numbered past the runs already in *output_dir*
    (:meth:`ScenarioQueue.add`).
    *map_pack* is the content pack *map_name* is mounted from
    (:func:`build_map_pack`).

    This is the extracted orchestration logic that was previously embedded
    inside :func:`run_scenario`.  Downstream projects can use this to execute
    their own ``BaseScenario`` subclasses without duplicating the queue setup::

        scenario = MyCustomScenario(ego, config=my_cfg, ...)
        result = run_scenario_with_queue(
            scenario, host="localhost", port=2000,
            output_dir=Path("outputs"),
        )
    """
    queue = ScenarioQueue(
        host=host,
        port=port,
        tm_port=tm_port,
        xodr_path=xodr_path,
        overwrite_xodr=overwrite_xodr,
        opendrive_path=opendrive_path,
        lanelet2_path=lanelet2_path,
        map_name=map_name,
        cooldown_seconds=cooldown_seconds,
        cooldown_max_retries=cooldown_max_retries,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        max_tick_rate_hz=max_tick_rate_hz,
        projector_type=projector_type,
        traffic_backend=traffic_backend,
        odd=odd,
        autoware_launcher=autoware_launcher,
        server_extra_args=server_extra_args,
        map_source=map_source,
        rerun=rerun,
        t4=t4,
        world_reset=world_reset,
        reload_every=reload_every,
        t4scene=t4scene,
        backends=backends,
        map_pack=map_pack,
    )
    queue.add(scenario, case=case, file_stem=file_stem)
    with queue:
        results = queue.run_all()
    return results[0]


def _server_number(
    cfg: DictConfig, key: str, default: float | None, *, positive: bool
) -> float | None:
    """``server.<key>`` of *cfg* (*default* when absent), checked: *None* for
    null, else a finite number -- an int or a float, not a bool -- that is
    above 0 when *positive*, at least 0 otherwise.

    Raises:
        ValueError: naming the key.
    """
    value = (cfg.get("server") or {}).get(key, default)
    if value is None:
        return None
    bound = "> 0" if positive else ">= 0"
    if not _finite_number(value) or (value <= 0 if positive else value < 0):
        raise ValueError(
            f"server.{key} must be a finite number {bound} (got {shown(value)})"
        )
    return float(value)


def _cooldown_seconds(cfg: DictConfig) -> float:
    """``server.cooldown_seconds``, checked: a finite number of seconds, 0 or
    more; absent or null is the default, 2 s -- ``ServerConfig``'s, the
    shipped ``server/localhost.yaml``'s and the docs' (#1267, #2161), not
    none.  A NaN or negative cooldown used to be read as none.

    Raises:
        ValueError: naming the key.
    """
    cooldown = _server_number(cfg, "cooldown_seconds", None, positive=False)
    return ServerConfig.cooldown_seconds if cooldown is None else cooldown


def _max_tick_rate_hz(cfg: DictConfig) -> float | None:
    """``server.max_tick_rate_hz``, checked: null (uncapped) or a finite rate
    above 0.  The runner used to read a NaN or non-positive rate as uncapped.

    Raises:
        ValueError: naming the key.
    """
    return _server_number(cfg, "max_tick_rate_hz", None, positive=True)


def _paced(cfg: DictConfig) -> bool:
    """Whether *cfg*'s ``server.max_tick_rate_hz`` caps the tick rate."""
    try:
        return _max_tick_rate_hz(cfg) is not None
    except ValueError:
        return False


def world_reset_settings(cfg: DictConfig) -> tuple[str, int]:
    """``server.world_reset`` and ``server.reload_every``, checked.

    Only an absent or null ``world_reset`` is ``reload``: an empty one,
    ``false`` or ``0`` is no strategy, and refused as any other (#2143).

    Raises:
        ValueError: naming the key, when ``world_reset`` is neither ``reload``
            nor ``soft``, or ``reload_every`` is not a whole number (``2`` or
            ``2.0``; not ``2.5`` or a boolean) or is negative.
    """
    from ..world_reset import (  # noqa: PLC0415
        RESET_RELOAD,
        RESET_STRATEGIES,
        check_world_reset,
    )

    server = cfg.get("server") or {}
    value = server.get("world_reset")
    if value is None:
        world_reset = RESET_RELOAD
    elif isinstance(value, str):
        world_reset = value
    else:
        # No strategy is spelt as anything but text (#2143).
        raise ValueError(
            f"server.world_reset must be one of {list(RESET_STRATEGIES)}, "
            f"got {shown(value)}."
        )
    # null is 0, as it always was.
    reload_every = (
        whole_number_setting("server.reload_every", server.get("reload_every", 0)) or 0
    )
    try:
        check_world_reset(world_reset, reload_every)
    except ValueError as exc:
        raise ValueError(f"server.{exc}") from None
    return world_reset, reload_every


def _fog_density_scale(cfg: DictConfig) -> float | None:
    """``server.fog_density_scale``, or *None* to keep the server's own.

    Raises:
        ValueError: naming the key, unless it is null or a finite number
            above 0, as a LiDAR's own ``fog_density_scale`` must be: without
            a LiDAR a NaN would reach the server's command line.
    """
    return _server_number(cfg, "fog_density_scale", None, positive=True)


def server_extra_args(cfg: DictConfig) -> list[str]:
    """The arguments a server the run starts is started with, from ``server``."""
    scale = _fog_density_scale(cfg)
    return [] if scale is None else fog_density_scale_args(scale)


def _with_fog_density_scale(driver: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """*driver* with ``server.fog_density_scale`` on every LiDAR that sets none."""
    scale = _fog_density_scale(cfg)
    lidars = driver.get("lidars")
    if scale is None or not lidars:
        return driver
    return {
        **driver,
        "lidars": [{"fog_density_scale": scale, **dict(lidar)} for lidar in lidars],
    }


def _map_source(cfg: DictConfig, map_paths: Any) -> str | None:
    """Where the map came from: ``map.directory``, else ``map.pack`` (as
    ``carla-pack:OWNER/REPO/PACK@VERSION``), else ``map.source``.

    The same order :func:`resolve_map_paths` resolves them in: a directory on
    this machine wins over a pack, and a pack over a source, so the map the
    run read is the one named.  The pack is named as *map_paths* resolved it
    (:attr:`~autoware_carla_scenario.maps.config.MapPaths.pack`), with its
    version, not as configured -- ``map.pack=OWNER/REPO`` searches a pack
    repository, and a pack without ``@VERSION`` is its newest -- so the log
    says which pack the run read (#2198).  A pack that carries no Lanelet2
    map leaves it to ``map.source``, which is then the source named.
    """
    # Read as resolve_map_paths reads them: a blank setting is unset (#2198).
    directory = _map_setting(cfg, "directory")
    if directory:
        return str(directory)
    source = _map_setting(cfg, "source")
    pack = getattr(map_paths, "pack", None)
    if isinstance(pack, str) and pack:
        resolved = getattr(map_paths, "resolved", None)
        # The pack's Lanelet2 map is read as a directory (no source); a
        # resolved source means the Lanelet2 map is map.source's.
        from_pack = resolved is not None and resolved.source is None
        if from_pack or not source:
            return f"carla-pack:{pack}"
    return str(source) if source else None


def build_map_pack(cfg: DictConfig, map_paths: Any) -> MapPack | None:
    """The content pack the run's CARLA map is mounted from (``map.pack``),
    or ``None`` when the config names none.

    The pack is the one *map_paths* (:func:`resolve_map_paths`) read the map
    files from -- its resolved reference and SHA-256 -- so the server mounts
    the very artifact the files came from; it is installed into
    ``server.packs_dir`` (:func:`~autoware_carla_scenario.maps.pack.packs_dir`).

    Raises:
        ValueError: ``map.pack`` or ``map.pack_sha256`` is malformed, or
            ``server.packs_dir`` is not text or is blank (#2198).
    """
    reference = getattr(map_paths, "pack", None)
    if not isinstance(reference, str) or not reference:
        return None
    sha256 = getattr(map_paths, "pack_sha256", None)
    configured = (cfg.get("server") or {}).get("packs_dir")
    if configured is not None and not isinstance(configured, str):
        raise ValueError(f"server.packs_dir must be a path, got {configured!r}")
    return MapPack(
        reference,
        sha256=sha256 if isinstance(sha256, str) and sha256 else None,
        packs_dir=packs_dir(configured),
    )


def _block(cfg: DictConfig, name: str) -> Any:
    """*cfg*'s ``name`` block: an empty mapping when it is absent or null
    (the default, nothing set), the block itself when it is a mapping.

    One that is not a mapping (``rerun: true``, ``false``, ``0``, ``''``) is
    refused, as ``metrics`` is (:func:`metrics_heavy`), rather than read as
    unset -- ``false`` turned the block's switches off without a word -- or
    crash on ``.get`` (``rerun: true``, #2298).

    Raises:
        ValueError: naming the block, for a value that is not a mapping.
    """
    value = cfg.get(name)
    if value is None:
        return {}
    if not isinstance(value, (DictConfig, Mapping)):
        raise ValueError(f"{name} must be a mapping, got {shown(value)}")
    return value


def _switch(cfg: DictConfig, block: str, key: str) -> bool:
    """The true/false switch ``<block>.<key>`` of *cfg*, off when absent or
    null (#2188), as is every switch of a block that is absent or null.

    A boolean or its spelling (``"false"``, ``"yes"``, ``0``: what an
    ``${oc.env:VAR,false}`` or a quoted override gives), not its truthiness:
    ``bool("false")`` was true, and so was a typo (#1753).

    Raises:
        ValueError: naming the block, for a block that is not a mapping
            (:func:`_block`), or the key, for a value that spells neither
            (:func:`~autoware_carla_scenario.config_flags.check_flag`).
    """
    settings = _block(cfg, block)
    return check_flag(f"{block}.{key}", _given(settings, key, False))


def _rerun_settings(cfg: DictConfig) -> tuple[bool, bool]:
    """``rerun.enabled`` and ``rerun.strict`` (both off when absent).

    Raises:
        ValueError: Either is not true or false (:func:`_switch`).
    """
    return _switch(cfg, "rerun", "enabled"), _switch(cfg, "rerun", "strict")


def _rerun_draw_radius(cfg: DictConfig) -> float | None:
    """``rerun.draw_radius_m``: ``None`` when unset.

    An int too big for a float (``10**400``) is no finite distance either
    (:func:`_finite_number`, #1652), refused with the same message rather
    than the ``OverflowError`` of ``math.isfinite()`` (#1928).

    Raises:
        ValueError: It is not a finite distance, or ``rerun`` is not a
            mapping (:func:`_block`).
    """
    settings = _block(cfg, "rerun")
    value = settings.get("draw_radius_m", None)
    if value is None:
        return None
    if not _finite_number(value) or value < 0.0:
        raise ValueError(
            f"rerun.draw_radius_m must be a finite distance, got {value!r}"
        )
    return float(value)


#: ``rerun.lidar_max_points`` when it is absent or null: the default the
#: shipped config sets and docs/usage.md documents.
_DEFAULT_LIDAR_MAX_POINTS = 65536


def _rerun_lidar_max_points(cfg: DictConfig) -> int | None:
    """``rerun.lidar_max_points`` when ``rerun.lidar`` is set; ``None`` otherwise.

    Absent or null is the default, :data:`_DEFAULT_LIDAR_MAX_POINTS` (65536);
    an explicit 0 keeps every point of a sweep (#2298: null kept every point).
    An integral float (``1e5`` in YAML) or a string of an integer (what
    ``${oc.env:...}`` gives) counts as that integer, as ``int()`` read them
    before (#1226).

    Raises:
        ValueError: It is not a non-negative integer (a bool is not one),
            ``rerun.lidar`` is not true or false (:func:`_switch`), or
            ``rerun`` is not a mapping (:func:`_block`).
    """
    if not _switch(cfg, "rerun", "lidar"):
        return None
    value = _given(_block(cfg, "rerun"), "lidar_max_points", _DEFAULT_LIDAR_MAX_POINTS)
    points: int | None = None
    if isinstance(value, bool):
        points = None
    elif isinstance(value, int):
        points = value
    elif isinstance(value, float) and value.is_integer():
        points = int(value)
    elif isinstance(value, str):
        try:
            points = int(value)
        except ValueError:
            points = None
    if points is None or points < 0:
        raise ValueError(
            "rerun.lidar_max_points must be a non-negative integer "
            f"(0 keeps every point), got {value!r}"
        )
    return points


def build_rerun_converter(*cfgs: DictConfig) -> RerunConverter | None:
    """What converts each run's trajectory log into a Rerun recording, when
    any of *cfgs* sets ``rerun.enabled`` (strict when any sets
    ``rerun.strict``); ``None`` otherwise.  With ``rerun.lidar`` it records
    each run's LiDAR sweeps too (the largest ``rerun.lidar_max_points`` of
    those asking, 0 being all), and with ``rerun.cameras`` its camera frames.
    ``rerun.draw_radius_m`` is the largest of those asking for recordings; one
    of them without it means no limit, since one converter serves the batch.

    Called before anything runs, so a run asking for recordings without the
    ``rerun`` extra installed, or with a ``rerun.lidar_max_points`` that is not
    a non-negative integer or a ``rerun.draw_radius_m`` that is not a finite
    distance in any of its cases, is refused at once.

    Raises:
        ValueError: A case asking for LiDAR sweeps sets a
            ``rerun.lidar_max_points`` that is not a non-negative integer, or
            a case asking for recordings a ``rerun.draw_radius_m`` that is not
            a finite distance.
        RerunUnavailableError: The extra is not installed.
        LidarRerunUnavailableError: ``rerun.lidar`` is set and the Rerun SDK
            is not installed.
    """
    settings = [_rerun_settings(cfg) for cfg in cfgs]
    if not any(enabled for enabled, _ in settings):
        return None
    lidar = [
        points
        for cfg, (enabled, _) in zip(cfgs, settings)
        if enabled and (points := _rerun_lidar_max_points(cfg)) is not None
    ]
    strict = any(strict for _, strict in settings)
    cameras = any(
        enabled and _switch(cfg, "rerun", "cameras")
        for cfg, (enabled, _) in zip(cfgs, settings)
    )
    sensors: dict[str, Any] = {}
    if lidar:
        sensors["lidar_max_points"] = 0 if 0 in lidar else max(lidar)
    if cameras:
        sensors["cameras"] = True
    radii = [
        _rerun_draw_radius(cfg) for cfg, (enabled, _) in zip(cfgs, settings) if enabled
    ]
    limits = [radius for radius in radii if radius is not None]
    if len(limits) == len(radii):
        sensors["draw_radius_m"] = max(limits)
    return RerunConverter(strict=strict, **sensors)


def build_t4_recorder(*cfgs: DictConfig) -> T4Recorder | None:
    """What writes each run as a T4 dataset, when the first of *cfgs* that
    sets ``t4.enabled`` does: with the driver policy's sensors and step
    (``driver``) and its ``t4`` settings; ``None`` otherwise.

    Called before anything runs, so a run asking for datasets without the
    ``t4`` extra installed is refused at once.

    Raises:
        T4UnavailableError: The extra is not installed.
        ValueError: The ``t4`` or ``driver`` settings are not valid.
    """
    settings = _t4_settings(*cfgs)
    return None if settings is None else T4Recorder(settings)


def _t4_settings(*cfgs: DictConfig) -> T4Settings | None:
    """The settings :func:`build_t4_recorder` builds its recorder with, read
    and checked, needing no extra; ``None`` when no case sets ``t4.enabled``.

    Raises:
        ValueError: The ``t4`` or ``driver`` settings are not valid.
    """
    from autoware_carla_scenario.driver import DriverClientConfig  # noqa: PLC0415

    for cfg in cfgs:
        if not _switch(cfg, "t4", "enabled"):
            continue
        t4_cfg = cfg.get("t4") or {}
        driver_cfg = cfg.get("driver")
        if driver_cfg is None:
            raise ValueError(
                "t4.enabled takes its sensors from the 'driver' config group "
                "(driver.lidars, driver.cameras): add 'driver: default' to the "
                "defaults list"
            )
        driver = DriverClientConfig.from_mapping(
            _with_fog_density_scale(_to_dict(driver_cfg), cfg)
        )
        return T4Settings.from_driver(
            driver, _to_dict(t4_cfg) if isinstance(t4_cfg, DictConfig) else t4_cfg
        )
    return None


def build_backend_pool(cfg: DictConfig) -> BackendPool:
    """What starts the backends the run's scenarios need, per ``backends``.

    Each backend reads its own block (``backends.ardy``, ...); one a scenario
    does not need is never started.
    """
    backends_cfg = cfg.get("backends")
    return BackendPool(None if backends_cfg is None else _to_dict(backends_cfg))


def build_t4scene_exporter(*cfgs: DictConfig) -> T4SceneExporter | None:
    """What writes each run's trajectory log as a T4Scene as the run ends,
    when the first of *cfgs* that sets ``t4scene.enabled`` does, with its
    ``step_s`` and ``max_distance_m``; ``None`` otherwise.  An absent or null
    ``step_s`` is 0.1 s (#2161) and an absent ``max_distance_m`` 100 m; a null
    ``max_distance_m`` takes every road user, as it always did.

    Raises:
        ValueError: ``t4scene.step_s`` is a bool, not a number or not a
            positive multiple of the tick, or ``t4scene.max_distance_m`` is a
            bool, not a number or not positive and finite.
    """
    for cfg in cfgs:
        if not _switch(cfg, "t4scene", "enabled"):
            continue
        t4scene_cfg = cfg.get("t4scene") or {}
        # Handed over as read: the exporter refuses a bool, which float()
        # would turn into a 1 s step or a 1 m distance (#1502, #1231).
        return T4SceneExporter(
            step_s=_given(t4scene_cfg, "step_s", 0.1),
            max_distance_m=t4scene_cfg.get("max_distance_m", 100.0),
        )
    return None


def canvas_config(*cfgs: DictConfig) -> str | None:
    """The carla_canvas configuration *cfgs* name (``canvas.config``), or ``None``.

    An absent or null ``canvas.config`` names none; anything else must be a
    non-empty path (``""`` or ``false`` is refused, not read as unset; #2188).

    Raises:
        ValueError: naming ``canvas.config``, for a value that is no path;
            and when they name more than one: a process loads one.
    """
    names: set[str] = set()
    for cfg in cfgs:
        value = (cfg.get("canvas") or {}).get("config")
        if value is None:
            continue
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"canvas.config must be a file path, got {value!r}")
        names.add(value)
    if len(names) > 1:
        raise ValueError(
            f"the cases name {len(names)} canvas.config files ({sorted(names)}); "
            "the sensors of one process are rendered by one"
        )
    return next(iter(names), None)


def apply_canvas(*cfgs: DictConfig) -> Path | None:
    """Have the sensors *cfgs*' ``canvas.config`` names rendered by carla_canvas
    (:func:`autoware_carla_scenario.canvas.use_canvas`); its path, or ``None``.

    Called before anything runs: typesafe_carla loads the plugin when the
    first sensor listens.
    """
    return use_canvas(canvas_config(*cfgs))


def metrics_heavy(cfg: DictConfig) -> bool | None:
    """*cfg*'s ``metrics.heavy``, read strictly
    (:func:`~autoware_carla_scenario.config_flags.check_flag`: ``"false"`` is
    off, a typo refused, #1761); ``None`` when it is unset or null, which
    leaves the decision to ``$AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS``.

    A ``metrics`` block that is absent or null is the default (nothing
    set); one that is not a mapping (``false``, ``0``, ``''``, a list) is
    refused rather than read as unset.

    Raises:
        ValueError: naming ``metrics``, for a block that is not a mapping, or
            ``metrics.heavy``, for a value that is not a switch.
    """
    block = cfg.get("metrics")
    if block is None:
        return None
    if not isinstance(block, (DictConfig, Mapping)):
        raise ValueError(f"metrics must be a mapping, got {shown(block)}")
    value = block.get("heavy")
    if value is None:
        return None
    return check_flag("metrics.heavy", value)


def apply_metrics(*cfgs: DictConfig) -> bool:
    """Turn the heavy CommonRoad metrics on or off as the first of *cfgs*
    that sets ``metrics.heavy`` (to ``true`` or ``false``) says; when none
    does, as ``$AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS`` says (on unless it is
    ``0``).  Whether they are on.

    The user's variable is left as it is: the decision goes to
    ``$AUTOWARE_CARLA_SCENARIO_HEAVY_METRICS_DECIDED``
    (:data:`~autoware_carla_scenario.commonroad.session.HEAVY_METRICS_DECIDED_ENV`),
    which carries it to the runs' own processes and is never read here.  Each
    call (a run of a Hydra ``--multirun``) decides afresh from its *cfgs* and
    the user's variable: one case's ``metrics.heavy`` does not carry over to
    the next case's."""
    from ..commonroad.session import (  # noqa: PLC0415
        HEAVY_METRICS_DECIDED_ENV,
        _heavy_metrics_from_user,
        configure_commonroad_metrics,
    )

    heavy: bool | None = None
    for cfg in cfgs:
        heavy = metrics_heavy(cfg)
        if heavy is not None:
            break
    if heavy is None:
        heavy = _heavy_metrics_from_user()
    os.environ[HEAVY_METRICS_DECIDED_ENV] = "1" if heavy else "0"
    configure_commonroad_metrics(heavy_metrics=heavy)
    return heavy


def _block_key(cfg: DictConfig, name: str) -> str:
    """*cfg*'s ``name`` block as a string two equal blocks share."""
    value = cfg.get(name)
    if isinstance(value, DictConfig):
        value = _to_dict(value)
    return json.dumps(value, sort_keys=True, default=str)


def batch_settings_cases(
    configs: list[DictConfig], *, heavy_metrics: bool
) -> dict[str, Any]:
    """Which of a batch's *configs* decide its batch-level settings, by index:
    what a SLURM batch's manifest records (``batch_settings``), so that every
    task decides them as a local batch does from all its cases, whichever of
    them the task runs (slurm/worker.py).

    - ``first_case``: the case whose config the queue is built from
      (:func:`_batch_queue`: the server, traffic, ``t4``, ``t4scene``, world
      reset and backend settings), the batch's first;
    - ``rerun_cases``: a case for each distinct ``rerun`` block, none when no
      case sets ``rerun.enabled``: :func:`build_rerun_converter` of these is
      that of every case;
    - ``canvas_case``: the first case naming a ``canvas.config``, or ``None``
      (:func:`canvas_config`);
    - ``heavy_metrics``: what :func:`apply_metrics` decided for the batch.

    A few cases' configs, not every one's: a task composes these and applies
    the same builders to them, at its node's paths.
    """
    rerun_cases: list[int] = []
    if any(_rerun_settings(cfg)[0] for cfg in configs):
        seen: set[str] = set()
        for index, cfg in enumerate(configs):
            key = _block_key(cfg, "rerun")
            if key not in seen:
                seen.add(key)
                rerun_cases.append(index)
    canvas_case = next(
        (
            index
            for index, cfg in enumerate(configs)
            if (cfg.get("canvas") or {}).get("config")
        ),
        None,
    )
    return {
        "first_case": 0,
        "rerun_cases": rerun_cases,
        "canvas_case": canvas_case,
        "heavy_metrics": bool(heavy_metrics),
    }


def rerun_errors(
    results: list[ScenarioResult], conversions: list[RerunConversion]
) -> dict[int, str]:
    """Why each case's Rerun recording failed, by its position in *results*:
    the cases whose ``rerun_recording`` is among the failed *conversions*.

    A recording converted more than once (a retried case, which writes under
    the stem of the attempt it replaces) goes by its last conversion: the
    checked :func:`.rerun_report.rerun_errors`, over the cases' recordings."""
    return _rerun_errors([result.rerun_recording for result in results], conversions)


def _finish_rerun(rerun: RerunConverter | None) -> list[RerunConversion]:
    """Wait for *rerun*'s conversions, end its thread, and print how they went."""
    if rerun is None:
        return []
    conversions = rerun.close()
    print(rerun_report(conversions))  # noqa: T201
    return conversions


#: What a run's ``video`` settings are now: none of them does anything.
VIDEO_REMOVED = (
    "video.* is set ({settings}), but autoware_carla_scenario renders no video "
    "any more (removed in 15.0.0): it is ignored. To see a run, make a Rerun "
    "recording of its {{Scenario}}.trajectory.jsonl and open that in the Rerun "
    "viewer: pip install 'autoware-carla-scenario[rerun,viewer]' and run with "
    "rerun.enabled=true (or, after a run, python -m "
    "autoware_carla_scenario.rerun <output dir>), then rerun {{Scenario}}.rrd; "
    "the CARLA recording ({{Scenario}}.log) is still written, for CARLA's "
    "replayer (docs/usage.md#seeing-a-run)."
)


def _video_settings(cfg: DictConfig) -> dict[str, Any]:
    """The ``video.*`` values *cfg* sets (none, in a config of 15.0.0 on)."""
    video = cfg.get("video")
    if video is None:
        return {}
    container = OmegaConf.to_container(video) if OmegaConf.is_config(video) else video
    if not isinstance(container, dict):
        return {}
    return {str(key): value for key, value in container.items() if value is not None}


def warn_if_video_set(*cfgs: DictConfig) -> bool:
    """Warn once, saying what to use instead, when any of *cfgs* sets
    ``video.*``, whose video is no longer rendered; whether one did."""
    settings: dict[str, Any] = {}
    for cfg in cfgs:
        settings = _video_settings(cfg)
        if settings:
            break
    if not settings:
        return False
    message = VIDEO_REMOVED.format(
        settings=", ".join(f"video.{k}={v}" for k, v in sorted(settings.items()))
    )
    logger.warning(message)
    print(f"Warning: {message}", file=sys.stderr)  # noqa: T201
    return True


def _odd_case(cfg: DictConfig) -> OddCase | None:
    """The case drawn from the ODD that the config runs (``odd_case``, set by
    a sweep drawing from it), or ``None``."""
    value = cfg.get(ODD_CASE_KEY)
    return None if value is None else decode_case(str(value))


def _odd_spec(cfg: DictConfig) -> str | None:
    """The ``odd`` the config names (docs/odd.md), or ``None`` for the default."""
    value = cfg.get("odd")
    return None if value is None else str(value)


def _to_dict(cfg_node: DictConfig) -> dict:  # type: ignore[type-arg]
    """Convert an OmegaConf node to a plain dict (typed helper)."""
    container = OmegaConf.to_container(cfg_node, resolve=True)
    assert isinstance(container, dict)  # noqa: S101
    return container


def _is_multirun() -> bool:
    """Return ``True`` when the Hydra job running is one of a multirun.

    The mode Hydra settled on, however it was set (``--multirun``, a
    ``hydra.mode=MULTIRUN`` override or a config); what ``sys.argv`` asks for
    when no Hydra job has set it.
    """
    if HydraConfig.initialized():
        return HydraConfig.get().mode == RunMode.MULTIRUN
    return _multirun_requested_in_argv(sys.argv)


def _find_scenario_yaml(name: str) -> Path:
    """Return the YAML path for scenario *name* across registered conf dirs.

    Falls back to the built-in dir (for display only) when the file is not
    found on disk, e.g. for a config coming from Hydra's ConfigStore.
    """
    for conf_dir in get_conf_dirs():
        candidate = conf_dir / "scenario" / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    return _CONF_DIR / "scenario" / f"{name}.yaml"


def _resolve_scenario_glob(pattern: str) -> list[str]:
    """Glob ``conf/scenario/{pattern}.yaml`` across every registered conf dir.

    Each returned name is a Hydra config path relative to ``scenario/`` without
    the ``.yaml`` suffix (e.g. ``"intersection_passing/left_turn"``).  Matches
    from external scenario packages are included alongside the built-ins.  The
    viewer's launcher resolves its patterns with the same
    :func:`~autoware_carla_scenario.scenario_glob.resolve_scenario_glob`.
    An invalid pattern (a ``**`` that is not a whole path component, a
    leading ``/`` or a ``..`` component), like
    one that matches nothing, is reported on stderr and exits with status 1.
    """
    conf_dirs = get_conf_dirs()
    try:
        names = resolve_scenario_glob(pattern, conf_dirs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if not names:
        print(  # noqa: T201
            f"Error: no scenario configs match pattern '{pattern}' "
            f"under {[str(d) for d in scenario_dirs(conf_dirs)]}",
            file=sys.stderr,
        )
        sys.exit(1)
    return names


def _compose_config(scenario_name: str, overrides: list[str]) -> DictConfig:
    """Build a resolved Hydra config for *scenario_name* using the Compose API."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"scenario={scenario_name}", *overrides],
        )
    return cfg


def _compose_case(name: str, overrides: list[str]) -> DictConfig:
    """:func:`_compose_config` of a batch's scenario *name*; a command line
    that does not compose (a misspelled key, a value of the wrong type) is
    reported as one ``Error: <name>: ...`` message, with the closest key, and
    exits 2, as ``scenario-check`` and ``scenario-expand`` report it (#2294).
    ``HYDRA_FULL_ERROR=1`` raises it instead.
    """
    from .compose_errors import compose_or_report  # noqa: PLC0415

    cfg = compose_or_report(name, overrides, named=True)
    if cfg is None:
        sys.exit(2)
    return cfg  # type: ignore[no-any-return]


def _has_sweep(cfg: DictConfig) -> bool:
    """Whether *cfg* is a logical scenario: one with a sweep to expand.

    Raises:
        ValueError: If its ``sweep`` is neither null nor a mapping
            (:func:`~autoware_carla_scenario.sweeper.expand.check_sweep_shape`),
            which :func:`check_batch_configs` refuses before anything runs
            (#2279).
    """
    from ..sweeper.expand import check_sweep_shape  # noqa: PLC0415 -- loads Lanelet2

    sweep = OmegaConf.select(cfg, "sweep")
    # Its shape only: an interpolation inside it, which only expanding the
    # sweep reads, is left unresolved, as a run that does not expand it
    # always left it.
    check_sweep_shape(sweep)
    if sweep is None:
        return False
    if isinstance(sweep, DictConfig):
        return any(
            OmegaConf.select(sweep, key)
            for key in ("constraints", "route", "odd_sample")
        )
    return any(sweep.get(key) for key in ("constraints", "route", "odd_sample"))


def _expand_batch(
    scenario_names: list[str],
    overrides: list[str],
    *,
    documents: dict[str, Any] | None = None,
) -> list[BatchCase]:
    """Every concrete case of *scenario_names*, in order: each scenario's sweep
    expanded the way the lanelet sweeper expands it (``sweep.constraints``,
    ``sweep.route``, ``sweep.odd_sample``), a scenario without one as itself.

    Scenarios that draw from one ODD go on from each other's draws: a later
    one aims at what the earlier ones left uncovered (``strategy: coverage``)
    and does not draw their values again.

    Exits 1 before anything runs when a scenario's config sets a value its
    runs would refuse, or it is composed on another map than its document
    names (:func:`check_batch_configs`, checked before its sweep reads any
    map), its ``controls:`` are invalid,
    its sweep is invalid (a ``sweep.case_index`` past its cases, a constraint
    or binding that does not parse, a Lanelet2 map that is missing, unreadable
    or cannot be resolved, ...: the ``ValueError`` expand_config
    refuses it with) or its sweep expands to no case, as the lanelet sweeper
    refuses one (#1407): every such scenario is reported, each named, since a
    batch missing a scenario it was asked for must not look like a complete
    one. A scenario without a sweep is one case, never refused.
    ``HYDRA_FULL_ERROR=1`` raises an invalid sweep's error, or a refused
    config value's, instead.  *documents* are the scenario documents already
    loaded, by name (:func:`check_scenario_maps`), filled here."""
    from ..odd.controls import ControlsError, format_diagnostics  # noqa: PLC0415
    from ..odd.sampler import OddDraws  # noqa: PLC0415
    from ..sweeper.expand import (  # noqa: PLC0415 -- loads Lanelet2
        EMPTY_SWEEP_MESSAGE,
        expand_config,
    )

    drawn = OddDraws()
    cases: list[BatchCase] = []
    # Each refused scenario's lines, in order: all of them are reported.
    errors: list[str] = []
    for name in scenario_names:
        # Composed outside the try: a ValueError there is no sweep's.
        cfg = _compose_case(name, overrides)
        try:
            # Before the sweep resolves, fetches or searches any map, as the
            # lanelet sweeper checks its config before it expands it: one
            # composed on another map than its document names would be
            # searched on that one (#1428), and a value every case would
            # refuse is refused once its map is read (#1430).  What only a
            # case's own overrides set is checked once the cases are composed
            # (run_batch).
            check_batch_configs([cfg], [name], documents=documents)
        except RunConfigError as exc:
            if os.environ.get("HYDRA_FULL_ERROR") == "1":
                raise
            message = str(exc)
            if not message.startswith(f"{name}: "):
                message = f"{name}: {message}"
            errors.append(f"Error: {message}")
            continue
        try:
            expanded = expand_config(cfg, overrides, drawn=drawn)
        except ControlsError as exc:
            errors.extend(format_diagnostics(exc.diagnostics, name))
            continue
        except ValueError as exc:
            if os.environ.get("HYDRA_FULL_ERROR") == "1":
                raise
            errors.append(f"Error: {name}: {exc}")
            continue
        logger.info("%s: %d case(s)", name, len(expanded))
        if not expanded:
            errors.append(f"Error: {name}: {EMPTY_SWEEP_MESSAGE}")
        cases.extend((name, case) for case in expanded)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)  # noqa: T201
        sys.exit(1)
    return cases


def _log_batch_plan(
    scenario_names: list[str],
    configs: list[DictConfig],
    overrides: list[str],
    *,
    labels: list[str] | None = None,
    case_overrides: list[list[str]] | None = None,
) -> None:
    """Log which YAML configs will be loaded and their resolved parameters."""
    sep = "=" * 60
    thin = "-" * 60
    logger.info(sep)
    logger.info(
        "Batch execution plan: %d scenario(s)%s",
        len(scenario_names),
        f"  (extra overrides: {overrides})" if overrides else "",
    )
    logger.info(sep)

    for i, (name, cfg) in enumerate(zip(scenario_names, configs), 1):
        yaml_path = _find_scenario_yaml(name)
        logger.info(thin)
        logger.info(
            "[%d/%d] %s", i, len(scenario_names), labels[i - 1] if labels else name
        )
        logger.info("  config file : %s", yaml_path)
        if case_overrides and case_overrides[i - 1]:
            logger.info("  sweep case  : %s", " ".join(case_overrides[i - 1]))
        logger.info("  map         : %s", cfg.map.name)
        logger.info("  server      : %s:%s", cfg.server.host, cfg.server.port)
        logger.info("  TM port     : %s", _traffic_manager_port_text(cfg))
        goal_pose = build_goal_pose(cfg)
        # Each as the config gives it (%s): a null is its default, as the
        # build reads it, and a quoted id is shown, not a logging error.
        logger.info(
            "  ego         : %s (%s km/h) spawn=lanelet:%s s:%s goal=%s",
            _given(cfg.ego, "vehicle_type", EgoVehicleConfig.vehicle_type),
            _given(cfg.ego, "initial_speed_kmh", EgoVehicleConfig.initial_speed_kmh),
            cfg.ego.get("spawn_lanelet_id"),
            _given(cfg.ego, "spawn_s", 0.0),
            (
                f"lanelet:{goal_pose.lanelet_id} s:{goal_pose.s:.1f}"
                if goal_pose is not None
                else "none"
            ),
        )
        # Log all scenario-specific parameters.
        logger.info("  scenario parameters:")
        scenario_dict = OmegaConf.to_container(cfg.scenario, resolve=True)
        assert isinstance(scenario_dict, dict)  # noqa: S101
        for key, value in scenario_dict.items():
            logger.info("    %-30s = %s", key, value)

    logger.info(sep)


def _reserve_output_dir(root: str) -> Path:
    """Create a Hydra-style timestamped directory under *root* that is this
    run's alone, and return it relative to the working directory.

    The directory is created with ``mkdir``, which fails when it exists, so
    two runs started in the same second never share one (#1223, #1359).  The
    first gets ``<root>/YYYY-MM-DD/HH-MM-SS``, the next ``HH-MM-SS-1``, then
    ``HH-MM-SS-2``, and so on.

    Relative, as Hydra's own ``outputs/...`` and ``multirun/...`` defaults
    are, so it can be handed to Hydra as an override whatever characters the
    working directory's path holds.

    Reads the clock, and reserves the directory with the checked
    :func:`~autoware_carla_scenario.run_dirs.reserve_timestamped_dir`.
    """
    from datetime import datetime  # noqa: PLC0415

    now = datetime.now()  # noqa: DTZ005
    return Path(reserve_timestamped_dir(Path("."), root, now))


def _make_batch_output_dir() -> Path:
    """Create a Hydra-style timestamped output directory for batch runs.

    The directory is this batch's alone (:func:`_reserve_output_dir`): two
    batches started in the same second never share one (#1223).  The first
    gets ``HH-MM-SS``, the next ``HH-MM-SS-1``, then ``HH-MM-SS-2``, and so on.

    Returns:
        Absolute path to the created directory
        (e.g. ``outputs/2026-03-13/12-00-00/``).
    """
    return _reserve_output_dir("outputs").resolve()


def _reserve_hydra_output_dir(argv: list[str]) -> tuple[list[str], Path | None]:
    """Reserve this Hydra run's output directory before Hydra starts (#1359).

    Hydra's defaults, ``outputs/<date>/<time>`` for a run and
    ``multirun/<date>/<time>`` for a sweep, are created with ``exist_ok``:
    two runs started in the same second would share one.  This creates the
    directory exclusively, as a batch's is (:func:`_reserve_output_dir`), and
    returns *argv* with a ``hydra.run.dir`` (or, for ``--multirun``,
    ``hydra.sweep.dir``) override naming it, and the directory.

    *argv* is returned unchanged, with ``None``, when it sets that directory
    itself (the UI runner's ``hydra.run.dir=multirun/<date>/<time>/<i>``, a
    user's own) or asks Hydra for something other than a job (``--help``,
    ``--cfg``, ...).
    """
    if asks_for_no_job(argv):
        return argv, None
    multirun = _multirun_requested_in_argv(argv)
    key = "hydra.sweep.dir" if multirun else "hydra.run.dir"
    if _sets_hydra_key(argv, key):
        return argv, None
    reserved = _reserve_output_dir("multirun" if multirun else "outputs")
    return [*argv, f"{key}={reserved.as_posix()}"], reserved


def _batch_queue(
    first_cfg: DictConfig,
    map_paths: Any,
    output_dir: Path,
    extra_server_args: list[str] | None = None,
    rerun: RerunConverter | None = None,
) -> ScenarioQueue:
    """The queue a batch runs its cases in, on one CARLA, from its first config.

    *map_paths* is what :func:`resolve_map_paths` gave for the batch's map;
    *extra_server_args* are added to the server's own (a SLURM task's ports);
    *rerun* converts each case's trajectory log as the case ends
    (:func:`build_rerun_converter`); the first config's ``t4`` writes each
    case as a T4 dataset (:func:`build_t4_recorder`).
    """
    cooldown = _cooldown_seconds(first_cfg)
    max_retries = cooldown_max_retries(first_cfg)
    world_reset, reload_every = world_reset_settings(first_cfg)
    return ScenarioQueue(
        host=first_cfg.server.host,
        port=first_cfg.server.port,
        tm_port=_traffic_manager_port(first_cfg),
        xodr_path=map_paths.install_xodr,
        overwrite_xodr=map_paths.overwrite_xodr,
        opendrive_path=map_paths.opendrive_path,
        lanelet2_path=map_paths.lanelet2_path,
        map_name=map_paths.name,
        cooldown_seconds=cooldown,
        cooldown_max_retries=max_retries,
        output_dir=output_dir,
        max_tick_rate_hz=_max_tick_rate_hz(first_cfg),
        projector_type=map_paths.projector_type,
        traffic_backend=build_traffic_backend(first_cfg),
        odd=_odd_spec(first_cfg),
        autoware_launcher=build_autoware_launcher(first_cfg),
        server_extra_args=[*server_extra_args(first_cfg), *(extra_server_args or [])],
        map_source=_map_source(first_cfg, map_paths),
        rerun=rerun,
        t4=build_t4_recorder(first_cfg),
        world_reset=world_reset,
        reload_every=reload_every,
        t4scene=build_t4scene_exporter(first_cfg),
        backends=build_backend_pool(first_cfg),
        map_pack=build_map_pack(first_cfg, map_paths),
        # The tasks of a SLURM batch write into one directory, each counting
        # its stacks from 0: a case's stem tells their logs apart (#2247).
        stack_logs_by_stem=True,
    )


def cooldown_max_retries(cfg: DictConfig) -> int:
    """``server.cooldown_max_retries``, checked: a whole number (``2`` or
    ``2.0``; not ``1.9`` or a boolean), 0 or more; absent or null is 0, no
    retry (#2161).

    Raises:
        ValueError: naming the key, when it is not a whole number, or when it
            is negative: a zero-attempt loop would record every scenario as
            errored without running it.
    """
    value = _given(cfg.get("server") or {}, "cooldown_max_retries", 0)
    retries = whole_number_setting("server.cooldown_max_retries", value, minimum=0)
    if retries is None:
        raise ValueError(
            f"server.cooldown_max_retries must be a whole number, got {value!r}"
        )
    return retries


def _check_port_setting(
    cfg: DictConfig, section: str, *, null_is_default: bool = True
) -> None:
    """Refuse ``{section}.port`` of *cfg* unless it is a TCP port: an integer
    (not a bool) within 1-65535.  Unset is left to the reader, as it was, and
    so is null when *null_is_default*: ``traffic_manager.port: null`` is the
    TrafficManager's default port (:data:`DEFAULT_TM_PORT`,
    :func:`_traffic_manager_port`).  ``server.port`` has no default to fall
    back on, so null is refused there (``null_is_default=False``, #2089).
    Without this a port of 0, 70000, ``true`` or null reaches the CARLA
    client and is refused only once a server is up.

    Raises:
        ValueError: naming the key.
    """
    node = cfg.get(section)
    if not isinstance(node, (DictConfig, Mapping)) or "port" not in node:
        return
    port = node.get("port")
    if port is None:
        if null_is_default:
            return
        raise ValueError(f"{section}.port must be an integer within 1-65535 (got None)")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(
            f"{section}.port must be an integer within 1-65535 (got {port!r})"
        )


def _traffic_manager_port(cfg: DictConfig) -> int:
    """``traffic_manager.port`` of *cfg*, the port the runner's TrafficManager
    and the scenario's entities are given: :data:`DEFAULT_TM_PORT` when it is
    null, as the traffic backend reads it (:func:`build_traffic_backend`,
    ``traffic.options.port``).  Checked by :func:`_check_port_setting`."""
    port = cfg.traffic_manager.port
    return DEFAULT_TM_PORT if port is None else int(port)


def _check_host_setting(cfg: DictConfig) -> None:
    """Refuse ``server.host`` of *cfg* unless it is a non-blank string, when
    it is set (null included).  The queue launches CARLA only for a local
    host and uses the server already listening at any other
    (:class:`~autoware_carla_scenario.CarlaServerManager`); a null or blank
    host names neither, and would otherwise reach the CARLA client (#1984).

    Raises:
        ValueError: naming the key.
    """
    node = cfg.get("server")
    if not isinstance(node, (DictConfig, Mapping)) or "host" not in node:
        return
    host = node.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(
            "server.host must be the CARLA server's host name or address "
            f"(localhost for one this run launches), got {host!r}"
        )


def _check_batch_queue_settings(cfg: DictConfig) -> None:
    """Refuse what :func:`_batch_queue` would refuse of *cfg*, building nothing
    that starts: no queue, no CARLA, no Autoware.

    Raises:
        ValueError: naming the key, for a server, traffic or Autoware launcher
            setting the queue would refuse.
    """
    _check_host_setting(cfg)
    _check_port_setting(cfg, "server", null_is_default=False)
    _check_port_setting(cfg, "traffic_manager")
    _cooldown_seconds(cfg)
    cooldown_max_retries(cfg)
    _max_tick_rate_hz(cfg)
    world_reset_settings(cfg)
    # server.fog_density_scale, checked whether or not a LiDAR reads it.
    server_extra_args(cfg)
    build_traffic_backend(cfg)
    launcher = build_autoware_launcher(cfg)
    if launcher is not None:
        _check_autoware_liveness(_autoware_bridge_config(cfg), launcher)


def _check_batch_writer_settings(configs: list[DictConfig]) -> None:
    """Refuse what :func:`build_rerun_converter`, :func:`build_t4_recorder`
    and :func:`build_t4scene_exporter` refuse of a batch's *configs*, read as
    they read them -- ``rerun.lidar_max_points`` of every case asking for
    recordings and their ``rerun.draw_radius_m`` (#1928), the ``t4`` and
    ``driver`` settings and the ``t4scene`` settings of the first case
    enabling each -- without the extras those need, which
    :func:`check_run_extras` asks for.  And every case's switches
    (``rerun.enabled``, ``.strict``, ``.lidar`` and ``.cameras``,
    ``t4.enabled``, ``t4scene.enabled``): true or false, or a spelling of
    either (:func:`_switch`), so that ``rerun.enabled=ture`` is refused in
    one line before anything runs, whichever case of the batch sets it
    (#1753).

    Raises:
        ValueError: naming the key of the first value refused.
    """
    for cfg in configs:
        if _rerun_settings(cfg)[0]:
            _rerun_lidar_max_points(cfg)
            _rerun_draw_radius(cfg)
        _switch(cfg, "rerun", "lidar")
        _switch(cfg, "rerun", "cameras")
        _switch(cfg, "t4", "enabled")
        _switch(cfg, "t4scene", "enabled")
    _t4_settings(*configs)
    # Holds nothing that starts: built to be checked, and dropped.
    build_t4scene_exporter(*configs)


class RunConfigError(ValueError):
    """A config value a run refuses before anything starts
    (:func:`check_batch_configs`): the same message, naming the key, as the
    ``ValueError`` it was raised as.

    Deterministic, so a case of a sweep refused for it is refused again on a
    retry, and so is every case after it with the same value.
    """


#: The ``map`` settings that decide which map a run is on: what
#: :func:`resolve_map_paths` reads (and :func:`_map_source`, the queue's
#: record of where the map came from).  A batch shares one CARLA world, built
#: from its first case's map, so its cases must agree on all of them.
#: ``no_3d_model_lanelet_ids`` is not among them: only a sweep's constraints
#: read it (``${map.no_3d_model_lanelet_ids}``), each case's own, when the
#: sweep is expanded.
_BATCH_MAP_KEYS = (
    "name",
    "directory",
    "source",
    "lanelet2_path",
    "xodr_path",
    "projector_type",
    "overwrite_xodr",
    "pack",
    "pack_sha256",
)


def _map_setting(cfg: DictConfig, key: str) -> str | bool | None:
    """``map.<key>`` of *cfg* as :func:`resolve_map_paths` reads it: a string
    stripped, ``None`` when unset or empty, and a boolean or a value that is
    neither text nor a number refused
    (:func:`~autoware_carla_scenario.maps.config.map_setting`, #2165);
    ``overwrite_xodr`` a flag, off when unset and otherwise checked
    (``check_flag``, #2143).

    Raises:
        ValueError: naming ``map.<key>``, when the value is refused.
    """
    map_cfg = cfg.get("map") or {}
    if key == "overwrite_xodr":
        value = map_cfg.get(key)
        return value is not None and check_flag(f"map.{key}", value)
    return map_setting(map_cfg, key)


def check_batch_map(configs: list[DictConfig], labels: list[str] | None = None) -> None:
    """Refuse a batch whose cases would run on different maps.

    The batch's one queue -- and each SLURM task's -- is built on the map its
    first case resolves to (:func:`_batch_queue`), so a case setting another
    ``map.source``, ``map.lanelet2_path`` or any other of
    :data:`_BATCH_MAP_KEYS` would silently run on the first case's map.  The
    settings are compared as configured: nothing is fetched or read.

    Args:
        configs: The cases' configs.
        labels: The cases' names, for the message; ``case <n>`` without them.

    Raises:
        RunConfigError: naming the first key that differs and the cases
            setting each of its values.
        ValueError: naming ``map.<key>`` and the value, when a case's is
            refused (:func:`_map_setting`).
    """
    names = labels if labels is not None else []
    for key in _BATCH_MAP_KEYS:
        cases: dict[str | bool | None, list[str]] = {}
        for i, cfg in enumerate(configs):
            label = names[i] if i < len(names) else f"case {i + 1}"
            cases.setdefault(_map_setting(cfg, key), []).append(label)
        if len(cases) > 1:
            found = "; ".join(
                f"{value!r} in {', '.join(labels_)}" for value, labels_ in cases.items()
            )
            raise RunConfigError(
                f"batch scenarios must share the same map, but map.{key} "
                f"differs: {found}.  Set it on the command line rather than "
                "in a scenario's config"
            )


def check_scenario_maps(
    configs: list[DictConfig],
    labels: list[str] | None = None,
    *,
    documents: dict[str, Any] | None = None,
) -> None:
    """Refuse a case whose declarative scenario names a map of its own that
    the composed config is not on (#1424).

    An exported scenario package whose document names its map by
    ``map.source`` or ``map.directory`` ships that map as a group of its own,
    ``map=<id>_map``
    (:func:`~autoware_carla_scenario.authoring.hydra_config.own_map_group`).
    The ``map`` group is composed after the ``scenario`` one, so a run that
    leaves the override out composes the default group, whose keys win over
    the ones the scenario's config writes: the run would load another map's
    Lanelet2 and OpenDRIVE, and every lanelet id the document names would be
    one of that map's.  The composed ``map`` is therefore compared with the
    document's, the document the package registered with its builder
    (:func:`~autoware_carla_scenario.registry.register_scenario_document`),
    as :func:`~autoware_carla_scenario.authoring.hydra_config.own_map_mismatch`
    compares them.  A scenario with no registered document, or one on a
    built-in group, is not checked; neither is one whose document cannot be
    loaded, which its builder refuses with its own message.

    Args:
        configs: The cases' configs.
        labels: The cases' names, for the message; none for a single run.
        documents: The documents already loaded, by scenario name, filled
            here: a batch shares one across its checks, so each scenario's
            document is loaded once.

    Raises:
        RunConfigError: naming the scenario, what differs and the
            ``map=<id>_map`` to run it with.
    """
    from autoware_carla_scenario.authoring.hydra_config import (  # noqa: PLC0415
        scenario_map_refusal,
    )

    names = labels if labels is not None else []
    # Loaded once a scenario, however many of the batch's cases run it.
    loaded: dict[str, Any] = {} if documents is None else documents
    for i, cfg in enumerate(configs):
        refusal = scenario_map_refusal(cfg, loaded)
        if refusal is None:
            continue
        prefix = f"{names[i]}: " if i < len(names) else ""
        raise RunConfigError(f"{prefix}{refusal}")


def check_batch_configs(
    configs: list[DictConfig],
    labels: list[str] | None = None,
    *,
    documents: dict[str, Any] | None = None,
) -> None:
    """Refuse the configs of a batch's cases for the values a run reads of its
    config before it starts, building nothing that starts (no queue, no CARLA,
    no Autoware, no Rerun or T4 writer) and needing no map.

    First that the cases share a map (:func:`check_batch_map`, which names
    them by *labels*), and that a declarative scenario's case is on the map
    its document names (:func:`check_scenario_maps`).

    The ``rerun``, ``t4``, ``driver`` and ``t4scene`` settings, as the batch
    reads them (:func:`_check_batch_writer_settings`); each case's timeout,
    ``typecheck`` mode (:func:`~autoware_carla_scenario.typecheck.mode.typecheck_mode`),
    ``ego.entity`` (:func:`ego_entity_name`), seeds (``seed``, ``sweep.odd_sample.seed`` and ``.count``)
    and the shape of ``sweep`` (a mapping or null),
    ``background_traffic`` (:func:`background_traffic_settings`),
    ``environment`` (:func:`environment_settings`) and the form of its
    ``rules`` (:func:`_check_weather_rules`), ``driver`` settings of an
    ``ego.entity=carla_driver`` case (:func:`_check_driver_settings`), ``ros2``
    settings of an ``ego.entity=ros2`` case (:func:`_check_ros2_settings`), the
    ``backends`` settings of the backends this package ships
    (:func:`_check_backend_settings`), the ``scenario`` settings its config
    class refuses (:func:`_check_scenario_config`), the ego's and entities'
    numbers (``ego.initial_speed_kmh``, the spawn and goal poses,
    ``entity.spawn_retry_*_step`` and ``entity.spawn_retry_max_count``:
    :func:`_check_ego_numbers`), ``ego.vehicle_type``
    (:func:`_ego_vehicle_type`), lanelet ids (``ego.spawn_lanelet_id``,
    ``ego.goal_lanelet_id``, ``ego.waypoint_lanelet_ids``:
    :func:`_check_ego_lanelet_ids`), ``ego.native_autoware_interface``
    (:func:`_native_autoware_interface`), ``odd_case``,
    ``canvas.config`` (:func:`canvas_config`) and ``metrics.heavy``
    (:func:`metrics_heavy`); and the server, traffic and Autoware launcher settings its
    queue is built from (:func:`_batch_queue`, :func:`run_scenario`), once for
    each distinct set of them.  A batch is refused for these before any
    scenario is built or the map is resolved (#1430): a local one would refuse
    them only once its queue was built, a SLURM one only on its nodes, once
    submitted; and a lanelet sweep before any case is forked (each case would
    refuse them in turn).

    Args:
        configs: The cases' configs.
        labels: The cases' names: a value a case is refused for is headed by
            its name (``<label>: ...``), as the map checks head theirs; none
            for a single run.
        documents: The scenario documents already loaded, by name, shared
            with :func:`check_scenario_maps` (and filled by it).

    Raises:
        RunConfigError: naming the key of the first value refused.
    """
    check_batch_map(configs, labels)
    check_scenario_maps(configs, labels, documents=documents)
    try:
        # The batch's: read off the first case enabling each writer.
        _check_batch_writer_settings(configs)
    except RunConfigError:
        raise
    except ValueError as exc:
        raise RunConfigError(str(exc)) from exc
    names = labels if labels is not None else []
    queue_settings: set[str] = set()
    for i, cfg in enumerate(configs):
        try:
            run_timeout_seconds(cfg)
            # When the scenario is statically checked, and which ego drives
            # it: each refused only once the scenario was built (#2089).
            typecheck_mode(cfg)
            ego_entity_name(cfg)
            _check_seeds(cfg)
            background_traffic_settings(cfg)
            environment_settings(cfg)
            _check_weather_rules(cfg)
            _check_driver_settings(cfg)
            _check_ros2_settings(cfg)
            _check_backend_settings(cfg)
            _check_scenario_config(cfg)
            _odd_case(cfg)
            build_ground_projection(cfg)
            _check_ego_numbers(cfg)
            _check_ego_vehicle_type(cfg)
            _check_ego_lanelet_ids(cfg)
            _native_autoware_interface(cfg)
            # Each case's alone: that the cases name one is run_batch's.
            canvas_config(cfg)
            # Every case's, though apply_metrics reads only the first set.
            metrics_heavy(cfg)
            # Once a distinct setting: a sweep's cases mostly share them.  The
            # ego's entity too: build_autoware_launcher reads the launcher only
            # for an ``ego.entity=autoware`` case.
            settings: dict[str, Any] = {
                name: _to_dict(cfg[name]) if cfg.get(name) is not None else None
                for name in ("server", "traffic", "traffic_manager", "autoware")
            }
            settings["ego.entity"] = str(_ego_entity_setting(cfg))
            key = json.dumps(
                settings,
                sort_keys=True,
                default=str,
            )
            if key not in queue_settings:
                queue_settings.add(key)
                _check_batch_queue_settings(cfg)
        except ValueError as exc:
            # Named by its case, as the map checks name it: every case of a
            # batch is checked, and the one refused need not be the first.
            prefix = f"{names[i]}: " if i < len(names) else ""
            raise RunConfigError(f"{prefix}{exc}") from exc


def check_run_config(cfg: DictConfig) -> None:
    """:func:`check_batch_configs` of one run's *cfg*.

    Raises:
        RunConfigError: naming the key of the first value refused.
    """
    check_batch_configs([cfg])


def check_run_extras(cfg: DictConfig) -> None:
    """Refuse *cfg* when it asks for Rerun recordings or T4 datasets without
    the extra that writes them, as a run does before anything starts.  The
    converter and the recorder are built to be checked and dropped: the
    converter's thread is ended.

    Raises:
        RerunUnavailableError: ``rerun.enabled`` without the ``rerun`` extra.
        T4UnavailableError: ``t4.enabled`` without the ``t4`` extra.
    """
    rerun = build_rerun_converter(cfg)
    if rerun is not None:
        rerun.close()
    build_t4_recorder(cfg)


def _batch_launcher(configs: list[DictConfig]) -> str:
    """``launcher.type`` the batch's configs share: ``local`` or ``slurm``.

    Exits with an error when they differ, name an unknown launcher, or select
    SLURM where there is no ``sbatch``.
    """
    from ..slurm.settings import launcher_type, require_sbatch  # noqa: PLC0415

    try:
        kinds = {launcher_type(cfg) for cfg in configs}
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if len(kinds) > 1:
        print(  # noqa: T201
            "Error: batch scenarios must share launcher.type; set it on the "
            "command line rather than in a scenario's config",
            file=sys.stderr,
        )
        sys.exit(1)
    (kind,) = kinds
    if kind == "slurm":
        try:
            require_sbatch()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    return kind


def run_batch(
    scenario_names: list[str],
    overrides: list[str],
    *,
    expand: bool = False,
    resume_from: int = 0,
    build_scenario_fn: BuildScenarioFn | None = None,
    wait: bool | None = None,
    resume_into: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Compose configs, build scenarios, and run them in a single queue.

    With *expand* (``--multirun``) each scenario's sweep is expanded and every
    case is run; without it a scenario runs once, as configured. *resume_from*
    is the 1-based case to start at, as for the lanelet sweeper.

    With *resume_into* (``--resume-into DIR``) the remaining cases run in
    that directory, the killed batch's, rather than a new one, so that the
    batch is gated as a whole (#1957): *resume_from* defaults to its first
    case without a result, its ``batch_results.json`` is rewritten from the
    entries of the cases before that and the cases run now, and the outputs
    the cases run again left there (under the stems the summary records for
    them) are removed before they run
    (:mod:`~autoware_carla_scenario.batch_resume`).  A directory that is not
    such a batch's, or lists other cases, is refused (exit 2) before
    anything is built or removed.  A local batch only.

    Every case of a local batch is named by a stem given before it runs, a
    local batch's sequential stems (``slurm.stems.sequential_stems``), which
    its summary records (``stem``).

    With ``launcher=slurm`` the cases run as SLURM array tasks instead
    (docs/slurm.md); *wait* overrides ``launcher.wait`` (``--no-wait``).

    With *output_dir* (``hydra.run.dir=``, or ``hydra.sweep.dir=`` with
    ``--multirun``) the batch writes there, created once its cases are built,
    rather than in a new ``outputs/<date>/<time>``: a directory that is not
    empty, or one given with *resume_into*, is refused (exit 2) before
    anything is composed (:func:`_batch_output_dir_refusal`, #2294).

    A command line that does not compose (a misspelled key) is reported as
    one ``Error:`` line naming the scenario, and exits 2
    (:func:`~autoware_carla_scenario.examples.compose_errors.compose_or_report`).

    Raises:
        RunConfigError: a config value a case would refuse
            (:func:`check_batch_configs`), before any scenario is built or the
            map is resolved.  The ``scenario`` CLI prints it as an ``Error:``
            line and exits 1.
    """
    from ..slurm.settings import (  # noqa: PLC0415
        launcher_requested_in_argv,
        require_sbatch,
    )

    if output_dir is not None:
        refusal = _batch_output_dir_refusal(output_dir, resume_into)
        if refusal is not None:
            print(f"Error: {refusal}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    if launcher_requested_in_argv(overrides) == "slurm":
        # Before the sweep is expanded, which can take a while.
        try:
            require_sbatch()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    # Each scenario's document, loaded once for every check of the batch.
    documents: dict[str, Any] = {}
    if expand:
        cases = _expand_batch(scenario_names, overrides, documents=documents)
    else:
        cases = [(name, list(overrides)) for name in scenario_names]
    # Labelled before resuming, so a case keeps the name it had in the run
    # being resumed.
    labels = _case_labels(cases)
    # What each case runs with beyond the command line: its sweep case.
    case_overrides = [case[: len(case) - len(overrides)] for _, case in cases]
    # The batch resumed into its own directory: checked before anything is
    # built, and its first case without a result is where it resumes unless
    # --resume-from says otherwise (#1957).
    resumed: ResumedBatch | None = None
    if resume_into is not None:
        try:
            resumed = read_resumed_batch(
                resume_into.resolve(), labels, case_overrides if expand else None
            )
            if resume_from == 0:
                first = resumed.first_pending()
                if first is None:
                    raise ResumeIntoError(
                        f"{resumed.directory}: every case of the batch has a "
                        "result; pass --resume-from N to run its cases from N "
                        "again"
                    )
                resume_from = first + 1
            resumed.check_kept(min(max(resume_from - 1, 0), len(cases)), labels)
        except ResumeIntoError as exc:
            print(f"Error: --resume-into {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    # The whole batch's cases, for the summary of a batch resumed into its
    # own directory, which lists the cases it keeps too.
    all_labels, all_case_overrides = list(labels), list(case_overrides)
    skip = 0
    if resume_from > 1:
        skip = min(resume_from - 1, len(cases))
        logger.info(
            "Resuming from case %d -- skipping %d/%d case(s).",
            resume_from,
            skip,
            len(cases),
        )
        cases, labels, case_overrides = (
            cases[skip:],
            labels[skip:],
            case_overrides[skip:],
        )
    if not cases:
        print("Error: no scenario case left to run", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    configs = [_compose_case(name, case) for name, case in cases]

    # Every value a case reads of its config before it starts -- that the
    # cases share a map, the queue is built on the first case's (#1241); that
    # each is on the map its document names; the ground-projection ray, a ray
    # that cannot hit spawns nothing (#1416); the queue's server, traffic and
    # Autoware launcher settings, the run timeout, the writers' settings, ...
    # -- for every case, before anything is built or the map is resolved: a
    # local batch would refuse it only once its queue was built, and a SLURM
    # task only on its node, once the batch was submitted (#977, #1430).
    # Raised, as RunConfigError: the CLI reports it as an ``Error:`` line,
    # headed by the case's name, and HYDRA_FULL_ERROR=1 raises it (_dispatch).
    check_batch_configs(configs, labels, documents=documents)

    if not expand:
        # After the configs are checked: a sweep that is not a mapping is
        # refused there, naming its case (#2279).
        swept = [name for name, cfg in zip(labels, configs) if _has_sweep(cfg)]
        if swept:
            logger.warning(
                "%s ha%s a sweep (sweep.constraints, sweep.route or "
                "sweep.odd_sample) that a batch runs once, as configured; add "
                "--multirun to run every case of it",
                ", ".join(swept),
                "s" if len(swept) == 1 else "ve",
            )

    # One launcher for the whole batch: the queue prepares it (and puts its
    # logs in the batch's output directory) and every ego starts its stacks
    # with it, so a scenario configuring an Autoware of its own is refused.
    launcher_cfgs = {
        json.dumps(
            _to_dict(cfg.autoware.launcher)
            if cfg.get("autoware") is not None
            and cfg.autoware.get("launcher") is not None
            else None,
            sort_keys=True,
            default=str,
        )
        for cfg in configs
    }
    if len(launcher_cfgs) > 1:
        print(  # noqa: T201
            "Error: batch scenarios must share autoware.launcher; set it on the "
            "command line rather than in a scenario's config",
            file=sys.stderr,
        )
        sys.exit(1)

    launcher = _batch_launcher(configs)
    if resumed is not None and launcher == "slurm":
        print(  # noqa: T201
            "Error: --resume-into is for a local batch; a SLURM batch's cases "
            "are run again with scenario-slurm",
            file=sys.stderr,
        )
        sys.exit(2)
    if wait is not None and launcher != "slurm":
        print("Error: --no-wait is for launcher=slurm only", file=sys.stderr)  # noqa: T201
        sys.exit(1)

    # Before anything is built: a batch asking for Rerun recordings without
    # the extra installed is refused at once (RerunUnavailableError).  Under
    # SLURM each task converts its own cases (slurm/worker.py), with what
    # this decides for the batch (batch_settings_cases), so this only checks.
    rerun = build_rerun_converter(*configs)
    # Likewise a batch asking for T4 datasets without the `t4` extra, and
    # T4Scenes at a step that does not fit the tick.
    build_t4_recorder(*configs)
    build_t4scene_exporter(*configs)
    if rerun is not None and launcher == "slurm":
        rerun.close()
        rerun = None
    # Under SLURM the file is the nodes' (slurm/worker.py): only one is named.
    if launcher == "slurm":
        canvas_config(*configs)
    else:
        apply_canvas(*configs)
    heavy_metrics = apply_metrics(*configs)

    # Log detailed execution plan before building anything.
    _log_batch_plan(
        [name for name, _ in cases],
        configs,
        overrides,
        labels=labels,
        case_overrides=case_overrides,
    )

    # Build (and statically check) every scenario before anything reads the
    # config values or fetches the map, so a bad value is refused first.
    scenarios: list[BaseScenario] = []
    for i, (name, cfg) in enumerate(zip(labels, configs), 1):
        logger.info("Building scenario [%d/%d]: %s", i, len(labels), name)
        _ego, scenario = build_scenario(cfg, build_scenario_fn=build_scenario_fn)
        scenarios.append(scenario)

    first_cfg = configs[0]
    # Any case of the batch: a scenario's own YAML may be what sets it.
    warn_if_video_set(*configs)
    # The ODD every run is measured against, checked once when it is Python.
    check_odd(_odd_spec(first_cfg), typecheck_mode(first_cfg))

    map_paths = resolve_map_paths(first_cfg.map)

    # Each case's file stem, given before it runs and recorded in the summary
    # (#1957): a batch resumed into its own directory keeps those of the
    # batch it resumes, so that it replaces the outputs of the cases it runs
    # again and of no other.
    scenario_classes = [type(s).__name__ for s in scenarios]
    kept: list[KeptResult] = []
    if resumed is not None:
        try:
            resumed.check_stems(skip, scenario_classes)
        except ResumeIntoError as exc:
            print(f"Error: --resume-into {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
        all_stems = resumed.stems
        kept = [KeptResult(entry) for entry in resumed.entries[:skip]]
        # The batch's own directory, not a new one.
        output_dir = resumed.directory
    else:
        all_stems = sequential_stems(scenario_classes)
        all_labels, all_case_overrides = list(labels), list(case_overrides)
        if output_dir is not None:
            # The directory the command line names (hydra.run.dir=), checked
            # empty before anything was composed (#2294).
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir.resolve()
        else:
            # Batch mode bypasses @hydra.main, so we create the output
            # directory ourselves following Hydra's timestamped convention.
            output_dir = _make_batch_output_dir()
    run_stems = all_stems[len(kept) :]
    kept_errored = [i for i, result in enumerate(kept) if result.errored]

    if launcher == "slurm":
        # The cases run as SLURM array tasks, each on a CARLA of its own, and
        # write their outputs here under the names this batch would give them
        # (docs/slurm.md).
        from ..slurm.submit import submit_batch  # noqa: PLC0415

        del scenarios
        sys.exit(
            submit_batch(
                scenario_names=scenario_names,
                cases=cases,
                labels=labels,
                case_overrides=case_overrides,
                configs=configs,
                scenario_classes=scenario_classes,
                overrides=overrides,
                expand=expand,
                output_dir=output_dir,
                build_scenario_fn=build_scenario_fn,
                wait=wait,
                batch_settings=batch_settings_cases(
                    configs, heavy_metrics=heavy_metrics
                ),
            )
        )

    # A case that raises becomes a failed result and the batch goes on: its
    # crash report goes to the batch's directory, and runs that case again
    # (_CaseCrashReports).
    crash_report.set_output_dir(output_dir)
    case_reports = _CaseCrashReports(
        output_dir,
        [[f"scenario={name}", *case] for name, case in cases],
        configs,
    )
    queue = _batch_queue(first_cfg, map_paths, output_dir, rerun=rerun)

    for cfg, scenario, stem in zip(configs, scenarios, run_stems):
        # The runner's own fail-safe, each scenario's own: without it the
        # queue default (60 s) caps every run, which is shorter than an
        # Autoware stack needs to localize, route and engage.
        queue.add(
            scenario,
            timeout_seconds=run_timeout_seconds(cfg),
            case=_odd_case(cfg),
            file_stem=stem,
        )
    # From here the queue alone holds the scenarios, and lets each go once it
    # has run: a batch reports results, and a finished scenario kept to the
    # end keeps everything its run built (its entities, conditions, the ego's
    # driver), case after case.
    del scenarios, scenario

    logger.info("All %d scenario(s) built. Starting execution...", len(labels))

    # The cases run so far, for batch_results.json: rewritten as each case
    # ends, so that the viewer lists a batch still running, or cut short
    # (Ctrl-C, a SIGTERM), as the batch it is, with the cases it ran (#1566),
    # and the cases it has not run yet as pending, so that a batch that dies
    # mid-run (a segfault, the OOM killer, SIGKILL) lists them too, and
    # scenario-gate does not judge the cases that survived alone (#1831).
    # It is first written before the first case runs, every case pending, so
    # that a batch that dies in its first case leaves it too (#1833).
    # The summary below writes it once more, whole, when the batch ends.
    # A batch resumed into its own directory writes the cases it keeps first,
    # as they were, and the cases it runs after them (#1957).
    done: list[Any] = list(kept)

    def errored_cases(errored: list[int]) -> list[int]:
        return kept_errored + [len(kept) + i for i in errored]

    def write_summary() -> None:
        write_batch_result_json(
            all_labels,
            done,
            output_dir,
            all_case_overrides if expand else None,
            errored=errored_cases(queue.errored_indices()),
            pending=True,
            stems=all_stems,
        )

    def write_done() -> None:
        try:
            write_summary()
        except Exception:
            # The batch goes on: its summary is written again at the end.
            logger.warning("Could not write the batch's results so far", exc_info=True)

    def case_done(index: int, result: ScenarioResult) -> None:
        done.append(result)
        write_done()
        case_reports.enter(index + 1)

    finished = False
    try:
        with queue:
            # Every case pending, once the queue has started: its pre-run
            # checks (a backend the host has no room for, an Autoware
            # workspace that cannot run, CARLA not coming up) refuse a batch
            # before any case runs, and such a batch leaves no summary.  One
            # that dies from here on -- in its first case too -- lists every
            # case it did not finish, and scenario-gate refuses it rather than
            # skip it (#1833).
            if resumed is None:
                write_done()
            else:
                # Only once the summary lists the cases run again as pending
                # do the outputs their earlier run left under their stems
                # (its result, gate record, coverage file, ...) go, so that
                # each case is judged on its new run alone, not counted
                # twice.  A summary that cannot be rewritten (a full disk)
                # still lists them as they were: nothing is removed, and the
                # batch stops (exit 2), rather than leave a summary claiming
                # results whose outputs are gone.  From here on a summary
                # that cannot be written leaves the last one written, whose
                # cases not finished since are pending: never a result
                # without its outputs.
                try:
                    write_summary()
                except Exception as exc:  # noqa: BLE001 -- reported, exit 2
                    print(  # noqa: T201
                        f"Error: --resume-into {output_dir}: the batch summary "
                        f"cannot be rewritten ({exc}); nothing was removed or run",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                for stem in run_stems:
                    remove_run_outputs(output_dir, stem)
            # A case that cannot run (a spawn point without ground, say) is
            # recorded as errored, and the rest of the batch still runs.
            case_reports.enter(0)
            results = queue.run_all(
                release=True, continue_on_error=True, on_result=case_done
            )
            errored = queue.errored_indices()
        finished = True
    finally:
        case_reports.close()
        if not finished and len(done) > len(kept):
            # Cut short: the cases it ran, as the last case's end wrote them.
            # (One cut short before any case ended keeps the summary written
            # before the first, every case pending.)
            write_done()
        # Each log was converted as its case ended; wait for the last ones.
        conversions = rerun.close() if rerun is not None else []

    all_passed = _print_summary(
        all_labels,
        [*kept, *results],
        output_dir=output_dir,
        overrides=all_case_overrides if expand else None,
        errored=errored_cases(errored),
        rerun_errors={
            # The kept cases' recordings that failed, as their entries say.
            **{i: r.rerun_error for i, r in enumerate(kept) if r.rerun_error},
            **{
                len(kept) + i: why
                for i, why in rerun_errors(results, conversions).items()
            },
        },
        stems=all_stems,
    )
    sys.exit(0 if all_passed else 1)


class _CaseCrashReports:
    """The crash report context of each case of a local batch, as a SLURM
    task's worker gives each of its cases (:func:`crash_report.run_context`):
    a report of a case is in the batch's directory, and its ``repro.sh`` runs
    that case again, with its own overrides and config.

    The queue runs every case in one ``run_all()``: case *i*'s context is
    entered before it starts (:meth:`enter`: at the start, and from the
    queue's ``on_result`` of the case before), and left when the next is
    entered or the batch ends (:meth:`close`).
    """

    def __init__(
        self, output_dir: Path, overrides: list[list[str]], configs: list[Any]
    ) -> None:
        self._output_dir = output_dir
        self._overrides = overrides
        self._configs = configs
        self._stack = ExitStack()

    def enter(self, index: int) -> None:
        """Leave the context entered last, and enter case *index*'s (none
        past the last case)."""
        self.close()
        if not crash_report.installed() or index >= len(self._overrides):
            return
        self._stack.enter_context(
            crash_report.run_context(
                self._output_dir,
                overrides=self._overrides[index],
                config_yaml=config_yaml(self._configs[index]),
            )
        )

    def close(self) -> None:
        """Leave the context entered last, if any."""
        self._stack.close()


def config_seed(cfg: DictConfig) -> int | None:
    """The run's seed, the config's top-level ``seed``; ``None`` when it has none.

    One number for every random draw of the run (:mod:`autoware_carla_scenario.seed`):
    the scenario's :attr:`~autoware_carla_scenario.BaseScenario.random_seed`, and
    through it the traffic backend's and CARLA's pedestrians'; the scenario's
    actions, ``background_traffic`` and the driver policy that name no seed of
    their own; and the cases a sweep draws from the ODD.  A config that predates
    the key (an exported scenario package's) leaves the scenario its own seed.

    Raises:
        ValueError: When it is not a whole number from 0 to
            :data:`~autoware_carla_scenario.seed.MAX_SEED` (``2**31 - 1``)
            (:func:`~autoware_carla_scenario.odd.sampler.seed_setting`):
            ``1.9`` and ``true`` are not seed 1.
    """
    return seed_setting("seed", cfg.get("seed"))


def _check_seeds(cfg: DictConfig) -> None:
    """Refuse the run's ``seed`` and its sweep's ``odd_sample`` ``seed`` and
    ``count`` unless each is a whole number (a seed from 0 to ``2**31 - 1``,
    a count at least 1), and its ``odd_sample`` ``tire_friction`` and
    ``visibility`` unless each is true or false, naming the key, as expanding
    the sweep and building the scenario would refuse them.

    And the sweep itself, unless it is null or a mapping (:func:`_has_sweep`).

    Raises:
        ValueError: naming the key of the value refused.
    """
    config_seed(cfg)
    _has_sweep(cfg)  # before its odd_sample is looked up in it
    odd_sample = OmegaConf.select(cfg, "sweep.odd_sample")
    if isinstance(odd_sample, DictConfig):
        check_sample_settings(odd_sample)


def _check_driver_settings(cfg: DictConfig) -> None:
    """Refuse the ``driver`` settings of an ``ego.entity=carla_driver`` case
    as :func:`build_ego_entity` would (``driver.random_seed``, a LiDAR's
    ``noise_seed``, ``driver.control.kp``, ...), building only the configs:
    nothing dials a policy.

    And ``t4.enabled`` with ``driver.lidars`` but not
    ``driver.send_renderer_data``: the dataset of a driver entity takes its
    sweeps from the LiDARs the entity attaches for the policy's
    ``renderer_data``, which it attaches only with it, so the dataset would
    have no point cloud (no sample at all with no camera) -- refused as
    ``driver.t4_scene`` is without it (#2208).

    Raises:
        ValueError: When :class:`~autoware_carla_scenario.driver.DriverClientConfig`
            or the trajectory follower's ``driver.control`` refuses them, or
            ``t4.enabled`` asks for the driver's LiDARs without
            ``driver.send_renderer_data``.
    """
    if _ego_entity_setting(cfg) != "carla_driver":
        return
    driver_cfg = cfg.get("driver")
    if driver_cfg is None:
        return  # Refused, with its remedy, by build_ego_entity.
    from autoware_carla_scenario.driver import DriverClientConfig  # noqa: PLC0415

    driver_dict = _with_fog_density_scale(_to_dict(driver_cfg), cfg)
    _control_config(driver_dict, "driver")
    driver = DriverClientConfig.from_mapping(driver_dict)
    if (
        _switch(cfg, "t4", "enabled")
        and driver.lidars
        and not driver.send_renderer_data
    ):
        raise ValueError(
            "t4.enabled takes the LiDAR sweeps of an ego.entity=carla_driver "
            "from its renderer_data: driver.lidars needs "
            "driver.send_renderer_data"
        )


def _control_config(section: Mapping[str, Any], name: str) -> "ControlConfig":
    """The trajectory follower's gains of the ``control`` entry of
    *section*, the run's *name* (``driver`` or ``ros2``) settings:
    ``control: null`` (or none) means the default gains; anything else but a
    mapping is refused (``driver.control must be a mapping, got 'abc'``).

    :meth:`~autoware_carla_scenario.driver.ControlConfig.from_mapping` names
    the key it refuses, not the section it is read from, as
    :class:`~autoware_carla_scenario.AutowareBridgeConfig` does: its error is
    headed with ``<name>.control.`` here (``driver.control.kp must be a
    number, got '0.5'``).

    Raises:
        ValueError: naming ``<name>.control.<key>``.
    """
    from autoware_carla_scenario.driver import ControlConfig  # noqa: PLC0415
    from autoware_carla_scenario.value_text import shown  # noqa: PLC0415

    control = section.get("control")
    if control is None:
        control = {}
    if not isinstance(control, Mapping):
        # from_mapping's errors start with a key; this one is the section's.
        raise ValueError(f"{name}.control must be a mapping, got {shown(control)}")
    try:
        return ControlConfig.from_mapping(control)
    except ValueError as exc:
        raise ValueError(f"{name}.control.{exc}") from None


def _check_ros2_settings(cfg: DictConfig) -> None:
    """Refuse the ``ros2`` settings of an ``ego.entity=ros2`` case as
    :func:`build_ego_entity` would (``ros2.domain_id`` above 232, a quoted
    ``priority``, ...), building only the configs: nothing joins a DDS domain,
    which the entity first does once the ego has spawned.

    Raises:
        ValueError: When the trajectory follower's ``ros2.control`` refuses
            them, naming ``ros2.control.<key>``, or
            :class:`~autoware_carla_scenario.ros2.Ros2EgoConfig` does, headed
            ``ros2:``.
    """
    if _ego_entity_setting(cfg) != "ros2":
        return
    ros2_cfg = cfg.get("ros2")
    if ros2_cfg is None:
        return  # Refused, with its remedy, by build_ego_entity.
    from autoware_carla_scenario.ros2 import Ros2EgoConfig  # noqa: PLC0415

    ros2_dict = _to_dict(ros2_cfg)
    _control_config(ros2_dict, "ros2")
    try:
        Ros2EgoConfig.from_mapping(ros2_dict)
    except ValueError as exc:
        raise ValueError(f"ros2: {exc}") from None


def _check_backend_settings(cfg: DictConfig) -> None:
    """Refuse the ``backends`` settings of a backend this package ships
    (``backends.ardy``) as the queue's preparer would once a scenario needed
    it (``startup_timeout_s: -1``, ``pull: sometimes``, ``docker_args`` as a
    string, ...): its ``from_settings`` only reads them -- it pulls no image
    and starts no container.  A backend of another package is built, and its
    settings checked, by the preparer; only a block of its that is neither
    null nor a mapping (``backends.<name>: false``) is refused here, as the
    preparer's ``BackendPool`` would refuse it once the queue was built
    (#2182).

    Raises:
        ValueError: naming the key (``backends.<name>.<key>``).
    """
    backends_cfg = cfg.get("backends")
    if backends_cfg is None:
        return
    if not isinstance(backends_cfg, DictConfig):
        raise ValueError(f"backends must be a mapping, got {backends_cfg!r}")
    for key, block in backends_cfg.items():
        if block is not None and not isinstance(block, DictConfig):
            raise ValueError(f"backends.{key!s} must be a mapping, got {block!r}")
    from autoware_carla_scenario.pedestrian.ardy import ArdyBackend  # noqa: PLC0415

    for backend in (ArdyBackend,):
        name = backend.backend_name
        settings = backends_cfg.get(name)
        if settings is None:
            continue
        if not isinstance(settings, DictConfig):
            raise ValueError(f"backends.{name} must be a mapping, got {settings!r}")
        backend.from_settings(_to_dict(settings))


def _check_scenario_config(cfg: DictConfig) -> None:
    """Build the config of a scenario registered with
    :func:`~autoware_carla_scenario.register_scenario`, as its builder does,
    so its ``__post_init__`` refuses a value (``scenario.nade_is_probability``
    above 1, #1458) before anything starts.  Nothing is built of a scenario
    with a custom builder or a declarative one.

    A key the class does not take (one not among its
    :func:`dataclasses.fields`) is left out: the build refuses it, with the
    scenario's own context.  Only those are: a value of the wrong type
    (``scenario.timeout_seconds=null``, a quoted ``'10'``, #1818) is
    refused, as any other value the class refuses.

    Raises:
        ValueError: When the scenario's config class refuses a value; one it
            refuses with a ``TypeError`` (which names no key, as a class of
            a scenario package's own may) is headed by ``scenario`` and the
            class's name.
    """
    scenario = cfg.get("scenario")
    if not isinstance(scenario, DictConfig) or scenario.get("name") is None:
        return
    classes = get_scenario_classes(str(scenario.name))
    if classes is None:
        return
    _, config_cls = classes
    settings = _to_dict(scenario)
    if not dataclasses.is_dataclass(config_cls):
        # No fields to tell an unknown key by: a TypeError may be one, which
        # the build refuses with its context.
        try:
            config_cls(**settings)
        except TypeError:
            pass
        return
    known = {f.name for f in dataclasses.fields(config_cls) if f.init}
    try:
        config_cls(**{key: value for key, value in settings.items() if key in known})
    except TypeError as exc:
        raise ValueError(
            f"scenario: {config_cls.__name__} refused the settings: {exc}"
        ) from exc


def build_scenario(
    cfg: DictConfig,
    *,
    build_scenario_fn: BuildScenarioFn | None = None,
) -> tuple[EgoConfig, BaseScenario]:
    """Instantiate the correct scenario class based on ``cfg.scenario.name``.

    Parameters
    ----------
    cfg:
        Resolved Hydra config containing ``scenario.name`` and related keys.
    build_scenario_fn:
        Optional callable that completely replaces the default registry
        lookup.  When provided, it is called as ``build_scenario_fn(cfg)``
        and its return value is forwarded to the caller.
    """
    # Before anything is built: an action built with no seed of its own takes
    # the run's when it is constructed.
    seed = config_seed(cfg)
    set_run_seed(DEFAULT_SEED if seed is None else seed)
    if build_scenario_fn is not None:
        ego, scenario = build_scenario_fn(cfg)
        if seed is not None:
            scenario.random_seed = seed
        _set_scenario_hash(cfg, scenario)
        _apply_ego_config(cfg, scenario)
        add_background_traffic(cfg, scenario)
        add_environment(cfg, scenario)
        add_effect_measures(cfg, scenario)
        return ego, scenario

    # Validate the name before doing any expensive work.
    scenario_name: str = cfg.scenario.name
    builder = get_scenario_builder(scenario_name)
    if builder is None:
        registered = sorted(get_scenario_registry())
        msg = (
            f"Unknown scenario name: {scenario_name!r}. "
            f"Registered scenarios: {registered}"
        )
        raise ValueError(msg)

    scenario_dict = _to_dict(cfg.scenario)
    # Compile the scenario before building anything of it: one that does not
    # type-check is refused here, before the runner touches CARLA.
    check_registered_scenario(scenario_name, scenario_dict, typecheck_mode(cfg))

    ego, spawn_pose, ground_projection = build_ego_and_spawn(cfg)
    scenario = builder(ego, scenario_dict, spawn_pose, ground_projection)
    if seed is not None:
        scenario.random_seed = seed
    _set_scenario_hash(cfg, scenario)
    # Built once: an Autoware ego holds a bridge server, and two of those cannot
    # hold the same address.
    ego_entity = build_ego_entity(cfg)
    if ego_entity is not None:
        scenario.ego_entity = ego_entity
    add_background_traffic(cfg, scenario)
    add_environment(cfg, scenario)
    add_effect_measures(cfg, scenario)
    return ego, scenario


def _set_scenario_hash(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Give *scenario* the digest of the concrete scenario *cfg* runs, its
    seed aside (:func:`~autoware_carla_scenario.scenario_hash.scenario_hash`):
    the runner writes it into the run's trajectory log, where
    ``scenario-wosac`` tells the rollouts of one scenario by it."""
    # Only the sections it reads are resolved: the rest may interpolate what
    # only a Hydra job has (its output directory).
    config: dict[str, Any] = {}
    for key in SCENARIO_SECTIONS:
        if key in cfg:
            node = cfg[key]
            config[key] = (
                OmegaConf.to_container(node, resolve=True)
                if OmegaConf.is_config(node)
                else node
            )
    # And the few settings of the other sections that say what is simulated
    # (server.fog_density_scale), each resolved on its own.
    for section, key in SCENARIO_SETTINGS:
        value = OmegaConf.select(cfg, f"{section}.{key}", default=None)
        if value is not None:
            config.setdefault(section, {})[key] = value
    scenario_class = type(scenario)
    scenario.scenario_hash = scenario_hash(
        config, f"{scenario_class.__module__}.{scenario_class.__qualname__}"
    )


def add_effect_measures(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Keep on *scenario* the measure of each effect its ``controls`` write.

    A control is keyed by the scenario measure it sets; one with a
    ``quantity`` writes that quantity of the rules -- an effect a sweep draws
    per case (docs/odd.md, "A scenario's nodes") -- so the measure is what
    reads, in the run, what the case drew: quantity -> measure key
    (:attr:`BaseScenario._effect_measures`).  A run of a drawn case measures
    them beside the ODD, for the drift view.  Nothing when the config has no
    ``controls`` or none writes a quantity.
    """
    node = cfg.get("controls")
    # Unresolved: only each control's quantity, a name, is read here, and its
    # other fields may interpolate keys a check of the config has not built.
    raw = None if node is None else OmegaConf.to_container(node, resolve=False)
    controls = raw if isinstance(raw, dict) else {}
    scenario._effect_measures = {
        str(spec["quantity"]): str(measure)
        for measure, spec in controls.items()
        if isinstance(spec, dict) and spec.get("quantity")
    }


def add_environment(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Set the weather and the sun the config's ``environment`` names, if any.

    An :class:`~autoware_carla_scenario.EnvironmentAction` registered for
    initialization, so the world is in it before the run starts.  It is
    registered before the scenario's own ``setup()`` runs, so a scenario that
    sets the weather itself has the last word.  Nothing is registered when
    every field is ``null`` (the default); one is when ``rules`` alone is
    set, so the run gives its vehicles the grip of its rules (#1937).

    The conditions it names -- ``rainfall_rate_mm_h``,
    ``fog_visibility_range_m``, ``time_of_day``, ``wind_speed_kmh``,
    ``cloud_cover`` -- are worked out by the weather rules into CARLA's
    weather fields and the visibility, once, here
    (:func:`~autoware_carla_scenario.weather_rules.resolve_environment`):
    the action applies numbers, and records the conditions but the time of
    day as what the run was set to, which the ODD measures.  A CARLA field
    the config sets itself is kept.

    ``environment.rules`` names the weather rules the run follows: CARLA's
    weather, the grip and the visibility the conditions leave
    (:mod:`autoware_carla_scenario.weather_rules`) -- a file (a path, relative
    to the directory the run was started in, or an ``https://`` URL) merged
    over the default rules, or ``{default, vehicles}`` for rules that differ
    by vehicle.  The default rules when ``null``.  Followed here, to work
    the conditions out, and kept on *scenario*, which the runner follows at
    the start of each of its runs: a batch builds every case before any
    runs, and each runs under its own rules, not the last case's (#1936).

    Raises:
        ValueError: An ``environment.<key>`` is no setting
            (:func:`environment_settings`), is not a number, or is one
            :class:`~autoware_carla_scenario.EnvironmentAction` refuses (not
            finite, or outside the range its argument takes); the message
            names the key.
    """
    from autoware_carla_scenario.weather_rules import (  # noqa: PLC0415
        check_weather_rules,
        resolve_environment,
        use_weather_rules,
    )

    # Any: EnvironmentAction takes its other arguments by keyword too.
    settings: dict[str, Any] = dict(environment_settings(cfg))
    env_cfg = cfg.get("environment")
    raw = {} if env_cfg is None else _to_dict(env_cfg)
    rules = _rules_locations(raw.get("rules"))
    use_weather_rules(rules)
    # What the runner follows when the scenario runs (scenario_runner).
    default, vehicles = check_weather_rules(rules)
    scenario._weather_rules_default = None if default is None else str(default)
    scenario._weather_rules_vehicles = [(p, str(f)) for p, f in vehicles]
    if not settings and default is None and not vehicles:
        return
    # Once, before the run: the action applies numbers. The time of day and
    # the visibility quantile are the rules' alone; the cloud cover is
    # recorded as the run's, which the ODD measures (rain may render the sky
    # cloudier, #1930).
    settings = resolve_environment(settings)
    settings.pop("time_of_day", None)
    settings.pop("visibility_quantile", None)
    from autoware_carla_scenario import EnvironmentAction  # noqa: PLC0415

    scenario.register_init(EnvironmentAction(**settings, label="config.environment"))


def environment_settings(cfg: DictConfig) -> dict[str, Any]:
    """The config's ``environment`` settings but ``rules``, checked: the ones
    set, by name, each a number (an int or a float; not a boolean or a
    string, which ``float()`` would read) that
    :class:`~autoware_carla_scenario.EnvironmentAction` takes, or that the
    weather rules read (``visibility_quantile``) -- but ``time_of_day`` and
    ``cloud_cover``, one of the taxonomy's labels (``day``, ``dusk``,
    ``night``, ``dawn``; ``clear``, ``partly_cloudy``, ``overcast``).

    Raises:
        ValueError: naming the key, as :func:`add_environment` documents,
            and for a key that is none of these (``environment.rain``), which
            ``EnvironmentAction(**settings)`` would refuse only with a
            ``TypeError`` once the scenario was built (#2265).
    """
    import inspect  # noqa: PLC0415

    from autoware_carla_scenario import EnvironmentAction  # noqa: PLC0415
    from autoware_carla_scenario.actions.environment import (  # noqa: PLC0415
        _refusal,
    )
    from autoware_carla_scenario.odd.probes import (  # noqa: PLC0415
        CLOUD_COVERS,
        TIMES_OF_DAY,
    )

    # The settings that take one of the taxonomy's labels, not a number.
    label_settings = {"time_of_day": TIMES_OF_DAY, "cloud_cover": CLOUD_COVERS}

    env_cfg = cfg.get("environment")
    raw = {} if env_cfg is None else _to_dict(env_cfg)
    raw.pop("rules", None)
    # What the action takes, but its own (when and how it runs), and what
    # the weather rules read before it (add_environment).
    known = {
        name
        for name in inspect.signature(EnvironmentAction.__init__).parameters
        if name not in ("self", "condition", "timing", "label", "once")
    } | {"time_of_day", "visibility_quantile"}
    settings: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(
                f"environment.{key} is no environment setting, got {value!r}: "
                f"the settings are rules, {', '.join(sorted(known))}"
            )
        if value is None:
            continue
        if key in label_settings:
            labels = label_settings[key]
            if value not in labels:
                raise ValueError(
                    f"environment.{key} must be one of {list(labels)}, got {value!r}"
                )
            settings[key] = str(value)
            continue
        # A bool is an int to Python, and ``float("15")`` would read a string.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"environment.{key} must be a number, got {value!r}")
        try:
            number = float(value)
        except OverflowError:
            # An int too big for a float (10**400): not finite (#1652).
            raise ValueError(
                f"environment.{key} must be finite, got {shown(value)}"
            ) from None
        refusal = _refusal(str(key), number)
        if refusal != "":
            raise ValueError(f"environment.{key} {refusal}")
        settings[str(key)] = number
    return settings


def _check_weather_rules(cfg: DictConfig) -> None:
    """Refuse the config's ``environment.rules`` when it is not of the form
    :func:`add_environment` reads (a file, or ``{default, vehicles}`` of
    files), reading no rule file.

    Raises:
        ValueError: An ``openodd.RulesError`` naming the key.
    """
    from autoware_carla_scenario.weather_rules import (  # noqa: PLC0415
        check_weather_rules,
    )

    env_cfg = cfg.get("environment")
    raw = {} if env_cfg is None else _to_dict(env_cfg)
    check_weather_rules(raw.get("rules"))


def _rules_locations(spec: Any) -> Any:
    """``environment.rules`` with every path made absolute against the
    directory the run was started in (a URL as it is): a file, or
    ``{default, vehicles}`` (anything else is left to refuse)."""
    from hydra.utils import to_absolute_path  # noqa: PLC0415

    def where(location: Any) -> Any:
        # A blank one as it is too, for check_weather_rules to refuse
        # (to_absolute_path would make it the run's directory).
        if (
            not isinstance(location, str)
            or not location.strip()
            or location.startswith(("https://", "http://"))
        ):
            return location
        return to_absolute_path(location)

    if isinstance(spec, dict):
        out = dict(spec)
        if "default" in out:
            out["default"] = where(out["default"])
        if isinstance(out.get("vehicles"), dict):
            out["vehicles"] = {k: where(v) for k, v in out["vehicles"].items()}
        return out
    return where(spec)


class BackgroundTraffic(NamedTuple):
    """The ``background_traffic`` settings a run reads as numbers, checked
    (:func:`background_traffic_settings`)."""

    #: ``seed``; ``None`` follows the run's.
    seed: int | None
    initial_vehicles: int
    #: ``None``: no limit.
    max_vehicles: int | None
    vehicles_per_minute: float
    speed_kmh: float
    min_gap_m: float
    #: ``source.stop_after_seconds``; ``None``: the source keeps adding.
    source_stop_after_seconds: float | None = None
    #: ``sink.stop_after_seconds``; ``None``: the sink keeps removing.
    sink_stop_after_seconds: float | None = None
    #: ``source.constraints``, as the config gives them (each parsed by
    #: :func:`~autoware_carla_scenario.actions.background_traffic.parse_constraints`
    #: to check it); empty: no source.
    source_constraints: tuple[Any, ...] = ()
    #: ``sink.constraints``, likewise; empty: no sink.
    sink_constraints: tuple[Any, ...] = ()


def _float_of_text(value: Any) -> float:
    """``float(str(value))``, the way ``background_traffic`` reads a number
    (``"30"`` reads, ``true`` does not); an int of more digits than
    ``sys.get_int_max_str_digits()``, which ``str()`` refuses, is infinite,
    as ``float(str())`` of a shorter one too big for a float is (#1660).

    Raises:
        ValueError: For a value that is not a number.
    """
    try:
        text = str(value)
    except ValueError:
        if isinstance(value, int):
            return -math.inf if value < 0 else math.inf
        raise  # A value holding such an int ([10**5000]): not a number.
    return float(text)


def background_traffic_settings(cfg: DictConfig) -> BackgroundTraffic | None:
    """The numbers of the config's ``background_traffic``, checked; ``None``
    when it is off (``enabled`` false or the block absent or null), when none
    is read.  A block that is not a mapping (``true``, ``false``, ``0``,
    ``''``) is refused (:func:`_block`, #2298).

    ``seed``, ``source.initial_vehicles`` and ``source.max_vehicles`` are
    whole numbers, not negative (``2`` or ``2.0``; not ``1.9``, a boolean or
    a string that is no integer: ``int()`` alone would read ``1.9`` and
    ``true`` as 1); ``seed`` and ``max_vehicles`` may be null (the run's
    seed, no limit), and a null ``initial_vehicles`` places none.
    ``source.speed_kmh`` is finite and at least 1 km/h
    (:func:`~autoware_carla_scenario.traffic.base.check_background_speed`),
    and ``source.vehicles_per_minute`` and ``source.min_gap_m`` finite and
    not negative; each of these three absent or null is its default (30 km/h,
    15 m, none a minute; #2161).  ``source.stop_after_seconds`` and
    ``sink.stop_after_seconds`` are null (no stop, the default) or a positive,
    finite number of seconds -- each checked whether or not the source or sink
    it ends is built (``vehicles_per_minute`` 0, no sink constraints), as the
    other numbers are (#2188).  ``source.constraints`` and
    ``sink.constraints`` are lists of sweep constraints (or a single
    constraint mapping), each parsed here, so one the sweeper does not know is
    refused before anything starts; null or an empty list builds no source
    or sink, and ``false``, ``''`` or anything else that is no list is
    refused (#2191).  ``source`` and ``sink`` themselves are null (none) or a
    mapping; ``false``, ``''`` or ``0`` is refused rather than read as an
    empty block.  Read by :func:`add_background_traffic` and checked up
    front by :func:`check_batch_configs`.

    Raises:
        ValueError: naming the key of the first value refused.
    """
    background = cfg.get("background_traffic")
    if background is None or not _switch(cfg, "background_traffic", "enabled"):
        return None
    settings = _to_dict(background)

    def block(key: str) -> Mapping[str, Any]:
        # null or absent (the default): no source or sink.  false, '' or 0 is
        # refused rather than read as an empty block, which turned the source
        # or sink off without a word (#2143, #2191).
        value = settings.get(key)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(
                f"background_traffic.{key} must be a mapping, not {value!r}"
            )
        return value

    source = block("source")
    sink = block("sink")
    seed = seed_setting("background_traffic.seed", settings.get("seed"))
    initial = whole_number_setting(
        "background_traffic.source.initial_vehicles",
        source.get("initial_vehicles", 0),
        minimum=0,
    )
    max_vehicles = whole_number_setting(
        "background_traffic.source.max_vehicles",
        source.get("max_vehicles"),
        minimum=0,
    )
    # Refused here, naming the key, whether or not a source is built from it:
    # 0 would be traffic that never moves (#967).
    speed_value = _given(source, "speed_kmh", 30.0)
    try:
        speed = _float_of_text(speed_value)
    except ValueError:
        # Not float()'s "could not convert string to float: ..." (#2161).
        raise ValueError(
            f"background_traffic.source.speed_kmh: {shown(speed_value)} is not a "
            "number"
        ) from None
    try:
        speed_kmh = check_background_speed(speed)
    except ValueError as exc:
        raise ValueError(
            f"background_traffic.source.speed_kmh: {shown(speed_value)} is not a "
            f"background speed ({exc})"
        ) from exc

    def finite_amount(key: str, default: float) -> float:
        # Refused here, naming the key: NaN would fail `rate > 0` and be
        # dropped without a word, infinity would crash the run (#1321).
        value = _given(source, key, default)
        try:
            amount = _float_of_text(value)
        except ValueError as exc:
            raise ValueError(
                f"background_traffic.source.{key}: {shown(value)} is not a number"
            ) from exc
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                f"background_traffic.source.{key}: {shown(value)} must be finite "
                "and not negative"
            )
        return amount

    def stop_after(block: Any, key: str) -> float | None:
        # null (the default): no stop.  Anything else a positive, finite
        # number of seconds, as ElapsedTimeCondition takes it (#1660, #2188).
        seconds = block.get("stop_after_seconds")
        if seconds is None:
            return None
        try:
            return ElapsedTimeCondition(
                _float_of_text(seconds), label=f"background_{key}_ends"
            ).duration_seconds
        except ValueError as exc:
            raise ValueError(
                f"background_traffic.{key}.stop_after_seconds: {shown(seconds)} "
                f"is not a duration ({exc})"
            ) from exc

    def constraints(block: Any, key: str) -> tuple[Any, ...]:
        # null (the default): none.  false or '' is refused rather than
        # read as none, which turned the source or sink off (#2143, #2191);
        # a single mapping is a list of one, as constraints_from_text reads it.
        value = block.get("constraints")
        if value is None:
            return ()
        parse_constraints(value, f"background_traffic.{key}.constraints")
        return (value,) if isinstance(value, Mapping) else tuple(value)

    min_gap_m = finite_amount("min_gap_m", 15.0)
    rate = finite_amount("vehicles_per_minute", 0.0)
    source_constraints = constraints(source, "source")
    sink_constraints = constraints(sink, "sink")
    return BackgroundTraffic(
        seed=seed,
        # null places none, as 0 does.
        initial_vehicles=initial or 0,
        max_vehicles=max_vehicles,
        vehicles_per_minute=rate,
        speed_kmh=speed_kmh,
        min_gap_m=min_gap_m,
        source_stop_after_seconds=stop_after(source, "source"),
        sink_stop_after_seconds=stop_after(sink, "sink"),
        source_constraints=source_constraints,
        sink_constraints=sink_constraints,
    )


def add_background_traffic(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Register the run's background traffic on *scenario*, when it asks for any.

    ``background_traffic`` in the config (off by default) becomes up to three
    actions, whatever the scenario:

    * a :class:`TrafficSourceAction` registered for initialization, placing
      ``source.initial_vehicles`` before the run starts -- under ``traffic=sumo``
      before SUMO's warm-up, which spreads them along their routes;
    * another on the tick loop adding ``source.vehicles_per_minute``, ended by
      ``source.stop_after_seconds`` when given;
    * a :class:`TrafficSinkAction` removing them on ``sink.constraints``, ended
      by ``sink.stop_after_seconds`` when given.

    So ``background_traffic.enabled=true`` turns it on for any run, and
    ``background_traffic.source.vehicles_per_minute=0`` keeps it to the vehicles
    placed during initialization.

    Raises:
        ValueError: naming the key, for a value
            :func:`background_traffic_settings` refuses (a
            ``stop_after_seconds`` that is not a positive, finite number of
            seconds among them).
    """
    checked = background_traffic_settings(cfg)
    if checked is None:
        return
    settings = _to_dict(cfg["background_traffic"])
    # A mapping or null: anything else was refused above.
    source = settings.get("source") or {}
    # null (the default) follows the run's seed, the scenario's.
    seed = scenario.random_seed if checked.seed is None else checked.seed

    def until(seconds: float | None, label: str) -> ElapsedTimeCondition | None:
        # Checked by background_traffic_settings (#2188).
        return None if seconds is None else ElapsedTimeCondition(seconds, label=label)

    max_vehicles = checked.max_vehicles
    constraints = checked.source_constraints
    sink_constraints = checked.sink_constraints
    speed_kmh = checked.speed_kmh
    min_gap_m = checked.min_gap_m
    rate = checked.vehicles_per_minute

    def source_action(
        *, initial: int, rate: float, seed: int, label: str
    ) -> TrafficSourceAction:
        return TrafficSourceAction(
            constraints,
            initial_vehicles=initial,
            vehicles_per_minute=rate,
            max_vehicles=max_vehicles,
            speed_kmh=speed_kmh,
            min_gap_m=min_gap_m,
            blueprint=source.get("blueprint"),
            seed=seed,
            label=label,
            until=(
                until(checked.source_stop_after_seconds, "background_source_ends")
                if rate > 0
                else None
            ),
        )

    initial = checked.initial_vehicles
    if constraints and initial > 0:
        scenario.register_init(
            source_action(
                initial=initial, rate=0.0, seed=seed, label="background_initial"
            )
        )
    if constraints and rate > 0:
        scenario.register_pre_tick(
            source_action(
                initial=0,
                rate=rate,
                # One past the initial vehicles' seed, wrapped into the seed
                # range: MAX_SEED's successor is 0, not a seed refused (#1560).
                seed=(seed + 1) % (MAX_SEED + 1),
                label="background_source",
            )
        )
    if sink_constraints:
        scenario.register_pre_tick(
            TrafficSinkAction(
                sink_constraints,
                label="background_sink",
                until=until(checked.sink_stop_after_seconds, "background_sink_ends"),
            )
        )
    logger.info(
        "Background traffic: %d at start, %.1f/min after, sink %s",
        initial if constraints else 0,
        rate if constraints else 0.0,
        "on" if sink_constraints else "off",
    )


def run_scenario(
    cfg: DictConfig,
    *,
    build_scenario_fn: BuildScenarioFn | None = None,
) -> ScenarioResult:
    """Build and execute a scenario from a resolved Hydra config.

    Parameters
    ----------
    cfg:
        Resolved Hydra config.
    build_scenario_fn:
        Optional callable forwarded to :func:`build_scenario`.

    Returns the :class:`ScenarioResult` so that callers (including Hydra
    multirun) can inspect it without the process being terminated.

    Hydra changes the working directory to its output directory
    (e.g. ``outputs/YYYY-MM-DD/HH-MM-SS/``) before this function is
    called, so all relative paths resolve inside that directory.
    """
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    warn_if_video_set(cfg)
    # Before anything is built: a config value the run would refuse later
    # (RunConfigError, which a case the lanelet sweeper forked exits 2 for).
    check_run_config(cfg)
    # Refused at once without the extras.
    rerun = build_rerun_converter(cfg)
    build_t4_recorder(cfg)
    build_t4scene_exporter(cfg)
    try:
        apply_canvas(cfg)
        apply_metrics(cfg)
        return _run_scenario(cfg, rerun, build_scenario_fn=build_scenario_fn)
    finally:
        _finish_rerun(rerun)


def _queue_retries(cfg: DictConfig) -> int:
    """How often the run's queue retries a run that raised.

    ``server.cooldown_max_retries``, except in a case the lanelet sweeper
    forked: the sweeper retries that case itself, so the queue does not.
    """
    # Loads Lanelet2, as the run does anyway.
    from ..sweeper.isolation import (  # noqa: PLC0415
        SWEEPER_OWNS_RETRIES_ENV,
    )

    if os.environ.get(SWEEPER_OWNS_RETRIES_ENV):
        return 0
    return cooldown_max_retries(cfg)


def _run_scenario(
    cfg: DictConfig,
    rerun: RerunConverter | None,
    *,
    build_scenario_fn: BuildScenarioFn | None = None,
) -> ScenarioResult:
    """:func:`run_scenario`, with its Rerun converter, if any."""
    _ego, scenario = build_scenario(cfg, build_scenario_fn=build_scenario_fn)
    # The ODD the run is measured against, checked when it is Python.
    check_odd(_odd_spec(cfg), typecheck_mode(cfg))

    map_paths = resolve_map_paths(cfg.map)
    cooldown = _cooldown_seconds(cfg)
    cooldown_max_retries = _queue_retries(cfg)
    world_reset, reload_every = world_reset_settings(cfg)

    # Retrieve the Hydra output directory (works regardless of
    # ``hydra.job.chdir`` which defaults to False since Hydra 1.2).
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    # The stem the run's files are named by, settled here and handed to the
    # runner, so that the path printed below is the one it wrote: past the
    # class name once an earlier run in this directory has taken it (a
    # sweeper's retry in the same job directory, a reused run directory).
    file_stem = _run_file_stem(output_dir, type(scenario).__name__)

    result = run_scenario_with_queue(
        scenario,
        host=cfg.server.host,
        port=cfg.server.port,
        tm_port=_traffic_manager_port(cfg),
        xodr_path=map_paths.install_xodr,
        overwrite_xodr=map_paths.overwrite_xodr,
        opendrive_path=map_paths.opendrive_path,
        lanelet2_path=map_paths.lanelet2_path,
        map_name=map_paths.name,
        cooldown_seconds=cooldown,
        cooldown_max_retries=cooldown_max_retries,
        output_dir=output_dir,
        # The runner's own fail-safe: the queue default (60 s) is shorter than
        # an Autoware stack needs to localize, route and engage.
        timeout_seconds=run_timeout_seconds(cfg),
        max_tick_rate_hz=_max_tick_rate_hz(cfg),
        projector_type=map_paths.projector_type,
        traffic_backend=build_traffic_backend(cfg),
        odd=_odd_spec(cfg),
        autoware_launcher=build_autoware_launcher(cfg),
        case=_odd_case(cfg),
        server_extra_args=server_extra_args(cfg),
        map_source=_map_source(cfg, map_paths),
        rerun=rerun,
        t4=build_t4_recorder(cfg),
        world_reset=world_reset,
        reload_every=reload_every,
        t4scene=build_t4scene_exporter(cfg),
        backends=build_backend_pool(cfg),
        file_stem=file_stem,
        map_pack=build_map_pack(cfg, map_paths),
    )

    status = "PASSED" if result.passed else "FAILED"
    print(f"{status}: {result.message} ({result.elapsed_seconds:.2f}s)")  # noqa: T201
    print(_result_json_line(output_dir, file_stem))  # noqa: T201
    return result


#: The cases of a ``--multirun`` run in this process (Hydra's basic sweeper)
#: that failed, and how many ran: :func:`main` exits 1 when any failed.  The
#: lanelet sweeper runs its cases in forked children, which this does not see;
#: it raises ``SweepFailedError`` itself.
_multirun_failed: list[str] = []
_multirun_ran: list[str] = []


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point that dispatches to the selected scenario."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    marker_dir = _case_errored_dir()
    # Around the handlers too: what the last one captures is the run's.
    with _crash_report_context(cfg):
        _hydra_case(cfg, marker_dir)


def _hydra_case(cfg: DictConfig, marker_dir: Path | None) -> None:
    """:func:`_hydra_main`'s run of *cfg*, its verdict and its errors."""
    try:
        result = run_scenario(cfg)
    except (ScenarioTypeError, RerunUnavailableError, T4UnavailableError) as exc:
        # A refusal ends a --multirun too, with the same 2: the next case
        # would be refused for the same reason.  Hydra lets SystemExit through
        # its basic sweeper, and the lanelet sweeper stops on a case's 2.
        _mark_case_errored(marker_dir, exc, exit_code=2)
        print(exc, file=sys.stderr)  # noqa: T201
        sys.exit(2)
    except RunConfigError as exc:
        # A case the lanelet sweeper forked is refused too: every retry of it
        # and every case after it would be refused for the same value.  The
        # sweeper checks its config before forking any (#1233); this catches
        # a value only a case's own overrides set.  Elsewhere a config error
        # exits 1, as it always has (docs/usage.md, "Exit Codes").
        from ..sweeper.isolation import (  # noqa: PLC0415
            SWEEPER_OWNS_RETRIES_ENV,
        )

        if not os.environ.get(SWEEPER_OWNS_RETRIES_ENV):
            _mark_case_errored(marker_dir, exc, exit_code=1)
            if _record_basic_sweeper_error(exc):
                return
            raise
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(2)
    except Exception as exc:
        # Before Hydra's own report of it, which ends a single run with exit
        # 1 and leaves main() nothing to report.  A refusal is not a crash:
        # the clauses above take those.
        crash_report.capture(exc)
        _mark_case_errored(marker_dir, exc, exit_code=1)
        if _record_basic_sweeper_error(exc):
            return
        raise
    # Ended with a verdict, passed or failed: the run wrote its result (and
    # gate record), and is judged; the marker saying it did not finish goes.
    _clear_case_errored(marker_dir)
    # Only exit for single-run mode. In --multirun, Hydra calls this
    # function repeatedly; sys.exit() would kill the entire sweep.  Hydra
    # discards what a job returns, so the verdict is recorded for main().
    if not _is_multirun():
        sys.exit(0 if result.passed else 1)
    case = " ".join(HydraConfig.get().overrides.task)
    _multirun_ran.append(case)
    if not result.passed:
        _multirun_failed.append(case)


@contextmanager
def _crash_report_context(cfg: DictConfig) -> Iterator[None]:
    """Crash reports of the ``with`` block (a native crash's too) are the
    Hydra run's: in its output directory, with its overrides and its config
    (:func:`crash_report.run_context`)."""
    if not crash_report.installed():
        yield
        return
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # noqa: BLE001 - the run goes on, reported elsewhere
        logger.warning("Cannot tell the run's output directory", exc_info=True)
        yield
        return
    with crash_report.run_context(
        output_dir, overrides=_task_overrides(), config_yaml=config_yaml(cfg)
    ):
        yield


def config_yaml(cfg: Any) -> str | None:
    """*cfg* as YAML for a crash report; ``None`` when reports hold no
    config (:func:`crash_report.wants`) or it cannot be written."""
    if not crash_report.wants("config"):
        return None
    try:
        return OmegaConf.to_yaml(cfg)
    except Exception:  # noqa: BLE001 - reported without its config
        return None


#: The ``_target_`` of Hydra's basic sweeper and launcher, which run every
#: case in this process (:func:`_record_basic_sweeper_error`).
_BASIC_SWEEPER_TARGET = "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"
_BASIC_LAUNCHER_TARGET = "hydra._internal.core_plugins.basic_launcher.BasicLauncher"


def _record_basic_sweeper_error(exc: BaseException) -> bool:
    """Record *exc*, what a ``--multirun`` case of Hydra's basic sweeper
    raised, as that case's error, with its traceback logged, and return
    ``True``; ``False`` (the caller re-raises) outside such a case.

    Hydra's basic sweeper runs every case before it re-raises the first
    error, so a raise would only lose the others' verdicts and
    :func:`_exit_if_multirun_failed`'s summary: the case is counted as
    failed (errored) instead, the sweep goes on, and :func:`main` exits 1
    with the summary naming it (#2294).  A single run raises as it always
    has; a case the lanelet sweeper forked (``SWEEPER_OWNS_RETRIES_ENV``)
    too, since the sweeper reads its exit status and retries it.  So does a
    case of any other sweeper or launcher (an installed plugin's: Optuna,
    joblib, submitit): one may run it in another process, where what is
    recorded here never reaches :func:`main`, and only the raise reports it.
    """
    from ..sweeper.isolation import (  # noqa: PLC0415
        SWEEPER_OWNS_RETRIES_ENV,
    )

    if (
        not HydraConfig.initialized()
        or not _is_multirun()
        or os.environ.get(SWEEPER_OWNS_RETRIES_ENV)
    ):
        return False
    hydra_cfg = HydraConfig.get()
    if (
        hydra_cfg.sweeper.get("_target_") != _BASIC_SWEEPER_TARGET
        or hydra_cfg.launcher.get("_target_") != _BASIC_LAUNCHER_TARGET
    ):
        return False
    case = " ".join(_task_overrides())
    logger.error(
        "Case %s raised %s; the sweep goes on",
        case or "(no overrides)",
        type(exc).__name__,
        exc_info=exc,
    )
    _multirun_ran.append(case)
    _multirun_failed.append(_errored_case(case, type(exc).__name__, str(exc)))
    return True


def _case_errored_dir() -> Path | None:
    """The job directory a ``--multirun`` case of Hydra's basic sweeper
    leaves its marker in (``case_errored.json``,
    :mod:`~autoware_carla_scenario.gate.errored`), given the marker saying
    the case did not finish, which replaces any an earlier run of it left
    there; ``None`` outside such a case.

    The marker is replaced by the one of a case that raised
    (:func:`_mark_case_errored`), or removed once the case ends with a
    verdict (:func:`_clear_case_errored`): left behind, it is the trace of a
    case whose sweep died with it -- a segfault, the OOM killer, ``SIGKILL``,
    Ctrl-C (#1833).  Hydra's basic sweeper creates each job's directory as
    it reaches the job, and so the cases it never reached leave nothing
    (docs/gate.md).

    A case the lanelet sweeper forked (``SWEEPER_OWNS_RETRIES_ENV``) leaves
    it alone: the sweeper owns its retries, and so its marker.
    """
    from ..sweeper.isolation import (  # noqa: PLC0415
        SWEEPER_OWNS_RETRIES_ENV,
    )

    if not _is_multirun() or os.environ.get(SWEEPER_OWNS_RETRIES_ENV):
        return None
    try:
        directory = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # noqa: BLE001 - the case runs anyway
        logger.warning("Cannot tell the case's output directory", exc_info=True)
        return None
    try:
        write_case_unfinished(directory, overrides=_task_overrides(), attempt=1)
    except Exception:  # noqa: BLE001 - the case runs anyway
        logger.warning("Cannot write the case's unfinished marker", exc_info=True)
        try:
            # At least no stale marker of an earlier run is left to say this
            # one raised.
            remove_case_errored(directory)
        except Exception:  # noqa: BLE001
            logger.warning("Cannot clear the case's error marker", exc_info=True)
            return None
    return directory


def _task_overrides() -> list[str]:
    """The overrides of the ``--multirun`` case running, as Hydra holds
    them; none when they cannot be read."""
    try:
        return [str(o) for o in HydraConfig.get().overrides.task]
    except Exception:  # noqa: BLE001 - the marker is written anyway
        return []


def _clear_case_errored(directory: Path | None) -> None:
    """Remove the marker of a ``--multirun`` case that ended with a verdict
    from *directory* (:func:`_case_errored_dir`).  Nothing when *directory*
    is ``None``; never raises."""
    if directory is None:
        return
    try:
        remove_case_errored(directory)
    except Exception:  # noqa: BLE001 - the verdict is what is reported
        logger.warning("Cannot remove the case's unfinished marker", exc_info=True)


def _mark_case_errored(
    directory: Path | None, exc: BaseException, *, exit_code: int
) -> None:
    """Leave the marker of a ``--multirun`` case that raised (*exc*) in
    *directory* (:func:`_case_errored_dir`), so that ``scenario-gate`` does
    not judge the cases that survived alone (#1830).  Nothing when
    *directory* is ``None``; never raises."""
    if directory is None:
        return
    try:
        write_case_errored(
            directory,
            overrides=_task_overrides(),
            exit_code=exit_code,
            attempts=1,
            reason=f"raised {type(exc).__name__}: {exc}",
        )
    except Exception:  # noqa: BLE001 - what the case raised is what is reported
        logger.warning("Cannot write the case's error marker", exc_info=True)


def _exit_if_multirun_failed() -> None:
    """Exit 1 when a ``--multirun`` case run in this process failed.

    After the sweep, so a failed case does not stop the ones after it.
    """
    if not _multirun_failed:
        return
    for line in _sweep_failed_lines(_multirun_failed, len(_multirun_ran)):
        print(line, file=sys.stderr)  # noqa: T201
    sys.exit(1)


class _Terminated(KeyboardInterrupt):
    """The ``KeyboardInterrupt`` a ``SIGTERM`` raises (:func:`_raise_interrupt`).

    It carries the signal, so the run exits ``128 + signal`` (143) rather than
    as Python ends one Ctrl-C stopped.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signal.Signals(signum).name}")
        self.signum = signum


def _raise_interrupt(signum: int, _frame: object) -> None:
    """End the run as Ctrl-C does, so its ``with`` blocks stop what they started.

    Once: a second SIGTERM (``docker stop``, a CI runner and systemd may send
    more than one) must not cut that clean-up short, ``atexit``'s included.
    As ``slurm.worker._raise_interrupt`` does for a SLURM task.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise _Terminated(signum)


@contextmanager
def _interrupt_on_sigterm() -> Iterator[None]:
    """SIGTERM raises :class:`_Terminated`, a ``KeyboardInterrupt``, in the block (#707).

    Python's default SIGTERM ends the process without unwinding: neither
    ``with queue`` nor ``atexit`` runs, so the Autoware stack and CARLA are
    stopped only by their die-with-parent watchdogs, without the run's own
    clean-up (its results, its Rerun recording).  Installed only over the
    default action, in the main thread: an embedding program's own handler is
    left alone.  The default is put back when the block ends, unless a SIGTERM
    came (it stays ignored for the rest of the clean-up) or the block put a
    handler of its own in place.
    """
    if (
        threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    ):
        yield
        return
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        yield
    finally:
        if signal.getsignal(signal.SIGTERM) is _raise_interrupt:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


#: How :func:`main` hands ``--resume-from`` to the sweeper
#: (``LaneletConstraintSweeper`` reads it).
RESUME_FROM_ENV = "SWEEP_RESUME_FROM"


@contextmanager
def _environ_for_run(name: str, value: str | None) -> Iterator[None]:
    """Set the environment variable *name* to *value* for the ``with`` block,
    and put back what it was (unset included) when it ends, however it ends.

    *value* ``None`` leaves the variable as it is.
    """
    if value is None:
        yield
        return
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


#: What a run raises when it refuses its configuration: the user's to fix, not
#: a crash, and so not reported (:func:`crash_report.reporting`).
_REFUSALS: tuple[type[BaseException], ...] = (
    RunConfigError,
    ScenarioTypeError,
    RerunUnavailableError,
    T4UnavailableError,
)


def _quiet_interrupt(exc: KeyboardInterrupt) -> None:
    """Let *exc*, re-raised, end the process without a traceback (#2333).

    Python still ends the process as it ends one Ctrl-C stopped: once it has
    finalized (``atexit`` included), it kills itself with ``SIGINT``, so a
    parent sees a death by signal (a shell's 130), not an exit status.  Only
    *exc* is silenced: any other uncaught exception goes to the hook as it
    did, and an embedding program that catches *exc* sees it as before.

    The hook puts the one it replaced back the first time it is called, and
    a hook an earlier call left (its caller caught that interrupt) is
    replaced rather than wrapped: calling ``main()`` again in one process
    neither stacks hooks nor keeps every interrupt (and its frames) alive.
    """
    previous = sys.excepthook
    if isinstance(previous, _QuietInterruptHook):
        previous = previous.previous
    sys.excepthook = _QuietInterruptHook(exc, previous)


class _QuietInterruptHook:
    """``sys.excepthook`` silencing one interrupt (:func:`_quiet_interrupt`)."""

    def __init__(self, exc: KeyboardInterrupt, previous: Any) -> None:
        self.exc = exc
        self.previous = previous

    def __call__(
        self, kind: type[BaseException], value: BaseException, tb: Any
    ) -> None:
        if sys.excepthook is self:
            sys.excepthook = self.previous
        if value is not self.exc:
            self.previous(kind, value, tb)


def main() -> None:
    """CLI entry point: detect glob patterns and dispatch accordingly.

    A ``SIGTERM`` stops the run as Ctrl-C does, and it exits 143.  Ctrl-C
    prints ``scenario: interrupted`` rather than a traceback, and the process
    still dies by ``SIGINT`` (:func:`_quiet_interrupt`), once the run has
    stopped what it started.  What it raises or a native crash leaves a crash
    report (:mod:`..crash_report`).
    """
    # Before the crash reporter puts its log buffer on the root logger: with
    # a handler there already, basicConfig() would do nothing, and a batch
    # would log nothing to the console (#2380).  A run through Hydra
    # replaces this with its own logging, as before.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        with (
            crash_report.reporting(list(sys.argv), ignore=_REFUSALS),
            _interrupt_on_sigterm(),
        ):
            _main()
    except _Terminated as exc:
        print(f"Interrupted ({exc})", file=sys.stderr)  # noqa: T201
        sys.exit(128 + exc.signum)
    except KeyboardInterrupt as exc:
        print("scenario: interrupted", file=sys.stderr)  # noqa: T201
        _quiet_interrupt(exc)
        raise


def _main() -> None:
    # First thing in main(), before the run starts its own threads (see
    # _cap_malloc_arenas). Threads the imports start have not allocated yet,
    # so they hold no arenas: glibc ignores a cap set once it has more.
    _cap_malloc_arenas()
    # Discover external scenario packages (entry-point plugins) before we touch
    # the config search path or the scenario registry, so their scenarios and
    # conf dirs are available to both the glob and single-run code paths.
    load_scenario_plugins()

    # Extract --resume-from and --resume-into before Hydra sees the argv.
    resume_from, cleaned_argv = _extract_resume_from(sys.argv)
    resume_into, cleaned_argv = _extract_resume_into(cleaned_argv)
    sys.argv = cleaned_argv

    # Pass via environment variable so the sweeper can read it without
    # going through Hydra's CLI parser (which rejects unknown overrides).
    # For this run only: a later main() in the same process (tests, an
    # embedding runner) without --resume-from must not resume from it.
    with _environ_for_run(
        RESUME_FROM_ENV, str(resume_from) if resume_from > 0 else None
    ):
        _dispatch(resume_from, resume_into)


def _extract_batch_output_dir(
    argv: list[str], *, multirun: bool
) -> tuple[Path | None, list[str]]:
    """The batch output directory *argv* names, and *argv* without it.

    ``hydra.run.dir=`` names it, or with ``--multirun`` (*multirun*)
    ``hydra.sweep.dir=``, as for a run through Hydra (#2294); ``None`` when
    it names none.  Every ``hydra.run.dir`` and ``hydra.sweep.dir`` override
    is taken off: neither is a case's setting.  The other mode's key is
    ignored with a warning, as Hydra ignores it.  A ``${now:...}``
    interpolation in it is resolved, as Hydra resolves it.

    Exits 2 when the value cannot be resolved or is empty.
    """
    wanted = _SWEEP_DIR_KEY if multirun else _RUN_DIR_KEY
    value, kept, ignored = _split_batch_output_dir(argv, multirun=multirun)
    for arg in ignored:
        logger.warning(
            "%s is ignored: a batch%s writes to %s",
            arg,
            " run with --multirun" if multirun else " without --multirun",
            wanted,
        )
    if value is None:
        return None, kept
    text = _batch_dir_text(value)
    if "${" in text:
        from hydra.core.utils import setup_globals  # noqa: PLC0415

        setup_globals()
        try:
            text = str(OmegaConf.create({"dir": text}).dir)
        except Exception as exc:  # noqa: BLE001 -- reported, exit 2
            print(  # noqa: T201
                f"Error: {wanted}={value}: cannot resolve it for a batch ({exc})",
                file=sys.stderr,
            )
            sys.exit(2)
    if not text:
        print(f"Error: {wanted}= names no directory", file=sys.stderr)  # noqa: T201
        sys.exit(2)
    return Path(text), kept


def _warn_if_sweep_not_expanded(scenario: str | None, overrides: list[str]) -> None:
    """Warn when *scenario* (the default one when ``None``), composed with
    the command line's *overrides*, has a sweep that Hydra's basic sweeper
    does not expand, and the command line sweeps nothing of its own: such a
    ``--multirun`` without ``hydra/sweeper=`` runs one job, the scenario as
    configured, not its cases (#2294), where the same scenario given as a
    glob runs every case.

    A command line that sweeps a value itself (``seed=1,2``,
    ``'seed=range(0,32)'``: the rollouts of one scenario, docs/wosac.md) asks
    for those jobs, of the scenario as configured, and is not warned.
    Composed without what Hydra reads itself (``hydra...``, flags); nothing
    is said when it does not compose: the run reports that itself.
    """
    from hydra.core.override_parser.overrides_parser import (  # noqa: PLC0415
        OverridesParser,
    )

    plain = _plain_overrides(overrides)
    try:
        parsed = OverridesParser.create().parse_overrides(plain)
    except Exception:  # noqa: BLE001 -- the run reports it
        logger.debug("Cannot parse the command line's overrides", exc_info=True)
        return
    if any(override.is_sweep_override() for override in parsed):
        return
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=(
                    [f"scenario={scenario}", *plain] if scenario is not None else plain
                ),
            )
        swept = _has_sweep(cfg)
    except Exception:  # noqa: BLE001 -- the run reports it
        logger.debug("Cannot tell whether the scenario has a sweep", exc_info=True)
        return
    finally:
        GlobalHydra.instance().clear()
    if swept:
        logger.warning(
            "%s has a sweep (sweep.constraints, sweep.route or sweep.odd_sample) "
            "that Hydra's basic sweeper does not expand: --multirun runs one job, "
            "the scenario as configured; add "
            "hydra/sweeper=lanelet_constraint, or name the scenario with a glob "
            "(a batch runs every case of each scenario's sweep with "
            "--multirun), to run every case of it",
            scenario if scenario is not None else "The scenario",
        )


def _dispatch(resume_from: int, resume_into: str | None = None) -> None:
    """Run what ``sys.argv`` (``--resume-from`` and ``--resume-into`` taken
    off) selects: a batch of the scenarios a glob or a SLURM launcher names,
    or one through Hydra.  ``--resume-into`` is a local batch's alone: a run
    through Hydra refuses it (exit 2).  ``--resume-from`` is a batch's (a
    glob, or ``launcher=slurm``) or the lanelet sweeper's (``--multirun
    hydra/sweeper=lanelet_constraint``): a single run or Hydra's basic
    sweeper would ignore it, and refuses it (exit 2).  ``launcher=slurm``
    runs a batch, so it needs ``scenario=``: without one it is refused
    (exit 2) rather than running the default scenario here (#2089)."""
    # --no-wait: a launcher=slurm batch is submitted, and this exits.
    no_wait = "--no-wait" in sys.argv
    if no_wait:
        sys.argv = [arg for arg in sys.argv if arg != "--no-wait"]

    from ..slurm.settings import launcher_requested_in_argv  # noqa: PLC0415

    scenario_value, remaining = _extract_scenario_override(sys.argv)
    # A SLURM launcher runs a batch, so a single scenario selected with it runs
    # as one too: --multirun's cases, or the scenario once, as SLURM tasks.
    slurm = launcher_requested_in_argv(remaining) == "slurm"
    if slurm and scenario_value is None and not asks_for_no_job(sys.argv):
        # @hydra.main would compose the default scenario and run it here,
        # ignoring the launcher (#2089).
        print(  # noqa: T201
            "Error: launcher=slurm runs a batch: name its scenarios with "
            "scenario=<name or glob>",
            file=sys.stderr,
        )
        sys.exit(2)
    # --help, --cfg job, ... ask Hydra for no job: @hydra.main answers them
    # rather than a batch (#2294), for the first scenario a glob matches.
    no_job = asks_for_no_job(sys.argv)
    if no_job and scenario_value is not None and _is_glob_pattern(scenario_value):
        names = _resolve_scenario_glob(scenario_value)
        print(  # noqa: T201 -- stderr: --cfg prints the config to stdout
            f"Note: scenario={scenario_value} matches {len(names)} scenario(s); "
            f"this is Hydra's answer for the first, {names[0]}",
            file=sys.stderr,
        )
        sys.argv = [
            f"scenario={names[0]}" if arg == f"scenario={scenario_value}" else arg
            for arg in sys.argv
        ]
        scenario_value = names[0]
    if (
        scenario_value is not None
        and (_is_glob_pattern(scenario_value) or slurm)
        and not no_job
    ):
        # --multirun runs every case of each scenario's sweep in the batch's
        # one queue; the sweeper it would pick is the batch itself.  Read off
        # argv: no Hydra job runs here (one an earlier main() left behind
        # in this process is not this run's).
        expand = _multirun_requested_in_argv(sys.argv)
        remaining = [
            arg
            for arg in remaining
            if arg not in ("--multirun", "-m")
            and not _is_multirun_mode_override(arg)
            and not arg.startswith("hydra/sweeper=")
        ]
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
        if _is_glob_pattern(scenario_value):
            scenario_names = _resolve_scenario_glob(scenario_value)
            logger.info(
                "Glob matched %d scenario(s): %s",
                len(scenario_names),
                scenario_names,
            )
        else:
            scenario_names = [scenario_value]
        # hydra.run.dir= (hydra.sweep.dir= with --multirun): the batch's
        # output directory, not a case's override (#2294).
        batch_dir, remaining = _extract_batch_output_dir(remaining, multirun=expand)
        # Each only when given.
        extra: dict[str, Any] = {}
        if no_wait:
            extra["wait"] = False
        if resume_into is not None:
            extra["resume_into"] = Path(resume_into)
        if batch_dir is not None:
            extra["output_dir"] = batch_dir
        try:
            run_batch(
                scenario_names,
                remaining,
                expand=expand,
                resume_from=resume_from,
                **extra,
            )
        except (ScenarioTypeError, RerunUnavailableError, T4UnavailableError) as exc:
            print(exc, file=sys.stderr)  # noqa: T201
            sys.exit(2)
        except RunConfigError as exc:
            # A config value a case would refuse (check_batch_configs), before
            # anything was built: the key's line, not a traceback (#1430).  A
            # configuration error, exit 1, as a single run's is and as the
            # batch's other refusals are; HYDRA_FULL_ERROR=1 raises it.
            if os.environ.get("HYDRA_FULL_ERROR") == "1":
                raise
            print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(1)
    else:
        if no_wait:
            print("Error: --no-wait is for launcher=slurm only", file=sys.stderr)  # noqa: T201
            sys.exit(1)
        if resume_into is not None:
            print(  # noqa: T201
                "Error: --resume-into resumes a local batch (a scenario glob); "
                "a sweep resumes into its own directory with hydra.sweep.dir=",
                file=sys.stderr,
            )
            sys.exit(2)
        if resume_from > 0 and not (
            _multirun_requested_in_argv(sys.argv)
            and _lanelet_sweeper_requested_in_argv(sys.argv)
        ):
            # Only the lanelet sweeper reads SWEEP_RESUME_FROM: a single run
            # or Hydra's basic sweeper would run every case (#2089).
            print(  # noqa: T201
                "Error: --resume-from resumes a batch (a scenario glob, or "
                "launcher=slurm) or a lanelet sweep (--multirun "
                "hydra/sweeper=lanelet_constraint); a single run or Hydra's "
                "basic sweeper would ignore it",
                file=sys.stderr,
            )
            sys.exit(2)
        # Single run goes through @hydra.main.  External conf dirs are added to
        # the search path by AutowareScenarioSearchPathPlugin (discovered via
        # the hydra_plugins namespace), so nothing needs threading here.
        # A sweep's verdict is this call's alone, not an earlier main()'s in
        # this process (tests, an embedding runner).
        _multirun_failed.clear()
        _multirun_ran.clear()
        if (
            not no_job
            and _multirun_requested_in_argv(sys.argv)
            and not _sets_hydra_key(sys.argv, "hydra/sweeper")
        ):
            _warn_if_sweep_not_expanded(scenario_value, remaining)
        # This run's own output directory, reserved before Hydra creates
        # Hydra's timestamped one with exist_ok (#1359).
        sys.argv, reserved = _reserve_hydra_output_dir(sys.argv)
        try:
            _hydra_main()
        finally:
            if reserved is not None:
                _remove_if_empty(reserved)
        _exit_if_multirun_failed()


if __name__ == "__main__":
    main()
