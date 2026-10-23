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

import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typesafe_carla.carla as carla
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from autoware_carla_scenario import (
    BaseScenario,
    EgoConfig,
    EgoVehicle,
    ElapsedTimeCondition,
    GroundProjectionConfig,
    Lanelet2Pose,
    ScenarioQueue,
    SpawnTransform,
    TrafficSinkAction,
    TrafficSourceAction,
)
from autoware_carla_scenario.autoware_stack.launcher import AutowareLauncher
from autoware_carla_scenario.batch_report import print_summary as _print_summary
from autoware_carla_scenario.canvas import use_canvas
from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.constants import DEFAULT_TM_PORT
from autoware_carla_scenario.maps import resolve_map_paths
from autoware_carla_scenario.odd.route import PlannedRoute, RouteError
from autoware_carla_scenario.odd.sampler import ODD_CASE_KEY, OddCase, decode_case
from autoware_carla_scenario.rerun import (
    RerunConversion,
    RerunConverter,
    RerunUnavailableError,
)
from autoware_carla_scenario.rerun import report as rerun_report
from autoware_carla_scenario.server import fog_density_scale_args
from autoware_carla_scenario.t4 import T4Recorder, T4Settings, T4UnavailableError
from autoware_carla_scenario.typecheck import ScenarioTypeError
from autoware_carla_scenario.typecheck.mode import (
    check_odd,
    check_registered_scenario,
    typecheck_mode,
)
from autoware_carla_scenario.traffic import (
    TrafficBackend,
    TrafficConfig,
    available_backends,
    build_backend,
    load_traffic_backend_plugins,
)
from autoware_carla_scenario.registry import (
    BuildScenarioFn,
    get_conf_dirs,
    get_scenario_builder,
    get_scenario_registry,
    load_scenario_plugins,
    register_conf_dir,
    register_scenario,
)

from .configs import (
    IntersectionPassingConfig,
    LaneChangeConfig,
    TemporaryStopConfig,
    PedestrianDartOutConfig,
    CutInConfig,
    TrafficLightComplianceConfig,
)
from .intersection_passing import IntersectionPassingScenario
from .lane_change import LaneChangeScenario
from .temporary_stop import TemporaryStopScenario
from .pedestrian_dart_out import PedestrianDartOutScenario
from .cut_in import CutInScenario
from .traffic_light_compliance import TrafficLightComplianceScenario

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


# Register built-in scenarios.
register_scenario(
    "intersection_passing", IntersectionPassingScenario, IntersectionPassingConfig
)
register_scenario(
    "traffic_light_compliance",
    TrafficLightComplianceScenario,
    TrafficLightComplianceConfig,
)
register_scenario("lane_change", LaneChangeScenario, LaneChangeConfig)
register_scenario("temporary_stop", TemporaryStopScenario, TemporaryStopConfig)
register_scenario(
    "pedestrian_dart_out", PedestrianDartOutScenario, PedestrianDartOutConfig
)
register_scenario("cut_in", CutInScenario, CutInConfig)


# ---------------------------------------------------------------------------
# Process memory
# ---------------------------------------------------------------------------

#: glibc's ``mallopt`` parameter for the number of malloc arenas.
_M_ARENA_MAX = -8

#: The arenas a run keeps: few enough that freed memory is reused rather than
#: stranded in an arena no thread allocates from again.
_ARENA_MAX = 2


def _arena_max_set_by_user(environ: Mapping[str, str]) -> bool:
    """Return whether the environment already says how many arenas to keep."""
    if environ.get("MALLOC_ARENA_MAX"):
        return True
    return "arena_max" in environ.get("GLIBC_TUNABLES", "")


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
    ground_projection = GroundProjectionConfig(
        ray_distance_upper=float(cfg.entity.ground_projection_ray_distance_upper),
        ray_distance_lower=float(cfg.entity.ground_projection_ray_distance_lower),
    )
    ego = EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        ),
        vehicle_type=cfg.ego.vehicle_type,
        initial_speed_kmh=float(cfg.ego.initial_speed_kmh),
        spawn_retry_max_count=int(cfg.entity.spawn_retry_max_count),
        spawn_retry_t_step=float(cfg.entity.spawn_retry_t_step),
        spawn_retry_z_step=float(cfg.entity.spawn_retry_z_step),
        goal_pose=build_goal_pose(cfg),
        waypoint_poses=build_waypoint_poses(cfg),
    )
    spawn_pose = Lanelet2Pose(
        lanelet_id=cfg.ego.spawn_lanelet_id,
        s=cfg.ego.spawn_s,
    )
    return ego, spawn_pose, ground_projection


def build_goal_pose(cfg: DictConfig) -> Lanelet2Pose | None:
    """Extract the ego's goal pose from ``ego.goal_lanelet_id`` / ``ego.goal_s``.

    Returns ``None`` when no goal is configured, which does not yet mean the run
    has none: a scenario may derive its own in ``setup()``.
    """
    ego_cfg = cfg.get("ego") or {}
    goal_lanelet_id = ego_cfg.get("goal_lanelet_id")
    if goal_lanelet_id is None:
        return None
    return Lanelet2Pose(
        lanelet_id=int(goal_lanelet_id),
        s=float(ego_cfg.get("goal_s", 0.0)),
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
    lanelet_ids = ego_cfg.get("waypoint_lanelet_ids") or []
    return [Lanelet2Pose(lanelet_id=int(x), s=0.0) for x in lanelet_ids]


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
    spawn_lanelet_id = ego_cfg.get("spawn_lanelet_id")
    if spawn_lanelet_id is None:
        raise RouteError("no ego.spawn_lanelet_id: the route has no start")
    start = Lanelet2Pose(
        lanelet_id=int(spawn_lanelet_id),
        s=float(ego_cfg.get("spawn_s", 0.0)),
    )
    goal = build_goal_pose(cfg)
    scenario_cfg = cfg.get("scenario") or {}
    expected = scenario_cfg.get("expected_route_lanelet_ids") or []
    if goal is None and expected:
        goal = Lanelet2Pose(lanelet_id=int(expected[-1]), s=0.0)
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

    Raises:
        ValueError: If ``autoware.launcher.type`` names an unknown launcher.
    """
    ego_cfg = cfg.get("ego") or {}
    if str(ego_cfg.get("entity", "autopilot")) != "autoware":
        return None
    autoware_cfg = cfg.get("autoware")
    launcher_cfg = None if autoware_cfg is None else autoware_cfg.get("launcher")
    if launcher_cfg is None:
        return None
    settings = _to_dict(launcher_cfg)
    kind = str(settings.pop("type", "none"))
    if kind == "none":
        return None

    key = json.dumps({"type": kind, **settings}, sort_keys=True, default=str)
    cached = _AUTOWARE_LAUNCHERS.get(key)
    if cached is not None:
        return cached

    from autoware_carla_scenario.autoware_stack import (  # noqa: PLC0415
        CommandAutowareLauncher,
        DockerAutowareConfig,
        DockerAutowareLauncher,
    )

    launcher: AutowareLauncher
    if kind == "docker":
        fields = DockerAutowareConfig.__dataclass_fields__
        options = {k: v for k, v in settings.items() if k in fields}
        options["workspace"] = Path(str(options.get("workspace", "~/autoware")))
        for name in ("launch", "colcon_args", "docker_args", "ros_domain_ids"):
            if name in options:
                options[name] = tuple(options[name] or ())
        if "bridge_parameters" in options:
            options["bridge_parameters"] = dict(options["bridge_parameters"] or {})
        launcher = DockerAutowareLauncher(DockerAutowareConfig(**options))
    elif kind == "command":
        launcher = CommandAutowareLauncher(
            tuple(str(arg) for arg in settings.get("command") or ()),
            env={str(k): str(v) for k, v in (settings.get("env") or {}).items()},
            stop_timeout_s=float(settings.get("stop_timeout_s", 20.0)),
            bridge_parameters=dict(settings.get("bridge_parameters") or {}),
        )
    else:
        msg = (
            f"Unknown autoware.launcher.type: {kind!r}. "
            "Expected one of: 'none', 'docker', 'command'."
        )
        raise ValueError(msg)
    _AUTOWARE_LAUNCHERS[key] = launcher
    return launcher


def build_ego_entity(cfg: DictConfig) -> EgoVehicle | None:
    """Build the ego entity selected by ``cfg.ego.entity``.

    Returns ``None`` for ``"autopilot"``, letting the scenario fall back to its default
    :class:`~autoware_carla_scenario.entity.ego.EgoVehicle`. A config with no ``ego``
    group at all selects ``"autopilot"`` too: an injected ``build_scenario_fn`` supplies
    its own :class:`EgoConfig`, so it has no reason to carry the group Hydra would.

    Raises:
        ValueError: If ``cfg.ego.entity`` names an unknown entity.
    """
    ego_cfg = cfg.get("ego") or {}
    entity = str(ego_cfg.get("entity", "autopilot"))

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
            AutowareBridgeConfig,
            AutowareEgoEntity,
            GrpcAutowareBridgeServer,
        )

        autoware_cfg = cfg.get("autoware")
        bridge_cfg = AutowareBridgeConfig(
            **{
                key: value
                for key, value in (
                    _to_dict(autoware_cfg) if autoware_cfg is not None else {}
                ).items()
                if key in AutowareBridgeConfig.__dataclass_fields__
            }
        )
        # autostart=False: in a batch every scenario is built before the first
        # one runs, and two bridges cannot hold the same address at once.  The
        # entity starts this one when its own scenario starts.
        scenario_cfg = cfg.get("scenario") or {}
        return AutowareEgoEntity(
            bridge_cfg,
            bridge=GrpcAutowareBridgeServer(bridge_cfg, autostart=False),
            launcher=build_autoware_launcher(cfg),
            episode_name=str(scenario_cfg.get("name", "")),
        )

    if entity == "carla_driver":
        from autoware_carla_scenario import CarlaDriverEntity  # noqa: PLC0415
        from autoware_carla_scenario.driver import (  # noqa: PLC0415
            ControlConfig,
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
        control = ControlConfig.from_mapping(driver_dict.get("control", {}))
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
        )

        bridge_cfg = AutowareBridgeConfig(
            **checked_options(AutowareBridgeConfig, bridge_dict)
        )
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
        from autoware_carla_scenario.driver import ControlConfig  # noqa: PLC0415
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

        if (cfg.get("server") or {}).get("max_tick_rate_hz") is None:
            # Freshness is judged in simulation time while the nodes publish on
            # the wall clock: unpaced, a 10 Hz node's commands land more than
            # stale_s apart and its source flaps between fresh and stale.
            logger.warning(
                "ego.entity=ros2 without server.max_tick_rate_hz: the run is not "
                "paced to real time, so ROS 2 nodes may look stale; set "
                "server.max_tick_rate_hz=20 (docs/ros2.md, Timing)"
            )
        ros2_dict = _to_dict(ros2_cfg)
        control = ControlConfig.from_mapping(ros2_dict.get("control") or {})
        return Ros2EgoEntity(Ros2EgoConfig.from_mapping(ros2_dict), control)

    msg = (
        f"Unknown ego.entity: {entity!r}. "
        "Expected one of: 'autopilot', 'autoware', 'carla_driver', 'ros2'."
    )
    raise ValueError(msg)


def _legacy_traffic_manager_options(cfg: DictConfig) -> dict[str, int]:
    """Return the TrafficManager options a config predating the group states.

    Only ``traffic_manager.port`` was ever settable that way, and a config that
    does not set it means the framework default.
    """
    legacy = cfg.get("traffic_manager")
    port = None if legacy is None else legacy.get("port")
    return {} if port is None else {"port": int(port)}


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

    # A key left at null is a key the config did not set, not an override of
    # the backend's own default -- the group declares its keys so that plain
    # `traffic.options.x=y` overrides work under Hydra's struct mode, and an
    # interpolation that resolved to nothing leaves the backend's default alone.
    options = {key: value for key, value in config.options.items() if value is not None}

    backend = build_backend(
        config.backend, options, config.vehicles, config.pedestrians, config.bicycles
    )
    logger.info(
        "Traffic backend: %s (registered: %s)", config.backend, available_backends()
    )
    return backend


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
    )
    queue.add(scenario, case=case)
    with queue:
        results = queue.run_all()
    return results[0]


def _optional_float(value: object) -> float | None:
    """Read an optional numeric config value that may be absent or null."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def _fog_density_scale(cfg: DictConfig) -> float | None:
    """``server.fog_density_scale``, or *None* to keep the server's own."""
    return _optional_float((cfg.get("server") or {}).get("fog_density_scale"))


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


def _map_source(cfg: DictConfig) -> str | None:
    """Where the map came from: ``map.source``, else ``map.directory``."""
    map_cfg = cfg.get("map") or {}
    for key in ("source", "directory"):
        value = map_cfg.get(key)
        if value:
            return str(value)
    return None


def _rerun_settings(cfg: DictConfig) -> tuple[bool, bool]:
    """``rerun.enabled`` and ``rerun.strict`` (both off when absent)."""
    rerun_cfg = cfg.get("rerun") or {}
    return bool(rerun_cfg.get("enabled", False)), bool(rerun_cfg.get("strict", False))


def _rerun_lidar_max_points(cfg: DictConfig) -> int | None:
    """``rerun.lidar_max_points`` when ``rerun.lidar`` is set; ``None`` otherwise."""
    rerun_cfg = cfg.get("rerun") or {}
    if not rerun_cfg.get("lidar", False):
        return None
    return int(rerun_cfg.get("lidar_max_points", 0) or 0)


def build_rerun_converter(*cfgs: DictConfig) -> RerunConverter | None:
    """What converts each run's trajectory log into a Rerun recording, when
    any of *cfgs* sets ``rerun.enabled`` (strict when any sets
    ``rerun.strict``); ``None`` otherwise.  With ``rerun.lidar`` it records
    each run's LiDAR sweeps too (the largest ``rerun.lidar_max_points`` of
    those asking, 0 being all), and with ``rerun.cameras`` its camera frames.

    Called before anything runs, so a run asking for recordings without the
    ``rerun`` extra installed is refused at once.

    Raises:
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
        enabled and bool((cfg.get("rerun") or {}).get("cameras", False))
        for cfg, (enabled, _) in zip(cfgs, settings)
    )
    sensors: dict[str, Any] = {}
    if lidar:
        sensors["lidar_max_points"] = 0 if 0 in lidar else max(lidar)
    if cameras:
        sensors["cameras"] = True
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
    from autoware_carla_scenario.driver import DriverClientConfig  # noqa: PLC0415

    for cfg in cfgs:
        t4_cfg = cfg.get("t4") or {}
        if not t4_cfg.get("enabled", False):
            continue
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
        settings = T4Settings.from_driver(
            driver, _to_dict(t4_cfg) if isinstance(t4_cfg, DictConfig) else t4_cfg
        )
        return T4Recorder(settings)
    return None


def canvas_config(*cfgs: DictConfig) -> str | None:
    """The carla_canvas configuration *cfgs* name (``canvas.config``), or ``None``.

    Raises:
        ValueError: They name more than one: a process loads one.
    """
    names = {
        str(value) for cfg in cfgs if (value := (cfg.get("canvas") or {}).get("config"))
    }
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


def rerun_errors(
    results: list[ScenarioResult], conversions: list[RerunConversion]
) -> dict[int, str]:
    """Why each case's Rerun recording failed, by its position in *results*:
    the cases whose ``rerun_recording`` is among the failed *conversions*."""
    failed = {str(c.recording): c.error for c in conversions if c.error is not None}
    return {
        i: failed[result.rerun_recording]
        for i, result in enumerate(results)
        if result.rerun_recording is not None and result.rerun_recording in failed
    }


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


def _is_glob_pattern(value: str) -> bool:
    """Return ``True`` if *value* contains glob metacharacters."""
    return any(ch in value for ch in ("*", "?", "["))


def _is_multirun() -> bool:
    """Return ``True`` when running under Hydra ``--multirun``."""
    return "--multirun" in sys.argv or "-m" in sys.argv


def _extract_scenario_override(argv: list[str]) -> tuple[str | None, list[str]]:
    """Parse *argv* to extract the ``scenario=…`` value.

    Returns:
        A 2-tuple of ``(scenario_value, remaining_overrides)``.
        *scenario_value* is ``None`` when no ``scenario=`` argument is found.
    """
    scenario_value: str | None = None
    remaining: list[str] = []
    for arg in argv[1:]:  # skip argv[0] (program name)
        if arg.startswith("scenario="):
            scenario_value = arg[len("scenario=") :]
        else:
            remaining.append(arg)
    return scenario_value, remaining


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
    from external scenario packages are included alongside the built-ins.
    """
    names: set[str] = set()
    searched_dirs: list[Path] = []
    for conf_dir in get_conf_dirs():
        scenario_dir = conf_dir / "scenario"
        if not scenario_dir.is_dir():
            continue
        searched_dirs.append(scenario_dir)
        for m in scenario_dir.glob(f"{pattern}.yaml"):
            rel = m.relative_to(scenario_dir).with_suffix("")
            # Hydra config names always use forward slashes, so normalise
            # away any OS-specific separator.
            names.add(rel.as_posix())
    if not names:
        print(  # noqa: T201
            f"Error: no scenario configs match pattern '{pattern}' "
            f"under {[str(d) for d in searched_dirs]}"
        )
        sys.exit(1)
    return sorted(names)


def _compose_config(scenario_name: str, overrides: list[str]) -> DictConfig:
    """Build a resolved Hydra config for *scenario_name* using the Compose API."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[f"scenario={scenario_name}", *overrides],
        )
    return cfg


#: One run of a batch: the scenario's config name and the overrides it runs
#: with -- the command line's, after those of its sweep case, if any.
BatchCase = tuple[str, list[str]]


def _has_sweep(cfg: DictConfig) -> bool:
    """Whether *cfg* is a logical scenario: one with a sweep to expand."""
    sweep = OmegaConf.select(cfg, "sweep")
    return sweep is not None and any(
        OmegaConf.select(sweep, key) for key in ("constraints", "route", "odd_sample")
    )


def _expand_batch(scenario_names: list[str], overrides: list[str]) -> list[BatchCase]:
    """Every concrete case of *scenario_names*, in order: each scenario's sweep
    expanded the way the lanelet sweeper expands it (``sweep.constraints``,
    ``sweep.route``, ``sweep.odd_sample``), a scenario without one as itself.

    Scenarios that draw from one ODD go on from each other's draws: a later
    one aims at what the earlier ones left uncovered (``strategy: coverage``)
    and does not draw their values again."""
    from ..odd.sampler import OddDraws  # noqa: PLC0415
    from ..sweeper.expand import expand_config  # noqa: PLC0415 -- loads Lanelet2

    drawn = OddDraws()
    cases: list[BatchCase] = []
    for name in scenario_names:
        expanded = expand_config(
            _compose_config(name, overrides), overrides, drawn=drawn
        )
        logger.info("%s: %d case(s)", name, len(expanded))
        cases.extend((name, case) for case in expanded)
    return cases


def _case_labels(cases: list[BatchCase]) -> list[str]:
    """A name per case for the summary: the scenario's, numbered ``#k`` when
    its sweep gave it more than one case."""
    totals: dict[str, int] = {}
    for name, _ in cases:
        totals[name] = totals.get(name, 0) + 1
    seen: dict[str, int] = {}
    labels: list[str] = []
    for name, _ in cases:
        seen[name] = seen.get(name, 0) + 1
        labels.append(f"{name}#{seen[name]}" if totals[name] > 1 else name)
    return labels


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
        logger.info("  TM port     : %s", cfg.traffic_manager.port)
        goal_pose = build_goal_pose(cfg)
        logger.info(
            "  ego         : %s (%.1f km/h) spawn=lanelet:%d s:%.1f goal=%s",
            cfg.ego.vehicle_type,
            cfg.ego.initial_speed_kmh,
            cfg.ego.spawn_lanelet_id,
            cfg.ego.spawn_s,
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


def _make_batch_output_dir() -> Path:
    """Create a Hydra-style timestamped output directory for batch runs.

    Returns:
        Absolute path to the created directory
        (e.g. ``outputs/2026-03-13/12-00-00/``).
    """
    from datetime import datetime  # noqa: PLC0415

    now = datetime.now()  # noqa: DTZ005
    output_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


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
    cooldown = float(first_cfg.server.get("cooldown_seconds", 0.0))
    cooldown_max_retries = int(first_cfg.server.get("cooldown_max_retries", 0))
    return ScenarioQueue(
        host=first_cfg.server.host,
        port=first_cfg.server.port,
        tm_port=first_cfg.traffic_manager.port,
        xodr_path=map_paths.install_xodr,
        overwrite_xodr=map_paths.overwrite_xodr,
        opendrive_path=map_paths.opendrive_path,
        lanelet2_path=map_paths.lanelet2_path,
        map_name=map_paths.name,
        cooldown_seconds=cooldown,
        cooldown_max_retries=cooldown_max_retries,
        output_dir=output_dir,
        max_tick_rate_hz=_optional_float(first_cfg.server.get("max_tick_rate_hz")),
        projector_type=map_paths.projector_type,
        traffic_backend=build_traffic_backend(first_cfg),
        odd=_odd_spec(first_cfg),
        autoware_launcher=build_autoware_launcher(first_cfg),
        server_extra_args=[*server_extra_args(first_cfg), *(extra_server_args or [])],
        map_source=_map_source(first_cfg),
        rerun=rerun,
        t4=build_t4_recorder(first_cfg),
    )


def _batch_launcher(configs: list[DictConfig]) -> str:
    """``launcher.type`` the batch's configs share: ``local`` or ``slurm``.

    Exits with an error when they differ, name an unknown launcher, or select
    SLURM where there is no ``sbatch``.
    """
    from ..slurm.settings import launcher_type, require_sbatch  # noqa: PLC0415

    try:
        kinds = {launcher_type(cfg) for cfg in configs}
    except ValueError as exc:
        print(f"Error: {exc}")  # noqa: T201
        sys.exit(1)
    if len(kinds) > 1:
        print(  # noqa: T201
            "Error: batch scenarios must share launcher.type; set it on the "
            "command line rather than in a scenario's config"
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
) -> None:
    """Compose configs, build scenarios, and run them in a single queue.

    With *expand* (``--multirun``) each scenario's sweep is expanded and every
    case is run; without it a scenario runs once, as configured. *resume_from*
    is the 1-based case to start at, as for the lanelet sweeper.

    With ``launcher=slurm`` the cases run as SLURM array tasks instead
    (docs/slurm.md); *wait* overrides ``launcher.wait`` (``--no-wait``).
    """
    from ..slurm.settings import (  # noqa: PLC0415
        launcher_requested_in_argv,
        require_sbatch,
    )

    if launcher_requested_in_argv(overrides) == "slurm":
        # Before the sweep is expanded, which can take a while.
        try:
            require_sbatch()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
            sys.exit(2)
    if expand:
        cases = _expand_batch(scenario_names, overrides)
    else:
        cases = [(name, list(overrides)) for name in scenario_names]
    # Labelled before resuming, so a case keeps the name it had in the run
    # being resumed.
    labels = _case_labels(cases)
    # What each case runs with beyond the command line: its sweep case.
    case_overrides = [case[: len(case) - len(overrides)] for _, case in cases]
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
        print("Error: no scenario case left to run")  # noqa: T201
        sys.exit(1)
    configs = [_compose_config(name, case) for name, case in cases]

    if not expand:
        swept = [name for name, cfg in zip(labels, configs) if _has_sweep(cfg)]
        if swept:
            logger.warning(
                "%s ha%s a sweep (sweep.constraints, sweep.route or "
                "sweep.odd_sample) that a batch runs once, as configured; add "
                "--multirun to run every case of it",
                ", ".join(swept),
                "s" if len(swept) == 1 else "ve",
            )

    # Validate all configs share the same map (shared CARLA server constraint).
    map_names = {str(cfg.map.name) for cfg in configs}
    if len(map_names) > 1:
        print(  # noqa: T201
            f"Error: batch scenarios must share the same map, but found: {map_names}"
        )
        sys.exit(1)

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
            "command line rather than in a scenario's config"
        )
        sys.exit(1)

    launcher = _batch_launcher(configs)
    if wait is not None and launcher != "slurm":
        print("Error: --no-wait is for launcher=slurm only")  # noqa: T201
        sys.exit(1)

    # Before anything is built: a batch asking for Rerun recordings without
    # the extra installed is refused at once (RerunUnavailableError).  Under
    # SLURM each task converts its own cases (slurm/worker.py), so this only
    # checks.
    rerun = build_rerun_converter(*configs)
    # Likewise a batch asking for T4 datasets without the `t4` extra.
    build_t4_recorder(*configs)
    if rerun is not None and launcher == "slurm":
        rerun.close()
        rerun = None
    # Under SLURM the file is the nodes' (slurm/worker.py): only one is named.
    if launcher == "slurm":
        canvas_config(*configs)
    else:
        apply_canvas(*configs)

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

    # Batch mode bypasses @hydra.main, so we create the output directory
    # ourselves following Hydra's timestamped convention.
    output_dir = _make_batch_output_dir()

    if launcher == "slurm":
        # The cases run as SLURM array tasks, each on a CARLA of its own, and
        # write their outputs here under the names this batch would give them
        # (docs/slurm.md).
        from ..slurm.submit import submit_batch  # noqa: PLC0415

        scenario_classes = [type(s).__name__ for s in scenarios]
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
            )
        )

    queue = _batch_queue(first_cfg, map_paths, output_dir, rerun=rerun)

    for cfg, scenario in zip(configs, scenarios):
        # The runner's own fail-safe, each scenario's own: without it the
        # queue default (60 s) caps every run, which is shorter than an
        # Autoware stack needs to localize, route and engage.
        queue.add(
            scenario,
            timeout_seconds=float(cfg.scenario.get("timeout_seconds", 60.0)),
            case=_odd_case(cfg),
        )
    # From here the queue alone holds the scenarios, and lets each go once it
    # has run: a batch reports results, and a finished scenario kept to the
    # end keeps everything its run built (its entities, conditions, the ego's
    # driver), case after case.
    del scenarios, scenario

    logger.info("All %d scenario(s) built. Starting execution...", len(labels))

    try:
        with queue:
            # A case that cannot run (a spawn point without ground, say) is
            # recorded as errored, and the rest of the batch still runs.
            results = queue.run_all(release=True, continue_on_error=True)
            errored = queue.errored_indices()
    finally:
        # Each log was converted as its case ended; wait for the last ones.
        conversions = rerun.close() if rerun is not None else []

    all_passed = _print_summary(
        labels,
        results,
        output_dir=output_dir,
        overrides=case_overrides if expand else None,
        errored=errored,
        rerun_errors=rerun_errors(results, conversions),
    )
    sys.exit(0 if all_passed else 1)


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
    if build_scenario_fn is not None:
        ego, scenario = build_scenario_fn(cfg)
        _apply_ego_config(cfg, scenario)
        add_background_traffic(cfg, scenario)
        add_environment(cfg, scenario)
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
    # Built once: an Autoware ego holds a bridge server, and two of those cannot
    # hold the same address.
    ego_entity = build_ego_entity(cfg)
    if ego_entity is not None:
        scenario.ego_entity = ego_entity
    add_background_traffic(cfg, scenario)
    add_environment(cfg, scenario)
    return ego, scenario


def add_environment(cfg: DictConfig, scenario: BaseScenario) -> None:
    """Set the weather and the sun the config's ``environment`` names, if any.

    An :class:`~autoware_carla_scenario.EnvironmentAction` registered for
    initialization, so the world is in it before the run starts.  It is
    registered before the scenario's own ``setup()`` runs, so a scenario that
    sets the weather itself has the last word.  Nothing is registered when
    every field is ``null`` (the default).
    """
    env_cfg = cfg.get("environment")
    if env_cfg is None:
        return
    settings: dict[str, Any] = {
        str(key): float(value)
        for key, value in _to_dict(env_cfg).items()
        if value is not None
    }
    if not settings:
        return
    from autoware_carla_scenario import EnvironmentAction  # noqa: PLC0415

    scenario.register_init(EnvironmentAction(**settings, label="config.environment"))


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
    """
    background = cfg.get("background_traffic")
    if background is None or not background.get("enabled", False):
        return
    settings = _to_dict(background)
    source = settings.get("source") or {}
    sink = settings.get("sink") or {}
    seed = int(settings.get("seed", 0))

    def until(seconds: object, label: str) -> ElapsedTimeCondition | None:
        if seconds is None:
            return None
        return ElapsedTimeCondition(float(str(seconds)), label=label)

    max_vehicles = source.get("max_vehicles")
    constraints = source.get("constraints") or []

    def source_action(
        *, initial: int, rate: float, seed: int, label: str
    ) -> TrafficSourceAction:
        return TrafficSourceAction(
            constraints,
            initial_vehicles=initial,
            vehicles_per_minute=rate,
            max_vehicles=None if max_vehicles is None else int(max_vehicles),
            speed_kmh=float(source.get("speed_kmh", 30.0)),
            min_gap_m=float(source.get("min_gap_m", 15.0)),
            blueprint=source.get("blueprint"),
            seed=seed,
            label=label,
            until=(
                until(source.get("stop_after_seconds"), "background_source_ends")
                if rate > 0
                else None
            ),
        )

    initial = int(source.get("initial_vehicles", 0))
    rate = float(source.get("vehicles_per_minute", 0.0))
    if constraints and initial > 0:
        scenario.register_init(
            source_action(
                initial=initial, rate=0.0, seed=seed, label="background_initial"
            )
        )
    if constraints and rate > 0:
        scenario.register_pre_tick(
            source_action(
                initial=0, rate=rate, seed=seed + 1, label="background_source"
            )
        )
    if sink.get("constraints"):
        scenario.register_pre_tick(
            TrafficSinkAction(
                sink["constraints"],
                label="background_sink",
                until=until(sink.get("stop_after_seconds"), "background_sink_ends"),
            )
        )
    logger.info(
        "Background traffic: %d at start, %.1f/min after, sink %s",
        initial if constraints else 0,
        rate if constraints else 0.0,
        "on" if sink.get("constraints") else "off",
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
    # Before anything is built: refused at once without the extras.
    rerun = build_rerun_converter(cfg)
    build_t4_recorder(cfg)
    try:
        apply_canvas(cfg)
        return _run_scenario(cfg, rerun, build_scenario_fn=build_scenario_fn)
    finally:
        _finish_rerun(rerun)


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
    cooldown = float(cfg.server.get("cooldown_seconds", 0.0))
    cooldown_max_retries = int(cfg.server.get("cooldown_max_retries", 0))

    # Retrieve the Hydra output directory (works regardless of
    # ``hydra.job.chdir`` which defaults to False since Hydra 1.2).
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    result = run_scenario_with_queue(
        scenario,
        host=cfg.server.host,
        port=cfg.server.port,
        tm_port=cfg.traffic_manager.port,
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
        timeout_seconds=float(cfg.scenario.get("timeout_seconds", 60.0)),
        max_tick_rate_hz=_optional_float(cfg.server.get("max_tick_rate_hz")),
        projector_type=map_paths.projector_type,
        traffic_backend=build_traffic_backend(cfg),
        odd=_odd_spec(cfg),
        autoware_launcher=build_autoware_launcher(cfg),
        case=_odd_case(cfg),
        server_extra_args=server_extra_args(cfg),
        map_source=_map_source(cfg),
        rerun=rerun,
        t4=build_t4_recorder(cfg),
    )

    status = "PASSED" if result.passed else "FAILED"
    print(f"{status}: {result.message} ({result.elapsed_seconds:.2f}s)")  # noqa: T201
    scenario_name = type(scenario).__name__
    json_path = (output_dir / f"{scenario_name}_result.json").resolve()
    print(f"Result JSON: {json_path}")  # noqa: T201
    return result


@hydra.main(version_base=None, config_path="conf", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """Hydra entry point that dispatches to the selected scenario."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        result = run_scenario(cfg)
    except (ScenarioTypeError, RerunUnavailableError, T4UnavailableError) as exc:
        if _is_multirun():
            raise
        print(exc, file=sys.stderr)  # noqa: T201
        sys.exit(2)
    # Only exit for single-run mode. In --multirun, Hydra calls this
    # function repeatedly; sys.exit() would kill the entire sweep.
    if not _is_multirun():
        sys.exit(0 if result.passed else 1)


def _extract_resume_from(argv: list[str]) -> tuple[int, list[str]]:
    """Extract ``--resume-from N`` from *argv* and return the value and cleaned argv.

    Returns:
        A 2-tuple of ``(resume_from, remaining_argv)``.
        *resume_from* is 0 when the flag is absent.
    """
    resume_from = 0
    remaining: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--resume-from":
            if i + 1 < len(argv):
                try:
                    resume_from = int(argv[i + 1])
                except ValueError:
                    print(  # noqa: T201
                        f"Error: --resume-from requires an integer, got '{argv[i + 1]}'"
                    )
                    sys.exit(1)
                skip_next = True
            else:
                print("Error: --resume-from requires a value")  # noqa: T201
                sys.exit(1)
        elif arg.startswith("--resume-from="):
            try:
                resume_from = int(arg.split("=", 1)[1])
            except ValueError:
                print(  # noqa: T201
                    f"Error: --resume-from requires an integer, got '{arg.split('=', 1)[1]}'"
                )
                sys.exit(1)
        else:
            remaining.append(arg)
    return resume_from, remaining


def main() -> None:
    """CLI entry point: detect glob patterns and dispatch accordingly."""
    # First thing in main(), before the run starts its own threads (see
    # _cap_malloc_arenas). Threads the imports start have not allocated yet,
    # so they hold no arenas: glibc ignores a cap set once it has more.
    _cap_malloc_arenas()
    # Discover external scenario packages (entry-point plugins) before we touch
    # the config search path or the scenario registry, so their scenarios and
    # conf dirs are available to both the glob and single-run code paths.
    load_scenario_plugins()

    # Extract --resume-from before Hydra sees the argv.
    resume_from, cleaned_argv = _extract_resume_from(sys.argv)
    sys.argv = cleaned_argv

    # Pass via environment variable so the sweeper can read it without
    # going through Hydra's CLI parser (which rejects unknown overrides).
    if resume_from > 0:
        os.environ["SWEEP_RESUME_FROM"] = str(resume_from)

    # --no-wait: a launcher=slurm batch is submitted, and this exits.
    no_wait = "--no-wait" in sys.argv
    if no_wait:
        sys.argv = [arg for arg in sys.argv if arg != "--no-wait"]

    from ..slurm.settings import launcher_requested_in_argv  # noqa: PLC0415

    scenario_value, remaining = _extract_scenario_override(sys.argv)
    # A SLURM launcher runs a batch, so a single scenario selected with it runs
    # as one too: --multirun's cases, or the scenario once, as SLURM tasks.
    slurm = launcher_requested_in_argv(remaining) == "slurm"
    if scenario_value is not None and (_is_glob_pattern(scenario_value) or slurm):
        # --multirun runs every case of each scenario's sweep in the batch's
        # one queue; the sweeper it would pick is the batch itself.
        expand = _is_multirun()
        remaining = [
            arg
            for arg in remaining
            if arg not in ("--multirun", "-m") and not arg.startswith("hydra/sweeper=")
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
        try:
            if no_wait:
                run_batch(
                    scenario_names,
                    remaining,
                    expand=expand,
                    resume_from=resume_from,
                    wait=False,
                )
            else:
                run_batch(
                    scenario_names, remaining, expand=expand, resume_from=resume_from
                )
        except (ScenarioTypeError, RerunUnavailableError, T4UnavailableError) as exc:
            print(exc, file=sys.stderr)  # noqa: T201
            sys.exit(2)
    else:
        if no_wait:
            print("Error: --no-wait is for launcher=slurm only")  # noqa: T201
            sys.exit(1)
        # Single run goes through @hydra.main.  External conf dirs are added to
        # the search path by AutowareScenarioSearchPathPlugin (discovered via
        # the hydra_plugins namespace), so nothing needs threading here.
        _hydra_main()


if __name__ == "__main__":
    main()
