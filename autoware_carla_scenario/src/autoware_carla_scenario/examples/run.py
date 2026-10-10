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
    uv run scenario scenario='intersection_passing/left_*'

    # Select a different map
    uv run scenario scenario=intersection_passing/straight map=nishishinjuku

    # Override individual parameters
    uv run scenario scenario=intersection_passing/straight scenario.timeout_seconds=15.0

    # Override server connection
    uv run scenario scenario=intersection_passing/straight server.host=192.168.1.100 server.port=3000
"""

from __future__ import annotations

import json
import logging
import os
import sys
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
from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.constants import DEFAULT_TM_PORT
from autoware_carla_scenario.maps import resolve_map_paths
from autoware_carla_scenario.odd.route import PlannedRoute, RouteError
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
        launcher = DockerAutowareLauncher(DockerAutowareConfig(**options))
    elif kind == "command":
        launcher = CommandAutowareLauncher(
            tuple(str(arg) for arg in settings.get("command") or ()),
            env={str(k): str(v) for k, v in (settings.get("env") or {}).items()},
            stop_timeout_s=float(settings.get("stop_timeout_s", 20.0)),
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

        driver_dict = _to_dict(driver_cfg)
        control = ControlConfig.from_mapping(driver_dict.get("control", {}))
        return CarlaDriverEntity(DriverClientConfig.from_mapping(driver_dict), control)

    msg = (
        f"Unknown ego.entity: {entity!r}. "
        "Expected one of: 'autopilot', 'autoware', 'carla_driver'."
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

    backend = build_backend(config.backend, options)
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
) -> ScenarioResult:
    """Run a single pre-built scenario using :class:`ScenarioQueue`.

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
    )
    queue.add(scenario)
    with queue:
        results = queue.run_all()
    return results[0]


def _optional_float(value: object) -> float | None:
    """Read an optional numeric config value that may be absent or null."""
    return None if value is None else float(value)  # type: ignore[arg-type]


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


def _write_batch_result_json(
    names: list[str],
    results: list[ScenarioResult],
    output_dir: Path,
) -> Path:
    """Write a machine-readable JSON summary to *output_dir* and return the path."""
    import json  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_results.json"
    json_results = [
        {"scenario": name, **result.to_dict()} for name, result in zip(names, results)
    ]
    json_path.write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_path


def _print_summary(
    names: list[str],
    results: list[ScenarioResult],
    output_dir: Path = Path("scenario_outputs"),
) -> bool:
    """Print a formatted result table and return ``True`` if all passed.

    A machine-readable JSON file is also written to *output_dir*.
    """
    sep = "=" * 60
    thin = "-" * 60
    print(f"\n{sep}")  # noqa: T201
    print("Batch Scenario Results")  # noqa: T201
    print(sep)  # noqa: T201
    for name, result in zip(names, results):
        tag = "PASS" if result.passed else "FAIL"
        print(f"  [{tag}] {name} ({result.elapsed_seconds:.1f}s)")  # noqa: T201
        if not result.passed:
            for line in result.message.splitlines():
                print(f"         {line}")  # noqa: T201
        if result.condition_statuses:
            max_label_len = max(len(cs.label) for cs in result.condition_statuses)
            for cs in result.condition_statuses:
                mark = "OK" if cs.satisfied else "NG"
                padded = cs.label.ljust(max_label_len)
                print(f"    [{mark}] {padded} : {cs.message}")  # noqa: T201

    json_path = _write_batch_result_json(names, results, output_dir)
    print(thin)  # noqa: T201
    print(f"Result JSON: {json_path}")  # noqa: T201

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(thin)  # noqa: T201
    print(f"{passed}/{total} scenarios passed")  # noqa: T201
    print(sep)  # noqa: T201
    return passed == total


def _log_batch_plan(
    scenario_names: list[str],
    configs: list[DictConfig],
    overrides: list[str],
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
        logger.info("[%d/%d] %s", i, len(scenario_names), name)
        logger.info("  config file : %s", yaml_path)
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


def run_batch(
    scenario_names: list[str],
    overrides: list[str],
    *,
    build_scenario_fn: BuildScenarioFn | None = None,
) -> None:
    """Compose configs, build scenarios, and run them in a single queue."""
    configs = [_compose_config(name, overrides) for name in scenario_names]

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

    # Log detailed execution plan before building anything.
    _log_batch_plan(scenario_names, configs, overrides)

    # Build (and statically check) every scenario before anything reads the
    # config values or fetches the map, so a bad value is refused first.
    scenarios: list[BaseScenario] = []
    for i, (name, cfg) in enumerate(zip(scenario_names, configs), 1):
        logger.info("Building scenario [%d/%d]: %s", i, len(scenario_names), name)
        _ego, scenario = build_scenario(cfg, build_scenario_fn=build_scenario_fn)
        scenarios.append(scenario)

    first_cfg = configs[0]
    # The ODD every run is measured against, checked once when it is Python.
    check_odd(_odd_spec(first_cfg), typecheck_mode(first_cfg))

    map_paths = resolve_map_paths(first_cfg.map)

    cooldown = float(first_cfg.server.get("cooldown_seconds", 0.0))
    cooldown_max_retries = int(first_cfg.server.get("cooldown_max_retries", 0))

    # Batch mode bypasses @hydra.main, so we create the output directory
    # ourselves following Hydra's timestamped convention.
    output_dir = _make_batch_output_dir()

    queue = ScenarioQueue(
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
    )

    for cfg, scenario in zip(configs, scenarios):
        # The runner's own fail-safe, each scenario's own: without it the
        # queue default (60 s) caps every run, which is shorter than an
        # Autoware stack needs to localize, route and engage.
        queue.add(
            scenario,
            timeout_seconds=float(cfg.scenario.get("timeout_seconds", 60.0)),
        )

    logger.info("All %d scenario(s) built. Starting execution...", len(scenario_names))

    with queue:
        results = queue.run_all()

    all_passed = _print_summary(scenario_names, results, output_dir=output_dir)
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
    except ScenarioTypeError as exc:
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

    scenario_value, remaining = _extract_scenario_override(sys.argv)
    if scenario_value is not None and _is_glob_pattern(scenario_value):
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
        scenario_names = _resolve_scenario_glob(scenario_value)
        logger.info(
            "Glob matched %d scenario(s): %s",
            len(scenario_names),
            scenario_names,
        )
        try:
            run_batch(scenario_names, remaining)
        except ScenarioTypeError as exc:
            print(exc, file=sys.stderr)  # noqa: T201
            sys.exit(2)
    else:
        # Single run goes through @hydra.main.  External conf dirs are added to
        # the search path by AutowareScenarioSearchPathPlugin (discovered via
        # the hydra_plugins namespace), so nothing needs threading here.
        _hydra_main()


if __name__ == "__main__":
    main()
