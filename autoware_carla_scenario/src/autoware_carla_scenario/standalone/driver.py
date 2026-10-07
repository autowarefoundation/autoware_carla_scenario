"""The program a standalone scenario binary is compiled from.

The static check's driver (:mod:`..typecheck.driver`) builds a scenario and
calls ``setup()`` so Codon checks what that reaches; this one builds the same
scenario from the same resolved config and hands it to the runtime's runner
(``codon/autoware_carla_scenario/runner.codon``), which runs it against a
CARLA server. Everything the Python runner reads from the composed config is
rendered into the program as a literal -- the scenario's own values exactly
as the checker renders them, and the ego, spawn pose, ground projection, map
and server settings :func:`~autoware_carla_scenario.examples.run.build_ego_and_spawn`
and :func:`~autoware_carla_scenario.examples.run.run_scenario` take from it --
so the binary needs neither Hydra nor the YAML it was built from. The server
address, ports, output directory and timeout can still be overridden on its
command line (``--help``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..typecheck.driver import (
    _UNKNOWN,
    _all_fields_have_defaults,
    _at,
    _field_hints,
    _Imports,
    _render,
)

__all__ = ["MAIN_MODULE", "RunSettings", "render_main"]

#: File name of the generated program in the build workspace.
MAIN_MODULE = "_acs_main.py"


@dataclass(frozen=True)
class Pose:
    """A Lanelet2 pose, as the config names one."""

    lanelet_id: int
    s: float


@dataclass(frozen=True)
class RunSettings:
    """What the binary takes from the composed config besides ``scenario.*``."""

    scenario_name: str  # type(scenario).__name__: the output files' stem
    vehicle_type: str
    initial_speed_kmh: float
    spawn_retry_max_count: int
    spawn_retry_t_step: float
    spawn_retry_z_step: float
    spawn_pose: Pose
    goal_pose: Pose | None
    waypoint_poses: tuple[Pose, ...]
    ray_distance_upper: float
    ray_distance_lower: float
    host: str
    port: int
    tm_port: int
    timeout_seconds: float
    max_tick_rate_hz: float | None
    map_name: str
    overwrite_xodr: bool
    xodr_env_var: str


def _pose(p: Pose) -> str:
    return f"Lanelet2Pose({p.lanelet_id!r}, {float(p.s)!r})"


def render_main(
    scenario_cls: type,
    config_cls: type,
    scenario_dict: dict[str, Any],
    run: RunSettings,
) -> str:
    """The program that builds *scenario_cls* and runs it with *run*."""
    imports = _Imports()
    scenario_name = imports.name(scenario_cls)
    config_name = imports.name(config_cls)
    hints = _field_hints(config_cls)
    try:
        built: Any = config_cls(**copy.deepcopy(scenario_dict))
    except Exception:  # noqa: BLE001 - the program's constructor call reports it
        built = _UNKNOWN

    by_assignment = _all_fields_have_defaults(config_cls)
    lines = [
        "def _acs_build_scenario():",
        f"    config = {config_name}()"
        if by_assignment
        else f"    config = {config_name}(",
    ]
    for key, value in scenario_dict.items():
        rendered = _render(value, hints.get(key, Any), imports, _at(built, key))
        lines.append(
            f"    config.{key} = {rendered}"
            if by_assignment
            else f"        {key}={rendered},"
        )
    if not by_assignment:
        lines.append("    )")
    goal = "None" if run.goal_pose is None else _pose(run.goal_pose)
    waypoints = ", ".join(_pose(p) for p in run.waypoint_poses)
    lines += [
        "    ego = EgoConfig(",
        "        SpawnTransform(carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))),",
        f"        vehicle_type={run.vehicle_type!r},",
        f"        initial_speed_kmh={float(run.initial_speed_kmh)!r},",
        f"        spawn_retry_max_count={int(run.spawn_retry_max_count)!r},",
        f"        spawn_retry_t_step={float(run.spawn_retry_t_step)!r},",
        f"        spawn_retry_z_step={float(run.spawn_retry_z_step)!r},",
        f"        goal_pose={goal},",
        f"        waypoint_poses=List[Lanelet2Pose]([{waypoints}]),",
        "    )",
        f"    return {scenario_name}(",
        "        ego,",
        "        config=config,",
        f"        spawn_pose={_pose(run.spawn_pose)},",
        "        ground_projection=GroundProjectionConfig(",
        f"            ray_distance_upper={float(run.ray_distance_upper)!r},",
        f"            ray_distance_lower={float(run.ray_distance_lower)!r},",
        "        ),",
        "    )",
        "",
        "",
        "def _acs_options() -> RunOptions:",
        "    opts = RunOptions()",
        f"    opts.scenario_name = {run.scenario_name!r}",
        f"    opts.host = {run.host!r}",
        f"    opts.port = {int(run.port)!r}",
        f"    opts.tm_port = {int(run.tm_port)!r}",
        f"    opts.timeout_seconds = {float(run.timeout_seconds)!r}",
        f"    opts.max_tick_rate_hz = {float(run.max_tick_rate_hz or 0.0)!r}",
        f"    opts.map_name = {run.map_name!r}",
        f"    opts.overwrite_xodr = {bool(run.overwrite_xodr)!r}",
        f"    opts.xodr_env_var = {run.xodr_env_var!r}",
        "    return opts",
        "",
        "",
        "sys.exit(main(_acs_build_scenario, _acs_options()))",
    ]
    head = [
        "import _acs_env  # first: before typesafe_carla loads its native library",
        "import sys",
        "import typesafe_carla.carla as carla",
        "from autoware_carla_scenario import (EgoConfig, GroundProjectionConfig, Lanelet2Pose,",
        "                                     SpawnTransform)",
        "from autoware_carla_scenario.runner import RunOptions",
        "from autoware_carla_scenario.main import main",
        *imports.lines,
        "",
        "",
    ]
    return "\n".join(head + lines) + "\n"
