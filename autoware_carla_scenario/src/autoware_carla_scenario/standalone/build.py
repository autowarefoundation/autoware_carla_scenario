"""Compile a registered scenario into a standalone binary (docs/standalone.md).

:func:`build_standalone` composes the scenario's config exactly as
``uv run scenario`` does, and produces a bundle that runs it with nothing but
a CARLA server::

    <out>/bin/<scenario>     the native binary
    <out>/lib/               libtypesafe_carla_ffi.so and Codon's runtime
    <out>/share/map.acsmap   the map, baked (bake.py)
    <out>/share/map.xodr     the OpenDRIVE, for a map that installs it

The build reuses the static check (docs/typecheck.md) end to end: the same
modules of the scenario's package are collected and rewritten for Codon
(:mod:`..typecheck.check`, :mod:`..typecheck.transform`), and the config is
rendered the same way (:mod:`..typecheck.driver`). What differs is what they
are compiled against: the runtime in ``codon/``, whose functions run, instead
of the model, whose functions only type-check. A scenario that uses part of
the API the runtime does not implement yet fails to build, at the line that
uses it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..typecheck.check import (
    Diagnostic,
    _collect,
    _parse_output,
    _Sources,
    _Workspace,
    find_supported_codon,
)
from ..typecheck.toolchain import Toolchain, codon_environment, codon_path_dir
from ..typecheck.transform import class_declarations, undeclared_attributes
from .bake import BakeStats, bake_map
from .driver import MAIN_MODULE, Pose, RunSettings, render_main
from .transform import transform_for_runtime

__all__ = [
    "BuildError",
    "BuildResult",
    "RUNTIME_LIBRARIES",
    "set_rpath",
    "BuildPlan",
    "build_bundle",
    "build_standalone",
    "plan_build",
    "compile_program",
    "prepare_runtime_workspace",
    "runtime_dir",
]

#: Codon's runtime libraries a binary links (libcodonrt and what it needs).
RUNTIME_LIBRARIES = (
    "libcodonrt.so",
    "libomp.so",
    "libgfortran.so.5",
    "libquadmath.so.0",
    "libgcc_s.so.1",
)
DEFAULT_TIMEOUT_SECONDS = 1200.0
#: Where a bundle's binary finds its libraries: <bundle>/lib.
BUNDLE_RPATH = "$ORIGIN/../lib"


def runtime_dir() -> Path:
    """The Codon runtime library (the workspace's base)."""
    return Path(__file__).resolve().parent / "codon"


class BuildError(RuntimeError):
    """The scenario cannot be built into a standalone binary."""

    def __init__(
        self, message: str, diagnostics: list[Diagnostic] | None = None
    ) -> None:
        self.diagnostics = list(diagnostics or [])
        text = message
        for d in self.diagnostics:
            text += "\n  " + d.format().replace("\n", "\n  ")
        super().__init__(text)


@dataclass
class BuildResult:
    """What :func:`build_standalone` produced."""

    out_dir: Path
    executable: Path
    map_data: BakeStats
    compile_seconds: float
    libraries: list[Path] = field(default_factory=list)


def _runtime_declarations() -> dict[str, set[str]]:
    import ast  # noqa: PLC0415

    classes: dict[str, tuple[list[str], set[str]]] = {}
    for path in sorted(runtime_dir().rglob("*.codon")):
        try:
            tree = ast.parse(path.read_text())
        except (
            SyntaxError
        ):  # Codon-only syntax (`from C import ...`): no scenario class
            continue
        classes.update(class_declarations(tree))

    def closure(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        if name not in classes or name in seen:
            return set()
        bases, own = classes[name]
        out = set(own)
        for base in bases:
            out |= closure(base.rsplit(".", 1)[-1], seen | {name})
        return out

    return {name: closure(name) for name in classes}


def prepare_runtime_workspace(root: Path, codon_path: Path | None = None) -> None:
    """Fill *root* with the runtime and typesafe_carla's Codon library: a
    directory to compile a program against (as CODON_PATH)."""
    codon_path = codon_path or codon_path_dir()
    shutil.copytree(runtime_dir(), root, dirs_exist_ok=True)
    for entry in codon_path.iterdir():
        (root / entry.name).symlink_to(entry.resolve())


def _build_workspace(
    root: Path, sources: _Sources, program: str, codon_path: Path
) -> tuple[_Workspace, list[Diagnostic]]:
    """typecheck.check._build_workspace, on the runtime instead of the model."""
    prepare_runtime_workspace(root, codon_path)
    ws = _Workspace(root, None)
    problems: list[Diagnostic] = []
    declared = _runtime_declarations()
    for module in sources.modules.values():
        for cls, (_bases, own) in class_declarations(module.tree).items():
            declared.setdefault(cls, set()).update(own)
    for name, module in sorted(sources.modules.items()):
        rel = Path(*name.split("."))
        rel = rel / "__init__.py" if module.is_package else rel.with_suffix(".py")
        for attr in undeclared_attributes(module.tree, declared):
            problems.append(Diagnostic(attr.message(), str(module.path), attr.lineno))
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(transform_for_runtime(module.text))
        ws.files[rel.as_posix()] = str(module.path)
    for rel_name in list(ws.files):
        parent = Path(rel_name).parent
        while parent != Path("."):
            if (
                not (root / parent / "__init__.py").exists()
                and not (root / parent / "__init__.codon").exists()
            ):
                (root / parent / "__init__.codon").write_text("")
            parent = parent.parent
    (root / MAIN_MODULE).write_text(program)
    return ws, problems


def compile_program(
    workspace: Path,
    program: str,
    output: Path,
    *,
    toolchain: Toolchain,
    rpath: str | None = BUNDLE_RPATH,
    release: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """``codon build -exe`` *program* (a file in *workspace*) into *output*."""
    args = [str(toolchain.executable), "build", "-exe"]
    if release:
        args.append("-release")
    if rpath is not None:
        # DT_RPATH, not RUNPATH: dlopen() calls Codon's runtime makes (the
        # CARLA client library) search it too (typesafe_carla.cli.add_rpath).
        args.append(f"--linker-flags=-Wl,--disable-new-dtags,-rpath,{rpath}")
    args += ["-o", str(output), program]
    env = codon_environment(toolchain, workspace)
    return subprocess.run(  # noqa: S603 - a fixed compiler invocation
        args,
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def set_rpath(executable: Path, rpath: str) -> str:
    """Replace *executable*'s DT_RPATH with *rpath*, in place; return the old one.

    ``codon build`` puts its own library directories (on the build machine)
    ahead of the RPATH it is given, so a bundle would load Codon's runtime
    from wherever the toolchain was installed when the build ran, if that
    path exists where it runs. The string is rewritten where it is, so the new
    one must not be longer than the old.
    """
    import struct  # noqa: PLC0415

    data = bytearray(executable.read_bytes())
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise BuildError(f"{executable}: not a 64-bit little-endian ELF file")
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        for i in range(e_shnum)
    ]
    dynamic = next((s for s in sections if s[1] == 6), None)  # SHT_DYNAMIC
    if dynamic is None:
        raise BuildError(f"{executable}: no dynamic section")
    dynstr = sections[dynamic[6]]  # sh_link: its string table
    for off in range(dynamic[4], dynamic[4] + dynamic[5], 16):
        tag, value = struct.unpack_from("<qQ", data, off)
        if tag == 0:
            break
        if tag in (15, 29):  # DT_RPATH, DT_RUNPATH
            start = dynstr[4] + value
            end = data.index(b"\0", start)
            old = data[start:end].decode()
            new = rpath.encode()
            if len(new) > end - start:
                raise BuildError(
                    f"{executable}: RPATH {rpath!r} is longer than {old!r}"
                )
            data[start : start + len(new)] = new
            data[start + len(new)] = 0
            executable.write_bytes(bytes(data))
            return old
    raise BuildError(f"{executable}: no RPATH to rewrite")


def _setting(cfg: Any, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        node = node.get(part) if hasattr(node, "get") else getattr(node, part, None)
    return default if node is None else node


def _run_settings(cfg: Any, scenario_cls: type, map_paths: Any) -> RunSettings:
    """What the Python runner takes from *cfg*, refusing what the runtime cannot run."""
    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios
    from ..maps.opendrive import map_asset_env_var  # noqa: PLC0415
    from ..traffic.traffic_manager import TrafficManagerBackend  # noqa: PLC0415

    entity = str(_setting(cfg, "ego.entity", "autopilot"))
    if entity != "autopilot":
        raise BuildError(
            f"ego.entity={entity}: the standalone runtime runs the autopilot ego only"
        )
    if bool(_setting(cfg, "background_traffic.enabled", False)):
        raise BuildError(
            "background_traffic.enabled: not supported by the standalone runtime yet"
        )
    backend = run.build_traffic_backend(cfg)
    if not isinstance(backend, TrafficManagerBackend):
        raise BuildError(
            f"traffic={backend.name}: the standalone runtime drives traffic with "
            "TrafficManager only"
        )
    ego, spawn_pose, ground_projection = run.build_ego_and_spawn(cfg)
    map_name = map_paths.name or str(_setting(cfg, "map.name", "") or "")
    goal = ego.goal_pose
    return RunSettings(
        scenario_name=scenario_cls.__name__,
        vehicle_type=ego.vehicle_type,
        initial_speed_kmh=ego.initial_speed_kmh,
        spawn_retry_max_count=ego.spawn_retry_max_count,
        spawn_retry_t_step=ego.spawn_retry_t_step,
        spawn_retry_z_step=ego.spawn_retry_z_step,
        spawn_pose=Pose(int(spawn_pose.lanelet_id), float(spawn_pose.s)),
        goal_pose=None if goal is None else Pose(int(goal.lanelet_id), float(goal.s)),
        waypoint_poses=tuple(
            Pose(int(p.lanelet_id), float(p.s)) for p in ego.waypoint_poses
        ),
        ray_distance_upper=ground_projection.ray_distance_upper,
        ray_distance_lower=ground_projection.ray_distance_lower,
        host=str(_setting(cfg, "server.host", "localhost")),
        port=int(_setting(cfg, "server.port", 2000)),
        tm_port=backend.port,
        timeout_seconds=float(_setting(cfg, "scenario.timeout_seconds", 60.0)),
        max_tick_rate_hz=run._optional_float(_setting(cfg, "server.max_tick_rate_hz")),
        map_name=map_name,
        overwrite_xodr=bool(_setting(cfg, "map.overwrite_xodr", False))
        and map_paths.install_xodr is not None,
        xodr_env_var=map_asset_env_var(map_name) if map_name else "",
    )


def _cached_bake(
    xodr: Path, lanelet2: Path, dest: Path, projector_type: str | None
) -> BakeStats:
    """bake_map(), cached by the content of its inputs (a sweep builds one map many times)."""
    import hashlib  # noqa: PLC0415

    from .bake import FORMAT_VERSION  # noqa: PLC0415

    digest = hashlib.sha256(f"{FORMAT_VERSION}:{projector_type}".encode())
    digest.update(xodr.read_bytes())
    digest.update(lanelet2.read_bytes())
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "autoware_carla_scenario"
        / "standalone"
    )
    cached = cache_root / f"{digest.hexdigest()}.acsmap"
    if not cached.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(f".{os.getpid()}.tmp")
        bake_map(xodr, lanelet2, partial, projector_type=projector_type)
        partial.replace(cached)  # atomic: concurrent builds never read half a file
    shutil.copyfile(cached, dest)
    text = dest.read_text(encoding="utf-8")
    return BakeStats(
        dest,
        sum(1 for line in text.splitlines() if line.startswith("lanelet ")),
        sum(1 for line in text.splitlines() if line.startswith("road ")),
        len(text.encode()),
    )


def build_bundle(
    scenario_cls: type,
    config_cls: type,
    scenario_dict: dict[str, Any],
    settings: RunSettings,
    *,
    xodr: Path,
    lanelet2: Path,
    out_dir: Path,
    executable_name: str,
    projector_type: str | None = None,
    toolchain: Toolchain | None = None,
    release: bool = True,
    keep_workspace: Path | None = None,
) -> BuildResult:
    """Compile *scenario_cls* built with ``config_cls(**scenario_dict)`` into a bundle.

    Raises:
        BuildError: The scenario does not build against the runtime.
    """
    name = f"{scenario_cls.__module__}.{scenario_cls.__qualname__}"
    tc = toolchain or find_supported_codon()
    codon_path = codon_path_dir()
    out = Path(out_dir)
    for sub in ("bin", "lib", "share"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    roots = sorted({scenario_cls.__module__, config_cls.__module__})
    sources = _collect(roots, tc, codon_path)
    if sources.problems:
        raise BuildError(f"{name}: cannot be compiled", sources.problems)
    program = render_main(scenario_cls, config_cls, scenario_dict, settings)

    executable = out / "bin" / executable_name
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="acs-standalone-") as tmp:
        root = Path(keep_workspace) if keep_workspace is not None else Path(tmp)
        if keep_workspace is not None and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        ws, problems = _build_workspace(root, sources, program, codon_path)
        if problems:
            raise BuildError(f"{name}: cannot be compiled", problems)
        proc = compile_program(
            root, MAIN_MODULE, executable.resolve(), toolchain=tc, release=release
        )
        if proc.returncode != 0:
            output = proc.stdout + proc.stderr
            diagnostics = _parse_output(output, ws) or [
                Diagnostic(f"codon build failed: {output.strip().splitlines()[-1:]}")
            ]
            raise BuildError(f"{name}: does not build", diagnostics)
    seconds = time.monotonic() - started

    stats = _cached_bake(xodr, lanelet2, out / "share" / "map.acsmap", projector_type)
    shutil.copyfile(xodr, out / "share" / "map.xodr")

    from typesafe_carla import paths as tsc_paths  # noqa: PLC0415

    libraries = [Path(tsc_paths.native_library())]
    for lib_dir in tc.library_dirs():
        libraries += [lib_dir / n for n in RUNTIME_LIBRARIES if (lib_dir / n).exists()]
    copied: list[Path] = []
    for lib in libraries:
        dest = out / "lib" / lib.name
        shutil.copy2(lib, dest)
        copied.append(dest)
    set_rpath(executable, BUNDLE_RPATH)
    os.chmod(executable, 0o755)
    return BuildResult(out, executable, stats, seconds, copied)


@dataclass
class BuildPlan:
    """A composed scenario config, ready for :func:`build_bundle` (``plan.build()``)."""

    scenario_cls: type
    config_cls: type
    scenario_dict: dict[str, Any]
    settings: RunSettings
    xodr: Path
    lanelet2: Path
    out_dir: Path
    executable_name: str
    projector_type: str | None = None

    def build(
        self,
        *,
        toolchain: Toolchain | None = None,
        release: bool = True,
        keep_workspace: Path | None = None,
    ) -> BuildResult:
        return build_bundle(
            self.scenario_cls,
            self.config_cls,
            self.scenario_dict,
            self.settings,
            xodr=self.xodr,
            lanelet2=self.lanelet2,
            out_dir=self.out_dir,
            executable_name=self.executable_name,
            projector_type=self.projector_type,
            toolchain=toolchain,
            release=release,
            keep_workspace=keep_workspace,
        )


def plan_build(
    config_name: str, overrides: list[str] | None = None, out_dir: Path | None = None
) -> BuildPlan:
    """Compose *config_name* as ``uv run scenario`` does, for :func:`build_bundle`.

    Hydra composes one config at a time per process: plan the builds one after
    the other, then build them side by side.

    Raises:
        BuildError: The config selects something the runtime cannot run.
    """
    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios
    from ..maps.config import resolve_map_paths  # noqa: PLC0415
    from ..registry import get_scenario_classes, load_scenario_plugins  # noqa: PLC0415

    load_scenario_plugins()
    cfg = run._compose_config(config_name, list(overrides or []))
    name = str(cfg.scenario.name)
    classes = get_scenario_classes(name)
    if classes is None:
        raise BuildError(
            f"scenario {name!r} is not registered with register_scenario(): a custom "
            "builder (register_scenario_builder) cannot be compiled"
        )
    scenario_cls, config_cls = classes
    map_paths = resolve_map_paths(cfg.map)
    settings = _run_settings(cfg, scenario_cls, map_paths)
    xodr = map_paths.xodr_path or map_paths.opendrive_path
    if map_paths.lanelet2_path is None or xodr is None or not Path(xodr).exists():
        raise BuildError(
            "the config names no Lanelet2 map and OpenDRIVE file to bake (an OpenDRIVE "
            "read back from CARLA has to be captured by one Python run first)"
        )
    stem = config_name.replace("/", "_")
    return BuildPlan(
        scenario_cls,
        config_cls,
        run._to_dict(cfg.scenario),
        settings,
        xodr=Path(xodr),
        lanelet2=Path(map_paths.lanelet2_path),
        out_dir=Path(out_dir) if out_dir is not None else Path("dist") / stem,
        executable_name=stem,
        projector_type=map_paths.projector_type,
    )


def build_standalone(
    config_name: str,
    overrides: list[str] | None = None,
    out_dir: Path | None = None,
    *,
    toolchain: Toolchain | None = None,
    release: bool = True,
    keep_workspace: Path | None = None,
) -> BuildResult:
    """Build the scenario config *config_name* (as ``scenario=`` takes it).

    Args:
        config_name: A scenario config, e.g. ``lane_change/left``.
        overrides: Hydra overrides, as the runner takes them.
        out_dir: The bundle directory (``dist/<config name>`` by default).
        toolchain: The Codon to compile with (found as the static check finds it).
        release: Compile with optimizations.
        keep_workspace: Write the Codon workspace here instead of a temporary
            directory (to look at what was compiled).

    Raises:
        BuildError: The scenario cannot be built (an unsupported setting, or
            part of the API the runtime does not implement yet).
    """
    plan = plan_build(config_name, overrides, out_dir)
    return plan.build(
        toolchain=toolchain, release=release, keep_workspace=keep_workspace
    )
