"""``scenario-build``: compile scenario configs into standalone binaries.

    uv run scenario-build scenario=lane_change/left
    uv run scenario-build scenario='lane_change/*' --out dist
    uv run scenario-build scenario=intersection_passing/straight scenario.timeout_seconds=20

Each ``scenario=`` config (a name or a glob, as ``scenario`` takes) is composed
with the remaining overrides exactly as the runner composes it, and built into
``<out>/<config>/`` (docs/standalone.md): ``bin/<config>`` runs it against a
CARLA server with nothing else installed. The exit status is 0 when every
config built, 1 when one did not, and 2 when there is no Codon compiler.
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..typecheck.check import find_supported_codon
from ..typecheck.toolchain import ToolchainError
from .build import BuildError, BuildPlan, BuildResult, plan_build

__all__ = ["main"]


def _pop_option(args: list[str], name: str, default: str) -> str:
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            value = args[i + 1]
            del args[i : i + 2]
            return value
        if arg.startswith(name + "="):
            del args[i]
            return arg.split("=", 1)[1]
    return default


def main(argv: list[str] | None = None) -> int:
    """Build the scenario configs *argv* selects (``sys.argv[1:]`` by default)."""
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)  # noqa: T201
        return 0
    out_root = Path(_pop_option(args, "--out", "dist"))
    jobs = int(_pop_option(args, "--jobs", str(min(4, os.cpu_count() or 1))))
    try:
        toolchain = find_supported_codon()
    except ToolchainError as exc:
        print(f"scenario-build: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios
    from ..registry import load_scenario_plugins  # noqa: PLC0415

    load_scenario_plugins()
    pattern, overrides = run._extract_scenario_override(["scenario-build", *args])
    names = run._resolve_scenario_glob(pattern or "**/*")

    # Hydra composes one config at a time: plan every build first, then run the
    # compiles side by side (each a CPU core and ~1 GB of memory).
    plans: list[BuildPlan | BuildError] = []
    for name in names:
        try:
            plans.append(plan_build(name, overrides, out_root / name.replace("/", "_")))
        except BuildError as exc:
            plans.append(exc)

    def build(plan: BuildPlan | BuildError) -> BuildResult | BuildError:
        if isinstance(plan, BuildError):
            return plan
        try:
            return plan.build(toolchain=toolchain)
        except BuildError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(build, plans))

    failed = 0
    for name, result in zip(names, results):
        if isinstance(result, BuildError):
            print(f"[FAILED] {name}: {result}")  # noqa: T201
            failed += 1
        else:
            print(  # noqa: T201
                f"[ok] {name}: {result.executable} ({result.compile_seconds:.0f}s)"
            )
    if failed:
        print(f"{failed} of {len(names)} scenario config(s) did not build")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
