"""``scenario-check``: compile scenarios with Codon without running them.

    uv run scenario-check                                   # every scenario config
    uv run scenario-check scenario=intersection_passing/straight
    uv run scenario-check scenario='lane_change/*' scenario.timeout_seconds=20
    uv run scenario-check --library                         # the framework itself
    uv run scenario-check --library autoware_carla_scenario.kinematics.vector

Each ``scenario=`` config (a name or a glob, as ``scenario`` takes) is composed
with the remaining overrides, exactly as the runner composes it, and checked
the way the runner checks it before a run (docs/typecheck.md).  The exit
status is 0 when every scenario passed, 1 when one failed (or its ``controls:`` are
invalid), and 2 when there is
no Codon compiler to check with or the command line does not compose: no config
matches ``scenario=``, or an override names a key the config does not have
(the message suggests the closest key; ``+key=value`` adds a new one).  ``HYDRA_FULL_ERROR=1``
shows the traceback of such an error.
``scenario='*'`` is every config, like ``scenario='*/*'`` or no ``scenario=``.
A package author checks only theirs: ``scenario='my_package/*'``.

``--library`` checks the framework's own modules instead: those
``typecheck/library.py`` lists as checked, or the modules named after it with
the checked modules they import (docs/typecheck.md, "Checking the framework
itself").
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .check import find_supported_codon, typecheck_library
from .mode import typecheck_registered
from .toolchain import ToolchainError

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Check the scenario configs *argv* selects (``sys.argv[1:]`` by default)."""
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)  # noqa: T201
        return 0
    try:
        toolchain = find_supported_codon()
    except ToolchainError as exc:
        print(f"scenario-check: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if "--library" in args:
        modules = [a for a in args if a != "--library"]
        result = typecheck_library(modules or None, toolchain=toolchain)
        print(f"[{'ok' if result.ok else 'FAILED'}] {result.format()}")  # noqa: T201
        return 0 if result.ok else 1

    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios
    from ..examples.compose_errors import (  # noqa: PLC0415
        compose_or_report,
        resolve_scenarios,
        suggest,
    )
    from ..registry import (  # noqa: PLC0415
        get_scenario_builder,
        get_scenario_registry,
        load_scenario_plugins,
    )

    load_scenario_plugins()
    pattern, overrides = run._extract_scenario_override(["scenario-check", *args])
    if pattern in (None, "*"):
        pattern = "**/*"  # `*` alone matches no `group/name`
    names = resolve_scenarios(pattern)
    if names is None:
        return 2
    cfgs = []
    for name in names:
        cfg = compose_or_report(name, overrides)
        if cfg is None:
            return 2
        cfgs.append(cfg)

    # The sweep reads `controls:` only when it expands; a wrong one is an
    # authoring error, reported here with its config path (#546).
    from omegaconf import OmegaConf  # noqa: PLC0415

    from ..odd.controls import validate_controls  # noqa: PLC0415

    control_errors: dict[str, int] = {}
    for config_name, cfg in zip(names, cfgs):
        node = OmegaConf.select(cfg, "controls")
        resolved = None if node is None else OmegaConf.to_container(node, resolve=True)
        diagnostics = validate_controls(resolved, cfg)
        for d in diagnostics:
            tag = "FAILED" if d.severity == "error" else "warning"
            print(f"[{tag}] {config_name}: {d.path}: {d.message}")  # noqa: T201
        errors = sum(d.severity == "error" for d in diagnostics)
        if errors:
            control_errors[config_name] = errors

    def check(cfg: Any) -> Any:
        return typecheck_registered(
            str(cfg.scenario.name), run._to_dict(cfg.scenario), toolchain=toolchain
        )

    # Each check is a separate `codon build`: run a few side by side (each
    # takes a CPU core and some memory of its own).
    # A config whose controls are wrong is reported above, not compiled.
    to_check = [c for n, c in zip(names, cfgs) if n not in control_errors]
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        checked = iter(list(pool.map(check, to_check)))
    results: list[Any] = [None if n in control_errors else next(checked) for n in names]

    failed = len(control_errors)
    for config_name, cfg, result in zip(names, cfgs, results):
        scenario_name = str(cfg.scenario.name)
        if config_name in control_errors:
            continue
        if result is None and get_scenario_builder(scenario_name) is None:
            known = sorted(get_scenario_registry())
            hint = suggest(scenario_name, known)
            print(  # noqa: T201
                f"[FAILED] {config_name}: unknown scenario name {scenario_name!r} "
                f"({hint}known names: {', '.join(known)})"
            )
            failed += 1
            continue
        if result is None:
            print(f"{config_name}: {scenario_name!r} has a custom builder: not checked")  # noqa: T201
            continue
        status = "ok" if result.ok else "FAILED"
        print(f"[{status}] {config_name}: {result.format()}")  # noqa: T201
        failed += not result.ok
    if failed:
        print(f"{failed} of {len(names)} scenario config(s) failed the static check")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
