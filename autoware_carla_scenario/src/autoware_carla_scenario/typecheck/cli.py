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
invalid, or the checker itself raised on it: ``[FAILED] <config>: checker
error: ...``, with the traceback on stderr, and the other configs are still
checked and reported), and 2 when there is
no Codon compiler to compile a config with, a config could not be checked (``[SKIPPED]``
says why) and none failed, or the command line does not compose: no config
matches ``scenario=``, or an override names a key the config does not have
(the message suggests the closest key; ``+key=value`` adds a new one; each
config of a glob that does not compose is reported, named).  ``HYDRA_FULL_ERROR=1``
shows the traceback of such an error.
``scenario='*'`` is every config, like ``scenario='*/*'`` or no ``scenario=``.
A package author checks only theirs: ``scenario='my_package/*'``.

``--library`` checks the framework's own modules instead: those
``typecheck/library.py`` lists as checked, or the modules named after it with
the checked modules they import (docs/typecheck.md, "Checking the framework
itself").  A name that is not a module of the package, or that it EXCLUDES,
is refused (``[FAILED]``, exit status 1) before Codon is looked for.
Likewise, the configs are composed, their ``controls:`` validated and their
scenario names looked up before Codon is looked for, and Codon is looked for
only when a config is to be compiled: a wrong control or an unknown scenario
name fails the command (exit status 1) with Codon or without, and configs
whose scenarios all have a custom builder need no Codon.

The flags are ``-h``/``--help`` and ``--library``.  Any other argument that
starts with ``-`` is refused with ``Error: scenario-check does not take
<flag>`` and exit 2, rather than read as an override (or, after
``--library``, as a module name).  Overrides (``key=value``, ``+key=value``,
``++key=value``, ``~key``) are passed on as the runner passes them.  Ctrl-C
prints ``scenario-check: interrupted`` and exits 130.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .check import find_supported_codon, refuse_library_modules, typecheck_library
from .mode import typecheck_registered
from .toolchain import Toolchain, ToolchainError

__all__ = ["main"]


def _checker_error(what: str, exc: BaseException) -> str:
    """The ``[FAILED]`` line of a check that raised; its traceback goes to stderr.

    A checker error fails that check only, exit status 1 as for any failed
    check (#2091): the other configs are still checked and reported.
    """
    print(  # noqa: T201
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        end="",
        file=sys.stderr,
    )
    return f"[FAILED] {what}: checker error: {type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    """Check the scenario configs *argv* selects (``sys.argv[1:]`` by default).

    Ctrl-C prints one line, not a traceback, and exits 130 (#2333).
    """
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("scenario-check: interrupted", file=sys.stderr)  # noqa: T201
        return 130


def _main(argv: list[str] | None) -> int:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in args):
        print(__doc__)  # noqa: T201
        return 0
    # Hydra would read a flag it does not know as an override, and report it
    # once for every config (#2333); after --library, it is no module name.
    for arg in args:
        if arg.startswith("-") and arg != "--library":
            # --library takes no value: `--library=x` is named whole.
            flag = arg if arg.startswith("--library=") else arg.split("=", 1)[0]
            print(f"Error: scenario-check does not take {flag}", file=sys.stderr)  # noqa: T201
            return 2
    library = "--library" in args
    modules = [a for a in args if a != "--library"]
    if library and modules:
        # Refused the same with Codon or without (#2288).
        refusal = refuse_library_modules(modules)
        if refusal is not None:
            print(f"[FAILED] {refusal.format()}")  # noqa: T201
            return 1
    if library:
        try:
            library_toolchain = find_supported_codon()
        except ToolchainError as exc:
            print(f"scenario-check: {exc}", file=sys.stderr)  # noqa: T201
            return 2
        try:
            result = typecheck_library(modules or None, toolchain=library_toolchain)
        except Exception as exc:  # noqa: BLE001 - reported as a failed check
            print(_checker_error("library check", exc))  # noqa: T201
            return 1
        if result.skipped is not None:
            # No verdict either way, as for a skipped config (#2249).
            print(f"[SKIPPED] {result.format()}")  # noqa: T201
            return 2
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
        get_scenario_classes,
        get_scenario_registry,
        load_scenario_plugins,
    )
    from ..scenario_glob import is_scenario_glob  # noqa: PLC0415

    load_scenario_plugins()
    pattern, overrides = run._extract_scenario_override(["scenario-check", *args])
    if pattern in (None, "*"):
        pattern = "**/*"  # `*` alone matches no `group/name`
    names = resolve_scenarios(pattern)
    if names is None:
        return 2
    # Each config of a glob is composed and may fail on its own: every
    # failure is reported, each headed by its config's name, as
    # scenario-expand reports them (#2249), before the command gives up.
    globbed = is_scenario_glob(pattern)
    cfgs = []
    composed = True
    for name in names:
        cfg = compose_or_report(name, overrides, named=globbed)
        if cfg is None:
            composed = False
        cfgs.append(cfg)
    if not composed:
        return 2

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

    # Composing the configs, validating their controls and telling which
    # scenario names are unknown or have a custom builder needs no Codon: it
    # is looked for only now, and only when a config is to be compiled, so
    # what fails without compiling fails the command (exit status 1) with
    # Codon or without (#2511, #2605), as a refused --library module does
    # (#2288), and a run of custom builders only needs no Codon at all.
    unknown: set[str] = set()
    custom: set[str] = set()
    for config_name, cfg in zip(names, cfgs):
        if config_name in control_errors:
            continue
        scenario_name = str(cfg.scenario.name)
        if get_scenario_builder(scenario_name) is None:
            unknown.add(config_name)
        elif get_scenario_classes(scenario_name) is None:
            custom.add(config_name)
    to_check = [
        (n, c)
        for n, c in zip(names, cfgs)
        if n not in control_errors and n not in unknown and n not in custom
    ]
    toolchain: Toolchain | None = None
    no_codon = False
    if to_check:
        try:
            toolchain = find_supported_codon()
        except ToolchainError as exc:
            print(f"scenario-check: {exc}", file=sys.stderr)  # noqa: T201
            no_codon = True
            to_check = []

    def check(cfg: Any) -> Any:
        # An exception is this config's failure, not the command's: it is
        # returned, so the pool does not re-raise it and drop every other
        # config's result (#2091).
        try:
            return typecheck_registered(
                str(cfg.scenario.name), run._to_dict(cfg.scenario), toolchain=toolchain
            )
        except Exception as exc:  # noqa: BLE001 - reported as a failed check
            return exc

    # Each check is a separate `codon build`: run a few side by side (each
    # takes a CPU core and some memory of its own).
    # A config whose controls are wrong is reported above, not compiled.
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        checked = list(pool.map(check, [c for _, c in to_check]))
    results: dict[str, Any] = {n: r for (n, _), r in zip(to_check, checked)}

    failed = len(control_errors)
    skipped = 0
    for config_name, cfg in zip(names, cfgs):
        scenario_name = str(cfg.scenario.name)
        if config_name in control_errors:
            continue
        if config_name in unknown:
            known = sorted(get_scenario_registry())
            hint = suggest(scenario_name, known)
            print(  # noqa: T201
                f"[FAILED] {config_name}: unknown scenario name {scenario_name!r} "
                f"({hint}known names: {', '.join(known)})"
            )
            failed += 1
            continue
        if config_name in custom:
            print(f"{config_name}: {scenario_name!r} has a custom builder: not checked")  # noqa: T201
            continue
        if config_name not in results:
            continue  # no Codon to compile it with, said above
        result = results[config_name]
        if isinstance(result, Exception):
            print(_checker_error(config_name, result))  # noqa: T201
            failed += 1
            continue
        if result is None:
            # Registered with a custom builder since it was classified.
            print(f"{config_name}: {scenario_name!r} has a custom builder: not checked")  # noqa: T201
            continue
        if result.skipped is not None:
            # No verdict either way: never a pass (#2249).
            print(f"[SKIPPED] {config_name}: {result.format()}")  # noqa: T201
            skipped += 1
            continue
        status = "ok" if result.ok else "FAILED"
        print(f"[{status}] {config_name}: {result.format()}")  # noqa: T201
        failed += not result.ok
    if failed:
        print(f"{failed} of {len(names)} scenario config(s) failed the static check")  # noqa: T201
    if skipped:
        print(f"{skipped} of {len(names)} scenario config(s) could not be checked")  # noqa: T201
    return 1 if failed else 2 if skipped or no_codon else 0


if __name__ == "__main__":
    sys.exit(main())
