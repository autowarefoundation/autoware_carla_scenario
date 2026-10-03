"""``scenario-check``: compile scenarios with Codon without running them.

    uv run scenario-check                                   # every scenario config
    uv run scenario-check scenario=intersection_passing/straight
    uv run scenario-check scenario='lane_change/*' scenario.timeout_seconds=20

Each ``scenario=`` config (a name or a glob, as ``scenario`` takes) is composed
with the remaining overrides, exactly as the runner composes it, and checked
the way the runner checks it before a run (docs/typecheck.md).  The exit
status is 0 when every scenario passed, 1 when one failed, and 2 when there is
no Codon compiler to check with.
"""

from __future__ import annotations

import logging
import sys

from .check import typecheck_scenario
from .toolchain import ToolchainError, find_codon

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
        toolchain = find_codon()
    except ToolchainError as exc:
        print(f"scenario-check: {exc}", file=sys.stderr)  # noqa: T201
        return 2

    from ..examples import run  # noqa: PLC0415 - registers the built-in scenarios
    from ..registry import get_scenario_classes, load_scenario_plugins  # noqa: PLC0415

    load_scenario_plugins()
    pattern, overrides = run._extract_scenario_override(["scenario-check", *args])
    names = run._resolve_scenario_glob(pattern or "**/*")

    failed = 0
    for config_name in names:
        cfg = run._compose_config(config_name, overrides)
        scenario_name = str(cfg.scenario.name)
        classes = get_scenario_classes(scenario_name)
        if classes is None:
            print(f"{config_name}: {scenario_name!r} has a custom builder: not checked")  # noqa: T201
            continue
        result = typecheck_scenario(
            classes[0], classes[1], run._to_dict(cfg.scenario), toolchain=toolchain
        )
        status = "ok" if result.ok else "FAILED"
        print(f"[{status}] {config_name}: {result.format()}")  # noqa: T201
        failed += not result.ok
    if failed:
        print(f"{failed} of {len(names)} scenario config(s) failed the static check")  # noqa: T201
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
