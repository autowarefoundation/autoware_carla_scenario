"""What the ``scenario`` command reads off its command line and environment.

The ``--resume-from N`` flag (:func:`_extract_resume_from`), the
``scenario=`` override and whether it is a glob pattern
(:func:`_extract_scenario_override`, :func:`_is_glob_pattern`), the name each
case of a batch goes by in the summary (:func:`_case_labels`), and whether
the environment already caps glibc's malloc arenas
(:func:`_arena_max_set_by_user`).

Split out of ``examples.run``, which re-exports every name here and keeps
what needs Hydra and omegaconf, so that the library check compiles it
(``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

from ..scenario_glob import is_scenario_glob

__all__ = [
    "BatchCase",
    "_arena_max_set_by_user",
    "_case_labels",
    "_extract_resume_from",
    "_extract_scenario_override",
    "_is_glob_pattern",
]


#: One run of a batch: the scenario's config name and the overrides it runs
#: with -- the command line's, after those of its sweep case, if any.
BatchCase = tuple[str, list[str]]


def _arena_max_set_by_user(environ: Mapping[str, str]) -> bool:
    """Return whether the environment already says how many arenas to keep."""
    if environ.get("MALLOC_ARENA_MAX"):
        return True
    return "arena_max" in environ.get("GLIBC_TUNABLES", "")


def _is_glob_pattern(value: str) -> bool:
    """Return ``True`` if *value* contains glob metacharacters.

    :func:`~autoware_carla_scenario.scenario_glob.is_scenario_glob`, the test
    the viewer's launcher applies too.
    """
    return is_scenario_glob(value)


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
