"""What the ``scenario`` command reads off its command line and environment.

The ``--resume-from N`` and ``--resume-into DIR`` flags
(:func:`_extract_resume_from`, :func:`_extract_resume_into`), the
``scenario=`` override and whether it is a glob pattern
(:func:`_extract_scenario_override`, :func:`_is_glob_pattern`), the name each
case of a batch goes by in the summary (:func:`_case_labels`), and whether
the environment already caps glibc's malloc arenas
(:func:`_arena_max_set_by_user`), and the ``hydra.run.dir=`` or
``hydra.sweep.dir=`` a batch writes to (:func:`_split_batch_output_dir`,
:func:`_batch_dir_text`, #2364), and the overrides a command line composes
a scenario with (:func:`_plain_overrides`) and the lines a ``--multirun``
names its failed cases in (:func:`_errored_case`,
:func:`_sweep_failed_lines`, #2386).

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
    "_batch_dir_text",
    "_case_labels",
    "_extract_resume_from",
    "_extract_resume_into",
    "_extract_scenario_override",
    "_errored_case",
    "_is_glob_pattern",
    "_plain_overrides",
    "_split_batch_output_dir",
    "_sweep_failed_lines",
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


#: The exit status of a bad ``--resume-from``: a usage error, refused before
#: anything runs (docs/usage.md, "Exit Codes").
_USAGE_EXIT_CODE = 2


def _resume_from_value(text: str) -> int:
    """The case *text*, the value of ``--resume-from``, names: a whole number
    of at least 1, or exit 2 after an ``Error:`` line.

    ``--resume-from N`` starts at case *N*, counted from 1, so 0 and negative
    values are refused rather than silently running every case (#1784).
    """
    value = 0
    try:
        value = int(text)
    except ValueError:
        print(  # noqa: T201
            f"Error: --resume-from requires an integer, got '{text}'", file=sys.stderr
        )
        sys.exit(_USAGE_EXIT_CODE)
    if value < 1:
        print(  # noqa: T201
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            f"start at), got '{text}'",
            file=sys.stderr,
        )
        sys.exit(_USAGE_EXIT_CODE)
    return value


def _extract_resume_from(argv: list[str]) -> tuple[int, list[str]]:
    """Extract ``--resume-from N`` from *argv* and return the value and cleaned argv.

    A value that is not an integer, or is below 1, is a usage error: an
    ``Error:`` line and exit 2 (:func:`_resume_from_value`).

    Returns:
        A 2-tuple of ``(resume_from, remaining_argv)``.
        *resume_from* is 0 when the flag is absent, at least 1 otherwise.
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
                resume_from = _resume_from_value(argv[i + 1])
                skip_next = True
            else:
                print("Error: --resume-from requires a value", file=sys.stderr)  # noqa: T201
                sys.exit(_USAGE_EXIT_CODE)
        elif arg.startswith("--resume-from="):
            resume_from = _resume_from_value(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
    return resume_from, remaining


def _extract_resume_into(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract ``--resume-into DIR`` from *argv* and return the directory and
    cleaned argv.

    ``--resume-into DIR`` runs a local batch's remaining cases in *DIR*, the
    directory of the batch it resumes, rather than a new one, so that the
    batch is judged as a whole (``examples.run.run_batch``, docs/usage.md
    "Resume from a Specific Scenario").  The last one given wins, as for
    ``--resume-from``.  A flag without a value (or with an empty one) is a
    usage error: an ``Error:`` line and exit 2.

    Returns:
        A 2-tuple of ``(directory, remaining_argv)``.  *directory* is
        ``None`` when the flag is absent.
    """
    directory: str | None = None
    remaining: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        value = ""
        if arg == "--resume-into":
            if i + 1 < len(argv):
                value = argv[i + 1]
                skip_next = True
        elif arg.startswith("--resume-into="):
            value = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
            continue
        if not value:
            print(  # noqa: T201
                "Error: --resume-into requires a value (the directory of the "
                "batch to resume)",
                file=sys.stderr,
            )
            sys.exit(_USAGE_EXIT_CODE)
        directory = value
    return directory, remaining


#: The Hydra keys naming a run's and a sweep's output directory, which a
#: batch reads as its own (``examples.run._extract_batch_output_dir``).
_RUN_DIR_KEY = "hydra.run.dir"
_SWEEP_DIR_KEY = "hydra.sweep.dir"


def _split_batch_output_dir(
    argv: list[str], *, multirun: bool
) -> tuple[str | None, list[str], list[str]]:
    """The batch output directory *argv* names, *argv* without it, and the
    overrides of the other mode's key, which are ignored.

    ``hydra.run.dir=`` names it, or with ``--multirun`` (*multirun*)
    ``hydra.sweep.dir=``; the last one given counts, ``None`` when none is.
    Every ``hydra.run.dir`` and ``hydra.sweep.dir`` override (``+``, ``++``
    and ``~`` ones too) is taken off: neither is a case's setting.  The value
    is as given (:func:`_batch_dir_text` strips it).
    """
    wanted = _SWEEP_DIR_KEY if multirun else _RUN_DIR_KEY
    value: str | None = None
    kept: list[str] = []
    ignored: list[str] = []
    for arg in argv:
        name, _, given = arg.lstrip("+~").partition("=")
        if name != _RUN_DIR_KEY and name != _SWEEP_DIR_KEY:
            kept.append(arg)
            continue
        if arg.startswith("~"):
            continue
        if name == wanted:
            value = given
        else:
            ignored.append(arg)
    return value, kept, ignored


def _batch_dir_text(value: str) -> str:
    """*value* (a ``hydra.*.dir=`` override's) without the whitespace and
    quotes around it."""
    return value.strip().strip("'\"")


def _errored_case(case: str, type_name: str, message: str) -> str:
    """How a ``--multirun`` *case* that raised an exception of *type_name*
    with *message* is listed among the failed ones: its first line (the
    type alone when it says nothing)."""
    lines = message.strip().splitlines()
    reason = f"{type_name}: {lines[0]}" if lines else type_name
    return f"{case} (errored: {reason})"


def _sweep_failed_lines(failed: list[str], ran: int) -> list[str]:
    """The lines a ``--multirun`` that ran *ran* cases, of which *failed*
    failed, ends with: a summary, then each failed case."""
    lines = [f"Sweep failed: {len(failed)} of {ran} case(s) failed:"]
    for case in failed:
        lines.append(f"  {case}")
    return lines


def _plain_overrides(overrides: list[str]) -> list[str]:
    """*overrides* without what Hydra reads itself: flags (``-...``) and
    ``hydra...`` overrides (``+``, ``++`` and ``~`` ones too)."""
    return [
        arg
        for arg in overrides
        if not arg.startswith("-") and not arg.lstrip("+~").startswith("hydra")
    ]
