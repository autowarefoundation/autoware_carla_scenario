"""examples.argv, split out of examples.run so the library check compiles it
(#752): what the ``scenario`` command reads off its command line and
environment, with the values master's examples.run gave, and examples.run
still exporting every moved name, as the same objects."""

from __future__ import annotations

import contextlib
import io
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.examples import argv, run

#: What examples.run held before the split that now lives in examples.argv.
_MOVED = (
    "BatchCase",
    "_arena_max_set_by_user",
    "_case_labels",
    "_extract_resume_from",
    "_extract_scenario_override",
    "_is_glob_pattern",
)


#: What examples.argv has gained since, which examples.run imports too.
_ADDED = (
    "_extract_resume_into",
    "_batch_dir_text",
    "_split_batch_output_dir",
    "_errored_case",
    "_plain_overrides",
    "_sweep_failed_lines",
)


@pytest.mark.parametrize("name", _MOVED + _ADDED)
def test_run_re_exports_each_moved_name(name: str) -> None:
    assert getattr(run, name) is getattr(argv, name)


def test_argv_exports_exactly_the_moved_names() -> None:
    assert sorted(argv.__all__) == sorted(_MOVED + _ADDED)


# ---------------------------------------------------------------------------
# --resume-from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["scenario"], (0, ["scenario"])),
        (["scenario", "--resume-from", "3", "a=1"], (3, ["scenario", "a=1"])),
        (["scenario", "--resume-from=4", "x"], (4, ["scenario", "x"])),
        # Repeated: the last one wins, whichever its form.
        (["scenario", "--resume-from", "1", "--resume-from=2"], (2, ["scenario"])),
        (["scenario", "--resume-from=2", "--resume-from", "5"], (5, ["scenario"])),
        # int() takes a sign, surrounding blanks and underscores.
        (["scenario", "--resume-from", "+3"], (3, ["scenario"])),
        (["scenario", "--resume-from", " 7 "], (7, ["scenario"])),
        (["scenario", "--resume-from", "1_0"], (10, ["scenario"])),
        # Only the exact flag: a longer one is left as it is.
        (["scenario", "--resume-fromage=1"], (0, ["scenario", "--resume-fromage=1"])),
        # argv[0] is not skipped.
        (["--resume-from", "9"], (9, [])),
    ],
)
def test_extract_resume_from(given: list[str], expected: tuple[int, list[str]]) -> None:
    assert argv._extract_resume_from(given) == expected


@pytest.mark.parametrize(
    ("given", "message"),
    [
        (["scenario", "--resume-from"], "Error: --resume-from requires a value"),
        (
            ["scenario", "--resume-from", "abc"],
            "Error: --resume-from requires an integer, got 'abc'",
        ),
        (
            ["scenario", "--resume-from=x"],
            "Error: --resume-from requires an integer, got 'x'",
        ),
        (
            ["scenario", "--resume-from="],
            "Error: --resume-from requires an integer, got ''",
        ),
        (
            ["scenario", "--resume-from=1=2"],
            "Error: --resume-from requires an integer, got '1=2'",
        ),
        # The value after the flag is taken whatever it is, another flag too.
        (
            ["scenario", "--resume-from", "--resume-from", "3"],
            "Error: --resume-from requires an integer, got '--resume-from'",
        ),
        (
            ["scenario", "--resume-from", "2.5"],
            "Error: --resume-from requires an integer, got '2.5'",
        ),
        # Below 1: it starts at case N, counted from 1, so these would run
        # every case without a word (#1784).
        (
            ["scenario", "--resume-from", "0"],
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            "start at), got '0'",
        ),
        (
            ["scenario", "--resume-from", "-3"],
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            "start at), got '-3'",
        ),
        (
            ["scenario", "--resume-from=-1"],
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            "start at), got '-1'",
        ),
        (
            ["scenario", "--resume-from=0"],
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            "start at), got '0'",
        ),
        # A valid one first does not excuse a bad one after it.
        (
            ["scenario", "--resume-from", "3", "--resume-from", "0"],
            "Error: --resume-from must be at least 1 (the 1-indexed case to "
            "start at), got '0'",
        ),
    ],
)
def test_extract_resume_from_refuses_a_missing_non_integer_or_below_1_value(
    given: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A usage error: exit 2, before anything runs (#1784)."""
    with pytest.raises(SystemExit) as exc:
        argv._extract_resume_from(given)
    assert exc.value.code == 2
    assert capsys.readouterr().err == message + "\n"


# ---------------------------------------------------------------------------
# scenario=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["scenario"], (None, [])),
        ([], (None, [])),
        (["scenario", "scenario=a/b"], ("a/b", [])),
        # Only a plain override: `+`/`++` ones are left to Hydra.
        (["scenario", "+scenario=a/b"], (None, ["+scenario=a/b"])),
        (["scenario", "++scenario=a/b"], (None, ["++scenario=a/b"])),
        # Quotes the shell left are kept.
        (["scenario", "scenario='a/*'"], ("'a/*'", [])),
        (["scenario", 'scenario="a/b"'], ('"a/b"', [])),
        # Repeated: the last one wins; the others are dropped.
        (["scenario", "scenario=a", "scenario=b", "x=1"], ("b", ["x=1"])),
        (["scenario", "scenario="], ("", [])),
        (["scenario", "scenario=a=b"], ("a=b", [])),
        (["scenario", "scenario"], (None, ["scenario"])),
        (["scenario", "Scenario=a"], (None, ["Scenario=a"])),
        (
            ["scenario", "scenario.timeout_seconds=1", "scenario=c"],
            ("c", ["scenario.timeout_seconds=1"]),
        ),
        # argv[0] is skipped, whatever it is.
        (["scenario=first", "y"], (None, ["y"])),
    ],
)
def test_extract_scenario_override(
    given: list[str], expected: tuple[str | None, list[str]]
) -> None:
    assert argv._extract_scenario_override(given) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a/b", False),
        ("", False),
        ("a/*", True),
        ("a/**", True),
        ("a/?", True),
        ("a/[lr]", True),
        ("]", False),
        ("a{b}", False),
    ],
)
def test_is_glob_pattern(value: str, expected: bool) -> None:
    assert argv._is_glob_pattern(value) is expected


# ---------------------------------------------------------------------------
# The malloc arenas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({"MALLOC_ARENA_MAX": "2"}, True),
        ({"MALLOC_ARENA_MAX": "0"}, True),
        # Set but empty is not set.
        ({"MALLOC_ARENA_MAX": ""}, False),
        ({"malloc_arena_max": "2"}, False),
        ({"GLIBC_TUNABLES": "glibc.malloc.arena_max=2"}, True),
        (
            {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0:glibc.malloc.arena_max=4"},
            True,
        ),
        ({"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"}, False),
        ({"GLIBC_TUNABLES": "glibc.malloc.arena_test=8"}, False),
        ({"GLIBC_TUNABLES": ""}, False),
        ({"MALLOC_ARENA_MAX": "", "GLIBC_TUNABLES": "glibc.malloc.arena_max=1"}, True),
    ],
)
def test_arena_max_set_by_user(environ: dict[str, str], expected: bool) -> None:
    assert argv._arena_max_set_by_user(environ) is expected


# ---------------------------------------------------------------------------
# Case labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cases", "expected"),
    [
        ([], []),
        ([("a", [])], ["a"]),
        ([("a", []), ("b", ["x=1"])], ["a", "b"]),
        ([("a", ["1"]), ("a", ["2"])], ["a#1", "a#2"]),
        # Numbered in their order, apart from the scenarios between them.
        (
            [("a", []), ("b", []), ("a", []), ("c", []), ("a", [])],
            ["a#1", "b", "a#2", "c", "a#3"],
        ),
        ([("a#1", []), ("a", [])], ["a#1", "a"]),
    ],
)
def test_case_labels(cases: list[argv.BatchCase], expected: list[str]) -> None:
    assert argv._case_labels(cases) == expected


# ---------------------------------------------------------------------------
# --resume-into (#1957)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["scenario"], (None, ["scenario"])),
        (
            ["scenario", "--resume-into", "out/d", "a=1"],
            ("out/d", ["scenario", "a=1"]),
        ),
        (["scenario", "--resume-into=out/d", "x"], ("out/d", ["scenario", "x"])),
        # Repeated: the last one wins.
        (["scenario", "--resume-into", "a", "--resume-into=b"], ("b", ["scenario"])),
        # Only the exact flag; --resume-from is left for its own parser.
        (
            ["scenario", "--resume-intox=1", "--resume-from", "2"],
            (None, ["scenario", "--resume-intox=1", "--resume-from", "2"]),
        ),
    ],
)
def test_extract_resume_into(
    given: list[str], expected: tuple[str | None, list[str]]
) -> None:
    assert argv._extract_resume_into(given) == expected


@pytest.mark.parametrize(
    "given", [["scenario", "--resume-into"], ["scenario", "--resume-into="]]
)
def test_resume_into_without_a_value_is_a_usage_error(
    given: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        argv._extract_resume_into(given)
    assert raised.value.code == 2
    assert capsys.readouterr().err.startswith("Error: --resume-into requires a value")


# ---------------------------------------------------------------------------
# hydra.run.dir / hydra.sweep.dir of a batch, moved from examples.run (#2364)
# ---------------------------------------------------------------------------
#
# ``_old_extract_batch_output_dir`` is examples.run._extract_batch_output_dir()
# before #2364, verbatim but for the names, the logger (a list it appends
# to) and the ``${...}`` resolution, which did not move and is left out: the
# values below hold none.

_OLD_RUN_DIR_KEY = "hydra.run.dir"
_OLD_SWEEP_DIR_KEY = "hydra.sweep.dir"


def _old_extract_batch_output_dir(
    argv: list[str], *, multirun: bool, warnings: list[str]
) -> tuple[Path | None, list[str]]:
    wanted = _OLD_SWEEP_DIR_KEY if multirun else _OLD_RUN_DIR_KEY
    value: str | None = None
    kept: list[str] = []
    for arg in argv:
        name, _, given = arg.lstrip("+~").partition("=")
        if name not in (_OLD_RUN_DIR_KEY, _OLD_SWEEP_DIR_KEY):
            kept.append(arg)
            continue
        if arg.startswith("~"):
            continue
        if name == wanted:
            value = given
        else:
            warnings.append(
                "%s is ignored: a batch%s writes to %s"
                % (
                    arg,
                    " run with --multirun" if multirun else " without --multirun",
                    wanted,
                )
            )
    if value is None:
        return None, kept
    text = value.strip().strip("'\"")
    assert "${" not in text
    if not text:
        print(f"Error: {wanted}= names no directory", file=sys.stderr)
        sys.exit(2)
    return Path(text), kept


_BATCH_ARGS = (
    "hydra.run.dir=out",
    "hydra.sweep.dir=sweep",
    "+hydra.run.dir=plus",
    "++hydra.sweep.dir=plusplus",
    "~hydra.run.dir",
    "~hydra.sweep.dir=x",
    "hydra.run.dir=",
    "hydra.sweep.dir=  ",
    "hydra.run.dir='quoted dir'",
    'hydra.sweep.dir="dq"',
    "hydra.run.dir=a=b",
    "hydra.run.dirx=1",
    "hydra.run=1",
    "hydra.job.chdir=true",
    "seed=3",
    "scenario=demo",
    "--multirun",
    "+~hydra.run.dir=odd",
    "hydra.sweep.dir= 'x' ",
)


def _batch_outcome(call: Any) -> tuple[object, str]:
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            return call(), stderr.getvalue()
    except SystemExit as exc:
        return ("exit", exc.code), stderr.getvalue()


def test_the_batch_output_dir_is_split_off_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new_warnings: list[str] = []

    class _Logger:
        def warning(self, message: str, *args: object) -> None:
            new_warnings.append(message % args)

    monkeypatch.setattr(run, "logger", _Logger())
    rng = random.Random(2364)
    for _ in range(3000):
        args = [rng.choice(_BATCH_ARGS) for _ in range(rng.randrange(0, 6))]
        multirun = rng.random() < 0.5
        old_warnings: list[str] = []
        new_warnings.clear()
        old = _batch_outcome(
            lambda: _old_extract_batch_output_dir(
                args, multirun=multirun, warnings=old_warnings
            )
        )
        new = _batch_outcome(
            lambda: run._extract_batch_output_dir(args, multirun=multirun)
        )
        assert new == old, args
        assert new_warnings == old_warnings, args


def test_the_split_returns_the_value_the_kept_and_the_ignored() -> None:
    assert argv._split_batch_output_dir(
        ["seed=1", "hydra.sweep.dir=s", "+hydra.run.dir= 'r' ", "~hydra.run.dir"],
        multirun=False,
    ) == (" 'r' ", ["seed=1"], ["hydra.sweep.dir=s"])
    assert argv._batch_dir_text(" 'r' ") == "r"
    assert run._RUN_DIR_KEY is argv._RUN_DIR_KEY
    assert run._SWEEP_DIR_KEY is argv._SWEEP_DIR_KEY


# ---------------------------------------------------------------------------
# A --multirun's failed cases and the overrides it composes, moved from
# examples.run (#2386)
# ---------------------------------------------------------------------------
#
# ``_old_errored`` is the tail of examples.run._record_multirun_error(),
# ``_old_exit_if_multirun_failed`` examples.run._exit_if_multirun_failed()
# (the lists handed in) and ``_old_plain`` the filter of
# examples.run._warn_if_sweep_not_expanded(), before #2386, verbatim.


def _old_errored(case: str, exc: BaseException) -> str:
    lines = str(exc).strip().splitlines()
    reason = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    return f"{case} (errored: {reason})"


def _old_exit_if_multirun_failed(
    _multirun_failed: list[str], _multirun_ran: list[str]
) -> None:
    if not _multirun_failed:
        return
    print(  # noqa: T201
        f"Sweep failed: {len(_multirun_failed)} of {len(_multirun_ran)} "
        "case(s) failed:",
        file=sys.stderr,
    )
    for case in _multirun_failed:
        print(f"  {case}", file=sys.stderr)  # noqa: T201
    sys.exit(1)


def _old_plain(overrides: list[str]) -> list[str]:
    return [
        arg
        for arg in overrides
        if not arg.startswith("-") and not arg.lstrip("+~").startswith("hydra")
    ]


_MESSAGES = (
    "",
    "   ",
    "boom",
    "  boom  \n",
    "first\nsecond",
    "\n\nthird",
    "a\r\nb",
    "a\rb",
    "a\x0bb",
    "a\x0cb",
    "a\x1cb",
    "a\x85b",
    "a\u2028b",
    "\u2029",
    "tab\tin",
    "unicode \u00e9",
)

_EXCEPTIONS = (ValueError, RuntimeError, KeyError, OSError, KeyboardInterrupt)


def test_the_errored_case_is_the_same() -> None:
    rng = random.Random(2386)
    for _ in range(3000):
        case = rng.choice(["", "seed=1", "scenario=cut_in seed=2", "x=' a '"])
        args: tuple[Any, ...] = tuple(
            rng.choice(_MESSAGES) for _ in range(rng.randint(0, 2))
        )
        exc = rng.choice(_EXCEPTIONS)(*args)
        want = _old_errored(case, exc)
        assert argv._errored_case(case, type(exc).__name__, str(exc)) == want


def test_the_sweep_summary_is_the_same() -> None:
    rng = random.Random(2386)
    for _ in range(500):
        failed = [rng.choice(_MESSAGES) for _ in range(rng.randint(1, 4))]
        ran = [rng.choice(_MESSAGES) for _ in range(rng.randint(0, 6))]
        old_err = io.StringIO()
        with contextlib.redirect_stderr(old_err), pytest.raises(SystemExit) as old:
            _old_exit_if_multirun_failed(failed, ran)
        new_err = io.StringIO()
        with contextlib.redirect_stderr(new_err), pytest.raises(SystemExit) as new:
            for line in argv._sweep_failed_lines(failed, len(ran)):
                print(line, file=sys.stderr)  # noqa: T201
            sys.exit(1)
        assert new_err.getvalue() == old_err.getvalue()
        assert new.value.code == old.value.code == 1


def test_run_exits_with_the_sweep_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run, "_multirun_failed", ["seed=1 (errored: E: x)"])
    monkeypatch.setattr(run, "_multirun_ran", ["seed=1", "seed=2"])
    with pytest.raises(SystemExit) as raised:
        run._exit_if_multirun_failed()
    assert raised.value.code == 1
    assert capsys.readouterr().err == (
        "Sweep failed: 1 of 2 case(s) failed:\n  seed=1 (errored: E: x)\n"
    )
    monkeypatch.setattr(run, "_multirun_failed", [])
    run._exit_if_multirun_failed()


_PLAIN_ARGS = (
    "seed=3",
    "scenario=demo",
    "--multirun",
    "-m",
    "-",
    "",
    "hydra.run.dir=out",
    "+hydra.job.chdir=true",
    "++hydra/sweeper=basic",
    "~hydra.sweep.dir",
    "hydra/launcher=basic",
    "hydrant=1",
    "my.hydra=1",
    " hydra=1",
    "+-x=1",
    "~seed",
    "+~hydra=odd",
    "'seed=range(0,32)'",
    "seed=1,2",
)


def test_the_plain_overrides_are_the_same() -> None:
    rng = random.Random(2386)
    for _ in range(2000):
        overrides = [rng.choice(_PLAIN_ARGS) for _ in range(rng.randint(0, 8))]
        assert argv._plain_overrides(overrides) == _old_plain(overrides)
