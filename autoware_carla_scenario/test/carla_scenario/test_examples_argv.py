"""examples.argv, split out of examples.run so the library check compiles it
(#752): what the ``scenario`` command reads off its command line and
environment, with the values master's examples.run gave, and examples.run
still exporting every moved name, as the same objects."""

from __future__ import annotations

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


@pytest.mark.parametrize("name", _MOVED)
def test_run_re_exports_each_moved_name(name: str) -> None:
    assert getattr(run, name) is getattr(argv, name)


def test_argv_exports_exactly_the_moved_names() -> None:
    assert sorted(argv.__all__) == sorted(_MOVED)


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
        (["scenario", "--resume-from", "-2"], (-2, ["scenario"])),
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
    ],
)
def test_extract_resume_from_refuses_a_missing_or_non_integer_value(
    given: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        argv._extract_resume_from(given)
    assert exc.value.code == 1
    assert capsys.readouterr().out == message + "\n"


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
