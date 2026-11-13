"""The bytes of a given-up job's marker (``gate.errored``, ``case_errored.json``).

Pins, byte for byte, what write_case_errored() and write_case_unfinished()
write: the members in order, indented by two, non-ASCII text as it is, no
newline at the end, and the ``unfinished`` member only on the marker a job is
given before an attempt; and what they raise, and leave behind, for what JSON
or UTF-8 cannot hold.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.gate.errored import (
    CASE_ERRORED,
    CASE_ERRORED_SCHEMA,
    UNFINISHED_REASON,
    read_case_errored,
    write_case_errored,
    write_case_unfinished,
)


def _expected(
    overrides: Any,
    exit_code: Any,
    attempts: Any,
    reason: Any,
    unfinished: Any = False,
) -> str:
    """What the marker held when it was written as one dict."""
    data: dict[str, Any] = {
        "schema": CASE_ERRORED_SCHEMA,
        "overrides": list(overrides),
        "exit_code": exit_code,
        "attempts": attempts,
        "reason": reason,
    }
    if unfinished:
        data["unfinished"] = True
    return json.dumps(data, indent=2, ensure_ascii=False)


def test_the_marker_of_a_job_given_up_byte_for_byte(tmp_path: Path) -> None:
    path = write_case_errored(
        tmp_path / "job" / "3",
        overrides=["+case=3", "scenario.speed=1.5"],
        exit_code=-9,
        attempts=2,
        reason="exit code -9",
    )
    assert path == tmp_path / "job" / "3" / CASE_ERRORED
    assert path.read_bytes() == (
        b"{\n"
        b'  "schema": "autoware_carla_scenario.case_errored/1",\n'
        b'  "overrides": [\n'
        b'    "+case=3",\n'
        b'    "scenario.speed=1.5"\n'
        b"  ],\n"
        b'  "exit_code": -9,\n'
        b'  "attempts": 2,\n'
        b'  "reason": "exit code -9"\n'
        b"}"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert [p.name for p in path.parent.iterdir()] == [CASE_ERRORED]


def test_the_unfinished_marker_byte_for_byte(tmp_path: Path) -> None:
    path = write_case_unfinished(tmp_path, overrides=[], attempt=1)
    assert (
        path.read_bytes()
        == (
            "{\n"
            '  "schema": "autoware_carla_scenario.case_errored/1",\n'
            '  "overrides": [],\n'
            '  "exit_code": null,\n'
            '  "attempts": 1,\n'
            f'  "reason": "{UNFINISHED_REASON}",\n'
            '  "unfinished": true\n'
            "}"
        ).encode()
    )


def test_a_marker_replaces_the_one_there(tmp_path: Path) -> None:
    write_case_unfinished(tmp_path, overrides=["a"], attempt=1)
    path = write_case_errored(
        tmp_path, overrides=["a"], exit_code=None, attempts=1, reason="timed out"
    )
    assert path.read_text(encoding="utf-8") == _expected(["a"], None, 1, "timed out")
    assert '"unfinished"' not in path.read_text(encoding="utf-8")


_ODD_TEXT = [
    "",
    "日本語のオーバーライド",
    'quote " backslash \\ slash /',
    "control \x00\x01\x1f\x7f tab\t newline\n return\r",
    "   separators",
    "emoji \U0001f697",
    "  spaces  ",
    "{}[],:",
]


@pytest.mark.parametrize(
    ("overrides", "exit_code", "attempts", "reason", "unfinished"),
    [
        ([], None, 0, "", False),
        ([], 0, 1, "x", True),
        (_ODD_TEXT, 1, 3, "理由: 終了コード 1", False),
        (_ODD_TEXT, None, 1, _ODD_TEXT[3], True),
        (["+case=1"], 2**80, 2**70, "huge", False),
        (["+case=1"], -(2**80), -1, "negative", False),
        (("tuple", "of", "overrides"), 1, 1, "a tuple", False),
        (iter(["an", "iterator"]), 1, 1, "an iterator", False),
        ("ab", 1, 1, "a str is its characters", False),
        # Values of other types than the signature's, as a caller may pass.
        ([1, 2.5, None, True, ["nested", {"k": "v"}]], True, 1.5, None, 1),
        (["x"], 1.0, float("nan"), float("inf"), "truthy"),
        (["x"], -0.0, float("-inf"), {"a": [1, {"b": None}]}, 0),
        (["x"], 1, 1, "falsy unfinished", ""),
        (["x"], 1, 1, "falsy unfinished", []),
    ],
    ids=[
        "empty",
        "unfinished",
        "odd-text",
        "odd-text-unfinished",
        "huge-ints",
        "negative-ints",
        "tuple",
        "iterator",
        "str-overrides",
        "other-types",
        "floats",
        "zeros",
        "empty-str-unfinished",
        "empty-list-unfinished",
    ],
)
def test_the_marker_is_the_json_of_its_members(
    tmp_path: Path,
    overrides: Any,
    exit_code: Any,
    attempts: Any,
    reason: Any,
    unfinished: Any,
) -> None:
    listed = list(overrides)
    again: Any = (
        iter(listed) if not isinstance(overrides, (list, tuple, str)) else overrides
    )
    path = write_case_errored(
        tmp_path,
        overrides=again,
        exit_code=exit_code,
        attempts=attempts,
        reason=reason,
        unfinished=unfinished,
    )
    text = path.read_bytes().decode("utf-8")
    assert text == _expected(listed, exit_code, attempts, reason, unfinished)
    assert not text.endswith("\n")


def test_the_overrides_are_copied_not_held(tmp_path: Path) -> None:
    overrides = ["a"]
    write_case_errored(
        tmp_path, overrides=overrides, exit_code=1, attempts=1, reason="r"
    )
    assert overrides == ["a"]
    assert read_case_errored(tmp_path / CASE_ERRORED) == {
        "schema": CASE_ERRORED_SCHEMA,
        "overrides": ["a"],
        "exit_code": 1,
        "attempts": 1,
        "reason": "r",
    }


@pytest.mark.parametrize(
    ("overrides", "exit_code", "reason", "error", "message"),
    [
        (
            [object()],
            1,
            "r",
            TypeError,
            "Object of type object is not JSON serializable",
        ),
        (
            ["a"],
            {1, 2},
            "r",
            TypeError,
            "Object of type set is not JSON serializable",
        ),
        (
            # The first member JSON cannot hold is the one reported.
            [b"bytes"],
            {1, 2},
            "r",
            TypeError,
            "Object of type bytes is not JSON serializable",
        ),
        (
            ["a"],
            1,
            {(1, 2): "tuple key"},
            TypeError,
            "keys must be str, int, float, bool or None, not tuple",
        ),
        (
            # A lone surrogate is JSON, but not UTF-8.
            ["\ud800"],
            1,
            "r",
            UnicodeEncodeError,
            "surrogates not allowed",
        ),
        (
            ["a"],
            1,
            "lone \udfff",
            UnicodeEncodeError,
            "surrogates not allowed",
        ),
    ],
    ids=[
        "object-override",
        "set-exit-code",
        "first-member-reported",
        "tuple-key",
        "surrogate-override",
        "surrogate-reason",
    ],
)
def test_what_json_or_utf8_cannot_hold_raises_and_writes_nothing(
    tmp_path: Path,
    overrides: list[Any],
    exit_code: Any,
    reason: Any,
    error: type[Exception],
    message: str,
) -> None:
    job = tmp_path / "job"
    with pytest.raises(error) as raised:
        write_case_errored(
            job, overrides=overrides, exit_code=exit_code, attempts=1, reason=reason
        )
    assert message in str(raised.value)
    # The directory is made first; no marker and no scratch file is left.
    assert job.is_dir()
    assert list(job.iterdir()) == []


def test_a_failed_write_keeps_the_marker_there(tmp_path: Path) -> None:
    before = write_case_unfinished(tmp_path, overrides=["a"], attempt=1).read_bytes()
    with pytest.raises(TypeError):
        write_case_errored(
            tmp_path,
            overrides=[object()],  # type: ignore[list-item]
            exit_code=1,
            attempts=1,
            reason="r",
        )
    assert (tmp_path / CASE_ERRORED).read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [CASE_ERRORED]


def test_an_override_nested_too_deep_raises_recursion_error(tmp_path: Path) -> None:
    deep: Any = "leaf"
    for _ in range(100_000):
        deep = [deep]
    with pytest.raises(RecursionError):
        write_case_errored(
            tmp_path, overrides=[deep], exit_code=1, attempts=1, reason="r"
        )
    assert list(tmp_path.iterdir()) == []


def test_unfinished_keeps_its_signature() -> None:
    with pytest.raises(TypeError):
        write_case_unfinished(Path("."), ["a"], 1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        write_case_errored(Path("."), ["a"], None, 1, "r")  # type: ignore[misc]


def test_the_exit_code_none_is_null(tmp_path: Path) -> None:
    path = write_case_errored(
        tmp_path, overrides=["a"], exit_code=None, attempts=1, reason="r"
    )
    assert '"exit_code": null,\n' in path.read_text(encoding="utf-8")
