"""What ``slurm.manifest`` writes and reads of a SLURM batch's files.

Pinned against files a hand, another version, a crash or a full disk may
have left: values of every JSON type where another is expected, keys missing,
added or repeated, numbers JSON lets Python read in more than one way (NaN,
infinities, ``-0.0``, integers no float holds), text that is not ASCII or not
UTF-8.  The bytes written, the values read back, the messages raised and
logged, and the files left behind are compared whole.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import manifest
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    MANIFEST_VERSIONS,
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    SubmissionRecord,
    manifest_of_text,
    submission_record_of,
    no_job_note,
    read_json,
    read_submission_record,
    readable_result,
    resubmit_target,
    sbatch_in_flight,
    sbatch_timed_out,
    verdict_of_document,
    write_json,
)
from autoware_carla_scenario.slurm.records import JsonMembers

_LOGGER = "autoware_carla_scenario.slurm.manifest"


def _submission(record: dict[str, Any]) -> SubmissionRecord:
    """The submission record of the members of *record*."""
    return submission_record_of(JsonMembers.from_value(record))


#: A document with a value of each kind JSON writes, and the keys json turns
#: into strings (an int, a float, a bool, None).
_DOCUMENT: dict[Any, Any] = {
    "b": 1,
    "a": [1.5, -0.0, math.nan, math.inf, -math.inf, 10**30, 0, -7],
    "é": 'ü \x00"\\/\t',
    "nested": {"t": True, "f": False, "n": None, "empty": {}, "list": []},
    "tuple": (1, "x"),
    1: "int key",
    2.5: "float key",
    False: "bool key",
    None: "none key",
}

_DOCUMENT_BYTES = (
    "{\n"
    '  "b": 1,\n'
    '  "a": [\n'
    "    1.5,\n"
    "    -0.0,\n"
    "    NaN,\n"
    "    Infinity,\n"
    "    -Infinity,\n"
    "    1000000000000000000000000000000,\n"
    "    0,\n"
    "    -7\n"
    "  ],\n"
    '  "é": "ü \\u0000\\"\\\\/\\t",\n'
    '  "nested": {\n'
    '    "t": true,\n'
    '    "f": false,\n'
    '    "n": null,\n'
    '    "empty": {},\n'
    '    "list": []\n'
    "  },\n"
    '  "tuple": [\n'
    "    1,\n"
    '    "x"\n'
    "  ],\n"
    '  "1": "int key",\n'
    '  "2.5": "float key",\n'
    '  "false": "bool key",\n'
    '  "null": "none key"\n'
    "}\n"
).encode()


def _layout(tmp_path: Path) -> BatchLayout:
    return BatchLayout(tmp_path / "batch")


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def umask_022() -> Iterator[None]:
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


def test_write_json_writes_these_bytes(tmp_path: Path) -> None:
    path = tmp_path / "new" / "dir" / "doc.json"
    write_json(path, _DOCUMENT)
    assert path.read_bytes() == _DOCUMENT_BYTES
    assert _mode(path) == 0o600
    assert _names(path.parent) == ["doc.json"]


@pytest.mark.parametrize(
    ("data", "text"),
    [
        ("x", '"x"\n'),
        ("", '""\n'),
        (1.0, "1.0\n"),
        (-0.0, "-0.0\n"),
        (1e308 * 10, "Infinity\n"),
        (1e300 * 10, "1e+301\n"),
        (10**400, "1" + "0" * 400 + "\n"),
        (True, "true\n"),
        (None, "null\n"),
        ([], "[]\n"),
        ({}, "{}\n"),
        ((), "[]\n"),
        ([[[]]], "[\n  [\n    []\n  ]\n]\n"),
        ({"k": {"k": 0.1}}, '{\n  "k": {\n    "k": 0.1\n  }\n}\n'),
        ("日本語 \U0001f600", '"日本語 \U0001f600"\n'),
    ],
)
def test_write_json_writes_any_value_so(tmp_path: Path, data: Any, text: str) -> None:
    path = tmp_path / "doc.json"
    write_json(path, data)
    assert path.read_bytes() == text.encode()


def test_write_json_replaces_a_file(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text("old, and longer than the new one", encoding="utf-8")
    path.chmod(0o644)
    write_json(path, {"x": 1})
    assert path.read_bytes() == b'{\n  "x": 1\n}\n'
    assert _mode(path) == 0o600
    assert _names(tmp_path) == ["doc.json"]


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (
            {"a": object()},
            TypeError,
            "Object of type object is not JSON serializable",
        ),
        ({"a": {1, 2}}, TypeError, "Object of type set is not JSON serializable"),
        (
            {(1, 2): 1},
            TypeError,
            "keys must be str, int, float, bool or None, not tuple",
        ),
        (b"bytes", TypeError, "Object of type bytes is not JSON serializable"),
    ],
)
def test_write_json_leaves_the_old_file_when_it_cannot_write(
    tmp_path: Path, data: Any, error: type[Exception], message: str
) -> None:
    path = tmp_path / "doc.json"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(error) as info:
        write_json(path, data)
    assert str(info.value) == message
    assert path.read_text(encoding="utf-8") == "old"
    assert _names(tmp_path) == ["doc.json"]


def test_write_json_refuses_a_circular_document(tmp_path: Path) -> None:
    looped: list[Any] = []
    looped.append(looped)
    path = tmp_path / "doc.json"
    # str(), not match=: from 3.14.10 json adds a note ("when serializing
    # list item 0"), which match= reads as part of the message.
    with pytest.raises(ValueError) as info:
        write_json(path, looped)
    assert str(info.value) == "Circular reference detected"
    assert _names(tmp_path) == []


def test_write_json_refuses_a_lone_surrogate(tmp_path: Path) -> None:
    # ensure_ascii=False writes the str itself, which UTF-8 cannot encode.
    path = tmp_path / "doc.json"
    with pytest.raises(UnicodeEncodeError):
        write_json(path, {"s": "\ud800"})
    assert _names(tmp_path) == []


def test_write_json_makes_the_directory_before_it_fails(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "doc.json"
    with pytest.raises(TypeError):
        write_json(path, object())
    assert path.parent.is_dir()
    assert _names(path.parent) == []


# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------


def test_read_json_reads_what_json_holds(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_bytes(
        b'{"a": NaN, "b": -Infinity, "c": -0.0, "d": 1E400, '
        b'"e": 100000000000000000000000000001, "a": "again", '
        b'"\\u00e9": "\\ud83d\\ude00", "f": 1.0}'
    )
    data = read_json(path)
    assert list(data) == ["a", "b", "c", "d", "e", "é", "f"]
    assert data["a"] == "again"
    assert data["b"] == -math.inf
    assert math.copysign(1.0, data["c"]) == -1.0
    assert data["d"] == math.inf
    assert data["e"] == 100000000000000000000000000001
    assert isinstance(data["e"], int)
    assert data["é"] == "\U0001f600"
    assert isinstance(data["f"], float)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (b"", json.JSONDecodeError),
        (b"{not json", json.JSONDecodeError),
        (b"\xff\xfe{}", UnicodeDecodeError),
        (b"\xef\xbb\xbf{}", json.JSONDecodeError),
    ],
)
def test_read_json_raises_for_what_is_not_json(
    tmp_path: Path, content: bytes, error: type[Exception]
) -> None:
    path = tmp_path / "doc.json"
    path.write_bytes(content)
    with pytest.raises(error):
        read_json(path)


def test_read_json_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# read_result and read_task: an object, or None
# ---------------------------------------------------------------------------

_NOT_AN_OBJECT = [
    b"",
    b"   ",
    b"{not json",
    b'{"a": 1',
    b"[1, 2]",
    b"[]",
    b'"text"',
    b"null",
    b"true",
    b"1",
    b"NaN",
    b"\xff\xfe{}",
    b'{"a": "\xff"}',
    b"\xef\xbb\xbf{}",
    b'{"a": 1} trailing',
]


@pytest.mark.parametrize("content", _NOT_AN_OBJECT)
def test_read_result_and_read_task_are_none_for_what_is_no_object(
    tmp_path: Path, content: bytes
) -> None:
    layout = _layout(tmp_path)
    for path in (layout.result(3), layout.task(1, 2)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert layout.read_result(3) is None
    assert layout.read_task(1, 2) is None


def test_read_result_and_read_task_are_none_when_missing(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    assert layout.read_result(0) is None
    assert layout.read_task(0, 0) is None
    # A directory where the file would be is no file either.
    layout.result(0).mkdir(parents=True)
    layout.task(0, 0).mkdir(parents=True)
    assert layout.read_result(0) is None
    assert layout.read_task(0, 0) is None


def test_read_result_and_read_task_read_the_object_as_json_holds_it(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    content = (
        b'{"z": 1, "result": {"passed": true, "elapsed_seconds": NaN, '
        b'"message": "\xc3\xbc"}, "z": [1, 2.0, -0.0, Infinity], '
        b'"big": 123456789012345678901234567890, "extra": {"k": null}}'
    )
    for path in (layout.result(0), layout.task(0, 0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    outcome = layout.read_result(0)
    assert outcome is not None
    task = layout.read_task(0, 0)
    assert task is not None
    assert (
        outcome.members.keys() == task.members.keys() == ["z", "result", "big", "extra"]
    )
    for record in (
        {key: task.get(key) for key in task.members.keys()},
        {key: outcome.get(key) for key in outcome.members.keys()},
    ):
        assert record["z"][:2] == [1, 2.0]
        assert isinstance(record["z"][1], float)
        assert math.copysign(1.0, record["z"][2]) == -1.0
        assert record["z"][3] == math.inf
        assert math.isnan(record["result"]["elapsed_seconds"])
        assert record["result"]["message"] == "ü"
        assert record["big"] == 123456789012345678901234567890
        assert record["extra"] == {"k": None}


def test_read_result_raises_for_a_document_nested_too_deeply(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.result(0).parent.mkdir(parents=True)
    depth = 200_000
    layout.result(0).write_text('{"a": ' + "[" * depth + "]" * depth + "}")
    with pytest.raises(RecursionError):
        layout.read_result(0)


def test_read_result_of_an_integer_too_long_to_read(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.result(0).parent.mkdir(parents=True)
    layout.result(0).write_text('{"a": ' + "1" * 5000 + "}")
    if hasattr(sys, "get_int_max_str_digits"):
        # Python refuses to read it (a ValueError), so there is no record.
        assert layout.read_result(0) is None
    else:  # pragma: no cover - a Python without the limit
        record = layout.read_result(0)
        assert record is not None
        assert record.get("a") == int("1" * 5000)


# ---------------------------------------------------------------------------
# read_submission_record and submission_record
# ---------------------------------------------------------------------------


def _warnings(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    return [(r.name, r.levelno, r.getMessage()) for r in caplog.records]


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (b"", "Expecting value: line 1 column 1 (char 0)"),
        (
            b"{not json",
            "Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)",
        ),
        (
            b"\xef\xbb\xbf{}",
            "Unexpected UTF-8 BOM (decode using utf-8-sig): "
            "line 1 column 1 (char 0)",
        ),
        (
            b"\xff",
            "'utf-8' codec can't decode byte 0xff in position 0: " "invalid start byte",
        ),
        (b'{"a": 1} {', "Extra data: line 1 column 10 (char 9)"),
    ],
)
def test_a_submission_record_that_is_not_json_is_none_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes, why: str
) -> None:
    layout = _layout(tmp_path)
    path = layout.submission(4)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert read_submission_record(path) is None
        assert layout.submission_record(4) is None
    message = f"Ignoring {path}: it cannot be read as a submission record ({why})"
    assert _warnings(caplog) == [(_LOGGER, logging.WARNING, message)] * 2


@pytest.mark.parametrize(
    "content", [b"[]", b"[{}]", b'"x"', b"null", b"0", b"false", b"NaN"]
)
def test_a_submission_record_that_is_no_object_is_none_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: bytes
) -> None:
    layout = _layout(tmp_path)
    path = layout.submission(0)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert layout.submission_record(0) is None
    message = (
        f"Ignoring {path}: it cannot be read as a submission record "
        "(it holds JSON that is not an object)"
    )
    assert _warnings(caplog) == [(_LOGGER, logging.WARNING, message)]


def test_a_missing_submission_record_is_none_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = _layout(tmp_path)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert layout.submission_record(0) is None
        assert read_submission_record(tmp_path / "nowhere" / "0.json") is None
    assert caplog.records == []


def test_a_directory_as_a_submission_record_is_none_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = _layout(tmp_path)
    path = layout.submission(0)
    path.mkdir(parents=True)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert layout.submission_record(0) is None
    message = (
        f"Ignoring {path}: it cannot be read as a submission record "
        f"([Errno 21] Is a directory: '{path}')"
    )
    assert _warnings(caplog) == [(_LOGGER, logging.WARNING, message)]


def test_a_submission_record_reads_as_json_holds_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = _layout(tmp_path)
    path = layout.submission(2)
    path.parent.mkdir(parents=True)
    path.write_bytes(
        b'{"index": 2, "job_id": null, "index": "two", "submitted_at": 1E400, '
        b'"tasks": [[0, 1]], "x": {"\\u00e9": -0.0}}'
    )
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        record = layout.submission_record(2)
    assert caplog.records == []
    assert record is not None
    assert {key: record.get(key) for key in record.members.keys()} == {
        "index": "two",
        "job_id": None,
        "submitted_at": math.inf,
        "tasks": [[0, 1]],
        "x": {"é": -0.0},
    }
    assert record.members.keys() == ["index", "job_id", "submitted_at", "tasks", "x"]
    assert math.copysign(1.0, record["x"]["é"]) == -1.0


def test_read_submission_is_none_for_json_no_object_and_raises_as_read_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(FileNotFoundError):
        layout.read_submission(0)
    layout.submission(0).parent.mkdir(parents=True)
    layout.submission(0).write_text("[1, NaN]")
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        assert layout.read_submission(0) is None
    assert caplog.records == []
    layout.submission(1).write_text("{")
    with pytest.raises(json.JSONDecodeError):
        layout.read_submission(1)
    layout.submission(2).write_text('{"job_id": 7, "chunks": [[NaN]]}')
    record = layout.read_submission(2)
    assert record is not None
    assert record["job_id"] == 7 and math.isnan(record.get("chunks")[0][0])
    with pytest.raises(KeyError):
        record["cases"]


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------


def _write_manifest(layout: BatchLayout, content: str | bytes) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode()
    layout.manifest.write_bytes(content)


def _versions_known() -> str:
    return ", ".join(str(v) for v in MANIFEST_VERSIONS)


def test_the_versions_read() -> None:
    assert MANIFEST_VERSION == 2
    assert MANIFEST_VERSIONS == (1, 2)
    assert _versions_known() == "1, 2"


def test_a_batch_without_a_manifest_is_no_slurm_batch(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(FileNotFoundError) as info:
        layout.read_manifest()
    assert type(info.value) is FileNotFoundError
    assert str(info.value) == (
        f"{layout.batch_dir} holds no SLURM batch (no {layout.manifest})"
    )
    layout.manifest.mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as again:
        layout.read_manifest()
    assert str(again.value) == str(info.value)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("", json.JSONDecodeError),
        ("{", json.JSONDecodeError),
        ('{"version": 1} x', json.JSONDecodeError),
        (b"\xff{}", UnicodeDecodeError),
    ],
)
def test_a_manifest_that_is_not_json_raises_as_json_does(
    tmp_path: Path, content: str | bytes, error: type[Exception]
) -> None:
    layout = _layout(tmp_path)
    _write_manifest(layout, content)
    with pytest.raises(error) as info:
        layout.read_manifest()
    assert type(info.value) is error


@pytest.mark.parametrize("content", ["[]", "[1]", '"x"', "null", "2", "true", "NaN"])
def test_a_manifest_that_is_no_object_is_refused(tmp_path: Path, content: str) -> None:
    layout = _layout(tmp_path)
    _write_manifest(layout, content)
    with pytest.raises(ValueError) as info:
        layout.read_manifest()
    assert type(info.value) is ValueError
    assert str(info.value) == (
        f"{layout.manifest} is not a SLURM batch manifest "
        "(it holds JSON that is not an object)"
    )


@pytest.mark.parametrize(
    ("version", "shown"),
    [
        (None, "None"),
        ("missing", "None"),
        ("true", "True"),
        ("false", "False"),
        ("0", "0"),
        ("3", "3"),
        ("-1", "-1"),
        ("1.5", "1.5"),
        ("0.0", "0.0"),
        ("-0.0", "-0.0"),
        ("NaN", "nan"),
        ("Infinity", "inf"),
        ("-Infinity", "-inf"),
        ("1E400", "inf"),
        ("100000000000000000000000000001", "100000000000000000000000000001"),
        ('"1"', "1"),
        ('"2"', "2"),
        ('""', ""),
        ("[1]", "[1]"),
        ("[]", "[]"),
        ('{"v": 1}', "{'v': 1}"),
        ('"\\u00e9"', "é"),
    ],
)
def test_a_manifest_of_a_version_not_read_is_refused(
    tmp_path: Path, version: str | None, shown: str
) -> None:
    layout = _layout(tmp_path)
    if version is None:
        _write_manifest(layout, '{"version": null, "cases": []}')
    elif version == "missing":
        _write_manifest(layout, '{"cases": []}')
    else:
        _write_manifest(layout, f'{{"cases": [], "version": {version}}}')
    with pytest.raises(ValueError) as info:
        layout.read_manifest()
    assert type(info.value) is ValueError
    assert str(info.value) == (
        f"{layout.manifest} is a version {shown} manifest; this "
        "autoware_carla_scenario reads version(s) 1, 2"
    )


def test_a_manifest_whose_version_is_too_long_to_read_raises(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_manifest(layout, '{"version": ' + "1" * 5000 + "}")
    if hasattr(sys, "get_int_max_str_digits"):
        with pytest.raises(ValueError) as info:
            layout.read_manifest()
        assert "manifest" not in str(info.value)
    else:  # pragma: no cover - a Python without the limit
        with pytest.raises(ValueError, match="is a version 1111"):
            layout.read_manifest()


@pytest.mark.parametrize("version", ["1", "2", "1.0", "2.0", "1e0", "2E0"])
def test_a_manifest_of_a_version_read_reads_as_json_holds_it(
    tmp_path: Path, version: str
) -> None:
    layout = _layout(tmp_path)
    _write_manifest(
        layout,
        '{"cases": [{"index": 0, "config": {"a": NaN, "b": [1, -0.0]}}], '
        f'"version": {version}, "batch_dir": "/x/\\u00e9", "extra": {{"k": 1}}, '
        '"cases": [{"index": 1}], "big": 123456789012345678901234567890}',
    )
    data = layout.read_manifest()
    assert data.keys() == ["cases", "version", "batch_dir", "extra", "big"]
    assert data.members.keys() == ["version", "batch_dir"]
    assert data["cases"] == [{"index": 1}]
    assert [case.index() for case in data.cases()] == [1]
    assert data["version"] == json.loads(version)
    assert type(data["version"]) is type(json.loads(version))
    assert data["batch_dir"] == "/x/é"
    assert data["extra"] == {"k": 1}
    assert data["big"] == 123456789012345678901234567890


def test_read_manifest_gives_a_new_manifest_each_time(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_manifest(layout, '{"version": 1, "cases": []}')
    first = layout.read_manifest()
    first["cases"].append("changed")
    again = layout.read_manifest()
    assert again["cases"] == []
    assert again.keys() == ["version", "cases"]
    assert again.cases() == []


# ---------------------------------------------------------------------------
# reserve_submission
# ---------------------------------------------------------------------------

_SUBMISSION: dict[str, Any] = {
    "job_id": None,
    "cases": [[0, 1], [2]],
    "submitted_at": 1.5,
    "note": "ü  ",
    "nan": math.nan,
    "big": 10**30,
}

_SUBMISSION_BYTES = (
    "{\n"
    '  "job_id": null,\n'
    '  "cases": [\n'
    "    [\n"
    "      0,\n"
    "      1\n"
    "    ],\n"
    "    [\n"
    "      2\n"
    "    ]\n"
    "  ],\n"
    '  "submitted_at": 1.5,\n'
    '  "note": "ü  ",\n'
    '  "nan": NaN,\n'
    '  "big": 1000000000000000000000000000000,\n'
    '  "index": 0\n'
    "}\n"
).encode()


def test_reserve_submission_creates_the_first_record(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    record = dict(_SUBMISSION)
    assert layout.reserve_submission(record) == 0
    assert record["index"] == 0
    assert list(record) == [*_SUBMISSION, "index"]
    path = layout.submission(0)
    assert path.read_bytes() == _SUBMISSION_BYTES
    assert _mode(path) == 0o600
    assert _names(path.parent) == ["0.json"]


def test_reserve_submission_keeps_an_index_key_where_it_is(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    layout.submission(0).parent.mkdir(parents=True)
    layout.submission(0).write_text("{}")
    layout.submission(1).write_text("not even json")
    record = {"index": "old", "job_id": "7"}
    assert layout.reserve_submission(record) == 2
    assert record == {"index": 2, "job_id": "7"}
    assert layout.submission(2).read_bytes() == (
        b'{\n  "index": 2,\n  "job_id": "7"\n}\n'
    )
    assert layout.submission(1).read_text() == "not even json"
    assert _names(layout.submission(0).parent) == ["0.json", "1.json", "2.json"]


def _link_then(
    monkeypatch: pytest.MonkeyPatch, before: Callable[[str, Path], None] | None
) -> list[tuple[str, str]]:
    """Wrap ``os.link``: run *before* (src, dst) ahead of the real call."""
    calls: list[tuple[str, str]] = []
    real = os.link

    def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        calls.append((os.path.basename(src), os.path.basename(dst)))
        if before is not None:
            before(str(src), Path(dst))
        real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)
    return calls


def test_reserve_submission_passes_over_an_index_taken_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)

    def other_submitter(src: str, dst: Path) -> None:
        if dst.name == "0.json":
            dst.write_text("theirs")

    calls = _link_then(monkeypatch, other_submitter)
    record: dict[str, Any] = {"job_id": None}
    assert layout.reserve_submission(record) == 1
    assert record == {"job_id": None, "index": 1}
    assert [dst for _, dst in calls] == ["0.json", "1.json"]
    assert all(src.endswith(".tmp") for src, _ in calls)
    assert layout.submission(0).read_text() == "theirs"
    assert layout.submission(1).read_bytes() == (
        b'{\n  "job_id": null,\n  "index": 1\n}\n'
    )
    assert _names(layout.submission(0).parent) == ["0.json", "1.json"]


def test_reserve_submission_takes_a_link_whose_reply_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NFS retries a link whose reply was lost: it was made, and the retry
    # reports EEXIST. The temporary file's second link tells.
    layout = _layout(tmp_path)
    real = os.link

    def link(src: Any, dst: Any) -> None:
        real(src, dst)
        raise FileExistsError(errno.EEXIST, "File exists")

    monkeypatch.setattr(os, "link", link)
    record: dict[str, Any] = {}
    assert layout.reserve_submission(record) == 0
    assert layout.submission(0).read_bytes() == b'{\n  "index": 0\n}\n'
    assert _names(layout.submission(0).parent) == ["0.json"]


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP])
def test_reserve_submission_without_hard_links_creates_the_file_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, umask_022: None, code: int
) -> None:
    layout = _layout(tmp_path)

    def link(src: Any, dst: Any) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "link", link)
    record = dict(_SUBMISSION)
    assert layout.reserve_submission(record) == 0
    path = layout.submission(0)
    assert path.read_bytes() == _SUBMISSION_BYTES
    assert _mode(path) == 0o600
    assert _names(path.parent) == ["0.json"]


class _LinkError(OSError):
    """An OSError of another class, as a library may raise."""


@pytest.mark.parametrize(
    ("make", "fallback"),
    [
        # Python compares the code by ==: a subclass, a bool or a float equal
        # to a code is taken; a code of another type (a str) and a code given
        # by keyword-less construction of more values are read as Python
        # reads them.
        (lambda: PermissionError(errno.EPERM, "Operation not permitted"), True),
        (lambda: _LinkError(errno.ENOTSUP, "Not supported"), True),
        (lambda: OSError(float(errno.EOPNOTSUPP), "Not supported"), True),
        (lambda: OSError(True, "one"), True),
        (lambda: OSError(str(errno.EPERM), "a str"), False),
        (lambda: OSError(errno.EPERM, "x", "file", 0, "other"), True),
        (lambda: OSError(errno.EPERM), False),
        (lambda: _LinkError(), False),
    ],
    ids=["permission-error", "subclass", "float", "bool", "str", "five", "one", "none"],
)
def test_reserve_submission_falls_back_by_the_code_python_compares(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    umask_022: None,
    make: Callable[[], OSError],
    fallback: bool,
) -> None:
    # Pins _link_exclusive()'s test of the link error's errno (#2124), which
    # the library check compiles: what Python reads of it is unchanged.
    layout = _layout(tmp_path)
    error = make()

    def link(src: Any, dst: Any) -> None:
        raise error

    monkeypatch.setattr(os, "link", link)
    record: dict[str, Any] = {}
    if fallback:
        assert layout.reserve_submission(record) == 0
        assert layout.submission(0).read_bytes() == b'{\n  "index": 0\n}\n'
        assert _mode(layout.submission(0)) == 0o600
        assert _names(layout.submission(0).parent) == ["0.json"]
    else:
        with pytest.raises(OSError) as info:
            layout.reserve_submission(record)
        assert info.value is error
        assert _names(layout.submission(0).parent) == []


def test_reserve_submission_without_hard_links_passes_over_a_taken_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    seen: list[str] = []

    def link(src: Any, dst: Any) -> None:
        seen.append(Path(dst).name)
        if Path(dst).name == "0.json":
            Path(dst).write_text("theirs")
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", link)
    record: dict[str, Any] = {"a": 1}
    assert layout.reserve_submission(record) == 1
    assert seen == ["0.json", "1.json"]
    assert layout.submission(0).read_text() == "theirs"
    assert layout.submission(1).read_bytes() == b'{\n  "a": 1,\n  "index": 1\n}\n'
    assert _names(layout.submission(0).parent) == ["0.json", "1.json"]


def test_reserve_submission_without_hard_links_raises_any_other_open_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only EEXIST passes over an index: any other error of the O_EXCL open
    # is raised, and the temporary file is removed.
    layout = _layout(tmp_path)
    error = PermissionError(errno.EACCES, "Permission denied")
    real_open = os.open

    def link(src: Any, dst: Any) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path).name == "0.json":
            raise error
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "open", open_)
    record: dict[str, Any] = {}
    with pytest.raises(PermissionError) as info:
        layout.reserve_submission(record)
    assert info.value is error
    assert record == {"index": 0}
    assert _names(layout.submission(0).parent) == []


@pytest.mark.parametrize("code", [errno.EACCES, errno.EXDEV, errno.ENOSPC, None])
def test_reserve_submission_raises_any_other_link_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int | None
) -> None:
    layout = _layout(tmp_path)
    error = OSError("no errno") if code is None else OSError(code, os.strerror(code))

    def link(src: Any, dst: Any) -> None:
        raise error

    monkeypatch.setattr(os, "link", link)
    record: dict[str, Any] = {}
    with pytest.raises(OSError) as info:
        layout.reserve_submission(record)
    assert info.value is error
    assert record == {"index": 0}
    assert _names(layout.submission(0).parent) == []


def test_reserve_submission_of_what_json_cannot_write_raises(
    tmp_path: Path,
) -> None:
    # The record is serialized after its temporary file is made, and outside
    # what removes it: the temporary file stays.
    layout = _layout(tmp_path)
    record: dict[str, Any] = {"x": object()}
    with pytest.raises(TypeError) as info:
        layout.reserve_submission(record)
    assert str(info.value) == "Object of type object is not JSON serializable"
    assert record["index"] == 0
    left = _names(layout.submission(0).parent)
    assert len(left) == 1
    assert left[0].startswith(".0.json.") and left[0].endswith(".tmp")


# ---------------------------------------------------------------------------
# What a submission record and a result record say
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("submission", "now", "expected"),
    [
        ({"job_id": "1", "submitted_at": 100.0}, 100.0, False),
        ({"job_id": None, "submitted_at": 100.0}, 100.0, True),
        ({"job_id": "", "submitted_at": 100.0}, 699.9, True),
        ({"job_id": 0, "submitted_at": 100}, 700.0, False),
        ({"submitted_at": True}, 1.0, True),
        ({"submitted_at": "100"}, 100.0, False),
        ({"submitted_at": None}, 100.0, False),
        ({}, 100.0, False),
        ({"submitted_at": math.nan}, 100.0, False),
        ({"submitted_at": math.inf}, 100.0, True),
        ({"submitted_at": -math.inf}, 100.0, False),
        ({"job_id": [], "submitted_at": 0}, 599.0, True),
        ({"job_id": [0], "submitted_at": 0}, 599.0, False),
    ],
)
def test_sbatch_in_flight(
    submission: dict[str, Any], now: float, expected: bool
) -> None:
    assert sbatch_in_flight(_submission(submission), now) is expected


def test_sbatch_in_flight_of_an_integer_no_float_holds_raises() -> None:
    with pytest.raises(OverflowError):
        sbatch_in_flight(_submission({"submitted_at": 10**400}), 0.0)


def test_sbatch_in_flight_reads_the_clock_when_no_time_is_given() -> None:
    import time

    assert sbatch_in_flight(_submission({"submitted_at": time.time()})) is True
    old = time.time() - SBATCH_IN_FLIGHT_SECONDS - 1
    assert sbatch_in_flight(_submission({"submitted_at": old})) is False


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ({"sbatch_timed_out": True}, True),
        ({"sbatch_timed_out": True, "job_id": "5"}, False),
        ({"sbatch_timed_out": 1}, False),
        ({"sbatch_timed_out": "true"}, False),
        ({"sbatch_timed_out": False}, False),
        ({}, False),
        ({"sbatch_timed_out": True, "job_id": ""}, True),
    ],
)
def test_sbatch_timed_out(submission: dict[str, Any], expected: bool) -> None:
    assert sbatch_timed_out(_submission(submission)) is expected


@pytest.mark.parametrize(
    ("submission", "note"),
    [
        (
            {"sbatch_timed_out": True, "sbatch_timed_out_at": "12:00"},
            "no job id: sbatch timed out at 12:00; the job may exist all the "
            "same: check `squeue --me`",
        ),
        (
            {"sbatch_timed_out": True, "sbatch_timed_out_at": ""},
            "no job id: sbatch timed out; the job may exist all the same: "
            "check `squeue --me`",
        ),
        (
            {"sbatch_timed_out": True, "sbatch_timed_out_at": 1.5},
            "no job id: sbatch timed out at 1.5; the job may exist all the "
            "same: check `squeue --me`",
        ),
        ({"submitted_at": 100.0}, "no job id yet (sbatch in flight)"),
        (
            {"submitted_at": -500.0},
            "no job id: its submitter stopped before sbatch returned, more "
            "than 600 s ago (a stale record)",
        ),
    ],
)
def test_no_job_note(submission: dict[str, Any], note: str) -> None:
    assert no_job_note(_submission(submission), 100.0) == note


_STATUS = {"label": "l", "satisfied": True, "message": "m"}


@pytest.mark.parametrize(
    ("result", "readable"),
    [
        ({"message": "", "elapsed_seconds": 0}, True),
        ({"message": "", "elapsed_seconds": 1.5}, True),
        ({"message": "", "elapsed_seconds": math.nan}, True),
        ({"message": "", "elapsed_seconds": math.inf}, True),
        ({"message": "", "elapsed_seconds": 10**400}, False),
        ({"message": "", "elapsed_seconds": True}, False),
        ({"message": "", "elapsed_seconds": "1"}, False),
        ({"message": "", "elapsed_seconds": None}, False),
        ({"message": "", "elapsed_seconds": 1}, True),
        ({"elapsed_seconds": 1}, False),
        ({"message": 1, "elapsed_seconds": 1}, False),
        ({"message": "", "elapsed_seconds": 1, "condition_statuses": None}, True),
        ({"message": "", "elapsed_seconds": 1, "condition_statuses": []}, True),
        ({"message": "", "elapsed_seconds": 1, "condition_statuses": [_STATUS]}, True),
        ({"message": "", "elapsed_seconds": 1, "condition_statuses": "s"}, False),
        ({"message": "", "elapsed_seconds": 1, "condition_statuses": {}}, False),
        (
            {
                "message": "",
                "elapsed_seconds": 1,
                "condition_statuses": [{**_STATUS, "satisfied": 1}],
            },
            False,
        ),
        (
            {
                "message": "",
                "elapsed_seconds": 1,
                "condition_statuses": [{**_STATUS, "label": None}],
            },
            False,
        ),
        (
            {
                "message": "",
                "elapsed_seconds": 1,
                "condition_statuses": [{"label": "l", "satisfied": False}],
            },
            False,
        ),
        ({"message": "", "elapsed_seconds": 1, "failure": None}, True),
        ({"message": "", "elapsed_seconds": 1, "failure": {}}, True),
        ({"message": "", "elapsed_seconds": 1, "failure": []}, False),
        ({"message": "", "elapsed_seconds": 1, "failure": {"at": "x"}}, False),
        ({"message": "", "elapsed_seconds": 1, "failure": {"at": 10**400}}, False),
        (
            {"message": "", "elapsed_seconds": 1, "failure": {"also_triggered": 3}},
            False,
        ),
        (
            {"message": "", "elapsed_seconds": 1, "failure": {"category": "nope"}},
            True,
        ),
        ([], False),
        (None, False),
        ("x", False),
    ],
)
def test_readable_result(result: Any, readable: bool) -> None:
    if isinstance(result, dict):
        assert readable_result(JsonMembers.from_value(result)) is readable
    # A result that is no object holds no verdict.
    record = {
        "result": {"passed": True, **result} if isinstance(result, dict) else result
    }
    assert verdict_of_document(record) is (True if readable else None)


@pytest.mark.parametrize(
    ("record", "verdict"),
    [
        ({"result": {"passed": True, "message": "", "elapsed_seconds": 1}}, True),
        ({"result": {"passed": False, "message": "", "elapsed_seconds": 1}}, False),
        ({"result": {"passed": 1, "message": "", "elapsed_seconds": 1}}, None),
        ({"result": {"passed": "true", "message": "", "elapsed_seconds": 1}}, None),
        ({"result": {"passed": True, "elapsed_seconds": 1}}, None),
        ({"result": {"message": "", "elapsed_seconds": 1}}, None),
        ({"result": []}, None),
        ({"result": None}, None),
        ({}, None),
        (None, None),
        ([], None),
    ],
)
def test_recorded_verdict(record: Any, verdict: bool | None) -> None:
    assert verdict_of_document(record) is verdict


@pytest.mark.parametrize(
    ("record", "target"),
    [
        (
            {"batch_dir_relative": "o/d/t", "batch_dir": "/submit/o/d/t"},
            "/submit/o/d/t",
        ),
        ({"batch_dir_relative": "", "batch_dir": "/submit/o/d/t"}, "LAYOUT"),
        ({"batch_dir_relative": "o/d/t", "batch_dir": ""}, "LAYOUT"),
        ({"batch_dir_relative": "o/d/t", "batch_dir": 5}, "5"),
        ({"batch_dir": "/submit/o/d/t"}, "LAYOUT"),
        ({}, "LAYOUT"),
    ],
)
def test_resubmit_target(tmp_path: Path, record: dict[str, Any], target: str) -> None:
    layout = _layout(tmp_path)
    expected = str(layout.batch_dir) if target == "LAYOUT" else target
    manifest = manifest_of_text(json.dumps(record))
    assert manifest is not None
    assert resubmit_target(layout, manifest) == expected


def test_config_files_names_the_files_of_a_config(tmp_path: Path) -> None:
    present = tmp_path / "a.txt"
    present.write_text("a")
    cache: dict[str, str] = {}
    found = manifest.config_files(
        {
            "x": str(present),
            "y": [str(present), "relative.txt", str(tmp_path / "missing")],
            "z": {"w": str(tmp_path)},
            "n": 1,
        },
        cache,
    )
    digest = manifest.file_sha256(present)
    assert found == {"x": digest, "y[0]": digest}
    assert cache == {
        str(present): digest,
        str(tmp_path / "missing"): "",
        str(tmp_path): "",
    }
    assert manifest.config_files(str(present)) == {"<root>": digest}
    assert manifest.config_files(str(present), prefix="p") == {"p": digest}
