"""How a SLURM batch's task status (``slurm/tasks/<k>-<t>.json``) is read.

Pinned against the statuses a task (``slurm/worker.py``'s ``_Task.update``),
a hand, another version, a crash or a killed task may have left: each key the
task writes (``submission``, ``task``, ``job_id``, ``array_job_id``, ``host``,
``pid``, ``state``, ``started``, ``cases``, ``done``, ``current``, and those
it adds on first use, in the order of its run: ``updated``, ``ros_domains``,
``ports``, ``finished``, ``error``, ``traceback``, ``peak_*_rss_mb``),
missing or of every JSON type -- NaN, infinities, a number no float holds, an
integer no float holds exactly, non-ASCII text -- keys in another order or
named twice, and files that hold no status.  Each is compared with what the
readers of a batch made of the status when it was read as a ``dict`` (the
oracles below): what ``read_task`` reads, the task's line of ``status``, the
message ``collect`` gives a case with no outcome (and what it prints and
writes), and the viewer's batch status (its tasks, and the state of a case
with no outcome).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import cli, collect as collect_mod
from autoware_carla_scenario.slurm.collect import (
    _UNFINISHED_TASK_STATES,
    _advice,
    _last_submission_of,
    collect,
)
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    BatchLayout,
    no_job_note,
    sbatch_in_flight,
    sbatch_timed_out,
    write_json,
)
from autoware_carla_scenario.slurm.queue import ACTIVE_STATES

# -- The corpus -------------------------------------------------------------------

#: A member left out of its object.
_ABSENT = "<absent>"

#: A value of every JSON type, as its text in the file.
_VALUES: list[str] = [
    _ABSENT,
    "null",
    "true",
    "false",
    "0",
    "1",
    "-1",
    "2",
    "1.0",
    "1.5",
    "-0.0",
    "NaN",
    "Infinity",
    "-Infinity",
    "1e400",
    "123456789012345678901234567890",
    "1" + "0" * 400,
    '""',
    '"text"',
    '"1"',
    '"\\u00fc ber \\u2603"',
    '"running"',
    '"done"',
    "[]",
    "[1]",
    '["a", "b"]',
    "{}",
    '{"rpc": 2000}',
    '{"k": "v"}',
]

#: The keys a task's status holds, in the order the task writes them.
_KEYS = (
    "submission",
    "task",
    "job_id",
    "array_job_id",
    "host",
    "pid",
    "state",
    "started",
    "cases",
    "done",
    "current",
    "updated",
    "ros_domains",
    "ports",
    "finished",
    "error",
    "traceback",
    "peak_rss_mb",
    "peak_worker_rss_mb",
    "peak_children_rss_mb",
)


def _object(members: dict[str, str]) -> str:
    """The text of the object of *members* (each a value's JSON text)."""
    return (
        "{"
        + ", ".join(
            f"{json.dumps(key)}: {value}"
            for key, value in members.items()
            if value != _ABSENT
        )
        + "}"
    )


def _members(**values: str) -> dict[str, str]:
    """A task's status that failed while running case 1, as the task writes
    it, with *values* in place of (or after) its own."""
    members = {
        "submission": "0",
        "task": "0",
        "job_id": '"70"',
        "array_job_id": '"70"',
        "host": '"node\\u00fc"',
        "pid": "4242",
        "state": '"running"',
        "started": '"2026-11-12T00:00:00"',
        "cases": "[0, 1, 2]",
        "done": "[0]",
        "current": "1",
        "updated": '"2026-11-12T00:01:00"',
        "ros_domains": '{"0": 11}',
        "ports": '{"rpc": 2000, "streaming": 2001}',
        "finished": '"2026-11-12T00:02:00"',
        "error": '"RuntimeError: boom \\u00fc"',
        "traceback": '"Traceback ...\\n"',
        "peak_rss_mb": "12.5",
        "peak_worker_rss_mb": "10.0",
        "peak_children_rss_mb": "2.5",
    }
    members.update(values)
    return members


def _status(**values: str) -> str:
    return _object(_members(**values))


def _corpus() -> list[tuple[str, str]]:
    """(name, the text of task 0's status)."""
    corpus: list[tuple[str, str]] = []
    for key in _KEYS:
        for value in _VALUES:
            corpus.append((f"{key}-{value}", _status(**{key: value})))
    for value in _VALUES:
        corpus.append((f"ports-rpc-{value}", _status(ports=_object({"rpc": value}))))
    for state in ("starting", "waiting", "building", "running", "done", "failed"):
        for error in (_ABSENT, '""', '"x"', "0", "[1]"):
            corpus.append(
                (
                    f"state-{state}-error-{error}",
                    _status(state=json.dumps(state), error=error),
                )
            )
    # What a task writes at each step of its run, in its order.
    started = {key: value for key, value in _members().items() if key in _KEYS[:11]}
    corpus += [
        (
            "just-started",
            _object({**started, "state": '"starting"', "current": "null"}),
        ),
        (
            "waiting",
            _object({**started, "state": '"waiting"', "updated": '"u"'}),
        ),
        (
            "building",
            _object(
                {
                    **started,
                    "state": '"building"',
                    "updated": '"u"',
                    "ros_domains": '{"0": 11}',
                    "ports": '{"rpc": 2000}',
                }
            ),
        ),
        (
            "ports-before-domains",
            _object(
                {
                    **started,
                    "state": '"running"',
                    "updated": '"u"',
                    "ports": '{"rpc": 2000}',
                    "ros_domains": '{"0": 11}',
                }
            ),
        ),
        (
            "done-with-rss",
            _object(
                {
                    **started,
                    "state": '"done"',
                    "current": "null",
                    "done": "[0, 1, 2]",
                    "updated": '"u"',
                    "finished": '"f"',
                    "peak_rss_mb": "12.5",
                    "peak_worker_rss_mb": "10.0",
                    "peak_children_rss_mb": "2.5",
                }
            ),
        ),
        (
            "failed-before-ports",
            _object(
                {
                    **started,
                    "state": '"failed"',
                    "updated": '"u"',
                    "finished": '"f"',
                    "error": '"OSError: no"',
                    "traceback": '"t"',
                    "peak_rss_mb": "0.0",
                }
            ),
        ),
        (
            "cancelled",
            _object(
                {
                    **started,
                    "state": '"cancelled"',
                    "updated": '"u"',
                    "finished": '"f"',
                    "error": '"SIGTERM"',
                }
            ),
        ),
        (
            "reversed",
            _object(dict(reversed(list(_members().items())))),
        ),
        ("only-state", '{"state": "running"}'),
        ("only-error", '{"error": "e"}'),
        ("empty-object", "{}"),
        (
            "extra-keys",
            '{"z": [1, {"y": NaN}], "state": "failed", "error": "e", '
            '"\\u00fc": [Infinity, -0.0], "a": null}',
        ),
    ]
    # Keys named twice: the first place and the last value.
    corpus += [
        ("twice-state", '{"state": "done", "error": "e", "state": "running"}'),
        ("twice-error", '{"error": "a", "state": "failed", "error": ""}'),
        ("twice-current", '{"current": 1, "state": "running", "current": 2}'),
        ("twice-host", '{"host": "a", "host": ["b"]}'),
    ]
    # Files that hold no status: the task has none.
    corpus += [
        ("not-json", "{not json"),
        ("truncated", _status()[:-7]),
        ("empty", ""),
        ("array", "[" + _status() + "]"),
        ("string", '"x"'),
        ("number", "1"),
        ("null", "null"),
        ("trailing", _status() + " x"),
        ("integer-too-long", '{"state": ' + "1" * 5000 + "}"),
    ]
    return corpus


_CORPUS = _corpus()
_BYTES: dict[str, bytes] = {
    "bom": b"\xef\xbb\xbf" + _status().encode(),
    "not-utf-8": _status().encode().replace(b"node", b"no\xffde"),
    "latin-1": _status(host='"n\\u00fc"').replace("\\u00fc", "ü").encode("latin-1"),
}


def _all_contents() -> list[tuple[str, bytes]]:
    return [(name, text.encode("utf-8")) for name, text in _CORPUS] + list(
        _BYTES.items()
    )


_CONTENTS = _all_contents()
_IDS = [name for name, _ in _CONTENTS]


def _old_status(content: bytes | None) -> dict[str, Any] | None:
    """The status as it was read: a dict, or None."""
    if content is None:
        return None
    try:
        data = json.loads(content.decode("utf-8"))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# -- The batch ------------------------------------------------------------------


def _passing_record(index: int) -> str:
    return json.dumps(
        {
            "index": index,
            "label": f"s#{index + 1}",
            "stem": f"stem{index}",
            "result": {
                "passed": True,
                "message": "ok",
                "elapsed_seconds": 1.5,
                "condition_statuses": [],
            },
            "errored": False,
        }
    )


def _batch(tmp_path: Path, content: bytes | None, job: Any = "70") -> BatchLayout:
    """A batch of three cases in one task of submission 0 (whose job is
    *job*), whose cases 0 and 2 passed, whose case 1 has no outcome, and whose
    task left *content* as its status (none when None)."""
    layout = BatchLayout(tmp_path / "batch")
    write_json(
        layout.manifest,
        {
            "version": MANIFEST_VERSION,
            "cases": [
                {"index": i, "label": f"s#{i + 1}", "stem": f"stem{i}"}
                for i in range(3)
            ],
            "launcher": {},
        },
    )
    layout.result(0).parent.mkdir(parents=True, exist_ok=True)
    layout.result(0).write_text(_passing_record(0), encoding="utf-8")
    layout.result(2).write_text(_passing_record(2), encoding="utf-8")
    write_json(
        layout.submission(0),
        {"index": 0, "job_id": job, "chunks": [[0, 1, 2]], "submitted_at": 1.0},
    )
    layout.task(0, 0).parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        layout.task(0, 0).write_bytes(content)
    return layout


def _outcome(call: Any) -> Any:
    """What *call* returns, or the type and text of what it raises."""
    try:
        return ("returns", call())
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


def _same(actual: Any, expected: Any) -> None:
    """Equal, NaN equal to NaN and -0.0 told from 0.0: compared as JSON text
    (repr for what JSON cannot write)."""

    def text(value: Any) -> str:
        try:
            return json.dumps(value, indent=1, ensure_ascii=False)
        except (TypeError, ValueError):
            return repr(value)

    assert text(actual) == text(expected)


def _keys(status: Any) -> list[str]:
    """The keys of the status read_task read, in their order (a dict, as it
    was read before, or a typed record of its members)."""
    if isinstance(status, dict):
        return list(status)
    return list(status.members.keys())


def _has(status: Any, key: str) -> bool:
    """Whether the status read_task read has the member *key*."""
    if isinstance(status, dict):
        return key in status
    return bool(status.has(key))


# -- What read_task reads -----------------------------------------------------------


def test_the_corpus_covers_every_kind_of_status() -> None:
    statuses = [_old_status(content) for _, content in _CONTENTS]
    assert statuses.count(None) == 12
    assert len(statuses) > 650


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_read_task_reads_each_status(tmp_path: Path, name: str, content: bytes) -> None:
    layout = _batch(tmp_path, content)
    old = _old_status(content)
    status = layout.read_task(0, 0)
    assert (status is None) is (old is None)
    if status is None or old is None:
        return
    assert _keys(status) == list(old)
    for key in (*_KEYS, "z", "a", "rpc"):
        _same(status.get(key), old.get(key))
        assert _has(status, key) is (key in old)
    for key in old:
        _same(status.get(key), old[key])


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_typed_accessors_of_each_status(
    tmp_path: Path, name: str, content: bytes
) -> None:
    """TaskStatus's state_text(), error and is_at() read what its readers
    read of the status as a dict."""
    layout = _batch(tmp_path, content)
    old = _old_status(content)
    status = layout.read_task(0, 0)
    if status is None or old is None:
        return
    for default in ("not started", "never started", "?"):
        assert status.state_text(default) == f"{old.get('state', default)}"
    assert status.error == (f"{old['error']}" if old.get("error") else None)
    for case in (0, 1, 2, -1, 2**53 + 1, 10**30, 10**400):
        assert status.is_at(case) is bool(old.get("current") == case)


def test_read_task_is_none_for_a_missing_status_or_a_directory(
    tmp_path: Path,
) -> None:
    layout = _batch(tmp_path, None)
    assert layout.read_task(0, 0) is None
    assert layout.read_task(3, 1) is None
    layout.task(0, 0).mkdir()
    assert layout.read_task(0, 0) is None


def test_read_task_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for name, content in _CONTENTS[-12:]:
        layout = _batch(tmp_path / name, content)
        with caplog.at_level("DEBUG"):
            assert layout.read_task(0, 0) is None
    assert caplog.records == []


# -- status --------------------------------------------------------------------------


def _old_task_line(old: dict[str, Any] | None, chunk: list[int]) -> str:
    """The task's line of ``scenario-slurm status``, as it read a dict."""
    status = old or {}
    ports = status.get("ports") or {}
    rss = status.get("peak_rss_mb")
    return (
        f"    task 0: {status.get('state', 'not started')}"
        f"{' on ' + status['host'] if status.get('host') else ''}"
        f"{' port ' + str(ports['rpc']) if ports.get('rpc') else ''}"
        f", {len(status.get('done') or [])}/{len(chunk)} case(s)"
        f"{f', peak RSS {rss} MB' if rss else ''}"
        f"{' -- ' + status['error'] if status.get('error') else ''}"
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_status_prints_each_task_s_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
) -> None:
    layout = _batch(tmp_path, content)
    monkeypatch.setattr(cli, "squeue_states", lambda job: {})
    old = _old_status(content)
    expected = _outcome(lambda: _old_task_line(old, [0, 1, 2]))
    actual = _outcome(lambda: cli._status(layout))
    out = capsys.readouterr().out
    head = (
        f"Batch {layout.batch_dir}: 3 case(s)\n"
        "  2 with a result (2 passed, 0 errored), 1 without\n"
        "  submission 0: job 70, 1 task(s), 0 case(s); in the queue: ended\n"
    )
    if expected[0] == "raises":
        assert actual == expected
        assert out == head
    else:
        assert actual == ("returns", 0)
        assert out == head + expected[1] + "\n"


# -- collect: a case with no outcome ------------------------------------------------


def _old_missing_message(
    layout: BatchLayout, case: int, target: str
) -> tuple[str, str, int | None]:
    """collect._missing_message as it read the task's status as a dict."""
    where = _last_submission_of(layout, case)
    assert where is not None
    index, task, submission = where
    status = _old_status(
        layout.task(index, task).read_bytes()
        if layout.task(index, task).is_file()
        else None
    )
    state = "never started" if status is None else status.get("state", "?")
    job = submission.get("job_id")
    note = "" if job else no_job_note(submission)
    error = f"{status['error']}" if status is not None and status.get("error") else None
    unfinished = bool(job) and status is not None and state in _UNFINISHED_TASK_STATES
    timed_out = (
        not job and sbatch_timed_out(submission) and sbatch_in_flight(submission)
    )
    in_flight = not job and not timed_out and sbatch_in_flight(submission)
    return _advice(
        layout,
        target,
        submitted=True,
        index=index,
        task=task,
        job=f"{job}",
        has_job=bool(job),
        state=f"{state}",
        has_status=status is not None,
        unfinished=unfinished,
        error=error,
        note=note,
        timed_out=timed_out,
        in_flight=in_flight,
    )


@pytest.mark.parametrize("job", ["70", None])
@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_message_of_a_case_with_no_outcome(
    tmp_path: Path, name: str, content: bytes, job: Any
) -> None:
    layout = _batch(tmp_path, content, job)
    target = str(layout.batch_dir)
    _same(
        _outcome(lambda: collect_mod._missing_message(layout, 1, target)),
        _outcome(lambda: _old_missing_message(layout, 1, target)),
    )


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_what_collect_prints_and_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
) -> None:
    """collect gives the case with no outcome the message above: what it
    prints and writes is what it did with the message of the status read as
    a dict."""

    def run() -> tuple[Any, ...]:
        code = _outcome(lambda: collect(layout))
        captured = capsys.readouterr()
        written = [
            path.read_bytes() if path.is_file() else None
            for path in (
                layout.batch_dir / "batch_results.json",
                layout.root / "collected.json",
            )
        ]
        for path in (
            layout.batch_dir / "batch_results.json",
            layout.root / "collected.json",
        ):
            path.unlink(missing_ok=True)
        return (code, captured.out, captured.err, written)

    layout = _batch(tmp_path, content)
    capsys.readouterr()
    actual = run()
    monkeypatch.setattr(collect_mod, "_missing_message", _old_missing_message)
    assert actual == run()


# -- The viewer's batch status ------------------------------------------------------


def _old_task_entry(old: dict[str, Any] | None) -> dict[str, Any]:
    status = old or {}
    return {
        "task": 0,
        "cases": [0, 1, 2],
        "squeue": None,
        "state": status.get("state", "not started"),
        "host": status.get("host"),
        "done": len(status.get("done") or []),
        "current": status.get("current"),
        "error": status.get("error"),
        "peak_rss_mb": status.get("peak_rss_mb"),
    }


def _old_case(
    old: dict[str, Any] | None, states: dict[int, str] | None
) -> tuple[str, str]:
    """The state and message of case 1, which has no outcome, as the viewer
    read its task's status as a dict."""
    from autoware_carla_scenario.ui.slurm_case_state import case_state
    from autoware_carla_scenario.ui.slurm_launch import _ACTIVE_STATES

    status = old or {}
    squeue_known = states is not None
    queued = None if states is None else states.get(0)
    at_case = False
    task_state, task_error = "", None
    if queued in ACTIVE_STATES:
        at_case = queued == "RUNNING" and bool(status.get("current") == 1)
    elif squeue_known:
        task_state = f"{status.get('state', 'never started')}"
        if status.get("error"):
            task_error = f"{status['error']}"
    return case_state(
        False,
        False,
        False,
        True,
        queued,
        squeue_known,
        at_case,
        0,
        "70",
        task_state,
        task_error,
        _ACTIVE_STATES,
    )


@pytest.mark.parametrize(
    "states",
    [{}, {0: "RUNNING"}, {0: "PENDING"}, None],
    ids=["ended", "running", "pending", "unknown"],
)
@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_viewer_s_batch_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    content: bytes,
    states: dict[int, str] | None,
) -> None:
    from autoware_carla_scenario.ui import slurm_launch

    layout = _batch(tmp_path, content)
    monkeypatch.setattr(slurm_launch, "squeue_states", lambda job: states)
    old = _old_status(content)
    expected_entry = _outcome(lambda: _old_task_entry(old))
    expected_case = _outcome(lambda: _old_case(old, states))
    actual = _outcome(lambda: slurm_launch.batch_status(layout.batch_dir))
    # The tasks are read first, then the cases.
    for expected in (expected_entry, expected_case):
        if expected[0] == "raises":
            assert actual == expected
            return
    assert actual[0] == "returns"
    batch = actual[1]
    entry = expected_entry[1]
    entry["squeue"] = None if states is None else states.get(0)
    _same(batch["submissions"][0]["tasks"], [entry])
    # The very values json.loads() read: of their type, NaN as NaN.
    for key in ("state", "host", "current", "error", "peak_rss_mb"):
        value = batch["submissions"][0]["tasks"][0][key]
        assert type(value) is type(entry[key])
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(entry[key])
        else:
            assert value == entry[key]
    case = batch["cases"][1]
    assert (case["state"], case["message"]) == expected_case[1]


def test_the_viewer_reads_no_status_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.ui import slurm_launch

    layout = _batch(tmp_path, None)
    monkeypatch.setattr(slurm_launch, "squeue_states", lambda job: {})
    batch = slurm_launch.batch_status(layout.batch_dir)
    assert batch is not None
    assert batch["submissions"][0]["tasks"] == [
        _old_task_entry(None) | {"squeue": None}
    ]
    assert (batch["cases"][1]["state"], batch["cases"][1]["message"]) == _old_case(
        None, {}
    )


def test_nan_and_negative_zero_are_compared_as_json_writes_them() -> None:
    _same([math.nan, -0.0], [math.nan, -0.0])
    with pytest.raises(AssertionError):
        _same([-0.0], [0.0])


# -- A status nested deep -----------------------------------------------------------


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


@pytest.mark.parametrize("key", ["x", "done", "current", "ports", "host"])
@pytest.mark.parametrize("depth", [500, 5000, 100_000])
def test_a_status_nested_deep_is_read_as_json_loads_reads_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    depth: int,
) -> None:
    """A status nested deeper than ``json``'s encoder written in Python lays
    out is read wherever ``json.loads()`` reads it, and raises what it raises
    where it does not: a RecursionError, which reading it as a ``dict`` let
    through too."""
    from autoware_carla_scenario.ui import slurm_launch

    content = _status(**{key: _nested(depth)}).encode()
    layout = _batch(tmp_path, content)
    monkeypatch.setattr(cli, "squeue_states", lambda job: {})
    monkeypatch.setattr(slurm_launch, "squeue_states", lambda job: {})
    try:
        old = json.loads(content.decode("utf-8"))
    except RecursionError:
        for read in (
            lambda: layout.read_task(0, 0),
            lambda: cli._status(layout),
            lambda: slurm_launch.batch_status(layout.batch_dir),
            lambda: collect_mod._missing_message(layout, 1, "t"),
        ):
            with pytest.raises(RecursionError):
                read()
        return
    status = layout.read_task(0, 0)
    assert status is not None
    assert json.dumps(status.get(key)) == json.dumps(old[key])
    assert len(status.get("done") or []) == len(old["done"])
    if key in ("x", "done", "current"):
        assert cli._status(layout) == 0
        assert capsys.readouterr().out.splitlines()[-1] == _old_task_line(
            old, [0, 1, 2]
        )
        batch = slurm_launch.batch_status(layout.batch_dir)
        assert batch is not None
        assert batch["cases"][1]["state"] == "missing"
        assert collect_mod._missing_message(layout, 1, "t") == _old_missing_message(
            layout, 1, "t"
        )
