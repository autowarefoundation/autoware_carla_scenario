"""How a SLURM batch's submission record (``slurm/submissions/<k>.json``) is read.

Pinned against the records a submitter (``slurm/submit.py``'s
``submit_cases``), a hand, another version, a crash or a killed submitter may
have left: each key a submitter writes (``index``, ``created``,
``submitted_at``, ``cases``, ``chunks``, ``launcher``, ``job_id``,
``collect_job_id``, then ``sbatch``, and either ``sbatch_timed_out`` and
``sbatch_timed_out_at`` or ``collect_sbatch`` and ``collect_error``), missing
or of every JSON type -- NaN, infinities, a number no float holds, an integer
no float holds exactly, non-ASCII text -- in a record whose sbatch returned,
one still in flight and one whose sbatch timed out, records older than
``submitted_at``, keys named twice, and files that hold no record.

Each is compared with what the readers of a batch made of the record when it
was read as a ``dict`` (the ``_old_*`` oracles below, the code that read it):
what ``submission_record`` reads and logs, whether its sbatch is in flight or
timed out and why it has no job id (with a clock that counts its reads), which
submission last gave a case, the message ``collect`` gives a case with no
outcome; and, with the record read as a ``dict`` by the oracles in place of
the readers, what ``status``, ``cancel``, a resubmit's ``_busy``, ``wait``
(``submit.follow``), ``after_submit``, a task's start (``worker.run_task``),
``collect`` and the viewer's batch status print, log, write and return.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import autoware_carla_scenario.slurm.manifest as manifest_mod
from autoware_carla_scenario.slurm import cli, collect as collect_mod, submit
from autoware_carla_scenario.slurm.collect import (
    _UNFINISHED_TASK_STATES,
    _advice,
    _last_submission_of,
)
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    no_job_note,
    sbatch_in_flight,
    sbatch_timed_out,
    write_json,
)
from autoware_carla_scenario.slurm.settings import SlurmSettings

_LOGGER = "autoware_carla_scenario.slurm.manifest"

#: What the fake clock says, unless a test says otherwise.
_NOW = 2_000_000_000.0

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
    '"70"',
    "[]",
    "[1]",
    '["a", "b"]',
    "{}",
    '{"1": 1}',
    '{"k": "v"}',
]

#: More values of the keys that hold cases or times.
_EXTRA: dict[str, list[str]] = {
    "chunks": [
        "[[0, 1], [2]]",
        "[[2], [0, 1]]",
        "[[1]]",
        "[[1.0]]",
        "[[true]]",
        "[[1.5]]",
        "[[NaN, 1]]",
        '[["1"]]',
        '[{"1": 1}]',
        "[{}]",
        "[1]",
        "[null]",
        "[true]",
        '[[0], "x"]',
        '["x", [1]]',
        "[[[1]]]",
        "[[], [1, 2]]",
        "[[0, 1, 2]]",
        "[[2, 1]]",
        "[[0], [1], [2], [1]]",
        '"abc"',
    ],
    "cases": ["[0, 1, 2]", "[1, 2]", '"abc"', "[[]]"],
    "submitted_at": [
        "1999999401",
        "1999999400",
        "1999999400.0",
        "1999999400.5",
        "1999999999.9999999",
        "2000000000",
        "3000000000",
    ],
    "sbatch_timed_out_at": ['"2026-11-12T00:00:10"', "0.0"],
    "collect_error": ['"sbatch: error: boom \\u00fc"'],
}

#: The keys a submission record holds, in the order the submitter writes them.
_KEYS = (
    "index",
    "created",
    "submitted_at",
    "cases",
    "chunks",
    "launcher",
    "job_id",
    "collect_job_id",
    "sbatch",
    "sbatch_timed_out",
    "sbatch_timed_out_at",
    "collect_sbatch",
    "collect_error",
)

#: What a submitter writes when it reserves the index, before sbatch.
_RESERVED: dict[str, str] = {
    "index": "0",
    "created": '"2026-11-12T00:00:00"',
    "submitted_at": "1999999900.0",
    "cases": "[0, 1, 2]",
    "chunks": "[[0, 1], [2]]",
    "launcher": '{"wait": false, "partition": "p\\u00fc"}',
    "job_id": "null",
    "collect_job_id": "null",
}

_SBATCH = '["sbatch", "--array=0-1", "/b/slurm/jobs/0.sh"]'

#: The records of a submission, by what became of its sbatch.
_BASES: dict[str, dict[str, str]] = {
    # sbatch returned a job id.
    "done": {**_RESERVED, "job_id": '"70"', "sbatch": _SBATCH},
    # sbatch has not returned yet.
    "in-flight": dict(_RESERVED),
    # sbatch timed out (#1791).
    "timed-out": {
        **_RESERVED,
        "sbatch": _SBATCH,
        "sbatch_timed_out": "true",
        "sbatch_timed_out_at": '"2026-11-12T00:00:10"',
    },
}


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


def _record(base: str = "done", **values: str) -> str:
    """The record *base* with *values* in place of (or after) its own."""
    return _object({**_BASES[base], **values})


def _corpus() -> list[tuple[str, str]]:
    """(name, the text of submission 0's record)."""
    corpus: list[tuple[str, str]] = []
    for base in _BASES:
        for key in _KEYS:
            for value in _VALUES + _EXTRA.get(key, []):
                corpus.append((f"{base}-{key}-{value}", _record(base, **{key: value})))
    collected = {**_BASES["done"], "collect_sbatch": _SBATCH}
    corpus += [
        ("collected", _object({**collected, "collect_job_id": '"71"'})),
        (
            "collect-failed",
            _object({**collected, "collect_error": '"sbatch: error: no"'}),
        ),
        (
            "collect-failed-before-sbatch",
            _object({**_BASES["done"], "collect_error": '"OSError: no"'}),
        ),
        # Before submitted_at (#1352): never in flight.
        (
            "old-no-submitted-at",
            _object({k: v for k, v in _BASES["done"].items() if k != "submitted_at"}),
        ),
        (
            "old-no-job",
            _object(
                {k: v for k, v in _BASES["in-flight"].items() if k != "submitted_at"}
            ),
        ),
        ("old-minimal", '{"index": 0, "job_id": "70", "chunks": [[0, 1], [2]]}'),
        ("old-minimal-no-job", '{"index": 0, "job_id": null, "chunks": [[0, 1, 2]]}'),
        ("reversed", _object(dict(reversed(list(_BASES["timed-out"].items()))))),
        ("empty-object", "{}"),
        (
            "extra-keys",
            '{"z": [1, {"y": NaN}], "job_id": null, "submitted_at": 1999999999, '
            '"\\u00fc": [Infinity, -0.0], "chunks": [[1, 2]], "a": null}',
        ),
    ]
    # Keys named twice: the first place and the last value.
    corpus += [
        ("twice-job", '{"job_id": "70", "chunks": [[1]], "job_id": null}'),
        (
            "twice-submitted-at",
            '{"submitted_at": 1999999900, "job_id": null, "submitted_at": "x"}',
        ),
        ("twice-chunks", '{"chunks": [[1]], "job_id": "7", "chunks": [[2]]}'),
        (
            "twice-timed-out",
            '{"sbatch_timed_out": false, "job_id": null, "submitted_at": '
            '1999999900, "sbatch_timed_out": true}',
        ),
    ]
    # Files that hold no record.
    corpus += [
        ("not-json", "{not json"),
        ("truncated", _record()[:-7]),
        ("empty", ""),
        ("array", "[" + _record() + "]"),
        ("string", '"x"'),
        ("number", "1"),
        ("null", "null"),
        ("trailing", _record() + " x"),
        ("integer-too-long", '{"job_id": ' + "1" * 5000 + "}"),
    ]
    return corpus


_CORPUS = _corpus()
_BYTES: dict[str, bytes] = {
    "bom": b"\xef\xbb\xbf" + _record().encode(),
    "not-utf-8": _record().encode().replace(b"sbatch", b"sb\xffatch"),
    "latin-1": _record(created='"\\u00fc"').replace("\\u00fc", "ü").encode("latin-1"),
}


def _all_contents() -> list[tuple[str, bytes]]:
    return [(name, text.encode("utf-8")) for name, text in _CORPUS] + list(
        _BYTES.items()
    )


_CONTENTS = _all_contents()
_IDS = [name for name, _ in _CONTENTS]


# -- The record as it was read: a dict ------------------------------------------------


class _Clock:
    """``time`` as the readers see it: a fixed time, counting its reads."""

    def __init__(self, now: float = _NOW) -> None:
        self.now = now
        self.reads = 0

    def time(self) -> float:
        self.reads += 1
        return self.now


def _old_read(path: Path) -> dict[str, Any] | None:
    """read_submission_record as it read a record: a dict, or None (with the
    warning it logged)."""
    logger = logging.getLogger(_LOGGER)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record (%s)", path, exc
        )
        return None
    if not isinstance(record, dict):
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record "
            "(it holds JSON that is not an object)",
            path,
        )
        return None
    return record


def _old_in_flight(submission: dict[str, Any], now: float | None = None) -> bool:
    if submission.get("job_id"):
        return False
    started = submission.get("submitted_at")
    if not isinstance(started, (int, float)):
        return False
    now = manifest_mod.time.time() if now is None else now
    return now - float(started) < SBATCH_IN_FLIGHT_SECONDS


def _old_timed_out(submission: dict[str, Any]) -> bool:
    return not submission.get("job_id") and submission.get("sbatch_timed_out") is True


def _old_note(submission: dict[str, Any], now: float | None = None) -> str:
    if _old_timed_out(submission):
        at = submission.get("sbatch_timed_out_at")
        return (
            "no job id: sbatch timed out"
            + (f" at {at}" if at else "")
            + "; the job may exist all the same: check `squeue --me`"
        )
    if _old_in_flight(submission, now):
        return "no job id yet (sbatch in flight)"
    return (
        "no job id: its submitter stopped before sbatch returned, more than "
        f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s ago (a stale record)"
    )


def _old_last_submission_of(
    layout: BatchLayout, case: int
) -> tuple[int, int, dict[str, Any]] | None:
    for index in reversed(layout.submissions()):
        submission = _old_read(layout.submission(index))
        if submission is None:
            continue
        for task, chunk in enumerate(submission.get("chunks") or []):
            if case in chunk:
                return index, task, submission
    return None


def _old_missing_message(
    layout: BatchLayout, case: int, target: str
) -> tuple[str, str, int | None]:
    """collect._missing_message as it read the submission as a dict."""
    where = _old_last_submission_of(layout, case)
    if where is None:
        return _advice(
            layout,
            target,
            submitted=False,
            index=0,
            task=0,
            job="",
            has_job=False,
            state="",
            has_status=False,
            unfinished=False,
            error=None,
            note="",
            timed_out=False,
            in_flight=False,
        )
    index, task, submission = where
    status = layout.read_task(index, task)
    state = (
        "never started"
        if status is None
        else status.get("state")
        if status.has("state")
        else "?"
    )
    job = submission.get("job_id")
    note = "" if job else _old_note(submission)
    error = None if status is None else status.error
    unfinished = bool(job) and status is not None and state in _UNFINISHED_TASK_STATES
    timed_out = not job and _old_timed_out(submission) and _old_in_flight(submission)
    in_flight = not job and not timed_out and _old_in_flight(submission)
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


def _old_read_submission(layout: BatchLayout, index: int) -> Any:
    """What cli._active_jobs read raw of a record: a dict, or None for JSON
    that is no object (which it took as one it cannot read)."""
    data = json.loads(layout.submission(index).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


@contextmanager
def _old_reading(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every reader of the batch reading the submission records as a dict."""
    from autoware_carla_scenario.ui import slurm_launch

    with monkeypatch.context() as patch:
        patch.setattr(
            BatchLayout,
            "submission_record",
            lambda self, index: _old_read(self.submission(index)),
        )
        patch.setattr(BatchLayout, "read_submission", _old_read_submission)
        for module in (manifest_mod, cli, collect_mod, submit, slurm_launch):
            for name, old in (
                ("sbatch_in_flight", _old_in_flight),
                ("sbatch_timed_out", _old_timed_out),
                ("no_job_note", _old_note),
            ):
                if hasattr(module, name):
                    patch.setattr(module, name, old)
        patch.setattr(collect_mod, "_last_submission_of", _old_last_submission_of)
        patch.setattr(collect_mod, "_missing_message", _old_missing_message)
        yield


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


def _batch(
    tmp_path: Path, content: bytes | None, task: str | None = '{"state": "running"}'
) -> BatchLayout:
    """A batch of three cases, whose case 0 passed and cases 1 and 2 have no
    outcome, with *content* as submission 0's record (none when None), and
    *task* as task 0 of submission 0's status (none when None)."""
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
    layout.submission(0).parent.mkdir(parents=True, exist_ok=True)
    if content is not None:
        layout.submission(0).write_bytes(content)
    layout.task(0, 0).parent.mkdir(parents=True, exist_ok=True)
    if task is not None:
        layout.task(0, 0).write_text(task, encoding="utf-8")
    return layout


def _outcome(call: Callable[[], Any]) -> Any:
    """What *call* returns, or the type and text of what it raises."""
    try:
        return ("returns", call())
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


def _text(value: Any) -> str:
    """*value* as JSON text (repr for what JSON cannot write): NaN equal to
    NaN and -0.0 told from 0.0."""
    try:
        return json.dumps(value, indent=1, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _same(actual: Any, expected: Any) -> None:
    assert _text(actual) == _text(expected)


def _keys(record: Any) -> list[str]:
    """The keys of a record submission_record read, in their order (a dict,
    as it was read before, or a typed record of its members)."""
    if isinstance(record, dict):
        return list(record)
    return list(record.members.keys())


def _logs(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    return [(r.name, r.levelno, r.getMessage()) for r in caplog.records]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """The readers' clock, which reads _NOW and counts its reads."""
    fake = _Clock()
    monkeypatch.setattr(manifest_mod, "time", SimpleNamespace(time=fake.time))
    monkeypatch.setattr(cli, "time", SimpleNamespace(time=fake.time))
    return fake


# -- What submission_record reads ------------------------------------------------------


def test_the_corpus_covers_every_kind_of_record(tmp_path: Path) -> None:
    path = tmp_path / "0.json"
    records = []
    for _, content in _CONTENTS:
        path.write_bytes(content)
        records.append(_old_read(path))
    assert records.count(None) == 12
    assert len(records) > 1200


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_submission_record_reads_each_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, content: bytes
) -> None:
    layout = _batch(tmp_path, content)
    path = layout.submission(0)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        old = _old_read(path)
    expected_logs = _logs(caplog)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        record = layout.submission_record(0)
        direct = manifest_mod.read_submission_record(path)
    assert _logs(caplog) == expected_logs * 2
    assert (record is None) is (old is None)
    assert (direct is None) is (old is None)
    if record is None or old is None:
        return
    assert _keys(record) == list(old)
    for key in (*_KEYS, "z", "a", "rpc"):
        _same(record.get(key), old.get(key))
    for key in old:
        _same(record.get(key), old[key])
        _same(record[key], old[key])
    # What json.loads() gave, of its type.
    for key in old:
        assert type(record.get(key)) is type(old[key])


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_what_a_record_says_of_its_sbatch(
    tmp_path: Path, clock: _Clock, name: str, content: bytes
) -> None:
    """sbatch_in_flight, sbatch_timed_out and no_job_note of each record, at
    each time and on the clock, as of the record read as a dict: what they
    return or raise, and how often they read the clock."""
    layout = _batch(tmp_path, content)
    old = _old_read(layout.submission(0))
    record = layout.submission_record(0)
    if record is None or old is None:
        assert record is None and old is None
        return
    for now in (None, 0.0, 600.0, 1999999400.0, 1999999400.5, _NOW, 3e9):
        for new, oracle in (
            (lambda: sbatch_in_flight(record, now), lambda: _old_in_flight(old, now)),
            (lambda: no_job_note(record, now), lambda: _old_note(old, now)),
        ):
            before = clock.reads
            expected = _outcome(oracle)
            expected_reads = clock.reads - before
            before = clock.reads
            assert _outcome(new) == expected
            assert clock.reads - before == expected_reads
    before = clock.reads
    assert _outcome(lambda: sbatch_timed_out(record)) == _outcome(
        lambda: _old_timed_out(old)
    )
    assert clock.reads == before


# -- Which submission last gave a case ------------------------------------------------


def _where(found: Any) -> Any:
    """(submission, task, the record's keys) of what _last_submission_of found."""
    if found is None:
        return None
    index, task, record = found
    return (index, task, _keys(record))


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_submission_that_last_gave_each_case(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, content: bytes
) -> None:
    """With submission 0 giving every case and submission 1 the record, which
    gave each case last: what it returns or raises, and what it logs (the
    records read until one gave it)."""
    layout = _batch(tmp_path, b'{"index": 0, "job_id": "60", "chunks": [[0, 1, 2]]}')
    layout.submission(1).write_bytes(content)
    for case in (0, 1, 2, 3, -1):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            expected = _outcome(lambda: _where(_old_last_submission_of(layout, case)))
        expected_logs = _logs(caplog)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            actual = _outcome(lambda: _where(_last_submission_of(layout, case)))
        assert actual == expected
        assert _logs(caplog) == expected_logs


# -- collect: a case with no outcome ------------------------------------------------


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_the_message_of_a_case_with_no_outcome(
    tmp_path: Path, clock: _Clock, name: str, content: bytes
) -> None:
    layout = _batch(tmp_path, content)
    target = str(layout.batch_dir)
    for case in (1, 2):
        before = clock.reads
        expected = _outcome(lambda: _old_missing_message(layout, case, target))
        expected_reads = clock.reads - before
        before = clock.reads
        _same(
            _outcome(lambda: collect_mod._missing_message(layout, case, target)),
            expected,
        )
        assert clock.reads - before == expected_reads


@pytest.mark.parametrize(
    "task",
    [
        None,
        '{"state": "running"}',
        '{"state": "waiting", "error": "e"}',
        '{"state": "done"}',
        '{"state": "failed", "error": ["x"]}',
        '{"state": ["running"]}',
        '{"state": {"a": 1}}',
        '{"state": 1}',
        '{"state": null}',
        '{"error": "e"}',
        "{}",
        "[]",
    ],
)
@pytest.mark.parametrize(
    "record",
    [
        _record("done"),
        _record("done", job_id="0"),
        _record("done", job_id="[1]"),
        _record("in-flight"),
        _record("in-flight", submitted_at="1"),
        _record("timed-out"),
        _record("timed-out", submitted_at="1"),
        _record("timed-out", sbatch_timed_out_at='""'),
        '{"chunks": [[0, 1]], "job_id": "7"}',
    ],
)
def test_the_message_of_a_case_with_no_outcome_and_its_task_s_status(
    tmp_path: Path, clock: _Clock, record: str, task: str | None
) -> None:
    layout = _batch(tmp_path, record.encode(), task)
    target = str(layout.batch_dir)
    before = clock.reads
    expected = _outcome(lambda: _old_missing_message(layout, 1, target))
    expected_reads = clock.reads - before
    before = clock.reads
    _same(_outcome(lambda: collect_mod._missing_message(layout, 1, target)), expected)
    assert clock.reads - before == expected_reads


# -- Every reader of the batch --------------------------------------------------------


def _readers(layout: BatchLayout) -> dict[str, Callable[[], Any]]:
    """Each reader of the batch's submission records, as a call."""
    from autoware_carla_scenario.slurm import worker
    from autoware_carla_scenario.ui import slurm_launch

    batch = str(layout.batch_dir)
    return {
        "status": lambda: cli._status(layout),
        "cancel": lambda: cli._cancel(argparse.Namespace(batch_dir=batch)),
        "busy": lambda: cli._busy(layout),
        "busy-below-0": lambda: cli._busy(layout, below=0),
        "active-jobs": lambda: cli._active_jobs(layout),
        "follow": lambda: submit.follow(layout, 0, SlurmSettings(poll_seconds=0.0)),
        "after-submit": lambda: submit.after_submit(
            layout, 0, SlurmSettings(wait=False)
        ),
        "run-task": lambda: worker.run_task(layout.manifest, 0, 99),
        "missing-cases": lambda: [
            collect_mod._missing_message(layout, i, batch)
            for i in collect_mod.missing_cases(layout)
        ],
        "collect": lambda: collect_mod.collect(layout),
        "batch-status": lambda: _text(slurm_launch.batch_status(layout.batch_dir)),
    }


def _run_readers(
    layout: BatchLayout,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> dict[str, Any]:
    """What each reader returns or raises, prints, logs and writes, and how
    often it reads the clock."""
    seen: dict[str, Any] = {}
    written = (
        layout.batch_dir / "batch_results.json",
        layout.root / "collected.json",
    )
    for name, call in _readers(layout).items():
        capsys.readouterr()
        caplog.clear()
        before = clock.reads
        with caplog.at_level(logging.DEBUG):
            outcome = _outcome(call)
        captured = capsys.readouterr()
        files = [path.read_bytes() if path.is_file() else None for path in written]
        for path in written:
            path.unlink(missing_ok=True)
        seen[name] = (
            _text(outcome),
            captured.out,
            captured.err,
            _logs(caplog),
            files,
            clock.reads - before,
        )
    return seen


@pytest.fixture
def quiet_slurm(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """squeue lists no task of any job, and scancel cancels (what it was
    asked to cancel)."""
    from autoware_carla_scenario.ui import slurm_launch

    for module in (cli, submit, slurm_launch):
        monkeypatch.setattr(module, "squeue_states", lambda job: {})
    cancelled: list[list[str]] = []

    def cancel(jobs: list[str]) -> bool:
        cancelled.append(list(jobs))
        return True

    monkeypatch.setattr(cli, "cancel_jobs", cancel)
    return cancelled


@pytest.mark.parametrize(("name", "content"), _CONTENTS, ids=_IDS)
def test_what_every_reader_makes_of_each_record(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: list[list[str]],
    name: str,
    content: bytes,
) -> None:
    layout = _batch(tmp_path, content)
    actual = _run_readers(layout, clock, capsys, caplog)
    cancelled = list(quiet_slurm)
    quiet_slurm.clear()
    with _old_reading(monkeypatch):
        expected = _run_readers(layout, clock, capsys, caplog)
    assert actual == expected
    assert cancelled == quiet_slurm


@pytest.mark.parametrize(
    "records",
    [
        [_record("done"), _record("in-flight", chunks="[[1, 2]]")],
        [_record("timed-out"), "{not json", _record("done", chunks="[[2]]")],
        [_record("done", job_id='"60"'), None, _record("timed-out", chunks="[[1]]")],
        [_record("in-flight"), "[]", _record("done", job_id="null", chunks="[[0]]")],
        [_record("done"), _record("done", job_id='"71"', chunks='"x"')],
        [_record("done", chunks="[1]"), _record("done", job_id='"71"', chunks="[[2]]")],
        [_record("done", submitted_at="1"), _record("in-flight", submitted_at="1")],
    ],
    ids=lambda records: str(len(records)),
)
def test_what_every_reader_makes_of_a_batch_of_many_submissions(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: list[list[str]],
    records: list[str | None],
) -> None:
    layout = _batch(tmp_path, None)
    for index, record in enumerate(records):
        if record is not None:
            layout.submission(index).write_text(record, encoding="utf-8")
    actual = _run_readers(layout, clock, capsys, caplog)
    with _old_reading(monkeypatch):
        expected = _run_readers(layout, clock, capsys, caplog)
    assert actual == expected


def test_the_readers_of_a_batch_with_no_record_and_a_directory(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: list[list[str]],
) -> None:
    layout = _batch(tmp_path, None)
    layout.submission(1).mkdir()
    layout.submission(2).write_text(_record("done"), encoding="utf-8")
    actual = _run_readers(layout, clock, capsys, caplog)
    with _old_reading(monkeypatch):
        expected = _run_readers(layout, clock, capsys, caplog)
    assert actual == expected
    assert actual["status"][1].count("unreadable record") == 1


# -- Exact text -----------------------------------------------------------------------


def test_status_says_why_a_submission_has_no_job_id(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    quiet_slurm: list[list[str]],
) -> None:
    layout = _batch(tmp_path, _record("done").encode(), None)
    for index, record in enumerate(
        [
            _record("in-flight"),
            _record("timed-out"),
            _record("timed-out", submitted_at="1"),
            _record("in-flight", submitted_at="1"),
            '{"index": 5, "job_id": null, "chunks": [[0]]}',
            "[]",
        ],
        start=1,
    ):
        layout.submission(index).write_text(record, encoding="utf-8")
    capsys.readouterr()
    assert cli._status(layout) == 0
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("  submission")
    ]
    stale = (
        "no job id: its submitter stopped before sbatch returned, more than "
        "600 s ago (a stale record)"
    )
    timed_out = (
        "no job id: sbatch timed out at 2026-11-12T00:00:10; the job may exist "
        "all the same: check `squeue --me`"
    )
    assert lines == [
        "  submission 0: job 70, 2 task(s), 3 case(s); in the queue: ended",
        "  submission 1: 2 task(s), 3 case(s); no job id yet (sbatch in flight)",
        f"  submission 2: 2 task(s), 3 case(s); {timed_out}",
        f"  submission 3: 2 task(s), 3 case(s); {timed_out}",
        f"  submission 4: 2 task(s), 3 case(s); {stale}",
        f"  submission 5: 1 task(s), 0 case(s); {stale}",
        f"  submission 6: unreadable record {layout.submission(6)} "
        "(empty, not JSON, or not a JSON object)",
    ]
    # In flight: one read of the clock each; timed out, none for the note.
    assert clock.reads == 2


def test_a_resubmit_is_refused_for_a_submission_in_flight_or_timed_out(
    tmp_path: Path, clock: _Clock, quiet_slurm: list[list[str]]
) -> None:
    layout = _batch(tmp_path, _record("in-flight").encode())
    layout.submission(1).write_text(_record("timed-out"), encoding="utf-8")
    layout.submission(2).write_text(_record("done"), encoding="utf-8")
    assert cli._active_jobs(layout) == ([], [], [0], [], [1])
    assert clock.reads == 2
    reason = cli._busy(layout)
    assert reason == (
        "submission(s) 0 of this batch are being submitted (sbatch has not "
        "returned yet), by another resubmit or the viewer; try again once it "
        "has, or pass --force. A record left by a submitter that died is "
        "ignored 600 s after it was made"
    )
    layout.submission(0).write_text(
        _record("in-flight", submitted_at="1"), encoding="utf-8"
    )
    assert cli._busy(layout) == (
        "the sbatch of submission(s) 1 of this batch timed out, and SLURM may "
        "have taken the job all the same: check `squeue --me` (scancel the job "
        "if it is there and you want its cases run again), then pass --force, "
        "or try again once the record is ignored, 600 s after it was made"
    )
    assert cli._busy(layout, below=1) is None


def test_an_unreadable_record_refuses_a_resubmit_only_while_young(
    tmp_path: Path, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    import os

    layout = _batch(tmp_path, b"{not json")
    path = layout.submission(0)
    os.utime(path, (_NOW - 10, _NOW - 10))
    with caplog.at_level(logging.DEBUG):
        assert cli._active_jobs(layout) == ([], [], [], [0], [])
    assert caplog.records == []
    os.utime(path, (_NOW - 700, _NOW - 700))
    with caplog.at_level(logging.DEBUG):
        assert cli._active_jobs(layout) == ([], [], [], [], [])
    assert _logs(caplog) == [
        (
            "autoware_carla_scenario.slurm.cli",
            logging.WARNING,
            f"Ignoring {path}: it cannot be read as a submission record, "
            "and was last modified 700 s ago",
        )
    ]
    assert clock.reads == 2


def test_the_viewer_notes_why_a_submission_has_no_job_id(
    tmp_path: Path, clock: _Clock, quiet_slurm: list[list[str]]
) -> None:
    from autoware_carla_scenario.ui import slurm_launch

    layout = _batch(tmp_path, _record("timed-out").encode())
    layout.submission(1).write_text(_record("in-flight"), encoding="utf-8")
    batch = slurm_launch.batch_status(layout.batch_dir)
    assert batch is not None
    first, second = batch["submissions"]
    assert (first["note"], first["sbatch_timed_out"], first["in_flight"]) == (
        "no job id: sbatch timed out at 2026-11-12T00:00:10; the job may exist "
        "all the same: check `squeue --me`",
        True,
        False,
    )
    assert (second["note"], second["sbatch_timed_out"], second["in_flight"]) == (
        "no job id yet (sbatch in flight)",
        False,
        True,
    )
    assert second["active"] is True
    assert [c["state"] for c in batch["cases"]] == ["passed", "pending", "pending"]


# -- A record nested deep -----------------------------------------------------------


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


@pytest.mark.parametrize(
    "key", ["x", "chunks-element", "job_id", "submitted_at", "sbatch_timed_out_at"]
)
@pytest.mark.parametrize("depth", [500, 5000, 100_000])
def test_a_record_nested_deep_is_read_as_json_loads_reads_it(
    tmp_path: Path,
    clock: _Clock,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    quiet_slurm: list[list[str]],
    key: str,
    depth: int,
) -> None:
    """A record nested deeper than ``json``'s encoder written in Python lays
    out is read wherever ``json.loads()`` reads it, and raises what it raises
    where it does not: a RecursionError, which reading it as a ``dict`` let
    through too."""
    if key == "chunks-element":
        text = _record("timed-out", chunks=f"[[1, {_nested(depth)}], [2]]")
    else:
        text = _record("timed-out", **{key: _nested(depth)})
    layout = _batch(tmp_path, text.encode())
    try:
        old = json.loads(text)
    except RecursionError:
        with pytest.raises(RecursionError):
            layout.submission_record(0)
        with pytest.raises(RecursionError):
            collect_mod._missing_message(layout, 1, "t")
        return
    record = layout.submission_record(0)
    assert record is not None
    assert _keys(record) == list(old)
    if depth < 5000:
        actual = _run_readers(layout, clock, capsys, caplog)
        with _old_reading(monkeypatch):
            expected = _run_readers(layout, clock, capsys, caplog)
        assert actual == expected
    else:
        _same(
            _outcome(lambda: collect_mod._missing_message(layout, 1, "t")),
            _outcome(lambda: _old_missing_message(layout, 1, "t")),
        )
