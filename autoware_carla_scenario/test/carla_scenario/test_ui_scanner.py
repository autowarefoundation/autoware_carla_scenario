"""The viewer's session scan survives a malformed result file (#785).

One result JSON that is not an object, not UTF-8, or names no scenario, or a
``batch_results.json`` with entries that are not objects, must not break the
scan of every other session (``scan_sessions`` feeds the session list). Nor
must a ``.hydra/overrides.yaml`` that is not UTF-8 break the session and case
pages (#1631).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.sweeper.lanelet_constraint_sweeper import _read_verdict
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import results, scanner

DATE = "2026-01-01"
BAD = "00-00-00"
GOOD = "00-00-01"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _session(base: Path, time: str, name: str, data: bytes) -> Path:
    directory = base / "outputs" / DATE / time
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(data)
    return directory


def _with_good_session(base: Path) -> None:
    _session(
        base,
        GOOD,
        "Good_result.json",
        json.dumps({"passed": True, "scenario": "Good"}).encode(),
    )


def _listed(base: Path) -> dict[str, str]:
    return {s.time: s.scenario_name for s in scanner.scan_sessions(base)}


@pytest.mark.parametrize(
    "data",
    [b"[1, 2]", b"\xff\xfe{", b'"text"', b"null"],
    ids=["list", "not-utf8", "string", "null"],
)
def test_a_single_result_that_is_no_json_object_lists_without_a_result(
    tmp_path: Path, data: bytes
) -> None:
    """Not passed, and without a result on the session list and the session
    page alike (#2285)."""
    _with_good_session(tmp_path)
    _session(tmp_path, BAD, "X_result.json", data)

    sessions = {s.time: s for s in scanner.scan_sessions(tmp_path)}

    assert sessions[GOOD].scenario_name == "Good"
    assert sessions[GOOD].passed_count == 1
    assert sessions[GOOD].pending_count == 0
    assert sessions[BAD].scenario_name == "X"
    assert (
        sessions[BAD].passed_count,
        sessions[BAD].pending_count,
        sessions[BAD].total_count,
    ) == (0, 1, 1)
    (item,) = scanner.load_session(tmp_path, "single", DATE, BAD)
    assert (item.index, item.scenario_name, item.passed, item.pending) == (
        0,
        "X",
        None,
        True,
    )
    assert item.message == "Failed to read result JSON"
    assert scanner.load_scenario(tmp_path, "single", DATE, BAD, 0) is None


@pytest.mark.parametrize("scenario", [None, "", 5, ["a"]])
def test_a_result_naming_no_scenario_is_named_after_its_file(
    tmp_path: Path, scenario: object
) -> None:
    _with_good_session(tmp_path)
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps({"passed": True, "scenario": scenario}).encode(),
    )

    assert _listed(tmp_path) == {GOOD: "Good", BAD: "X"}
    (item,) = scanner.load_session(tmp_path, "single", DATE, BAD)
    assert item.scenario_name == "X"
    assert item.passed is True


def test_result_fields_of_the_wrong_type_are_left_out(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps(
            {
                "passed": "yes",
                "elapsed_seconds": "slow",
                "message": 3,
                "condition_statuses": [1, {"label": None, "details": [2]}],
            }
        ).encode(),
    )

    (item,) = scanner.load_session(tmp_path, "single", DATE, BAD)
    assert (item.passed, item.elapsed_seconds, item.message) == (None, None, "3")
    view = scanner.load_scenario(tmp_path, "single", DATE, BAD, 0)
    assert view is not None
    assert [c.label for c in view.condition_statuses] == [""]


def test_a_batch_summary_of_no_objects_is_skipped(tmp_path: Path) -> None:
    _with_good_session(tmp_path)
    _session(tmp_path, BAD, "batch_results.json", b"[1]")

    assert _listed(tmp_path) == {GOOD: "Good"}
    assert scanner.load_session(tmp_path, "batch", DATE, BAD) == []
    assert scanner.load_scenario(tmp_path, "batch", DATE, BAD, 0) is None


def test_batch_entries_that_are_not_objects_keep_the_others_position(
    tmp_path: Path,
) -> None:
    _with_good_session(tmp_path)
    _session(
        tmp_path,
        BAD,
        "batch_results.json",
        json.dumps(
            [
                {"scenario": None, "passed": True},
                7,
                {"scenario": "a", "passed": False, "elapsed_seconds": 1.5},
            ]
        ).encode(),
    )

    sessions = {s.time: s for s in scanner.scan_sessions(tmp_path)}
    assert sessions[GOOD].scenario_name == "Good"
    assert sessions[BAD].scenario_name == "a, unknown"
    assert (sessions[BAD].passed_count, sessions[BAD].total_count) == (1, 2)

    items = scanner.load_session(tmp_path, "batch", DATE, BAD)
    assert [(i.index, i.scenario_name, i.passed) for i in items] == [
        (0, "unknown", True),
        (2, "a", False),
    ]
    view = scanner.load_scenario(tmp_path, "batch", DATE, BAD, 2)
    assert view is not None
    assert view.elapsed_seconds == 1.5


def test_results_reads_malformed_batch_entries(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "batch_results.json",
        json.dumps(
            [
                {
                    "scenario": None,
                    "passed": "no",
                    "elapsed_seconds": "x",
                    "overrides": 3,
                },
                [1],
                {"scenario": "a", "passed": True, "elapsed_seconds": 2},
            ]
        ).encode(),
    )

    cases = results.load_results(tmp_path, "batch", DATE, BAD, metrics=False)

    assert [(c.index, c.label, c.passed, c.elapsed_seconds) for c in cases] == [
        (0, "unknown", None, None),
        (2, "a", True, 2.0),
    ]
    assert cases[0].overrides == []
    assert cases[1].metrics == {"elapsed_s": 2.0}


def test_results_reads_a_single_result_of_wrong_types(tmp_path: Path) -> None:
    _session(
        tmp_path,
        BAD,
        "X_result.json",
        json.dumps({"passed": 1, "elapsed_seconds": "x"}).encode(),
    )

    (case,) = results.load_results(tmp_path, "single", DATE, BAD, metrics=False)

    assert (case.passed, case.elapsed_seconds, case.metrics) == (None, None, {})


def test_the_sweeper_reads_no_verdict_off_a_result_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "X_result.json").write_bytes(b"\xff\xfe{")

    assert _read_verdict(tmp_path) is None


@pytest.mark.parametrize("stray", ["²", "٣"])
def test_a_job_directory_named_by_a_non_ascii_digit_is_skipped(
    tmp_path: Path, stray: str
) -> None:
    """#1612: ``"²".isdigit()`` is true but ``int("²")`` raises, which took
    down the session list; ``"٣"`` (an Arabic-Indic 3) is no job either."""
    session = tmp_path / "multirun" / DATE / GOOD
    (session / "0").mkdir(parents=True)
    (session / "0" / "Good_result.json").write_text(
        json.dumps({"passed": True, "scenario": "Good"})
    )
    (session / stray).mkdir()

    (summary,) = scanner.scan_sessions(tmp_path)
    (item,) = scanner.load_session(tmp_path, "multirun", DATE, GOOD)
    (case,) = results.load_results(tmp_path, "multirun", DATE, GOOD, metrics=False)

    assert (summary.passed_count, summary.total_count) == (1, 1)
    assert (item.index, item.scenario_name, item.passed) == (0, "Good", True)
    assert case.passed is True


def _multirun_job(base: Path, overrides: bytes) -> Path:
    """A one-job multirun session whose ``.hydra/overrides.yaml`` is
    *overrides*."""
    job = base / "multirun" / DATE / GOOD / "0"
    (job / ".hydra").mkdir(parents=True)
    (job / ".hydra" / "overrides.yaml").write_bytes(overrides)
    (job / "Good_result.json").write_text(
        json.dumps({"passed": True, "scenario": "Good"})
    )
    return job


@pytest.mark.parametrize(
    "overrides",
    [b"- a=\xff\n", b"\xff\xfe- a=1\n", b"- [unclosed\n", b"{a: 1}\n"],
    ids=["not-utf8", "utf16-bom", "not-yaml", "not-a-list"],
)
def test_overrides_that_cannot_be_read_are_none(
    tmp_path: Path, overrides: bytes
) -> None:
    """#1631: ``read_text`` raised ``UnicodeDecodeError`` (a ``ValueError``)
    out of the session and case pages; ``results`` already read no overrides."""
    job = _multirun_job(tmp_path, overrides)

    (item,) = scanner.load_session(tmp_path, "multirun", DATE, GOOD)
    view = scanner.load_scenario(tmp_path, "multirun", DATE, GOOD, 0)
    (case,) = results.load_results(tmp_path, "multirun", DATE, GOOD, metrics=False)

    assert scanner._read_overrides(job) == results._hydra_overrides(job) == []
    assert (item.scenario_name, item.passed, item.overrides) == ("Good", True, [])
    assert view is not None
    assert view.overrides == []
    assert case.passed is True


def test_override_items_are_read_as_strings(tmp_path: Path) -> None:
    """Items YAML reads as other types are strings, as ``results`` reads them."""
    job = _multirun_job(tmp_path, b"- a=1\n- 5\n- null\n")

    assert scanner._read_overrides(job) == ["a=1", "5", "None"]
    assert scanner._read_overrides(job) == results._hydra_overrides(job)


def test_the_pages_of_a_run_with_overrides_not_utf8_are_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1631: one damaged run made the session and case pages answer 500."""
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    _multirun_job(tmp_path, b"- a=\xff\n")
    client = TestClient(app_module.app, base_url="http://127.0.0.1")

    session = client.get(f"/session/multirun/{DATE}/{GOOD}")
    case = client.get(f"/session/multirun/{DATE}/{GOOD}/0")

    assert session.status_code == 200, session.text
    assert case.status_code == 200, case.text


def test_sessions_of_the_same_second_list_by_number(tmp_path: Path) -> None:
    """#1845: newest first, ``-10`` before ``-9`` (a text sort reverses
    them), the bare time last."""
    for time in ("12-00-00", "12-00-00-1", "12-00-00-9", "12-00-00-10"):
        _session(
            tmp_path,
            time,
            "Good_result.json",
            json.dumps({"passed": True, "scenario": "Good"}).encode(),
        )
    assert [s.time for s in scanner.scan_sessions(tmp_path)] == [
        "12-00-00-10",
        "12-00-00-9",
        "12-00-00-1",
        "12-00-00",
    ]


def _unreadable(monkeypatch: pytest.MonkeyPatch, *directories: Path) -> None:
    """Make listing *directories* raise, as for another user's directory of
    mode 0700: chmod does nothing for root, which the tests may run as."""
    real = Path.iterdir

    def iterdir(self: Path) -> Iterator[Path]:
        if self in directories:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)


@pytest.mark.parametrize("root", ["outputs", "multirun"])
def test_a_directory_it_cannot_list_is_left_out_of_the_session_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    root: str,
) -> None:
    """#2183: an unreadable date directory, or session directory, is skipped
    with a warning; every other session is still listed."""
    _with_good_session(tmp_path)
    other = "2026-01-02"
    multirun_job = tmp_path / "multirun" / DATE / BAD / "0"
    multirun_job.mkdir(parents=True)
    (multirun_job / "Good_result.json").write_text(
        json.dumps({"passed": True, "scenario": "Good"})
    )
    (tmp_path / root / other / GOOD).mkdir(parents=True)
    _unreadable(monkeypatch, tmp_path / root / other, multirun_job.parent)

    with caplog.at_level("WARNING", logger=scanner.__name__):
        sessions = scanner.scan_sessions(tmp_path)

    assert [(s.session_type, s.date, s.time) for s in sessions] == [
        ("single", DATE, GOOD)
    ]
    assert sum("Cannot list" in r.getMessage() for r in caplog.records) == 2


def test_a_multirun_session_it_cannot_list_has_no_item_and_no_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2183: the session page of an unreadable multirun session is served,
    with no case, and the session is read again once it can be listed."""
    job = _multirun_job(tmp_path, b"- a=1\n")
    _unreadable(monkeypatch, job.parent)
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    client = TestClient(app_module.app, base_url="http://127.0.0.1")

    assert scanner.load_session(tmp_path, "multirun", DATE, GOOD) == []
    assert results.load_results(tmp_path, "multirun", DATE, GOOD, metrics=False) == []
    page = client.get(f"/session/multirun/{DATE}/{GOOD}")
    assert page.status_code == 200, page.text

    monkeypatch.undo()
    (item,) = scanner.load_session(tmp_path, "multirun", DATE, GOOD)
    assert item.scenario_name == "Good"


def _counts_shown(client: TestClient, session_type: str, time: str) -> tuple[str, str]:
    """The counts the session list shows for the session at *time*, and the
    ones its own page shows, each as its text with the tags and spacing
    taken out."""
    import re

    def text(html: str) -> str:
        return " ".join(re.sub(r"<[^>]+>", " ", html).split())

    index = client.get("/")
    assert index.status_code == 200, index.text
    link = f"/session/{session_type}/{DATE}/{time}"
    row = index.text[index.text.index(f'href="{link}"') :]
    row = row[: row.index("</tr>")]
    listed = re.search(r"\d+/\d+ passed( \d+ pending)?", text(row))
    assert listed is not None, row
    page = client.get(link)
    assert page.status_code == 200, page.text
    header = page.text[page.text.index('<div class="text-right">') :]
    header = header[: header.index("<!-- Pass rate bar")]
    return listed.group(0), text(header)


@pytest.mark.parametrize("session_type", ["multirun", "single"])
def test_the_session_list_and_the_session_page_count_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_type: str
) -> None:
    """#2285: the session list counted a missing or unreadable result as
    failed, the session page as without a result, and a single run whose
    result is corrupt showed "0/0 All passed" on its page.  Both count it
    as without a result now, the page lists it, and it is never all passed;
    a result without a verdict counts as failed on both."""
    if session_type == "multirun":
        # A passing job, one that crashed (no result) and one whose result
        # names no verdict.
        session = tmp_path / "multirun" / DATE / GOOD
        for index in (0, 1, 2):
            (session / str(index) / ".hydra").mkdir(parents=True)
        (session / "0" / "foo_result.json").write_text(
            json.dumps({"scenario": "foo", "passed": True, "message": "ok"})
        )
        (session / "2" / "foo_result.json").write_text(
            json.dumps({"scenario": "foo", "passed": "yes", "message": "?"})
        )
        want_list = "1/3 passed 1 pending"
        want_page = "1/3 1 failed, 1 without a result"
        want_items = [(0, True, False), (1, None, True), (2, None, False)]
    else:
        _session(tmp_path, GOOD, "foo_result.json", b"{truncated")
        want_list = "0/1 passed 1 pending"
        want_page = "0/1 1 without a result"
        want_items = [(0, None, True)]
    monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    client = TestClient(app_module.app, base_url="http://127.0.0.1")

    listed, page = _counts_shown(client, session_type, GOOD)

    assert listed == want_list
    assert page == want_page
    assert "All passed" not in page
    items = scanner.load_session(tmp_path, session_type, DATE, GOOD)
    assert [(i.index, i.passed, i.pending) for i in items] == want_items


@pytest.mark.parametrize("chunks", [[5], 5, [None]], ids=["number", "not-list", "null"])
def test_a_slurm_batch_whose_later_submission_record_is_malformed_still_lists(
    tmp_path: Path, chunks: object
) -> None:
    """#2285: whether a SLURM case's outcome is stale reads the later
    submissions' records, and one whose ``chunks`` hold no list raises a
    TypeError there; the outcome is then shown as it is, and the session list
    still lists every session."""
    from autoware_carla_scenario.slurm.manifest import MANIFEST_VERSION, BatchLayout
    from autoware_carla_scenario.ui import cases

    _with_good_session(tmp_path)
    layout = BatchLayout(tmp_path / "outputs" / DATE / BAD)
    for path, document in (
        (
            layout.manifest,
            {
                "version": MANIFEST_VERSION,
                "cases": [{"index": 0, "label": "a"}],
                "launcher": {},
            },
        ),
        (layout.submission(0), {"index": 0, "chunks": [[0]]}),
        (layout.submission(1), {"index": 1, "chunks": chunks}),
        (
            layout.result(0),
            {
                "submission": 0,
                "result": {
                    "passed": True,
                    "message": "ok",
                    "elapsed_seconds": 1.0,
                    "condition_statuses": [],
                },
            },
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))

    found = cases.batch_entries(layout.batch_dir)

    assert found is not None
    assert [e["passed"] for e in found[0]] == [True]
    sessions = {s.time: s for s in scanner.scan_sessions(tmp_path)}
    assert sessions[GOOD].passed_count == 1
    assert (sessions[BAD].passed_count, sessions[BAD].total_count) == (1, 1)
