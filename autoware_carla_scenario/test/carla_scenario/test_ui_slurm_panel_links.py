"""The SLURM panel reads no batch record through a symbolic link (#2549).

A batch whose own manifest is real may still hold a case outcome
(``slurm/results/<i>.json``), a task status (``slurm/tasks/<k>-<t>.json``)
or a submission record (``slurm/submissions/<k>.json``) that is a link, or
sits in a directory that is one, naming a file outside the viewer's base.
The session page reads none of them (``ui.cases._own_member``, #2543); nor
do the panel's status API (``BatchLayout(..., own_files=True)``) and its
actions (``scenario-slurm --own-files``). The ``scenario-slurm`` commands
run from a shell follow links, as they always have.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoware_carla_scenario.slurm import cli
from autoware_carla_scenario.slurm.manifest import BatchLayout
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import cases, scanner

DATE = "2026-11-25"
TIME = "12-00-00"
STATUS = f"/api/slurm/{DATE}/{TIME}/status"


@pytest.fixture
def base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "base"
    root.mkdir()
    monkeypatch.setattr(app_module, "BASE_PATH", root)
    monkeypatch.setenv("VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    scanner.clear_cache()
    yield root
    scanner.clear_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app, base_url="http://127.0.0.1")


@pytest.fixture
def scancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A SLURM whose squeue lists no job and whose scancel logs what it was
    asked to cancel: the file it logs to."""
    bin_dir = tmp_path / "slurm-bin"
    bin_dir.mkdir()
    log = tmp_path / "scancel.log"
    for tool, body in (
        ("squeue", "exit 0"),
        ("sbatch", "exit 0"),
        ("scancel", f'echo "$@" >> "{log}"'),
    ):
        script = bin_dir / tool
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _batch(root: Path) -> Path:
    """A SLURM batch of two cases, its own manifest real: submission 0 (job
    1000) gave case 0 to task 0 and case 1 to task 1; case 1 has passed,
    case 0 has no outcome yet."""
    batch = root / "outputs" / DATE / TIME
    slurm = batch / "slurm"
    _write(
        slurm / "manifest.json",
        {
            "version": 2,
            "created": "2026-11-25T03:00:00+00:00",
            "scenario_names": ["s"],
            "launcher": {"type": "slurm"},
            "cases": [
                {"index": i, "label": f"s#{i + 1}", "case_overrides": [], "stem": s}
                for i, s in enumerate(["Case", "Case-1"])
            ],
        },
    )
    _write(
        slurm / "submissions" / "0.json",
        {"index": 0, "job_id": "1000", "cases": [0, 1], "chunks": [[0], [1]]},
    )
    _write(slurm / "tasks" / "0-1.json", {"state": "done", "host": "node1"})
    _write(
        slurm / "results" / "1.json",
        {
            "index": 1,
            "submission": 0,
            "result": {"passed": True, "message": "ok", "elapsed_seconds": 1.0},
        },
    )
    return batch


def _outside(tmp_path: Path) -> Path:
    """``slurm`` records outside the viewer's base, each naming SECRET."""
    outside = tmp_path / "outside" / "slurm"
    _write(
        outside / "results" / "0.json",
        {
            "index": 0,
            "submission": 0,
            "host": "SECRET_HOST",
            "result": {
                "passed": True,
                "message": "SECRET_MESSAGE",
                "elapsed_seconds": 1.0,
            },
        },
    )
    _write(
        outside / "tasks" / "0-0.json",
        {"state": "failed", "host": "SECRET_HOST", "error": "SECRET_ERROR"},
    )
    _write(
        outside / "submissions" / "0.json",
        {"index": 0, "job_id": "666", "cases": [0, 1], "chunks": [[0], [1]]},
    )
    return outside


#: What is linked, the record under ``slurm/`` and the link it becomes:
#: the file itself, or the directory holding it.
_LINKS = {
    "outcome": ("results/0.json", "results/0.json"),
    "results": ("results/0.json", "results"),
    "task": ("tasks/0-0.json", "tasks/0-0.json"),
    "tasks": ("tasks/0-0.json", "tasks"),
    "submission": ("submissions/0.json", "submissions/0.json"),
    "submissions": ("submissions/0.json", "submissions"),
}


def _link(batch: Path, outside: Path, linked: str) -> None:
    """Replace *batch*'s record (or its directory, with the batch's other
    records in it) with a link to *outside*'s."""
    record, link = _LINKS[linked]
    slurm = batch / "slurm"
    target = slurm / link
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)
    target.symlink_to(outside / link)
    assert (slurm / record).is_file()


def _read(layout: BatchLayout, linked: str) -> object:
    """The linked record, as *layout* reads it."""
    if linked in ("outcome", "results"):
        return layout.read_result(0)
    if linked in ("task", "tasks"):
        return layout.read_task(0, 0)
    return layout.submission_record(0)


@pytest.mark.parametrize("linked", sorted(_LINKS))
def test_the_panel_shows_no_linked_record(
    base: Path,
    tmp_path: Path,
    client: TestClient,
    scancelled: Path,
    linked: str,
) -> None:
    batch = _batch(base)
    _link(batch, _outside(tmp_path), linked)
    # The command line follows the link, as it always has; the viewer's
    # layout does not.
    assert _read(BatchLayout(batch), linked) is not None
    assert _read(BatchLayout(batch, own_files=True), linked) is None
    response = client.get(STATUS)
    assert response.status_code == 200
    assert "SECRET" not in response.text and "666" not in response.text
    status = response.json()
    case = status["cases"][0]
    # The linked outcome's passed verdict is not the case's.
    assert case["state"] != "passed" and "host" not in case
    if linked != "results":
        assert status["cases"][1]["state"] == "passed"
    [submission] = status["submissions"] or [None]
    if linked == "submissions":
        # No submission of the batch's own: none listed, no case given.
        assert submission is None
        assert case["state"] == "missing"
    elif linked == "submission":
        assert submission["unreadable"]
        assert "is reached through a symbolic link" in submission["note"]
    else:
        assert submission["job_id"] == "1000"
        task = submission["tasks"][0]
        if linked in ("task", "tasks"):
            assert task["state"] == "not started" and task["host"] is None
    # The session page agrees: case 0 is pending, nothing linked is shown.
    page = client.get(f"/session/batch/{DATE}/{TIME}")
    assert page.status_code == 200 and "SECRET" not in page.text


@pytest.mark.parametrize("linked", ["outcome", "results"])
def test_collect_from_the_panel_copies_no_linked_outcome(
    base: Path, tmp_path: Path, client: TestClient, linked: str
) -> None:
    batch = _batch(base)
    _link(batch, _outside(tmp_path), linked)
    response = client.post(f"/api/slurm/{DATE}/{TIME}/collect")
    assert response.status_code == 200, response.text
    summary = (batch / "batch_results.json").read_text()
    assert "SECRET" not in summary and "SECRET" not in response.text
    entries = json.loads(summary)
    assert entries[0]["passed"] is not True
    # From a shell, collect reads through the link, as it always has.
    cli.main(["collect", str(batch)])
    assert "SECRET_MESSAGE" in (batch / "batch_results.json").read_text()


@pytest.mark.parametrize("linked", ["submission", "submissions"])
def test_cancel_from_the_panel_cancels_no_linked_job(
    base: Path, tmp_path: Path, client: TestClient, scancelled: Path, linked: str
) -> None:
    batch = _batch(base)
    _link(batch, _outside(tmp_path), linked)
    response = client.post(f"/api/slurm/{DATE}/{TIME}/cancel")
    assert response.status_code == 200, response.text
    assert "666" not in (scancelled.read_text() if scancelled.exists() else "")
    # From a shell, cancel reads the linked record, as it always has.
    cli.main(["cancel", str(batch)])
    assert "666" in scancelled.read_text()


def test_resubmit_with_own_files_takes_a_linked_outcome_as_missing(
    base: Path,
    tmp_path: Path,
    scancelled: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = _batch(base)
    _link(batch, _outside(tmp_path), "outcome")
    assert cli.main(["--own-files", "resubmit", "--dry-run", str(batch)]) == 0
    assert "Resubmitting 1 case(s): s#1" in capsys.readouterr().out
    # Without it, the linked outcome's verdict counts.
    assert cli.main(["resubmit", "--dry-run", str(batch)]) == 0
    assert "Nothing to resubmit" in capsys.readouterr().out


def test_the_viewer_reads_no_linked_submission_to_call_an_outcome_stale(
    base: Path, tmp_path: Path
) -> None:
    """``slurm.collect._stale``, as ``ui.cases`` calls it: a later
    submission record that is a link does not make case 1's outcome stale."""
    batch = _batch(base)
    outside = tmp_path / "outside"
    _write(
        outside / "1.json",
        {"index": 1, "job_id": "1001", "cases": [1], "chunks": [[1]]},
    )
    (batch / "slurm" / "submissions" / "1.json").symlink_to(outside / "1.json")
    _write(batch / "slurm" / "tasks" / "1-0.json", {"state": "failed"})
    assert cases._stale(BatchLayout(batch), 1) is not None
    assert cases._stale(BatchLayout(batch, own_files=True), 1) is None
    found = cases.batch_entries(batch)
    assert found is not None
    entry = found[0][1]
    assert entry["passed"] is True and not entry.get("pending")


def test_own_refuses_only_links_below_the_batch(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / "slurm" / "results").mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    # The batch directory itself may be reached through a link: its path
    # is the viewer's to check (``cases.session_dir``).
    layout = BatchLayout(linked, own_files=True)
    assert layout.own(layout.result(0))
    (real / "slurm" / "results" / "0.json").symlink_to(tmp_path / "x")
    assert not layout.own(layout.result(0))
    assert BatchLayout(linked).own(layout.result(0))
