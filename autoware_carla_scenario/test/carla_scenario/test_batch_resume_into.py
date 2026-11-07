"""A local batch resumed into its own directory is gated as a whole (#1957).

A batch resumed with ``--resume-from`` alone ran its remaining cases in a new
directory: the killed batch's ``batch_results.json`` kept them pending, so
``scenario-gate`` over both directories refused (exit 2), and with
``--allow-unrecorded`` counted a case killed after writing its gate record
twice.  ``--resume-into DIR`` runs them in the killed batch's directory,
rewrites its summary and replaces the outputs of the cases it runs again.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.gate.criteria import find_gate_records
from autoware_carla_scenario.gate.evaluate import EXIT_ERROR, EXIT_PASSED
from autoware_carla_scenario.gate.evaluate import main as gate_main
from autoware_carla_scenario.gate.evaluate import unrecorded_cases
from autoware_carla_scenario.gate.record import GateCase, GateRecord, write_gate_record

#: A scenario without a sweep of its own: drawing from the ODD needs no map.
_CONCRETE = "intersection_passing/straight_3npc"
_SWEEP = ["+sweep.odd_sample.count=4"]
#: The stems of its four cases: the stand-in scenarios are ``object``s.
_STEMS = ["object", "object-1", "object-2", "object-3"]


class _Killed(KeyboardInterrupt):
    """The batch dies mid-case: what its last summary said is what it leaves."""


def _gate_record(passed: bool = True) -> GateRecord:
    return GateRecord(
        scenario="left_turn",
        title="Left turn",
        digest="d" * 64,
        gate={"min_success_rate": 0.5},
        case=GateCase(
            map="Town",
            index=0,
            lanelet_ids=(1, 2, 3),
            case_count=4,
            segment_kinds=("lane", "junction", "lane"),
            segment_ends=(40.0, 60.0, 100.0),
        ),
        passed=passed,
        message="ok" if passed else "collision",
        elapsed_seconds=12.0,
        failure_category=None if passed else "collision",
        ego=[(i * 0.1, float(i), 0.0, 10.0) for i in range(101)],
    )


def _write_run(directory: Path, stem: str, passed: bool = True) -> None:
    """What a run of a logical scenario leaves under *stem*: its gate record,
    trajectory log and result."""
    write_gate_record(directory / f"{stem}_gate.json", _gate_record(passed))
    (directory / f"{stem}.trajectory.jsonl").write_text("{}\n")
    (directory / f"{stem}_result.json").write_text(json.dumps({"passed": passed}))


class _Batch:
    """Runs ``run_batch`` on a fake queue that writes each case's outputs
    under the stem it is given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from autoware_carla_scenario.examples import run

        self.run = run
        self.root = tmp_path / "outputs"
        self.monkeypatch = monkeypatch
        self.added: list[str | None] = []
        monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
        monkeypatch.setattr(run, "build_scenario", lambda *_a, **_k: (None, object()))
        monkeypatch.setattr(
            run,
            "resolve_map_paths",
            lambda _m: SimpleNamespace(
                install_xodr=None,
                overwrite_xodr=False,
                opendrive_path=None,
                lanelet2_path=None,
                name="M",
                projector_type=None,
            ),
        )
        monkeypatch.setattr(run, "build_traffic_backend", lambda _cfg: None)

    def __call__(
        self,
        outcomes: list[bool | None],
        *,
        new_dir: str | None = None,
        kill_at: int | None = None,
        record_before_kill: bool = False,
        **kwargs: Any,
    ) -> Path | None:
        """Run the batch: case *i* passes, fails, or (``None``) errors
        without writing anything; at case *kill_at* it dies, after writing
        that case's gate record with *record_before_kill*.  A new batch
        directory is *new_dir* (under ``outputs/2026-01-01``)."""
        added = self.added
        added.clear()

        class FakeQueue:
            def __init__(self, *, output_dir: Path, **_: object) -> None:
                self._dir = output_dir
                self._errored: list[int] = []

            def add(
                self, *_: object, file_stem: str | None = None, **__: object
            ) -> None:
                added.append(file_stem)

            def __enter__(self) -> FakeQueue:
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def run_all(
                self,
                *,
                on_result: Callable[[int, ScenarioResult], None] | None = None,
                **_: object,
            ) -> list[ScenarioResult]:
                assert on_result is not None
                results = []
                for i, (stem, passed) in enumerate(zip(added, outcomes)):
                    assert stem is not None
                    if i == kill_at:
                        if record_before_kill:
                            write_gate_record(
                                self._dir / f"{stem}_gate.json", _gate_record()
                            )
                        raise _Killed
                    if passed is None:
                        self._errored.append(i)
                        result = ScenarioResult(False, "Errored", 0.0)
                    else:
                        _write_run(self._dir, stem, passed)
                        result = ScenarioResult(passed, "", float(i + 1))
                    results.append(result)
                    on_result(i, result)
                return results

            def errored_indices(self) -> list[int]:
                return list(self._errored)

        def make_dir() -> Path:
            assert new_dir is not None, "a batch resumed into its own directory"
            path = self.root / "2026-01-01" / new_dir
            path.mkdir(parents=True)
            return path

        self.monkeypatch.setattr(self.run, "ScenarioQueue", FakeQueue)
        self.monkeypatch.setattr(self.run, "_make_batch_output_dir", make_dir)
        with pytest.raises((SystemExit, _Killed)) as raised:
            self.run.run_batch([_CONCRETE], list(_SWEEP), expand=True, **kwargs)
        self.exit = raised.value
        return None

    @property
    def first(self) -> Path:
        return self.root / "2026-01-01" / "12-00-00"


def _killed_batch(batch: _Batch, *, record_before_kill: bool = True) -> Path:
    """A four-case batch killed in its third case: two passed, two pending."""
    batch(
        [True, True, True, True],
        new_dir="12-00-00",
        kill_at=2,
        record_before_kill=record_before_kill,
    )
    assert isinstance(batch.exit, _Killed)
    return batch.first


def _entries(directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / "batch_results.json").read_text())


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Batch:
    return _Batch(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# The stems a batch gives its cases, and records
# ---------------------------------------------------------------------------


def test_a_batch_names_each_case_s_stem_and_records_it(batch: _Batch) -> None:
    directory = _killed_batch(batch)
    assert batch.added == _STEMS
    entries = _entries(directory)
    assert [e["stem"] for e in entries] == _STEMS
    assert [e.get("pending", False) for e in entries] == [False, False, True, True]


def test_an_errored_case_keeps_its_stem(batch: _Batch) -> None:
    """A case that raised before writing anything does not hand its stem to
    the next case of its class: each case's outputs are its own."""
    batch([True, None, True, True], new_dir="12-00-00")
    assert isinstance(batch.exit, SystemExit)
    assert [e["stem"] for e in _entries(batch.first)] == _STEMS
    assert not (batch.first / "object-1_result.json").exists()
    assert (batch.first / "object-2_result.json").exists()


# ---------------------------------------------------------------------------
# The issue: resumed, the batch is gated as a whole
# ---------------------------------------------------------------------------


def test_a_batch_resumed_into_its_directory_is_gated_as_a_whole(
    batch: _Batch,
) -> None:
    directory = _killed_batch(batch)
    kept = _entries(directory)[:2]
    assert gate_main([str(batch.root)]) == EXIT_ERROR

    # Resumed at its first case without a result: the third.
    batch([True, True], resume_into=directory)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 0
    assert batch.added == _STEMS[2:]
    # No other directory.
    assert [p.name for p in (batch.root / "2026-01-01").iterdir()] == ["12-00-00"]

    entries = _entries(directory)
    assert entries[:2] == kept
    assert [e["stem"] for e in entries] == _STEMS
    assert [e.get("pending", False) for e in entries] == [False] * 4
    assert [e["passed"] for e in entries] == [True] * 4
    assert all(e["overrides"] for e in entries)

    assert unrecorded_cases(batch.root) == []
    # Each case once.
    assert len(find_gate_records([batch.root])) == 4
    assert gate_main([str(batch.root)]) == EXIT_PASSED


def test_a_killed_case_s_gate_record_is_replaced_not_counted_twice(
    batch: _Batch,
) -> None:
    """The case killed after writing its gate record errors when run again,
    writing nothing: its earlier record is gone, and the gate refuses the
    errored case rather than judge its killed run."""
    directory = _killed_batch(batch, record_before_kill=True)
    assert (directory / "object-2_gate.json").is_file()

    batch([None, True], resume_into=directory)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 1
    assert not (directory / "object-2_gate.json").exists()
    entries = _entries(directory)
    assert [e.get("error", False) for e in entries] == [False, False, True, False]
    assert len(find_gate_records([batch.root])) == 3
    (line,) = unrecorded_cases(batch.root)
    assert "1 of 4 case(s)" in line
    # Judged over the cases that ran, the killed run is not among them.
    assert gate_main([str(batch.root), "--allow-unrecorded"]) == EXIT_PASSED
    assert len(find_gate_records([batch.root])) == 3


def test_resume_from_runs_a_finished_case_again_in_place(batch: _Batch) -> None:
    """``--resume-from 2`` runs the second case again: its outputs are
    replaced, the first case's are left alone."""
    directory = _killed_batch(batch)
    (directory / "object-1.sumo").mkdir()
    (directory / "object-1.sumo" / "fcd.xml").write_text("<fcd/>")
    first = (directory / "object_gate.json").read_text()

    batch([False, True, True], resume_into=directory, resume_from=2)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 1
    assert batch.added == _STEMS[1:]
    assert not (directory / "object-1.sumo").exists()
    assert (directory / "object_gate.json").read_text() == first
    entries = _entries(directory)
    assert [e["passed"] for e in entries] == [True, False, True, True]
    assert len(find_gate_records([batch.root])) == 4


def test_a_resumed_batch_killed_again_keeps_its_cases_pending(
    batch: _Batch,
) -> None:
    directory = _killed_batch(batch)
    batch([True, True], resume_into=directory, kill_at=1)
    assert isinstance(batch.exit, _Killed)
    entries = _entries(directory)
    assert [e.get("pending", False) for e in entries] == [False, False, False, True]
    assert [e["stem"] for e in entries] == _STEMS
    (line,) = unrecorded_cases(batch.root)
    assert "1 of 4 case(s)" in line


def test_the_resumed_batch_through_the_cli(
    batch: _Batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _killed_batch(batch)
    seen: dict[str, Any] = {}
    monkeypatch.setenv("SWEEP_RESUME_FROM", "0")
    monkeypatch.setattr(
        batch.run, "run_batch", lambda names, overrides, **kw: seen.update(kw)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scenario",
            "--multirun",
            "scenario=intersection_passing/straight_3np*",
            *_SWEEP,
            "--resume-into",
            str(directory),
        ],
    )
    batch.run.main()
    assert seen == {"expand": True, "resume_from": 0, "resume_into": directory}


# ---------------------------------------------------------------------------
# Refused: nothing removed, nothing run
# ---------------------------------------------------------------------------


def _listing(directory: Path) -> dict[str, str]:
    return {
        p.name: p.read_text() if p.is_file() else ""
        for p in sorted(directory.iterdir())
    }


def _refused(batch: _Batch, capsys: Any, directory: Path, **kwargs: Any) -> str:
    before = _listing(directory) if directory.is_dir() else None
    batch([True, True, True, True], resume_into=directory, **kwargs)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 2
    assert batch.added == []
    if before is not None:
        assert _listing(directory) == before
    err: str = capsys.readouterr().err
    assert err.startswith("Error: --resume-into ")
    return err


def test_no_such_directory_is_refused(
    batch: _Batch, capsys: Any, tmp_path: Path
) -> None:
    assert "no such directory" in _refused(batch, capsys, tmp_path / "nowhere")


def test_a_directory_without_a_summary_is_refused(
    batch: _Batch, capsys: Any, tmp_path: Path
) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "object_gate.json").write_text("{}")
    assert "holds no batch_results.json" in _refused(batch, capsys, tmp_path / "run")


def test_an_unreadable_summary_is_refused(batch: _Batch, capsys: Any) -> None:
    directory = _killed_batch(batch)
    (directory / "batch_results.json").write_text("[{")
    assert "cannot be read" in _refused(batch, capsys, directory)


def test_a_batch_of_other_cases_is_refused(
    batch: _Batch, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _killed_batch(batch)
    monkeypatch.setattr(sys.modules[__name__], "_SWEEP", ["+sweep.odd_sample.count=3"])
    assert "lists 4 case(s), this batch has 3" in _refused(batch, capsys, directory)


def test_a_batch_of_other_sweep_cases_is_refused(batch: _Batch, capsys: Any) -> None:
    directory = _killed_batch(batch)
    entries = _entries(directory)
    entries[3]["overrides"] = ["scenario.speed=1"]
    (directory / "batch_results.json").write_text(json.dumps(entries))
    assert "case 4" in _refused(batch, capsys, directory)


def test_a_summary_without_stems_is_refused(batch: _Batch, capsys: Any) -> None:
    """An older batch's summary names no stem: which outputs belong to the
    cases run again cannot be told, so nothing is removed."""
    directory = _killed_batch(batch)
    entries = _entries(directory)
    for entry in entries:
        del entry["stem"]
    (directory / "batch_results.json").write_text(json.dumps(entries))
    err = _refused(batch, capsys, directory)
    assert "names no file stem" in err
    assert "nothing is removed" in err


@pytest.mark.parametrize("stem", ["../object-2", "", "object"])
def test_an_unsafe_or_repeated_stem_is_refused(
    batch: _Batch, capsys: Any, stem: str
) -> None:
    directory = _killed_batch(batch)
    entries = _entries(directory)
    entries[2]["stem"] = stem
    (directory / "batch_results.json").write_text(json.dumps(entries))
    assert "not a file name of its own" in _refused(batch, capsys, directory)


def test_a_stem_of_another_scenario_class_is_refused(
    batch: _Batch, capsys: Any
) -> None:
    directory = _killed_batch(batch)
    entries = _entries(directory)
    entries[3]["stem"] = "CutInScenario-3"
    (directory / "batch_results.json").write_text(json.dumps(entries))
    assert "not a run of object" in _refused(batch, capsys, directory)


def test_resuming_past_a_case_without_a_result_is_refused(
    batch: _Batch, capsys: Any
) -> None:
    directory = _killed_batch(batch)
    err = _refused(batch, capsys, directory, resume_from=4)
    assert "case 3" in err and "--resume-from 3" in err


def test_a_finished_batch_without_resume_from_is_refused(
    batch: _Batch, capsys: Any
) -> None:
    batch([True] * 4, new_dir="12-00-00")
    assert "every case of the batch has a result" in _refused(
        batch, capsys, batch.first
    )


def test_a_slurm_batch_s_directory_is_refused(batch: _Batch, capsys: Any) -> None:
    directory = _killed_batch(batch)
    (directory / "slurm").mkdir()
    (directory / "slurm" / "manifest.json").write_text("{}")
    assert "SLURM" in _refused(batch, capsys, directory)


def test_a_single_run_refuses_resume_into(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, tmp_path: Path
) -> None:
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("SWEEP_RESUME_FROM", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["scenario", f"scenario={_CONCRETE}", "--resume-into", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as raised:
        run.main()
    assert raised.value.code == 2
    assert "--resume-into resumes a local batch" in capsys.readouterr().err


def test_a_summary_that_cannot_be_rewritten_removes_nothing(
    batch: _Batch, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk when the summary is rewritten before the outputs go: the
    summary still says the cases run again passed, so their outputs stay and
    the batch stops (exit 2); it never leaves results without outputs, which
    scenario-gate would judge as they are."""
    batch([True, True, True, True], new_dir="12-00-00")
    directory = batch.first
    before = _listing(directory)

    def full(*_: object, **__: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(batch.run, "write_batch_result_json", full)
    capsys.readouterr()
    batch([True, True, True], resume_into=directory, resume_from=2, kill_at=0)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 2
    err = capsys.readouterr().err
    assert "the batch summary cannot be rewritten" in err
    assert "nothing was removed" in err
    assert _listing(directory) == before
    assert len(find_gate_records([batch.root])) == 4
    assert gate_main([str(batch.root)]) == EXIT_PASSED


def test_a_kept_case_keeps_its_failure_and_failed_recording(
    batch: _Batch, capsys: Any
) -> None:
    """The summary's table lists a kept failed case's category, and counts a
    kept recording that failed to convert."""
    directory = _killed_batch(batch)
    entries = _entries(directory)
    entries[1].update(
        passed=False,
        failure={"category": "collision", "message": "hit", "at": 1.0},
        rerun_recording="object-1.rrd",
        rerun_error="conversion failed",
    )
    (directory / "batch_results.json").write_text(json.dumps(entries))
    capsys.readouterr()
    batch([True, True], resume_into=directory)
    assert isinstance(batch.exit, SystemExit) and batch.exit.code == 1
    out = capsys.readouterr().out
    assert "[collision]" in out
    assert "[FAILED] intersection_passing/straight_3npc#2: conversion failed" in out
    kept = _entries(directory)[1]
    assert kept["rerun_error"] == "conversion failed"
    assert kept["failure"]["category"] == "collision"
