"""A batch writes ``batch_results.json`` as each case ends, atomically (#1566).

The viewer takes a directory for a batch only when it holds that file; a batch
that wrote it only at its end was listed, while it ran or once cut short
(Ctrl-C, a SIGTERM), as a single run of its first case.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoware_carla_scenario import atomic_file
from autoware_carla_scenario.atomic_file import write_text_atomically
from autoware_carla_scenario.conditions import ScenarioResult
from autoware_carla_scenario.ui import scanner

#: A scenario without a sweep of its own: drawing from the ODD needs no map.
_CONCRETE = "intersection_passing/straight_3npc"
DATE = "2026-01-01"
TIME = "12-00-00"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    scanner.clear_cache()
    yield
    scanner.clear_cache()


def _run_batch_cut_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cases_run: list[bool]
) -> Path:
    """Run a three-case batch whose queue runs *cases_run* (passed or not),
    then is interrupted; return its output directory."""
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    output_dir = tmp_path / "outputs" / DATE / TIME
    output_dir.mkdir(parents=True)

    class FakeQueue:
        def __init__(self, **_: object) -> None:
            self._errored: list[int] = []

        def add(self, *_: object, **__: object) -> None:
            pass

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
            for i, passed in enumerate(cases_run):
                result = ScenarioResult(
                    passed=passed, message="", elapsed_seconds=float(i)
                )
                on_result(i, result)
            raise KeyboardInterrupt

        def errored_indices(self) -> list[int]:
            return list(self._errored)

    def no_summary(*_: object, **__: object) -> bool:
        raise AssertionError("a batch cut short prints no summary")

    monkeypatch.setattr(run, "build_scenario", lambda *_a, **_k: (None, object()))
    monkeypatch.setattr(run, "ScenarioQueue", FakeQueue)
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
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: output_dir)
    monkeypatch.setattr(run, "_print_summary", no_summary)

    with pytest.raises(KeyboardInterrupt):
        run.run_batch([_CONCRETE], ["+sweep.odd_sample.count=3"], expand=True)
    return output_dir


def test_an_interrupted_batch_leaves_the_cases_it_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = _run_batch_cut_short(monkeypatch, tmp_path, [True, False])

    entries = json.loads((output_dir / "batch_results.json").read_text())
    assert [(e["scenario"], e["passed"]) for e in entries] == [
        (f"{_CONCRETE}#1", True),
        (f"{_CONCRETE}#2", False),
    ]
    assert all(e["overrides"] for e in entries)
    # No scratch file is left beside it.
    assert [p.name for p in output_dir.iterdir()] == ["batch_results.json"]

    # The viewer lists it as the batch it is, not a single run.
    (session,) = scanner.scan_sessions(tmp_path)
    assert session.session_type == "batch"
    assert (session.passed_count, session.total_count) == (1, 2)
    items = scanner.load_session(tmp_path, "batch", DATE, TIME)
    assert [i.passed for i in items] == [True, False]


def test_a_batch_interrupted_before_any_case_ended_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = _run_batch_cut_short(monkeypatch, tmp_path, [])

    assert list(output_dir.iterdir()) == []


def test_an_atomic_write_that_fails_leaves_the_old_file_and_no_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "batch_results.json"
    path.write_text("[]", encoding="utf-8")

    def interrupted(*_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_file.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_text_atomically(path, '[{"passed": true}]')

    assert path.read_text(encoding="utf-8") == "[]"
    assert [p.name for p in tmp_path.iterdir()] == ["batch_results.json"]


def test_an_atomic_write_replaces_the_file_whole(tmp_path: Path) -> None:
    path = tmp_path / "batch_results.json"
    path.write_text("old, and longer than the new text", encoding="utf-8")

    write_text_atomically(path, "[]")

    assert path.read_text(encoding="utf-8") == "[]"
    assert os.stat(path).st_mode & 0o777 == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["batch_results.json"]
