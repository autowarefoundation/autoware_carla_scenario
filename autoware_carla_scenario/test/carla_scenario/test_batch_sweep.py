"""A glob batch run with ``--multirun`` runs every case of each scenario's sweep."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import DictConfig

from autoware_carla_scenario.scenario_runner import _run_file_stem

#: A scenario without a sweep of its own: drawing from the ODD needs no map.
_CONCRETE = "intersection_passing/straight_3npc"


class _Batch:
    """What a faked batch built and summarised."""

    def __init__(self) -> None:
        self.configs: list[DictConfig] = []
        self.labels: list[str] = []
        self.overrides: list[list[str]] | None = None


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch) -> _Batch:
    """Fake out everything ``run_batch`` touches past composing and building."""
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    seen = _Batch()

    def fake_build(cfg: DictConfig, **_: object) -> tuple[None, object]:
        seen.configs.append(cfg)
        return None, object()

    class FakeQueue:
        def __init__(self, **_: object) -> None:
            pass

        def add(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> "FakeQueue":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def run_all(self) -> list[object]:
            return []

    def fake_summary(
        names: list[str],
        _results: object,
        output_dir: object = None,
        overrides: list[list[str]] | None = None,
    ) -> bool:
        seen.labels = names
        seen.overrides = overrides
        return True

    monkeypatch.setattr(run, "build_scenario", fake_build)
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
    monkeypatch.setattr(run, "_make_batch_output_dir", lambda: None)
    monkeypatch.setattr(run, "_print_summary", fake_summary)
    return seen


def _weather(cfg: DictConfig) -> tuple[Any, Any]:
    env = cfg.get("environment") or {}
    return env.get("precipitation"), env.get("sun_altitude_angle")


def test_an_expanded_batch_runs_each_case_the_odd_sampler_draws(batch: _Batch) -> None:
    from autoware_carla_scenario.examples import run

    with pytest.raises(SystemExit):
        run.run_batch([_CONCRETE], ["+sweep.odd_sample.count=3"], expand=True)

    assert batch.labels == [f"{_CONCRETE}#{k}" for k in (1, 2, 3)]
    # Each case is built with the settings drawn for it, not the defaults.
    weathers = [_weather(cfg) for cfg in batch.configs]
    assert all(w != (None, None) for w in weathers)
    assert len(set(weathers)) == 3
    assert batch.overrides is not None
    assert all(
        any(o.startswith("environment.") for o in case) for case in batch.overrides
    )
    # The command line's own overrides are not repeated as the case's.
    assert all("+sweep.odd_sample.count=3" not in case for case in batch.overrides)


def test_a_batch_without_multirun_runs_each_scenario_once_and_says_so(
    batch: _Batch, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.examples import run

    with caplog.at_level(logging.WARNING), pytest.raises(SystemExit):
        run.run_batch([_CONCRETE], ["+sweep.odd_sample.count=3"])

    assert batch.labels == [_CONCRETE]
    assert _weather(batch.configs[0]) == (None, None)
    assert batch.overrides is None
    assert "add --multirun" in caplog.text


def test_resuming_an_expanded_batch_keeps_each_case_its_number(batch: _Batch) -> None:
    from autoware_carla_scenario.examples import run

    with pytest.raises(SystemExit):
        run.run_batch(
            [_CONCRETE], ["+sweep.odd_sample.count=3"], expand=True, resume_from=2
        )

    assert batch.labels == [f"{_CONCRETE}#2", f"{_CONCRETE}#3"]
    assert len(batch.configs) == 2


def test_a_concrete_scenario_resumed_past_its_end_is_refused(batch: _Batch) -> None:
    from autoware_carla_scenario.examples import run

    with pytest.raises(SystemExit) as exc:
        run.run_batch([_CONCRETE], [], expand=True, resume_from=2)

    assert exc.value.code == 1
    assert batch.configs == []


def test_multirun_on_a_glob_expands_the_batch_instead_of_reaching_hydra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples import run

    calls: list[tuple[list[str], list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        run,
        "run_batch",
        lambda names, overrides, **kw: calls.append((names, overrides, kw)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scenario",
            "--multirun",
            "hydra/sweeper=lanelet_constraint",
            "scenario=intersection_passing/straight*",
            "map=town10hd_opt",
            "--resume-from",
            "3",
        ],
    )
    run.main()

    (names, overrides, kw) = calls[0]
    assert _CONCRETE in names
    assert overrides == ["map=town10hd_opt"]
    assert kw == {"expand": True, "resume_from": 3}


def test_each_run_of_one_scenario_class_keeps_files_of_its_own(tmp_path: Path) -> None:
    assert _run_file_stem(tmp_path, "CutInScenario") == "CutInScenario"
    (tmp_path / "CutInScenario.trajectory.jsonl").touch()
    assert _run_file_stem(tmp_path, "CutInScenario") == "CutInScenario-1"
    (tmp_path / "CutInScenario-1_result.json").touch()
    assert _run_file_stem(tmp_path, "CutInScenario") == "CutInScenario-2"
    assert _run_file_stem(tmp_path, "LaneChangeScenario") == "LaneChangeScenario"
