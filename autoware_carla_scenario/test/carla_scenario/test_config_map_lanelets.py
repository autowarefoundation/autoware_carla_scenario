"""A config written for one map's lanelet ids, refused on another (#2253).

The shipped adversity cases are placed on Town10HD_Opt; composed on the
default map (Nishi-Shinjuku), lanelet 719 is not there and 324/446 name other
lanelets.  Two checks, both before anything starts and without CARLA:

* ``scenario.required_map`` -- the ``map.name`` a case was written for -- is
  compared with the composed map (``required_map_refusal``, through
  ``scenario_map_refusal``: a run, a batch, ``scenario-expand`` and the
  viewer's sweep resolver).
* every lanelet id a config names is looked up in the Lanelet2 map the run
  resolved (``check_config_lanelets``).

A scenario a glob picked that does not fit the map is skipped with a warning
instead, and the rest run (``map_fit.skip_off_map_scenarios``): a batch, a
SLURM batch, ``scenario-expand`` and the viewer's launcher.  A scenario named
alone is still refused; a glob whose every scenario is skipped is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from autoware_carla_scenario import map_fit

from autoware_carla_scenario.authoring.hydra_config import (
    map_groups_named,
    required_map_refusal,
    scenario_map_refusal,
)
from autoware_carla_scenario.examples import expand, run
from autoware_carla_scenario.examples.configs import AdversityConfig
from autoware_carla_scenario.ui import app as app_module
from autoware_carla_scenario.ui import runner

#: The nishishinjuku fixture map, kept at the repository root.
_OSM = Path(__file__).resolve().parents[3] / "data" / "nishishinjuku.osm"

_ADVERSITY_CASES = run._resolve_scenario_glob("adversity/*")


def _nishishinjuku() -> Any:
    """The resolved map of the default group, as ``resolve_map_paths`` gives
    it, without resolving anything."""
    return SimpleNamespace(name="NishishinjukuMap", lanelet2_path=_OSM)


class TestRequiredMap:
    def test_every_shipped_adversity_case_names_town10hd_opt(self) -> None:
        assert len(_ADVERSITY_CASES) == 6
        for case in _ADVERSITY_CASES:
            cfg = run._compose_config(case, ["map=town10hd_opt"])
            assert cfg.scenario.required_map == "Town10HD_Opt"
            assert cfg.map.name == "Town10HD_Opt"

    @pytest.mark.parametrize("case", _ADVERSITY_CASES)
    def test_a_shipped_case_is_accepted_on_its_map(self, case: str) -> None:
        cfg = run._compose_config(case, ["map=town10hd_opt"])
        assert required_map_refusal(cfg) is None
        assert scenario_map_refusal(cfg) is None
        run.check_scenario_maps([cfg])

    @pytest.mark.parametrize("case", _ADVERSITY_CASES)
    def test_a_shipped_case_is_refused_on_the_default_map(self, case: str) -> None:
        cfg = run._compose_config(case, [])
        refusal = scenario_map_refusal(cfg)
        assert refusal is not None
        assert "requires map 'Town10HD_Opt'" in refusal
        assert "map.name is 'NishishinjukuMap'" in refusal
        assert refusal.endswith("Run it with map=town10hd_opt")
        with pytest.raises(run.RunConfigError, match="requires map 'Town10HD_Opt'"):
            run.check_run_config(cfg)

    def test_a_batch_names_the_case_refused(self) -> None:
        cfg = run._compose_config("adversity/cut_in", [])
        with pytest.raises(
            run.RunConfigError,
            match=r"^adversity/cut_in: scenario 'adversity \(cut_in\)'",
        ):
            run.check_scenario_maps([cfg], ["adversity/cut_in"])

    @pytest.mark.parametrize("value", ["null", "''"])
    def test_no_required_map_is_no_requirement(self, value: str) -> None:
        cfg = run._compose_config(
            "adversity/cut_in", [f"scenario.required_map={value}"]
        )
        assert required_map_refusal(cfg) is None

    def test_a_map_with_no_group_is_named_without_one(self) -> None:
        cfg = OmegaConf.create(
            {
                "scenario": {"name": "adversity", "required_map": "Elsewhere"},
                "map": {"name": "NishishinjukuMap"},
            }
        )
        refusal = required_map_refusal(cfg)
        assert refusal is not None
        assert refusal.endswith("Run it on a map group whose map.name is 'Elsewhere'")

    def test_the_groups_of_a_map_are_found_by_its_name(self) -> None:
        assert map_groups_named("Town10HD_Opt") == ["town10hd_opt"]
        assert map_groups_named("NishishinjukuMap") == ["nishishinjuku"]

    @pytest.mark.parametrize("value", ["", "  ", True, 10])
    def test_a_required_map_that_is_no_name_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"^scenario\.required_map must be"):
            AdversityConfig(required_map=value)

    def test_the_default_config_requires_no_map(self) -> None:
        assert AdversityConfig().required_map is None


class TestConfigLanelets:
    def test_the_map_s_lanelets_are_read_off_the_osm(self) -> None:
        ids = run.map_lanelet_ids(_OSM)
        assert ids is not None
        assert {324, 446} <= ids
        assert 719 not in ids
        assert run.map_lanelet_ids(str(_OSM)) is ids

    @pytest.mark.parametrize(
        "path", [None, MagicMock(), "/nonexistent/lanelet2_map.osm"]
    )
    def test_no_map_to_read_is_not_checked(self, path: Any) -> None:
        assert run.map_lanelet_ids(path) is None

    def test_a_file_with_no_lanelet_is_not_checked(self, tmp_path: Path) -> None:
        osm = tmp_path / "empty.osm"
        osm.write_text("<osm><node id='1' lat='0' lon='0'/></osm>")
        assert run.map_lanelet_ids(osm) is None
        broken = tmp_path / "broken.osm"
        broken.write_text("<osm><relation")
        assert run.map_lanelet_ids(broken) is None

    def test_lanelets_are_the_relations_lanelet2_loads(self, tmp_path: Path) -> None:
        """Every ``type=lanelet`` relation, a JOSM one with a negative id too,
        but none marked ``action=delete`` (Lanelet2 skips those) and no other
        relation (an area, a regulatory element)."""
        osm = tmp_path / "map.osm"
        osm.write_text(
            "<osm>"
            "<relation id='5'><tag k='type' v='lanelet'/></relation>"
            "<relation id='-6' action='modify'><tag k='type' v='lanelet'/></relation>"
            "<relation id='7' action='delete'><tag k='type' v='lanelet'/></relation>"
            "<relation id='8'><tag k='type' v='multipolygon'/></relation>"
            "<relation id='9'><tag k='type' v='regulatory_element'/></relation>"
            "</osm>"
        )
        assert run.map_lanelet_ids(osm) == frozenset({5, -6})

    def test_the_ego_s_and_the_npc_s_lanelets_are_named(self) -> None:
        cfg = run._compose_config(
            "adversity/traffic_rule",
            [
                "map=town10hd_opt",
                "ego.goal_lanelet_id=5",
                "ego.waypoint_lanelet_ids=[7,8]",
            ],
        )
        assert run.config_lanelet_ids(cfg) == [
            ("ego.spawn_lanelet_id", 719),
            ("ego.goal_lanelet_id", 5),
            ("ego.waypoint_lanelet_ids[0]", 7),
            ("ego.waypoint_lanelet_ids[1]", 8),
            ("scenario.npc_lanelet_id", 719),
        ]

    def test_nade_s_npc_lanelet_is_not_read(self) -> None:
        cfg = run._compose_config("adversity/nade", ["map=town10hd_opt"])
        assert run.config_lanelet_ids(cfg) == [("ego.spawn_lanelet_id", 324)]

    def test_another_scenario_s_keys_are_not_read(self) -> None:
        cfg = OmegaConf.create(
            {"scenario": {"name": "cut_in", "npc_lanelet_id": 0}, "ego": {}}
        )
        assert run.config_lanelet_ids(cfg) == []

    def test_a_lanelet_not_on_the_map_is_refused(self) -> None:
        cfg = run._compose_config(
            "adversity/traffic_rule", ["scenario.required_map=null"]
        )
        with pytest.raises(run.RunConfigError) as raised:
            run.check_config_lanelets([cfg], _nishishinjuku(), ["case"])
        message = str(raised.value)
        assert message.startswith("case: no lanelet of map 'NishishinjukuMap'")
        assert "ego.spawn_lanelet_id=719, scenario.npc_lanelet_id=719" in message

    def test_lanelets_on_the_map_are_accepted(self) -> None:
        cfg = run._compose_config("adversity/cut_in", ["scenario.required_map=null"])
        run.check_config_lanelets([cfg], _nishishinjuku())

    def test_a_map_with_no_lanelet2_file_is_not_checked(self) -> None:
        cfg = run._compose_config(
            "adversity/traffic_rule", ["scenario.required_map=null"]
        )
        run.check_config_lanelets([cfg], SimpleNamespace(name="x", lanelet2_path=None))

    def test_a_run_is_refused_once_its_map_is_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = run._compose_config(
            "adversity/traffic_rule", ["scenario.required_map=null"]
        )
        monkeypatch.setattr(
            run, "build_scenario", lambda *_a, **_k: (None, MagicMock())
        )
        monkeypatch.setattr(run, "check_odd", lambda *_a, **_k: None)
        monkeypatch.setattr(run, "resolve_map_paths", lambda _map: _nishishinjuku())
        queue = MagicMock(side_effect=AssertionError("the run started"))
        monkeypatch.setattr(run, "run_scenario_with_queue", queue)
        with pytest.raises(run.RunConfigError, match="no lanelet of map"):
            run._run_scenario(cfg, None)
        queue.assert_not_called()


class _Selected(Exception):
    """Raised where a batch has checked its cases, with their labels."""


def _stop_at_the_check(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The labels of the cases each batch checks; the batch stops there."""
    seen: list[list[str]] = []

    def check(_configs: Any, labels: Any = None, **_: Any) -> None:
        seen.append(list(labels or []))
        raise _Selected

    monkeypatch.setattr(run, "check_batch_configs", check)
    return seen


def _compose(overrides: list[str]) -> Any:
    return lambda name: run._compose_config(name, overrides)


class TestGlobSelection:
    def test_a_glob_batch_on_the_default_map_skips_adversity_and_runs_the_rest(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen = _stop_at_the_check(monkeypatch)
        names = run._resolve_scenario_glob("*/*")
        with caplog.at_level("WARNING", logger=map_fit.__name__):
            with pytest.raises(_Selected):
                run.run_batch(names, [], pattern="*/*")
        [labels] = seen
        assert labels == [n for n in names if not n.startswith("adversity/")]
        assert len(labels) == len(names) - len(_ADVERSITY_CASES)
        for case in _ADVERSITY_CASES:
            assert any(
                record.getMessage().startswith(f"Skipping {case} (picked by ")
                and "requires map 'Town10HD_Opt'" in record.getMessage()
                for record in caplog.records
            ), case

    def test_an_adversity_case_named_alone_is_still_refused(self) -> None:
        with pytest.raises(run.RunConfigError, match="requires map 'Town10HD_Opt'"):
            run.run_batch(["lane_change/left", "adversity/cut_in"], [])

    def test_a_glob_whose_every_scenario_is_skipped_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen = _stop_at_the_check(monkeypatch)
        with pytest.raises(SystemExit) as exited:
            run.run_batch(_ADVERSITY_CASES, [], pattern="adversity/*")
        assert exited.value.code == 1
        assert seen == []
        err = capsys.readouterr().err
        assert "Error: no scenario that scenario=adversity/* matches fits map" in err
        assert "'NishishinjukuMap'" in err

    def test_the_cli_passes_a_glob_on_and_not_a_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            run, "run_batch", lambda names, overrides, **kw: calls.append(kw)
        )
        monkeypatch.setattr(run.sys, "argv", ["scenario", "scenario=adversity/*"])
        run._dispatch(0)
        monkeypatch.setattr(
            run.sys,
            "argv",
            ["scenario", "scenario=adversity/cut_in", "launcher=slurm"],
        )
        run._dispatch(0)
        assert calls[0]["pattern"] == "adversity/*"
        assert "pattern" not in calls[1]

    def test_a_lanelet_not_on_the_map_skips_a_glob_s_scenario(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        overrides = ["scenario.required_map=null"]
        with caplog.at_level("WARNING", logger=map_fit.__name__):
            kept = map_fit.skip_off_map_scenarios(
                "adversity/*",
                _ADVERSITY_CASES,
                _compose(overrides),
                lambda _map: _nishishinjuku(),
            )
        assert "adversity/traffic_rule" not in kept
        assert "adversity/cut_in" in kept
        assert any(
            "Skipping adversity/traffic_rule" in record.getMessage()
            and "ego.spawn_lanelet_id=719" in record.getMessage()
            for record in caplog.records
        )

    def test_a_scenario_that_does_not_compose_is_kept(self) -> None:
        assert map_fit.skip_off_map_scenarios(
            "*/*", ["a/b"], lambda _name: None, lambda _map: _nishishinjuku()
        ) == ["a/b"]

    def test_a_map_that_does_not_resolve_is_not_checked(self) -> None:
        cfg = run._compose_config(
            "adversity/traffic_rule", ["scenario.required_map=null"]
        )

        def unresolved(_map: Any) -> Any:
            raise ValueError("no such map")

        assert map_fit.map_fit_refusal(cfg, unresolved) is None

    def test_the_lanelets_a_sweep_picks_are_not_checked_when_it_is_expanded(
        self,
    ) -> None:
        cfg = OmegaConf.create(
            {
                "scenario": {"name": "lane_change"},
                "map": {"name": "NishishinjukuMap"},
                "ego": {"spawn_lanelet_id": 719, "goal_lanelet_id": 999999999},
                "sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}},
            }
        )
        resolve = lambda _map: _nishishinjuku()  # noqa: E731
        concrete = map_fit.map_fit_refusal(cfg, resolve)
        assert concrete is not None
        assert "ego.spawn_lanelet_id=719, ego.goal_lanelet_id=999999999" in concrete
        swept = map_fit.map_fit_refusal(cfg, resolve, expanding=True)
        assert swept is not None
        assert "ego.spawn_lanelet_id" not in swept
        assert "ego.goal_lanelet_id=999999999" in swept
        cfg.sweep = {"route": {"type": "x"}}
        assert map_fit.map_fit_refusal(cfg, resolve, expanding=True) is None


class TestExpandSelection:
    def test_a_glob_s_scenario_off_the_map_is_skipped(
        self, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger=map_fit.__name__):
            status = expand.main(["scenario=adversity/*", "scenario.required_map=null"])
        assert status == 0
        listed = [
            json.loads(line)["scenario"]
            for line in capsys.readouterr().out.splitlines()
        ]
        assert "adversity/traffic_rule" not in listed
        assert "adversity/cut_in" in listed
        assert any(
            record.getMessage().startswith("Skipping adversity/traffic_rule")
            for record in caplog.records
        )

    def test_a_glob_whose_every_scenario_is_skipped_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert expand.main(["scenario=adversity/*"]) == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert "Error: no scenario that scenario=adversity/* matches fits map" in err

    def test_a_case_named_alone_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert expand.main(["scenario=adversity/cut_in"]) == 1
        assert "requires map 'Town10HD_Opt'" in capsys.readouterr().err


class TestViewerSelection:
    @pytest.fixture
    def started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> list[list[list[str]]]:
        runs: list[list[list[str]]] = []

        def fake_start_run(overrides_list: list[list[str]], **_: Any) -> bool:
            runs.append(overrides_list)
            return True

        monkeypatch.setattr(runner, "start_run", fake_start_run)
        monkeypatch.setattr(app_module, "BASE_PATH", tmp_path)
        return runs

    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app_module.app, base_url="http://127.0.0.1")

    def test_a_glob_s_scenarios_off_the_map_are_not_run(
        self, started: list[list[list[str]]], client: TestClient
    ) -> None:
        response = client.post(
            "/api/run", json={"scenario": "*/*", "extra_overrides": []}
        )
        assert response.json() == {"status": "started"}
        [jobs] = started
        names = [job[0].removeprefix("scenario=") for job in jobs]
        assert names
        assert not [name for name in names if name.startswith("adversity/")]

    def test_a_glob_whose_every_scenario_is_skipped_is_refused(
        self, started: list[list[list[str]]], client: TestClient
    ) -> None:
        response = client.post(
            "/api/run", json={"scenario": "adversity/*", "extra_overrides": []}
        )
        assert response.status_code == 400
        assert response.json()["message"].startswith(
            "no scenario that scenario=adversity/* matches fits map"
        )
        assert started == []
