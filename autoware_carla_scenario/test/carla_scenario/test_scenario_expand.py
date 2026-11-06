"""Expanding a logical scenario into its concrete scenarios (``scenario-expand``)."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.sweeper import bindings as bindings_module
from autoware_carla_scenario.sweeper import expand as expand_module
from autoware_carla_scenario.sweeper.expand import expand_config, expand_sweep


#: A map source pinned to a commit, so resolving it fetches nothing first.
_PINNED_SOURCE = (
    "git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
    "@5058c6683a1077a99aa5fdca5e6ecff9d71fcfb1#autoware_maps/Town10HD_Opt"
)


class TestExpandSweep:
    """The constraint machinery is stood in for; what is under test is the wiring."""

    @pytest.fixture
    def _matching(self, monkeypatch: pytest.MonkeyPatch):
        def _set(ids, bindings=None):
            monkeypatch.setattr(expand_module, "parse_constraint", lambda c: c)
            monkeypatch.setattr(
                expand_module, "create_routing_graph", lambda m: "graph"
            )
            monkeypatch.setattr(
                expand_module, "find_matching_lanelets", lambda c, m, g: list(ids)
            )
            monkeypatch.setattr(
                expand_module,
                "parse_binding",
                lambda key, cfg: SimpleNamespace(target_key=key, resolve=bindings[key]),
            )

        return _set

    def test_one_case_per_matching_lanelet_with_its_bindings(self, _matching) -> None:
        _matching(
            [5, 6],
            {
                "ego.spawn_s": lambda lid, m, g: SimpleNamespace(
                    value=lid * 2.0, lanelet_id_override=None
                )
            },
        )

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{"type": "is_junction"}]},
                "bindings": {"ego.spawn_s": {"type": "stop_line_offset"}},
            },
            lanelet_map=object(),
            arguments=["map=town10hd_opt"],
        )

        assert cases == [
            ["ego.spawn_lanelet_id=5", "ego.spawn_s=10.0", "map=town10hd_opt"],
            ["ego.spawn_lanelet_id=6", "ego.spawn_s=12.0", "map=town10hd_opt"],
        ]

    def test_a_binding_can_move_the_pick_and_a_failing_one_drops_it(
        self, _matching
    ) -> None:
        def _resolve(lid, m, g):
            if lid == 6:
                raise ValueError("no stop line ahead")
            return SimpleNamespace(value=1.5, lanelet_id_override=50)

        _matching([5, 6], {"ego.spawn_s": _resolve})

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{}]},
                "bindings": {"ego.spawn_s": {}},
            },
            lanelet_map=object(),
        )

        assert cases == [["ego.spawn_lanelet_id=50", "ego.spawn_s=1.5"]]

    @staticmethod
    def _walk(value, override):
        return lambda lid, m, g: SimpleNamespace(
            value=value, lanelet_id_override=override
        )

    def test_another_slots_walk_back_moves_its_own_lanelet_not_the_ego(
        self, _matching
    ) -> None:
        # #1184: the ego fits 2 m before the stop line on the pick (29); the
        # NPC, 300 m before it, walks back to 54.  Only the NPC moves.
        _matching(
            [29],
            {
                "ego.spawn_s": self._walk(8.9, None),
                "scenario.spawn_overrides.npc1.s": self._walk(21.7, 54),
                "scenario.spawn_overrides.npc1.lanelet_id": self._walk(54, None),
            },
        )

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{}]},
                "bindings": {
                    "ego.spawn_s": {"type": "stop_line_offset", "offset": 2},
                    "scenario.spawn_overrides.npc1.s": {
                        "type": "stop_line_offset",
                        "offset": 300,
                    },
                    "scenario.spawn_overrides.npc1.lanelet_id": {
                        "type": "stop_line_approach",
                        "offset": 300,
                    },
                },
            },
            lanelet_map=object(),
        )

        assert cases == [
            [
                "ego.spawn_lanelet_id=29",
                "ego.spawn_s=8.9",
                "scenario.spawn_overrides.npc1.s=21.7",
                "scenario.spawn_overrides.npc1.lanelet_id=54",
            ]
        ]

    def test_a_walk_back_writes_its_spawns_lanelet_when_nothing_else_does(
        self, _matching
    ) -> None:
        _matching(
            [29],
            {
                "ego.spawn_s": self._walk(3.0, 40),
                "scenario.spawn_overrides.npc1.s": self._walk(21.7, 54),
            },
        )

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{}]},
                "bindings": {
                    "ego.spawn_s": {},
                    "scenario.spawn_overrides.npc1.s": {},
                },
            },
            lanelet_map=object(),
        )

        # The ego's own walk-back still moves the sweep target.
        assert cases == [
            [
                "ego.spawn_lanelet_id=40",
                "ego.spawn_s=3.0",
                "scenario.spawn_overrides.npc1.s=21.7",
                "scenario.spawn_overrides.npc1.lanelet_id=54",
            ]
        ]

    def test_exclude_drops_a_case_by_the_lanelet_its_walk_back_lands_on(
        self, _matching
    ) -> None:
        # The pick (29, 30) is fine; the spawn stop_line_offset walks 29's case
        # back onto, 40, is a lanelet CARLA does not draw.
        _matching(
            [29, 30],
            {
                "ego.spawn_s": lambda lid, m, g: SimpleNamespace(
                    value=3.0, lanelet_id_override=lid + 11
                )
            },
        )

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{}]},
                "bindings": {"ego.spawn_s": {}},
                "exclude": {"ego.spawn_lanelet_id": [40]},
            },
            lanelet_map=object(),
        )

        assert cases == [["ego.spawn_lanelet_id=41", "ego.spawn_s=3.0"]]

    def test_exclude_reads_a_list_the_way_the_case_writes_it(self, _matching) -> None:
        _matching(
            [5, 6],
            {
                "scenario.expected_route_lanelet_ids": lambda lid,
                m,
                g: SimpleNamespace(value=[lid, lid + 1], lanelet_id_override=None)
            },
        )

        cases = expand_sweep(
            {
                "constraints": {"ego.spawn_lanelet_id": [{}]},
                "bindings": {"scenario.expected_route_lanelet_ids": {}},
                "exclude": {"scenario.expected_route_lanelet_ids": [[6, 7]]},
            },
            lanelet_map=object(),
        )

        assert cases == [
            ["ego.spawn_lanelet_id=5", "scenario.expected_route_lanelet_ids=[5,6]"]
        ]

    def test_the_ego_walk_back_moves_its_lanelet_when_another_slot_is_swept(
        self, _matching
    ) -> None:
        _matching([7], {"ego.spawn_s": self._walk(3.0, 40)})

        cases = expand_sweep(
            {
                "constraints": {"ego.goal_lanelet_id": [{}]},
                "bindings": {"ego.spawn_s": {}},
            },
            lanelet_map=object(),
        )

        assert cases == [
            ["ego.goal_lanelet_id=7", "ego.spawn_s=3.0", "ego.spawn_lanelet_id=40"]
        ]

    def test_bindings_that_disagree_about_a_lanelet_are_refused(
        self, _matching
    ) -> None:
        # The approach is worked out for a shorter offset than the spawn's s:
        # it stays on the pick while the s walks back to 54.
        _matching(
            [29],
            {
                "scenario.spawn_overrides.npc1.s": self._walk(21.7, 54),
                "scenario.spawn_overrides.npc1.lanelet_id": self._walk(29, None),
            },
        )

        with pytest.raises(ValueError, match="disagree") as error:
            expand_sweep(
                {
                    "constraints": {"ego.spawn_lanelet_id": [{}]},
                    "bindings": {
                        "scenario.spawn_overrides.npc1.s": {},
                        "scenario.spawn_overrides.npc1.lanelet_id": {},
                    },
                },
                lanelet_map=object(),
            )
        assert "scenario.spawn_overrides.npc1.lanelet_id" in str(error.value)
        assert "54" in str(error.value) and "29" in str(error.value)

    def test_a_walk_back_of_a_key_that_is_no_spawns_s_is_refused(
        self, _matching
    ) -> None:
        _matching([29], {"scenario.param_overrides.p.gap": self._walk(1.0, 54)})

        with pytest.raises(ValueError, match="no lanelet key"):
            expand_sweep(
                {
                    "constraints": {"ego.spawn_lanelet_id": [{}]},
                    "bindings": {"scenario.param_overrides.p.gap": {}},
                },
                lanelet_map=object(),
            )

    def test_no_constraints_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            expand_sweep({"constraints": {}}, lanelet_map=object())

    def test_a_second_constraint_key_is_refused(self, _matching) -> None:
        # Two keys used to be ANDed into one search with only the last key
        # written, silently leaving the first at its default (#673).
        _matching([5], {})
        sweep = {
            "constraints": {
                "ego.spawn_lanelet_id": [{"type": "turn_direction", "value": "right"}],
                "scenario.npc_lanelet_id": [{"type": "has_adjacent", "value": "left"}],
            }
        }
        with pytest.raises(ValueError, match="one parameter") as error:
            expand_sweep(sweep, lanelet_map=object())
        assert "ego.spawn_lanelet_id" in str(error.value)
        assert "scenario.npc_lanelet_id" in str(error.value)


class TestCaseIndex:
    """``sweep.case_index`` keeps one case of the expansion."""

    @pytest.fixture
    def _three_lanelets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(expand_module, "parse_constraint", lambda c: c)
        monkeypatch.setattr(expand_module, "create_routing_graph", lambda m: "graph")
        monkeypatch.setattr(
            expand_module, "find_matching_lanelets", lambda c, m, g: [5, 6, 7]
        )

    def _sweep(self, case_index: object) -> dict:
        return {
            "constraints": {"ego.spawn_lanelet_id": [{"type": "is_junction"}]},
            "case_index": case_index,
        }

    @pytest.mark.usefixtures("_three_lanelets")
    def test_only_the_case_at_the_index_is_kept(self) -> None:
        cases = expand_sweep(self._sweep(1), object(), ["map=town10hd_opt"])
        assert cases == [["ego.spawn_lanelet_id=6", "map=town10hd_opt"]]

    @pytest.mark.usefixtures("_three_lanelets")
    def test_none_keeps_every_case(self) -> None:
        assert len(expand_sweep(self._sweep(None), object())) == 3

    @pytest.mark.usefixtures("_three_lanelets")
    def test_an_index_past_the_cases_is_refused(self) -> None:
        with pytest.raises(ValueError, match="case_index 3 is out of range"):
            expand_sweep(self._sweep(3), object())

    # A case index int() would misread (1.9, true) or not name (#1819).
    @pytest.mark.parametrize(
        ("value", "why"),
        [
            (True, "must be a whole number, got True"),
            (1.9, "must be a whole number, got 1.9"),
            ("abc", "must be a whole number, got 'abc'"),
            (-1, "must be >= 0, got -1"),
        ],
    )
    def test_an_index_that_is_no_whole_number_is_refused(
        self, value: object, why: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(*args: object) -> list[int]:
            raise AssertionError("the sweep was searched")

        monkeypatch.setattr(expand_module, "parse_constraint", lambda c: c)
        monkeypatch.setattr(expand_module, "create_routing_graph", lambda m: "graph")
        monkeypatch.setattr(expand_module, "find_matching_lanelets", never)
        with pytest.raises(ValueError) as error:
            expand_sweep(self._sweep(value), object())
        message = str(error.value)
        assert message == f"sweep.case_index {why}"
        assert "\n" not in message

    @pytest.mark.parametrize("value", [2, "2", 2.0])
    @pytest.mark.usefixtures("_three_lanelets")
    def test_a_whole_number_or_its_string_is_accepted(self, value: object) -> None:
        assert expand_sweep(self._sweep(value), object()) == [
            ["ego.spawn_lanelet_id=7"]
        ]


class TestExpandConfig:
    def test_a_concrete_scenario_expands_to_itself_without_loading_a_map(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_map(*_args, **_kwargs):
            raise AssertionError("a concrete scenario needs no map")

        monkeypatch.setattr("autoware_carla_scenario.maps.resolve_map_paths", _no_map)

        cfg = OmegaConf.create(
            {"scenario": {"name": "x"}, "map": {"name": "Town10HD_Opt"}}
        )

        assert expand_config(cfg) == [[]]
        assert expand_config(cfg, ["ego.spawn_s=3.0"]) == [["ego.spawn_s=3.0"]]

    # A concrete config's `controls:` are checked as a sweep's are, and as
    # `scenario-check` checks them (#1191): they used to be read only when
    # the config had a sweep.
    def test_a_concrete_scenario_with_invalid_controls_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.odd.controls import ControlsError

        def _no_map(*_args, **_kwargs):
            raise AssertionError("checking controls needs no map")

        monkeypatch.setattr("autoware_carla_scenario.maps.resolve_map_paths", _no_map)
        cfg = OmegaConf.create(
            {
                "scenario": {"name": "x", "npc_gap_m": 10.0},
                "map": {"name": "Town10HD_Opt"},
                "controls": {"vehicle_ahead_gap_m": {"key": "scenario.npc_gapm"}},
            }
        )
        with pytest.raises(ControlsError) as caught:
            expand_config(cfg)
        assert [d.path for d in caught.value.diagnostics] == [
            "controls.vehicle_ahead_gap_m.key"
        ]

    def test_a_concrete_scenario_with_valid_controls_expands_to_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_map(*_args, **_kwargs):
            raise AssertionError("a concrete scenario needs no map")

        monkeypatch.setattr("autoware_carla_scenario.maps.resolve_map_paths", _no_map)
        cfg = OmegaConf.create(
            {
                "scenario": {"name": "x", "npc_gap_m": 10.0},
                "map": {"name": "Town10HD_Opt"},
                "controls": {
                    "vehicle_ahead_gap_m": {
                        "key": "scenario.npc_gap_m",
                        "range": [2, 30],
                    }
                },
            }
        )
        assert expand_config(cfg) == [[]]
        assert expand_config(cfg, ["ego.spawn_s=3.0"]) == [["ego.spawn_s=3.0"]]

    def test_a_second_constraint_key_is_refused_before_the_map_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_map(*_args, **_kwargs):
            raise AssertionError("a refused sweep must not fetch its map")

        monkeypatch.setattr("autoware_carla_scenario.maps.resolve_map_paths", _no_map)
        cfg = OmegaConf.create(
            {
                "map": {"name": "Town10HD_Opt"},
                "sweep": {
                    "constraints": {
                        "ego.spawn_lanelet_id": [{"type": "is_junction"}],
                        "scenario.npc_lanelet_id": [{"type": "is_junction"}],
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="one parameter"):
            expand_config(cfg)

    def test_an_unreadable_map_is_refused_as_an_invalid_sweep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # A directory: Lanelet2's parser refuses it with a bare RuntimeError.
        monkeypatch.setattr(
            "autoware_carla_scenario.maps.resolve_map_paths",
            lambda _m: SimpleNamespace(
                lanelet2_path=tmp_path, xodr_path=None, projector_type=None
            ),
        )
        cfg = OmegaConf.create(
            {"sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}}}
        )
        with pytest.raises(ValueError, match="Could not read the Lanelet2 map"):
            expand_config(cfg)

    def test_the_sweeper_refuses_a_second_key_before_the_map_is_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock, patch

        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        # The ValueError itself, not the one line it is reported as (#1705).
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(
            {"sweep": {"constraints": {"a.id": [{}], "b.id": [{}]}}}
        )
        sweeper.launcher = MagicMock()
        with (
            patch.object(mod, "load_sweep_map") as resolve,
            pytest.raises(ValueError, match="one parameter"),
        ):
            sweeper.sweep([])
        resolve.assert_not_called()

    # A map that cannot be resolved, fetched or read is refused by the Hydra
    # sweeper (a single scenario's --multirun) as scenario-expand refuses it:
    # a ValueError naming the map, its cause kept, before any case runs
    # (#1447).  It used to escape as a MapResolutionError, a MapCacheError or
    # Lanelet2's RuntimeError.  HYDRA_FULL_ERROR=1 raises that ValueError
    # rather than print it as one line and exit 1 (#1705).
    @staticmethod
    def _sweep_on(
        map_cfg: dict, monkeypatch: pytest.MonkeyPatch
    ) -> pytest.ExceptionInfo:
        from unittest.mock import MagicMock

        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        def _no_expansion(*_args, **_kwargs):
            raise AssertionError("a refused map must not be expanded")

        monkeypatch.setattr(mod, "expand_sweep", _no_expansion)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(
            {"map": map_cfg, "sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}}}
        )
        sweeper.launcher = MagicMock()
        with pytest.raises(ValueError, match="^LaneletConstraintSweeper: ") as caught:
            sweeper.sweep([])
        sweeper.launcher.launch.assert_not_called()
        return caught

    def test_the_sweeper_refuses_a_map_directory_that_does_not_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from autoware_carla_scenario.maps import MapResolutionError

        missing = tmp_path / "no_map"
        caught = self._sweep_on({"directory": str(missing)}, monkeypatch)
        assert f"Map directory {missing} does not exist" in str(caught.value)
        assert isinstance(caught.value.__cause__, MapResolutionError)

    def test_the_sweeper_refuses_a_map_that_cannot_be_fetched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.maps import MapCacheError

        def _unreachable(_map_cfg):
            raise MapCacheError("could not clone the map repository")

        monkeypatch.setattr(
            "autoware_carla_scenario.maps.resolve_map_paths", _unreachable
        )
        caught = self._sweep_on({"name": "Town10HD_Opt"}, monkeypatch)
        assert str(caught.value) == (
            "LaneletConstraintSweeper: could not clone the map repository"
        )
        assert isinstance(caught.value.__cause__, MapCacheError)

    def test_the_sweeper_refuses_a_directory_as_the_lanelet2_map(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        caught = self._sweep_on({"lanelet2_path": str(tmp_path)}, monkeypatch)
        assert f"Could not read the Lanelet2 map {tmp_path}" in str(caught.value)
        assert isinstance(caught.value.__cause__, (OSError, RuntimeError))

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({"map": {}}, "No 'sweep' section found in config."),
            (
                {"sweep": {"constraints": {"a.id": [{}], "b.id": [{}]}}},
                "one parameter",
            ),
            (
                {
                    "map": {"directory": "/nonexistent/no_map"},
                    "sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}},
                },
                "LaneletConstraintSweeper: Map directory /nonexistent/no_map "
                "does not exist",
            ),
        ],
    )
    def test_the_sweeper_reports_an_invalid_sweep_in_one_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        config: dict,
        message: str,
    ) -> None:
        """#1705: without HYDRA_FULL_ERROR, `Error: <scenario>: <why>` on
        stderr and exit 1, as scenario-expand prints it."""
        from unittest.mock import MagicMock

        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)
        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(
            {
                **config,
                "hydra": {"runtime": {"choices": {"scenario": "cut_in/left"}}},
            }
        )
        sweeper.launcher = MagicMock()
        with pytest.raises(SystemExit) as caught:
            sweeper.sweep([])
        assert caught.value.code == 1
        lines = capsys.readouterr().err.splitlines()
        assert len(lines) == 1, lines
        assert lines[0].startswith("Error: cut_in/left: "), lines
        assert message in lines[0], lines
        sweeper.launcher.launch.assert_not_called()

    #: A sweep whose `controls` have two errors, on a map that must not be
    #: fetched: they are refused before it is (#1710).
    _BAD_CONTROLS: dict = {
        "map": {"source": "https://example.com/never-cloned.git"},
        "sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}},
        "controls": {
            "vehicle_ahead_gap_m": 5,
            "ego_speed_mps": {"key": "scenario.npc_gapm"},
        },
    }

    def test_the_sweeper_reports_invalid_controls_one_line_per_problem(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """#1710: as scenario-expand prints them, before the map is loaded."""
        from unittest.mock import MagicMock, patch

        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)
        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(
            {
                **self._BAD_CONTROLS,
                "scenario": {"npc_gap_m": 10.0},
                "hydra": {"runtime": {"choices": {"scenario": "cut_in/left"}}},
            }
        )
        sweeper.launcher = MagicMock()
        with (
            patch.object(mod, "load_sweep_map") as load,
            pytest.raises(SystemExit) as caught,
        ):
            sweeper.sweep([])
        assert caught.value.code == 1
        load.assert_not_called()
        sweeper.launcher.launch.assert_not_called()
        lines = capsys.readouterr().err.splitlines()
        assert lines == [
            "cut_in/left: controls.vehicle_ahead_gap_m: "
            "expected a key or a mapping, got 5",
            "cut_in/left: controls.ego_speed_mps.key: 'scenario.npc_gapm' is "
            "not a key of the config (did you mean 'scenario.npc_gap_m'?)",
        ], lines

    def test_the_sweeper_raises_invalid_controls_with_hydra_full_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import MagicMock, patch

        from autoware_carla_scenario.odd.controls import ControlsError
        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(self._BAD_CONTROLS)
        sweeper.launcher = MagicMock()
        with (
            patch.object(mod, "load_sweep_map") as load,
            pytest.raises(ControlsError) as caught,
        ):
            sweeper.sweep([])
        load.assert_not_called()
        assert [d.path for d in caught.value.diagnostics] == [
            "controls.vehicle_ahead_gap_m",
            "controls.ego_speed_mps.key",
        ]

    def test_a_map_cache_root_that_is_no_directory_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # The cache root's own OSError, from resolving a `map.source` (#1447).
        from autoware_carla_scenario.maps.cache import (
            AUTOWARE_DATA_DIR_ENV,
            CACHE_ROOT_ENV,
        )

        data_dir = tmp_path / "autoware_data"
        data_dir.write_text("")
        monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
        monkeypatch.setenv(AUTOWARE_DATA_DIR_ENV, str(data_dir))
        source = _PINNED_SOURCE
        cfg = OmegaConf.create(
            {
                "map": {"source": source},
                "sweep": {"constraints": {"ego.spawn_lanelet_id": [{}]}},
            }
        )
        expected = f"Could not resolve the map map.source={source}: "
        with pytest.raises(ValueError, match="^" + re.escape(expected)) as caught:
            expand_config(cfg)
        assert isinstance(caught.value.__cause__, NotADirectoryError)


@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_expands_a_packaged_logical_scenario(
    capsys: pytest.CaptureFixture,
) -> None:
    """End to end on the built-in traffic-light scenario and the Nishishinjuku map:
    the same cases the Hydra sweeper would run, as JSON on stdout."""
    from autoware_carla_scenario.examples.expand import main

    assert main(["scenario=traffic_light_compliance/traffic_light_compliance"]) == 0

    raw = capsys.readouterr().out
    # One scenario, one line: the JSON object and its newline.
    assert raw.endswith("}\n")
    assert raw.count("\n") == 1
    out = json.loads(raw)
    cases = out["cases"]
    assert out["scenario"] == "traffic_light_compliance/traffic_light_compliance"
    assert cases, "the constraints match lanelets on this map"
    for case in cases:
        lanelet, spawn_s = case
        assert lanelet.startswith("ego.spawn_lanelet_id=")
        assert spawn_s.startswith("ego.spawn_s=")
    # The constraints exclude lanelet 222 explicitly.
    assert "ego.spawn_lanelet_id=222" not in {c[0] for c in cases}


# `scenario-expand` takes the same arguments as `scenario`, glob included
# (#1160): it used to compose the glob as a config name and fail.
@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_expands_every_scenario_a_glob_matches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    assert main(["scenario=intersection_passing/*"]) == 0
    captured = capsys.readouterr()
    assert "Could not find" not in captured.err
    listings = [json.loads(line) for line in captured.out.splitlines()]
    # One listing per scenario, headed by its name, in the batch's order.
    assert [listing["scenario"] for listing in listings] == [
        "intersection_passing/left_turn",
        "intersection_passing/right_turn",
        "intersection_passing/right_turn_sweep",
        "intersection_passing/straight",
        "intersection_passing/straight_3npc",
    ]
    for listing in listings:
        assert listing["cases"], listing["scenario"]
    assert {"scenario": "intersection_passing/straight_3npc", "cases": [[]]} in (
        listings
    )


@pytest.mark.usefixtures("at_repo_root")
def test_each_glob_listing_is_what_its_scenario_expands_to_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    overrides = ["scenario.timeout_seconds=7"]
    assert main(["scenario=intersection_passing/straight*", *overrides]) == 0
    globbed = capsys.readouterr().out

    alone = ""
    for name in ("intersection_passing/straight", "intersection_passing/straight_3npc"):
        assert main([f"scenario={name}", *overrides]) == 0
        alone += capsys.readouterr().out
    assert globbed == alone


# Scenarios on one ODD go on from each other's draws in a batch, and so in
# `scenario-expand` of the same glob (#1405): it used to draw each afresh.
@pytest.mark.usefixtures("at_repo_root")
def test_a_glob_listing_is_the_cases_the_batch_runs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.examples.expand import main

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    overrides = ["+sweep.odd_sample.count=3"]
    assert main(["scenario=cut_in/*", *overrides]) == 0
    listings = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    expanded = [(listing["scenario"], listing["cases"]) for listing in listings]

    names = ["cut_in/left", "cut_in/right"]
    batch = run._expand_batch(names, overrides)
    # The batch's cases carry the command line's overrides; the listing's
    # cases are what is added to them.
    assert expanded == [
        (
            name,
            [[o for o in case if o not in overrides] for n, case in batch if n == name],
        )
        for name in names
    ]

    def draws(cases: list[list[str]]) -> set[tuple[str, ...]]:
        return {
            tuple(o for o in case if o.startswith("environment.")) for case in cases
        }

    # The second scenario does not draw the first one's settings again.
    assert not draws(expanded[0][1]) & draws(expanded[1][1])


# A scenario package's conf dir reached Hydra's search path as a
# percent-encoded ``file:`` URI, which Hydra never unquotes, so a package under a
# path with a space or a non-ASCII character could not be composed although the
# glob resolver listed its names (#1543).
@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize("directory", ["my pkgs é", "100% #1 ?x"])
def test_the_cli_composes_a_package_under_a_path_a_uri_would_escape(
    directory: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario import registry
    from autoware_carla_scenario.examples.expand import main

    template = Path("examples/scenario_package_template/src/my_scenario_package")
    conf_dir = tmp_path / directory / "my_scenario_package" / "conf"
    shutil.copytree(template / "conf", conf_dir)
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry.get_conf_dirs()))
    registry.register_conf_dir(conf_dir)

    assert main(["scenario=reach_goal/default", "map=nishishinjuku"]) == 0
    captured = capsys.readouterr()
    assert "Could not find" not in captured.err
    assert json.loads(captured.out)["scenario"] == "reach_goal/default"


def test_a_bad_scenario_in_a_glob_prints_no_listing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every matched scenario's problem is reported, and stdout is all the
    listings or none: never a fan-out that silently lacks a scenario."""
    from autoware_carla_scenario.examples.expand import main

    code = main(
        [
            "scenario=cut_in/*",
            "+sweep.odd_sample.count=2",
            "controls.vehicle_ahead_gap_m.range=[30,2]",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "cut_in/left: controls.vehicle_ahead_gap_m.range" in captured.err
    assert "cut_in/right: controls.vehicle_ahead_gap_m.range" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# A sweep that expands to no case is refused, not listed as `"cases": []`
# with exit 0 (#1407), as the lanelet sweeper and a batch refuse it.
@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize(
    "scenario", ["intersection_passing/straight_3npc", "intersection_passing/str*"]
)
def test_the_cli_refuses_a_sweep_that_expands_to_no_case(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    from autoware_carla_scenario.examples import expand as expand_cli

    # A sweep matching nothing on the map: odd_sample.count=0 is refused for
    # its count now (#1437).
    monkeypatch.setattr(expand_cli, "expand_config", lambda *_, **__: [])
    code = expand_cli.main([f"scenario={scenario}", "+sweep.odd_sample.count=1"])
    captured = capsys.readouterr()
    assert code == 1
    assert (
        "Error: intersection_passing/straight_3npc: the sweep expands to no case "
        "(its constraints, route or ODD sample match nothing on this map)"
    ) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# A seed or an ODD sample count that is no whole number is refused, named,
# as an invalid sweep is, rather than truncated (1.9 as seed 1, 2.7 as 2
# cases) or raised as a traceback (seed=abc) (#1437).
@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["seed=abc"], "seed must be a whole number, got 'abc'"),
        (["seed=1.9"], "seed must be a whole number, got 1.9"),
        (["seed=true"], "seed must be a whole number, got True"),
        (["seed=-1"], "seed must not be negative, got -1"),
        (
            ["+sweep.odd_sample.count=2.7"],
            "sweep.odd_sample.count must be a whole number of at least 1, got 2.7",
        ),
        (
            ["+sweep.odd_sample.count=0"],
            "sweep.odd_sample.count must be a whole number of at least 1, got 0",
        ),
        (
            ["+sweep.odd_sample.count=2", "+sweep.odd_sample.seed=-2"],
            "sweep.odd_sample.seed must not be negative, got -2",
        ),
        (
            ["+sweep.odd_sample.count=2", "+sweep.odd_sample.tire_friction=ture"],
            "sweep.odd_sample.tire_friction must be true or false, got 'ture'",
        ),
        (
            ["+sweep.odd_sample.count=2", "+sweep.odd_sample.visibility=2"],
            "sweep.odd_sample.visibility must be true or false, got 2",
        ),
    ],
)
def test_the_cli_refuses_a_seed_or_count_that_is_no_whole_number(
    capsys: pytest.CaptureFixture[str], overrides: list[str], message: str
) -> None:
    from autoware_carla_scenario.examples.expand import main

    name = "intersection_passing/straight_3npc"
    code = main([f"scenario={name}", *overrides])
    captured = capsys.readouterr()
    assert code == 1
    assert f"Error: {name}: {message}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# An invalid sweep is reported, named, for every scenario of a glob (#1412):
# its ValueError used to escape as a traceback, hiding the other scenarios.
_OUT_OF_RANGE = ["+sweep.odd_sample.count=2", "++sweep.case_index=99"]


@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_reports_an_invalid_sweep_of_each_scenario(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(["scenario=cut_in/*", *_OUT_OF_RANGE])
    captured = capsys.readouterr()
    assert code == 1
    for name in ("cut_in/left", "cut_in/right"):
        assert (
            f"Error: {name}: sweep.case_index 99 is out of range: the sweep "
            "expands to 2 case(s)"
        ) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# A case index that is no whole number is refused, naming the key, for every
# scenario of a glob (#1819): int() read 1.9 as 1 and failed on "abc" unnamed.
@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_reports_a_case_index_that_is_no_whole_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(
        ["scenario=cut_in/*", "+sweep.odd_sample.count=2", "++sweep.case_index=1.9"]
    )
    captured = capsys.readouterr()
    assert code == 1
    for name in ("cut_in/left", "cut_in/right"):
        assert (
            f"Error: {name}: sweep.case_index must be a whole number, got 1.9\n"
        ) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("at_repo_root")
def test_hydra_full_error_raises_an_invalid_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples.expand import main

    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(ValueError, match="case_index 99 is out of range"):
        main(["scenario=cut_in/left", *_OUT_OF_RANGE])


# A missing Lanelet2 map is refused, named, for every scenario of a glob
# (#1444): load_map's FileNotFoundError used to escape as a traceback.
@pytest.mark.usefixtures("at_repo_root")
@pytest.mark.parametrize("glob", [False, True])
def test_the_cli_reports_a_missing_map_of_each_scenario(
    capsys: pytest.CaptureFixture[str], tmp_path, glob: bool
) -> None:
    from autoware_carla_scenario.examples.expand import main

    missing = tmp_path / "missing.osm"
    names = ["lane_change/left", "lane_change/right"] if glob else ["lane_change/left"]
    scenario = "lane_change/*" if glob else "lane_change/left"
    code = main([f"scenario={scenario}", f"map.lanelet2_path={missing}"])
    captured = capsys.readouterr()
    assert code == 1
    for name in names:
        assert f"Error: {name}: Lanelet2 file not found: {missing}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# A map that cannot be resolved -- a `map.directory` that does not exist, a
# `map.source` that cannot be fetched -- is refused the same way (#1444).
@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_reports_a_map_directory_that_does_not_exist(
    capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    from autoware_carla_scenario.examples.expand import main

    missing = tmp_path / "no_map"
    code = main(["scenario=lane_change/*", f"+map.directory={missing}"])
    captured = capsys.readouterr()
    assert code == 1
    for name in ("lane_change/left", "lane_change/right"):
        assert f"Error: {name}: Map directory {missing} does not exist" in (
            captured.err
        )
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_reports_a_map_that_cannot_be_fetched(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.examples.expand import main
    from autoware_carla_scenario.maps import MapCacheError

    def _unreachable(_map_cfg):
        raise MapCacheError("could not clone the map repository")

    monkeypatch.setattr("autoware_carla_scenario.maps.resolve_map_paths", _unreachable)
    code = main(["scenario=lane_change/*"])
    captured = capsys.readouterr()
    assert code == 1
    for name in ("lane_change/left", "lane_change/right"):
        assert f"Error: {name}: could not clone the map repository" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


# A map cache root that is no directory -- AUTOWARE_DATA_DIR naming a file --
# is refused the same way, naming the map (#1447): its NotADirectoryError
# used to escape as a traceback.
@pytest.mark.usefixtures("at_repo_root")
def test_the_cli_reports_a_map_cache_root_that_is_no_directory(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from autoware_carla_scenario.examples.expand import main
    from autoware_carla_scenario.maps.cache import (
        AUTOWARE_DATA_DIR_ENV,
        CACHE_ROOT_ENV,
    )

    data_dir = tmp_path / "autoware_data"
    data_dir.write_text("")
    monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
    monkeypatch.setenv(AUTOWARE_DATA_DIR_ENV, str(data_dir))
    # Town10HD_Opt's source is pinned to a commit: nothing is fetched.
    code = main(["scenario=lane_change/*", "map=town10hd_opt"])
    captured = capsys.readouterr()
    assert code == 1
    for name in ("lane_change/left", "lane_change/right"):
        assert f"Error: {name}: Could not resolve the map map.source=" in captured.err
    assert "Not a directory" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("at_repo_root")
def test_hydra_full_error_raises_a_missing_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from autoware_carla_scenario.examples.expand import main

    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(ValueError, match="Lanelet2 file not found") as caught:
        main(["scenario=lane_change/left", f"map.lanelet2_path={tmp_path}/no.osm"])
    assert isinstance(caught.value.__cause__, FileNotFoundError)


# `scenario=` given twice: the last wins, as for `scenario` and Hydra (#1412);
# it used to take the first.
def test_the_cli_takes_the_last_scenario_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(
        [
            "scenario=intersection_passing/straight",
            "scenario=intersection_passing/straight_3npc",
        ]
    )
    assert code == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing == {"scenario": "intersection_passing/straight_3npc", "cases": [[]]}


# A scenario without a sweep has its `controls:` checked too (#1191).
@pytest.mark.parametrize("override", ["+controls.x=5", "+controls.x.rnge=[1,2]"])
def test_the_cli_refuses_a_concrete_scenario_with_invalid_controls(
    capsys: pytest.CaptureFixture[str], override: str
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(["scenario=intersection_passing/straight_3npc", override])
    captured = capsys.readouterr()
    assert code == 1
    assert "intersection_passing/straight_3npc: controls.x" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_a_glob_that_does_not_compose_prints_no_listing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    assert main(["scenario=intersection_passing/*", "foo.bar=1"]) == 2
    captured = capsys.readouterr()
    # Each scenario's error names it, so the reader can tell them apart.
    for name in ("left_turn", "straight_3npc"):
        assert (
            f"Error: intersection_passing/{name}: could not override 'foo.bar'"
            in captured.err
        )
    assert captured.out == ""


# A command line that does not compose (#993): a message and 2, as
# `scenario-check` prints, never a traceback; stdout stays the JSON's.
def test_the_cli_names_the_closest_scenario_configs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    assert main(["scenario=nonexistent/x"]) == 2
    captured = capsys.readouterr()
    assert "no scenario configs match pattern 'nonexistent/x'" in captured.err
    assert "scenario configs: " in captured.err
    assert "intersection_passing/straight" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""

    assert main(["scenario=intersection_passing/straigth"]) == 2
    assert "did you mean: intersection_passing/straight" in capsys.readouterr().err


def test_the_cli_names_an_unknown_override_key_with_a_suggestion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples.expand import main

    code = main(["scenario=intersection_passing/straight", "foo.bar=1"])
    captured = capsys.readouterr()
    assert code == 2
    assert "could not override 'foo.bar'" in captured.err
    assert "+foo.bar=1" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""

    code = main(["scenario=intersection_passing/straight", "scenario.timout_seconds=3"])
    assert code == 2
    assert "did you mean 'scenario.timeout_seconds'?" in capsys.readouterr().err


def test_hydra_full_error_shows_the_cli_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra.errors import HydraException

    from autoware_carla_scenario.examples.expand import main

    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(HydraException):
        main(["scenario=intersection_passing/straight", "foo.bar=1"])


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_the_cli_help_is_the_usage_on_stdout(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.examples import expand

    assert expand.main([flag]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == (expand.__doc__ or "").strip()
    assert "scenario-expand scenario=" in captured.out
    assert captured.err == ""


def test_expand_re_exports_the_checked_walk_back_lanelet_key() -> None:
    """#1697: the function moved to the checked sweeper.bindings."""
    assert (
        expand_module._walk_back_lanelet_key is bindings_module._walk_back_lanelet_key
    )
