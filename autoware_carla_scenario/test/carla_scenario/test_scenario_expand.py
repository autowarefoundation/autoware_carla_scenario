"""Expanding a logical scenario into its concrete scenarios (``scenario-expand``)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.sweeper import expand as expand_module
from autoware_carla_scenario.sweeper.expand import expand_config, expand_sweep


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

    def test_the_sweeper_refuses_a_second_key_before_the_map_is_fetched(
        self,
    ) -> None:
        from unittest.mock import MagicMock, patch

        from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

        sweeper = mod.LaneletConstraintSweeper()
        sweeper.config = OmegaConf.create(
            {"sweep": {"constraints": {"a.id": [{}], "b.id": [{}]}}}
        )
        sweeper.launcher = MagicMock()
        with (
            patch.object(mod, "resolve_map_paths") as resolve,
            pytest.raises(ValueError, match="one parameter"),
        ):
            sweeper.sweep([])
        resolve.assert_not_called()


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
