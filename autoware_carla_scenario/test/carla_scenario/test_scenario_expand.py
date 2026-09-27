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

    def test_no_constraints_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            expand_sweep({"constraints": {}}, lanelet_map=object())


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


def test_the_cli_expands_a_packaged_logical_scenario(
    capsys: pytest.CaptureFixture,
) -> None:
    """End to end on the built-in traffic-light scenario and the Nishishinjuku map:
    the same cases the Hydra sweeper would run, as JSON on stdout."""
    from autoware_carla_scenario.examples.expand import main

    assert main(["scenario=traffic_light_compliance/traffic_light_compliance"]) == 0

    out = json.loads(capsys.readouterr().out)
    cases = out["cases"]
    assert out["scenario"] == "traffic_light_compliance/traffic_light_compliance"
    assert cases, "the constraints match lanelets on this map"
    for case in cases:
        lanelet, spawn_s = case
        assert lanelet.startswith("ego.spawn_lanelet_id=")
        assert spawn_s.startswith("ego.spawn_s=")
    # The constraints exclude lanelet 222 explicitly.
    assert "ego.spawn_lanelet_id=222" not in {c[0] for c in cases}
