"""The run settings of #2188: a null setting is its default, and a value
that is not one is refused before the run, naming the key (#2143)."""

from __future__ import annotations

import re
from typing import Any

import pytest
from omegaconf import OmegaConf

from .test_video_removed import _shipped


def _traffic(overrides: list[str]) -> Any:
    return _shipped(["background_traffic.enabled=true", *overrides])


class TestBackgroundTrafficStopAfterSeconds:
    def test_null_is_no_stop(self) -> None:
        from autoware_carla_scenario.examples import run

        settings = run.background_traffic_settings(_traffic([]))
        assert settings is not None
        assert settings.source_stop_after_seconds is None
        assert settings.sink_stop_after_seconds is None

    def test_a_duration_is_carried_on_the_settings(self) -> None:
        from autoware_carla_scenario.examples import run

        settings = run.background_traffic_settings(
            _traffic(
                [
                    "background_traffic.source.stop_after_seconds=12.5",
                    "background_traffic.sink.stop_after_seconds='7'",
                ]
            )
        )
        assert settings is not None
        assert settings.source_stop_after_seconds == 12.5
        assert settings.sink_stop_after_seconds == 7.0

    @pytest.mark.parametrize("block", ["source", "sink"])
    @pytest.mark.parametrize(
        ("value", "shown"),
        [("abc", "'abc'"), ("false", "False"), ("0", "0"), ("''", "''")],
    )
    def test_a_value_that_is_no_duration_is_a_run_config_error(
        self, block: str, value: str, shown: str
    ) -> None:
        """Refused before the run (an ``Error:`` line, exit 2 for a case the
        lanelet sweeper forked), not by add_background_traffic once the
        scenario is built."""
        from autoware_carla_scenario.examples import run

        cfg = _traffic([f"background_traffic.{block}.stop_after_seconds={value}"])
        match = (
            rf"^background_traffic\.{block}\.stop_after_seconds: "
            rf"{re.escape(shown)} is not a duration"
        )
        with pytest.raises(ValueError, match=match):
            run.background_traffic_settings(cfg)
        with pytest.raises(run.RunConfigError, match=match):
            run.check_run_config(cfg)
        with pytest.raises(run.RunConfigError, match=rf"^case: {match[1:]}"):
            run.check_batch_configs([cfg], ["case"])

    @pytest.mark.parametrize("block", ["source", "sink"])
    def test_it_is_refused_when_nothing_it_ends_is_built(self, block: str) -> None:
        """With no rate and no sink, the value was never read at all."""
        from autoware_carla_scenario.examples import run

        cfg = _traffic(
            [
                "background_traffic.source.vehicles_per_minute=0",
                "background_traffic.sink.constraints=null",
                f"background_traffic.{block}.stop_after_seconds=abc",
            ]
        )
        with pytest.raises(
            run.RunConfigError,
            match=rf"^background_traffic\.{block}\.stop_after_seconds: 'abc'",
        ):
            run.check_run_config(cfg)


class TestGroundProjectionNull:
    @pytest.mark.parametrize(
        ("key", "field"),
        [
            ("ground_projection_ray_distance_upper", "ray_distance_upper"),
            ("ground_projection_ray_distance_lower", "ray_distance_lower"),
        ],
    )
    def test_null_is_the_default(self, key: str, field: str) -> None:
        from autoware_carla_scenario.examples import run
        from autoware_carla_scenario.scenario_config import EntityConfig

        cfg = _shipped([f"entity.{key}=null"])
        ray = run.build_ground_projection(cfg)
        assert getattr(ray, field) == getattr(EntityConfig, key)
        run.check_run_config(cfg)

    def test_empty_text_is_still_refused(self) -> None:
        from autoware_carla_scenario.examples import run

        with pytest.raises(
            run.RunConfigError,
            match=r"^entity\.ground_projection_ray_distance_upper must be a number",
        ):
            run.check_run_config(
                _shipped(["entity.ground_projection_ray_distance_upper=''"])
            )


class TestNativeAutowareInterfaceSwitch:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("null", False), ("'false'", False), ("0", False), ("yes", True)],
    )
    def test_it_reads_as_every_other_switch(self, value: str, expected: bool) -> None:
        from autoware_carla_scenario.examples import run

        cfg = _shipped(
            [
                "ego.entity=autoware",
                f"ego.native_autoware_interface={value}",
                "autoware.launcher.type=none",
            ]
        )
        assert run._native_autoware_interface(cfg) is expected
        run.check_run_config(cfg)

    def test_a_typo_is_refused(self) -> None:
        from autoware_carla_scenario.examples import run

        with pytest.raises(
            run.RunConfigError,
            match=r"^ego\.native_autoware_interface must be true or false, got 'ture'",
        ):
            run.check_run_config(_shipped(["ego.native_autoware_interface=ture"]))


class TestSwitchNull:
    @pytest.mark.parametrize(
        ("block", "key"),
        [
            ("rerun", "enabled"),
            ("rerun", "strict"),
            ("rerun", "lidar"),
            ("rerun", "cameras"),
            ("t4", "enabled"),
            ("t4scene", "enabled"),
            ("background_traffic", "enabled"),
        ],
    )
    def test_null_is_off(self, block: str, key: str) -> None:
        from autoware_carla_scenario.examples import run

        assert run._switch(OmegaConf.create({block: {key: None}}), block, key) is False
        run.check_run_config(_shipped([f"{block}.{key}=null"]))

    def test_empty_text_is_still_refused(self) -> None:
        from autoware_carla_scenario.examples import run

        with pytest.raises(ValueError, match=r"^rerun\.enabled must be true or false"):
            run._switch(
                OmegaConf.create({"rerun": {"enabled": ""}}), "rerun", "enabled"
            )


class TestPlannedRouteDerivedGoal:
    @staticmethod
    def _cfg(expected: list[Any]) -> Any:
        return OmegaConf.create(
            {
                "ego": {"spawn_lanelet_id": 1},
                "scenario": {"expected_route_lanelet_ids": expected},
            }
        )

    @pytest.mark.parametrize("value", [242.7, "abc", True])
    def test_a_goal_that_is_no_lanelet_id_is_refused_by_its_key(
        self, value: object
    ) -> None:
        from autoware_carla_scenario.examples import run

        with pytest.raises(
            ValueError, match=r"^scenario\.expected_route_lanelet_ids\[1\]"
        ):
            run.build_planned_route(self._cfg([1, value]))

    @pytest.mark.parametrize("value", [242, 242.0, "242"])
    def test_a_whole_number_is_the_goal(self, value: object) -> None:
        from autoware_carla_scenario.examples import run

        route = run.build_planned_route(self._cfg([1, value]))
        assert route.goal is not None and route.goal.lanelet_id == 242


class TestCanvasConfig:
    @staticmethod
    def _cfg(value: object) -> Any:
        return OmegaConf.create({"canvas": {"config": value}})

    def test_null_names_none(self) -> None:
        from autoware_carla_scenario.examples import run

        assert run.canvas_config(self._cfg(None)) is None

    @pytest.mark.parametrize(("value", "shown"), [("", "''"), (False, "False")])
    def test_a_value_that_is_no_path_is_refused(
        self, value: object, shown: str
    ) -> None:
        from autoware_carla_scenario.examples import run

        message = rf"^canvas\.config must be a file path, got {re.escape(shown)}$"
        with pytest.raises(ValueError, match=message):
            run.canvas_config(self._cfg(value))
        cfg = _shipped([f"canvas.config={shown}"])
        with pytest.raises(run.RunConfigError, match=message):
            run.check_run_config(cfg)
