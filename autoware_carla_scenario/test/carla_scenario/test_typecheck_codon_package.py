"""The static check of this package's own scenarios, and the runner's step.

Every built-in scenario config, the scenario package template and a scaffolded
package compile with Codon, and the runner and the batch refuse a wrong
scenario before building it. These are the compile checks of the package's own
code: unlike the checker's own tests (``test_typecheck_codon.py`` and the
others marked ``codon_selftest``), every pull request runs them. Skipped where
no Codon compiler is installed.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from autoware_carla_scenario.typecheck import (
    ScenarioTypeError,
    available_toolchain,
    typecheck_scenario,
)

from .test_typecheck_codon import _write_case

pytestmark = [
    pytest.mark.codon,
    pytest.mark.skipif(available_toolchain() is None, reason="no Codon compiler"),
]

_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE_SRC = _REPO / "examples" / "scenario_package_template" / "src"


# ---------------------------------------------------------------------------
# What passes
# ---------------------------------------------------------------------------


def _scenario_configs() -> list[str]:
    from autoware_carla_scenario.examples import run

    return run._resolve_scenario_glob("**/*")


@pytest.mark.parametrize("config_name", _scenario_configs())
def test_every_built_in_scenario_config_compiles(config_name: str) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.registry import get_scenario_classes

    cfg = run._compose_config(config_name, [])
    classes = get_scenario_classes(str(cfg.scenario.name))
    assert classes is not None
    result = typecheck_scenario(*classes, run._to_dict(cfg.scenario))
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


def test_the_scenario_package_template_compiles() -> None:
    sys.path.insert(0, str(_TEMPLATE_SRC))
    try:
        from my_scenario_package.configs import ReachGoalConfig
        from my_scenario_package.reach_goal import ReachGoalScenario
    finally:
        sys.path.remove(str(_TEMPLATE_SRC))
    conf = _TEMPLATE_SRC / "my_scenario_package" / "conf" / "scenario" / "reach_goal"
    values = yaml.safe_load((conf / "default.yaml").read_text())["scenario"]
    result = typecheck_scenario(ReachGoalScenario, ReachGoalConfig, values)
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


def test_a_scaffolded_scenario_package_compiles(tmp_path: Path) -> None:
    from autoware_carla_scenario.scaffold import create_scenario_package

    root = create_scenario_package("typecheck_scaffold_pkg", output_dir=tmp_path).root
    src = root / "src"
    sys.path.insert(0, str(src))
    try:
        configs = importlib.import_module("typecheck_scaffold_pkg.configs")
        scenario = importlib.import_module("typecheck_scaffold_pkg.typecheck_scaffold")
    finally:
        sys.path.remove(str(src))
    conf = src / "typecheck_scaffold_pkg" / "conf" / "scenario" / "typecheck_scaffold"
    values = yaml.safe_load((conf / "default.yaml").read_text())["scenario"]
    config_cls = next(v for k, v in vars(configs).items() if k.endswith("Config"))
    scenario_cls = next(
        v
        for k, v in vars(scenario).items()
        if k.endswith("Scenario") and k != "BaseScenario"
    )
    result = typecheck_scenario(scenario_cls, config_cls, values)
    assert result.ok, result.format()
    # Every function is called by the check: none is left out (#192).
    assert not result.warnings, result.format()


# ---------------------------------------------------------------------------
# The runner's step
# ---------------------------------------------------------------------------


def test_build_scenario_refuses_a_wrong_scenario_before_building_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.registry import register_scenario, unregister_scenario

    scenario, config, _ = _write_case(
        tmp_path, "self.register_fail_condition(TimeoutCondition(1.0))\n"
    )
    register_scenario("typecheck_case", scenario, config)

    def must_not_run(_cfg: Any) -> Any:
        raise AssertionError("the scenario was built before it was checked")

    monkeypatch.setattr(run, "build_ego_and_spawn", must_not_run)
    try:
        with pytest.raises(ScenarioTypeError, match=re.escape("label is a required")):
            run.build_scenario(
                OmegaConf.create({"scenario": {"name": "typecheck_case"}})
            )
        # Switched off, the runner goes on to build it.
        with pytest.raises(AssertionError, match="built before it was checked"):
            run.build_scenario(
                OmegaConf.create(
                    {"typecheck": "off", "scenario": {"name": "typecheck_case"}}
                )
            )
    finally:
        unregister_scenario("typecheck_case")


def test_a_batch_refuses_a_wrong_value_before_reading_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples import run

    def must_not_run(_cfg: Any) -> Any:
        raise AssertionError("the map was resolved before the scenarios were checked")

    monkeypatch.setattr(run, "resolve_map_paths", must_not_run)
    with pytest.raises(ScenarioTypeError, match=re.escape("scenario.timeout_seconds")):
        run.run_batch(
            ["lane_change/left", "lane_change/right"],
            ["scenario.timeout_seconds=fast"],
        )


def test_scenario_check_fails_an_unknown_scenario_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.typecheck.cli import main

    assert main(["scenario=lane_change/left", "scenario.name=no_such_scenario"]) == 1
    assert "unknown scenario name 'no_such_scenario'" in capsys.readouterr().out
