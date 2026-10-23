"""The static check of this package's scenarios (``scenario-check``).

Compiles each scenario with Codon, built with the values of its YAML config,
as ``uv run scenario-check scenario='reach_goal/*'`` does.  Add a scenario by
adding its config class, its class and its YAML to ``CASES``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from autoware_carla_scenario.typecheck import find_supported_codon, typecheck_scenario
from autoware_carla_scenario.typecheck.toolchain import ToolchainError

from my_scenario_package.configs import ReachGoalConfig
from my_scenario_package.reach_goal import ReachGoalScenario

CONF = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "my_scenario_package"
    / "conf"
    / "scenario"
)

#: (scenario class, config class, the scenario's YAML configs)
CASES = [
    (ReachGoalScenario, ReachGoalConfig, sorted((CONF / "reach_goal").glob("*.yaml")))
]


@pytest.mark.parametrize(
    ("scenario_cls", "config_cls", "path"),
    [(s, c, p) for s, c, paths in CASES for p in paths],
    ids=lambda value: value.stem if isinstance(value, Path) else "",
)
def test_the_scenario_compiles(
    scenario_cls: type, config_cls: type, path: Path
) -> None:
    try:
        find_supported_codon()
    except ToolchainError as exc:
        pytest.skip(str(exc))
    values = yaml.safe_load(path.read_text())["scenario"]
    result = typecheck_scenario(scenario_cls, config_cls, values)
    assert result.ok, result.format()
