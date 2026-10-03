"""The static check with Codon: correct scenarios compile, wrong ones are refused.

Skipped where no Codon compiler is installed (the ``codon`` extra, which the
dev group includes on Linux x86_64).
"""

from __future__ import annotations

import importlib
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from autoware_carla_scenario.typecheck import (
    ScenarioTypeError,
    TypeCheckResult,
    available_toolchain,
    typecheck_scenario,
)

pytestmark = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)

_REPO = Path(__file__).resolve().parents[3]
_TEMPLATE_SRC = _REPO / "examples" / "scenario_package_template" / "src"

_CONFIG = """
from dataclasses import dataclass, field


@dataclass
class CaseConfig:
    name: str = "case"
    goal_lanelet_ids: list[int] = field(default_factory=lambda: [460, 265])
    timeout_seconds: float = 10.0
    min_speed_kmh: float | None = None
"""

_HEADER = """
from __future__ import annotations

import logging

import carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    AndCondition,
    BaseCondition,
    BaseScenario,
    ComparisonRule,
    EgoConfig,
    ElapsedTimeCondition,
    GroundProjectionConfig,
    LaneChangeDirection,
    Lanelet2Pose,
    ScenarioResult,
    SpeedCondition,
    StickyCondition,
    TickTiming,
    TimeoutCondition,
    TurnAction,
    TurnDirection,
    snap_to_carla_road,
)

from .configs import CaseConfig

logger = logging.getLogger(__name__)


class CaseScenario(BaseScenario):
    _config: CaseConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: CaseConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection)
        self._config = config or CaseConfig()

    def is_done(self) -> bool:
        return False

    def setup(self) -> None:
        cfg = self._config
"""

_VALID_SETUP = """
        self.derive_goal_from_route(cfg.goal_lanelet_ids)
        self._setup_ego_spawn()
        self.register_pre_tick(
            TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT, timing=TickTiming.PRE_TICK)
        )
        if cfg.min_speed_kmh is not None:
            self.register_fail_condition(
                AndCondition(
                    [
                        ElapsedTimeCondition(0.3, label="speed_check_delay"),
                        SpeedCondition(
                            entity_name=EGO_ROLE_NAME,
                            value=cfg.min_speed_kmh / 3.6,
                            rule=ComparisonRule.LESS_THAN,
                            label="ego_min_speed",
                        ),
                    ]
                )
            )
        self.register_fail_condition(TimeoutCondition(cfg.timeout_seconds, label="timeout"))
        logger.info("goal lanelets: %s", cfg.goal_lanelet_ids)
"""

_counter = 0


def _write_case(tmp_path: Path, setup: str, extra: str = "") -> tuple[type, type, Path]:
    """A scenario package in *tmp_path* whose setup() body is *setup*."""
    global _counter
    _counter += 1
    package = f"acs_typecheck_case_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "configs.py").write_text(_CONFIG)
    scenario = root / "scenario.py"
    scenario.write_text(
        _HEADER + textwrap.indent(textwrap.dedent(setup), " " * 8) + extra
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module(f"{package}.scenario")
        configs = importlib.import_module(f"{package}.configs")
    finally:
        sys.path.remove(str(tmp_path))
    return module.CaseScenario, configs.CaseConfig, scenario


def _line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in {path}")


def _only_error(result: TypeCheckResult) -> Any:
    assert not result.ok, result.format()
    assert result.diagnostics, result.output
    return result.diagnostics[0]


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


def test_a_correct_scenario_compiles(tmp_path: Path) -> None:
    scenario, config, _ = _write_case(tmp_path, _VALID_SETUP)
    result = typecheck_scenario(scenario, config, {"min_speed_kmh": 5})
    assert result.ok, result.format()


def test_a_custom_condition_is_checked_through_its_check_method(tmp_path: Path) -> None:
    custom = """

class AfterCondition(BaseCondition):
    _after: float

    def __init__(self, after: float) -> None:
        super().__init__("after")
        self._after = after

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        if elapsed < self._after:
            return None
        return ScenarioResult(passed=True, message="done", elapsed_seconds=elapsed)
"""
    setup = "self.register_pass_condition(StickyCondition(AfterCondition(3.0)))\n"
    scenario, config, _ = _write_case(tmp_path, setup, custom)
    assert typecheck_scenario(scenario, config).ok

    broken = custom.replace('message="done"', "message=elapsed")
    scenario, config, path = _write_case(tmp_path, setup, broken)
    error = _only_error(typecheck_scenario(scenario, config))
    assert error.path == str(path)
    assert error.line == _line_of(path, "message=elapsed")


# ---------------------------------------------------------------------------
# What is refused, and where it is reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        (
            "self.register_fail_condition(TimeoutCondition(10.0))\n",
            "TimeoutCondition(10.0)",
            "TimeoutCondition(): label is a required keyword argument",
        ),
        (
            'self.register_fail_condition(TimeoutCondition("10", label="t"))\n',
            'TimeoutCondition("10"',
            "no function 'TimeoutCondition.__init__'",
        ),
        (
            "self.register_pre_tick(TurnAction(EGO_ROLE_NAME, LaneChangeDirection.LEFT))\n",
            "LaneChangeDirection.LEFT",
            "no function 'TurnAction.__init__'",
        ),
        (
            "self.register_pass_condition(TurnAction(EGO_ROLE_NAME, TurnDirection.LEFT))\n",
            "register_pass_condition",
            "'TurnAction' does not match expected type 'BaseCondition'",
        ),
        (
            'self.register_fail_condition(SpeedCondition(42, 1.0, ComparisonRule.LESS_THAN, label="s"))\n',
            "SpeedCondition(42",
            "expected an EntityRole or a str",
        ),
        (
            "snap_to_carla_road(Lanelet2Pose(1, 0.0), self.world).to_carla_transfrom()\n",
            "to_carla_transfrom",
            "has no attribute 'to_carla_transfrom'",
        ),
        (
            "self.derive_goal_from_route(cfg.goal_lanelet_id)\n",
            "cfg.goal_lanelet_id)",
            "has no attribute 'goal_lanelet_id'",
        ),
    ],
    ids=[
        "missing-label",
        "str-for-float",
        "wrong-enum",
        "action-as-condition",
        "int-role",
        "method-typo",
        "config-field-typo",
    ],
)
def test_a_wrong_scenario_is_refused_at_its_line(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    scenario, config, path = _write_case(tmp_path, setup)
    error = _only_error(typecheck_scenario(scenario, config))
    assert message in error.message, error.format()
    assert error.path == str(path)
    assert error.line == _line_of(path, needle)


def test_a_config_value_of_the_wrong_type_is_reported_at_its_key(
    tmp_path: Path,
) -> None:
    scenario, config, _ = _write_case(tmp_path, _VALID_SETUP)
    error = _only_error(
        typecheck_scenario(scenario, config, {"timeout_seconds": "fast"})
    )
    assert error.path == "scenario.timeout_seconds"
    assert "'str' does not match expected type 'float'" in error.message

    error = _only_error(typecheck_scenario(scenario, config, {"timeout_secs": 3}))
    assert error.path == "scenario.timeout_secs"


def test_an_undeclared_attribute_is_refused_before_compiling(tmp_path: Path) -> None:
    scenario, config, path = _write_case(tmp_path, "self._count = 3\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "CaseScenario._count is assigned without a type" in error.message
    assert "`_count: int`" in error.message
    assert error.line == _line_of(path, "self._count = 3")


def test_a_module_without_a_codon_model_is_named(tmp_path: Path) -> None:
    scenario, config, path = _write_case(tmp_path, "import json\njson.dumps({})\n")
    error = _only_error(typecheck_scenario(scenario, config))
    assert "json has no Codon model" in error.message
    assert error.path == str(path)


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
