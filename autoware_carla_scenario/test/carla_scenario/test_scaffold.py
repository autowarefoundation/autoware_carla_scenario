"""Unit tests for the scenario-package generator.

These exercise :mod:`autoware_carla_scenario.scaffold` end-to-end.  The
generator has no CARLA dependency (it only renders Jinja2 templates), so the
tests -- including generating a package and byte-compiling its Python files --
run in any environment.  ``TestOneGoalLanelet`` also runs the generated
scenario's ``setup()`` (and the package template's), with stand-ins for the
steps that need a live CARLA world.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from autoware_carla_scenario.scaffold import (
    create_scenario_package,
    resolve_names,
)


class TestResolveNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (
                "reach_goal_pkg",
                ("reach_goal_pkg", "reach-goal-pkg", "reach_goal", "ReachGoalScenario"),
            ),
            (
                "my-cool-scenario-package",
                (
                    "my_cool_scenario_package",
                    "my-cool-scenario-package",
                    "my_cool_scenario",
                    "MyCoolScenario",  # no "ScenarioScenario" doubling
                ),
            ),
            (
                "ParkingLot",
                ("parking_lot", "parking-lot", "parking_lot", "ParkingLotScenario"),
            ),
        ],
    )
    def test_name_derivation(
        self, raw: str, expected: tuple[str, str, str, str]
    ) -> None:
        names = resolve_names(raw)
        assert (
            names["package_name"],
            names["distribution_name"],
            names["scenario_name"],
            names["scenario_class"],
        ) == expected

    def test_explicit_scenario_name(self) -> None:
        names = resolve_names("my_pkg", scenario_name="overtake")
        assert names["scenario_name"] == "overtake"
        assert names["scenario_class"] == "OvertakeScenario"
        assert names["config_class"] == "OvertakeConfig"

    @pytest.mark.parametrize("bad", ["123", "class", "!!!", "", "café"])
    def test_invalid_names_raise(self, bad: str) -> None:
        with pytest.raises(ValueError):
            resolve_names(bad)


class TestReservedScenarioNames:
    """A scenario module may not take the name of another generated file (#761)."""

    def test_reserved_names_follow_the_manifest(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _reserved_scenario_names,
        )

        assert _reserved_scenario_names() == {
            "__init__",
            "__all__",
            "configs",
            "conf",
            "register",
            "register_conf_dir",
            "register_scenario",
        }

    @pytest.mark.parametrize(
        "scenario",
        ["configs", "conf", "Configs", "configs-", "register", "RegisterScenario"],
    )
    def test_explicit_reserved_scenario_name_raises(self, scenario: str) -> None:
        with pytest.raises(ValueError, match="is reserved"):
            resolve_names("my_pkg", scenario_name=scenario)

    @pytest.mark.parametrize("package", ["configs_package", "configs-pkg", "conf"])
    def test_derived_reserved_scenario_name_raises(self, package: str) -> None:
        with pytest.raises(ValueError, match="is reserved"):
            resolve_names(package)

    def test_nothing_written(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'configs' is reserved"):
            create_scenario_package(
                "my_pkg", output_dir=tmp_path, scenario_name="configs"
            )
        assert not (tmp_path / "my_pkg").exists()

    @pytest.mark.parametrize("argv", [["my_pkg", "-s", "configs"], ["configs_package"]])
    def test_cli_reports_cleanly(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main([*argv, "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: scenario name 'configs' is reserved")
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "scenario", ["config", "configs_check", "my_configs", "registered", "path"]
    )
    def test_near_misses_still_work(self, tmp_path: Path, scenario: str) -> None:
        result = create_scenario_package(
            "my_pkg", output_dir=tmp_path, scenario_name=scenario
        )
        pkg = result.root / "src" / "my_pkg"
        configs = (pkg / "configs.py").read_text(encoding="utf-8")
        assert f"class {result.names['config_class']}" in configs
        module = (pkg / f"{scenario}.py").read_text(encoding="utf-8")
        assert f"class {result.names['scenario_class']}" in module


class TestScenarioModuleShadowing:
    """A generated class may not replace a name ``scenario.py`` imports (#809)."""

    def test_the_imports_are_read_off_the_template(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _scenario_module_globals,
        )

        taken = _scenario_module_globals()
        assert {"BaseScenario", "EgoConfig", "GroundProjectionConfig"} <= taken
        assert "logger" in taken
        assert "PlaceholderScenario" not in taken
        assert "PlaceholderConfig" not in taken

    @pytest.mark.parametrize(
        ("scenario", "shadowed"),
        [
            ("ego", "EgoConfig"),
            ("ground_projection", "GroundProjectionConfig"),
            ("GroundProjection", "GroundProjectionConfig"),
            ("base", "BaseScenario"),
            ("base_scenario", "BaseScenario"),
        ],
    )
    def test_shadowing_scenario_name_raises(self, scenario: str, shadowed: str) -> None:
        with pytest.raises(ValueError, match=f"is reserved: its class '{shadowed}'"):
            resolve_names("my_pkg", scenario_name=scenario)

    def test_derived_shadowing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="'EgoConfig'"):
            resolve_names("ego_package")

    def test_nothing_written(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is reserved"):
            create_scenario_package("my_pkg", output_dir=tmp_path, scenario_name="ego")
        assert not (tmp_path / "my_pkg").exists()

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["my_pkg", "-s", "ego", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: scenario name 'ego' is reserved")
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "scenario", ["ego_follow", "my_ego", "baseline", "ground", "and_condition"]
    )
    def test_near_misses_still_work(self, tmp_path: Path, scenario: str) -> None:
        result = create_scenario_package(
            "my_pkg", output_dir=tmp_path, scenario_name=scenario
        )
        module = (result.root / "src" / "my_pkg" / f"{scenario}.py").read_text(
            encoding="utf-8"
        )
        assert f"class {result.names['scenario_class']}(BaseScenario)" in module


class TestCreateScenarioPackage:
    def test_generates_valid_package(self, tmp_path: Path) -> None:
        result = create_scenario_package("reach_goal_pkg", output_dir=tmp_path)
        root = result.root

        assert root == tmp_path / "reach_goal_pkg"
        assert result.names["scenario_name"] == "reach_goal"
        pkg = root / "src" / "reach_goal_pkg"

        # Expected files exist.
        assert (root / "pyproject.toml").is_file()
        assert (root / "README.md").is_file()
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "reach_goal.py").is_file()
        assert (pkg / "configs.py").is_file()
        yaml_path = pkg / "conf" / "scenario" / "reach_goal" / "default.yaml"
        assert yaml_path.is_file()

        # Python renders to compilable source.
        for py in (pkg / "__init__.py", pkg / "reach_goal.py", pkg / "configs.py"):
            compile(py.read_text(encoding="utf-8"), str(py), "exec")

        # pyproject carries the right distribution name + entry point.
        # (Assert on text rather than parse: 3.10 is still the floor of the
        # supported range and has no stdlib tomllib.)
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        assert 'name = "reach-goal-pkg"' in pyproject
        assert 'reach_goal_pkg = "reach_goal_pkg:register"' in pyproject
        assert '[project.entry-points."autoware_carla_scenario.scenarios"]' in pyproject

        # YAML binds the scenario name (assert on text; avoids a yaml stub dep).
        yaml_text = yaml_path.read_text(encoding="utf-8")
        assert 'name: "reach_goal"' in yaml_text

    def test_mentions_the_static_check(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["reach_goal_pkg", "-o", str(tmp_path)]) == 0
        assert "scenario-check scenario=reach_goal/default" in capsys.readouterr().out
        root = tmp_path / "reach_goal_pkg"
        readme = (root / "README.md").read_text(encoding="utf-8")
        assert "scenario-check scenario='reach_goal/*'" in readme
        test = root / "tests" / "test_typecheck.py"
        compile(test.read_text(encoding="utf-8"), str(test), "exec")
        assert "typecheck_scenario" in test.read_text(encoding="utf-8")
        assert 'dev = ["pytest"]' in (root / "pyproject.toml").read_text()

    def test_refuses_existing_without_force(self, tmp_path: Path) -> None:
        create_scenario_package("dup_pkg", output_dir=tmp_path)
        with pytest.raises(FileExistsError):
            create_scenario_package("dup_pkg", output_dir=tmp_path)

    def test_the_refusal_names_how_each_entry_point_overwrites(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The Python API names ``force=True``; ``scenario-new``, ``--force`` (#993)."""
        from autoware_carla_scenario.scaffold import PackageExistsError  # noqa: PLC0415
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        create_scenario_package("dup_pkg", output_dir=tmp_path)
        with pytest.raises(PackageExistsError, match="force=True") as raised:
            create_scenario_package("dup_pkg", output_dir=tmp_path)
        assert raised.value.root == tmp_path / "dup_pkg"

        assert main(["dup_pkg", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "--force" in err
        assert "force=True" not in err
        assert str(tmp_path / "dup_pkg") in err

    def test_force_overwrites(self, tmp_path: Path) -> None:
        create_scenario_package("dup_pkg", output_dir=tmp_path)
        # Should not raise when force=True.
        result = create_scenario_package("dup_pkg", output_dir=tmp_path, force=True)
        assert result.root.is_dir()


# ---------------------------------------------------------------------------
# One goal lanelet (#1068)
# ---------------------------------------------------------------------------

_TEMPLATE_SRC = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "scenario_package_template"
    / "src"
)


def _import_from(src: Path, *names: str) -> list[ModuleType]:
    """Import *names* with *src* on ``sys.path`` for the import only."""
    sys.path.insert(0, str(src))
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(str(src))


def _set_up(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    scenario_cls: Any,
    config: Any,
) -> Any:
    """Run ``setup()`` with stand-ins for what needs a live CARLA world."""
    from autoware_carla_scenario import EgoConfig, Lanelet2Pose  # noqa: PLC0415
    from autoware_carla_scenario.conditions.always_true import (  # noqa: PLC0415
        AlwaysTrueCondition,
    )
    from autoware_carla_scenario.entity._spawn import (  # noqa: PLC0415
        SpawnPointIndex,
    )

    # The lane-position check needs the loaded map.
    monkeypatch.setattr(
        module,
        "EntityLanePositionCondition",
        lambda entity, pose, label: AlwaysTrueCondition(label=label),
    )
    scenario = scenario_cls(
        EgoConfig(SpawnPointIndex(0)), Lanelet2Pose(lanelet_id=242, s=25.0), config
    )
    monkeypatch.setattr(scenario, "derive_goal_from_route", lambda ids: None)
    monkeypatch.setattr(scenario, "_setup_ego_spawn", lambda: None)
    scenario.setup()
    return scenario


def _assert_goals_pass_condition(scenario: Any, goals: list[int]) -> None:
    from autoware_carla_scenario import (  # noqa: PLC0415
        AndCondition,
        StickyCondition,
    )

    (passing,) = scenario._pass_conditions
    if len(goals) == 1:
        # The single sticky itself: AndCondition refuses fewer than two.
        assert isinstance(passing, StickyCondition)
        assert passing.label == f"ego_on_lanelet_{goals[0]}_sticky"
    else:
        assert isinstance(passing, AndCondition)


class TestOneGoalLanelet:
    """A single goal lanelet is a valid goal (#1068)."""

    @pytest.mark.parametrize("goals", [[265], [460, 265]])
    def test_the_template_sets_up(
        self, monkeypatch: pytest.MonkeyPatch, goals: list[int]
    ) -> None:
        module, configs = _import_from(
            _TEMPLATE_SRC,
            "my_scenario_package.reach_goal",
            "my_scenario_package.configs",
        )
        scenario = _set_up(
            monkeypatch,
            module,
            module.ReachGoalScenario,
            configs.ReachGoalConfig(goal_lanelet_ids=goals),
        )
        _assert_goals_pass_condition(scenario, goals)

    def test_the_template_refuses_no_goal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, configs = _import_from(
            _TEMPLATE_SRC,
            "my_scenario_package.reach_goal",
            "my_scenario_package.configs",
        )
        with pytest.raises(ValueError, match="at least one lanelet"):
            _set_up(
                monkeypatch,
                module,
                module.ReachGoalScenario,
                configs.ReachGoalConfig(goal_lanelet_ids=[]),
            )

    @pytest.mark.parametrize("goals", [[265], [460, 265]])
    def test_a_scaffolded_scenario_sets_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, goals: list[int]
    ) -> None:
        result = create_scenario_package("one_goal_pkg", output_dir=tmp_path)
        module, configs = _import_from(
            result.root / "src", "one_goal_pkg.one_goal", "one_goal_pkg.configs"
        )
        try:
            scenario = _set_up(
                monkeypatch,
                module,
                getattr(module, result.names["scenario_class"]),
                getattr(configs, result.names["config_class"])(goal_lanelet_ids=goals),
            )
            _assert_goals_pass_condition(scenario, goals)
        finally:
            for name in [m for m in sys.modules if m.split(".")[0] == "one_goal_pkg"]:
                del sys.modules[name]
