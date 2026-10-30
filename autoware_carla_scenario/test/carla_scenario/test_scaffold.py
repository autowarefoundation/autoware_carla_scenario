"""Unit tests for the scenario-package generator.

These exercise :mod:`autoware_carla_scenario.scaffold` end-to-end.  The
generator has no CARLA dependency (it only renders Jinja2 templates), so the
tests -- including generating a package and byte-compiling its Python files --
run in any environment.
"""

from __future__ import annotations

from pathlib import Path

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

        assert _reserved_scenario_names() == {"__init__", "configs", "conf"}

    @pytest.mark.parametrize("scenario", ["configs", "conf", "Configs", "configs-"])
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

    @pytest.mark.parametrize("scenario", ["config", "configs_check", "my_configs"])
    def test_near_misses_still_work(self, tmp_path: Path, scenario: str) -> None:
        result = create_scenario_package(
            "my_pkg", output_dir=tmp_path, scenario_name=scenario
        )
        pkg = result.root / "src" / "my_pkg"
        configs = (pkg / "configs.py").read_text(encoding="utf-8")
        assert f"class {result.names['config_class']}" in configs
        module = (pkg / f"{scenario}.py").read_text(encoding="utf-8")
        assert f"class {result.names['scenario_class']}" in module


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

    def test_force_overwrites(self, tmp_path: Path) -> None:
        create_scenario_package("dup_pkg", output_dir=tmp_path)
        # Should not raise when force=True.
        result = create_scenario_package("dup_pkg", output_dir=tmp_path, force=True)
        assert result.root.is_dir()
