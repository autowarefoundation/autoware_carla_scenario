"""Unit tests for the scenario-package generator.

These exercise :mod:`autoware_carla_scenario.scaffold` end-to-end.  The
generator has no CARLA dependency (it only renders Jinja2 templates), so the
tests -- including generating a package and byte-compiling its Python files --
run in any environment.  ``TestOneGoalLanelet`` also runs the generated
scenario's ``setup()`` (and the package template's), with stand-ins for the
steps that need a live CARLA world.
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# tomllib landed in 3.11; tomli is the framework's own dependency on 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
    import tomli as tomllib

from autoware_carla_scenario.scaffold import (
    create_scenario_package,
    resolve_names,
)


class TestHydraSelectableScenarioNames:
    """``scenario-new`` refuses a scenario name Hydra cannot select (#1858)."""

    @pytest.mark.parametrize("scenario", ["café", "Café", "naïve_run", "ｆｏｏ"])
    def test_a_non_ascii_name_is_refused(self, scenario: str) -> None:
        with pytest.raises(ValueError, match="Hydra's override grammar"):
            resolve_names("my_pkg", scenario_name=scenario)

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["my_pkg", "-s", "café", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: scenario name 'café' must match")
        assert "Hydra's override grammar" in err
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("argv", "scenario"),
        [
            (["reach_goal_pkg"], "reach_goal"),
            (["my-cool-scenario-package"], "my_cool_scenario"),
            (["ParkingLot"], "parking_lot"),
            (["my_pkg", "-s", "MyScenario"], "my_scenario"),
            (["my_pkg", "-s", "cut-in.v2"], "cut_in_v2"),
            (["my_pkg", "-s", "HTTPServer"], "httpserver"),
            (["my_pkg", "-s", "fooBar9Baz"], "foo_bar9_baz"),
            (["my_pkg", "-s", "a1"], "a1"),
        ],
    )
    def test_the_printed_override_parses(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        scenario: str,
    ) -> None:
        from hydra.core.override_parser.overrides_parser import (  # noqa: PLC0415
            OverridesParser,
        )

        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main([*argv, "-o", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        overrides = [word for word in out.split() if word.startswith("scenario=")]
        assert overrides
        assert set(overrides) == {f"scenario={scenario}/default"}
        override = OverridesParser.create().parse_override(overrides[0])
        assert override.key_or_group == "scenario"
        assert override.value() == f"{scenario}/default"


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


class TestReservedPackageNames:
    """A package may not take a name that clashes once installed (#1451)."""

    @pytest.mark.parametrize(
        "package",
        [
            "autoware-carla-scenario",
            "autoware_carla_scenario",
            "AutowareCarlaScenario",
            "autoware.carla.scenario",
        ],
    )
    def test_framework_name_raises(self, package: str) -> None:
        with pytest.raises(ValueError, match="framework's own"):
            resolve_names(package)

    @pytest.mark.parametrize("package", ["json", "typing", "Typing", "asyncio"])
    def test_stdlib_name_raises(self, package: str) -> None:
        with pytest.raises(ValueError, match="standard library module"):
            resolve_names(package)

    @pytest.mark.parametrize(
        "package", ["openodd", "lanelet2", "typesafe_carla", "omegaconf", "hydra"]
    )
    def test_dependency_name_raises(self, package: str) -> None:
        with pytest.raises(ValueError, match="which autoware_carla_scenario depends"):
            resolve_names(package)

    def test_dependencies_are_read_off_the_metadata(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _DEPENDENCY_IMPORT_NAMES,
            _dependency_import_names,
        )

        # The installed metadata covers every name of the documented floor.
        names = _dependency_import_names()
        assert names >= _DEPENDENCY_IMPORT_NAMES
        # ... and an optional extra (``sumo``'s ``traci``) is not a dependency.
        assert "traci" not in names

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["autoware-carla-scenario", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith(
            "Error: package name 'autoware_carla_scenario' is reserved"
        )
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "package", ["my_json_pkg", "openodd_scenarios_pkg", "carla_pkg", "lanes"]
    )
    def test_ordinary_names_still_work(self, tmp_path: Path, package: str) -> None:
        result = create_scenario_package(package, output_dir=tmp_path)
        assert result.names["package_name"] == package
        assert (result.root / "pyproject.toml").is_file()


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


class TestBuiltinScenarioNames:
    """A generated package may not take a built-in scenario's name (#1802)."""

    def test_the_list_matches_the_built_in_registry(self) -> None:
        """What ``examples.run`` registers, read in a fresh interpreter.

        A fresh one so that what other tests register (or an installed
        scenario package, which the runner only loads later) does not count.
        """
        import json  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        from autoware_carla_scenario.scaffold.names import (  # noqa: PLC0415
            BUILTIN_SCENARIO_NAMES,
        )

        code = (
            "import json\n"
            "from autoware_carla_scenario import registry\n"
            "before = set(registry.get_scenario_registry())\n"
            "import autoware_carla_scenario.examples.run\n"
            "after = set(registry.get_scenario_registry())\n"
            "print(json.dumps(sorted(after - before)))\n"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        registered = json.loads(result.stdout.strip().splitlines()[-1])
        assert registered == sorted(BUILTIN_SCENARIO_NAMES)
        assert list(BUILTIN_SCENARIO_NAMES) == sorted(BUILTIN_SCENARIO_NAMES)

    @pytest.mark.parametrize(
        "scenario", ["lane_change", "cut_in", "LaneChange", "cut-in", "adversity"]
    )
    def test_a_built_in_name_is_refused(self, scenario: str) -> None:
        with pytest.raises(ValueError, match="names a built-in scenario"):
            resolve_names("my_pkg", scenario_name=scenario)

    def test_a_derived_built_in_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'lane_change' is reserved"):
            resolve_names("lane_change_package")

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["my_pkg", "--scenario", "lane_change", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: scenario name 'lane_change' is reserved")
        assert "built-in scenario" in err
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("scenario", ["my_lane_change", "lane_change_merge"])
    def test_near_misses_still_work(self, scenario: str) -> None:
        assert resolve_names("my_pkg", scenario_name=scenario)["scenario_name"] == (
            scenario
        )

    def test_the_check_does_not_import_the_runner(self) -> None:
        """``scenario-new`` reads the list without importing the runner or CARLA."""
        import subprocess  # noqa: PLC0415

        code = (
            "import sys\n"
            "from autoware_carla_scenario.scaffold.generator import resolve_names\n"
            "try:\n"
            "    resolve_names('my_pkg', scenario_name='lane_change')\n"
            "except ValueError:\n"
            "    pass\n"
            "loaded = [m for m in ('autoware_carla_scenario.examples.run',"
            " 'typesafe_carla.carla') if m in sys.modules]\n"
            "print(loaded)\n"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        assert result.stdout.strip().splitlines()[-1] == "[]"


class TestBuiltinScenarioGroups:
    """Nor a built-in scenario config group's (#1976): the package's configs
    would share the Hydra group, ``scenario=<group>/*`` would run the built-in
    ones too, and a built-in config would shadow the package's of that name."""

    @staticmethod
    def _conf_tree_groups() -> list[str]:
        import autoware_carla_scenario  # noqa: PLC0415

        conf = (
            Path(autoware_carla_scenario.__file__).parent
            / "examples"
            / "conf"
            / "scenario"
        )
        return sorted(
            entry.name
            for entry in conf.iterdir()
            if entry.is_dir() and not entry.name.startswith((".", "_"))
        )

    def test_the_groups_follow_the_conf_tree(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _builtin_scenario_groups,
        )
        from autoware_carla_scenario.scaffold.names import (  # noqa: PLC0415
            BUILTIN_SCENARIO_NAMES,
        )

        groups = _builtin_scenario_groups()
        assert list(groups) == self._conf_tree_groups()
        assert "lane_change_fail" in groups
        # Every builder has its configs there too.
        assert set(BUILTIN_SCENARIO_NAMES) <= set(groups)

    def test_hidden_and_private_directories_are_no_groups(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario.scaffold import generator  # noqa: PLC0415

        for name in ("lane_change", "__pycache__", ".hidden", "_private"):
            (tmp_path / name).mkdir()
        (tmp_path / "top.yaml").write_text("")
        monkeypatch.setattr(generator, "_BUILTIN_SCENARIO_CONF", tmp_path)
        generator._builtin_scenario_groups.cache_clear()
        try:
            assert generator._builtin_scenario_groups() == ("lane_change",)
        finally:
            generator._builtin_scenario_groups.cache_clear()

    def test_a_group_that_is_no_builder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="built-in scenario config group"):
            resolve_names("lane_change_fail_pkg", scenario_name="lane_change_fail")

    def test_a_derived_group_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'lane_change_fail' is reserved"):
            resolve_names("lane_change_fail_package")

    def test_every_group_is_refused(self) -> None:
        for scenario in self._conf_tree_groups():
            with pytest.raises(ValueError, match="is reserved"):
                resolve_names("my_pkg", scenario_name=scenario)

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        argv = ["my_pkg", "--scenario", "lane_change_fail", "-o", str(tmp_path)]
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: scenario name 'lane_change_fail' is reserved")
        assert "conf/scenario/lane_change_fail/" in err
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []


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


class TestFrameworkTopLevelModules:
    """A package may not hide another module the framework ships (#2031)."""

    def test_the_list_matches_the_framework_wheel(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _FRAMEWORK_TOP_LEVEL_MODULES,
        )

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        shipped = {
            module.split(".", 1)[0]
            for module in pyproject["tool"]["uv"]["build-backend"]["module-name"]
        }
        assert shipped - {"autoware_carla_scenario"} == set(
            _FRAMEWORK_TOP_LEVEL_MODULES
        )

    @pytest.mark.parametrize(
        "package", ["hydra_plugins", "hydra-plugins", "HydraPlugins"]
    )
    def test_hydra_plugins_is_refused(self, package: str) -> None:
        with pytest.raises(
            ValueError,
            match="'hydra_plugins' is reserved: autoware_carla_scenario ships",
        ):
            resolve_names(package)

    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["hydra_plugins", "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: package name 'hydra_plugins' is reserved")
        assert "namespace package" in err
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("package", ["hydra_plugins_pkg", "my_hydra_plugins"])
    def test_ordinary_names_still_work(self, tmp_path: Path, package: str) -> None:
        result = create_scenario_package(package, output_dir=tmp_path)
        assert result.names["package_name"] == package


class TestTemplateImportShadowing:
    """A scenario module may not replace a module the package imports (#2031)."""

    def test_the_imports_are_read_off_the_templates(self) -> None:
        from autoware_carla_scenario.scaffold.generator import (  # noqa: PLC0415
            _template_imported_modules,
        )

        imported = _template_imported_modules()
        assert {"math", "dataclasses", "logging", "autoware_carla_scenario"} <= imported
        assert "__future__" not in imported
        # tests/test_typecheck.py is outside the import package.
        assert "pytest" not in imported

    @pytest.mark.parametrize("scenario", ["math", "logging", "dataclasses", "Math"])
    def test_shadowing_scenario_name_raises(self, scenario: str) -> None:
        with pytest.raises(
            ValueError,
            match=f"'{scenario.lower()}' is reserved: the generated package imports",
        ):
            resolve_names("my_pkg", scenario_name=scenario)

    def test_derived_shadowing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="imports the module 'logging'"):
            resolve_names("logging_pkg")

    @pytest.mark.parametrize("scenario", ["math", "logging", "dataclasses"])
    def test_cli_reports_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], scenario: str
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["my_pkg", "-s", scenario, "-o", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith(f"Error: scenario name '{scenario}' is reserved")
        assert "Traceback" not in err
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "scenario", ["signal", "queue", "math_check", "logging_run", "my_dataclasses"]
    )
    def test_ordinary_names_still_work(self, tmp_path: Path, scenario: str) -> None:
        result = create_scenario_package(
            "my_pkg", output_dir=tmp_path, scenario_name=scenario
        )
        assert (result.root / "src" / "my_pkg" / f"{scenario}.py").is_file()


class TestOneLineDescription:
    """``project.description`` must be a single line (PEP 621, #2031)."""

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("line one\nline two", "line one line two"),
            ("line one\r\n\r\n  line two\r", "line one line two"),
            ("a\u2028b\u2029c\x85d", "a b c d"),
            ('one "line"', 'one "line"'),
        ],
    )
    def test_the_pyproject_gets_one_line(
        self, tmp_path: Path, description: str, expected: str
    ) -> None:
        result = create_scenario_package(
            "desc_pkg", output_dir=tmp_path, description=description
        )
        pyproject = tomllib.loads(
            (result.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert pyproject["project"]["description"] == expected
        # The docstring keeps the description as given.
        init = result.root / "src" / "desc_pkg" / "__init__.py"
        docstring = ast.get_docstring(
            ast.parse(init.read_text(encoding="utf-8")), clean=False
        )
        assert docstring is not None
        assert docstring.startswith(description)

    def test_the_cli_folds_it(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        assert main(["desc_pkg", "-d", "line one\nline two", "-o", str(tmp_path)]) == 0
        pyproject = tomllib.loads(
            (tmp_path / "desc_pkg" / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert pyproject["project"]["description"] == "line one line two"

    @pytest.mark.parametrize("description", [None, "", "\n", " \r\n "])
    def test_a_blank_description_gets_the_default(
        self, description: str | None
    ) -> None:
        names = resolve_names("desc_pkg", description=description)
        assert (
            names["description"] == "Desc scenario package for autoware_carla_scenario"
        )


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

    @pytest.mark.parametrize("with_configs", [True, False])
    def test_the_static_check_fails_without_a_config(
        self, tmp_path: Path, with_configs: bool
    ) -> None:
        """An empty conf dir fails the package's tests rather than skip them (#1214).

        Only the guard runs (``-k``): the compile test needs Codon and takes
        minutes, and with no config it has no case to run anyway.
        """
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        root = create_scenario_package("guard_pkg", output_dir=tmp_path).root
        if not with_configs:
            for path in (root / "src" / "guard_pkg" / "conf").rglob("*.yaml"):
                path.unlink()
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "--rootdir",
                str(root),
                "-k",
                "test_every_scenario_has_a_config",
                "tests/test_typecheck.py",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        output = result.stdout + result.stderr
        if with_configs:
            assert result.returncode == 0, output
        else:
            assert result.returncode == 1, output
            assert "no *.yaml config under" in output

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

    def test_force_replaces_the_whole_package(self, tmp_path: Path) -> None:
        """An earlier scenario's files do not survive ``force`` (#1696)."""
        first = create_scenario_package(
            "dup_pkg", scenario_name="alpha", output_dir=tmp_path
        )
        src = first.root / "src" / "dup_pkg"
        assert (src / "alpha.py").is_file()
        (first.root / "NOTES.txt").write_text("added by hand", encoding="utf-8")

        result = create_scenario_package(
            "dup_pkg", scenario_name="beta", output_dir=tmp_path, force=True
        )

        assert (src / "beta.py").is_file()
        assert (src / "conf" / "scenario" / "beta" / "default.yaml").is_file()
        assert not (src / "alpha.py").exists()
        assert not (src / "conf" / "scenario" / "alpha").exists()
        assert not (result.root / "NOTES.txt").exists()
        # No staging or set-aside directory is left beside the package.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dup_pkg"]

    def test_force_refuses_a_file_in_the_way(self, tmp_path: Path) -> None:
        (tmp_path / "dup_pkg").write_text("not a package", encoding="utf-8")
        with pytest.raises(FileExistsError, match="is not a directory"):
            create_scenario_package("dup_pkg", output_dir=tmp_path, force=True)
        assert (tmp_path / "dup_pkg").read_text(encoding="utf-8") == "not a package"

    @staticmethod
    def _fail_on_write(monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
        """Make the *nth* ``Path.write_text`` call raise ``OSError``."""
        real = Path.write_text
        calls = {"n": 0}

        def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
            calls["n"] += 1
            if calls["n"] == nth:
                raise OSError(28, "No space left on device", str(self))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write_text)

    def test_a_failed_write_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._fail_on_write(monkeypatch, nth=3)
        with pytest.raises(OSError, match="No space left"):
            create_scenario_package("dup_pkg", output_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

        # So the next run needs no --force.
        monkeypatch.undo()
        assert create_scenario_package("dup_pkg", output_dir=tmp_path).root.is_dir()

    def test_a_failed_forced_write_keeps_the_old_package(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = create_scenario_package(
            "dup_pkg", scenario_name="alpha", output_dir=tmp_path
        )
        before = sorted(str(p.relative_to(first.root)) for p in first.root.rglob("*"))
        self._fail_on_write(monkeypatch, nth=3)
        with pytest.raises(OSError, match="No space left"):
            create_scenario_package(
                "dup_pkg", scenario_name="beta", output_dir=tmp_path, force=True
            )
        after = sorted(str(p.relative_to(first.root)) for p in first.root.rglob("*"))
        assert after == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dup_pkg"]

    def test_a_package_that_cannot_be_moved_aside_leaves_no_holder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = create_scenario_package(
            "dup_pkg", scenario_name="alpha", output_dir=tmp_path
        )
        before = sorted(str(p.relative_to(first.root)) for p in first.root.rglob("*"))
        real = os.replace

        def replace(src: Any, dst: Any) -> None:
            if Path(src) == first.root:
                raise PermissionError(13, "Permission denied", str(src))
            real(src, dst)

        monkeypatch.setattr(os, "replace", replace)
        with pytest.raises(PermissionError):
            create_scenario_package(
                "dup_pkg", scenario_name="beta", output_dir=tmp_path, force=True
            )
        after = sorted(str(p.relative_to(first.root)) for p in first.root.rglob("*"))
        assert after == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dup_pkg"]

    @staticmethod
    def _occupy_while_rendering(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
        """Make another process's directory appear at *root* on the first write."""
        real = Path.write_text

        def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
            if not root.exists():
                root.mkdir()
                real(root / "theirs.txt", "not ours", encoding="utf-8")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write_text)

    def test_a_directory_that_appears_meanwhile_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``force``, a target created during rendering is not replaced."""
        from autoware_carla_scenario.scaffold import PackageExistsError  # noqa: PLC0415

        root = tmp_path / "dup_pkg"
        self._occupy_while_rendering(monkeypatch, root)
        with pytest.raises(PackageExistsError):
            create_scenario_package("dup_pkg", output_dir=tmp_path)
        assert sorted(p.name for p in root.iterdir()) == ["theirs.txt"]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dup_pkg"]

    def test_the_move_itself_refuses_an_occupied_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A target appearing after the last check is refused by the rename."""
        from autoware_carla_scenario.scaffold import (  # noqa: PLC0415
            PackageExistsError,
            generator,
        )

        root = tmp_path / "dup_pkg"
        monkeypatch.setattr(generator, "_occupied", lambda _root: False)
        self._occupy_while_rendering(monkeypatch, root)
        with pytest.raises(PackageExistsError):
            create_scenario_package("dup_pkg", output_dir=tmp_path)
        assert sorted(p.name for p in root.iterdir()) == ["theirs.txt"]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dup_pkg"]

    def test_output_dir_expands_the_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        monkeypatch.setenv("HOME", str(tmp_path))
        result = create_scenario_package("dup_pkg", output_dir="~/pkgs")
        assert result.root == tmp_path / "pkgs" / "dup_pkg"
        assert result.root.is_dir()

        assert main(["dup_pkg", "-o", "~/cli"]) == 0
        assert (tmp_path / "cli" / "dup_pkg" / "pyproject.toml").is_file()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cli", "pkgs"]

    def test_main_reports_an_os_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A filesystem error is ``Error: ...`` and exit code 1, not a traceback."""
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        not_a_dir = tmp_path / "a_file"
        not_a_dir.write_text("", encoding="utf-8")
        assert main(["dup_pkg", "-o", str(not_a_dir)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: ")
        assert str(not_a_dir) in err
        assert "Traceback" not in err

    def test_main_reports_a_failed_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from autoware_carla_scenario.scaffold.generator import main  # noqa: PLC0415

        self._fail_on_write(monkeypatch, nth=1)
        assert main(["dup_pkg", "-o", str(tmp_path)]) == 1
        assert capsys.readouterr().err.startswith("Error: [Errno 28] No space left")
        assert list(tmp_path.iterdir()) == []


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
    *,
    goal_ego: bool = False,
) -> Any:
    """Run ``setup()`` with stand-ins for what needs a live CARLA world.

    With *goal_ego*, the ego is one that plans its own route and the
    scenario's own ``derive_goal_from_route`` runs.
    """
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
    if goal_ego:
        # An ego that plans its own route, as `entity=autoware` is.  The spawn
        # snap needs the world, so it is stood in for by the one thing it asks
        # of the goal: `register_route_to_goal` calls require_goal(), which
        # refuses such an ego with no goal (#1377).
        from autoware_carla_scenario.entity.ego import EgoVehicle  # noqa: PLC0415

        monkeypatch.setattr(
            scenario,
            "ego_type",
            type("_RoutingEgo", (EgoVehicle,), {"requires_goal": True}),
        )
        monkeypatch.setattr(scenario, "_setup_ego_spawn", scenario.require_goal)
    else:
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


class TestGeneratedConfig:
    """The generated config refuses a bad timeout, as the template's does (#1818)."""

    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
    def test_a_timeout_not_positive_is_refused(
        self, tmp_path: Path, value: float
    ) -> None:
        result = create_scenario_package("timeout_pkg", output_dir=tmp_path)
        (configs,) = _import_from(result.root / "src", "timeout_pkg.configs")
        try:
            config_cls = getattr(configs, result.names["config_class"])
            with pytest.raises(
                ValueError,
                match=r"^scenario\.timeout_seconds must be finite and positive, ",
            ):
                config_cls(timeout_seconds=value)
            assert config_cls(timeout_seconds=8.0).timeout_seconds == 8.0
            assert config_cls().timeout_seconds == 10.0
        finally:
            for name in [m for m in sys.modules if m.split(".")[0] == "timeout_pkg"]:
                del sys.modules[name]

    def test_the_check_is_the_templates(self, tmp_path: Path) -> None:
        """The same ``__post_init__`` as the scenario package template's."""
        result = create_scenario_package("same_pkg", output_dir=tmp_path)
        generated = (result.root / "src" / "same_pkg" / "configs.py").read_text(
            encoding="utf-8"
        )
        template = (_TEMPLATE_SRC / "my_scenario_package" / "configs.py").read_text(
            encoding="utf-8"
        )
        marker = "    def __post_init__"
        assert marker in template
        assert (
            generated[generated.index(marker) :] == template[template.index(marker) :]
        )


class TestGoalFromRoute:
    """A scenario-new package names its goal, as the template does (#1377).

    Run with an ego that plans its own route and no stand-in for
    ``derive_goal_from_route``: without the call, ``setup()`` refuses it.
    """

    @pytest.mark.parametrize("goals", [[265], [460, 265]])
    def test_the_template_derives_its_goal(
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
            goal_ego=True,
        )
        assert scenario.goal_pose is not None
        assert scenario.goal_pose.lanelet_id == goals[-1]

    @pytest.mark.parametrize("goals", [[265], [460, 265]])
    def test_a_scaffolded_scenario_derives_its_goal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, goals: list[int]
    ) -> None:
        result = create_scenario_package("goal_ego_pkg", output_dir=tmp_path)
        module, configs = _import_from(
            result.root / "src", "goal_ego_pkg.goal_ego", "goal_ego_pkg.configs"
        )
        try:
            scenario = _set_up(
                monkeypatch,
                module,
                getattr(module, result.names["scenario_class"]),
                getattr(configs, result.names["config_class"])(goal_lanelet_ids=goals),
                goal_ego=True,
            )
            assert scenario.goal_pose is not None
            assert scenario.goal_pose.lanelet_id == goals[-1]
            assert scenario.require_goal() is scenario.goal_pose
        finally:
            for name in [m for m in sys.modules if m.split(".")[0] == "goal_ego_pkg"]:
                del sys.modules[name]
