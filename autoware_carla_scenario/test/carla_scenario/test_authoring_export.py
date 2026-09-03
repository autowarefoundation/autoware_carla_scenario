"""Scenario Package export -- the reproducibility guarantees, in tests.

The expensive end-to-end check (``uv lock`` + ``uv sync --locked`` + the
generated package's own tests) needs the network and about a minute, so it is
marked ``slow``. Everything that can be asserted from the generated files
themselves runs unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# tomllib landed in 3.11; the workspace is capped at 3.10 by the CARLA wheel.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.11+ has it in the stdlib
    import tomli as tomllib

from autoware_carla_scenario.authoring.framework_pin import (
    CONVERTER_DISTRIBUTION,
    DISTRIBUTION,
    Pin,
    PinResolutionError,
    normalize_repository_url,
    resolve_framework_pin,
)
from autoware_carla_scenario.authoring.hydra_config import (
    build_scenario_config,
    spawn_lanelet_key,
    spawn_s_key,
    swept_entity,
)
from autoware_carla_scenario.authoring.package_export import (
    PackageExportError,
    export_package,
    package_names,
)
from autoware_carla_scenario.authoring.starter import new_document


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """Export the starter scenario without locking (fast, offline)."""
    result = export_package(
        new_document(),
        tmp_path,
        dev_mode=True,
        lock=False,
        verify=False,
        run_tests=False,
    )
    return result.root


class TestHydraConfig:
    def test_ego_spawn_uses_the_frameworks_own_keys(self) -> None:
        """The sweeper already overrides these; an authored scenario must too."""
        document = new_document()
        ego = document.ego
        assert ego is not None
        assert spawn_lanelet_key(ego) == "ego.spawn_lanelet_id"
        assert spawn_s_key(ego) == "ego.spawn_s"

    def test_npc_spawn_keys_are_declared_so_hydra_accepts_overrides(self) -> None:
        document = new_document()
        config = build_scenario_config(document)
        npc = document.entity("npc1")
        assert npc is not None
        assert spawn_lanelet_key(npc) == "scenario.spawn_overrides.npc1.lanelet_id"
        assert config["scenario"]["spawn_overrides"]["npc1"] == {
            "lanelet_id": npc.spawn.lanelet_id,
            "s": npc.spawn.s.value,
        }

    def test_sweep_section_matches_the_sweepers_yaml_shape(self) -> None:
        config = build_scenario_config(new_document())
        constraints = config["sweep"]["constraints"]
        assert list(constraints) == ["scenario.spawn_overrides.npc1.lanelet_id"]
        assert (
            constraints["scenario.spawn_overrides.npc1.lanelet_id"][0]["type"] == "and"
        )
        assert config["sweep"]["bindings"] == {
            "scenario.spawn_overrides.npc1.s": {
                "type": "stop_line_offset",
                "offset": 15.0,
            }
        }

    def test_the_generated_constraints_parse_with_the_sweeper(self) -> None:
        """The whole point of reusing the sweeper's syntax."""
        from autoware_carla_scenario.sweeper.constraints import parse_constraint

        config = build_scenario_config(new_document())
        for target in config["sweep"]["constraints"].values():
            for entry in target:
                assert parse_constraint(entry) is not None

    def test_no_sweep_section_without_a_constraint_search(self) -> None:
        document = new_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.spawn.mode = "fixed"
        assert swept_entity(document) is None
        assert "sweep" not in build_scenario_config(document)

    def test_empty_map_fields_are_left_to_the_map_group(self) -> None:
        """An empty exclusion list must fall through, not shadow the group's."""
        config = build_scenario_config(new_document())
        assert "no_3d_model_lanelet_ids" not in config["map"]


class TestFrameworkPin:
    def test_a_branch_is_never_emitted(self) -> None:
        pin = Pin(kind="git", repository="https://example.invalid/r", commit="a" * 40)
        source = pin.uv_source()
        assert source is not None
        assert "branch" not in source
        assert source["rev"] == "a" * 40

    def test_version_pins_are_exact(self) -> None:
        pin = Pin(kind="version", version="1.2.3")
        assert pin.requirement() == f"{DISTRIBUTION}==1.2.3"
        assert pin.uv_source() is None

    def test_the_companion_pin_matches_the_frameworks_kind(self) -> None:
        pin = Pin(
            kind="git",
            repository="https://example.invalid/r",
            commit="a" * 40,
            subdirectory="autoware_carla_scenario",
        )
        companion = pin.companion()
        assert companion.distribution == CONVERTER_DISTRIBUTION
        assert companion.kind == "git"
        assert companion.commit == pin.commit
        assert companion.subdirectory == "autoware_lanelet2_to_opendrive"

    def test_the_companion_of_a_path_pin_is_the_sibling_checkout(self) -> None:
        pin = Pin(kind="path", path="/w/autoware_carla_scenario")
        assert pin.companion().path == "/w/autoware_lanelet2_to_opendrive"

    def test_a_path_pin_is_not_reproducible(self) -> None:
        assert not Pin(kind="path", path="/tmp/x").reproducible
        assert Pin(kind="version", version="1.0").reproducible

    def test_ssh_remotes_are_normalised_to_https(self) -> None:
        assert (
            normalize_repository_url("git@github.com:owner/repo.git")
            == "https://github.com/owner/repo"
        )

    def test_an_explicit_version_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCENARIO_EXPORT_FRAMEWORK_VERSION", "9.9.9")
        pin = resolve_framework_pin()
        assert pin.kind == "version"
        assert pin.version == "9.9.9"

    def test_no_immutable_pin_and_no_dev_mode_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better to refuse than to ship a package pinned to nothing."""
        import autoware_carla_scenario.authoring.framework_pin as module

        monkeypatch.delenv("SCENARIO_EXPORT_FRAMEWORK_VERSION", raising=False)
        monkeypatch.setattr(module, "_resolve_git_pin", lambda _root: None)
        with pytest.raises(PinResolutionError):
            resolve_framework_pin()

    def test_dev_mode_yields_a_local_path_with_a_warning(self) -> None:
        pin = resolve_framework_pin(dev_mode=True)
        assert pin.kind == "path"
        assert pin.warnings


class TestGeneratedPackage:
    def test_naming(self) -> None:
        names = package_names(new_document())
        assert names["scenario_id"] == "cut_in"
        assert names["package_name"] == "cut_in_scenario"
        assert names["distribution_name"] == "cut-in-scenario"

    def test_expected_files_exist(self, package: Path) -> None:
        for relative in (
            "pyproject.toml",
            "README.md",
            ".python-version",
            "scenario/document.yaml",
            "scenario/manifest.yaml",
            "conf/scenario/cut_in.yaml",
            "src/cut_in_scenario/__init__.py",
            "src/cut_in_scenario/scenario.py",
            "tests/test_scenario.py",
        ):
            assert (package / relative).is_file(), relative

    def test_python_version_is_an_exact_patch_version(self, package: Path) -> None:
        import platform

        recorded = (package / ".python-version").read_text().strip()
        assert recorded == platform.python_version()
        assert len(recorded.split(".")) == 3

    def test_pyproject_declares_the_workspace_and_nothing_transitive(
        self, package: Path
    ) -> None:
        """Transitive dependencies belong in uv.lock, not in the manifest.

        Both workspace projects are declared, though: the framework imports the
        converter at module scope without declaring it, so a package that named
        only the framework could not import it.
        """
        data = tomllib.loads((package / "pyproject.toml").read_text())
        assert set(data["project"]["dependencies"]) == {
            DISTRIBUTION,
            CONVERTER_DISTRIBUTION,
        }
        assert data["project"]["requires-python"]

    def test_both_projects_are_pinned_the_same_way(self, package: Path) -> None:
        """The two halves of the workspace must not drift apart."""
        data = tomllib.loads((package / "pyproject.toml").read_text())
        sources = data["tool"]["uv"]["sources"]
        framework = sources[DISTRIBUTION]
        converter = sources[CONVERTER_DISTRIBUTION]
        assert set(framework) == set(converter)
        if "git" in framework:
            assert framework["git"] == converter["git"]
            assert framework["rev"] == converter["rev"]
            assert framework["subdirectory"] != converter["subdirectory"]
        else:
            assert framework["path"] != converter["path"]

    def test_pyproject_pins_uv_when_its_version_is_known(self, package: Path) -> None:
        import shutil

        data = tomllib.loads((package / "pyproject.toml").read_text())
        if shutil.which("uv"):
            assert data["tool"]["uv"]["required-version"].startswith("==")
        else:
            assert "required-version" not in data.get("tool", {}).get("uv", {})

    def test_the_document_round_trips_into_the_package(self, package: Path) -> None:
        from autoware_carla_scenario.authoring.persistence import load_document

        assert load_document(package / "scenario/document.yaml").id == "cut_in"

    def test_the_hydra_config_is_package_global(self, package: Path) -> None:
        text = (package / "conf/scenario/cut_in.yaml").read_text()
        assert text.splitlines()[0] == "# @package _global_"
        assert yaml.safe_load(text)["scenario"]["name"] == "cut_in"

    def test_manifest_records_only_observed_values(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert manifest["format_version"] == 1
        assert manifest["scenario"]["id"] == "cut_in"
        assert manifest["runtime"]["python"]
        assert "uv" in manifest["runtime"]
        assert manifest["autoware_carla_scenario"]["source"] in (
            "git",
            "version",
            "path",
        )
        assert manifest["files"]["document"] == "scenario/document.yaml"

    def test_skipping_the_lock_is_recorded_as_a_caveat(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert any("not reproducible" in note for note in manifest["notes"])


class TestExportRefusals:
    def test_an_invalid_document_is_refused(self, tmp_path: Path) -> None:
        document = new_document()
        document.assertions.pass_conditions = []
        with pytest.raises(PackageExportError):
            export_package(document, tmp_path, dev_mode=True, lock=False)

    def test_an_occupied_destination_is_refused_without_force(
        self, tmp_path: Path
    ) -> None:
        options = {"dev_mode": True, "lock": False, "verify": False, "run_tests": False}
        export_package(new_document(), tmp_path, **options)
        with pytest.raises(PackageExportError):
            export_package(new_document(), tmp_path, **options)
        export_package(new_document(), tmp_path, force=True, **options)

    def test_a_failed_lock_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A package whose dependencies never resolved is not a success."""
        import autoware_carla_scenario.authoring.package_export as module

        def _fail(_root: Path) -> str:
            raise PackageExportError("dependency locking failed", log="boom")

        monkeypatch.setattr(module, "_lock", _fail)
        with pytest.raises(PackageExportError):
            export_package(new_document(), tmp_path, dev_mode=True)
        assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
class TestExportSelfCheck:
    """The full guarantee: the exported directory really does sync and test."""

    def test_uv_sync_locked_and_package_tests_succeed(self, tmp_path: Path) -> None:
        import shutil

        if shutil.which("uv") is None:
            pytest.skip("uv is required to lock the exported package")

        result = export_package(new_document(), tmp_path, dev_mode=True)
        assert result.locked, result.log
        assert result.verified, result.log
        assert result.tested, result.log
        assert (result.root / "uv.lock").is_file()
        # Build output must not travel with the package.
        assert not (result.root / ".venv").exists()
        assert not (result.root / ".pytest_cache").exists()
