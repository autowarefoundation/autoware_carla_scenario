"""Scenario Package export -- the reproducibility guarantees, in tests.

The expensive end-to-end check (``uv lock`` + ``uv sync --locked`` + the
generated package's own tests + the wheelhouse) needs the network and some
minutes, so it is marked ``slow``. Everything that can be asserted from the
generated files themselves runs unconditionally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# tomllib landed in 3.11, and 3.10 is still the floor of the supported range.
# Branching on sys.version_info rather than catching ImportError keeps mypy from
# reading the fallback as a redefinition when it checks against 3.11+.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only taken on 3.10
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
    ExportResult,
    PackageExportError,
    export_package,
    package_names,
)
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.wheelhouse import (
    WheelhouseError,
    build_wheelhouse,
    carla_extra,
    carla_wheels,
)


#: Options that keep an export offline: no lock, so no sync, no tests and no
#: wheelhouse -- everything those steps would need the network for.
OFFLINE = {
    "dev_mode": True,
    "lock": False,
    "verify": False,
    "run_tests": False,
}


@pytest.fixture
def package(tmp_path: Path) -> Path:
    """Export the starter scenario without locking (fast, offline)."""
    return export_package(new_document(), tmp_path, **OFFLINE).root


def _document_with_goal(lanelet_id: int | None, s: float = 0.0):
    """The starter document, with the ego's goal replaced -- or removed."""
    from autoware_carla_scenario.authoring.models import GoalSpec

    document = new_document()
    ego = document.ego
    assert ego is not None
    ego.goal = None if lanelet_id is None else GoalSpec(lanelet_id=lanelet_id, s=s)
    return document


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

    def test_the_egos_goal_uses_the_frameworks_own_keys(self) -> None:
        """The goal reaches the runner the way the spawn does: as ego.* keys."""
        config = build_scenario_config(_document_with_goal(265, 12.5))

        assert config["ego"]["goal_lanelet_id"] == 265
        assert config["ego"]["goal_s"] == 12.5

    def test_the_document_says_which_stack_drives_the_ego(self) -> None:
        """``ego.entity`` is what selects the stack, so the document writes it."""
        document = _document_with_goal(265)
        ego = document.ego
        assert ego is not None
        ego.driven_by = "autoware"

        assert build_scenario_config(document)["ego"]["entity"] == "autoware"
        assert build_scenario_config(new_document())["ego"]["entity"] == "autopilot"

    def test_the_starter_carries_its_goal_into_the_config(self) -> None:
        config = build_scenario_config(new_document())
        assert config["ego"]["goal_lanelet_id"] > 0

    def test_an_ego_with_no_goal_writes_no_goal_keys(self) -> None:
        # A document whose ego has no goal is a validation error, but the
        # renderer states only what the document says: an emitted key would
        # shadow the ego group's own null with a lanelet nobody chose.
        config = build_scenario_config(_document_with_goal(None))

        assert "goal_lanelet_id" not in config["ego"]
        assert "goal_s" not in config["ego"]

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
            "scenario/manifest.yaml",
            "src/cut_in_scenario/__init__.py",
            "src/cut_in_scenario/scenario.py",
            "tests/test_scenario.py",
        ):
            assert (package / relative).is_file(), relative

    def test_the_document_and_config_live_inside_the_module(
        self, package: Path
    ) -> None:
        """Anything beside the module is dropped when the wheel is built.

        A wheel holding a scenario package with no scenario in it installs
        perfectly and fails at run time, so where these two files sit is the
        difference between a shippable package and a broken one.
        """
        module = package / "src" / "cut_in_scenario"
        assert (module / "document.yaml").is_file()
        assert (module / "conf" / "scenario" / "cut_in.yaml").is_file()
        assert not (package / "scenario" / "document.yaml").exists()
        assert not (package / "conf").exists()

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
        declared = {
            requirement.split("[")[0] for requirement in data["project"]["dependencies"]
        }
        assert declared == {DISTRIBUTION, CONVERTER_DISTRIBUTION}
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

    def test_the_package_is_its_own_pytest_rootdir(self, package: Path) -> None:
        """pytest searches upwards, so an unpacked package would inherit config.

        The workspace this repository is one of sets ``addopts = "-n auto
        --testmon"``; a package unpacked anywhere under such a project would
        pick that up and fail before collecting a test, on plugins it has no
        reason to install.
        """
        data = tomllib.loads((package / "pyproject.toml").read_text())
        assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]

    def test_pyproject_pins_uv_when_its_version_is_known(self, package: Path) -> None:
        import shutil

        data = tomllib.loads((package / "pyproject.toml").read_text())
        if shutil.which("uv"):
            assert data["tool"]["uv"]["required-version"].startswith("==")
        else:
            assert "required-version" not in data.get("tool", {}).get("uv", {})

    def test_the_document_round_trips_into_the_package(self, package: Path) -> None:
        from autoware_carla_scenario.authoring.persistence import load_document

        document = package / "src/cut_in_scenario/document.yaml"
        assert load_document(document).id == "cut_in"

    def test_the_hydra_config_is_package_global(self, package: Path) -> None:
        text = (package / "src/cut_in_scenario/conf/scenario/cut_in.yaml").read_text()
        assert text.splitlines()[0] == "# @package _global_"
        assert yaml.safe_load(text)["scenario"]["name"] == "cut_in"

    def test_manifest_records_only_observed_values(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert manifest["format_version"] == 2
        assert manifest["scenario"]["id"] == "cut_in"
        assert manifest["runtime"]["python"]
        assert "uv" in manifest["runtime"]
        assert manifest["autoware_carla_scenario"]["source"] in (
            "git",
            "version",
            "path",
        )
        assert manifest["files"]["document"] == "src/cut_in_scenario/document.yaml"

    def test_skipping_the_lock_is_recorded_as_a_caveat(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert any("not reproducible" in note for note in manifest["notes"])


class TestExportRefusals:
    def test_an_invalid_document_is_refused(self, tmp_path: Path) -> None:
        document = new_document()
        document.assertions.pass_conditions = []
        with pytest.raises(PackageExportError):
            export_package(document, tmp_path, **OFFLINE)

    def test_an_occupied_destination_is_refused_without_force(
        self, tmp_path: Path
    ) -> None:
        export_package(new_document(), tmp_path, **OFFLINE)
        with pytest.raises(PackageExportError):
            export_package(new_document(), tmp_path, **OFFLINE)
        export_package(new_document(), tmp_path, force=True, **OFFLINE)

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

    def test_a_forced_unlocked_export_clears_a_previous_wheelhouse(
        self, tmp_path: Path
    ) -> None:
        """`force` replaces both directories, including the one not rebuilt.

        An unlocked export has no wheelhouse to put there, so a previous
        export's would otherwise stay -- stale wheels beside a fresh manifest
        that says the package has none.
        """
        stale = tmp_path / "cut_in_scenario_wheelhouse"
        stale.mkdir()
        (stale / "old-0.0.1-py3-none-any.whl").write_text("", encoding="utf-8")

        result = export_package(new_document(), tmp_path, force=True, **OFFLINE)
        assert result.wheelhouse is None
        assert not stale.exists()

    def test_a_failed_final_check_takes_the_wheelhouse_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`uv lock --check` runs after both directories are in place.

        A wheelhouse left behind by a failed export looks finished, and blocks
        the next export unless it is forced.
        """
        import autoware_carla_scenario.authoring.package_export as module

        def _lock(root: Path) -> str:
            (root / "uv.lock").write_text("# stub\n", encoding="utf-8")
            return ""

        def _wheelhouse(_root: Path, destination: Path, **_: Any) -> Any:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "stub-0.1.0-py3-none-any.whl").write_text("")
            return module.Wheelhouse(
                root=destination, distribution="stub", version="0.1.0"
            )

        def _refuse(_root: Path) -> str:
            raise PackageExportError("lock no longer matches")

        monkeypatch.setattr(module, "_lock", _lock)
        monkeypatch.setattr(module, "_verify_sync", lambda _root: "")
        monkeypatch.setattr(module, "_run_tests", lambda _root: (True, ""))
        monkeypatch.setattr(module, "build_wheelhouse", _wheelhouse)
        monkeypatch.setattr(module, "_check_lock", _refuse)

        with pytest.raises(PackageExportError):
            export_package(new_document(), tmp_path, dev_mode=True)
        assert list(tmp_path.iterdir()) == []


class TestVendoredCarlaClient:
    """The client is on no index, so a package that does not carry it cannot run."""

    def test_the_repositorys_wheel_is_found_and_matches_the_extra(self) -> None:
        wheels = carla_wheels()
        if not wheels:
            pytest.skip("no CARLA wheel is vendored in this checkout")
        assert all(wheel.suffix == ".whl" for wheel in wheels)
        # Exactly one client: the repository also vendors the legacy 0.9.16
        # wheel, and two mutually exclusive clients in one wheelhouse is not a
        # wheelhouse anybody can install.
        assert len({wheel.name.split("-")[1] for wheel in wheels}) == 1

    def test_the_vendored_wheels_cover_the_declared_python_range(self) -> None:
        """`requires-python` claims to stop where the client stops. Check it.

        The ceiling is a hand-written number in one file and the wheels it
        describes are files in another, and nothing but this makes them move
        together. Widen the range without vendoring a wheel and the failure
        lands on whoever installs the export, not on whoever widened it.
        """
        import re

        from autoware_carla_scenario.authoring.framework_pin import (
            framework_source_root,
        )

        wheels = carla_wheels()
        pyproject = framework_source_root() / "pyproject.toml"
        if not wheels or not pyproject.is_file():
            pytest.skip("not a source checkout with vendored CARLA wheels")

        declared = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
        floor = re.search(r">=\s*3\.(\d+)", declared)
        ceiling = re.search(r"<\s*3\.(\d+)", declared)
        assert floor and ceiling, declared

        supported = {
            f"cp3{minor}" for minor in range(int(floor.group(1)), int(ceiling.group(1)))
        }
        vendored = {wheel.name.split("-")[2] for wheel in wheels}
        assert vendored == supported, (
            f"{declared} says {sorted(supported)}, carla_wheels/ has "
            f"{sorted(vendored)} -- vendor the wheel or narrow the range"
        )

    def test_an_export_vendors_it_and_asks_for_it(self, package: Path) -> None:
        if not carla_wheels():
            pytest.skip("no CARLA wheel is vendored in this checkout")
        data = tomllib.loads((package / "pyproject.toml").read_text())
        assert f"{DISTRIBUTION}[{carla_extra()}]" in data["project"]["dependencies"]
        # Relative, so the package resolves wherever it is copied.
        assert data["tool"]["uv"]["find-links"] == ["carla_wheels"]
        assert list((package / "carla_wheels").glob("carla-*.whl"))

    def test_the_extra_is_asked_for_even_with_no_wheel_to_vendor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vendoring answers *where from*, not *whether*.

        The legacy client is published to PyPI, so nothing has to be vendored
        for it -- and an export that dropped the extra because it found no
        local wheel would build a wheelhouse with no `carla` in it and report
        success. A scenario that cannot import carla cannot run.
        """
        import autoware_carla_scenario.authoring.package_export as module

        monkeypatch.setattr(module, "carla_wheels", lambda _extra: [])
        result = export_package(new_document(), tmp_path, **OFFLINE)
        data = tomllib.loads((result.root / "pyproject.toml").read_text())
        assert f"{DISTRIBUTION}[{carla_extra()}]" in data["project"]["dependencies"]
        # Nothing to point a relative find-links at.
        assert "find-links" not in data.get("tool", {}).get("uv", {})
        assert any("No local wheel was vendored" in w for w in result.warnings)


class TestShippedRequirements:
    """`-r requirements.txt` has to install with `--no-index` like the rest."""

    def test_direct_references_become_the_version_that_was_built(self) -> None:
        """A git or path source sends pip to the network whatever --find-links says.

        `uv export` keeps those sources as direct references, so copying its
        output verbatim would ship a file that clones the repository -- on a
        machine chosen for having no network.
        """
        from autoware_carla_scenario.authoring.wheelhouse import (
            _pin_direct_references,
        )

        exported = "\n".join(
            [
                "annotated-types==0.8.0",
                f"{DISTRIBUTION}[carla] @ git+https://example.com/r@abc"
                "#subdirectory=autoware_carla_scenario",
                "    # via cut-in-scenario",
                f"{CONVERTER_DISTRIBUTION} @ file:///somewhere/local",
                "colorama==0.4.6 ; sys_platform == 'win32'",
                "unbuilt @ git+https://example.com/nope",
            ]
        )
        rewritten = _pin_direct_references(
            exported,
            (
                "annotated_types-0.8.0-py3-none-any.whl",
                "autoware_carla_scenario-0.1.0-py3-none-any.whl",
                "autoware_lanelet2_to_opendrive-2.62.0-py3-none-any.whl",
            ),
        ).splitlines()

        assert f"{DISTRIBUTION}[carla]==0.1.0" in rewritten
        assert f"{CONVERTER_DISTRIBUTION}==2.62.0" in rewritten
        # Markers travel; comments and unbuilt requirements are left alone.
        assert "colorama==0.4.6 ; sys_platform == 'win32'" in rewritten
        assert "    # via cut-in-scenario" in rewritten
        assert "unbuilt @ git+https://example.com/nope" in rewritten
        assert not any(" @ git+https://example.com/r" in line for line in rewritten)


class TestWheelhouseRefusals:
    def test_a_wheelhouse_needs_a_lock(self, tmp_path: Path) -> None:
        """It *is* the lockfile resolved into wheels; there is nothing else to build."""
        with pytest.raises(WheelhouseError):
            build_wheelhouse(
                tmp_path,
                tmp_path / "out",
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
            )

    def test_a_non_empty_destination_is_refused(self, tmp_path: Path) -> None:
        """A stale wheel left in the directory would be installed."""
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        destination = tmp_path / "out"
        destination.mkdir()
        (destination / "stale-1.0-py3-none-any.whl").write_text("", encoding="utf-8")
        with pytest.raises(WheelhouseError):
            build_wheelhouse(
                tmp_path,
                destination,
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
            )

    def test_skipping_the_lock_records_why_there_is_no_wheelhouse(
        self, tmp_path: Path
    ) -> None:
        """There is a wheelhouse exactly when there is a lock to resolve."""
        result = export_package(new_document(), tmp_path, **OFFLINE)
        assert result.wheelhouse is None
        assert any("No wheelhouse was built" in w for w in result.warnings)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> ExportResult:
    """One real export, shared by the checks below.

    It locks, syncs, runs the generated package's own tests and builds some
    seventy wheels -- a minute of network that neither check should pay twice.
    """
    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv is required to lock the exported package")
    destination = tmp_path_factory.mktemp("export")
    return export_package(new_document(), destination, dev_mode=True)


@pytest.mark.slow
class TestExportSelfCheck:
    """The full guarantee: the export really does sync, test and install."""

    def test_uv_sync_locked_and_package_tests_succeed(
        self, exported: ExportResult
    ) -> None:
        result = exported
        assert result.locked, result.log
        assert result.verified, result.log
        assert result.tested, result.log
        assert (result.root / "uv.lock").is_file()
        # Build output must not travel with the package.
        assert not (result.root / ".venv").exists()
        assert not (result.root / ".pytest_cache").exists()

        wheelhouse = result.wheelhouse
        assert wheelhouse is not None, result.log
        assert wheelhouse.root.is_dir()
        # The scenario's own wheel, the framework, the converter and the client
        # -- the four that are not on any index between them.
        names = " ".join(wheelhouse.wheels)
        assert "cut_in_scenario-" in names
        assert "autoware_carla_scenario-" in names
        assert "autoware_lanelet2_to_opendrive-" in names
        if carla_wheels():
            assert "carla-" in names
        assert (wheelhouse.root / "requirements.txt").is_file()
        assert (wheelhouse.root / "README.md").is_file()

        # The manifest is read after the export, so it has to name the
        # directory that is there -- not the temporary one it was built in.
        recorded = result.manifest["wheelhouse"]
        assert recorded["directory"] == wheelhouse.root.name
        assert (result.root.parent / recorded["directory"]).is_dir()

        # `-r requirements.txt` is documented as an offline install, so nothing
        # in it may send pip to a network.
        shipped = (wheelhouse.root / "requirements.txt").read_text()
        assert " @ git+" not in shipped
        assert " @ file://" not in shipped

    def test_the_wheelhouse_installs_with_pip_and_nothing_else(
        self, exported: ExportResult, tmp_path: Path
    ) -> None:
        """The whole point: no uv, no git, no index, no resolution.

        ``--no-index`` is what makes this a real check rather than a slow way of
        installing from PyPI: if a single wheel were missing, pip has nowhere
        else to look and the install fails.
        """
        import subprocess

        from autoware_carla_scenario.authoring.uv_tool import run_uv
        from autoware_carla_scenario.authoring.wheelhouse import venv_python

        result = exported
        assert result.wheelhouse is not None, result.log

        # `uv venv --seed` rather than the stdlib `venv`: the consumer's venv
        # needs pip in it, and uv is already required by this test.
        target = tmp_path / "venv"
        created = run_uv(
            tmp_path,
            "venv",
            str(target),
            "--python",
            result.wheelhouse.python_tag,
            "--seed",
            timeout=300,
        )
        assert created.returncode == 0, created.stdout + created.stderr
        python = venv_python(target)
        installed = subprocess.run(  # noqa: S603
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(result.wheelhouse.root),
                result.wheelhouse.distribution,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr

        # The document and the Hydra config have to be *in* the wheel: an
        # installed scenario has no project directory to read them out of.
        probe = subprocess.run(  # noqa: S603
            [
                str(python),
                "-c",
                "import cut_in_scenario as p;"
                "assert p.DOCUMENT_PATH.is_file(), p.DOCUMENT_PATH;"
                "assert p.CONF_DIR.is_dir(), p.CONF_DIR",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
        assert (venv_python(target).parent / "scenario").exists()


class TestFreeFormTextReachesTheManifest:
    """Titles and descriptions are prose, and prose ends up in TOML."""

    @pytest.mark.parametrize(
        "description",
        [
            'NPC1 cuts in "hard" on the ego',
            "line one\nline two",
            "back\\slash",
        ],
    )
    def test_a_description_survives_into_a_parsable_pyproject(
        self, tmp_path: Path, description: str
    ) -> None:
        document = new_document()
        document.description = description
        result = export_package(document, tmp_path, **OFFLINE)
        parsed = tomllib.loads((result.root / "pyproject.toml").read_text())
        assert parsed["project"]["description"] == description
