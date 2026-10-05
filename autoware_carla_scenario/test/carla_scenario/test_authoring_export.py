"""Scenario Package export -- the reproducibility guarantees, in tests.

The expensive end-to-end check (``uv lock`` + ``uv sync --locked`` + the
generated package's own tests + the wheelhouse) needs the network and some
minutes, so it is marked ``slow``. Everything that can be asserted from the
generated files themselves runs unconditionally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import jinja2
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
    TESTED_PYTHONS,
    WheelhouseError,
    build_wheelhouse,
    supported_pythons,
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
        """Transitive dependencies belong in uv.lock, not in the manifest."""
        data = tomllib.loads((package / "pyproject.toml").read_text())
        declared = {
            requirement.split("[")[0] for requirement in data["project"]["dependencies"]
        }
        assert declared == {DISTRIBUTION}
        assert data["project"]["requires-python"]

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
        assert manifest["format_version"] == 3
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

    def test_a_uv_timeout_arrives_as_an_export_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A raw TimeoutExpired flies past every handler that rolls back.

        The one around the final lock check runs after both directories are in
        place, so what it left behind was a finished-looking export that also
        blocked the next one.
        """
        import subprocess

        import autoware_carla_scenario.authoring.package_export as module

        def _timeout(*_args: Any, **_kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired("uv", 1)

        monkeypatch.setattr(module, "run_uv", _timeout)
        with pytest.raises(PackageExportError) as caught:
            module._run_uv(tmp_path, "lock", "--check", timeout=1)
        assert "did not finish" in str(caught.value)

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


class TestCarlaClient:
    """The client is typesafe_carla, a plain dependency of the framework."""

    def test_an_export_asks_for_no_extra_and_vendors_nothing(
        self, package: Path
    ) -> None:
        """typesafe-carla is on PyPI and the framework depends on it directly.

        No client extra to request, and no wheel that has to travel with the
        package -- so no relative find-links and no vendored directory.
        """
        data = tomllib.loads((package / "pyproject.toml").read_text())
        requirements = data["project"]["dependencies"]
        assert not [r for r in requirements if r.startswith(f"{DISTRIBUTION}[")]
        assert "find-links" not in data.get("tool", {}).get("uv", {})
        assert not (package / "carla_wheels").exists()

    def test_the_readme_says_how_the_client_is_built(self, package: Path) -> None:
        """The CPython package is compiled on the target, not shipped in a wheel.

        An offline install that does not know that finds out on its first
        scenario run, fifteen minutes into a build that needs `cc`.
        """
        readme = (package / "README.md").read_text()
        assert "typesafe-codon pycarla" in readme
        assert "carla-*.whl" not in readme
        assert "extra" not in readme

    def test_tested_pythons_match_the_frameworks_requires_python(self) -> None:
        """Two hand-written statements of CI's range; nothing else ties them.

        A generated package inherits the framework's `requires-python`, and the
        wheelhouse is resolved for each of :data:`TESTED_PYTHONS` it admits.
        Widen one without the other and the wheelhouse silently covers fewer
        interpreters than the package claims to support, or more than CI tests.
        """
        from autoware_carla_scenario.authoring.framework_pin import (
            framework_source_root,
        )

        pyproject = framework_source_root() / "pyproject.toml"
        if not pyproject.is_file():
            pytest.skip("not a source checkout")
        assert supported_pythons(framework_source_root()) == list(TESTED_PYTHONS)

        # ...and the range admits nothing outside them either.
        from packaging.specifiers import SpecifierSet

        declared = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
        admitted = [
            f"3.{minor}"
            for minor in range(6, 30)
            if SpecifierSet(declared).contains(f"3.{minor}")
        ]
        assert admitted == list(TESTED_PYTHONS), declared


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
                # uv writes a path source as a bare URL, name and all omitted.
                "file:///somewhere/local/local_helper",
                "colorama==0.4.6 ; sys_platform == 'win32'",
                "unbuilt @ git+https://example.com/nope",
            ]
        )
        rewritten = _pin_direct_references(
            exported,
            (
                "annotated_types-0.8.0-py3-none-any.whl",
                "autoware_carla_scenario-0.1.0-py3-none-any.whl",
                "local_helper-2.62.0-py3-none-any.whl",
            ),
        ).splitlines()

        assert f"{DISTRIBUTION}[carla]==0.1.0" in rewritten
        assert "local-helper==2.62.0" in rewritten
        assert not any(line.startswith("file:") for line in rewritten)
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

    def test_a_timed_out_tool_leaves_no_half_built_wheelhouse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`TimeoutExpired` is not a WheelhouseError and used to fly straight past.

        The destination has the project wheel in it by then, so what was left
        behind both broke the "nothing partial survives" guarantee and made the
        next attempt fail on a non-empty directory.
        """
        import subprocess

        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        destination = tmp_path / "out"

        def _timeout(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "half-0.1.0-py3-none-any.whl").write_text("")
            raise subprocess.TimeoutExpired("uv", 1)

        monkeypatch.setattr(module, "_export_requirements", _timeout)
        with pytest.raises(WheelhouseError) as caught:
            build_wheelhouse(
                tmp_path,
                destination,
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
            )
        assert "did not finish" in str(caught.value)
        assert not destination.exists()

    @pytest.mark.parametrize(
        "failure",
        [
            OSError("no space left"),
            # Rendered with StrictUndefined: a variable added to the template
            # and forgotten at the call site raises this, not an OSError.
            jinja2.TemplateError("undefined variable"),
        ],
        ids=["disk", "template"],
    )
    def test_a_failure_writing_the_install_files_cleans_up_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """The last two files are written onto a disk that just took 160 MB.

        A wheelhouse missing its `requirements.txt` must not be what a failed
        build leaves behind -- and the next attempt would be refused for
        finding a non-empty directory.
        """
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        destination = tmp_path / "out"

        def _wheels_but_no_files(*_args: Any, **_kwargs: Any) -> str:
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "done-1.0-py3-none-any.whl").write_text("")
            return ""

        monkeypatch.setattr(module, "_export_requirements", lambda _r: ("", ""))
        monkeypatch.setattr(module, "_build_project_wheel", _wheels_but_no_files)
        monkeypatch.setattr(
            module, "_builder_environment", lambda *_a: (tmp_path / "venv", "")
        )
        monkeypatch.setattr(module, "_download_wheels", lambda *_a: "")
        monkeypatch.setattr(
            module,
            "_write_install_files",
            lambda *_a, **_k: (_ for _ in ()).throw(failure),
        )

        with pytest.raises(WheelhouseError) as caught:
            build_wheelhouse(
                tmp_path,
                destination,
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
            )
        assert "did not finish" in str(caught.value)
        assert not destination.exists()

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

    def test_a_destination_that_is_a_file_is_refused_cleanly(
        self, tmp_path: Path
    ) -> None:
        """`iterdir()` on a regular file raises NotADirectoryError.

        That would leave a function documented to raise WheelhouseError for an
        unusable destination through a different door.
        """
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        occupied = tmp_path / "out"
        occupied.write_text("not a directory", encoding="utf-8")
        with pytest.raises(WheelhouseError) as caught:
            build_wheelhouse(
                tmp_path,
                occupied,
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
            )
        assert "not a directory" in str(caught.value)

    def test_skipping_the_lock_records_why_there_is_no_wheelhouse(
        self, tmp_path: Path
    ) -> None:
        """There is a wheelhouse exactly when there is a lock to resolve."""
        result = export_package(new_document(), tmp_path, **OFFLINE)
        assert result.wheelhouse is None
        assert any("No wheelhouse was built" in w for w in result.warnings)


class TestWheelhouseInterpreters:
    """A wheelhouse covers every interpreter the package can run under.

    Humble's Python is 3.10 and Jazzy's is 3.12. A wheelhouse resolved for one
    of them is a wheelhouse the other cannot install, and Ubuntu 22.04 has no
    `python3.12` to install out of the archive -- so the directory holds both.
    """

    def test_requires_python_picks_the_tested_interpreters(
        self, tmp_path: Path
    ) -> None:
        """The tested interpreters the package's `requires-python` admits."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "nothing"\nrequires-python = ">=3.10,<3.13"\n',
            encoding="utf-8",
        )
        assert supported_pythons(tmp_path) == ["3.10", "3.11", "3.12"]

    def test_an_open_range_stops_at_what_ci_tests(self, tmp_path: Path) -> None:
        """`>=3.11` is not a list: nothing past the tested range is built for."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "nothing"\nrequires-python = ">=3.11"\n',
            encoding="utf-8",
        )
        assert supported_pythons(tmp_path) == [
            version for version in TESTED_PYTHONS if version != "3.10"
        ]

    @pytest.mark.parametrize(
        "pyproject",
        [
            None,
            '[project]\nname = "nothing"\n',
            '[project]\nname = "nothing"\nrequires-python = "not a specifier"\n',
            '[project]\nname = "nothing"\nrequires-python = ">=4"\n',
            "this is not toml = = =",
        ],
    )
    def test_nothing_to_read_says_nothing(
        self, tmp_path: Path, pyproject: str | None
    ) -> None:
        """The caller falls back to `.python-version` on an empty answer."""
        if pyproject is not None:
            (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        assert supported_pythons(tmp_path) == []

    def test_one_resolution_pass_is_made_per_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Markers and wheel tags are both resolved by the running interpreter.

        One pass cannot produce a directory that installs under another Python,
        so the passes -- not just the recorded tags -- are what this checks.
        """
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "nothing"\nrequires-python = ">=3.10,!=3.11.*"\n',
            encoding="utf-8",
        )

        built: list[str] = []
        filled: list[Path] = []

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            built.append(python)
            return parent / f"builder-{python}", ""

        def _download(venv: Path, *_args: Any) -> str:
            filled.append(venv)
            return ""

        monkeypatch.setattr(module, "_export_requirements", lambda _r: ("", ""))
        monkeypatch.setattr(module, "_build_project_wheel", lambda *_a: "")
        monkeypatch.setattr(module, "_builder_environment", _environment)
        monkeypatch.setattr(module, "_download_wheels", _download)

        wheelhouse = build_wheelhouse(
            tmp_path,
            tmp_path / "out",
            distribution="nothing",
            version="0.1.0",
            run_command="scenario scenario=nothing",
        )

        assert built == ["3.10", "3.12"]
        assert wheelhouse.python_tags == ("3.10", "3.12")
        # Each pass fills the one directory from its own environment; a shared
        # builder venv would mean the second pass resolved nothing new.
        assert len(set(filled)) == 2
        assert "3.10, 3.12" in (wheelhouse.root / "README.md").read_text()

    def test_an_explicit_interpreter_overrides_requires_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building for one Python stays possible -- it is just not the default."""
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "nothing"\nrequires-python = ">=3.10,!=3.11.*"\n',
            encoding="utf-8",
        )

        built: list[str] = []

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            built.append(python)
            return parent, ""

        monkeypatch.setattr(module, "_export_requirements", lambda _r: ("", ""))
        monkeypatch.setattr(module, "_build_project_wheel", lambda *_a: "")
        monkeypatch.setattr(module, "_builder_environment", _environment)
        monkeypatch.setattr(module, "_download_wheels", lambda *_a: "")

        wheelhouse = build_wheelhouse(
            tmp_path,
            tmp_path / "out",
            distribution="nothing",
            version="0.1.0",
            run_command="scenario scenario=nothing",
            pythons=("3.11",),
        )
        assert built == ["3.11"]
        assert wheelhouse.python_tags == ("3.11",)

    def test_without_requires_python_the_recorded_interpreter_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A package that declares no range is built for its `.python-version`."""
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        (tmp_path / ".python-version").write_text("3.12\n", encoding="utf-8")

        built: list[str] = []

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            built.append(python)
            return parent, ""

        monkeypatch.setattr(module, "_export_requirements", lambda _r: ("", ""))
        monkeypatch.setattr(module, "_build_project_wheel", lambda *_a: "")
        monkeypatch.setattr(module, "_builder_environment", _environment)
        monkeypatch.setattr(module, "_download_wheels", lambda *_a: "")

        wheelhouse = build_wheelhouse(
            tmp_path,
            tmp_path / "out",
            distribution="nothing",
            version="0.1.0",
            run_command="scenario scenario=nothing",
        )
        assert built == ["3.12"]
        assert wheelhouse.python_tags == ("3.12",)


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
        # The scenario's own wheel and the framework -- the two that are not
        # on any index -- and the CARLA client with the Codon toolchain it
        # pins, without which the scenario cannot import carla.
        names = " ".join(wheelhouse.wheels)
        assert "cut_in_scenario-" in names
        assert "autoware_carla_scenario-" in names
        assert "typesafe_carla-" in names
        assert "typesafe_carla_toolchain-" in names
        assert (wheelhouse.root / "requirements.txt").is_file()
        readme = (wheelhouse.root / "README.md").read_text()
        # The venv layout is the exporting platform's, because that is the only
        # platform these wheels install on.
        expected_bin = "Scripts" if os.name == "nt" else "bin"
        assert f".venv/{expected_bin}/pip install --no-index" in readme
        # The wheelhouse is what leaves the machine, so it carries its own
        # provenance rather than leaving it in the tree the editor deletes.
        travelling = yaml.safe_load((wheelhouse.root / "manifest.yaml").read_text())
        assert (
            travelling["autoware_carla_scenario"]
            == (result.manifest["autoware_carla_scenario"])
        )
        # ...and its file map names things that are actually here, not paths
        # into the package the editor is about to delete.
        for key in ("manifest", "requirements", "readme", "scenario_wheel"):
            named = travelling["files"][key]
            assert (wheelhouse.root / named).is_file(), f"{key}: {named}"

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
        # uv writes a path source as the bare URL, with no ` @ ` in it at all.
        assert not [
            line
            for line in shipped.splitlines()
            if line.startswith(("file:", "git+", "http:", "https:", "/", "./", "../"))
        ], shipped

    def test_the_wheelhouse_installs_with_pip_and_nothing_else(
        self, exported: ExportResult, tmp_path: Path
    ) -> None:
        """The whole point: no uv, no git, no index, no resolution.

        ``--no-index`` is what makes this a real check rather than a slow way of
        installing from PyPI: if a single wheel were missing, pip has nowhere
        else to look and the install fails.

        Done once per interpreter the wheelhouse claims, because that claim is
        the feature: a directory that installs under 3.10 and 3.12 is what lets
        one export run on Humble and on Jazzy. Checking only the exporting
        machine's Python would pass just as happily on a wheelhouse missing
        every other one's wheels.
        """
        import subprocess

        from autoware_carla_scenario.authoring.uv_tool import run_uv
        from autoware_carla_scenario.authoring.wheelhouse import venv_python

        result = exported
        assert result.wheelhouse is not None, result.log
        assert result.wheelhouse.python_tags, result.log

        for interpreter in result.wheelhouse.python_tags:
            # `uv venv --seed` rather than the stdlib `venv`: the consumer's
            # venv needs pip in it, and uv is already required by this test.
            target = tmp_path / f"venv-{interpreter}"
            created = run_uv(
                tmp_path,
                "venv",
                str(target),
                "--python",
                interpreter,
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
            assert (
                installed.returncode == 0
            ), f"Python {interpreter}: {installed.stdout}{installed.stderr}"

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
