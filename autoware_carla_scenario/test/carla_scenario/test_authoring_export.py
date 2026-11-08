"""Scenario Package export -- the reproducibility guarantees, in tests.

The expensive end-to-end check (``uv lock`` + ``uv sync --locked`` + the
generated package's own tests + the wheelhouse) needs the network and some
minutes, so it is marked ``slow``. Everything that can be asserted from the
generated files themselves runs unconditionally.
"""

from __future__ import annotations

import os
import re
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
from autoware_carla_scenario.authoring.compiler import compile_document
from autoware_carla_scenario.authoring.package_names import (
    framework_name_problem,
    scenario_id_problem,
)
from autoware_carla_scenario.authoring.validator import (
    EXPORT_REFUSAL_NOTE,
    validate_document,
)
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


def _in_set_values(tree: Any) -> list[Any]:
    """Every ``in_set`` constraint's ``values`` in *tree*, in order."""
    found: list[Any] = []
    if isinstance(tree, dict):
        if tree.get("type") == "in_set":
            found.append(tree.get("values"))
        for value in tree.values():
            found.extend(_in_set_values(value))
    elif isinstance(tree, list):
        for value in tree:
            found.extend(_in_set_values(value))
    return found


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

    def test_an_empty_exclusion_list_keeps_the_reference(self) -> None:
        """Left empty, the list is the group's: Hydra resolves the reference."""
        config = build_scenario_config(new_document())
        assert _in_set_values(config["sweep"]) == ["${map.no_3d_model_lanelet_ids}"]

    def test_a_listed_exclusion_list_replaces_the_reference(self) -> None:
        """On a built-in group the ``map`` keys lose to the group, so a list the
        author typed is written into the constraints that name it (#1523)."""
        document = new_document()
        document.map.no_3d_model_lanelet_ids = [1, 2]
        config = build_scenario_config(document)
        assert _in_set_values(config["sweep"]) == [[1, 2]]
        assert "no_3d_model_lanelet_ids" not in config["map"]
        # The document itself still names the reference.
        assert "${map.no_3d_model_lanelet_ids}" in str(document.model_dump(mode="json"))


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

    def test_a_token_in_an_https_remote_is_dropped(self) -> None:
        """The pin lands in a package meant to be shared (#1045)."""
        url = "https://oauth2:glpat-SECRET@gitlab.example.com/team/repo.git"
        normalized = normalize_repository_url(url)
        assert normalized == "https://gitlab.example.com/team/repo"
        assert "SECRET" not in normalized

    def test_user_and_password_in_an_https_remote_are_dropped(self) -> None:
        normalized = normalize_repository_url(
            "https://alice:hunter2@example.com:8443/o/r.git"
        )
        assert normalized == "https://example.com:8443/o/r"
        assert "hunter2" not in normalized
        assert "alice" not in normalized

    def test_an_ssh_url_with_a_port_drops_user_and_port(self) -> None:
        assert (
            normalize_repository_url("ssh://git@gitlab.example.com:2222/team/repo.git")
            == "https://gitlab.example.com/team/repo"
        )

    def test_an_scp_remote_is_normalised_as_before(self) -> None:
        assert (
            normalize_repository_url("git@gitlab.example.com:team/repo.git")
            == "https://gitlab.example.com/team/repo"
        )

    @pytest.mark.parametrize(
        "url",
        ["https://github.com/owner/repo", "https://github.com/owner/repo.git"],
    )
    def test_a_plain_https_remote_is_unchanged(self, url: str) -> None:
        assert normalize_repository_url(url) == "https://github.com/owner/repo"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://tok@[::1/o/r.git", "https://[::1/o/r"),
            ("https://tok@host]/o/r", "https://host]/o/r"),
            ("https://tok@[::1]:8443/o/r.git", "https://[::1]:8443/o/r"),
        ],
    )
    def test_a_malformed_ipv6_remote_does_not_crash(
        self, url: str, expected: str
    ) -> None:
        """``urlsplit`` raises on these; the export must not, nor keep the token."""
        assert normalize_repository_url(url) == expected

    def test_a_credentialed_origin_never_reaches_the_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import autoware_carla_scenario.authoring.framework_pin as module

        monkeypatch.setenv(
            module.REPOSITORY_ENV, "https://user:TOKEN@example.com/o/r.git"
        )
        pin = module._resolve_git_pin(module.framework_source_root())
        if pin is None:
            pytest.skip("not a git checkout")
        assert pin.repository == "https://example.com/o/r"
        assert "TOKEN" not in repr(pin.uv_source())
        assert "TOKEN" not in repr(pin.manifest())

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
        monkeypatch.setattr(module, "_index_release_version", lambda: None)
        with pytest.raises(PinResolutionError):
            resolve_framework_pin()

    def test_dev_mode_yields_a_local_path_with_a_warning(self) -> None:
        pin = resolve_framework_pin(dev_mode=True)
        assert pin.kind == "path"
        assert pin.warnings


def _git_repo(root: Path, *, origin: str) -> Path:
    """Make *root* a git repository with one commit and an ``origin``."""
    import subprocess  # noqa: PLC0415

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for args in (
        ("init", "-q"),
        ("add", "-A"),
        ("commit", "-q", "--no-gpg-sign", "-m", "init"),
        ("remote", "add", "origin", origin),
    ):
        subprocess.run(  # noqa: S603
            ["git", "-C", str(root), *args],  # noqa: S607
            check=True,
            capture_output=True,
            env=env,
        )
    return root


def _head(root: Path) -> str:
    import subprocess  # noqa: PLC0415

    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="needs git")
class TestFrameworkPinSourceTree:
    """A commit or path pin needs the framework's own source tree (#1976)."""

    @pytest.fixture
    def user_venv(self, tmp_path: Path) -> Path:
        """The ``lib/python3.12`` of a wheel install in a user's own repo."""
        repo = tmp_path / "user_project"
        lib = repo / ".venv" / "lib" / "python3.12"
        (lib / "site-packages" / "autoware_carla_scenario").mkdir(parents=True)
        (lib / "site-packages" / "autoware_carla_scenario" / "__init__.py").write_text(
            ""
        )
        (repo / "pyproject.toml").write_text('[project]\nname = "user-project"\n')
        _git_repo(repo, origin="https://github.com/someone/user_project")
        return lib

    @pytest.fixture
    def checkout(self, tmp_path: Path) -> Path:
        """A framework-like checkout: ``<repo>/autoware_carla_scenario``."""
        repo = tmp_path / "framework"
        project = repo / "autoware_carla_scenario"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "Autoware_Carla.Scenario"\nversion = "0"\n'
        )
        _git_repo(repo, origin="git@github.com:owner/autoware_carla_scenario.git")
        return project

    @staticmethod
    def _use_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> Any:
        import autoware_carla_scenario.authoring.framework_pin as module

        monkeypatch.delenv(module.VERSION_ENV, raising=False)
        monkeypatch.delenv(module.REPOSITORY_ENV, raising=False)
        monkeypatch.setattr(module, "framework_source_root", lambda: root)
        monkeypatch.setattr(module, "_index_release_version", lambda: None)
        return module

    def test_a_venv_inside_a_users_repo_is_no_checkout(
        self, monkeypatch: pytest.MonkeyPatch, user_venv: Path
    ) -> None:
        module = self._use_root(monkeypatch, user_venv)
        assert module._resolve_git_pin(user_venv) is None
        with pytest.raises(PinResolutionError, match=module.VERSION_ENV):
            resolve_framework_pin()

    def test_dev_mode_refuses_a_venv_too(
        self, monkeypatch: pytest.MonkeyPatch, user_venv: Path
    ) -> None:
        self._use_root(monkeypatch, user_venv)
        with pytest.raises(PinResolutionError, match="not one"):
            resolve_framework_pin(dev_mode=True)

    def test_a_project_of_another_name_is_no_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "other"
        repo.mkdir()
        (repo / "pyproject.toml").write_text('[project]\nname = "other"\n')
        _git_repo(repo, origin="https://github.com/someone/other")
        module = self._use_root(monkeypatch, repo)
        assert module._resolve_git_pin(repo) is None

    def test_an_unreadable_pyproject_is_no_checkout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        repo = tmp_path / "broken"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project\n")
        _git_repo(repo, origin="https://github.com/someone/broken")
        module = self._use_root(monkeypatch, repo)
        assert module._resolve_git_pin(repo) is None

    def test_a_framework_checkout_is_pinned_by_commit(
        self, monkeypatch: pytest.MonkeyPatch, checkout: Path
    ) -> None:
        self._use_root(monkeypatch, checkout)
        pin = resolve_framework_pin()
        assert pin.kind == "git"
        assert pin.repository == "https://github.com/owner/autoware_carla_scenario"
        assert pin.commit == _head(checkout)
        assert pin.subdirectory == "autoware_carla_scenario"

    def test_dev_mode_pins_a_framework_checkout_by_path(
        self, monkeypatch: pytest.MonkeyPatch, checkout: Path
    ) -> None:
        self._use_root(monkeypatch, checkout)
        pin = resolve_framework_pin(dev_mode=True)
        assert pin.kind == "path"
        assert pin.path == str(checkout)

    def test_an_index_install_is_pinned_by_version(
        self, monkeypatch: pytest.MonkeyPatch, user_venv: Path
    ) -> None:
        module = self._use_root(monkeypatch, user_venv)
        monkeypatch.setattr(module, "_index_release_version", lambda: "1.2.3")
        pin = resolve_framework_pin()
        assert pin.kind == "version"
        assert pin.requirement() == f"{DISTRIBUTION}==1.2.3"


class TestIndexReleaseVersion:
    """Which installs count as a published release (#1976)."""

    class _Distribution:
        def __init__(self, version: str, direct_url: str | None) -> None:
            self.version = version
            self._direct_url = direct_url

        def read_text(self, filename: str) -> str | None:
            return self._direct_url if filename == "direct_url.json" else None

    @pytest.mark.parametrize(
        ("version", "direct_url", "expected"),
        [
            ("1.2.3", None, "1.2.3"),
            ("1.2.3", '{"url": "file:///w/x.whl", "archive_info": {}}', None),
            ("1.2.3", '{"url": "file:///src", "dir_info": {"editable": true}}', None),
            ("1.2.3+local", None, None),
        ],
    )
    def test_only_an_index_release_counts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        version: str,
        direct_url: str | None,
        expected: str | None,
    ) -> None:
        from importlib import metadata  # noqa: PLC0415

        import autoware_carla_scenario.authoring.framework_pin as module

        fake = self._Distribution(version, direct_url)
        monkeypatch.setattr(metadata, "distribution", lambda _name: fake)
        assert module._index_release_version() == expected


class TestFrameworkVersionEnv:
    """``SCENARIO_EXPORT_FRAMEWORK_VERSION`` is stripped and validated (#1976)."""

    @pytest.mark.parametrize("value", ["", "  ", "\t\n"])
    def test_a_blank_value_counts_as_unset(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        import autoware_carla_scenario.authoring.framework_pin as module

        sentinel = Pin(
            kind="git", repository="https://example.invalid/r", commit="a" * 40
        )
        monkeypatch.setenv(module.VERSION_ENV, value)
        monkeypatch.setattr(module, "_resolve_git_pin", lambda _root: sentinel)
        assert resolve_framework_pin() is sentinel

    def test_surrounding_whitespace_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SCENARIO_EXPORT_FRAMEWORK_VERSION", " 1.2.3\n")
        assert resolve_framework_pin().requirement() == f"{DISTRIBUTION}==1.2.3"

    @pytest.mark.parametrize("value", ["latest", ">=1.0", "1.2.3; extra", "1..2"])
    def test_an_invalid_value_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("SCENARIO_EXPORT_FRAMEWORK_VERSION", value)
        with pytest.raises(PinResolutionError, match="not a valid version"):
            resolve_framework_pin()


class TestGeneratedPackage:
    def test_naming(self) -> None:
        names = package_names(new_document())
        assert names["scenario_id"] == "my_cut_in"
        assert names["package_name"] == "my_cut_in_scenario"
        assert names["distribution_name"] == "my-cut-in-scenario"

    def test_expected_files_exist(self, package: Path) -> None:
        for relative in (
            "pyproject.toml",
            "README.md",
            ".python-version",
            "scenario/manifest.yaml",
            "src/my_cut_in_scenario/__init__.py",
            "src/my_cut_in_scenario/scenario.py",
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
        module = package / "src" / "my_cut_in_scenario"
        assert (module / "document.yaml").is_file()
        assert (module / "conf" / "scenario" / "my_cut_in.yaml").is_file()
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

    def test_an_empty_parametrization_fails(self, package: Path) -> None:
        """Computed cases that come out empty fail, not skip (#1214)."""
        data = tomllib.loads((package / "pyproject.toml").read_text())
        ini = data["tool"]["pytest"]["ini_options"]
        assert ini["empty_parameter_set_mark"] == "fail_at_collect"

    def test_pyproject_pins_uv_when_its_version_is_known(self, package: Path) -> None:
        import shutil

        data = tomllib.loads((package / "pyproject.toml").read_text())
        if shutil.which("uv"):
            assert data["tool"]["uv"]["required-version"].startswith("==")
        else:
            assert "required-version" not in data.get("tool", {}).get("uv", {})

    def test_the_document_round_trips_into_the_package(self, package: Path) -> None:
        from autoware_carla_scenario.authoring.persistence import load_document

        document = package / "src/my_cut_in_scenario/document.yaml"
        assert load_document(document).id == "my_cut_in"

    def test_the_hydra_config_is_package_global(self, package: Path) -> None:
        text = (
            package / "src/my_cut_in_scenario/conf/scenario/my_cut_in.yaml"
        ).read_text()
        assert text.splitlines()[0] == "# @package _global_"
        assert yaml.safe_load(text)["scenario"]["name"] == "my_cut_in"

    def test_manifest_records_only_observed_values(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert manifest["format_version"] == 3
        assert manifest["scenario"]["id"] == "my_cut_in"
        assert manifest["runtime"]["python"]
        assert "uv" in manifest["runtime"]
        assert manifest["autoware_carla_scenario"]["source"] in (
            "git",
            "version",
            "path",
        )
        assert manifest["files"]["document"] == "src/my_cut_in_scenario/document.yaml"

    def test_skipping_the_lock_is_recorded_as_a_caveat(self, package: Path) -> None:
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert any("not reproducible" in note for note in manifest["notes"])


#: The framework's own Hydra config directory, the primary config of a run.
_FRAMEWORK_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)

_MAP_SOURCE = (
    "git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
    "@5058c6683a1077a99aa5fdca5e6ecff9d71fcfb1#autoware_maps/Town10HD_Opt"
)


def _document_on_own_map(kind: str):
    """The starter document, on a map named by *kind* (source or directory).

    As the editor leaves it: the starter's Nishishinjuku files cleared, the
    CARLA map name taken from the new map, the group still the starter's.
    """
    document = new_document()
    if kind == "source":
        document.map.source = _MAP_SOURCE
    else:
        document.map.directory = "/maps/Town10HD_Opt"
    document.map.name = "Town10HD_Opt"
    document.map.lanelet2_path = None
    document.map.xodr_path = None
    document.map.no_3d_model_lanelet_ids = [7, 11]
    assert document.map.group == "nishishinjuku"
    return document


def _compose_exported(package: Path, *overrides: str) -> Any:
    """Compose a run of the exported *package* as the ``scenario`` CLI does.

    The framework's config is the primary one; the package's ``conf/`` reaches
    Hydra's search path through ``register_conf_dir``, as the package's
    ``register()`` puts it there once installed.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from autoware_carla_scenario.registry import (
        register_conf_dir,
        unregister_conf_dir,
    )

    conf_dir = package / "src" / "my_cut_in_scenario" / "conf"
    register_conf_dir(conf_dir)
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(_FRAMEWORK_CONF.resolve()), version_base=None
        ):
            return compose(
                config_name="config", overrides=["scenario=my_cut_in", *overrides]
            )
    finally:
        unregister_conf_dir(conf_dir)


class TestExportedMapGroup:
    """A scenario on a source or directory map runs on that map (#1422).

    The framework composes the ``map`` group after the ``scenario`` one, so the
    ``map`` keys of the scenario's config lose to whichever group the run
    selects: the export ships the document's map as a group of its own.
    """

    @pytest.mark.parametrize("kind", ["source", "directory"])
    def test_the_exported_run_resolves_the_documents_map(
        self, tmp_path: Path, kind: str
    ) -> None:
        document = _document_on_own_map(kind)
        package = export_package(document, tmp_path, **OFFLINE).root

        group = (
            package
            / "src"
            / "my_cut_in_scenario"
            / "conf"
            / "map"
            / "my_cut_in_map.yaml"
        )
        assert group.read_text().splitlines()[0] == "# @package _global_"

        cfg = _compose_exported(package, "map=my_cut_in_map")
        resolved = cfg.map
        assert resolved.name == "Town10HD_Opt"
        assert resolved.lanelet2_path is None
        assert resolved.xodr_path is None
        assert resolved.projector_type is None
        assert resolved.overwrite_xodr is False
        assert list(resolved.no_3d_model_lanelet_ids) == [7, 11]
        if kind == "source":
            assert resolved.source == _MAP_SOURCE
            assert resolved.directory is None
        else:
            assert resolved.directory == "/maps/Town10HD_Opt"
            assert resolved.source is None

    def test_installing_an_opendrive_is_left_to_the_run(self, tmp_path: Path) -> None:
        """The group never installs one; a run can still ask for it."""
        document = _document_on_own_map("source")
        document.map.xodr_path = "/maps/Town10HD_Opt.xodr"
        package = export_package(document, tmp_path, **OFFLINE).root

        cfg = _compose_exported(package, "map=my_cut_in_map")
        assert cfg.map.xodr_path == "/maps/Town10HD_Opt.xodr"
        assert cfg.map.overwrite_xodr is False
        asked = _compose_exported(
            package, "map=my_cut_in_map", "map.overwrite_xodr=true"
        )
        assert asked.map.overwrite_xodr is True

    def test_the_package_says_to_run_it_with_its_own_group(
        self, tmp_path: Path
    ) -> None:
        result = export_package(_document_on_own_map("source"), tmp_path, **OFFLINE)
        readme = (result.root / "README.md").read_text()
        assert "scenario scenario=my_cut_in map=my_cut_in_map" in readme
        assert "map=nishishinjuku" not in readme
        init = (result.root / "src/my_cut_in_scenario/__init__.py").read_text()
        assert "map=my_cut_in_map" in init
        manifest = yaml.safe_load((result.root / "scenario/manifest.yaml").read_text())
        assert (
            manifest["files"]["map_config"]
            == "src/my_cut_in_scenario/conf/map/my_cut_in_map.yaml"
        )

    def test_a_built_in_group_ships_no_map_group(self, package: Path) -> None:
        assert not (package / "src" / "my_cut_in_scenario" / "conf" / "map").exists()
        assert "map=nishishinjuku" in (package / "README.md").read_text()
        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert "map_config" not in manifest["files"]

        cfg = _compose_exported(package, "map=nishishinjuku")
        assert cfg.map.name == "NishishinjukuMap"
        assert cfg.map.overwrite_xodr is True
        assert cfg.map.lanelet2_path


def _group_exclusions(group: str) -> list[int]:
    """The ``no_3d_model_lanelet_ids`` the framework's *group* file lists."""
    data = yaml.safe_load((_FRAMEWORK_CONF / "map" / f"{group}.yaml").read_text())
    return [int(v) for v in data["map"]["no_3d_model_lanelet_ids"]]


class TestExcludedLaneletIds:
    """The editor's "Excluded lanelet IDs" on a built-in map group (#1523).

    A non-empty list replaces the group's, an empty one inherits it -- in a
    run of the export and in the editor's preview alike.
    """

    @staticmethod
    def _effective(package: Path) -> list[Any]:
        from omegaconf import OmegaConf

        cfg = _compose_exported(package, "map=nishishinjuku")
        sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
        return _in_set_values(sweep)

    def test_an_empty_list_inherits_the_groups(self, tmp_path: Path) -> None:
        group = _group_exclusions("nishishinjuku")
        assert len(group) > 2
        package = export_package(new_document(), tmp_path, **OFFLINE).root
        assert self._effective(package) == [group]

    def test_a_listed_one_replaces_the_groups(self, tmp_path: Path) -> None:
        document = new_document()
        document.map.no_3d_model_lanelet_ids = [1, 2]
        package = export_package(document, tmp_path, **OFFLINE).root
        assert self._effective(package) == [[1, 2]]

    def test_the_effective_list_follows_the_same_rule(self) -> None:
        from autoware_carla_scenario.authoring.hydra_config import (
            effective_no_3d_model_lanelet_ids,
        )

        document = new_document()
        assert effective_no_3d_model_lanelet_ids(document) == _group_exclusions(
            "nishishinjuku"
        )
        document.map.no_3d_model_lanelet_ids = [1, 2]
        assert effective_no_3d_model_lanelet_ids(document) == [1, 2]
        # A document on its own map group keeps its own list, empty or not.
        own = _document_on_own_map("source")
        own.map.no_3d_model_lanelet_ids = []
        assert effective_no_3d_model_lanelet_ids(own) == []

    @pytest.mark.usefixtures("at_repo_root")
    @pytest.mark.parametrize("listed", [False, True])
    def test_the_preview_matches_what_the_sweep_enumerates(
        self, tmp_path: Path, listed: bool
    ) -> None:
        """The preview's match count is the sweep's, list or no list."""
        from autoware_carla_scenario.authoring.hydra_config import swept_slot
        from autoware_carla_scenario.editor import map_preview
        from autoware_carla_scenario.sweeper.constraints import (
            create_routing_graph,
            find_matching_lanelets,
            parse_constraint,
        )
        from autoware_carla_scenario.sweeper.expand import load_sweep_map
        from omegaconf import OmegaConf

        document = new_document()
        if listed:
            # Lanelets the group excludes are back in; one it does not is out.
            document.map.no_3d_model_lanelet_ids = [1, 2]
        slot = swept_slot(document)
        assert slot is not None
        preview = map_preview.evaluate_slot(document, slot, load_map=True)
        assert preview.map_loaded, preview.error

        package = export_package(document, tmp_path, **OFFLINE).root
        cfg = _compose_exported(package, "map=nishishinjuku")
        sweep = OmegaConf.to_container(cfg.sweep, resolve=True)
        assert isinstance(sweep, dict)
        constraints = [
            parse_constraint(entry)
            for entries in sweep["constraints"].values()
            for entry in entries
        ]
        lanelet_map = load_sweep_map(cfg.map)
        swept = find_matching_lanelets(
            constraints, lanelet_map, create_routing_graph(lanelet_map)
        )

        assert swept
        assert sorted(preview.matched_ids) == sorted(swept)


@pytest.fixture
def scenario_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The scenario registry, restored after the test: an exported
    ``my_cut_in`` package registers its document into it.

    ``examples.run`` registers the built-in scenarios when it is first
    imported: imported before the registry is copied, or they would be
    registered into the copy and gone once it is restored."""
    import autoware_carla_scenario.examples.run  # noqa: F401
    from autoware_carla_scenario import registry

    for name in ("_SCENARIO_REGISTRY", "_SCENARIO_CLASSES", "_SCENARIO_DOCUMENTS"):
        monkeypatch.setattr(registry, name, dict(getattr(registry, name)))
    monkeypatch.setattr(registry, "_CONF_DIRS", list(registry._CONF_DIRS))
    return registry


class TestRunOnTheDocumentsMap:
    """A run of a scenario on a source or directory map that composes another
    map group is refused before anything starts, naming ``map=<id>_map``
    (#1424)."""

    @staticmethod
    def _check(cfg: Any) -> None:
        from autoware_carla_scenario.examples.run import check_run_config

        check_run_config(cfg)

    @pytest.mark.parametrize("kind", ["source", "directory"])
    def test_the_default_map_group_is_refused(
        self, tmp_path: Path, kind: str, scenario_registry: Any
    ) -> None:
        from autoware_carla_scenario.examples.run import RunConfigError

        document = _document_on_own_map(kind)
        package = export_package(document, tmp_path, **OFFLINE).root
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)

        # The default group (nishishinjuku) wins over the scenario's map keys.
        cfg = _compose_exported(package)
        assert cfg.map.name == "NishishinjukuMap"
        with pytest.raises(RunConfigError, match=r"map=my_cut_in_map") as raised:
            self._check(cfg)
        assert "map.name is 'NishishinjukuMap'" in str(raised.value)

        self._check(_compose_exported(package, "map=my_cut_in_map"))

    def test_another_revision_of_the_map_is_refused(
        self, tmp_path: Path, scenario_registry: Any
    ) -> None:
        """A group on the same CARLA map, from another source, is another map."""
        from autoware_carla_scenario.examples.run import RunConfigError

        document = _document_on_own_map("source")
        document.map.source = _MAP_SOURCE.replace("5058c66", "0000000")
        package = export_package(document, tmp_path, **OFFLINE).root
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)

        with pytest.raises(RunConfigError, match=r"map\.source.*map=my_cut_in_map"):
            self._check(_compose_exported(package, "map=town10hd_opt"))
        self._check(_compose_exported(package, "map=my_cut_in_map"))

    def test_a_local_copy_of_the_same_map_is_accepted(
        self, tmp_path: Path, scenario_registry: Any
    ) -> None:
        """Explicit files and a directory are how a run points at its own copy."""
        document = _document_on_own_map("source")
        package = export_package(document, tmp_path, **OFFLINE).root
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)

        self._check(
            _compose_exported(
                package,
                "map=my_cut_in_map",
                "map.lanelet2_path=/copy/lanelet2_map.osm",
                "map.xodr_path=/copy/Town10HD_Opt.xodr",
            )
        )
        self._check(
            _compose_exported(package, "map=my_cut_in_map", "map.directory=/copy")
        )

    def test_a_document_on_a_built_in_group_is_not_checked(
        self, package: Path, scenario_registry: Any
    ) -> None:
        document = new_document()
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)
        self._check(_compose_exported(package))
        # Its group is the map; a run choosing another is the run's choice.
        self._check(_compose_exported(package, "map=town10hd_opt"))

    def test_the_exported_package_registers_its_document(
        self, tmp_path: Path, scenario_registry: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What ``register()`` puts in the registry is what the run checks."""
        from autoware_carla_scenario.examples.run import RunConfigError

        package = export_package(_document_on_own_map("source"), tmp_path, **OFFLINE)
        monkeypatch.syspath_prepend(str(package.root / "src"))
        import importlib

        try:
            importlib.import_module("my_cut_in_scenario").register()
            document = scenario_registry.get_scenario_document("my_cut_in")
            assert document is not None and document.map.source == _MAP_SOURCE
            with pytest.raises(RunConfigError, match=r"map=my_cut_in_map"):
                self._check(_compose_exported(package.root))
            self._check(_compose_exported(package.root, "map=my_cut_in_map"))
        finally:
            for module in [
                m for m in sys.modules if m.startswith("my_cut_in_scenario")
            ]:
                del sys.modules[module]

    def test_a_batch_names_the_case(
        self, tmp_path: Path, scenario_registry: Any
    ) -> None:
        from autoware_carla_scenario.examples.run import (
            RunConfigError,
            check_batch_configs,
        )

        document = _document_on_own_map("directory")
        package = export_package(document, tmp_path, **OFFLINE).root
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)
        with pytest.raises(RunConfigError, match=r"^case-a: .*map=my_cut_in_map"):
            check_batch_configs([_compose_exported(package)], ["case-a"])

    def test_a_batch_loads_each_document_once(
        self, tmp_path: Path, scenario_registry: Any
    ) -> None:
        from autoware_carla_scenario.examples.run import check_batch_configs

        document = _document_on_own_map("source")
        package = export_package(document, tmp_path, **OFFLINE).root
        loads: list[str] = []

        def load() -> Any:
            loads.append(document.id)
            return document

        scenario_registry.register_scenario_document("my_cut_in", load)
        cfg = _compose_exported(package, "map=my_cut_in_map")
        check_batch_configs([cfg, cfg, cfg], ["a", "b", "c"])
        assert loads == ["my_cut_in"]

    @pytest.mark.parametrize(
        "spelling",
        [
            # A browser URL of the same revision and directory.
            "https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps"
            "/tree/5058c6683a1077a99aa5fdca5e6ecff9d71fcfb1/autoware_maps/Town10HD_Opt",
            _MAP_SOURCE + "/",
        ],
    )
    def test_another_spelling_of_the_documents_source_is_accepted(
        self, tmp_path: Path, scenario_registry: Any, spelling: str
    ) -> None:
        document = _document_on_own_map("source")
        package = export_package(document, tmp_path, **OFFLINE).root
        scenario_registry.register_scenario_document("my_cut_in", lambda: document)
        self._check(
            _compose_exported(package, "map=my_cut_in_map", f"map.source='{spelling}'")
        )

    def test_an_unparsable_source_is_compared_as_text(self) -> None:
        from autoware_carla_scenario.authoring.hydra_config import own_map_mismatch

        document = _document_on_own_map("source")
        document.map.source = "not a source"
        on_it = {"name": "Town10HD_Opt", "source": " not a source "}
        assert own_map_mismatch(document, on_it) is None
        other = {"name": "Town10HD_Opt", "source": _MAP_SOURCE}
        assert "map.source" in (own_map_mismatch(document, other) or "")


class TestEveryPathRefusesAnotherMapBeforeReadingIt:
    """The own-map refusal (#1424) comes before any map is resolved, fetched or
    searched on every path that composes a scenario and expands or plans it
    without running it: a ``--multirun`` batch, ``scenario-expand``, the
    viewer's sweep resolver and ``scenario-odd route`` (#1428).

    The document is registered by the scenario packages' loading alone, as an
    installed package's ``register()`` does: a path that checked before
    loading them would find no document and let the run through.
    """

    _REFUSAL = (
        r"scenario 'my_cut_in' runs on the map its document names, but map\.name "
        r"is 'NishishinjukuMap'.*map=my_cut_in_map"
    )

    @pytest.fixture
    def exported_package(
        self,
        tmp_path: Path,
        scenario_registry: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Any:
        """Load the packages: puts the exported package's conf dir on the
        search path and registers its document, as its ``register()`` would."""
        # Imported before expand_config is patched: it binds the name on import.
        import autoware_carla_scenario.examples.expand  # noqa: F401
        from autoware_carla_scenario.sweeper import expand as sweeper_expand

        document = _document_on_own_map("source")
        package = export_package(document, tmp_path, **OFFLINE).root

        def load_scenario_plugins() -> None:
            scenario_registry.register_conf_dir(
                package / "src" / "my_cut_in_scenario" / "conf"
            )
            scenario_registry.register_scenario_document("my_cut_in", lambda: document)

        monkeypatch.setattr(
            scenario_registry, "load_scenario_plugins", load_scenario_plugins
        )

        def expand_config(*_: Any, **__: Any) -> list[list[str]]:
            raise AssertionError("expanded the sweep on the composed map")

        monkeypatch.setattr(sweeper_expand, "expand_config", expand_config)
        return load_scenario_plugins

    def test_a_multirun_batch(
        self,
        exported_package: Any,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from autoware_carla_scenario.examples import run

        exported_package()
        with pytest.raises(SystemExit) as exited:
            run.run_batch(["my_cut_in"], [], expand=True)
        assert exited.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("Error: my_cut_in: scenario 'my_cut_in'")
        assert re.search(self._REFUSAL, err)

        # On its own map the sweep is expanded.
        from autoware_carla_scenario.sweeper import expand as sweeper_expand

        monkeypatch.setattr(sweeper_expand, "expand_config", lambda *_, **__: [[]])
        assert run._expand_batch(["my_cut_in"], ["map=my_cut_in_map"]) == [
            ("my_cut_in", [])
        ]

    def test_the_scenario_cli_loads_the_packages_before_the_batch_checks(
        self,
        exported_package: Any,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from autoware_carla_scenario.examples import run

        monkeypatch.setattr(run, "load_scenario_plugins", exported_package)
        monkeypatch.setenv("SWEEP_RESUME_FROM", "0")
        monkeypatch.setattr(
            sys, "argv", ["scenario", "--multirun", "scenario=my_cut_in*"]
        )
        with pytest.raises(SystemExit) as exited:
            run.main()
        assert exited.value.code == 1
        assert re.search(self._REFUSAL, capsys.readouterr().err)

    def test_scenario_expand(
        self,
        exported_package: Any,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from autoware_carla_scenario.examples import expand

        monkeypatch.setattr(expand, "load_scenario_plugins", exported_package)
        monkeypatch.setattr(
            expand,
            "expand_config",
            lambda *_, **__: pytest.fail("expanded the sweep on the composed map"),
        )
        assert expand.main(["scenario=my_cut_in"]) == 1
        out, err = capsys.readouterr()
        assert out == ""
        assert err.startswith("Error: my_cut_in: scenario 'my_cut_in'")
        assert re.search(self._REFUSAL, err)

        monkeypatch.setattr(expand, "expand_config", lambda *_, **__: [[]])
        assert expand.main(["scenario=my_cut_in", "map=my_cut_in_map"]) == 0

    def test_the_viewers_sweep_resolver(
        self, exported_package: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.ui import sweep_resolver

        monkeypatch.setattr(sweep_resolver, "load_scenario_plugins", exported_package)
        with pytest.raises(sweep_resolver.SweepResolutionError) as raised:
            sweep_resolver.resolve_sweep("my_cut_in")
        assert str(raised.value).startswith("my_cut_in: scenario 'my_cut_in'")
        assert re.search(self._REFUSAL, str(raised.value))

        from autoware_carla_scenario.sweeper import expand as sweeper_expand

        monkeypatch.setattr(sweeper_expand, "expand_config", lambda *_, **__: [[]])
        assert sweep_resolver.resolve_sweep("my_cut_in", ["map=my_cut_in_map"]) == [
            ["scenario=my_cut_in"]
        ]

    def test_the_viewers_sweep_resolver_reports_what_the_check_raises(
        self, exported_package: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error reading the composed config is the resolver's own error."""
        from omegaconf.errors import InterpolationResolutionError

        from autoware_carla_scenario.authoring import hydra_config
        from autoware_carla_scenario.ui import sweep_resolver

        def refusal(*_: Any, **__: Any) -> None:
            raise InterpolationResolutionError("map.name: no such key")

        monkeypatch.setattr(sweep_resolver, "load_scenario_plugins", exported_package)
        monkeypatch.setattr(hydra_config, "scenario_map_refusal", refusal)
        with pytest.raises(
            sweep_resolver.SweepResolutionError,
            match=r"^my_cut_in: InterpolationResolutionError: map\.name: no such key",
        ):
            sweep_resolver.resolve_sweep("my_cut_in")

    def test_scenario_odd_route(
        self,
        exported_package: Any,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from autoware_carla_scenario.examples import run
        from autoware_carla_scenario.odd.cli import main as odd_main

        def planned(*_: Any, **__: Any) -> None:
            raise AssertionError("planned the route on the composed map")

        monkeypatch.setattr(run, "build_planned_route", planned)
        assert odd_main(["route", "default", "my_cut_in"]) == 2
        out = capsys.readouterr().out
        assert "my_cut_in: NOT PLANNED: scenario 'my_cut_in'" in out
        assert re.search(self._REFUSAL, out)
        assert "planned the route" not in out

        # On its own map the route is planned.
        assert odd_main(["route", "default", "my_cut_in", "map=my_cut_in_map"]) == 2
        assert "planned the route" in capsys.readouterr().out


#: Scenario ids an export accepts, each with the Python package it is named
#: after.  Before #1358 ``__`` and a trailing ``_`` were what the package-name
#: sanitising rewrote, and so where the scenario id and the names derived from
#: it could part (#1072); export now refuses them instead (see below).
_SCENARIO_IDS = [
    ("stop_line", "stop_line_scenario"),
    ("my_cut_in", "my_cut_in_scenario"),
    ("scenario", "scenario_scenario"),
    ("autoware", "autoware_scenario"),
    ("autoware_carla_x", "autoware_carla_x_scenario"),
    ("scenario_one", "scenario_one_scenario"),
    # Not the framework's package any more: ``_scenario`` is always added.
    ("autoware_carla_scenario", "autoware_carla_scenario_scenario"),
    # ``_scenario`` is always added: ``stop_line`` already has its package.
    ("stop_line_scenario", "stop_line_scenario_scenario"),
    ("new_scenario", "new_scenario_scenario"),
]

#: Ids the model's pattern accepts whose package another id already has
#: (#1358): each with the id it would collide with and a phrase the refusal
#: must contain.
_COLLIDING_IDS = [
    ("stop__line", "stop_line", "contains `__`"),
    ("stop_line_", "stop_line", "ends in `_`"),
]


class TestPackageNameCollisions:
    """No two exportable ids share a package, nor take the framework's (#1358)."""

    @staticmethod
    def _id_issues(scenario_id: str) -> list[Any]:
        report = validate_document(new_document(scenario_id))
        return [issue for issue in report.issues if issue.path == "id"]

    @pytest.mark.parametrize(("scenario_id", "other", "phrase"), _COLLIDING_IDS)
    def test_a_colliding_id_is_a_validation_warning(
        self, scenario_id: str, other: str, phrase: str
    ) -> None:
        # Without the refusal both would be packaged under the same name.
        assert (
            package_names(new_document(scenario_id))["package_name"]
            == package_names(new_document(other))["package_name"]
        )
        assert scenario_id_problem(other) is None
        issues = self._id_issues(scenario_id)
        assert [issue.severity for issue in issues] == ["warning"]
        assert phrase in issues[0].message
        assert f"`{other}_scenario`" in issues[0].message
        assert EXPORT_REFUSAL_NOTE in issues[0].message

    @pytest.mark.parametrize("scenario_id", ["stop__line", "stop_line_"])
    def test_a_refused_id_still_validates_and_compiles(self, scenario_id: str) -> None:
        """A package exported before #1358 runs ``compile_document`` on every
        run and in its own tests, so the id must not become an error."""
        document = new_document(scenario_id)
        assert validate_document(document).ok
        assert compile_document(document) is not None

    @pytest.mark.parametrize(("scenario_id", "_other", "phrase"), _COLLIDING_IDS)
    def test_a_colliding_id_is_not_exported(
        self, tmp_path: Path, scenario_id: str, _other: str, phrase: str
    ) -> None:
        with pytest.raises(PackageExportError, match=re.escape(phrase)) as caught:
            export_package(new_document(scenario_id), tmp_path, **OFFLINE)
        assert "Cannot export this scenario" in str(caught.value)
        assert EXPORT_REFUSAL_NOTE not in str(caught.value)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        "scenario_id", ["autoware_carla", "autoware__carla", "autoware_carla_"]
    )
    def test_the_frameworks_name_is_refused(
        self, tmp_path: Path, scenario_id: str
    ) -> None:
        """An id error, as before #1358 (#1695), and the only issue the id gets."""
        problem = framework_name_problem(scenario_id)
        assert problem is not None
        assert "'autoware_carla_scenario'" in problem
        assert "framework's own" in problem
        issues = self._id_issues(scenario_id)
        assert [issue.severity for issue in issues] == ["error"]
        assert issues[0].message == problem
        with pytest.raises(PackageExportError, match="framework's own"):
            export_package(new_document(scenario_id), tmp_path, **OFFLINE)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(("scenario_id", "package_name"), _SCENARIO_IDS)
    def test_an_accepted_id_has_its_own_package(
        self, scenario_id: str, package_name: str
    ) -> None:
        assert scenario_id_problem(scenario_id) is None
        assert self._id_issues(scenario_id) == []
        assert package_names(new_document(scenario_id))["package_name"] == (
            package_name
        )

    def test_accepted_ids_map_to_distinct_distributions(self) -> None:
        ids = [scenario_id for scenario_id, _ in _SCENARIO_IDS]
        names = {package_names(new_document(i))["distribution_name"] for i in ids}
        assert len(names) == len(ids)


class TestScenarioIdAgreement:
    """Every place an export names its scenario uses ``document.id`` (#1072)."""

    @staticmethod
    def _export(tmp_path: Path, scenario_id: str) -> Path:
        return export_package(new_document(scenario_id), tmp_path, **OFFLINE).root

    @pytest.mark.parametrize(("scenario_id", "package_name"), _SCENARIO_IDS)
    def test_the_scenario_id_is_the_document_id(
        self, scenario_id: str, package_name: str
    ) -> None:
        names = package_names(new_document(scenario_id))
        assert names["scenario_id"] == scenario_id
        assert names["package_name"] == package_name
        assert names["distribution_name"] == package_name.replace("_", "-")

    @pytest.mark.parametrize(("scenario_id", "package_name"), _SCENARIO_IDS)
    def test_builder_conf_and_test_agree(
        self, tmp_path: Path, scenario_id: str, package_name: str
    ) -> None:
        package = self._export(tmp_path, scenario_id)
        module = package / "src" / package_name

        conf_files = list((module / "conf" / "scenario").glob("*.yaml"))
        assert [p.name for p in conf_files] == [f"{scenario_id}.yaml"]
        conf = yaml.safe_load(conf_files[0].read_text())
        assert conf["scenario"]["name"] == scenario_id

        init = (module / "__init__.py").read_text()
        assert f'register_scenario_builder("{scenario_id}", build_scenario)' in init

        test = (package / "tests" / "test_scenario.py").read_text()
        assert f'assert document.id == "{scenario_id}"' in test
        assert f'get_scenario_builder("{scenario_id}")' in test

        manifest = yaml.safe_load((package / "scenario/manifest.yaml").read_text())
        assert manifest["scenario"]["id"] == scenario_id

    @pytest.mark.parametrize("scenario_id", ["stop_line"])
    def test_the_exported_packages_own_tests_pass(
        self, tmp_path: Path, scenario_id: str
    ) -> None:
        """Run the generated tests against this environment's framework.

        The locked run (``uv sync`` from the package's own lock) needs the
        network and is covered by the ``slow`` end-to-end test; putting the
        package's ``src`` on the path checks the same assertions offline.
        """
        import subprocess

        package = self._export(tmp_path, scenario_id)
        env = {**os.environ, "PYTHONPATH": str(package / "src")}
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
                "tests",
            ],
            cwd=package,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


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
        stale = tmp_path / "my_cut_in_scenario_wheelhouse"
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

    def test_the_readme_names_the_client_and_its_prebuilt_package(
        self, package: Path
    ) -> None:
        """The client is typesafe_carla, whose wheel carries its CPython package.

        The README has to name the import a scenario uses -- never the official
        `carla` package -- and tell an offline installer that nothing has to be
        built on the target, so nobody installs `cc` or runs a build for nothing.
        """
        readme = (package / "README.md").read_text()
        assert "import typesafe_carla.carla as carla" in readme
        assert "prebuilt" in readme
        assert "carla-*.whl" not in readme
        assert "extra" not in readme

    def test_tested_pythons_match_the_frameworks_requires_python(self) -> None:
        """Three hand-written statements of one range; nothing else ties them.

        A generated package inherits the framework's `requires-python`, the
        wheelhouse is resolved for each of :data:`TESTED_PYTHONS` it admits, and
        CI's test matrix is what "tested" means. Widen one without the others
        and the wheelhouse silently covers fewer interpreters than the package
        claims to support, or more than CI tests.
        """
        from autoware_carla_scenario.authoring.framework_pin import (
            framework_source_root,
        )

        pyproject = framework_source_root() / "pyproject.toml"
        if not pyproject.is_file():
            pytest.skip("not a source checkout")
        from packaging.specifiers import SpecifierSet

        declared = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
        admitted = [
            f"3.{minor}"
            for minor in range(6, 30)
            if SpecifierSet(declared).contains(f"3.{minor}")
        ]
        assert admitted == list(TESTED_PYTHONS), declared

        ci = framework_source_root().parent / ".github" / "workflows" / "ci.yml"
        matrix = yaml.safe_load(ci.read_text())["jobs"]["test"]["strategy"]["matrix"]
        assert matrix["python-version"] == list(TESTED_PYTHONS)


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
                "    # via my-cut-in-scenario",
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
        assert "    # via my-cut-in-scenario" in rewritten
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


def _write_pyproject(root: Path, requires_python: str) -> None:
    """A minimal package at *root* declaring *requires_python*."""
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "nothing"\nrequires-python = "{requires_python}"\n',
        encoding="utf-8",
    )


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
        _write_pyproject(tmp_path, ">=3.10,<3.13")
        assert supported_pythons(tmp_path) == ["3.10", "3.11", "3.12"]

    def test_an_open_range_stops_at_what_ci_tests(self, tmp_path: Path) -> None:
        """`>=3.11` is not a list: nothing past the tested range is built for."""
        _write_pyproject(tmp_path, ">=3.11")
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
        _write_pyproject(tmp_path, ">=3.10,!=3.11.*,<3.13")

        built: list[str] = []
        filled: list[Path] = []

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            built.append(python)
            return parent / f"builder-{python}", ""

        def _download(venv: Path, _requirements: Path, wheels: Path) -> str:
            filled.append(venv)
            python = venv.name.removeprefix("builder-")
            (wheels / "shared-1.0-py3-none-any.whl").write_text("same")
            (wheels / f"only-{python}-1.0-py3-none-any.whl").write_text(python)
            return f"pass {python}\n"

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

        # The passes run side by side, so in no particular order.
        assert sorted(built) == ["3.10", "3.12"]
        assert wheelhouse.python_tags == ("3.10", "3.12")
        # Each pass resolves from its own environment; a shared builder venv
        # would mean the second pass resolved nothing new.
        assert len(set(filled)) == 2
        # Each into a directory of its own, gathered after: what the two
        # share once, and what only one of them resolved.
        assert wheelhouse.wheels == (
            "only-3.10-1.0-py3-none-any.whl",
            "only-3.12-1.0-py3-none-any.whl",
            "shared-1.0-py3-none-any.whl",
        )
        assert wheelhouse.log.index("pass 3.10") < wheelhouse.log.index("pass 3.12")
        assert "3.10, 3.12" in (wheelhouse.root / "README.md").read_text()

    def test_an_interpreter_named_twice_is_one_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two passes of one Python would race for its builder venv."""
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        built: list[str] = []

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            built.append(python)
            return parent / f"builder-{python}", ""

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
            pythons=("3.12", "3.10", "3.12"),
        )
        assert sorted(built) == ["3.10", "3.12"]
        assert wheelhouse.python_tags == ("3.12", "3.10")
        # The passes' own directories are gone once their wheels moved up.
        assert not list((tmp_path / "out").glob(".pass-*"))

    def test_a_failed_pass_names_its_environment_and_leaves_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")

        def _environment(parent: Path, python: str) -> tuple[Path, str]:
            return parent / f"builder-{python}", f"venv for {python}\n"

        def _download(venv: Path, *_args: Any) -> str:
            if venv.name.endswith("3.12"):
                raise module.WheelhouseError("no wheel", "pip said no\n")
            return "pass 3.10\n"

        monkeypatch.setattr(module, "_export_requirements", lambda _r: ("", ""))
        monkeypatch.setattr(module, "_build_project_wheel", lambda *_a: "")
        monkeypatch.setattr(module, "_builder_environment", _environment)
        monkeypatch.setattr(module, "_download_wheels", _download)

        with pytest.raises(module.WheelhouseError) as raised:
            build_wheelhouse(
                tmp_path,
                tmp_path / "out",
                distribution="nothing",
                version="0.1.0",
                run_command="scenario scenario=nothing",
                pythons=("3.10", "3.12"),
            )
        log = raised.value.log
        assert log.index("pass 3.10") < log.index("venv for 3.12")
        assert log.index("venv for 3.12") < log.index("pip said no")
        assert not (tmp_path / "out").exists()

    def test_an_explicit_interpreter_overrides_requires_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Building for one Python stays possible -- it is just not the default."""
        import autoware_carla_scenario.authoring.wheelhouse as module

        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        _write_pyproject(tmp_path, ">=3.10,!=3.11.*")

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
        assert "my_cut_in_scenario-" in names
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
        from concurrent.futures import ThreadPoolExecutor

        from autoware_carla_scenario.authoring.uv_tool import run_uv
        from autoware_carla_scenario.authoring.wheelhouse import venv_python

        result = exported
        assert result.wheelhouse is not None, result.log
        assert result.wheelhouse.python_tags, result.log
        wheelhouse = result.wheelhouse

        def install(interpreter: str) -> None:
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
                    str(wheelhouse.root),
                    wheelhouse.distribution,
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
                    "import my_cut_in_scenario as p;"
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

        # Each interpreter installs into a venv of its own: side by side.
        tags = wheelhouse.python_tags
        with ThreadPoolExecutor(max_workers=len(tags)) as pool:
            for done in [pool.submit(install, interpreter) for interpreter in tags]:
                done.result()


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
