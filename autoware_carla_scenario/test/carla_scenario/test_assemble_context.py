"""Tests for the build-context assembler shipped with the packing action.

`assemble-context.sh` decides which files reach the Docker build, and a file it
leaves out is not loud: the image either fails to build in CI (a license file
the build backend looks for) or builds without part of the package (a config
group that happens to share a name with a build directory). Both are pinned
here against a package scaffolded the way users create them.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from autoware_carla_scenario.scaffold.generator import main as scenario_new

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT / ".github" / "actions" / "pack-scenario-image" / "assemble-context.sh"
)
_FRAMEWORK = "autoware_carla_scenario"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="the assembler needs bash and tar",
)


def _has_tomllib_python() -> bool:
    """Whether the assembler can read TOML, as its own lookup decides."""
    for name in ("python3", "python3.14", "python3.13", "python3.12", "python3.11"):
        interpreter = shutil.which(name)
        if interpreter is not None and (
            subprocess.run(
                [interpreter, "-c", "import tomllib"], capture_output=True
            ).returncode
            == 0
        ):
            return True
    return False


@pytest.fixture(scope="module")
def context(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A context assembled from a package whose scenario is named `dist`.

    The package also carries a top-level `dist/`, which is build output and
    must stay out of the context.
    """
    generated = tmp_path_factory.mktemp("generated")
    assert scenario_new(["dist_pkg", "-s", "dist", "-o", str(generated)]) == 0
    package = generated / "dist_pkg"
    assert (package / "src" / "dist_pkg" / "conf" / "scenario" / "dist").is_dir()
    (package / "dist").mkdir()
    (package / "dist" / "dist_pkg-0.1.0-py3-none-any.whl").write_bytes(b"stale")
    # Named on tar's command line now, so pin that the excludes still apply.
    for ignored in (".git", ".venv"):
        (package / ignored).mkdir()
        (package / ignored / "marker").write_text("")
    (package / ".gitignore").write_text("dist/\n")

    out = tmp_path_factory.mktemp("context")
    subprocess.run(
        ["bash", str(_SCRIPT), "--scenario", str(package), "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_nested_dist_directory_is_kept(context: Path) -> None:
    group = context / "scenario" / "src" / "dist_pkg" / "conf" / "scenario" / "dist"
    assert (group / "default.yaml").is_file()


def test_top_level_dist_directory_is_dropped(context: Path) -> None:
    assert not (context / "scenario" / "dist").exists()


def test_top_level_vcs_and_venv_are_dropped_but_dotfiles_kept(context: Path) -> None:
    assert not (context / "scenario" / ".git").exists()
    assert not (context / "scenario" / ".venv").exists()
    assert (context / "scenario" / ".gitignore").is_file()


def test_member_carries_the_files_its_manifest_names(context: Path) -> None:
    member = context / "framework" / _FRAMEWORK
    assert (member / "README.md").is_file()
    assert (member / "LICENSE").read_bytes() == (
        _REPO_ROOT / _FRAMEWORK / "LICENSE"
    ).read_bytes()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv builds the context")
def test_framework_wheel_builds_with_its_license(context: Path, tmp_path: Path) -> None:
    """The Dockerfile's wheelhouse step, run against the assembled context."""
    wheels = tmp_path / "wheels"
    result = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--all-packages",
            "--out-dir",
            str(wheels),
            "framework",
        ],
        cwd=context,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    (wheel,) = wheels.glob(f"{_FRAMEWORK}-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names), names


@pytest.mark.skipif(
    not _has_tomllib_python(), reason="without tomllib the assembler guesses"
)
@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ('license-files = ["LICENSE*"]', "matches no file"),
        ('license-files = ["../LICENSE"]', "points outside the member"),
        ('readme = "README.md"', "does not exist"),
        ('license = { file = "COPYING" }', "does not exist"),
    ],
)
def test_missing_or_outside_member_file_fails_loudly(
    tmp_path: Path, declaration: str, message: str
) -> None:
    """A file the backend will look for must not be skipped quietly.

    Skipping it would only move the failure into the image build.
    """
    framework = tmp_path / "framework"
    (framework / "member" / "src" / "member").mkdir(parents=True)
    (framework / "member" / "src" / "member" / "__init__.py").write_text("")
    (framework / "LICENSE").write_text("license\n")
    (framework / "uv.lock").write_text("")
    (framework / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["member"]\n'
    )
    (framework / "member" / "pyproject.toml").write_text(
        f'[project]\nname = "member"\n{declaration}\n'
    )
    package = tmp_path / "package"
    package.mkdir()
    (package / "pyproject.toml").write_text('[project]\nname = "package"\n')

    result = subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            "--scenario",
            str(package),
            "--framework",
            str(framework),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert message in result.stderr
