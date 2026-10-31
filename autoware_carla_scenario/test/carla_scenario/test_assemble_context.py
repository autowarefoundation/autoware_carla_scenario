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
