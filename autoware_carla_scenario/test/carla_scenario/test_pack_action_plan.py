"""The packing action pushes an image only after its smoke test passed.

``action.yml`` builds twice: a build that is loaded and smoke-tested and never
pushes, then a build -- all cache hits in the same builder -- that pushes.
These tests pin that order and run the "Plan the build" step's shell under
the input combinations that pick which builds run and for which platforms
(#2055).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_ACTION = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "actions"
    / "pack-scenario-image"
    / "action.yml"
)


@pytest.fixture(scope="module")
def action() -> dict[str, Any]:
    return yaml.safe_load(_ACTION.read_text())


def _step(action: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in action["runs"]["steps"] if step.get("id") == step_id)


def _index(action: dict[str, Any], name: str) -> int:
    names = [step["name"] for step in action["runs"]["steps"]]
    return names.index(name)


def test_loaded_build_never_pushes(action: dict[str, Any]) -> None:
    build = _step(action, "build")
    assert build["with"]["push"] is False
    assert build["with"]["platforms"] == "${{ steps.plan.outputs.local-platforms }}"


def test_push_runs_after_the_smoke_test(action: dict[str, Any]) -> None:
    push = _step(action, "push")
    assert push["with"]["push"] == "${{ inputs.push }}"
    assert push["with"]["load"] is False
    assert push["with"]["platforms"] == "${{ inputs.platforms }}"
    assert _index(action, "Build image") < _index(action, "Smoke-test the image")
    assert _index(action, "Smoke-test the image") < _index(action, "Push image")
    assert action["outputs"]["digest"]["value"] == "${{ steps.push.outputs.digest }}"


def test_both_builds_build_the_same_image(action: dict[str, Any]) -> None:
    """Same context, Dockerfile, args and cache: the push is all cache hits."""
    build, push = _step(action, "build")["with"], _step(action, "push")["with"]
    differing = {"platforms", "push", "load"}
    assert set(build) == set(push)
    for key in set(build) - differing:
        assert build[key] == push[key], key


def _plan(
    tmp_path: Path,
    *,
    platforms: str = "",
    load: str = "true",
    push: str = "false",
    smoke_test: str = "true",
    runner_arch: str = "X64",
    action: dict[str, Any],
) -> dict[str, str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    output = tmp_path / "github_output"
    output.write_text("")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_SHA": "0123456789abcdef",
        "RUNNER_ARCH": runner_arch,
        "IMAGE": "ghcr.io/example/my-scenario",
        "TAGS_IN": "",
        "CACHE_SCOPE_IN": "",
        "PLATFORMS": platforms,
        "LOAD": load,
        "PUSH": push,
        "SMOKE_TEST": smoke_test,
    }
    result = subprocess.run(
        [bash, "-c", _step(action, "plan")["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs: dict[str, str] = {}
    lines = iter(output.read_text().splitlines())
    for line in lines:
        if "<<" in line:
            name, marker = line.split("<<", 1)
            value = []
            for inner in lines:
                if inner == marker:
                    break
                value.append(inner)
            outputs[name] = "\n".join(value)
        else:
            name, _, value_ = line.partition("=")
            outputs[name] = value_
    return outputs


def test_single_platform_push_builds_tests_then_pushes(
    tmp_path: Path, action: dict[str, Any]
) -> None:
    out = _plan(tmp_path, push="true", action=action)
    assert out["local-build"] == "true"
    assert out["local-platforms"] == ""
    assert out["smoke-platform"] == ""
    assert out["final-build"] == "true"


def test_no_push_is_a_single_build(tmp_path: Path, action: dict[str, Any]) -> None:
    out = _plan(tmp_path, action=action)
    assert out["local-build"] == "true"
    assert out["final-build"] == "false"


def test_push_without_load_or_test_skips_the_local_build(
    tmp_path: Path, action: dict[str, Any]
) -> None:
    out = _plan(tmp_path, push="true", load="false", smoke_test="false", action=action)
    assert out["local-build"] == "false"
    assert out["final-build"] == "true"


@pytest.mark.parametrize(
    ("runner_arch", "platforms", "expected"),
    [
        ("X64", "linux/arm64, linux/amd64", "linux/amd64"),
        ("ARM64", "linux/amd64,linux/arm64", "linux/arm64"),
        # The runner's own is not listed: the first, under QEMU.
        ("X64", "linux/arm64,linux/riscv64", "linux/arm64"),
    ],
)
def test_multi_platform_tests_one_platform_then_builds_all(
    tmp_path: Path,
    action: dict[str, Any],
    runner_arch: str,
    platforms: str,
    expected: str,
) -> None:
    out = _plan(
        tmp_path,
        platforms=platforms,
        push="true",
        runner_arch=runner_arch,
        action=action,
    )
    assert out["local-build"] == "true"
    assert out["local-platforms"] == expected
    assert out["smoke-platform"] == expected
    assert out["final-build"] == "true"


def test_multi_platform_without_load_builds_all_at_once(
    tmp_path: Path, action: dict[str, Any]
) -> None:
    platforms = "linux/amd64,linux/arm64"
    out = _plan(
        tmp_path, platforms=platforms, load="false", smoke_test="false", action=action
    )
    assert out["local-build"] == "true"
    assert out["local-platforms"] == platforms
    assert out["final-build"] == "false"


def test_multi_platform_test_without_push_still_builds_every_platform(
    tmp_path: Path, action: dict[str, Any]
) -> None:
    out = _plan(tmp_path, platforms="linux/amd64,linux/arm64", action=action)
    assert out["local-platforms"] == "linux/amd64"
    assert out["final-build"] == "true"


def test_smoke_test_without_load_is_rejected(
    tmp_path: Path, action: dict[str, Any]
) -> None:
    with pytest.raises(AssertionError, match="requires load=true"):
        _plan(tmp_path, load="false", action=action)
