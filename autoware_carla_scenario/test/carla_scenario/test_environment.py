"""What a run ran with: the environment and the lock beside its result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import Requirement

from autoware_carla_scenario import environment
from autoware_carla_scenario.environment import (
    PinSource,
    RunEnvironment,
    _pin,
    _source,
    capture,
    write,
)


def test_the_framework_and_its_dependencies_are_found() -> None:
    env = capture()
    framework = env.by_name["autoware-carla-scenario"]
    assert framework["version"]
    assert "openodd-py" in env.by_name and env.summary()["packages"]["openodd-py"]
    assert len(env.id) == 64 and env.id == capture().id


def test_a_lock_pins_every_distribution(tmp_path: Path) -> None:
    summary = write(tmp_path, "Demo")
    assert summary is not None
    lock = (tmp_path / "Demo_requirements.lock").read_text().splitlines()
    assert lock[1] == f"# environment id: {summary['id']}"
    pins = [line for line in lock if line and not line.startswith("#")]
    assert len(pins) == len(capture().packages)
    # From an index (``openodd-py==``) or a git source (``openodd-py @ git+``).
    assert any(line.startswith(("openodd-py==", "openodd-py @ ")) for line in pins)
    doc = json.loads((tmp_path / "Demo_environment.json").read_text())
    assert doc["schema"] == "autoware_carla_scenario.environment/1"
    assert doc["id"] == summary["id"]
    assert {d["name"] for d in doc["distributions"]} >= {
        "autoware-carla-scenario",
        "openodd-py",
    }


def test_how_each_source_is_pinned() -> None:
    assert _pin("a", "1.0", PinSource(kind="index")) == "a==1.0"
    assert _pin(
        "a",
        "1.0",
        PinSource(kind="vcs", vcs="git", url="https://x/a.git", commit="abc"),
    ) == ("a @ git+https://x/a.git@abc")
    assert (
        _pin("a", "1.0", PinSource(kind="archive", url="https://x/a.whl"))
        == "a @ https://x/a.whl"
    )
    assert _pin(
        "a",
        "1.0",
        PinSource(kind="editable", path="/src/a", commit="abc", dirty=True),
    ) == ("a==1.0  # editable from /src/a, git abc with uncommitted changes")


class _Scenario:
    pass


def test_the_scenario_s_package_is_named(monkeypatch: Any) -> None:
    env = RunEnvironment(
        [
            {
                "name": "autoware-carla-scenario",
                "version": "1.0",
                "source": {"kind": "index"},
            },
            {
                "name": "town10-scenarios",
                "version": "0.3.0",
                "source": {"kind": "index"},
            },
        ]
    )
    _Scenario.__module__ = "town10_scenarios.cut_in"
    monkeypatch.setattr(
        environment.metadata,
        "packages_distributions",
        lambda: {"town10_scenarios": ["town10_scenarios"]},
    )
    assert env.summary(_Scenario())["scenario"]["name"] == "town10-scenarios"
    assert "# scenario: town10-scenarios==0.3.0" in env.lock(_Scenario())


def test_a_failure_is_not_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    assert write(blocked / "out", "Demo") is None


class _Dist:
    """A distribution that only has a ``direct_url.json``."""

    def __init__(self, direct_url: dict[str, Any]) -> None:
        self._raw = json.dumps(direct_url)

    def read_text(self, name: str) -> str | None:
        return self._raw if name == "direct_url.json" else None


REPOSITORY = "https://github.com/autowarefoundation/autoware_carla_scenario.git"
COMMIT = "9e15ebae" + "0" * 32


def _pin_of(direct_url: dict[str, Any]) -> tuple[PinSource, str]:
    source = _source(_Dist(direct_url))  # type: ignore[arg-type]
    pin = _pin("autoware-carla-scenario", "2.63.0", source)
    # Whatever the lock writes has to be a requirement pip and uv accept (a
    # requirements file drops a comment after whitespace).
    assert Requirement(pin.split("  #")[0]).name == "autoware-carla-scenario"
    return source, pin


def test_a_git_pin_keeps_its_subdirectory() -> None:
    # Issue #682: the framework lives in a subdirectory of its repository, and
    # a lock line without it names the workspace root, which has no project.
    source, pin = _pin_of(
        {
            "url": REPOSITORY,
            "vcs_info": {
                "vcs": "git",
                "commit_id": COMMIT,
                "requested_revision": "master",
            },
            "subdirectory": "autoware_carla_scenario",
        }
    )
    assert source.subdirectory == "autoware_carla_scenario"
    assert source.requested == "master"
    assert pin == (
        f"autoware-carla-scenario @ git+{REPOSITORY}@{COMMIT}"
        "#subdirectory=autoware_carla_scenario"
    )


def test_a_git_pin_without_a_commit_falls_back_to_the_requested_revision() -> None:
    _, pin = _pin_of(
        {"url": REPOSITORY, "vcs_info": {"vcs": "git", "requested_revision": "v1"}}
    )
    # A branch or tag can move, so the lock says the pin is not a commit.
    assert pin == (
        f"autoware-carla-scenario @ git+{REPOSITORY}@v1"
        "  # requested v1, no commit recorded: may move"
    )


@pytest.mark.parametrize(
    ("archive_info", "fragment"),
    [
        ({}, "#subdirectory=pkg"),
        (
            {"hashes": {"sha256": "ab" * 32, "md5": "cd" * 16}},
            f"#sha256={'ab' * 32}&subdirectory=pkg",
        ),
        ({"hash": "sha256=" + "ef" * 32}, f"#sha256={'ef' * 32}&subdirectory=pkg"),
    ],
    ids=["no-hash", "hashes", "legacy-hash"],
)
def test_an_archive_pin_keeps_its_hash_and_subdirectory(
    archive_info: dict[str, Any], fragment: str
) -> None:
    source, pin = _pin_of(
        {
            "url": "https://x/repo.tar.gz",
            "archive_info": archive_info,
            "subdirectory": "pkg",
        }
    )
    assert source.subdirectory == "pkg"
    assert pin == f"autoware-carla-scenario @ https://x/repo.tar.gz{fragment}"


def test_a_local_directory_names_its_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    source = _source(
        _Dist(  # type: ignore[arg-type]
            {
                "url": tmp_path.as_uri(),
                "dir_info": {"editable": True},
                "subdirectory": "pkg",
            }
        )
    )
    assert source.kind == "editable"
    assert source.path == str(tmp_path / "pkg")
