"""What a run ran with: the environment and the lock beside its result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoware_carla_scenario import environment
from autoware_carla_scenario.environment import RunEnvironment, _pin, capture, write


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
    assert any(line.startswith("openodd-py==") for line in pins)
    doc = json.loads((tmp_path / "Demo_environment.json").read_text())
    assert doc["schema"] == "autoware_carla_scenario.environment/1"
    assert doc["id"] == summary["id"]
    assert {d["name"] for d in doc["distributions"]} >= {
        "autoware-carla-scenario",
        "openodd-py",
    }


def test_how_each_source_is_pinned() -> None:
    assert _pin("a", "1.0", {"kind": "index"}) == "a==1.0"
    assert _pin(
        "a",
        "1.0",
        {"kind": "vcs", "vcs": "git", "url": "https://x/a.git", "commit": "abc"},
    ) == ("a @ git+https://x/a.git@abc")
    assert (
        _pin("a", "1.0", {"kind": "archive", "url": "https://x/a.whl"})
        == "a @ https://x/a.whl"
    )
    assert _pin(
        "a",
        "1.0",
        {"kind": "editable", "path": "/src/a", "commit": "abc", "dirty": True},
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
