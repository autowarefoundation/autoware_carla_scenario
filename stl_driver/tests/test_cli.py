# SPDX-License-Identifier: Apache-2.0
"""The command line refuses out-of-range numbers before it warms anything up.

A bad ``--steps`` used to surface as an ``IndexError`` from the metrics, and
only after the minute-long torch warm-up; these tests pin the argparse error
that replaces it, and that valid values still reach the code that uses them.
"""

from __future__ import annotations

from typing import Any

import pytest
from carla_driver_interface import server as driver_server

from stl_driver import cli, policy


class _FakePolicy:
    """Stands in for EgoPolicy: records construction and warm-up, compiles nothing."""

    instances: list[_FakePolicy] = []

    def __init__(self, cruise_speed_mps: float | None = None) -> None:
        self.cruise_speed_mps = cruise_speed_mps
        self.warmed_up = False
        self.name = "fake"
        self.rules: list[object] = []
        _FakePolicy.instances.append(self)

    def warm_up(self) -> None:
        self.warmed_up = True


@pytest.fixture
def fake_policy(monkeypatch: pytest.MonkeyPatch) -> type[_FakePolicy]:
    _FakePolicy.instances = []
    monkeypatch.setattr(policy, "EgoPolicy", _FakePolicy)
    return _FakePolicy


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run_server(policy: object, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(driver_server, "run_server", fake_run_server)
    return calls


def _refused(
    capsys: pytest.CaptureFixture[str], argv: list[str], fake_policy: type[_FakePolicy]
) -> str:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    assert fake_policy.instances == [], "the policy was built before the argument was checked"
    return capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "-120", "1.5", "ten", ""])
def test_demo_steps_must_be_positive_int(
    capsys: pytest.CaptureFixture[str], fake_policy: type[_FakePolicy], value: str
) -> None:
    err = _refused(capsys, ["demo", "--steps", value], fake_policy)
    assert "--steps" in err
    assert "positive integer" in err


@pytest.mark.parametrize("value", ["0", "-3", "2.0", "many"])
def test_serve_max_workers_must_be_positive_int(
    capsys: pytest.CaptureFixture[str],
    fake_policy: type[_FakePolicy],
    served: list[dict[str, Any]],
    value: str,
) -> None:
    err = _refused(capsys, ["serve", "--max-workers", value], fake_policy)
    assert "--max-workers" in err
    assert "positive integer" in err
    assert served == []


@pytest.mark.parametrize("value", ["0", "-1", "65536", "http"])
def test_serve_port_must_be_in_range(
    capsys: pytest.CaptureFixture[str],
    fake_policy: type[_FakePolicy],
    served: list[dict[str, Any]],
    value: str,
) -> None:
    err = _refused(capsys, ["serve", "--port", value], fake_policy)
    assert "--port" in err
    assert "port" in err
    assert served == []


@pytest.mark.parametrize("command", ["serve", "demo"])
@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "-inf", "fast"])
def test_cruise_speed_must_be_finite_and_positive(
    capsys: pytest.CaptureFixture[str],
    fake_policy: type[_FakePolicy],
    served: list[dict[str, Any]],
    command: str,
    value: str,
) -> None:
    err = _refused(capsys, [command, f"--cruise-speed={value}"], fake_policy)
    assert "--cruise-speed" in err
    assert served == []


def test_defaults_are_unchanged() -> None:
    parser = cli.build_parser()
    serve = parser.parse_args(["serve"])
    assert (serve.host, serve.port, serve.max_workers, serve.cruise_speed) == (
        "0.0.0.0",
        50051,
        8,
        None,
    )
    demo = parser.parse_args(["demo"])
    assert (demo.steps, demo.cruise_speed, demo.metrics_json) == (120, None, None)


def test_demo_accepts_valid_values() -> None:
    args = cli.build_parser().parse_args(["demo", "--steps", "1", "--cruise-speed", "4.5"])
    assert args.steps == 1
    assert args.cruise_speed == 4.5


def test_serve_passes_valid_values_through(
    fake_policy: type[_FakePolicy], served: list[dict[str, Any]]
) -> None:
    code = cli.main(
        [
            "serve",
            "--port",
            "65535",
            "--max-workers",
            "1",
            "--cruise-speed",
            "12",
            "--host",
            "127.0.0.1",
        ]
    )
    assert code == 0
    assert served == [{"port": 65535, "host": "127.0.0.1", "max_workers": 1}]
    (built,) = fake_policy.instances
    assert built.warmed_up
    assert built.cruise_speed_mps == 12.0
