"""``scenario-hils --setup``: the playbook variables derived from a
configuration's devices, and the playbooks run with them (ansible-runner
replaced by a recorder; the playbooks themselves are hilspy's, tested there)."""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from autoware_carla_scenario.hils import parse_config
from autoware_carla_scenario.hils import host_setup as host_setup_module
from autoware_carla_scenario.hils.cli import example_config, main
from autoware_carla_scenario.hils.host_setup import (
    HostSetupError,
    LoopbackCamera,
    VcanInterface,
    host_setup,
    run_host_setup,
)


def _config(**devices: str) -> Any:
    return parse_config(
        {"devices": {name: {"url": url} for name, url in devices.items()}}
    )


def test_the_example_needs_vcan0_and_one_camera(tmp_path: Path) -> None:
    import yaml

    setup = host_setup(parse_config(yaml.safe_load(example_config()), tmp_path))
    assert setup.vcan == (VcanInterface("vcan0", fd=False),)
    assert setup.cameras == (LoopbackCamera(10, "front_camera"),)
    assert dict(setup.untouched).keys() == {"gnss", "lidar_top"}


def test_the_playbook_variables() -> None:
    setup = host_setup(
        _config(
            body_can="can://vcan0?fd",
            chassis_can="can://vcan1",
            rear="v4l2:///dev/video12?width=640",
            front="v4l2:///dev/video10",
            lidar="udp://192.168.1.201:2368",
        )
    )
    assert setup.playbooks(user="hils") == [
        (
            "vcan.yml",
            {
                "vcan_interfaces": [
                    {"name": "vcan0", "fd": True},
                    {"name": "vcan1", "fd": False},
                ]
            },
        ),
        (
            "v4l2loopback.yml",
            {
                # By device number, each labelled with its device's name.
                "v4l2loopback_numbers": [10, 12],
                "v4l2loopback_labels": ["front", "rear"],
                "v4l2loopback_exclusive_caps": True,
                "hilspy_user": "hils",
                "hilspy_build": False,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("query", "fd"),
    [
        ("fd", True),
        ("fd=1", True),
        ("fd=true", True),
        ("fd=off", False),
        ("echo", False),
    ],
)
def test_can_fd_is_read_as_hilspy_reads_it(query: str, fd: bool) -> None:
    setup = host_setup(_config(bus=f"can://vcan0?{query}"))
    assert setup.vcan == (VcanInterface("vcan0", fd=fd),)


def test_an_interface_shared_by_devices_is_fd_if_one_of_them_is() -> None:
    setup = host_setup(_config(a="can://vcan0", b="can://vcan0?fd", c="can://vcan0"))
    assert setup.vcan == (VcanInterface("vcan0", fd=True),)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("can://vcan0?fd=1", "can://vcan0?fd=maybe"),
        ("can://vcan0?fd=maybe", "can://vcan0?fd=1"),
    ],
)
def test_every_device_on_a_shared_interface_has_its_fd_checked(
    first: str, second: str
) -> None:
    devices = {"a": first, "b": second}
    bad = next(name for name, url in devices.items() if url.endswith("maybe"))
    with pytest.raises(HostSetupError) as error:
        host_setup(_config(**devices))
    assert str(error.value) == f"devices.{bad}.url: fd='maybe' is not a boolean"


@pytest.mark.parametrize(
    ("first", "second", "fd"),
    [
        ("fd=1", "fd=0", True),
        ("fd=0", "fd=1", True),
        ("fd=true", "fd=off", True),
        ("fd=no", "fd", True),
        ("fd=0", "fd=false", False),
        ("echo", "fd=no", False),
    ],
)
def test_a_valid_pair_on_a_shared_interface(first: str, second: str, fd: bool) -> None:
    setup = host_setup(_config(a=f"can://vcan0?{first}", b=f"can://vcan0?{second}"))
    assert setup.vcan == (VcanInterface("vcan0", fd=fd),)


def test_physical_and_network_devices_are_left_alone() -> None:
    setup = host_setup(
        _config(
            adapter="can://can0",
            gnss="tcp://0.0.0.0:28784?listen",
            gps="serial:///dev/ttyUSB0?baud=115200",
        )
    )
    assert setup.playbooks() == []
    assert [device for device, _ in setup.untouched] == ["adapter", "gnss", "gps"]
    assert "can0" in dict(setup.untouched)["adapter"]


def test_a_label_is_cut_to_what_v4l2loopback_holds() -> None:
    name = 'a"long,camera' + "x" * 40
    (camera,) = host_setup(_config(**{name: "v4l2:///dev/video3"})).cameras
    assert camera.label.startswith("a_long_camera")
    assert len(camera.label.encode()) == 31


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ({"cam": "v4l2:///dev/camera"}, "/dev/videoN"),
        ({"a": "v4l2:///dev/video10", "b": "v4l2:///dev/video10"}, "already"),
        ({"bus": "can://vcan0?fd=maybe"}, "not a boolean"),
        ({"bus": "can://vcan_far_too_long0"}, "not an interface name"),
    ],
)
def test_devices_it_cannot_create(devices: Dict[str, str], message: str) -> None:
    with pytest.raises(HostSetupError, match=message):
        host_setup(_config(**devices))


class _Recorder:
    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.runs: List[Tuple[Path, str, Dict[str, Any]]] = []

    def __call__(
        self, playbook_dir: Path, playbook: str, variables: Dict[str, Any]
    ) -> int:
        self.runs.append((playbook_dir, playbook, variables))
        return self.status


@pytest.fixture
def playbooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hilspy that ships the playbooks (the installed one may predate them)."""
    for name in ("vcan.yml", "v4l2loopback.yml"):
        (tmp_path / name).write_text("[]\n")
    monkeypatch.setattr(host_setup_module, "playbook_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    return tmp_path


def test_run_runs_each_playbook_with_its_variables(playbooks: Path) -> None:
    recorder = _Recorder()
    out = io.StringIO()
    config = _config(bus="can://vcan0?fd", cam="v4l2:///dev/video10")
    assert run_host_setup(config, runner=recorder, out=out) == 0
    assert [(d, p) for d, p, _ in recorder.runs] == [
        (playbooks, "vcan.yml"),
        (playbooks, "v4l2loopback.yml"),
    ]
    assert "vcan vcan0 (CAN FD)" in out.getvalue()
    assert "v4l2loopback /dev/video10 ('cam')" in out.getvalue()


def test_run_stops_at_a_failed_playbook(playbooks: Path) -> None:
    recorder = _Recorder(status=2)
    config = _config(bus="can://vcan0", cam="v4l2:///dev/video10")
    assert run_host_setup(config, runner=recorder, out=io.StringIO()) == 2
    assert [p for _, p, _ in recorder.runs] == ["vcan.yml"]


def test_run_with_nothing_to_do_needs_no_playbook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        host_setup_module,
        "playbook_dir",
        lambda: pytest.fail("looked for the playbooks"),
    )
    out = io.StringIO()
    assert run_host_setup(_config(gnss="tcp://0.0.0.0:1"), out=out) == 0
    assert "nothing to set up" in out.getvalue()


def test_the_ansible_runner_call(
    playbooks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: List[Dict[str, Any]] = []

    def run(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(rc=0)

    fake = types.ModuleType("ansible_runner")
    fake.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ansible_runner", fake)
    monkeypatch.setattr(host_setup_module, "_become_password", lambda: "secret")
    runner = host_setup_module.AnsibleRunner(check=True)
    assert runner(playbooks, "vcan.yml", {"vcan_interfaces": []}) == 0
    assert runner(playbooks, "v4l2loopback.yml", {}) == 0
    (first, second) = calls
    assert first["project_dir"] == str(playbooks) and first["playbook"] == "vcan.yml"
    assert first["cmdline"] == "--ask-become-pass --check --diff"
    assert list(first["passwords"].values()) == ["secret"]
    host = first["inventory"]["all"]["hosts"]["localhost"]
    assert host["ansible_python_interpreter"] == "/usr/bin/python3"
    assert second["playbook"] == "v4l2loopback.yml"


def test_without_ansible_runner_it_says_which_extra(
    playbooks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "ansible_runner", None)
    with pytest.raises(HostSetupError, match="hils-setup"):
        host_setup_module.AnsibleRunner()(playbooks, "vcan.yml", {})


def test_a_hilspy_without_playbooks_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Files:
        def joinpath(self, *parts: str) -> Any:
            return tmp_path.joinpath(*parts)

    monkeypatch.setattr(host_setup_module.resources, "files", lambda _: _Files())
    with pytest.raises(HostSetupError, match="hilspy 0.17"):
        host_setup_module.playbook_dir()
    (tmp_path / "ansible").mkdir()
    for name in ("vcan.yml", "v4l2loopback.yml"):
        (tmp_path / "ansible" / name).write_text("[]\n")
    assert host_setup_module.playbook_dir() == tmp_path / "ansible"


# -- the command ----------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "hils.yaml"
    path.write_text(text)
    return path


def test_setup_vars_prints_the_variables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path,
        "devices:\n  bus: {url: 'can://vcan0?fd'}\n  cam: {url: 'v4l2:///dev/video10'}\n",
    )
    assert main([str(path), "--setup-vars"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["vcan.yml"] == {"vcan_interfaces": [{"name": "vcan0", "fd": True}]}
    assert printed["v4l2loopback.yml"]["v4l2loopback_labels"] == ["cam"]


def test_setup_runs_the_playbooks_in_check_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: List[Tuple[Any, bool]] = []

    def run(config: Any, check: bool) -> int:
        seen.append((config, check))
        return 0

    monkeypatch.setattr(host_setup_module, "run_host_setup", run)
    path = _write(tmp_path, "devices:\n  bus: {url: 'can://vcan0'}\n")
    assert main([str(path), "--setup", "--check"]) == 0
    assert main([str(path), "--setup"]) == 0
    assert [check for _, check in seen] == [True, False]


def test_setup_reports_a_device_it_cannot_create(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "devices:\n  cam: {url: 'v4l2:///dev/camera'}\n")
    assert main([str(path), "--setup-vars"]) == 1
    assert "/dev/videoN" in capsys.readouterr().err
