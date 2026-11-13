"""``scenario-hils --setup``: prepare the HILS PC for the devices a
configuration names.

The virtual devices the bridge writes to have to exist before it starts: a
``can://vcan0`` device needs the ``vcan0`` interface, a ``v4l2:///dev/video10``
camera the v4l2loopback device ``/dev/video10``. hilspy ships ansible
playbooks that create them (``hilspy/ansible/vcan.yml`` and
``v4l2loopback.yml``, hilspy's README, "Host setup playbooks"); this module
derives their variables from the configuration's ``devices``
(:func:`host_setup`) and runs them on this PC through ansible-runner
(:func:`run_host_setup`)::

    devices:
      vehicle_can: {url: "can://vcan0?fd"}          -> vcan.yml: vcan0, CAN FD MTU
      front_camera: {url: "v4l2:///dev/video10"}    -> v4l2loopback.yml: 10, "front_camera"

Only virtual devices are set up: a ``can://`` interface whose name starts with
``vcan`` and every ``v4l2://`` device. A physical CAN adapter (``can0``), the
network the UDP and TCP devices use and a serial port are the PC's own; they
are listed as left alone.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .config import HilsConfig

# What the devices need is the checked hils.host_devices's: the same objects
# (docs/typecheck.md).
from .host_devices import _FALSE as _FALSE
from .host_devices import _INTERFACE as _INTERFACE
from .host_devices import _LABEL_BYTES as _LABEL_BYTES
from .host_devices import _TRUE as _TRUE
from .host_devices import _VIDEO_DEVICE as _VIDEO_DEVICE
from .host_devices import VCAN_PREFIX as VCAN_PREFIX
from .host_devices import HostSetup as HostSetup
from .host_devices import HostSetupError as HostSetupError
from .host_devices import LoopbackCamera as LoopbackCamera
from .host_devices import VcanInterface as VcanInterface
from .host_devices import _flag as _flag
from .host_devices import _label as _label
from .host_devices import _user as _user
from .host_devices import host_devices

__all__ = [
    "HostSetup",
    "HostSetupError",
    "LoopbackCamera",
    "VcanInterface",
    "host_setup",
    "playbook_dir",
    "run_host_setup",
]

#: The playbooks' inventory: this PC, through the system Python (it has the apt
#: bindings the playbooks' ``apt`` tasks need; a virtual environment does not).
_LOCALHOST: Dict[str, Any] = {
    "all": {
        "hosts": {
            "localhost": {
                "ansible_connection": "local",
                "ansible_python_interpreter": "/usr/bin/python3",
            }
        }
    }
}


def host_setup(config: HilsConfig) -> HostSetup:
    """The virtual devices *config*'s ``devices`` need
    (:func:`~.host_devices.host_devices` of each device's name and URL, read
    in turn).

    Raises:
        HostSetupError: A device the playbooks cannot create: a ``v4l2://``
            path that is not ``/dev/videoN``, two cameras on one device, a
            ``vcan`` name Linux does not take, or an ``fd`` flag that is not
            a boolean.
    """
    return host_devices((device.name, device.url) for device in config.devices)


def playbook_dir() -> Path:
    """The directory of hilspy's playbooks (in the installed hilspy).

    Raises:
        HostSetupError: hilspy is not installed, or is older than the release
            that ships its playbooks.
    """
    try:
        root = resources.files("hilspy").joinpath("ansible")
    except ModuleNotFoundError as exc:
        raise HostSetupError(
            "the setup runs hilspy's playbooks: install the `hils-setup` extra "
            "(uvx --from 'autoware-carla-scenario[hils-setup]' scenario-hils --setup ...)"
        ) from exc
    if not (
        root.joinpath("vcan.yml").is_file()
        and root.joinpath("v4l2loopback.yml").is_file()
    ):
        raise HostSetupError(
            "this hilspy ships no host setup playbooks (hilspy/ansible/vcan.yml); "
            "they come with hilspy 0.17"
        )
    return Path(str(root))


class Runner(Protocol):
    """Runs one playbook with its variables; its exit status."""

    def __call__(
        self, playbook_dir: Path, playbook: str, variables: Dict[str, Any]
    ) -> int: ...


def _become_password() -> Optional[str]:
    """sudo's password, or ``None`` when root or sudo needs none."""
    if os.geteuid() == 0:
        return None
    try:
        free = (
            subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
        )
    except FileNotFoundError as exc:
        raise HostSetupError("the setup needs root or sudo") from exc
    return None if free else getpass.getpass("sudo password: ")


class AnsibleRunner:
    """Runs hilspy's playbooks on this PC through ansible-runner, asking for
    sudo's password at most once."""

    def __init__(self, check: bool = False) -> None:
        self.check = check
        self._asked = False
        self._password: Optional[str] = None

    def __call__(
        self, playbook_dir: Path, playbook: str, variables: Dict[str, Any]
    ) -> int:
        try:
            import ansible_runner  # noqa: PLC0415
        except ImportError as exc:
            raise HostSetupError(
                "the setup runs ansible: install the `hils-setup` extra "
                "(uvx --from 'autoware-carla-scenario[hils-setup]' scenario-hils --setup ...)"
            ) from exc
        if not self._asked:
            self._password, self._asked = _become_password(), True
        cmdline: List[str] = []
        passwords: Dict[str, str] = {}
        if self._password is not None:
            cmdline.append("--ask-become-pass")
            passwords[r"^BECOME password.*:\s*?$"] = self._password
        if self.check:
            cmdline += ["--check", "--diff"]
        with tempfile.TemporaryDirectory(prefix="scenario-hils-setup-") as data:
            result = ansible_runner.run(
                private_data_dir=data,
                project_dir=str(playbook_dir),
                playbook=playbook,
                inventory=_LOCALHOST,
                extravars=variables,
                cmdline=" ".join(cmdline) or None,
                passwords=passwords,
                quiet=False,
            )
        return int(result.rc)


def run_host_setup(
    config: HilsConfig,
    *,
    check: bool = False,
    runner: Optional[Runner] = None,
    out: Any = None,
) -> int:
    """Set this PC up for *config*'s virtual devices; the exit status.

    Prints what it derived, then runs each of hilspy's playbooks that has
    something to do (*check*: ansible's ``--check --diff``, which changes
    nothing). *runner* runs a playbook (by default :class:`AnsibleRunner`).

    Raises:
        HostSetupError: See :func:`host_setup`, :func:`playbook_dir` and
            :class:`AnsibleRunner`.
    """
    out = sys.stdout if out is None else out
    if sys.platform != "linux":
        raise HostSetupError("the setup (vcan, v4l2loopback) is Linux-only")
    setup = host_setup(config)
    runs = setup.playbooks()
    print(setup.describe() or "  (no devices)", file=out)
    if not runs:
        print("nothing to set up: no vcan or v4l2 device", file=out)
        return 0
    directory = playbook_dir()
    run = runner if runner is not None else AnsibleRunner(check=check)
    for playbook, variables in runs:
        print(f"{playbook}: {json.dumps(variables)}", file=out)
        status = run(directory, playbook, variables)
        if status != 0:
            return status
    if setup.cameras and not check:
        print(
            "v4l2:// needs hilspy built with its `v4l2` feature (hilspy's README, "
            '"Cameras"); log out and in (or `newgrp video`) if you were just '
            "added to `video`",
            file=out,
        )
    return 0
