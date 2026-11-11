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
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import parse_qsl, urlsplit

from .config import HilsConfig

__all__ = [
    "HostSetup",
    "HostSetupError",
    "LoopbackCamera",
    "VcanInterface",
    "host_setup",
    "playbook_dir",
    "run_host_setup",
]

#: The prefix of the ``can://`` interfaces that are virtual (``vcan0``).
VCAN_PREFIX = "vcan"
#: The longest v4l2loopback card label (its ``card_label`` buffer holds 32
#: bytes with the terminating NUL).
_LABEL_BYTES = 31
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
_INTERFACE = re.compile(r"[A-Za-z0-9_.-]{1,15}")
_VIDEO_DEVICE = re.compile(r"/dev/video([0-9]+)")
#: hilspy's spellings of a boolean option (``take_bool`` in its core).
_TRUE = ("", "1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class HostSetupError(Exception):
    """The setup cannot run (a device it cannot create, no ansible-runner, a
    hilspy without the playbooks)."""


@dataclass(frozen=True)
class VcanInterface:
    """A virtual CAN interface (``vcan.yml``'s ``vcan_interfaces`` item)."""

    name: str
    #: CAN FD: the interface gets the CAN FD MTU (72), else the classic one.
    fd: bool = False


@dataclass(frozen=True)
class LoopbackCamera:
    """A v4l2loopback device, ``/dev/video<number>``, labelled *label*."""

    number: int
    label: str


@dataclass(frozen=True)
class HostSetup:
    """What a configuration needs on the PC.

    Attributes:
        vcan: The virtual CAN interfaces, in the order the devices name them.
        cameras: The v4l2loopback devices, by number.
        untouched: ``(device, why)`` of the devices the setup leaves alone.
    """

    vcan: Tuple[VcanInterface, ...] = ()
    cameras: Tuple[LoopbackCamera, ...] = ()
    untouched: Tuple[Tuple[str, str], ...] = ()

    def playbooks(self, user: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """``(playbook, variables)`` to run, in order: hilspy's playbooks and
        the variables derived from the devices. *user* joins the ``video``
        group (by default the user who ran ``sudo``, else this one)."""
        runs: List[Tuple[str, Dict[str, Any]]] = []
        if self.vcan:
            runs.append(
                (
                    "vcan.yml",
                    {
                        "vcan_interfaces": [
                            {"name": i.name, "fd": i.fd} for i in self.vcan
                        ]
                    },
                )
            )
        if self.cameras:
            runs.append(
                (
                    "v4l2loopback.yml",
                    {
                        "v4l2loopback_numbers": [c.number for c in self.cameras],
                        "v4l2loopback_labels": [c.label for c in self.cameras],
                        "v4l2loopback_exclusive_caps": True,
                        "hilspy_user": user if user is not None else _user(),
                        # The wheel hilspy is installed from has no v4l2
                        # feature to build; the bridge only needs the devices.
                        "hilspy_build": False,
                    },
                )
            )
        return runs

    def describe(self) -> str:
        """What the setup does, one device a line."""
        lines = [f"  vcan {i.name}" + (" (CAN FD)" if i.fd else "") for i in self.vcan]
        lines += [
            f"  v4l2loopback /dev/video{c.number} ({c.label!r})" for c in self.cameras
        ]
        lines += [f"  left alone: {device}: {why}" for device, why in self.untouched]
        return "\n".join(lines)


def _user() -> str:
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _flag(query: str, key: str, where: str) -> bool:
    values = [v for k, v in parse_qsl(query, keep_blank_values=True) if k == key]
    if not values:
        return False
    value = values[-1].lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise HostSetupError(f"{where}: {key}={values[-1]!r} is not a boolean")


def _label(name: str) -> str:
    """*name* as a v4l2loopback card label: no quote or comma (they would end
    it in the module's options), at most 31 bytes."""
    label = re.sub(r'[",]', "_", name)
    while len(label.encode("utf-8")) > _LABEL_BYTES:
        label = label[:-1]
    return label


def host_setup(config: HilsConfig) -> HostSetup:
    """The virtual devices *config*'s ``devices`` need.

    Raises:
        HostSetupError: A device the playbooks cannot create: a ``v4l2://``
            path that is not ``/dev/videoN``, two cameras on one device, or a
            ``vcan`` name Linux does not take.
    """
    vcan: Dict[str, bool] = {}
    cameras: Dict[int, LoopbackCamera] = {}
    untouched: List[Tuple[str, str]] = []
    for device in config.devices:
        where = f"devices.{device.name}.url"
        url = urlsplit(device.url)
        if url.scheme == "can":
            name = url.netloc or url.path
            if not name.startswith(VCAN_PREFIX):
                untouched.append(
                    (device.name, f"{name} is not a vcan interface (a CAN adapter)")
                )
                continue
            if not _INTERFACE.fullmatch(name):
                raise HostSetupError(
                    f"{where}: {name!r} is not an interface name (at most 15 "
                    "letters, digits, '_', '.' or '-')"
                )
            vcan[name] = vcan.get(name, False) or _flag(url.query, "fd", where)
        elif url.scheme == "v4l2":
            match = _VIDEO_DEVICE.fullmatch(url.netloc + url.path)
            if match is None:
                raise HostSetupError(
                    f"{where}: v4l2loopback creates /dev/videoN devices, not "
                    f"{url.netloc + url.path!r}"
                )
            number = int(match.group(1))
            if number in cameras:
                raise HostSetupError(
                    f"{where}: /dev/video{number} is already "
                    f"{cameras[number].label!r}'s; give each camera its own device"
                )
            cameras[number] = LoopbackCamera(number, _label(device.name))
        else:
            untouched.append((device.name, f"{url.scheme}:// needs no virtual device"))
    return HostSetup(
        vcan=tuple(VcanInterface(name, fd) for name, fd in vcan.items()),
        cameras=tuple(cameras[n] for n in sorted(cameras)),
        untouched=tuple(untouched),
    )


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
