"""The virtual devices a HILS configuration's devices need: what
``scenario-hils --setup`` (:mod:`.host_setup`) creates.

A ``can://vcan0`` device needs the ``vcan0`` interface, a
``v4l2:///dev/video10`` camera the v4l2loopback device ``/dev/video10``.
:func:`host_devices` derives them from each device's name and transport URL,
validating every device in order, and :class:`HostSetup` holds them, with the
devices it leaves alone (a physical CAN adapter, the network, a serial
port)::

    vehicle_can  can://vcan0?fd          -> vcan0, CAN FD
    front_camera v4l2:///dev/video10     -> /dev/video10, labelled "front_camera"

Free of heavy imports (no yaml, no ansible-runner), so the library check
compiles it (docs/typecheck.md); :mod:`.host_setup` reads the devices off a
configuration and runs hilspy's playbooks.
"""

from __future__ import annotations

import getpass
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

__all__ = [
    "HostSetup",
    "HostSetupError",
    "LoopbackCamera",
    "VCAN_PREFIX",
    "VcanInterface",
    "host_devices",
]

#: The prefix of the ``can://`` interfaces that are virtual (``vcan0``).
VCAN_PREFIX = "vcan"
#: The longest v4l2loopback card label (its ``card_label`` buffer holds 32
#: bytes with the terminating NUL).
_LABEL_BYTES = 31
_INTERFACE = re.compile(r"[A-Za-z0-9_.-]{1,15}")
_VIDEO_DEVICE = re.compile(r"/dev/video([0-9]+)")
#: hilspy's spellings of a boolean option (``take_bool`` in its core).
_TRUE = ("", "1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


class HostSetupError(Exception):
    """The setup cannot run (a device it cannot create, no ansible-runner, a
    hilspy without the playbooks)."""

    # Codon 0.19 raises no subclass of a built-in exception without a
    # constructor of its own (docs/typecheck.md). This one hands on what it
    # is given, so Python builds it as Exception's would: the same args,
    # message and repr for any arguments, and the same TypeError for a
    # keyword.
    def __init__(self, *args: str, **kwargs: str) -> None:
        super().__init__(*args, **kwargs)


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


def host_devices(devices: Iterable[Tuple[str, str]]) -> HostSetup:
    """The virtual devices *devices*, ``(name, url)`` in a configuration's
    order, need.  Each is validated in turn: the first a playbook cannot
    create raises, and the devices after it are not read.

    Raises:
        HostSetupError: A device the playbooks cannot create: a ``v4l2://``
            path that is not ``/dev/videoN``, two cameras on one device, a
            ``vcan`` name Linux does not take, or an ``fd`` flag that is not
            a boolean (on every device of an interface).
    """
    vcan: Dict[str, bool] = {}
    cameras: Dict[int, LoopbackCamera] = {}
    untouched: List[Tuple[str, str]] = []
    for device, address in devices:
        where = f"devices.{device}.url"
        url = urlsplit(address)
        if url.scheme == "can":
            name = url.netloc or url.path
            if not name.startswith(VCAN_PREFIX):
                untouched.append(
                    (device, f"{name} is not a vcan interface (a CAN adapter)")
                )
                continue
            if not _INTERFACE.fullmatch(name):
                raise HostSetupError(
                    f"{where}: {name!r} is not an interface name (at most 15 "
                    "letters, digits, '_', '.' or '-')"
                )
            # Read the flag first: every device's ``fd`` is validated, even
            # when an earlier one on the interface already made it FD.
            fd = _flag(url.query, "fd", where)
            vcan[name] = vcan.get(name, False) or fd
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
            cameras[number] = LoopbackCamera(number, _label(device))
        else:
            untouched.append((device, f"{url.scheme}:// needs no virtual device"))
    return HostSetup(
        vcan=tuple(VcanInterface(name, fd) for name, fd in vcan.items()),
        cameras=tuple(cameras[n] for n in sorted(cameras)),
        untouched=tuple(untouched),
    )
