"""How a launcher runs the ``scenario_bridge`` node next to the stack it starts.

The node (:mod:`autoware_carla_scenario.autoware_bridge.ros_bridge`) is
Autoware's end of the ``AutowareBridge``: it pulls the mission and drives
Autoware's startup through the AD API.  It runs with Autoware's ROS 2
environment and Python, not this package's, so what it is handed is the
directory of :mod:`autoware_carla_scenario.autoware_bridge` -- which needs only
``grpcio`` and ``protobuf`` -- placed as ``autoware_carla_scenario/autoware_bridge``
under a directory on ``PYTHONPATH``.  Without an ``__init__.py`` beside it,
``autoware_carla_scenario`` is a namespace package there, so nothing else of the
framework is imported.
"""

from __future__ import annotations

import hashlib
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "BRIDGE_APT_PACKAGES",
    "BRIDGE_MODULE",
    "bridge_command",
    "bridge_package_dir",
    "bridge_ros_args",
    "host_python_root",
]

#: What ``python3 -m`` runs.
BRIDGE_MODULE: str = "autoware_carla_scenario.autoware_bridge.ros_bridge"

#: What the node needs beyond ROS 2 and Autoware's message packages, as the
#: Ubuntu packages a ROS 2 distribution's Python takes them from.
BRIDGE_APT_PACKAGES: tuple[str, ...] = ("python3-grpcio", "python3-protobuf")


def bridge_package_dir() -> Path:
    """The installed :mod:`autoware_carla_scenario.autoware_bridge` directory."""
    from .. import autoware_bridge  # noqa: PLC0415 - only its location is wanted

    return Path(autoware_bridge.__file__).resolve().parent


def bridge_ros_args(bridge_address: str, parameters: Mapping[str, Any]) -> list[str]:
    """The node's ``--ros-args``: the bridge's address, then *parameters*."""
    args = ["--ros-args", "-p", f"bridge_address:={bridge_address}"]
    for name, value in parameters.items():
        if isinstance(value, bool):
            value = str(value).lower()
        args += ["-p", f"{name}:={value}"]
    return args


def bridge_command(
    python_root: str, bridge_address: str, parameters: Mapping[str, Any]
) -> str:
    """A shell command running the node, with *python_root* put on ``PYTHONPATH``."""
    root = shlex.quote(python_root)
    argv = [
        "python3",
        "-m",
        BRIDGE_MODULE,
        *bridge_ros_args(bridge_address, parameters),
    ]
    return f'PYTHONPATH={root}"${{PYTHONPATH:+:$PYTHONPATH}}" {shlex.join(argv)}'


def host_python_root(cache_dir: Path | None = None) -> Path:
    """A directory to put on ``PYTHONPATH`` to run the node on this host.

    It holds ``autoware_carla_scenario/autoware_bridge`` as a link to the
    installed package, and nothing else, so another Python -- the ROS 2
    distribution's -- imports only that.  Kept under the user cache, one per
    installed location.
    """
    package = bridge_package_dir()
    base = (
        cache_dir
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "autoware_carla_scenario"
        / "ros_bridge"
    )
    root = base / hashlib.sha256(str(package).encode()).hexdigest()[:16]
    link = root / "autoware_carla_scenario" / "autoware_bridge"
    if not link.is_symlink() or link.resolve() != package:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(package, target_is_directory=True)
    return root
