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
import math
import os
import random
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "BRIDGE_APT_PACKAGES",
    "BRIDGE_MODULE",
    "MISSION_POLL_TIMEOUT_S",
    "ROUTE_WATCH_PERIOD_S",
    "bridge_command",
    "bridge_package_dir",
    "bridge_ros_args",
    "host_python_root",
    "launcher_bridge_parameters",
    "liveness_timeout_problem",
    "min_liveness_timeout_s",
]

#: What ``python3 -m`` runs.
BRIDGE_MODULE: str = "autoware_carla_scenario.autoware_bridge.ros_bridge"

#: What the node needs beyond ROS 2 and Autoware's message packages, as the
#: Ubuntu packages a ROS 2 distribution's Python takes them from.
BRIDGE_APT_PACKAGES: tuple[str, ...] = ("python3-grpcio", "python3-protobuf")


#: The node's ``route_watch_period_s`` when it is not given: how often, in
#: wall-clock seconds, it asks for the mission once Autoware is ready.
ROUTE_WATCH_PERIOD_S: float = 1.0

#: The node's ``mission_poll_timeout_s`` when it is not given: how long one
#: ``GetMission`` may take.
MISSION_POLL_TIMEOUT_S: float = 1.0


def bridge_package_dir() -> Path:
    """The installed :mod:`autoware_carla_scenario.autoware_bridge` directory."""
    from ..autoware_bridge import __file__ as bridge_file  # noqa: PLC0415 - only its location is wanted

    return Path(bridge_file).resolve().parent


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
    # Concatenated: Codon 0.19 cannot parse `{{` in an f-string.
    return "PYTHONPATH=" + root + '"${PYTHONPATH:+:$PYTHONPATH}" ' + shlex.join(argv)


def host_python_root(cache_dir: Path | None = None) -> Path:
    """A directory to put on ``PYTHONPATH`` to run the node on this host.

    It holds ``autoware_carla_scenario/autoware_bridge`` as a link to the
    installed package, and nothing else, so another Python -- the ROS 2
    distribution's -- imports only that.  Kept under the user cache, one per
    installed location.

    Safe for processes that make it at once (runs started together from one
    install, #2079): the link is made under a name of this process's own and
    renamed onto ``autoware_bridge``, which replaces whatever is there in one
    step, so no process sees it missing or finds it already made.
    """
    package = bridge_package_dir()
    if cache_dir is not None:
        base = cache_dir
    else:
        default_cache = Path.home() / ".cache"
        cache_home = os.environ.get("XDG_CACHE_HOME")
        cache = Path(cache_home) if cache_home is not None else default_cache
        base = cache / "autoware_carla_scenario" / "ros_bridge"
    root = base / hashlib.sha256(str(package).encode()).hexdigest()[:16]
    link = root / "autoware_carla_scenario" / "autoware_bridge"
    if not link.is_symlink() or link.resolve() != package:
        link.parent.mkdir(parents=True, exist_ok=True)
        # Unique to this process: the pid, and random bits for a pid reused
        # on another host sharing the cache (an NFS home).  A Random of its
        # own, seeded afresh: the module's would advance a stream a scenario
        # may have seeded, and repeat the same bits wherever it was seeded alike.
        unique = f"{os.getpid()}.{random.Random().getrandbits(32)}"
        tmp = link.with_name(f".{link.name}.{unique}.tmp")
        tmp.symlink_to(package, target_is_directory=True)
        try:
            os.replace(tmp, link)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    return root


def min_liveness_timeout_s(
    route_watch_period_s: float, mission_poll_timeout_s: float
) -> float:
    """The smallest ``liveness_timeout_s`` that cannot fail a healthy node.

    Once ready, the node asks for the mission every *route_watch_period_s* of
    wall-clock time, and each ask may take up to *mission_poll_timeout_s*, so
    a healthy node's asks reach the scenario at most their sum apart.  Twice
    that, so that one lost ask -- a poll that timed out, an executor thread
    that was late -- is not taken for a dead stack.  ``0.0`` when the node
    does not ask once ready (*route_watch_period_s* of 0 or less), which the
    scenario does not watch.
    """
    if route_watch_period_s <= 0.0:
        return 0.0
    return 2.0 * (route_watch_period_s + max(mission_poll_timeout_s, 0.0))


def _seconds_parameter(
    parameters: Mapping[str, Any], name: str, default: float
) -> float:
    """*parameters*' *name* as seconds; *default* when it is not set.

    A value that is not a number (one the node would refuse itself) reads as
    the default: the node's own error says what is wrong with it.
    """
    value = parameters.get(name)
    if value is None or isinstance(value, bool):
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    return seconds if math.isfinite(seconds) else default


def liveness_timeout_problem(
    liveness_timeout_s: float, parameters: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Why *liveness_timeout_s* would fail a healthy node run with *parameters*.

    *parameters* are the ``scenario_bridge`` node's ROS parameters, as a
    launcher's ``bridge_parameters`` hold them; ``route_watch_period_s`` and
    ``mission_poll_timeout_s`` not among them are the node's defaults
    (:data:`ROUTE_WATCH_PERIOD_S`, :data:`MISSION_POLL_TIMEOUT_S`).  ``None``
    when it is enough (:func:`min_liveness_timeout_s`), when the liveness check
    is off (``0``), when the node does not ask once ready, and when there are
    no *parameters* to compare with (a node started elsewhere).
    """
    if parameters is None or liveness_timeout_s <= 0.0:
        return None
    period = _seconds_parameter(
        parameters, "route_watch_period_s", ROUTE_WATCH_PERIOD_S
    )
    poll_timeout = _seconds_parameter(
        parameters, "mission_poll_timeout_s", MISSION_POLL_TIMEOUT_S
    )
    needed = min_liveness_timeout_s(period, poll_timeout)
    if liveness_timeout_s >= needed:
        return None
    return (
        f"liveness_timeout_s ({liveness_timeout_s:g} s) would end a healthy run: "
        "the scenario_bridge node asks for the mission every "
        f"route_watch_period_s ({period:g} s), each ask taking up to "
        f"mission_poll_timeout_s ({poll_timeout:g} s), so liveness_timeout_s "
        f"must be at least twice their sum, {needed:g} s. Raise "
        "liveness_timeout_s, lower route_watch_period_s or "
        "mission_poll_timeout_s in the launcher's bridge_parameters, or set "
        "liveness_timeout_s to 0 to turn the check off"
    )


def launcher_bridge_parameters(launcher: object) -> Optional[Mapping[str, Any]]:
    """The ``scenario_bridge`` node's ROS parameters *launcher* starts it with.

    A launcher's ``bridge_parameters`` (``CommandAutowareLauncher``), or its
    config's (``DockerAutowareLauncher``); ``None`` for a launcher with
    neither, whose node the framework cannot see.
    """
    parameters = getattr(launcher, "bridge_parameters", None)
    if parameters is None:
        parameters = getattr(
            getattr(launcher, "config", None), "bridge_parameters", None
        )
    return parameters if isinstance(parameters, Mapping) else None
