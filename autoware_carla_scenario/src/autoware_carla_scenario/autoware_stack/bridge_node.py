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

from ..utils.cache_dir import user_cache_dir

__all__ = [
    "BRIDGE_APT_PACKAGES",
    "BRIDGE_MODULE",
    "BRIDGE_PROTOBUF_REQUIREMENT",
    "DOUBLE_PARAMETERS",
    "INTEGER_PARAMETERS",
    "MISSION_POLL_TIMEOUT_S",
    "PARENT_PID_PARAMETER",
    "POSITIVE_PARAMETERS",
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

#: The protobuf runtime the node's generated code needs, as pip takes it:
#: ``autoware_bridge/_proto`` is protobuf 4.25 gencode (it imports
#: ``google.protobuf.internal.builder``, which protobuf 3.20 added), the
#: package's own floor (``pyproject.toml``).  Ubuntu 22.04's
#: ``python3-protobuf`` (3.12, the one ROS 2 Humble's images have) is too old
#: for it, so the dependency image installs this over it with pip when the
#: one it has cannot import the generated code (#2217); Ubuntu 24.04's (3.21)
#: can.
BRIDGE_PROTOBUF_REQUIREMENT: str = "protobuf>=4.25.0,<8"


#: The node's ``route_watch_period_s`` when it is not given: how often, in
#: wall-clock seconds, it asks for the mission once Autoware is ready.
ROUTE_WATCH_PERIOD_S: float = 1.0

#: The node's ``mission_poll_timeout_s`` when it is not given: how long one
#: ``GetMission`` may take.
MISSION_POLL_TIMEOUT_S: float = 1.0

#: The node's parameters declared as doubles (``ros_bridge/node.py``,
#: ``_declare_parameters()``).  rclpy refuses an INTEGER override of a DOUBLE
#: declaration, so a whole number given for one of these (``2``) is passed as
#: ``2.0`` (#2187).
DOUBLE_PARAMETERS: tuple[str, ...] = (
    "mission_poll_timeout_s",
    "route_watch_period_s",
    "rpc_timeout_s",
    "step_timeout_s",
    "tick_period_s",
)

#: The node's parameters declared as integers (``ros_bridge/node.py``,
#: ``_declare_parameters()``).  rclpy refuses a DOUBLE override of an INTEGER
#: declaration, and YAML or Hydra may well read ``10.0`` for one: a whole
#: float is passed as the int (``10``), and any other float refused (#2217).
INTEGER_PARAMETERS: tuple[str, ...] = ("reroute_max_attempts",)

#: The node's parameter :func:`bridge_command` sets, to the pid of the shell
#: that starts the stack (``$$``): the node shuts down once that process is
#: gone, as a node started in the background with ``(... &)`` is no child of
#: the stack and would otherwise outlive a run killed outright (#2405).
PARENT_PID_PARAMETER: str = "parent_pid"

#: The double parameters that must be positive: the node waits on, or ticks
#: at, each.  A NaN, zero or negative one would never let a readiness report
#: through (``rpc_timeout_s``), expire every startup request as soon as it is
#: sent (``step_timeout_s``), or reach ``create_timer`` unchecked
#: (``tick_period_s``) (#2275).  ``route_watch_period_s`` is not among them:
#: 0 or less turns the re-route watch off.
POSITIVE_PARAMETERS: tuple[str, ...] = (
    "mission_poll_timeout_s",
    "rpc_timeout_s",
    "step_timeout_s",
    "tick_period_s",
)


def bridge_package_dir() -> Path:
    """The installed :mod:`autoware_carla_scenario.autoware_bridge` directory."""
    from ..autoware_bridge import __file__ as bridge_file  # noqa: PLC0415 - only its location is wanted

    return Path(bridge_file).resolve().parent


def bridge_ros_args(bridge_address: str, parameters: Mapping[str, Any]) -> list[str]:
    """The node's ``--ros-args``: the bridge's address, then *parameters*.

    A parameter set to ``None`` is left out: it is the node's default, as a
    null is everywhere in the config.  A whole number for a parameter the node
    declares as a double (:data:`DOUBLE_PARAMETERS`) is passed as a float,
    which rclpy takes where it refuses the integer; a whole float for one it
    declares as an integer (:data:`INTEGER_PARAMETERS`) is passed as the int.

    Raises:
        ValueError: naming the parameter and the value, for a value the node
            would refuse, or misbehave with, as it starts, after the scenario
            has: a boolean or a string (``'5'``, which reaches ROS as an
            integer or a string) for a number; a NaN or an infinity for a
            double; one that is not positive for a
            :data:`POSITIVE_PARAMETERS` one; a float that is not a whole
            number for an integer parameter; any :data:`PARENT_PID_PARAMETER`,
            which :func:`bridge_command` sets.
    """
    args = ["--ros-args", "-p", f"bridge_address:={bridge_address}"]
    for name, value in parameters.items():
        if value is None:
            continue
        if name == PARENT_PID_PARAMETER:
            raise ValueError(
                f"{name} is not a bridge parameter to set: the launcher sets it "
                f"to the pid of the shell that starts the stack, got {value!r}"
            )
        number = name in DOUBLE_PARAMETERS or name in INTEGER_PARAMETERS
        if number and (isinstance(value, bool) or not isinstance(value, (int, float))):
            # A bool is passed as ``:=true`` and a string as itself, which
            # rclpy refuses for a number, or reads as another type.
            raise ValueError(
                f"{name} must be a number, as the scenario_bridge node "
                f"declares it, got {value!r}"
            )
        if isinstance(value, bool):
            value = str(value).lower()
        elif name in DOUBLE_PARAMETERS:
            seconds = float(value)
            if not math.isfinite(seconds):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if name in POSITIVE_PARAMETERS and seconds <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            value = seconds
        elif isinstance(value, float) and name in INTEGER_PARAMETERS:
            if not math.isfinite(value) or value != math.floor(value):
                raise ValueError(
                    f"{name} must be a whole number, as the scenario_bridge "
                    f"node declares it an integer, got {value!r}"
                )
            value = int(value)
        args += ["-p", f"{name}:={value}"]
    return args


def bridge_command(
    python_root: str, bridge_address: str, parameters: Mapping[str, Any]
) -> str:
    """A shell command running the node, with *python_root* put on ``PYTHONPATH``.

    The node is given the pid of the shell the command runs in
    (:data:`PARENT_PID_PARAMETER`, ``$$``, which a ``( ... &)`` subshell
    expands to its parent's pid) and shuts down once that shell is gone: the
    ``bash -c`` that ``exec``s ``ros2 launch``, whose pid is the stack's.
    """
    root = shlex.quote(python_root)
    argv = [
        "python3",
        "-m",
        BRIDGE_MODULE,
        *bridge_ros_args(bridge_address, parameters),
    ]
    # Concatenated: Codon 0.19 cannot parse `{{` in an f-string.
    # `$$` left unquoted, for the shell to expand.
    return (
        "PYTHONPATH="
        + root
        + '"${PYTHONPATH:+:$PYTHONPATH}" '
        + shlex.join(argv)
        + " -p "
        + PARENT_PID_PARAMETER
        + ":=$$"
    )


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
        base = user_cache_dir() / "autoware_carla_scenario" / "ros_bridge"
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

    A value that is not a finite number (a boolean, a string, a NaN), which
    :func:`bridge_ros_args` refuses, reads as the default: its error says what
    is wrong with it.
    """
    value = parameters.get(name)
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    seconds = float(value)
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
