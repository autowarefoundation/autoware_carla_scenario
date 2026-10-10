"""Bring an Autoware stack up for one scenario and tear it down after it.

A scenario run against Autoware is only as good as the Autoware it starts
from.  Localization, the route, the operation mode, the planners' memory of
the last run and every node's own state carry over from one run to the next
unless the stack is started afresh, so the framework starts it itself: once
per scenario, from the outside, and removes it -- every process -- when the
scenario ends.  That is what lets a :class:`~autoware_carla_scenario.ScenarioQueue`
or a sweep run scenario after scenario without anyone restarting Autoware by
hand.

:class:`AutowareLauncher` is the contract
:class:`~autoware_carla_scenario.entity.autoware_entity.AutowareEgoEntity`
drives: :meth:`~AutowareLauncher.prepare` once per batch,
:meth:`~AutowareLauncher.start` / :meth:`~AutowareLauncher.stop` once per
scenario, :meth:`~AutowareLauncher.poll` on every tick in between.  Two
implementations ship:

* :class:`~autoware_carla_scenario.autoware_stack.docker.DockerAutowareLauncher`
  runs a local Autoware workspace in its dev container image -- builds it, and
  starts each scenario's stack in a container of its own;
* :class:`CommandAutowareLauncher` runs any command as a process group, for an
  Autoware installed on the host (and for tests).

Nothing here imports ROS 2: the stack is a child process, and the only thing
the framework says to it is on its command line (and over the bridge).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "AutowareEpisode",
    "AutowareLauncher",
    "AutowareStackError",
    "CommandAutowareLauncher",
    "ProcessGroup",
    "episode_log_path",
    "fill_placeholders",
]


class AutowareStackError(RuntimeError):
    """The Autoware stack could not be prepared or started."""


@dataclass(frozen=True)
class AutowareEpisode:
    """The scenario a launcher started a stack for.

    Attributes:
        index: 0-based count of the stacks the launcher has started, so two
            runs of the same scenario in one batch keep separate logs.
        name: The scenario's name, for logs and container names.
        bridge_address: ``host:port`` of the bridge server the stack's
            ``scenario_bridge`` node is to dial.  The framework is already
            serving the mission there when the stack starts.
    """

    index: int
    name: str
    bridge_address: str


@runtime_checkable
class AutowareLauncher(Protocol):
    """Starts and stops one Autoware stack per scenario.

    At most one stack runs at a time: :meth:`start` is only called once the
    previous one has been stopped.
    """

    def prepare(self, log_dir: Optional[Path] = None) -> None:
        """Get ready to start stacks: build, check, clean up.  Idempotent.

        Called once before a batch runs, so a workspace that cannot run a
        scenario is refused before any scenario starts.

        Args:
            log_dir: Where the stacks' output goes from here on (the batch's
                output directory); ``None`` keeps the launcher's own.

        Raises:
            AutowareStackError: If no stack could be started from here.
        """
        ...

    def start(self, *, name: str, bridge_address: str) -> AutowareEpisode:
        """Start a stack and return without waiting for it to come up.

        The stack takes the better part of a minute to come up; the entity
        waits for it over the bridge, while the world keeps ticking.

        Args:
            name: The scenario's name, for logs.
            bridge_address: ``host:port`` of the bridge server the stack's
                ``scenario_bridge`` node is to dial.

        Returns:
            The episode the stack was started for, numbered by the launcher.

        Raises:
            AutowareStackError: If the stack could not be started.
        """
        ...

    def poll(self) -> Optional[int]:
        """The stack's exit code once it has exited, ``None`` while it runs.

        Must be cheap: it is called on every world tick.
        """
        ...

    def stop(self) -> None:
        """Stop the running stack and everything it started.  Idempotent.

        Returns once nothing of it is left running.  Never raises.
        """
        ...

    def close(self) -> None:
        """Release whatever :meth:`prepare` set up.  Idempotent; never raises."""
        ...


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


class ProcessGroup:
    """A child process started in a session of its own, stopped as a group.

    ``ros2 launch`` starts every node as a child of its own, and a node can
    outlive a launch that died badly; stopping the whole process group is what
    leaves nothing behind.  The group is asked to stop with *stop_signal*
    (``SIGINT``, which ``ros2 launch`` shuts its nodes down on), then
    ``SIGTERM``, then ``SIGKILL``, each after *stop_timeout_s*.

    Args:
        argv: The command to run.
        env: The child's environment; ``None`` inherits this process's.
        log_path: Where its stdout and stderr go; ``None`` discards them.
        stop_signal: The first signal :meth:`stop` sends.
        stop_timeout_s: How long :meth:`stop` waits after each signal.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Optional[Mapping[str, str]] = None,
        log_path: Optional[Path] = None,
        stop_signal: signal.Signals = signal.SIGINT,
        stop_timeout_s: float = 20.0,
    ) -> None:
        self._stop_signal = stop_signal
        self._stop_timeout_s = stop_timeout_s
        self._log: Optional[IO[bytes]] = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("ab")
        try:
            self._process = subprocess.Popen(  # noqa: S603 - argv comes from the user's own config
                list(argv),
                env=None if env is None else dict(env),
                stdin=subprocess.DEVNULL,
                stdout=self._log if self._log is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._close_log()
            raise AutowareStackError(f"Could not run {argv[0]!r}: {exc}") from exc
        # start_new_session makes the child its group's leader.
        self._pgid = self._process.pid

    @property
    def pid(self) -> int:
        """The child's process id (and its group's)."""
        return self._process.pid

    def poll(self) -> Optional[int]:
        """The child's exit code once it has exited, ``None`` while it runs."""
        return self._process.poll()

    def stop(self) -> Optional[int]:
        """Stop the group; returns the child's exit code.  Never raises."""
        for sig in (self._stop_signal, signal.SIGTERM, signal.SIGKILL):
            if self._process.poll() is not None:
                break
            _signal_group(self._pgid, sig)
            try:
                self._process.wait(timeout=self._stop_timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Process %d did not exit %.0f s after %s",
                    self._process.pid,
                    self._stop_timeout_s,
                    sig.name,
                )
        # The leader may be gone while a node it started is not: whatever is
        # left of the group goes too.
        _signal_group(self._pgid, signal.SIGKILL)
        self._close_log()
        return self._process.poll()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


@dataclass
class CommandAutowareLauncher:
    """Start each scenario's stack by running a command on this host.

    For an Autoware installed on the host rather than in a container::

        CommandAutowareLauncher(
            [
                "bash", "-c",
                "source ~/autoware/install/setup.bash && exec ros2 launch "
                "autoware_launch e2e_simulator.launch.xml ... "
                "bridge_address:={bridge_address}",
            ],
        )

    ``{bridge_address}``, ``{index}`` and ``{name}`` in any argument are
    replaced with the episode's (see :class:`AutowareEpisode`); the address is
    in the environment as ``AUTOWARE_BRIDGE_ADDRESS`` too.

    Attributes:
        argv: The command, with the placeholders above.
        env: Variables added to this process's environment for the command.
        log_dir: Where each stack's output goes, as
            ``autoware-<index>-<name>.log``; ``None`` discards it.  A
            ``log_dir`` passed to :meth:`prepare` replaces it.
        stop_timeout_s: How long stopping waits after each signal.
    """

    argv: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    log_dir: Optional[Path] = None
    stop_timeout_s: float = 20.0

    _process: Optional[ProcessGroup] = field(default=None, init=False, repr=False)
    _episodes: int = field(default=0, init=False, repr=False)

    def prepare(self, log_dir: Optional[Path] = None) -> None:
        if log_dir is not None:
            self.log_dir = log_dir
        if not self.argv:
            raise AutowareStackError("CommandAutowareLauncher needs a command to run")

    def start(self, *, name: str, bridge_address: str) -> AutowareEpisode:
        if self._process is not None:
            raise AutowareStackError(
                "An Autoware stack is already running; stop it before starting another"
            )
        episode = AutowareEpisode(self._episodes, name, bridge_address)
        self._episodes += 1
        argv = [fill_placeholders(arg, episode) for arg in self.argv]
        env = {
            **os.environ,
            **self.env,
            "AUTOWARE_BRIDGE_ADDRESS": episode.bridge_address,
        }
        logger.info("Starting the Autoware stack for %s: %s", episode.name, argv)
        self._process = ProcessGroup(
            argv,
            env=env,
            log_path=episode_log_path(self.log_dir, episode),
            stop_timeout_s=self.stop_timeout_s,
        )
        return episode

    def poll(self) -> Optional[int]:
        return None if self._process is None else self._process.poll()

    def stop(self) -> None:
        if self._process is None:
            return
        code = self._process.stop()
        logger.info("Autoware stack stopped (exit code %s)", code)
        self._process = None

    def close(self) -> None:
        self.stop()


def episode_log_path(
    log_dir: Optional[Path], episode: AutowareEpisode
) -> Optional[Path]:
    """``<log_dir>/autoware-<index>-<name>.log``, or ``None`` without a directory."""
    if log_dir is None:
        return None
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in episode.name)
    return log_dir / f"autoware-{episode.index:03d}-{safe_name}.log"


def fill_placeholders(text: str, episode: AutowareEpisode) -> str:
    """Replace ``{bridge_address}``, ``{index}`` and ``{name}`` in *text*.

    Only those: every other brace is the command's own -- a shell's
    ``${VAR}``, a Python one-liner's dict -- and stays as written.
    """
    for key, value in (
        ("bridge_address", episode.bridge_address),
        ("index", str(episode.index)),
        ("name", episode.name),
    ):
        text = text.replace("{" + key + "}", value)
    return text
