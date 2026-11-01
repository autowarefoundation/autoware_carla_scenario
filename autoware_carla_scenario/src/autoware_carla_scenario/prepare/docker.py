"""A backend run as a docker container on this host.

:class:`DockerBackend` pulls its image if the host lacks it, runs it with the
GPU it was placed on, publishes its port on loopback, follows its log into
the run's output directory, and waits until it answers
(:meth:`DockerBackend.ready`, a subclass's).  Each container is labelled with
the process that started it, so one a killed run left behind is removed the
next time a backend of its name starts -- and only that one: two runs on one
host (two SLURM jobs) each keep their own.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, ClassVar, Optional

from . import BackendError, Resources

logger = logging.getLogger(__name__)

__all__ = ["DockerBackend", "DockerSettings"]

#: The label every backend container carries: ``<host>:<pid>`` of its starter.
OWNER_LABEL = "autoware_carla_scenario.backend.owner"
#: The label naming the backend.
NAME_LABEL = "autoware_carla_scenario.backend.name"


@dataclass(frozen=True)
class DockerSettings:
    """How a backend's container runs.

    Attributes:
        image: The image, with its tag.
        container_port: The port the service listens on inside.
        gpus: ``"auto"`` (the GPU the preparer placed it on), ``"all"``,
            ``"none"``, or what ``docker run --gpus`` takes (``device=1``).
        env: Environment variables set in the container.
        pass_env: Variables passed on from this process when it has them
            (a token, a proxy).
        mounts: ``host:container[:ro]`` volumes; ``~`` in the host path is
            expanded.
        docker_args: More ``docker run`` arguments, before the image.
        command: The container's command (``()``: the image's).
        pull: ``missing`` (pull when the host lacks the image), ``always``
            or ``never``.
        startup_timeout_s: How long it may take to answer -- a first start
            downloads the model.
        resources: What it takes of the host.
        docker: The docker binary.
    """

    image: str
    container_port: int
    gpus: str = "auto"
    env: dict[str, str] = field(default_factory=dict)
    pass_env: tuple[str, ...] = ()
    mounts: tuple[str, ...] = ()
    docker_args: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    pull: str = "missing"
    startup_timeout_s: float = 600.0
    resources: Resources = Resources()
    docker: str = "docker"


class DockerBackend:
    """A backend in a container (:class:`~autoware_carla_scenario.prepare.Backend`).

    A subclass names itself (``backend_name``), builds its settings
    (``from_settings``) and says when the service answers (:meth:`ready`).
    """

    backend_name: ClassVar[str] = ""

    _settings: DockerSettings
    _container: Optional[str]
    _port: int
    _log: Optional[IO[bytes]]
    _follower: Optional["subprocess.Popen[bytes]"]

    def __init__(self, settings: DockerSettings) -> None:
        self._settings = settings
        self._container = None
        self._port = 0
        self._log = None
        self._follower = None

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def settings(self) -> DockerSettings:
        return self._settings

    def resources(self) -> Resources:
        return self._settings.resources

    def address(self) -> str:
        return f"127.0.0.1:{self._port}"

    def ready(self, timeout_s: float) -> bool:
        """Whether the service answers, waiting at most *timeout_s*."""
        raise NotImplementedError

    def start(self, log_dir: Optional[Path] = None, gpu: Optional[int] = None) -> None:
        settings = self._settings
        self._check_docker()
        self._remove_leftovers()
        self._ensure_image()
        self._port = _free_port()
        name = f"acs-{self.name}-{socket.gethostname()}-{os.getpid()}-{self._port}"
        argv = [
            settings.docker,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            f"{OWNER_LABEL}={_owner()}",
            "--label",
            f"{NAME_LABEL}={self.name}",
            "--publish",
            f"127.0.0.1:{self._port}:{settings.container_port}",
            *_gpu_args(settings.gpus, gpu),
        ]
        for key, value in settings.env.items():
            argv += ["--env", f"{key}={value}"]
        for key in settings.pass_env:
            if key in os.environ:
                argv += ["--env", key]
        for mount in settings.mounts:
            host, _, rest = mount.partition(":")
            argv += ["--volume", f"{os.path.expanduser(host)}:{rest}"]
        argv += [*settings.docker_args, settings.image, *settings.command]
        logger.info("Starting backend '%s': %s", self.name, " ".join(argv))
        out = _docker(argv, f"start the '{self.name}' backend")
        self._container = out.strip() or name
        self._follow(log_dir)
        self._wait_ready()

    def close(self) -> None:
        container = self._container
        self._container = None
        if container is not None:
            try:
                subprocess.run(
                    [self._settings.docker, "stop", "--time", "10", container],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                logger.warning("Could not stop backend container %s", container)
        follower = self._follower
        self._follower = None
        if follower is not None:
            follower.terminate()
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.kill()
        log = self._log
        self._log = None
        if log is not None:
            log.close()

    # ------------------------------------------------------------------

    def _check_docker(self) -> None:
        try:
            runtimes = _docker(
                [self._settings.docker, "info", "--format", "{{json .Runtimes}}"],
                "reach the docker daemon",
            )
        except BackendError as error:
            raise BackendError(
                f"backend '{self.name}' runs in docker, which is not usable here: "
                f"{error}"
            ) from None
        if self._settings.gpus != "none" and "nvidia" not in runtimes:
            logger.warning(
                "docker on this host lists no nvidia runtime; backend '%s' may "
                "not get its GPU",
                self.name,
            )

    def _remove_leftovers(self) -> None:
        """Remove this backend's containers whose starter is gone."""
        docker = self._settings.docker
        listing = _docker(
            [
                docker,
                "ps",
                "--all",
                "--filter",
                f"label={NAME_LABEL}={self.name}",
                "--format",
                f'{{{{.ID}}}} {{{{.Label "{OWNER_LABEL}"}}}}',
            ],
            "list backend containers",
        )
        host = socket.gethostname()
        for line in listing.splitlines():
            container, _, owner = line.partition(" ")
            owner_host, _, pid = owner.rpartition(":")
            if owner_host != host or not pid.isdigit() or _alive(int(pid)):
                continue
            logger.warning(
                "Removing backend container %s a stopped run (%s) left",
                container,
                owner,
            )
            subprocess.run(
                [docker, "rm", "--force", container], capture_output=True, check=False
            )

    def _ensure_image(self) -> None:
        settings = self._settings
        if settings.pull == "never":
            return
        if settings.pull == "missing":
            present = subprocess.run(
                [settings.docker, "image", "inspect", settings.image],
                capture_output=True,
                check=False,
            )
            if present.returncode == 0:
                return
        logger.info("Pulling %s for backend '%s'", settings.image, self.name)
        _docker(
            [settings.docker, "pull", settings.image], f"pull {settings.image}", None
        )

    def _follow(self, log_dir: Optional[Path]) -> None:
        if log_dir is None or self._container is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(log_dir / f"backend_{self.name}.log", "ab")
        self._follower = subprocess.Popen(
            [self._settings.docker, "logs", "--follow", self._container],
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def _wait_ready(self) -> None:
        settings = self._settings
        deadline = time.monotonic() + settings.startup_timeout_s
        while time.monotonic() < deadline:
            if not self._running():
                raise BackendError(
                    f"backend '{self.name}' ({settings.image}) exited while starting:\n"
                    f"{self._tail()}"
                )
            if self.ready(timeout_s=5.0):
                return
            time.sleep(2.0)
        tail = self._tail()
        self.close()
        raise BackendError(
            f"backend '{self.name}' ({settings.image}) did not answer within "
            f"{settings.startup_timeout_s:.0f} s:\n{tail}"
        )

    def _running(self) -> bool:
        if self._container is None:
            return False
        state = subprocess.run(
            [
                self._settings.docker,
                "inspect",
                "--format",
                "{{.State.Running}}",
                self._container,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return state.returncode == 0 and state.stdout.strip() == "true"

    def _tail(self) -> str:
        if self._container is None:
            return ""
        logs = subprocess.run(
            [self._settings.docker, "logs", "--tail", "40", self._container],
            capture_output=True,
            text=True,
            check=False,
        )
        return (logs.stdout + logs.stderr).strip()


def _docker(argv: list[str], what: str, timeout: Optional[float] = 120.0) -> str:
    """Run a docker command, its stdout; :class:`BackendError` if it fails."""
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        raise BackendError(f"could not {what}: no '{argv[0]}' on this host") from None
    except subprocess.TimeoutExpired:
        raise BackendError(
            f"could not {what}: '{' '.join(argv[:3])}' timed out"
        ) from None
    if done.returncode != 0:
        raise BackendError(
            f"could not {what}: {done.stderr.strip() or done.stdout.strip()}"
        )
    return done.stdout


def _gpu_args(gpus: str, placed: Optional[int]) -> list[str]:
    if gpus == "none":
        return []
    if gpus == "auto":
        return ["--gpus", "all" if placed is None else f"device={placed}"]
    return ["--gpus", gpus]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
