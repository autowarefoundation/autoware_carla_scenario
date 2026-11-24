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
import math
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, Any, ClassVar, Mapping, Optional

from ..seed import check_whole_number

# The ``<host>:<pid>`` owner label, and whether the process it names runs,
# are the checked utils.owner_label's, which the Autoware stack's containers
# carry too (#2182).
from ..utils.owner_label import owner_alive
from ..utils.owner_label import owner_value as _owner
from ..value_text import shown
from . import BackendError, Resources

logger = logging.getLogger(__name__)

__all__ = [
    "PULL_POLICIES",
    "DockerBackend",
    "DockerSettings",
    "backend_log_name",
    "positive_seconds",
    "read_docker_settings",
    "setting_key",
    "string_list",
    "whole_mib",
]

#: The label every backend container carries: ``<host>:<pid>`` of its starter.
OWNER_LABEL = "autoware_carla_scenario.backend.owner"
#: The label naming the backend.
NAME_LABEL = "autoware_carla_scenario.backend.name"

#: How long a docker command that only reads or removes (``image inspect``,
#: a leftover's ``rm``) may take: a daemon that stops answering must not hang
#: the run (#1785).
_CALL_TIMEOUT_S = 60.0
#: How long reading the end of a container's log for an error may take.
_TAIL_TIMEOUT_S = 10.0
#: The least time a call made while waiting for the container to answer is
#: given, however close the deadline is.
_MIN_CALL_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class DockerSettings:
    """How a backend's container runs.

    Attributes:
        image: The image, with its tag.
        container_port: The port the service listens on inside.
        gpus: ``"auto"`` (the GPU the preparer placed it on), ``"all"``,
            ``"none"``, or what ``docker run --gpus`` takes (``device=1``,
            ``"device=0,1"``, a count).  The preparer places on a GPU only a
            backend that asks for GPU memory (``resources.gpu_memory_mb``
            above 0, and the host checked): ``"auto"`` with none
            (``gpu_memory_mb: 0``) passes ``--gpus all``, every GPU, as
            ``"all"`` does; ``"none"`` passes none.  Any other value is
            checked on the GPUs docker gives the container: every one for
            ``"all"``, those listed for ``device=``, and the first *N* for a
            count *N* (docker's ``--gpus N``).
        env: Environment variables set in the container.
        pass_env: Variables passed on from this process when it has them
            (a token, a proxy).
        mounts: ``host:container[:ro]`` volumes; ``~`` in the host path is
            expanded.
        docker_args: More ``docker run`` arguments, before the image.
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
    pull: str = "missing"
    startup_timeout_s: float = 600.0
    resources: Resources = Resources()
    docker: str = "docker"


#: What ``pull`` takes: pull when the host lacks the image, every start, or
#: never.
PULL_POLICIES = ("missing", "always", "never")


def setting_key(backend: str, key: str) -> str:
    """The name of *backend*'s setting *key* in a run's config
    (``backends.<backend>.<key>``), as an error names it."""
    return f"backends.{backend}.{key}"


def positive_seconds(key: str, value: Any) -> float:
    """*value*, a timeout in seconds, checked: a number, finite and above 0.

    A timeout of 0 or less puts a deadline in the past (a container started,
    then torn down at once), NaN never ends a comparison, and infinity waits
    forever.

    Raises:
        ValueError: naming *key*.
    """
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{key} must be a number of seconds, got {value!r}")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number of seconds, got {value!r}")
    except OverflowError:
        # An int too big for a float (10**400): not finite (#1652).
        raise ValueError(
            f"{key} must be finite and above 0, got {shown(value)}"
        ) from None
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{key} must be finite and above 0, got {value!r}")
    return seconds


def whole_mib(key: str, value: Any) -> int:
    """*value*, an amount of memory in MiB, checked: a whole number
    (:func:`~autoware_carla_scenario.seed.check_whole_number`), not negative.

    Raises:
        ValueError: naming *key*.
    """
    number = check_whole_number(key, value)
    if number < 0:
        raise ValueError(f"{key} must not be negative, got {value!r}")
    return number


def string_list(key: str, value: Any) -> tuple[str, ...]:
    """*value*, a list setting (``docker_args``), as a tuple of strings.

    ``None`` is an empty list.  A string is refused rather than split into
    its characters (``'--shm-size=8g'``), and so is a mapping, or an item
    that is itself a list or a mapping.

    Raises:
        ValueError: naming *key*.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise ValueError(f"{key} must be a list, got {value!r}")
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"{key} must be a list, got {value!r}")
    for item in items:
        if isinstance(item, Mapping) or isinstance(item, (list, tuple)):
            raise ValueError(f"{key} must be a list of strings, got {value!r}")
    return tuple(str(item) for item in items)


def read_docker_settings(
    backend: str,
    settings: Mapping[str, Any],
    *,
    image: str,
    container_port: int,
    env: Optional[Mapping[str, str]] = None,
    pass_env: tuple[str, ...] = (),
    mounts: tuple[str, ...] = (),
    startup_timeout_s: float = 600.0,
    gpu_memory_mb: int = 0,
    memory_mb: int = 0,
) -> DockerSettings:
    """The :class:`DockerSettings` of a backend's block of the run's config
    (``backends.<backend>``), every key optional, checked.

    It reads the keys every container backend shares: ``image``, ``gpus``,
    ``pull`` (one of :data:`PULL_POLICIES`), ``docker_args`` (a list),
    ``startup_timeout_s`` (finite, above 0), ``gpu_memory_mb`` and
    ``memory_mb`` (whole numbers, not negative).  The keyword arguments are
    its defaults, and what the backend sets itself (its port, environment and
    mounts); a backend checks its own keys.  Checking reads nothing of the
    host and starts nothing, so a run can refuse a setting before any image
    is pulled or container started.  ``image`` is text, and ``gpus`` text or
    a count (``docker run --gpus 2``), neither blank; an absent or null one
    is its default, while ``false`` or a number of the wrong kind is refused
    rather than handed to ``docker`` as ``"False"`` (#2143).  A ``gpus``
    count is 1 or more (``gpus: 0`` is ``none``), and a backend that takes
    GPU memory has a ``gpus`` that says which GPUs docker gives it, since the
    memory is checked on each of them (#2233).  Every key
    absent or null is its default (#2158): ``pull: null`` is ``missing``,
    not refused.

    Raises:
        ValueError: naming the key (``backends.<backend>.<key>``) of the
            first value refused.
    """
    image_value = settings.get("image")
    if image_value is None:
        image_value = image
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError(
            f"{setting_key(backend, 'image')} must be an image name, "
            f"got {image_value!r}"
        )
    gpus_value = settings.get("gpus")
    if gpus_value is None:
        gpus_value = "auto"
    if isinstance(gpus_value, int) and not isinstance(gpus_value, bool):
        gpus_value = str(gpus_value)  # A count, as docker run --gpus takes.
    if not isinstance(gpus_value, str) or not gpus_value.strip():
        raise ValueError(
            f"{setting_key(backend, 'gpus')} must be auto, all, none or what "
            f"docker run --gpus takes (device=N, a count), got {gpus_value!r}"
        )
    # Stored stripped, so the GPU check and docker read the same text (#2263).
    gpus_value = gpus_value.strip()
    if _gpu_request(gpus_value)[0] == _BAD_COUNT:
        raise ValueError(
            f"{setting_key(backend, 'gpus')} must be a count of 1 or more, "
            f"got {gpus_value!r}; none is gpus: none"
        )
    pull = settings.get("pull")
    if pull is None:
        pull = "missing"
    if not isinstance(pull, str) or pull not in PULL_POLICIES:
        raise ValueError(
            f"{setting_key(backend, 'pull')} must be one of "
            f"{', '.join(PULL_POLICIES)}, got {pull!r}"
        )
    startup_value = settings.get("startup_timeout_s")
    if startup_value is None:
        startup_value = startup_timeout_s
    gpu_memory_value = settings.get("gpu_memory_mb")
    if gpu_memory_value is None:
        gpu_memory_value = gpu_memory_mb
    memory_value = settings.get("memory_mb")
    if memory_value is None:
        memory_value = memory_mb
    docker_args = string_list(
        setting_key(backend, "docker_args"), settings.get("docker_args")
    )
    startup = positive_seconds(setting_key(backend, "startup_timeout_s"), startup_value)
    gpu_memory = whole_mib(setting_key(backend, "gpu_memory_mb"), gpu_memory_value)
    host_memory = whole_mib(setting_key(backend, "memory_mb"), memory_value)
    # The GPU memory is checked on the GPUs docker gives the container, so
    # those must be known (#2233).
    if gpu_memory > 0 and _gpu_request(gpus_value)[0] == _UNCHECKED:
        raise ValueError(_unchecked_gpus(backend, gpus_value))
    return DockerSettings(
        image=image_value,
        container_port=container_port,
        gpus=gpus_value,
        env=dict(env or {}),
        pass_env=pass_env,
        mounts=mounts,
        docker_args=docker_args,
        pull=pull,
        startup_timeout_s=startup,
        resources=Resources(gpu_memory_mb=gpu_memory, memory_mb=host_memory),
    )


def backend_log_name(name: str) -> str:
    """The file backend *name*'s container log is followed into.

    ``backend_<name>.log``; in a SLURM array task,
    ``backend_<name>-<array job id>_<task id>.log``, the task's own as its
    ``<array job id>_<task id>.out`` is: every task of a batch starts its own
    container and writes into the batch's directory (#2247).
    """
    job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if job and task:
        return f"backend_{name}-{job}_{task}.log"
    return f"backend_{name}.log"


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

    def resources(self) -> Resources:
        """The settings' resources, on the GPUs ``gpus`` gives the container
        (#2233): none for ``none``, the one of ``device=N``, those of
        ``device=N,M``, the first *N* for a count *N* (as ``docker run --gpus
        N`` gives), every one for ``all``, and the one the preparer places it
        on for ``auto``.

        Raises:
            ValueError: If ``gpus`` is a count below 1, or names GPUs that
                cannot be told (a UUID, other ``--gpus`` options) for a
                backend that takes GPU memory.
        """
        settings = self._settings
        kind, devices = _gpu_request(settings.gpus)
        if kind == _NONE:
            return replace(settings.resources, gpu_memory_mb=0)
        if kind == _BAD_COUNT:
            raise ValueError(
                f"{setting_key(self.name, 'gpus')} must be a count of 1 or more, "
                f"got {settings.gpus!r}; none is gpus: none"
            )
        if kind == _DEVICES:
            if len(devices) == 1:
                return replace(settings.resources, gpu=devices[0])
            return replace(settings.resources, gpus=tuple(devices))
        if kind == _ALL:
            return replace(settings.resources, every_gpu=True)
        if kind == _UNCHECKED and settings.resources.gpu_memory_mb > 0:
            raise ValueError(_unchecked_gpus(self.name, settings.gpus))
        return settings.resources

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
            # Not --rm: one that exits while starting keeps its logs for the
            # error; close() removes it.
            "--detach",
            # Never a pull of its own (#1960): _ensure_image() has pulled what
            # `pull` asks for, with no timeout, and `never` means the image is
            # the host's.  docker run's own default (`missing`) would pull a
            # missing image under _docker()'s 120 s timeout.
            "--pull",
            "never",
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
        argv += [*settings.docker_args, settings.image]
        logger.info("Starting backend '%s': %s", self.name, " ".join(argv))
        # docker run often creates the container and then fails to start it
        # (a port taken since _free_port(), a GPU error, its timeout, Ctrl-C):
        # the container is removed by name whatever stops the call (#1960).
        try:
            out = _docker(argv, f"start the '{self.name}' backend")
        except BaseException:
            self._remove(name)
            raise
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
            # Whatever the stop did (#1960): a stop that timed out leaves the
            # container running, and rm --force kills it.
            self._remove(container)
        follower = self._follower
        self._follower = None
        if follower is not None:
            follower.terminate()
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.kill()
                try:
                    follower.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "The log follower of backend '%s' did not exit", self.name
                    )
        log = self._log
        self._log = None
        if log is not None:
            log.close()

    # ------------------------------------------------------------------

    def _remove(self, container: str) -> None:
        """``docker rm --force`` *container*, at most :data:`_CALL_TIMEOUT_S`;
        a failure is logged, never raised -- it must not stand in for the
        error that made the caller remove it."""
        try:
            subprocess.run(
                [self._settings.docker, "rm", "--force", container],
                capture_output=True,
                timeout=_CALL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.warning("Could not remove backend container %s", container)

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
        if self._settings.gpus.strip() != _NONE and "nvidia" not in runtimes:
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
        for line in listing.splitlines():
            container, _, owner = line.partition(" ")
            # Another host's, a malformed label's (None) or a live run's.
            if owner_alive(owner) is not False:
                continue
            logger.warning(
                "Removing backend container %s a stopped run (%s) left",
                container,
                owner,
            )
            try:
                _docker(
                    [docker, "rm", "--force", container],
                    f"remove container {container}",
                    _CALL_TIMEOUT_S,
                )
            except BackendError as error:
                # Another run's leftover: one that cannot be removed now is
                # removed by the next start, and does not stop this one.
                logger.warning("%s", error)

    def _ensure_image(self) -> None:
        settings = self._settings
        if settings.pull == "never":
            return
        if settings.pull == "missing":
            argv = [settings.docker, "image", "inspect", settings.image]
            try:
                present = _run(argv, _CALL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                raise BackendError(
                    f"could not look for {settings.image}: '{' '.join(argv[:3])}' "
                    f"timed out"
                ) from None
            except OSError as error:
                raise BackendError(
                    f"could not look for {settings.image}: {error}"
                ) from None
            if present.returncode == 0:
                return
        logger.info("Pulling %s for backend '%s'", settings.image, self.name)
        _docker(
            [settings.docker, "pull", settings.image], f"pull {settings.image}", None
        )

    def _follow(self, log_dir: Optional[Path]) -> None:
        if log_dir is None or self._container is None:
            return
        # An OSError (no docker binary, a fork that fails with EMFILE or
        # ENOMEM, a log dir that cannot be written) is the backend's error,
        # not a raw traceback (#1790); the caller's close() removes the
        # container and closes the log.
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log = (log_dir / backend_log_name(self.name)).open("ab")
            self._follower = subprocess.Popen(
                [self._settings.docker, "logs", "--follow", self._container],
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            raise BackendError(
                f"could not follow the log of backend '{self.name}': {error}"
            ) from None

    def _wait_ready(self) -> None:
        settings = self._settings
        deadline = time.monotonic() + settings.startup_timeout_s
        # Every call is capped by the time left (#1785): a docker that stops
        # answering ends in the deadline, not in a run that hangs.
        left = deadline - time.monotonic()
        while left > 0.0:
            if self._exited(max(left, _MIN_CALL_TIMEOUT_S)):
                raise BackendError(
                    f"backend '{self.name}' ({settings.image}) exited while starting:\n"
                    f"{self._tail()}"
                )
            left = deadline - time.monotonic()
            if self.ready(timeout_s=min(5.0, max(left, _MIN_CALL_TIMEOUT_S))):
                return
            left = deadline - time.monotonic()
            if left > 0.0:
                time.sleep(min(2.0, left))
            left = deadline - time.monotonic()
        try:
            tail = self._tail()
        finally:
            # A failure to read the log must not leave the container running.
            self.close()
        raise BackendError(
            f"backend '{self.name}' ({settings.image}) did not answer within "
            f"{settings.startup_timeout_s:.0f} s:\n{tail}"
        )

    def _exited(self, timeout_s: float) -> bool:
        """Whether docker says the container is not running.

        A docker that does not answer within *timeout_s* says nothing: the
        container is taken as not running *yet*, and the caller's deadline
        decides.
        """
        if self._container is None:
            return True
        try:
            state = _run(
                [
                    self._settings.docker,
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    self._container,
                ],
                timeout_s,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "docker did not say within %.0f s whether backend '%s' runs",
                timeout_s,
                self.name,
            )
            return False
        except OSError as error:
            # The caller's close() removes the container (#1790).
            raise BackendError(
                f"could not ask docker whether backend '{self.name}' runs: {error}"
            ) from None
        return not (state.returncode == 0 and state.stdout.strip() == "true")

    def _tail(self) -> str:
        """The end of the container's log, or a note that docker did not give
        it within :data:`_TAIL_TIMEOUT_S` or could not be run."""
        if self._container is None:
            return ""
        timeout_s = _TAIL_TIMEOUT_S
        try:
            logs = _run(
                [self._settings.docker, "logs", "--tail", "40", self._container],
                timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"(no log: docker did not give it within {timeout_s:.0f} s)"
        except OSError as error:
            return f"(no log: could not run docker: {error})"
        return (logs.stdout + logs.stderr).strip()


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """A docker command run for its answer, whatever its exit status, within
    *timeout* seconds.

    Its output, the backend's own (a log) among it, is any bytes: what is not
    UTF-8 is replaced rather than fail to report the error it explains
    (#1346).

    Raises:
        subprocess.TimeoutExpired: when it does not finish in time.
        OSError: when it cannot be run.
    """
    return subprocess.run(
        argv,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _docker(argv: list[str], what: str, timeout: Optional[float] = 120.0) -> str:
    """Run a docker command, its stdout; :class:`BackendError` if it fails."""
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        raise BackendError(f"could not {what}: no '{argv[0]}' on this host") from None
    except subprocess.TimeoutExpired:
        raise BackendError(
            f"could not {what}: '{' '.join(argv[:3])}' timed out"
        ) from None
    except OSError as error:
        raise BackendError(f"could not {what}: {error}") from None
    if done.returncode != 0:
        raise BackendError(
            f"could not {what}: {done.stderr.strip() or done.stdout.strip()}"
        )
    return done.stdout


#: What ``docker run --gpus`` gives a container (:func:`_gpu_request`).
_NONE = "none"
_AUTO = "auto"
_ALL = "all"
_DEVICES = "devices"
_BAD_COUNT = "bad count"
_UNCHECKED = "unchecked"


def _whole(text: str) -> bool:
    """Whether *text* is a whole number, a ``-`` before its digits or not."""
    digits = text[1:] if text.startswith("-") else text
    return digits != "" and digits.isascii() and digits.isdigit()


def _gpu_request(gpus: str) -> tuple[str, list[int]]:
    """Which GPUs a ``gpus`` setting gives the container, and their indices.

    ``none``, ``auto`` and ``all`` are themselves; ``device=N[,M...]``
    (quoted or not, as ``docker run --gpus '"device=0,1"'`` takes it) is
    those indices; a count *N* (``N`` or ``count=N``) the first *N*, as
    docker hands ``--gpus N`` the GPUs ``0`` to ``N-1``, and ``count=all``
    every one.  A count below 1 is :data:`_BAD_COUNT`; anything else (a
    UUID, other ``--gpus`` options) is :data:`_UNCHECKED`: the GPUs it
    gives cannot be told here.  An index named twice (``device=0,0``) is one
    GPU, listed once, so its memory is not booked twice (#2263).
    """
    text = gpus.strip()
    devices: list[int] = []
    if text == _NONE or text == _AUTO or text == _ALL:
        return text, devices
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if text.startswith("count="):
        text = text[len("count=") :].strip()
        if text == _ALL:
            return _ALL, devices
    if _whole(text):
        count = int(text)
        if count < 1:
            return _BAD_COUNT, devices
        for index in range(count):
            devices.append(index)
        return _DEVICES, devices
    if text.startswith("device="):
        for item in text[len("device=") :].split(","):
            item = item.strip()
            if not (item != "" and item.isascii() and item.isdigit()):
                return _UNCHECKED, []
            index = int(item)
            if index not in devices:
                devices.append(index)
        if devices:
            return _DEVICES, devices
    return _UNCHECKED, []


def _unchecked_gpus(backend: str, gpus: str) -> str:
    """The error for a ``gpus`` whose GPUs cannot be checked for the GPU
    memory a backend takes."""
    return (
        f"{setting_key(backend, 'gpus')} {gpus!r} does not say which GPUs "
        f"docker gives the container, so the {setting_key(backend, 'gpu_memory_mb')} "
        "it takes cannot be checked on them: set gpus to auto, all, device=N "
        "(or \"device=N,M\") or a count, or gpu_memory_mb to 0"
    )


def _gpu_args(gpus: str, placed: Optional[int]) -> list[str]:
    """``docker run``'s ``--gpus`` for a ``gpus`` setting, read as
    :func:`_gpu_request` reads it: around its surrounding spaces (#2263)."""
    text = gpus.strip()
    if text == _NONE:
        return []
    if text == _AUTO:
        return ["--gpus", "all" if placed is None else f"device={placed}"]
    return ["--gpus", text]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
