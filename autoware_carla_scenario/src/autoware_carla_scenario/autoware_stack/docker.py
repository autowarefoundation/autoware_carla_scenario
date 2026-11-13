"""Run a local Autoware workspace in its dev container, one container per scenario.

The workspace is the one a developer already works in: a clone of
``autowarefoundation/autoware`` built in its dev container.  The launcher reads
which image that dev container runs and where it mounts the workspace
(:mod:`.devcontainer`), builds the workspace there when asked to, and starts
each scenario's stack in a container of its own, removed with everything in it
when the scenario ends::

    launcher = DockerAutowareLauncher(
        DockerAutowareConfig(
            workspace=Path("~/autoware"),
            launch=(
                "autoware_launch", "e2e_simulator.launch.xml",
                "map_path:=/home/aw/autoware_map/Town01",
                "simulator_type:=carla",
                "scenario_mode:=true",
            ),
        )
    )

Each stack's container also runs the ``scenario_bridge`` node (:mod:`.bridge_node`),
the framework's own, from this package mounted into it; it needs ``grpcio`` and
``protobuf`` in the image.  And the dev container's image is what its sources
were set up against when it was built: a workspace whose packages have since
gained dependencies (``rosdep`` keys in their ``package.xml``) is run in a dev
container those were installed into by hand, and a fresh container from the
image does not have them.  So both are installed into an image of its own, made
once from the dev container's and kept for as long as the ``package.xml`` files
and the image do not change.

A workspace whose ``autoware_carla_interface`` has no ``scenario_mode`` -- one
built from a branch without scenario support -- cannot run a scenario at all,
and is refused by :meth:`DockerAutowareLauncher.prepare` before any scenario
starts, unless the configured launch file declares ``scenario_mode`` itself.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence

from ..constants import MAX_ROS_DOMAIN_ID
from .bridge_node import (
    BRIDGE_APT_PACKAGES,
    bridge_command,
    bridge_package_dir,
)
from .devcontainer import DEFAULT_WORKSPACE_MOUNT, DevContainer, read_devcontainer
from .launcher import (
    DEFAULT_BRIDGE_PARAMETERS,
    AutowareEpisode,
    AutowareStackError,
    ProcessGroup,
    _check_argument_list,
    _check_stop_timeout_s,
    episode_log_path,
    fill_placeholders,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BuildPolicy",
    "DEPENDENCY_IMAGE_REPOSITORY",
    "DockerAutowareConfig",
    "DockerAutowareLauncher",
    "OWNER_LABEL",
    "STACK_LABEL",
]

#: Every container this launcher starts carries this label, which is how one
#: left behind by a run that was killed is found and removed by the next.
STACK_LABEL: str = "autoware_carla_scenario.stack"

#: And this one, ``<host>:<pid>`` of the process that started it: a leftover
#: is one whose process is gone, not a live stack of another run of the same
#: user on the host (a SLURM task beside this one, #1862).  A host and pid
#: name one process only while the launchers on a host share its pid
#: namespace, which is assumed: a launcher run in a container of its own
#: (a pid namespace of its own) that keeps the host's name would look the
#: others' pids up in the wrong namespace and could take a live owner for
#: a dead one.  One with a host name of its own is another host's to the
#: rest, whose running stacks are left alone (remove_leftover_containers()).
OWNER_LABEL: str = "autoware_carla_scenario.owner"

#: The states of a container that runs nothing, which one whose owner cannot
#: be told alive or dead is removed in (not ``running``, ``paused`` or
#: ``restarting``: those may be another run's live stack).
_STOPPED_STATES: tuple[str, ...] = ("exited", "created", "dead")

#: When the workspace is built: ``auto`` when it has never been built
#: (``install/setup.bash`` is missing), ``always`` before every batch, ``never``
#: (the workspace is built by hand, from the dev container).
BuildPolicy = Literal["auto", "always", "never"]

#: Images holding a workspace's dependencies are tagged
#: ``<this>:<digest of the base image and the workspace's package.xml files>``.
DEPENDENCY_IMAGE_REPOSITORY: str = "autoware-carla-scenario-deps"

#: The package whose launch file takes ``scenario_mode``.
_INTERFACE_PACKAGE = "autoware_carla_interface"

#: Where the ``scenario_bridge`` node's package is mounted in a stack's
#: container: the directory put on ``PYTHONPATH``.
_BRIDGE_PYTHON_ROOT = "/opt/autoware_carla_scenario/python"

#: How long a captured docker call (``ps``, ``image inspect``, ``rm``) may
#: take before it is given up on: a daemon that stops answering fails the
#: call rather than hang the batch between scenarios (#1790).
_DOCKER_TIMEOUT_S = 60.0

#: How long ``docker commit`` of the dependency image may take: it writes the
#: container's whole layer (apt packages and, with ``rosdep``, the workspace's
#: dependencies, gigabytes of them), once per digest.  The workspace's build
#: and the dependencies' install stream their output and are not bounded.
_DOCKER_COMMIT_TIMEOUT_S = 1800.0

#: The lock file, in the workspace, that the dependency image's install and
#: the workspace's build are run under: one run on the host (or on any host
#: the workspace is shared with, where the file system locks across hosts)
#: at a time, so SLURM tasks sharing a workspace do not ``colcon build`` it
#: together (#1862).  In the workspace, not in a per-user cache directory:
#: the runs to keep apart are those on one workspace, on whichever node.
#: Taken only when there is something to make or build.
_BUILD_LOCK_NAME = ".autoware_carla_scenario-build.lock"

#: How long a run waits for another's build of the workspace before failing.
_BUILD_LOCK_TIMEOUT_S = 4 * 3600.0

#: How often the build lock is tried while another run holds it.
_BUILD_LOCK_POLL_S = 2.0


def _shapes_the_cli(key: str) -> bool:
    """Whether variable *key* changes what the Docker CLI process does.

    ``autoware.launcher.env`` is passed by name (#1961), its value in the CLI's
    environment; one of these would reach the CLI (or the wrapper that starts
    it) as well as the container, so it stays on the command line.
    """
    upper = key.upper()
    return (
        key in ("PATH", "HOME", "TMPDIR", "SHELL")
        or key.startswith("DOCKER_")
        or key.startswith("LD_")
        or key.startswith("PYTHON")
        or upper.endswith("_PROXY")
        or upper.startswith("SSL_")
    )


def _label_value() -> str:
    """The value of :data:`STACK_LABEL` on this user's containers."""
    return f"uid-{os.getuid()}"


def _owner_value() -> str:
    """The value of :data:`OWNER_LABEL` on this process's containers."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _process_alive(pid: int) -> bool:
    """Whether process *pid* exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's, but alive
    return True


def _owner_alive(owner: str) -> Optional[bool]:
    """Whether the process an :data:`OWNER_LABEL` value names still runs;
    None when that cannot be told (no label, a malformed one, another host's)."""
    host, _, pid = owner.rpartition(":")
    if not host or not pid.isdigit():
        return None
    if host != socket.gethostname():
        return None
    return _process_alive(int(pid))


@dataclass(frozen=True)
class DockerAutowareConfig:
    """How to build and start an Autoware workspace in its dev container.

    Attributes:
        workspace: The ``autowarefoundation/autoware`` clone on this host.
        launch: The ``ros2 launch`` arguments (package, file, ``name:=value``
            arguments), with ``scenario_mode:=true``.  ``{bridge_address}``,
            ``{index}`` and ``{name}`` are replaced with the episode's (see
            :class:`~autoware_carla_scenario.autoware_stack.AutowareEpisode`;
            ``{name}`` with only ``A-Z a-z 0-9 . _ -``, as in the log's file
            name).
        devcontainer: The dev container variant under ``.devcontainer/`` whose
            image and workspace mount are used.
        image: The image to use instead of the dev container's; its mount and
            environment are still the dev container's, when there is one.
        workspace_mount: Where to mount the workspace instead of where the dev
            container does.  It must be where the workspace was built.
        build: When to build the workspace; see :data:`BuildPolicy`.
        rosdep: Install the workspace's dependencies (``rosdep install
            --from-paths src``) into the image made from the dev container's
            (which gets the ``scenario_bridge`` node's packages either way).
            Made once and reused until a ``package.xml`` or the dev container's
            image changes.
        bridge_parameters: ROS parameters of the ``scenario_bridge`` node
            besides its address (``auto_engage``, ``initialize_localization``,
            ``require_localization_initialized``, ...).
        colcon_args: Arguments to ``colcon build``.
        mounts: Further ``host path -> container path`` mounts.  A host path
            that does not exist is skipped with a warning (Docker would create
            it, owned by root).
        env: Variables set in the container, over the dev container's own.
            Passed to ``docker run`` by name, their values in the CLI's
            environment, so they do not show in ``ps`` (but one the CLI reads
            itself, ``PATH`` or ``DOCKER_HOST`` say, does).
        ros_domain_ids: ``ROS_DOMAIN_ID`` for each scenario in turn, cycling;
            empty keeps the dev container's.  Moving to another domain keeps a
            stack from discovering what is left of the last one before its
            participants time out.  Each is an ``int`` from 0 to 232, the
            range ROS 2 can use (0 to 101 is safe on Linux; see
            :data:`~autoware_carla_scenario.constants.MAX_ROS_DOMAIN_ID`).
        gpus: Docker's ``--gpus`` value; ``None`` for no GPU.
        user: Docker's ``--user`` value.  ``None`` leaves it to the image, which
            maps its ``aw`` user to ``HOST_UID``/``HOST_GID`` (set to this
            user's ids) so the files a build writes stay this user's.
        docker_args: Further ``docker run`` arguments.
        docker: The Docker CLI.
        stop_timeout_s: How long stopping waits after each signal, positive
            and finite.
        container_prefix: Container names start with this.

    Raises:
        ValueError: Naming the field, when ``build`` is not a
            :data:`BuildPolicy`, ``launch``, ``colcon_args`` or
            ``docker_args`` is a ``str`` (or ``bytes``) rather than a sequence
            of arguments, ``launch`` is empty, ``stop_timeout_s`` is not
            positive and finite, a ``ros_domain_ids`` entry is a boolean or
            outside 0 to 232, or ``rosdep`` is not a ``bool``.
    """

    workspace: Path
    launch: Sequence[str]
    devcontainer: str = "universe-devel-cuda"
    image: Optional[str] = None
    workspace_mount: Optional[str] = None
    build: BuildPolicy = "auto"
    rosdep: bool = True
    bridge_parameters: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_BRIDGE_PARAMETERS)
    )
    colcon_args: Sequence[str] = (
        "--symlink-install",
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=Release",
    )
    mounts: Mapping[str, str] = field(
        default_factory=lambda: {"~/autoware_data": "/home/aw/autoware_data"}
    )
    env: Mapping[str, str] = field(default_factory=dict)
    ros_domain_ids: Sequence[int] = ()
    gpus: Optional[str] = "all"
    user: Optional[str] = None
    docker_args: Sequence[str] = ()
    docker: str = "docker"
    stop_timeout_s: float = 20.0
    container_prefix: str = "acs-autoware"

    def __post_init__(self) -> None:
        if self.build not in ("auto", "always", "never"):
            raise ValueError(
                f"build must be 'auto', 'always' or 'never', not {self.build!r}"
            )
        # A str would be run a character per argument.
        _check_argument_list("launch", self.launch)
        _check_argument_list("colcon_args", self.colcon_args)
        _check_argument_list("docker_args", self.docker_args)
        if not self.launch:
            raise ValueError(
                "launch is required: the ros2 launch package, file and arguments "
                "that start Autoware in scenario mode"
            )
        # Checked here: an id out of range would only be refused inside the
        # container, by the stack's DDS, once a scenario has started.
        for domain_id in self.ros_domain_ids:
            if isinstance(domain_id, bool) or not (0 <= domain_id <= MAX_ROS_DOMAIN_ID):
                raise ValueError(
                    "ros_domain_ids must be ROS domain ids, whole numbers from 0 "
                    f"to {MAX_ROS_DOMAIN_ID}, got {domain_id}"
                )
        _check_stop_timeout_s(self.stop_timeout_s)
        # Read by truthiness (it runs rosdep and feeds the image's hash): a
        # "false" would run it.  The run's config reads its spellings
        # strictly (config_flags.check_flag) before it gets here (#1761).
        if not isinstance(self.rosdep, bool):
            raise ValueError(f"rosdep must be true or false, got {self.rosdep!r}")


class DockerAutowareLauncher:
    """Starts each scenario's Autoware stack in a container of its own.

    Satisfies :class:`~autoware_carla_scenario.autoware_stack.AutowareLauncher`.

    Args:
        config: How to build and start the workspace.
        log_dir: Where the build's and each stack's output goes; a ``log_dir``
            passed to :meth:`prepare` replaces it.
    """

    _config: DockerAutowareConfig
    _workspace: Path
    _log_dir: Optional[Path]
    _container: Optional[DevContainer]
    _process: Optional[ProcessGroup]
    _container_name: Optional[str]
    _prepared: bool
    _episodes: int

    def __init__(
        self, config: DockerAutowareConfig, *, log_dir: Optional[Path] = None
    ) -> None:
        self._config = config
        self._workspace = config.workspace.expanduser().resolve()
        self._log_dir = log_dir
        self._container = None
        self._process = None
        self._container_name = None
        self._prepared = False
        self._episodes = 0

    @property
    def config(self) -> DockerAutowareConfig:
        """How the workspace is built and started."""
        return self._config

    # ------------------------------------------------------------------
    # AutowareLauncher
    # ------------------------------------------------------------------

    def prepare(self, log_dir: Optional[Path] = None) -> None:
        if log_dir is not None:
            self._log_dir = log_dir
        if self._prepared:
            return
        if not (self._workspace / "src").is_dir():
            raise AutowareStackError(
                f"{self._workspace} is not an Autoware workspace (it has no src/)"
            )
        self._container = self._resolve_container()
        logger.info(
            "Autoware workspace %s: image %s, mounted at %s",
            self._workspace,
            self._container.image,
            self._container.workspace_mount,
        )
        self.remove_leftover_containers()
        build = self._config.build == "always" or (
            self._config.build == "auto" and not self._is_built()
        )
        if build or self._is_built():
            base = self._container.image
            tag = self._dependency_tag(base)
            if not build and self._has_image(tag):
                # Nothing to make: no lock to wait for.
                logger.info("Workspace dependencies: %s", tag)
                self._container = replace(self._container, image=tag)
            else:
                lock = self._lock_workspace()
                try:
                    # Asked again: another run may have made it meanwhile.
                    self._container = replace(
                        self._container, image=self._dependency_image(base, tag)
                    )
                    # Or built the workspace, while this one waited.
                    if build and self._config.build == "auto" and self._is_built():
                        build = False
                    if build:
                        self._build()
                finally:
                    self._unlock_workspace(lock)
        if not self._is_built():
            raise AutowareStackError(
                f"{self._workspace} has not been built ({self._setup_script()} is "
                "missing). Build it, or set build to 'auto'."
            )
        if not self.has_scenario_support():
            raise AutowareStackError(
                f"{self._workspace} was built without scenario support: the launch "
                f"file of {_INTERFACE_PACKAGE} takes no scenario_mode, and neither "
                "does the configured launch file. Build a branch that has it."
            )
        self._prepared = True

    def start(self, *, name: str, bridge_address: str) -> AutowareEpisode:
        if self._process is not None:
            raise AutowareStackError(
                "An Autoware stack is already running; stop it before starting another"
            )
        self.prepare()
        episode = AutowareEpisode(self._episodes, name, bridge_address)
        self._episodes += 1
        assert self._container is not None  # noqa: S101 - set by prepare()
        container = f"{self._config.container_prefix}-{os.getpid()}-{episode.index:03d}"
        launch = [fill_placeholders(arg, episode) for arg in self._config.launch]
        if TYPE_CHECKING:
            # bridge_command() takes the ROS parameters as `Mapping[str, Any]`,
            # which Codon cannot express (typecheck/library.py: UNCALLED).
            bridge = _BRIDGE_PYTHON_ROOT
        else:
            bridge = bridge_command(
                _BRIDGE_PYTHON_ROOT,
                episode.bridge_address,
                self._config.bridge_parameters,
            )
        # The bridge node runs beside the launch, in the background: it goes
        # with the container when the launch ends.
        script = (
            f"source {shlex.quote(self._setup_script_in_container())} && "
            f"({bridge} &) && "
            f"exec ros2 launch {shlex.join(launch)}"
        )
        extra_env = {"AUTOWARE_BRIDGE_ADDRESS": episode.bridge_address}
        if self._config.ros_domain_ids:
            ids = self._config.ros_domain_ids
            extra_env["ROS_DOMAIN_ID"] = str(ids[episode.index % len(ids)])
        mount = f"{_BRIDGE_PYTHON_ROOT}/autoware_carla_scenario/autoware_bridge"
        argv = [
            *self._run_argv(
                name=container,
                extra_env=extra_env,
                volumes={str(bridge_package_dir()): f"{mount}:ro"},
            ),
            "bash",
            "-c",
            script,
        ]
        logger.info(
            "Starting Autoware for %s in container %s: ros2 launch %s",
            episode.name,
            container,
            shlex.join(launch),
        )
        self._container_name = container
        # SIGINT reaches ros2 launch through the Docker CLI's signal proxy, and
        # it shuts its nodes down on that; the container is removed afterwards
        # whatever happened.
        self._process = ProcessGroup(
            argv,
            env=self._run_env(extra_env),
            log_path=episode_log_path(self._log_dir, episode),
            stop_timeout_s=self._config.stop_timeout_s,
        )
        return episode

    def poll(self) -> Optional[int]:
        if self._process is None:
            return None
        return self._process.poll()

    def stop(self) -> None:
        if self._process is not None:
            # SIGINT, through the CLI's signal proxy, is ros2 launch's cue to
            # shut its nodes down.  Past that, removing the container is what
            # stops everything in it: signalling the CLI harder does not.
            code = self._process.stop(escalate=False)
            logger.info("Autoware stack stopped (exit code %s)", code)
            self._process = None
        if self._container_name is not None:
            self._docker("rm", "--force", self._container_name, check=False)
            self._container_name = None

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def has_scenario_support(self) -> bool:
        """Whether the workspace can follow a scenario's world, ego and clock.

        True when the built ``autoware_carla_interface`` launch takes
        ``scenario_mode``, or when the configured launch file itself declares
        it -- a launch that brings up Autoware and another CARLA interface
        (``autoware_carla_interface_v2``'s ``autoware_carla_scenario.launch.xml``,
        say) in scenario mode.
        """
        candidates = [
            self._installed_launch(
                _INTERFACE_PACKAGE, f"{_INTERFACE_PACKAGE}.launch.xml"
            )
        ]
        launch_args = list(self._config.launch)
        if (
            len(launch_args) >= 2
            and ":=" not in launch_args[0]
            and ":=" not in launch_args[1]
        ):
            candidates.append(self._installed_launch(launch_args[0], launch_args[1]))
        for launch in candidates:
            try:
                if 'name="scenario_mode"' in self._on_host(launch).read_text():
                    return True
            except OSError:
                continue
        return False

    def _installed_launch(self, package: str, launch_file: str) -> Path:
        return (
            self._workspace
            / "install"
            / package
            / "share"
            / package
            / "launch"
            / launch_file
        )

    def _on_host(self, path: Path) -> Path:
        """*path*, its links into the workspace's mount followed on this host.

        A ``--symlink-install`` build links ``install/`` to the sources by
        their paths in the container, which do not exist here.
        """
        mount = (
            self._container.workspace_mount
            if self._container is not None
            else DEFAULT_WORKSPACE_MOUNT
        )
        for _ in range(40):  # A link chain, not a loop.
            if not path.is_symlink():
                return path
            target = Path(os.readlink(path))
            if not target.is_absolute():
                target = path.parent / target
            elif target.is_relative_to(mount):
                target = self._workspace / target.relative_to(mount)
            path = target
        return path

    def remove_leftover_containers(self) -> None:
        """Remove the containers this user's launchers started whose process is gone.

        A run killed before its scenario ended leaves its stack running -- and
        its nodes publishing into the next run's ROS graph.  Containers carry
        the user's id in their label, so another user's runs are left alone,
        and the host and pid of the process that started them
        (:data:`OWNER_LABEL`), so the live stack of another run of this
        user's (a SLURM task beside this one) is left alone too: only those
        whose process no longer runs on this host are removed.  One without
        an owner (started by a version before the label, or by another host
        whose processes cannot be seen from here) is removed only once it
        runs nothing (``exited``, ``created``, ``dead``), and only named in a
        warning otherwise.  The owner is looked up in this process's pid
        namespace (see :data:`OWNER_LABEL`).
        """
        listed = self._docker(
            "ps",
            "--all",
            "--filter",
            f"label={STACK_LABEL}={_label_value()}",
            "--format",
            '{{.ID}}\t{{.State}}\t{{.Label "' + OWNER_LABEL + '"}}',
            check=False,
        )
        ids: list[str] = []
        unknown: list[str] = []
        for line in listed.stdout.splitlines():
            fields = line.split("\t")
            container = fields[0].strip()
            if not container:
                continue
            state = ""
            if len(fields) > 1:
                state = fields[1].strip()
            owner = ""
            if len(fields) > 2:
                owner = fields[2].strip()
            alive = _owner_alive(owner)
            if alive is None:
                if state in _STOPPED_STATES:
                    ids.append(container)
                else:
                    unknown.append(container)
            elif not alive:
                ids.append(container)
        if unknown:
            logger.warning(
                "Leaving %d Autoware container(s) still up whose process cannot be "
                "told alive or not (no %s label, or another host's): %s. Remove "
                "them with `docker rm --force` if no run of yours uses them.",
                len(unknown),
                OWNER_LABEL,
                " ".join(unknown),
            )
        if ids:
            logger.warning("Removing %d leftover Autoware container(s)", len(ids))
            # A list, not `*ids`: Codon 0.19 unpacks only a tuple into a call.
            self._run_docker(["rm", "--force"] + ids, check=False)

    def _resolve_container(self) -> DevContainer:
        config = self._config
        try:
            container = read_devcontainer(self._workspace, config.devcontainer)
        except ValueError as exc:
            if config.image is None:
                raise AutowareStackError(str(exc)) from exc
            # An image named outright needs no dev container to come from.
            container = DevContainer(image=config.image)
        image = container.image
        if config.image:
            image = config.image
        workspace_mount = container.workspace_mount
        if config.workspace_mount:
            workspace_mount = config.workspace_mount
        return DevContainer(
            image=image,
            workspace_mount=workspace_mount,
            environment=container.environment,
        )

    def _lock_workspace(self) -> int:
        """Wait for, and take, the workspace's build lock; its descriptor.

        Raises:
            AutowareStackError: Another run held it for longer than
                :data:`_BUILD_LOCK_TIMEOUT_S`.
        """
        import fcntl  # noqa: PLC0415
        import time  # noqa: PLC0415

        path = self._workspace / _BUILD_LOCK_NAME
        try:
            # Read-only is enough for flock, and lets an existing file
            # on a read-only mount lock too.
            # 0o644: Codon has no octal literal.
            fd = os.open(path, os.O_RDONLY | os.O_CREAT, 420)
        except OSError as exc:
            logger.warning(
                "Building %s without a lock (%s): another run may build it "
                "at the same time",
                self._workspace,
                exc,
            )
            return -1
        # Read here, not as a default: a default is bound once.
        timeout = _BUILD_LOCK_TIMEOUT_S
        deadline = time.monotonic() + timeout
        waiting = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                pass
            except OSError as exc:
                os.close(fd)
                logger.warning(
                    "Building %s without a lock (%s): another run may build "
                    "it at the same time",
                    self._workspace,
                    exc,
                )
                return -1
            if time.monotonic() >= deadline:
                os.close(fd)
                raise AutowareStackError(
                    f"Another run has been building {self._workspace} for "
                    f"more than {timeout:.0f} s (it holds {path}); not "
                    "building it beside it"
                )
            if not waiting:
                logger.info(
                    "Waiting for another run to finish building %s (%s)",
                    self._workspace,
                    path,
                )
                waiting = True
            time.sleep(_BUILD_LOCK_POLL_S)

    def _unlock_workspace(self, fd: int) -> None:
        """Let the build lock *fd* (from :meth:`_lock_workspace`) go."""
        if fd >= 0:
            os.close(fd)

    def _setup_script(self) -> Path:
        return self._workspace / "install" / "setup.bash"

    def _setup_script_in_container(self) -> str:
        assert self._container is not None  # noqa: S101 - set by prepare()
        return f"{self._container.workspace_mount}/install/setup.bash"

    def _is_built(self) -> bool:
        return self._setup_script().is_file()

    def _dependency_digest(self, base_image: str) -> str:
        """What the dependency image is made from: the base image, the bridge's
        packages and, with ``rosdep``, every package.xml."""
        listed = self._docker(
            "image", "inspect", "--format", "{{.Id}}", base_image, check=True
        )
        digest = hashlib.sha256(listed.stdout.strip().encode())
        digest.update(" ".join(BRIDGE_APT_PACKAGES).encode())
        if not self._config.rosdep:
            return digest.hexdigest()[:16]
        digest.update(b"rosdep\0")
        source = self._workspace / "src"
        for path in sorted(source.rglob("package.xml")):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    def _dependency_tag(self, base_image: str) -> str:
        """The tag of the dependency image made from *base_image*."""
        return f"{DEPENDENCY_IMAGE_REPOSITORY}:{self._dependency_digest(base_image)}"

    def _has_image(self, tag: str) -> bool:
        """Whether Docker has the image *tag*."""
        # A non-zero exit is "no such image"; no answer at all is not, and
        # read as one it would reinstall and recommit an image that may well
        # exist, through an install that is not bounded (#1797).
        inspected = self._docker(
            "image", "inspect", tag, check=False, timeout_is_error=True
        )
        return inspected.returncode == 0

    def _dependency_image(self, base_image: str, tag: Optional[str] = None) -> str:
        """The image with what the stack needs installed, made if missing.

        That is the ``scenario_bridge`` node's packages and, with ``rosdep``,
        the workspace's dependencies.  *tag* is :meth:`_dependency_tag`'s,
        when the caller has it.
        """
        if tag is None:
            tag = self._dependency_tag(base_image)
        if self._has_image(tag):
            logger.info("Workspace dependencies: %s", tag)
            return tag
        name = f"{self._config.container_prefix}-{os.getpid()}-deps"
        # The image's aw user has password-less sudo, which rosdep uses for
        # apt; unresolvable keys (-r) are logged, not fatal: the dev container
        # runs with them missing as well.
        script = (
            "sudo apt-get update && "
            f"sudo apt-get install -y --no-install-recommends {shlex.join(BRIDGE_APT_PACKAGES)}"
        )
        if self._config.rosdep:
            script += (
                ' && rosdep update --rosdistro "${ROS_DISTRO}" && '
                f"cd {shlex.quote(self._container_mount())} && "
                "rosdep install -y -r --from-paths src --ignore-src "
                '--rosdistro "${ROS_DISTRO}"'
            )
        argv = [
            *self._run_argv(
                name=name,
                extra_env={"DEBIAN_FRONTEND": "noninteractive"},
                remove=False,
                image=base_image,
            ),
            "bash",
            "-c",
            script,
        ]
        logger.info(
            "Installing what the stack of %s needs into %s, once",
            self._workspace,
            tag,
        )
        try:
            code = self._stream(
                argv,
                "autoware-deps.log",
                self._run_env({"DEBIAN_FRONTEND": "noninteractive"}),
            )
            if code != 0:
                raise AutowareStackError(
                    f"Installing the dependencies of {self._workspace} failed "
                    f"(exit code {code}); see autoware-deps.log"
                )
            self._docker(
                "commit",
                "--change",
                'CMD ["/bin/bash"]',
                "--change",
                f"LABEL {STACK_LABEL}=",
                name,
                tag,
                check=True,
                timeout_s=_DOCKER_COMMIT_TIMEOUT_S,
            )
        finally:
            self._docker("rm", "--force", name, check=False)
        return tag

    def _container_mount(self) -> str:
        assert self._container is not None  # noqa: S101 - set by _resolve_container()
        return self._container.workspace_mount

    def _stream(
        self,
        argv: Sequence[str],
        log_name: str,
        env: Optional[dict[str, str]] = None,
    ) -> int:
        """Run *argv*, its output to this process's and to *log_name* in the log dir.

        *env* is its environment (:meth:`_run_env`); ``None`` is this process's.
        """
        log_path: Optional[Path] = None
        if self._log_dir is not None:
            log_path = self._log_dir / log_name
        if log_path is not None:
            logger.info("Output in %s", log_path)
        log: Optional[IO[str]] = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
        try:
            process: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603 - argv comes from the user's own config
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # gcc echoes source lines, any bytes: replace what is not
                # UTF-8 rather than abort the build mid-stream (#1346).
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None  # noqa: S101 - stdout=PIPE
            for line in process.stdout:
                sys.stdout.write(line)
                if log is not None:
                    log.write(line)
            return process.wait()
        except OSError as exc:
            raise AutowareStackError(
                f"Could not run {self._config.docker!r}: {exc}"
            ) from exc
        finally:
            if log is not None:
                log.close()

    def _build(self) -> None:
        assert self._container is not None  # noqa: S101 - set by _resolve_container()
        name = f"{self._config.container_prefix}-{os.getpid()}-build"
        script = (
            'source "/opt/ros/${ROS_DISTRO}/setup.bash" && '
            f"cd {shlex.quote(self._container.workspace_mount)} && "
            f"colcon build {shlex.join(self._config.colcon_args)}"
        )
        argv = [*self._run_argv(name=name, extra_env={}), "bash", "-c", script]
        logger.info(
            "Building %s in %s (colcon build %s)",
            self._workspace,
            self._container.image,
            shlex.join(self._config.colcon_args),
        )
        try:
            code = self._stream(argv, "autoware-build.log", self._run_env({}))
        finally:
            self._docker("rm", "--force", name, check=False)
        if code != 0:
            raise AutowareStackError(
                f"colcon build failed in {self._workspace} (exit code {code})"
            )

    # ------------------------------------------------------------------
    # Docker
    # ------------------------------------------------------------------

    def _run_argv(
        self,
        *,
        name: str,
        extra_env: dict[str, str],
        remove: bool = True,
        image: Optional[str] = None,
        volumes: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """``docker run`` up to the image, for a container named *name*.

        *remove* is ``--rm``; *image* replaces the one the stack runs in;
        *volumes* are further ``host: container[:options]`` mounts.  The
        ``autoware.launcher.env`` variables of :meth:`_named_env` are passed
        by name: the CLI is run with :meth:`_run_env` (*extra_env* the same).
        """
        assert self._container is not None  # noqa: S101 - set by prepare()
        config = self._config
        env: dict[str, str] = dict(self._container.environment)
        # The dev container's display belongs to a desktop session, not to a
        # batch run; RViz and friends are the launch's to turn off.
        if "DISPLAY" in env:
            del env["DISPLAY"]
        env["HOST_UID"] = str(os.getuid())
        env["HOST_GID"] = str(os.getgid())
        env.update(config.env)
        env.update(extra_env)

        argv = [
            config.docker,
            "run",
            *(["--rm"] if remove else []),
            "--name",
            name,
            "--label",
            f"{STACK_LABEL}={_label_value()}",
            "--label",
            f"{OWNER_LABEL}={_owner_value()}",
            # tini as PID 1 hands SIGINT on to ros2 launch whatever the image's
            # entrypoint does with it.
            "--init",
            # The bridge, CARLA and the ROS graph are all reached on the host's
            # network, as the dev container reaches them.
            "--network",
            "host",
            "--workdir",
            self._container.workspace_mount,
            "--volume",
            f"{self._workspace}:{self._container.workspace_mount}",
        ]
        for host, target in config.mounts.items():
            host_path = Path(host).expanduser()
            if not host_path.exists():
                logger.warning("Not mounting %s: it does not exist", host_path)
                continue
            argv += ["--volume", f"{host_path.resolve()}:{target}"]
        if volumes:
            for host, target in volumes.items():
                argv += ["--volume", f"{host}:{target}"]
        if config.gpus is not None:
            argv += ["--gpus", config.gpus]
        if config.user is not None:
            argv += ["--user", config.user]
        named = self._named_env(extra_env)
        for key, value in env.items():
            if key in named:
                argv += ["--env", key]
            else:
                argv += ["--env", f"{key}={value}"]
        argv += list(config.docker_args)
        if image:
            argv.append(image)
        else:
            argv.append(self._container.image)
        return argv

    def _named_env(self, extra_env: dict[str, str]) -> dict[str, str]:
        """The ``autoware.launcher.env`` variables passed to ``docker run`` by
        name, their values in the CLI's environment (#1961): on its command
        line, any user of the host could read them (``ps``).

        Not one *extra_env* overrides, nor one the CLI would heed itself
        (:func:`_shapes_the_cli`).
        """
        named: dict[str, str] = {}
        for key, value in self._config.env.items():
            if key not in extra_env and not _shapes_the_cli(key):
                named[key] = value
        return named

    def _run_env(self, extra_env: dict[str, str]) -> Optional[dict[str, str]]:
        """The environment of the CLI running :meth:`_run_argv`'s command,
        with the same *extra_env*: this process's, with the variables passed
        by name; ``None`` (this process's) when there are none."""
        named = self._named_env(extra_env)
        if len(named) == 0:
            return None
        env = os.environ.copy()
        env.update(named)
        return env

    def _docker(
        self,
        *args: str,
        check: bool,
        timeout_s: Optional[float] = None,
        timeout_is_error: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return self._run_docker(
            list(args),
            check=check,
            timeout_s=timeout_s,
            timeout_is_error=timeout_is_error,
        )

    def _run_docker(
        self,
        args: list[str],
        *,
        check: bool,
        timeout_s: Optional[float] = None,
        timeout_is_error: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """The Docker CLI run with *args*, its output captured, within
        *timeout_s* seconds (:data:`_DOCKER_TIMEOUT_S` when None).

        A call that fails to run or to finish raises
        :class:`AutowareStackError` with *check*; without, it is logged and
        answers as a failed one (exit status 1).  *timeout_is_error* raises
        on a call that does not finish even without *check*: for a caller
        that reads a non-zero exit as an answer ("no such image"), which no
        answer at all must not pass for.
        """
        # Read here, not as a default: a default is bound once, at definition.
        timeout = _DOCKER_TIMEOUT_S
        if timeout_s is not None:
            timeout = timeout_s
        try:
            completed = subprocess.run(  # noqa: S603 - argv comes from the user's own config
                [self._config.docker] + args,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run() has killed the docker CLI by now.
            if check or timeout_is_error:
                raise AutowareStackError(
                    f"{self._config.docker} {' '.join(args)} did not finish "
                    f"within {timeout:.0f} s"
                ) from None
            logger.warning(
                "%s %s did not finish within %.0f s; going on without it",
                self._config.docker,
                " ".join(args),
                timeout,
            )
            return subprocess.CompletedProcess([self._config.docker] + args, 1, "", "")
        except OSError as exc:
            if check:
                raise AutowareStackError(
                    f"Could not run {self._config.docker!r}: {exc}"
                ) from exc
            logger.warning("Could not run %s: %s", self._config.docker, exc)
            return subprocess.CompletedProcess([self._config.docker] + args, 1, "", "")
        if check and completed.returncode != 0:
            raise AutowareStackError(
                f"{self._config.docker} {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed
