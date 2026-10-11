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
starts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional

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
    episode_log_path,
    fill_placeholders,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BuildPolicy",
    "DEPENDENCY_IMAGE_REPOSITORY",
    "DockerAutowareConfig",
    "DockerAutowareLauncher",
    "STACK_LABEL",
]

#: Every container this launcher starts carries this label, which is how one
#: left behind by a run that was killed is found and removed by the next.
STACK_LABEL: str = "autoware_carla_scenario.stack"

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


def _label_value() -> str:
    """The value of :data:`STACK_LABEL` on this user's containers."""
    return f"uid-{os.getuid()}"


@dataclass(frozen=True)
class DockerAutowareConfig:
    """How to build and start an Autoware workspace in its dev container.

    Attributes:
        workspace: The ``autowarefoundation/autoware`` clone on this host.
        launch: The ``ros2 launch`` arguments (package, file, ``name:=value``
            arguments), with ``scenario_mode:=true``.  ``{bridge_address}``,
            ``{index}`` and ``{name}`` are replaced with the episode's (see
            :class:`~autoware_carla_scenario.autoware_stack.AutowareEpisode`).
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
        ros_domain_ids: ``ROS_DOMAIN_ID`` for each scenario in turn, cycling;
            empty keeps the dev container's.  Moving to another domain keeps a
            stack from discovering what is left of the last one before its
            participants time out.
        gpus: Docker's ``--gpus`` value; ``None`` for no GPU.
        user: Docker's ``--user`` value.  ``None`` leaves it to the image, which
            maps its ``aw`` user to ``HOST_UID``/``HOST_GID`` (set to this
            user's ids) so the files a build writes stay this user's.
        docker_args: Further ``docker run`` arguments.
        docker: The Docker CLI.
        stop_timeout_s: How long stopping waits after each signal.
        container_prefix: Container names start with this.
    """

    workspace: Path
    launch: Sequence[str]
    devcontainer: str = "universe-devel-cuda"
    image: Optional[str] = None
    workspace_mount: Optional[str] = None
    build: BuildPolicy = "auto"
    rosdep: bool = True
    bridge_parameters: Mapping[str, object] = field(
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
        if not self.launch:
            raise ValueError(
                "launch is required: the ros2 launch package, file and arguments "
                "that start Autoware in scenario mode"
            )


class DockerAutowareLauncher:
    """Starts each scenario's Autoware stack in a container of its own.

    Satisfies :class:`~autoware_carla_scenario.autoware_stack.AutowareLauncher`.

    Args:
        config: How to build and start the workspace.
        log_dir: Where the build's and each stack's output goes; a ``log_dir``
            passed to :meth:`prepare` replaces it.
    """

    def __init__(
        self, config: DockerAutowareConfig, *, log_dir: Optional[Path] = None
    ) -> None:
        self._config = config
        self._workspace = config.workspace.expanduser().resolve()
        self._log_dir = log_dir
        self._container: Optional[DevContainer] = None
        self._process: Optional[ProcessGroup] = None
        self._container_name: Optional[str] = None
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
            self._container = replace(
                self._container, image=self._dependency_image(self._container.image)
            )
        if build:
            self._build()
        if not self._is_built():
            raise AutowareStackError(
                f"{self._workspace} has not been built ({self._setup_script()} is "
                "missing). Build it, or set build to 'auto'."
            )
        if not self.has_scenario_support():
            raise AutowareStackError(
                f"{self._workspace} was built without scenario support: the launch "
                f"file of {_INTERFACE_PACKAGE} takes no scenario_mode. Build a "
                "branch that has it."
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
        bridge = bridge_command(
            _BRIDGE_PYTHON_ROOT, episode.bridge_address, self._config.bridge_parameters
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
            log_path=episode_log_path(self._log_dir, episode),
            stop_timeout_s=self._config.stop_timeout_s,
        )
        return episode

    def poll(self) -> Optional[int]:
        return None if self._process is None else self._process.poll()

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
        """Whether the built ``autoware_carla_interface`` launch takes ``scenario_mode``."""
        launch = (
            self._workspace
            / "install"
            / _INTERFACE_PACKAGE
            / "share"
            / _INTERFACE_PACKAGE
            / "launch"
            / f"{_INTERFACE_PACKAGE}.launch.xml"
        )
        try:
            return 'name="scenario_mode"' in self._on_host(launch).read_text()
        except OSError:
            return False

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
        """Remove every container this user's launchers started and nothing removed.

        A run killed before its scenario ended leaves its stack running -- and
        its nodes publishing into the next run's ROS graph.  Containers carry
        the user's id in their label, so another user's runs are left alone;
        two batches of one user's on one host are not supported.
        """
        listed = self._docker(
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label={STACK_LABEL}={_label_value()}",
            check=False,
        )
        ids = listed.stdout.split()
        if ids:
            logger.warning("Removing %d leftover Autoware container(s)", len(ids))
            self._docker("rm", "--force", *ids, check=False)

    def _resolve_container(self) -> DevContainer:
        config = self._config
        try:
            container = read_devcontainer(self._workspace, config.devcontainer)
        except ValueError as exc:
            if config.image is None:
                raise AutowareStackError(str(exc)) from exc
            # An image named outright needs no dev container to come from.
            container = DevContainer(image=config.image)
        return DevContainer(
            image=config.image or container.image,
            workspace_mount=config.workspace_mount or container.workspace_mount,
            environment=container.environment,
        )

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

    def _dependency_image(self, base_image: str) -> str:
        """The image with what the stack needs installed, made if missing.

        That is the ``scenario_bridge`` node's packages and, with ``rosdep``,
        the workspace's dependencies.
        """
        tag = f"{DEPENDENCY_IMAGE_REPOSITORY}:{self._dependency_digest(base_image)}"
        if self._docker("image", "inspect", tag, check=False).returncode == 0:
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
            code = self._stream(argv, "autoware-deps.log")
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
            )
        finally:
            self._docker("rm", "--force", name, check=False)
        return tag

    def _container_mount(self) -> str:
        assert self._container is not None  # noqa: S101 - set by _resolve_container()
        return self._container.workspace_mount

    def _stream(self, argv: Sequence[str], log_name: str) -> int:
        """Run *argv*, its output to this process's and to *log_name* in the log dir."""
        log_path = None if self._log_dir is None else self._log_dir / log_name
        if log_path is not None:
            logger.info("Output in %s", log_path)
        log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
        try:
            process = subprocess.Popen(  # noqa: S603 - argv comes from the user's own config
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
            code = self._stream(argv, "autoware-build.log")
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
        extra_env: Mapping[str, str],
        remove: bool = True,
        image: Optional[str] = None,
        volumes: Optional[Mapping[str, str]] = None,
    ) -> list[str]:
        """``docker run`` up to the image, for a container named *name*.

        *remove* is ``--rm``; *image* replaces the one the stack runs in;
        *volumes* are further ``host: container[:options]`` mounts.
        """
        assert self._container is not None  # noqa: S101 - set by prepare()
        config = self._config
        env: dict[str, str] = dict(self._container.environment)
        # The dev container's display belongs to a desktop session, not to a
        # batch run; RViz and friends are the launch's to turn off.
        env.pop("DISPLAY", None)
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
        for host, target in (volumes or {}).items():
            argv += ["--volume", f"{host}:{target}"]
        if config.gpus is not None:
            argv += ["--gpus", config.gpus]
        if config.user is not None:
            argv += ["--user", config.user]
        for key, value in env.items():
            argv += ["--env", f"{key}={value}"]
        argv += list(config.docker_args)
        argv.append(image or self._container.image)
        return argv

    def _docker(self, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(  # noqa: S603 - argv comes from the user's own config
                [self._config.docker, *args],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            if check:
                raise AutowareStackError(
                    f"Could not run {self._config.docker!r}: {exc}"
                ) from exc
            logger.warning("Could not run %s: %s", self._config.docker, exc)
            return subprocess.CompletedProcess([self._config.docker, *args], 1, "", "")
        if check and completed.returncode != 0:
            raise AutowareStackError(
                f"{self._config.docker} {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed
