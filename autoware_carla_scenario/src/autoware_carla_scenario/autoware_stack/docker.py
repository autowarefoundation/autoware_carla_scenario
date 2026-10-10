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
                "bridge_address:={bridge_address}",
            ),
        )
    )

A workspace whose ``autoware_carla_interface`` has no ``scenario_bridge`` --
one built from a branch without scenario support -- cannot run a scenario at
all, and is refused by :meth:`DockerAutowareLauncher.prepare` before any
scenario starts.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .devcontainer import DevContainer, read_devcontainer
from .launcher import (
    AutowareEpisode,
    AutowareStackError,
    ProcessGroup,
    episode_log_path,
    fill_placeholders,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BuildPolicy",
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

#: The package that hosts the scenario bridge (autoware_universe#13319).
_INTERFACE_PACKAGE = "autoware_carla_interface"


@dataclass(frozen=True)
class DockerAutowareConfig:
    """How to build and start an Autoware workspace in its dev container.

    Attributes:
        workspace: The ``autowarefoundation/autoware`` clone on this host.
        launch: The ``ros2 launch`` arguments (package, file, ``name:=value``
            arguments).  ``{bridge_address}``, ``{index}`` and ``{name}`` are
            replaced with the episode's (see
            :class:`~autoware_carla_scenario.autoware_stack.AutowareEpisode`):
            the launch has to hand ``{bridge_address}`` to its
            ``scenario_bridge`` node.
        devcontainer: The dev container variant under ``.devcontainer/`` whose
            image and workspace mount are used.
        image: The image to use instead of the dev container's.
        workspace_mount: Where to mount the workspace instead of where the dev
            container does.  It must be where the workspace was built.
        build: When to build the workspace; see :data:`BuildPolicy`.
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
                "that start Autoware with its scenario_bridge"
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
        if self._config.build == "always" or (
            self._config.build == "auto" and not self._is_built()
        ):
            self._build()
        if not self._is_built():
            raise AutowareStackError(
                f"{self._workspace} has not been built ({self._setup_script()} is "
                "missing). Build it, or set build to 'auto'."
            )
        if not self.has_scenario_support():
            raise AutowareStackError(
                f"{self._workspace} was built without scenario support: "
                f"{_INTERFACE_PACKAGE} has no scenario_bridge. Build a branch that "
                "has it (autowarefoundation/autoware_universe#13319)."
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
        script = (
            f"source {shlex.quote(self._setup_script_in_container())} && "
            f"exec ros2 launch {shlex.join(launch)}"
        )
        extra_env = {"AUTOWARE_BRIDGE_ADDRESS": episode.bridge_address}
        if self._config.ros_domain_ids:
            ids = self._config.ros_domain_ids
            extra_env["ROS_DOMAIN_ID"] = str(ids[episode.index % len(ids)])
        argv = [
            *self._run_argv(name=container, extra_env=extra_env),
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
            code = self._process.stop()
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
        """Whether the built ``autoware_carla_interface`` has a ``scenario_bridge``."""
        package = self._workspace / "install" / _INTERFACE_PACKAGE
        if not package.is_dir():
            return False
        return any(
            path.name.startswith("scenario_bridge") for path in package.rglob("*")
        )

    def remove_leftover_containers(self) -> None:
        """Remove every container a launcher started and nothing removed.

        A run killed before its scenario ended leaves its stack running -- and
        its nodes publishing into the next run's ROS graph.
        """
        listed = self._docker(
            "ps", "--all", "--quiet", "--filter", f"label={STACK_LABEL}", check=False
        )
        ids = listed.stdout.split()
        if ids:
            logger.warning("Removing %d leftover Autoware container(s)", len(ids))
            self._docker("rm", "--force", *ids, check=False)

    def _resolve_container(self) -> DevContainer:
        config = self._config
        if config.image is not None:
            container = DevContainer(image=config.image)
        else:
            try:
                container = read_devcontainer(self._workspace, config.devcontainer)
            except ValueError as exc:
                raise AutowareStackError(str(exc)) from exc
        if config.workspace_mount is not None:
            container = DevContainer(
                image=container.image,
                workspace_mount=config.workspace_mount,
                environment=container.environment,
            )
        return container

    def _setup_script(self) -> Path:
        return self._workspace / "install" / "setup.bash"

    def _setup_script_in_container(self) -> str:
        assert self._container is not None  # noqa: S101 - set by prepare()
        return f"{self._container.workspace_mount}/install/setup.bash"

    def _is_built(self) -> bool:
        return self._setup_script().is_file()

    def _build(self) -> None:
        assert self._container is not None  # noqa: S101 - set by _resolve_container()
        name = f"{self._config.container_prefix}-{os.getpid()}-build"
        script = (
            'source "/opt/ros/${ROS_DISTRO}/setup.bash" && '
            f"cd {shlex.quote(self._container.workspace_mount)} && "
            f"colcon build {shlex.join(self._config.colcon_args)}"
        )
        argv = [*self._run_argv(name=name, extra_env={}), "bash", "-c", script]
        log_path = (
            None if self._log_dir is None else self._log_dir / "autoware-build.log"
        )
        logger.info(
            "Building %s in %s (colcon build %s)%s",
            self._workspace,
            self._container.image,
            shlex.join(self._config.colcon_args),
            "" if log_path is None else f"; output in {log_path}",
        )
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
            code = process.wait()
        except OSError as exc:
            raise AutowareStackError(
                f"Could not run {self._config.docker!r}: {exc}"
            ) from exc
        finally:
            if log is not None:
                log.close()
            self._docker("rm", "--force", name, check=False)
        if code != 0:
            raise AutowareStackError(
                f"colcon build failed in {self._workspace} (exit code {code})"
            )

    # ------------------------------------------------------------------
    # Docker
    # ------------------------------------------------------------------

    def _run_argv(self, *, name: str, extra_env: Mapping[str, str]) -> list[str]:
        """``docker run`` up to the image, for a container named *name*."""
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
            "--rm",
            "--name",
            name,
            "--label",
            f"{STACK_LABEL}=autoware",
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
        if config.gpus is not None:
            argv += ["--gpus", config.gpus]
        if config.user is not None:
            argv += ["--user", config.user]
        for key, value in env.items():
            argv += ["--env", f"{key}={value}"]
        argv += list(config.docker_args)
        argv.append(self._container.image)
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
