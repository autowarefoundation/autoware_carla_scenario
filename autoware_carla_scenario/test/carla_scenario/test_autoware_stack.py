"""Tests for :mod:`autoware_carla_scenario.autoware_stack`.

No Docker and no Autoware: the dev container is a configuration written to a
temporary workspace, the Docker CLI is a script that records what it was asked
and plays a container, and a stack is a plain process group.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from autoware_carla_scenario.autoware_stack.bridge_node import (
    bridge_package_dir,
    host_python_root,
)
from autoware_carla_scenario.autoware_stack import (
    DEPENDENCY_IMAGE_REPOSITORY,
    STACK_LABEL,
    AutowareLauncher,
    AutowareStackError,
    CommandAutowareLauncher,
    DevContainerError,
    DockerAutowareConfig,
    DockerAutowareLauncher,
    ProcessGroup,
    read_devcontainer,
)


def _wait_for(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A child nothing has reaped yet is gone for every purpose here.
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Dev container
# ---------------------------------------------------------------------------

_DEVCONTAINER_JSON = """\
// The dev container, as autowarefoundation/autoware writes it: JSONC.
{
  "name": "autoware:universe-devel-cuda",
  /* Compose holds the image. */
  "dockerComposeFile": "../../docker/devcontainer/universe-devel-cuda.compose.yaml",
  "service": "autoware-devel",
  "remoteUser": "aw", // VS Code connects as aw
  "postCreateCommand": "echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc",
}
"""

_COMPOSE_YAML = """\
services:
  autoware-devel:
    image: ghcr.io/autowarefoundation/autoware:universe-devel-cuda-${ROS_DISTRO:-jazzy}
    network_mode: host
    environment:
      - HOST_UID=${HOST_UID:-1000}
      - ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
      - CMAKE_CUDA_ARCHITECTURES=86;89
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ../..:/home/aw/autoware
    working_dir: /home/aw/autoware
"""


def _workspace(tmp_path: Path, *, built: bool = False, scenario: bool = False) -> Path:
    ws = tmp_path / "autoware"
    (ws / "src").mkdir(parents=True)
    devcontainer = ws / ".devcontainer" / "universe-devel-cuda"
    devcontainer.mkdir(parents=True)
    (devcontainer / "devcontainer.json").write_text(_DEVCONTAINER_JSON)
    compose = ws / "docker" / "devcontainer"
    compose.mkdir(parents=True)
    (compose / "universe-devel-cuda.compose.yaml").write_text(_COMPOSE_YAML)
    if built:
        _install(ws, scenario=scenario)
    return ws


def _install(ws: Path, *, scenario: bool) -> None:
    (ws / "install").mkdir(exist_ok=True)
    (ws / "install" / "setup.bash").write_text("")
    lib = (
        ws / "install" / "autoware_carla_interface" / "lib" / "autoware_carla_interface"
    )
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "carla_autoware").write_text("")
    launch = (
        ws
        / "install"
        / "autoware_carla_interface"
        / "share"
        / "autoware_carla_interface"
        / "launch"
    )
    launch.mkdir(parents=True, exist_ok=True)
    arg = '<arg name="scenario_mode" default="false"/>' if scenario else ""
    (launch / "autoware_carla_interface.launch.xml").write_text(
        f"<launch>{arg}</launch>"
    )


def test_devcontainer_image_and_mount_come_from_compose(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)

    container = read_devcontainer(ws, "universe-devel-cuda", env={})

    assert container.image == (
        "ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy"
    )
    assert container.workspace_mount == "/home/aw/autoware"
    assert container.environment["CMAKE_CUDA_ARCHITECTURES"] == "86;89"
    assert container.environment["HOST_UID"] == "1000"


def test_devcontainer_variables_resolve_against_the_environment(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)

    container = read_devcontainer(
        ws, "universe-devel-cuda", env={"ROS_DISTRO": "humble"}
    )

    assert container.image.endswith("universe-devel-cuda-humble")


def test_devcontainer_image_named_directly(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / ".devcontainer" / "custom").mkdir(parents=True)
    (ws / ".devcontainer" / "custom" / "devcontainer.json").write_text(
        '{"image": "example/autoware:dev", "workspaceFolder": "/ws"}'
    )

    container = read_devcontainer(ws, "custom", env={})

    assert container.image == "example/autoware:dev"
    assert container.workspace_mount == "/ws"


def test_missing_devcontainer_variant_lists_the_ones_there(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)

    with pytest.raises(DevContainerError, match="universe-devel-cuda"):
        read_devcontainer(ws, "core-devel", env={})


# ---------------------------------------------------------------------------
# Process group
# ---------------------------------------------------------------------------


def test_process_group_stop_takes_the_children_too(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    # A parent that ignores SIGINT and a child of its own, as a launch that
    # does not shut down cleanly would leave.
    group = ProcessGroup(
        [
            "bash",
            "-c",
            f"trap '' INT; sleep 60 & echo $! > {pid_file}; wait",
        ],
        stop_timeout_s=0.5,
    )
    _wait_for(lambda: pid_file.exists() and pid_file.read_text().strip() != "")
    child = int(pid_file.read_text())
    assert group.poll() is None

    group.stop()

    assert group.poll() is not None
    _wait_for(lambda: not _alive(child))


def test_process_group_logs_output(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "out.log"
    group = ProcessGroup(["bash", "-c", "echo hello"], log_path=log)
    _wait_for(lambda: group.poll() is not None)
    group.stop()

    assert log.read_text().strip() == "hello"


def test_process_group_reports_a_missing_command() -> None:
    with pytest.raises(AutowareStackError, match="no-such-command"):
        ProcessGroup(["no-such-command-for-sure"])


# ---------------------------------------------------------------------------
# Command launcher
# ---------------------------------------------------------------------------


def test_command_launcher_fills_in_the_episode(tmp_path: Path) -> None:
    out = tmp_path / "args.json"
    script = (
        "import json, os, sys; "
        f"json.dump({{'argv': sys.argv[1:], 'env': os.environ['AUTOWARE_BRIDGE_ADDRESS']}}, "
        f"open({str(out)!r}, 'w'))"
    )
    launcher = CommandAutowareLauncher(
        [sys.executable, "-c", script, "{bridge_address}", "{index}", "{name}"]
    )
    assert isinstance(launcher, AutowareLauncher)
    launcher.prepare(log_dir=tmp_path)

    episodes = []
    for _ in range(2):
        episode = launcher.start(name="left_turn", bridge_address="localhost:1234")
        episodes.append(episode)
        _wait_for(lambda: launcher.poll() is not None)
        assert launcher.poll() == 0
        assert json.loads(out.read_text()) == {
            "argv": ["localhost:1234", str(episode.index), "left_turn"],
            "env": "localhost:1234",
        }
        launcher.stop()

    assert [e.index for e in episodes] == [0, 1]
    assert (tmp_path / "autoware-000-left_turn.log").exists()
    assert (tmp_path / "autoware-001-left_turn.log").exists()


def test_command_launcher_runs_one_stack_at_a_time() -> None:
    launcher = CommandAutowareLauncher(["sleep", "30"])
    launcher.start(name="a", bridge_address="localhost:1")
    try:
        with pytest.raises(AutowareStackError, match="already running"):
            launcher.start(name="b", bridge_address="localhost:1")
    finally:
        launcher.close()
    assert launcher.poll() is None


# ---------------------------------------------------------------------------
# Docker launcher, against a Docker CLI that only pretends
# ---------------------------------------------------------------------------

_FAKE_DOCKER = """\
#!{python}
import json, os, signal, sys, time

args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")

if args[0] == "ps":
    print(os.environ.get("FAKE_DOCKER_PS", ""))
elif args[:2] == ["image", "inspect"]:
    if "--format" in args:
        print("sha256:" + args[-1].replace("/", "_"))
    else:
        tags = os.environ.get("FAKE_DOCKER_IMAGES", "").split()
        sys.exit(0 if args[-1] in tags else 1)
elif args[0] == "run":
    script = args[-1]
    if "apt-get install" in script:
        if "rosdep install" in script:
            print("#All required rosdeps installed successfully")
        else:
            print("Setting up python3-grpcio")
        sys.exit(int(os.environ.get("FAKE_DOCKER_ROSDEP_EXIT", "0")))
    if "colcon build" in script:
        ws = os.environ["FAKE_DOCKER_WS"]
        lib = os.path.join(ws, "install", "autoware_carla_interface", "lib",
                           "autoware_carla_interface")
        os.makedirs(lib, exist_ok=True)
        open(os.path.join(ws, "install", "setup.bash"), "w").close()
        share = os.path.join(ws, "install", "autoware_carla_interface", "share",
                             "autoware_carla_interface", "launch")
        os.makedirs(share, exist_ok=True)
        arg = ('<arg name="scenario_mode"/>'
               if os.environ.get("FAKE_DOCKER_SCENARIO") == "1" else "")
        with open(os.path.join(share, "autoware_carla_interface.launch.xml"), "w") as f:
            f.write("<launch>" + arg + "</launch>")
        print("Summary: 1 package finished")
        sys.exit(int(os.environ.get("FAKE_DOCKER_BUILD_EXIT", "0")))
    # A stack: runs until it is told to stop, as ros2 launch does.
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    marker = os.environ.get("FAKE_DOCKER_RUNNING")
    if marker:
        open(marker, "w").close()
    while True:
        time.sleep(0.05)
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docker = tmp_path / "bin" / "docker"
    docker.parent.mkdir()
    docker.write_text(_FAKE_DOCKER.format(python=sys.executable))
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))

    def calls() -> list[list[str]]:
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return docker, calls


def _docker_launcher(ws: Path, docker: Path, **kwargs) -> DockerAutowareLauncher:
    options = dict(
        workspace=ws,
        launch=(
            "autoware_launch",
            "e2e_simulator.launch.xml",
            "bridge_address:={bridge_address}",
        ),
        docker=str(docker),
        mounts={},
        stop_timeout_s=2.0,
        rosdep=False,
    )
    options.update(kwargs)
    return DockerAutowareLauncher(DockerAutowareConfig(**options))  # type: ignore[arg-type]


def test_docker_launcher_builds_an_unbuilt_workspace(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_SCENARIO", "1")
    launcher = _docker_launcher(ws, docker)

    launcher.prepare(log_dir=tmp_path / "out")

    build = next(c for c in calls() if c[0] == "run" and "colcon build" in c[-1])
    assert f"{ws}:/home/aw/autoware" in build
    # Built in the image the stack runs in: the dev container's, with what the
    # stack needs installed.
    (tag,) = _dependency_tags(calls())
    assert tag in build
    assert "colcon build --symlink-install" in build[-1]
    assert (tmp_path / "out" / "autoware-build.log").read_text().startswith("Summary")
    # A build container is removed whatever happened to it.
    assert any(c[0] == "rm" and c[-1].endswith("-build") for c in calls())


def test_docker_launcher_refuses_a_failed_build(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_BUILD_EXIT", "2")

    with pytest.raises(AutowareStackError, match="colcon build failed"):
        _docker_launcher(ws, docker).prepare()


def test_docker_launcher_does_not_build_when_told_not_to(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path)

    with pytest.raises(AutowareStackError, match="has not been built"):
        _docker_launcher(ws, docker, build="never").prepare()
    assert not any(c[0] == "run" for c in calls())


def test_docker_launcher_refuses_a_branch_without_scenario_support(
    tmp_path: Path, fake_docker
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=False)

    with pytest.raises(AutowareStackError, match="without scenario support"):
        _docker_launcher(ws, docker).prepare()


def test_docker_launcher_removes_leftover_containers(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    monkeypatch.setenv("FAKE_DOCKER_PS", "abc123\ndef456")

    _docker_launcher(ws, docker).prepare()

    assert [
        "ps",
        "--all",
        "--quiet",
        "--filter",
        f"label={STACK_LABEL}=uid-{os.getuid()}",
    ] in calls()
    assert ["rm", "--force", "abc123", "def456"] in calls()


def test_docker_launcher_starts_and_removes_a_container_per_scenario(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(
        ws, docker, ros_domain_ids=(7, 8), env={"FOO": "bar"}, gpus=None
    )
    assert isinstance(launcher, AutowareLauncher)
    launcher.prepare(log_dir=tmp_path / "out")

    names: list[str] = []
    for expected_domain in ("7", "8", "7"):
        episode = launcher.start(name="cut_in", bridge_address="localhost:50052")
        _wait_for(lambda: len(_stack_runs(calls())) == len(names) + 1)
        run = _stack_runs(calls())[-1]
        assert launcher.poll() is None
        launcher.stop()
        name = run[run.index("--name") + 1]
        names.append(name)

        assert run[:3] == ["run", "--rm", "--name"]
        assert f"{STACK_LABEL}=uid-{os.getuid()}" in run
        assert "--init" in run
        assert run[run.index("--network") + 1] == "host"
        assert f"{ws}:/home/aw/autoware" in run
        assert "--gpus" not in run
        assert f"ROS_DOMAIN_ID={expected_domain}" in run
        assert "FOO=bar" in run
        assert f"HOST_UID={os.getuid()}" in run
        assert "AUTOWARE_BRIDGE_ADDRESS=localhost:50052" in run
        assert run[-3:-1] == ["bash", "-c"]
        # The framework's scenario_bridge node, mounted from this package, runs
        # beside the launch.
        assert (
            f"{bridge_package_dir()}:/opt/autoware_carla_scenario/python/"
            "autoware_carla_scenario/autoware_bridge:ro"
        ) in run
        assert run[-1] == (
            "source /home/aw/autoware/install/setup.bash && "
            '(PYTHONPATH=/opt/autoware_carla_scenario/python"${PYTHONPATH:+:$PYTHONPATH}" '
            "python3 -m autoware_carla_scenario.autoware_bridge.ros_bridge --ros-args "
            "-p bridge_address:=localhost:50052 -p use_sim_time:=true &) && "
            "exec ros2 launch autoware_launch e2e_simulator.launch.xml "
            "bridge_address:=localhost:50052"
        )
        assert ["rm", "--force", name] in calls()
        assert (tmp_path / "out" / f"autoware-{episode.index:03d}-cut_in.log").exists()

    assert len(set(names)) == 3
    assert launcher.poll() is None


def _stack_runs(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "run" and c[-1].startswith("source")]


def test_docker_launcher_reports_a_stack_that_exits(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, _calls = fake_docker
    running = tmp_path / "running"
    monkeypatch.setenv("FAKE_DOCKER_RUNNING", str(running))
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    launcher.start(name="x", bridge_address="localhost:1")
    process = launcher._process
    assert process is not None
    # Its SIGINT handler is installed once the marker exists.
    _wait_for(running.exists)

    os.killpg(process.pid, signal.SIGINT)
    _wait_for(lambda: launcher.poll() is not None)

    assert launcher.poll() == 0
    launcher.close()


def test_docker_config_requires_a_launch() -> None:
    with pytest.raises(ValueError, match="launch is required"):
        DockerAutowareConfig(workspace=Path("/ws"), launch=())


def test_docker_config_rejects_an_unknown_build_policy() -> None:
    with pytest.raises(ValueError, match="build must be"):
        DockerAutowareConfig(workspace=Path("/ws"), launch=("a",), build="sometimes")  # type: ignore[arg-type]


def test_compose_override_files_merge_environment_and_volumes(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    devcontainer = ws / ".devcontainer" / "universe-devel-cuda" / "devcontainer.json"
    devcontainer.write_text(
        devcontainer.read_text().replace(
            '"dockerComposeFile": "../../docker/devcontainer/universe-devel-cuda.compose.yaml",',
            '"dockerComposeFile": ["../../docker/devcontainer/universe-devel-cuda.compose.yaml",'
            ' "../../docker/devcontainer/override.yaml"],',
        )
    )
    (ws / "docker" / "devcontainer" / "override.yaml").write_text(
        "services:\n"
        "  autoware-devel:\n"
        "    environment:\n"
        "      - EXTRA=1\n"
        "      - FROM_HOST\n"
        "      - NOT_ON_HOST\n"
        "    volumes:\n"
        "      - /data:/data\n"
    )

    container = read_devcontainer(ws, "universe-devel-cuda", env={"FROM_HOST": "h"})

    # The base file's mount and environment survive the override.
    assert container.workspace_mount == "/home/aw/autoware"
    assert container.environment["CMAKE_CUDA_ARCHITECTURES"] == "86;89"
    assert container.environment["EXTRA"] == "1"
    # A bare key takes the host's value, and is left out when it has none.
    assert container.environment["FROM_HOST"] == "h"
    assert "NOT_ON_HOST" not in container.environment


def test_the_mount_wins_over_a_workspace_folder_below_it(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    devcontainer = ws / ".devcontainer" / "universe-devel-cuda" / "devcontainer.json"
    devcontainer.write_text(
        devcontainer.read_text().replace(
            '"service": "autoware-devel",',
            '"service": "autoware-devel", "workspaceFolder": "/home/aw/autoware/src",',
        )
    )

    container = read_devcontainer(ws, "universe-devel-cuda", env={})

    assert container.workspace_mount == "/home/aw/autoware"


def test_an_image_override_keeps_the_dev_containers_mount(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker, image="example/autoware:mine")
    launcher.start(name="x", bridge_address="localhost:1")
    _wait_for(lambda: len(_stack_runs(calls())) == 1)
    launcher.stop()

    deps = next(c for c in calls() if c[0] == "run" and "apt-get install" in c[-1])
    assert "example/autoware:mine" in deps
    run = _stack_runs(calls())[0]
    assert _dependency_tags(calls())[0] in run
    assert f"{ws}:/home/aw/autoware" in run
    assert "CMAKE_CUDA_ARCHITECTURES=86;89" in run


def test_a_stack_dies_with_the_process_that_started_it(tmp_path: Path) -> None:
    """A sweeper kills a job past its timeout; the job's stack must not outlive it."""
    pid_file = tmp_path / "stack.pid"
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from autoware_carla_scenario.autoware_stack import ProcessGroup
        group = ProcessGroup(["bash", "-c", "echo $$ > {pid_file}; exec sleep 60"])
        import time
        while not Path("{pid_file}").exists() or not Path("{pid_file}").read_text().strip():
            time.sleep(0.02)
        print("started", flush=True)
        time.sleep(60)
        """
    )
    job = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert job.stdout is not None
        assert job.stdout.readline().strip() == "started"
        stack = int(pid_file.read_text())
        assert _alive(stack)
        job.kill()
        job.wait()
        _wait_for(lambda: not _alive(stack))
    finally:
        job.kill()


# ---------------------------------------------------------------------------
# The workspace's dependencies, installed into an image of their own
# ---------------------------------------------------------------------------


def _dependency_tags(calls: list[list[str]]) -> list[str]:
    return [c[-1] for c in calls if c[0] == "commit"]


def test_the_workspaces_dependencies_go_into_an_image_the_stack_runs_in(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    (ws / "src" / "pkg").mkdir(parents=True)
    (ws / "src" / "pkg" / "package.xml").write_text("<exec_depend>a</exec_depend>")
    launcher = _docker_launcher(ws, docker, rosdep=True)

    launcher.prepare(log_dir=tmp_path / "out")
    launcher.start(name="x", bridge_address="localhost:1")
    _wait_for(lambda: len(_stack_runs(calls())) == 1)
    launcher.stop()

    rosdep = next(c for c in calls() if c[0] == "run" and "rosdep install" in c[-1])
    # Made from the dev container's image, and kept to be committed.
    assert "ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy" in rosdep
    assert "--rm" not in rosdep
    (tag,) = _dependency_tags(calls())
    assert tag.startswith(f"{DEPENDENCY_IMAGE_REPOSITORY}:")
    assert tag in _stack_runs(calls())[0]
    assert "python3-grpcio python3-protobuf" in rosdep[-1]
    assert any(c[0] == "rm" and c[-1].endswith("-deps") for c in calls())
    assert "rosdeps installed" in (tmp_path / "out" / "autoware-deps.log").read_text()


def test_the_dependency_image_is_made_once_per_set_of_package_xmls(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    (ws / "src" / "pkg").mkdir(parents=True)
    xml = ws / "src" / "pkg" / "package.xml"
    xml.write_text("<exec_depend>a</exec_depend>")
    _docker_launcher(ws, docker, rosdep=True).prepare()
    (tag,) = _dependency_tags(calls())

    # Made already: used as it is.
    monkeypatch.setenv("FAKE_DOCKER_IMAGES", tag)
    _docker_launcher(ws, docker, rosdep=True).prepare()
    assert _dependency_tags(calls()) == [tag]

    # A dependency added: made again, under another tag.
    xml.write_text("<exec_depend>a</exec_depend><exec_depend>b</exec_depend>")
    _docker_launcher(ws, docker, rosdep=True).prepare()
    tags = _dependency_tags(calls())
    assert len(tags) == 2 and tags[1] != tag


def test_a_failed_rosdep_install_is_refused(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    monkeypatch.setenv("FAKE_DOCKER_ROSDEP_EXIT", "1")

    with pytest.raises(AutowareStackError, match="Installing the dependencies"):
        _docker_launcher(ws, docker, rosdep=True).prepare()
    assert _dependency_tags(calls()) == []
    assert any(c[0] == "rm" and c[-1].endswith("-deps") for c in calls())


def test_without_rosdep_the_image_still_gets_the_bridges_packages(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, rosdep=False).prepare()

    deps = next(c for c in calls() if c[0] == "run" and "apt-get install" in c[-1])
    assert "python3-grpcio python3-protobuf" in deps[-1]
    assert "rosdep" not in deps[-1]


def test_a_built_workspace_without_scenario_mode_is_refused(
    tmp_path: Path, fake_docker
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=False)

    with pytest.raises(AutowareStackError, match="takes no scenario_mode"):
        _docker_launcher(ws, docker).prepare()


def test_command_launcher_fills_in_the_scenario_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    out = tmp_path / "bridge.txt"
    launcher = CommandAutowareLauncher(
        ["bash", "-c", f"echo {{scenario_bridge}} > {out}"],
        bridge_parameters={"use_sim_time": True, "auto_engage": False},
    )
    launcher.prepare()
    launcher.start(name="x", bridge_address="localhost:7")
    _wait_for(lambda: launcher.poll() is not None)
    launcher.stop()

    written = out.read_text()
    assert "python3 -m autoware_carla_scenario.autoware_bridge.ros_bridge" in written
    assert "-p bridge_address:=localhost:7" in written
    assert "-p use_sim_time:=true -p auto_engage:=false" in written


def test_the_host_python_root_holds_the_bridge_package_alone(tmp_path: Path) -> None:
    """Another Python imports the bridge from it without the framework's own package."""
    import site

    root = host_python_root(tmp_path)
    assert host_python_root(tmp_path) == root  # kept, not made again
    code = (
        "import sys; "
        "import autoware_carla_scenario.autoware_bridge.ros_bridge.client; "
        "import autoware_carla_scenario as p; "
        "print(p.__file__, 'typesafe_carla' in sys.modules)"
    )
    # -S: no site-packages, so not the framework installed in this venv; only
    # grpc and protobuf from its site directory, put after the root.
    path = os.pathsep.join([str(root), *site.getsitepackages()])
    done = subprocess.run(
        [sys.executable, "-S", "-c", code],
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        check=True,
    )
    # A namespace package: no __init__ of the framework's was run.
    assert done.stdout.split() == ["None", "False"]


def test_a_symlink_install_is_read_through_the_workspace_mount(
    tmp_path: Path, fake_docker
) -> None:
    """``--symlink-install`` links install/ to the sources by their container paths."""
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=False)
    source = ws / "src" / "autoware_carla_interface" / "launch"
    source.mkdir(parents=True)
    (source / "autoware_carla_interface.launch.xml").write_text(
        '<launch><arg name="scenario_mode"/></launch>'
    )
    installed = (
        ws
        / "install"
        / "autoware_carla_interface"
        / "share"
        / "autoware_carla_interface"
        / "launch"
        / "autoware_carla_interface.launch.xml"
    )
    installed.unlink()
    installed.symlink_to(
        "/home/aw/autoware/src/autoware_carla_interface/launch/"
        "autoware_carla_interface.launch.xml"
    )

    launcher = _docker_launcher(ws, docker)
    launcher.prepare()
    assert launcher.has_scenario_support()
