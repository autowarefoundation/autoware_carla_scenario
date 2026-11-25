"""Tests for :mod:`autoware_carla_scenario.autoware_stack`.

No Docker and no Autoware: the dev container is a configuration written to a
temporary workspace, the Docker CLI is a script that records what it was asked
and plays a container, and a stack is a plain process group.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.autoware_stack.bridge_node import (
    BRIDGE_PROTOBUF_REQUIREMENT,
    MISSION_POLL_TIMEOUT_S,
    ROUTE_WATCH_PERIOD_S,
    DOUBLE_PARAMETERS,
    INTEGER_PARAMETERS,
    POSITIVE_PARAMETERS,
    bridge_package_dir,
    bridge_ros_args,
    host_python_root,
    launcher_bridge_parameters,
    liveness_timeout_problem,
    min_liveness_timeout_s,
)
from autoware_carla_scenario.autoware_stack import (
    DEPENDENCY_IMAGE_REPOSITORY,
    OWNER_LABEL,
    STACK_LABEL,
    AutowareEpisode,
    AutowareLauncher,
    AutowareStackError,
    CommandAutowareLauncher,
    DevContainerError,
    DockerAutowareConfig,
    DockerAutowareLauncher,
    ProcessGroup,
    read_devcontainer,
)

from autoware_carla_scenario.autoware_stack import docker as docker_module
from autoware_carla_scenario.autoware_stack.launcher import (
    episode_log_path,
    fill_placeholders,
)

from ._stale_pids import signal_group, signal_unreaped_group


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
    except (FileNotFoundError, ProcessLookupError):
        # The process was reaped between kill(0) and the read: /proc raises
        # ESRCH mid-read rather than failing the open.
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


def test_an_unreadable_devcontainer_is_a_devcontainer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #2187: not a raw PermissionError (root reads any file, so read_text
    # refuses it here).
    ws = _workspace(tmp_path)
    config = ws / ".devcontainer" / "universe-devel-cuda" / "devcontainer.json"
    read_text = Path.read_text

    def refusing(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == config:
            raise PermissionError(13, "Permission denied", str(self))
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refusing)
    with pytest.raises(DevContainerError, match="Could not read .*Permission denied"):
        read_devcontainer(ws, "universe-devel-cuda", env={})


@pytest.mark.parametrize("document", ["[]", '"image"', "3", "null"])
def test_a_devcontainer_that_is_not_an_object_is_a_devcontainer_error(
    tmp_path: Path, document: str
) -> None:
    ws = tmp_path / "ws"
    (ws / ".devcontainer" / "custom").mkdir(parents=True)
    (ws / ".devcontainer" / "custom" / "devcontainer.json").write_text(document)

    with pytest.raises(DevContainerError, match="not a JSON object"):
        read_devcontainer(ws, "custom", env={})


def test_an_unreadable_devcontainer_falls_back_on_a_configured_image(
    tmp_path: Path, fake_docker
) -> None:
    # The DevContainerError lets _resolve_container take the image named.
    docker, _calls = fake_docker
    ws = tmp_path / "ws"
    (ws / ".devcontainer" / "universe-devel-cuda").mkdir(parents=True)
    (ws / ".devcontainer" / "universe-devel-cuda" / "devcontainer.json").write_text(
        "[]"
    )
    launcher = _docker_launcher(ws, docker, image="example/autoware:mine")

    assert launcher._resolve_container().image == "example/autoware:mine"


def test_a_compose_that_is_not_a_mapping_names_no_service(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    compose = ws / "docker" / "devcontainer" / "universe-devel-cuda.compose.yaml"
    compose.write_text("- a\n- b\n")

    with pytest.raises(DevContainerError, match="No service 'autoware-devel'"):
        read_devcontainer(ws, "universe-devel-cuda", env={})


@pytest.mark.parametrize("overridden", [False, True], ids=["one-file", "override"])
def test_compose_environment_values_are_interpolated_once(
    tmp_path: Path, overridden: bool
) -> None:
    # #2187: Compose interpolates a value once: a host value passed through
    # (`- KEY`) keeps a `${X}` in it.
    ws = _workspace(tmp_path)
    compose = ws / "docker" / "devcontainer" / "universe-devel-cuda.compose.yaml"
    compose.write_text(
        compose.read_text().replace(
            "      - CMAKE_CUDA_ARCHITECTURES=86;89\n",
            "      - CMAKE_CUDA_ARCHITECTURES=86;89\n" "      - FROM_HOST\n",
        )
    )
    if overridden:
        devcontainer = (
            ws / ".devcontainer" / "universe-devel-cuda" / "devcontainer.json"
        )
        devcontainer.write_text(
            devcontainer.read_text().replace(
                '"dockerComposeFile": "../../docker/devcontainer/universe-devel-cuda.compose.yaml",',
                '"dockerComposeFile": ["../../docker/devcontainer/universe-devel-cuda.compose.yaml",'
                ' "../../docker/devcontainer/override.yaml"],',
            )
        )
        (ws / "docker" / "devcontainer" / "override.yaml").write_text(
            "services:\n  autoware-devel:\n    environment:\n      - EXTRA=1\n"
        )

    container = read_devcontainer(
        ws, "universe-devel-cuda", env={"FROM_HOST": "a${X}b", "X": "expanded"}
    )

    assert container.environment["FROM_HOST"] == "a${X}b"
    assert container.environment["HOST_UID"] == "1000"


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


#: How long the test waits for the background run before it calls that a
#: hang: a bound on a hang only, far past what a loaded machine takes.
_HANG_S = 300.0

_BACKGROUND_PARENT = """\
import json, os, signal, sys, time
from pathlib import Path
from autoware_carla_scenario.autoware_stack import ProcessGroup

out = Path(sys.argv[1])
child_out = out.with_suffix(".child")
child = (
    "import json, os, signal, sys, time\\n"
    "json.dump({'int_ignored': signal.getsignal(signal.SIGINT) == signal.SIG_IGN,"
    " 'quit_default': signal.getsignal(signal.SIGQUIT) == signal.SIG_DFL},"
    " open(sys.argv[1] + '.tmp', 'w'))\\n"
    "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\\n"
    "time.sleep(120)\\n"
)
group = ProcessGroup([sys.executable, "-c", child, str(child_out)], stop_timeout_s=60.0)
# The pid and its start time (field 22 of stat), so the test's teardown can
# tell the group from one that reused its id once it is gone (#812).
with open(f"/proc/{group.pid}/stat") as stat:
    started = stat.read().rsplit(")", 1)[1].split()[19]
pid_tmp = out.with_suffix(".pid.tmp")
pid_tmp.write_text(f"{group.pid} {started}")
pid_tmp.replace(out.with_suffix(".pid"))
# For as long as the child runs: however slowly a loaded machine starts it,
# it is stopped only once it reported, or the result would be about a child
# stopped before it could (the test's own wait bounds a hang).  waitid with
# WNOWAIT leaves an exited child unreaped, for stop() to find as it is.
while not child_out.exists() and (
    os.waitid(os.P_PID, group.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
):
    time.sleep(0.05)
code = group.stop()
json.dump(
    {
        "parent_int_ignored": signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
        "parent_quit_ignored": signal.getsignal(signal.SIGQUIT) == signal.SIG_IGN,
        "child": json.loads(child_out.read_text()) if child_out.exists() else None,
        "code": code,
    },
    open(out, "w"),
)
"""


def test_process_group_child_hears_sigint_from_a_background_run(
    tmp_path: Path,
) -> None:
    """A run with SIGINT ignored (a background job) still stops its stack with it (#759).

    The shell starts a background job with ``SIGINT`` and ``SIGQUIT`` ignored;
    the stack must not inherit that, or the ``SIGINT`` :meth:`stop` sends is
    lost and only the ``SIGTERM`` after it, a long timeout later, stops it.
    """
    script = tmp_path / "parent.py"
    script.write_text(_BACKGROUND_PARENT)
    out = tmp_path / "result.json"
    pid_file = out.with_suffix(".pid")

    # Started as a shell starts a background job: SIGINT and SIGQUIT ignored
    # by a trap, which survives the exec.  Not preexec_fn, which forces a
    # real fork() in this multithreaded worker and runs at-fork hooks (gRPC's,
    # say) and Python code in the child before the exec (#2257).
    parent = subprocess.Popen(  # noqa: S603 - a fixed test command
        [
            "sh",
            "-c",
            'trap \'\' INT QUIT; exec "$0" "$@"',
            sys.executable,
            str(script),
            str(out),
        ],
        start_new_session=True,
    )
    try:
        # Bounds a hang only: the parent waits for the child to start however
        # long that takes, then 60 s (stop_timeout_s) more if SIGINT went
        # unheeded, which the exit code below then reports.
        parent.wait(timeout=_HANG_S)
    finally:
        # Neither group is signalled once its leader is gone and its id may
        # belong to another test's process (#812).
        signal_unreaped_group(parent)
        try:
            pgid, started = (int(field) for field in pid_file.read_text().split())
        except (FileNotFoundError, ValueError):
            pass
        else:
            signal_group(pgid, (pgid, started))
        parent.wait()

    assert parent.returncode == 0, (
        f"the background parent exited with {parent.returncode}"
        " (its traceback, if any, is in the captured stderr)"
    )
    result = json.loads(out.read_text())
    assert result["parent_int_ignored"] is True
    assert result["parent_quit_ignored"] is True
    assert result["child"] == {"int_ignored": False, "quit_default": True}
    # Python dies of the SIGINT it did not catch; a SIGTERM would be -15.
    assert result["code"] == -signal.SIGINT


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_process_group_child_starts_with_no_signal_ignored(tmp_path: Path) -> None:
    """The Python wrapper hands the command no ignored signal of its own.

    Python ignores ``SIGPIPE`` and ``SIGXFSZ`` as it starts; a command the
    wrapper ``exec``s must not inherit that, any more than ``SIGINT``.
    """
    log = tmp_path / "status.log"
    group = ProcessGroup(["grep", "SigIgn", "/proc/self/status"], log_path=log)
    _wait_for(lambda: group.poll() is not None)
    group.stop()
    ignored = int(log.read_text().split()[-1], 16)
    for sig in (signal.SIGINT, signal.SIGQUIT, signal.SIGPIPE, signal.SIGXFSZ):
        assert not ignored & (1 << (sig - 1)), sig.name


def test_process_group_runs_beside_a_module_that_shadows_the_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper imports nothing from the run's directory or ``PYTHONPATH``
    (#1961): a scenario package's ``signal.py`` would have it exit 1."""
    shadows = tmp_path / "pkg"
    (shadows / "ctypes").mkdir(parents=True)
    for module in (shadows / "signal.py", shadows / "ctypes" / "__init__.py"):
        module.write_text("raise ImportError('the shadowing module was imported')\n")
    monkeypatch.chdir(shadows)
    monkeypatch.setenv("PYTHONPATH", str(shadows))
    log = tmp_path / "out.log"
    group = ProcessGroup(
        ["bash", "-c", 'echo "ran with PYTHONPATH=$PYTHONPATH"'], log_path=log
    )
    _wait_for(lambda: group.poll() is not None)
    assert group.poll() == 0, log.read_text()
    group.stop()
    # The command it execs keeps the environment.
    assert log.read_text().strip() == f"ran with PYTHONPATH={shadows}"


def test_process_group_logs_output(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "out.log"
    group = ProcessGroup(["bash", "-c", "echo hello"], log_path=log)
    _wait_for(lambda: group.poll() is not None)
    group.stop()

    assert log.read_text().strip() == "hello"


@pytest.mark.parametrize(
    "env", [{"A=B": "1"}, {"A": "x\0y"}, {"A\0": "1"}], ids=["equals", "nul", "nul-key"]
)
def test_process_group_reports_an_environment_it_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """#2079: Popen's ValueError is an AutowareStackError, and the log is closed."""
    opened = []
    path_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = path_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", recording_open)
    with pytest.raises(AutowareStackError, match="Could not run 'true'"):
        ProcessGroup(["true"], env={**os.environ, **env}, log_path=tmp_path / "out.log")
    assert len(opened) == 1
    assert opened[0].closed


def test_process_group_reports_a_log_it_cannot_open(tmp_path: Path) -> None:
    # #2187: an unwritable log directory, or a name too long, is an
    # AutowareStackError, not a raw OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    with pytest.raises(AutowareStackError, match="Could not open the log"):
        ProcessGroup(["true"], log_path=blocker / "logs" / "out.log")
    with pytest.raises(AutowareStackError, match="Could not open the log"):
        ProcessGroup(["true"], log_path=tmp_path / ("x" * 300 + ".log"))


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


def test_command_launcher_names_its_log_after_the_stem_it_is_given(
    tmp_path: Path,
) -> None:
    """#2247: the tasks of a SLURM batch each count their stacks from 0 and
    run cases of one scenario name: a case's stem tells their logs apart."""
    launcher = CommandAutowareLauncher([sys.executable, "-c", "print('up')"])
    launcher.prepare(log_dir=tmp_path)
    for stem in ("CutIn-3", None):
        launcher.use_log_stem(stem)
        launcher.start(name="cut in", bridge_address="localhost:1")
        _wait_for(lambda: launcher.poll() is not None)
        launcher.stop()
    assert (tmp_path / "autoware-CutIn-3.log").read_text() == "up\n"
    # Without a stem, as before: the second stack, counted from 0.
    assert (tmp_path / "autoware-001-cut_in.log").exists()
    episode = AutowareEpisode(7, "x", "localhost:1")
    assert (
        episode_log_path(tmp_path, episode, "a b/c") == tmp_path / "autoware-a_b_c.log"
    )
    assert episode_log_path(tmp_path, episode, "") == tmp_path / "autoware-007-x.log"
    assert episode_log_path(None, episode, "s") is None


def test_command_launcher_runs_one_stack_at_a_time() -> None:
    launcher = CommandAutowareLauncher(["sleep", "30"])
    launcher.start(name="a", bridge_address="localhost:1")
    try:
        with pytest.raises(AutowareStackError, match="already running"):
            launcher.start(name="b", bridge_address="localhost:1")
    finally:
        launcher.close()
    assert launcher.poll() is None


def test_command_launcher_keeps_shell_syntax_in_the_name_out_of_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``{name}`` in a ``bash -c`` script runs nothing the name spells (#1853)."""
    # The command runs here, so the name can make the marker by a short name.
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "marker"
    out = tmp_path / "name.txt"
    name = "x; touch marker $(touch marker) `touch marker` | && > <"
    launcher = CommandAutowareLauncher(
        [
            "bash",
            "-c",
            f'echo {{name}} > {out} && printf %s "$AUTOWARE_SCENARIO_NAME" > {out}.raw',
        ]
    )
    launcher.prepare(log_dir=tmp_path)
    episode = launcher.start(name=name, bridge_address="localhost:1")
    _wait_for(lambda: launcher.poll() is not None)
    assert launcher.poll() == 0
    launcher.stop()

    safe = "".join(
        c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in name
    )
    assert out.read_text() == safe + "\n"
    assert not marker.exists()
    # The name as given, for a script that quotes it.
    assert Path(f"{out}.raw").read_text() == name
    # Named as the log is.
    assert episode_log_path(tmp_path, episode) == tmp_path / f"autoware-000-{safe}.log"
    assert (tmp_path / f"autoware-000-{safe}.log").exists()


def test_fill_placeholders_makes_the_name_shell_safe() -> None:
    episode = AutowareEpisode(3, "a b;c$(d)'e\"f{g}é/", "localhost:9")
    assert fill_placeholders("{name}", episode) == "a_b_c__d__e_f_g___"
    assert fill_placeholders(
        "{name}", AutowareEpisode(0, "temporary_stop-2.v1", "h:1")
    ) == ("temporary_stop-2.v1")


def test_fill_placeholders_does_not_expand_what_it_put_in() -> None:
    """A placeholder in a value stays as it is: the text is scanned once (#1853)."""
    episode = AutowareEpisode(2, "{scenario_bridge}{index}", "{name}")
    filled = fill_placeholders(
        "{bridge_address} {name} {scenario_bridge} {index}",
        episode,
        {"scenario_bridge": "run {index} {name}"},
    )
    assert filled == "{name} _scenario_bridge__index_ run {index} {name} 2"


def test_fill_placeholders_keeps_its_own_keys_over_extra() -> None:
    episode = AutowareEpisode(4, "a;b", "h:1")
    filled = fill_placeholders(
        "{name} {index} {bridge_address} {more}",
        episode,
        {"name": "x;y", "index": "9", "bridge_address": "z", "more": "m"},
    )
    assert filled == "a_b 4 h:1 m"


def test_fill_placeholders_leaves_other_braces_alone() -> None:
    episode = AutowareEpisode(1, "n", "h:1")
    assert fill_placeholders("${HOME} {x} {{index}} {} {index", episode) == (
        "${HOME} {x} {1} {} {index"
    )
    assert fill_placeholders("{index}{index}{nam{name}", episode) == "11{namn"


# ---------------------------------------------------------------------------
# Docker launcher, against a Docker CLI that only pretends
# ---------------------------------------------------------------------------

_FAKE_DOCKER = """\
#!{python}
import json, os, signal, sys, time

args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
# What reaches a container of each variable passed by name (--env KEY): the
# value this CLI process has, as docker reads it.
if args[0] == "run" and os.environ.get("FAKE_DOCKER_ENV_LOG"):
    named = dict(
        (value, os.environ.get(value))
        for flag, value in zip(args, args[1:])
        if flag == "--env" and "=" not in value
    )
    with open(os.environ["FAKE_DOCKER_ENV_LOG"], "a") as log:
        log.write(json.dumps([args[-1], named]) + "\\n")

# A daemon that stops answering (#1790): the commands that start with one of
# FAKE_DOCKER_HANGS (comma-separated) sleep here, in this process, for a
# bounded time -- subprocess.run() kills this process on its timeout.
for hang in filter(None, os.environ.get("FAKE_DOCKER_HANGS", "").split(",")):
    if " ".join(args).startswith(hang):
        time.sleep(30)

if args[0] == "ps":
    print(os.environ.get("FAKE_DOCKER_PS", ""))
elif args[:2] == ["image", "inspect"]:
    if "--format" in args:
        # A base image the daemon lacks until it is pulled (#2501), or a
        # daemon that fails otherwise.
        error = os.environ.get("FAKE_DOCKER_INSPECT_ERROR")
        if error:
            sys.stderr.write(error + "\\n")
            sys.exit(1)
        pulled_log = os.environ.get("FAKE_DOCKER_PULLED", "")
        pulled = []
        if pulled_log and os.path.exists(pulled_log):
            pulled = open(pulled_log).read().split()
        missing = os.environ.get("FAKE_DOCKER_MISSING", "").split()
        if args[-1] in missing and args[-1] not in pulled:
            sys.stderr.write("Error: No such image: " + args[-1] + "\\n")
            sys.exit(1)
        if ".Config.User" in args[args.index("--format") + 1]:
            print(os.environ.get("FAKE_DOCKER_BASE_CONFIG", "\\t"))
        else:
            print("sha256:" + args[-1].replace("/", "_"))
    else:
        tags = os.environ.get("FAKE_DOCKER_IMAGES", "").split()
        sys.exit(0 if args[-1] in tags else 1)
elif args[0] == "pull":
    print("Pulling from " + args[-1])
    code = int(os.environ.get("FAKE_DOCKER_PULL_EXIT", "0"))
    if code == 0 and os.environ.get("FAKE_DOCKER_PULLED"):
        with open(os.environ["FAKE_DOCKER_PULLED"], "a") as pulled:
            pulled.write(args[-1] + "\\n")
    sys.exit(code)
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
    # The dependency images' locks go below it (#2501), not into ~/.cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

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
    (build_log,) = (tmp_path / "out").glob("autoware-build-*.log")
    assert build_log.read_text().startswith("Summary")
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


def test_build_output_that_is_not_utf8_is_streamed_with_replacements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # gcc echoes source lines whatever their bytes: one that is not UTF-8
    # neither aborts the build's output nor truncates its log (#1346).
    launcher = _docker_launcher(_workspace(tmp_path), tmp_path / "docker")
    launcher._log_dir = tmp_path / "out"
    child = (
        "import sys; "
        "sys.stdout.buffer.write(b'warning: \\xff in a source line\\nafter\\n'); "
        "sys.exit(3)"
    )

    code = launcher._stream([sys.executable, "-c", child], "autoware-build.log")

    assert code == 3
    log = (tmp_path / "out" / "autoware-build.log").read_text(encoding="utf-8")
    assert log == "warning: \ufffd in a source line\nafter\n"
    assert "warning: \ufffd in a source line" in capsys.readouterr().out


class _FailingOutput:
    """A child's stdout whose iteration raises *error* after one line."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.closed = False

    def __iter__(self) -> Any:
        yield "first\n"
        raise self._error

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    """A Popen whose output fails mid-stream; records what was done to it."""

    instances: list[_FakePopen] = []

    def __init__(self, error: BaseException) -> None:
        self.stdout = _FailingOutput(error)
        self.killed = False
        self.waited = False
        _FakePopen.instances.append(self)

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: Any = None) -> int:
        self.waited = True
        return -9


@pytest.mark.parametrize(
    "error",
    [BrokenPipeError(32, "Broken pipe"), KeyboardInterrupt()],
    ids=["broken-pipe", "interrupt"],
)
def test_a_build_whose_output_fails_mid_stream_is_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    # #2187: the child is killed, reaped and its pipe closed, not leaked.
    launcher = _docker_launcher(_workspace(tmp_path), tmp_path / "docker")
    launcher._log_dir = tmp_path / "out"
    _FakePopen.instances.clear()
    monkeypatch.setattr(
        docker_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(error),
    )

    with pytest.raises((AutowareStackError, KeyboardInterrupt)):
        launcher._stream(["docker", "build"], "autoware-build.log")

    (process,) = _FakePopen.instances
    assert process.killed
    assert process.waited
    assert process.stdout.closed
    # The log is closed, with what came before the failure.
    assert (tmp_path / "out" / "autoware-build.log").read_text() == "first\n"


def test_a_write_failure_mid_stream_is_not_docker_failing_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #2321: a full disk while writing the build's output is said to be that,
    # not "Could not run 'docker'", which means docker itself did not start.
    launcher = _docker_launcher(_workspace(tmp_path), tmp_path / "docker")
    launcher._log_dir = tmp_path / "out"
    _FakePopen.instances.clear()
    monkeypatch.setattr(
        docker_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakePopen(OSError(28, "No space left on device")),
    )

    with pytest.raises(AutowareStackError) as raised:
        launcher._stream(["docker", "build"], "autoware-build.log")

    assert "Could not run" not in str(raised.value)
    assert "Could not stream the output of" in str(raised.value)
    assert "No space left on device" in str(raised.value)
    (process,) = _FakePopen.instances
    assert process.killed and process.waited


def test_a_command_that_cannot_start_is_docker_failing_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _docker_launcher(_workspace(tmp_path), tmp_path / "docker")
    launcher._log_dir = tmp_path / "out"

    def no_docker(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(docker_module.subprocess, "Popen", no_docker)

    with pytest.raises(AutowareStackError, match="Could not run"):
        launcher._stream(["docker", "build"], "autoware-build.log")


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


def _dead_pid() -> int:
    """The pid of a process that has exited (and been reaped)."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_docker_launcher_removes_leftover_containers(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only those whose process is gone: a live stack of another run of this
    # user's on the host (a SLURM task beside this one) is left alone (#1862).
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    host = socket.gethostname()
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lines = [
            f"dead01\trunning\t{host}:{_dead_pid()}",
            f"live01\trunning\t{host}:{live.pid}",
            f"mine01\trunning\t{host}:{os.getpid()}",
            # Another host's process cannot be seen from here.
            f"other1\trunning\tsome-other-host-{os.getpid()}:{live.pid}",
            f"other2\texited\tsome-other-host-{os.getpid()}:{live.pid}",
            # Started before the owner label: removed only once it has stopped.
            "legacy1\texited\t",
            "legacy2\trunning\t",
            "legacy3\tcreated",
            "legacy4\tdead\t",
            # Up, if not running: may be another run's live stack.
            "legacy5\tpaused\t",
            "legacy6\trestarting\t",
            f"other3\tpaused\tsome-other-host-{os.getpid()}:{live.pid}",
        ]
        monkeypatch.setenv("FAKE_DOCKER_PS", "\n".join(lines))

        _docker_launcher(ws, docker).prepare()
    finally:
        live.kill()
        live.wait()

    assert [
        "ps",
        "--all",
        "--filter",
        f"label={STACK_LABEL}=uid-{os.getuid()}",
        "--format",
        '{{.ID}}\t{{.State}}\t{{.Label "' + OWNER_LABEL + '"}}',
    ] in calls()
    removed = [c for c in calls() if c[:2] == ["rm", "--force"] and len(c) > 3]
    assert removed == [
        ["rm", "--force", "dead01", "other2", "legacy1", "legacy3", "legacy4"]
    ]
    assert "other1 legacy2 legacy5 legacy6 other3" in caplog.text


def test_docker_launcher_labels_its_containers_with_their_owner(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    launcher.start(name="x", bridge_address="localhost:1")
    try:
        _wait_for(lambda: len(_stack_runs(calls())) == 1)
    finally:
        launcher.close()
    run = _stack_runs(calls())[0]
    assert f"{OWNER_LABEL}={socket.gethostname()}:{os.getpid()}" in run


def test_docker_launcher_waits_for_another_runs_build(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two runs on one workspace (SLURM tasks sharing a node) build it one at
    # a time, under a lock in the workspace (#1862).
    import fcntl

    docker, calls = fake_docker
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_SCENARIO", "1")
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.05)
    lock = ws / ".autoware_carla_scenario-build.lock"
    # Another process holds the lock (a child, so flock is not shared with
    # this one's descriptors).
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            f"f = open({str(lock)!r}, 'w')\n"
            "fcntl.flock(f, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(float(sys.argv[1]))\n",
            "1.0",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        began = time.monotonic()
        _docker_launcher(ws, docker, build="always").prepare()
        waited = time.monotonic() - began
    finally:
        holder.kill()
        holder.wait()
    assert waited >= 0.5
    assert any(c[0] == "run" and "colcon build" in c[-1] for c in calls())
    # The lock is let go: another run takes it at once.
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _hold_lock(path: Path) -> subprocess.Popen[str]:
    """A process holding an exclusive flock on *path* (opened read-only)."""
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, sys, time\n"
            "f = open(sys.argv[1], 'rb')\n"
            "fcntl.flock(f, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(float(sys.argv[2]))\n",
            str(path),
            "1.0",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "locked"
    return holder


def test_docker_launcher_takes_no_build_lock_with_nothing_to_make(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Built, and its dependency image made: another run's build lock is not
    # waited for, and nothing warns.
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    launcher._container = launcher._resolve_container()
    tag = launcher._dependency_tag(launcher._container.image)
    launcher._container = None
    monkeypatch.setenv("FAKE_DOCKER_IMAGES", tag)
    lock = ws / ".autoware_carla_scenario-build.lock"
    lock.touch()
    holder = _hold_lock(lock)
    try:
        began = time.monotonic()
        launcher.prepare()
        assert time.monotonic() - began < 0.9
    finally:
        holder.kill()
        holder.wait()
    assert launcher._container is not None
    assert launcher._container.image == tag
    assert not any(c[0] in ("run", "commit") for c in calls())
    assert "without a lock" not in caplog.text
    assert "Waiting for another run" not in caplog.text


@pytest.mark.parametrize("code", [errno.EACCES, errno.EROFS], ids=["eacces", "erofs"])
def test_docker_launcher_locks_a_read_only_build_lock_file(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    # Where the lock file cannot be opened for writing -- an existing one of
    # mode 0444, or on a read-only mount -- it is opened read-only, which a
    # local file system locks too (#2187).  os.open refuses the write itself:
    # root ignores the file's mode.
    docker, calls = fake_docker
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_SCENARIO", "1")
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.05)
    lock = ws / ".autoware_carla_scenario-build.lock"
    lock.touch()
    lock.chmod(0o444)
    flags: list[int] = []
    real_open = os.open

    def recording_open(path, flag, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(path) == lock:
            flags.append(flag)
            if flag & (os.O_WRONLY | os.O_RDWR):
                raise OSError(code, os.strerror(code), str(path))
        return real_open(path, flag, *args, **kwargs)

    monkeypatch.setattr(docker_module.os, "open", recording_open)
    holder = _hold_lock(lock)
    try:
        began = time.monotonic()
        _docker_launcher(ws, docker, build="always").prepare()
        waited = time.monotonic() - began
    finally:
        holder.kill()
        holder.wait()
    assert waited >= 0.5
    # Each try: read-write first, then read-only.
    assert flags
    assert flags[0] == os.O_RDWR | os.O_CREAT
    assert flags[-1] == os.O_RDONLY
    assert any(c[0] == "run" and "colcon build" in c[-1] for c in calls())


def test_docker_launcher_gives_up_on_a_build_lock_held_too_long(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.05)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_TIMEOUT_S", 0.3)
    lock = ws / ".autoware_carla_scenario-build.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, time\n"
            f"f = open({str(lock)!r}, 'w')\n"
            "fcntl.flock(f, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(AutowareStackError, match="Another run has been building"):
            _docker_launcher(ws, docker, build="always").prepare()
    finally:
        holder.kill()
        holder.wait()
    # The dependency image is the host's, made without the workspace's lock
    # (#2501); the build is not run.
    assert not any(c[0] == "run" and "colcon build" in c[-1] for c in calls())


def test_docker_launcher_builds_serially_across_processes(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two launchers in two processes, both told to build: their builds do
    # not overlap.
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    spans = tmp_path / "spans"
    slow_docker = tmp_path / "bin" / "slow-docker"
    slow_docker.write_text(
        f"#!/bin/sh\n"
        f'case "$*" in *"colcon build"*) '
        f'echo "start $(date +%s.%N)" >> {spans}; sleep 0.5; '
        f'echo "end $(date +%s.%N)" >> {spans};; esac\n'
        f'exec {docker} "$@"\n'
    )
    slow_docker.chmod(0o755)
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from autoware_carla_scenario.autoware_stack import (
            DockerAutowareConfig, DockerAutowareLauncher,
        )
        from autoware_carla_scenario.autoware_stack import docker
        docker._BUILD_LOCK_POLL_S = 0.05
        DockerAutowareLauncher(DockerAutowareConfig(
            workspace=Path({str(ws)!r}), launch=("p", "f.launch.xml"),
            docker={str(slow_docker)!r}, mounts={{}}, rosdep=False,
            build="always",
        )).prepare()
        """
    )
    env = {
        **os.environ,
        "FAKE_DOCKER_WS": str(ws),
        "FAKE_DOCKER_SCENARIO": "1",
    }
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for run in runs:
        _out, err = run.communicate(timeout=120)
        assert run.returncode == 0, err
    events = [line.split() for line in spans.read_text().splitlines()]
    assert [kind for kind, _ in events] == ["start", "end", "start", "end"]


def _flock_raising(
    monkeypatch: pytest.MonkeyPatch, *errors: BaseException
) -> list[int]:
    """Make ``fcntl.flock`` raise *errors* in turn (then lock); its calls."""
    import fcntl

    operations: list[int] = []
    pending = list(errors)

    def _flock(fd: int, operation: int) -> None:
        operations.append(operation)
        if pending:
            raise pending.pop(0)

    monkeypatch.setattr(fcntl, "flock", _flock)
    return operations


def test_the_build_lock_is_opened_read_write_created_mode_644(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    opened: list[tuple[Any, ...]] = []
    real_open = os.open

    def _open(path: Any, flags: int, mode: int = 0o777) -> int:
        opened.append((path, flags, mode))
        return real_open(path, flags, mode)

    monkeypatch.setattr(docker_module.os, "open", _open)
    operations = _flock_raising(monkeypatch)
    old = os.umask(0)
    try:
        fd = launcher._lock_workspace()
    finally:
        os.umask(old)
    try:
        assert fd >= 0
        lock = ws / ".autoware_carla_scenario-build.lock"
        # Read-write: on NFS an exclusive flock needs it (#2187).
        assert opened == [(lock, os.O_RDWR | os.O_CREAT, 0o644)]
        assert lock.stat().st_mode & 0o777 == 0o644
        assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB]
    finally:
        launcher._unlock_workspace(fd)


def test_a_build_lock_open_error_other_than_unwritable_is_not_retried_read_only(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    flags: list[int] = []

    def _open(path: Any, flag: int, *args: Any) -> int:
        flags.append(flag)
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(docker_module.os, "open", _open)
    assert launcher._lock_workspace() == -1
    assert flags == [os.O_RDWR | os.O_CREAT]
    assert "no space" in caplog.text


def test_a_build_lock_that_cannot_be_opened_builds_unlocked(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)

    def _open(*args: Any) -> int:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(docker_module.os, "open", _open)
    assert launcher._lock_workspace() == -1
    assert f"Building {ws} without a lock" in caplog.text
    assert "denied" in caplog.text


@pytest.mark.parametrize("waiting", [False, True], ids=["at-once", "waiting"])
def test_a_build_lock_flock_error_closes_the_file_and_builds_unlocked(
    waiting: bool,
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import fcntl

    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.01)
    errors: list[BaseException] = [OSError(37, "no locks")]
    if waiting:
        errors.insert(0, BlockingIOError())
    operations = _flock_raising(monkeypatch, *errors)
    closed: list[int] = []
    real_close = os.close

    def _close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(docker_module.os, "close", _close)
    caplog.set_level(logging.INFO, logger="autoware_carla_scenario")
    assert launcher._lock_workspace() == -1
    assert len(closed) == 1
    assert operations == [fcntl.LOCK_EX | fcntl.LOCK_NB] * len(errors)
    assert f"Building {ws} without a lock" in caplog.text
    assert "no locks" in caplog.text
    assert ("Waiting for another run to finish building" in caplog.text) is waiting


def test_a_build_lock_interrupt_is_raised(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.01)
    error = KeyboardInterrupt()
    _flock_raising(monkeypatch, BlockingIOError(), error)
    closed = _closes_recorded(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as raised:
        launcher._lock_workspace()
    assert raised.value is error
    assert len(closed) == 1


def test_a_build_lock_interrupt_while_waiting_closes_the_file(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The caller's try/finally only starts once _lock_workspace returns, so
    # an interrupt in its sleep must close the descriptor itself (#2327).
    import time

    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    _flock_raising(monkeypatch, *[BlockingIOError() for _ in range(1000)])
    closed = _closes_recorded(monkeypatch)
    error = KeyboardInterrupt()

    def _sleep(seconds: float) -> None:
        raise error

    monkeypatch.setattr(time, "sleep", _sleep)
    with pytest.raises(KeyboardInterrupt) as raised:
        launcher._lock_workspace()
    assert raised.value is error
    assert len(closed) == 1
    # Really closed, not just recorded: the descriptor is gone.
    with pytest.raises(OSError):
        os.fstat(closed[0])


def test_a_build_lock_taken_after_a_wait_is_handed_over_open(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.01)
    _flock_raising(monkeypatch, BlockingIOError(), BlockingIOError())
    closed = _closes_recorded(monkeypatch)
    fd = launcher._lock_workspace()
    try:
        assert fd >= 0
        assert closed == []
        os.fstat(fd)  # still open
    finally:
        launcher._unlock_workspace(fd)
    assert closed == [fd]


def _closes_recorded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the descriptors ``docker`` closes (and close them)."""
    closed: list[int] = []
    real_close = os.close

    def _close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(docker_module.os, "close", _close)
    return closed


def test_a_build_lock_held_too_long_closes_the_file(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, _calls = fake_docker
    ws = _workspace(tmp_path)
    launcher = _docker_launcher(ws, docker)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.01)
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_TIMEOUT_S", 0.05)
    _flock_raising(monkeypatch, *[BlockingIOError() for _ in range(1000)])
    closed: list[int] = []
    real_close = os.close

    def _close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(docker_module.os, "close", _close)
    lock = ws / ".autoware_carla_scenario-build.lock"
    with pytest.raises(AutowareStackError) as raised:
        launcher._lock_workspace()
    assert str(raised.value) == (
        f"Another run has been building {ws} for more than 0 s (it holds "
        f"{lock}); not building it beside it"
    )
    assert len(closed) == 1


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
        # Passed by name, its value in the CLI's environment (#1961).
        assert run[run.index("FOO") - 1] == "--env"
        assert "FOO=bar" not in run
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
            "-p bridge_address:=localhost:50052 -p use_sim_time:=true "
            "-p parent_pid:=$$ &) && "
            "exec ros2 launch autoware_launch e2e_simulator.launch.xml "
            "bridge_address:=localhost:50052"
        )
        assert ["rm", "--force", name] in calls()
        assert (tmp_path / "out" / f"autoware-{episode.index:03d}-cut_in.log").exists()

    assert len(set(names)) == 3
    assert launcher.poll() is None

    # Named after the case's stem while one is given (#2247).
    launcher.use_log_stem("CutIn-2")
    launcher.start(name="cut_in", bridge_address="localhost:50052")
    _wait_for(lambda: len(_stack_runs(calls())) == 4)
    launcher.stop()
    assert (tmp_path / "out" / "autoware-CutIn-2.log").exists()
    assert not (tmp_path / "out" / "autoware-003-cut_in.log").exists()


def test_launcher_env_values_reach_the_container_but_not_the_command_line(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``env`` is passed by name, its values in the CLI's environment: on the
    command line any user of the host could read them in ``ps`` (#1961)."""
    caplog.set_level(logging.DEBUG)
    docker, calls = fake_docker
    env_log = tmp_path / "docker-env.log"
    monkeypatch.setenv("FAKE_DOCKER_ENV_LOG", str(env_log))
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_SCENARIO", "1")
    launcher = _docker_launcher(
        ws,
        docker,
        env={
            "API_TOKEN": "s3cret-token",
            "DEBIAN_FRONTEND": "readline",
            "HTTPS_PROXY": "http://proxy:3128",
        },
        gpus=None,
    )
    launcher.prepare(log_dir=tmp_path / "out")
    launcher.start(name="x", bridge_address="localhost:1")
    # The fake CLI writes what it was handed before it waits to be stopped.
    _wait_for(lambda: len(env_log.read_text().splitlines()) == 3)
    launcher.stop()
    launcher.close()

    runs = [c for c in calls() if c[0] == "run"]
    named = [json.loads(line) for line in env_log.read_text().splitlines()]
    # The dependency image, the build and the stack.
    assert len(runs) == len(named) == 3
    # The dependency image's install gets none of it: `docker commit` would
    # keep it in the image (#2501).
    deps, (deps_script, deps_by_name) = runs[0], named[0]
    assert deps_script == deps[-1]
    assert "apt-get install" in deps[-1]
    assert "--env" not in deps
    assert deps_by_name == {}
    assert "export DEBIAN_FRONTEND=noninteractive && " in deps[-1]
    for run, (script, by_name) in zip(runs[1:], named[1:]):
        assert script == run[-1]
        assert "s3cret-token" not in " ".join(run)
        assert run[run.index("API_TOKEN") - 1] == "--env"
        assert by_name["API_TOKEN"] == "s3cret-token"
        # One the CLI would heed itself stays on the command line.
        assert "HTTPS_PROXY=http://proxy:3128" in run
        assert "HTTPS_PROXY" not in by_name
    assert "s3cret-token" not in caplog.text


def test_a_docker_rm_that_hangs_does_not_hang_stop(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A stuck daemon: stop()'s `docker rm --force` is given up on with a
    # warning, and the batch goes on to its next scenario (#1790).
    docker, calls = fake_docker
    monkeypatch.setattr(docker_module, "_DOCKER_TIMEOUT_S", 1.0)
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    launcher.start(name="x", bridge_address="localhost:1")
    try:
        _wait_for(lambda: len(_stack_runs(calls())) == 1)
        # Only now: the dependency image's own cleanup removes a container.
        monkeypatch.setenv("FAKE_DOCKER_HANGS", "rm")
        began = time.monotonic()
        launcher.stop()
        assert time.monotonic() - began < 2.0 + 15.0
    finally:
        launcher.close()
    assert any(c[:2] == ["rm", "--force"] for c in calls())
    assert "did not finish within 1 s" in caplog.text
    assert launcher._container_name is None


def test_a_docker_ps_that_hangs_warns_instead_of_hanging_prepare(
    tmp_path: Path,
    fake_docker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The leftover containers' removal is best effort: a `docker ps` that does
    # not answer is logged and prepare() goes on (#1790).
    docker, calls = fake_docker
    monkeypatch.setattr(docker_module, "_DOCKER_TIMEOUT_S", 1.0)
    monkeypatch.setenv("FAKE_DOCKER_HANGS", "ps")
    ws = _workspace(tmp_path, built=True, scenario=True)
    began = time.monotonic()
    _docker_launcher(ws, docker).prepare()
    assert time.monotonic() - began < 15.0
    assert any(c[0] == "ps" for c in calls())
    assert "did not finish within 1 s" in caplog.text


def test_an_image_inspect_that_hangs_fails_prepare_in_bounded_time(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The base image's id is needed for the dependency image: a call that
    # does not answer fails prepare() rather than hang it (#1790).
    docker, calls = fake_docker
    monkeypatch.setattr(docker_module, "_DOCKER_TIMEOUT_S", 1.0)
    monkeypatch.setenv("FAKE_DOCKER_HANGS", "image inspect")
    ws = _workspace(tmp_path, built=True, scenario=True)
    began = time.monotonic()
    with pytest.raises(AutowareStackError, match="did not finish within 1 s"):
        _docker_launcher(ws, docker).prepare()
    assert time.monotonic() - began < 15.0
    assert not any(c[0] == "run" for c in calls())


def test_a_dependency_image_inspect_that_hangs_is_not_taken_for_a_missing_image(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The base image answers but the dependency image's tag does not: no
    # answer is not "missing", so nothing is installed or committed (#1797).
    docker, calls = fake_docker
    monkeypatch.setattr(docker_module, "_DOCKER_TIMEOUT_S", 1.0)
    monkeypatch.setenv(
        "FAKE_DOCKER_HANGS", f"image inspect {DEPENDENCY_IMAGE_REPOSITORY}:"
    )
    ws = _workspace(tmp_path, built=True, scenario=True)
    began = time.monotonic()
    with pytest.raises(AutowareStackError, match="did not finish within 1 s"):
        _docker_launcher(ws, docker).prepare()
    assert time.monotonic() - began < 15.0
    # The tag was asked about, after the base image answered.
    assert any(
        c[:2] == ["image", "inspect"]
        and c[-1].startswith(f"{DEPENDENCY_IMAGE_REPOSITORY}:")
        for c in calls()
    )
    assert not any(c[0] in ("run", "commit") for c in calls())


def test_the_dependency_image_commit_gets_the_longer_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _docker_launcher(_workspace(tmp_path), tmp_path / "docker")
    timeouts: dict[str, float] = {}

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeouts[argv[1]] = kwargs["timeout"]  # type: ignore[assignment]
        code = 1 if argv[1:3] == ["image", "inspect"] and len(argv) == 4 else 0
        return subprocess.CompletedProcess(argv, code, "sha256:x\n", "")

    monkeypatch.setattr(docker_module.subprocess, "run", run)
    monkeypatch.setattr(launcher, "_stream", lambda argv, log_name, env=None: 0)
    launcher._container = launcher._resolve_container()
    launcher._dependency_image("img:1", launcher._dependency_tag("img:1"))
    assert timeouts["commit"] == docker_module._DOCKER_COMMIT_TIMEOUT_S
    assert timeouts["image"] == timeouts["rm"] == docker_module._DOCKER_TIMEOUT_S
    assert docker_module._DOCKER_COMMIT_TIMEOUT_S > docker_module._DOCKER_TIMEOUT_S


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


@pytest.mark.parametrize("field", ["launch", "colcon_args", "docker_args"])
@pytest.mark.parametrize("value", ["--shm-size=8g", b"--shm-size=8g"])
def test_docker_config_refuses_a_string_argument_list(
    field: str, value: str | bytes
) -> None:
    """#1484: a str would be run a character per argument."""
    options: dict[str, object] = {"launch": ("a",), field: value}
    with pytest.raises(ValueError, match=rf"^{field} must be a sequence of arguments"):
        DockerAutowareConfig(workspace=Path("/ws"), **options)  # type: ignore[arg-type]


def test_docker_config_takes_argument_lists_as_lists_and_tuples() -> None:
    config = DockerAutowareConfig(
        workspace=Path("/ws"),
        launch=["autoware_launch", "x.launch.xml"],
        colcon_args=(),
        docker_args=["--shm-size=8g"],
    )
    assert list(config.launch) == ["autoware_launch", "x.launch.xml"]
    assert list(config.docker_args) == ["--shm-size=8g"]


@pytest.mark.parametrize("argv", ["ros2 launch x y", b"ros2 launch x y"])
def test_command_launcher_refuses_a_string_command(argv: str | bytes) -> None:
    """#1484: ``"ros2 launch x y"`` may be one argument or four; it is refused."""
    with pytest.raises(ValueError, match=r"^argv must be a sequence of arguments"):
        CommandAutowareLauncher(argv)  # type: ignore[arg-type]


#: #1450: a NaN never times out (the stack is never escalated), and 0 or less
#: skips the grace a stack is given to stop.
_BAD_STOP_TIMEOUTS = [float("nan"), float("inf"), -5.0, 0.0]


@pytest.mark.parametrize("stop_timeout_s", _BAD_STOP_TIMEOUTS)
def test_docker_config_rejects_a_stop_timeout_that_is_not_positive_and_finite(
    stop_timeout_s: float,
) -> None:
    with pytest.raises(ValueError, match="^stop_timeout_s must be positive and finite"):
        DockerAutowareConfig(
            workspace=Path("/ws"), launch=("a",), stop_timeout_s=stop_timeout_s
        )


@pytest.mark.parametrize(
    ("field", "value", "wanted"),
    [
        ("devcontainer", None, "non-empty text"),
        ("devcontainer", "", "non-empty text"),
        ("docker", None, "non-empty text"),
        ("docker", " ", "non-empty text"),
        ("container_prefix", None, "non-empty text"),
        ("image", "", "null or non-empty text"),
        ("workspace_mount", False, "null or non-empty text"),
        ("gpus", 2, "null or non-empty text"),
        ("user", "", "null or non-empty text"),
    ],
)
def test_docker_config_rejects_a_text_field_that_is_no_text(
    field: str, value: object, wanted: str
) -> None:
    """#2161: a None devcontainer failed at prepare time with a TypeError, a
    None docker reached subprocess, a None container_prefix named containers
    "None-..."."""
    with pytest.raises(ValueError, match=rf"^{field} must be {wanted}, got "):
        DockerAutowareConfig(
            workspace=Path("/ws"),
            launch=("a",),
            **{field: value},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("stop_timeout_s", [True, False, "20", None])
def test_command_launcher_rejects_a_stop_timeout_that_is_no_number_naming_it(
    stop_timeout_s: object,
) -> None:
    """#2275: True was a 1 s wait, and "20" a TypeError naming no field."""
    with pytest.raises(
        ValueError, match=rf"^stop_timeout_s must be a number, got {stop_timeout_s!r}$"
    ):
        CommandAutowareLauncher(["true"], stop_timeout_s=stop_timeout_s)  # type: ignore[arg-type]


@pytest.mark.parametrize("stop_timeout_s", _BAD_STOP_TIMEOUTS)
def test_command_launcher_rejects_a_stop_timeout_that_is_not_positive_and_finite(
    stop_timeout_s: float,
) -> None:
    with pytest.raises(ValueError, match="^stop_timeout_s must be positive and finite"):
        CommandAutowareLauncher(["true"], stop_timeout_s=stop_timeout_s)


@pytest.mark.parametrize("stop_timeout_s", _BAD_STOP_TIMEOUTS)
def test_process_group_rejects_a_stop_timeout_before_it_starts_anything(
    stop_timeout_s: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autoware_carla_scenario.autoware_stack.launcher as launcher_module

    monkeypatch.setattr(
        launcher_module,
        "_start_session",
        lambda *_a, **_k: pytest.fail("started the command"),
    )
    with pytest.raises(ValueError, match="^stop_timeout_s must be positive and finite"):
        ProcessGroup([sys.executable, "-c", "pass"], stop_timeout_s=stop_timeout_s)


@pytest.mark.parametrize("ids", [(-3,), (233,), (7, 999), (True,), (7.5,), ("7",)])
def test_docker_config_rejects_a_ros_domain_id_out_of_range(
    ids: tuple[int, ...],
) -> None:
    """#1450: refused here, not by the stack's DDS inside the container."""
    with pytest.raises(ValueError, match="^ros_domain_ids must be ROS domain ids"):
        DockerAutowareConfig(workspace=Path("/ws"), launch=("a",), ros_domain_ids=ids)


def test_docker_config_takes_the_ros_domain_id_range_and_a_small_timeout() -> None:
    config = DockerAutowareConfig(
        workspace=Path("/ws"),
        launch=("a",),
        ros_domain_ids=(0, 101, 232),
        stop_timeout_s=0.1,
    )
    assert tuple(config.ros_domain_ids) == (0, 101, 232)
    assert config.stop_timeout_s == 0.1


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
    (deps_log,) = (tmp_path / "out").glob("autoware-deps-*.log")
    assert "rosdeps installed" in deps_log.read_text()


@pytest.mark.parametrize(
    ("virtual_env", "as_root"),
    [("/opt/autoware-venv", " --as-root pip:false"), ("", "")],
)
def test_rosdep_keeps_pip_keys_in_the_images_venv(
    tmp_path: Path, fake_docker, virtual_env: str, as_root: str
) -> None:
    # An image with Autoware's venv (VIRTUAL_ENV set, PIP_REQUIRE_VIRTUALENV)
    # refuses rosdep's `sudo pip3`; one without it keeps rosdep's default.
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, rosdep=True).prepare(log_dir=tmp_path / "out")

    script = next(c for c in calls() if c[0] == "run" and "rosdep install" in c[-1])[-1]
    rosdep = script[
        script.index("rosdep install") : script.index(
            docker_module._protobuf_upgrade_script()
        )
    ]
    expanded = subprocess.run(
        ["bash", "-c", f'printf %s "{rosdep}"'],
        env={"VIRTUAL_ENV": virtual_env, "ROS_DISTRO": "jazzy"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert expanded == (
        "rosdep install -y -r --from-paths src --ignore-src --rosdistro jazzy" + as_root
    )


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


def test_package_xmls_reached_through_symlinks_feed_the_digest(
    tmp_path: Path, fake_docker
) -> None:
    """rosdep (catkin_pkg) follows directory symlinks, so a repository linked
    into ``src/`` changes the dependency image's tag too (#2474); a symlink
    loop is walked once."""
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    repo = tmp_path / "elsewhere" / "repo"
    (repo / "pkg").mkdir(parents=True)
    xml = repo / "pkg" / "package.xml"
    xml.write_text("<exec_depend>a</exec_depend>")
    (ws / "src" / "repo").symlink_to(repo, target_is_directory=True)
    (repo / "pkg" / "loop").symlink_to(repo, target_is_directory=True)
    (ws / "src" / "own").mkdir()
    (ws / "src" / "own" / "package.xml").write_text("<exec_depend>c</exec_depend>")

    assert docker_module._package_manifests(ws / "src") == [  # noqa: SLF001
        "own/package.xml",
        "repo/pkg/package.xml",
    ]
    launcher = _docker_launcher(ws, docker, rosdep=True)
    launcher.prepare()
    before = launcher._dependency_digest("base")  # noqa: SLF001
    assert launcher._dependency_digest("base") == before  # noqa: SLF001
    xml.write_text("<exec_depend>a</exec_depend><exec_depend>b</exec_depend>")
    assert launcher._dependency_digest("base") != before  # noqa: SLF001


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


def test_the_dependency_image_keeps_nothing_of_the_run_that_made_it(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``docker commit`` stores the install container's environment, user and
    working directory in an image every user of the host shares (#2501)."""
    docker, calls = fake_docker
    monkeypatch.setenv("FAKE_DOCKER_BASE_CONFIG", "aw\t/home/aw")
    ws = _workspace(tmp_path, built=True, scenario=True)
    extra = tmp_path / "extra"
    extra.mkdir()
    _docker_launcher(
        ws,
        docker,
        rosdep=True,
        env={"API_TOKEN": "s3cret-token", "HTTPS_PROXY": "http://proxy:3128"},
        user="1234:1234",
        mounts={str(extra): "/extra"},
        docker_args=("--env-file", "/secrets.env"),
    ).prepare(log_dir=tmp_path / "out")

    deps = next(c for c in calls() if c[0] == "run" and "apt-get install" in c[-1])
    image = deps.index("ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy")
    options = deps[:image]
    for absent in ("--env", "--user", "--gpus", "--env-file", "--rm"):
        assert absent not in options
    assert "API_TOKEN" not in " ".join(deps)
    assert "HOST_UID" not in " ".join(deps)
    # The dev container's containerEnv.
    assert "CMAKE_CUDA_ARCHITECTURES" not in " ".join(deps)
    assert f"{extra}:/extra" not in options
    assert options[options.index("--volume") + 1] == f"{ws}:/home/aw/autoware"
    commit = next(c for c in calls() if c[0] == "commit")
    changes = [commit[i + 1] for i, a in enumerate(commit) if a == "--change"]
    # The base image's own, read from it.
    assert "USER aw" in changes
    assert "WORKDIR /home/aw" in changes
    assert f"LABEL {OWNER_LABEL}=" in changes
    assert [
        "image",
        "inspect",
        "--format",
        "{{.Config.User}}\t{{.Config.WorkingDir}}",
        "ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy",
    ] in calls()


def test_a_base_image_without_user_or_workdir_commits_dockers_defaults(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, user="1234:1234").prepare()
    commit = next(c for c in calls() if c[0] == "commit")
    assert "USER root" in commit
    assert "WORKDIR /" in commit


def test_the_dependency_image_version_feeds_its_digest(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image made before its install stopped getting the run's settings
    may hold a token: it is made again, under another tag (#2501)."""
    docker, _calls = fake_docker
    launcher = _docker_launcher(_workspace(tmp_path), docker)
    launcher._container = launcher._resolve_container()  # noqa: SLF001
    before = launcher._dependency_digest("base")  # noqa: SLF001
    monkeypatch.setattr(docker_module, "_DEPENDENCY_IMAGE_VERSION", 1)
    assert launcher._dependency_digest("base") != before  # noqa: SLF001


def test_a_missing_base_image_is_pulled(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SLURM node where the dev container was never opened (#2501)."""
    docker, calls = fake_docker
    base = "ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy"
    monkeypatch.setenv("FAKE_DOCKER_MISSING", base)
    monkeypatch.setenv("FAKE_DOCKER_PULLED", str(tmp_path / "pulled"))
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    launcher.prepare(log_dir=tmp_path / "out")

    recorded = calls()
    pull = recorded.index(["pull", base])
    inspects = [
        i
        for i, c in enumerate(recorded)
        if c == ["image", "inspect", "--format", "{{.Id}}", base]
    ]
    assert inspects[0] < pull < inspects[1]
    assert launcher._container is not None  # noqa: SLF001
    assert launcher._container.image.startswith(DEPENDENCY_IMAGE_REPOSITORY)  # noqa: SLF001
    # The pull's output and the install's, in the run's dependency log.
    (log,) = (tmp_path / "out").glob("autoware-deps-*.log")
    text = log.read_text()
    assert text.index(f"Pulling from {base}") < text.index("python3-grpcio")


def test_a_base_image_that_cannot_be_pulled_names_the_log(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    base = "ghcr.io/autowarefoundation/autoware:universe-devel-cuda-jazzy"
    monkeypatch.setenv("FAKE_DOCKER_MISSING", base)
    monkeypatch.setenv("FAKE_DOCKER_PULL_EXIT", "1")
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker)
    with pytest.raises(AutowareStackError, match="Pulling .* failed") as raised:
        launcher.prepare(log_dir=tmp_path / "out")
    (log,) = (tmp_path / "out").glob("autoware-deps-*.log")
    assert str(log) in str(raised.value)
    assert not any(c[0] in ("run", "commit") for c in calls())


def test_a_base_image_inspect_failing_otherwise_is_not_pulled(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker, calls = fake_docker
    monkeypatch.setenv(
        "FAKE_DOCKER_INSPECT_ERROR",
        "permission denied while trying to connect to the Docker daemon socket",
    )
    ws = _workspace(tmp_path, built=True, scenario=True)
    with pytest.raises(AutowareStackError, match="permission denied"):
        _docker_launcher(ws, docker).prepare()
    assert not any(c[0] in ("pull", "run", "commit") for c in calls())


def _image_lock(launcher: DockerAutowareLauncher, cache: Path) -> Path:
    launcher._container = launcher._resolve_container()  # noqa: SLF001
    tag = launcher._dependency_tag(launcher._container.image)  # noqa: SLF001
    launcher._container = None  # noqa: SLF001
    name = docker_module._safe_name(f"{socket.gethostname()}-{tag}") + ".lock"
    return cache / "autoware_carla_scenario" / "locks" / name


def test_the_dependency_image_is_made_under_a_host_lock_not_the_workspaces(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image is the host's daemon's: making it does not wait for the
    shared workspace's lock (another node's build), only for another run
    making it on this host (#2501)."""
    docker, calls = fake_docker
    monkeypatch.setattr(docker_module, "_BUILD_LOCK_POLL_S", 0.05)
    ws = _workspace(tmp_path, built=True, scenario=True)
    workspace_lock = ws / ".autoware_carla_scenario-build.lock"
    workspace_lock.touch()
    holder = _hold_lock(workspace_lock)
    try:
        began = time.monotonic()
        _docker_launcher(ws, docker).prepare()
        assert time.monotonic() - began < 0.9
    finally:
        holder.kill()
        holder.wait()
    assert len(_dependency_tags(calls())) == 1

    other = tmp_path / "other"
    other.mkdir()
    ws2 = _workspace(other, built=True, scenario=True)
    launcher = _docker_launcher(ws2, docker, rosdep=True)
    image_lock = _image_lock(launcher, tmp_path / "cache")
    image_lock.parent.mkdir(parents=True, exist_ok=True)
    image_lock.touch()
    holder = _hold_lock(image_lock)
    try:
        began = time.monotonic()
        launcher.prepare()
        waited = time.monotonic() - began
    finally:
        holder.kill()
        holder.wait()
    assert waited >= 0.5
    assert len(_dependency_tags(calls())) == 2


def test_dependency_images_are_made_one_at_a_time_on_a_host(
    tmp_path: Path, fake_docker
) -> None:
    """Two runs on one host making the same image: one makes it, the other
    waits and uses it."""
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    spans = tmp_path / "spans"
    images = tmp_path / "images"
    images.write_text("")
    wrapper = tmp_path / "bin" / "image-docker"
    # A docker whose committed images stay: `image inspect <tag>` finds them.
    wrapper.write_text(
        "#!/bin/sh\n"
        'for last in "$@"; do :; done\n'
        f'case "$1" in commit) echo "$last" >> {images};; esac\n'
        f'case "$*" in *"apt-get install"*) '
        f'echo "start $(date +%s.%N)" >> {spans}; sleep 0.5; '
        f'echo "end $(date +%s.%N)" >> {spans};; esac\n'
        f'FAKE_DOCKER_IMAGES="$(cat {images})" exec {docker} "$@"\n'
    )
    wrapper.chmod(0o755)
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from autoware_carla_scenario.autoware_stack import (
            DockerAutowareConfig, DockerAutowareLauncher,
        )
        from autoware_carla_scenario.autoware_stack import docker
        docker._BUILD_LOCK_POLL_S = 0.05
        DockerAutowareLauncher(DockerAutowareConfig(
            workspace=Path({str(ws)!r}), launch=("p", "f.launch.xml"),
            docker={str(wrapper)!r}, mounts={{}}, rosdep=False,
        )).prepare()
        """
    )
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for run in runs:
        _out, err = run.communicate(timeout=120)
        assert run.returncode == 0, err
    events = [line.split()[0] for line in spans.read_text().splitlines()]
    assert events == ["start", "end"]
    assert len(images.read_text().split()) == 1


def test_the_deps_and_build_logs_are_the_runs_own(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tasks of a SLURM batch share its directory: each writes logs of its
    own, and an error names the one it wrote (#2501)."""
    docker, _calls = fake_docker
    out = tmp_path / "out"
    (out / "autoware-deps.log").parent.mkdir(parents=True)
    (out / "autoware-deps.log").write_text("another task's\n")
    ws = _workspace(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_WS", str(ws))
    monkeypatch.setenv("FAKE_DOCKER_BUILD_EXIT", "2")
    launcher = _docker_launcher(ws, docker)
    with pytest.raises(AutowareStackError, match="colcon build failed") as raised:
        launcher.prepare(log_dir=out)
    owner = docker_module._safe_name(f"{socket.gethostname()}:{os.getpid()}")
    build_log = out / f"autoware-build-{owner}.log"
    assert build_log.read_text().startswith("Summary")
    assert str(build_log) in str(raised.value)
    assert (out / f"autoware-deps-{owner}.log").exists()
    assert (out / "autoware-deps.log").read_text() == "another task's\n"

    monkeypatch.setenv("FAKE_DOCKER_ROSDEP_EXIT", "1")
    with pytest.raises(
        AutowareStackError, match="Installing the dependencies"
    ) as raised:
        _docker_launcher(ws, docker, rosdep=True).prepare(log_dir=out)
    assert str(out / f"autoware-deps-{owner}.log") in str(raised.value)


def test_without_rosdep_the_image_still_gets_the_bridges_packages(
    tmp_path: Path, fake_docker
) -> None:
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, rosdep=False).prepare()

    deps = next(c for c in calls() if c[0] == "run" and "apt-get install" in c[-1])
    assert "python3-grpcio python3-protobuf" in deps[-1]
    assert "rosdep" not in deps[-1]


def test_the_image_gets_a_protobuf_the_generated_code_imports(
    tmp_path: Path, fake_docker
) -> None:
    """#2217: Ubuntu 22.04's python3-protobuf (3.12) cannot import the node's
    protobuf 4.25 gencode, so the apt step brings it up when it is too old."""
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, rosdep=False).prepare()

    deps = next(c for c in calls() if c[0] == "run" and "apt-get install" in c[-1])
    assert docker_module._protobuf_upgrade_script() in deps[-1]
    assert "pip install 'protobuf>=4.25.0,<8'" in deps[-1]


def test_the_protobuf_step_comes_after_rosdep(tmp_path: Path, fake_docker) -> None:
    """It checks the protobuf the image ends up with, whatever rosdep installed."""
    docker, calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    _docker_launcher(ws, docker, rosdep=True).prepare()

    deps = next(c for c in calls() if c[0] == "run" and "rosdep install" in c[-1])
    assert deps[-1].endswith(docker_module._protobuf_upgrade_script())


def _fake_tools(bin_dir: Path, log: Path, *, has_builder: bool) -> None:
    """``python3``, ``sudo`` and ``apt-get`` that log what they are asked."""
    bin_dir.mkdir()
    import_exit = 0 if has_builder else 1
    tools = {
        "python3": (
            f'if [ "$1" = "-c" ]; then exit {import_exit}; fi\n'
            f'echo "python3 $*" >> {log}\n'
        ),
        "sudo": f'echo "sudo" >> {log}\nexec "$@"\n',
        "apt-get": f'echo "apt-get $*" >> {log}\n',
    }
    for name, body in tools.items():
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(0o755)


@pytest.mark.parametrize(
    ("has_builder", "venv", "expected"),
    [
        (True, None, []),
        (
            False,
            None,
            [
                "sudo",
                "apt-get install -y --no-install-recommends python3-pip",
                "sudo",
                f"python3 -m pip install {BRIDGE_PROTOBUF_REQUIREMENT}",
            ],
        ),
        (False, "/opt/venv", [f"python3 -m pip install {BRIDGE_PROTOBUF_REQUIREMENT}"]),
    ],
)
def test_the_protobuf_step_installs_only_when_the_runtime_is_too_old(
    tmp_path: Path, has_builder: bool, venv: str | None, expected: list[str]
) -> None:
    """pip runs only for a protobuf without ``internal.builder``: into
    Autoware's venv when the image has one, else system-wide with sudo."""
    log = tmp_path / "calls.log"
    log.touch()
    _fake_tools(tmp_path / "bin", log, has_builder=has_builder)
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"}
    if venv is not None:
        env["VIRTUAL_ENV"] = venv
    subprocess.run(
        ["bash", "-c", "true" + docker_module._protobuf_upgrade_script()],
        env=env,
        check=True,
    )
    assert log.read_text().splitlines() == expected


def test_the_protobuf_requirement_is_the_packages_own_floor() -> None:
    """The node runs the vendored gencode the package itself needs
    protobuf>=4.25 for (pyproject.toml)."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    assert BRIDGE_PROTOBUF_REQUIREMENT in dependencies


def test_the_protobuf_requirement_feeds_the_dependency_images_digest(
    tmp_path: Path, fake_docker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image made before the protobuf step (#2217) is made again."""
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=True)
    launcher = _docker_launcher(ws, docker, rosdep=False)
    launcher.prepare()
    before = launcher._dependency_digest("base")  # noqa: SLF001
    monkeypatch.setattr(docker_module, "BRIDGE_PROTOBUF_REQUIREMENT", "protobuf>=9")
    assert launcher._dependency_digest("base") != before  # noqa: SLF001


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


def test_the_scenario_bridge_is_given_the_pid_of_the_shell_that_starts_the_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2405: ``$$`` in the ``( ... &)`` subshell is the ``bash -c``'s pid,
    the one that ``exec``s the launch, so the node ends with the stack."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "args.txt"
    python3 = bin_dir / "python3"
    python3.write_text(
        f'#!/bin/sh\necho "$@" > {args_file}.tmp\nmv {args_file}.tmp {args_file}\n'
    )
    python3.chmod(0o755)
    shell_pid = tmp_path / "shell_pid.txt"
    launcher = CommandAutowareLauncher(
        [
            "bash",
            "-c",
            f"({{scenario_bridge}} &) && echo $$ > {shell_pid} && exec sleep 30",
        ],
        env={"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    )
    launcher.prepare()
    launcher.start(name="x", bridge_address="localhost:7")
    try:
        _wait_for(lambda: args_file.exists() and shell_pid.exists())
        args = args_file.read_text().split()
        pid = shell_pid.read_text().strip()
        assert args[-2:] == ["-p", f"parent_pid:={pid}"]
        assert launcher.poll() is None  # the shell, now the launch, lives on
    finally:
        launcher.stop()


def test_parent_pid_is_not_a_bridge_parameter_to_set() -> None:
    """#2405: the launcher sets it, to the pid of the stack's shell."""
    with pytest.raises(ValueError, match=r"^parent_pid is not a bridge parameter"):
        bridge_ros_args("h:1", {"parent_pid": 1})


def test_the_host_python_root_replaces_a_stale_link(tmp_path: Path) -> None:
    """A link to another install is replaced, and no temporary link is left."""
    root = host_python_root(tmp_path)
    link = root / "autoware_carla_scenario" / "autoware_bridge"
    link.unlink()
    link.symlink_to(tmp_path, target_is_directory=True)

    import random

    state = random.getstate()
    assert host_python_root(tmp_path) == root
    # The temporary name's bits leave the module's (seedable) stream alone.
    assert random.getstate() == state
    assert link.resolve() == bridge_package_dir()
    assert sorted(p.name for p in link.parent.iterdir()) == ["autoware_bridge"]


# Forks processes that make the link at the same moment, in each of the
# directories given. Run in a fresh interpreter, not in the test's own: that is
# an xdist worker with threads (and grpc's fork handlers), and a fork of it
# could inherit a lock one of them held and hang until the CI job timed out.
_MAKE_AT_ONCE = """
import os
import sys
import traceback
from pathlib import Path

from autoware_carla_scenario.autoware_stack.bridge_node import host_python_root

processes = 12
failed = False
for cache in sys.argv[1:]:
    gate, release = os.pipe()
    children = []
    for _ in range(processes):
        child = os.fork()
        if child == 0:  # every child waits for the same moment
            status = 1
            try:
                os.close(release)
                os.read(gate, 1)
                host_python_root(Path(cache))
                status = 0
            except BaseException:
                traceback.print_exc()
            finally:
                os._exit(status)
        children.append(child)
    os.close(gate)
    os.close(release)  # EOF: releases every child at once
    failed |= any(os.waitpid(child, 0)[1] for child in children)
sys.exit(1 if failed else 0)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
def test_the_host_python_root_is_made_by_processes_at_once(tmp_path: Path) -> None:
    """#2079: runs started together all get the link, none an error."""
    caches = [tmp_path / f"trial{trial}" for trial in range(24)]
    done = subprocess.run(
        [sys.executable, "-c", _MAKE_AT_ONCE, *map(str, caches)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert (done.returncode, done.stderr) == (0, "")

    for cache in caches:
        root = host_python_root(cache)
        link = root / "autoware_carla_scenario" / "autoware_bridge"
        assert link.is_symlink()
        assert link.resolve() == bridge_package_dir()
        assert sorted(p.name for p in link.parent.iterdir()) == ["autoware_bridge"]


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


def test_docker_launcher_accepts_a_launch_file_that_takes_scenario_mode(
    tmp_path: Path, fake_docker
) -> None:
    """Another CARLA interface's own launch can carry scenario support."""
    docker, _calls = fake_docker
    ws = _workspace(tmp_path, built=True, scenario=False)
    launch = (
        ws
        / "install"
        / "autoware_carla_interface_v2"
        / "share"
        / "autoware_carla_interface_v2"
        / "launch"
    )
    launch.mkdir(parents=True)
    (launch / "autoware_carla_scenario.launch.xml").write_text(
        '<launch><arg name="scenario_mode" default="true"/></launch>'
    )
    launcher = _docker_launcher(
        ws,
        docker,
        launch=(
            "autoware_carla_interface_v2",
            "autoware_carla_scenario.launch.xml",
            "map_path:=/maps/Town10HD_Opt",
        ),
    )
    assert launcher.has_scenario_support()
    assert not _docker_launcher(ws, docker).has_scenario_support()


# ---------------------------------------------------------------------------
# #2051: liveness_timeout_s against the node's route_watch_period_s
# ---------------------------------------------------------------------------


def test_the_smallest_liveness_timeout_takes_one_lost_poll() -> None:
    assert min_liveness_timeout_s(1.0, 1.0) == 4.0
    assert min_liveness_timeout_s(60.0, 1.0) == 122.0
    assert min_liveness_timeout_s(0.0, 1.0) == 0.0  # the node stops polling
    assert min_liveness_timeout_s(-1.0, 1.0) == 0.0
    assert (
        min_liveness_timeout_s(ROUTE_WATCH_PERIOD_S, MISSION_POLL_TIMEOUT_S) < 30.0
    ), "the defaults suit the default liveness_timeout_s"


@pytest.mark.parametrize(
    "parameters",
    [
        None,  # a node the framework does not start
        {},
        {"use_sim_time": True},
        {"route_watch_period_s": 14.0},
        {"route_watch_period_s": 0},
        {"route_watch_period_s": "not a number"},  # bridge_ros_args() refuses it
        {"route_watch_period_s": True},
        {"route_watch_period_s": float("nan")},
    ],
)
def test_a_liveness_timeout_the_node_cannot_trip_is_no_problem(parameters) -> None:
    assert liveness_timeout_problem(30.0, parameters) is None


def test_a_liveness_timeout_the_node_would_trip_is_a_problem() -> None:
    problem = liveness_timeout_problem(30.0, {"route_watch_period_s": 60})
    assert problem is not None
    assert "route_watch_period_s (60 s)" in problem
    assert "122 s" in problem
    assert liveness_timeout_problem(0.0, {"route_watch_period_s": 60}) is None
    assert liveness_timeout_problem(3.0, {}) is not None  # under the defaults' 4 s


def test_the_launchers_parameters_are_found(tmp_path: Path) -> None:
    command = CommandAutowareLauncher(["true"], bridge_parameters={"a": 1})
    docker = DockerAutowareLauncher(
        DockerAutowareConfig(
            workspace=tmp_path, launch=["x"], bridge_parameters={"b": 2}
        )
    )
    assert launcher_bridge_parameters(command) == {"a": 1}
    assert launcher_bridge_parameters(docker) == {"b": 2}
    assert launcher_bridge_parameters(object()) is None


@pytest.mark.parametrize("value", [10.0, 1e1])
def test_a_whole_float_for_an_integer_parameter_is_passed_as_an_int(
    value: float,
) -> None:
    """#2217: ``reroute_max_attempts:=10.0`` is refused by rclpy, killing the node."""
    assert INTEGER_PARAMETERS == ("reroute_max_attempts",)
    args = bridge_ros_args("h:1", {"reroute_max_attempts": value})
    assert args[3:] == ["-p", "reroute_max_attempts:=10"]


@pytest.mark.parametrize("value", [10.5, float("nan"), float("inf")])
def test_a_fraction_for_an_integer_parameter_is_refused_naming_it(
    value: float,
) -> None:
    with pytest.raises(ValueError, match=r"^reroute_max_attempts must be a whole"):
        bridge_ros_args("h:1", {"reroute_max_attempts": value})


@pytest.mark.parametrize("name", ["reroute_max_attempts", "rpc_timeout_s"])
@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_for_a_number_parameter_is_refused_naming_it(
    name: str, value: bool
) -> None:
    """Not passed as ``:=true``, which rclpy refuses as the node starts."""
    assert name in INTEGER_PARAMETERS + DOUBLE_PARAMETERS
    with pytest.raises(ValueError, match=rf"^{name} must be a number, .* got {value}"):
        bridge_ros_args("h:1", {name: value})


@pytest.mark.parametrize("value", ["60", True, float("nan")])
def test_a_seconds_parameter_bridge_ros_args_refuses_reads_as_the_default(
    value: object,
) -> None:
    """#2275: as bridge_ros_args() refuses it, with an error of its own."""
    assert liveness_timeout_problem(4.0, {"route_watch_period_s": value}) is None
    assert liveness_timeout_problem(3.0, {"route_watch_period_s": value}) is not None


@pytest.mark.parametrize("name", ["reroute_max_attempts", "rpc_timeout_s"])
@pytest.mark.parametrize("value", ["5", "5.0", ""])
def test_a_string_for_a_number_parameter_is_refused_naming_it(
    name: str, value: str
) -> None:
    """#2275: ``rpc_timeout_s:=5`` reached ROS as an INTEGER, which rclpy
    refuses for a DOUBLE, and the run waited out boot_timeout_s."""
    with pytest.raises(
        ValueError, match=rf"^{name} must be a number, .* got {value!r}$"
    ):
        bridge_ros_args("h:1", {name: value})


@pytest.mark.parametrize("name", DOUBLE_PARAMETERS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_double_parameter_that_is_not_finite_is_refused_naming_it(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=rf"^{name} must be finite, got {value!r}$"):
        bridge_ros_args("h:1", {name: value})


@pytest.mark.parametrize("name", POSITIVE_PARAMETERS)
@pytest.mark.parametrize("value", [0, 0.0, -1, -0.5])
def test_a_wait_or_period_that_is_not_positive_is_refused_naming_it(
    name: str, value: float
) -> None:
    """#2275: a negative rpc_timeout_s never let a readiness report through, a
    step_timeout_s of 0 re-sent every startup request on each pass, and a
    tick_period_s of 0 reached create_timer."""
    with pytest.raises(ValueError, match=rf"^{name} must be positive, got {value!r}$"):
        bridge_ros_args("h:1", {name: value})


def test_the_positive_parameters_are_double_parameters() -> None:
    assert set(POSITIVE_PARAMETERS) == set(DOUBLE_PARAMETERS) - {"route_watch_period_s"}


@pytest.mark.parametrize("value", [0, -1.0])
def test_a_route_watch_period_of_zero_or_less_turns_the_watch_off(
    value: float,
) -> None:
    """Not refused: the node stops polling once ready, as it did before
    re-routing (ros_bridge/node.py)."""
    args = bridge_ros_args("h:1", {"route_watch_period_s": value})
    assert args[3:] == ["-p", f"route_watch_period_s:={float(value)}"]


def test_the_nodes_defaults_pass() -> None:
    args = bridge_ros_args(
        "h:1",
        {
            "use_sim_time": True,
            "rpc_timeout_s": 5.0,
            "mission_poll_timeout_s": 1.0,
            "step_timeout_s": 10.0,
            "route_watch_period_s": 1.0,
            "reroute_max_attempts": 10,
            "tick_period_s": 0.5,
        },
    )
    assert len(args) == 3 + 2 * 7


def test_a_null_bridge_parameter_is_the_nodes_default() -> None:
    # #2187: not `auto_engage:=None`, which the node's bool refuses.
    args = bridge_ros_args("h:1", {"auto_engage": None, "use_sim_time": True})
    assert args == [
        "--ros-args",
        "-p",
        "bridge_address:=h:1",
        "-p",
        "use_sim_time:=true",
    ]


def test_a_whole_number_for_a_double_parameter_is_passed_as_a_float() -> None:
    # #2187: rclpy refuses an INTEGER override of a DOUBLE declaration.
    args = bridge_ros_args(
        "h:1",
        {
            "route_watch_period_s": 2,
            "tick_period_s": 0.25,
            "reroute_max_attempts": 3,  # an integer parameter stays one
            "map_frame": "map",
        },
    )
    assert args[3:] == [
        "-p",
        "route_watch_period_s:=2.0",
        "-p",
        "tick_period_s:=0.25",
        "-p",
        "reroute_max_attempts:=3",
        "-p",
        "map_frame:=map",
    ]


def test_the_double_parameters_are_the_nodes_double_declarations() -> None:
    # The table follows ros_bridge/node.py: every parameter it declares with a
    # float default, and no other.
    import ast

    # Read, not imported: it imports rclpy.
    node = bridge_package_dir() / "ros_bridge" / "node.py"
    tree = ast.parse(node.read_text(encoding="utf-8"))
    doubles = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "declare_parameter"
        and len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, float)
    }
    assert doubles
    assert set(DOUBLE_PARAMETERS) == doubles


@pytest.mark.parametrize("xdg", [None, "", "relative/cache"])
def test_the_host_python_root_ignores_an_unset_empty_or_relative_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str | None
) -> None:
    # #2187: not under the current directory.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    if xdg is None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    root = host_python_root()
    assert root.is_relative_to(
        tmp_path / "home" / ".cache" / "autoware_carla_scenario" / "ros_bridge"
    )
