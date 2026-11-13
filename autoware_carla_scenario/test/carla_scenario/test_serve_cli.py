"""The command lines of ``viewer`` and ``scenario-editor`` (#712).

Both used to ignore their arguments: ``viewer --help`` or
``scenario-editor --port 8080`` started the server on ``0.0.0.0`` instead.
``uvicorn.run`` is patched, so nothing here serves.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from autoware_carla_scenario import editor, ui
from autoware_carla_scenario.serve_cli import (
    check_port,
    is_loopback,
    port,
    warn_if_exposed,
)

_ENV = (
    "VIEWER_HOST",
    "VIEWER_PORT",
    "VIEWER_BASE_PATH",
    "SCENARIO_EDITOR_HOST",
    "SCENARIO_EDITOR_PORT",
    "SCENARIO_EDITOR_DRAFTS",
    "SCENARIO_EDITOR_EXPORT_DIR",
)

_MAINS: dict[str, Callable[[list[str] | None], None]] = {
    "viewer": ui.main,
    "scenario-editor": editor.main,
}
_PREFIX = {"viewer": "VIEWER", "scenario-editor": "SCENARIO_EDITOR"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The calls to ``uvicorn.run``, which serves nothing here."""
    import uvicorn

    calls: list[dict[str, Any]] = []

    def run(app: object, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", run)
    return calls


@pytest.fixture
def base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the viewer's BASE_PATH, which ``--base-path`` sets."""
    from autoware_carla_scenario.ui import app as app_module

    monkeypatch.setattr(app_module, "BASE_PATH", app_module.BASE_PATH)


@pytest.fixture(autouse=True)
def viewer_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the Host names the viewer answers to, which ``--host`` sets."""
    from autoware_carla_scenario.ui import app as app_module

    monkeypatch.setattr(app_module.HOSTS, "bound_host", app_module.HOSTS.bound_host)


@pytest.mark.parametrize("prog", sorted(_MAINS))
class TestBothCommands:
    def test_help_exits_without_serving(
        self,
        prog: str,
        served: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog](["--help"])
        assert exc.value.code == 0
        assert served == []
        out = capsys.readouterr().out
        assert f"usage: {prog}" in out
        assert "--host" in out and "--port" in out

    def test_an_unknown_argument_is_an_error(
        self, prog: str, served: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog](["--no-such-option"])
        assert exc.value.code == 2
        assert served == []

    def test_host_and_port_reach_uvicorn(
        self, prog: str, served: list[dict[str, Any]], base_path: None
    ) -> None:
        _MAINS[prog](["--host", "127.0.0.2", "--port", "8080"])
        assert len(served) == 1
        assert served[0]["host"] == "127.0.0.2"
        assert served[0]["port"] == 8080

    def test_the_default_host_is_loopback(
        self, prog: str, served: list[dict[str, Any]], base_path: None
    ) -> None:
        _MAINS[prog]([])
        assert served[0]["host"] == "127.0.0.1"
        assert served[0]["port"] == (9000 if prog == "viewer" else 9100)

    def test_the_environment_still_configures_it(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_HOST", "127.0.0.3")
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "8181")
        _MAINS[prog]([])
        assert (served[0]["host"], served[0]["port"]) == ("127.0.0.3", 8181)

    def test_a_flag_wins_over_the_environment(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_HOST", "127.0.0.3")
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "8181")
        _MAINS[prog](["--host", "127.0.0.4", "--port", "8282"])
        assert (served[0]["host"], served[0]["port"]) == ("127.0.0.4", 8282)

    def test_a_malformed_port_is_an_error(
        self,
        prog: str,
        served: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", "eighty")
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog]([])
        assert exc.value.code == 2
        assert served == []

    @pytest.mark.parametrize("bad", ["0", "-1", "70000", "abc"])
    def test_an_out_of_range_port_flag_is_a_usage_error(
        self,
        prog: str,
        bad: str,
        served: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Out of range used to reach uvicorn, whose asyncio bind raised an
        # OverflowError traceback (#1774).
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog](["--port", bad])
        assert exc.value.code == 2
        assert served == []
        err = capsys.readouterr().err
        assert f"usage: {prog}" in err
        assert "argument --port" in err

    @pytest.mark.parametrize("bad", ["0", "-1", "70000", "abc"])
    def test_an_out_of_range_port_variable_is_a_usage_error(
        self,
        prog: str,
        bad: str,
        served: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", bad)
        with pytest.raises(SystemExit) as exc:
            _MAINS[prog]([])
        assert exc.value.code == 2
        assert served == []
        err = capsys.readouterr().err
        assert f"usage: {prog}" in err
        assert "argument --port" in err

    @pytest.mark.parametrize("edge", [1, 65535])
    def test_the_ends_of_the_port_range_are_served(
        self,
        prog: str,
        edge: int,
        served: list[dict[str, Any]],
        base_path: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _MAINS[prog](["--port", str(edge)])
        monkeypatch.setenv(f"{_PREFIX[prog]}_PORT", str(edge))
        _MAINS[prog]([])
        assert [call["port"] for call in served] == [edge, edge]

    def test_every_interface_is_served_with_a_warning(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _MAINS[prog](["--host", "0.0.0.0"])  # noqa: S104
        assert served[0]["host"] == "0.0.0.0"  # noqa: S104
        assert "WARNING" in capsys.readouterr().err

    def test_loopback_is_served_without_a_warning(
        self,
        prog: str,
        served: list[dict[str, Any]],
        base_path: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _MAINS[prog](["--host", "127.0.0.1"])
        assert "WARNING" not in capsys.readouterr().err


class TestViewer:
    def test_base_path_reaches_the_app(
        self, served: list[dict[str, Any]], base_path: None, tmp_path: Path
    ) -> None:
        from autoware_carla_scenario.ui import app as app_module

        ui.main(["--base-path", str(tmp_path)])
        assert app_module.BASE_PATH == tmp_path.resolve()

    def test_the_bound_host_reaches_the_app(
        self, served: list[dict[str, Any]], base_path: None
    ) -> None:
        """The viewer answers the Host it is bound to (web_guard, #1844)."""
        from autoware_carla_scenario.ui import app as app_module

        ui.main(["--host", "box.lan"])
        assert app_module.HOSTS.bound_host == "box.lan"
        assert app_module.HOSTS.allows("box.lan")
        assert not app_module.HOSTS.allows("evil.example")

    def test_fetch_rerun_web_viewer_still_fetches_instead_of_serving(
        self, served: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetched: list[list[str]] = []
        monkeypatch.setattr(ui, "_fetch_web_viewer", fetched.append)
        ui.main(["--fetch-rerun-web-viewer", "--from", "x.tgz", "--version", "1"])
        assert fetched == [["--from", "x.tgz", "--version", "1"]]
        assert served == []

    def test_fetch_help_is_the_fetch_commands_own(
        self, served: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            ui.main(["--fetch-rerun-web-viewer", "--help"])
        assert exc.value.code == 0
        assert "--from" in capsys.readouterr().out
        assert served == []

    def test_a_mistyped_fetch_flag_is_an_error(
        self, served: list[dict[str, Any]]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            ui.main(["--fetch-rerun-web-viewr"])
        assert exc.value.code == 2
        assert served == []


class TestEditor:
    def test_directories_reach_the_app(
        self,
        served: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from autoware_carla_scenario.editor import app as app_module

        made: list[dict[str, Any]] = []

        def create_app(**kwargs: Any) -> object:
            made.append(kwargs)
            return object()

        monkeypatch.setattr(app_module, "create_app", create_app)
        monkeypatch.setenv("SCENARIO_EDITOR_EXPORT_DIR", str(tmp_path / "env"))
        editor.main(["--drafts", str(tmp_path / "drafts")])
        assert made == [
            {
                "draft_dir": tmp_path / "drafts",
                "export_dir": tmp_path / "env",
                "host": "127.0.0.1",
            }
        ]

    def test_the_bound_host_reaches_the_app(
        self, served: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The editor answers the Host it is bound to (web_guard, #1844)."""
        from autoware_carla_scenario.editor import app as app_module

        made: list[dict[str, Any]] = []

        def create_app(**kwargs: Any) -> object:
            made.append(kwargs)
            return object()

        monkeypatch.setattr(app_module, "create_app", create_app)
        editor.main(["--host", "box.lan"])
        assert made[0]["host"] == "box.lan"


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("[::1]", True),
        ("::ffff:127.0.0.1", True),
        ("localhost", True),
        ("LOCALHOST", True),
        ("0.0.0.0", False),  # noqa: S104
        ("::", False),
        ("[::]", False),
        ("192.168.0.1", False),
        ("::ffff:192.168.0.1", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_is_loopback(host: str, loopback: bool) -> None:
    assert is_loopback(host) is loopback


def test_the_warning_brackets_an_ipv6_address(
    capsys: pytest.CaptureFixture[str],
) -> None:
    warn_if_exposed("viewer", "::", 9000, "do things")
    assert "listening on [::]:9000" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("0", "must be within 1-65535, got 0"),
        ("-1", "must be within 1-65535, got -1"),
        ("65536", "must be within 1-65535, got 65536"),
        ("abc", "expected an integer, got 'abc'"),
        ("2000.7", "expected an integer, got '2000.7'"),
    ],
)
def test_port_refuses_what_is_not_a_tcp_port(text: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        port(text)


def test_port_accepts_the_tcp_range() -> None:
    assert [port(text) for text in ("1", "2000", "65535")] == [1, 2000, 65535]


@pytest.mark.parametrize("number", [0, -1, 65536, 67536])
def test_check_port_refuses_a_number_outside_the_tcp_range(number: int) -> None:
    # The range check on its own, for a port no argparse option reads (#1781).
    with pytest.raises(ValueError, match=f"must be within 1-65535, got {number}"):
        check_port(number)


def test_check_port_returns_a_number_in_the_tcp_range() -> None:
    assert [check_port(n) for n in (1, 2000, 65535)] == [1, 2000, 65535]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("prog", sorted(_MAINS))
def test_the_console_script_answers_help(prog: str) -> None:
    """The reproduction of #712, run as the console script runs it.

    Before the fix, ``--help`` started the server, which never exits; the
    port is a free one so that server could not collide with another.
    """
    module = "ui" if prog == "viewer" else "editor"
    env = {**os.environ, f"{_PREFIX[prog]}_PORT": str(_free_port())}
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.argv[0] = {prog!r}; "
                f"from autoware_carla_scenario.{module} import main; main()",
                "--help",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"`{prog} --help` served instead of printing its usage")
    assert result.returncode == 0, result.stderr
    assert f"usage: {prog}" in result.stdout
    assert "Uvicorn running" not in result.stderr


# ---------------------------------------------------------------------------
# Pins of what serve_cli does, byte for byte, kept as it is when its checkable
# part moves into a module the library check compiles.
# ---------------------------------------------------------------------------


def test_serve_cli_exports_the_same_names() -> None:
    from autoware_carla_scenario import serve_cli

    assert serve_cli.__all__ == [
        "DEFAULT_HOST",
        "add_bind_arguments",
        "check_port",
        "is_loopback",
        "port",
        "warn_if_exposed",
    ]
    assert serve_cli.DEFAULT_HOST == "127.0.0.1"


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        # The name, upper or lower case, with any dots at its end.
        ("localhost.", True),
        ("Localhost.", True),
        ("LOCALHOST.", True),
        ("localhost..", True),
        ("localhost.localdomain", False),
        ("x.localhost", False),
        ("[localhost]", False),
        # IPv4: all of 127.0.0.0/8, written as ipaddress reads it only.
        ("127.255.255.255", True),
        ("128.0.0.1", False),
        ("127.1", False),
        ("127.0.0.01", False),
        (" 127.0.0.1", False),
        ("127.0.0.1 ", False),
        # Brackets are stripped from either end, however many.
        ("[127.0.0.1]", True),
        ("[[::1]]", True),
        ("::1]", True),
        # IPv6: ::1 however it is written, an IPv4-mapped loopback address,
        # a scoped one; not another embedding of 127.0.0.1.
        ("0:0:0:0:0:0:0:1", True),
        ("::1%lo", True),
        ("[::ffff:127.0.0.1]", True),
        ("::ffff:7f00:1", True),
        ("::ffff:127.0.0.1%lo", True),
        ("::ffff:127.0.0.1%1", True),
        ("::ffff:0:127.0.0.1", False),
        ("::127.0.0.1", False),
        ("64:ff9b::7f00:1", False),
        ("2002:7f00:1::", False),
        ("fe80::1%eth0", False),
    ],
)
def test_is_loopback_reads_hosts_as_before(host: str, loopback: bool) -> None:
    assert is_loopback(host) is loopback


@pytest.mark.parametrize(
    ("host", "shown"),
    [
        ("0.0.0.0", "0.0.0.0"),  # noqa: S104
        ("::", "[::]"),
        ("[::]", "[::]"),
        ("fe80::1%eth0", "[fe80::1%eth0]"),
        ("box.lan", "box.lan"),
        ("", ""),
        ("::ffff:192.168.0.1", "[::ffff:192.168.0.1]"),
    ],
)
def test_warn_if_exposed_writes_one_line_to_stderr(
    host: str, shown: str, capsys: pytest.CaptureFixture[str]
) -> None:
    warn_if_exposed("prog", host, 1234, "edit drafts")
    out, err = capsys.readouterr()
    assert out == ""
    assert err == (
        f"prog: WARNING: listening on {shown}:1234 with no authentication; "
        "anyone who can reach this port can edit drafts. Pass --host "
        "127.0.0.1 to serve this machine only.\n"
    )


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "::1", "[::1]", "localhost", "LOCALHOST."]
)
def test_warn_if_exposed_says_nothing_of_loopback(
    host: str, capsys: pytest.CaptureFixture[str]
) -> None:
    warn_if_exposed("prog", host, 1234, "edit drafts")
    assert capsys.readouterr() == ("", "")


def test_warn_if_exposed_writes_to_the_stderr_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    warn_if_exposed("viewer", "box.lan", 9000, "read the recordings")
    assert stream.getvalue() == (
        "viewer: WARNING: listening on box.lan:9000 with no authentication; "
        "anyone who can reach this port can read the recordings. Pass --host "
        "127.0.0.1 to serve this machine only.\n"
    )


@pytest.mark.parametrize(
    ("number", "message"),
    [
        (0, "must be within 1-65535, got 0"),
        (65536, "must be within 1-65535, got 65536"),
    ],
)
def test_check_port_raises_a_plain_value_error(number: int, message: str) -> None:
    with pytest.raises(ValueError) as exc:
        check_port(number)
    assert type(exc.value) is ValueError
    assert exc.value.args == (message,)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("0", "must be within 1-65535, got 0"),
        ("70000", "must be within 1-65535, got 70000"),
        ("abc", "expected an integer, got 'abc'"),
        ("", "expected an integer, got ''"),
        ("1.0", "expected an integer, got '1.0'"),
    ],
)
def test_port_raises_an_argument_type_error_without_a_cause(
    text: str, message: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        port(text)
    assert exc.value.args == (message,)
    assert exc.value.__cause__ is None
    assert exc.value.__suppress_context__ is True


def test_port_reads_what_int_reads() -> None:
    assert [port(text) for text in (" 80 ", "8_080", "+443", "٨٠")] == [
        80,
        8080,
        443,
        80,
    ]


def _bind_parser(
    monkeypatch: pytest.MonkeyPatch, **env: str
) -> argparse.ArgumentParser:
    from autoware_carla_scenario.serve_cli import add_bind_arguments

    for name in ("PIN_HOST", "PIN_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    parser = argparse.ArgumentParser(prog="pin")
    add_bind_arguments(
        parser, host_env="PIN_HOST", port_env="PIN_PORT", default_port=9100
    )
    return parser


def test_add_bind_arguments_adds_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _bind_parser(monkeypatch)
    actions = {a.dest: a for a in parser._actions}
    assert sorted(actions) == ["help", "host", "port"]
    host, port_action = actions["host"], actions["port"]
    assert host.option_strings == ["--host"]
    assert host.default == "127.0.0.1"
    assert host.type is None
    assert host.help == (
        "address to bind (env PIN_HOST; default 127.0.0.1, this machine only). "
        "0.0.0.0 serves every interface, with no authentication"
    )
    assert port_action.option_strings == ["--port"]
    assert port_action.default == "9100"
    assert port_action.type is port
    assert port_action.help == "port to listen on, 1-65535 (env PIN_PORT; default 9100)"
    assert vars(parser.parse_args([])) == {"host": "127.0.0.1", "port": 9100}


@pytest.mark.parametrize(
    ("env", "argv", "parsed"),
    [
        (
            {"PIN_HOST": "box.lan", "PIN_PORT": "81"},
            [],
            {"host": "box.lan", "port": 81},
        ),
        ({"PIN_HOST": "", "PIN_PORT": ""}, [], {"host": "127.0.0.1", "port": 9100}),
        (
            {"PIN_HOST": "box.lan", "PIN_PORT": "81"},
            ["--host", "::", "--port", "82"],
            {"host": "::", "port": 82},
        ),
        ({}, ["--host="], {"host": "", "port": 9100}),
    ],
)
def test_add_bind_arguments_reads_the_environment_then_the_flags(
    env: dict[str, str],
    argv: list[str],
    parsed: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _bind_parser(monkeypatch, **env)
    assert vars(parser.parse_args(argv)) == parsed


@pytest.mark.parametrize(
    ("env", "argv", "error"),
    [
        ({}, ["--port", "0"], "argument --port: must be within 1-65535, got 0"),
        ({}, ["--port", "x"], "argument --port: expected an integer, got 'x'"),
        (
            {"PIN_PORT": "99999"},
            [],
            "argument --port: must be within 1-65535, got 99999",
        ),
        ({}, ["--port"], "argument --port: expected one argument"),
    ],
)
def test_add_bind_arguments_refuses_a_bad_port_with_exit_2(
    env: dict[str, str],
    argv: list[str],
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _bind_parser(monkeypatch, **env)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.endswith(f"\npin: error: {error}\n")
    assert err.startswith("usage: pin [-h] [--host HOST] [--port PORT]\n")
