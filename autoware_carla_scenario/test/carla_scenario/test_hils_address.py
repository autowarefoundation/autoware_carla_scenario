"""The HILS bridge's addresses take IPv6 (issue #1288).

``parse_address`` used to split on the last ``:``, so ``::1`` was host ``:``
on port 1, ``[::1]:7700`` kept its brackets (which no socket takes) and
``[::1]`` raised ``int('1]')``.
"""

from __future__ import annotations

import queue
import socket
from pathlib import Path
from typing import Dict

import pytest

from autoware_carla_scenario.entity.hils_entity import HilsEgoConfig
from autoware_carla_scenario.hils.config import (
    HilsConfigError,
    load_config,
    parse_config,
)
from autoware_carla_scenario.hils.link import RunnerLink
from autoware_carla_scenario.hils.wire import (
    DEFAULT_PORT,
    format_address,
    parse_address,
)

_DEVICES = {"devices": {"can": {"url": "mem://x"}}}


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        # host:port and a bare host.
        ("127.0.0.1:7701", ("127.0.0.1", 7701)),
        ("localhost:0", ("localhost", 0)),
        ("localhost:65535", ("localhost", 65535)),
        ("hils-pc", ("hils-pc", DEFAULT_PORT)),
        ("10.0.0.2", ("10.0.0.2", DEFAULT_PORT)),
        (":7701", ("0.0.0.0", 7701)),
        # [host]:port and [host].
        ("[::1]:7701", ("::1", 7701)),
        ("[::]:0", ("::", 0)),
        ("[::1]", ("::1", DEFAULT_PORT)),
        ("[fe80::1%eth0]:7701", ("fe80::1%eth0", 7701)),
        ("[localhost]:7701", ("localhost", 7701)),
        # An IPv6 address without brackets is all host.
        ("::1", ("::1", DEFAULT_PORT)),
        ("::", ("::", DEFAULT_PORT)),
        ("fe80::1", ("fe80::1", DEFAULT_PORT)),
        ("fe80::1:7701", ("fe80::1:7701", DEFAULT_PORT)),
        ("2001:db8::8a2e:370:7334", ("2001:db8::8a2e:370:7334", DEFAULT_PORT)),
    ],
)
def test_parse_address_forms(address: str, expected: tuple) -> None:
    assert parse_address(address) == expected


def test_parse_address_default_port() -> None:
    assert parse_address("[::1]", 9) == ("::1", 9)
    assert parse_address("::1", 9) == ("::1", 9)
    assert parse_address("host", 9) == ("host", 9)


@pytest.mark.parametrize(
    ("address", "message"),
    [
        ("host:", r"the port after ':' is empty"),
        ("[::1]:", r"the port after ':' is empty"),
        ("host:http", r"the port 'http' is not a whole number"),
        ("host:-1", r"the port '-1' is not a whole number"),
        ("host:+1", r"the port '\+1' is not a whole number"),
        ("host: 1", r"the port ' 1' is not a whole number"),
        ("host:1.5", r"the port '1.5' is not a whole number"),
        ("host:²", r"is not a whole number"),
        ("[::1]:x", r"the port 'x' is not a whole number"),
        ("host:65536", r"the port 65536 is not within 0-65535"),
        ("[::1]:99999", r"the port 99999 is not within 0-65535"),
        ("[::1", r"'\[' is never closed by '\]'"),
        ("[]:7700", r"no host between '\[' and '\]'"),
        ("[]", r"no host between"),
        ("[::1]7700", r"expected ':port' after '\]', got '7700'"),
        ("[::1]]:7700", r"expected ':port' after '\]'"),
        ("[[::1]]:7700", r"a '\[' inside the brackets"),
        ("::1]:7700", r"brackets go around the host only"),
        ("host]:7700", r"brackets go around the host only"),
    ],
)
def test_parse_address_refuses(address: str, message: str) -> None:
    with pytest.raises(ValueError, match=message) as info:
        parse_address(address)
    assert repr(address) in str(info.value)


@pytest.mark.parametrize(
    ("host", "port", "text"),
    [
        ("127.0.0.1", 7700, "127.0.0.1:7700"),
        ("localhost", 0, "localhost:0"),
        ("::1", 7700, "[::1]:7700"),
        ("fe80::1%eth0", 1, "[fe80::1%eth0]:1"),
    ],
)
def test_format_address_round_trips(host: str, port: int, text: str) -> None:
    assert format_address(host, port) == text
    assert parse_address(text) == (host, port)


@pytest.mark.parametrize(
    "listen",
    ["[::1]:7701", "[::]", "::1", "0.0.0.0:0", "localhost", ":7700"],
)
def test_config_takes_listen_forms(listen: str) -> None:
    config = parse_config({**_DEVICES, "runner": {"listen": listen}})
    assert config.runner_listen == listen


@pytest.mark.parametrize(
    ("listen", "message"),
    [
        ("0.0.0.0:70000", r"the port 70000 is not within 0-65535"),
        ("[::1]:http", r"the port 'http' is not a whole number"),
        ("[::1", r"never closed"),
        ("host:", r"the port after ':' is empty"),
    ],
)
def test_config_refuses_bad_listen(listen: str, message: str) -> None:
    with pytest.raises(HilsConfigError, match=r"runner\.listen: .*" + message):
        parse_config({**_DEVICES, "runner": {"listen": listen}})


def test_load_config_refuses_bad_listen_early(tmp_path: Path) -> None:
    """The loader names the setting: a bad port is not a crash at bind time."""
    path = tmp_path / "hils.yaml"
    path.write_text(
        'runner:\n  listen: "[::1]:99999"\ndevices:\n  can: {url: "mem://x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        HilsConfigError, match=r"runner\.listen: address '\[::1\]:99999'"
    ):
        load_config(path)


def test_load_config_takes_bracketed_ipv6(tmp_path: Path) -> None:
    path = tmp_path / "hils.yaml"
    path.write_text(
        'runner:\n  listen: "[::1]:7701"\ndevices:\n  can: {url: "mem://x"}\n',
        encoding="utf-8",
    )
    assert load_config(path).runner_listen == "[::1]:7701"


def _ipv6_loopback() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _ipv6_loopback(), reason="no IPv6 loopback")
@pytest.mark.parametrize("address", ["[::1]:0", "::1"])
def test_runner_link_listens_on_ipv6(address: str) -> None:
    if address == "::1":
        # The default port may be taken on this host; bracket form covers 0.
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind(("::1", DEFAULT_PORT))
        except OSError:
            pytest.skip(f"port {DEFAULT_PORT} is taken")
    link = RunnerLink(address, queue.Queue())
    try:
        host, port = link.address
        assert host == "::1"
        # The run's side connects the same way the HILS entity does.
        with socket.create_connection((host, port), timeout=2.0):
            pass
    finally:
        link.close()


def test_runner_link_listens_on_ipv4() -> None:
    link = RunnerLink("127.0.0.1:0", queue.Queue())
    try:
        host, port = link.address
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        link.close()


def test_hils_entity_host_takes_brackets() -> None:
    assert HilsEgoConfig(host="[::1]").host == "::1"
    assert HilsEgoConfig(host="::1").host == "::1"
    assert HilsEgoConfig(host="hils-pc").host == "hils-pc"


@pytest.mark.parametrize("host", ["[::1]:7700", "[::1", "::1]"])
def test_hils_entity_host_refuses_a_port(host: str) -> None:
    with pytest.raises(ValueError, match=r"hils\.host"):
        HilsEgoConfig(host=host)


@pytest.mark.parametrize("port", [0, -1, 65536, True])
def test_hils_entity_port_in_range(port: int) -> None:
    with pytest.raises(ValueError, match=r"hils\.port"):
        HilsEgoConfig(port=port)


@pytest.mark.parametrize(
    ("address", "where", "family"),
    [
        ("[::1]:7701", ("::1", 7701), socket.AF_INET6),
        ("::", ("::", DEFAULT_PORT), socket.AF_INET6),
        ("127.0.0.1:7701", ("127.0.0.1", 7701), socket.AF_INET),
        ("localhost", ("localhost", DEFAULT_PORT), socket.AF_INET),
    ],
)
def test_runner_link_binds_the_address_family(
    monkeypatch: pytest.MonkeyPatch,
    address: str,
    where: tuple,
    family: socket.AddressFamily,
) -> None:
    """Without IPv6 here, check what the link asks the socket module for."""
    import autoware_carla_scenario.hils.link as link_module

    asked: Dict[str, object] = {}

    class _Server:
        def settimeout(self, _: float) -> None:
            pass

        def close(self) -> None:
            pass

    def create_server(address: tuple, **kwargs: object) -> _Server:
        asked["address"] = address
        asked.update(kwargs)
        return _Server()

    monkeypatch.setattr(link_module.socket, "create_server", create_server)
    RunnerLink(address, queue.Queue()).close()
    assert asked["address"] == where
    assert asked["family"] == family
