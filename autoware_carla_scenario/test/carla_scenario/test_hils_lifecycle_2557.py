"""The HILS bridge survives a device that fails to open and a CARLA server
that goes down while it runs (#2557).

* A device whose channel cannot be opened stops the bridge before anything
  reaches the ECUs: the devices opened before it are closed again, no
  ``shutdown`` step is sent, and ``scenario-hils`` says so in one line.
* :meth:`HilsBridge.close` closes everything even when a ``shutdown`` step
  fails with an error that is not a step's own.
* A CARLA server that cannot be reached is waited for: the quiet period's
  refresh of the world, the scan for an ego and a run's ``start`` do not end
  the bridge, and a run whose start cannot read CARLA is answered ``error``.
"""

from __future__ import annotations

import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from autoware_carla_scenario.hils.cli import main
from autoware_carla_scenario.hils.config import parse_config

from ._fake_world import ActorsById, FakeActorList

hilspy = pytest.importorskip("hilspy")

from autoware_carla_scenario.hils.bridge import (  # noqa: E402
    HilsBridge,
    HilsOpenError,
)

# -- fakes ---------------------------------------------------------------------


class _Channel:
    """A device's channel that records what is sent and whether it closed."""

    def __init__(self, url: str, log: List[Tuple[str, str]]) -> None:
        self.url = url
        self._log = log

    def send(self, *args: Any) -> None:
        self._log.append(("send", self.url))

    def recv(self, timeout: float) -> None:
        time.sleep(min(timeout, 0.01))
        return None

    def close(self) -> None:
        self._log.append(("close", self.url))


def _opener(
    log: List[Tuple[str, str]], failing: str, error: BaseException
) -> Callable[..., Any]:
    def open_channel(url: str, codec: Optional[str], **options: Any) -> _Channel:
        if url == failing:
            raise error
        return _Channel(url, log)

    return open_channel


def _two_devices(**lifecycle: Any) -> Dict[str, Any]:
    return {
        "runner": {"listen": "127.0.0.1:0"},
        "devices": {"a": {"url": "can://vcan0"}, "b": {"url": "can://vcan1"}},
        "lifecycle": {
            "startup": [{"send": "a", "data": "01"}],
            "shutdown": [{"send": "b", "data": "02"}, {"send": "a", "data": "03"}],
            **lifecycle,
        },
    }


class _Ego:
    type_id = "vehicle.test"

    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self.attributes = {"role_name": "Ego"}
        self.is_alive = True


class _World(ActorsById):
    """A world that never ticks; *down* makes every call to it fail as a
    CARLA server that is gone does."""

    def __init__(self, down: threading.Event, fail: Tuple[str, ...] = ()) -> None:
        self.actors = FakeActorList([_Ego(1)])
        self._down = down
        self._fail = fail

    def _check(self, what: str) -> None:
        if self._down.is_set() or what in self._fail:
            raise RuntimeError(
                f"time-out of 1000ms while waiting for the simulator ({what})"
            )

    def wait_for_tick(self, timeout: float) -> Any:
        time.sleep(min(timeout, 0.02))
        raise RuntimeError("time-out while waiting for the simulator")

    def get_actors(self) -> FakeActorList:
        self._check("get_actors")
        return FakeActorList(a for a in self.actors if a.is_alive)

    def get_map(self) -> Any:
        self._check("get_map")
        return types.SimpleNamespace()

    def get_snapshot(self) -> Any:
        self._check("get_snapshot")
        return types.SimpleNamespace(
            frame=1,
            delta_seconds=0.05,
            timestamp=types.SimpleNamespace(elapsed_seconds=0.05),
            find=lambda actor_id: None,
        )


class _Carla:
    """typesafe_carla's ``carla`` module, as far as the bridge reaches it."""

    def __init__(self, down: threading.Event, fail: Tuple[str, ...] = ()) -> None:
        self.down = down
        self.fail = fail
        self.worlds: List[_World] = []
        self.refused = 0
        carla = self

        class Client:
            def __init__(self, host: str, port: int) -> None:
                pass

            def set_timeout(self, timeout: float) -> None:
                pass

            def get_world(self) -> _World:
                if carla.down.is_set():
                    carla.refused += 1
                    raise RuntimeError(
                        "time-out of 1000ms while waiting for the simulator"
                    )
                world = _World(carla.down, carla.fail)
                carla.worlds.append(world)
                return world

        self.Client = Client


def _one_device(**extra: Any) -> Dict[str, Any]:
    return {
        "runner": {"listen": None},
        "devices": {"a": {"url": "can://vcan0"}},
        **extra,
    }


# -- a device that fails to open -------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        hilspy.Error("unknown codec"),
        OSError(19, 'cannot open SocketCAN interface "vcan1": No such device'),
    ],
    ids=["hilspy-error", "os-error"],
)
def test_a_device_that_fails_to_open_closes_the_others(error: BaseException) -> None:
    log: List[Tuple[str, str]] = []
    bridge = HilsBridge(
        parse_config(_two_devices()),
        carla_module=object(),
        open_channel=_opener(log, "can://vcan1", error),
    )
    with pytest.raises(HilsOpenError, match=r"^b: can://vcan1: ") as raised:
        bridge.startup()
    assert raised.value.__cause__ is error
    # Device a opened, and closed again; nothing reached the ECUs.
    assert log == [("close", "can://vcan0")]
    assert bridge.link_address is None
    # No shutdown step is sent (no startup step was), and nothing raises.
    bridge.close()
    assert log == [("close", "can://vcan0")]


def test_the_command_reports_a_device_that_fails_to_open_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    bus = "mem://hils-2557-open"
    ecu = hilspy.open(bus)
    try:
        document = {
            "carla": {"port": 1},
            "runner": {"listen": "127.0.0.1:0"},
            "devices": {
                "a": {"url": bus},
                # nmea takes no such option: hilspy.open refuses it.
                "b": {
                    "url": "mem://hils-2557-other",
                    "codec": "nmea",
                    "options": {"bogus": 1},
                },
            },
            "lifecycle": {
                "startup": [{"send": "a", "data": "01"}],
                "shutdown": [{"send": "a", "data": "02"}],
            },
        }
        path = tmp_path / "hils.yaml"
        path.write_text(yaml.safe_dump(document))
        assert main([str(path)]) == 1
        assert ecu.recv(0.2) is None
    finally:
        ecu.close()
    err = capsys.readouterr().err
    assert err.startswith("scenario-hils: b: mem://hils-2557-other: ")
    assert "bogus" in err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


# -- close() -----------------------------------------------------------------------


def test_close_closes_everything_when_a_shutdown_step_raises() -> None:
    log: List[Tuple[str, str]] = []
    bridge = HilsBridge(
        parse_config(_two_devices()),
        carla_module=object(),
        open_channel=_opener(log, "", RuntimeError()),
    )
    bridge.startup()
    assert bridge.link_address is not None
    run_step = bridge._run_step
    calls: List[Any] = []

    def failing_first(step: Any, *args: Any) -> None:
        calls.append(step)
        if len(calls) == 1 and step.device == "b":
            raise KeyError(step.device)
        run_step(step, *args)

    bridge._run_step = failing_first  # type: ignore[method-assign,assignment]
    bridge.close()
    # The second shutdown step still ran, and every channel closed.
    assert ("send", "can://vcan0") in log[1:]
    assert sorted(e for e in log if e[0] == "close") == [
        ("close", "can://vcan0"),
        ("close", "can://vcan1"),
    ]
    assert bridge.link_address is None


def test_close_closes_everything_when_ending_the_scenario_fails() -> None:
    log: List[Tuple[str, str]] = []
    bridge = HilsBridge(
        parse_config(_two_devices()),
        carla_module=object(),
        open_channel=_opener(log, "", RuntimeError()),
    )
    bridge.startup()

    def finish() -> None:
        raise RuntimeError("the scenario could not end")

    bridge._active = True
    bridge._finish = finish  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="could not end"):
        bridge.close()
    # The shutdown steps ran, and everything closed.
    assert log.count(("send", "can://vcan0")) == 2
    assert ("send", "can://vcan1") in log
    assert ("close", "can://vcan0") in log and ("close", "can://vcan1") in log
    assert bridge.link_address is None
    # The scenario is over: a second close() neither ends it again nor sends.
    assert not bridge.active
    sent = len(log)
    bridge.close()
    assert len(log) == sent


# -- a CARLA server that goes down ---------------------------------------------------


def test_the_bridge_waits_for_a_carla_that_goes_down_and_comes_back() -> None:
    down = threading.Event()
    carla = _Carla(down)
    bridge = HilsBridge(
        parse_config(_one_device(carla={"timeout": 0.1})),
        carla_module=carla,
        open_channel=_opener([], "", RuntimeError()),
    )
    bridge.connect()
    first = bridge._world

    def quiet_step() -> bool:
        # A quiet period has passed: the next missed tick refreshes the world.
        bridge._last_tick = time.monotonic() - 1.0
        return bridge.step(0.0)

    down.set()
    # Not CARLA's RuntimeError: the world held is kept, and waited on.
    assert quiet_step() is False
    assert quiet_step() is False
    assert carla.refused == 2
    assert bridge._world is first
    down.clear()
    assert quiet_step() is False
    # Once back, the world was taken anew.
    assert len(carla.worlds) == 2
    assert bridge._world is carla.worlds[-1]
    bridge.close()


def test_the_scan_for_an_ego_waits_for_a_carla_that_is_down() -> None:
    down = threading.Event()
    bridge = HilsBridge(
        parse_config(_one_device()),
        carla_module=_Carla(down),
        open_channel=_opener([], "", RuntimeError()),
    )
    bridge.connect()
    down.set()
    bridge._find_ego()
    assert not bridge.active
    down.clear()
    bridge._next_scan = 0.0
    bridge._find_ego()
    assert bridge.active and bridge.ego is not None
    bridge.close()


def _started(carla: _Carla) -> Tuple[HilsBridge, List[Dict[str, Any]]]:
    bridge = HilsBridge(
        parse_config(_one_device()),
        carla_module=carla,
        open_channel=_opener([], "", RuntimeError()),
    )
    bridge.connect()
    sent: List[Dict[str, Any]] = []
    bridge._link = types.SimpleNamespace(send=sent.append)  # type: ignore[assignment]
    return bridge, sent


@pytest.mark.parametrize("fail", ["get_map", "get_snapshot"])
def test_a_start_that_cannot_read_carla_is_answered_error(fail: str) -> None:
    bridge, sent = _started(_Carla(threading.Event(), (fail,)))
    bridge._handle(("start", {"ego_id": 1, "scenario": "t"}))
    assert [m["type"] for m in sent] == ["error"]
    assert "CARLA cannot be read" in sent[0]["message"]
    assert fail in sent[0]["message"]
    # No scenario is left waiting on.
    assert not bridge.active
    assert bridge.ego is None
    bridge._link = None
    bridge.close()


def test_a_start_while_carla_is_down_is_answered_error() -> None:
    down = threading.Event()
    bridge, sent = _started(_Carla(down))
    down.set()
    bridge._handle(("start", {"ego_id": 1, "scenario": "t"}))
    assert [m["type"] for m in sent] == ["error"]
    assert "CARLA cannot be read" in sent[0]["message"]
    assert not bridge.active
    bridge._link = None
    bridge.close()
