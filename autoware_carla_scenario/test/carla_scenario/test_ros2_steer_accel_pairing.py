"""A ``steer_accel`` source pairs each cycle's steering with that cycle's acceleration.

Regression tests for #688: the drain used to take every queued steering sample
before any acceleration sample, so a poll holding two cycles, or a cycle split
across two polls, paired steering with the wrong cycle's acceleration, and the
offset never recovered.

The readers are fakes standing in for ``cyclone._reader`` / ``cyclone._take``,
so no DDS traffic is needed; each sample carries the ``sample_info`` with the
``source_timestamp`` Cyclone annotates every taken sample with.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest

cyclone = pytest.importorskip("autoware_carla_scenario.ros2.cyclone")

from autoware_carla_scenario.ros2.commands import Command, SteerAccelCommand  # noqa: E402
from autoware_carla_scenario.ros2.config import CommandSourceConfig  # noqa: E402

#: The publisher's cycle, as VisionPilot's 10 Hz.
_CYCLE_NS = 100_000_000
#: Within one cycle, the acceleration is written this long after the steering.
_GAP_NS = 1_000_000


class _FakeReader:
    def __init__(self) -> None:
        self.queue: List[Any] = []


Publish = Callable[[str, float, int], None]
Drain = Callable[[], List[Command]]


@pytest.fixture
def drain(monkeypatch: pytest.MonkeyPatch) -> Tuple[Drain, Publish]:
    readers: Dict[str, _FakeReader] = {}

    def fake_reader(participant: Any, name: str, message_type: Any, depth: int) -> Any:
        readers[name] = _FakeReader()
        return readers[name]

    def fake_take(reader: _FakeReader, depth: int) -> List[Any]:
        taken, reader.queue = reader.queue[:depth], reader.queue[depth:]
        return taken

    monkeypatch.setattr(cyclone, "_reader", fake_reader)
    monkeypatch.setattr(cyclone, "_take", fake_take)
    source = CommandSourceConfig(
        name="vp", kind="steer_accel", steering_topic="/s", acceleration_topic="/a"
    )
    fn = cyclone._source_drain(None, source)

    def publish(topic: str, value: float, stamp_ns: int) -> None:
        readers[topic].queue.append(
            SimpleNamespace(
                data=value, sample_info=SimpleNamespace(source_timestamp=stamp_ns)
            )
        )

    return fn, publish


def _cycle(
    publish: Publish, k: int, *, steering: bool = True, acceleration: bool = True
) -> None:
    """Publish cycle *k*: steering ``k / 10`` then acceleration ``k``."""
    if steering:
        publish("/s", k / 10, k * _CYCLE_NS)
    if acceleration:
        publish("/a", float(k), k * _CYCLE_NS + _GAP_NS)


def _pairs(commands: Sequence[Command]) -> List[Tuple[float, float]]:
    out = []
    for command in commands:
        assert isinstance(command, SteerAccelCommand)
        out.append((command.steering_rad, command.acceleration_mps2))
    return out


def _expected(*cycles: int) -> List[Tuple[float, float]]:
    return [(k / 10, float(k)) for k in cycles]


def test_two_cycles_in_one_poll(drain: Tuple[Drain, Publish]) -> None:
    fn, publish = drain
    _cycle(publish, 1)
    _cycle(publish, 2)
    got = _pairs(fn())
    _cycle(publish, 3)
    got += _pairs(fn())
    assert got == _expected(1, 2, 3)


def test_a_cycle_split_across_polls(drain: Tuple[Drain, Publish]) -> None:
    fn, publish = drain
    _cycle(publish, 1, acceleration=False)  # steering before the tick...
    got = _pairs(fn())
    assert got == []
    _cycle(publish, 1, steering=False)  # ...its acceleration after it
    _cycle(publish, 2)
    got += _pairs(fn())
    _cycle(publish, 3)
    got += _pairs(fn())
    assert got == _expected(1, 2, 3)


def test_the_acceleration_written_first_still_pairs(
    drain: Tuple[Drain, Publish],
) -> None:
    fn, publish = drain
    for k in (1, 2):
        publish("/a", float(k), k * _CYCLE_NS)
        publish("/s", k / 10, k * _CYCLE_NS + _GAP_NS)
    assert _pairs(fn()) == _expected(1, 2)


def test_a_lost_steering_message_drops_one_cycle_only(
    drain: Tuple[Drain, Publish],
) -> None:
    fn, publish = drain
    got: List[Tuple[float, float]] = []
    for k in range(1, 6):
        _cycle(publish, k, steering=k != 2)
        got += _pairs(fn())
    assert got == _expected(1, 3, 4, 5)


def test_a_lost_acceleration_message_drops_one_cycle_only(
    drain: Tuple[Drain, Publish],
) -> None:
    fn, publish = drain
    got: List[Tuple[float, float]] = []
    for k in range(1, 6):
        _cycle(publish, k, acceleration=k != 2)
        got += _pairs(fn())
    assert got == _expected(1, 3, 4, 5)


def test_a_lost_message_inside_a_burst_drops_one_cycle_only(
    drain: Tuple[Drain, Publish],
) -> None:
    fn, publish = drain
    _cycle(publish, 1)
    _cycle(publish, 2, acceleration=False)
    _cycle(publish, 3)
    got = _pairs(fn())
    _cycle(publish, 4, steering=False)
    _cycle(publish, 5)
    got += _pairs(fn())
    assert got == _expected(1, 3, 5)


def test_a_silent_topic_does_not_grow_or_shift_the_pairing(
    drain: Tuple[Drain, Publish],
) -> None:
    fn, publish = drain
    for k in range(1, 50):
        _cycle(publish, k, acceleration=False)
        assert fn() == []
    _cycle(publish, 50)
    _cycle(publish, 51)
    assert _pairs(fn()) == _expected(50, 51)
