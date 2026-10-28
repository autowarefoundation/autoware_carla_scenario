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

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

cyclone = pytest.importorskip("autoware_carla_scenario.ros2.cyclone")

from autoware_carla_scenario.ros2.commands import Command, SteerAccelCommand  # noqa: E402
from autoware_carla_scenario.ros2.config import CommandSourceConfig  # noqa: E402

_MS = 1_000_000
#: The publisher's cycle, as VisionPilot's 10 Hz.
_CYCLE_NS = 100 * _MS


class _FakeReader:
    def __init__(self) -> None:
        self.queue: List[Any] = []


class _Rig:
    """A ``steer_accel`` source's drain over two fake readers.

    Args:
        gap_ns: How long after a cycle's steering :meth:`cycle` writes its
            acceleration.
        pair_window_s: The source's ``pair_window_s``.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gap_ns: int,
        pair_window_s: Optional[float] = None,
    ) -> None:
        self.gap_ns = gap_ns
        readers: Dict[str, _FakeReader] = {}

        def fake_reader(
            participant: Any, name: str, message_type: Any, depth: int
        ) -> Any:
            readers[name] = _FakeReader()
            return readers[name]

        def fake_take(reader: _FakeReader, depth: int) -> List[Any]:
            taken, reader.queue = reader.queue[:depth], reader.queue[depth:]
            return taken

        monkeypatch.setattr(cyclone, "_reader", fake_reader)
        monkeypatch.setattr(cyclone, "_take", fake_take)
        source = CommandSourceConfig(
            name="vp",
            kind="steer_accel",
            steering_topic="/s",
            acceleration_topic="/a",
            pair_window_s=pair_window_s,
        )
        self._drain = cyclone._source_drain(None, source)
        self._readers = readers

    def publish(self, topic: str, value: float, stamp_ns: int) -> None:
        self._readers[topic].queue.append(
            SimpleNamespace(
                data=value, sample_info=SimpleNamespace(source_timestamp=stamp_ns)
            )
        )

    def cycle(
        self, k: int, *, steering: bool = True, acceleration: bool = True
    ) -> None:
        """Publish cycle *k*: steering ``k / 10``, then acceleration ``k``."""
        if steering:
            self.publish("/s", k / 10, k * _CYCLE_NS)
        if acceleration:
            self.publish("/a", float(k), k * _CYCLE_NS + self.gap_ns)

    def drain(self) -> List[Tuple[float, float]]:
        return _pairs(self._drain())


def _pairs(commands: Sequence[Command]) -> List[Tuple[float, float]]:
    out = []
    for command in commands:
        assert isinstance(command, SteerAccelCommand)
        out.append((command.steering_rad, command.acceleration_mps2))
    return out


def _expected(*cycles: int) -> List[Tuple[float, float]]:
    return [(k / 10, float(k)) for k in cycles]


@pytest.fixture(params=[1 * _MS, 40 * _MS], ids=["back_to_back", "40ms_apart"])
def rig(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    """The default (measured) window, with the two halves of a cycle written
    back to back, as VisionPilot does, or 40 ms apart, as two timers might:
    both must pair every cycle with no ``pair_window_s`` set."""
    return _Rig(monkeypatch, int(request.param))


def test_two_cycles_in_one_poll(rig: _Rig) -> None:
    rig.cycle(1)
    rig.cycle(2)
    got = rig.drain()
    rig.cycle(3)
    got += rig.drain()
    assert got == _expected(1, 2, 3)


def test_a_cycle_split_across_polls(rig: _Rig) -> None:
    rig.cycle(1, acceleration=False)  # steering before the tick...
    got = rig.drain()
    assert got == []
    rig.cycle(1, steering=False)  # ...its acceleration after it
    rig.cycle(2)
    got += rig.drain()
    rig.cycle(3)
    got += rig.drain()
    assert got == _expected(1, 2, 3)


def test_the_acceleration_written_first_still_pairs(rig: _Rig) -> None:
    for k in (1, 2, 3):
        rig.publish("/a", float(k), k * _CYCLE_NS)
        rig.publish("/s", k / 10, k * _CYCLE_NS + rig.gap_ns)
    assert rig.drain() == _expected(1, 2, 3)


def test_equal_stamps_pair_across_topics(rig: _Rig) -> None:
    for k in (1, 2, 3):
        rig.publish("/a", float(k), k * _CYCLE_NS)
        rig.publish("/s", k / 10, k * _CYCLE_NS)
    assert rig.drain() == _expected(1, 2, 3)


def test_a_lost_steering_message_drops_one_cycle_only(rig: _Rig) -> None:
    got: List[Tuple[float, float]] = []
    for k in range(1, 8):
        rig.cycle(k, steering=k != 4)
        got += rig.drain()
    assert got == _expected(1, 2, 3, 5, 6, 7)


def test_a_lost_acceleration_message_drops_one_cycle_only(rig: _Rig) -> None:
    got: List[Tuple[float, float]] = []
    for k in range(1, 8):
        rig.cycle(k, acceleration=k != 4)
        got += rig.drain()
    assert got == _expected(1, 2, 3, 5, 6, 7)


def test_a_lost_message_inside_a_burst_drops_one_cycle_only(rig: _Rig) -> None:
    rig.cycle(1)
    rig.cycle(2)
    got = rig.drain()
    rig.cycle(3, acceleration=False)
    rig.cycle(4)
    got += rig.drain()
    rig.cycle(5, steering=False)
    rig.cycle(6)
    got += rig.drain()
    assert got == _expected(1, 2, 4, 6)


def test_a_silent_topic_does_not_grow_or_shift_the_pairing(rig: _Rig) -> None:
    for k in range(1, 50):
        rig.cycle(k, acceleration=False)
        assert rig.drain() == []
    rig.cycle(50)
    rig.cycle(51)
    assert rig.drain() == _expected(50, 51)


def test_a_rate_change_is_followed(rig: _Rig) -> None:
    # 10 Hz, then 20 Hz with the same gap ratio: the measured window follows.
    got: List[Tuple[float, float]] = []
    for k in range(1, 5):
        rig.cycle(k)
        got += rig.drain()
    for k in range(5, 30):
        stamp = 4 * _CYCLE_NS + (k - 4) * _CYCLE_NS // 2
        rig.publish("/s", k / 10, stamp)
        rig.publish("/a", float(k), stamp + rig.gap_ns // 2)
        got += rig.drain()
    assert got == _expected(*range(1, 30))


def test_a_fixed_window_too_small_for_the_publisher_drops_every_cycle_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The halves are written 40 ms apart but the window says 20 ms: nothing
    # pairs, and the log says so instead of the ego going silently commandless.
    clock = [1000.0]
    monkeypatch.setattr(cyclone.time, "monotonic", lambda: clock[0])
    rig = _Rig(monkeypatch, 40 * _MS, pair_window_s=0.02)
    with caplog.at_level(logging.WARNING, logger=cyclone.__name__):
        for k in range(1, 6):
            rig.cycle(k)
            assert rig.drain() == []
            clock[0] += 0.1
        # Rate-limited: one warning for the first drop, none for the next ones
        # until the interval has passed.
        warnings = [r for r in caplog.records if "pair_window_s" in r.getMessage()]
        assert len(warnings) == 1
        assert "'vp'" in warnings[0].getMessage()
        assert "20.0 ms" in warnings[0].getMessage()
        clock[0] += cyclone._DROP_REPORT_INTERVAL_S
        rig.cycle(6)
        rig.drain()
        warnings = [r for r in caplog.records if "pair_window_s" in r.getMessage()]
        assert len(warnings) == 2
        # Every drop since the first report, counted.
        assert "dropped 10 " in warnings[1].getMessage()


def test_a_fixed_window_that_fits_pairs_every_cycle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rig = _Rig(monkeypatch, 40 * _MS, pair_window_s=0.05)
    got: List[Tuple[float, float]] = []
    with caplog.at_level(logging.WARNING, logger=cyclone.__name__):
        for k in range(1, 6):
            rig.cycle(k)
            got += rig.drain()
    assert got == _expected(1, 2, 3, 4, 5)
    assert not caplog.records
