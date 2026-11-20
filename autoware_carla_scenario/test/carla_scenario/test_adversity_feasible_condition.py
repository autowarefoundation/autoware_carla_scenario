"""AdversityFeasibleCondition on a CARLA world (#392).

The synthetic crossroads of :mod:`._route_maps` stands in the MapManager with
no offset, so CARLA's frame is the map's with y flipped; the actors are fakes.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.adversity import (
    AdversarialCommand,
    AdversityKind,
    AdversityLocation,
    snapshot,
)
from autoware_carla_scenario.conditions.adversity_feasible import (
    AdversityFeasibleCondition,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.adversity.nade import (
    stream_seed,
    trigger_key,
    trigger_stream_seed,
)
from autoware_carla_scenario.seed import MAX_SEED, run_seed

from . import _route_maps as rm

INNER = -1.75


def _vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _actor(
    actor_id: int,
    role: str,
    x: float,
    y: float,
    speed: float,
    type_id: str = "vehicle.audi.tt",
    *,
    box_x: float = 0.0,
    box_y: float = 0.0,
) -> Any:
    # A map-frame (x, y) facing East, as CARLA has it: y flipped.  The box's
    # centre is (box_x, box_y) off the origin, in CARLA's frame (y right).
    transform = SimpleNamespace(
        location=_vector(x, -y, 0.0),
        rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0),
        get_forward_vector=lambda: _vector(1.0),
    )
    return SimpleNamespace(
        id=actor_id,
        type_id=type_id,
        attributes={"role_name": role},
        bounding_box=SimpleNamespace(
            extent=_vector(2.5, 1.0, 0.8), location=_vector(box_x, box_y)
        ),
        get_transform=lambda: transform,
        get_velocity=lambda: _vector(speed),
        get_acceleration=lambda: _vector(),
        get_speed_limit=lambda: 50.0,
    )


class _Actors(list):
    def filter(self, pattern: str) -> "_Actors":
        return _Actors(a for a in self if a.type_id.startswith(pattern.rstrip("*")))


def _world(*actors: Any, frame: int = 1) -> Any:
    return SimpleNamespace(
        get_actors=lambda: _Actors(actors),
        get_snapshot=lambda: SimpleNamespace(frame=frame),
    )


@pytest.fixture(autouse=True)
def crossroads(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    lanelet_map = rm.crossroads()
    manager._lanelet_map = lanelet_map
    manager._routing_graph = rm.routing_graph(lanelet_map)
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    monkeypatch.setattr(
        snapshot, "find_traffic_lights_for_lanelet2_id", lambda world, i: []
    )
    yield
    MapManager._instance = saved


def test_it_holds_when_the_trigger_fires_and_keeps_the_command() -> None:
    condition = AdversityFeasibleCondition(
        "npc1", AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, label="brake"
    )
    world = _world(
        _actor(1, "npc1", -40.0, INNER, 10.0), _actor(2, "npc2", -55.0, INNER, 10.0)
    )
    result = condition.check(world, 3.0)
    assert result is not None and result.passed
    assert "may brake check" in result.message
    assert condition.command == AdversarialCommand.accelerate(-9.0, 2.0)
    assert condition.scene is not None and set(condition.scene.agents) == {
        "npc1",
        "npc2",
    }


def test_it_holds_only_at_its_location() -> None:
    condition = AdversityFeasibleCondition(
        "npc1", AdversityKind.BRAKE_CHECK, AdversityLocation.INTERSECTION, label="brake"
    )
    far = _world(
        _actor(1, "npc1", -40.0, INNER, 10.0), _actor(2, "npc2", -55.0, INNER, 10.0)
    )
    assert condition.check(far, 0.0) is None and condition.command is None
    near = _world(
        _actor(1, "npc1", -20.0, INNER, 10.0), _actor(2, "npc2", -35.0, INNER, 10.0)
    )
    assert condition.check(near, 0.1) is not None


def test_it_does_not_hold_without_its_entity_or_its_trigger() -> None:
    condition = AdversityFeasibleCondition(
        "npc1", AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, label="brake"
    )
    assert condition.check(_world(_actor(2, "npc2", -55.0, INNER, 10.0)), 0.0) is None
    assert condition.check(_world(_actor(1, "npc1", -40.0, INNER, 10.0)), 0.0) is None


def test_a_jaywalker_is_seeded() -> None:
    def run(seed: int) -> Any:
        condition = AdversityFeasibleCondition(
            "p1",
            AdversityKind.JAYWALKING,
            AdversityLocation.ANY,
            label="jay",
            seed=seed,
        )
        walker = _actor(7, "p1", -40.0, -12.0, 0.0, type_id="walker.pedestrian.0001")
        assert condition.check(_world(walker), 0.0) is not None
        assert condition.command is not None
        return condition.command.path[-1]

    three = run(3)
    assert three == run(3) != run(4)


@pytest.mark.parametrize("value", [2.5, math.nan, math.inf, True, -1, "2"])
def test_a_seed_must_be_a_whole_number_not_negative(value: Any) -> None:
    # #1495: int() read 2.5 as 2 and True as 1, and a negative seed passed.
    with pytest.raises(ValueError, match="seed"):
        AdversityFeasibleCondition(
            "p1", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="j", seed=value
        )


def test_a_whole_float_seed_is_read_as_an_int_and_none_is_the_runs() -> None:
    two: Any = 2.0
    condition = AdversityFeasibleCondition(
        "p1", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="j", seed=two
    )
    key = trigger_key("p1\x1fjaywalking\x1fj")
    assert condition._seed == trigger_stream_seed(2, key)  # noqa: SLF001
    assert type(condition._seed) is int  # noqa: SLF001
    default = AdversityFeasibleCondition(
        "p1", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="j"
    )
    assert default._seed == trigger_stream_seed(run_seed(), key)  # noqa: SLF001


def _first_draws(condition: AdversityFeasibleCondition) -> list[float]:
    return [condition._rng.random() for _ in range(4)]  # noqa: SLF001


def test_two_jaywalkers_following_the_runs_seed_draw_apart() -> None:
    # #2226: both were seeded random.Random(run seed), so their speeds and
    # waits were the same numbers.
    def jaywalker(name: str, label: str = "jay") -> AdversityFeasibleCondition:
        return AdversityFeasibleCondition(
            name, AdversityKind.JAYWALKING, AdversityLocation.ANY, label=label
        )

    first, second = jaywalker("p1"), jaywalker("p2")
    assert first._seed != second._seed  # noqa: SLF001
    assert _first_draws(first) != _first_draws(second)
    # Two actions on one pedestrian, told apart by their labels.
    assert _first_draws(jaywalker("p1", "a")) != _first_draws(jaywalker("p1", "b"))
    # The same condition again draws the same numbers.
    assert _first_draws(jaywalker("p1")) == _first_draws(jaywalker("p1"))


def test_a_conditions_stream_is_none_of_nades_nor_the_raw_seeds() -> None:
    # #2226: at seed s the condition drew from Random(s), which is NADE's
    # sampling stream at seed s / 3.
    keys = [trigger_key(text) for text in ("a", "b", "p1\x1fjaywalking\x1fjay")]
    conditions = {trigger_stream_seed(s, k) for s in range(200) for k in keys}
    assert len(conditions) == 600
    nade = {stream_seed(s, i) for s in range(200) for i in range(3)}
    assert not conditions & nade
    assert min(conditions) > stream_seed(MAX_SEED, 2) > MAX_SEED
    assert trigger_stream_seed(MAX_SEED, (1 << 30) - 1) < 2**62
    for seed in range(10):
        condition = AdversityFeasibleCondition(
            "p1", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="j", seed=seed
        )
        nade_draws = [random.Random(stream_seed(seed, 0)).random() for _ in range(1)]
        assert _first_draws(condition)[:1] != nade_draws
    with pytest.raises(ValueError, match="key"):
        trigger_stream_seed(0, 1 << 30)
    with pytest.raises(ValueError, match="seed"):
        trigger_stream_seed(MAX_SEED + 1, 0)


def test_a_trigger_key_is_the_same_in_every_process() -> None:
    # Not hash(), which CPython salts per process.
    code = (
        "from autoware_carla_scenario.adversity.nade import trigger_key;"
        "print(trigger_key('p1\\x1fjaywalking\\x1fjay'))"
    )
    keys = {
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(salt)},
        ).stdout.strip()
        for salt in (1, 2)
    }
    assert keys == {str(trigger_key("p1\x1fjaywalking\x1fjay"))}
    # Pinned, so a change of the scheme is a deliberate one.
    # The FNV-1a offset basis, folded to 30 bits.
    assert trigger_key("") == (2166136261 ^ (2166136261 >> 30)) & ((1 << 30) - 1)
    assert 0 <= trigger_key("\u6b69\u884c\u8005") < 1 << 30


def test_one_scene_a_frame_for_everyone_who_asks() -> None:
    world = _world(
        _actor(1, "npc1", -40.0, INNER, 10.0), _actor(2, "npc2", -55.0, INNER, 10.0)
    )
    first = snapshot.scene_at(world)
    assert snapshot.scene_at(world) is first
    world.get_snapshot = lambda: SimpleNamespace(frame=2)
    assert snapshot.scene_at(world) is not first
    brake = AdversityFeasibleCondition(
        "npc1", AdversityKind.BRAKE_CHECK, AdversityLocation.ANY, label="a"
    )
    follow = AdversityFeasibleCondition(
        "npc2", AdversityKind.IGNORE_LEADER, AdversityLocation.ANY, label="b"
    )
    brake.check(world, 0.0)
    follow.check(world, 0.0)
    assert brake.scene is snapshot.scene_at(world)


def test_its_details() -> None:
    condition = AdversityFeasibleCondition(
        "npc1", AdversityKind.CUT_IN, AdversityLocation.HIGHWAY, label="cut"
    )
    assert condition.get_details() == {
        "entity_name": "npc1",
        "kind": "cut_in",
        "location": "highway",
    }
    assert condition.kind is AdversityKind.CUT_IN
    assert condition.location is AdversityLocation.HIGHWAY
