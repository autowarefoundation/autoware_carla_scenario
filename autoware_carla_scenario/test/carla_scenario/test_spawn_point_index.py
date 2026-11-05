"""A spawn-point index is a whole number from 0 to the map's count - 1 (#1491).

Python's negative indexing would otherwise spawn ``-1`` at the last spawn
point, ``-3`` at the first of three, and ``True`` at index 1.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from autoware_carla_scenario.entity._spawn import (
    SpawnPointIndex,
    spawn_vehicle_actor,
)

_BLUEPRINT = "vehicle.mini.cooper"


class TestTheIndexIsCheckedWhenBuilt:
    @pytest.mark.parametrize("value", [0, 1, 42])
    def test_a_whole_number_not_negative_is_kept(self, value: int) -> None:
        assert SpawnPointIndex(value).value == value

    def test_an_integral_float_is_read_as_an_int(self) -> None:
        """YAML may read ``3.0``; it is the index 3, kept as an int."""
        index = SpawnPointIndex(3.0)  # type: ignore[arg-type]
        assert index.value == 3
        assert type(index.value) is int

    def test_a_numpy_integer_is_read_as_an_int(self) -> None:
        index = SpawnPointIndex(np.int64(2))  # type: ignore[arg-type]
        assert index.value == 2
        assert type(index.value) is int

    @pytest.mark.parametrize("value", [-1, -3, -4, -2.0])
    def test_a_negative_index_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="must not be negative") as raised:
            SpawnPointIndex(value)
        assert repr(value) in str(raised.value)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_refused(self, value: bool) -> None:
        with pytest.raises(ValueError, match="SpawnPointIndex value") as raised:
            SpawnPointIndex(value)
        assert repr(value) in str(raised.value)

    @pytest.mark.parametrize("value", ["3", 3.5, None, float("nan"), float("inf")])
    def test_a_non_integer_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="SpawnPointIndex value"):
            SpawnPointIndex(value)

    def test_equal_indices_compare_equal(self) -> None:
        assert SpawnPointIndex(3.0) == SpawnPointIndex(3)  # type: ignore[arg-type]


def _spawn_points(count: int) -> List[SimpleNamespace]:
    return [
        SimpleNamespace(location=SimpleNamespace(x=float(i), y=0.0, z=0.0))
        for i in range(count)
    ]


class _StubWorld:
    """Just enough of a carla.World for spawn_vehicle_actor: no CARLA."""

    def __init__(self, spawn_points: List[SimpleNamespace]) -> None:
        self._map = SimpleNamespace(get_spawn_points=lambda: list(spawn_points))
        blueprint = SimpleNamespace(id=_BLUEPRINT, set_attribute=lambda *_: None)
        self._library = SimpleNamespace(
            filter=lambda _pattern: [blueprint], find=lambda _id: blueprint
        )
        self.spawned_at: List[Any] = []

    def get_blueprint_library(self) -> SimpleNamespace:
        return self._library

    def get_map(self) -> SimpleNamespace:
        return self._map

    def try_spawn_actor(self, _blueprint: Any, transform: Any) -> Any:
        self.spawned_at.append(transform)
        return SimpleNamespace(transform=transform)


@pytest.fixture(autouse=True)
def _world_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """The world's own map, whatever map a MapManager may hold."""
    monkeypatch.setattr(
        "autoware_carla_scenario.entity._spawn.run_carla_map",
        lambda world: world.get_map(),
    )


class TestTheIndexIsCheckedAtSpawn:
    @pytest.mark.parametrize("value", [0, 1, 2])
    def test_an_index_in_range_spawns_at_that_point(self, value: int) -> None:
        points = _spawn_points(3)
        world = _StubWorld(points)

        spawn_vehicle_actor(
            world,  # type: ignore[arg-type]
            _BLUEPRINT,
            "npc",
            SpawnPointIndex(value),
        )

        assert world.spawned_at == [points[value]]

    def test_an_index_past_the_last_point_is_refused(self) -> None:
        world = _StubWorld(_spawn_points(3))

        with pytest.raises(ValueError, match=r"Spawn index 3 is out of range"):
            spawn_vehicle_actor(
                world,  # type: ignore[arg-type]
                _BLUEPRINT,
                "npc",
                SpawnPointIndex(3),
            )
        assert world.spawned_at == []

    @pytest.mark.parametrize("value", [-1, -3, -4])
    def test_a_negative_index_that_slipped_past_the_constructor_is_refused(
        self, value: int
    ) -> None:
        """The backstop: an index set around ``__post_init__`` does not count
        from the end of the spawn points."""
        index = SpawnPointIndex(0)
        object.__setattr__(index, "value", value)
        world = _StubWorld(_spawn_points(3))

        with pytest.raises(ValueError, match=rf"Spawn index {value} is out of range"):
            spawn_vehicle_actor(
                world,  # type: ignore[arg-type]
                _BLUEPRINT,
                "npc",
                index,
            )
        assert world.spawned_at == []

    def test_a_map_with_no_spawn_points_refuses_index_0(self) -> None:
        world = _StubWorld([])

        with pytest.raises(ValueError, match=r"Spawn index 0 is out of range"):
            spawn_vehicle_actor(
                world,  # type: ignore[arg-type]
                _BLUEPRINT,
                "npc",
                SpawnPointIndex(0),
            )
