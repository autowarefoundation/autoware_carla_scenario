"""A run's pedestrian navigation draws do not depend on its place in a batch (#2448).

:func:`~autoware_carla_scenario.entity._walker_ai.has_navmesh` draws
navigation locations to find out whether the map has a mesh, once per world.
Probed lazily at a run's first ``drive()``, after the run's seed, it took
draws from the first run on a world and not from the ones after it.
:func:`~autoware_carla_scenario.entity._walker_ai.seed_navigation` probes
before seeding, so every run draws the same sequence after its seed.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

import pytest

from autoware_carla_scenario.entity import _walker_ai


class _NavigationWorld:
    """A world whose navigation draws come from one generator, which
    ``set_pedestrians_seed`` reseeds; it counts its draws and seeds."""

    id = 2448

    def __init__(self, misses: int = 0) -> None:
        self._random = random.Random(0)
        self.draws = 0
        self.seeds: list[int] = []
        #: How many draws find nothing before one finds a location.
        self._misses = misses

    def set_pedestrians_seed(self, seed: int) -> None:
        self.seeds.append(seed)
        self._random.seed(seed)

    def get_random_location_from_navigation(self) -> Optional[float]:
        self.draws += 1
        value = self._random.random()
        if self._misses > 0:
            self._misses -= 1
            return None
        return value


@pytest.fixture(autouse=True)
def _forget() -> Iterator[None]:
    _walker_ai._NAVMESH.clear()
    _walker_ai._NAVMESH_SAID.clear()
    yield
    _walker_ai._NAVMESH.clear()
    _walker_ai._NAVMESH_SAID.clear()


def _run(world: _NavigationWorld, seed: int) -> list[Optional[float]]:
    """A run as the runner starts one: the navigation seeded, then the run's
    walkers ask for the mesh (drive()) and draw."""
    _walker_ai.seed_navigation(world, seed)  # type: ignore[arg-type]
    assert _walker_ai.has_navmesh(world)  # type: ignore[arg-type]
    return [world.get_random_location_from_navigation() for _ in range(5)]


def test_consecutive_runs_on_one_world_draw_the_same_after_their_seed() -> None:
    # The probe misses twice before the mesh answers: three draws, all of
    # them in the first run on the world.
    world = _NavigationWorld(misses=2)
    first = _run(world, 7)
    probed = world.draws - 5
    assert probed == 3
    second = _run(world, 7)
    assert world.draws == probed + 10  # the second run probed nothing
    assert world.seeds == [7, 7]
    assert first == second
    # And the same as a run alone on a fresh world.
    assert _run(_NavigationWorld(), 7) == first


def test_the_probe_draws_before_the_seed() -> None:
    world = _NavigationWorld()
    _walker_ai.seed_navigation(world, 11)  # type: ignore[arg-type]
    # Probed (one draw), then seeded: the next draw is the seed's first.
    assert world.draws == 1 and world.seeds == [11]
    assert world.get_random_location_from_navigation() == random.Random(11).random()


def test_a_reloaded_world_is_probed_again_before_its_seed() -> None:
    world = _NavigationWorld()
    _walker_ai.seed_navigation(world, 3)  # type: ignore[arg-type]
    # A reload: a new episode, a new world id.
    reloaded = _NavigationWorld()
    reloaded.id = 2449  # type: ignore[misc]
    _walker_ai.seed_navigation(reloaded, 3)  # type: ignore[arg-type]
    assert reloaded.draws == 1 and reloaded.seeds == [3]
    assert reloaded.get_random_location_from_navigation() == random.Random(3).random()


def test_a_missing_mesh_is_said_only_when_a_walker_needs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every probe misses: the map has no mesh.
    world = _NavigationWorld(misses=_walker_ai._NAVMESH_PROBES)
    with caplog.at_level(logging.WARNING):
        _walker_ai.seed_navigation(world, 5)  # type: ignore[arg-type]
        _walker_ai.seed_navigation(world, 5)  # type: ignore[arg-type]
        assert world.draws == _walker_ai._NAVMESH_PROBES  # probed once
        assert not any("navigation mesh" in r.getMessage() for r in caplog.records)
        # A walker asks (drive()): said once.
        assert not _walker_ai.has_navmesh(world)  # type: ignore[arg-type]
        assert not _walker_ai.has_navmesh(world)  # type: ignore[arg-type]
    assert sum("navigation mesh" in r.getMessage() for r in caplog.records) == 1
