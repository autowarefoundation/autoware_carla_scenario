"""The clearance checks are on one level of road: a road user on a bridge
over the spot, or in an underpass beneath it, is not in the way (#1588),
while one further along a climbing road still is."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

from autoware_carla_scenario.utils import clearance

_EPISODES = itertools.count(158800)


class _World:
    """A world of road users ``id: (type id, x, y, z)``, each facing +x, its
    box 2.4 m by 1.0 m by 0.8 m half extents centred 0.8 m above its origin
    (a car standing on the road at its origin's height)."""

    def __init__(self, actors: dict[int, tuple[str, float, float, float]]) -> None:
        self.actors = actors
        self.id = next(_EPISODES)  # a fresh episode: nothing cached
        self.frame = 1

    def get_snapshot(self) -> Any:
        def seen(actor_id: int, x: float, y: float, z: float) -> Any:
            return SimpleNamespace(
                id=actor_id,
                get_transform=lambda: SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y, z=z),
                    rotation=SimpleNamespace(yaw=0.0),
                ),
                get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )

        class _Snapshot(list):  # type: ignore[type-arg]
            frame = self.frame

        return _Snapshot(seen(i, x, y, z) for i, (_, x, y, z) in self.actors.items())

    def get_actor(self, actor_id: int) -> Any:
        if actor_id not in self.actors:
            return None
        return SimpleNamespace(
            id=actor_id,
            type_id=self.actors[actor_id][0],
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=2.4, y=1.0, z=0.8),
                location=SimpleNamespace(x=0.0, y=0.0, z=0.8),
            ),
        )


def _car_at(z: float, x: float = 5.0) -> _World:
    return _World({1: ("vehicle.parked", x, 0.0, z)})


def test_a_car_on_a_bridge_over_the_spot_does_not_take_it() -> None:
    """The issue's repro: a car 8 m up, right over the spot."""
    world = _car_at(8.0)
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=0.0) is None
    # Put on the bridge, it is taken.
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=8.0) == 1
    # With no height given, every level is one, as before.
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0) == 1


def test_a_car_under_the_spot_does_not_take_it() -> None:
    world = _car_at(0.0)
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=6.0) is None
    # A box a little off the road it stands on is on the same level.
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=1.5) == 1


def test_one_put_beside_itself_on_its_own_level_counts_its_own_height() -> None:
    """The one being put stands on the road at *z* with its own box's
    height: a car's bottom 3.9 m above its roof is on another level, 0.5 m
    above it on the same."""
    world = _World(
        {
            1: ("vehicle.parked", 5.0, 0.0, 5.5),
            2: ("vehicle.truck", -50.0, 0.0, -0.5),  # the one being put
        }
    )
    # Car 2's box is 1.6 m tall: its top at 1.6 m, 3.9 m under the car.
    assert clearance.overlaps(world, 2, 5.0, 0.0, 0.0, 2.4, 1.0, z=0.0) is None
    # Its top 1.6 m over the road at 4.4 m: 0.5 m under the car's bottom.
    assert clearance.overlaps(world, 2, 5.0, 0.0, 0.0, 2.4, 1.0, z=3.4) == 1


def test_a_spot_claimed_on_another_level_does_not_take_it() -> None:
    world = _World({2: ("vehicle.parked", -50.0, 0.0, 0.0)})
    clearance.claim_spot(world, 2, 5.0, 0.0, 0.0, 2.4, 1.0, z=8.0)
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=0.0) is None
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=8.0) == 2
    # A claim with no height is on every level.
    clearance.claim_spot(world, 3, 5.0, 0.0, 0.0, 2.4, 1.0)
    assert clearance.overlaps(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=0.0) == 3


def test_a_car_on_a_bridge_ahead_is_not_in_the_way() -> None:
    """The issue's repro: looking 10 m ahead under a car 8 m up."""
    world = _car_at(8.0)
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        assert look(world, 99, 0.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=0.0) is None
        found = look(world, 99, 0.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=8.0)
        assert found is not None and found.id == 1
        # With no height given, every level is one, as before.
        found = look(world, 99, 0.0, 0.0, 0.0, "vehicle", 10.0, 1.0)
        assert found is not None and found.id == 1


def test_a_car_far_up_a_climbing_road_ahead_is_in_the_way() -> None:
    """40 m on and 3 m up is a 7.5 % climb, a road's: the car is on the
    mover's road.  3 m up 5 m on is a bridge."""
    far = _car_at(3.0, x=40.0)
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        found = look(far, 99, 0.0, 0.0, 0.0, "vehicle", 50.0, 1.0, z=0.0)
        assert found is not None and found.id == 1
    near = _car_at(3.0, x=5.0)
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        assert look(near, 99, 0.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=0.0) is None


def test_a_piece_of_path_further_on_allows_for_the_climb_to_it() -> None:
    """Looking along a path in pieces, the height read where the mover
    stands: a car 5 m into a piece 30 m on, 3 m up, is on its road."""
    world = _car_at(3.0, x=35.0)
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        assert look(world, 99, 30.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=0.0) is None
        found = look(world, 99, 30.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=0.0, along=30.0)
        assert found is not None and found.id == 1


def test_the_mover_s_own_box_is_its_height() -> None:
    """The mover's own box, read from the world, is what is compared."""
    world = _World(
        {
            1: ("vehicle.parked", 5.0, 0.0, 2.5),
            7: ("walker.pedestrian", 0.0, 0.0, 0.0),
        }
    )
    # Its box (centred 0.8 m above its origin here) spans 0.0 to 1.6 m: the
    # car's bottom at 2.5 m is 0.9 m above it, on its level.
    found = clearance.first_in_way(world, 7, 0.0, 0.0, 0.0, "vehicle", 10.0, 1.0, z=0.0)
    assert found is not None and found.id == 1


def test_a_place_clear_condition_is_clear_under_a_bridge() -> None:
    from autoware_carla_scenario.conditions import place_clear as place_module

    world = _car_at(8.0)
    assert place_module._occupant(world, 5.0, 0.0, 0.0, 2.0, 1.0, 0.3, 0.0) == -1  # type: ignore[arg-type]
    assert place_module._occupant(world, 5.0, 0.0, 0.0, 2.0, 1.0, 0.3, 8.0) == 1  # type: ignore[arg-type]
    assert place_module._occupant(world, 5.0, 0.0, 0.0, 2.0, 1.0, 0.3, None) == 1  # type: ignore[arg-type]


def test_a_spawn_is_not_taken_by_a_car_on_a_bridge_over_it() -> None:
    from autoware_carla_scenario.actions import spawn as spawn_module

    world = _car_at(8.0)
    assert spawn_module._taken(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=0.0) == -1  # type: ignore[arg-type]
    assert spawn_module._taken(world, 99, 5.0, 0.0, 0.0, 2.4, 1.0, z=8.0) == 1  # type: ignore[arg-type]


def test_levels_are_told_apart_by_where_each_stands_not_by_the_gap() -> None:
    """20 m on and 3 m up is steeper than a road (2 m at 10 %): another
    level, though the gap over the mover's 1.6 m tall box is only 1.4 m."""
    world = _World(
        {
            1: ("vehicle.parked", 20.0, 0.0, 3.0),
            7: ("vehicle.mover", 0.0, 0.0, 0.0),
        }
    )
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        assert look(world, 7, 0.0, 0.0, 0.0, "vehicle", 25.0, 1.0, z=0.0) is None
    # 30 m on, 3 m up is a 10 % climb: the mover's road.
    world.actors[1] = ("vehicle.parked", 30.0, 0.0, 3.0)
    world.frame = 2
    for look in (clearance.first_box_in_way, clearance.first_in_way):
        found = look(world, 7, 0.0, 0.0, 0.0, "vehicle", 35.0, 1.0, z=0.0)
        assert found is not None and found.id == 1
