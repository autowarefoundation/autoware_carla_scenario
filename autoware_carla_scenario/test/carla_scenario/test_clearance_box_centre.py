"""The clearance checks take each road user's outline where its bounding box
is, not on its origin: CARLA puts a car's box ahead of its origin (#884)."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.utils import clearance

_EPISODES = itertools.count(88400)


class _World:
    """A world of actors ``id: (type id, x, y, yaw, extent, offset)``: the
    origin's place and heading, the box's half length and width, and where
    the box sits from the origin (forward, right)."""

    def __init__(
        self,
        actors: dict[
            int,
            tuple[str, float, float, float, tuple[float, float], tuple[float, float]],
        ],
    ) -> None:
        self.actors = actors
        self.id = next(_EPISODES)  # a fresh episode: nothing cached

    def get_snapshot(self) -> Any:
        def seen(actor_id: int, x: float, y: float, yaw: float) -> Any:
            return SimpleNamespace(
                id=actor_id,
                get_transform=lambda: SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y, z=0.0),
                    rotation=SimpleNamespace(yaw=yaw),
                ),
                get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
            )

        class _Snapshot(list):  # type: ignore[type-arg]
            frame = 1

        return _Snapshot(
            seen(i, x, y, yaw) for i, (_, x, y, yaw, _e, _o) in self.actors.items()
        )

    def get_actor(self, actor_id: int) -> Any:
        if actor_id not in self.actors:
            return None
        type_id, _x, _y, _yaw, extent, offset = self.actors[actor_id]
        return SimpleNamespace(
            id=actor_id,
            type_id=type_id,
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=extent[0], y=extent[1], z=0.8),
                location=SimpleNamespace(x=offset[0], y=offset[1], z=0.8),
            ),
        )


def _parked_car(yaw: float = 0.0, offset: float = 1.3) -> _World:
    """A car with its origin at (0, 0), its 4.8 m box *offset* m ahead of
    the origin: along x, its body spans [-1.1, 3.7] facing +x."""
    return _World({1: ("vehicle.parked", 0.0, 0.0, yaw, (2.4, 1.0), (offset, 0.0))})


def test_a_walker_by_a_car_s_front_overlaps_it() -> None:
    """The issue's repro: a walker 0.4 m inside the car's front."""
    world = _parked_car()
    assert clearance.overlaps(world, 99, 3.3, 0.0, 0.0, 0.3, 0.3, 0.0) == 1
    # Behind its rear bumper (-1.1), though within 2.4 m of its origin.
    assert clearance.overlaps(world, 99, -1.8, 0.0, 0.0, 0.3, 0.3, 0.0) is None


def test_a_car_s_front_is_in_the_way_ahead() -> None:
    """The issue's repro: looking 1.0 m ahead from x=2.6, inside its body."""
    world = _parked_car()
    found = clearance.first_box_in_way(world, 99, 2.6, 0.0, 0.0, "vehicle", 1.0, 0.3)
    assert found is not None and found.id == 1
    # Reported where its box is, not its origin.
    assert (found.x, found.y) == pytest.approx((1.3, 0.0))
    # The 1.1 m behind its rear (-1.1) to -0.2 is clear.
    assert (
        clearance.first_box_in_way(world, 99, -2.5, 0.0, 0.0, "vehicle", 1.2, 0.3)
        is None
    )


def test_a_turned_car_s_box_is_turned_with_it() -> None:
    """Facing +y (CARLA's right of +x), its box sits 1.3 m along +y."""
    world = _parked_car(yaw=90.0)
    assert clearance.overlaps(world, 99, 0.0, 3.3, 0.0, 0.3, 0.3, 0.0) == 1
    assert clearance.overlaps(world, 99, 0.0, -1.8, 0.0, 0.3, 0.3, 0.0) is None
    found = clearance.first_box_in_way(world, 99, 0.0, 2.6, 90.0, "vehicle", 1.0, 0.3)
    assert found is not None and found.id == 1
    assert (found.x, found.y) == pytest.approx((0.0, 1.3))
    assert clearance.box_centre(world, 1, 10.0, 0.0, 90.0) == pytest.approx((10.0, 1.3))


def test_a_sideways_offset_is_to_the_right_in_carla_s_frame() -> None:
    world = _World({1: ("vehicle.odd", 0.0, 0.0, 0.0, (2.0, 1.0), (0.0, 2.0))})
    assert clearance.overlaps(world, 99, 0.0, 2.9, 0.0, 0.3, 0.3, 0.0) == 1
    assert clearance.overlaps(world, 99, 0.0, -1.2, 0.0, 0.3, 0.3, 0.0) is None
    yawed = math.radians(30.0)
    assert clearance.box_centre(world, 1, 0.0, 0.0, 30.0) == pytest.approx(
        (-2.0 * math.sin(yawed), 2.0 * math.cos(yawed))
    )


def test_an_actor_with_its_box_on_its_origin_is_as_before() -> None:
    world = _parked_car(offset=0.0)
    assert clearance.overlaps(world, 99, 2.5, 0.0, 0.0, 0.3, 0.3, 0.0) == 1
    assert clearance.overlaps(world, 99, 3.3, 0.0, 0.0, 0.3, 0.3, 0.0) is None
    found = clearance.first_box_in_way(world, 99, 2.6, 0.0, 0.0, "vehicle", 1.0, 0.3)
    assert found is None
    found = clearance.first_box_in_way(world, 99, 1.6, 0.0, 0.0, "vehicle", 1.0, 0.3)
    assert found is not None and (found.x, found.y) == pytest.approx((0.0, 0.0))
    assert clearance.box_centre(world, 1, 5.0, 6.0, 45.0) == (5.0, 6.0)


def test_a_box_with_no_offset_to_read_is_on_its_origin() -> None:
    """A test double's box with no location: on its origin, as before."""
    actor = SimpleNamespace(bounding_box=SimpleNamespace(extent=None))
    transform = SimpleNamespace(
        location=SimpleNamespace(x=4.0, y=5.0), rotation=SimpleNamespace(yaw=30.0)
    )
    assert clearance.actor_box_centre(actor, transform) == (4.0, 5.0)  # type: ignore[arg-type]


def test_no_actor_is_centred_on_its_origin() -> None:
    """A prop not spawned yet (-1) has no box to read."""
    assert clearance.box_centre(_parked_car(), -1, 3.0, 4.0, 90.0) == (3.0, 4.0)


def test_a_spawn_checks_its_own_box_where_it_will_stand() -> None:
    """A car put facing +x with its origin at x: its body spans
    [x - 1.1, x + 3.7] (its box 1.3 m ahead of its origin), checked against
    a walker standing at 0 -- its own box on its origin."""
    from autoware_carla_scenario.actions import spawn as spawn_module

    world = _World(
        {
            1: ("walker.pedestrian.0001", 0.0, 0.0, 0.0, (0.3, 0.3), (0.0, 0.0)),
            # Out of the way until it is put.
            2: ("vehicle.entering", -50.0, 0.0, 0.0, (2.4, 1.0), (1.3, 0.0)),
        }
    )
    # Its front at -0.3 (and 0.3 m margin): on the walker, at -0.3 to 0.3.
    assert spawn_module._taken(world, 2, -4.0, 0.0, 0.0, 2.4, 1.0) == 1  # type: ignore[arg-type]
    # Its front at -0.8: clear.
    assert spawn_module._taken(world, 2, -4.5, 0.0, 0.0, 2.4, 1.0) == -1  # type: ignore[arg-type]
    # Its rear at 1.4, though its origin is within 2.4 m of the walker.
    assert spawn_module._taken(world, 2, 2.5, 0.0, 0.0, 2.4, 1.0) == -1  # type: ignore[arg-type]
    assert spawn_module._taken(world, 2, 1.3, 0.0, 0.0, 2.4, 1.0) == 1  # type: ignore[arg-type]


def test_a_spot_held_is_held_where_the_box_will_stand() -> None:
    from autoware_carla_scenario.actions import spawn as spawn_module

    world = _World({2: ("vehicle.entering", -50.0, 0.0, 0.0, (2.4, 1.0), (1.3, 0.0))})
    clearance._CACHE.claims = []
    clearance._CACHE.claims_frame = -1
    try:
        spawn_module._hold(world, 2, 10.0, 0.0, 0.0, 2.4, 1.0)  # type: ignore[arg-type]
        # Its body from 8.9 to 13.7: a walker at 13.3 stands in it.
        assert clearance.overlaps(world, 99, 13.3, 0.0, 0.0, 0.3, 0.3, 0.0) == 2
        assert clearance.overlaps(world, 99, 8.3, 0.0, 0.0, 0.3, 0.3, 0.0) is None
    finally:
        clearance._CACHE.claims = []
        clearance._CACHE.claims_frame = -1


def test_a_place_clear_condition_sees_a_car_s_front_on_its_place() -> None:
    """The place 3 m ahead of a parked car's origin, 1 m long: on its
    bonnet, not clear of it."""
    from autoware_carla_scenario.conditions import place_clear as place_module

    world = _parked_car()
    assert place_module._occupant(world, 3.3, 0.0, 0.0, 0.5, 0.5, 0.0) == 1  # type: ignore[arg-type]
    assert place_module._occupant(world, -2.0, 0.0, 0.0, 0.5, 0.5, 0.0) == -1  # type: ignore[arg-type]


def test_first_in_way_finds_a_car_by_its_box_and_reports_its_centre() -> None:
    """Each actor is a circle (radius 2.6) round its box's centre (1.3), so
    from -1.3 to 3.9 along x, not round its origin (-2.6 to 2.6)."""
    world = _parked_car()
    # Looking back (-x) 1 m from 4.5: its circle's edge at 3.9 is 0.6 m on.
    found = clearance.first_in_way(world, 99, 4.5, 0.0, 180.0, "vehicle", 1.0, 0.3)
    assert found is not None and found.id == 1
    assert (found.x, found.y) == pytest.approx((1.3, 0.0))
    assert found.ahead == pytest.approx(3.2)
    # Looking on (+x) 2.5 m from -4.5: its circle's edge at -1.3 is 3.2 m on.
    assert (
        clearance.first_in_way(world, 99, -4.5, 0.0, 0.0, "vehicle", 2.5, 0.3) is None
    )


class _Driven:
    """A vehicle the framework drives, origin at (0, 0) facing +x, its box
    1.3 m ahead of its origin (4.8 m long: its front at 3.7)."""

    def __init__(self, world: _World) -> None:
        self.id = 2
        self._world = world
        self.bounding_box = SimpleNamespace(
            extent=SimpleNamespace(x=2.4, y=1.0, z=0.8),
            location=SimpleNamespace(x=1.3, y=0.0, z=0.8),
        )

    def get_world(self) -> _World:
        return self._world

    def get_transform(self) -> Any:
        return SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )


def test_a_driven_vehicle_looks_for_a_walker_from_its_box_s_centre() -> None:
    """Standing, it keeps 4 m from its box's centre (1.3) clear of walkers:
    a walker at 5.0 is within that, though not within 4 m of its origin."""
    from autoware_carla_scenario.actions import _trajectory_run as run

    walker = ("walker.pedestrian.0001", 5.0, 0.0, 0.0, (0.3, 0.3), (0.0, 0.0))
    world = _World(
        {1: walker, 2: ("vehicle.driven", 0.0, 0.0, 0.0, (2.4, 1.0), (1.3, 0.0))}
    )
    actor = _Driven(world)
    assert run._walker_in_way(actor, actor.get_transform(), 0.0)  # type: ignore[arg-type]
    world = _World(
        {
            1: ("walker.pedestrian.0001", 6.0, 0.0, 0.0, (0.3, 0.3), (0.0, 0.0)),
            2: ("vehicle.driven", 0.0, 0.0, 0.0, (2.4, 1.0), (1.3, 0.0)),
        }
    )
    actor = _Driven(world)
    assert not run._walker_in_way(actor, actor.get_transform(), 0.0)  # type: ignore[arg-type]


def test_a_driven_vehicle_looks_for_a_vehicle_from_its_box_s_centre() -> None:
    """Standing, it looks 4.4 m on from its box's centre (1.3): a car (a
    circle of radius 2.6 round its origin, its box on it) at 8.0 is in its
    way, at 8.6 not -- looked for from its origin, it would be at 7.0."""
    from autoware_carla_scenario.actions import _trajectory_run as run

    def world_with_car_at(x: float) -> _World:
        return _World(
            {
                1: ("vehicle.ahead", x, 0.0, 0.0, (2.4, 1.0), (0.0, 0.0)),
                2: ("vehicle.driven", 0.0, 0.0, 0.0, (2.4, 1.0), (1.3, 0.0)),
            }
        )

    actor = _Driven(world_with_car_at(8.0))
    assert run._vehicle_in_way(actor, actor.get_transform(), 0.0)  # type: ignore[arg-type]
    actor = _Driven(world_with_car_at(8.6))
    assert not run._vehicle_in_way(actor, actor.get_transform(), 0.0)  # type: ignore[arg-type]


def test_a_vehicle_led_out_looks_on_from_its_box_s_centre(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its reach is measured from its box's centre, 1.3 m ahead of the
    origin its place on the path is: 2 m on from it looks to 3.3."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.final_approach import FinalApproach

    from .test_routing_trajectory_action import _Actor as _Vehicle

    looked: list[float] = []

    def first_box_in_way(
        world: Any,
        self_id: int,
        x: float,
        y: float,
        heading_deg: float,
        kind: str,
        distance: float,
        half_width: float,
    ) -> None:
        looked.append(x + distance)
        return None

    monkeypatch.setattr(clearance, "first_box_in_way", first_box_in_way)
    vehicle = _Vehicle(speed=0.0)  # at the origin, facing +x
    vehicle.bounding_box = SimpleNamespace(
        extent=SimpleNamespace(x=2.4, y=1.0, z=0.9),
        location=SimpleNamespace(x=1.3, y=0.0, z=0.7),
    )
    vehicle.get_world = lambda: None  # type: ignore[attr-defined]
    path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 20)]
    approach = FinalApproach(vehicle, carla.Location(0.0, 0.0, 0.0), path, stop=False)  # type: ignore[arg-type]
    assert not approach._in_way("walker", 2.0, 1.5)
    assert max(looked) == pytest.approx(3.3)
