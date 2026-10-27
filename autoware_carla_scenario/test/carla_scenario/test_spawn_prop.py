"""SpawnAction with a prop: one added to the world and taken away by name
(#447; PropAction until 1.0).

The Lanelet2 position is placed on the synthetic crossroads of
:mod:`._route_maps`, standing in the MapManager with no offset (CARLA's frame
is the map's with y flipped): W_IN1 runs east from x = -110 at y = -1.75.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario import DespawnAction, SpawnAction
from autoware_carla_scenario.actions import spawn as prop_module
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.trajectory.model import RelativeLanePose

from . import _route_maps as rm
from .test_adversity_feasible_condition import crossroads as crossroads  # noqa: F401


class _Prop:
    def __init__(self, blueprint: str, transform: Any) -> None:
        self.blueprint, self.transform = blueprint, transform
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class _World:
    id = 1

    def __init__(self, ground: Optional[float] = 0.5) -> None:
        self.ground = ground
        self.spawned: list[_Prop] = []

    def get_blueprint_library(self) -> Any:
        def find(name: str) -> list[str]:
            return [name] if name.startswith("static.prop.") else []

        return SimpleNamespace(filter=find)

    def get_map(self) -> Any:
        if self.ground is None:
            return SimpleNamespace(get_waypoint=lambda *a, **k: None)
        ground = SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(z=self.ground))
        )
        return SimpleNamespace(get_waypoint=lambda *a, **k: ground)

    def get_actors(self) -> list[Any]:
        return []

    def try_spawn_actor(self, blueprint: str, transform: Any) -> Optional[_Prop]:
        prop = _Prop(blueprint, transform)
        self.spawned.append(prop)
        return prop


@pytest.fixture(autouse=True)
def _no_props_standing() -> Iterator[None]:
    prop_module._STANDING.clear()
    yield
    prop_module._STANDING.clear()


def _add(name: str, position: Any, **kwargs: Any) -> SpawnAction:
    return SpawnAction(name=name, position=position, **kwargs)


def test_a_prop_is_added_on_the_road_and_removed_by_name() -> None:
    world = _World()
    added = _add(
        "box", CarlaWorldPose(10.0, -3.0, 9.0, yaw=90.0), blueprint="static.prop.box01"
    )
    added.tick(world, 0.0)  # type: ignore[arg-type]
    (box,) = world.spawned
    assert box.blueprint == "static.prop.box01"
    location = box.transform.location
    # On the road surface (0.5), a little above it; the pose's own z unused.
    assert (location.x, location.y, location.z) == pytest.approx((10.0, -3.0, 0.55))
    assert box.transform.rotation.yaw == pytest.approx(90.0)
    assert prop_module.standing_props() == ["box"]
    removed = DespawnAction(name="box")
    removed.tick(world, 0.1)  # type: ignore[arg-type]
    assert box.destroyed and prop_module.standing_props() == []
    # Removing it again finds nothing.
    assert not prop_module.remove_prop("box")


def test_off_the_ground_it_stands_at_its_own_height() -> None:
    world = _World()
    _add("sign", CarlaWorldPose(1.0, 2.0, 3.0, 0.0), on_ground=False).apply(world)  # type: ignore[arg-type]
    assert world.spawned[0].transform.location.z == pytest.approx(3.0)


def test_a_lanelet2_position_is_placed_on_its_lane(crossroads: Any) -> None:  # noqa: F811
    world = _World()
    assert _add("cone", Lanelet2Pose(rm.W_IN1, 20.0, 0.0)).apply(world)  # type: ignore[arg-type]
    location = world.spawned[0].transform.location
    assert (location.x, location.y) == pytest.approx((-90.0, 1.75), abs=0.05)


def test_a_name_added_again_is_replaced() -> None:
    world = _World()
    _add("cone", CarlaWorldPose(0.0, 0.0, 0.0)).apply(world)  # type: ignore[arg-type]
    _add("cone", CarlaWorldPose(5.0, 0.0, 0.0)).apply(world)  # type: ignore[arg-type]
    old, new = world.spawned
    assert old.destroyed and not new.destroyed
    assert prop_module.standing_props() == ["cone"]


def test_the_runner_clears_every_prop_away() -> None:
    world = _World()
    for name in ("a", "b"):
        _add(name, CarlaWorldPose(0.0, 0.0, 0.0)).apply(world)  # type: ignore[arg-type]
    prop_module.clear_props()
    assert all(p.destroyed for p in world.spawned)
    assert prop_module.standing_props() == []


def test_many_props_share_the_library_and_the_ground() -> None:
    world = _World(ground=None)
    library = world.get_blueprint_library()
    for i in range(3):
        _add(f"c{i}", CarlaWorldPose(float(i), 0.0, 0.0)).apply(
            world,  # type: ignore[arg-type]
            library,
            ground=lambda x, y: 2.0,
        )
    assert [p.transform.location.z for p in world.spawned] == pytest.approx([2.05] * 3)


def test_what_cannot_be_added() -> None:
    world = _World()
    with pytest.raises(ValueError):
        SpawnAction(name="cone")
    with pytest.raises(ValueError):
        SpawnAction("npc1", name="cone", position=CarlaWorldPose(0.0, 0.0, 0.0))
    assert not _add(
        "x", CarlaWorldPose(0.0, 0.0, 0.0, 0.0), blueprint="vehicle.no_such"
    ).apply(world)  # type: ignore[arg-type]
    # Relative to an entity that is not in the world: nowhere to stand.
    assert not _add("y", RelativeLanePose(ds=5.0, entity_ref="npc1")).apply(world)  # type: ignore[arg-type]
    assert world.spawned == [] and prop_module.standing_props() == []
