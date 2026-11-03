"""SpawnAction placed against an entity (#1276): a ``RelativeLanePose``, or a
route pose without an anchor.

Its reference is read as a condition reads a second entity: one parked under
the map (spawned hidden, not brought in yet) is not in the scene, so nothing
is placed against it -- not a point under the road.  With no ``yaw`` of its
own, such a place faces along the lane it is placed on, not world 0 deg.

An entity brought in on the very frame a place is written against it is
not seen there yet (a client sees a new transform from the next tick on):
such a place is placed against where it was put, on that frame only.

Placed on the synthetic crossroads of :mod:`._route_maps` (CARLA's frame is
the map's with y flipped): the S road's northbound lane ``S_IN`` runs at map
x = +1.75 from y = -110 to -10, so CARLA yaw -90.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import SpawnAction, entity_role
from autoware_carla_scenario.actions import _trajectory_run
from autoware_carla_scenario.actions import spawn as spawn_module
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.trajectory import resolve as resolve_module
from autoware_carla_scenario.trajectory.model import RelativeLanePose, RouteLanePose

from . import _route_maps as rm
from .test_adversity_feasible_condition import crossroads as crossroads  # noqa: F401

#: CARLA yaw (deg) along ``S_IN``.
_NORTHBOUND = -90.0


class _Prop:
    def __init__(self, blueprint: str, transform: Any) -> None:
        self.blueprint, self.transform = blueprint, transform

    def destroy(self) -> None:
        pass


def _actor(role: str, x: float, y: float, z: float = 0.0) -> MagicMock:
    actor = MagicMock()
    actor.id = 7
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(x, y, z)
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x, y, z), carla.Rotation(yaw=_NORTHBOUND)
    )
    return actor


class _World:
    id = 1276

    def __init__(self, *actors: Any) -> None:
        self.actors = list(actors)
        self.spawned: list[_Prop] = []
        self.frame = 10

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self.frame)

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(filter=lambda name: [name])

    def get_map(self) -> Any:
        ground = SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(z=0.0))
        )
        return SimpleNamespace(get_waypoint=lambda *a, **k: ground)

    def get_actors(self) -> list[Any]:
        return self.actors

    def get_actor(self, actor_id: int) -> Any:
        for actor in self.actors:
            if actor.id == actor_id:
                return actor
        return None

    def try_spawn_actor(self, blueprint: str, transform: Any) -> Optional[_Prop]:
        prop = _Prop(blueprint, transform)
        self.spawned.append(prop)
        return prop


@pytest.fixture(autouse=True)
def _no_props_standing() -> Iterator[None]:
    # The actor ids found by role name too: another test's world has this id.
    spawn_module._STANDING.clear()
    spawn_module._BROUGHT_IN.clear()
    entity_role._ACTOR_IDS.clear()
    yield
    spawn_module._STANDING.clear()
    spawn_module._BROUGHT_IN.clear()
    entity_role._ACTOR_IDS.clear()


@pytest.fixture
def unoccupied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every place is clear, and claiming one does nothing."""
    monkeypatch.setattr(spawn_module, "_taken", lambda *a, **k: -1)
    monkeypatch.setattr(spawn_module, "_hold", lambda *a, **k: None)


def _on_s_in(z: float = 0.0) -> MagicMock:
    """``npc1`` 50 m up ``S_IN`` (map y = -50), at *z*."""
    return _actor("npc1", 1.75, 50.0, z)


_AHEAD = RelativeLanePose(ds=5.0, entity_ref="npc1")


class TestAHiddenReference:
    def test_a_prop_is_not_put_under_or_over_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        world = _World(_on_s_in(z=-500.0))
        with caplog.at_level(logging.WARNING, logger=spawn_module.__name__):
            assert not SpawnAction(name="cone", position=_AHEAD).apply(world)  # type: ignore[arg-type]
        assert world.spawned == []
        assert "'npc1' is not in the scene" in caplog.text

    def test_an_entity_has_no_place_against_it(self, unoccupied: None) -> None:
        world = _World(_on_s_in(z=-500.0))
        action = SpawnAction("npc2", position=_AHEAD)
        # None: brought in where it was declared, as against an absent one.
        assert action._place(world, 9, 2.0, 1.0) is None  # type: ignore[arg-type]

    def test_it_is_treated_as_an_absent_one(self, unoccupied: None) -> None:
        action = SpawnAction("npc2", position=_AHEAD)
        assert action._place(_World(), 9, 2.0, 1.0) is None  # type: ignore[arg-type]

    def test_once_in_the_scene_it_is_placed_against(self, unoccupied: None) -> None:
        action = SpawnAction("npc2", position=_AHEAD)
        place = action._place(_World(_on_s_in()), 9, 2.0, 1.0)  # type: ignore[arg-type]
        assert place is not None
        x, y, _z, _yaw = place
        assert (x, y) == pytest.approx((1.75, 45.0), abs=0.05)


@pytest.fixture
def brought_in(monkeypatch: pytest.MonkeyPatch, unoccupied: None) -> MagicMock:
    """``npc1``, hidden: parked 500 m under its declared spawn on ``S_IN``;
    a SpawnAction of it finds it, and putting it moves nothing (as on a
    CARLA client, until the next tick)."""
    npc1 = _on_s_in(z=-500.0)
    monkeypatch.setattr(_trajectory_run, "_entity_actor", lambda name: npc1)
    monkeypatch.setattr(_trajectory_run, "_is_walker", lambda found: False)
    monkeypatch.setattr(spawn_module, "_put", lambda *a, **k: None)
    return npc1


#: Where ``npc1`` is brought in: 30 m up ``S_IN`` (map y = -80), not where it
#: is parked.
_ENTRY = Lanelet2Pose(rm.S_IN, 30.0, 0.0)


class TestBroughtInOnTheSameFrame:
    def test_a_place_against_it_is_placed_where_it_was_put(
        self, brought_in: MagicMock
    ) -> None:
        world = _World(brought_in)
        assert SpawnAction("npc1", position=_ENTRY).apply(world)  # type: ignore[arg-type]
        # The same frame: the world still shows npc1 parked under the map.
        assert SpawnAction(name="cone", position=_AHEAD).apply(world)  # type: ignore[arg-type]
        (cone,) = world.spawned
        location = cone.transform.location
        assert (location.x, location.y) == pytest.approx((1.75, 75.0), abs=0.05)
        assert cone.transform.rotation.yaw == pytest.approx(_NORTHBOUND, abs=0.1)

    def test_on_a_later_frame_it_is_read_where_it_stands(
        self, brought_in: MagicMock
    ) -> None:
        world = _World(brought_in)
        assert SpawnAction("npc1", position=_ENTRY).apply(world)  # type: ignore[arg-type]
        world.frame += 1
        # Where it stands now, not where it was put a frame ago.
        brought_in.get_location.return_value = carla.Location(1.75, 60.0, 0.0)
        brought_in.get_transform.return_value = carla.Transform(
            carla.Location(1.75, 60.0, 0.0), carla.Rotation(yaw=_NORTHBOUND)
        )
        assert SpawnAction(name="cone", position=_AHEAD).apply(world)  # type: ignore[arg-type]
        location = world.spawned[0].transform.location
        assert (location.x, location.y) == pytest.approx((1.75, 55.0), abs=0.05)

    def test_on_a_later_frame_still_hidden_it_is_absent(
        self, brought_in: MagicMock
    ) -> None:
        world = _World(brought_in)
        assert SpawnAction("npc1", position=_ENTRY).apply(world)  # type: ignore[arg-type]
        world.frame += 1
        assert not SpawnAction(name="cone", position=_AHEAD).apply(world)  # type: ignore[arg-type]
        assert world.spawned == []

    def test_the_runner_forgets_it_between_runs(self, brought_in: MagicMock) -> None:
        world = _World(brought_in)
        assert SpawnAction("npc1", position=_ENTRY).apply(world)  # type: ignore[arg-type]
        spawn_module.clear_props()
        assert spawn_module._BROUGHT_IN == {}


class TestTheHeading:
    def test_a_relative_pose_without_a_yaw_faces_along_its_lane(
        self, unoccupied: None
    ) -> None:
        action = SpawnAction("npc2", position=_AHEAD)
        place = action._place(_World(_on_s_in()), 9, 2.0, 1.0)  # type: ignore[arg-type]
        assert place is not None
        assert place[3] == pytest.approx(_NORTHBOUND, abs=0.1)

    def test_a_prop_faces_along_its_lane_too(self) -> None:
        world = _World(_on_s_in())
        assert SpawnAction(name="cone", position=_AHEAD).apply(world)  # type: ignore[arg-type]
        (cone,) = world.spawned
        location = cone.transform.location
        assert (location.x, location.y) == pytest.approx((1.75, 45.0), abs=0.05)
        assert cone.transform.rotation.yaw == pytest.approx(_NORTHBOUND, abs=0.1)

    def test_an_explicit_yaw_is_kept(self, unoccupied: None) -> None:
        # 0.5 rad to the left of the lane: anticlockwise on the map, so
        # clockwise in CARLA's y-flipped frame.
        turned = RelativeLanePose(ds=5.0, yaw=0.5, entity_ref="npc1")
        action = SpawnAction("npc2", position=turned)
        place = action._place(_World(_on_s_in()), 9, 2.0, 1.0)  # type: ignore[arg-type]
        assert place is not None
        assert place[3] == pytest.approx(_NORTHBOUND - math.degrees(0.5), abs=0.1)

    def test_a_route_pose_without_a_yaw_faces_along_its_lane(
        self, monkeypatch: pytest.MonkeyPatch, unoccupied: None
    ) -> None:
        # A route pose counted from the ego, placed 5 m ahead of it on a lane
        # heading 30 deg: with no yaw of its own it resolves to none.
        def resolve(
            position: Any, reference: Optional[Any] = None
        ) -> tuple[float, float, Optional[float], Optional[float]]:
            assert isinstance(position, RouteLanePose) and reference is not None
            ego = reference("Ego")
            yaw = None if position.yaw is None else 30.0 + position.yaw
            return (ego.x + position.ds, ego.y, ego.z, yaw)

        monkeypatch.setattr(resolve_module, "resolve_position", resolve)
        monkeypatch.setattr(spawn_module, "resolve_position", resolve)
        action = SpawnAction("npc2", position=RouteLanePose(ds=5.0))
        place = action._place(_World(_actor("Ego", 10.0, 0.0)), 9, 2.0, 1.0)  # type: ignore[arg-type]
        assert place is not None
        assert place == pytest.approx((15.0, 0.0, 0.0, 30.0))


class TestAnAbsolutePlace:
    def test_it_is_resolved_as_before(
        self, monkeypatch: pytest.MonkeyPatch, unoccupied: None
    ) -> None:
        along: list[Any] = []

        def never(position: Any, reference: Optional[Any] = None) -> Any:
            along.append(position)
            raise AssertionError("an absolute place is not resolved along a lane")

        monkeypatch.setattr(spawn_module, "resolve_along_lane", never)
        world = _World()
        fixed = SpawnAction("npc2", position=CarlaWorldPose(3.0, 4.0, 1.0, yaw=45.0))
        assert fixed._place(world, 9, 2.0, 1.0) == pytest.approx((3.0, 4.0, 1.0, 45.0))  # type: ignore[arg-type]
        on_lane = SpawnAction("npc2", position=Lanelet2Pose(rm.S_IN, 20.0, 0.0))
        place = on_lane._place(world, 9, 2.0, 1.0)  # type: ignore[arg-type]
        assert place is not None
        assert place[3] == pytest.approx(_NORTHBOUND, abs=0.1)
        assert along == []
