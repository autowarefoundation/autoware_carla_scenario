"""Replaying a transcribed T4 scene: who is spawned where, and what drives them.

The map is stood in for by its projector offset alone (all a map-frame pose
needs to reach CARLA) and the world by a fake that records what is spawned, so
these tests are about the setup the replay builds rather than the simulation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EgoConfig,
    Lanelet2Pose,
    SpawnPointIndex,
    SpawnTransform,
    T4Category,
    T4ReplayScenario,
    TrajectoryFollowingMode,
    read_t4_scene,
    replay_t4_objects,
)
from autoware_carla_scenario.actions.follow_trajectory import HIDDEN_DEPTH_M
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.entity.registry import (
    clear_entities,
    find_entity_by_role_name,
)
from autoware_carla_scenario.trajectory.t4 import T4SceneTranscription

from .test_t4_transcription import _write_scene

#: The projector offset of the stand-in map: map (1000, 2000) is CARLA (0, 0).
_OFFSET = (1000.0, 2000.0)
_ROAD_Z = 3.0


class _Actor:
    _next = 0

    def __init__(self, blueprint: Any, transform: carla.Transform) -> None:
        _Actor._next += 1
        self.id = _Actor._next
        self.type_id = blueprint.id
        self.attributes = {"role_name": blueprint.role_name}
        self.transform = transform
        self.physics = True

    def set_simulate_physics(self, enabled: bool) -> None:
        self.physics = enabled


class _Blueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id
        self.role_name = ""

    def has_attribute(self, name: str) -> bool:
        return name == "role_name"

    def set_attribute(self, name: str, value: str) -> None:
        if name == "role_name":
            self.role_name = value


class _Library:
    def filter(self, pattern: str) -> list[_Blueprint]:
        return [_Blueprint("vehicle.tesla.model3"), _Blueprint("vehicle.lincoln.mkz")]

    def find(self, blueprint_id: str) -> _Blueprint:
        return _Blueprint(blueprint_id)


class _World:
    def __init__(self) -> None:
        self.spawned: list[_Actor] = []

    def get_blueprint_library(self) -> _Library:
        return _Library()

    def get_map(self) -> Any:
        ground = SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(z=_ROAD_Z))
        )
        return SimpleNamespace(get_waypoint=lambda *a, **k: ground)

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> _Actor:
        actor = _Actor(blueprint, transform)
        self.spawned.append(actor)
        return actor

    def by_role(self, role: str) -> _Actor:
        (actor,) = [a for a in self.spawned if a.attributes["role_name"] == role]
        return actor


class _Client:
    def __init__(self, world: _World) -> None:
        self.world = world

    def get_world(self) -> _World:
        return self.world


@pytest.fixture(autouse=True)
def stand_in_map() -> Iterator[None]:
    manager = MapManager.get_instance()
    manager._mgrs_offset = _OFFSET
    manager._z_offset = 0.0
    yield
    MapManager.reset()
    clear_entities()


@pytest.fixture
def scene(tmp_path: Path) -> T4SceneTranscription:
    return read_t4_scene(_write_scene(tmp_path))


def _scenario(
    scene: T4SceneTranscription, world: _World, **kwargs: Any
) -> T4ReplayScenario:
    scenario = T4ReplayScenario(scene, **kwargs)
    scenario.set_client(_Client(world))  # type: ignore[arg-type]
    return scenario


class TestReplayObjects:
    def test_every_road_user_is_spawned_and_followed(
        self, scene: T4SceneTranscription
    ) -> None:
        world = _World()
        actions = replay_t4_objects(_scenario(scene, world), scene)
        assert {action.label for action in actions} == {
            "follow_t4_vehicle_0",
            "follow_t4_pedestrian_1",
        }
        assert find_entity_by_role_name("t4_vehicle_0") is not None
        assert world.by_role("t4_vehicle_0").type_id == "vehicle.tesla.model3"
        assert world.by_role("t4_pedestrian_1").type_id == "walker.pedestrian.0001"

    def test_one_seen_from_the_start_stands_where_it_was_seen(
        self, scene: T4SceneTranscription
    ) -> None:
        world = _World()
        replay_t4_objects(_scenario(scene, world), scene)
        car = world.by_role("t4_vehicle_0")
        # Map (1010, 2005) is CARLA (10, -5), on the road and a little above it.
        assert (car.transform.location.x, car.transform.location.y) == pytest.approx(
            (10.0, -5.0)
        )
        assert car.transform.location.z == pytest.approx(_ROAD_Z + 0.3)
        assert car.physics is True

    def test_one_seen_late_waits_out_of_the_world(
        self, scene: T4SceneTranscription
    ) -> None:
        world = _World()
        replay_t4_objects(_scenario(scene, world), scene)
        walker = world.by_role("t4_pedestrian_1")
        # PedestrianEntity lifts its first try by 0.3 m.
        assert walker.transform.location.z == pytest.approx(
            _ROAD_Z - HIDDEN_DEPTH_M + 0.3
        )
        assert walker.physics is False

    def test_follow_mode_does_not_hide(self, scene: T4SceneTranscription) -> None:
        world = _World()
        replay_t4_objects(
            _scenario(scene, world),
            scene,
            following_mode=TrajectoryFollowingMode.FOLLOW,
        )
        assert world.by_role("t4_pedestrian_1").physics is True

    def test_categories_and_blueprints_are_chosen(
        self, scene: T4SceneTranscription
    ) -> None:
        world = _World()
        actions = replay_t4_objects(
            _scenario(scene, world),
            scene,
            categories=[T4Category.VEHICLE],
            blueprint_for=lambda track: "vehicle.lincoln.mkz",
            role_prefix="rec",
        )
        assert [action.label for action in actions] == ["follow_rec_vehicle_0"]
        assert world.by_role("rec_vehicle_0").type_id == "vehicle.lincoln.mkz"


class TestReplayScenario:
    def test_the_ego_starts_where_the_recording_did(
        self, scene: T4SceneTranscription, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.coordinate.transform.to_lanelet2",
            lambda pose: Lanelet2Pose(lanelet_id=7, s=1.0),
        )
        world = _World()
        scenario = _scenario(scene, world)
        scenario.setup()
        location = scenario.ego_config.spawn_location
        assert isinstance(location, SpawnTransform)
        spawn = location.value
        # Map (1000, 2001.4) is CARLA (0, -1.4), facing North: CARLA yaw -90.
        assert (spawn.location.x, spawn.location.y) == pytest.approx((0.0, -1.4))
        assert spawn.rotation.yaw == pytest.approx(-90.0)
        assert scenario.goal_pose == Lanelet2Pose(lanelet_id=7, s=1.0)
        assert len(scenario.object_actions) == 2
        assert [c.label for c in scenario._pass_conditions] == ["t4_scene_end"]

    def test_a_goal_the_config_names_is_kept(self, scene: T4SceneTranscription) -> None:
        goal = Lanelet2Pose(lanelet_id=3, s=0.0)
        world = _World()
        scenario = _scenario(
            scene,
            world,
            ego_config=EgoConfig(SpawnPointIndex(0), goal_pose=goal),
        )
        scenario.setup()
        assert scenario.goal_pose == goal

    def test_the_ego_can_be_replayed_too(
        self, scene: T4SceneTranscription, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "autoware_carla_scenario.coordinate.transform.to_lanelet2",
            lambda pose: Lanelet2Pose(lanelet_id=7, s=1.0),
        )
        world = _World()
        scenario = _scenario(scene, world, replay_ego=True)
        scenario.setup()
        labels = [action.label for action in scenario._pre_tick_actions]
        assert "follow_t4_ego" in labels

    def test_a_saved_transcription_is_read_by_path(
        self, scene: T4SceneTranscription, tmp_path: Path
    ) -> None:
        path = scene.save(tmp_path / "scene.json")
        assert T4ReplayScenario(path).transcription == scene
