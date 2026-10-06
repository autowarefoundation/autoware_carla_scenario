"""Entities that enter the world partway through a run."""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import EgoConfig, SpawnEntityAction, SpawnTransform
from autoware_carla_scenario.actions import spawn_entity as spawn_module
from autoware_carla_scenario.authoring.models import ActionNode, ScenarioDocument
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coordinate.poses import Lanelet2Pose
from autoware_carla_scenario.declarative import DeclarativeScenario


class _Scenario:
    def __init__(self) -> None:
        self.spawned: list[tuple[str, Optional[Lanelet2Pose]]] = []

    def spawn_entity(self, role_name: str, world: Any, pose: Any) -> None:
        self.spawned.append((role_name, pose))


def _world(*roles: str) -> MagicMock:
    world = MagicMock()
    actors = []
    for role in roles:
        actor = MagicMock()
        actor.attributes = {"role_name": role}
        actor.get_location.return_value = carla.Location(1.0, 2.0, 0.0)
        actors.append(actor)
    world.get_actors.return_value = actors
    return world


class TestTheAction:
    def test_without_an_anchor_it_spawns_where_authored(self) -> None:
        scenario = _Scenario()
        SpawnEntityAction("npc1", scenario).execute(_world("ego"))
        assert scenario.spawned == [("npc1", None)]

    def test_with_an_anchor_it_spawns_along_the_road_from_it_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import autoware_carla_scenario.coordinate.map_manager as map_manager
        import autoware_carla_scenario.coordinate.transform as transform
        import autoware_carla_scenario.sweeper.bindings as bindings

        monkeypatch.setattr(
            transform, "to_lanelet2", lambda pose: Lanelet2Pose(lanelet_id=183, s=12.0)
        )
        monkeypatch.setattr(map_manager.MapManager, "get_instance", MagicMock)
        asked: dict[str, Any] = {}

        def route_offset_pose(lanelet_id, lanelet_map, graph, *, distance, side):
            asked.update(lanelet_id=lanelet_id, distance=distance, side=side)
            return 184, 42.0

        monkeypatch.setattr(bindings, "route_offset_pose", route_offset_pose)
        scenario = _Scenario()
        SpawnEntityAction(
            "npc1", scenario, anchor="ego", gap_m=30.0, side="right"
        ).execute(_world("ego"))
        # Measured from where the anchor is now: s=12 on 183, plus 30.
        assert asked == {"lanelet_id": 183, "distance": 42.0, "side": "right"}
        assert scenario.spawned == [("npc1", Lanelet2Pose(lanelet_id=184, s=42.0))]

    def test_an_anchor_not_in_the_world_spawns_nothing(self) -> None:
        scenario = _Scenario()
        SpawnEntityAction("npc1", scenario, anchor="ego").execute(_world())
        assert scenario.spawned == []

    def test_an_unknown_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="side"):
            SpawnEntityAction("npc1", _Scenario(), side="up")

    def test_the_module_offers_every_route_offset_side(self) -> None:
        from autoware_carla_scenario.sweeper.bindings import ROUTE_OFFSET_SIDES

        assert spawn_module.SPAWN_SIDES == ROUTE_OFFSET_SIDES


def _deferred_document(*, with_card: bool = True) -> ScenarioDocument:
    document = new_document()
    npc = document.entity("npc1")
    assert npc is not None
    npc.deferred = True
    npc.spawn.t = 1.5
    npc.spawn.heading = 0.25
    if with_card:
        document.actions.append(
            ActionNode(
                type="spawn_entity",
                actor="npc1",
                params={"anchor": None, "gap_m": 0.0, "side": "same"},
            )
        )
    return document


def _errors(document: ScenarioDocument) -> list[str]:
    return [issue.message for issue in validate_document(document).errors]


class TestValidation:
    def test_a_deferred_entity_with_a_spawn_card_is_valid(self) -> None:
        assert _errors(_deferred_document()) == []

    def test_a_deferred_entity_nothing_spawns_is_refused(self) -> None:
        assert any(
            "no Spawn card" in m for m in _errors(_deferred_document(with_card=False))
        )

    def test_a_spawn_card_on_an_entity_already_there_is_refused(self) -> None:
        document = _deferred_document()
        npc = document.entity("npc1")
        assert npc is not None
        npc.deferred = False
        assert any("mark it Deferred" in m for m in _errors(document))

    def test_a_repeating_spawn_card_is_refused(self) -> None:
        document = _deferred_document()
        document.actions[-1].once = False
        assert any("fires once" in m for m in _errors(document))

    def test_the_ego_cannot_be_deferred(self) -> None:
        document = new_document()
        ego = document.ego
        assert ego is not None
        ego.deferred = True
        assert any("ego starts the run" in m for m in _errors(document))


class TestTheScenario:
    @staticmethod
    def _scenario(document: ScenarioDocument) -> DeclarativeScenario:
        return DeclarativeScenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
                )
            ),
            spawn_pose=Lanelet2Pose(lanelet_id=183, s=0.0),
            document=document,
        )

    def test_a_deferred_entity_is_not_spawned_at_the_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scenario = self._scenario(_deferred_document())
        spawned: list[str] = []
        monkeypatch.setattr(DeclarativeScenario, "world", property(lambda self: None))
        monkeypatch.setattr(
            scenario, "_spawn", lambda entity, world, pose: spawned.append(entity.id)
        )
        scenario._spawn_npcs()
        assert spawned == []

    def test_spawning_mid_run_keeps_the_authored_offset_and_heading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        scenario = self._scenario(_deferred_document())
        spawned: list[Lanelet2Pose] = []
        monkeypatch.setattr(
            scenario, "_spawn", lambda entity, world, pose: spawned.append(pose)
        )
        role = scenario._compiled.role_of("npc1")
        scenario.spawn_entity(role, None, Lanelet2Pose(lanelet_id=184, s=42.0))
        assert spawned == [Lanelet2Pose(lanelet_id=184, s=42.0, t=1.5, heading=0.25)]

    def test_an_unknown_role_is_refused(self) -> None:
        scenario = self._scenario(_deferred_document())
        with pytest.raises(LookupError):
            scenario.spawn_entity("npc99", None)
