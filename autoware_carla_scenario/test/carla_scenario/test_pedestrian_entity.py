"""Pedestrians: the kind, the blueprint that follows it, and what they may do.

Three things are being kept honest here.  A pedestrian is not a vehicle with a
different model, so the blueprint has to follow the kind and the validator has
to say so when it does not.  Nothing drives a walker, so the actions a vehicle
performs must be refused on one.  And a walker moves because an action tells it
to, which is the last test group.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    PedestrianEntity,
    PedestrianEntityConfig,
    SpawnTransform,
    WalkStraightAction,
)
from autoware_carla_scenario.authoring.models import (
    ActionNode,
    Entity,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.entity.registry import (
    register_entity,
    unregister_entity,
)


def _errors(document: ScenarioDocument) -> list[str]:
    return [issue.message for issue in validate_document(document).errors]


def _document(*, actor_kind: str = "pedestrian", action: str = "walk_straight"):
    return ScenarioDocument(
        id="crossing",
        entities=[
            Entity(id="ego", kind="ego"),
            Entity(id="other", kind=actor_kind),  # type: ignore[arg-type]
        ],
        actions=[ActionNode(id="a1", type=action, actor="other")],
    )


class TestKindAndBlueprint:
    def test_a_pedestrian_gets_a_walker_blueprint(self) -> None:
        assert Entity(id="w").kind == "vehicle"
        assert Entity(id="w", kind="pedestrian").vehicle_type.startswith("walker.")

    def test_switching_kind_moves_the_blueprint_with_it(self) -> None:
        """Otherwise the card would keep a car's blueprint and fail to spawn."""
        entity = Entity(id="w")
        entity.kind = "pedestrian"
        assert entity.vehicle_type.startswith("walker.")

    def test_a_chosen_blueprint_is_never_overwritten(self) -> None:
        entity = Entity(
            id="w", kind="pedestrian", vehicle_type="walker.pedestrian.0042"
        )
        assert entity.vehicle_type == "walker.pedestrian.0042"

    def test_the_wrong_family_is_a_document_error(self) -> None:
        document = ScenarioDocument(
            id="crossing",
            entities=[
                Entity(id="ego", kind="ego"),
                Entity(
                    id="w", kind="pedestrian", vehicle_type="walker.pedestrian.0001"
                ),
            ],
        )
        document.entities[1].vehicle_type = "vehicle.tesla.model3"
        assert any("blueprint" in message for message in _errors(document))


class TestWhatAPedestrianMayDo:
    @pytest.mark.parametrize("action", ["lane_change", "turn", "routing"])
    def test_a_vehicle_action_is_refused_on_a_pedestrian(self, action: str) -> None:
        messages = _errors(_document(action=action))
        assert any("cannot be performed by a pedestrian" in m for m in messages)

    def test_walking_is_refused_on_a_vehicle(self) -> None:
        messages = _errors(_document(actor_kind="vehicle"))
        assert any("cannot be performed by a vehicle" in m for m in messages)

    def test_walking_is_accepted_on_a_pedestrian(self) -> None:
        messages = _errors(_document())
        assert not any("cannot be performed" in m for m in messages)

    def test_a_pedestrian_may_not_have_a_goal(self) -> None:
        """Already the rule for every non-ego entity; asserted for the new kind."""
        from autoware_carla_scenario.authoring.models import GoalSpec

        document = _document()
        document.entities[1].goal = GoalSpec(lanelet_id=5)
        assert any("goal" in m.lower() for m in _errors(document))


class TestSpawning:
    def test_the_walker_is_lifted_clear_of_the_ground(self) -> None:
        """A walker flush with the surface is refused as colliding with it."""
        world = MagicMock()
        blueprint = MagicMock()
        blueprint.has_attribute.return_value = True
        world.get_blueprint_library.return_value.find.return_value = blueprint

        entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1",
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=1.0, y=2.0, z=0.0))
                ),
            )
        )
        entity.spawn(world)

        placed = world.try_spawn_actor.call_args[0][1]
        assert placed.location.z > 0.0

    def test_a_refused_spawn_point_raises(self) -> None:
        world = MagicMock()
        world.get_blueprint_library.return_value.find.return_value = MagicMock()
        world.try_spawn_actor.return_value = None

        entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1",
                spawn_location=SpawnTransform(carla.Transform()),
            )
        )
        with pytest.raises(RuntimeError, match="could not spawn"):
            entity.spawn(world)

    def test_an_unavailable_blueprint_raises(self) -> None:
        world = MagicMock()
        world.get_blueprint_library.return_value.find.side_effect = IndexError
        entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1",
                spawn_location=SpawnTransform(carla.Transform()),
                walker_type="walker.pedestrian.9999",
            )
        )
        with pytest.raises(ValueError, match="not available"):
            entity.spawn(world)


class TestWalking:
    @pytest.fixture
    def walker(self) -> Iterator[PedestrianEntity]:
        entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1",
                spawn_location=SpawnTransform(carla.Transform()),
            )
        )
        actor = MagicMock()
        transform = MagicMock()
        transform.get_forward_vector.return_value = carla.Vector3D(1.0, 0.0, 0.0)
        actor.get_transform.return_value = transform
        entity._walker = actor
        register_entity("walker1", entity)
        yield entity
        unregister_entity("walker1")

    def test_the_action_sends_the_walker_forward(
        self, walker: PedestrianEntity
    ) -> None:
        WalkStraightAction(entity_name="walker1", speed_ms=1.4).execute(MagicMock())

        actor = walker.actor
        assert actor is not None
        control = actor.apply_control.call_args[0][0]
        assert control.speed == pytest.approx(1.4)
        assert control.direction.x == pytest.approx(1.0)

    def test_zero_speed_stops_the_walker(self, walker: PedestrianEntity) -> None:
        WalkStraightAction(entity_name="walker1", speed_ms=0.0).execute(MagicMock())
        actor = walker.actor
        assert actor is not None
        assert actor.apply_control.call_args[0][0].speed == pytest.approx(0.0)

    def test_a_negative_speed_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            WalkStraightAction(entity_name="walker1", speed_ms=-1.0)

    def test_commanding_a_vehicle_is_a_no_op_rather_than_a_crash(self) -> None:
        """The validator is the guard; the runtime still must not fall over."""
        register_entity("npc1", MagicMock(spec=[]))
        try:
            WalkStraightAction(entity_name="npc1").execute(MagicMock())
        finally:
            unregister_entity("npc1")

    def test_an_unspawned_pedestrian_does_not_crash(self) -> None:
        entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker2",
                spawn_location=SpawnTransform(carla.Transform()),
            )
        )
        entity.walk_straight(1.4)
