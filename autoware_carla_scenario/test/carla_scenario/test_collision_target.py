"""Collision targeting: which collision the condition is actually about.

The sensor is faked by calling the listener directly, which is what CARLA does
from its own thread.  The behaviour under test is the filter, and in particular
*where* it sits: the first collision is latched, so a collision that is not the
one named must never be recorded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    CollisionCondition,
    CollisionTargetType,
    ScenarioResult,
)


def _other(type_id: str, role_name: str | None = None) -> MagicMock:
    actor = MagicMock()
    actor.type_id = type_id
    actor.attributes = {"role_name": role_name} if role_name else {}
    return actor


def _event(other: MagicMock, impulse: float = 100.0) -> MagicMock:
    event = MagicMock()
    event.other_actor = other
    event.normal_impulse = carla.Vector3D(impulse, 0.0, 0.0)
    return event


def _verdict(condition: CollisionCondition) -> "ScenarioResult | None":
    """Check the condition against a world with no ego, so nothing re-attaches."""
    world = MagicMock()
    world.get_actors.return_value.filter.return_value = []
    return condition.check(world, 1.0)


class TestUnrestricted:
    def test_any_collision_still_fires(self) -> None:
        """The default is unchanged: existing scenarios keep their meaning."""
        condition = CollisionCondition(label="any")
        condition._on_collision(_event(_other("static.prop.streetbarrier")))
        assert _verdict(condition) is not None

    def test_the_impulse_floor_still_applies(self) -> None:
        condition = CollisionCondition(min_impulse=50.0, label="any")
        condition._on_collision(_event(_other("vehicle.tesla.model3"), impulse=10.0))
        assert _verdict(condition) is None


class TestEntityTarget:
    def test_a_collision_with_the_named_entity_fires(self) -> None:
        condition = CollisionCondition(target="npc1", label="hit_npc1")
        condition._on_collision(
            _event(_other("vehicle.tesla.model3", role_name="npc1"))
        )
        result = _verdict(condition)
        assert result is not None
        assert not result.passed

    def test_a_collision_with_another_entity_does_not(self) -> None:
        condition = CollisionCondition(target="npc1", label="hit_npc1")
        condition._on_collision(
            _event(_other("vehicle.tesla.model3", role_name="npc2"))
        )
        assert _verdict(condition) is None

    def test_an_unwanted_collision_does_not_consume_the_latch(self) -> None:
        """Clipping a kerb must not stop the collision under test being seen.

        The condition records only the first collision, so a filter applied at
        the verdict rather than at the event would let any earlier bump make
        the real one invisible.
        """
        condition = CollisionCondition(target="npc1", label="hit_npc1")
        condition._on_collision(_event(_other("static.prop.streetbarrier")))
        assert _verdict(condition) is None

        condition._on_collision(
            _event(_other("vehicle.tesla.model3", role_name="npc1"))
        )
        assert _verdict(condition) is not None


class TestTypeTarget:
    @pytest.mark.parametrize(
        ("target_type", "type_id", "expected"),
        [
            (CollisionTargetType.PEDESTRIAN, "walker.pedestrian.0001", True),
            (CollisionTargetType.PEDESTRIAN, "vehicle.tesla.model3", False),
            (CollisionTargetType.VEHICLE, "vehicle.tesla.model3", True),
            (CollisionTargetType.VEHICLE, "static.prop.streetbarrier", False),
            (CollisionTargetType.STATIC, "static.prop.streetbarrier", True),
            (CollisionTargetType.STATIC, "walker.pedestrian.0001", False),
        ],
    )
    def test_a_class_of_object_is_matched_by_its_blueprint_prefix(
        self, target_type: CollisionTargetType, type_id: str, expected: bool
    ) -> None:
        condition = CollisionCondition(target_type=target_type, label="typed")
        condition._on_collision(_event(_other(type_id)))
        assert (_verdict(condition) is not None) is expected

    def test_any_matches_everything(self) -> None:
        assert CollisionTargetType.ANY.matches("walker.pedestrian.0001")
        assert CollisionTargetType.ANY.matches(None)


class TestConstruction:
    def test_naming_both_an_entity_and_a_type_is_refused(self) -> None:
        """Two vocabularies for one assertion; one of them would be ignored."""
        with pytest.raises(ValueError, match="not both"):
            CollisionCondition(
                target="npc1",
                target_type=CollisionTargetType.PEDESTRIAN,
                label="both",
            )

    def test_an_entity_with_an_explicit_any_is_allowed(self) -> None:
        CollisionCondition(
            target="npc1", target_type=CollisionTargetType.ANY, label="fine"
        )


class TestAuthoring:
    """The mistake has to be catchable before a live server sees it."""

    @staticmethod
    def _document(params: dict[str, object]):
        from autoware_carla_scenario.authoring.models import (
            Assertions,
            ConditionNode,
            Entity,
            ScenarioDocument,
        )

        return ScenarioDocument(
            id="s",
            entities=[Entity(id="ego", kind="ego"), Entity(id="npc1")],
            assertions=Assertions(
                fail=[ConditionNode(id="c1", type="collision", params=params)]
            ),
        )

    def _both_named(self, params: dict[str, object]) -> bool:
        from autoware_carla_scenario.authoring.validator import validate_document

        return any(
            "not both" in issue.message
            for issue in validate_document(self._document(params)).errors
        )

    def test_naming_both_is_a_document_error(self) -> None:
        """Otherwise it saves, compiles, exports and fails on the simulator."""
        assert self._both_named({"target": "npc1", "target_type": "PEDESTRIAN"})

    @pytest.mark.parametrize(
        "params",
        [{}, {"target": "npc1"}, {"target_type": "PEDESTRIAN"}],
        ids=["neither", "entity", "type"],
    )
    def test_one_or_neither_is_accepted(self, params: dict[str, object]) -> None:
        assert not self._both_named(params)

    def test_the_card_shows_which_entity_was_named(self) -> None:
        """Two differently-targeted cards must not render identically.

        The canvas renders only the fields the visual names, so a target left
        out of it is a target the author cannot see or check.
        """
        from autoware_carla_scenario.authoring import registry

        spec = registry.get_condition_spec("collision")
        assert spec is not None
        assert spec.visual.target == "target"


class TestDetails:
    def test_details_report_what_was_named(self) -> None:
        by_entity = CollisionCondition(target="npc1", label="a")
        assert by_entity.get_details()["target"] == "npc1"

        by_type = CollisionCondition(
            target_type=CollisionTargetType.PEDESTRIAN, label="b"
        )
        assert by_type.get_details()["target_type"] == "PEDESTRIAN"

        unrestricted = CollisionCondition(label="c")
        details = unrestricted.get_details()
        assert "target" not in details and "target_type" not in details
