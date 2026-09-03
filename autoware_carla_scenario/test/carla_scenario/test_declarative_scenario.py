"""DeclarativeScenario builds the framework's own runtime objects.

The point of the declarative path is that it adds no runtime of its own, so
these tests assert the *types* that come out of the builders: an authored
trigger must be the same ``AndCondition`` over the same leaf conditions a
hand-written scenario would have registered.

Spawning and ticking need a live CARLA server and are covered by the
integration suite; everything here runs against constructors only.
"""

from __future__ import annotations

import carla
import pytest

from autoware_carla_scenario import (
    AndCondition,
    CollisionCondition,
    EgoConfig,
    EntityDistanceCondition,
    Lanelet2Pose,
    SpawnTransform,
    StickyCondition,
    TimeoutCondition,
    TimeToCollisionCondition,
)
from autoware_carla_scenario.actions import LaneChangeAction, TickTiming
from autoware_carla_scenario.authoring.builders import (
    instantiate_action,
    instantiate_condition,
)
from autoware_carla_scenario.authoring.compiler import BuildContext, compile_document
from autoware_carla_scenario.authoring.models import ConditionNode
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.declarative import (
    DeclarativeScenario,
    DeclarativeScenarioConfig,
)


def _ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        )
    )


def _scenario(document=None, config=None) -> DeclarativeScenario:
    return DeclarativeScenario(
        _ego_config(),
        spawn_pose=Lanelet2Pose(lanelet_id=183, s=0.0),
        config=config,
        document=document or new_document(),
    )


class TestConstruction:
    def test_the_document_is_compiled_up_front(self) -> None:
        """An invalid document should not cost a CARLA session to discover."""
        scenario = _scenario()
        assert scenario.compiled.roles == {"ego": "Ego", "npc1": "npc1"}
        assert len(scenario.compiled.actions) == 1

    def test_an_invalid_document_raises_at_construction(self) -> None:
        from autoware_carla_scenario.authoring.compiler import CompilationError

        document = new_document()
        document.assertions.pass_conditions = []
        with pytest.raises(CompilationError):
            _scenario(document)

    def test_timeout_defaults_to_the_document(self) -> None:
        assert _scenario().timeout_seconds == 30.0

    def test_hydra_can_override_the_timeout(self) -> None:
        config = DeclarativeScenarioConfig(name="cut_in", timeout_seconds=7.5)
        assert _scenario(config=config).timeout_seconds == 7.5

    def test_a_missing_document_path_is_reported_clearly(self, tmp_path) -> None:
        config = DeclarativeScenarioConfig(
            name="x", document_path=str(tmp_path / "nope.yaml")
        )
        with pytest.raises(ValueError, match="not found"):
            DeclarativeScenario(
                _ego_config(),
                spawn_pose=Lanelet2Pose(lanelet_id=1, s=0.0),
                config=config,
            )

    def test_a_document_loads_from_a_path(self, tmp_path) -> None:
        from autoware_carla_scenario.authoring.persistence import save_document

        path = save_document(new_document(), tmp_path / "document.yaml")
        config = DeclarativeScenarioConfig(name="cut_in", document_path=str(path))
        assert _scenario_from(config).document.id == "cut_in"

    def test_spawn_overrides_reach_the_entity(self) -> None:
        """This is how a swept NPC spawn arrives from the sweeper."""
        config = DeclarativeScenarioConfig(
            name="cut_in", spawn_overrides={"npc1": {"lanelet_id": 999, "s": 3.5}}
        )
        scenario = _scenario(config=config)
        npc = scenario.document.entity("npc1")
        assert npc is not None
        assert npc.spawn.lanelet_id == 999
        assert npc.spawn.s.value == 3.5

    def test_spawn_overrides_for_an_unknown_entity_are_ignored(self) -> None:
        config = DeclarativeScenarioConfig(
            name="cut_in", spawn_overrides={"ghost": {"lanelet_id": 1}}
        )
        assert _scenario(config=config).document.entity("ghost") is None


def _scenario_from(config: DeclarativeScenarioConfig) -> DeclarativeScenario:
    return DeclarativeScenario(
        _ego_config(), spawn_pose=Lanelet2Pose(lanelet_id=183, s=0.0), config=config
    )


class TestBuildersProduceFrameworkObjects:
    def test_a_trigger_becomes_the_framework_condition_tree(self) -> None:
        compiled = compile_document(new_document())
        ctx = BuildContext(scenario=None, client=None, tm_port=8000)
        trigger = compiled.actions[0].trigger
        assert trigger is not None

        condition = instantiate_condition(trigger, ctx)
        assert isinstance(condition, AndCondition)
        children = condition.get_details()["children"]
        assert [c["condition_type"] for c in children] == [
            EntityDistanceCondition.__name__,
            TimeToCollisionCondition.__name__,
        ]

    def test_an_action_becomes_the_framework_action(self) -> None:
        compiled = compile_document(new_document())
        ctx = BuildContext(scenario=None, client=None, tm_port=8123)
        action = instantiate_action(compiled.actions[0], ctx)
        assert isinstance(action, LaneChangeAction)
        assert action.timing is TickTiming.PRE_TICK
        assert action.label == "Cut in"

    def test_assertions_become_framework_conditions(self) -> None:
        document = new_document()
        document.assertions.pass_conditions = [
            ConditionNode(
                type="sticky",
                children=[
                    ConditionNode(
                        type="elapsed_time",
                        params={"rule": "greater_than_or_equal", "duration_seconds": 5},
                    )
                ],
            )
        ]
        compiled = compile_document(document)
        ctx = BuildContext(scenario=None, client=None)

        passes = [instantiate_condition(c, ctx) for c in compiled.pass_conditions]
        fails = [instantiate_condition(c, ctx) for c in compiled.fail_conditions]
        assert isinstance(passes[0], StickyCondition)
        assert isinstance(fails[0], CollisionCondition)
        assert isinstance(fails[1], TimeoutCondition)

    def test_every_condition_gets_a_non_empty_label(self) -> None:
        """BaseCondition rejects an empty label, so the compiler must supply one."""
        compiled = compile_document(new_document())
        ctx = BuildContext(scenario=None, client=None)
        trigger = compiled.actions[0].trigger
        assert trigger is not None
        assert instantiate_condition(trigger, ctx).label

    def test_an_unregistered_builder_is_reported(self) -> None:
        from dataclasses import replace

        from autoware_carla_scenario.authoring import builders

        compiled = compile_document(new_document())
        broken = compiled.fail_conditions[0]
        broken = type(broken)(
            spec=replace(broken.spec, builder="build_nothing"),
            node=broken.node,
            params=broken.params,
        )
        with pytest.raises(LookupError, match="build_nothing"):
            builders.instantiate_condition(broken, BuildContext(scenario=None))
