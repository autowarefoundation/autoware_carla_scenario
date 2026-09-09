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


class TestTheEgoGoal:
    """A document names where the ego is going; the ego config carries it."""

    @staticmethod
    def _document_with_goal(lanelet_id: int | None = 265, s: float = 12.5):
        """The starter document, with the goal replaced -- or removed."""
        from autoware_carla_scenario.authoring.models import GoalSpec

        document = new_document()
        ego = document.ego
        assert ego is not None
        ego.goal = None if lanelet_id is None else GoalSpec(lanelet_id=lanelet_id, s=s)
        return document

    def test_the_documents_goal_lands_on_the_ego_config(self) -> None:
        scenario = _scenario(self._document_with_goal())

        assert scenario.goal_pose is not None
        assert (scenario.goal_pose.lanelet_id, scenario.goal_pose.s) == (265, 12.5)
        assert scenario.ego_config.goal_pose is scenario.goal_pose

    def test_a_goal_already_on_the_config_wins(self) -> None:
        """A CLI override reaches the scenario as a goal on the ego config."""
        ego_config = EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
            ),
            goal_pose=Lanelet2Pose(lanelet_id=42, s=1.0),
        )
        scenario = DeclarativeScenario(
            ego_config,
            spawn_pose=Lanelet2Pose(lanelet_id=183, s=0.0),
            document=self._document_with_goal(),
        )

        assert scenario.goal_pose is not None
        assert scenario.goal_pose.lanelet_id == 42

    def test_an_autoware_document_without_a_goal_does_not_run(self) -> None:
        """Autoware will not move without one, so the document cannot compile."""
        from autoware_carla_scenario.authoring.compiler import CompilationError

        document = self._document_with_goal(lanelet_id=None)
        ego = document.ego
        assert ego is not None
        ego.driven_by = "autoware"

        with pytest.raises(CompilationError, match="no goal"):
            _scenario(document)

    def test_a_trafficmanager_document_without_a_goal_runs(self) -> None:
        # The ego is driven for it and reads no goal: the run is about what
        # happens on the way.
        scenario = _scenario(self._document_with_goal(lanelet_id=None))

        assert scenario.goal_pose is None

    def test_the_starter_document_brings_its_own_goal(self) -> None:
        assert _scenario().goal_pose is not None


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


class TestRoutingActionBuilder:
    """A goal card compiles into the action that hands an entity its mission.

    The ego needs a goal during initialization -- the runner waits for it to be
    ready before the loop starts, and it is not ready until it has routed.  That
    is a constraint on *the ego's first* routing, not on the action: re-routing
    part-way through a run, or routing another entity once something has
    happened, is an ordinary tick-loop action.
    """

    @staticmethod
    def _document_with_a_goal(actor: str):
        from autoware_carla_scenario.authoring.models import ActionNode

        document = new_document()
        document.actions = [
            ActionNode(
                type="routing",
                title="Drive to the far side",
                actor=actor,
                params={"goal_lanelet_id": 265, "goal_s": 4.0},
            )
        ]
        return document

    def test_a_goal_on_the_ego_becomes_a_routing_action(self) -> None:
        from autoware_carla_scenario.actions import RoutingAction

        compiled = compile_document(self._document_with_a_goal("ego"))

        action = instantiate_action(compiled.actions[0], BuildContext(scenario=None))

        assert isinstance(action, RoutingAction)
        assert action.goal.lanelet_id == 265
        assert action.goal.s == 4.0
        assert action.label == "Drive to the far side"

    def test_the_action_names_its_entity_rather_than_holding_one(self) -> None:
        """Like every other action, and for the same reason.

        The document is compiled before the scenario has spawned anything, so an
        action handed an entity now would be handed nothing.  It carries the
        role and looks the entity up when it runs.
        """
        from autoware_carla_scenario.actions import RoutingAction
        from autoware_carla_scenario.constants import EGO_ROLE_NAME

        compiled = compile_document(self._document_with_a_goal("ego"))

        action = instantiate_action(compiled.actions[0], BuildContext(scenario=None))

        assert isinstance(action, RoutingAction)
        assert action.entity_name == str(EGO_ROLE_NAME)

    def test_the_card_carries_the_role_an_author_drew_it_on(self) -> None:
        """Routing an NPC is allowed: the entity decides whether it means anything.

        A TrafficManager NPC takes no destination, so ``route_to`` is a no-op
        for it -- which is better than refusing the compile over a card that
        simply does nothing.
        """
        from autoware_carla_scenario.actions import RoutingAction

        compiled = compile_document(self._document_with_a_goal("npc1"))

        action = instantiate_action(compiled.actions[0], BuildContext(scenario=None))

        assert isinstance(action, RoutingAction)
        assert action.entity_name == "npc1"

    def test_a_re_route_can_live_on_the_tick_loop(self) -> None:
        """The phase and ``once`` come from the document, not from the action.

        Only the ego's *first* goal has to be handed over during initialization;
        a later one is triggered like any other action.
        """
        from autoware_carla_scenario.actions import TickTiming

        document = self._document_with_a_goal("ego")
        document.actions[0].phase = "post_tick"
        document.actions[0].once = False
        compiled = compile_document(document)

        action = instantiate_action(compiled.actions[0], BuildContext(scenario=None))

        assert action.timing is TickTiming.POST_TICK
        assert action._once is False


class TestTheInitPhaseIsAPhaseAndNotATick:
    """An init action is registered before the loop, not on it.

    The runner waits for the ego to report ready before the tick loop starts,
    and an Autoware ego only becomes ready once it has been routed.  A goal
    registered on the loop would be delivered to a stack the runner is already
    done waiting for, so which register_* an action reaches is the whole of
    whether it works.
    """

    def test_routing_lands_in_init_by_default(self) -> None:
        from autoware_carla_scenario.authoring.registry import get_action_spec

        spec = get_action_spec("routing")
        assert spec is not None
        assert spec.default_phase == "init"

    def test_an_init_action_is_kept_out_of_the_steps(self) -> None:
        from autoware_carla_scenario.authoring.models import (
            ActionNode,
            ScenarioDocument,
        )

        document = ScenarioDocument(id="s", title="s")
        document.actions.append(
            ActionNode(id="a_init", type="routing", actor=None, phase="init")
        )
        document.actions.append(
            ActionNode(id="a_step", type="traffic_signal", actor=None)
        )

        assert [a.id for a in document.init_actions(None)] == ["a_init"]
        stepped = [a.id for slot in document.action_slots(None) for a in slot]
        assert stepped == ["a_step"]

    def test_a_draft_written_before_the_rename_still_loads(self) -> None:
        from autoware_carla_scenario.authoring.models import ActionNode

        # `extra="forbid"` would otherwise reject every draft on disk.
        assert ActionNode.model_validate(
            {"type": "t", "timing": "post_tick"}
        ).phase == ("post_tick")
