"""Unit tests for BaseScenario callback registration and ordering."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import (
    BaseAction,
    BaseCondition,
    BaseScenario,
    EgoConfig,
    Lanelet2Pose,
    PedestrianEntity,
    PedestrianEntityConfig,
    ScenarioResult,
    SpawnTransform,
    VehicleEntity,
    VehicleEntityConfig,
)
from autoware_carla_scenario.entity.registry import (
    clear_entities,
    find_entity_by_role_name,
    find_pedestrian_entity,
    find_vehicle_entity,
)


# ---------------------------------------------------------------------------
# Minimal concrete scenario for testing
# ---------------------------------------------------------------------------


class _SimpleScenario(BaseScenario):
    """Concrete scenario that records callback invocation order."""

    def __init__(self, ego_config: EgoConfig) -> None:
        super().__init__(ego_config)
        self.setup_called = False
        self.call_log: List[str] = []

    def setup(self) -> None:
        self.setup_called = True

    def is_done(self) -> bool:
        return False


class _CountingCondition(BaseCondition):
    """Records how many times check() is called before returning a result."""

    def __init__(self, trigger_after: int, passed: bool) -> None:
        self.trigger_after = trigger_after
        self.passed = passed
        self.call_count = 0

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        self.call_count += 1
        if self.call_count >= self.trigger_after:
            return ScenarioResult(
                passed=self.passed,
                message="triggered",
                elapsed_seconds=elapsed,
            )
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(carla.Transform(carla.Location(x=0, y=0, z=0))),
        vehicle_type="vehicle.mini.cooper",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseScenario:
    def test_setup_is_called(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        scenario.set_client(MagicMock())
        scenario.setup()
        assert scenario.setup_called

    def test_register_pre_tick_appends_callback(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cb = MagicMock()
        scenario.register_pre_tick(cb)
        assert cb in scenario._pre_tick_callbacks

    def test_register_post_tick_appends_callback(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cb = MagicMock()
        scenario.register_post_tick(cb)
        assert cb in scenario._post_tick_callbacks

    def test_multiple_pre_tick_callbacks_ordered(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        order: List[int] = []
        scenario.register_pre_tick(lambda w: order.append(1))
        scenario.register_pre_tick(lambda w: order.append(2))
        world = MagicMock()
        for cb in scenario._pre_tick_callbacks:
            cb(world)
        assert order == [1, 2]

    def test_register_pass_condition(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cond = _CountingCondition(trigger_after=1, passed=True)
        scenario.register_pass_condition(cond)
        assert cond in scenario._pass_conditions

    def test_register_fail_condition(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cond = _CountingCondition(trigger_after=1, passed=False)
        scenario.register_fail_condition(cond)
        assert cond in scenario._fail_conditions

    def test_ego_config_stored(self) -> None:
        cfg = _make_ego_config()
        scenario = _SimpleScenario(cfg)
        assert scenario.ego_config is cfg

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseScenario(_make_ego_config())  # type: ignore[abstract]

    def test_counting_condition_triggers_after_n_calls(self) -> None:
        cond = _CountingCondition(trigger_after=3, passed=True)
        world = MagicMock()
        assert cond.check(world, 0.0) is None
        assert cond.check(world, 0.5) is None
        result = cond.check(world, 1.0)
        assert result is not None
        assert result.passed is True

    def test_register_pre_tick_with_base_action_tracks_action(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        action = _NoOpAction()
        scenario.register_pre_tick(action)
        assert action in scenario._pre_tick_actions
        assert action not in scenario._pre_tick_callbacks

    def test_register_post_tick_with_base_action_tracks_action(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        action = _NoOpAction()
        scenario.register_post_tick(action)
        assert action in scenario._post_tick_actions
        assert action not in scenario._post_tick_callbacks

    def test_register_pre_tick_plain_callable_does_not_track_action(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cb = MagicMock()
        scenario.register_pre_tick(cb)
        assert len(scenario._pre_tick_actions) == 0
        assert cb in scenario._pre_tick_callbacks

    def test_register_post_tick_plain_callable_does_not_track_action(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        cb = MagicMock()
        scenario.register_post_tick(cb)
        assert len(scenario._post_tick_actions) == 0
        assert cb in scenario._post_tick_callbacks

    def test_base_action_tick_receives_elapsed(self) -> None:
        condition = _ElapsedRecordingCondition()
        action = _NoOpAction(condition=condition)
        world = MagicMock()
        action.tick(world, 5.0)
        assert condition.last_elapsed == 5.0


# ---------------------------------------------------------------------------
# Test helpers for BaseAction tests
# ---------------------------------------------------------------------------


class _NoOpAction(BaseAction):
    """Minimal concrete action for testing."""

    def __init__(self, condition: Optional[BaseCondition] = None) -> None:
        super().__init__(label="noop", condition=condition)
        self.executed = False

    def execute(self, world: object) -> None:  # type: ignore[override]
        self.executed = True


class _ElapsedRecordingCondition(BaseCondition):
    """Condition that records the elapsed value it receives."""

    def __init__(self) -> None:
        super().__init__(label="elapsed_recorder")
        self.last_elapsed: Optional[float] = None

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        self.last_elapsed = elapsed
        return None


# ---------------------------------------------------------------------------
# Ego entity selection
# ---------------------------------------------------------------------------


class TestCreateEgo:
    """`ScenarioRunner` calls `create_ego()` to obtain the entity it spawns."""

    def test_defaults_to_a_traffic_manager_ego(self) -> None:
        from autoware_carla_scenario.entity.ego import EgoVehicle

        ego = _SimpleScenario(_make_ego_config()).create_ego()
        assert isinstance(ego, EgoVehicle)
        assert ego.use_autopilot is True

    def test_ego_type_is_instantiated_when_given(self) -> None:
        from autoware_carla_scenario.entity.externally_driven import (
            ExternallyDrivenEgo,
        )

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_type = ExternallyDrivenEgo
        ego = scenario.create_ego()
        assert isinstance(ego, ExternallyDrivenEgo)
        assert ego.use_autopilot is False

    def test_a_prebuilt_entity_takes_precedence(self) -> None:
        """Entities needing constructor arguments are supplied as instances."""
        from autoware_carla_scenario.entity.externally_driven import (
            ExternallyDrivenEgo,
        )
        from autoware_carla_scenario.entity.ego import EgoVehicle

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_type = EgoVehicle
        prebuilt = ExternallyDrivenEgo()
        scenario.ego_entity = prebuilt

        assert scenario.create_ego() is prebuilt

    def test_create_ego_returns_the_same_instance_each_call(self) -> None:
        from autoware_carla_scenario.entity.externally_driven import (
            ExternallyDrivenEgo,
        )

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_entity = ExternallyDrivenEgo()
        assert scenario.create_ego() is scenario.create_ego()

    def test_default_ego_never_requests_termination(self) -> None:
        assert (
            _SimpleScenario(_make_ego_config()).create_ego().termination_requested
            is False
        )


# ---------------------------------------------------------------------------
# TestConfigureAutowareMission – a packaged scenario has to supply a mission
# ---------------------------------------------------------------------------


class TestRouteToGoal:
    """``ego.entity=autoware`` gives the scenario an ego that plans a route.

    Autoware localizes at an initial pose and drives to a goal; without both it
    never moves.  The scenario owns those poses because only it knows where the
    ego spawns and where the run is meant to end, so ``_setup_ego_spawn()``
    registers a :class:`RoutingAction` for them -- and a config that selected
    this entity without a goal is refused during setup rather than at the start
    of the run.

    The hand-over is an init action rather than a direct call so that it sits
    with the other things a scenario sets up before the loop, and so that the
    runner performs it at one defined point.
    """

    @staticmethod
    def _autoware_entity():
        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.entity import AutowareEgoEntity

        return AutowareEgoEntity(bridge=FakeAutowareBridge())

    @staticmethod
    def _initial_pose():
        from autoware_carla_scenario.coordinate import CarlaWorldPose

        return CarlaWorldPose(x=1.0, y=2.0, z=3.0, yaw=45.0)

    def test_an_ego_that_drives_itself_is_routed_too(self) -> None:
        # The action is a no-op for an ego the TrafficManager drives, but the
        # scenario still says where the run was aiming: every ego has a goal.
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(_make_ego_config())
        scenario.goal_pose = Lanelet2Pose(lanelet_id=141, s=0.0)

        scenario.register_route_to_goal(self._initial_pose())

        assert len(scenario._init_actions) == 1

    def test_an_ego_driven_for_it_registers_nothing_without_a_goal(self) -> None:
        # No destination named, and none needed: there is no mission to hand
        # over, so nothing is registered and nothing is refused.
        scenario = _SimpleScenario(_make_ego_config())

        scenario.register_route_to_goal(self._initial_pose())

        assert scenario._init_actions == []

    def test_an_autoware_ego_without_a_goal_is_refused(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_entity = self._autoware_entity()

        with pytest.raises(ValueError, match="plans its own route"):
            scenario.register_route_to_goal(self._initial_pose())

    def test_a_goal_registers_a_routing_action(self) -> None:
        from autoware_carla_scenario import RoutingAction
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_entity = self._autoware_entity()
        scenario.goal_pose = Lanelet2Pose(lanelet_id=123, s=4.0)

        scenario.register_route_to_goal(self._initial_pose())

        assert len(scenario._init_actions) == 1
        action = scenario._init_actions[0]
        assert isinstance(action, RoutingAction)
        assert action.goal is not None
        assert action.goal.lanelet_id == 123
        # Registered, not performed: the runner runs init actions once the ego
        # actor exists, and nothing has reached the entity yet.
        assert scenario.ego_entity is not None
        assert scenario.ego_entity.handshake._goal_pose is None  # type: ignore[union-attr]

    def test_the_registered_action_reaches_the_entity_by_name(self) -> None:
        """The scenario registers; the action finds the entity; the entity routes.

        Each of the three does one thing, so this checks the seam rather than
        the arithmetic: that the action registered for the ego reaches the ego.
        What the entity then does with the goal -- snapping it and putting it in
        Autoware's map frame -- is pinned in ``test_autoware_ego_entity.py``,
        where that code now lives.
        """
        from autoware_carla_scenario.constants import EGO_ROLE_NAME
        from autoware_carla_scenario.coordinate import Lanelet2Pose
        from autoware_carla_scenario.entity.registry import (
            clear_entities,
            register_entity,
        )

        routed: list[Lanelet2Pose] = []

        class _Ego:
            def route_to(self, world, goal, *, initial_pose=None, **_) -> None:
                routed.append(goal)

        entity = _Ego()
        clear_entities()
        register_entity(EGO_ROLE_NAME, entity)
        try:
            scenario = _SimpleScenario(_make_ego_config())
            scenario.ego_entity = self._autoware_entity()
            scenario.goal_pose = Lanelet2Pose(lanelet_id=123, s=4.0)
            scenario.set_client(MagicMock())
            scenario.register_route_to_goal(self._initial_pose())

            scenario.run_init(MagicMock())
        finally:
            clear_entities()

        assert [g.lanelet_id for g in routed] == [123]


# ---------------------------------------------------------------------------
# Tests for the rule every scenario is held to
# ---------------------------------------------------------------------------


class TestRequireGoal:
    """One statement of "every scenario says where its ego is going"."""

    def test_a_goal_is_returned(self) -> None:
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(_make_ego_config())
        scenario.goal_pose = Lanelet2Pose(lanelet_id=141, s=0.0)

        assert scenario.require_goal() is scenario.goal_pose

    def test_an_ego_driven_for_it_may_have_no_goal(self) -> None:
        # A cut-in or a red-light run is about what happens on the way; the
        # TrafficManager reads no goal, so naming none is not an error.
        scenario = _SimpleScenario(_make_ego_config())

        assert scenario.require_goal() is None

    def test_an_ego_that_plans_its_own_route_is_refused_without_one(self) -> None:
        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.entity import AutowareEgoEntity

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_entity = AutowareEgoEntity(bridge=FakeAutowareBridge())

        with pytest.raises(ValueError, match="plans its own route"):
            scenario.require_goal()

    def test_the_rule_is_read_off_the_ego_type_when_no_entity_was_built(
        self,
    ) -> None:
        from autoware_carla_scenario.entity import AutowareEgoEntity

        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_type = AutowareEgoEntity

        assert scenario.ego_requires_goal is True

    def test_the_rule_is_read_off_the_entity_itself(self) -> None:
        """A driver-policy ego given a bridge needs a goal; its class does not."""
        from unittest.mock import MagicMock

        from autoware_carla_scenario.autoware_bridge import FakeAutowareBridge
        from autoware_carla_scenario.driver.base import DriverClientConfig
        from autoware_carla_scenario.entity import CarlaDriverEntity

        config = DriverClientConfig(cameras=())
        scenario = _SimpleScenario(_make_ego_config())
        scenario.ego_entity = CarlaDriverEntity(config, client=MagicMock())
        assert scenario.ego_requires_goal is False

        scenario.ego_entity = CarlaDriverEntity(
            config, client=MagicMock(), bridge=FakeAutowareBridge()
        )
        assert scenario.ego_requires_goal is True


class TestDeriveGoalFromRoute:
    """A scenario that declares its route has already named its destination."""

    def test_the_last_lanelet_of_the_route_becomes_the_goal(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())

        scenario.derive_goal_from_route([460, 265])

        assert scenario.goal_pose is not None
        assert scenario.goal_pose.lanelet_id == 265

    def test_a_goal_that_arrived_another_way_wins(self) -> None:
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(_make_ego_config())
        scenario.goal_pose = Lanelet2Pose(lanelet_id=141, s=0.0)

        scenario.derive_goal_from_route([460, 265])

        assert scenario.goal_pose.lanelet_id == 141

    def test_an_empty_route_names_nothing(self) -> None:
        # Left for require_goal() to refuse, rather than invented here.
        scenario = _SimpleScenario(_make_ego_config())

        scenario.derive_goal_from_route([])

        assert scenario.goal_pose is None


# ---------------------------------------------------------------------------
# Tests for the goal carried by the ego config
# ---------------------------------------------------------------------------


class TestGoalOnTheEgoConfig:
    """Where the ego is going belongs to the ego, next to where it starts.

    ``scenario.goal_pose`` is a view onto ``ego_config.goal_pose``, so a goal
    set either way is the same goal, and an Autoware ego cannot be configured
    without one.
    """

    def test_an_ego_config_has_no_goal_by_default(self) -> None:
        # An ego that drives itself plans no route and needs no destination.
        assert _make_ego_config().goal_pose is None

    def test_the_scenario_reads_the_goal_off_its_ego_config(self) -> None:
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        goal = Lanelet2Pose(lanelet_id=265, s=12.5)
        scenario = _SimpleScenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0, y=0, z=0))
                ),
                goal_pose=goal,
            )
        )

        assert scenario.goal_pose is goal

    def test_assigning_the_goal_writes_it_onto_the_ego_config(self) -> None:
        # The CLI runner and a scenario that derives its goal in setup() both
        # assign the attribute; there is still only one goal.
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(_make_ego_config())
        scenario.goal_pose = Lanelet2Pose(lanelet_id=123, s=4.0)

        assert scenario.ego_config.goal_pose is scenario.goal_pose
        assert scenario.ego_config.goal_pose is not None
        assert scenario.ego_config.goal_pose.lanelet_id == 123

    def test_a_config_that_carries_the_goal_routes_without_any_assignment(
        self,
    ) -> None:
        from autoware_carla_scenario import RoutingAction
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        scenario = _SimpleScenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0, y=0, z=0))
                ),
                goal_pose=Lanelet2Pose(lanelet_id=265, s=12.5),
            )
        )

        scenario.register_route_to_goal()

        assert len(scenario._init_actions) == 1
        action = scenario._init_actions[0]
        assert isinstance(action, RoutingAction)
        assert action.goal is not None
        assert action.goal.lanelet_id == 265


class TestInitPhase:
    """``register_init`` is the phase before the loop, in the shape of the loop.

    Same two forms as ``register_pre_tick`` -- an action or a plain callable --
    but run once, by the runner, before the clock starts.
    """

    def test_callbacks_and_actions_both_run_once(self) -> None:
        world = MagicMock()
        calls: List[str] = []

        class _RecordingAction(BaseAction):
            def execute(self, world: object) -> None:
                calls.append("action")

        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_init(lambda _world: calls.append("callback"))
        scenario.register_init(_RecordingAction(label="init_action"))

        scenario.run_init(world)

        assert calls == ["callback", "action"]

    def test_an_init_action_that_fired_is_complete_once_init_is_over(self) -> None:
        # No tick steps its lifecycle: what waits on its completing (the route
        # of an entity a spawn card brought in) would otherwise never start.
        from autoware_carla_scenario.action_state import ActionState

        calls: List[str] = []

        class _RecordingAction(BaseAction):
            def execute(self, world: object) -> None:
                calls.append("action")

        action = _RecordingAction(label="init_action")
        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_init(action)
        scenario.run_init(MagicMock())
        assert calls == ["action"]
        assert action.state is ActionState.COMPLETE

    def test_an_init_action_does_not_run_on_the_tick_loop(self) -> None:
        # The registration lists are separate: an init action is not a pre/post
        # tick action, so the loop never sees it.
        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_init(lambda _world: None)
        assert scenario._pre_tick_callbacks == []
        assert scenario._post_tick_callbacks == []

    def test_nothing_registered_is_a_no_op(self) -> None:
        scenario = _SimpleScenario(_make_ego_config())
        scenario.run_init(MagicMock())


class TestConstruction:
    """Behaviour the static check's rewrites of BaseScenario must keep."""

    def test_the_random_seed_defaults_to_the_class_default(self) -> None:
        # The default is written as a literal for Codon; it must stay equal.
        scenario = _SimpleScenario(_make_ego_config())
        assert scenario.random_seed == BaseScenario.DEFAULT_RANDOM_SEED

    @pytest.mark.parametrize(
        "given",
        [
            None,
            [],
            (Lanelet2Pose(lanelet_id=1, s=0.0), Lanelet2Pose(lanelet_id=2, s=0.0)),
            (Lanelet2Pose(lanelet_id=n, s=0.0) for n in (1, 2)),
        ],
        ids=["none", "empty", "tuple", "generator"],
    )
    def test_waypoint_poses_take_any_iterable_as_a_new_list(
        self, given: Optional[Iterable[Lanelet2Pose]]
    ) -> None:
        config = EgoConfig(
            spawn_location=SpawnTransform(carla.Transform(carla.Location())),
            waypoint_poses=given,  # type: ignore[arg-type]
        )
        assert isinstance(config.waypoint_poses, list)
        assert [p.lanelet_id for p in config.waypoint_poses] == (
            [] if given is None or given == [] else [1, 2]
        )
        assert config.waypoint_poses is not given

    def test_the_backend_reaches_actions_of_every_phase(self) -> None:
        received: List[str] = []

        class _Driven(BaseAction):
            def execute(self, world: object) -> None:
                pass

            def set_traffic_backend(self, backend: object) -> None:
                received.append(self.label)

        class _NotDriven(BaseAction):
            set_traffic_backend = None

            def execute(self, world: object) -> None:
                pass

        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_init(_Driven(label="init"))
        scenario.register_pre_tick(_Driven(label="pre"))
        scenario.register_post_tick(_Driven(label="post"))
        scenario.register_post_tick(_NotDriven(label="none"))
        assert received == []
        scenario.set_traffic_backend(MagicMock())
        assert received == ["init", "pre", "post"]


class TestEntityRegistration:
    """register_entity / register_pedestrian put the entity in the registry
    by kind, and a duck-typed double of neither kind under every lookup, as
    registry.register_entity always has."""

    @pytest.fixture(autouse=True)
    def _empty_registry(self) -> Iterator[None]:
        clear_entities()
        yield
        clear_entities()

    def test_a_vehicle_is_registered_as_a_vehicle(self) -> None:
        npc = VehicleEntity(
            VehicleEntityConfig(
                role_name="npc1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_entity(npc)

        assert find_vehicle_entity("npc1") is npc
        assert find_pedestrian_entity("npc1") is None

    def test_a_pedestrian_is_registered_as_a_pedestrian(self) -> None:
        walker = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        scenario = _SimpleScenario(_make_ego_config())
        scenario.register_pedestrian(walker)

        assert find_pedestrian_entity("walker1") is walker
        assert find_vehicle_entity("walker1") is None

    def test_a_pedestrian_is_handed_the_runs_traffic_backend(self) -> None:
        walker = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        backend = MagicMock()
        scenario = _SimpleScenario(_make_ego_config())
        scenario.set_traffic_backend(backend)
        scenario.register_pedestrian(walker)

        backend.adopt_pedestrian.assert_called_once_with(walker)
        assert walker._traffic_backend is backend

    @pytest.mark.parametrize("method", ["register_entity", "register_pedestrian"])
    def test_a_double_of_neither_kind_is_found_by_every_lookup(
        self, method: str
    ) -> None:
        double = MagicMock()
        double.role_name = "x"
        scenario = _SimpleScenario(_make_ego_config())
        getattr(scenario, method)(double)

        assert find_entity_by_role_name("x") is double
        assert find_vehicle_entity("x") is double
        assert find_pedestrian_entity("x") is double
