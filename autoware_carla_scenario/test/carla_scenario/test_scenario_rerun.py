"""A retry, or a re-run, of the same scenario object starts from a clean slate.

``ScenarioQueue`` retries a scenario whose run raised on the same object, and
``run_all()`` may be called again on the same scenarios; either way
``ScenarioRunner.run_scenario`` calls ``setup()`` on it once more.  What the
earlier run registered -- its conditions (a latched one among them), actions,
callbacks, entities, coverage and measures -- must be gone by then, and what
``__init__`` registered must still be there, once (#661).
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseAction,
    BaseCondition,
    BaseScenario,
    EgoConfig,
    ScenarioQueue,
    ScenarioResult,
    ScenarioRunner,
    SpawnTransform,
    StickyCondition,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.environment import follow_tire_friction


def _ego_config() -> EgoConfig:
    return EgoConfig(
        spawn_location=SpawnTransform(carla.Transform(carla.Location(x=0, y=0, z=0))),
        vehicle_type="vehicle.mini.cooper",
    )


class _Never(BaseCondition):
    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        return None


class _Reached(BaseCondition):
    """Passes once armed: what the test arms stands for the world getting there."""

    def __init__(self) -> None:
        super().__init__(label="reached")
        self.armed = False

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if not self.armed:
            return None
        return ScenarioResult(passed=True, message="reached", elapsed_seconds=elapsed)


class _Noop(BaseAction):
    def __init__(self, label: str, *, grip: bool = False) -> None:
        super().__init__(label)
        self._grip = grip

    @property
    def sets_tire_friction(self) -> bool:
        return self._grip

    def execute(self, world: Any) -> None:
        pass


class _Npc:
    """A stand-in for a spawned NPC: all ``register_entity`` reads of it."""

    actor = None
    initial_speed_kmh = 0.0

    def __init__(self, role_name: str) -> None:
        self.role_name = role_name

    def set_traffic_backend(self, backend: Any) -> None:
        pass

    def set_client(self, client: Any, tm_port: int) -> None:
        pass


def _noop(world: Any) -> None:
    pass


class _Registers(BaseScenario):
    """Registers one of everything in ``setup()``, and a few in ``__init__``."""

    STABILIZE_TICKS = 0

    reached: _Reached

    def __init__(self, ego_entity: Any = None) -> None:
        super().__init__(_ego_config(), ego_entity=ego_entity)
        self.register_fail_condition(_Never(label="from_init"))
        self.register_post_tick(_Noop("from_init"))
        self.register_measure("from_init", lambda w: 1.0)
        self.register_cover("from_init", lambda w: 0.0, values=[False, True])

    def setup(self) -> None:
        self.reached = _Reached()
        self.register_pass_condition(StickyCondition(self.reached, label="reached"))
        self.register_fail_condition(_Never(label="collided"))
        self.register_init(_Noop("init_action"))
        self.register_init(_noop)
        self.register_pre_tick(_Noop("pre_action", grip=True))
        self.register_pre_tick(_noop)
        self.register_post_tick(_Noop("post_action"))
        self.register_post_tick(_noop)
        self.register_entity(_Npc("npc1"))  # type: ignore[arg-type]
        self.register_measure("from_setup", lambda w: 2.0)
        self.register_cover("speed", lambda w: 0.0, range=(0, 10), every=5)
        self.register_cover("braking", lambda w: False, values=[False, True])
        self.register_cross("speed_x_braking", ["speed", "braking"])

    def is_done(self) -> bool:
        return False


def _registered(scenario: BaseScenario) -> dict[str, Any]:
    """Everything a run registers on *scenario*, by label where it has one."""
    return {
        "pass": [c.label for c in scenario._pass_conditions],
        "fail": [c.label for c in scenario._fail_conditions],
        "init_actions": [a.label for a in scenario._init_actions],
        "pre_tick_actions": [a.label for a in scenario._pre_tick_actions],
        "post_tick_actions": [a.label for a in scenario._post_tick_actions],
        "init_callbacks": len(scenario._init_callbacks),
        "pre_tick_callbacks": len(scenario._pre_tick_callbacks),
        "post_tick_callbacks": len(scenario._post_tick_callbacks),
        "tire_friction": scenario._post_tick_callbacks.count(follow_tire_friction),
        "entities": [e.role_name for e in scenario._entities],
        "cover": [i.name for i in scenario._cover_items],
        "cross": [c.name for c in scenario._cross_items],
        "measures": sorted(scenario._measures),
    }


def _attempt(scenario: BaseScenario) -> None:
    """What ``run_scenario`` does to the scenario before and in ``setup()``."""
    scenario._reset_registrations()
    scenario.setup()


# ---------------------------------------------------------------------------
# BaseScenario
# ---------------------------------------------------------------------------


class TestResetRegistrations:
    def test_a_second_attempt_registers_exactly_what_the_first_did(self) -> None:
        scenario = _Registers()
        _attempt(scenario)
        first = _registered(scenario)

        _attempt(scenario)

        assert _registered(scenario) == first

    def test_one_copy_of_each_registration(self) -> None:
        scenario = _Registers()
        _attempt(scenario)
        _attempt(scenario)
        _attempt(scenario)

        registered = _registered(scenario)
        assert registered["pass"] == ["reached"]
        assert registered["fail"] == ["from_init", "collided"]
        assert registered["init_actions"] == ["init_action"]
        assert registered["pre_tick_actions"] == ["pre_action"]
        assert registered["post_tick_actions"] == ["from_init", "post_action"]
        assert registered["init_callbacks"] == 1
        assert registered["pre_tick_callbacks"] == 1
        # The callback and follow_tire_friction, which the grip action brought.
        assert registered["post_tick_callbacks"] == 2
        assert registered["tire_friction"] == 1
        assert registered["entities"] == ["npc1"]
        assert registered["cover"] == ["from_init", "speed", "braking"]
        assert registered["cross"] == ["speed_x_braking"]

    def test_what_init_registered_stays(self) -> None:
        scenario = _Registers()
        _attempt(scenario)
        _attempt(scenario)

        assert "from_init" in scenario._measures
        assert [c.label for c in scenario._fail_conditions][0] == "from_init"
        assert scenario._post_tick_actions[0].label == "from_init"
        assert scenario._cover_items[0].name == "from_init"

    def test_the_first_attempt_keeps_everything_init_registered(self) -> None:
        scenario = _Registers()
        scenario._reset_registrations()

        assert [c.label for c in scenario._fail_conditions] == ["from_init"]
        assert [i.name for i in scenario._cover_items] == ["from_init"]
        assert "from_init" in scenario._measures

    def test_a_condition_that_latched_in_an_earlier_attempt_is_gone(self) -> None:
        """The issue's case: a sticky pass latched, then the attempt raised."""
        scenario = _Registers()
        _attempt(scenario)
        scenario.reached.armed = True
        assert scenario._pass_conditions[0].check(None, 3.0) is not None

        _attempt(scenario)  # the retry: its own _Reached is not armed

        assert [c.check(None, 0.05) for c in scenario._pass_conditions] == [None]

    def test_the_constructor_state_is_kept(self) -> None:
        ego = MagicMock()
        scenario = _Registers(ego_entity=ego)
        scenario.random_seed = 7
        scenario.goal_pose = None
        _attempt(scenario)
        _attempt(scenario)

        assert scenario.ego_entity is ego
        assert scenario.random_seed == 7
        assert scenario.ego_config.vehicle_type == "vehicle.mini.cooper"


# ---------------------------------------------------------------------------
# ScenarioRunner and ScenarioQueue
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Ends a run where its clock would start, once everything is registered."""


def _fake_ego() -> MagicMock:
    ego = MagicMock()
    ego.requires_goal = False
    ego.attaches_to_existing_actor = True
    ego.carried_actor_ids = frozenset()
    ego.is_initialized = True
    ego.termination_requested = False
    ego.use_autopilot = True
    return ego


def _runner(tmp_path: Any) -> ScenarioRunner:
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), max_tick_rate_hz=None)
    runner.output_dir = tmp_path
    runner._world = MagicMock()
    return runner


def _seen_at_the_clock(
    scenario: BaseScenario, error: Exception
) -> tuple[list[dict[str, Any]], Any]:
    """A ``_ScenarioClock`` stand-in that records what *scenario* has
    registered when its run's clock would start, then raises *error*."""
    seen: list[dict[str, Any]] = []

    def clock(world: Any) -> None:
        seen.append(_registered(scenario))
        raise error

    return seen, clock


class TestRunScenarioStartsFromACleanSlate:
    def test_running_the_same_scenario_twice(self, tmp_path: Any) -> None:
        runner = _runner(tmp_path)
        scenario = _Registers(ego_entity=_fake_ego())
        seen, clock = _seen_at_the_clock(scenario, _Stop("clock"))

        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch("autoware_carla_scenario.scenario_runner._ScenarioClock", clock),
        ):
            for _ in range(2):
                with pytest.raises(_Stop):
                    runner.run_scenario(
                        scenario, timeout_seconds=12.0, reset_world=False
                    )

        assert len(seen) == 2
        assert seen[1] == seen[0]
        # The runner's own, again on every run: after setup()'s, once each.
        assert seen[1]["fail"] == [
            "from_init",
            "collided",
            "ego_existence",
            "default_timeout",
        ]

    def test_entry_speeds_held_before_the_run_are_forgotten(
        self, tmp_path: Any
    ) -> None:
        """A spawn card ticked outside any run holds an entry speed for an
        actor id; the next run starts without it (#697), and without a
        warm-up flag left set."""
        from autoware_carla_scenario.entity import vehicle_entity

        vehicle_entity.hold_entry_speed(17, 5.0)
        vehicle_entity._set_warmup_done(True)
        runner = _runner(tmp_path)
        scenario = _Registers(ego_entity=_fake_ego())
        held: list[dict[int, float]] = []

        def clock(world: Any) -> None:
            held.append(vehicle_entity.entry_speeds())
            raise _Stop("clock")

        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch("autoware_carla_scenario.scenario_runner._ScenarioClock", clock),
        ):
            # A MagicMock world hands set_initial_speed a mock actor for id
            # 17, which carla.Vector3D refuses: had the speed been kept, the
            # run would raise TypeError before its clock.
            with pytest.raises(_Stop):
                runner.run_scenario(scenario, timeout_seconds=12.0, reset_world=False)

        assert held == [{}]

    def test_the_ego_s_base_link_is_only_this_run_s(self, tmp_path: Any) -> None:
        """Where the last run's ego had its base_link is gone before this run's
        ego starts, and this run's is gone when the run ends -- even when the
        ego's on_scenario_end does not clear it (#739)."""
        from autoware_carla_scenario.ego_base_link import (
            ego_base_link,
            set_ego_base_link,
        )

        set_ego_base_link(-9.0)
        runner = _runner(tmp_path)
        ego = _fake_ego()
        before_start: list[Any] = []

        def on_scenario_start(world: Any) -> None:
            before_start.append(ego_base_link())
            set_ego_base_link(-1.4)

        ego.on_scenario_start.side_effect = on_scenario_start
        scenario = _Registers(ego_entity=ego)
        at_clock: list[Any] = []

        def clock(world: Any) -> None:
            at_clock.append(ego_base_link())
            raise _Stop("clock")

        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch("autoware_carla_scenario.scenario_runner._ScenarioClock", clock),
        ):
            with pytest.raises(_Stop):
                runner.run_scenario(scenario, timeout_seconds=12.0, reset_world=False)

        assert before_start == [None]
        assert at_clock == [-1.4]
        assert ego_base_link() is None

    def test_the_ego_s_base_link_is_gone_though_a_later_cleanup_step_raises(
        self, tmp_path: Any
    ) -> None:
        """The run forgets its ego's base_link right after the ego's teardown,
        so a cleanup step after it that raises does not leave it set (#739)."""
        from autoware_carla_scenario.ego_base_link import (
            ego_base_link,
            set_ego_base_link,
        )

        runner = _runner(tmp_path)
        ego = _fake_ego()
        ego.on_scenario_start.side_effect = lambda world: set_ego_base_link(-1.4)
        scenario = _Registers(ego_entity=ego)

        def clock(world: Any) -> None:
            raise _Stop("clock")

        def clear_props() -> None:
            if ego.on_scenario_end.called:
                raise _Stop("cleanup")

        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch("autoware_carla_scenario.scenario_runner._ScenarioClock", clock),
            patch("autoware_carla_scenario.scenario_runner.clear_props", clear_props),
        ):
            with pytest.raises(_Stop, match="cleanup"):
                runner.run_scenario(scenario, timeout_seconds=12.0, reset_world=False)

        ego.on_scenario_end.assert_called_once()
        assert ego_base_link() is None

    def test_a_queue_retry_sees_only_its_own_registrations(self, tmp_path: Any) -> None:
        """The first attempt raises after setup(); the retry runs the same
        object and sees one copy of each registration."""
        queue = ScenarioQueue(server=MagicMock(), cooldown_max_retries=1)
        queue._runner = _runner(tmp_path)
        scenario = _Registers(ego_entity=_fake_ego())
        seen, clock = _seen_at_the_clock(scenario, RuntimeError("CARLA went away"))
        queue.add(scenario, timeout_seconds=12.0)

        with (
            patch("autoware_carla_scenario.scenario_runner.prepare_measures"),
            patch("autoware_carla_scenario.scenario_runner._ScenarioClock", clock),
            patch.object(ScenarioRunner, "reset_world"),
        ):
            results = queue.run_all(continue_on_error=True)

        assert len(seen) == 2  # both attempts got as far as the clock
        assert seen[1] == seen[0]
        assert seen[1]["pass"] == ["reached"]
        assert seen[1]["entities"] == ["npc1"]
        assert results[0].passed is False  # errored after every attempt


# ---------------------------------------------------------------------------
# What __init__ registered starts every run as it was built (#928)
# ---------------------------------------------------------------------------


class _Scripted(BaseCondition):
    """Passes while *holds* says so at the run's time: the world's script."""

    def __init__(self, label: str, holds: Any) -> None:
        super().__init__(label=label)
        self._holds = holds

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if not self._holds(elapsed):
            return None
        return ScenarioResult(passed=True, message=self.label, elapsed_seconds=elapsed)


class _Counted(BaseAction):
    """A ``once`` action counting the times it was executed."""

    def __init__(self) -> None:
        super().__init__("counted", condition=_Scripted("at_1s", lambda t: t >= 1.0))
        self.executions = 0

    def execute(self, world: Any) -> None:
        self.executions += 1


class _Sensor:
    """A collision sensor's client handle."""

    def __init__(self) -> None:
        self.listener: Any = None
        self.destroyed = False

    def listen(self, callback: Any) -> None:
        self.listener = callback

    def stop(self) -> None:
        self.listener = None

    def destroy(self) -> bool:
        self.destroyed = True
        return True


class _World:
    """The run's world: an ego, and the collision sensors attached to it."""

    def __init__(self) -> None:
        from autoware_carla_scenario.constants import EGO_ROLE_NAME

        self.ego = MagicMock()
        self.ego.attributes = {"role_name": str(EGO_ROLE_NAME)}
        self.sensors: list[_Sensor] = []

    def get_actors(self) -> MagicMock:
        actors = MagicMock()
        actors.filter.return_value = [self.ego]
        return actors

    def get_blueprint_library(self) -> MagicMock:
        return MagicMock()

    def spawn_actor(self, blueprint: Any, transform: Any, attach_to: Any) -> _Sensor:
        sensor = _Sensor()
        self.sensors.append(sensor)
        return sensor

    def collide(self) -> None:
        """The ego hits a vehicle: every listening sensor hears it."""
        from types import SimpleNamespace

        event = SimpleNamespace(
            normal_impulse=SimpleNamespace(x=500.0, y=0.0, z=0.0),
            other_actor=SimpleNamespace(type_id="vehicle.audi.tt", attributes={}),
        )
        for sensor in self.sensors:
            if sensor.listener is not None:
                sensor.listener(event)


class _RegistersInInit(BaseScenario):
    """Registers its conditions and a ``once`` action in ``__init__`` only."""

    STABILIZE_TICKS = 0

    def __init__(self) -> None:
        from autoware_carla_scenario import (
            CollisionCondition,
            PersistentCondition,
            PlaceClearCondition,
        )
        from autoware_carla_scenario.trajectory.model import RouteLanePose

        super().__init__(_ego_config())
        # Visits its goal from 0.5 s to 1 s, and stops for good at 2 s.
        self.sticky = StickyCondition(
            _Scripted("at_goal", lambda t: 0.5 <= t < 1.0), label="visited_goal"
        )
        self.persistent = PersistentCondition(
            _Scripted("stopped", lambda t: t >= 2.0), duration=1.0, label="stopped"
        )
        self.collision = CollisionCondition(label="collided")
        self.place = PlaceClearCondition(
            RouteLanePose(ds=5.0, anchor="start"), 4.0, 2.0, label="clear"
        )
        self.action = _Counted()
        self.register_pass_condition(self.sticky)
        self.register_pass_condition(self.persistent)
        self.register_pass_condition(self.place)
        self.register_fail_condition(self.collision)
        self.register_post_tick(self.action)

    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return False


def _run(scenario: _RegistersInInit, world: _World) -> dict[str, Any]:
    """One run as the runner drives it: reset, setup, the ticks (the ego
    collides at 3.5 s), the conditions closed.  What it saw, by condition."""
    first: dict[str, float] = {}
    scenario._reset_registrations()
    scenario.setup()
    for step in range(1, 101):
        t = step * 0.05
        if abs(t - 3.5) < 1e-9:
            world.collide()
        for action in scenario._post_tick_actions:
            action.tick(world, t)  # type: ignore[arg-type]
        for condition in scenario._pass_conditions + scenario._fail_conditions:
            result = condition.check(world, t)  # type: ignore[arg-type]
            if result is not None and condition.label not in first:
                first[condition.label] = round(result.elapsed_seconds, 2)
    scenario.close_conditions()
    return {
        "first": first,
        "executions": scenario.action.executions,
        "state": scenario.action.state,
        "sensors": [(s.destroyed, s.listener is None) for s in world.sensors],
    }


class TestInitRegistrationsStartEveryRunAsBuilt:
    def test_a_second_run_behaves_like_the_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sticky, Persistent, Collision, PlaceClear and a once action
        registered in __init__: the second run of the same object sees what
        the first saw, each of them deciding it afresh (#928)."""
        from autoware_carla_scenario.conditions import place_clear as place_module
        from autoware_carla_scenario.trajectory import resolve as resolve_module

        # Each run's route puts the place somewhere of its own.
        route_start = {"x": 10.0}
        monkeypatch.setattr(
            resolve_module,
            "resolve_position",
            lambda position, reference=None: (route_start["x"], 0.0, 0.0, 0.0),
        )
        looked_at: list[float] = []

        def occupant(world: Any, x: float, *rest: float) -> int:
            looked_at.append(x)
            return -1

        monkeypatch.setattr(place_module, "_occupant", occupant)
        scenario = _RegistersInInit()

        first = _run(scenario, _World())
        executions = scenario.action.executions
        route_start["x"] = 20.0
        looked_at.clear()
        second = _run(scenario, _World())

        assert first["first"] == {
            "visited_goal": 0.5,
            "stopped": 3.0,
            "clear": 0.05,
            "collided": 3.5,
        }
        assert second["first"] == first["first"]
        assert executions == 1
        assert second["executions"] == 2  # once in each run
        assert second["state"] is first["state"]
        # A sensor of its own in each run, released at its end.
        assert first["sensors"] == [(True, True)]
        assert second["sensors"] == [(True, True)]
        assert set(looked_at) == {20.0}  # placed on the second run's route

    def test_a_retry_after_a_latched_attempt_is_not_decided_by_it(self) -> None:
        """The first attempt latched the sticky condition, collided and fired
        the action, then raised: the retry starts with none of it."""
        scenario = _RegistersInInit()
        world = _World()
        scenario._reset_registrations()
        scenario.setup()
        for t in (0.5, 1.0, 2.0, 3.0):
            scenario.action.tick(world, t)  # type: ignore[arg-type]
            for condition in scenario._pass_conditions:
                condition.check(world, t)  # type: ignore[arg-type]
        scenario.collision.check(world, 3.0)  # type: ignore[arg-type]
        world.collide()
        assert scenario.collision.check(world, 3.05) is not None  # type: ignore[arg-type]
        scenario.close_conditions()

        scenario._reset_registrations()

        assert scenario.sticky.check(None, 0.05) is None  # type: ignore[arg-type]
        assert scenario.persistent.check(None, 2.5) is None  # type: ignore[arg-type]
        assert scenario.collision.check(world, 0.05) is None  # type: ignore[arg-type]
        assert scenario.action.state is ActionState.STANDBY
        assert not scenario.action.done
        assert scenario.sticky.to_summary_dict()["satisfied"] is False


class TestCollisionClosedWithinARun:
    def test_a_closed_condition_attaches_no_sensor_until_reset(self) -> None:
        """Closed, the condition is not checked again in that run -- it
        attaches no sensor -- and reset() opens it for the next one, which
        attaches its own (#928)."""
        from autoware_carla_scenario import CollisionCondition

        world = _World()
        condition = CollisionCondition(label="collided")
        condition.check(world, 0.0)  # type: ignore[arg-type]
        world.collide()
        latched = condition.check(world, 0.05)  # type: ignore[arg-type]
        assert latched is not None
        condition.close()

        for step in range(1, 60):
            condition.check(world, 0.05 * step)  # type: ignore[arg-type]
        assert len(world.sensors) == 1
        # What the run saw stays for its report.
        assert condition.get_details()["collision_elapsed_seconds"] == 0.05

        condition.reset()

        assert condition.check(world, 0.0) is None  # type: ignore[arg-type]
        assert len(world.sensors) == 2
        assert world.sensors[1].listener is not None
        assert "other_actor_type" not in condition.get_details()


class TestActionsStartEveryRunAsBuilt:
    def test_a_spawn_enters_at_its_given_speed_again(self) -> None:
        from autoware_carla_scenario.actions.spawn import SpawnAction

        action = SpawnAction(entity_name="npc1", speed=8.0, label="spawn")
        action._speed = 3.0  # slowed behind a road user of the earlier run
        action.reset()
        assert action._speed == 8.0

    def test_nade_samples_each_run_from_a_weight_of_one(self) -> None:
        from autoware_carla_scenario.actions.adversarial_traffic import (
            AdversarialTrafficAction,
        )

        action = AdversarialTrafficAction([], seed=5)
        action._sampler.weight = 0.25
        action._sampler._sampled_at = 30.0
        action.reset()
        assert action.weight == 1.0
        assert not action._sampler.cooling(0.0)

    def test_a_route_starts_from_its_given_offset_again(self) -> None:
        from autoware_carla_scenario.actions.routing import RoutingAction
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        action = RoutingAction("npc1", goal=Lanelet2Pose(1, 5.0), label="route")
        action._initial_distance_offset = 12.0  # where the last run's spawn stood
        action._begun = True
        action._released = True
        action.reset()
        assert action._initial_distance_offset == 0.0
        assert not action._begun
        assert not action._released
