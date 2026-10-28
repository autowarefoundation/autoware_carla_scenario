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
