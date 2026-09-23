"""The set-speed action: what reaches the entity, and when the run is over.

The entity is faked and registered, so these tests are about the action's own
decisions -- what it commands, how a rate is walked, and what ends the run --
rather than about the TrafficManager.

The rate tests are all about one property: the commanded target is stepped from
the vehicle's **actual** speed, so it can never run ahead of what the vehicle
achieved.  A vehicle that is not keeping up is therefore not chased by a target
it never reached; it simply takes longer.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import SetSpeedAction
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.set_speed import ARRIVAL_TOLERANCE_KMH
from autoware_carla_scenario.entity.registry import (
    register_entity,
    unregister_entity,
)

if TYPE_CHECKING:
    from autoware_carla_scenario.authoring.models import ScenarioDocument

_KMH_PER_MS = 3.6


class _RecordingVehicle:
    """An entity that records every speed it is told to hold.

    Its actor is what both the action and the arrival condition read, so the
    speed set here is the one the closed loop sees.
    """

    def __init__(self, speed_kmh: float = 0.0, *, role_name: str = "npc1") -> None:
        self.commanded: list[float] = []
        self.speed_kmh = speed_kmh
        self.actor = MagicMock()
        self.actor.attributes = {"role_name": role_name}
        self.actor.get_velocity.side_effect = lambda: carla.Vector3D(
            self.speed_kmh / _KMH_PER_MS, 0.0, 0.0
        )

    def set_speed(self, world: object, speed_kmh: float) -> None:
        self.commanded.append(speed_kmh)


class _FakeWorld:
    """A world that knows how long a tick took and which actors are in it."""

    def __init__(
        self, *vehicles: _RecordingVehicle, delta_seconds: float = 0.1
    ) -> None:
        self._vehicles = vehicles
        self._delta_seconds = delta_seconds

    def get_snapshot(self) -> MagicMock:
        snapshot = MagicMock()
        snapshot.timestamp.delta_seconds = self._delta_seconds
        return snapshot

    def get_actors(self) -> list[MagicMock]:
        return [vehicle.actor for vehicle in self._vehicles]


@pytest.fixture
def vehicle() -> Iterator["_RecordingVehicle"]:
    """A vehicle registered as ``npc1`` for the duration of one test."""
    entity = _RecordingVehicle()
    register_entity("npc1", entity)
    yield entity
    unregister_entity("npc1")


def _drive(
    action: SetSpeedAction,
    world: object,
    ticks: int,
    *,
    follows: "_RecordingVehicle | None" = None,
    step_seconds: float = 0.1,
) -> None:
    """Tick *action* *ticks* times.

    When *follows* is given the vehicle is moved to whatever it was last
    commanded, which is the best case a rate limit can be asked to handle: a
    vehicle that tracks its target exactly.
    """
    elapsed = 0.0
    for _ in range(ticks):
        action.tick(world, elapsed)
        if follows is not None and follows.commanded:
            follows.speed_kmh = follows.commanded[-1]
        elapsed += step_seconds


class TestImmediateChange:
    def test_the_target_is_commanded_once(self, vehicle: _RecordingVehicle) -> None:
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0)

        _drive(action, _FakeWorld(vehicle), ticks=10)

        assert vehicle.commanded == [10.0]

    def test_nothing_is_re_sent(self) -> None:
        """`set_desired_speed` holds what it was given; repeating says it again."""
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0)

        assert action.reissues_while_running is False

    def test_it_is_complete_immediately(self, vehicle: _RecordingVehicle) -> None:
        """Nothing is left to wait for: handing over a target is not a manoeuvre."""
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0)
        world = _FakeWorld(vehicle)

        action.tick(world, 0.0)
        assert action.state is ActionState.START_TRANSITION
        action.tick(world, 0.1)
        assert action.state is ActionState.COMPLETE


class TestRateLimitedChange:
    def test_the_command_is_re_sent_every_tick(self) -> None:
        """A target that is walked has to be moved to move at all."""
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=50.0, rate_kmh_s=36.0
        )

        assert action.reissues_while_running is True

    def test_each_step_is_the_rate_times_the_tick(
        self, vehicle: _RecordingVehicle
    ) -> None:
        # 36 km/h/s over a 0.1 s tick is 3.6 km/h per tick.
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=50.0, rate_kmh_s=36.0
        )

        _drive(action, _FakeWorld(vehicle), ticks=4, follows=vehicle)

        assert [round(value, 6) for value in vehicle.commanded] == [
            3.6,
            7.2,
            10.8,
            14.4,
        ]

    def test_the_command_never_runs_ahead_of_the_vehicle(
        self, vehicle: _RecordingVehicle
    ) -> None:
        """A vehicle that cannot keep up is not chased by an open-loop target.

        The vehicle here never moves -- held up by traffic, a speed limit, or
        a TrafficManager that will not oblige -- so every step is taken from
        the same standstill and the command stays one step ahead rather than
        walking off to the target on its own.
        """
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=50.0, rate_kmh_s=36.0
        )

        _drive(action, _FakeWorld(vehicle), ticks=5)

        assert all(math.isclose(value, 3.6) for value in vehicle.commanded)

    def test_a_rate_slows_a_deceleration_too(self) -> None:
        # 10 m/s == 36 km/h, braking to a stop at 36 km/h/s over 0.1 s ticks.
        entity = _RecordingVehicle(speed_kmh=36.0)
        register_entity("npc1", entity)
        try:
            action = SetSpeedAction(
                entity_name="npc1", target_speed_kmh=0.0, rate_kmh_s=36.0
            )
            _drive(action, _FakeWorld(entity), ticks=3, follows=entity)
        finally:
            unregister_entity("npc1")

        assert [round(value, 6) for value in entity.commanded] == [32.4, 28.8, 25.2]

    def test_the_walk_ends_exactly_on_the_target(
        self, vehicle: _RecordingVehicle
    ) -> None:
        """No overshoot, whatever the tick alignment."""
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=10.0, rate_kmh_s=36.0
        )

        _drive(action, _FakeWorld(vehicle), ticks=8, follows=vehicle)

        assert max(vehicle.commanded) <= 10.0
        assert vehicle.commanded[-1] == 10.0


class TestWhenARateLimitedRunIsOver:
    """`completeState` means the vehicle got there, not that commands stopped."""

    def test_it_stays_running_until_the_vehicle_arrives(
        self, vehicle: _RecordingVehicle
    ) -> None:
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=50.0, rate_kmh_s=3.6
        )
        world = _FakeWorld(vehicle)

        _drive(action, world, ticks=5, follows=vehicle)

        assert action.state is ActionState.RUNNING

    def test_arriving_ends_the_run(self, vehicle: _RecordingVehicle) -> None:
        action = SetSpeedAction(
            entity_name="npc1", target_speed_kmh=10.0, rate_kmh_s=36.0
        )
        world = _FakeWorld(vehicle)

        _drive(action, world, ticks=10, follows=vehicle)

        assert vehicle.speed_kmh == 10.0
        assert action.state is ActionState.COMPLETE

    def test_close_enough_counts_as_arrived(self) -> None:
        """Whatever drives the vehicle sits near its target, not exactly on it."""
        entity = _RecordingVehicle(speed_kmh=10.0 - ARRIVAL_TOLERANCE_KMH / 2.0)
        register_entity("npc1", entity)
        try:
            action = SetSpeedAction(
                entity_name="npc1", target_speed_kmh=10.0, rate_kmh_s=36.0
            )
            # The vehicle is left where it is: nothing but the tolerance can
            # end this run.
            _drive(action, _FakeWorld(entity), ticks=4)
            state = action.state
        finally:
            unregister_entity("npc1")

        assert state is ActionState.COMPLETE

    def test_the_exact_target_is_the_last_value_commanded(self) -> None:
        """What the backend keeps must be what the scenario asked for.

        The backend holds the last target it was given forever, so a run that
        completes one step short leaves an intermediate value in force.  It
        bites whenever `rate_kmh_s * delta_seconds` is smaller than the arrival
        band -- here 3 km/h/s over a 0.05 s tick is 0.15 km/h against a 0.5
        km/h band -- and a brake to a stop is the case that shows it: the
        action would report `completeState` while the vehicle crawled on.
        """
        entity = _RecordingVehicle(speed_kmh=1.0)
        register_entity("npc1", entity)
        try:
            action = SetSpeedAction(
                entity_name="npc1", target_speed_kmh=0.0, rate_kmh_s=3.0
            )
            _drive(
                action,
                _FakeWorld(entity, delta_seconds=0.05),
                ticks=30,
                follows=entity,
                step_seconds=0.05,
            )
            state = action.state
        finally:
            unregister_entity("npc1")

        assert state is ActionState.COMPLETE
        assert entity.commanded[-1] == 0.0

    def test_a_given_until_replaces_the_arrival_condition(
        self, vehicle: _RecordingVehicle
    ) -> None:
        """The default is a default, so a scenario can say `for two seconds`."""
        from autoware_carla_scenario.conditions.elapsed_time import ElapsedTimeCondition

        action = SetSpeedAction(
            entity_name="npc1",
            target_speed_kmh=50.0,
            rate_kmh_s=3.6,
            until=ElapsedTimeCondition(duration_seconds=0.25, label="past_two_ticks"),
        )

        # The vehicle never gets near 50 km/h, so only the given `until` can
        # end this run.
        _drive(action, _FakeWorld(vehicle), ticks=6, follows=vehicle)

        assert action.state is ActionState.COMPLETE


class TestMissingEntity:
    def test_an_immediate_change_completes_rather_than_hanging(self) -> None:
        """A missing NPC must not look like a manoeuvre still under way."""
        action = SetSpeedAction(entity_name="nobody", target_speed_kmh=10.0)
        world = MagicMock()

        action.tick(world, 0.0)
        action.tick(world, 0.1)
        assert action.state is ActionState.COMPLETE

    def test_a_missing_entity_is_reported_rather_than_crashing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        action = SetSpeedAction(entity_name="nobody", target_speed_kmh=10.0)

        with caplog.at_level("WARNING"):
            action.tick(MagicMock(), 0.0)

        assert "not found" in caplog.text


class TestSelfDrivenEgos:
    """An ego the TrafficManager does not drive must refuse, not pretend."""

    @pytest.mark.parametrize("class_name", ["AutowareEgoEntity", "CarlaDriverEntity"])
    def test_the_class_overrides_set_speed(self, class_name: str) -> None:
        """Inherited, it would send the command to the TrafficManager.

        The runner puts these egos in `skip_actor_ids`, so the command would
        reach an actor the TrafficManager does not control -- while the action
        reported progress and completion, which is worse than refusing.

        Asserted on the class's own namespace rather than by calling it,
        because inheriting the method is exactly the defect: a call would
        succeed either way and only the TrafficManager would know.
        """
        import autoware_carla_scenario as acs

        entity_class = getattr(acs, class_name)
        assert "set_speed" in vars(entity_class), (
            f"{class_name} inherits set_speed from BackendDriven, so a speed "
            f"command would go to the TrafficManager"
        )

    def test_a_refusal_is_logged_rather_than_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario import CarlaDriverEntity

        with caplog.at_level("WARNING"):
            CarlaDriverEntity.set_speed(MagicMock(), MagicMock(), 30.0)
        assert "not driven by the TrafficManager" in caplog.text


class TestInitPhase:
    def _document(self, params: dict[str, object]) -> "ScenarioDocument":
        from autoware_carla_scenario.authoring.models import (
            ActionNode,
            Entity,
            ScenarioDocument,
        )

        return ScenarioDocument(
            id="s",
            entities=[Entity(id="ego", kind="ego")],
            actions=[
                ActionNode(
                    id="a1",
                    type="set_speed",
                    actor="ego",
                    phase="init",
                    params=params,
                )
            ],
        )

    def test_a_rate_in_init_is_a_document_error(self) -> None:
        """`init` performs each action once, so the walk would never advance."""
        from autoware_carla_scenario.authoring.validator import validate_document

        document = self._document({"target_speed_kmh": 10.0, "rate_kmh_s": 5.0})

        assert any(
            "never be walked to its target" in issue.message
            for issue in validate_document(document).errors
        )

    def test_an_immediate_change_in_init_is_fine(self) -> None:
        """Setting a speed once, before anything moves, is what init is for."""
        from autoware_carla_scenario.authoring.validator import validate_document

        document = self._document({"target_speed_kmh": 10.0, "rate_kmh_s": None})

        assert not any(
            "never be walked to its target" in issue.message
            for issue in validate_document(document).errors
        )

    def test_a_blank_rate_in_init_is_fine(self) -> None:
        """The editor writes an empty field as `""`, not as `None`."""
        from autoware_carla_scenario.authoring.validator import validate_document

        document = self._document({"target_speed_kmh": 10.0, "rate_kmh_s": ""})

        assert not any(
            "never be walked to its target" in issue.message
            for issue in validate_document(document).errors
        )


class TestConstruction:
    def test_a_negative_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            SetSpeedAction(entity_name="npc1", target_speed_kmh=-1.0)

    def test_a_zero_rate_is_refused_rather_than_never_arriving(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0, rate_kmh_s=0.0)

    def test_a_negative_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0, rate_kmh_s=-1.0)

    def test_a_rate_that_is_told_not_to_reissue_is_refused(self) -> None:
        """The seam carries a target and not a rate, so the walk must re-send.

        Accepting it would send one step and then wait forever for an arrival
        that nothing was still driving towards -- and no backend could take
        over, because `set_speed` is handed that intermediate value rather than
        the rate or the final target.
        """
        with pytest.raises(ValueError, match="must reissue"):
            SetSpeedAction(
                entity_name="npc1",
                target_speed_kmh=50.0,
                rate_kmh_s=10.0,
                reissue=False,
            )

    def test_an_immediate_change_may_still_be_told_to_reissue(self) -> None:
        """The injection point survives: only the impossible corner is closed.

        A backend that keeps nothing needs the same target sent again, and that
        is exactly what `reissue` is for.
        """
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=50.0, reissue=True)

        assert action.reissues_while_running is True
