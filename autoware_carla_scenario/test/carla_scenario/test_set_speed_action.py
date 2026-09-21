"""The set-speed action: what reaches the entity, and when the action is over.

The entity is faked and registered, so these tests are about the action's own
decisions -- what it commands, how it interpolates a ramp, and how long it
stays running -- rather than about the TrafficManager.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import SetSpeedAction, SpeedTransition
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.entity.registry import (
    register_entity,
    unregister_entity,
)


class _RecordingVehicle:
    """An entity that records every speed it is told to hold."""

    def __init__(self, speed_ms: float = 0.0) -> None:
        self.commanded: list[float] = []
        self.actor = MagicMock()
        self.actor.get_velocity.return_value = carla.Vector3D(speed_ms, 0.0, 0.0)

    def set_speed(self, world: object, speed_kmh: float) -> None:
        self.commanded.append(speed_kmh)


@pytest.fixture
def vehicle() -> Iterator["_RecordingVehicle"]:
    """A vehicle registered as ``npc1`` for the duration of one test."""
    entity = _RecordingVehicle()
    register_entity("npc1", entity)
    yield entity
    unregister_entity("npc1")


def _drive(action: SetSpeedAction, world: object, until: float, step: float) -> None:
    """Tick *action* from 0 to *until* in increments of *step*."""
    elapsed = 0.0
    while elapsed <= until + 1e-9:
        action.tick(world, elapsed)
        elapsed += step


class TestStep:
    def test_the_target_is_commanded_once(self, vehicle: _RecordingVehicle) -> None:
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0)
        world = MagicMock()

        _drive(action, world, until=1.0, step=0.1)

        assert vehicle.commanded == [10.0]

    def test_a_step_is_complete_immediately(self, vehicle: _RecordingVehicle) -> None:
        """Nothing is left to wait for: obeying a target is not the action working."""
        action = SetSpeedAction(entity_name="npc1", target_speed_kmh=10.0)
        world = MagicMock()

        action.tick(world, 0.0)
        assert action.state is ActionState.START_TRANSITION
        action.tick(world, 0.1)
        assert action.state is ActionState.COMPLETE


class TestLinearRamp:
    def test_the_target_is_interpolated_from_the_current_speed(self) -> None:
        # 10 m/s == 36 km/h, ramping to 0 over 2 s.
        entity = _RecordingVehicle(speed_ms=10.0)
        register_entity("npc1", entity)
        try:
            action = SetSpeedAction(
                entity_name="npc1",
                target_speed_kmh=0.0,
                transition=SpeedTransition.LINEAR,
                duration=2.0,
            )
            _drive(action, MagicMock(), until=2.0, step=1.0)
        finally:
            unregister_entity("npc1")

        # Commanded at t=1.0 (half way) and t=2.0 (the end).  Nothing is
        # commanded on the tick the action fires: that tick is the start
        # transition, and the ramp has not advanced by then.
        assert len(entity.commanded) == 2
        assert math.isclose(entity.commanded[0], 18.0)
        assert math.isclose(entity.commanded[1], 0.0)

    def test_the_action_stays_running_for_the_whole_ramp(
        self, vehicle: _RecordingVehicle
    ) -> None:
        """So an `action_state` condition waits for the manoeuvre, not the command."""
        action = SetSpeedAction(
            entity_name="npc1",
            target_speed_kmh=50.0,
            transition=SpeedTransition.LINEAR,
            duration=2.0,
        )
        world = MagicMock()

        action.tick(world, 0.0)
        action.tick(world, 0.5)
        assert action.state is ActionState.RUNNING
        action.tick(world, 1.9)
        assert action.state is ActionState.RUNNING

        action.tick(world, 2.0)
        assert action.state is ActionState.END_TRANSITION
        action.tick(world, 2.1)
        assert action.state is ActionState.COMPLETE

    def test_the_ramp_ends_exactly_on_the_target(
        self, vehicle: _RecordingVehicle
    ) -> None:
        """No overshoot and no stopping short, whatever the tick alignment."""
        action = SetSpeedAction(
            entity_name="npc1",
            target_speed_kmh=40.0,
            transition=SpeedTransition.LINEAR,
            duration=1.0,
        )
        _drive(action, MagicMock(), until=1.3, step=0.3)

        assert vehicle.commanded[-1] == 40.0
        assert max(vehicle.commanded) <= 40.0


class TestMissingEntity:
    def test_an_unknown_entity_completes_rather_than_hanging(self) -> None:
        """A missing NPC must not look like a manoeuvre still under way."""
        action = SetSpeedAction(
            entity_name="nobody",
            target_speed_kmh=10.0,
            transition=SpeedTransition.LINEAR,
            duration=5.0,
        )
        world = MagicMock()

        action.tick(world, 0.0)
        action.tick(world, 0.1)
        assert action.state is ActionState.COMPLETE


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
    def test_a_linear_ramp_in_init_is_a_document_error(self) -> None:
        """`init` performs each action once, so a ramp would never advance."""
        from autoware_carla_scenario.authoring.models import (
            ActionNode,
            Entity,
            ScenarioDocument,
        )
        from autoware_carla_scenario.authoring.validator import validate_document

        document = ScenarioDocument(
            id="s",
            entities=[Entity(id="ego", kind="ego")],
            actions=[
                ActionNode(
                    id="a1",
                    type="set_speed",
                    actor="ego",
                    phase="init",
                    params={
                        "target_speed_kmh": 10.0,
                        "transition": "linear",
                        "duration": 2.0,
                    },
                )
            ],
        )
        assert any(
            "never advance" in issue.message
            for issue in validate_document(document).errors
        )

    def test_a_step_in_init_is_fine(self) -> None:
        """Setting a speed once, before anything moves, is what init is for."""
        from autoware_carla_scenario.authoring.models import (
            ActionNode,
            Entity,
            ScenarioDocument,
        )
        from autoware_carla_scenario.authoring.validator import validate_document

        document = ScenarioDocument(
            id="s",
            entities=[Entity(id="ego", kind="ego")],
            actions=[
                ActionNode(
                    id="a1",
                    type="set_speed",
                    actor="ego",
                    phase="init",
                    params={"target_speed_kmh": 10.0, "transition": "step"},
                )
            ],
        )
        assert not any(
            "never advance" in issue.message
            for issue in validate_document(document).errors
        )


class TestConstruction:
    def test_a_negative_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            SetSpeedAction(entity_name="npc1", target_speed_kmh=-1.0)

    def test_a_linear_transition_needs_a_duration(self) -> None:
        with pytest.raises(ValueError, match="positive duration"):
            SetSpeedAction(
                entity_name="npc1",
                target_speed_kmh=10.0,
                transition=SpeedTransition.LINEAR,
            )

    def test_a_step_with_a_duration_is_refused_rather_than_resolved(self) -> None:
        """Saying both is a mistake; picking one silently hides it."""
        with pytest.raises(ValueError, match="takes no duration"):
            SetSpeedAction(
                entity_name="npc1",
                target_speed_kmh=10.0,
                transition=SpeedTransition.STEP,
                duration=2.0,
            )
