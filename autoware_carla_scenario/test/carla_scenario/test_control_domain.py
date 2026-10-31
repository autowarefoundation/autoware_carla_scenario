"""Actions that command the same control domain of an entity conflict.

OpenSCENARIO's control domains (longitudinal, lateral, lighting, animation):
an action takes the ones it commands as it fires, and one fired to command a
domain another action holds raises :class:`ControlConflictError` instead of
commanding anything.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.base import BaseAction, control_holder
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult
from autoware_carla_scenario.control_domain import (
    MOVEMENT,
    ControlConflictError,
    ControlDomain,
)

LON = ControlDomain.LONGITUDINAL
LAT = ControlDomain.LATERAL


class _Switch(BaseCondition):
    """Satisfied while it is on."""

    def __init__(self, on: bool = False) -> None:
        super().__init__(label="switch")
        self.on = on

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        if not self.on:
            return None
        return ScenarioResult(passed=True, message="on", elapsed_seconds=elapsed)


class _Command(BaseAction):
    """An action commanding *domains* of *entity*; it runs on until *until*
    is on, if it has one."""

    def __init__(
        self,
        label: str,
        domains: tuple[ControlDomain, ...],
        entity: str = "npc",
        until: Optional[_Switch] = None,
        once: bool = True,
        overrides: bool = False,
    ) -> None:
        super().__init__(label=label, until=until, once=once, overrides=overrides)
        self.entity = entity
        self.domains = domains
        self.executed = 0

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        return [(self.entity, domain) for domain in self.domains]

    def execute(self, world: object) -> None:
        self.executed += 1


class TestActions:
    def test_a_conflict_names_both_actions_and_takes_nothing(self) -> None:
        _Command("lane", (LAT,), until=_Switch()).tick(None, 0.0)
        walk = _Command("walk", MOVEMENT)
        with pytest.raises(ControlConflictError) as raised:
            walk.tick(None, 1.0)
        assert (raised.value.entity, raised.value.domain) == ("npc", LAT)
        assert raised.value.holder == "_Command 'lane'"
        assert raised.value.claimant == "_Command 'walk'"
        assert control_holder("npc", LON, 1.0) is None

    def test_other_domains_and_entities_are_free(self) -> None:
        _Command("speed", (LON,), until=_Switch()).tick(None, 0.0)
        _Command("lane", (LAT,), until=_Switch()).tick(None, 0.0)
        _Command("other", (LON,), entity="other", until=_Switch()).tick(None, 0.0)
        assert control_holder("npc", LAT, 9.0) == "_Command 'lane'"
        assert control_holder("npc", ControlDomain.LIGHTING, 9.0) is None

    def test_a_running_action_blocks_another_on_its_domain(self) -> None:
        holding = _Command("speed", (LON,), until=_Switch())
        holding.tick(None, 0.0)
        assert holding.state is ActionState.START_TRANSITION
        second = _Command("speed2", (LON,))
        with pytest.raises(ControlConflictError, match="speed2.*'npc'.*speed"):
            second.tick(None, 1.0)
        assert second.executed == 0

    def test_movement_takes_both_domains(self) -> None:
        _Command("walk", MOVEMENT, until=_Switch()).tick(None, 0.0)
        with pytest.raises(ControlConflictError):
            _Command("lane", (LAT,)).tick(None, 1.0)

    def test_a_finished_run_gives_its_domains_back(self) -> None:
        switch = _Switch()
        holding = _Command("speed", (LON,), until=switch)
        holding.tick(None, 0.0)
        holding.tick(None, 0.05)
        switch.on = True
        holding.tick(None, 0.1)
        assert holding.state is ActionState.END_TRANSITION
        _Command("speed2", (LON,)).tick(None, 0.15)

    def test_two_commands_in_one_tick_conflict(self) -> None:
        _Command("lane", (LAT,)).tick(None, 2.0)
        with pytest.raises(ControlConflictError):
            _Command("turn", (LAT,)).tick(None, 2.0)

    def test_commands_on_successive_ticks_do_not(self) -> None:
        first = _Command("speed", (LON,))
        first.tick(None, 2.0)
        _Command("speed2", (LON,)).tick(None, 2.05)
        first.tick(None, 2.05)
        assert first.state is ActionState.COMPLETE

    def test_a_repeating_action_takes_its_domain_again(self) -> None:
        repeating = _Command("speed", (LON,), until=_Switch(on=True), once=False)
        for i in range(6):
            repeating.tick(None, i * 0.05)
        assert repeating.executed >= 2

    def test_closing_an_action_gives_its_domains_back(self) -> None:
        holding = _Command("speed", (LON,), until=_Switch())
        holding.tick(None, 0.0)
        holding.close()
        assert control_holder("npc", LON, 1.0) is None

    def test_an_action_on_no_entity_takes_nothing(self) -> None:
        _Command("weather", ()).tick(None, 0.0)
        _Command("weather2", ()).tick(None, 0.0)


class TestOverride:
    def test_an_overriding_action_stops_the_holder(self) -> None:
        holding = _Command("walk", MOVEMENT, until=_Switch())
        holding.tick(None, 0.0)
        holding.tick(None, 0.05)
        assert holding.state is ActionState.RUNNING
        taking = _Command("speed", (LON,), until=_Switch(), overrides=True)
        taking.tick(None, 0.1)
        assert taking.executed == 1
        assert holding.state is ActionState.COMPLETE
        assert control_holder("npc", LON, 0.1) == "_Command 'speed'"
        # The stopped action gave its lateral domain back too.
        _Command("lane", (LAT,)).tick(None, 0.15)
        # And is not run again.
        holding.tick(None, 0.2)
        assert holding.executed == 1

    def test_an_overriding_action_takes_a_domain_commanded_this_tick(self) -> None:
        _Command("speed", (LON,)).tick(None, 1.0)
        _Command("speed2", (LON,), overrides=True).tick(None, 1.0)

    def test_an_action_held_over_still_refuses_one_that_does_not_override(
        self,
    ) -> None:
        _Command("speed", (LON,), until=_Switch(), overrides=True).tick(None, 0.0)
        with pytest.raises(ControlConflictError):
            _Command("speed2", (LON,)).tick(None, 0.05)


class _Recording(_Command):
    """A :class:`_Command` that remembers being stopped by an override."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stopped_in: Optional[object] = None

    def _stopped(self, world: Any) -> None:
        self.stopped_in = world


class TestRunsThatHandOff:
    def test_an_overridden_action_undoes_what_its_run_left(self) -> None:
        world = object()
        holding = _Recording("cut_in", MOVEMENT, until=_Switch())
        holding.tick(world, 0.0)
        _Command("brake", (LON,), until=_Switch(), overrides=True).tick(world, 0.5)
        assert holding.stopped_in is world

    def test_an_init_action_left_running_controls_nothing(self) -> None:
        """Nothing ticks an init action again: a route handed to the ego's
        driver there must not keep the ego's lateral domain for the run."""
        route = _Command("route", (LAT,), until=_Switch())
        route.tick(None, 0.0)
        route._settle_start(None, 0.0)
        assert route.state is ActionState.RUNNING
        assert control_holder("npc", LAT, 0.0) is None
        _Command("turn", (LAT,)).tick(None, 0.0)

    def test_an_adversity_s_own_path_takes_no_domain(self) -> None:
        from autoware_carla_scenario.actions._adversarial_execution import (
            AdversarialExecution,
        )
        from autoware_carla_scenario.adversity.geometry import PredictedState

        path = [
            PredictedState(x=0.0, y=0.0, yaw=0.0, speed=1.0, t=0.0),
            PredictedState(x=1.0, y=0.0, yaw=0.0, speed=1.0, t=1.0),
        ]
        execution = AdversarialExecution("npc", path)
        assert execution._follow._claims_control is False
