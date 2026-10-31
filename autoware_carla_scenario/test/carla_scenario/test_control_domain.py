"""Actions that command the same control domain of an entity conflict.

OpenSCENARIO's control domains (longitudinal, lateral, lighting, animation):
an action takes the ones it commands as it fires, and one fired to command a
domain another action holds raises :class:`ControlConflictError` instead of
commanding anything.
"""

from __future__ import annotations

from typing import Optional

import pytest

from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.actions.base import BaseAction
from autoware_carla_scenario.conditions.base import BaseCondition, ScenarioResult
from autoware_carla_scenario.control_domain import (
    MOVEMENT,
    ControlConflictError,
    ControlDomain,
    ControlSlots,
    control_slots,
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
    ) -> None:
        super().__init__(label=label, until=until, once=once)
        self.entity = entity
        self.domains = domains
        self.executed = 0

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        return [(self.entity, domain) for domain in self.domains]

    def execute(self, world: object) -> None:
        self.executed += 1


class TestSlots:
    def test_a_running_hold_blocks_until_released(self) -> None:
        slots = ControlSlots()
        owner, _ = slots.take("a", [("npc", LON)], 0.0, runs=True)
        with pytest.raises(ControlConflictError) as raised:
            slots.take("b", [("npc", LON)], 5.0, runs=False)
        assert (raised.value.entity, raised.value.domain) == ("npc", LON)
        assert (raised.value.holder, raised.value.claimant) == ("a", "b")
        slots.release(owner)
        slots.take("b", [("npc", LON)], 5.0, runs=False)

    def test_a_command_holds_its_tick_alone(self) -> None:
        slots = ControlSlots()
        slots.take("a", [("npc", LON)], 1.0, runs=False)
        with pytest.raises(ControlConflictError):
            slots.take("b", [("npc", LON)], 1.0, runs=False)
        slots.take("b", [("npc", LON)], 1.05, runs=False)

    def test_other_domains_and_entities_are_free(self) -> None:
        slots = ControlSlots()
        slots.take("a", [("npc", LON)], 0.0, runs=True)
        slots.take("b", [("npc", LAT)], 0.0, runs=True)
        slots.take("c", [("other", LON)], 0.0, runs=True)
        assert slots.holder("npc", LAT, 9.0) == "b"
        assert slots.holder("npc", ControlDomain.LIGHTING, 9.0) is None

    def test_a_refused_claim_takes_nothing(self) -> None:
        slots = ControlSlots()
        slots.take("a", [("npc", LAT)], 0.0, runs=True)
        with pytest.raises(ControlConflictError):
            slots.take("b", [("npc", d) for d in MOVEMENT], 0.0, runs=True)
        assert slots.holder("npc", LON, 0.0) is None

    def test_an_override_releases_everything_the_holder_held(self) -> None:
        slots = ControlSlots()
        held, _ = slots.take("walk", [("npc", d) for d in MOVEMENT], 0.0, runs=True)
        owner, overridden = slots.take(
            "speed", [("npc", LON)], 1.0, runs=True, overrides=True
        )
        assert overridden == [held]
        assert slots.holder("npc", LON, 1.0) == "speed"
        assert slots.holder("npc", LAT, 1.0) is None


def _Overriding(*args: object, **kwargs: object) -> _Command:  # noqa: N802
    """A :class:`_Command` that overrides."""
    command = _Command(*args, **kwargs)  # type: ignore[arg-type]
    command.overrides = True
    return command


class TestActions:
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
        assert control_slots().holder("npc", LON, 1.0) is None

    def test_an_action_on_no_entity_takes_nothing(self) -> None:
        _Command("weather", ()).tick(None, 0.0)
        _Command("weather2", ()).tick(None, 0.0)


class TestOverride:
    def test_an_overriding_action_stops_the_holder(self) -> None:
        holding = _Command("walk", MOVEMENT, until=_Switch())
        holding.tick(None, 0.0)
        holding.tick(None, 0.05)
        assert holding.state is ActionState.RUNNING
        taking = _Overriding("speed", (LON,), until=_Switch())
        taking.tick(None, 0.1)
        assert taking.executed == 1
        assert holding.state is ActionState.COMPLETE
        assert control_slots().holder("npc", LON, 0.1) == "_Command 'speed'"
        # The stopped action gave its lateral domain back too.
        _Command("lane", (LAT,)).tick(None, 0.15)
        # And is not run again.
        holding.tick(None, 0.2)
        assert holding.executed == 1

    def test_an_overriding_action_takes_a_domain_commanded_this_tick(self) -> None:
        _Command("speed", (LON,)).tick(None, 1.0)
        _Overriding("speed2", (LON,)).tick(None, 1.0)

    def test_an_action_held_over_still_refuses_one_that_does_not_override(
        self,
    ) -> None:
        _Overriding("speed", (LON,), until=_Switch()).tick(None, 0.0)
        with pytest.raises(ControlConflictError):
            _Command("speed2", (LON,)).tick(None, 0.05)
