"""The junction cycle: phases, the amber between them, and who waits on whom.

CARLA's lights are faked, because what is under test is not that ``set_state``
reaches a simulator.  It is the cycle -- that a phase holds for its declared
duration and then hands over, that a jump restarts the duration without
stopping the cycle, and that a controller offset from another one starts when
that one says so.  Those are the parts a scenario's timing actually rests on.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch

import carla
import pytest

from autoware_carla_scenario import (
    TrafficSignalControllerAction,
    TrafficSignalControllerCondition,
)
from autoware_carla_scenario.authoring.models import (
    SIGNAL_STATE_NAMES,
    SignalControllerRef,
    SignalPhaseRef,
    SignalStateRef,
)
from autoware_carla_scenario.signals import (
    STATE_NAMES,
    Phase,
    PhaseState,
    SignalController,
    build_controllers,
    clear_signal_controllers,
    register_signal_controller,
)

_GREEN = carla.TrafficLightState.Green
_YELLOW = carla.TrafficLightState.Yellow
_RED = carla.TrafficLightState.Red

_NORTH = 1001
"""Lanelet2 regulatory element id of the north approach."""

_EAST = 1002
"""...and of the east approach, which conflicts with it."""


def _light(actor_id: int) -> MagicMock:
    light = MagicMock()
    light.id = actor_id
    light.get_state.return_value = _RED
    return light


class _World:
    """Just enough CARLA world to hold a clock the test advances."""

    def __init__(self) -> None:
        self.now = 0.0

    def get_snapshot(self) -> MagicMock:
        snapshot = MagicMock()
        snapshot.timestamp.elapsed_seconds = self.now
        return snapshot


@pytest.fixture
def junction() -> Iterator[dict[int, MagicMock]]:
    """Two conflicting approaches, each resolving to one light."""
    lights = {_NORTH: _light(1), _EAST: _light(2)}

    def _find(world: object, lanelet2_id: int) -> list[MagicMock]:
        found = lights.get(lanelet2_id)
        return [found] if found is not None else []

    with patch(
        "autoware_carla_scenario.signals.controller"
        ".find_traffic_lights_for_lanelet2_id",
        _find,
    ):
        yield lights


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear_signal_controllers()
    yield
    clear_signal_controllers()


def _cycle(name: str = "crossing") -> SignalController:
    """North green 5s, north amber 2s, all red 1s, east green 5s."""
    return SignalController(
        name=name,
        phases=(
            Phase(
                "ns_green",
                5.0,
                (PhaseState(_NORTH, "green"), PhaseState(_EAST, "red")),
            ),
            Phase(
                "ns_amber",
                2.0,
                (PhaseState(_NORTH, "yellow"), PhaseState(_EAST, "red")),
            ),
            Phase(
                "all_red",
                1.0,
                (PhaseState(_NORTH, "red"), PhaseState(_EAST, "red")),
            ),
            Phase(
                "ew_green",
                5.0,
                (PhaseState(_NORTH, "red"), PhaseState(_EAST, "green")),
            ),
        ),
    )


def _run_to(controller: SignalController, world: _World, when: float) -> None:
    """Tick the controller at 0.5 s up to and including *when*."""
    while world.now <= when + 1e-9:
        controller.tick(world)
        world.now = round(world.now + 0.5, 6)


class TestTheCycle:
    def test_the_first_phase_shows_on_the_first_tick(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()
        controller.tick(world)

        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_green"
        junction[_NORTH].set_state.assert_called_with(_GREEN)
        junction[_EAST].set_state.assert_called_with(_RED)

    def test_a_phase_holds_for_its_declared_duration(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()

        _run_to(controller, world, 4.5)
        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_green"

    def test_amber_comes_between_the_two_greens(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """The interval the old single-signal design could not express at all.

        It is where an ego decides whether to stop or to go, so a cycle that
        jumped green straight to red would leave the decision untestable.
        """
        controller, world = _cycle(), _World()

        _run_to(controller, world, 5.0)
        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_amber"
        junction[_NORTH].set_state.assert_called_with(_YELLOW)
        # The conflicting approach stays red *through* the amber: an amber is
        # not a moment when nobody has the junction.
        junction[_EAST].set_state.assert_called_with(_RED)

    def test_the_cycle_returns_to_its_first_phase(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()

        # 5 + 2 + 1 + 5 = 13 seconds of cycle.
        _run_to(controller, world, 13.0)
        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_green"

    def test_every_phase_is_reached_in_order(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()
        seen: list[str] = []

        while world.now <= 13.0:
            controller.tick(world)
            phase = controller.current_phase
            assert phase is not None
            if not seen or seen[-1] != phase.name:
                seen.append(phase.name)
            world.now = round(world.now + 0.5, 6)

        assert seen == ["ns_green", "ns_amber", "all_red", "ew_green", "ns_green"]

    def test_a_phase_sets_every_signal_it_names_not_only_the_changed_one(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """What makes it a phase rather than a set of edits.

        Both lights are written on every change, so no earlier phase can leave
        one behind in a state the author never wrote.
        """
        controller, world = _cycle(), _World()

        _run_to(controller, world, 5.0)  # into the amber
        assert junction[_NORTH].set_state.call_count == 2
        assert junction[_EAST].set_state.call_count == 2

    def test_the_junction_is_frozen_so_carla_does_not_also_drive_it(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()
        controller.tick(world)

        for light in junction.values():
            light.freeze.assert_called_with(True)

    def test_a_zero_length_phase_is_passed_straight_through(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """An all-red clearance a scenario wants declared but not waited on."""
        controller = SignalController(
            name="c",
            phases=(
                Phase("clear", 0.0, (PhaseState(_NORTH, "red"),)),
                Phase("go", 5.0, (PhaseState(_NORTH, "green"),)),
            ),
        )
        world = _World()

        controller.tick(world)
        controller.tick(world)

        assert controller.current_phase is not None
        assert controller.current_phase.name == "go"


class TestJumpingToAPhase:
    def test_the_named_phase_shows_at_once(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()
        controller.tick(world)

        assert controller.change_phase_to("ew_green", world) is True
        assert controller.current_phase is not None
        assert controller.current_phase.name == "ew_green"
        junction[_EAST].set_state.assert_called_with(_GREEN)

    def test_the_jumped_to_phase_gets_its_whole_duration(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """Jumping restarts the clock rather than inheriting what was left."""
        controller, world = _cycle(), _World()
        _run_to(controller, world, 4.0)  # 1 s left of ns_green

        controller.change_phase_to("ns_green", world)
        _run_to(controller, world, 8.0)

        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_green"

    def test_the_cycle_carries_on_from_there(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """A junction forced green does not thereby stay green.

        Holding it is a second decision; this one only says where to start.
        """
        controller, world = _cycle(), _World()
        controller.tick(world)
        controller.change_phase_to("ns_green", world)

        _run_to(controller, world, 6.0)
        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_amber"

    def test_an_unknown_phase_is_refused_and_changes_nothing(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        controller, world = _cycle(), _World()
        controller.tick(world)

        with caplog.at_level("WARNING"):
            assert controller.change_phase_to("nonesuch", world) is False

        assert controller.current_phase is not None
        assert controller.current_phase.name == "ns_green"
        assert "no phase named 'nonesuch'" in caplog.text


class TestOffsetControllers:
    """A green wave: the corridor's junctions declared as offsets, not timed."""

    def test_a_referencing_controller_waits_out_its_delay(
        self, junction: dict[int, MagicMock]
    ) -> None:
        first = _cycle("first")
        second = SignalController(
            name="second",
            phases=(Phase("go", 5.0, (PhaseState(_EAST, "green"),)),),
            delay_seconds=3.0,
            reference=first,
        )
        world = _World()

        first.tick(world)
        second.tick(world)
        assert second.current_phase is None

        world.now = 2.5
        second.tick(world)
        assert second.current_phase is None

        world.now = 3.0
        second.tick(world)
        assert second.current_phase is not None

    def test_the_delay_is_measured_from_the_reference_not_from_the_run(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """Which is the whole point: the corridor keeps its shape.

        A first junction that itself started late carries the offset along, so
        the wave does not collapse onto the run's start.
        """
        first = _cycle("first")
        second = SignalController(
            name="second",
            phases=(Phase("go", 5.0, (PhaseState(_EAST, "green"),)),),
            delay_seconds=3.0,
            reference=first,
        )
        world = _World()

        world.now = 100.0
        first.tick(world)
        second.tick(world)
        assert second.current_phase is None

        world.now = 103.0
        second.tick(world)
        assert second.current_phase is not None
        assert second.started_at == 103.0

    def test_a_controller_with_no_reference_starts_with_the_run(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller, world = _cycle(), _World()
        world.now = 42.0
        controller.tick(world)

        assert controller.started_at == 42.0


class TestBuildingFromTheDocument:
    def test_the_declared_cycle_is_built_in_order(self) -> None:
        declared = [
            SignalControllerRef(
                name="crossing",
                phases=[
                    SignalPhaseRef(
                        name="ns_green",
                        duration_seconds=5.0,
                        states=[
                            SignalStateRef(
                                lanelet2_regulatory_element_id=_NORTH, state="green"
                            )
                        ],
                    ),
                    SignalPhaseRef(
                        name="ns_amber",
                        duration_seconds=2.0,
                        states=[
                            SignalStateRef(
                                lanelet2_regulatory_element_id=_NORTH, state="yellow"
                            )
                        ],
                    ),
                ],
            )
        ]

        (built,) = build_controllers(declared)

        assert built.name == "crossing"
        assert [p.name for p in built.phases] == ["ns_green", "ns_amber"]
        assert built.phases[1].duration_seconds == 2.0
        assert built.phases[0].states[0].state == "green"

    def test_a_reference_is_resolved_to_the_controller_it_names(self) -> None:
        declared = [
            SignalControllerRef(
                name="first",
                phases=[SignalPhaseRef(name="go", duration_seconds=5.0, states=[])],
            ),
            SignalControllerRef(
                name="second",
                phases=[SignalPhaseRef(name="go", duration_seconds=5.0, states=[])],
                delay_seconds=3.0,
                reference="first",
            ),
        ]

        first, second = build_controllers(declared)

        assert second.reference is first

    def test_an_unresolvable_reference_drops_the_offset_rather_than_the_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The validator rejects this; reaching here means something bypassed it.

        A run that loses one junction's offset is better than one that will not
        start at all.
        """
        declared = [
            SignalControllerRef(
                name="only",
                phases=[SignalPhaseRef(name="go", duration_seconds=5.0, states=[])],
                delay_seconds=3.0,
                reference="nonesuch",
            )
        ]

        with caplog.at_level("WARNING"):
            (built,) = build_controllers(declared)

        assert built.reference is None
        assert "which no controller on this map declares" in caplog.text

    def test_the_document_and_the_runtime_agree_on_the_state_names(self) -> None:
        """Two lists, kept apart on purpose, pinned to each other here.

        The authoring package must not import CARLA and the runtime must not
        import the authoring package, so the words exist twice; this is what
        stops them drifting.
        """
        assert set(SIGNAL_STATE_NAMES) == set(STATE_NAMES)

    def test_the_older_cards_spelling_of_a_colour_still_works(self) -> None:
        """``Green`` is how the single-signal card spells it.

        A document that mixes the two spellings should not mean two different
        things, so the phase table takes either.
        """
        declared = [
            SignalControllerRef(
                name="c",
                phases=[
                    SignalPhaseRef(
                        name="go",
                        duration_seconds=5.0,
                        states=[
                            SignalStateRef(
                                lanelet2_regulatory_element_id=_NORTH, state="Green"
                            )
                        ],
                    )
                ],
            )
        ]

        (built,) = build_controllers(declared)

        assert built.phases[0].states[0].state == "green"


class TestUnresolvedSignals:
    def test_a_signal_the_map_does_not_have_is_warned_about(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A phase naming a signal this map has not got is wrong every cycle."""
        controller = SignalController(
            name="c",
            phases=(Phase("go", 5.0, (PhaseState(9999, "green"),)),),
        )
        world = _World()

        with caplog.at_level("WARNING"):
            controller.tick(world)

        assert "does not resolve" in caplog.text
        assert "9999" in caplog.text

    def test_an_unknown_state_leaves_that_signal_alone(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        controller = SignalController(
            name="c",
            phases=(
                Phase(
                    "go",
                    5.0,
                    (PhaseState(_NORTH, "chartreuse"), PhaseState(_EAST, "red")),
                ),
            ),
        )
        world = _World()

        with caplog.at_level("WARNING"):
            controller.tick(world)

        junction[_NORTH].set_state.assert_not_called()
        junction[_EAST].set_state.assert_called_once_with(_RED)
        assert "unknown state" in caplog.text


class TestAction:
    def test_it_shows_the_named_phase(self, junction: dict[int, MagicMock]) -> None:
        controller = _cycle()
        register_signal_controller(controller)
        world = _World()
        controller.tick(world)

        TrafficSignalControllerAction(controller="crossing", phase="ew_green").execute(
            world
        )

        assert controller.current_phase is not None
        assert controller.current_phase.name == "ew_green"

    def test_an_unknown_controller_is_warned_about_and_touches_nothing(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        register_signal_controller(_cycle())

        with caplog.at_level("WARNING"):
            TrafficSignalControllerAction(
                controller="nonesuch", phase="ns_green"
            ).execute(_World())

        for light in junction.values():
            light.set_state.assert_not_called()
        assert "no controller named 'nonesuch'" in caplog.text


class TestCondition:
    def test_it_holds_while_the_named_phase_is_showing(
        self, junction: dict[int, MagicMock]
    ) -> None:
        controller = _cycle()
        register_signal_controller(controller)
        world = _World()
        controller.tick(world)

        condition = TrafficSignalControllerCondition(
            controller="crossing", phase="ns_green", label="ns"
        )
        result = condition.check(world, 1.0)

        assert result is not None
        assert result.passed

    def test_another_phase_is_not_it(self, junction: dict[int, MagicMock]) -> None:
        controller = _cycle()
        register_signal_controller(controller)
        world = _World()
        controller.tick(world)

        condition = TrafficSignalControllerCondition(
            controller="crossing", phase="ew_green", label="ew"
        )
        assert condition.check(world, 1.0) is None

    def test_the_amber_is_a_phase_that_can_be_waited_for(
        self, junction: dict[int, MagicMock]
    ) -> None:
        """The reason the pair exists: an interval with a name.

        A single-signal check sees a colour; this sees the junction's own
        account of what it is doing, which is what "the ego faced an amber"
        actually means.
        """
        controller = _cycle()
        register_signal_controller(controller)
        world = _World()
        condition = TrafficSignalControllerCondition(
            controller="crossing", phase="ns_amber", label="amber"
        )

        _run_to(controller, world, 4.5)
        assert condition.check(world, 4.5) is None

        _run_to(controller, world, 5.0)
        assert condition.check(world, 5.0) is not None

    def test_details_name_the_controller_and_the_phase(self) -> None:
        condition = TrafficSignalControllerCondition(
            controller="crossing", phase="ns_green", label="ns"
        )
        assert condition.get_details() == {
            "controller": "crossing",
            "phase": "ns_green",
        }


class TestWhyItDidNotFire:
    """A condition that never fires has to say which of the reasons it is.

    All of them return ``None`` -- an action treats any non-``None`` result as
    its trigger firing, so a failing result here would start the action this
    condition exists to hold back -- so the logs are the only thing that tells
    a mistyped name apart from a phase that simply has not come round.
    """

    def test_an_unknown_controller_is_reported_once_not_every_tick(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        condition = TrafficSignalControllerCondition(
            controller="nowhere", phase="go", label="nowhere"
        )
        world = _World()

        with caplog.at_level("WARNING"):
            for _ in range(5):
                assert condition.check(world, 1.0) is None

        assert caplog.text.count("no controller named 'nowhere'") == 1

    def test_a_cycle_that_has_not_started_is_silent(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        """It is about to change by itself, which is not a fault to report."""
        first = _cycle("first")
        second = SignalController(
            name="second",
            phases=(Phase("go", 5.0, ()),),
            delay_seconds=3.0,
            reference=first,
        )
        register_signal_controller(second)
        world = _World()
        second.tick(world)

        condition = TrafficSignalControllerCondition(
            controller="second", phase="go", label="second"
        )

        with caplog.at_level("DEBUG"):
            assert condition.check(world, 1.0) is None

        assert caplog.text == ""

    def test_the_wrong_phase_names_the_one_that_is_showing(
        self, junction: dict[int, MagicMock], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Debug rather than warning: on most ticks some other phase is up.

        That is an answer, not a fault.
        """
        controller = _cycle()
        register_signal_controller(controller)
        world = _World()
        controller.tick(world)

        condition = TrafficSignalControllerCondition(
            controller="crossing", phase="ew_green", label="ew"
        )

        with caplog.at_level("DEBUG"):
            assert condition.check(world, 1.0) is None

        assert "is in phase 'ns_green', not 'ew_green'" in caplog.text
