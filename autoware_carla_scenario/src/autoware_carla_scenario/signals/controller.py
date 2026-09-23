"""A traffic signal controller: the thing that cycles a junction's phases.

The framework already had an action that sets one light and a condition that
reads one, which is enough to make a junction inconsistent -- two conflicting
approaches green is a state no real road reaches -- and not enough to describe
how a junction actually behaves.  A real junction runs a *cycle*: north green
for twenty seconds, north amber for three, everything red for one, then the
same for east.  The amber is not a detail.  It is where an ego decides whether
to stop or to go, so a framework that cannot express it cannot test the
decision.

This is that cycle, modelled the way OpenSCENARIO models it and the way
``scenario_simulator_v2`` implements it:

* a :class:`Phase` names a duration and the state every signal it controls
  holds for it;
* a :class:`SignalController` owns an ordered list of phases and walks them in
  a loop, each for its own duration;
* a phase can also be jumped to by name, which is what
  ``TrafficSignalControllerAction`` does.

Time is the world's
-------------------
Durations are simulated seconds, read from the world rather than from a wall
clock, because a scenario is measured on the world's clock and on nothing else
(see :class:`~autoware_carla_scenario.scenario_runner._ScenarioClock`).  Only
*differences* are ever taken, so the world's own elapsed time serves directly
and the controller needs no reference to the run's start.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import carla

from ..coordinate.traffic_light import find_traffic_lights_for_lanelet2_id

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "STATE_NAMES",
    "Phase",
    "PhaseState",
    "SignalController",
    "build_controllers",
]

#: The signal states a phase may name, and what each is in CARLA.
#:
#: Spelled in the document as lower-case words rather than as CARLA enum
#: members, so a scenario file does not name a simulator's API -- the same rule
#: the traffic seam follows.  ``off`` and ``unknown`` are CARLA's own: a dark
#: signal and one whose state cannot be established are different things, and a
#: junction under test may legitimately contain either.
STATE_NAMES: "dict[str, carla.TrafficLightState]" = {
    "green": carla.TrafficLightState.Green,
    "yellow": carla.TrafficLightState.Yellow,
    "red": carla.TrafficLightState.Red,
    "off": carla.TrafficLightState.Off,
    "unknown": carla.TrafficLightState.Unknown,
}


@dataclass(frozen=True)
class PhaseState:
    """One signal's state for the duration of a phase.

    Attributes:
        lanelet2_regulatory_element_id: The signal, named by the Lanelet2
            regulatory element it belongs to -- the same id, and the same name
            for it, that every other signal primitive here takes.
        state: One of :data:`STATE_NAMES`.
    """

    lanelet2_regulatory_element_id: int
    state: str


@dataclass(frozen=True)
class Phase:
    """One step of a cycle: what every controlled signal shows, and for how long.

    A phase states every signal it controls, not only the ones that change.
    That is what makes it a phase rather than a set of edits: applying it puts
    the junction into a known whole, so no combination of earlier phases can
    leave a light behind in a state the author never wrote.

    Attributes:
        name: What the scenario calls this phase.  Unique within a controller.
        duration_seconds: How long it holds before the cycle moves on.  Zero is
            allowed and means the cycle passes straight through -- useful for
            an all-red clearance a scenario wants declared but not waited on.
        states: The signals and their states.
    """

    name: str
    duration_seconds: float
    states: "tuple[PhaseState, ...]"


@dataclass
class SignalController:
    """A junction's cycle, running on simulated time.

    Attributes:
        name: What the scenario calls this controller.
        phases: The cycle, in order.  Walked circularly.
        delay_seconds: How long after *reference* starts its first phase this
            controller starts its own.  Zero without a *reference* means it
            starts with the run.
        reference: Another controller this one is offset from, which is how a
            progressive system -- a green wave along a corridor -- is written.
    """

    name: str
    phases: "tuple[Phase, ...]"
    delay_seconds: float = 0.0
    reference: Optional["SignalController"] = None

    #: Index of the phase now showing, or ``None`` before the cycle starts.
    _current: Optional[int] = field(default=None, init=False, repr=False)
    #: World time the current phase began, in simulated seconds.
    _phase_started_at: float = field(default=0.0, init=False, repr=False)
    #: World time this controller's first phase began.  A controller that
    #: references this one measures its own delay from here.
    _started_at: Optional[float] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> Optional[Phase]:
        """The phase now showing, or ``None`` before the cycle has started."""
        if self._current is None:
            return None
        return self.phases[self._current]

    @property
    def started_at(self) -> Optional[float]:
        """Simulated time this controller began its first phase."""
        return self._started_at

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def tick(self, world: "carla.World") -> None:
        """Advance the cycle if it is due, and hold it where it is otherwise.

        Registered as a pre-tick callback, so this runs once per tick before
        anything observes the world.  It reads the world's own simulated time
        rather than being handed one: only differences are taken, so the run's
        start cancels out and there is nothing to keep in step.
        """
        now = _simulated_now(world)

        if self._current is None:
            if self._may_start(now):
                self._enter(0, world, now)
            return

        if now - self._phase_started_at >= self.phases[self._current].duration_seconds:
            self._enter((self._current + 1) % len(self.phases), world, now)

    def change_phase_to(self, phase_name: str, world: "carla.World") -> bool:
        """Jump to the named phase, restarting its duration from now.

        This is what an action does.  The cycle carries on from there, so a
        scenario that forces a junction green does not also freeze it a phase
        later -- holding it is a separate decision, made by whatever the
        scenario does next.

        Returns:
            ``True`` when the phase was found and applied.
        """
        for index, phase in enumerate(self.phases):
            if phase.name == phase_name:
                self._enter(index, world, _simulated_now(world))
                return True
        logger.warning(
            "SignalController '%s': no phase named '%s'; it has %s",
            self.name,
            phase_name,
            ", ".join(repr(p.name) for p in self.phases) or "none",
        )
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _may_start(self, now: float) -> bool:
        """Whether the cycle may begin, given any controller it waits on.

        A controller with no reference starts as soon as it is first ticked.
        One with a reference starts *delay* seconds after that reference began
        its own first phase, so a corridor's junctions can be offset into a
        green wave by declaring the offsets rather than by timing actions.
        """
        if self.reference is None:
            return True
        reference_started = self.reference.started_at
        if reference_started is None:
            return False
        return now - reference_started >= self.delay_seconds

    def _enter(self, index: int, world: "carla.World", now: float) -> None:
        """Show phase *index* and start its clock."""
        self._current = index
        self._phase_started_at = now
        if self._started_at is None:
            self._started_at = now
        _apply(self.phases[index], world, self.name)


def _apply(phase: Phase, world: "carla.World", controller_name: str) -> None:
    """Set every signal the phase names, and freeze it there.

    Frozen because CARLA runs its own cycle otherwise, and a junction the
    simulator is also driving is not one the scenario can assert about.  The
    controller re-applies on every phase change, so freezing costs nothing it
    would otherwise have.
    """
    unresolved: list[int] = []
    for entry in phase.states:
        state = STATE_NAMES.get(entry.state)
        if state is None:
            # The validator rejects this while the document is being written,
            # so reaching here means something bypassed it.  One signal left
            # alone beats a run that dies mid-cycle on a KeyError.
            logger.warning(
                "SignalController '%s': phase '%s' asks for unknown state "
                "'%s'; leaving signal %d alone",
                controller_name,
                phase.name,
                entry.state,
                entry.lanelet2_regulatory_element_id,
            )
            continue
        lights = find_traffic_lights_for_lanelet2_id(
            world, entry.lanelet2_regulatory_element_id
        )
        if not lights:
            unresolved.append(entry.lanelet2_regulatory_element_id)
            continue
        for light in lights:
            light.set_state(state)
            light.freeze(True)

    if unresolved:
        # Warned rather than debugged: a phase that names a signal the map does
        # not have is a document that describes a junction this map has not
        # got, and every cycle of the run will be wrong in the same way.
        logger.warning(
            "SignalController '%s': phase '%s' names %d signal(s) the map does "
            "not resolve: %s",
            controller_name,
            phase.name,
            len(unresolved),
            ", ".join(str(i) for i in unresolved),
        )


def _simulated_now(world: "carla.World") -> float:
    """Return the world's own elapsed simulated seconds."""
    return float(world.get_snapshot().timestamp.elapsed_seconds)


def build_controllers(declared: "list[Any]") -> "tuple[SignalController, ...]":
    """Turn the map's declared controllers into running ones.

    Resolves ``reference`` from a name to the controller object, which is why
    this builds them all together rather than one at a time: a green wave is a
    graph, and half of it is not runnable.

    A reference naming a controller that does not exist is dropped with a
    warning rather than raising.  The validator rejects it while the document
    is being written, so reaching here means something bypassed that, and a run
    that loses one junction's offset is better than a run that will not start.

    Args:
        declared: ``MapRef.traffic_signal_controllers`` -- typed loosely so
            this module stays independent of the authoring package.

    Returns:
        The controllers, in declaration order.
    """
    built: "dict[str, SignalController]" = {}
    for spec in declared:
        built[spec.name] = SignalController(
            name=spec.name,
            phases=tuple(
                Phase(
                    name=phase.name,
                    duration_seconds=float(phase.duration_seconds),
                    states=tuple(
                        PhaseState(
                            lanelet2_regulatory_element_id=int(
                                entry.lanelet2_regulatory_element_id
                            ),
                            # Lower-cased here rather than demanded of the
                            # author: the older single-signal card spells the
                            # same colours as CARLA's enum members, and a
                            # document that mixes 'Green' and 'green' should
                            # not mean two different things.
                            state=str(entry.state).lower(),
                        )
                        for entry in phase.states
                    ),
                )
                for phase in spec.phases
            ),
            delay_seconds=float(spec.delay_seconds),
        )

    for spec in declared:
        if spec.reference is None:
            continue
        reference = built.get(spec.reference)
        if reference is None:
            logger.warning(
                "SignalController '%s' is offset from '%s', which no "
                "controller on this map declares; it will start with the run "
                "instead",
                spec.name,
                spec.reference,
            )
            continue
        built[spec.name].reference = reference

    return tuple(built.values())
