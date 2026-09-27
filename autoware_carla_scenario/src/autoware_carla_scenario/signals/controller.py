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

Groups are resolved here, not carried
-------------------------------------
A phase in the document usually names *signal groups* -- the movements the map
says change as one -- rather than individual regulatory elements.  That is an
authoring concern: :func:`build_controllers` expands each group into one
:class:`PhaseState` per member, and everything from there down deals in single
signals.

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

    Always one signal: a phase the document wrote against a *group* is expanded
    into one of these per member at build time, so nothing downstream of
    :func:`build_controllers` has to know that groups exist.  Applying a phase
    stays a flat walk over signals, which is what it has to be in the end.

    Attributes:
        lanelet2_regulatory_element_id: The signal, named by the Lanelet2
            regulatory element it belongs to -- the same id, and the same name
            for it, that every other signal primitive here takes.
        state: One of :data:`STATE_NAMES`.
        group: The signal group this came from, when it came from one.  Carried
            for diagnostics only: a warning that says which *movement* failed to
            resolve is actionable, where one naming a bare id is a lookup.
    """

    lanelet2_regulatory_element_id: int
    state: str
    group: Optional[str] = None


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
    unresolved: list[str] = []
    for entry in phase.states:
        state = STATE_NAMES.get(entry.state)
        if state is None:
            # The validator rejects this while the document is being written,
            # so reaching here means something bypassed it.  One signal left
            # alone beats a run that dies mid-cycle on a KeyError.
            logger.warning(
                "SignalController '%s': phase '%s' asks for unknown state "
                "'%s'; leaving signal %s alone",
                controller_name,
                phase.name,
                entry.state,
                _describe(entry),
            )
            continue
        lights = find_traffic_lights_for_lanelet2_id(
            world, entry.lanelet2_regulatory_element_id
        )
        if not lights:
            unresolved.append(_describe(entry))
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
            ", ".join(unresolved),
        )


def _describe(entry: PhaseState) -> str:
    """Name a signal the way a reader can act on it.

    A bare id sends the reader to the map to find out which movement it was; the
    group name says it outright, so it is given whenever the phase named one.
    """
    if entry.group is None:
        return str(entry.lanelet2_regulatory_element_id)
    return f"{entry.lanelet2_regulatory_element_id} (group '{entry.group}')"


def _simulated_now(world: "carla.World") -> float:
    """Return the world's own elapsed simulated seconds."""
    return float(world.get_snapshot().timestamp.elapsed_seconds)


def _group_members(declared_groups: "list[Any]") -> "dict[str, tuple[int, ...]]":
    """Index the map's signal groups by name.

    A later declaration of an already-taken name wins and says so, rather than
    being merged into the first: the validator rejects the duplicate while the
    document is being written, and silently driving the union of two groups
    somebody meant as one would be a junction nobody wrote.
    """
    members: "dict[str, tuple[int, ...]]" = {}
    for group in declared_groups or ():
        name = str(group.name)
        if name in members:
            logger.warning(
                "Signal group '%s' is declared more than once on this map; "
                "the last declaration wins",
                name,
            )
        members[name] = tuple(int(i) for i in group.lanelet2_regulatory_element_ids)
    return members


def _phase_states(
    phase: "Any",
    controller_name: str,
    members: "dict[str, tuple[int, ...]]",
) -> "tuple[PhaseState, ...]":
    """Expand one declared phase into one :class:`PhaseState` per signal.

    Every complaint here is a warning rather than a raise, for the reason the
    rest of this module gives: the validator rejects each of these while the
    document is being written, so reaching them means something bypassed it, and
    a junction missing one movement is a better report than a run that would not
    start.
    """
    states: "list[PhaseState]" = []
    for entry in phase.states:
        # Lower-cased here rather than demanded of the author: the older
        # single-signal card spells the same colours as CARLA's enum members,
        # and a document that mixes 'Green' and 'green' should not mean two
        # different things.
        state = str(entry.state).lower()
        # A blank group reads as "not named", matching the validator: that is
        # what a form submits for an untouched field.
        group = (getattr(entry, "group", None) or "").strip() or None
        signal_id = getattr(entry, "lanelet2_regulatory_element_id", None)

        if group is None:
            if signal_id is None:
                logger.warning(
                    "SignalController '%s': phase '%s' has a state naming "
                    "neither a signal group nor a regulatory element; skipping it",
                    controller_name,
                    phase.name,
                )
                continue
            states.append(
                PhaseState(lanelet2_regulatory_element_id=int(signal_id), state=state)
            )
            continue

        if signal_id is not None:
            # The group is the movement the author was describing, so it is the
            # one kept; saying which was dropped is what makes the warning
            # actionable.
            logger.warning(
                "SignalController '%s': phase '%s' names both signal group '%s' "
                "and regulatory element %s; driving the group and ignoring the "
                "element",
                controller_name,
                phase.name,
                group,
                signal_id,
            )

        ids = members.get(group)
        if ids is None:
            logger.warning(
                "SignalController '%s': phase '%s' names signal group '%s', "
                "which this map does not declare; that movement will not be "
                "driven",
                controller_name,
                phase.name,
                group,
            )
            continue
        if not ids:
            logger.warning(
                "SignalController '%s': phase '%s' names signal group '%s', "
                "which drives no signal",
                controller_name,
                phase.name,
                group,
            )
        states.extend(
            PhaseState(lanelet2_regulatory_element_id=signal, state=state, group=group)
            for signal in ids
        )
    return tuple(states)


def build_controllers(
    declared: "list[Any]", signal_groups: "Optional[list[Any]]" = None
) -> "tuple[SignalController, ...]":
    """Turn the scenario's declared controllers into running ones.

    Resolves two things a declaration only names.  ``reference`` becomes the
    controller object, which is why this builds them all together rather than
    one at a time: a green wave is a graph, and half of it is not runnable.  And
    each phase state naming a *signal group* becomes one
    :class:`PhaseState` per regulatory element that group covers, so the
    running controller deals only in signals.

    A reference naming a controller that does not exist is dropped with a
    warning rather than raising.  The validator rejects it while the document
    is being written, so reaching here means something bypassed that, and a run
    that loses one junction's offset is better than a run that will not start.

    Args:
        declared: ``MapRef.traffic_signal_controllers`` -- typed loosely so
            this module stays independent of the authoring package.
        signal_groups: ``MapRef.signal_groups``, the map's statement of which
            signals move together.  Omit it for a document whose phases name
            regulatory elements directly.

    Returns:
        The controllers, in declaration order.
    """
    members = _group_members(signal_groups or [])
    built: "dict[str, SignalController]" = {}
    for spec in declared:
        built[spec.name] = SignalController(
            name=spec.name,
            phases=tuple(
                Phase(
                    name=phase.name,
                    duration_seconds=float(phase.duration_seconds),
                    states=_phase_states(phase, spec.name, members),
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
