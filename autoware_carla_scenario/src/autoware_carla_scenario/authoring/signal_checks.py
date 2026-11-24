"""The checks of a scenario's signal table: its groups, phases and offsets.

What :mod:`authoring.validator` checks of ``map.signal_groups`` and
``map.traffic_signal_controllers`` -- a signal group's name, members and
conflicts, a controller's name, phases, states and start delay, and the
references between controllers -- over plain dataclasses the validator builds
from the document's pydantic models.  Kept apart from the validator, which
imports pydantic, so the static check compiles these rules (#2364).  The
validator turns the ``(severity, path, message)`` triples they return back
into its issues, in the order they are returned, and still checks the actions
and conditions that name a controller and a phase itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "PERMISSIVE_SIGNAL_STATES",
    "SignalController",
    "SignalGroup",
    "SignalGroups",
    "SignalPhase",
    "SignalState",
    "SignalTableCheck",
    "check_signal_table",
]

#: The signal states that let traffic through.  Amber is one of them: a junction
#: showing amber to one movement still has vehicles in it, which is exactly why
#: a real one never shows amber to a movement that crosses it either.
PERMISSIVE_SIGNAL_STATES = frozenset({"green", "yellow"})


@dataclass(frozen=True)
class SignalGroup:
    """One entry of ``map.signal_groups``, as written.

    Attributes:
        name: What a phase calls the group.
        regulatory_element_ids: The Lanelet2 regulatory elements it drives.
        conflicts_with: The groups whose movements cross this one's.
    """

    name: str
    regulatory_element_ids: list[int] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SignalState:
    """One state of a phase: a group or a single signal, and its colour."""

    group: Optional[str] = None
    regulatory_element_id: Optional[int] = None
    state: str = "red"


@dataclass(frozen=True)
class SignalPhase:
    """One phase of a controller's cycle, as written."""

    name: str
    duration_seconds: float = 10.0
    states: list[SignalState] = field(default_factory=list)


@dataclass(frozen=True)
class SignalController:
    """One entry of ``map.traffic_signal_controllers``, as written."""

    name: str
    phases: list[SignalPhase] = field(default_factory=list)
    delay_seconds: float = 0.0
    reference: Optional[str] = None


@dataclass(frozen=True)
class SignalGroups:
    """The map's signal groups, resolved for the phase checks that follow.

    Attributes:
        members: Group name -> the regulatory element ids it drives.
        conflicts: Group name -> the groups whose movements cross it.  Closed
            symmetrically, so a crossing declared on one side only is still
            caught from the other.
    """

    members: dict[str, set[int]] = field(default_factory=dict)
    conflicts: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalTableCheck:
    """What :func:`check_signal_table` found.

    Attributes:
        issues: ``(severity, path, message)`` per finding, in the order found;
            the severity is ``"error"`` or ``"warning"``.
        controller_index: A controller's stripped name -> the index of the
            first controller with it, for each named controller; the one the
            references and the actions naming that name resolve to.
    """

    issues: list[tuple[str, str, str]] = field(default_factory=list)
    controller_index: dict[str, int] = field(default_factory=dict)


class _Issues:
    """Accumulates ``(severity, path, message)`` triples."""

    found: list[tuple[str, str, str]]

    def __init__(self) -> None:
        self.found = []

    def error(self, path: str, message: str) -> None:
        self.found.append(("error", path, message))

    def warn(self, path: str, message: str) -> None:
        self.found.append(("warning", path, message))


def _stripped(value: Optional[str]) -> str:
    """``(value or "").strip()``: a blank or absent name reads as ``""``."""
    if value is None:
        return ""
    return value.strip()


def check_signal_table(
    groups: list[SignalGroup],
    controllers: list[SignalController],
    state_names: list[str],
) -> SignalTableCheck:
    """Check the map's signal groups and the cycles run on them.

    The groups come first because the phases are checked against them -- both
    for a name that resolves and for the one rule the grouping exists to make
    checkable: that no phase lets two crossing movements through at once.
    Then each controller's name, phases and start delay, in turn, and last the
    references between controllers.

    Args:
        groups: ``map.signal_groups``, in order.
        controllers: ``map.traffic_signal_controllers``, in order.
        state_names: The signal states a phase may name.
    """
    out = _Issues()
    resolved = _check_signal_groups(out, groups)
    by_name: dict[str, int] = {}

    for index, controller in enumerate(controllers):
        path = f"map.traffic_signal_controllers[{index}]"
        name = controller.name.strip()
        if not name:
            out.error(path, "A signal controller needs a name.")
        elif name in by_name:
            out.error(path, f"Duplicate signal controller name {name!r}.")
        else:
            by_name[name] = index

        _check_signal_phases(out, path, controller, resolved, state_names)
        _check_signal_offset(out, path, controller)

    _check_signal_reference_targets(out, controllers, by_name)
    _check_signal_reference_cycles(out, controllers, by_name)
    return SignalTableCheck(issues=out.found, controller_index=by_name)


def _check_signal_groups(out: _Issues, declared: list[SignalGroup]) -> SignalGroups:
    """Check ``map.signal_groups`` and return it resolved.

    A group is a claim about the road -- these signals change together, those
    two movements cross -- so the mistakes worth catching here are the ones that
    make the claim incoherent.  The one that matters most is a signal in two
    groups: a phase could then ask for two colours on one light, and which it
    ended up showing would depend on the order the states happened to be written
    in.
    """
    members: dict[str, set[int]] = {}
    #: Which declaration each name came from, so the conflicts pass can tell a
    #: duplicate's entries from the ones belonging to the group that kept it.
    declared_at: dict[str, int] = {}
    owner_of: dict[int, str] = {}

    for index, group in enumerate(declared):
        path = f"map.signal_groups[{index}]"
        name = group.name.strip()
        if not name:
            out.error(path, "A signal group needs a name, since phases call it by one.")
            continue
        if name in members:
            out.error(
                path,
                f"Duplicate signal group name {name!r}; a phase naming it would "
                "be ambiguous.",
            )
            continue

        ids = set(group.regulatory_element_ids)
        if not ids:
            out.error(
                f"{path}.lanelet2_regulatory_element_ids",
                f"Signal group {name!r} drives no signal, so a phase naming it "
                "would set nothing.",
            )
        if len(ids) != len(group.regulatory_element_ids):
            out.warn(
                f"{path}.lanelet2_regulatory_element_ids",
                f"Signal group {name!r} names the same regulatory element more "
                "than once.",
            )
        for signal in sorted(ids):
            if signal in owner_of:
                existing = owner_of[signal]
                out.error(
                    f"{path}.lanelet2_regulatory_element_ids",
                    f"Regulatory element {signal} is in both signal group "
                    f"{existing!r} and {name!r}. A signal shows one colour, so "
                    "a phase naming both groups would not say which.",
                )
                continue
            owner_of[signal] = name
        members[name] = ids
        declared_at[name] = index

    conflicts = _check_signal_group_conflicts(out, declared, members, declared_at)
    return SignalGroups(members=members, conflicts=conflicts)


def _check_signal_group_conflicts(
    out: _Issues,
    declared: list[SignalGroup],
    members: dict[str, set[int]],
    declared_at: dict[str, int],
) -> dict[str, set[str]]:
    """Resolve ``conflicts_with`` into a symmetric relation, reporting bad names.

    Symmetric because a crossing is not directional: an author who wrote it once
    has said everything there is to say, and demanding the mirror entry would
    only create a second place for it to be forgotten.
    """
    conflicts: dict[str, set[str]] = {name: set() for name in members}
    for index, group in enumerate(declared):
        name = group.name.strip()
        if name not in declared_at or declared_at[name] != index:
            # Nameless, or a duplicate that lost the name.  Already reported, and
            # its conflicts are not the surviving group's to inherit.
            continue
        for written in group.conflicts_with:
            other = written.strip()
            path = f"map.signal_groups[{index}].conflicts_with"
            if other == name:
                out.error(
                    path,
                    f"Signal group {name!r} conflicts with itself, which would "
                    "leave no phase able to let it through.",
                )
                continue
            if other not in members:
                out.error(
                    path,
                    f"Signal group {name!r} conflicts with {other!r}, which this "
                    "map does not declare.",
                )
                continue
            conflicts[name].add(other)
            conflicts[other].add(name)
    return conflicts


def _check_signal_phases(
    out: _Issues,
    path: str,
    controller: SignalController,
    groups: SignalGroups,
    state_names: list[str],
) -> None:
    """Check one controller's cycle: its phases, their durations and states."""
    if not controller.phases:
        out.error(
            f"{path}.phases",
            f"Signal controller {controller.name!r} declares no phases, so it "
            "would never show anything.",
        )
        return

    seen: set[str] = set()
    for index, phase in enumerate(controller.phases):
        phase_path = f"{path}.phases[{index}]"
        name = phase.name.strip()
        if not name:
            out.error(phase_path, "A phase needs a name, since actions call it by one.")
        elif name in seen:
            out.error(
                phase_path,
                f"Signal controller {controller.name!r} declares two phases "
                f"named {name!r}; an action naming it would be ambiguous.",
            )
        seen.add(name)

        if not math.isfinite(phase.duration_seconds) or phase.duration_seconds < 0:
            out.error(
                f"{phase_path}.duration_seconds",
                "A phase duration cannot be negative, and must be finite.",
            )

        if not phase.states:
            # Not an error: a phase that holds the junction unchanged for a
            # while is a legitimate thing to write, even if it is rarely what
            # was meant.
            out.warn(
                f"{phase_path}.states",
                f"Phase {phase.name!r} sets no signal, so it only spends time.",
            )

        for state_index, entry in enumerate(phase.states):
            if entry.state.lower() not in state_names:
                out.error(
                    f"{phase_path}.states[{state_index}].state",
                    f"Unknown signal state {entry.state!r}; expected one of "
                    + ", ".join(sorted(state_names)),
                )
            _check_signal_phase_subject(
                out, f"{phase_path}.states[{state_index}]", phase, entry, groups
            )

        _check_signal_phase_agrees_with_itself(out, phase_path, phase, groups)
        _check_signal_phase_lets_no_conflict_through(out, phase_path, phase, groups)


def _check_signal_phase_subject(
    out: _Issues,
    path: str,
    phase: SignalPhase,
    entry: SignalState,
    groups: SignalGroups,
) -> None:
    """Every state names exactly one subject, and a named group must exist.

    Both at once is refused rather than resolved by precedence: a rule about
    which of the two wins is a rule nobody reading the document would know, and
    the author who wrote both did not have one in mind either.
    """
    # A blank is what a form submits for an untouched field, so it reads as
    # "not named" rather than as a group called "".
    group = _stripped(entry.group)
    signal = entry.regulatory_element_id

    if group and signal is not None:
        out.error(
            path,
            f"Phase {phase.name!r} names both signal group {group!r} and "
            f"regulatory element {signal} in one state. Name one: the group for "
            "the movement, the element for a single light.",
        )
        return
    if not group and signal is None:
        out.error(
            path,
            f"Phase {phase.name!r} has a state that names no signal -- give it "
            "either a signal group or a regulatory element id.",
        )
        return
    if group and group not in groups.members:
        if len(groups.members) > 0:
            declared = " Declared: " + ", ".join(
                repr(n) for n in sorted(groups.members)
            )
        else:
            declared = " This map declares none."
        out.error(
            f"{path}.group",
            f"Phase {phase.name!r} names signal group {group!r}, which "
            "map.signal_groups does not declare." + declared,
        )


def _check_signal_phase_agrees_with_itself(
    out: _Issues, path: str, phase: SignalPhase, groups: SignalGroups
) -> None:
    """One phase must not ask for two colours on the same signal.

    Applying a phase walks its states in order, so a repeat is not ambiguous at
    runtime -- the last one wins.  It is ambiguous to the reader, which is the
    problem: a phase is meant to be readable as the junction's whole state.
    """
    seen_groups: set[str] = set()
    seen_signals: set[int] = set()
    for entry in phase.states:
        group = _stripped(entry.group)
        if group:
            if group in seen_groups:
                out.error(
                    f"{path}.states",
                    f"Phase {phase.name!r} sets signal group {group!r} twice.",
                )
            seen_groups.add(group)
            continue
        element = entry.regulatory_element_id
        if element is None:
            continue
        signal = int(element)
        if signal in seen_signals:
            out.error(
                f"{path}.states",
                f"Phase {phase.name!r} sets regulatory element {signal} twice.",
            )
        seen_signals.add(signal)

    # A bare element that one of the named groups also covers is the one overlap
    # that is legitimate: overriding a single light out of a movement.  It is
    # worth saying out loud, because it only works in the order it is written.
    covered: set[int] = set()
    for group in seen_groups:
        if group in groups.members:
            covered.update(groups.members[group])
    for signal in sorted(seen_signals & covered):
        out.warn(
            f"{path}.states",
            f"Phase {phase.name!r} sets regulatory element {signal} on its own "
            "as well as through a signal group that covers it; whichever is "
            "written last is what the light shows.",
        )


def _check_signal_phase_lets_no_conflict_through(
    out: _Issues, path: str, phase: SignalPhase, groups: SignalGroups
) -> None:
    """No phase may let two conflicting movements through at once.

    This is the rule the whole grouping exists for.  Two conflicting approaches
    green is the state a junction assembled one light at a time can reach and a
    real road cannot, and until the map said which movements cross, nothing
    could check it -- an author listing twelve regulatory elements per phase had
    only their own care to rely on.

    It reads the groups a phase names, not the signals it ends up setting, and
    that is deliberate: naming a group is how an author invokes the map's
    statement about that movement, including this rule.  A scenario whose
    subject *is* a junction in an impossible state names the regulatory elements
    directly and is left alone.
    """
    permissive: dict[str, str] = {}
    for entry in phase.states:
        group = _stripped(entry.group)
        if not group or group not in groups.members:
            continue
        state = entry.state.lower()
        if state in PERMISSIVE_SIGNAL_STATES and group not in permissive:
            permissive[group] = state

    reported: set[tuple[str, str]] = set()
    for group in sorted(permissive):
        crossing: set[str] = set()
        if group in groups.conflicts:
            crossing = groups.conflicts[group]
        for other in sorted(crossing):
            if other not in permissive:
                continue
            first = group if group <= other else other
            second = other if group <= other else group
            if (first, second) in reported:
                continue
            reported.add((first, second))
            out.error(
                f"{path}.states",
                f"Phase {phase.name!r} shows {permissive[first]} to signal group "
                f"{first!r} and {permissive[second]} to {second!r} at the same "
                "time, but the map declares their movements as crossing. Real "
                "junctions put an all-red clearance between them. If a junction "
                "in this state is what the scenario tests, name the regulatory "
                "elements directly instead of the groups.",
            )


def _check_signal_offset(out: _Issues, path: str, controller: SignalController) -> None:
    """Check the controller's offset from another one.

    A delay measures from the moment its reference started, so a delay with
    nothing to measure from is not a controller that starts late -- it is one
    whose author expected an offset that silently does not exist.
    """
    if not math.isfinite(controller.delay_seconds) or controller.delay_seconds < 0:
        out.error(
            f"{path}.delay_seconds",
            "A start delay cannot be negative, and must be finite.",
        )
    # ``!= 0.0`` is the float's truth: NaN counts as a delay, -0.0 does not.
    if controller.delay_seconds != 0.0 and controller.reference is None:
        out.error(
            f"{path}.delay_seconds",
            f"Signal controller {controller.name!r} declares a start delay but "
            "no reference controller to measure it from.",
        )


def _check_signal_reference_targets(
    out: _Issues, declared: list[SignalController], by_name: dict[str, int]
) -> None:
    """Every ``reference`` must name a controller this map declares."""
    for index, controller in enumerate(declared):
        reference = controller.reference
        if reference is None:
            continue
        if reference == controller.name:
            out.error(
                f"map.traffic_signal_controllers[{index}].reference",
                f"Signal controller {controller.name!r} is offset from itself.",
            )
        elif reference not in by_name:
            out.error(
                f"map.traffic_signal_controllers[{index}].reference",
                f"Signal controller {controller.name!r} is offset from "
                f"{reference!r}, which this scenario does not declare.",
            )


def _check_signal_reference_cycles(
    out: _Issues, declared: list[SignalController], by_name: dict[str, int]
) -> None:
    """Reject controllers that wait on each other.

    Each controller starts once the one it references has, so a loop is a set
    of junctions none of which can ever begin -- a run where every light in the
    corridor stays on CARLA's own cycle and nothing says why.

    The walk follows each reference to the controller *by_name* resolves it to,
    by index, until a controller with no reference, a reference to no
    controller, or a name already walked.
    """
    reported: set[str] = set()
    for start in range(len(declared)):
        walked: list[str] = []
        current = start
        while True:
            controller = declared[current]
            reference = controller.reference
            if reference is None:
                break
            if controller.name in walked:
                loop = walked[walked.index(controller.name) :]
                key = " -> ".join(sorted(loop))
                if key not in reported:
                    reported.add(key)
                    out.error(
                        "map.traffic_signal_controllers",
                        "These signal controllers are offset from each other in "
                        "a loop, so none of them can start: "
                        + " -> ".join(loop + [controller.name]),
                    )
                break
            walked.append(controller.name)
            if reference not in by_name:
                break
            current = by_name[reference]
