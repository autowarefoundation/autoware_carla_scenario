"""The validator's signal-table and bound checks, moved into the checked
:mod:`~autoware_carla_scenario.authoring.signal_checks` and
:mod:`~autoware_carla_scenario.authoring.field_bounds` (#2364).

The validator read the document's pydantic models; the checks now take plain
dataclasses (or the bounds alone), rewritten for Codon (an ``Optional``
controller walked in a loop, ``dict.get()`` compared with an ``int``,
``(name or "")``).  These tests hold them to the code they replace over many
generated inputs, finding for finding and in the same order.
"""

from __future__ import annotations

import itertools
import math
import random
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.authoring import validator
from autoware_carla_scenario.authoring.field_bounds import (
    bound_text,
    bounds_text,
    traffic_mean_bounds,
    within_bounds,
)
from autoware_carla_scenario.authoring.models import (
    SIGNAL_STATE_NAMES,
    SignalControllerRef,
    SignalGroupRef,
    SignalPhaseRef,
    SignalStateRef,
)
from autoware_carla_scenario.authoring.signal_checks import (
    PERMISSIVE_SIGNAL_STATES,
    SignalController,
    SignalGroup,
    SignalPhase,
    SignalState,
    check_signal_table,
)
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import Issue, _Collector

# ---------------------------------------------------------------------------
# The code validator ran before #2364, verbatim but for the names (and
# `_check_signal_uses`, which stays in the validator, called through a hook).
# ---------------------------------------------------------------------------

_OLD_PERMISSIVE_SIGNAL_STATES = frozenset({"green", "yellow"})

#: What the old code handed `_check_signal_uses`, and what to call with it.
_USES: dict[str, Any] = {}


def _old_check_signal_uses(out: Any, document: Any, by_name: dict[str, Any]) -> None:
    _USES["by_name"] = dict(by_name)
    if _USES.get("real"):
        validator._check_signal_uses(out, document, by_name)


def _old_check_signal_controllers(out: Any, document: Any) -> None:
    groups = _old_check_signal_groups(out, document)
    declared = document.map.traffic_signal_controllers
    by_name: dict[str, Any] = {}

    for index, controller in enumerate(declared):
        path = f"map.traffic_signal_controllers[{index}]"
        name = (controller.name or "").strip()
        if not name:
            out.error(path, "A signal controller needs a name.")
        elif name in by_name:
            out.error(path, f"Duplicate signal controller name {name!r}.")
        else:
            by_name[name] = controller

        _old_check_signal_phases(out, path, controller, groups)
        _old_check_signal_offset(out, path, controller)

    _old_check_signal_reference_targets(out, declared, by_name)
    _old_check_signal_reference_cycles(out, declared, by_name)
    _old_check_signal_uses(out, document, by_name)


class _OldSignalGroups:
    def __init__(
        self, members: dict[str, set[int]], conflicts: dict[str, set[str]]
    ) -> None:
        self.members = members
        self.conflicts = conflicts


def _old_check_signal_groups(out: Any, document: Any) -> _OldSignalGroups:
    declared = document.map.signal_groups
    members: "dict[str, set[int]]" = {}
    declared_at: "dict[str, int]" = {}
    owner_of: "dict[int, str]" = {}

    for index, group in enumerate(declared):
        path = f"map.signal_groups[{index}]"
        name = (group.name or "").strip()
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

        ids = set(group.lanelet2_regulatory_element_ids)
        if not ids:
            out.error(
                f"{path}.lanelet2_regulatory_element_ids",
                f"Signal group {name!r} drives no signal, so a phase naming it "
                "would set nothing.",
            )
        if len(ids) != len(group.lanelet2_regulatory_element_ids):
            out.warn(
                f"{path}.lanelet2_regulatory_element_ids",
                f"Signal group {name!r} names the same regulatory element more "
                "than once.",
            )
        for signal in sorted(ids):
            existing = owner_of.get(signal)
            if existing is not None:
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

    conflicts = _old_check_signal_group_conflicts(out, declared, members, declared_at)
    return _OldSignalGroups(members=members, conflicts=conflicts)


def _old_check_signal_group_conflicts(
    out: Any,
    declared: "list[Any]",
    members: "dict[str, set[int]]",
    declared_at: "dict[str, int]",
) -> "dict[str, set[str]]":
    conflicts: "dict[str, set[str]]" = {name: set() for name in members}
    for index, group in enumerate(declared):
        name = (group.name or "").strip()
        if declared_at.get(name) != index:
            continue
        for other in group.conflicts_with:
            other = str(other).strip()
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


def _old_check_signal_phases(
    out: Any, path: str, controller: Any, groups: _OldSignalGroups
) -> None:
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
        name = (phase.name or "").strip()
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
            out.warn(
                f"{phase_path}.states",
                f"Phase {phase.name!r} sets no signal, so it only spends time.",
            )

        for state_index, entry in enumerate(phase.states):
            if str(entry.state).lower() not in SIGNAL_STATE_NAMES:
                out.error(
                    f"{phase_path}.states[{state_index}].state",
                    f"Unknown signal state {entry.state!r}; expected one of "
                    + ", ".join(sorted(SIGNAL_STATE_NAMES)),
                )
            _old_check_signal_phase_subject(
                out, f"{phase_path}.states[{state_index}]", phase, entry, groups
            )

        _old_check_signal_phase_agrees_with_itself(out, phase_path, phase, groups)
        _old_check_signal_phase_lets_no_conflict_through(out, phase_path, phase, groups)


def _old_check_signal_phase_subject(
    out: Any, path: str, phase: Any, entry: Any, groups: _OldSignalGroups
) -> None:
    group = (entry.group or "").strip()
    signal = entry.lanelet2_regulatory_element_id

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
        out.error(
            f"{path}.group",
            f"Phase {phase.name!r} names signal group {group!r}, which "
            "map.signal_groups does not declare."
            + (
                " Declared: " + ", ".join(repr(n) for n in sorted(groups.members))
                if groups.members
                else " This map declares none."
            ),
        )


def _old_check_signal_phase_agrees_with_itself(
    out: Any, path: str, phase: Any, groups: _OldSignalGroups
) -> None:
    seen_groups: set[str] = set()
    seen_signals: set[int] = set()
    for entry in phase.states:
        group = (entry.group or "").strip()
        if group:
            if group in seen_groups:
                out.error(
                    f"{path}.states",
                    f"Phase {phase.name!r} sets signal group {group!r} twice.",
                )
            seen_groups.add(group)
            continue
        if entry.lanelet2_regulatory_element_id is None:
            continue
        signal = int(entry.lanelet2_regulatory_element_id)
        if signal in seen_signals:
            out.error(
                f"{path}.states",
                f"Phase {phase.name!r} sets regulatory element {signal} twice.",
            )
        seen_signals.add(signal)

    covered = {
        signal for group in seen_groups for signal in groups.members.get(group, set())
    }
    for signal in sorted(seen_signals & covered):
        out.warn(
            f"{path}.states",
            f"Phase {phase.name!r} sets regulatory element {signal} on its own "
            "as well as through a signal group that covers it; whichever is "
            "written last is what the light shows.",
        )


def _old_check_signal_phase_lets_no_conflict_through(
    out: Any, path: str, phase: Any, groups: _OldSignalGroups
) -> None:
    permissive: "dict[str, str]" = {}
    for entry in phase.states:
        group = (entry.group or "").strip()
        if not group or group not in groups.members:
            continue
        state = str(entry.state).lower()
        if state in _OLD_PERMISSIVE_SIGNAL_STATES:
            permissive.setdefault(group, state)

    reported: "set[tuple[str, str]]" = set()
    for group in sorted(permissive):
        for other in sorted(groups.conflicts.get(group, set())):
            if other not in permissive:
                continue
            first, second = sorted((group, other))
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


def _old_check_signal_offset(out: Any, path: str, controller: Any) -> None:
    if not math.isfinite(controller.delay_seconds) or controller.delay_seconds < 0:
        out.error(
            f"{path}.delay_seconds",
            "A start delay cannot be negative, and must be finite.",
        )
    if controller.delay_seconds and controller.reference is None:
        out.error(
            f"{path}.delay_seconds",
            f"Signal controller {controller.name!r} declares a start delay but "
            "no reference controller to measure it from.",
        )


def _old_check_signal_reference_targets(
    out: Any, declared: "list[Any]", by_name: "dict[str, Any]"
) -> None:
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


def _old_check_signal_reference_cycles(
    out: Any, declared: "list[Any]", by_name: "dict[str, Any]"
) -> None:
    reported: set[str] = set()
    for controller in declared:
        walked: list[str] = []
        current = controller
        while current is not None and current.reference is not None:
            if current.name in walked:
                loop = walked[walked.index(current.name) :]
                key = " -> ".join(sorted(loop))
                if key not in reported:
                    reported.add(key)
                    out.error(
                        "map.traffic_signal_controllers",
                        "These signal controllers are offset from each other in "
                        "a loop, so none of them can start: "
                        + " -> ".join(loop + [current.name]),
                    )
                break
            walked.append(current.name)
            current = by_name.get(current.reference)


def _old_check_bounds(
    out: Any, path: str, spec: Any, number: float, object_id: str | None
) -> None:
    low, high = spec.exclusive_minimum, spec.maximum
    inclusive = spec.minimum
    if (
        (low is None or number > low)
        and (inclusive is None or number >= inclusive)
        and (high is None or number <= high)
    ):
        return
    out.error(
        f"{path}.{spec.name}", f"{spec.label} {_old_bounds_text(spec)}.", object_id
    )


def _old_bound_text(spec: Any, bound: float) -> str:
    if spec.kind == "int" and float(bound).is_integer():
        return str(int(bound))
    return f"{bound:g}"


def _old_bounds_text(spec: Any) -> str:
    low, inclusive, high = spec.exclusive_minimum, spec.minimum, spec.maximum
    if high is not None and (low is not None or inclusive is not None):
        if low is not None:
            opening = f"({_old_bound_text(spec, low)}"
        else:
            assert inclusive is not None
            opening = f"[{_old_bound_text(spec, inclusive)}"
        return f"must be in {opening}, {_old_bound_text(spec, high)}]"
    if high is not None:
        return f"must be at most {_old_bound_text(spec, high)}"
    if low is not None:
        return (
            "must be positive"
            if low == 0
            else f"must be above {_old_bound_text(spec, low)}"
        )
    assert inclusive is not None
    return (
        "must not be negative"
        if inclusive == 0
        else f"must be at least {_old_bound_text(spec, inclusive)}"
    )


def _old_traffic_mean_bounds() -> dict[str, tuple[float, float]]:
    from autoware_carla_scenario.traffic.parameters import (
        BICYCLE_BOUNDS,
        PEDESTRIAN_BOUNDS,
        VEHICLE_BOUNDS,
    )

    bounds = {f"{name}_mean": span for name, span in VEHICLE_BOUNDS.items()}
    for name, span in PEDESTRIAN_BOUNDS.items():
        field_name = "pedestrian_min_gap_m" if name == "min_gap_m" else name
        bounds[f"{field_name}_mean"] = span
    for name, span in BICYCLE_BOUNDS.items():
        bounds[f"bicycle_{name}_mean"] = span
    return bounds


# ---------------------------------------------------------------------------
# Random signal tables
# ---------------------------------------------------------------------------

_NAMES = ["", " ", "a", "b", "c", " a", "b ", "ns", "ew", "d"]
_UNKNOWN = ["zz", " zz", "q"]
_STATES = ["green", "yellow", "red", "off", "unknown", "Red", "GREEN", "blue", " red"]
_DURATIONS = [10.0, 0.0, -1.0, math.nan, math.inf, 2.5, -0.0]
_DELAYS = [0.0, -0.0, 1.0, math.nan, -2.0, math.inf, 3.0]


def _random_table(rng: random.Random) -> tuple[list[Any], list[Any]]:
    """Signal groups and controllers, as namespaces with raw (unstripped)
    names: duplicates, blanks, unknown groups and ids, conflicts, and
    references that chain and loop."""
    group_names = [rng.choice(_NAMES) for _ in range(rng.randint(0, 5))]
    groups = [
        SimpleNamespace(
            name=name,
            lanelet2_regulatory_element_ids=[
                rng.randint(1, 6) for _ in range(rng.randint(0, 4))
            ],
            conflicts_with=[
                rng.choice(_NAMES + _UNKNOWN + group_names)
                for _ in range(rng.randint(0, 3))
            ],
        )
        for name in group_names
    ]
    subjects = group_names + _UNKNOWN + ["", " ", "a"]
    controller_names = [rng.choice(_NAMES[:7]) for _ in range(rng.randint(0, 6))]
    controllers = []
    for name in controller_names:
        phases = []
        for _ in range(rng.choice([0, 1, 1, 2, 3])):
            states = []
            for _ in range(rng.randint(0, 5)):
                group = rng.choice([None, None, *subjects])
                element = rng.choice([None, None, 1, 2, 3, 7])
                states.append(
                    SimpleNamespace(
                        group=group,
                        lanelet2_regulatory_element_id=element,
                        state=rng.choice(_STATES),
                    )
                )
            phases.append(
                SimpleNamespace(
                    name=rng.choice(["", " ", "p", "q", "p ", "r"]),
                    duration_seconds=rng.choice(_DURATIONS),
                    states=states,
                )
            )
        controllers.append(
            SimpleNamespace(
                name=name,
                phases=phases,
                delay_seconds=rng.choice(_DELAYS),
                reference=rng.choice(
                    [None, None] + controller_names + _UNKNOWN + [" a", "b "]
                ),
            )
        )
    return groups, controllers


def _plain(groups: list[Any], controllers: list[Any]) -> Any:
    return check_signal_table(
        [
            SignalGroup(
                name=group.name,
                regulatory_element_ids=list(group.lanelet2_regulatory_element_ids),
                conflicts_with=[str(other) for other in group.conflicts_with],
            )
            for group in groups
        ],
        [
            SignalController(
                name=controller.name,
                phases=[
                    SignalPhase(
                        name=phase.name,
                        duration_seconds=phase.duration_seconds,
                        states=[
                            SignalState(
                                group=entry.group,
                                regulatory_element_id=entry.lanelet2_regulatory_element_id,
                                state=str(entry.state),
                            )
                            for entry in phase.states
                        ],
                    )
                    for phase in controller.phases
                ],
                delay_seconds=controller.delay_seconds,
                reference=controller.reference,
            )
            for controller in controllers
        ],
        list(SIGNAL_STATE_NAMES),
    )


def _triples(issues: list[Issue]) -> list[tuple[str, str, str]]:
    assert all(issue.object_id is None for issue in issues)
    return [(issue.severity, issue.path, issue.message) for issue in issues]


def test_permissive_states_are_unchanged() -> None:
    assert PERMISSIVE_SIGNAL_STATES == _OLD_PERMISSIVE_SIGNAL_STATES


@pytest.mark.parametrize("seed", range(40))
def test_check_signal_table_matches_the_old_checks(seed: int) -> None:
    """Over raw names, as the old code took any object with the attributes."""
    rng = random.Random(seed)
    for _ in range(100):
        groups, controllers = _random_table(rng)
        document = SimpleNamespace(
            map=SimpleNamespace(
                signal_groups=groups, traffic_signal_controllers=controllers
            )
        )
        out = _Collector()
        _USES.clear()
        _old_check_signal_controllers(out, document)

        checked = _plain(groups, controllers)

        assert checked.issues == _triples(out.issues)
        old_by_name = _USES["by_name"]
        assert list(checked.controller_index) == list(old_by_name)
        for name, index in checked.controller_index.items():
            assert controllers[index] is old_by_name[name]


def _random_document(rng: random.Random) -> Any:
    """A starter document holding a random table, through the pydantic models
    (which strip the names and lower-case the states)."""
    groups, controllers = _random_table(rng)
    document = new_document()
    document.map.signal_groups = [
        SignalGroupRef(
            name=group.name,
            lanelet2_regulatory_element_ids=group.lanelet2_regulatory_element_ids,
            conflicts_with=group.conflicts_with,
        )
        for group in groups
    ]
    document.map.traffic_signal_controllers = [
        SignalControllerRef(
            name=controller.name,
            phases=[
                SignalPhaseRef(
                    name=phase.name,
                    duration_seconds=phase.duration_seconds,
                    states=[
                        SignalStateRef(
                            group=entry.group,
                            lanelet2_regulatory_element_id=entry.lanelet2_regulatory_element_id,
                            state=entry.state,
                        )
                        for entry in phase.states
                    ],
                )
                for phase in controller.phases
            ],
            delay_seconds=controller.delay_seconds,
            reference=controller.reference,
        )
        for controller in controllers
    ]
    return document


@pytest.mark.parametrize("seed", range(10))
def test_validator_reports_what_it_reported(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the validator: the same issues, and the same controllers handed
    on to the check of their uses."""
    rng = random.Random(1000 + seed)
    seen: dict[str, Any] = {}
    real_uses = validator._check_signal_uses

    def uses(out: Any, document: Any, by_name: dict[str, Any]) -> None:
        seen["by_name"] = dict(by_name)
        real_uses(out, document, by_name)

    monkeypatch.setattr(validator, "_check_signal_uses", uses)
    for _ in range(40):
        document = _random_document(rng)
        old = _Collector()
        _USES.clear()
        _USES["real"] = True
        _old_check_signal_controllers(old, document)
        new = _Collector()
        validator._check_signal_controllers(new, document)
        assert new.issues == old.issues
        assert list(seen["by_name"]) == list(_USES["by_name"])
        for name, controller in seen["by_name"].items():
            assert controller is _USES["by_name"][name]


def test_a_reference_loop_and_a_chain() -> None:
    """The cycle walk by index reports a loop once, keeping raw names."""
    controllers = [
        SimpleNamespace(name=name, phases=[], delay_seconds=0.0, reference=ref)
        for name, ref in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a"), ("e", None)]
    ]
    checked = _plain([], controllers)
    loops = [m for _, p, m in checked.issues if p == "map.traffic_signal_controllers"]
    assert loops == [
        "These signal controllers are offset from each other in a loop, so none "
        "of them can start: a -> b -> c -> a"
    ]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

_BOUNDS: list[Any] = [None, 0, 0.0, -0.0, 1, 2.5, -3, 100.0, 2147483647, 1e300, 10**30]
_NUMBERS: list[Any] = [
    0,
    -0.0,
    1,
    -1,
    2.5,
    99.99,
    100,
    2147483647,
    2147483648,
    10**30,
    10**400,
    -(10**400),
    math.nan,
    math.inf,
    -math.inf,
    1e-9,
]


def _spec(kind: str, low: Any, inclusive: Any, high: Any) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        exclusive_minimum=low,
        minimum=inclusive,
        maximum=high,
        name="field",
        label="Field",
    )


def test_bounds_match_the_old_checks() -> None:
    for kind, low, inclusive, high in itertools.product(
        ["int", "number", "seed"], _BOUNDS, _BOUNDS, _BOUNDS
    ):
        spec: Any = _spec(kind, low, inclusive, high)
        for number in _NUMBERS:
            old = _Collector()
            _old_check_bounds(old, "params", spec, number, "node")
            new = _Collector()
            validator._check_bounds(new, "params", spec, number, "node")
            assert new.issues == old.issues
            assert within_bounds(number, low, inclusive, high) == (not old.issues)
            if old.issues:
                assert bounds_text(kind == "int", low, inclusive, high) == (
                    _old_bounds_text(spec)
                )


def test_bound_text_matches_the_old_text() -> None:
    rng = random.Random(7)
    bounds: list[Any] = [b for b in _BOUNDS if b is not None] + _NUMBERS
    bounds += [rng.uniform(-1e6, 1e6) for _ in range(200)]
    bounds += [rng.randint(-(10**20), 10**20) for _ in range(200)]
    for kind in ["int", "number"]:
        for bound in bounds:
            assert _outcome(bound_text, kind == "int", bound) == _outcome(
                _old_bound_text, _spec(kind, None, None, None), bound
            )


def _outcome(function: Any, *args: Any) -> Any:
    """What *function* returns, or the type of what it raises (an int bound
    too large for a float raises ``OverflowError`` in both)."""
    try:
        return function(*args)
    except Exception as error:  # noqa: BLE001 -- compared, not swallowed
        return type(error)


def test_traffic_mean_bounds_are_unchanged() -> None:
    assert traffic_mean_bounds() == _old_traffic_mean_bounds()
    assert list(traffic_mean_bounds()) == list(_old_traffic_mean_bounds())
    assert validator._traffic_mean_bounds is traffic_mean_bounds


def test_is_whole_is_float_is_integer() -> None:
    """The Codon spelling of ``float.is_integer()``."""
    from autoware_carla_scenario.authoring.field_bounds import _is_whole

    rng = random.Random(11)
    values = [0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e300, -1e300, 2.0**53, 2.0**53 + 2]
    values += [math.inf, -math.inf, math.nan, 5e-324, 1e15 + 0.5]
    values += [rng.uniform(-1e3, 1e3) for _ in range(500)]
    values += [float(rng.randint(-(10**18), 10**18)) for _ in range(500)]
    for value in values:
        assert _is_whole(value) == value.is_integer()
