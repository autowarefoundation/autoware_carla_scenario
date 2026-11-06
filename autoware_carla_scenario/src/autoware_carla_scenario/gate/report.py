"""The typed core of ``scenario-gate``'s report: its Markdown, and what changed.

The parts of :mod:`.evaluate` that read only the checked :mod:`.criteria`
types and the per-run records (:mod:`.record`): a verdict's and a
comparison's Markdown, a criterion's cell in a comparison, the criteria and
the cases compared from a baseline's runs to a candidate's, and the records
grouped by scenario.  :mod:`.evaluate`'s ``GateVerdict`` and
``GateComparison`` hold a pydantic ``GateSpec`` and delegate to these.  See
``docs/gate.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from .criteria import (
    CaseChange,
    CriterionChange,
    CriterionResult,
    _number,
    _warning_lines,
)
from .record import GateRecord

__all__ = [
    "case_changes",
    "comparison_markdown",
    "criterion_cell",
    "criterion_changes",
    "records_by_scenario",
    "verdict_markdown",
]


def verdict_markdown(
    scenario: str,
    title: str,
    passed: bool,
    passed_runs: int,
    runs: int,
    criteria: Sequence[CriterionResult],
    warnings: Sequence[str],
    errors: Sequence[str],
) -> str:
    """A logical scenario's gate, judged over its runs, as Markdown: the
    criteria's table, then the warnings and the errors."""
    mark = "PASSED" if passed else "FAILED"
    lines = [
        f"## {title or scenario} (`{scenario}`): {mark}",
        "",
        f"{passed_runs} of {runs} run(s) passed.",
        "",
        "| Criterion | Measured | Required | Result |",
        "|---|---|---|---|",
    ]
    for c in criteria:
        value = "-" if c.value is None else _number(c.value)
        result = "pass" if c.passed else "**FAIL**"
        detail = f" ({c.detail})" if c.detail else ""
        lines.append(
            f"| {c.name} | {value}{detail} | {c.comparison} "
            f"{_number(c.threshold)} | {result} |"
        )
    lines += _warning_lines(warnings)
    for e in errors:
        lines.append(f"- error: {e}")
    return "\n".join(lines)


def criterion_cell(result: Optional[CriterionResult]) -> str:
    """One side of a criterion in a comparison's table: its value, marked when
    it fails; ``-`` when that side did not judge it."""
    if result is None:
        return "-"
    value = "n/a" if result.value is None else _number(result.value)
    return value if result.passed else f"{value} (fail)"


def comparison_markdown(
    scenario: str,
    title: str,
    baseline: Optional[tuple[int, int, bool]],
    candidate: Optional[tuple[int, int, bool]],
    regressed: bool,
    criteria: Sequence[CriterionChange],
    newly_failing: Sequence[CaseChange],
    fixed: Sequence[CaseChange],
    warnings: Sequence[str],
) -> str:
    """One logical scenario's gate on a baseline's runs and a candidate's, as
    Markdown.

    *baseline* and *candidate* are each side's ``(passed runs, runs, whether
    the gate passed)``, ``None`` when that side has no run.
    """
    if candidate is None:
        verdict = "NOT RUN"
    else:
        verdict = "PASSED" if candidate[2] else "FAILED"
    trend = "REGRESSED" if regressed else "no regression"
    lines = [
        f"## {title or scenario} (`{scenario}`): {verdict}, {trend}",
        "",
    ]
    for name, side in (("Baseline", baseline), ("Candidate", candidate)):
        if side is None:
            lines.append(f"- {name}: no run")
        else:
            mark = "passed" if side[2] else "failed"
            lines.append(f"- {name}: {side[0]} of {side[1]} run(s) passed; gate {mark}")
    lines += [
        "",
        "| Criterion | Required | Baseline | Candidate | Change |",
        "|---|---|---|---|---|",
    ]
    for c in criteria:
        ref = c.candidate if c.candidate is not None else c.baseline
        assert ref is not None  # noqa: S101 -- one side judged it
        required = f"{ref.comparison} {_number(ref.threshold)}"
        change = c.change
        if change == "regressed":
            change = "**regressed**"
        lines.append(
            f"| {c.name} | {required} | {criterion_cell(c.baseline)} | "
            f"{criterion_cell(c.candidate)} | {change} |"
        )
    for heading, cases in (
        ("Newly failing cases", newly_failing),
        ("Fixed cases", fixed),
    ):
        if cases:
            lines += ["", f"{heading}:", ""]
            for case in cases:
                lines.append(
                    f"- {case.label()}: baseline {case.baseline[0]}/"
                    f"{case.baseline[1]} passed, candidate {case.candidate[0]}/"
                    f"{case.candidate[1]}"
                )
    lines += _warning_lines(warnings)
    return "\n".join(lines)


def criterion_changes(
    baseline: Sequence[CriterionResult], candidate: Sequence[CriterionResult]
) -> list[CriterionChange]:
    """Each criterion either side judged, compared: in the candidate's order,
    then what only the baseline judged."""
    before_by_name: dict[str, CriterionResult] = {}
    for c in baseline:
        before_by_name[c.name] = c
    after_by_name: dict[str, CriterionResult] = {}
    for c in candidate:
        after_by_name[c.name] = c
    order: list[str] = list(after_by_name)
    for name in before_by_name:
        if name not in after_by_name:
            order.append(name)
    out: list[CriterionChange] = []
    for name in order:
        out.append(
            CriterionChange(name, before_by_name.get(name), after_by_name.get(name))
        )
    return out


def case_changes(
    baseline: Mapping[tuple[str, tuple[int, ...]], tuple[int, int]],
    candidate: Mapping[tuple[str, tuple[int, ...]], tuple[int, int]],
) -> tuple[list[CaseChange], list[CaseChange], int]:
    """The cases both sides ran whose outcome changed, from each side's
    ``(passed runs, runs)`` of each case (``criteria._outcomes``).

    Returns:
        The cases every baseline run of passed and a candidate run of failed
        (newly failing), those a baseline run of failed and every candidate
        run of passed (fixed), each sorted by case, and how many cases the
        baseline ran that the candidate did not.
    """
    common: list[tuple[str, tuple[int, ...]]] = []
    missing = 0
    for key in baseline:
        if key in candidate:
            common.append(key)
        else:
            missing += 1
    common.sort()
    newly_failing: list[CaseChange] = []
    fixed: list[CaseChange] = []
    for key in common:
        b, a = baseline[key], candidate[key]
        change = CaseChange(key[0], key[1], b, a)
        if b[0] == b[1] and a[0] < a[1]:
            newly_failing.append(change)
        elif b[0] < b[1] and a[0] == a[1]:
            fixed.append(change)
    return newly_failing, fixed, missing


def records_by_scenario(
    records: Sequence[GateRecord], scenario: Optional[str] = None
) -> dict[str, list[GateRecord]]:
    """*records* grouped by scenario id, in the order each id first appears;
    only *scenario*'s when it is given."""
    groups: dict[str, list[GateRecord]] = {}
    for record in records:
        if scenario is None or record.scenario == scenario:
            if record.scenario not in groups:
                groups[record.scenario] = []
            groups[record.scenario].append(record)
    return groups
