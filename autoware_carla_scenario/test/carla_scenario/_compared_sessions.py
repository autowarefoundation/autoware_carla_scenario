"""Sessions to compare, and what a comparison of them gives, as plain data.

:func:`sessions` builds three sessions of ``ui.results.CaseResult`` that
cover what the comparison tells apart: cases matched by their overrides, by
the ODD values they were drawn at and by their label; every outcome; a case
run twice in one session; a case not drawn in a bucket of an axis; numeric
and named buckets; and metrics whose deltas are and are not significant.
:func:`snapshot` writes out everything ``compare.html`` reads of a
comparison, and the axes, for ``data/ui_results/comparison_golden.json``:
the comparison's output from before it was split out of ``ui.results``
(#787), which it must still give.
"""

from __future__ import annotations

from typing import Any

from autoware_carla_scenario.ui import results

SPEED = "dynamic.speed"
GAP = "dynamic.gap"
WEATHER = "scenery.weather"


def _case(
    index: int,
    label: str,
    passed: bool | None,
    *,
    errored: bool = False,
    overrides: list[str] | None = None,
    buckets: dict[str, str] | None = None,
    values: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> results.CaseResult:
    case: dict[str, Any] = {}
    if buckets is not None:
        case["buckets"] = buckets
    if values is not None:
        case["values"] = values
    return results.CaseResult(
        index=index,
        label=label,
        passed=passed,
        errored=errored,
        elapsed_seconds=None,
        message=f"{label} at {index}",
        overrides=list(overrides or []),
        files=None,
        coverage={"case": case} if case else None,
        metrics=dict(metrics or {}),
    )


def _session(outcomes: dict[str, bool | None], order: list[str]) -> list[Any]:
    """The cases named in *order*, each passing, failing or (``None``)
    erroring as *outcomes* says."""
    out = []
    for index, name in enumerate(order):
        out.append(_CASES[name](index, outcomes.get(name)))
    return out


def _by_overrides(number: int, speed: str, gap: str, weather: str) -> Any:
    def make(index: int, passed: bool | None) -> results.CaseResult:
        return _case(
            index,
            f"pkg/cut_in#{number}",
            passed,
            errored=passed is None,
            overrides=[
                f"ego.speed={number}",
                f"++odd_case=position{index}",
                "hydra.job.chdir=true",
                f"+npc.gap={number * 2}",
            ],
            buckets={SPEED: speed, GAP: gap, WEATHER: weather},
            metrics={
                "elapsed_s": 5.0 + number + index * 0.01,
                "min_gap_m": 2.0 + (index if number % 2 else 0) * 0.5,
                "max_speed_kmh": 30.0 + number,
            },
        )

    return make


def _by_values(number: int, speed: str, weather: str) -> Any:
    def make(index: int, passed: bool | None) -> results.CaseResult:
        return _case(
            index,
            f"pkg/follow#{number}",
            passed,
            buckets={SPEED: speed, WEATHER: weather},
            values={"speed": number * 1.5, "weather": weather},
            metrics={"distance_m": 100.0 + number * index},
        )

    return make


def _by_label(label: str) -> Any:
    def make(index: int, passed: bool | None) -> results.CaseResult:
        return _case(index, label, passed)

    return make


_CASES: dict[str, Any] = {
    "a1": _by_overrides(1, "[0, 10)", "[0, 5)", "clear"),
    "a2": _by_overrides(2, "[10, 20)", "[0, 5)", "rain"),
    "a10": _by_overrides(10, "[-10, 0)", "[5, 10)", "clear"),
    "a3": _by_overrides(3, "[0, 10)", "[5, 10)", "fog"),
    "a4": _by_overrides(4, "(-inf, -10)", "[0, 5)", "rain"),
    "b1": _by_values(1, "[10, 20)", "clear"),
    "b2": _by_values(2, "[0, 10)", "fog"),
    "c1": _by_label("pkg/plain"),
    "c2": _by_label("pkg/plain#7"),
}


def sessions() -> list[tuple[str, list[Any]]]:
    """Three sessions, oldest first."""
    first = _session(
        {"a1": True, "a2": True, "a10": False, "a3": True, "b1": False, "c1": True},
        ["a10", "a1", "a2", "a3", "b1", "c1", "c2"],
    )
    second = _session(
        {"a1": False, "a2": True, "a10": True, "a3": None, "a4": True, "b2": True},
        ["a4", "a3", "a2", "a1", "a10", "b2", "a2"],
    )
    third = _session(
        {"a1": True, "a2": False, "a10": True, "a3": True, "b1": True, "b2": True},
        ["b1", "b2", "a1", "a2", "a10", "a3", "c2"],
    )
    return [
        ("batch/2026-10-18/10-00-00", first),
        ("batch/2026-10-18/11-00-00", second),
        ("batch/2026-10-18/12-00-00", third),
    ]


def _run(run: Any) -> Any:
    if run is None:
        return None
    return [
        run.index,
        run.label,
        run.outcome,
        run.message,
        run.scenario,
        run.key,
        sorted(run.buckets.items()),
    ]


def snapshot(comparison: Any) -> dict[str, Any]:
    """What ``compare.html`` reads of *comparison*, as JSON."""
    count = len(comparison.sessions)
    return {
        "sessions": comparison.sessions,
        "x": comparison.x,
        "y": comparison.y,
        "x_labels": comparison.x_labels,
        "y_labels": comparison.y_labels,
        "totals": [list(t) for t in comparison.totals],
        "rates": [comparison.rate(i) for i in range(count)],
        "counts": {
            change: comparison.count(change)
            for change in ("regression", "fixed", "flaky", "new", "gone", "same")
        },
        "metrics": comparison.metrics,
        "cells": [
            [
                list(at),
                cell.x,
                cell.y,
                [list(t) for t in cell.tallies],
                cell.keys,
                cell.delta,
                [cell.rate(i) for i in range(count)],
            ]
            for at, cell in comparison.cells.items()
        ],
        "cases": [
            {
                "key": case.key,
                "change": case.change(),
                "first": _run(case.first),
                "runs": [_run(run) for run in case.runs],
                "ran": [_run(run) for run in case.ran],
                "delta": {m: case.delta(m) for m in results.METRICS},
                "significant": {m: case.significant(m) for m in results.METRICS},
            }
            for case in comparison.cases
        ],
    }


def snapshots() -> dict[str, Any]:
    """The comparisons of :func:`sessions` along each pair of axes, and the
    axes they default to."""
    every = [r for _, found in sessions() for r in found]
    out: dict[str, Any] = {
        "default_axes": list(results.default_axes(every)),
        "varying_attributes": results.varying_attributes(every),
        "default_axes_of_one": [
            list(results.default_axes(found)) for _, found in sessions()
        ],
    }
    axes: list[tuple[str | None, str | None]] = [
        (None, None),
        (GAP, SPEED),
        (WEATHER, None),
        (None, WEATHER),
        ("", ""),
        ("no.such", SPEED),
    ]
    for x, y in axes:
        out[f"{x}|{y}"] = snapshot(results.compare(sessions(), x=x, y=y))
    out["two"] = snapshot(results.compare(sessions()[1:]))
    out["one"] = snapshot(results.compare(sessions()[:1]))
    out["none"] = snapshot(results.compare([]))
    return out
