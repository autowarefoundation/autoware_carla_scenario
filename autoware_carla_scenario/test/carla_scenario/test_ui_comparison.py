"""Sessions compared case by case: ``ui.comparison``, the checked module
``ui.results`` hands its cases to (#787)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoware_carla_scenario.ui import comparison, results

from . import _compared_sessions

_GOLDEN = Path(__file__).parent / "data" / "ui_results" / "comparison_golden.json"

SPEED = "dynamic.speed"
GAP = "dynamic.gap"


def _case(
    index: int,
    label: str,
    passed: bool | None,
    *,
    errored: bool = False,
    buckets: dict[str, str] | None = None,
    metrics: dict[str, float] | None = None,
    key: str | None = None,
) -> comparison.ComparedCase:
    return comparison.ComparedCase(
        index=index,
        label=label,
        passed=passed,
        errored=errored,
        message=f"{label} at {index}",
        overrides=[],
        buckets=dict(buckets or {}),
        metrics=dict(metrics or {}),
        key=key or label,
    )


def test_a_case_s_outcome_and_scenario() -> None:
    assert _case(0, "pkg/cut_in#3", True).outcome == "pass"
    assert _case(0, "pkg/cut_in#3", False).outcome == "fail"
    assert _case(0, "pkg/cut_in#3", None).outcome == "none"
    assert _case(0, "pkg/cut_in#3", True, errored=True).outcome == "error"
    assert _case(0, "pkg/cut_in#3", None).scenario == "pkg/cut_in"
    assert _case(0, "pkg/cut_in", None).scenario == "pkg/cut_in"


def test_the_axes_are_the_attributes_with_the_most_buckets() -> None:
    cases = [
        _case(0, "a", True, buckets={GAP: "[0, 5)", SPEED: "[0, 10)", "w": "x"}),
        _case(1, "b", True, buckets={GAP: "[5, 10)", SPEED: "[10, 20)", "w": "x"}),
        _case(2, "c", True, buckets={SPEED: "[20, 30)"}),
    ]
    assert comparison.varying_attributes(cases) == [GAP, SPEED]
    assert comparison.default_axes(cases) == (SPEED, GAP)
    assert comparison.default_axes(cases[:1]) == (None, None)
    assert comparison.default_axes([]) == (None, None)


def test_a_change_is_read_between_the_first_and_last_runs() -> None:
    def change(*outcomes: bool | None | str) -> str:
        runs: list[comparison.ComparedCase | None] = []
        for i, outcome in enumerate(outcomes):
            if outcome == "-":
                runs.append(None)
            elif outcome == "error":
                runs.append(_case(i, "a", None, errored=True))
            else:
                runs.append(_case(i, "a", outcome))  # type: ignore[arg-type]
        return comparison.MatchedCase("a", runs).change()

    assert change(True, False) == "regression"
    assert change(True, "error") == "regression"
    assert change(False, True) == "fixed"
    assert change(True, False, True) == "flaky"
    assert change(False, True, "error") == "flaky"
    assert change(False, "error") == "same"
    assert change(True, "-", True) == "same"
    assert change(True, "-") == "gone"
    assert change("-", True) == "new"


def test_a_metric_delta_and_whether_it_is_more_than_noise() -> None:
    runs: list[comparison.ComparedCase | None] = [
        _case(0, "a", True, metrics={"min_gap_m": 2.0, "elapsed_s": 10.0}),
        None,
        _case(2, "a", True, metrics={"min_gap_m": 1.0, "elapsed_s": 10.4}),
    ]
    matched = comparison.MatchedCase("a", runs)
    assert matched.first is runs[0]
    assert matched.ran == [runs[0], runs[2]]
    assert matched.delta("min_gap_m") == -1.0
    assert matched.significant("min_gap_m")
    # 0.4 s of 10 s: under 5%.
    assert matched.delta("elapsed_s") == pytest.approx(0.4)
    assert not matched.significant("elapsed_s")
    assert matched.delta("distance_m") is None
    assert not matched.significant("distance_m")


def test_a_match_no_session_ran_has_no_first() -> None:
    with pytest.raises(StopIteration):
        comparison.MatchedCase("a", [None, None]).first


def test_buckets_read_in_order_numeric_ranges_by_their_lower_bound() -> None:
    labels = {"[10, 20)", "(-inf, -10)", "[-10, 0)", "named", "[0, 10)", "abc"}
    assert comparison._bucket_order(labels) == [
        "(-inf, -10)",
        "[-10, 0)",
        "[0, 10)",
        "[10, 20)",
        "abc",
        "named",
    ]
    assert comparison._case_number("pkg/cut_in#10") == (10, "pkg/cut_in#10")
    assert comparison._case_number("pkg/cut_in") == (0, "pkg/cut_in")


def test_sessions_are_compared_by_key_and_by_cell() -> None:
    base = [
        _case(0, "s#1", True, buckets={GAP: "[0, 5)"}, metrics={"elapsed_s": 1.0}),
        _case(1, "s#2", True, buckets={GAP: "[5, 10)"}),
        _case(2, "s#3", False),
    ]
    new = [
        _case(0, "s#2", False, buckets={GAP: "[5, 10)"}),
        _case(1, "s#1", True, buckets={GAP: "[0, 5)"}),
        _case(2, "s#4", True, buckets={GAP: "[0, 5)"}),
    ]
    compared = comparison.compare([("a", base), ("b", new)], x=GAP)
    assert [(c.key, c.change()) for c in compared.cases] == [
        ("s#2", "regression"),
        ("s#4", "new"),
        ("s#3", "gone"),
        ("s#1", "same"),
    ]
    assert compared.count("regression") == 1
    assert compared.totals == [(2, 3), (2, 3)]
    assert compared.rate(0) == pytest.approx(2 / 3)
    assert compared.x_labels == ["[0, 5)", "[5, 10)"]
    assert compared.y_labels == [""]
    # s#3 was drawn in no bucket of the gap: in no cell.
    assert compared.cells[("[0, 5)", "")].tallies == [(1, 1), (2, 2)]
    assert compared.cells[("[5, 10)", "")].delta == -1.0
    assert compared.cells[("[0, 5)", "")].keys == ["s#1", "s#4"]
    assert compared.metrics == ["elapsed_s"]


def test_results_re_exports_the_comparison() -> None:
    for name in (
        "METRICS",
        "CellComparison",
        "ComparedCase",
        "Comparison",
        "MatchedCase",
        "_bucket_order",
        "_case_number",
    ):
        assert getattr(results, name) is getattr(comparison, name), name


def test_a_case_result_is_compared_by_its_buckets_and_key() -> None:
    result = results.CaseResult(
        index=3,
        label="pkg/cut_in#4",
        passed=False,
        errored=False,
        elapsed_seconds=6.0,
        message="failed",
        overrides=["ego.speed=3", "++odd_case=x"],
        files=None,
        coverage={"case": {"buckets": {GAP: "[0, 5)"}, "values": {"a": 1}}},
        metrics={"elapsed_s": 6.0},
    )
    compared = result.compared()
    assert compared.buckets == result.buckets == {GAP: "[0, 5)"}
    assert compared.key == result.key == "pkg/cut_in|ego.speed=3"
    assert compared.outcome == result.outcome == "fail"
    assert compared.scenario == result.scenario
    assert (compared.index, compared.label, compared.message) == (
        3,
        "pkg/cut_in#4",
        "failed",
    )
    assert compared.metrics == {"elapsed_s": 6.0}


def test_the_comparison_gives_what_it_gave_before_it_was_split_out() -> None:
    """``data/ui_results/comparison_golden.json`` is what ``ui.results``'
    own comparison gave on these sessions before #787 moved it into
    ``ui.comparison``: everything ``compare.html`` reads of it."""
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(_compared_sessions.snapshots(), ensure_ascii=False))
    assert now.keys() == golden.keys()
    for name, snapshot in golden.items():
        assert now[name] == snapshot, name


def test_the_golden_comparison_covers_every_change() -> None:
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    counts = golden["None|None"]["counts"]
    assert all(
        counts[c] for c in ("regression", "fixed", "flaky", "new", "gone", "same")
    )
