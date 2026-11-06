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


def _runs(*sessions: object) -> list[list[comparison.ComparedCase]]:
    """Each session's runs of key ``a``: ``"-"`` none, a verdict (``True``,
    ``False``, ``None``, ``"error"``) one run, a tuple of them several."""
    out: list[list[comparison.ComparedCase]] = []
    for i, outcomes in enumerate(sessions):
        group = outcomes if isinstance(outcomes, tuple) else (outcomes,)
        runs: list[comparison.ComparedCase] = []
        for j, outcome in enumerate(group):
            if outcome == "-":
                continue
            if outcome == "error":
                runs.append(_case(10 * i + j, "a", None, errored=True))
            else:
                runs.append(_case(10 * i + j, "a", outcome))  # type: ignore[arg-type]
        out.append(runs)
    return out


def _change(*sessions: object) -> str:
    return comparison.MatchedCase("a", _runs(*sessions)).change()


def test_a_change_is_read_between_the_first_and_last_runs() -> None:
    assert _change(True, False) == "regression"
    assert _change(True, "error") == "regression"
    assert _change(False, True) == "fixed"
    assert _change(True, False, True) == "flaky"
    assert _change(False, True, "error") == "flaky"
    assert _change(False, "error") == "same"
    assert _change(True, "-", True) == "same"
    assert _change(True, "-") == "gone"
    assert _change("-", True) == "new"


def test_a_change_with_several_runs_in_a_session_counts_every_run() -> None:
    """A session's verdict on a key is read off every run of it: ``pass``
    when all passed, ``fail`` when none did, ``mixed`` (flaky) otherwise
    (#1825)."""
    # Every run passed, then none did: a regression; the other way: fixed.
    assert _change((True, True), (False, "error")) == "regression"
    assert _change((False, None), (True, True, True)) == "fixed"
    # The same verdict on every run of every session: the same.
    assert _change((True, True), True) == "same"
    assert _change((False, "error"), (False,)) == "same"
    # A session that both passed and failed it: flaky, whichever it is, and
    # whatever the others did -- even when it alone ran it.
    assert _change((False, True), (True, True)) == "flaky"
    assert _change((True, True), (True, False)) == "flaky"
    assert _change(True, (True, False), True) == "flaky"
    assert _change((True, False), "-") == "flaky"
    assert _change("-", ("error", True)) == "flaky"
    # Consistent sessions that disagree between the ends: flaky.
    assert _change((True, True), (False, False), (True,)) == "flaky"
    # One session's runs alone, each the same: new or gone.
    assert _change((True, True), "-") == "gone"
    assert _change("-", (False, False)) == "new"


def test_a_session_s_tally_and_verdict_on_a_key() -> None:
    matched = comparison.MatchedCase("a", _runs((False, True), "-", (True,), "error"))
    assert [matched.tally(i) for i in range(4)] == [(1, 2), (0, 0), (1, 1), (0, 1)]
    assert [matched.verdict(i) for i in range(4)] == ["mixed", "", "pass", "fail"]
    # The first run of the first session; each session's last.
    assert matched.first is matched.runs[0][0]
    assert matched.latest == [
        matched.runs[0][1],
        None,
        matched.runs[2][0],
        matched.runs[3][0],
    ]
    assert matched.ran == [*matched.runs[0], *matched.runs[2], *matched.runs[3]]


def test_a_metric_delta_and_whether_it_is_more_than_noise() -> None:
    runs: list[list[comparison.ComparedCase]] = [
        [_case(0, "a", True, metrics={"min_gap_m": 2.0, "elapsed_s": 10.0})],
        [],
        [_case(2, "a", True, metrics={"min_gap_m": 1.0, "elapsed_s": 10.4})],
    ]
    matched = comparison.MatchedCase("a", runs)
    assert matched.first is runs[0][0]
    assert matched.ran == [runs[0][0], runs[2][0]]
    assert matched.delta("min_gap_m") == -1.0
    assert matched.significant("min_gap_m")
    # 0.4 s of 10 s: under 5%.
    assert matched.delta("elapsed_s") == pytest.approx(0.4)
    assert not matched.significant("elapsed_s")
    assert matched.delta("distance_m") is None
    assert not matched.significant("distance_m")


def test_a_metric_delta_is_between_the_sessions_means() -> None:
    runs: list[list[comparison.ComparedCase]] = [
        [
            _case(0, "a", True, metrics={"min_gap_m": 1.0}),
            _case(1, "a", True, metrics={"min_gap_m": 3.0}),
            _case(2, "a", True),
        ],
        [_case(0, "a", True, metrics={"min_gap_m": 2.5})],
    ]
    matched = comparison.MatchedCase("a", runs)
    # Means 2.0 and 2.5; a run without the metric is left out.
    assert matched.delta("min_gap_m") == 0.5
    assert matched.significant("min_gap_m")


def test_a_match_no_session_ran_has_no_first() -> None:
    with pytest.raises(StopIteration):
        comparison.MatchedCase("a", [[], []]).first


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
    # #1612: "²".isdigit() is true but int("²") raises.
    assert comparison._case_number("pkg/cut_in#²") == (0, "pkg/cut_in#²")


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


def _cell_sums(compared: comparison.Comparison) -> list[tuple[int, int]]:
    sums = [(0, 0) for _ in compared.sessions]
    for cell in compared.cells.values():
        for i, (passed, runs) in enumerate(cell.tallies):
            sums[i] = (sums[i][0] + passed, sums[i][1] + runs)
    return sums


def test_every_run_of_a_key_a_session_repeats_is_kept_and_tallied() -> None:
    """#1825: a session that ran a key twice, failing then passing, keeps
    both runs; its cell tallies both, so the cells add up to the totals."""
    base = [
        _case(0, "s#1", False, buckets={GAP: "[0, 5)"}, key="k1"),
        _case(1, "s#2", True, buckets={GAP: "[5, 10)"}, key="k2"),
        _case(2, "s#1", True, buckets={GAP: "[0, 5)"}, key="k1"),
    ]
    new = [
        _case(0, "s#1", True, buckets={GAP: "[0, 5)"}, key="k1"),
        _case(1, "s#2", True, buckets={GAP: "[5, 10)"}, key="k2"),
    ]
    compared = comparison.compare([("a", base), ("b", new)], x=GAP)
    k1 = next(c for c in compared.cases if c.key == "k1")
    # The failure is not dropped.
    assert [r.outcome for r in k1.runs[0]] == ["fail", "pass"]
    assert k1.tally(0) == (1, 2)
    assert k1.change() == "flaky"
    assert compared.count("flaky") == 1
    assert compared.cells[("[0, 5)", "")].tallies == [(1, 2), (1, 1)]
    assert compared.cells[("[0, 5)", "")].keys == ["k1"]
    assert compared.totals == [(2, 3), (2, 2)]
    assert _cell_sums(compared) == compared.totals


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


def test_a_case_s_outcome_is_read_off_its_verdict() -> None:
    assert comparison.outcome_of(False, True) == "pass"
    assert comparison.outcome_of(False, False) == "fail"
    assert comparison.outcome_of(False, None) == "none"
    for passed in (True, False, None):
        assert comparison.outcome_of(True, passed) == "error"
    assert comparison.scenario_of("pkg/cut_in#3") == "pkg/cut_in"
    assert comparison.scenario_of("pkg/cut_in") == "pkg/cut_in"


def _key_before_1721(scenario: str, overrides: list[str]) -> str | None:
    """``CaseResult.key``'s override part as ``ui.results`` built it before
    it moved to ``ui.comparison.override_key`` (#1721)."""
    own = sorted(
        o.lstrip("+")
        for o in overrides
        if not o.lstrip("+").startswith(("odd_case=",))
        and not o.startswith(("hydra.", "hydra/"))
    )
    return f"{scenario}|" + "|".join(own) if own else None


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["ego.speed=3"],
        ["b=2", "a=1", "+c=3", "++d=4"],
        ["odd_case=abc"],
        ["+odd_case=abc", "++odd_case=abc"],
        ["hydra.run.dir=out", "hydra/launcher=slurm"],
        ["+hydra.job.name=x"],
        ["ego.speed=3", "++odd_case=x", "hydra.run.dir=o", "hydra/sweeper=basic"],
        ["z=1", "+hydra.verbose=true", "odd_case=7", "a=b|c", "scenario=pkg/x"],
    ],
)
def test_a_case_s_key_by_overrides_is_what_it_was(overrides: list[str]) -> None:
    """Sessions are matched by key: the moved code gives the same bytes."""
    expected = _key_before_1721("pkg/cut_in", overrides)
    assert comparison.override_key("pkg/cut_in", overrides) == expected
    result = results.CaseResult(
        index=0,
        label="pkg/cut_in#2",
        passed=None,
        errored=False,
        elapsed_seconds=None,
        message="",
        overrides=overrides,
        files=None,
        coverage={"case": {"values": {"b": 2, "a": [1, "x"]}}},
    )
    fallback = 'pkg/cut_in|{"a": [1, "x"], "b": 2}'
    assert result.key == (expected if expected is not None else fallback)


def test_the_comparison_gives_what_it_gave_before_it_was_split_out() -> None:
    """``data/ui_results/comparison_golden.json`` is what ``ui.results``'
    own comparison gave on these sessions before #787 moved it into
    ``ui.comparison``: everything ``compare.html`` reads of it. Regenerated
    for #1825, which keeps every run of a key a session ran twice (``a2``
    of the second session) instead of its last."""
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
