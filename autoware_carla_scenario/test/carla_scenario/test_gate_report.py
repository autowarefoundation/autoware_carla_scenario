"""Golden tests of the gate's reports (``scenario-gate``'s Markdown) and of
how ``compare`` and the grouping by scenario read the records.

The expected values were captured from the code before it moved into the
checked ``gate.report`` (#1738), so they pin its output byte for byte.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from autoware_carla_scenario.authoring.models import GateSpec
from autoware_carla_scenario.gate.evaluate import (
    CaseChange,
    CriterionChange,
    CriterionResult,
    GateComparison,
    GateVerdict,
    _by_scenario,
    _cell,
    compare,
    compare_all,
)
from autoware_carla_scenario.gate.record import GateCase, GateRecord


def _rec(
    *,
    scenario: str = "left_turn",
    passed: bool = True,
    lanelets: tuple[int, ...] = (1, 2),
    map_name: str = "Town",
    digest: str = "d",
    gate: Optional[dict[str, Any]] = None,
) -> GateRecord:
    return GateRecord(
        scenario=scenario,
        title="Left turn",
        digest=digest * 64,
        gate=gate if gate is not None else GateSpec(min_success_rate=0.5).to_dict(),
        case=GateCase(
            map=map_name,
            index=0,
            lanelet_ids=lanelets,
            case_count=4,
            segment_kinds=("lane",),
            segment_ends=(100.0,),
        ),
        passed=passed,
        message="",
        elapsed_seconds=1.0,
        failure_category=None if passed else "collision",
        ego=[(0.0, 0.0, 0.0, 10.0), (0.1, 100.0, 0.0, 10.0)],
    )


_OK = CriterionResult("min_runs", True, 3.0, 2.0, ">=", "3 run(s)")
_BAD = CriterionResult("min_success_rate", False, 1 / 3, 0.9, ">=", "1 of 3 passed")
_NONE = CriterionResult("max_lateral_spread_m", False, None, 0.25, "<=", "")
_GOOD = CriterionResult("min_success_rate", True, 0.95, 0.9, ">=", "19 of 20 passed")
_SPREAD = CriterionResult("max_lateral_spread_m", True, 0.123456, 0.25, "<=", "")


def _passed_verdict() -> GateVerdict:
    return GateVerdict(
        "s1",
        "First",
        GateSpec(),
        [_rec(), _rec()],
        criteria=[_OK, _GOOD],
        warnings=["w one", "w two"],
        errors=["e one"],
    )


def _failed_verdict() -> GateVerdict:
    return GateVerdict(
        "s2",
        "",
        GateSpec(),
        [_rec(), _rec(passed=False), _rec(passed=False)],
        criteria=[_OK, _BAD, _NONE],
    )


def _empty_verdict() -> GateVerdict:
    return GateVerdict("s3", "Empty", GateSpec(), [])


class TestVerdictMarkdown:
    def test_passed_with_warnings_and_errors(self) -> None:
        assert _passed_verdict().to_markdown() == (
            "## First (`s1`): PASSED\n"
            "\n"
            "2 of 2 run(s) passed.\n"
            "\n"
            "| Criterion | Measured | Required | Result |\n"
            "|---|---|---|---|\n"
            "| min_runs | 3 (3 run(s)) | >= 2 | pass |\n"
            "| min_success_rate | 0.95 (19 of 20 passed) | >= 0.9 | pass |\n"
            "\n"
            "- warning: w one\n"
            "- warning: w two\n"
            "- error: e one"
        )

    def test_failed_untitled_unmeasured(self) -> None:
        assert _failed_verdict().to_markdown() == (
            "## s2 (`s2`): FAILED\n"
            "\n"
            "1 of 3 run(s) passed.\n"
            "\n"
            "| Criterion | Measured | Required | Result |\n"
            "|---|---|---|---|\n"
            "| min_runs | 3 (3 run(s)) | >= 2 | pass |\n"
            "| min_success_rate | 0.3333 (1 of 3 passed) | >= 0.9 | **FAIL** |\n"
            "| max_lateral_spread_m | - | <= 0.25 | **FAIL** |"
        )

    def test_empty(self) -> None:
        assert _empty_verdict().to_markdown() == (
            "## Empty (`s3`): PASSED\n"
            "\n"
            "0 of 0 run(s) passed.\n"
            "\n"
            "| Criterion | Measured | Required | Result |\n"
            "|---|---|---|---|"
        )


class TestCell:
    @pytest.mark.parametrize(
        "result, expected",
        [
            (None, "-"),
            (_OK, "3"),
            (_BAD, "0.3333 (fail)"),
            (_NONE, "n/a (fail)"),
            (CriterionResult("x", True, None, 1.0, ">="), "n/a"),
        ],
    )
    def test_cell(self, result: Optional[CriterionResult], expected: str) -> None:
        assert _cell(result) == expected


class TestComparisonMarkdown:
    def test_regressed_with_cases_and_warnings(self) -> None:
        comparison = GateComparison(
            "s1",
            "First",
            baseline=_passed_verdict(),
            candidate=_failed_verdict(),
            criteria=[
                CriterionChange("min_runs", _OK, _OK),
                CriterionChange("min_success_rate", _GOOD, _BAD),
                CriterionChange("max_lateral_spread_m", None, _NONE),
                CriterionChange("max_speed_spread_mps", _SPREAD, None),
            ],
            newly_failing=[CaseChange("Town", (1, 2), (2, 2), (1, 3))],
            fixed=[
                CaseChange("", (7,), (0, 1), (2, 2)),
                CaseChange("Other", (), (1, 2), (1, 1)),
            ],
            warnings=["careful"],
        )
        assert comparison.to_markdown() == (
            "## First (`s1`): FAILED, REGRESSED\n"
            "\n"
            "- Baseline: 2 of 2 run(s) passed; gate passed\n"
            "- Candidate: 1 of 3 run(s) passed; gate failed\n"
            "\n"
            "| Criterion | Required | Baseline | Candidate | Change |\n"
            "|---|---|---|---|---|\n"
            "| min_runs | >= 2 | 3 | 3 | same |\n"
            "| min_success_rate | >= 0.9 | 0.95 | 0.3333 (fail) | **regressed** |\n"
            "| max_lateral_spread_m | <= 0.25 | - | n/a (fail) | new |\n"
            "| max_speed_spread_mps | <= 0.25 | 0.1235 | - | dropped |\n"
            "\n"
            "Newly failing cases:\n"
            "\n"
            "- Town [1,2]: baseline 2/2 passed, candidate 1/3\n"
            "\n"
            "Fixed cases:\n"
            "\n"
            "- ? [7]: baseline 0/1 passed, candidate 2/2\n"
            "- Other []: baseline 1/2 passed, candidate 1/1\n"
            "\n"
            "- warning: careful"
        )

    def test_not_run(self) -> None:
        comparison = GateComparison(
            "s1",
            "",
            baseline=_passed_verdict(),
            candidate=None,
            criteria=[CriterionChange("min_runs", _OK, None)],
        )
        assert comparison.to_markdown() == (
            "## s1 (`s1`): NOT RUN, REGRESSED\n"
            "\n"
            "- Baseline: 2 of 2 run(s) passed; gate passed\n"
            "- Candidate: no run\n"
            "\n"
            "| Criterion | Required | Baseline | Candidate | Change |\n"
            "|---|---|---|---|---|\n"
            "| min_runs | >= 2 | 3 | - | dropped |"
        )

    def test_no_baseline_no_regression(self) -> None:
        comparison = GateComparison(
            "s3",
            "Empty",
            baseline=None,
            candidate=_empty_verdict(),
        )
        assert comparison.to_markdown() == (
            "## Empty (`s3`): PASSED, no regression\n"
            "\n"
            "- Baseline: no run\n"
            "- Candidate: 0 of 0 run(s) passed; gate passed\n"
            "\n"
            "| Criterion | Required | Baseline | Candidate | Change |\n"
            "|---|---|---|---|---|"
        )


class TestCompare:
    def test_newly_failing_fixed_and_missing(self) -> None:
        baseline = [
            _rec(lanelets=(1,)),
            _rec(lanelets=(1,)),
            _rec(lanelets=(2,), passed=False),
            _rec(lanelets=(3,)),
            _rec(lanelets=(4,), map_name="A"),
            _rec(lanelets=(5,), map_name="B"),
        ]
        candidate = [
            _rec(lanelets=(1,)),
            _rec(lanelets=(1,), passed=False),
            _rec(lanelets=(2,)),
            _rec(lanelets=(9,)),
            _rec(lanelets=(3,), passed=True),
        ]
        result = compare(baseline, candidate)
        assert [(c.name, c.change) for c in result.criteria] == [
            ("min_runs", "worse"),
            ("min_success_rate", "worse"),
        ]
        assert result.newly_failing == [
            CaseChange("Town", (1,), (2, 2), (1, 2)),
        ]
        assert result.fixed == [CaseChange("Town", (2,), (0, 1), (1, 1))]
        assert result.warnings == ["2 case(s) the baseline ran did not run"]
        assert result.to_markdown() == (
            "## Left turn (`left_turn`): PASSED, REGRESSED\n"
            "\n"
            "- Baseline: 5 of 6 run(s) passed; gate passed\n"
            "- Candidate: 4 of 5 run(s) passed; gate passed\n"
            "\n"
            "| Criterion | Required | Baseline | Candidate | Change |\n"
            "|---|---|---|---|---|\n"
            "| min_runs | >= 1 | 6 | 5 | worse |\n"
            "| min_success_rate | >= 0.5 | 0.8333 | 0.8 | worse |\n"
            "\n"
            "Newly failing cases:\n"
            "\n"
            "- Town [1]: baseline 2/2 passed, candidate 1/2\n"
            "\n"
            "Fixed cases:\n"
            "\n"
            "- Town [2]: baseline 0/1 passed, candidate 1/1\n"
            "\n"
            "- warning: 2 case(s) the baseline ran did not run"
        )

    def test_without_a_candidate(self) -> None:
        result = compare([_rec(), _rec(lanelets=(3,), passed=False)], [])
        assert result.candidate is None
        assert [(c.name, c.change) for c in result.criteria] == [
            ("min_runs", "dropped"),
            ("min_success_rate", "dropped"),
        ]
        assert result.newly_failing == []
        assert result.fixed == []
        # The candidate did not run at all: no "did not run" warning.
        assert result.warnings == []

    def test_without_a_baseline(self) -> None:
        result = compare([], [_rec()])
        assert [(c.name, c.change) for c in result.criteria] == [
            ("min_runs", "new"),
            ("min_success_rate", "new"),
        ]
        assert result.warnings == []

    def test_baseline_warnings_come_first(self) -> None:
        baseline = [
            _rec(gate={"min_success_rate": 0.4}),
            _rec(gate={"min_success_rate": 0.5}, digest="e"),
        ]
        result = compare(baseline, [_rec(digest="f")])
        assert result.warnings == [
            "the runs ran 2 versions of the document",
            "the baseline's runs stated another gate; both are judged "
            "against the candidate's",
            "the baseline and the candidate ran different versions of the " "document",
        ]


class TestByScenario:
    def test_groups_in_first_seen_order(self) -> None:
        a1, b1, a2 = _rec(scenario="a"), _rec(scenario="b"), _rec(scenario="a")
        groups = _by_scenario([b1, a1, a2])
        assert list(groups) == ["b", "a"]
        assert groups["a"][0] is a1
        assert groups["a"][1] is a2
        assert groups["b"][0] is b1

    def test_only_the_scenario(self) -> None:
        a1, b1 = _rec(scenario="a"), _rec(scenario="b")
        groups = _by_scenario([a1, b1], "b")
        assert list(groups) == ["b"]
        assert groups["b"][0] is b1
        assert _by_scenario([a1], "c") == {}

    def test_empty(self) -> None:
        assert _by_scenario([]) == {}

    def test_compare_all_uses_it(self) -> None:
        result = compare_all(
            [_rec(scenario="b"), _rec(scenario="a")], [_rec(scenario="a")]
        )
        assert [(c.scenario, c.candidate is None) for c in result] == [
            ("a", False),
            ("b", True),
        ]
