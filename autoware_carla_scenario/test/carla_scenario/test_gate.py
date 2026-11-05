"""A logical scenario's gate: the IR, the run's record, and judging it (docs/gate.md)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from autoware_carla_scenario.authoring.models import (
    GateCoverageItem,
    GateSpec,
    LaneSegment,
    RouteSearch,
    ScenarioDocument,
)
from autoware_carla_scenario.authoring.persistence import load_document, save_document
from autoware_carla_scenario.authoring.starter import new_document
from autoware_carla_scenario.authoring.validator import validate_document
from autoware_carla_scenario.coverage import CoverageCollector, CoverItem
from autoware_carla_scenario.gate.evaluate import (
    EXIT_ERROR,
    EXIT_FAILED,
    EXIT_PASSED,
    compare,
    compare_all,
    consistency,
    find_gate_records,
    judge,
    judge_all,
    load_gate_records,
    main,
    station_values,
)
from autoware_carla_scenario.gate.probe import signed_lateral
from autoware_carla_scenario.gate.record import (
    GATE_RECORD_SCHEMA,
    GateCase,
    GateRecord,
    read_gate_record,
    write_gate_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drive(
    lateral: float = 0.0, speed: float = 10.0, start: float = 0.0, end: float = 100.0
) -> list[tuple[float, float, float, float]]:
    """Rows of an ego driving from route s *start* to *end*, 1 m a row."""
    rows = []
    for i, s in enumerate(range(int(start), int(end) + 1)):
        rows.append((i * 0.1, float(s), lateral, speed))
    return rows


def _record(
    *,
    passed: bool = True,
    lanelets: tuple[int, ...] = (1, 2, 3),
    map_name: str = "Town",
    case_count: int = 4,
    gate: Optional[dict[str, Any]] = None,
    ego: Optional[list[tuple[float, float, float, float]]] = None,
    ends: tuple[float, ...] = (40.0, 60.0, 100.0),
    scenario: str = "left_turn",
) -> GateRecord:
    return GateRecord(
        scenario=scenario,
        title="Left turn",
        digest="d" * 64,
        gate=gate if gate is not None else {"min_success_rate": 0.5},
        case=GateCase(
            map=map_name,
            index=0,
            lanelet_ids=lanelets,
            case_count=case_count,
            segment_kinds=("lane", "junction", "lane"),
            segment_ends=ends,
        ),
        passed=passed,
        message="ok" if passed else "collision",
        elapsed_seconds=12.0,
        failure_category=None if passed else "collision",
        ego=ego if ego is not None else _drive(),
    )


def _write(tmp_path: Path, name: str, record: GateRecord) -> Path:
    path = tmp_path / f"{name}_gate.json"
    record.files = {"coverage": f"{name}_coverage.json"}
    write_gate_record(path, record)
    record.path = path
    return path


def _route() -> RouteSearch:
    return RouteSearch(segments=[LaneSegment()])


# ---------------------------------------------------------------------------
# The IR
# ---------------------------------------------------------------------------


class TestGateSpec:
    def test_unset_thresholds_are_not_judged(self) -> None:
        gate = GateSpec(min_success_rate=0.9, max_lateral_spread_m=0.3)
        assert gate.judged() == [
            "min_runs",
            "min_success_rate",
            "max_lateral_spread_m",
        ]
        assert gate.judges_consistency

    @pytest.mark.parametrize(
        "field, value",
        [
            ("min_success_rate", 1.5),
            ("min_case_coverage", -0.1),
            ("max_speed_spread_mps", 0.0),
            ("min_runs", 0),
            ("min_runs", True),
            ("consistency_min_runs", 1),
            ("min_success_rate", math.nan),
        ],
    )
    def test_out_of_range_is_refused(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError):
            GateSpec(**{field: value})

    def test_a_typo_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            GateSpec.model_validate({"min_sucess_rate": 0.9})

    def test_the_document_round_trips(self, tmp_path: Path) -> None:
        document = new_document()
        document.route = _route()
        document.gate = GateSpec(
            min_success_rate=0.9,
            coverage_items=[GateCoverageItem(item="speed", min=0.5)],
        )
        path = tmp_path / "document.yaml"
        save_document(document, path)
        stored = document.to_yaml_dict()["gate"]
        assert "min_case_coverage" not in stored  # unset: left out
        assert load_document(path).gate == document.gate

    def test_no_gate_says_nothing(self) -> None:
        assert "gate" not in new_document().to_yaml_dict()


class TestValidation:
    @staticmethod
    def _messages(document: ScenarioDocument) -> list[tuple[str, str]]:
        return [
            (i.severity, i.message)
            for i in validate_document(document).issues
            if i.path.startswith("gate")
        ]

    def test_a_gate_needs_a_route_search(self) -> None:
        document = new_document()
        document.gate = GateSpec(min_success_rate=0.9)
        ((severity, message),) = self._messages(document)
        assert severity == "error"
        assert "route search" in message

    def test_a_gate_on_a_logical_scenario_is_fine(self) -> None:
        document = new_document()
        document.route = _route()
        document.gate = GateSpec(min_success_rate=0.9)
        assert self._messages(document) == []

    def test_a_gate_judging_only_runs_is_warned_about(self) -> None:
        document = new_document()
        document.route = _route()
        document.gate = GateSpec()
        ((severity, _),) = self._messages(document)
        assert severity == "warning"

    def test_an_item_named_twice_is_an_error(self) -> None:
        document = new_document()
        document.route = _route()
        document.gate = GateSpec(
            coverage_items=[
                GateCoverageItem(item="a", min=0.5),
                GateCoverageItem(item="a", min=0.6),
            ]
        )
        assert ("error", "Cover item 'a' is named twice.") in self._messages(document)

    def test_too_few_runs_for_the_spread_is_warned_about(self) -> None:
        document = new_document()
        document.route = _route()
        document.gate = GateSpec(max_lateral_spread_m=0.5, min_runs=1)
        assert [s for s, _ in self._messages(document)] == ["warning"]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_it_round_trips(self, tmp_path: Path) -> None:
        record = _record(passed=False)
        path = _write(tmp_path, "Left", record)
        raw = json.loads(path.read_text())
        assert raw["schema"] == GATE_RECORD_SCHEMA
        assert raw["ego"]["columns"] == ["t", "s", "lateral_m", "speed_mps"]
        read = read_gate_record(path)
        assert read.case == record.case
        # Rounded to 4 decimals as written.
        assert read.ego == [tuple(round(v, 4) for v in row) for row in record.ego]
        assert (read.passed, read.failure_category) == (False, "collision")
        assert read.file("coverage") == tmp_path / "Left_coverage.json"

    def test_another_schema_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "x_gate.json"
        path.write_text(json.dumps({"schema": "other"}))
        with pytest.raises(ValueError, match="not a gate record"):
            read_gate_record(path)

    def test_a_malformed_row_is_refused(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "x", _record())
        raw = json.loads(path.read_text())
        raw["ego"]["rows"][0] = [0.0, "nan", 0.0, 0.0]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="malformed ego row"):
            read_gate_record(path)

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("case", "m", "'case' is not a JSON object"),
            ("files", ["x"], "'files' is not a JSON object"),
            ("outcome", [True], "'outcome' is not a JSON object"),
            ("ego", "rows", "'ego' is not a JSON object"),
            ("gate", "strict", "'gate' is not a JSON object"),
        ],
    )
    def test_a_part_of_the_wrong_type_is_refused(
        self, tmp_path: Path, key: str, value: Any, match: str
    ) -> None:
        path = _write(tmp_path, "x", _record())
        raw = json.loads(path.read_text())
        raw[key] = value
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match=match):
            read_gate_record(path)

    @pytest.mark.parametrize(
        ("part", "key", "value"),
        [
            ("case", "lanelet_ids", 7),
            ("ego", "rows", {"0": [0, 0, 0, 0]}),
            ("ego", "columns", "tslv"),
        ],
    )
    def test_a_nested_list_of_the_wrong_type_is_refused(
        self, tmp_path: Path, part: str, key: str, value: Any
    ) -> None:
        path = _write(tmp_path, "x", _record())
        raw = json.loads(path.read_text())
        raw[part][key] = value
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="is not a JSON array"):
            read_gate_record(path)

    def test_a_row_that_is_not_an_array_is_refused(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "x", _record())
        raw = json.loads(path.read_text())
        raw["ego"]["rows"][0] = "0000"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="malformed ego row"):
            read_gate_record(path)

    @pytest.mark.parametrize("gate", ["missing", None])
    def test_a_record_without_a_gate_is_refused(
        self, tmp_path: Path, gate: Optional[str]
    ) -> None:
        path = _write(tmp_path, "x", _record())
        raw = json.loads(path.read_text())
        if gate == "missing":
            del raw["gate"]
        else:
            raw["gate"] = None
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="no 'gate' object"):
            read_gate_record(path)

    def test_an_empty_gate_is_still_a_gate(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "x", _record(gate={}))
        assert read_gate_record(path).gate == {}

    def test_a_case_that_is_not_an_object_is_refused_on_its_own(self) -> None:
        with pytest.raises(ValueError, match="the case is not a JSON object"):
            GateCase.from_dict("m")  # type: ignore[arg-type]

    def test_a_record_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a JSON object"):
            GateRecord.from_dict(["x"])  # type: ignore[arg-type]

    def test_a_malformed_record_does_not_stop_the_others(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", _record())
        bad = _write(tmp_path, "b", _record())
        raw = json.loads(bad.read_text())
        raw["case"] = "m"
        bad.write_text(json.dumps(raw))
        records, errors = load_gate_records(find_gate_records([tmp_path]))
        assert [r.path for r in records] == [tmp_path / "a_gate.json"]
        assert len(errors) == 1 and "b_gate.json" in errors[0]


class TestSignedLateral:
    def test_left_of_the_route_is_positive(self) -> None:
        # The route runs along +x; +y is its left.
        assert signed_lateral((0.0, 0.0, 0.0), 5.0, 1.5, 1.5) == 1.5
        assert signed_lateral((0.0, 0.0, 0.0), 5.0, -1.5, 1.5) == -1.5

    def test_the_heading_turns_the_sides(self) -> None:
        # Running along +y, +x is its right.
        assert signed_lateral((0.0, 0.0, math.pi / 2), 2.0, 0.0, 2.0) == -2.0

    def test_on_the_line_is_zero(self) -> None:
        assert signed_lateral((0.0, 0.0, 0.0), 3.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_stations_sit_at_fractions_of_each_segment(self) -> None:
        record = _record(ends=(10.0, 20.0))
        values = station_values(record, 2)
        # 2.5 and 7.5 in the first segment, 12.5 and 17.5 in the second.
        assert sorted(values) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_a_station_behind_the_spawn_is_not_passed(self) -> None:
        record = _record(ego=_drive(start=30.0), ends=(40.0, 60.0, 100.0))
        values = station_values(record, 4)
        # Segment 0's stations at 5, 15, 25 m lie behind a spawn at 30 m.
        assert [k for k in values if k[0] == 0] == [(0, 3)]

    def test_values_are_interpolated_between_rows(self) -> None:
        rows = [(0.0, 0.0, 0.0, 0.0), (1.0, 10.0, 1.0, 10.0)]
        record = _record(ego=rows, ends=(10.0,))
        assert station_values(record, 1)[(0, 0)] == pytest.approx((0.5, 5.0))

    def test_identical_drives_do_not_spread(self) -> None:
        spread = consistency([_record(), _record()], 10, 2)
        assert spread.stations == 30
        assert spread.lateral_spread_m == pytest.approx(0.0)
        assert spread.speed_spread_mps == pytest.approx(0.0)

    def test_the_largest_spread_is_found(self) -> None:
        wide = _drive(lateral=1.0, speed=12.0)
        spread = consistency([_record(), _record(ego=wide)], 10, 2)
        assert spread.lateral_spread_m == pytest.approx(0.5)
        assert spread.speed_spread_mps == pytest.approx(1.0)

    def test_matches_of_different_lengths_compare_by_fraction(self) -> None:
        # The same drive, through a pattern stretched to twice the length.
        long = _record(ego=_drive(end=200.0), ends=(80.0, 120.0, 200.0))
        spread = consistency([_record(), long], 5, 2)
        assert spread.stations == 15

    def test_a_station_too_few_runs_passed_is_not_measured(self) -> None:
        short = _record(ego=_drive(end=50.0))
        spread = consistency([_record(), short], 10, 2)
        # Segment 0 (0-40 m) and the first half of segment 1 (40-60 m).
        assert spread.stations == 15


class TestJudge:
    def test_success_rate_and_runs(self) -> None:
        gate = {"min_runs": 3, "min_success_rate": 0.6}
        verdict = judge([_record(gate=gate), _record(gate=gate, passed=False)])
        by_name = {c.name: c for c in verdict.criteria}
        assert not by_name["min_runs"].passed
        assert by_name["min_success_rate"].value == 0.5
        assert not by_name["min_success_rate"].passed
        assert not verdict.passed

    def test_case_coverage_counts_cases_per_map(self) -> None:
        gate = {"min_case_coverage": 0.5}
        verdict = judge(
            [
                _record(gate=gate, lanelets=(1,), case_count=2),
                _record(gate=gate, lanelets=(1,), case_count=2),  # a repeat
                _record(gate=gate, lanelets=(9,), case_count=2, map_name="Other"),
            ]
        )
        (_, case) = verdict.criteria
        assert case.value == pytest.approx(2 / 4)
        assert case.passed
        assert "2 of 4 case(s) on 2 map(s)" == case.detail

    def test_an_unknown_case_count_fails_the_case_coverage(self) -> None:
        verdict = judge([_record(gate={"min_case_coverage": 0.1}, case_count=0)])
        assert verdict.criteria[-1].value is None
        assert not verdict.passed

    def test_trajectory_spread(self) -> None:
        gate = {"max_lateral_spread_m": 0.3, "max_speed_spread_mps": 2.0}
        verdict = judge(
            [_record(gate=gate), _record(gate=gate, ego=_drive(lateral=1.0))]
        )
        by_name = {c.name: c for c in verdict.criteria}
        assert not by_name["max_lateral_spread_m"].passed
        assert by_name["max_speed_spread_mps"].passed
        assert "segment 0" in by_name["max_lateral_spread_m"].detail

    def test_a_spread_nothing_measured_fails(self) -> None:
        verdict = judge([_record(gate={"max_lateral_spread_m": 1.0})])
        assert verdict.criteria[-1].value is None
        assert "no station" in verdict.criteria[-1].detail
        assert not verdict.passed

    def test_a_given_gate_replaces_the_recorded_one(self) -> None:
        verdict = judge([_record(passed=False)], GateSpec(min_success_rate=0.0))
        assert verdict.passed

    def test_different_gates_are_warned_about(self, tmp_path: Path) -> None:
        records = [
            _record(gate={"min_success_rate": 0.1}),
            _record(gate={"min_success_rate": 0.2}),
        ]
        for i, record in enumerate(records):
            _write(tmp_path, f"r{i}", record)
        verdict = judge(records)
        assert any("different gates" in w for w in verdict.warnings)

    def test_records_of_two_scenarios_are_judged_apart(self) -> None:
        verdicts = judge_all([_record(), _record(scenario="other")])
        assert [v.scenario for v in verdicts] == ["left_turn", "other"]

    def test_coverage_is_merged_from_the_files_beside_the_records(
        self, tmp_path: Path
    ) -> None:
        gate = {
            "min_scenario_coverage": 0.5,
            "coverage_items": [{"item": "x", "min": 1.0}],
        }
        records = []
        for i, value in enumerate((1, 2)):
            record = _record(gate=gate)
            _write(tmp_path, f"run{i}", record)
            collector = CoverageCollector(
                [CoverItem("x", lambda world: world.value, values=[1, 2, 3])]
            )
            collector.end(_World(value), 0.0)
            collector.write(tmp_path / f"run{i}_coverage.json", "S")
            records.append(record)
        verdict = judge(records)
        by_name = {c.name: c for c in verdict.criteria}
        assert by_name["min_scenario_coverage"].value == pytest.approx(2 / 3)
        assert by_name["min_scenario_coverage"].passed
        assert not by_name["coverage_items[x]"].passed
        assert by_name["coverage_items[x]"].detail == "1 hole(s)"

    def test_no_coverage_file_fails_a_coverage_criterion(self, tmp_path: Path) -> None:
        record = _record(gate={"min_odd_coverage": 0.1})
        _write(tmp_path, "run", record)
        verdict = judge([record])
        assert not verdict.passed
        assert any("no coverage file" in w for w in verdict.warnings)


class _World:
    def __init__(self, value: Any) -> None:
        self.value = value


class TestCli:
    def test_a_passing_gate_exits_0(self, tmp_path: Path, capsys: Any) -> None:
        _write(tmp_path, "a", _record())
        assert main([str(tmp_path), "--json", str(tmp_path / "v.json")]) == EXIT_PASSED
        assert "PASSED" in capsys.readouterr().out
        verdicts = json.loads((tmp_path / "v.json").read_text())["verdicts"]
        assert verdicts[0]["passed"] is True

    def test_a_failing_gate_exits_1(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", _record(passed=False))
        markdown = tmp_path / "gate.md"
        assert main([str(tmp_path), "--markdown", str(markdown)]) == EXIT_FAILED
        assert "**FAIL**" in markdown.read_text()

    def test_nothing_to_judge_exits_2(self, tmp_path: Path) -> None:
        assert main([str(tmp_path)]) == EXIT_ERROR

    def test_an_unreadable_record_exits_2(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", _record())
        (tmp_path / "b_gate.json").write_text("{")
        assert main([str(tmp_path)]) == EXIT_ERROR

    @pytest.mark.parametrize(
        ("key", "value"),
        [("case", "m"), ("files", ["x"]), ("gate", None), ("outcome", 1)],
    )
    def test_a_malformed_record_exits_2(
        self, tmp_path: Path, capsys: Any, key: str, value: Any
    ) -> None:
        # A passing record beside it: the malformed one is reported as
        # unreadable (2), not as a failed gate (1), nor ignored (0).
        _write(tmp_path, "a", _record())
        path = _write(tmp_path, "b", _record())
        raw = json.loads(path.read_text())
        if value is None:
            del raw[key]
        else:
            raw[key] = value
        path.write_text(json.dumps(raw))
        assert main([str(tmp_path)]) == EXIT_ERROR
        assert "b_gate.json" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ('"index":0', '"index":1e999'),
            ('"lanelet_ids":[1,', '"lanelet_ids":[Infinity,'),
            ('"case_count":4', '"case_count":-Infinity'),
        ],
    )
    def test_an_infinite_integer_exits_2(
        self, tmp_path: Path, capsys: Any, old: str, new: str
    ) -> None:
        verdict = tmp_path / "v.json"
        _write(tmp_path, "a", _record())
        path = _write(tmp_path, "b", _record())
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))
        with pytest.raises(ValueError, match="b_gate.json"):
            read_gate_record(path)
        assert main([str(tmp_path), "--json", str(verdict)]) == EXIT_ERROR
        assert "b_gate.json" in capsys.readouterr().err
        # The other record is still judged.
        assert json.loads(verdict.read_text())["verdicts"][0]["runs"] == 1

    @pytest.mark.parametrize(
        "gate",
        [{"bogus_field": 1}, {"min_success_rate": "x"}, {"min_success_rate": 2.0}],
    )
    def test_a_malformed_gate_exits_2(
        self, tmp_path: Path, capsys: Any, gate: dict[str, Any]
    ) -> None:
        verdict = tmp_path / "v.json"
        _write(tmp_path, "a", _record())
        _write(tmp_path, "b", _record(gate=gate))
        assert main([str(tmp_path), "--json", str(verdict)]) == EXIT_ERROR
        assert "b_gate.json: not a valid gate" in capsys.readouterr().err
        # The other record is still judged.
        assert json.loads(verdict.read_text())["verdicts"][0]["runs"] == 1

    def test_a_malformed_gate_is_refused_with_a_document_too(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        # The document's gate replaces the records', but a record whose gate
        # is malformed is corrupt all the same.
        runs = tmp_path / "runs"
        _write(runs, "a", _record())
        _write(runs, "b", _record(gate={"bogus_field": 1}))
        document = new_document()
        document.id = "left_turn"
        document.route = _route()
        document.gate = GateSpec(min_success_rate=0.0)
        path = tmp_path / "document.yaml"
        save_document(document, path)
        assert main([str(runs), "--document", str(path)]) == EXIT_ERROR
        assert "b_gate.json: not a valid gate" in capsys.readouterr().err

    def test_deeply_nested_json_exits_2(self, tmp_path: Path, capsys: Any) -> None:
        verdict = tmp_path / "v.json"
        _write(tmp_path, "a", _record())
        path = tmp_path / "b_gate.json"
        path.write_text("[" * 200_000 + "]" * 200_000)
        with pytest.raises(ValueError, match="b_gate.json: cannot be read"):
            read_gate_record(path)
        assert main([str(tmp_path), "--json", str(verdict)]) == EXIT_ERROR
        assert "b_gate.json" in capsys.readouterr().err
        assert json.loads(verdict.read_text())["verdicts"][0]["runs"] == 1

    def test_a_huge_integer_literal_names_the_file(self, tmp_path: Path) -> None:
        # Past Python 3.11+'s digit limit json.loads raises ValueError; on
        # 3.10 it parses and the schema is refused. Either names the file.
        path = tmp_path / "b_gate.json"
        path.write_text('{"schema": ' + "1" * 5000 + "}")
        with pytest.raises(ValueError, match="b_gate.json: "):
            read_gate_record(path)
        assert main([str(tmp_path)]) == EXIT_ERROR

    def test_a_file_not_in_utf8_exits_2(self, tmp_path: Path, capsys: Any) -> None:
        verdict = tmp_path / "v.json"
        _write(tmp_path, "a", _record())
        path = tmp_path / "b_gate.json"
        path.write_bytes(b'{"schema": "\xff\xfe"}')
        with pytest.raises(ValueError, match="b_gate.json: cannot be read"):
            read_gate_record(path)
        assert main([str(tmp_path), "--json", str(verdict)]) == EXIT_ERROR
        assert "b_gate.json" in capsys.readouterr().err
        assert json.loads(verdict.read_text())["verdicts"][0]["runs"] == 1

    def test_a_record_without_a_gate_alone_exits_2(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "a", _record())
        raw = json.loads(path.read_text())
        del raw["gate"]
        path.write_text(json.dumps(raw))
        assert main([str(tmp_path)]) == EXIT_ERROR

    def test_the_documents_gate_rejudges_the_runs(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", _record(passed=False))
        document = new_document()
        document.id = "left_turn"
        document.route = _route()
        document.gate = GateSpec(min_success_rate=0.0)
        path = tmp_path / "document.yaml"
        save_document(document, path)
        assert main([str(tmp_path), "--document", str(path)]) == EXIT_PASSED

    def test_a_document_without_a_gate_exits_2(self, tmp_path: Path) -> None:
        _write(tmp_path, "a", _record())
        path = tmp_path / "document.yaml"
        save_document(new_document(), path)
        assert main([str(tmp_path), "--document", str(path)]) == EXIT_ERROR


class TestRegression:
    """Two sets of runs: a baseline's and a candidate's (degradation checks)."""

    GATE = {"min_success_rate": 0.5, "max_lateral_spread_m": 0.6}

    def _runs(self, *passed: bool, lateral: float = 0.0) -> list[GateRecord]:
        return [
            _record(
                gate=self.GATE,
                passed=ok,
                lanelets=(i,),
                ego=_drive(lateral=lateral * i),
            )
            for i, ok in enumerate(passed)
        ]

    def test_nothing_changed_is_no_regression(self) -> None:
        comparison = compare(self._runs(True, True), self._runs(True, True))
        assert not comparison.regressed
        assert comparison.passed
        assert {c.change for c in comparison.criteria} == {"same"}

    def test_a_criterion_that_passed_and_fails_regressed(self) -> None:
        comparison = compare(
            self._runs(True, True, True), self._runs(True, False, False)
        )
        by_name = {c.name: c for c in comparison.criteria}
        assert by_name["min_success_rate"].change == "regressed"
        assert by_name["min_success_rate"].delta == pytest.approx(1 / 3 - 1)
        assert comparison.regressed and not comparison.ok()

    def test_a_case_that_passed_and_fails_regressed_inside_the_gate(self) -> None:
        # Still above the success rate: the gate passes, the case regressed.
        comparison = compare(
            self._runs(True, True, True), self._runs(True, True, False)
        )
        assert comparison.candidate is not None and comparison.candidate.passed
        (case,) = comparison.newly_failing
        assert (case.lanelet_ids, case.baseline, case.candidate) == (
            (2,),
            (1, 1),
            (0, 1),
        )
        assert comparison.regressed
        # The gate decides by default; strict, the regression fails it.
        assert comparison.ok()
        assert not comparison.ok(strict=True)

    def test_a_fixed_case_is_reported(self) -> None:
        comparison = compare(self._runs(True, False), self._runs(True, True))
        assert [c.lanelet_ids for c in comparison.fixed] == [(1,)]
        by_name = {c.name: c for c in comparison.criteria}
        assert by_name["min_success_rate"].change == "better"
        assert not comparison.regressed

    def test_a_wider_spread_is_worse(self) -> None:
        comparison = compare(
            self._runs(True, True, lateral=0.2), self._runs(True, True, lateral=0.8)
        )
        by_name = {c.name: c for c in comparison.criteria}
        assert by_name["max_lateral_spread_m"].change == "worse"
        assert not comparison.regressed  # still within 0.6 m

    def test_both_are_judged_against_the_candidates_gate(self) -> None:
        baseline = [_record(gate={"min_success_rate": 0.1}, passed=False)]
        candidate = [_record(gate={"min_success_rate": 0.0}, passed=False)]
        comparison = compare(baseline, candidate)
        assert comparison.baseline is not None and comparison.baseline.passed
        assert any("another gate" in w for w in comparison.warnings)

    def test_a_scenario_the_candidate_did_not_run_regressed(self) -> None:
        (comparison,) = compare_all(self._runs(True), [])
        assert comparison.candidate is None
        assert comparison.regressed

    def test_a_scenario_new_to_the_candidate_is_judged_alone(self) -> None:
        (comparison,) = compare_all([], self._runs(True, True))
        assert comparison.baseline is None
        assert {c.change for c in comparison.criteria} == {"new"}
        assert comparison.passed


class TestRegressionCli:
    def _set(self, root: Path, *passed: bool) -> Path:
        root.mkdir()
        for i, ok in enumerate(passed):
            _write(root, f"r{i}", _record(passed=ok, lanelets=(i,)))
        return root

    def test_two_sets_with_no_regression_exit_0(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        base = self._set(tmp_path / "base", True, True)
        cand = self._set(tmp_path / "cand", True, True)
        out = tmp_path / "c.json"
        assert main([str(base), str(cand), "--json", str(out)]) == EXIT_PASSED
        assert "no regression" in capsys.readouterr().out
        payload = json.loads(out.read_text())
        assert payload["comparisons"][0]["regressed"] is False

    def test_a_newly_failing_case_inside_the_gate_exits_0(self, tmp_path: Path) -> None:
        # 1 of 2 passed: the gate's 0.5 success rate still holds.
        base = self._set(tmp_path / "base", True, True)
        cand = self._set(tmp_path / "cand", True, False)
        markdown = tmp_path / "c.md"
        argv = [str(base), str(cand), "--markdown", str(markdown)]
        assert main(argv) == EXIT_PASSED
        text = markdown.read_text()
        assert "REGRESSED" in text and "Newly failing cases" in text

    def test_fail_on_regression_makes_it_exit_1(self, tmp_path: Path) -> None:
        base = self._set(tmp_path / "base", True, True)
        cand = self._set(tmp_path / "cand", True, False)
        argv = [str(base), str(cand), "--fail-on-regression"]
        assert main(argv) == EXIT_FAILED

    def test_a_candidate_failing_the_gate_exits_1(self, tmp_path: Path) -> None:
        base = self._set(tmp_path / "base", True, True)
        cand = self._set(tmp_path / "cand", False, False)
        assert main([str(base), str(cand)]) == EXIT_FAILED

    def test_fail_on_regression_needs_two_sets(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exited:
            main([str(tmp_path), "--fail-on-regression"])
        assert exited.value.code == 2

    def test_more_than_two_sets_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exited:
            main([str(tmp_path), str(tmp_path), str(tmp_path)])
        assert exited.value.code == 2


class TestTheRunnerWritesTheRecord:
    """scenario_runner._write_gate_record: the scenario's record, and the
    outputs the run wrote beside it."""

    class _Scenario:
        def __init__(self, record: Optional[GateRecord]) -> None:
            self._record = record

        def gate_record(self, _result: Any, _stem: str) -> Optional[GateRecord]:
            return self._record

    def test_only_the_outputs_written_are_named(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.scenario_runner import _write_gate_record

        (tmp_path / "Left_result.json").write_text("{}")
        (tmp_path / "Left.trajectory.jsonl").write_text("")
        result = ScenarioResult(passed=True, message="", elapsed_seconds=1.0)
        _write_gate_record(tmp_path, "Left", self._Scenario(_record()), result)  # type: ignore[arg-type]
        record = read_gate_record(tmp_path / "Left_gate.json")
        assert record.files == {
            "result": "Left_result.json",
            "trajectory": "Left.trajectory.jsonl",
        }
        assert record.file("coverage") is None

    def test_a_scenario_without_a_record_writes_none(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.conditions import ScenarioResult
        from autoware_carla_scenario.scenario_runner import _write_gate_record

        result = ScenarioResult(passed=True, message="", elapsed_seconds=1.0)
        _write_gate_record(tmp_path, "X", self._Scenario(None), result)  # type: ignore[arg-type]
        _write_gate_record(tmp_path, "Y", object(), result)  # type: ignore[arg-type]
        assert not list(tmp_path.iterdir())
