"""Coverage: cover items, the collector, the ODD items and the merged report."""

from __future__ import annotations

import enum
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from autoware_carla_scenario.conditions import BaseCondition, ScenarioResult
from autoware_carla_scenario.coverage import (
    COVERAGE_SCHEMA,
    CoverageCollector,
    CoverGroup,
    CoverItem,
    CrossItem,
    SamplingEvent,
    merge_coverage,
)
from autoware_carla_scenario.coverage.items import ABOVE_RANGE, BELOW_RANGE
from autoware_carla_scenario.coverage.report import load_and_merge, main


class _Side(enum.Enum):
    LEFT = "left"
    RIGHT = "right"


def _item(name: str = "x", **kw: Any) -> CoverItem:
    kw.setdefault("expression", lambda world: world.value)
    return CoverItem(name=name, **kw)


class _World:
    """Just enough of a world for expressions that read ``world.value``."""

    def __init__(self, value: Any = None) -> None:
        self.value = value


class _Flag(BaseCondition):
    """Satisfied while ``world.flag`` is true."""

    def __init__(self) -> None:
        super().__init__("flag")

    def check(self, world: Any, elapsed: float) -> Optional[ScenarioResult]:
        if getattr(world, "flag", False):
            return ScenarioResult(passed=True, message="", elapsed_seconds=elapsed)
        return None


class TestCoverItem:
    def test_range_and_every_make_equal_buckets(self) -> None:
        # The DSL's own example: 10..130 kph every 10 is twelve buckets.
        item = _item(range=(10, 130), every=10)
        assert len(item.labels) == 12
        assert item.labels[0] == "[10, 20)"
        assert item.labels[-1] == "[120, 130]"

    def test_a_range_that_does_not_divide_ends_in_a_narrower_bucket(self) -> None:
        item = _item(range=(0, 25), every=10)
        assert item.labels == ["[0, 10)", "[10, 20)", "[20, 25]"]

    def test_n_edges_make_n_minus_one_buckets(self) -> None:
        item = _item(buckets=[0, 1, 5, 20])
        assert item.labels == ["[0, 1)", "[1, 5)", "[5, 20]"]

    def test_a_bucket_holds_its_lower_edge_and_the_last_its_upper_too(self) -> None:
        item = _item(buckets=[0, 10, 20])
        assert item.bucket_of(0) == "[0, 10)"
        assert item.bucket_of(10) == "[10, 20]"
        assert item.bucket_of(20) == "[10, 20]"
        assert item.bucket_of(-0.1) == BELOW_RANGE
        assert item.bucket_of(20.1) == ABOVE_RANGE

    def test_values_make_one_bucket_each(self) -> None:
        assert _item(values=_Side).labels == ["LEFT", "RIGHT"]
        assert _item(values=[False, True]).labels == ["false", "true"]
        assert _item(values=["dry", "wet"]).bucket_of("wet") == "wet"

    @pytest.mark.parametrize(
        "kw",
        [
            {},
            {"values": [1], "range": (0, 1), "every": 1},
            {"range": (0, 10)},
            {"every": 1.0, "values": [1]},
            {"range": (10, 0), "every": 1},
            {"range": (0, 10), "every": 0},
            {"buckets": [1]},
            {"buckets": [0, 2, 1]},
            {"values": []},
            {"values": [1, "1"]},
            {"values": [1], "target": 0},
        ],
    )
    def test_an_ill_defined_item_is_refused(self, kw: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _item(**kw)

    def test_a_cross_needs_items_sampled_on_the_same_event(self) -> None:
        a = _item("a", values=[1], event=SamplingEvent.TICK)
        b = _item("b", values=[1], event=SamplingEvent.END)
        with pytest.raises(ValueError, match="same event"):
            CrossItem("ab", [a, b])


class TestCollector:
    def test_an_end_item_is_sampled_once_at_the_end(self) -> None:
        collector = CoverageCollector([_item(values=[1, 2])])
        world = _World(1)
        collector.start(world, 0.0)
        for _ in range(3):
            collector.tick(world, 0.0)
        world.value = 2
        collector.end(world, 1.0)
        assert collector.to_dict("S")["items"][0]["hits"] == {"1": 0, "2": 1}

    def test_a_tick_item_counts_every_tick(self) -> None:
        collector = CoverageCollector([_item(values=[1, 2], event=SamplingEvent.TICK)])
        world = _World(1)
        for _ in range(3):
            collector.tick(world, 0.0)
        collector.end(world, 0.0)
        assert collector.to_dict("S")["items"][0]["hits"] == {"1": 3, "2": 0}

    def test_a_condition_event_samples_when_it_becomes_satisfied(self) -> None:
        collector = CoverageCollector([_item(values=[1, 2, 3], event=_Flag())])
        world = SimpleNamespace(value=1, flag=False)
        for value, flag in [(1, False), (2, True), (3, True), (3, False), (3, True)]:
            world.value, world.flag = value, flag
            collector.tick(world, 0.0)
        assert collector.to_dict("S")["items"][0]["hits"] == {"1": 0, "2": 1, "3": 1}

    def test_none_and_a_raising_expression_take_no_sample(self) -> None:
        def boom(world: Any) -> float:
            raise RuntimeError("weather unsupported")

        collector = CoverageCollector(
            [
                _item("none", values=[1], event=SamplingEvent.TICK),
                CoverItem("boom", boom, values=[1], event=SamplingEvent.TICK),
            ]
        )
        for _ in range(2):
            collector.tick(_World(None), 0.0)
        items = collector.to_dict("S")["items"]
        assert [i["samples"] for i in items] == [0, 0]

    def test_ignored_and_out_of_range_samples_are_counted_apart(self) -> None:
        collector = CoverageCollector(
            [
                _item(
                    "n",
                    buckets=[0, 10],
                    ignore=lambda v: v == 5,
                    event=SamplingEvent.TICK,
                ),
                _item("c", values=[1], event=SamplingEvent.TICK),
            ]
        )
        for value in (1, 5, 50, -3):
            collector.tick(_World(value), 0.0)
        n, c = collector.to_dict("S")["items"]
        assert n["hits"] == {"[0, 10]": 1}
        assert n["ignored"] == 1
        assert n["out_of_range"] == {ABOVE_RANGE: 1, BELOW_RANGE: 1}
        assert c["out_of_range"] == {"5": 1, "50": 1, "-3": 1}

    def test_a_cross_counts_the_buckets_hit_together(self) -> None:
        a = CoverItem(
            "a", lambda w: w.value[0], values=[1, 2], event=SamplingEvent.TICK
        )
        b = CoverItem(
            "b", lambda w: w.value[1], values=["x", "y"], event=SamplingEvent.TICK
        )
        collector = CoverageCollector([a, b], [CrossItem("ab", [a, b])])
        for value in [(1, "x"), (1, "x"), (2, "y"), (2, "z")]:
            collector.tick(_World(value), 0.0)
        cross = collector.to_dict("S")["crosses"][0]
        assert cross["hits"] == {"1 / x": 2, "1 / y": 0, "2 / x": 0, "2 / y": 1}

    def test_names_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            CoverageCollector([_item("a", values=[1]), _item("a", values=[2])])

    def test_the_file_is_written_in_the_coverage_schema(self, tmp_path: Path) -> None:
        collector = CoverageCollector([_item(values=[1])])
        path = tmp_path / "out" / "S_coverage.json"
        collector.write(path, "S")
        doc = json.loads(path.read_text())
        assert doc["schema"] == COVERAGE_SCHEMA
        assert doc["scenario"] == "S"


class TestScenarioRegistration:
    def _scenario(self) -> Any:
        import typesafe_carla.carla as carla

        from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform

        class _Scenario(BaseScenario):
            def setup(self) -> None:
                pass

            def is_done(self) -> bool:
                return True

        return _Scenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0, y=0, z=0))
                )
            )
        )

    def test_register_cover_and_cross(self) -> None:
        scenario = self._scenario()
        scenario.register_cover(
            "speed", lambda w: 1.0, range=(0, 10), every=5, event=SamplingEvent.TICK
        )
        scenario.register_cover(
            "side", lambda w: _Side.LEFT, values=_Side, event=SamplingEvent.TICK
        )
        scenario.register_cross("speed_x_side", ["speed", "side"])
        assert [i.name for i in scenario._cover_items] == ["speed", "side"]
        assert scenario._cross_items[0].items == scenario._cover_items
        assert scenario._cover_items[0].group is CoverGroup.SCENARIO

    def test_a_cross_of_an_unknown_item_is_refused(self) -> None:
        scenario = self._scenario()
        scenario.register_cover("a", lambda w: 1, values=[1])
        with pytest.raises(ValueError, match="no cover item named"):
            scenario.register_cross("ab", ["a", "b"])

    def test_a_name_is_registered_once(self) -> None:
        scenario = self._scenario()
        scenario.register_cover("a", lambda w: 1, values=[1])
        with pytest.raises(ValueError, match="already registered"):
            scenario.register_cover("a", lambda w: 1, values=[1])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _run(value: Any, scenario: str = "S", **kw: Any) -> dict[str, Any]:
    collector = CoverageCollector([_item("x", values=[1, 2, 3], **kw)])
    collector.end(_World(value), 0.0)
    return collector.to_dict(scenario)


class TestReport:
    def test_runs_merge_into_grades_and_holes(self) -> None:
        report = merge_coverage([_run(1), _run(1), _run(2, scenario="T")])
        (entry,) = report.entries
        assert entry.hits == {"1": 2, "2": 1, "3": 0}
        assert entry.runs == {"1": 2, "2": 1}
        assert entry.holes == ["3"]
        assert entry.grade == pytest.approx(2 / 3)
        assert report.runs == 3
        assert report.scenarios == {"S": 2, "T": 1}
        assert report.group_grade("scenario") == pytest.approx(2 / 3)

    def test_target_raises_the_hits_a_bucket_needs(self) -> None:
        report = merge_coverage([_run(1, target=2), _run(2, target=2)])
        assert report.entries[0].covered == []

    def test_an_item_defined_differently_is_not_merged(self) -> None:
        other = _run(1)
        other["items"][0]["buckets"] = ["1", "2"]
        other["items"][0]["hits"] = {"1": 1, "2": 0}
        report = merge_coverage([_run(1), other])
        assert [e.name for e in report.entries] == ["x", "x#2"]

    def test_a_file_of_another_schema_is_refused(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            merge_coverage([{"schema": "something-else"}])

    def test_odd_and_scenario_are_reported_apart(self) -> None:
        odd = CoverItem(
            "odd.a",
            lambda w: 1,
            values=[1],
            group=CoverGroup.ODD,
            event=SamplingEvent.END,
        )
        collector = CoverageCollector([odd, _item("x", values=[1, 2])])
        collector.end(_World(1), 0.0)
        report = merge_coverage([collector.to_dict("S")])
        assert report.groups() == ["odd", "scenario"]
        assert report.group_grade("odd") == 1.0
        assert report.group_grade("scenario") == 0.5
        markdown = report.to_markdown()
        assert "## ODD coverage: 100.0%" in markdown
        assert "## Scenario coverage: 50.0%" in markdown
        assert "| 2 (hole) | 0 | 0 |" in markdown

    def test_the_cli_merges_a_directory(self, tmp_path: Path) -> None:
        for i, value in enumerate((1, 2)):
            run_dir = tmp_path / "outputs" / f"run{i}"
            run_dir.mkdir(parents=True)
            (run_dir / "S_coverage.json").write_text(json.dumps(_run(value)))
        out_json = tmp_path / "report.json"
        out_md = tmp_path / "report.md"
        assert (
            main(
                [
                    str(tmp_path / "outputs"),
                    "--json",
                    str(out_json),
                    "--markdown",
                    str(out_md),
                ]
            )
            == 0
        )
        report = json.loads(out_json.read_text())
        assert report["runs"] == 2
        assert report["groups"]["scenario"]["grade"] == pytest.approx(2 / 3)
        assert "Holes" in out_md.read_text()
        assert load_and_merge([tmp_path / "outputs"]).runs == 2

    def test_the_cli_fails_when_there_is_nothing_to_merge(self, tmp_path: Path) -> None:
        assert main([str(tmp_path)]) == 1
