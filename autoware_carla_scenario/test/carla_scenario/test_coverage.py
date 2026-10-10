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


class TestReviewRegressions:
    def test_labels_that_cannot_be_told_apart_are_refused(self) -> None:
        with pytest.raises(ValueError, match="too close"):
            _item(buckets=[1.0, 1.0 + 1e-13, 1.0 + 2e-13, 1.0 + 3e-13])

    def test_nan_is_no_sample(self) -> None:
        collector = CoverageCollector([_item(buckets=[0, 1], event=SamplingEvent.TICK)])
        collector.tick(_World(float("nan")), 0.0)
        item = collector.to_dict("S")["items"][0]
        assert item["samples"] == 0 and item["out_of_range"] == {}

    def test_crosses_over_different_buckets_are_not_merged(self) -> None:
        def run(edges: list[float]) -> dict[str, Any]:
            a = CoverItem("a", lambda w: 0.5, buckets=edges, event=SamplingEvent.TICK)
            b = CoverItem("b", lambda w: 1, values=[1, 2], event=SamplingEvent.TICK)
            collector = CoverageCollector([a, b], [CrossItem("ab", [a, b])])
            collector.tick(_World(None), 0.0)
            return collector.to_dict("S")

        report = merge_coverage([run([0, 1, 2]), run([0, 5, 10])])
        assert sorted(e.name for e in report.entries if e.kind == "cross") == [
            "ab",
            "ab#2",
        ]

    def test_files_of_another_schema_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "S_coverage.json").write_text(json.dumps(_run(1)))
        (tmp_path / "merged_coverage.json").write_text(json.dumps({"runs": 1}))
        assert load_and_merge([tmp_path]).runs == 1

    def test_a_scenario_cannot_take_an_odd_name(self) -> None:
        scenario = TestScenarioRegistration()._scenario()
        with pytest.raises(ValueError, match="odd."):
            scenario.register_cover("odd.x", lambda w: 1, values=[1])


class TestExposure:
    """#25: seconds, meters and entries per bucket, besides hits."""

    @pytest.fixture
    def drive(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Run a tick item over ``(value, x, elapsed)`` steps; the ego is at ``x``."""
        from autoware_carla_scenario.odd import probes

        def run(
            steps: list[tuple[Any, Optional[float], float]],
            **kw: Any,
        ) -> dict[str, Any]:
            item = _item(values=["a", "b"], event=SamplingEvent.TICK, **kw)
            collector = CoverageCollector([item])
            world = _World()
            position: list[Optional[float]] = [0.0]
            monkeypatch.setattr(
                probes,
                "ego_position",
                lambda w: None if position[0] is None else (position[0], 0.0, 0.0),
            )
            collector.start(world, 0.0)
            for value, x, elapsed in steps:
                world.value = value
                position[0] = x
                collector.tick(world, elapsed)
            collector.end(world, steps[-1][2])
            return collector.to_dict("s")["items"][0]

        return run

    def test_a_tick_counts_the_step_since_the_previous_one(self, drive: Any) -> None:
        out = drive([("a", 10.0, 1.0), ("a", 30.0, 2.0), ("b", 35.0, 2.5)])
        assert out["hits"] == {"a": 2, "b": 1}
        assert out["seconds"] == {"a": 2.0, "b": 0.5}
        assert out["meters"] == {"a": 30.0, "b": 5.0}
        assert out["entries"] == {"a": 1, "b": 1}

    def test_a_stopped_ego_adds_time_but_no_distance(self, drive: Any) -> None:
        out = drive([("a", 0.0, t) for t in (1.0, 2.0, 3.0)])
        assert out["seconds"]["a"] == 3.0
        assert out["meters"]["a"] == 0.0

    def test_leaving_and_coming_back_is_a_new_entry(self, drive: Any) -> None:
        out = drive(
            [("a", 1.0, 1.0), ("b", 2.0, 2.0), ("a", 3.0, 3.0), (None, 4.0, 4.0)]
            + [("a", 5.0, 5.0)]
        )
        assert out["entries"] == {"a": 3, "b": 1}

    def test_a_teleport_adds_no_distance(self, drive: Any) -> None:
        out = drive([("a", 1.0, 1.0), ("a", 5000.0, 1.05)])
        assert out["meters"]["a"] == 1.0

    def test_no_ego_means_no_distance(self, drive: Any) -> None:
        out = drive([("a", None, 1.0), ("a", None, 2.0)])
        assert out["meters"]["a"] == 0.0
        assert out["seconds"]["a"] == 2.0

    def test_a_one_shot_sample_is_an_entry_of_no_duration(self) -> None:
        collector = CoverageCollector([_item(values=["a", "b"])])
        collector.start(_World(), 0.0)
        collector.tick(_World(), 5.0)
        collector.end(_World("a"), 5.0)
        out = collector.to_dict("s")["items"][0]
        assert out["entries"] == {"a": 1, "b": 0}
        assert out["seconds"] == {"a": 0.0, "b": 0.0}

    def test_a_cross_cell_has_exposure_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoware_carla_scenario.odd import probes

        monkeypatch.setattr(probes, "ego_position", lambda w: (w.x, 0.0, 0.0))
        x = _item("x", values=["a", "b"], event=SamplingEvent.TICK)
        y = _item(
            "y",
            values=["c"],
            event=SamplingEvent.TICK,
            expression=lambda world: "c",
        )
        collector = CoverageCollector([x, y], [CrossItem("xy", [x, y])])
        world = SimpleNamespace(value="a", x=0.0)
        collector.start(world, 0.0)
        world.x = 4.0
        collector.tick(world, 1.0)
        cross = collector.to_dict("s")["crosses"][0]
        assert cross["meters"]["a / c"] == 4.0
        assert cross["entries"]["a / c"] == 1

    def test_runs_sum_and_an_old_file_makes_the_measure_unknown(self) -> None:
        def doc(**measures: Any) -> dict[str, Any]:
            item = {
                **_item(values=["a"], event=SamplingEvent.TICK).describe(),
                "hits": {"a": 2},
                **measures,
            }
            return {"schema": COVERAGE_SCHEMA, "scenario": "s", "items": [item]}

        new = {"seconds": {"a": 1.5}, "meters": {"a": 10.0}, "entries": {"a": 1}}
        report = merge_coverage([doc(**new), doc(**new)])
        bucket = report.to_dict()["groups"]["scenario"]["entries"][0]["buckets"][0]
        assert (bucket["seconds"], bucket["meters"], bucket["entries"]) == (
            3.0,
            20.0,
            2,
        )
        assert "| a | 4 | 2 | 3.0 | 20 | 2 |" in report.to_markdown()
        old = merge_coverage([doc(**new), doc()])
        bucket = old.to_dict()["groups"]["scenario"]["entries"][0]["buckets"][0]
        assert bucket["meters"] is None
        assert "| a | 4 | 2 | ? | ? | ? |" in old.to_markdown()


class TestCriteria:
    """#26: cover_by and min_stay."""

    @pytest.fixture
    def drive(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from autoware_carla_scenario.odd import probes

        def run(steps: list[tuple[Any, float, float]], **kw: Any) -> dict[str, Any]:
            item = _item(values=["a", "b"], event=SamplingEvent.TICK, **kw)
            collector = CoverageCollector([item])
            world = SimpleNamespace(value=None, x=0.0)
            monkeypatch.setattr(probes, "ego_position", lambda w: (w.x, 0.0, 0.0))
            collector.start(world, 0.0)
            for value, x, elapsed in steps:
                world.value, world.x = value, x
                collector.tick(world, elapsed)
            collector.end(world, steps[-1][2])
            return {
                "schema": COVERAGE_SCHEMA,
                "scenario": "s",
                "items": collector.to_dict("s")["items"],
            }

        return run

    def test_a_bucket_can_be_covered_by_distance(self, drive: Any) -> None:
        doc = drive(
            [("a", 30.0, 1.0), ("a", 60.0, 2.0), ("b", 70.0, 3.0)],
            cover_by="meters",
            target=50,
        )
        entry = merge_coverage([doc]).entries[0]
        assert entry.covered == ["a"]
        assert entry.holes == ["b"]
        assert "| b (hole: 10/50 m) |" in merge_coverage([doc]).to_markdown()

    def test_a_short_stay_does_not_count(self, drive: Any) -> None:
        # "b" is clipped for 0.5 s between two long stays in "a".
        doc = drive(
            [("a", 10.0, 1.0), ("a", 20.0, 2.0), ("b", 25.0, 2.5)]
            + [("a", 35.0, 3.5), ("a", 45.0, 4.5)],
            cover_by="entries",
            min_stay=1.0,
        )
        raw = doc["items"][0]
        assert raw["entries"] == {"a": 2, "b": 1}
        assert raw["counted"]["entries"] == {"a": 2, "b": 0}
        assert raw["counted"]["meters"] == {"a": 40.0, "b": 0}
        entry = merge_coverage([doc]).entries[0]
        assert entry.holes == ["b"]
        assert "counting stays of 1 s or more" in merge_coverage([doc]).to_markdown()

    def test_the_last_stay_counts_when_long_enough(self, drive: Any) -> None:
        doc = drive([("b", 0.0, 1.0), ("b", 0.0, 3.0)], cover_by="seconds", target=3)
        assert merge_coverage([doc]).entries[0].covered == ["b"]
        doc = drive(
            [("b", 0.0, 1.0), ("b", 0.0, 3.0)],
            cover_by="seconds",
            target=3,
            min_stay=2.0,
        )
        assert doc["items"][0]["counted"]["seconds"]["b"] == 3.0

    def test_runs_sum_the_counted_amounts(self, drive: Any) -> None:
        doc = drive([("a", 30.0, 1.0)], cover_by="meters", target=50, min_stay=0.5)
        report = merge_coverage([doc, doc])
        assert report.entries[0].amount("a") == 60.0
        assert report.entries[0].covered == ["a"]

    @pytest.mark.parametrize(
        ("kw", "message"),
        [
            ({"cover_by": "laps"}, "cover_by must be one of"),
            ({"target": 0}, "target must be at least 1"),
            ({"cover_by": "seconds", "target": 0}, "target must be positive"),
            ({"min_stay": -1, "event": SamplingEvent.TICK}, "must not be negative"),
            ({"cover_by": "meters"}, "need an item sampled on SamplingEvent.TICK"),
            ({"min_stay": 1.0}, "need an item sampled on SamplingEvent.TICK"),
        ],
    )
    def test_criteria_that_cannot_be_met_are_refused(
        self, kw: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _item(values=["a"], **kw)

    def test_entries_need_no_tick(self) -> None:
        assert _item(values=["a"], cover_by="entries", target=2).cover_by == "entries"

    def test_criteria_change_the_merge_key(self, drive: Any) -> None:
        by_hits = drive([("a", 1.0, 1.0)])
        by_meters = drive([("a", 1.0, 1.0)], cover_by="meters")
        names = [e.name for e in merge_coverage([by_hits, by_meters]).entries]
        assert names == ["x", "x#2"]

    def test_an_odd_attribute_and_a_binding_take_criteria(self, tmp_path: Path) -> None:
        from autoware_carla_scenario.odd import OddAttribute, OpenOddError, load_openodd

        attribute = OddAttribute(
            "a", lambda w: 1, values=[1], cover_by="meters", target=5, min_stay=1
        )
        assert attribute.item is not None
        assert (attribute.item.cover_by, attribute.item.target) == ("meters", 5)
        taxonomy = "TAXONOMY:\n    wind_speed: float velocity\n"
        odd = load_openodd(
            taxonomy,
            bindings={
                "wind_speed": {
                    "probe": "ego_speed_kph",
                    "unit": "km/h",
                    "buckets": [0, 10, 20],
                    "cover_by": "meters",
                    "target": 100,
                    "min_stay": 2,
                }
            },
        )
        item = odd.attributes[0].item
        assert item is not None
        assert (item.cover_by, item.target, item.min_stay) == ("meters", 100.0, 2.0)
        with pytest.raises(OpenOddError, match=r"unknown keys \['targets'\]"):
            load_openodd(
                taxonomy, bindings={"wind_speed": {"probe": "rain", "targets": 3}}
            )
