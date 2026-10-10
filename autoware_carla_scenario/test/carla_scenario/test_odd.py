"""ODDs: the model, the probes, the OpenODD reader, the registry and the CLI."""

from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import autoware_carla_scenario.odd as odd_pkg
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.coverage import CoverageCollector, merge_coverage
from autoware_carla_scenario.odd import (
    OddAttribute,
    OddDefinition,
    OddModule,
    all_of,
    any_of,
    default_odd,
    module_holds,
    probes,
    register_odd,
    reset_probes,
    resolve_odd,
)
from autoware_carla_scenario.odd import registry as odd_registry
from autoware_carla_scenario.odd.cli import main as odd_main
from autoware_carla_scenario.odd.openodd import OpenOddError, load_openodd
from autoware_carla_scenario.odd.probes import (
    illumination_level,
    intensity_level,
    traffic_density_level,
)
from autoware_carla_scenario.odd.registry import odd_builder
from autoware_carla_scenario.odd.units import UnitError, convert
from autoware_carla_scenario.typecheck import model_dir


def _attr(name: str = "x", **kw: Any) -> OddAttribute:
    return OddAttribute(name, lambda world: world.values.get(name), **kw)


class _World(SimpleNamespace):
    """A world whose probes (made by :func:`_attr`) read ``values``."""


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_conversions(self) -> None:
        assert convert(36.0, "km/h", "m/s") == pytest.approx(10.0)
        assert convert(10.0, "m/s", "kph") == pytest.approx(36.0)
        assert convert(1.0, "km", "m") == 1000.0
        assert convert(50.0, "ms", "s") == pytest.approx(0.05)

    def test_no_unit_on_either_side_is_no_conversion(self) -> None:
        assert convert(3.0, "", "km/h") == 3.0
        assert convert(3.0, "m/s", "") == 3.0
        assert convert(3.0, "furlong", "furlong") == 3.0

    def test_mismatched_or_unknown_units_are_refused(self) -> None:
        with pytest.raises(UnitError, match="cannot convert"):
            convert(1.0, "m", "s")
        with pytest.raises(UnitError, match="unknown unit"):
            convert(1.0, "furlong", "m")


# ---------------------------------------------------------------------------
# Conditions, modules, the ODD
# ---------------------------------------------------------------------------


class TestConditions:
    def test_intervals_include_and_exclude_their_bounds(self) -> None:
        x = _attr(range=(0, 100), every=10)
        values = lambda v: {"x": v}  # noqa: E731
        assert x.between(10, 20).evaluate(values(10), {}) is True
        assert x.between(10, 20).evaluate(values(20), {}) is True
        assert x.greater_than(10).evaluate(values(10), {}) is False
        assert x.at_least(10).evaluate(values(10), {}) is True
        assert x.less_than(10).evaluate(values(10), {}) is False
        assert x.at_most(10).evaluate(values(10), {}) is True

    def test_is_in_compares_labels(self) -> None:
        location = _attr("loc", values=["urban", "nonurban"])
        assert location.is_in(["urban"]).evaluate({"loc": "urban"}, {}) is True
        assert location.is_in(["urban"]).evaluate({"loc": "nonurban"}, {}) is False
        flag = _attr("flag", values=[False, True])
        assert flag.equals(True).evaluate({"flag": True}, {}) is True

    def test_an_unknown_value_is_unknown_not_false(self) -> None:
        x = _attr(range=(0, 10), every=1)
        assert x.at_least(1).evaluate({"x": None}, {}) is None
        y = _attr("y", values=[1])
        # Unknown and false is false; unknown or true is true.
        both = all_of([x.at_least(1), y.equals(2)])
        assert both.evaluate({"x": None, "y": 1}, {}) is False
        either = any_of([x.at_least(1), y.equals(1)])
        assert either.evaluate({"x": None, "y": 1}, {}) is True


class TestModulesAndOdd:
    def _odd(self, **kw: Any) -> tuple[OddDefinition, dict[str, OddAttribute]]:
        a = {
            "speed": _attr("speed", unit="km/h", buckets=[0, 30, 60, 90]),
            "loc": _attr("loc", values=["urban", "nonurban"]),
            "rain": _attr("rain", values=["none", "light", "heavy"]),
        }
        modules = [
            OddModule(
                "roads",
                include_and=[a["loc"].is_in(["urban"]), a["speed"].between(0, 60)],
            ),
            OddModule(
                "weather", exclude_or=[a["rain"].is_in(["heavy"])], labels=["ok"]
            ),
            OddModule("never", include_and=[a["speed"].at_least(1000)], active=False),
        ]
        return OddDefinition("t", list(a.values()), modules, **kw), a

    def test_inside_when_every_active_module_holds(self) -> None:
        odd, _ = self._odd()
        inside = {"speed": 40, "loc": "urban", "rain": "none"}
        assert odd.evaluate(inside).inside is True
        assert odd.evaluate({**inside, "rain": "heavy"}).inside is False
        assert odd.evaluate({**inside, "speed": 70}).inside is False
        verdict = odd.evaluate({**inside, "rain": None})
        assert verdict.inside is None
        assert verdict.modules == {"roads": True, "weather": None, "never": False}

    def test_a_root_module_decides_and_labels_refer_to_modules(self) -> None:
        odd, a = self._odd()
        root = OddModule(
            "root", include_and=[module_holds("roads"), module_holds("ok")]
        )
        odd = OddDefinition("t", odd.attributes, [*odd.modules, root], root="root")
        inside = {"speed": 40, "loc": "urban", "rain": "none"}
        assert odd.evaluate(inside).inside is True
        assert odd.evaluate({**inside, "rain": "heavy"}).inside is False

    def test_without_modules_everything_is_inside(self) -> None:
        odd = OddDefinition("t", [_attr(values=[1])])
        assert odd.evaluate({}).inside is True

    def test_buckets_outside_the_odd(self) -> None:
        odd, a = self._odd()
        assert odd.outside_buckets(a["speed"]) == ["[60, 90]"]
        assert odd.outside_buckets(a["loc"]) == ["nonurban"]
        assert odd.outside_buckets(a["rain"]) == ["heavy"]
        items = {i.name: i for i in odd.cover_items()}
        assert items["odd.loc"].outside == ["nonurban"]

    def test_a_condition_across_attributes_rules_out_no_bucket(self) -> None:
        a = _attr("a", values=[1, 2])
        b = _attr("b", values=[1, 2])
        odd = OddDefinition(
            "t",
            [a, b],
            [OddModule("m", include_and=[any_of([a.equals(1), b.equals(1)])])],
        )
        assert odd.outside_buckets(a) == [] and odd.outside_buckets(b) == []

    def test_an_inactive_or_unrequired_module_rules_out_no_bucket(self) -> None:
        odd, a = self._odd(root="roads")
        assert odd.outside_buckets(a["rain"]) == []

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (
                lambda a: [OddModule("m"), OddModule("m")],
                "modules named more than once",
            ),
            (
                lambda a: [OddModule("m", include_and=[module_holds("nope")])],
                "neither a module nor a label",
            ),
            (
                lambda a: [
                    OddModule("p", include_and=[module_holds("q")]),
                    OddModule("q", include_and=[module_holds("p")]),
                ],
                "refer to each other",
            ),
            (
                lambda a: [
                    OddModule("m", include_and=[_attr("other", values=[1]).equals(1)])
                ],
                "not one of the ODD's attributes",
            ),
        ],
    )
    def test_an_ill_formed_odd_is_refused(self, build: Any, message: str) -> None:
        x = _attr(values=[1])
        with pytest.raises(ValueError, match=message):
            OddDefinition("t", [x], build(x))

    def test_a_missing_root_is_refused(self) -> None:
        with pytest.raises(ValueError, match="root module"):
            OddDefinition("t", [_attr(values=[1])], root="nope")


# ---------------------------------------------------------------------------
# The collector and the report, with an ODD
# ---------------------------------------------------------------------------


def _speed_odd() -> OddDefinition:
    speed = _attr("speed", unit="km/h", buckets=[0, 30, 60, 90])
    hidden = _attr("hidden")  # monitored, no buckets
    return OddDefinition(
        "speed_odd",
        [speed, hidden],
        [
            OddModule("slow", include_and=[speed.at_most(60)]),
            OddModule("visible", exclude_or=[hidden.equals(True)]),
        ],
    )


def _drive(speeds: list[Any], hidden: Any = False) -> dict[str, Any]:
    collector = CoverageCollector([], odd=_speed_odd())
    collector.start(_World(values={}), 0.0)
    for i, speed in enumerate(speeds, start=1):
        collector.tick(_World(values={"speed": speed, "hidden": hidden}), i * 0.5)
    collector.end(_World(values={}), len(speeds) * 0.5)
    return collector.to_dict("S")


class TestCollectorWithOdd:
    def test_each_tick_is_judged_inside_outside_or_unknown(self) -> None:
        doc = _drive([10, 70, 80, 20, None, 75])
        odd = doc["odd"]
        assert odd["name"] == "speed_odd"
        assert odd["ticks"] == {"inside": 2, "outside": 3, "unknown": 1}
        assert odd["seconds"] == {"inside": 1.0, "outside": 1.5, "unknown": 0.5}
        assert odd["module_ticks"]["slow"] == {"failed_ticks": 3, "unknown_ticks": 1}
        assert odd["out_intervals"] == [[0.5, 1.5], [2.5, 3.0]]
        assert odd["unmeasured"] == ["hidden"]

    def test_attributes_without_buckets_are_read_for_the_verdict(self) -> None:
        doc = _drive([10], hidden=True)
        assert doc["odd"]["ticks"]["outside"] == 1
        assert [i["name"] for i in doc["items"]] == ["odd.speed"]

    def test_the_odd_attributes_are_items_with_their_outside_buckets(self) -> None:
        (item,) = _drive([10])["items"]
        assert item["outside_odd"] == ["[60, 90]"]
        assert item["group"] == "odd"


class TestReportWithOdd:
    def test_buckets_outside_the_odd_are_not_targets(self) -> None:
        report = merge_coverage([_drive([10, 40]), _drive([70])])
        (entry,) = report.entries
        assert entry.targets == ["[0, 30)", "[30, 60)"]
        assert entry.grade == 1.0
        assert entry.holes == []

    def test_exposure_is_merged_per_odd(self) -> None:
        report = merge_coverage([_drive([10, 70]), _drive([10])])
        exposure = report.odds["speed_odd"]
        assert exposure.runs == 2
        assert exposure.runs_outside == 1
        assert exposure.ticks == {"inside": 2, "outside": 1, "unknown": 0}
        markdown = report.to_markdown()
        assert "## ODD: speed_odd" in markdown
        assert "Runs: 2, of which left the ODD: 1" in markdown
        assert "(outside ODD, reached)" in markdown
        assert "Monitored but not covered (no buckets): hidden" in markdown
        assert report.to_dict()["odds"]["speed_odd"]["runs_outside"] == 1

    def test_a_cross_cell_with_an_outside_bucket_is_not_a_target(self) -> None:
        from autoware_carla_scenario.coverage import CrossItem

        odd = _speed_odd()
        speed_item = odd.cover_items()[0]
        other = _attr("o", values=[1, 2]).item
        assert other is not None
        collector = CoverageCollector(
            [other], [CrossItem("c", [speed_item, other])], odd=odd
        )
        collector.tick(_World(values={"speed": 10, "o": 1, "hidden": False}), 0.5)
        report = merge_coverage([collector.to_dict("S")])
        cross = next(e for e in report.entries if e.kind == "cross")
        assert len(cross.targets) == 4  # 2 inside speed buckets x 2
        assert cross.covered == ["[0, 30) / 1"]


# ---------------------------------------------------------------------------
# Probes and the default ODD
# ---------------------------------------------------------------------------


class _Vec(SimpleNamespace):
    pass


class _Actor:
    def __init__(
        self,
        actor_id: int,
        type_id: str,
        role: str = "",
        at: tuple[float, float] = (0.0, 0.0),
        speed_mps: float = 0.0,
        speed_limit: float = 0.0,
    ) -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role}
        self._at = _Vec(x=at[0], y=at[1], z=0.0)
        self._v = _Vec(x=speed_mps, y=0.0, z=0.0)
        self._limit = speed_limit

    def get_location(self) -> _Vec:
        return self._at

    def get_velocity(self) -> _Vec:
        return self._v

    def get_speed_limit(self) -> float:
        return self._limit

    def get_transform(self) -> Any:
        raise RuntimeError("no Lanelet2 map in this test")


class _Lane:
    def __init__(self, lane_id: int) -> None:
        self.lane_id = lane_id
        self.lane_type = "LaneType.Driving"
        self.is_junction = False
        self._left: Any = None
        self._right: Any = None

    def get_left_lane(self) -> Any:
        return self._left

    def get_right_lane(self) -> Any:
        return self._right


class _OddWorld:
    def __init__(self, actors: list[_Actor], waypoint: Any, weather: Any) -> None:
        self._actors = actors
        self._waypoint = waypoint
        self._weather = weather
        self.frame = 1
        self.map_fetches = 0

    def get_snapshot(self) -> Any:
        return SimpleNamespace(frame=self.frame)

    def get_actors(self) -> list[_Actor]:
        return self._actors

    def get_map(self) -> Any:
        self.map_fetches += 1
        return SimpleNamespace(get_waypoint=lambda loc: self._waypoint)

    def get_weather(self) -> Any:
        if self._weather is None:
            raise RuntimeError("no weather on this server")
        return self._weather


def _odd_world(weather: Any = "default") -> _OddWorld:
    ego = _Actor(1, "vehicle.ego", str(EGO_ROLE_NAME), speed_mps=10.0, speed_limit=50)
    actors = [
        ego,
        _Actor(2, "vehicle.a", at=(10.0, 0.0)),
        _Actor(3, "vehicle.b", at=(500.0, 0.0)),
        _Actor(4, "walker.pedestrian.0001", at=(5.0, 5.0)),
    ]
    # Two lanes in the ego's direction, one the other way.
    own = _Lane(-1)
    own._left = _Lane(1)
    own._right = _Lane(-2)
    if weather == "default":
        weather = SimpleNamespace(
            sun_altitude_angle=45.0, precipitation=50.0, fog_density=0.0
        )
    return _OddWorld(actors, own, weather)


@pytest.fixture(autouse=True)
def _fresh_probes() -> Any:
    reset_probes()
    yield
    reset_probes()


def _hits(world: Any, odd: OddDefinition | None = None) -> dict[str, list[str]]:
    collector = CoverageCollector([], odd=odd or default_odd())
    collector.tick(world, 0.05)
    return {
        i["name"]: [b for b, n in i["hits"].items() if n]
        for i in collector.to_dict("S")["items"]
    }


class TestProbes:
    def test_levels(self) -> None:
        assert [
            illumination_level(a) for a in (45, 5, -3, -30)
        ] == probes.ILLUMINATION_LEVELS
        assert [intensity_level(i) for i in (0, 10, 50, 90)] == probes.INTENSITY_LEVELS
        assert [
            traffic_density_level(n) for n in (0, 2, 5, 6)
        ] == probes.TRAFFIC_DENSITY_LEVELS

    def test_the_default_odd_reads_the_world(self) -> None:
        assert _hits(_odd_world()) == {
            "odd.scenery.junction": ["false"],
            "odd.scenery.location": [],
            "odd.scenery.road_type": [],
            "odd.scenery.speed_limit": ["[50, 60)"],
            "odd.scenery.lane_count": ["2"],
            "odd.environment.illumination": ["day"],
            "odd.environment.rain": ["moderate"],
            "odd.environment.fog": ["none"],
            "odd.dynamic.ego_speed": ["[30, 40)"],
            "odd.dynamic.traffic_density": ["low"],
            "odd.dynamic.pedestrian_nearby": ["true"],
        }

    def test_lanelet_tags_come_first_when_a_map_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            probes,
            "_lanelet_attributes",
            lambda world: {"location": "urban", "subtype": "road", "speed_limit": "40"},
        )
        hits = _hits(_odd_world())
        assert hits["odd.scenery.location"] == ["urban"]
        assert hits["odd.scenery.road_type"] == ["road"]
        assert hits["odd.scenery.speed_limit"] == ["[40, 50)"]

    def test_a_server_without_weather_leaves_the_weather_unknown(self) -> None:
        collector = CoverageCollector([], odd=default_odd())
        collector.tick(_odd_world(weather=None), 0.05)
        samples = {i["name"]: i["samples"] for i in collector.to_dict("S")["items"]}
        assert samples["odd.environment.rain"] == 0
        assert samples["odd.dynamic.ego_speed"] == 1

    def test_without_an_ego_nothing_is_sampled(self) -> None:
        collector = CoverageCollector([], odd=default_odd())
        collector.tick(_OddWorld([], None, None), 0.05)
        assert all(i["samples"] == 0 for i in collector.to_dict("S")["items"])

    def test_the_carla_map_is_fetched_once_per_run(self) -> None:
        world = _odd_world()
        collector = CoverageCollector([], odd=default_odd())
        for frame in range(3):
            world.frame = frame
            collector.tick(world, 0.05 * frame)
        assert world.map_fetches == 1
        reset_probes()
        collector.tick(world, 1.0)
        assert world.map_fetches == 2


# ---------------------------------------------------------------------------
# OpenODD YAML
# ---------------------------------------------------------------------------

_OPENODD = """
ODD:
  name: urban_test
  root: root_odd
  text: Urban roads, fair weather
TAXONOMY:
  env:
    visibility: float distance
    weather:
      rain: [none, light, moderate, heavy]
      fog:
        severity:
          none:
            env.visibility: ">= 500 m"
          light:
            env.visibility: "[200 .. 500] m"
          dense:
            env.visibility: "< 200 m"
  road:
    location: [urban, nonurban, private]
    speed_limit: int speed
    junction: boolean
  ego:
    speed: float velocity
  scene:
    ego_pose: "vehicle/pose"
COVERAGE:
  road.location: {probe: lanelet_location}
  road.speed_limit: {probe: speed_limit_kph}
  road.junction: {probe: in_junction}
  env.weather.rain: {probe: rain}
  env.visibility: {probe: "%(module)s:visibility_m", unit: m}
  ego.speed: {probe: ego_speed_kph, range: [0, 120], every: 10}
MODULES:
  urban_only:
    TITLE: Urban roads
    INCLUDE_AND:
      road.location: [urban]
      road.speed_limit: "<= 16.7 m/s"
  weather:
    LABEL: good_weather
    EXCLUDE_OR:
      env.weather.rain: [heavy]
      env.weather.fog.severity: [dense]
  slow_at_junctions:
    INCLUDE_OR:
      road.junction: false
      AND:
        road.junction: true
        ego.speed: "<= 30 km/h"
  dusk:
    ACTIVE: false
    INCLUDE_AND:
      ego.speed: "< 30 km/h"
  root_odd:
    INCLUDE_AND:
      urban_only: true
      good_weather: true
      slow_at_junctions: true
"""

VISIBILITY = {"value": 1000.0}


def visibility_m(world: Any) -> float:
    return VISIBILITY["value"]


def _load(text: str = _OPENODD, **kw: Any) -> OddDefinition:
    return load_openodd(text % {"module": __name__}, **kw)


class TestOpenOdd:
    def test_it_reads_the_odd(self) -> None:
        odd = _load()
        assert odd.name == "urban_test"
        assert odd.root == "root_odd"
        assert [m.name for m in odd.modules] == [
            "urban_only",
            "weather",
            "slow_at_junctions",
            "dusk",
            "root_odd",
        ]
        assert next(m for m in odd.modules if m.name == "dusk").active is False

    def test_conditions_are_converted_into_the_probes_unit(self) -> None:
        odd = _load()
        urban = next(m for m in odd.modules if m.name == "urban_only")
        assert urban.describe()["include_and"] == [
            "road.location in [urban]",
            "road.speed_limit <= 60.12",
        ]

    def test_buckets_come_from_coverage_the_taxonomy_or_the_thresholds(self) -> None:
        items = {i.name: i for i in _load().cover_items()}
        assert items["odd.road.location"].labels == ["urban", "nonurban", "private"]
        assert items["odd.road.junction"].labels == ["false", "true"]
        assert len(items["odd.ego.speed"].labels) == 12
        # No buckets given: one each side of every threshold the modules test.
        assert items["odd.road.speed_limit"].labels == ["[-inf, 60.12)", "[60.12, inf]"]
        assert items["odd.env.visibility"].labels == [
            "[-inf, 200)",
            "[200, 500)",
            "[500, inf]",
        ]
        # A derived enumeration has one bucket per literal.
        assert items["odd.env.weather.fog.severity"].labels == [
            "none",
            "light",
            "dense",
        ]

    def test_the_root_and_its_labels_rule_buckets_out(self) -> None:
        items = {i.name: i for i in _load().cover_items()}
        assert items["odd.road.location"].outside == ["nonurban", "private"]
        assert items["odd.road.speed_limit"].outside == ["[60.12, inf]"]
        assert items["odd.env.weather.rain"].outside == ["heavy"]
        assert items["odd.env.weather.fog.severity"].outside == ["dense"]
        assert items["odd.ego.speed"].outside == []  # only in an OR

    def test_a_derived_enumeration_is_its_first_literal_that_holds(self) -> None:
        odd = _load()
        fog = next(a for a in odd.attributes if a.name == "env.weather.fog.severity")
        for visibility, level in ((1000.0, "none"), (300.0, "light"), (100.0, "dense")):
            VISIBILITY["value"] = visibility
            assert fog.probe(None) == level
        VISIBILITY["value"] = 1000.0

    def test_the_odd_is_judged_like_a_python_one(self) -> None:
        odd = _load()
        inside = {
            "road.location": "urban",
            "road.speed_limit": 50.0,
            "road.junction": True,
            "ego.speed": 20.0,
            "env.weather.rain": "none",
            "env.visibility": 800.0,
            "env.weather.fog.severity": "none",
        }
        assert odd.evaluate(inside).inside is True
        assert odd.evaluate({**inside, "ego.speed": 40.0}).inside is False
        assert odd.evaluate(
            {**inside, "road.junction": False, "ego.speed": 40.0}
        ).inside
        assert odd.evaluate({**inside, "env.weather.rain": None}).inside is None

    def test_an_attribute_with_no_probe_is_unknown_and_unmeasured(self) -> None:
        text = """
TAXONOMY:
  road:
    type: [motorway, residential]
MODULES:
  m:
    INCLUDE_AND:
      road.type: [motorway]
"""
        odd = load_openodd(text, name="n")
        assert odd.name == "n"
        assert odd.unmeasured() == ["road.type"]
        assert odd.evaluate(odd.sample(None)).inside is None

    def test_several_files_merge(self, tmp_path: Path) -> None:
        taxonomy = tmp_path / "taxonomy.yaml"
        taxonomy.write_text("TAXONOMY:\n  road:\n    location: [urban, nonurban]\n")
        modules = tmp_path / "odd.yaml"
        modules.write_text(
            "COVERAGE:\n  road.location: {probe: lanelet_location}\n"
            "MODULES:\n  m:\n    INCLUDE_AND:\n      road.location: [urban]\n"
        )
        odd = load_openodd(taxonomy, modules)
        assert odd.name == "taxonomy"
        assert odd.cover_items()[0].outside == ["nonurban"]

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (
                "TAXONOMY:\n  a: [x]\nMODULES:\n  m:\n    INCLUDE_AND:\n      b: [x]\n",
                "neither",
            ),
            (
                "TAXONOMY:\n  a: float length\nMODULES:\n  m:\n    INCLUDE_AND:\n      a: fast\n",
                "cannot read",
            ),
            (
                "TAXONOMY:\n  a: float length\nCOVERAGE:\n  a: {probe: ego_speed_kph}\n"
                "MODULES:\n  m:\n    INCLUDE_AND:\n      a: '< 3 m'\n",
                "cannot convert",
            ),
            ("TAXONOMY:\n  a: [x]\nCOVERAGE:\n  b: {probe: rain}\n", "not in TAXONOMY"),
            ("TAXONOMY:\n  a: [x]\nCOVERAGE:\n  a: {probe: nope}\n", "unknown probe"),
            ("TAXONOMY:\n  a: complex number\n", "cannot read the type"),
            ("- not a mapping\n", "must be a mapping"),
        ],
    )
    def test_a_bad_document_is_refused(self, text: str, message: str) -> None:
        with pytest.raises(OpenOddError, match=message):
            load_openodd(text)


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------


def small_odd() -> OddDefinition:
    return OddDefinition("small", [OddAttribute("a", probes.rain, values=["none"])])


class TestRegistry:
    def test_names_files_and_functions(self, tmp_path: Path) -> None:
        assert resolve_odd(None).name == "default"
        assert resolve_odd("default").name == "default"
        assert resolve_odd(f"{__name__}:small_odd").name == "small"
        path = tmp_path / "x.yaml"
        path.write_text("ODD: {name: from_yaml}\nTAXONOMY:\n  a: [x]\n")
        assert resolve_odd(str(path)).name == "from_yaml"
        odd = small_odd()
        assert resolve_odd(odd) is odd

    def test_a_registered_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(odd_registry, "_REGISTRY", {})
        register_odd("small", small_odd)
        assert resolve_odd("small").name == "small"
        assert odd_builder("small") is small_odd
        assert "small" in odd_registry.odd_names()
        with pytest.raises(ValueError, match="built-in"):
            register_odd("default", small_odd)

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no ODD named"):
            resolve_odd("no_such_odd")

    def test_a_builder_that_returns_something_else_is_refused(self) -> None:
        with pytest.raises(ValueError, match="returned"):
            resolve_odd(f"{__name__}:_not_an_odd")

    def test_builtin_and_yaml_odds_have_no_builder_to_compile(self) -> None:
        assert odd_builder(None) is None
        assert odd_builder("default") is None
        assert odd_builder("x.yaml") is None


def _not_an_odd() -> int:
    return 3


class TestCli:
    def test_check_a_yaml_odd(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("TAXONOMY:\n  a: [x]\n")
        assert odd_main(["check", str(path), "default"]) == 0
        assert "[ok]" in capsys.readouterr().out

    def test_check_a_bad_odd(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "x.yaml"
        path.write_text("TAXONOMY:\n  a: complex number\n")
        assert odd_main(["check", str(path)]) == 1
        assert "[FAILED]" in capsys.readouterr().out

    def test_show(self, capsys: Any) -> None:
        assert odd_main(["show", "default"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["name"] == "default"
        assert any(a["name"] == "dynamic.ego_speed" for a in shown["attributes"])

    def test_list(self, capsys: Any) -> None:
        assert odd_main(["list"]) == 0
        assert "default" in capsys.readouterr().out.split()


# ---------------------------------------------------------------------------
# The Codon model of autoware_carla_scenario.odd
# ---------------------------------------------------------------------------


def test_the_model_declares_the_probes_as_python_does() -> None:
    tree = ast.parse(
        (model_dir() / "autoware_carla_scenario" / "odd.codon").read_text()
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    constants = {
        node.target.id: node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    for name, node in functions.items():
        python = getattr(odd_pkg, name)
        assert [a.arg for a in node.args.args] == list(
            inspect.signature(python).parameters
        ), name
    for name, value in constants.items():
        assert getattr(odd_pkg, name) == ast.literal_eval(value), name
    probe_names = {n for n in probes.__all__ if n not in ("reset_probes",)}
    probe_functions = {n for n in probe_names if inspect.isfunction(getattr(probes, n))}
    probe_functions -= {
        "illumination_level",
        "intensity_level",
        "traffic_density_level",
    }
    assert probe_functions <= set(functions), probe_functions - set(functions)


def test_threshold_buckets_are_open_ended() -> None:
    # Guards the label format the report relies on for open-ended buckets.
    items = {i.name: i for i in _load().cover_items()}
    edges = items["odd.road.speed_limit"].edges
    assert edges[0] == -math.inf and edges[-1] == math.inf
