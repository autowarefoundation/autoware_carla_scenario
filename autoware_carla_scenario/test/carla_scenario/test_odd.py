"""ODDs: the model, the probes, the OpenODD reader, the registry and the CLI."""

from __future__ import annotations

import ast
import inspect
import json
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
from autoware_carla_scenario.odd.openodd import (
    OpenOddError,
    load_odd_binding,
    load_openodd,
)
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
                include_and=[a["loc"].is_in(["urban"]), a["speed"].less_than(60)],
            ),
            OddModule(
                "weather", exclude_or=[a["rain"].is_in(["heavy"])], labels=["ok"]
            ),
            OddModule("never", include_and=[a["speed"].at_least(1000)], active=False),
        ]
        return OddDefinition("t", list(a.values()), modules, **kw), a

    def test_unreferenced_active_modules_are_the_roots(self) -> None:
        odd, _ = self._odd()
        assert odd.roots == ["roads", "weather"]
        inside = {"speed": 40, "loc": "urban", "rain": "none"}
        assert odd.evaluate(inside).inside is True
        assert odd.evaluate({**inside, "rain": "heavy"}).inside is False
        assert odd.evaluate({**inside, "speed": 70}).inside is False

    def test_a_missing_value_does_not_put_a_situation_outside(self) -> None:
        # OpenODD's missing-value semantics: open world unless required.
        odd, _ = self._odd()
        verdict = odd.evaluate({"speed": 40, "loc": "urban", "rain": None})
        assert verdict.inside is True
        assert verdict.assumed is True
        assert verdict.modules == {"roads": True, "weather": None, "never": "inactive"}
        settled = odd.evaluate({"speed": 70, "loc": "urban", "rain": None})
        assert settled.inside is False and settled.assumed is False

    def test_unknown_makes_a_value_required(self) -> None:
        rain = _attr("rain", values=["none", "heavy"])
        odd = OddDefinition(
            "t", [rain], [OddModule("m", exclude_or=[rain.is_unknown()])]
        )
        assert odd.evaluate({"rain": None}).inside is False
        assert odd.evaluate({"rain": "none"}).inside is True

    def test_a_root_module_decides_and_labels_refer_to_modules(self) -> None:
        odd, a = self._odd()
        root = OddModule(
            "root", include_and=[module_holds("roads"), module_holds("ok")]
        )
        odd = OddDefinition("t", odd.attributes, [*odd.modules, root], roots=["root"])
        assert odd.roots == ["root"]
        inside = {"speed": 40, "loc": "urban", "rain": "none"}
        assert odd.evaluate(inside).inside is True
        assert odd.evaluate({**inside, "rain": "heavy"}).inside is False

    def test_a_reference_counts_a_module_out_of_the_roots(self) -> None:
        odd, a = self._odd()
        root = OddModule("root", exclude_or=[module_holds("ok", False)])
        odd = OddDefinition("t", odd.attributes, [*odd.modules, root])
        assert odd.roots == ["roads", "root"]  # weather is referenced via its label

    def test_a_condition_on_an_inactive_module_is_satisfied(self) -> None:
        odd, a = self._odd()
        for holds in (True, False):
            root = OddModule("root", include_and=[module_holds("never", holds)])
            with_root = OddDefinition(
                "t", odd.attributes, [*odd.modules, root], roots=["root"]
            )
            assert with_root.evaluate({"speed": 5}).inside is True

    def test_without_modules_everything_is_inside(self) -> None:
        odd = OddDefinition("t", [_attr(values=[1])])
        assert odd.evaluate({}).inside is True

    def test_buckets_outside_the_odd(self) -> None:
        odd, a = self._odd()
        assert odd.outside_buckets(a["speed"]) == ["[60, 90]"]
        assert odd.outside_buckets(a["loc"]) == ["nonurban"]
        assert odd.outside_buckets(a["rain"]) == ["heavy"]
        assert [i.name for i in odd.cover_items()] == [
            "odd.speed",
            "odd.loc",
            "odd.rain",
        ]

    def test_a_module_the_odd_excludes_rules_its_leaves_out(self) -> None:
        rain = _attr("rain", values=["none", "light", "heavy"])
        odd = OddDefinition(
            "t",
            [rain],
            [
                OddModule("bad_weather", include_or=[rain.is_in(["heavy"])]),
                OddModule("root", exclude_or=[module_holds("bad_weather")]),
            ],
        )
        assert odd.roots == ["root"]
        assert odd.outside_buckets(rain) == ["heavy"]

    def test_a_bucket_partly_inside_stays_a_target(self) -> None:
        speed = _attr("speed", buckets=[0, 30, 60, 90])
        odd = OddDefinition(
            "t", [speed], [OddModule("m", include_and=[speed.at_most(60)])]
        )
        # [60, 90] holds 60, which is inside.
        assert odd.outside_buckets(speed) == []
        odd = OddDefinition(
            "t",
            [speed],
            [OddModule("m", include_or=[speed.less_than(30), speed.greater_than(60)])],
        )
        assert odd.outside_buckets(speed) == ["[30, 60)"]

    def test_a_condition_across_attributes_rules_out_no_bucket(self) -> None:
        a = _attr("a", values=[1, 2])
        b = _attr("b", values=[1, 2])
        odd = OddDefinition(
            "t",
            [a, b],
            [OddModule("m", include_and=[any_of([a.equals(1), b.equals(1)])])],
        )
        assert odd.outside_buckets(a) == [] and odd.outside_buckets(b) == []

    def test_an_unrequired_module_rules_out_no_bucket(self) -> None:
        odd, a = self._odd(roots=["roads"])
        assert odd.outside_buckets(a["rain"]) == []

    def test_a_module_has_one_include_and_one_exclude_section(self) -> None:
        x = _attr(values=[1])
        with pytest.raises(ValueError, match="one include section"):
            OddModule("m", include_and=[x.equals(1)], include_or=[x.equals(1)])
        with pytest.raises(ValueError, match="one exclude section"):
            OddModule("m", exclude_and=[x.equals(1)], exclude_or=[x.equals(1)])

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
            (
                lambda a: [OddModule("m", labels=["x"])],
                "also a module's or an attribute's name",
            ),
        ],
    )
    def test_an_ill_formed_odd_is_refused(self, build: Any, message: str) -> None:
        x = _attr(values=[1])
        with pytest.raises(ValueError, match=message):
            OddDefinition("t", [x], build(x))

    def test_a_missing_root_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no root module"):
            OddDefinition("t", [_attr(values=[1])], roots=["nope"])


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
            OddModule("slow", include_and=[speed.less_than(60)]),
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
    def test_each_tick_is_judged_inside_assumed_or_outside(self) -> None:
        doc = _drive([10, 70, 80, 20, None, 75])
        odd = doc["odd"]
        assert odd["name"] == "speed_odd"
        assert odd["roots"] == ["slow", "visible"]
        assert odd["ticks"] == {"inside": 2, "assumed": 1, "outside": 3}
        assert odd["seconds"] == {"inside": 1.0, "assumed": 0.5, "outside": 1.5}
        assert odd["module_ticks"]["slow"] == {"failed_ticks": 3, "missing_ticks": 1}
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
        report = merge_coverage([_drive([10, 70]), _drive([10, None])])
        exposure = report.odds["speed_odd"]
        assert exposure.runs == 2
        assert exposure.runs_outside == 1
        assert exposure.ticks == {"inside": 2, "assumed": 1, "outside": 1}
        markdown = report.to_markdown()
        assert "## ODD: speed_odd" in markdown
        assert "Runs: 2, of which left the ODD: 1" in markdown
        assert "Inside only because values were missing: 0.5 s" in markdown
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
        self.road_id, self.section_id = 1, 0
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
            "_lanelet_tags",
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

# A taxonomy and modules in the shape of the standard's own examples
# (ASAM OpenODD 1.0, chapter 10: Code 170, 171, 189, 192).
_TAXONOMY = """
TAXONOMY:
    environment_conditions:
        rainfall_rate: float precipitation_rate
        rainfall_level:
            no_rain:
                rainfall_rate: "< 0.1 mm/h"
            light_rain:
                rainfall_rate: "[0.1 .. 2.5] mm/h"
            moderate_rain:
                rainfall_rate: "[2.5 .. 7.6] mm/h"
            heavy_rain:
                rainfall_rate: "> 7.6 mm/h"
        wind_speed: float velocity
        is_dangerous_wind:
            true:
                wind_speed: "> 50 km/h"
            false:
                wind_speed: "<= 50 km/h"
    scenery:
        road_type: [town_local, dead_end, town_expressway, expressway]
        current_road: road_type
        lane_count: integer count
        is_mixed_zone: boolean
        record_of_categoricals:
            surface: [dry, wet]
            marking: [solid, dashed]
    connectivity:
        downlink_latency: float time
        downlink_throughput: float bandwidth
    vehicle: reusable_pose
"""

_MODULES = """
IMPORT:
    - taxonomy.yml
ODD:
    odd1:
        TITLE: The baseline ODD
        INCLUDE_AND:
            low_speed_roads: true
            good_connectivity: true
        EXCLUDE_OR:
            bad_weather: true
MODULES:
    low_speed_roads:
        TITLE: Low speed traffic conditions
        INCLUDE_AND:
            road_type:
                - town_local
                - dead_end
            lane_count: "< 3"
            OR:
                rainfall_rate: "<= 2 mm/h"
                is_mixed_zone: false
    bad_weather_1:
        LABEL: bad_weather
        INCLUDE_OR:
            rainfall_level: ">= heavy_rain"
            is_dangerous_wind: true
    bad_weather_2:
        LABELS: bad_weather
        ACTIVE: "false"
        INCLUDE_AND:
            road_type: expressway
    good_connectivity:
        INCLUDE_AND:
            downlink_latency: "< 10 msec"
            downlink_throughput: "> 1 Mbps"
    required_data:
        EXCLUDE_OR:
            wind_speed: unknown
"""

RAIN = {"value": 0.0}
WIND = {"value": 1.0}


def rain_rate(world: Any) -> float:
    return RAIN["value"]


def wind_mps(world: Any) -> float:
    return WIND["value"]


def _write(tmp_path: Path, bindings: str = "") -> Path:
    (tmp_path / "taxonomy.yml").write_text(_TAXONOMY)
    (tmp_path / "odd.yml").write_text(_MODULES)
    binding = tmp_path / "urban.binding.yaml"
    binding.write_text(
        "openodd: [odd.yml]\n"
        "name: spec_example\n"
        "text: The standard's example\n"
        "probes:\n"
        "  road_type: {probe: lanelet_location}\n"
        f"  rainfall_rate: {{probe: '{__name__}:rain_rate', unit: mm/h}}\n"
        f"  wind_speed: {{probe: '{__name__}:wind_mps', unit: m/s}}\n"
        "  lane_count: {probe: lane_count, values: [1, 2, 3]}\n" + bindings
    )
    return binding


def _items(odd: OddDefinition) -> dict[str, Any]:
    return {i.name: i for i in odd.cover_items()}


def _outside(odd: OddDefinition) -> dict[str, list[str]]:
    return {a.name: odd.outside_buckets(a) for a in odd.attributes}


class TestOpenOdd:
    def test_it_reads_the_standards_shape(self, tmp_path: Path) -> None:
        odd = load_odd_binding(_write(tmp_path))
        assert odd.name == "spec_example"
        assert odd.text == "The standard's example"
        # Modules under ODD are root candidates; one no module refers to is a root.
        assert odd.roots == ["odd1"]
        modules = {m.name: m for m in odd.modules}
        assert modules["bad_weather_2"].active is False
        assert modules["bad_weather_1"].labels == ["bad_weather"]
        assert modules["bad_weather_2"].labels == ["bad_weather"]

    def test_concepts_are_named_by_id_and_units_are_converted(
        self, tmp_path: Path
    ) -> None:
        odd = load_odd_binding(_write(tmp_path))
        roads = next(m for m in odd.modules if m.name == "low_speed_roads")
        assert roads.describe()["include_and"] == [
            "scenery.road_type in [dead_end, town_local]",
            "scenery.lane_count < 3",
            "(environment_conditions.rainfall_rate <= 2 or "
            "scenery.is_mixed_zone == false)",
        ]
        attrs = {a.name: a for a in odd.attributes}
        dangerous = next(m for m in odd.modules if m.name == "bad_weather_1")
        assert "is_dangerous_wind in [true]" in dangerous.describe()["include_or"][1]
        # 50 km/h in the probe's m/s.
        assert attrs["environment_conditions.wind_speed"].item is not None
        assert _items(odd)["odd.environment_conditions.wind_speed"].labels == [
            "[-inf, 13.8889)",
            "[13.8889, inf]",
        ]

    def test_a_categorical_defined_by_ranges_is_ordered(self, tmp_path: Path) -> None:
        odd = load_odd_binding(_write(tmp_path))
        bad = next(m for m in odd.modules if m.name == "bad_weather_1")
        assert bad.describe()["include_or"][0] == (
            "environment_conditions.rainfall_level in [heavy_rain]"
        )
        level = next(a for a in odd.attributes if a.name.endswith("rainfall_level"))
        for rate, literal in (
            (0.0, "no_rain"),
            (2.5, "light_rain"),
            (9.0, "heavy_rain"),
        ):
            RAIN["value"] = rate
            assert level.probe(None) == literal
        RAIN["value"] = 0.0

    def test_buckets_and_what_the_odd_rules_out(self, tmp_path: Path) -> None:
        odd = load_odd_binding(_write(tmp_path))
        items, outside = _items(odd), _outside(odd)
        assert items["odd.scenery.road_type"].labels == [
            "town_local",
            "dead_end",
            "town_expressway",
            "expressway",
        ]
        assert outside["scenery.road_type"] == ["town_expressway", "expressway"]
        assert outside["scenery.lane_count"] == ["3"]
        # bad_weather (a label, one active module) must not hold.
        assert outside["environment_conditions.rainfall_level"] == ["heavy_rain"]
        assert outside["environment_conditions.is_dangerous_wind"] == ["true"]
        # Thresholds the ODD tests become bucket edges, in the probe's unit.
        assert items["odd.environment_conditions.rainfall_rate"].labels == [
            "[-inf, 0.1)",
            "[0.1, 2)",
            "[2, 2.5)",
            "[2.5, 7.6)",
            "[7.6, inf]",
        ]
        # No probe, no buckets.
        assert "odd.connectivity.downlink_latency" not in items

    def test_the_odd_is_judged_with_missing_value_semantics(
        self, tmp_path: Path
    ) -> None:
        odd = load_odd_binding(_write(tmp_path))
        values = odd.sample(None)
        values.update({"scenery.road_type": "town_local", "scenery.lane_count": 2})
        verdict = odd.evaluate(values)
        # Connectivity is never measured: open world, so inside, but assumed.
        assert verdict.inside is True and verdict.assumed is True
        assert odd.evaluate({**values, "scenery.lane_count": 3}).inside is False
        RAIN["value"] = 9.0
        assert (
            odd.evaluate(odd.sample(None) | {"scenery.road_type": "town_local"}).inside
            is False
        )
        RAIN["value"] = 0.0
        # required_data is a second root: a missing wind speed is outside.
        assert "required_data" not in odd.roots  # odd1 is the only ODD root

    def test_a_document_on_its_own_measures_nothing(self, tmp_path: Path) -> None:
        _write(tmp_path)
        odd = resolve_odd(str(tmp_path / "odd.yml"))
        assert odd.name == "odd"
        assert odd.cover_items() == []
        assert "scenery.road_type" in odd.unmeasured()

    def test_without_an_odd_section_unreferenced_modules_are_roots(self) -> None:
        modules = """
MODULES:
    roads:
        INCLUDE_AND:
            lane_count: "< 3"
    weather:
        LABEL: fair
        EXCLUDE_OR:
            rainfall_level: [heavy_rain]
    main:
        INCLUDE_AND:
            fair: true
"""
        odd = load_openodd(_TAXONOMY + modules, name="n")
        assert odd.roots == ["roads", "main"]

    def test_an_import_cycle_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "a.yml").write_text("IMPORT: [b.yml]\n")
        (tmp_path / "b.yml").write_text("IMPORT: [a.yml]\n")
        with pytest.raises(OpenOddError, match="IMPORT cycle"):
            load_openodd(tmp_path / "a.yml")

    @pytest.mark.parametrize(
        ("modules", "message"),
        [
            ("m: {INCLUDE_AND: {nope: [x]}}", "neither a concept"),
            ("m: {INCLUDE_AND: {lane_count: fast}}", "cannot read"),
            (
                "m: {INCLUDE_AND: {wind_speed: '< 3 m'}}",
                "measures length, not velocity",
            ),
            ("m: {INCLUDE_AND: {road_type: [motorway]}}", "has no literal motorway"),
            (
                "m: {INCLUDE_AND: {road_type: '< dead_end'}}",
                "literals defined by ranges",
            ),
            ("m: {INCLUDE_AND: {lane_count: '< 1.75*ego_width'}}", "unknown unit"),
            ("m: {INCLUDE_AND: {AND: {lane_count: 1}}}", "nests OR, not AND"),
            ("m: {INCLUDE_AND: {OR: {AND: {lane_count: 1}}}}", "nest one level only"),
            (
                "m: {INCLUDE_AND: {lane_count: 1}, INCLUDE_OR: {lane_count: 2}}",
                "at most one INCLUDE",
            ),
            ("m: {TITLE: empty}", "needs an INCLUDE"),
            ("m: {INCLUDE_AND: {lane_count: 1}, COLOR: red}", "unknown keys"),
            (
                "unknown_module: {INCLUDE_AND: {lane_count: 1}}",
                "must not contain 'unknown'",
            ),
            (
                "m: {LABEL: road_type, INCLUDE_AND: {lane_count: 1}}",
                "labels named like",
            ),
            (
                "m: {ACTIVE: maybe, INCLUDE_AND: {lane_count: 1}}",
                "expected true or false",
            ),
        ],
    )
    def test_a_bad_module_is_refused(self, modules: str, message: str) -> None:
        text = _TAXONOMY + "MODULES:\n    " + modules + "\n"
        with pytest.raises(OpenOddError, match=message):
            load_openodd(text, name="n")

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("TAXONOMY:\n  a: complex number\n", "cannot read the type"),
            ("TAXONOMY:\n  a: [x]\nCOVERAGE:\n  a: {probe: rain}\n", "binding file"),
            ("- not a mapping\n", "must be a mapping"),
        ],
    )
    def test_a_bad_document_is_refused(self, text: str, message: str) -> None:
        with pytest.raises(OpenOddError, match=message):
            load_openodd(text)

    def test_a_bad_binding_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(OpenOddError, match="unknown probe"):
            load_odd_binding(_write(tmp_path, "  is_mixed_zone: {probe: nope}\n"))
        with pytest.raises(OpenOddError, match="neither a concept"):
            load_odd_binding(_write(tmp_path, "  nope: {probe: rain}\n"))
        bad = tmp_path / "bad.yaml"
        bad.write_text("probes: {}\n")
        with pytest.raises(OpenOddError, match="under 'openodd'"):
            load_odd_binding(bad)

    def test_a_record_of_categoricals_is_not_a_derived_categorical(self) -> None:
        odd = load_openodd(_TAXONOMY, name="n")
        names = {a.name for a in odd.attributes}
        assert {
            "scenery.record_of_categoricals.surface",
            "scenery.record_of_categoricals.marking",
        } <= names
        # A concept typed by a categorical takes its literals; a record type is not followed.
        assert "scenery.current_road" in names
        assert "vehicle" not in names


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
        path.write_text("TAXONOMY:\n  a: [x]\n")
        assert resolve_odd(str(path)).name == "x"
        binding = tmp_path / "b.yaml"
        binding.write_text(
            "openodd: [x.yaml]\nname: bound\nprobes: {a: {probe: rain}}\n"
        )
        assert resolve_odd(str(binding)).name == "bound"
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
        assert any(a["name"] == "odd.dynamic.ego_speed" for a in shown["attributes"])

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
