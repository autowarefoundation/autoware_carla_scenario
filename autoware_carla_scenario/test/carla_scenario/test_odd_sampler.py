"""Sampling concrete scenarios from an ODD (:mod:`autoware_carla_scenario.odd.sampler`)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.coverage.collector import COVERAGE_SCHEMA
from autoware_carla_scenario.coverage.report import load_and_merge
from autoware_carla_scenario.odd import (
    INTENSITY_LEVELS,
    OddAttribute,
    OddDefinition,
    OddDraws,
    OddKnob,
    OddModule,
    OddSampler,
    default_odd,
    illumination,
    knobs_from_mapping,
    rain,
)
from autoware_carla_scenario.measures import VEHICLE_AHEAD_GAP_M
from autoware_carla_scenario.odd import scenario_measure
from autoware_carla_scenario.odd.probes import illumination_level, intensity_level
from autoware_carla_scenario.odd.sampler import (
    controls_from_mapping,
    decode_case,
    sampler_from_config,
)
from autoware_carla_scenario.sweeper.expand import expand_config, expand_sweep


def _value(overrides: list[str], key: str) -> float:
    for override in overrides:
        name, _, value = override.partition("=")
        if name == key:
            return float(value)
    raise AssertionError(f"{key} not in {overrides}")


def _speed() -> OddAttribute:
    return OddAttribute(
        "dynamic.lead_speed", lambda world: None, unit="km/h", buckets=[0, 30, 60, 90]
    )


def _weather_odd() -> OddDefinition:
    """Rain and illumination, both settable through the built-in knobs."""
    return OddDefinition(
        "weather",
        [
            OddAttribute("environment.rain", rain, values=INTENSITY_LEVELS),
            OddAttribute(
                "environment.illumination",
                illumination,
                values=["day", "low_sun", "twilight", "night"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# What is drawn
# ---------------------------------------------------------------------------


def test_built_in_probes_need_no_knob() -> None:
    sampler = OddSampler(default_odd(), seed=3)

    assert sampler.attributes == [
        "environment.illumination",
        "environment.rain",
        "environment.fog",
    ]


def test_a_drawn_value_reads_back_as_its_bucket() -> None:
    """The knob's ranges are the probes' own levels, read backwards."""
    for case in OddSampler(default_odd(), seed=7).sample(50):
        precipitation = _value(case.overrides, "environment.rain")
        fog_density = _value(case.overrides, "environment.fog_density")
        sun = _value(case.overrides, "environment.sun_altitude_angle")
        assert intensity_level(precipitation) == case.buckets["environment.rain"]
        assert intensity_level(fog_density) == case.buckets["environment.fog"]
        assert illumination_level(sun) == case.buckets["environment.illumination"]


def test_the_same_seed_draws_the_same_cases() -> None:
    first = [c.overrides for c in OddSampler(default_odd(), seed=11).sample(10)]
    again = [c.overrides for c in OddSampler(default_odd(), seed=11).sample(10)]
    other = [c.overrides for c in OddSampler(default_odd(), seed=12).sample(10)]

    assert first == again
    assert first != other


def test_nothing_outside_the_odd_is_drawn() -> None:
    base = _weather_odd()
    rain_attr = base.attributes[0]
    odd = OddDefinition(
        "weather",
        base.attributes,
        [OddModule("no_storms", exclude_or=[rain_attr.is_in(["heavy"])])],
    )

    buckets = Counter(
        c.buckets["environment.rain"] for c in OddSampler(odd, seed=1).sample(100)
    )

    assert "heavy" not in buckets
    assert set(buckets) == {"none", "light", "moderate"}


def test_a_combination_the_odd_rules_out_is_not_drawn() -> None:
    """No bucket is out on its own; only together: no rain at night."""
    base = _weather_odd()
    rain_attr, light = base.attributes
    odd = OddDefinition(
        "weather",
        base.attributes,
        [
            OddModule(
                "no_rain_at_night",
                exclude_and=[
                    rain_attr.is_in(["moderate", "heavy"]),
                    light.is_in(["night"]),
                ],
            )
        ],
    )

    for case in OddSampler(odd, seed=2).sample(200):
        assert not (
            case.buckets["environment.illumination"] == "night"
            and case.buckets["environment.rain"] in ("moderate", "heavy")
        )


def test_a_condition_finer_than_a_bucket_holds_for_the_value() -> None:
    """[30, 60) is partly inside a 0-45 km/h ODD; what is drawn in it is inside."""
    speed = _speed()
    odd = OddDefinition(
        "slow", [speed], [OddModule("slow", include_and=[speed.between(0, 45)])]
    )
    sampler = OddSampler(
        odd, {"dynamic.lead_speed": OddKnob("scenario.lead_speed_kmh")}, seed=5
    )

    cases = sampler.sample(100)

    assert {c.buckets["dynamic.lead_speed"] for c in cases} == {"[0, 30)", "[30, 60)"}
    assert all(0 <= c.values["dynamic.lead_speed"] <= 45 for c in cases)


def test_a_numeric_knob_scales_and_rounds() -> None:
    speed = _speed()
    odd = OddDefinition("speed", [speed])
    sampler = OddSampler(
        odd,
        {"dynamic.lead_speed": OddKnob("scenario.lead_speed_mps", scale=1 / 3.6)},
        seed=4,
    )
    for case in sampler.sample(30):
        written = _value(case.overrides, "scenario.lead_speed_mps")
        assert written == pytest.approx(
            case.values["dynamic.lead_speed"] / 3.6, abs=1e-3
        )

    integer = OddSampler(
        odd,
        {"dynamic.lead_speed": OddKnob("scenario.lead_speed_kmh", integer=True)},
        seed=4,
    )
    for case in integer.sample(30):
        (override,) = case.overrides
        assert override.split("=")[1].isdigit()


# ---------------------------------------------------------------------------
# Aiming at what earlier runs left uncovered
# ---------------------------------------------------------------------------


def _entry(name: str, buckets: list[str], hits: dict[str, int], **extra):
    def amount(bucket: str) -> float:
        return hits.get(bucket, 0)

    return SimpleNamespace(
        name=name,
        buckets=buckets,
        target=1.0,
        amount=amount,
        holes=[b for b in buckets if not hits.get(b)],
        **extra,
    )


def test_coverage_draws_the_holes_first() -> None:
    covered = {"none": 5, "light": 3, "moderate": 1}
    coverage = SimpleNamespace(
        entries=[_entry("odd.environment.rain", list(INTENSITY_LEVELS), covered)]
    )
    sampler = OddSampler(_weather_odd(), seed=0, strategy="coverage", coverage=coverage)

    first = sampler.sample(1)[0]

    assert first.buckets["environment.rain"] == "heavy"


def test_coverage_spreads_a_batch_over_the_buckets() -> None:
    sampler = OddSampler(_weather_odd(), seed=9, strategy="coverage")

    cases = sampler.sample(8)

    assert Counter(c.buckets["environment.rain"] for c in cases) == {
        label: 2 for label in INTENSITY_LEVELS
    }
    assert Counter(c.buckets["environment.illumination"] for c in cases) == {
        label: 2 for label in ("day", "low_sun", "twilight", "night")
    }


def test_an_uncovered_situation_is_aimed_at_first() -> None:
    base = _weather_odd()
    rain_attr, light = base.attributes
    odd = OddDefinition(
        "weather",
        base.attributes,
        [
            OddModule(
                "storm_at_night",
                include_and=[rain_attr.is_in(["heavy"]), light.is_in(["night"])],
                situation=True,
            )
        ],
    )
    coverage = SimpleNamespace(
        entries=[_entry("odd.situation.storm_at_night", ["holds"], {})]
    )
    sampler = OddSampler(odd, seed=0, strategy="coverage", coverage=coverage)

    first, second = sampler.sample(2)

    assert sampler.situation_holes == ["storm_at_night"]
    assert first.situation == "storm_at_night"
    assert first.buckets == {
        "environment.rain": "heavy",
        "environment.illumination": "night",
    }
    assert second.situation is None


def test_a_situation_no_knob_brings_about_is_given_up(caplog) -> None:
    speed = _speed()
    base = _weather_odd()
    odd = OddDefinition(
        "weather",
        [*base.attributes, speed],
        [
            OddModule(
                "impossible",
                include_and=[
                    speed.between(10, 20),
                    base.attributes[0].is_in(["heavy"]),
                ],
                situation=True,
            ),
            OddModule("bound", exclude_or=[base.attributes[0].is_in(["heavy"])]),
        ],
    )
    coverage = SimpleNamespace(
        entries=[_entry("odd.situation.impossible", ["holds"], {})]
    )
    sampler = OddSampler(
        odd, seed=0, strategy="coverage", coverage=coverage, max_tries=20
    )

    (case,) = sampler.sample(1)

    assert case.situation is None
    assert "impossible" in caplog.text


def test_coverage_is_read_from_earlier_runs_coverage_files(tmp_path: Path) -> None:
    document = {
        "schema": COVERAGE_SCHEMA,
        "scenario": "earlier",
        "items": [
            {
                "name": "odd.environment.rain",
                "group": "odd",
                "kind": "categorical",
                "event": "tick",
                "buckets": list(INTENSITY_LEVELS),
                "hits": {"none": 40, "light": 12, "moderate": 3},
                "target": 1,
            }
        ],
    }
    (tmp_path / "Earlier_coverage.json").write_text(json.dumps(document))

    sampler, count = sampler_from_config(
        {"count": 1, "strategy": "coverage", "coverage_from": [str(tmp_path)]},
        odd=_weather_odd(),
    )

    assert count == 1
    assert load_and_merge([tmp_path]).runs == 1
    assert sampler.sample(1)[0].buckets["environment.rain"] == "heavy"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_knobs_from_config() -> None:
    knobs = knobs_from_mapping(
        {
            "dynamic.lead_speed": "scenario.lead_speed_kmh",
            "environment.rain": {
                "key": "environment.precipitation",
                "values": {"none": 0, "heavy": [80, 100]},
            },
        }
    )

    assert knobs["dynamic.lead_speed"] == OddKnob("scenario.lead_speed_kmh")
    assert knobs["environment.rain"].values["heavy"] == [80, 100]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"x": {"values": {}}}, "key is required"),
        ({"x": {"key": "k", "colour": 1}}, "unknown fields"),
        ({"x": 3}, "expected a key or a mapping"),
        ({"x": {"key": "k", "values": {"a": [5, 1]}}}, "low, high"),
    ],
)
def test_bad_knobs_are_refused(raw, message) -> None:
    with pytest.raises(ValueError, match=message):
        knobs_from_mapping(raw)


def test_a_knob_given_replaces_the_default() -> None:
    sampler = OddSampler(
        _weather_odd(),
        {"environment.rain": OddKnob("scenario.rain", values={"heavy": 99})},
        seed=0,
    )

    (case,) = sampler.sample(1)

    # Only the bucket the knob gives a value to can be drawn.
    assert case.buckets["environment.rain"] == "heavy"
    assert "scenario.rain=99" in case.overrides


def test_refusals() -> None:
    with pytest.raises(ValueError, match="no attributes"):
        OddSampler(_weather_odd(), {"no.such": OddKnob("k")})
    with pytest.raises(ValueError, match="nothing of ODD"):
        OddSampler(OddDefinition("map", [_speed()]))
    with pytest.raises(ValueError, match="no buckets"):
        OddSampler(
            OddDefinition("x", [OddAttribute("a", lambda w: None)]), {"a": OddKnob("k")}
        )
    with pytest.raises(ValueError, match="strategy"):
        OddSampler(_weather_odd(), strategy="greedy")
    with pytest.raises(ValueError, match="unknown fields"):
        sampler_from_config({"count": 1, "cases": 3}, odd=_weather_odd())


# ---------------------------------------------------------------------------
# A sweep that draws from the ODD
# ---------------------------------------------------------------------------


def test_a_sweep_of_samples_alone_needs_no_map() -> None:
    cfg = OmegaConf.create(
        {"odd": "default", "sweep": {"odd_sample": {"count": 3, "seed": 1}}}
    )

    cases = expand_config(cfg, ["ego.entity=autoware"])

    assert len(cases) == 3
    for case in cases:
        assert case[-1] == "ego.entity=autoware"
        # The settings, then the case itself for the run to record.
        assert case[-2].startswith("++odd_case=")
        assert {o.split("=")[0] for o in case[:-2]} == {
            "environment.sun_altitude_angle",
            "environment.rain",
            "environment.fog_density",
            # Where the case's grip is in the range its rain leaves.
            "environment.tire_friction_quantile",
        }


def test_drawn_cases_take_the_lanelet_cases_in_turn(monkeypatch) -> None:
    from autoware_carla_scenario.sweeper import expand as expand_module

    monkeypatch.setattr(
        expand_module,
        "_expand_lanelets",
        lambda sweep, lanelet_map: [
            ["ego.spawn_lanelet_id=1"],
            ["ego.spawn_lanelet_id=2"],
        ],
    )
    odd_sample: dict[str, Any] = {"count": 3}
    sweep = {"constraints": {"ego.spawn_lanelet_id": [{}]}, "odd_sample": odd_sample}

    cases = expand_sweep(sweep, None, ["x=1"], odd="default")

    assert [c[0] for c in cases] == [
        "ego.spawn_lanelet_id=1",
        "ego.spawn_lanelet_id=2",
        "ego.spawn_lanelet_id=1",
    ]
    assert all(c[-1] == "x=1" and len(c) == 7 for c in cases)

    # Without a count, one drawn case per lanelet case.
    del odd_sample["count"]
    assert len(expand_sweep(sweep, None, (), odd="default")) == 2


def test_the_environment_config_sets_the_weather_before_the_run() -> None:
    from autoware_carla_scenario import EnvironmentAction
    from autoware_carla_scenario.examples.run import add_environment

    registered: list = []
    scenario = SimpleNamespace(register_init=registered.append)
    cfg = OmegaConf.create(
        {
            "environment": {
                "precipitation": 80,
                "sun_altitude_angle": -10,
                "fog_density": None,
            }
        }
    )

    add_environment(cfg, scenario)  # type: ignore[arg-type]

    (action,) = registered
    assert isinstance(action, EnvironmentAction)
    assert action.settings == {"precipitation": 80.0, "sun_altitude_angle": -10.0}

    registered.clear()
    add_environment(OmegaConf.create({"environment": {"fog_density": None}}), scenario)  # type: ignore[arg-type]
    add_environment(OmegaConf.create({}), scenario)  # type: ignore[arg-type]
    assert registered == []


def test_a_wet_road_in_the_config_follows_the_vehicles_spawned_later() -> None:
    from autoware_carla_scenario.actions.environment import follow_tire_friction
    from autoware_carla_scenario.examples.run import add_environment

    init: list = []
    post: list = []
    scenario = SimpleNamespace(
        register_init=init.append, register_post_tick=post.append
    )

    add_environment(
        OmegaConf.create({"environment": {"rain": 80, "tire_friction_quantile": 0.2}}),
        scenario,  # type: ignore[arg-type]
    )
    assert init[0].sets_tire_friction
    assert post == [follow_tire_friction]

    # Fog leaves the road's grip, and the vehicles, alone.
    init.clear()
    post.clear()
    add_environment(OmegaConf.create({"environment": {"fog_density": 40}}), scenario)  # type: ignore[arg-type]
    assert len(init) == 1 and post == []


def test_each_case_draws_its_grip_the_same_way_every_time() -> None:
    cfg = OmegaConf.create(
        {"odd": "default", "sweep": {"odd_sample": {"count": 8, "seed": 3}}}
    )

    def quantiles(cases: list[list[str]]) -> list[float]:
        return [
            float(o.split("=", 1)[1])
            for case in cases
            for o in case
            if o.startswith("environment.tire_friction_quantile=")
        ]

    first = quantiles(expand_config(cfg))
    assert len(first) == 8
    assert all(0.0 <= q <= 1.0 for q in first)
    assert len(set(first)) > 1
    assert quantiles(expand_config(cfg)) == first

    # The case carries it, for its run to record.
    case = expand_config(cfg)[0]
    sample, _space = decode_case(case[-1].split("=", 1)[1])
    assert any(
        o.startswith("environment.tire_friction_quantile=") for o in sample.overrides
    )


def test_a_sweep_can_leave_the_grip_at_the_middle() -> None:
    cfg = OmegaConf.create(
        {
            "odd": "default",
            "sweep": {"odd_sample": {"count": 2, "tire_friction": False}},
        }
    )

    for case in expand_config(cfg):
        assert not any("tire_friction" in o for o in case)


def test_a_case_that_names_its_grip_keeps_it() -> None:
    from autoware_carla_scenario.odd.sampler import OddSample, with_tire_friction

    named = OddSample(0, ["environment.tire_friction_quantile=0.9"], {}, {})
    drawn = OddSample(1, ["environment.rain=80"], {}, {})

    kept, added = with_tire_friction([named, drawn], seed=0)

    assert kept == named
    assert added.overrides[0] == "environment.rain=80"
    assert added.overrides[1].startswith("environment.tire_friction_quantile=")


# ---------------------------------------------------------------------------
# Edges a draw must not cross
# ---------------------------------------------------------------------------


def test_an_unbounded_bucket_needs_a_range_from_its_knob() -> None:
    """OpenODD threshold buckets reach -inf and inf: nothing to draw from there."""
    attr = OddAttribute("x.v", lambda world: None, buckets=[-math.inf, 50, math.inf])
    odd = OddDefinition("x", [attr])

    # Without a range, only the bounded side can be drawn -- and none is.
    with pytest.raises(ValueError, match="no bucket"):
        OddSampler(odd, {"x.v": OddKnob("x.v")})
    with pytest.raises(ValueError, match="no bucket"):
        OddSampler(
            OddDefinition(
                "y", [attr], [OddModule("m", include_and=[attr.at_least(50)])]
            ),
            {"x.v": OddKnob("x.v")},
        )
    ranged = OddSampler(
        odd,
        {"x.v": OddKnob("x.v", values={"[-inf, 50)": [0, 50], "[50, inf]": [50, 80]})},
    )

    for case in ranged.sample(40):
        value = _value(case.overrides, "x.v")
        assert math.isfinite(value) and 0 <= value <= 80


def test_categorical_values_are_tested_as_values_not_labels() -> None:
    lanes = OddAttribute("scenery.lanes", lambda world: None, values=[1, 2, 3])
    odd = OddDefinition(
        "two", [lanes], [OddModule("two", include_and=[lanes.equals(2)])]
    )
    sampler = OddSampler(
        odd,
        {"scenery.lanes": OddKnob("scenario.lanes", values={"1": 1, "2": 2, "3": 3})},
    )

    cases = sampler.sample(5)

    assert {c.buckets["scenery.lanes"] for c in cases} == {"2"}
    assert all(c.values["scenery.lanes"] == 2 for c in cases)
    assert all(c.overrides == ["scenario.lanes=2"] for c in cases)


class _Visibility:
    """An attribute derived from others, as an OpenODD categorical is."""

    def __call__(self, world: object) -> None:
        return None

    @staticmethod
    def from_values(values: dict) -> object:
        rain_value, fog_value = (
            values.get("environment.rain"),
            values.get("environment.fog"),
        )
        if not isinstance(rain_value, str) or not isinstance(fog_value, str):
            return None
        bad = {"moderate", "heavy"}
        return "poor" if rain_value in bad and fog_value in bad else "good"


def test_attributes_derived_from_drawn_ones_are_worked_out() -> None:
    from autoware_carla_scenario.odd import fog

    visibility = OddAttribute(
        "environment.visibility", _Visibility(), values=["good", "poor"]
    )
    odd = OddDefinition(
        "clear",
        [
            OddAttribute("environment.rain", rain, values=INTENSITY_LEVELS),
            OddAttribute("environment.fog", fog, values=INTENSITY_LEVELS),
            visibility,
        ],
        [OddModule("clear", exclude_or=[visibility.is_in(["poor"])])],
    )

    for case in OddSampler(odd, seed=0).sample(60):
        poor = {"moderate", "heavy"}
        assert not (
            case.buckets["environment.rain"] in poor
            and case.buckets["environment.fog"] in poor
        )


def test_rounding_does_not_carry_a_value_into_another_bucket() -> None:
    attr = OddAttribute("x.v", lambda world: None, buckets=[0, 0.4, 0.6, 1])
    sampler = OddSampler(
        OddDefinition("x", [attr]), {"x.v": OddKnob("x.v", integer=True)}, seed=0
    )

    for case in sampler.sample(30):
        (override,) = case.overrides
        written = int(override.split("=")[1])
        assert attr.item is not None
        assert attr.item.bucket_of(written) == case.buckets["x.v"]
    assert all(c.buckets["x.v"] != "[0.4, 0.6)" for c in sampler.sample(30))


def test_what_is_written_reads_back_exactly() -> None:
    """Rendered in full: 29.9999 must not come back as 30, a level up."""
    for case in OddSampler(default_odd(), seed=21).sample(300):
        precipitation = _value(case.overrides, "environment.rain")
        assert intensity_level(precipitation) == case.buckets["environment.rain"]


# ---------------------------------------------------------------------------
# Situations and coverage entries
# ---------------------------------------------------------------------------


def _storm_odd(*, speed_too: bool = False) -> OddDefinition:
    base = _weather_odd()
    rain_attr, light = base.attributes
    attributes = list(base.attributes)
    conditions = [
        rain_attr.is_in(["heavy", "moderate"]),
        light.is_in(["night", "twilight"]),
    ]
    if speed_too:
        speed = _speed()
        attributes.append(speed)
        conditions.append(speed.between(10, 20))
    return OddDefinition(
        "weather",
        attributes,
        [OddModule("storm_at_night", include_and=conditions, situation=True)],
    )


def test_a_situation_no_run_measured_is_aimed_at() -> None:
    sampler = OddSampler(_storm_odd(), seed=0, strategy="coverage")

    (case,) = sampler.sample(1)

    assert sampler.situation_holes == ["storm_at_night"]
    assert case.situation == "storm_at_night"
    assert case.buckets["environment.rain"] in ("heavy", "moderate")
    assert case.buckets["environment.illumination"] in ("night", "twilight")


def test_a_covered_situation_is_not_aimed_at() -> None:
    coverage = SimpleNamespace(
        entries=[_entry("odd.situation.storm_at_night", ["holds"], {"holds": 3})]
    )
    sampler = OddSampler(_storm_odd(), seed=0, strategy="coverage", coverage=coverage)

    assert sampler.situation_holes == []


def test_a_situation_on_what_no_knob_sets_gets_a_case_it_can_hold_in() -> None:
    sampler = OddSampler(_storm_odd(speed_too=True), seed=0, strategy="coverage")

    (case,) = sampler.sample(1)

    assert case.situation == "storm_at_night"
    assert "dynamic.lead_speed" not in case.buckets
    assert case.buckets["environment.rain"] in ("heavy", "moderate")


def test_coverage_of_another_definition_of_an_item_is_not_read() -> None:
    stale = _entry("odd.environment.rain", ["dry", "wet"], {"dry": 9, "wet": 9})
    variant = _entry(
        "odd.environment.rain#2",
        list(INTENSITY_LEVELS),
        {"none": 5, "light": 5, "moderate": 5},
    )
    sampler = OddSampler(
        _weather_odd(),
        seed=0,
        strategy="coverage",
        coverage=SimpleNamespace(entries=[stale, variant]),
    )

    assert sampler.sample(1)[0].buckets["environment.rain"] == "heavy"


def test_boolean_bucket_keys_from_yaml() -> None:
    knobs = knobs_from_mapping({"x": {"key": "k", "values": {True: 1, False: 0}}})

    assert set(knobs["x"].values) == {"true", "false"}


def test_a_zero_scale_is_refused() -> None:
    with pytest.raises(ValueError, match="scale"):
        OddKnob("k", scale=0)


def test_a_point_range_on_an_integer_knob_writes_an_integer() -> None:
    lanes = OddAttribute("scenery.lanes", lambda world: None, values=["few", "many"])
    sampler = OddSampler(
        OddDefinition("x", [lanes]),
        {"scenery.lanes": OddKnob("x.n", values={"many": [5, 5]}, integer=True)},
    )

    assert sampler.sample(1)[0].overrides == ["x.n=5"]


def test_an_inactive_situation_is_not_aimed_at() -> None:
    base = _storm_odd()
    (situation,) = base.modules
    situation.active = False
    sampler = OddSampler(
        OddDefinition("weather", base.attributes, [situation]),
        seed=0,
        strategy="coverage",
    )

    assert sampler.situation_holes == []
    assert sampler.sample(1)[0].situation is None


def test_coverage_counted_another_way_is_not_read() -> None:
    by_seconds = _entry(
        "odd.environment.rain",
        list(INTENSITY_LEVELS),
        {"none": 5, "light": 5, "moderate": 5},
        cover_by="seconds",
    )
    sampler = OddSampler(
        _weather_odd(),
        seed=0,
        strategy="coverage",
        coverage=SimpleNamespace(entries=[by_seconds]),
    )

    # The ODD counts hits; seconds say nothing about them.
    assert sampler.coverage_amounts == {}


# ---------------------------------------------------------------------------
# A scenario's own controls, joined to the ODD through measures
# ---------------------------------------------------------------------------


def _gap_odd(name: str = "dynamic.lead_gap") -> OddDefinition:
    """An ODD whose own taxonomy maps a gap attribute onto the gap measure."""
    gap = OddAttribute(
        name,
        scenario_measure(VEHICLE_AHEAD_GAP_M),
        unit="m",
        buckets=[0, 10, 20, 30, 50],
    )
    return OddDefinition("gap", [gap])


_GAP_CONTROL = {VEHICLE_AHEAD_GAP_M: OddKnob("scenario.npc_ahead_m", range=(6.0, 30.0))}


def test_a_control_is_joined_to_the_attribute_mapped_onto_its_measure() -> None:
    """The scenario names a measure, the ODD maps its own name onto it."""
    for name in ("dynamic.lead_gap", "my_taxonomy.headway"):
        sampler = OddSampler(_gap_odd(name), controls=_GAP_CONTROL, seed=0)
        (case,) = sampler.sample(1)
        assert sampler.attributes == [name]
        assert case.overrides[0].startswith("scenario.npc_ahead_m=")


def test_a_range_keeps_draws_to_what_the_scenario_can_stage() -> None:
    sampler = OddSampler(_gap_odd(), controls=_GAP_CONTROL, seed=0)

    cases = sampler.sample(60)

    # [30, 50) meets the range only at 30: not a bucket the scenario drives in.
    assert {c.buckets["dynamic.lead_gap"] for c in cases} == {
        "[0, 10)",
        "[10, 20)",
        "[20, 30)",
    }
    assert all(6.0 <= c.values["dynamic.lead_gap"] <= 30.0 for c in cases)


def test_a_point_range_is_that_point() -> None:
    sampler = OddSampler(
        _gap_odd(), controls={VEHICLE_AHEAD_GAP_M: OddKnob("k", range=(12.0, 12.0))}
    )

    assert sampler.sample(1)[0].overrides == ["k=12.0"]


def test_limits_keep_what_is_written_to_what_the_key_takes() -> None:
    """A speed written as the ego's plus a relative one stays a speed."""
    speed = OddAttribute(
        "dynamic.lead_relative_speed",
        scenario_measure("vehicle_ahead_relative_speed_kph"),
        unit="km/h",
        buckets=[-30, -15, -5, 5, 15],
    )
    knob = OddKnob(
        "scenario.npc_initial_speed_kmh",
        offset=5.0,
        range=(-15.0, 15.0),
        limits=(0.0, math.inf),
    )
    sampler = OddSampler(
        OddDefinition("speed", [speed]),
        controls={"vehicle_ahead_relative_speed_kph": knob},
        seed=0,
    )

    cases = sampler.sample(60)

    # An ego at 5 km/h leaves no lead 15 to 5 km/h slower than it.
    assert {c.buckets["dynamic.lead_relative_speed"] for c in cases} == {
        "[-5, 5)",
        "[5, 15]",
    }
    written = [float(c.overrides[0].split("=")[1]) for c in cases]
    assert all(0.0 <= w <= 20.0 for w in written)
    assert knob.reach == (-5.0, 15.0)


def test_limits_from_config_take_an_open_end() -> None:
    knobs = knobs_from_mapping({"x": {"key": "k", "limits": [0, None]}})

    assert knobs["x"].limits == (0.0, math.inf)
    with pytest.raises(ValueError, match="limits must be"):
        knobs_from_mapping({"x": {"key": "k", "limits": [5, 1]}})


def test_a_slow_ego_draws_no_negative_cut_in_speed() -> None:
    from autoware_carla_scenario.examples.run import _compose_config
    from autoware_carla_scenario.sweeper.expand import scenario_controls

    controls = scenario_controls(
        _compose_config("cut_in/left", ["ego.initial_speed_kmh=5.0"])
    )
    sampler, _ = sampler_from_config(
        {"count": 40, "seed": 1}, odd=_relative_speed_odd(), controls=controls
    )

    for case in sampler.sample(40):
        assert float(case.overrides[0].split("=")[1]) >= 0.0


def _relative_speed_odd() -> OddDefinition:
    speed = OddAttribute(
        "dynamic.lead_relative_speed",
        scenario_measure("vehicle_ahead_relative_speed_kph"),
        unit="km/h",
        buckets=[-15, -5, 5, 15],
    )
    return OddDefinition("speed", [speed])


def test_controls_of_measures_the_odd_does_not_map_are_unused() -> None:
    sampler, _ = sampler_from_config(
        {"count": 1},
        odd=_gap_odd(),
        controls={
            VEHICLE_AHEAD_GAP_M: {"key": "scenario.npc_ahead_m", "range": [6, 30]},
            "crossing_pedestrian_gap_m": {"key": "scenario.trigger_distance_m"},
            "a_measure_of_its_own": "scenario.x",
        },
    )

    assert sampler.attributes == ["dynamic.lead_gap"]


def test_an_attribute_not_mapped_onto_a_measure_is_not_controlled() -> None:
    probe_odd = OddDefinition(
        "probe", [OddAttribute("dynamic.lead_gap", lambda w: None, buckets=[0, 10, 20])]
    )
    with pytest.raises(ValueError, match="nothing of ODD"):
        OddSampler(probe_odd, controls=_GAP_CONTROL)


def test_sweep_knobs_replace_a_scenarios_controls() -> None:
    sampler, _ = sampler_from_config(
        {"count": 1, "knobs": {"dynamic.lead_gap": "sweep.key"}},
        odd=_gap_odd(),
        controls={VEHICLE_AHEAD_GAP_M: "scenario.npc_ahead_m"},
    )

    (case,) = sampler.sample(1)
    assert case.overrides[0].startswith("sweep.key=")
    # Knobs the sweep gives are for this ODD: a stray one is an error.
    with pytest.raises(ValueError, match="no attributes"):
        sampler_from_config({"count": 1, "knobs": {"no.such": "k"}}, odd=_gap_odd())


def test_the_example_scenarios_declare_their_controls_by_measure() -> None:
    from autoware_carla_scenario.examples.run import _compose_config
    from autoware_carla_scenario.sweeper.expand import scenario_controls

    cut_in = scenario_controls(_compose_config("cut_in/left", []))
    assert set(cut_in) == {VEHICLE_AHEAD_GAP_M, "vehicle_ahead_relative_speed_kph"}
    # The gap is drawn as the measure itself, which cut_in solves its NPC's
    # placement for.
    assert cut_in[VEHICLE_AHEAD_GAP_M]["key"] == "scenario.npc_gap_m"
    assert cut_in[VEHICLE_AHEAD_GAP_M]["target"] == "measure"
    # The relative speed is written on top of the ego's own, resolved.
    assert cut_in["vehicle_ahead_relative_speed_kph"]["offset"] == 30.0

    dart_out = scenario_controls(
        _compose_config("pedestrian_dart_out/pedestrian_dart_out", [])
    )
    assert set(dart_out) == {
        "crossing_pedestrian_gap_m",
        "crossing_pedestrian_speed_ms",
    }
    assert scenario_controls(_compose_config("lane_change/left", [])) == {}


def test_a_measure_targeted_control_writes_the_drawn_value_unchanged() -> None:
    sampler, _ = sampler_from_config(
        {"count": 30, "seed": 2},
        odd=_gap_odd(),
        controls={
            VEHICLE_AHEAD_GAP_M: {
                "key": "scenario.npc_gap_m",
                "target": "measure",
                "range": [6, 30],
            }
        },
    )

    for case in sampler.sample(30):
        (override,) = case.overrides
        key, value = override.split("=")
        assert key == "scenario.npc_gap_m"
        assert float(value) == pytest.approx(case.values["dynamic.lead_gap"])
        assert 6.0 <= float(value) <= 30.0


def test_a_control_is_a_raw_setting_unless_it_says_otherwise() -> None:
    raw = {"key": "scenario.npc_initial_speed_kmh", "offset": 30.0, "range": [-15, 15]}

    assert controls_from_mapping({"m": raw}) == knobs_from_mapping({"m": raw})
    assert controls_from_mapping({"m": {**raw, "target": "setting"}}) == (
        knobs_from_mapping({"m": raw})
    )
    assert controls_from_mapping({"m": "k"}) == knobs_from_mapping({"m": "k"})


@pytest.mark.parametrize(
    "mapping", [{"scale": 2.0}, {"offset": 1.0}, {"integer": True}, {"values": {}}]
)
def test_a_measure_targeted_control_maps_nothing(mapping: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="target: measure writes the measure"):
        controls_from_mapping({"m": {"key": "k", "target": "measure", **mapping}})


def test_a_control_target_is_setting_or_measure() -> None:
    with pytest.raises(ValueError, match="target must be one of"):
        controls_from_mapping({"m": {"key": "k", "target": "placement"}})


def test_sweep_knobs_may_target_the_measure_too() -> None:
    gap_knob = {"key": "sweep.gap", "target": "measure", "range": [8, 9]}
    sampler, _ = sampler_from_config(
        {"count": 1, "knobs": {"dynamic.lead_gap": gap_knob}},
        odd=_gap_odd(),
    )

    (case,) = sampler.sample(1)
    assert 8.0 <= float(case.overrides[0].split("=")[1]) <= 9.0


def test_a_scenarios_controls_are_drawn_without_naming_them_in_the_sweep() -> None:
    cfg = OmegaConf.create(
        {
            "odd": "default",
            "ego": {"initial_speed_kmh": 30.0},
            "controls": {
                "vehicle_ahead_relative_speed_kph": {
                    "key": "scenario.npc_initial_speed_kmh",
                    "offset": "${ego.initial_speed_kmh}",
                    "range": [-15, 15],
                }
            },
            "sweep": {"odd_sample": {"count": 20, "seed": 3}},
        }
    )

    for case in expand_config(cfg):
        npc_speed = _value(case, "scenario.npc_initial_speed_kmh")
        assert 15.0 <= npc_speed <= 45.0


def test_drawn_cases_take_route_matches_in_turn(monkeypatch) -> None:
    from autoware_carla_scenario.sweeper import expand as expand_module

    monkeypatch.setattr(
        expand_module,
        "expand_route",
        lambda route, lanelet_map, arguments=(): [["route=a"], ["route=b"]],
    )
    sweep = {"route": {"pattern": "x"}, "odd_sample": {"count": 3}}

    cases = expand_sweep(sweep, None, ["x=1"], odd="default")

    assert [c[0] for c in cases] == ["route=a", "route=b", "route=a"]
    assert all(c[-1] == "x=1" for c in cases)


# ---------------------------------------------------------------------------
# A batch: the cases earlier scenarios drew
# ---------------------------------------------------------------------------


def test_coverage_aims_at_the_buckets_earlier_draws_left_uncovered() -> None:
    earlier = OddSampler(_weather_odd(), seed=4, strategy="coverage").sample(2)

    later = OddSampler(
        _weather_odd(), seed=4, strategy="coverage", drawn=earlier
    ).sample(2)

    # Four cases, four rain levels: the later ones fill what the earlier left.
    for name in ("environment.rain", "environment.illumination"):
        assert len({c.buckets[name] for c in [*earlier, *later]}) == 4


def test_a_situation_an_earlier_case_was_aimed_at_is_not_aimed_at_again() -> None:
    (earlier,) = OddSampler(_storm_odd(), seed=0, strategy="coverage").sample(1)

    sampler = OddSampler(_storm_odd(), seed=0, strategy="coverage", drawn=[earlier])

    assert earlier.situation == "storm_at_night"
    assert sampler.situation_holes == []
    assert sampler.sample(1)[0].situation is None


def test_uniform_does_not_draw_the_earlier_cases_again() -> None:
    earlier = OddSampler(default_odd(), seed=5).sample(3)

    later = OddSampler(default_odd(), seed=5, drawn=earlier).sample(3)
    again = OddSampler(default_odd(), seed=5, drawn=earlier).sample(3)

    assert not {tuple(c.overrides) for c in later} & {
        tuple(c.overrides) for c in earlier
    }
    assert [c.overrides for c in later] == [c.overrides for c in again]


def test_nothing_drawn_before_is_a_sampler_of_its_own() -> None:
    alone = OddSampler(default_odd(), seed=5).sample(3)
    first = OddSampler(default_odd(), seed=5, drawn=[]).sample(3)

    assert [c.overrides for c in first] == [c.overrides for c in alone]


def test_draws_are_kept_per_odd() -> None:
    drawn = OddDraws()
    cases = OddSampler(_weather_odd(), seed=1).sample(2)

    drawn.add(_weather_odd(), cases)

    # One ODD built twice is one ODD; another starts afresh.
    assert drawn.of(_weather_odd()) == cases
    assert drawn.of(_storm_odd()) == []
    drawn.add(_weather_odd(), cases[:1])
    assert drawn.of(_weather_odd()) == [*cases, cases[0]]


def test_expanding_configs_in_turn_goes_on_from_the_earlier_draws() -> None:
    cfg = OmegaConf.create(
        {
            "odd": "default",
            "sweep": {"odd_sample": {"count": 3, "strategy": "coverage"}},
        }
    )
    drawn = OddDraws()

    first = expand_config(cfg, drawn=drawn)
    second = expand_config(cfg, drawn=drawn)

    assert len(drawn.of(default_odd())) == 6
    assert not {tuple(c) for c in first} & {tuple(c) for c in second}
    # Without a record, every expansion draws alike, as before.
    assert expand_config(cfg) == expand_config(cfg) == first


# ---------------------------------------------------------------------------
# Recording the cases and their outcomes; the flaky search
# ---------------------------------------------------------------------------


def _decoded(case: list[str]):
    from autoware_carla_scenario.odd.sampler import ODD_CASE_KEY, decode_case

    (encoded,) = [
        o.split("=", 1)[1] for o in case if o.startswith(f"++{ODD_CASE_KEY}=")
    ]
    return decode_case(encoded)


def test_a_drawn_case_travels_to_its_run_whole() -> None:
    cfg = OmegaConf.create(
        {"odd": "default", "sweep": {"odd_sample": {"count": 2, "seed": 1}}}
    )
    for case in expand_config(cfg):
        sample, space = _decoded(case)
        # What the run is given is the case's own settings.
        assert [o for o in case if not o.startswith("++")] == sample.overrides
        assert sample.flaky is None
        assert set(space.names) == set(sample.values)


def test_the_flaky_search_learns_from_the_runs_recorded_before(tmp_path: Path) -> None:
    from openodd import CoverageCollector

    sweep: dict[str, Any] = {
        "odd_sample": {"count": 20, "seed": 3, "strategy": "coverage"}
    }
    cfg = OmegaConf.create({"odd": "default", "sweep": sweep})
    # A first sweep, run: heavy fog fails, as the runner would record it.
    for i, case in enumerate(expand_config(cfg)):
        sample, space = _decoded(case)
        collector = CoverageCollector([])
        collector.record_case(sample, space)
        collector.record_outcome(sample.values["environment.fog"] != "heavy")
        (tmp_path / f"run{i}").mkdir()
        collector.write(tmp_path / f"run{i}" / "S_coverage.json", "S")

    sweep["odd_sample"]["coverage_from"] = [str(tmp_path)]
    sweep["odd_sample"]["flaky"] = {"explore_fraction": 1.0, "coverage_depth": 0}
    second = [
        _decoded(c)[0]
        for c in expand_config(OmegaConf.create({"odd": "default", "sweep": sweep}))
    ]
    assert all(s.flaky is not None for s in second)
    report = load_and_merge([tmp_path])
    assert len(report.cases) == 20 and all(c.passed is not None for c in report.cases)
    assert "## Flaky cases" in report.to_markdown()


def test_an_unknown_flaky_field_is_refused() -> None:
    from autoware_carla_scenario.odd.sampler import search_from_config

    with pytest.raises(ValueError, match="flaky: unknown fields"):
        search_from_config({"flaky": {"bogus": 1}}, odd=_weather_odd())
    with pytest.raises(ValueError, match="search_from_config"):
        sampler_from_config({"flaky": True}, odd=_weather_odd())


def test_a_run_reads_its_case_back_from_the_override() -> None:
    from autoware_carla_scenario.examples.run import _compose_config, _odd_case

    cfg = OmegaConf.create(
        {"odd": "default", "sweep": {"odd_sample": {"count": 1, "seed": 1}}}
    )
    case = expand_config(cfg)[0]
    drawn, space = _decoded(case)
    run_case = _odd_case(_compose_config("cut_in/left", case))
    assert run_case is not None
    assert run_case[0] == drawn and run_case[1] == space
    assert _odd_case(_compose_config("cut_in/left", [])) is None


# ---------------------------------------------------------------------------
# The road a case is on: the attributes the map decides
# ---------------------------------------------------------------------------


def _road_odd() -> OddDefinition:
    return OddDefinition(
        "road",
        [
            OddAttribute("scenery.junction", lambda world: None, values=[True, False]),
            OddAttribute(
                "scenery.speed_limit",
                lambda world: None,
                unit="km/h",
                buckets=[0, 30, 60, 90],
            ),
            # What a run sets, so there is something to draw without a map.
            OddAttribute("environment.rain", rain, values=INTENSITY_LEVELS),
        ],
    )


#: The value a lanelet reads, by its bucket, in the road tests.
_VALUES: dict[str, Any] = {
    "true": True,
    "false": False,
    "[0, 30)": 20.0,
    "[30, 60)": 50.0,
}


def _on_road(monkeypatch, buckets: list[dict[str, str]]) -> None:
    from autoware_carla_scenario.sweeper import expand as expand_module

    monkeypatch.setattr(
        expand_module,
        "_expand_lanelets",
        lambda sweep, lanelet_map: [
            [f"ego.spawn_lanelet_id={i + 1}"] for i in range(len(buckets))
        ],
    )
    readings = [
        {name: (label, _VALUES[label]) for name, label in b.items()} for b in buckets
    ]
    monkeypatch.setattr(
        expand_module,
        "_case_readings",
        lambda cases, definition, lanelet_map: readings,
    )


def _recorded(case: list[str]) -> Any:
    from autoware_carla_scenario.odd.sampler import ODD_CASE_KEY, decode_case

    encoded = next(o for o in case if o.startswith(f"++{ODD_CASE_KEY}="))
    sample, _space = decode_case(encoded.split("=", 1)[1])
    return sample


def _drawn(case: list[str]) -> dict[str, str]:
    return _recorded(case).buckets


def test_each_case_goes_on_a_lanelet_in_the_buckets_drawn(monkeypatch) -> None:
    buckets = [
        {"scenery.junction": "true", "scenery.speed_limit": "[30, 60)"},
        {"scenery.junction": "false", "scenery.speed_limit": "[30, 60)"},
        {"scenery.junction": "false", "scenery.speed_limit": "[0, 30)"},
    ]
    _on_road(monkeypatch, buckets)
    sweep = {
        "constraints": {"ego.spawn_lanelet_id": [{}]},
        "odd_sample": {"count": 12, "seed": 3},
    }

    cases = expand_sweep(sweep, object(), (), odd=_road_odd())

    by_lanelet = {f"ego.spawn_lanelet_id={i + 1}": b for i, b in enumerate(buckets)}
    for case in cases:
        drawn = _drawn(case)
        on = by_lanelet[case[0]]
        assert drawn["scenery.junction"] == on["scenery.junction"]
        assert drawn["scenery.speed_limit"] == on["scenery.speed_limit"]
        # The knobs' own setting is no override of the run.
        assert not any(o.startswith("_odd_map.") for o in case)
    # Only the buckets some lanelet is in are drawn, and every one of them.
    assert {_drawn(c)["scenery.speed_limit"] for c in cases} == {"[0, 30)", "[30, 60)"}
    assert {_drawn(c)["scenery.junction"] for c in cases} == {"true", "false"}


def test_without_a_map_the_cases_are_taken_in_turn(monkeypatch) -> None:
    _on_road(monkeypatch, [{"scenery.junction": "true"}, {"scenery.junction": "false"}])
    sweep = {
        "constraints": {"ego.spawn_lanelet_id": [{}]},
        "odd_sample": {"count": 4, "seed": 3},
    }

    cases = expand_sweep(sweep, None, (), odd=_road_odd())

    assert [c[0] for c in cases] == [
        "ego.spawn_lanelet_id=1",
        "ego.spawn_lanelet_id=2",
    ] * 2


def test_a_knob_the_sweep_gives_is_not_replaced_by_the_map(monkeypatch) -> None:
    from autoware_carla_scenario.sweeper.expand import _map_knobs

    knobs = _map_knobs(
        _road_odd(),
        [{"scenery.junction": "true", "scenery.speed_limit": "[30, 60)"}],
        {"scenery.junction": {"key": "x"}},
    )

    assert set(knobs) == {"scenery.speed_limit"}
    assert knobs["scenery.speed_limit"]["values"] == {"[30, 60)": [30.0, 60.0]}


def test_a_case_records_its_lanelet_s_values_not_the_draw(monkeypatch) -> None:
    _on_road(
        monkeypatch,
        [
            {"scenery.junction": "true", "scenery.speed_limit": "[30, 60)"},
            {"scenery.junction": "false", "scenery.speed_limit": "[0, 30)"},
        ],
    )
    sweep = {
        "constraints": {"ego.spawn_lanelet_id": [{}]},
        "odd_sample": {"count": 8, "seed": 5},
    }

    for case in expand_sweep(sweep, object(), (), odd=_road_odd()):
        recorded = _recorded(case)
        expected = 50.0 if case[0] == "ego.spawn_lanelet_id=1" else 20.0
        assert recorded.values["scenery.speed_limit"] == pytest.approx(expected)
        assert not any(o.startswith("_odd_map.") for o in recorded.overrides)


def test_cases_in_the_same_buckets_are_run_in_turn(monkeypatch) -> None:
    # One junction lanelet and three that are not: every draw that is not in
    # a junction goes on one of the three, the least run first.
    _on_road(
        monkeypatch,
        [{"scenery.junction": "true"}] + [{"scenery.junction": "false"}] * 3,
    )
    sweep = {
        "constraints": {"ego.spawn_lanelet_id": [{}]},
        "odd_sample": {"count": 12, "seed": 2},
    }

    cases = expand_sweep(sweep, object(), (), odd=_road_odd())

    off = Counter(c[0] for c in cases if _drawn(c)["scenery.junction"] == "false")
    assert set(off) == {f"ego.spawn_lanelet_id={i}" for i in (2, 3, 4)}
    assert max(off.values()) - min(off.values()) <= 1


def test_too_few_fitting_draws_go_on_the_nearest_case(caplog) -> None:
    from autoware_carla_scenario.sweeper.expand import _on_some_case

    def sample(junction: str, limit: str) -> Any:
        return SimpleNamespace(
            buckets={"scenery.junction": junction, "scenery.speed_limit": limit}
        )

    fits, misfit = sample("true", "[30, 60)"), sample("true", "[0, 30)")
    sampler = SimpleNamespace(sample=lambda count: [misfit, fits] + [misfit] * 38)
    buckets = [{"scenery.junction": "true", "scenery.speed_limit": "[30, 60)"}]

    chosen = _on_some_case(
        sampler, 2, buckets, ["scenery.junction", "scenery.speed_limit"]
    )

    assert chosen == [fits, misfit]
    assert "Only 1 of 40" in caplog.text


def test_a_derived_attribute_gets_no_knob_of_its_own() -> None:
    from autoware_carla_scenario.sweeper.expand import _map_knobs

    odd = _road_odd()
    derived = SimpleNamespace(
        name="scenery.junction", derived=True, item=odd.attributes[0].item
    )
    definition = SimpleNamespace(attributes=[derived, *odd.attributes[1:]])

    knobs = _map_knobs(
        definition,
        [{"scenery.junction": "true", "scenery.speed_limit": "[30, 60)"}],
        {},
    )

    assert set(knobs) == {"scenery.speed_limit"}
