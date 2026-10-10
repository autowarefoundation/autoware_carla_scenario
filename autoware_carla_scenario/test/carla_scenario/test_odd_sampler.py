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
    OddKnob,
    OddModule,
    OddSampler,
    default_odd,
    illumination,
    knobs_from_mapping,
    rain,
)
from autoware_carla_scenario.odd.probes import illumination_level, intensity_level
from autoware_carla_scenario.odd.sampler import sampler_from_config
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
        precipitation = _value(case.overrides, "environment.precipitation")
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
        assert {o.split("=")[0] for o in case[:-1]} == {
            "environment.sun_altitude_angle",
            "environment.precipitation",
            "environment.fog_density",
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
    assert all(c[-1] == "x=1" and len(c) == 5 for c in cases)

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
        precipitation = _value(case.overrides, "environment.precipitation")
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
