"""The weather rules: the fuzzy rule bases the grip and the visibility follow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from autoware_carla_scenario.actions.environment import tire_friction_range
from autoware_carla_scenario.visibility import visibility_range
from autoware_carla_scenario.weather_rules import (
    DEFAULT_RULES,
    WeatherRulesError,
    active_weather_rules,
    load_weather_rules,
    use_weather_rules,
    weather_range,
)


@pytest.fixture(autouse=True)
def _default_rules() -> object:
    use_weather_rules(None)
    yield
    use_weather_rules(None)


def _interpolate(rows: tuple[tuple[float, float, float], ...], at: float):
    """The tables the default rules replaced, read as they were."""
    if at <= rows[0][0]:
        return rows[0][1:]
    for (x0, lo0, hi0), (x1, lo1, hi1) in zip(rows, rows[1:]):
        if at <= x1:
            t = (at - x0) / (x1 - x0)
            return lo0 + t * (lo1 - lo0), hi0 + t * (hi1 - hi0)
    return rows[-1][1:]


_GRIP = ((0.0, 0.90, 1.10), (50.0, 0.65, 0.85), (100.0, 0.45, 0.65))
_FOG = (
    (0.0, 250.0, 400.0),
    (30.0, 120.0, 250.0),
    (70.0, 40.0, 100.0),
    (100.0, 15.0, 40.0),
)
_RAIN = (
    (0.0, 250.0, 400.0),
    (30.0, 180.0, 320.0),
    (70.0, 100.0, 220.0),
    (100.0, 60.0, 150.0),
)
_LIGHT = (
    (-90.0, 50.0, 150.0),
    (-6.0, 50.0, 150.0),
    (0.0, 250.0, 400.0),
    (90.0, 250.0, 400.0),
)


def _write(path: Path, quantities: dict) -> Path:
    path.write_text(yaml.safe_dump({"version": 1, "quantities": quantities}))
    return path


def _grip_rules(rules: list[dict], **water: object) -> dict:
    """A ``tire_friction`` of one rule base on ``water``: dry and wet."""
    return {
        "rule_bases": {
            "water": {
                "inputs": {
                    "water": {
                        "universe": [0, 100],
                        "terms": {
                            "dry": {"trapezoid": [0, 0, 0, 100]},
                            "wet": {"trapezoid": [0, 100, 100, 100]},
                        },
                        **water,
                    }
                },
                "rules": rules,
            }
        }
    }


_DRY_WET = [
    {"if": "(water IS dry)", "then": {"low": 0.9, "high": 1.1}},
    {"if": "(water IS wet)", "then": {"low": 0.5, "high": 0.7}},
]


class TestDefaultRules:
    @pytest.mark.parametrize(
        "wetness", [-10.0, 0.0, 12.5, 37.0, 50.0, 71.0, 100.0, 120.0]
    )
    @pytest.mark.parametrize("deposits", [0.0, 25.0, 80.0])
    def test_the_grip_reads_as_the_table_it_replaced(self, wetness, deposits) -> None:
        expected = _interpolate(_GRIP, max(wetness, deposits))
        assert tire_friction_range(wetness, deposits) == pytest.approx(expected)

    @pytest.mark.parametrize("fog", [0.0, 15.0, 50.0, 85.0, 100.0])
    @pytest.mark.parametrize("rain", [0.0, 40.0, 100.0])
    @pytest.mark.parametrize("sun", [-90.0, -6.0, -3.0, 0.0, 45.0])
    def test_the_visibility_is_the_shortest_bound_as_before(
        self, fog, rain, sun
    ) -> None:
        bounds = [
            _interpolate(_FOG, fog),
            _interpolate(_RAIN, rain),
            _interpolate(_LIGHT, sun),
        ]
        expected = (min(b[0] for b in bounds), min(b[1] for b in bounds))
        assert visibility_range(fog, rain, sun) == pytest.approx(expected)

    def test_water_is_derived_from_the_wetness_and_the_standing_water(self) -> None:
        assert weather_range(
            "tire_friction", {"wetness": 10.0, "precipitation_deposits": 50.0}
        ) == (pytest.approx(weather_range("tire_friction", {"water": 50.0})))

    def test_the_rules_say_where_they_come_from(self) -> None:
        rules = active_weather_rules()
        assert rules.path is None
        assert rules.source("visibility") == DEFAULT_RULES

    def test_an_unknown_quantity_is_refused(self) -> None:
        with pytest.raises(WeatherRulesError, match="unknown quantity 'grip'"):
            weather_range("grip", {})

    def test_an_input_not_given_is_refused(self) -> None:
        with pytest.raises(WeatherRulesError, match="reads fog_density"):
            weather_range(
                "visibility", {"precipitation": 0.0, "sun_altitude_angle": 0.0}
            )


class TestOtherRules:
    def test_a_file_replaces_the_quantities_it_names(self, tmp_path) -> None:
        path = _write(tmp_path / "rules.yaml", {"tire_friction": _grip_rules(_DRY_WET)})
        use_weather_rules(path)

        assert tire_friction_range(100.0) == pytest.approx((0.5, 0.7))
        assert tire_friction_range(50.0) == pytest.approx((0.7, 0.9))
        assert active_weather_rules().source("tire_friction") == path.resolve()
        # The visibility keeps the default rules.
        assert visibility_range(fog_density=100.0) == pytest.approx((15.0, 40.0))

        use_weather_rules(None)
        assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))

    def test_a_rule_may_read_more_than_one_input(self, tmp_path) -> None:
        quantity = _grip_rules(
            [
                {"if": "(water IS dry)", "then": {"low": 0.9, "high": 1.1}},
                {
                    "if": "(water IS wet) AND (NOT (cloudiness IS overcast))",
                    "then": {"low": 0.6, "high": 0.8},
                },
                {
                    "if": "(water IS wet) AND (cloudiness IS overcast)",
                    "then": {"low": 0.4, "high": 0.6},
                },
            ]
        )
        quantity["rule_bases"]["water"]["inputs"]["cloudiness"] = {
            "universe": [0, 100],
            "default": 0,
            "terms": {
                "clear": {"trapezoid": [0, 0, 0, 100]},
                "overcast": {"trapezoid": [0, 100, 100, 100]},
            },
        }
        use_weather_rules(_write(tmp_path / "rules.yaml", {"tire_friction": quantity}))

        assert weather_range("tire_friction", {"water": 100.0, "cloudiness": 0.0}) == (
            pytest.approx((0.6, 0.8))
        )
        assert weather_range(
            "tire_friction", {"water": 100.0, "cloudiness": 100.0}
        ) == (pytest.approx((0.4, 0.6)))
        # tire_friction_range() names no cloudiness: its default holds.
        assert tire_friction_range(100.0) == pytest.approx((0.6, 0.8))

    def test_combine_max_takes_the_longer_bound(self, tmp_path) -> None:
        def base(low: float) -> dict:
            return {
                "inputs": {
                    "fog_density": {
                        "universe": [0, 100],
                        "terms": {"any": {"trapezoid": [0, 0, 100, 100]}},
                    }
                },
                "rules": [
                    {"if": "(fog_density IS any)", "then": {"low": low, "high": 400}}
                ],
            }

        quantity = {
            "combine": "max",
            "rule_bases": {"a": base(100.0), "b": base(200.0)},
        }
        use_weather_rules(_write(tmp_path / "rules.yaml", {"visibility": quantity}))
        assert visibility_range() == pytest.approx((200.0, 400.0))

    def test_a_file_is_read_again_once_it_changes(self, tmp_path) -> None:
        path = _write(tmp_path / "rules.yaml", {"tire_friction": _grip_rules(_DRY_WET)})
        first = load_weather_rules(path)
        assert load_weather_rules(path) is first

        rules = [dict(r) for r in _DRY_WET]
        rules[1] = {"if": "(water IS wet)", "then": {"low": 0.3, "high": 0.5}}
        _write(path, {"tire_friction": _grip_rules(rules)})
        import os

        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert load_weather_rules(path) is not first
        assert load_weather_rules(path).range("tire_friction", {"water": 100.0}) == (
            pytest.approx((0.3, 0.5))
        )


class TestRefusedRules:
    @pytest.mark.parametrize(
        ("quantities", "match"),
        [
            ({"grip": {}}, "unknown quantities"),
            (
                {
                    "tire_friction": _grip_rules(
                        [{"if": "(water IS soaked)", "then": {"low": 0, "high": 1}}]
                    )
                },
                "water has no term 'soaked'",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        [
                            {
                                "if": "(fog_density IS dense)",
                                "then": {"low": 0, "high": 1},
                            }
                        ]
                    )
                },
                "'fog_density' is no input of this rule base",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        [{"if": "water IS dry", "then": {"low": 0, "high": 1}}]
                    )
                },
                r"expected clauses such as \(water IS wet\)",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        [{"if": "(water IS dry)", "then": {"low": 1, "high": 0}}]
                    )
                },
                "low must not be above high",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        [{"if": "(water IS dry)", "then": {"low": 1}}]
                    )
                },
                "expected low and high",
            ),
            # Only dry has a rule: no rule fires on a soaked road.
            (
                {
                    "tire_friction": _grip_rules(
                        [{"if": "(water IS dry)", "then": {"low": 0.9, "high": 1.1}}],
                        terms={
                            "dry": {"trapezoid": [0, 0, 0, 60]},
                            "wet": {"trapezoid": [40, 100, 100, 100]},
                        },
                    )
                },
                "no rule fires",
            ),
            # Nothing covers 40-60.
            (
                {
                    "tire_friction": _grip_rules(
                        _DRY_WET,
                        terms={
                            "dry": {"triangle": [0, 0, 40]},
                            "wet": {"triangle": [60, 100, 100]},
                        },
                    )
                },
                "no term covers water",
            ),
            (
                {"tire_friction": _grip_rules(_DRY_WET, universe=[100, 0])},
                "low must be below high",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        _DRY_WET, terms={"dry": {"triangle": [50, 0, 100]}}
                    )
                },
                "the points must not decrease",
            ),
            (
                {
                    "tire_friction": _grip_rules(
                        _DRY_WET, terms={"dry": {"bell": [0, 1]}}
                    )
                },
                "unknown term shape 'bell'",
            ),
            (
                {
                    "tire_friction": {
                        "combine": "mean",
                        **_grip_rules(_DRY_WET),
                    }
                },
                "expected min or max",
            ),
        ],
    )
    def test_a_malformed_file_is_refused(self, tmp_path, quantities, match) -> None:
        path = _write(tmp_path / "rules.yaml", quantities)
        with pytest.raises(WeatherRulesError, match=match):
            load_weather_rules(path)

    def test_an_unknown_input_is_refused(self, tmp_path) -> None:
        quantity = _grip_rules(_DRY_WET)
        inputs = quantity["rule_bases"]["water"]["inputs"]
        inputs["temperature"] = inputs.pop("water")
        with pytest.raises(WeatherRulesError, match="unknown input 'temperature'"):
            load_weather_rules(
                _write(tmp_path / "rules.yaml", {"tire_friction": quantity})
            )

    def test_another_version_is_refused(self, tmp_path) -> None:
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump({"version": 2, "quantities": {}}))
        with pytest.raises(WeatherRulesError, match="version must be 1"):
            load_weather_rules(path)

    def test_a_missing_file_is_refused(self, tmp_path) -> None:
        with pytest.raises(WeatherRulesError, match="cannot read weather rules"):
            load_weather_rules(tmp_path / "nowhere.yaml")


def test_the_run_config_names_the_rules_and_a_run_without_resets_them(
    tmp_path,
) -> None:
    from autoware_carla_scenario.examples.run import add_environment

    path = _write(tmp_path / "rules.yaml", {"tire_friction": _grip_rules(_DRY_WET)})
    registered: list = []
    scenario = SimpleNamespace(register_init=registered.append)

    add_environment(
        OmegaConf.create({"environment": {"rules": str(path), "wetness": 100}}),
        scenario,  # type: ignore[arg-type]
    )
    (action,) = registered
    # The rules are no weather setting.
    assert action.settings == {"wetness": 100.0}
    assert tire_friction_range(100.0) == pytest.approx((0.5, 0.7))

    registered.clear()
    add_environment(OmegaConf.create({"environment": {"rules": None}}), scenario)  # type: ignore[arg-type]
    assert registered == []
    assert active_weather_rules().path is None
    assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))
