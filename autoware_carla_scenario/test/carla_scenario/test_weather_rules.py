"""The weather rules: the CARLA backend's Bayesian network from the conditions
a run is given to CARLA's weather, the grip and the visibility, worked out
before a run; a run's rules over them, and rules that differ by vehicle."""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.actions.environment import (
    EnvironmentAction,
    apply_weather_grip,
    follow_tire_friction,
    forget_tire_friction,
    tire_friction,
    tire_friction_range,
    world_tire_friction,
)
from autoware_carla_scenario.odd import probes
from autoware_carla_scenario.odd.backend import CARLA_BACKEND
from autoware_carla_scenario.weather_rules import (
    DEFAULT_RULES,
    RulesError,
    check_weather_rules,
    resolve_environment,
    road_water,
    use_weather_rules,
    vehicle_at,
    vehicle_range,
    vehicle_rules_in_use,
    weather_at,
    weather_range,
    weather_rule_files,
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
# The visibility's parts, by the cause each reads.
_FOG = (
    (15.0, 12.0, 18.0),
    (50.0, 40.0, 60.0),
    (150.0, 120.0, 180.0),
    (400.0, 250.0, 400.0),
    (1000.0, 250.0, 400.0),
)
_RAIN = (
    (0.0, 250.0, 400.0),
    (2.5, 180.0, 320.0),
    (7.6, 100.0, 220.0),
    (25.0, 60.0, 150.0),
)
_LIGHT = {
    "day": (250.0, 400.0),
    "dusk": (150.0, 275.0),
    "night": (50.0, 150.0),
    "dawn": (150.0, 275.0),
}


def _grip_file(path: Path, flooded: tuple[float, float], extends: str = "") -> Path:
    """A rule file naming only the grip's distributions (by the wetness and
    by the standing water): their points are the default's."""
    parts = "".join(
        f"      {part}:\n        uniform:\n"
        f"          low: [0.90, 0.65, {flooded[0]}]\n"
        f"          high: [1.10, 0.85, {flooded[1]}]\n"
        for part in ("wetness", "standing_water")
    )
    path.write_text(
        "version: 2\n"
        + (f"extends: {extends}\n" if extends else "")
        + "quantities:\n  tire_friction:\n    parts:\n"
        + parts
    )
    return path


def _visibility_range(fog: float = 10000.0, rain: float = 0.0, time: str = "day"):
    return weather_range(
        "visibility",
        {"fog_visibility_range": fog, "rainfall_rate": rain, "time_of_day": time},
    )


class TestDefaultRules:
    @pytest.mark.parametrize(
        "wetness", [-10.0, 0.0, 12.5, 37.0, 50.0, 71.0, 100.0, 120.0]
    )
    @pytest.mark.parametrize("deposits", [0.0, 25.0, 80.0])
    def test_the_grip_reads_as_the_table_it_replaced(self, wetness, deposits) -> None:
        expected = _interpolate(_GRIP, max(wetness, deposits))
        assert tire_friction_range(wetness, deposits) == pytest.approx(expected)

    @pytest.mark.parametrize("fog", [10.0, 15.0, 30.0, 300.0, 1000.0, 10000.0])
    @pytest.mark.parametrize("rain", [0.0, 1.0, 7.6, 40.0])
    @pytest.mark.parametrize("time", ["day", "dusk", "night", "dawn"])
    def test_the_visibility_is_the_shortest_bound_of_the_causes(
        self, fog, rain, time
    ) -> None:
        bounds = [_interpolate(_FOG, fog), _interpolate(_RAIN, rain), _LIGHT[time]]
        expected = (min(b[0] for b in bounds), min(b[1] for b in bounds))
        assert _visibility_range(fog, rain, time) == pytest.approx(expected)

    def test_they_are_the_carla_backends_over_the_taxonomys_causes(self) -> None:
        assert CARLA_BACKEND.rules is not None
        assert CARLA_BACKEND.rules.files[0][0] == str(DEFAULT_RULES)
        assert CARLA_BACKEND.rules.version == 3
        assert set(CARLA_BACKEND.rules.input_concepts) == {
            "rainfall_rate",
            "fog_visibility_range",
            "time_of_day",
            "wind_speed",
            "cloudiness",
        }
        # The grip follows the wetter of the wetness and the standing water.
        assert weather_range(
            "tire_friction",
            {"carla_wetness": 10.0, "carla_precipitation_deposits": 50.0},
        ) == pytest.approx(weather_range("tire_friction", {"carla_wetness": 50.0}))

    @pytest.mark.parametrize(
        ("rain", "low", "high"),
        [(0.0, 0.0, 0.0), (2.5, 25.0, 35.0), (7.6, 60.0, 75.0), (25.0, 95.0, 100.0)],
    )
    def test_the_rainfall_rate_gives_carlas_precipitation(
        self, rain, low, high
    ) -> None:
        assert weather_range(
            "carla_precipitation", {"rainfall_rate": rain}
        ) == pytest.approx((low, high))

    @pytest.mark.parametrize(
        ("time", "altitude", "azimuth"),
        [
            ("day", (0, 90), (0, 360)),
            ("dusk", (-6, -0.01), (180, 359.99)),
            ("night", (-90, -6.01), (0, 360)),
            ("dawn", (-6, -0.01), (0, 179.99)),
        ],
    )
    def test_the_time_of_day_gives_a_sun_that_reads_back_as_it(
        self, time, altitude, azimuth
    ) -> None:
        inputs = {"time_of_day": time}
        assert weather_range("carla_sun_altitude", inputs) == pytest.approx(altitude)
        assert weather_range("carla_sun_azimuth", inputs) == pytest.approx(azimuth)
        for q in (0.0, 0.3, 0.7, 1.0):
            assert (
                probes.time_of_day_level(
                    weather_at("carla_sun_altitude", inputs, q),
                    weather_at("carla_sun_azimuth", inputs, q),
                )
                == time
            )

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_the_grip_at_a_quantile_is_its_uniform_distributions(self, q) -> None:
        low, high = tire_friction_range(70.0)
        assert tire_friction(70.0, quantile=q) == pytest.approx(low + q * (high - low))

    def test_the_visibilitys_parts_move_together_at_one_quantile(self) -> None:
        def at(fog: float, rain: float, time: str, q: float) -> float:
            return weather_at(
                "visibility",
                {
                    "fog_visibility_range": fog,
                    "rainfall_rate": rain,
                    "time_of_day": time,
                },
                q,
            )

        # Fog of 100 m: 80-120 m; light rain and the day are longer.  At the
        # quantile 0.5 each part is at its median, and the shortest holds.
        assert at(100.0, 1.0, "day", 0.5) == pytest.approx(100.0)
        # Night and light rain: the light's 50-150 m and the rain's.
        assert at(10000.0, 2.5, "night", 0.25) == pytest.approx(75.0)
        assert at(10000.0, 2.5, "night", 1.0) == pytest.approx(150.0)
        # The quantile is clamped.
        assert at(10000.0, 2.5, "night", 7.0) == pytest.approx(150.0)

    def test_an_old_fuzzy_rule_file_is_refused(self, tmp_path) -> None:
        path = tmp_path / "v1.yaml"
        path.write_text("version: 1\nquantities: {}\n")
        with pytest.raises(RulesError, match="version 1"):
            use_weather_rules(str(path))

    def test_the_manifest_follows_a_knob_to_the_rules(self) -> None:
        inputs = CARLA_BACKEND.manifest()["rule_inputs"]
        assert inputs["rainfall_rate"]["knob"] == "environment.rainfall_rate_mm_h"
        assert inputs["fog_visibility_range"]["knob"] == (
            "environment.fog_visibility_range_m"
        )
        assert inputs["time_of_day"]["knob"] == "environment.time_of_day"

    def test_an_unknown_quantity_is_refused(self) -> None:
        with pytest.raises(RulesError, match="unknown quantity 'grip'"):
            weather_range("grip", {})


class TestARunsRules:
    def test_a_file_naming_only_its_change_is_read_over_the_default(
        self, tmp_path
    ) -> None:
        path = _grip_file(tmp_path / "measured.yaml", (0.30, 0.50))
        use_weather_rules(str(path))
        assert tire_friction_range(100.0) == pytest.approx((0.30, 0.50))
        # The visibility keeps the default rules.
        assert _visibility_range(fog=15.0) == pytest.approx((12.0, 18.0))
        assert [Path(p).name for p, _ in weather_rule_files()] == [
            "weather_rules.yaml",
            "measured.yaml",
        ]
        use_weather_rules(None)
        assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))

    def test_a_url_is_fetched(self, monkeypatch, tmp_path) -> None:
        import openodd.rules

        text = _grip_file(tmp_path / "x.yaml", (0.20, 0.40)).read_text()
        asked: list[str] = []

        def fetch(url: str) -> str:
            asked.append(url)
            return text

        monkeypatch.setattr(openodd.rules, "fetch_url", fetch)
        use_weather_rules("https://example.com/rules/measured.yaml")
        assert asked == ["https://example.com/rules/measured.yaml"]
        assert tire_friction_range(100.0) == pytest.approx((0.20, 0.40))

    def test_a_file_reading_an_input_carla_has_not_is_refused(self, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            "version: 2\nquantities:\n  tire_friction:\n    parts:\n      t:\n"
            "        at: {temperature: [-20, 40]}\n"
            "        uniform: {low: 1, high: 1}\n"
        )
        with pytest.raises(RulesError, match="stands for no taxonomy concept"):
            use_weather_rules(str(path))


class TestRulesByVehicle:
    def test_each_vehicle_reads_the_first_entry_it_matches(self, tmp_path) -> None:
        use_weather_rules(
            {
                "default": str(_grip_file(tmp_path / "default.yaml", (0.40, 0.60))),
                "vehicles": {
                    "ego": str(_grip_file(tmp_path / "ego.yaml", (0.30, 0.50))),
                    "vehicle.mini.*": str(
                        _grip_file(tmp_path / "mini.yaml", (0.20, 0.40))
                    ),
                },
            }
        )
        assert vehicle_rules_in_use()
        flooded = {"carla_wetness": 100.0}
        assert vehicle_range(
            "tire_friction", flooded, "vehicle.mini.cooper", "Ego"
        ) == pytest.approx((0.30, 0.50))
        assert vehicle_range(
            "tire_friction", flooded, "vehicle.mini.cooper", "npc1"
        ) == pytest.approx((0.20, 0.40))
        assert vehicle_range(
            "tire_friction", flooded, "vehicle.tesla.model3", ""
        ) == pytest.approx((0.40, 0.60))
        assert vehicle_at(
            "tire_friction", flooded, 0.5, "vehicle.mini.cooper", "npc1"
        ) == pytest.approx(0.30)
        assert vehicle_at(
            "tire_friction", flooded, 0.5, "vehicle.tesla.model3", ""
        ) == pytest.approx(0.50)
        # The visibility is the default rules'.
        assert _visibility_range(fog=15.0) == pytest.approx((12.0, 18.0))

    def test_a_vehicles_file_extends_and_is_read_over_the_runs_default(
        self, tmp_path
    ) -> None:
        _grip_file(tmp_path / "shared.yaml", (0.35, 0.55))
        mini = tmp_path / "vehicles" / "mini.yaml"
        mini.parent.mkdir()
        mini.write_text("version: 2\nextends: ../shared.yaml\nquantities: {}\n")
        use_weather_rules({"vehicles": {"vehicle.mini.*": str(mini)}})
        assert vehicle_range(
            "tire_friction", {"carla_wetness": 100.0}, "vehicle.mini.cooper", ""
        ) == pytest.approx((0.35, 0.55))

    def test_malformed_settings_are_refused(self, tmp_path) -> None:
        with pytest.raises(RulesError, match=r"unknown fields \['vehicle'\]"):
            use_weather_rules({"vehicle": {}})
        with pytest.raises(RulesError, match="expected pattern -> file"):
            use_weather_rules({"vehicles": ["a.yaml"]})
        with pytest.raises(RulesError, match="expected a file or"):
            use_weather_rules(42)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "default",
        [
            5,
            True,
            ["a.yaml"],
            {"url": "https://example.com/rules.yaml", "sha256": "0" * 64},
        ],
    )
    def test_a_default_not_a_file_is_refused_and_changes_nothing(
        self, tmp_path, default
    ) -> None:
        # #1469: it was read as str() of itself, a path that does not exist.
        use_weather_rules(
            {
                "default": str(_grip_file(tmp_path / "measured.yaml", (0.3, 0.5))),
                "vehicles": {"ego": str(DEFAULT_RULES)},
            }
        )
        before = CARLA_BACKEND.active_rules
        with pytest.raises(
            RulesError, match=r"^environment\.rules\.default: expected a file"
        ):
            use_weather_rules({"default": default})
        assert CARLA_BACKEND.active_rules is before
        assert vehicle_rules_in_use()

    @pytest.mark.parametrize("vehicles", ["", [], 0, False, "a.yaml"])
    def test_vehicles_not_a_mapping_is_refused_even_when_empty(self, vehicles) -> None:
        with pytest.raises(RulesError, match="expected pattern -> file"):
            use_weather_rules({"vehicles": vehicles})

    @pytest.mark.parametrize("location", [5, None, {"url": "https://x"}])
    def test_a_vehicles_entry_not_a_file_is_refused_before_any_is_read(
        self, location
    ) -> None:
        with pytest.raises(
            RulesError, match=r"environment\.rules\.vehicles\.ego: expected a file"
        ):
            use_weather_rules(
                {"vehicles": {"vehicle.*": "missing.yaml", "ego": location}}
            )

    def test_the_form_is_checked_without_reading_a_file(self) -> None:
        assert check_weather_rules(None) == (None, [])
        assert check_weather_rules("missing.yaml") == ("missing.yaml", [])
        assert check_weather_rules(Path("m.yaml")) == (Path("m.yaml"), [])
        assert check_weather_rules({"default": None, "vehicles": None}) == (None, [])
        assert check_weather_rules(
            {"default": "a.yaml", "vehicles": {"ego": "b.yaml", 7: Path("c.yaml")}}
        ) == ("a.yaml", [("ego", "b.yaml"), ("7", Path("c.yaml"))])

    def test_without_vehicles_every_vehicle_reads_the_default(self) -> None:
        assert not vehicle_rules_in_use()
        assert vehicle_range(
            "tire_friction", {"carla_wetness": 100.0}, "vehicle.mini.cooper", "Ego"
        ) == pytest.approx((0.45, 0.65))


class _Vehicle:
    """A vehicle of a blueprint and a role, whose physics control is a copy
    each read, as CARLA's is."""

    def __init__(self, actor_id: int, type_id: str, role: str = "") -> None:
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role} if role else {}
        self.frictions = [3.5] * 4

    def get_physics_control(self) -> SimpleNamespace:
        return SimpleNamespace(
            wheels=[
                SimpleNamespace(friction_force_multiplier=f) for f in self.frictions
            ]
        )

    def apply_physics_control(self, physics: SimpleNamespace) -> None:
        self.frictions = [w.friction_force_multiplier for w in physics.wheels]


_world_ids = itertools.count(5000)


def _world(*vehicles: _Vehicle) -> MagicMock:
    world = MagicMock()
    world.id = next(_world_ids)
    weather = SimpleNamespace(
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wetness=0.0,
        wind_intensity=0.0,
        fog_density=0.0,
        fog_distance=0.0,
        sun_altitude_angle=45.0,
        sun_azimuth_angle=0.0,
    )
    world.get_weather.return_value = weather
    fleet = list(vehicles)
    world.fleet = fleet

    def actors(ids: "list[int] | None" = None) -> MagicMock:
        chosen = [v for v in fleet if ids is None or v.id in ids]
        listed = MagicMock()
        listed.filter.side_effect = lambda pattern: list(chosen)
        return listed

    world.get_actors.side_effect = actors
    world.get_snapshot.side_effect = lambda: [SimpleNamespace(id=v.id) for v in fleet]
    return world


class TestGripByVehicle:
    def test_each_vehicle_gets_its_own_grip_at_the_cases_quantile(
        self, tmp_path
    ) -> None:
        use_weather_rules(
            {
                "vehicles": {
                    "ego": str(_grip_file(tmp_path / "ego.yaml", (0.30, 0.50))),
                    "vehicle.mini.*": str(
                        _grip_file(tmp_path / "mini.yaml", (0.20, 0.40))
                    ),
                }
            }
        )
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        mini = _Vehicle(2, "vehicle.mini.cooper", "npc1")
        other = _Vehicle(3, "vehicle.tesla.model3", "npc2")
        world = _world(ego, mini, other)
        EnvironmentAction(wetness=100.0, tire_friction_quantile=0.0).execute(world)

        assert ego.frictions == pytest.approx([3.5 * 0.30] * 4)
        assert mini.frictions == pytest.approx([3.5 * 0.20] * 4)
        assert other.frictions == pytest.approx([3.5 * 0.45] * 4)
        # The ODD's probe reads the ego's.
        assert world_tire_friction(world.id) == pytest.approx(0.30)

        # A vehicle spawned later gets its own too.
        late = _Vehicle(4, "vehicle.mini.cooper_s")
        world.fleet.append(late)
        follow_tire_friction(world)
        assert late.frictions == pytest.approx([3.5 * 0.20] * 4)

    def test_a_fixed_grip_is_every_vehicles(self, tmp_path) -> None:
        use_weather_rules(
            {"vehicles": {"ego": str(_grip_file(tmp_path / "e.yaml", (0.3, 0.5)))}}
        )
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        EnvironmentAction(wetness=100.0, tire_friction=0.7).execute(world)
        assert ego.frictions == pytest.approx([3.5 * 0.7] * 4)
        assert world_tire_friction(world.id) == pytest.approx(0.7)

    def test_the_next_run_forgets_the_egos_own_grip(self, tmp_path) -> None:
        # Issue #899: a soft reset or a retry keeps the episode (the world).
        use_weather_rules(
            {"vehicles": {"ego": str(_grip_file(tmp_path / "e.yaml", (0.30, 0.50)))}}
        )
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        EnvironmentAction(wetness=100.0, tire_friction_quantile=0.0).execute(world)
        assert world_tire_friction(world.id) == pytest.approx(0.30)
        assert probes.tire_friction(world) == pytest.approx(0.30)

        # The next run, with neither rules nor a grip of its own.
        use_weather_rules(None)
        forget_tire_friction(world)
        assert world_tire_friction(world.id) is None
        assert probes.tire_friction(world) is None
        assert ego.frictions == pytest.approx([3.5] * 4)
        late = _Vehicle(2, "vehicle.mini.cooper", "npc1")
        world.fleet.append(late)
        follow_tire_friction(world)
        assert late.frictions == pytest.approx([3.5] * 4)

    def test_without_vehicle_rules_the_world_has_one_grip(self) -> None:
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        apply_weather_grip(world, {"carla_wetness": 100.0}, 1.0)
        assert ego.frictions == pytest.approx([3.5 * 0.65] * 4)
        assert world_tire_friction(world.id) == pytest.approx(0.65)


def _flat_grip_file(path: Path, low: float, high: float) -> Path:
    """A rule file whose grip is *low*-*high* on any road, dry or flooded."""
    parts = "".join(
        f"      {part}:\n        uniform:\n"
        f"          low: [{low}, {low}, {low}]\n"
        f"          high: [{high}, {high}, {high}]\n"
        for part in ("wetness", "standing_water")
    )
    path.write_text("version: 2\nquantities:\n  tire_friction:\n    parts:\n" + parts)
    return path


def _rules_v3(path: Path, part: str, at: str, low: str, high: str) -> Path:
    """A version 3 rule file adding the part *part* to the grip."""
    path.write_text(
        "version: 3\ntaxonomy: {bundled: default}\nquantities:\n"
        f"  tire_friction:\n    parts:\n      {part}:\n        at: {at}\n"
        f"        uniform:\n          low: {low}\n          high: {high}\n"
    )
    return path


class TestTheRunsGrip:
    """What the config's ``environment`` gives the vehicles: the grip of the
    run's rules whenever they differ from the default ones (#1937), read off
    the run's conditions and CARLA's fields (#1938)."""

    @pytest.fixture(autouse=True)
    def _fresh(self, tmp_path, monkeypatch) -> object:
        from autoware_carla_scenario.actions.environment import (
            forget_weather_conditions,
        )
        from autoware_carla_scenario.visibility import forget_visibility

        monkeypatch.chdir(tmp_path)
        forget_weather_conditions()
        forget_visibility()
        yield
        forget_weather_conditions()
        forget_visibility()

    def _run(self, environment: dict, world: MagicMock) -> list:
        """Build the config's action as a run does and fire it on *world*."""
        from autoware_carla_scenario.examples.run import add_environment

        registered: list = []
        scenario = SimpleNamespace(register_init=registered.append)
        add_environment(OmegaConf.create({"environment": environment}), scenario)  # type: ignore[arg-type]
        for action in registered:
            action.execute(world)
        return registered

    def test_vehicle_rules_with_only_a_time_of_day_give_the_ego_its_grip(
        self, tmp_path
    ) -> None:
        _flat_grip_file(tmp_path / "grip_low.yaml", 0.30, 0.40)
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        npc = _Vehicle(2, "vehicle.tesla.model3", "npc1")
        world = _world(ego, npc)
        self._run(
            {"rules": {"vehicles": {"ego": "grip_low.yaml"}}, "time_of_day": "night"},
            world,
        )
        # The median of the ego's rules; the others keep the dry road's.
        assert ego.frictions == pytest.approx([3.5 * 0.35] * 4)
        assert npc.frictions == pytest.approx([3.5] * 4)
        assert world_tire_friction(world.id) == pytest.approx(0.35)
        assert probes.tire_friction(world) == pytest.approx(0.35)

    def test_rules_alone_apply_their_grip(self, tmp_path) -> None:
        _flat_grip_file(tmp_path / "grip_low.yaml", 0.30, 0.40)
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        (action,) = self._run({"rules": "grip_low.yaml"}, world)
        world.set_weather.assert_not_called()
        assert ego.frictions == pytest.approx([3.5 * 0.35] * 4)
        assert probes.tire_friction(world) == pytest.approx(0.35)
        # A vehicle spawned later gets it too.
        from autoware_carla_scenario import BaseScenario

        class _Scenario(BaseScenario):
            def setup(self) -> None: ...

            def is_done(self) -> bool:
                return False

        scenario = _Scenario(MagicMock())
        scenario.register_init(action)
        assert follow_tire_friction in scenario._post_tick_callbacks

    def test_the_default_rules_without_road_water_keep_carlas_grip(self) -> None:
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        # A wet road the map left: the run sets no water, no quantile.
        world.get_weather.return_value.wetness = 50.0
        (action,) = self._run({"time_of_day": "night"}, world)
        assert not action.sets_tire_friction
        assert ego.frictions == pytest.approx([3.5] * 4)
        assert world_tire_friction(world.id) is None
        assert probes.tire_friction(world) is None

    def test_a_part_reading_the_rainfall_rate_reads_the_runs(self, tmp_path) -> None:
        _rules_v3(
            tmp_path / "rain.yaml",
            "rain_film",
            "{rainfall_rate: [0, 25]}",
            "[0.9, 0.3]",
            "[1.1, 0.4]",
        )
        use_weather_rules("rain.yaml")
        assert weather_range("tire_friction", {"rainfall_rate": 25}) == pytest.approx(
            (0.3, 0.4)
        )
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        self._run({"rules": "rain.yaml", "rainfall_rate_mm_h": 25.0}, world)
        assert world_tire_friction(world.id) == pytest.approx(0.35)
        assert ego.frictions == pytest.approx([3.5 * 0.35] * 4)

    def test_a_vehicles_part_reading_the_time_of_day_reads_carlas_sun(
        self, tmp_path
    ) -> None:
        _rules_v3(
            tmp_path / "dark.yaml",
            "dark",
            "{time_of_day: [day, dusk, night, dawn]}",
            "[0.9, 0.9, 0.5, 0.9]",
            "[1.1, 1.1, 0.6, 1.1]",
        )
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        self._run(
            {"rules": {"vehicles": {"ego": "dark.yaml"}}, "time_of_day": "night"},
            world,
        )
        assert ego.frictions == pytest.approx([3.5 * 0.55] * 4)
        assert probes.tire_friction(world) == pytest.approx(0.55)

    def test_a_sweep_case_and_a_run_get_the_same_grip(self, tmp_path) -> None:
        """A sweep case always names its grip's quantile; a run naming none
        gets the same grip at the median, road water or not."""
        _flat_grip_file(tmp_path / "grip_low.yaml", 0.30, 0.40)
        conditions = {
            "rules": "grip_low.yaml",
            "time_of_day": "dusk",
            "wind_speed_kmh": 30.0,
        }
        run = _world(_Vehicle(1, "vehicle.lincoln.mkz", "Ego"))
        self._run(conditions, run)
        case = _world(_Vehicle(1, "vehicle.lincoln.mkz", "Ego"))
        self._run({**conditions, "tire_friction_quantile": 0.5}, case)
        assert world_tire_friction(case.id) == pytest.approx(0.35)
        assert run.fleet[0].frictions == pytest.approx(case.fleet[0].frictions)


class TestALaterAction:
    """Under the run's own rules every EnvironmentAction sets the grip; one
    that names no grip, quantile or water (a scenario's own that sets the
    sun) keeps the run's quantile, and a grip named outright."""

    @pytest.fixture(autouse=True)
    def _rules(self, tmp_path) -> None:
        use_weather_rules(str(_flat_grip_file(tmp_path / "flat.yaml", 0.30, 0.40)))

    @pytest.mark.parametrize(
        ("first", "grip"),
        [
            ({"tire_friction_quantile": 0.0}, 0.30),
            ({"tire_friction": 0.5}, 0.50),
            ({"wetness": 80.0, "tire_friction_quantile": 1.0}, 0.40),
        ],
    )
    def test_a_sun_only_action_keeps_the_runs_grip(self, first, grip) -> None:
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        EnvironmentAction(**first).execute(world)
        assert world_tire_friction(world.id) == pytest.approx(grip)
        later = EnvironmentAction(sun_altitude_angle=-10.0)
        assert later.sets_tire_friction
        later.execute(world)
        assert world_tire_friction(world.id) == pytest.approx(grip)
        assert ego.frictions == pytest.approx([3.5 * grip] * 4)

    def test_a_first_action_naming_no_quantile_gets_the_median(self) -> None:
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        EnvironmentAction(sun_altitude_angle=-10.0).execute(world)
        assert world_tire_friction(world.id) == pytest.approx(0.35)

    def test_rain_starting_later_reads_the_grip_at_the_runs_quantile(self) -> None:
        use_weather_rules(None)
        ego = _Vehicle(1, "vehicle.lincoln.mkz", "Ego")
        world = _world(ego)
        EnvironmentAction(wetness=0.0, tire_friction_quantile=0.0).execute(world)
        assert world_tire_friction(world.id) == pytest.approx(0.90)
        EnvironmentAction(wetness=100.0).execute(world)
        assert world_tire_friction(world.id) == pytest.approx(0.45)


def test_a_vehicles_rules_read_causes_and_labels(tmp_path) -> None:
    """vehicle_at reads a vehicle's file with the causes among the numbers
    and the labels apart, as the checked environment action gives them."""
    rain = _rules_v3(
        tmp_path / "rain.yaml",
        "rain_film",
        "{rainfall_rate: [0, 25]}",
        "[0.9, 0.3]",
        "[1.1, 0.4]",
    )
    dark = _rules_v3(
        tmp_path / "dark.yaml",
        "dark",
        "{time_of_day: [day, dusk, night, dawn]}",
        "[0.9, 0.9, 0.5, 0.9]",
        "[1.1, 1.1, 0.6, 1.1]",
    )
    use_weather_rules({"vehicles": {"ego": str(rain), "vehicle.mini.*": str(dark)}})
    labels = {"time_of_day": "night", "cloudiness": "clear"}
    numbers = {
        "carla_wetness": 0.0,
        "carla_precipitation_deposits": 0.0,
        "rainfall_rate": 25.0,
    }
    assert vehicle_at(
        "tire_friction", numbers, 0.5, "vehicle.lincoln.mkz", "Ego", labels
    ) == pytest.approx(0.35)
    assert vehicle_at(
        "tire_friction", numbers, 0.5, "vehicle.mini.cooper", "", labels
    ) == pytest.approx(0.55)
    # Neither entry: the default rules, which read neither.
    assert vehicle_at(
        "tire_friction", numbers, 0.5, "vehicle.tesla.model3", "", labels
    ) == pytest.approx(1.0)


def test_the_run_config_names_the_rules_and_a_run_without_resets_them(
    tmp_path, monkeypatch
) -> None:
    from autoware_carla_scenario.examples.run import add_environment

    monkeypatch.chdir(tmp_path)
    _grip_file(tmp_path / "measured.yaml", (0.30, 0.50))
    _grip_file(tmp_path / "mini.yaml", (0.20, 0.40))
    registered: list = []
    scenario = SimpleNamespace(register_init=registered.append)

    # Relative to where the run starts.
    add_environment(
        OmegaConf.create({"environment": {"rules": "measured.yaml", "wetness": 100}}),
        scenario,  # type: ignore[arg-type]
    )
    (action,) = registered
    # The rules are no weather setting.
    assert action.settings == {"wetness": 100.0}
    assert tire_friction_range(100.0) == pytest.approx((0.30, 0.50))

    registered.clear()
    add_environment(
        OmegaConf.create(
            {"environment": {"rules": {"vehicles": {"vehicle.mini.*": "mini.yaml"}}}}
        ),
        scenario,  # type: ignore[arg-type]
    )
    assert vehicle_rules_in_use()
    assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))
    # Rules alone register an action, which gives the vehicles their grip
    # (#1937).
    (action,) = registered
    assert action.settings == {} and action.sets_tire_friction

    registered.clear()
    add_environment(OmegaConf.create({"environment": {"rules": None}}), scenario)  # type: ignore[arg-type]
    assert registered == []
    assert not vehicle_rules_in_use()
    assert CARLA_BACKEND.active_rules is CARLA_BACKEND.rules


def test_in_a_batch_each_case_runs_under_its_own_rules(tmp_path, monkeypatch) -> None:
    """A batch builds every case before any runs, and building one follows
    its rules; each run follows its own case's again when it starts, not the
    last case built's (#1936): the grip, the rules by vehicle, and the rules
    its coverage file records."""
    from autoware_carla_scenario.coverage.collector import CoverageCollector
    from autoware_carla_scenario.examples.run import add_environment
    from autoware_carla_scenario.scenario_runner import _follow_weather_rules

    monkeypatch.chdir(tmp_path)
    _grip_file(tmp_path / "grip_low.yaml", (0.30, 0.40))
    _grip_file(tmp_path / "mini.yaml", (0.20, 0.30))

    def build(environment: dict) -> SimpleNamespace:
        scenario = SimpleNamespace(register_init=lambda _action: None)
        add_environment(OmegaConf.create({"environment": environment}), scenario)  # type: ignore[arg-type]
        return scenario

    def files() -> list[str]:
        return [Path(p).name for p, _ in weather_rule_files()]

    def recorded() -> list[str]:
        coverage = CoverageCollector([]).to_dict("case")
        return [Path(p).name for p, _ in coverage["rules"]["files"]]

    a = build({"rules": "grip_low.yaml", "rainfall_rate_mm_h": 10.0})
    by_vehicle = build({"rules": {"vehicles": {"vehicle.mini.*": "mini.yaml"}}})
    b = build({"rules": None})
    # B, built last, left the default rules.
    assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))

    _follow_weather_rules(a)  # type: ignore[arg-type]
    assert tire_friction_range(100.0) == pytest.approx((0.30, 0.40))
    assert not vehicle_rules_in_use()
    assert files() == ["weather_rules.yaml", "grip_low.yaml"]
    assert recorded() == ["weather_rules.yaml", "grip_low.yaml"]

    _follow_weather_rules(by_vehicle)  # type: ignore[arg-type]
    assert vehicle_rules_in_use()
    assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))
    assert files() == ["weather_rules.yaml", "mini.yaml"]

    _follow_weather_rules(b)  # type: ignore[arg-type]
    assert not vehicle_rules_in_use()
    assert CARLA_BACKEND.active_rules is CARLA_BACKEND.rules
    assert recorded() == ["weather_rules.yaml"]


def test_a_scenario_built_without_the_config_runs_under_the_default_rules(
    tmp_path,
) -> None:
    import typesafe_carla.carla as carla

    from autoware_carla_scenario import BaseScenario, EgoConfig, SpawnTransform
    from autoware_carla_scenario.scenario_runner import _follow_weather_rules

    class _Scenario(BaseScenario):
        def setup(self) -> None:
            pass

        def is_done(self) -> bool:
            return True

    scenario = _Scenario(
        EgoConfig(spawn_location=SpawnTransform(carla.Transform(carla.Location())))
    )
    use_weather_rules(str(_grip_file(tmp_path / "left.yaml", (0.30, 0.40))))
    _follow_weather_rules(scenario)
    assert CARLA_BACKEND.active_rules is CARLA_BACKEND.rules
    assert tire_friction_range(100.0) == pytest.approx((0.45, 0.65))


def _composed(rules: object) -> object:
    from omegaconf import open_dict

    from autoware_carla_scenario.examples import run

    cfg = run._compose_config("adversity/nade", ["map=town10hd_opt"])
    with open_dict(cfg):
        cfg.environment.rules = rules
    return cfg


@pytest.mark.parametrize(
    ("rules", "match"),
    [
        ({"default": 5}, r"environment\.rules\.default: expected a file"),
        (
            {"default": {"url": "https://example.com/r.yaml", "sha256": "0" * 64}},
            r"environment\.rules\.default: expected a file",
        ),
        ({"vehicles": []}, r"environment\.rules\.vehicles: expected pattern"),
        ({"vehicles": {"ego": 5}}, r"environment\.rules\.vehicles\.ego: expected"),
        ({"vehicle": {}}, r"environment\.rules: unknown fields \['vehicle'\]"),
        (42, r"environment\.rules: expected a file or"),
    ],
)
def test_a_batch_is_refused_for_malformed_rules_before_it_runs(rules, match) -> None:
    from autoware_carla_scenario.examples import run

    # #1469: refused by the batch check, naming the case, before any run
    # reads a rule file.
    with pytest.raises(run.RunConfigError, match=f"^case: {match}"):
        run.check_batch_configs([_composed(rules)], ["case"])  # type: ignore[list-item]


def test_well_formed_rules_pass_the_batch_check_unread() -> None:
    from autoware_carla_scenario.examples import run

    for rules in (
        None,
        "missing.yaml",
        {"default": "https://example.com/r.yaml", "vehicles": {"ego": "x.yaml"}},
    ):
        run.check_batch_configs([_composed(rules)])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "name",
    [
        "CARLA_QUANTITIES",
        "CAUSE_SETTINGS",
        "EGO_KEY",
        "QUANTITIES",
        "WEATHER_INPUTS",
        "rule_values",
    ],
)
def test_weather_rules_re_exports_the_checked_weather_inputs(name: str) -> None:
    # #1732: the plain part moved to the checked weather_inputs; weather_rules
    # (and odd.backend through it) still names the same objects.
    from autoware_carla_scenario import weather_inputs, weather_rules

    assert getattr(weather_rules, name) is getattr(weather_inputs, name)
    assert name in weather_rules.__all__
    assert CARLA_BACKEND.rule_inputs is weather_inputs.WEATHER_INPUTS


@pytest.mark.parametrize("name", ["cloud_cover_level", "time_of_day_level"])
def test_the_probes_levels_are_the_checked_weather_inputs(name: str) -> None:
    # The levels the grip reads (#1938) and the ODD's probes read alike.
    from autoware_carla_scenario import weather_inputs

    assert getattr(probes, name) is getattr(weather_inputs, name)
    assert name in probes.__all__ and name in weather_inputs.__all__


def test_rule_values_reads_each_as_a_float() -> None:
    from autoware_carla_scenario.weather_inputs import rule_values

    assert rule_values({"carla_wetness": 30, "rainfall_rate": 2.5}) == {
        "carla_wetness": 30.0,
        "rainfall_rate": 2.5,
    }


def test_the_quantities_are_the_rules_and_every_carla_field_is_one() -> None:
    from autoware_carla_scenario.weather_inputs import CARLA_QUANTITIES, QUANTITIES

    assert CARLA_BACKEND.rules is not None
    assert set(QUANTITIES) == set(CARLA_BACKEND.rules.quantities)
    assert set(CARLA_QUANTITIES.values()) <= set(QUANTITIES)


class TestResolveEnvironment:
    def test_the_causes_give_carlas_weather_and_the_visibility_at_the_median(
        self,
    ) -> None:
        out = resolve_environment(
            {
                "rainfall_rate_mm_h": 2.5,
                "fog_visibility_range_m": 300.0,
                "time_of_day": "dusk",
            }
        )
        assert out["rainfall_rate_mm_h"] == 2.5 and out["time_of_day"] == "dusk"
        assert out["precipitation"] == pytest.approx(30.0)
        # What CARLA's rain presets pair with it, from the precipitation.
        assert out["cloudiness"] == pytest.approx(60.0)
        assert out["precipitation_deposits"] == pytest.approx(50.0)
        assert out["wetness"] == pytest.approx(30.0)
        assert 50.0 < out["fog_density"] < 95.0
        assert -6.0 <= out["sun_altitude_angle"] < 0.0
        assert 180.0 <= out["sun_azimuth_angle"] < 360.0
        # The visibility at its median, the least of its parts'.
        assert out["visibility_m"] == pytest.approx(
            weather_at(
                "visibility",
                {
                    "rainfall_rate": 2.5,
                    "fog_visibility_range": 300.0,
                    "time_of_day": "dusk",
                },
                0.5,
            )
        )

    def test_a_cause_not_named_leaves_its_fields_to_the_world(self) -> None:
        out = resolve_environment({"time_of_day": "night"})
        assert set(out) == {
            "time_of_day",
            "sun_altitude_angle",
            "sun_azimuth_angle",
            "visibility_m",
        }
        # The night's light bounds it; no rain, no fog.
        assert out["visibility_m"] == pytest.approx(100.0)

    def test_a_field_given_is_kept_and_what_follows_reads_it(self) -> None:
        out = resolve_environment({"rainfall_rate_mm_h": 25.0, "precipitation": 30.0})
        assert out["precipitation"] == 30.0
        assert out["wetness"] == pytest.approx(30.0)
        out = resolve_environment({"time_of_day": "day", "visibility_m": 42.0})
        assert out["visibility_m"] == 42.0

    def test_the_quantiles_drawn_are_the_ones_read(self) -> None:
        out = resolve_environment(
            {"time_of_day": "night", "visibility_quantile": 1.0},
            {"carla_sun_altitude": 0.0},
        )
        assert out["visibility_m"] == pytest.approx(150.0)
        assert out["sun_altitude_angle"] == pytest.approx(-90.0)

    def test_carlas_fields_alone_name_no_condition_and_get_no_visibility(
        self, caplog
    ) -> None:
        out = resolve_environment({"precipitation": 60.0, "fog_density": 80.0})
        assert "visibility_m" not in out
        # What follows from the precipitation in the rules still does.
        assert out["wetness"] == pytest.approx(60.0)
        assert "names no condition" in caplog.text


@pytest.mark.parametrize(
    ("patterns", "type_id", "role", "expected"),
    [
        ([], "vehicle.mini.cooper", "Ego", -1),
        (["ego"], "vehicle.mini.cooper", "Ego", 0),
        (["EGO"], "vehicle.mini.cooper", "Ego", 0),
        (["Ego"], "vehicle.mini.cooper", "ego", -1),
        # The ego key matches the role only, never the blueprint.
        (["ego"], "ego", "", -1),
        (["vehicle.mini.*", "ego"], "vehicle.mini.cooper", "Ego", 0),
        (["ego", "vehicle.mini.*"], "vehicle.mini.cooper", "Ego", 0),
        (["ego", "vehicle.mini.*"], "vehicle.mini.cooper", "", 1),
        (["vehicle.tesla.*", "vehicle.*"], "vehicle.mini.cooper", "", 1),
        (["vehicle.mini.cooper"], "vehicle.mini.cooper", "", 0),
        (["vehicle.mini.coope?"], "vehicle.mini.cooper", "", 0),
        (["vehicle.mini.coope?"], "vehicle.mini.coope", "", -1),
        (["vehicle.[mt]*"], "vehicle.mini.cooper", "", 0),
        (["vehicle.[!m]*"], "vehicle.mini.cooper", "", -1),
        (["vehicle.[!m]*"], "vehicle.tesla.model3", "", 0),
        (["vehicle.[a-l]*"], "vehicle.mini.cooper", "", -1),
        (["vehicle.[l-n]*"], "vehicle.mini.cooper", "", 0),
        # A `[` with no `]` is itself; `[[]` and `[*]` escape.
        (["vehicle.[mini"], "vehicle.[mini", "", 0),
        (["vehicle.[[]x"], "vehicle.[x", "", 0),
        (["vehicle.[*]"], "vehicle.*", "", 0),
        (["vehicle.[*]"], "vehicle.x", "", -1),
        # Case-sensitive, and of the whole name.
        (["Vehicle.*"], "vehicle.mini.cooper", "", -1),
        (["vehicle.mini"], "vehicle.mini.cooper", "", -1),
    ],
)
def test_vehicle_entry_is_the_first_pattern_the_vehicle_matches(
    patterns: list[str], type_id: str, role: str, expected: int
) -> None:
    import fnmatch

    from autoware_carla_scenario.weather_inputs import EGO_KEY, vehicle_entry

    assert vehicle_entry(patterns, type_id, role) == expected
    # As the loop weather_rules ran before the move (#1732).
    before = -1
    for index, pattern in enumerate(patterns):
        if pattern.lower() == EGO_KEY:
            if role == "Ego":
                before = index
                break
        elif fnmatch.fnmatchcase(type_id, pattern):
            before = index
            break
    assert before == expected


#: Rules under which no rain shows: CARLA's precipitation is 0 at any rate.
_DRY_RULES = """\
version: 3
quantities:
  carla_precipitation:
    parts:
      rain:
        uniform:
          low:  [0, 0, 0, 0, 0]
          high: [0, 0, 0, 0, 0]
"""


def _sweep_precipitation(rules: object = None) -> list[tuple[float, float]]:
    """``(rainfall rate, CARLA precipitation)`` of each case an ODD sweep
    draws, with ``environment.rules`` set to *rules*."""
    from autoware_carla_scenario.sweeper.expand import expand_config

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "environment": {"rules": rules},
            "sweep": {"odd_sample": {"count": 5, "seed": 1}},
        }
    )
    found = []
    for case in expand_config(cfg):
        settings = dict(o.split("=", 1) for o in case)
        found.append(
            (
                float(settings["environment.rainfall_rate_mm_h"]),
                float(settings["environment.precipitation"]),
            )
        )
    return found


def test_a_sweep_draws_carla_weather_by_the_configs_rules(tmp_path) -> None:
    """The CARLA weather a case is given follows the rules its run follows
    (``environment.rules``), not the default ones (#1924)."""
    path = tmp_path / "dry.yaml"
    path.write_text(_DRY_RULES)
    drawn = _sweep_precipitation(str(path))
    # The default rules give rain past 0.1 mm/h some precipitation.
    assert any(rate > 0.1 for rate, _ in drawn)
    assert all(p > 0.0 for rate, p in _sweep_precipitation() if rate > 0.1)
    for rate, precipitation in drawn:
        if rate > 0.1:
            assert precipitation == 0.0
    # Its vehicles' rules give a grip alone: the default rules decide here.
    assert _sweep_precipitation({"default": str(path)}) == drawn


def test_a_sweep_is_not_drawn_by_the_rules_an_earlier_run_left(tmp_path) -> None:
    """A run's rules stay on the process's backend after it; the next sweep
    draws by its own config's, not by them (#1924)."""
    path = tmp_path / "dry.yaml"
    path.write_text(_DRY_RULES)
    expected = _sweep_precipitation()
    use_weather_rules(str(path))
    assert _sweep_precipitation() == expected


class TestWindAndClouds:
    def test_the_wind_speed_gives_carlas_wind_intensity(self) -> None:
        assert weather_range("carla_wind_intensity", {"wind_speed": 20.0}) == (
            pytest.approx((30.0, 30.0))
        )
        assert weather_range("carla_wind_intensity", {"wind_speed": 12.5}) == (
            pytest.approx((8.0, 25.0))
        )
        out = resolve_environment({"wind_speed_kmh": 0.0})
        assert out["wind_intensity"] == 0.0

    # The taxonomy's wind buckets, km/h, and the level the `wind` probe reads
    # each one's CARLA wind intensity back as.
    _WIND_BUCKETS = (
        (0.0, 5.0, "none"),  # no_wind
        (5.0, 20.0, "light"),  # light_wind
        (20.0, 50.0, "moderate"),  # moderate_wind
        (50.0, 100.0, "heavy"),  # strong_wind (the knob's range ends at 100)
    )

    @pytest.mark.parametrize(("low", "high", "level"), _WIND_BUCKETS)
    def test_every_wind_speed_reads_back_as_its_bucket(
        self, low: float, high: float, level: str
    ) -> None:
        # Speeds across the bucket, its lower edge and just under its upper
        # one (the last bucket includes 100 km/h and what is held beyond).
        steps = 40
        speeds = [low + (high - low) * i / steps for i in range(steps)]
        speeds += [high - 1e-6, high - 1e-3]
        if level == "heavy":
            speeds += [high, 150.0]
        for speed in speeds:
            for q in (0.0, 0.5, 1.0):
                intensity = weather_at("carla_wind_intensity", {"wind_speed": speed}, q)
                assert (
                    probes.intensity_level(intensity) == level
                ), f"{speed} km/h at quantile {q} draws intensity {intensity}"

    @pytest.mark.parametrize("sky", ["clear", "partly_cloudy", "overcast"])
    def test_the_sky_drawn_reads_back_as_it(self, sky: str) -> None:
        for q in (0.0, 0.5, 1.0):
            cloudiness = weather_at("carla_cloudiness", {"cloudiness": sky}, q)
            assert probes.cloud_cover_level(cloudiness) == sky

    def test_rain_clouds_the_sky_whatever_the_cloud_cover(self) -> None:
        out = resolve_environment({"cloud_cover": "clear", "rainfall_rate_mm_h": 25.0})
        assert out["cloudiness"] > 90.0
        # A cloud cover alone leaves the rain to the world.
        out = resolve_environment({"cloud_cover": "partly_cloudy"})
        assert "precipitation" not in out and 20.0 <= out["cloudiness"] < 70.0


class TestRoadWater:
    """The road's water the rain leaves: what an EnvironmentAction that sets
    the rain alone applies, the config's own (#2063)."""

    @pytest.mark.parametrize("precipitation", [0.0, 30.0, 45.0, 100.0])
    def test_it_is_what_the_config_gets(self, precipitation: float) -> None:
        resolved = resolve_environment({"precipitation": precipitation})
        assert road_water({"precipitation": precipitation}) == {
            "wetness": pytest.approx(resolved["wetness"]),
            "precipitation_deposits": pytest.approx(resolved["precipitation_deposits"]),
        }

    def test_heavy_rain_floods_the_road(self) -> None:
        assert road_water({"precipitation": 100.0}) == {
            "wetness": pytest.approx(97.5),
            "precipitation_deposits": pytest.approx(90.0),
        }

    def test_water_given_is_kept(self) -> None:
        assert road_water({"precipitation": 100.0, "wetness": 5.0}) == {
            "precipitation_deposits": pytest.approx(90.0)
        }
        assert (
            road_water(
                {"precipitation": 100.0, "wetness": 5.0, "precipitation_deposits": 0.0}
            )
            == {}
        )

    def test_no_rain_names_no_water(self) -> None:
        assert road_water({"fog_density": 50.0}) == {}
        assert road_water({}) == {}

    def test_it_does_not_warn_of_no_condition(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            road_water({"precipitation": 100.0})
        assert "names no condition" not in caplog.text


def _wetness_by_rate_file(path: Path) -> Path:
    """A rule file whose wetness follows the rain and the rainfall rate, the
    drier of the two."""
    path.write_text(
        "version: 3\nquantities:\n  carla_wetness:\n    combine: min\n"
        "    parts:\n"
        "      rain:\n        at: {carla_precipitation: [0, 100]}\n"
        "        uniform:\n          low: [0, 95]\n          high: [0, 100]\n"
        "      rate:\n        at: {rainfall_rate: [0, 25]}\n"
        "        uniform:\n          low: [0, 10]\n          high: [0, 20]\n"
    )
    return path


class TestRoadWaterAtTheRunsConditions:
    """Rules whose water reads a condition read the run's, the action's own
    or the config's, as the config's ``environment`` does (#2063)."""

    @pytest.fixture(autouse=True)
    def _fresh(self, tmp_path) -> object:
        from autoware_carla_scenario.actions.environment import (
            forget_weather_conditions,
        )

        forget_weather_conditions()
        use_weather_rules(str(_wetness_by_rate_file(tmp_path / "wet.yaml")))
        yield
        forget_weather_conditions()

    def _configured(self) -> float:
        resolved = resolve_environment(
            {"precipitation": 100.0, "rainfall_rate_mm_h": 25.0}
        )
        assert resolved["wetness"] == pytest.approx(15.0)
        return float(resolved["wetness"])

    def test_road_water_reads_the_conditions_given(self) -> None:
        wetness = self._configured()
        assert road_water({"precipitation": 100.0}, {"rainfall_rate_mm_h": 25.0}) == {
            "wetness": pytest.approx(wetness),
            "precipitation_deposits": pytest.approx(90.0),
        }
        # Not given, the rate reads the rules' default: no rain falls.
        assert road_water({"precipitation": 100.0})["wetness"] == pytest.approx(0.0)

    def test_an_action_reads_its_own_rate(self) -> None:
        world = _world()
        EnvironmentAction(precipitation=100.0, rainfall_rate_mm_h=25.0).execute(world)
        assert world.get_weather.return_value.wetness == pytest.approx(
            self._configured()
        )

    def test_a_later_action_reads_the_rate_the_run_was_set_to(self) -> None:
        world = _world()
        EnvironmentAction(rainfall_rate_mm_h=25.0).execute(world)
        EnvironmentAction(precipitation=100.0).execute(world)
        assert world.get_weather.return_value.wetness == pytest.approx(
            self._configured()
        )


def _wetness_by_night_file(path: Path) -> Path:
    """A rule file whose wetness follows the rain, and is low by day."""
    path.write_text(
        "version: 3\nquantities:\n  carla_wetness:\n    combine: min\n"
        "    parts:\n"
        "      rain:\n        at: {carla_precipitation: [0, 100]}\n"
        "        uniform:\n          low: [0, 95]\n          high: [0, 100]\n"
        "      night:\n        at: {time_of_day: [day, night]}\n"
        "        uniform:\n          low: [0, 90]\n          high: [0, 100]\n"
    )
    return path


class TestRoadWaterAtTheRunsTimeOfDay:
    """The config's time of day is recorded only as the sun it gives: an
    action reads the time of day off the sun, as the grip does (#2063)."""

    @pytest.fixture(autouse=True)
    def _fresh(self, tmp_path) -> object:
        from autoware_carla_scenario.actions.environment import (
            forget_weather_conditions,
        )

        forget_weather_conditions()
        use_weather_rules(str(_wetness_by_night_file(tmp_path / "night.yaml")))
        yield
        forget_weather_conditions()

    def test_rain_at_night_gives_the_configs_water(self) -> None:
        resolved = resolve_environment({"precipitation": 100.0, "time_of_day": "night"})
        world = _world()
        weather = world.get_weather.return_value
        weather.sun_altitude_angle = float(resolved["sun_altitude_angle"])
        weather.sun_azimuth_angle = float(resolved["sun_azimuth_angle"])
        EnvironmentAction(precipitation=100.0).execute(world)
        assert weather.wetness == pytest.approx(resolved["wetness"])
        assert weather.wetness == pytest.approx(95.0)

    def test_the_actions_own_sun_holds(self) -> None:
        world = _world()  # The sun is up.
        EnvironmentAction(precipitation=100.0).execute(world)
        assert world.get_weather.return_value.wetness == pytest.approx(0.0)
        world = _world()
        EnvironmentAction(precipitation=100.0, sun_altitude_angle=-60.0).execute(world)
        assert world.get_weather.return_value.wetness == pytest.approx(95.0)
