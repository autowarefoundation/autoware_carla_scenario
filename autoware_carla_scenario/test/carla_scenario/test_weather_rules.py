"""The weather rules: the CARLA backend's conditional distributions for the
grip and the visibility, a run's over them, and rules that differ by vehicle."""

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
from autoware_carla_scenario.visibility import visibility, visibility_range
from autoware_carla_scenario.weather_rules import (
    DEFAULT_RULES,
    RulesError,
    use_weather_rules,
    vehicle_at,
    vehicle_range,
    vehicle_rules_in_use,
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


def _grip_file(path: Path, flooded: tuple[float, float], extends: str = "") -> Path:
    """A rule file naming only the grip's distribution: its points are the
    default's."""
    path.write_text(
        "version: 2\n"
        + (f"extends: {extends}\n" if extends else "")
        + "quantities:\n  tire_friction:\n    parts:\n      water:\n        uniform:\n"
        f"          low: [0.90, 0.65, {flooded[0]}]\n"
        f"          high: [1.10, 0.85, {flooded[1]}]\n"
    )
    return path


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

    def test_they_are_the_carla_backends_and_water_is_derived(self) -> None:
        assert CARLA_BACKEND.rules is not None
        assert CARLA_BACKEND.rules.files[0][0] == str(DEFAULT_RULES)
        assert weather_range(
            "tire_friction", {"wetness": 10.0, "precipitation_deposits": 50.0}
        ) == pytest.approx(weather_range("tire_friction", {"water": 50.0}))

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_the_grip_at_a_quantile_is_its_uniform_distributions(self, q) -> None:
        low, high = tire_friction_range(70.0)
        assert tire_friction(70.0, quantile=q) == pytest.approx(low + q * (high - low))

    def test_the_visibilitys_parts_move_together_at_one_quantile(self) -> None:
        # fog 80: 31.7-80 m; rain 30: 180-320 m; day: 250-400 m.  At the
        # quantile 0.5 each part is at its median, and the shortest holds.
        fog = _interpolate(_FOG, 80.0)
        assert visibility(80.0, 30.0, 45.0, 0.5) == pytest.approx((fog[0] + fog[1]) / 2)
        # Night and light rain: the light's 50-150 m and the rain's 180-320 m.
        assert visibility(0.0, 30.0, -10.0, 0.25) == pytest.approx(75.0)
        assert visibility(0.0, 30.0, -10.0, 1.0) == pytest.approx(150.0)
        # The quantile is clamped.
        assert visibility(0.0, 30.0, -10.0, 7.0) == pytest.approx(150.0)

    def test_an_old_fuzzy_rule_file_is_refused(self, tmp_path) -> None:
        path = tmp_path / "v1.yaml"
        path.write_text("version: 1\nquantities: {}\n")
        with pytest.raises(RulesError, match="version 1"):
            use_weather_rules(str(path))

    def test_the_manifest_follows_a_knob_to_the_rules(self) -> None:
        inputs = CARLA_BACKEND.manifest()["rule_inputs"]
        assert inputs["precipitation"]["knob"] == "environment.rain"
        assert inputs["fog_density"]["knob"] == "environment.fog_density"
        assert "wetness" in inputs["water"]["text"]

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
        assert visibility_range(fog_density=100.0) == pytest.approx((15.0, 40.0))
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
        with pytest.raises(RulesError, match="unknown input 'temperature'"):
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
        flooded = {"wetness": 100.0}
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
        assert visibility_range(fog_density=100.0) == pytest.approx((15.0, 40.0))

    def test_a_vehicles_file_extends_and_is_read_over_the_runs_default(
        self, tmp_path
    ) -> None:
        _grip_file(tmp_path / "shared.yaml", (0.35, 0.55))
        mini = tmp_path / "vehicles" / "mini.yaml"
        mini.parent.mkdir()
        mini.write_text("version: 2\nextends: ../shared.yaml\nquantities: {}\n")
        use_weather_rules({"vehicles": {"vehicle.mini.*": str(mini)}})
        assert vehicle_range(
            "tire_friction", {"wetness": 100.0}, "vehicle.mini.cooper", ""
        ) == pytest.approx((0.35, 0.55))

    def test_malformed_settings_are_refused(self, tmp_path) -> None:
        with pytest.raises(RulesError, match=r"unknown fields \['vehicle'\]"):
            use_weather_rules({"vehicle": {}})
        with pytest.raises(RulesError, match="expected pattern -> file"):
            use_weather_rules({"vehicles": ["a.yaml"]})
        with pytest.raises(RulesError, match="expected a file or"):
            use_weather_rules(42)  # type: ignore[arg-type]

    def test_without_vehicles_every_vehicle_reads_the_default(self) -> None:
        assert not vehicle_rules_in_use()
        assert vehicle_range(
            "tire_friction", {"wetness": 100.0}, "vehicle.mini.cooper", "Ego"
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
        apply_weather_grip(world, {"wetness": 100.0}, 1.0)
        assert ego.frictions == pytest.approx([3.5 * 0.65] * 4)
        assert world_tire_friction(world.id) == pytest.approx(0.65)


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

    add_environment(OmegaConf.create({"environment": {"rules": None}}), scenario)  # type: ignore[arg-type]
    assert registered == []
    assert not vehicle_rules_in_use()
    assert CARLA_BACKEND.active_rules is CARLA_BACKEND.rules
