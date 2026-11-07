"""The environment action: what it applies, and what it leaves alone.

The whole point of the action is the partial update -- a scenario that wants
rain must not silently reset the time of day -- so that is what most of these
tests are about.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import BaseScenario, EnvironmentAction
from autoware_carla_scenario.actions.environment import (
    apply_tire_friction,
    follow_tire_friction,
    forget_weather_conditions,
    tire_friction,
    tire_friction_range,
)


def _world(**current: float) -> MagicMock:
    """A world whose weather starts at *current*, defaulting to clear noon."""
    weather = carla.WeatherParameters(
        cloudiness=current.get("cloudiness", 0.0),
        precipitation=current.get("precipitation", 0.0),
        sun_altitude_angle=current.get("sun_altitude_angle", 70.0),
    )
    world = MagicMock()
    world.get_weather.return_value = weather
    return world


def _applied(world: MagicMock) -> carla.WeatherParameters:
    world.set_weather.assert_called_once()
    return world.set_weather.call_args[0][0]


class TestPartialUpdate:
    def test_only_the_named_fields_change(self) -> None:
        world = _world(sun_altitude_angle=15.0)
        EnvironmentAction(precipitation=60.0).execute(world)

        weather = _applied(world)
        assert math.isclose(weather.precipitation, 60.0)
        assert math.isclose(weather.sun_altitude_angle, 15.0)

    def test_the_action_reads_the_world_rather_than_a_default(self) -> None:
        """Building a fresh WeatherParameters would reset everything unset."""
        world = _world(cloudiness=80.0, sun_altitude_angle=-10.0)
        EnvironmentAction(fog_density=20.0).execute(world)

        weather = _applied(world)
        assert math.isclose(weather.cloudiness, 80.0)
        assert math.isclose(weather.sun_altitude_angle, -10.0)
        assert math.isclose(weather.fog_density, 20.0)

    def test_several_fields_at_once(self) -> None:
        world = _world()
        EnvironmentAction(
            cloudiness=90.0,
            precipitation=70.0,
            precipitation_deposits=50.0,
            fog_density=30.0,
            sun_altitude_angle=5.0,
        ).execute(world)

        weather = _applied(world)
        assert math.isclose(weather.cloudiness, 90.0)
        assert math.isclose(weather.precipitation, 70.0)
        assert math.isclose(weather.precipitation_deposits, 50.0)
        assert math.isclose(weather.fog_density, 30.0)
        assert math.isclose(weather.sun_altitude_angle, 5.0)


class TestNoSettings:
    def test_an_empty_action_touches_nothing(self) -> None:
        """Setting the weather to what it already is would look the same; this
        says so in the log instead, because an action that sets nothing is
        almost certainly an unfinished card rather than an intention."""
        world = _world()
        EnvironmentAction().execute(world)
        world.set_weather.assert_not_called()

    def test_an_empty_action_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            EnvironmentAction(label="unfinished").execute(_world())
        assert "unfinished" in caplog.text


class TestIntrospection:
    def test_settings_report_only_what_was_given(self) -> None:
        action = EnvironmentAction(precipitation=60.0, fog_density=20.0)
        assert action.settings == {"precipitation": 60.0, "fog_density": 20.0}

    def test_settings_are_a_copy(self) -> None:
        action = EnvironmentAction(precipitation=60.0)
        action.settings["precipitation"] = 0.0
        assert action.settings == {"precipitation": 60.0}


class TestConditions:
    """The conditions a run is set to are recorded, not applied: CARLA's
    fields render them, worked out before the run (resolve_environment)."""

    @pytest.fixture(autouse=True)
    def _forget(self) -> None:
        forget_weather_conditions()

    def test_they_are_recorded_for_the_odds_probes(self) -> None:
        from autoware_carla_scenario.odd import probes

        world = _world()
        world.id = 7
        assert probes.rainfall_rate_mm_h(world) is None
        action = EnvironmentAction(rainfall_rate_mm_h=3.0, fog_visibility_range_m=400.0)
        assert action.sets_conditions and action.settings == {}
        action.execute(world)
        world.set_weather.assert_not_called()
        assert probes.rainfall_rate_mm_h(world) == 3.0
        assert probes.fog_visibility_range_m(world) == 400.0
        forget_weather_conditions()
        assert probes.rainfall_rate_mm_h(world) is None
        assert probes.fog_visibility_range_m(world) is None

    def test_carlas_fields_alone_record_none(self) -> None:
        from autoware_carla_scenario.odd import probes

        world = _world()
        world.id = 7
        EnvironmentAction(precipitation=80.0).execute(world)
        assert probes.rainfall_rate_mm_h(world) is None

    def test_what_resolve_environment_works_out_is_applied(self) -> None:
        from autoware_carla_scenario.weather_rules import resolve_environment

        world = _world()
        settings = resolve_environment({"rainfall_rate_mm_h": 25.0})
        EnvironmentAction(**settings).execute(world)
        weather = _applied(world)
        # Heavy rain: what CARLA's HardRain preset pairs with it.
        assert weather.precipitation == pytest.approx(97.5)
        assert weather.cloudiness == pytest.approx(97.5, abs=3.0)
        assert weather.wetness == pytest.approx(97.5, abs=3.0)

    def test_they_are_keyword_only(self) -> None:
        """A twelfth positional argument has nowhere to go."""
        positional: list = [None] * 11
        with pytest.raises(TypeError):
            EnvironmentAction(*positional, 50.0)  # type: ignore[call-arg]


#: Every numeric argument of EnvironmentAction.
_NUMERIC_ARGUMENTS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
    "fog_distance",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "rainfall_rate_mm_h",
    "fog_visibility_range_m",
    "tire_friction",
    "tire_friction_quantile",
    "visibility_m",
)
_PERCENT_ARGUMENTS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
)


def _given(name: str, value: float) -> dict[str, Any]:
    """*name* set to *value*, as keyword arguments."""
    return {name: value}


class TestRefusedValues:
    """A value CARLA or the weather rules cannot use fails at construction,
    naming the argument, not later inside the run (#1322)."""

    @pytest.mark.parametrize("name", _NUMERIC_ARGUMENTS)
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_value_is_refused(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=rf"{name} must be finite"):
            EnvironmentAction(**_given(name, value))

    @pytest.mark.parametrize("name", _PERCENT_ARGUMENTS)
    @pytest.mark.parametrize("value", [-0.1, 100.1])
    def test_a_share_outside_0_100_is_refused(self, name: str, value: float) -> None:
        with pytest.raises(ValueError, match=rf"{name} must be within \[0, 100\]"):
            EnvironmentAction(**_given(name, value))

    @pytest.mark.parametrize("name", _PERCENT_ARGUMENTS)
    def test_the_ends_of_0_100_are_taken(self, name: str) -> None:
        assert EnvironmentAction(**_given(name, 0.0)).settings[name] == 0.0
        assert EnvironmentAction(**_given(name, 100.0)).settings[name] == 100.0

    def test_unbounded_fields_take_any_finite_value(self) -> None:
        settings = EnvironmentAction(
            fog_distance=-5.0, sun_altitude_angle=-90.0, sun_azimuth_angle=720.0
        ).settings
        assert settings == {
            "fog_distance": -5.0,
            "sun_altitude_angle": -90.0,
            "sun_azimuth_angle": 720.0,
        }

    def test_a_negative_rainfall_rate_or_no_fog_visibility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rainfall_rate_mm_h must not be negative"):
            EnvironmentAction(rainfall_rate_mm_h=-1.0)
        with pytest.raises(ValueError, match="fog_visibility_range_m must be positive"):
            EnvironmentAction(fog_visibility_range_m=0.0)


class TestTheEnvironmentConfig:
    """``examples/run.py``'s ``environment`` config names the key it refuses."""

    @staticmethod
    def _add(environment: dict) -> list:
        from omegaconf import OmegaConf

        from autoware_carla_scenario.examples.run import add_environment

        registered: list = []
        scenario = SimpleNamespace(register_init=registered.append)
        add_environment(OmegaConf.create({"environment": environment}), scenario)  # type: ignore[arg-type]
        return registered

    @pytest.mark.parametrize(
        "key",
        [
            "rainfall_rate_mm_h",
            "wetness",
            "fog_density",
            "tire_friction",
            "visibility_m",
        ],
    )
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_a_non_finite_value_names_the_key(self, key: str, value: str) -> None:
        with pytest.raises(ValueError, match=rf"environment\.{key} must be finite"):
            self._add({key: float(value)})

    @pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["huge", "-huge"])
    def test_a_whole_number_too_big_for_a_float_names_the_key(self, value: int) -> None:
        """#1652: ``float()`` raised OverflowError on it."""
        with pytest.raises(
            ValueError, match=r"^environment\.fog_density must be finite, got -?1000+$"
        ):
            self._add({"fog_density": value})

    def test_a_share_out_of_range_names_the_key(self) -> None:
        with pytest.raises(
            ValueError, match=r"environment\.cloudiness must be within \[0, 100\]"
        ):
            self._add({"cloudiness": 150})

    def test_a_value_that_is_not_a_number_names_the_key(self) -> None:
        with pytest.raises(ValueError, match=r"environment\.wetness must be a number"):
            self._add({"wetness": "soaked"})

    @pytest.mark.parametrize("value", [True, False, "15", "0.5"])
    def test_a_boolean_or_a_string_is_no_number(self, value: object) -> None:
        # float() read true as 1 and "15" as 15 (#1445).
        with pytest.raises(
            ValueError, match=r"^environment\.cloudiness must be a number"
        ):
            self._add({"cloudiness": value})

    @pytest.mark.parametrize(
        "override",
        ["environment.cloudiness=true", "environment.cloudiness='15'"],
    )
    def test_a_run_is_refused_for_it_before_anything_is_built(
        self, override: str
    ) -> None:
        from autoware_carla_scenario.examples import run

        from .test_video_removed import _shipped

        cfg = _shipped([override])
        with pytest.raises(
            run.RunConfigError, match=r"^environment\.cloudiness must be a number"
        ):
            run.check_run_config(cfg)

    def test_finite_values_are_taken(self) -> None:
        (action,) = self._add(
            {"rainfall_rate_mm_h": 25, "wetness": 40, "tire_friction": 0.8}
        )
        # Worked out from the rainfall rate, before the run.
        assert action.settings["precipitation"] == pytest.approx(97.5)
        # Set by the config: kept.
        assert action.settings["wetness"] == 40.0

    def test_a_time_of_day_is_one_of_the_taxonomys(self) -> None:
        (action,) = self._add({"time_of_day": "night"})
        assert action.settings["sun_altitude_angle"] < -6.0
        assert action.sets_visibility
        with pytest.raises(
            ValueError, match=r"environment\.time_of_day must be one of"
        ):
            self._add({"time_of_day": "noon"})


class _Vehicle:
    """A vehicle whose physics control is a copy each read, as CARLA's is."""

    def __init__(self, actor_id: int, friction: float = 3.5) -> None:
        self.id = actor_id
        self.frictions = [friction] * 4
        self.applied = 0

    def get_physics_control(self) -> SimpleNamespace:
        return SimpleNamespace(
            wheels=[
                SimpleNamespace(friction_force_multiplier=f) for f in self.frictions
            ]
        )

    def apply_physics_control(self, physics: SimpleNamespace) -> None:
        self.frictions = [w.friction_force_multiplier for w in physics.wheels]
        self.applied += 1


_world_ids = itertools.count(1000)


def _world_with(*vehicles: _Vehicle, **current: float) -> MagicMock:
    """A world (an episode of its own) holding *vehicles*, its road as *current*."""
    world = _world()
    weather = world.get_weather.return_value
    weather.wetness = current.get("wetness", 0.0)
    weather.precipitation_deposits = current.get("precipitation_deposits", 0.0)
    world.id = next(_world_ids)
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


class TestTireFrictionRange:
    def test_a_dry_road_is_centred_on_carlas_grip(self) -> None:
        assert tire_friction_range(0.0) == pytest.approx((0.90, 1.10))
        assert tire_friction(0.0, 0.0) == pytest.approx(1.0)

    def test_the_wetter_the_road_the_less_grip(self) -> None:
        wet = tire_friction_range(50.0)
        flooded = tire_friction_range(100.0, 90.0)
        assert wet == pytest.approx((0.65, 0.85))
        assert flooded == pytest.approx((0.45, 0.65))
        assert tire_friction_range(25.0) == pytest.approx((0.775, 0.975))

    def test_standing_water_counts_as_much_as_wetness(self) -> None:
        assert tire_friction_range(0.0, 50.0) == tire_friction_range(50.0, 0.0)

    def test_the_quantile_picks_a_point_in_the_range(self) -> None:
        assert tire_friction(100.0, 0.0, 0.0) == pytest.approx(0.45)
        assert tire_friction(100.0, 0.0, 1.0) == pytest.approx(0.65)
        assert tire_friction(100.0, 0.0, 2.0) == pytest.approx(0.65)


class TestTireFriction:
    def test_a_wet_road_scales_every_vehicles_wheels(self) -> None:
        world = _world_with(_Vehicle(1), _Vehicle(2, friction=2.0))
        EnvironmentAction(wetness=100.0, tire_friction_quantile=0.0).execute(world)

        # Wetness 100: the low end of a flooded road's range.
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.45] * 4)
        assert world.fleet[1].frictions == pytest.approx([2.0 * 0.45] * 4)

    def test_the_middle_of_the_range_by_default(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(wetness=100.0).execute(world)
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.55] * 4)

    def test_a_dry_road_at_its_middle_leaves_the_vehicles_alone(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(wetness=0.0, precipitation_deposits=0.0).execute(world)
        assert world.fleet[0].applied == 0

    def test_puddles_alone_set_the_grip(self) -> None:
        assert EnvironmentAction(precipitation_deposits=80.0).sets_tire_friction

    def test_weather_off_the_road_does_not_touch_the_grip(self) -> None:
        action = EnvironmentAction(fog_density=50.0, sun_altitude_angle=-10.0)
        assert not action.sets_tire_friction
        world = _world_with(_Vehicle(1))
        action.execute(world)
        world.get_actors.assert_not_called()

    def test_a_fixed_friction_wins_over_the_weather(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(
            wetness=100.0, tire_friction=0.3, tire_friction_quantile=1.0
        ).execute(world)
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.3] * 4)

    def test_the_grip_alone_needs_no_weather(self) -> None:
        world = _world_with(_Vehicle(1), wetness=50.0)
        EnvironmentAction(tire_friction_quantile=1.0).execute(world)
        world.set_weather.assert_not_called()
        # Read off the world's own wet road.
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.85] * 4)

    def test_a_second_factor_replaces_the_first(self) -> None:
        """Rain that starts mid-run must not compound the dry road's factor."""
        world = _world_with(_Vehicle(1))
        EnvironmentAction(tire_friction=0.5).execute(world)
        EnvironmentAction(tire_friction=0.8).execute(world)
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.8] * 4)

    def test_vehicles_spawned_later_follow(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(tire_friction=0.6).execute(world)
        world.fleet.append(_Vehicle(2))

        follow_tire_friction(world)
        follow_tire_friction(world)

        assert world.fleet[1].frictions == pytest.approx([3.5 * 0.6] * 4)
        assert world.fleet[1].applied == 1
        assert world.fleet[0].applied == 1
        # The second call found nothing new, so asked the server for nothing.
        assert world.get_actors.call_args_list[-1].args == ([2],)

    def test_the_same_factor_again_only_reaches_new_vehicles(self) -> None:
        world = _world_with(_Vehicle(1))
        apply_tire_friction(world, 0.6)
        apply_tire_friction(world, 0.6)
        assert world.get_actors.call_count == 1

    def test_following_does_nothing_before_a_grip_is_set(self) -> None:
        world = _world_with(_Vehicle(1))
        follow_tire_friction(world)
        world.get_actors.assert_not_called()

    def test_a_vehicle_without_physics_is_skipped(self) -> None:
        broken = MagicMock()
        broken.id = 7
        broken.get_physics_control.side_effect = RuntimeError("destroyed")
        world = _world_with(_Vehicle(1))
        world.fleet.insert(0, broken)
        apply_tire_friction(world, 0.5)
        assert world.fleet[1].frictions == pytest.approx([3.5 * 0.5] * 4)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"tire_friction": 0.0}, "positive"),
            ({"tire_friction_quantile": 1.5}, r"\[0, 1\]"),
        ],
    )
    def test_out_of_range_is_refused(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            EnvironmentAction(**kwargs)


class TestTheGripReadBack:
    def test_the_grip_given_the_wheels_is_what_the_probe_reads(self) -> None:
        from autoware_carla_scenario.actions.environment import world_tire_friction

        world = _world_with(_Vehicle(1))
        assert world_tire_friction(world.id) is None
        apply_tire_friction(world, 0.55)
        assert world_tire_friction(world.id) == 0.55
        assert world_tire_friction(None) is None

    def test_forgetting_the_grip_leaves_none_for_the_next_run(self) -> None:
        """A soft world reset keeps the episode: the runner forgets the grip
        the last run set, as it forgets the visibility (#795)."""
        from autoware_carla_scenario.actions.environment import (
            forget_tire_friction,
            world_tire_friction,
        )

        world = _world_with()
        apply_tire_friction(world, 0.55)
        forget_tire_friction(world)
        assert world_tire_friction(world.id) is None
        # A vehicle the next run spawns keeps the grip it was spawned with.
        world.fleet.append(_Vehicle(2))
        follow_tire_friction(world)
        assert world.fleet[0].frictions == pytest.approx([3.5] * 4)
        # A grip the next run sets reaches every vehicle again rather than
        # only the unseen ones.
        apply_tire_friction(world, 0.55)
        assert world_tire_friction(world.id) == 0.55
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.55] * 4)

    def test_a_vehicle_outliving_the_run_gets_its_spawned_grip_back(self) -> None:
        """An ego attached to an actor someone else spawned, or any vehicle of
        a retried run (no reset), survives the forgetting: it gets back the
        grip it was spawned with, and the next run's grip scales that rather
        than compounding with the last one's (#795)."""
        from autoware_carla_scenario.actions.environment import (
            forget_tire_friction,
        )

        ego = _Vehicle(1)
        world = _world_with(ego)
        apply_tire_friction(world, 0.55)
        assert ego.frictions == pytest.approx([3.5 * 0.55] * 4)
        forget_tire_friction(world)
        assert ego.frictions == pytest.approx([3.5] * 4)
        apply_tire_friction(world, 0.55)
        assert ego.frictions == pytest.approx([3.5 * 0.55] * 4)

    def test_forgetting_no_grip_asks_nothing_of_the_server(self) -> None:
        from autoware_carla_scenario.actions.environment import (
            forget_tire_friction,
        )

        world = _world_with(_Vehicle(1))
        forget_tire_friction(world)
        world.get_actors.assert_not_called()


class TestFollowingTheGrip:
    """A scenario with an action that sets the grip follows it after every tick."""

    def _scenario(self) -> BaseScenario:
        class _Scenario(BaseScenario):
            def setup(self) -> None: ...

            def is_done(self) -> bool:
                return False

        return _Scenario(MagicMock())

    def test_a_wet_road_is_followed_once(self) -> None:
        scenario = self._scenario()
        scenario.register_init(EnvironmentAction(wetness=80.0))
        scenario.register_pre_tick(EnvironmentAction(tire_friction=0.5))
        assert scenario._post_tick_callbacks.count(follow_tire_friction) == 1

    def test_fog_is_not_followed(self) -> None:
        scenario = self._scenario()
        scenario.register_init(EnvironmentAction(fog_density=40.0))
        assert follow_tire_friction not in scenario._post_tick_callbacks


class TestVisibility:
    """How far drivers see is the number the action is given: worked out from
    the conditions before the run (resolve_environment), not by the action."""

    @pytest.fixture(autouse=True)
    def _forget(self) -> None:
        from autoware_carla_scenario.visibility import forget_visibility

        forget_visibility()

    def _seen(self, world: MagicMock) -> float | None:
        from autoware_carla_scenario.visibility import world_visibility

        return world_visibility(world.id)

    def test_the_visibility_given_is_recorded(self) -> None:
        world = _world()
        world.id = 5
        action = EnvironmentAction(visibility_m=80.0)
        assert action.sets_visibility and not action.sets_tire_friction
        action.execute(world)
        world.set_weather.assert_not_called()
        assert self._seen(world) == 80.0

    def test_carlas_weather_alone_sets_none(self) -> None:
        world = _world()
        world.id = 5
        action = EnvironmentAction(fog_density=100.0, sun_altitude_angle=-30.0)
        assert not action.sets_visibility
        action.execute(world)
        assert self._seen(world) is None

    def test_the_conditions_give_it_before_the_run(self) -> None:
        from autoware_carla_scenario.weather_rules import resolve_environment

        world = _world()
        world.id = 5
        settings = resolve_environment({"fog_visibility_range_m": 15.0})
        EnvironmentAction(**settings).execute(world)
        # Thick fog: 12-18 m, at its median.
        assert self._seen(world) == pytest.approx(15.0)

    def test_the_probe_reads_it(self) -> None:
        from autoware_carla_scenario.odd import probes

        world = _world()
        world.id = 5
        assert probes.visibility(world) is None
        EnvironmentAction(visibility_m=60.0).execute(world)
        assert probes.visibility(world) == 60.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"visibility_m": 0.0}, "positive"),
            ({"tire_friction_quantile": -0.1}, r"\[0, 1\]"),
        ],
    )
    def test_out_of_range_is_refused(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            EnvironmentAction(**kwargs)


def test_the_wind_speed_is_recorded_and_the_cloud_cover_named() -> None:
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples.run import add_environment
    from autoware_carla_scenario.odd import probes

    forget_weather_conditions()
    registered: list = []
    scenario = SimpleNamespace(register_init=registered.append)
    add_environment(
        OmegaConf.create(
            {"environment": {"wind_speed_kmh": 30, "cloud_cover": "overcast"}}
        ),
        scenario,  # type: ignore[arg-type]
    )
    (action,) = registered
    assert action.settings["cloudiness"] >= 70.0
    assert probes.intensity_level(action.settings["wind_intensity"]) == "moderate"
    world = _world()
    world.id = 9
    action.execute(world)
    assert probes.wind_speed_kmh(world) == 30.0
    with pytest.raises(ValueError, match=r"environment\.cloud_cover must be one of"):
        add_environment(
            OmegaConf.create({"environment": {"cloud_cover": "stormy"}}),
            scenario,  # type: ignore[arg-type]
        )
    forget_weather_conditions()
