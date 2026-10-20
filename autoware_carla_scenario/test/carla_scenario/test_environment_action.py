"""The environment action: what it applies, and what it leaves alone.

The whole point of the action is the partial update -- a scenario that wants
rain must not silently reset the time of day -- so that is what most of these
tests are about.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import BaseScenario, EnvironmentAction
from autoware_carla_scenario.actions.environment import (
    apply_tire_friction,
    follow_tire_friction,
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


class TestRain:
    def test_rain_brings_the_clouds_and_the_wet_road_of_carlas_presets(self) -> None:
        """HardRain: precipitation 100, cloudiness 100, deposits 90, wetness 100."""
        world = _world()
        EnvironmentAction(rain=100.0).execute(world)

        weather = _applied(world)
        assert math.isclose(weather.precipitation, 100.0)
        assert math.isclose(weather.cloudiness, 100.0)
        assert math.isclose(weather.precipitation_deposits, 90.0)
        assert math.isclose(weather.wetness, 100.0)

    def test_between_presets_it_is_interpolated(self) -> None:
        settings = EnvironmentAction(rain=45.0).settings
        assert settings["precipitation"] == pytest.approx(45.0)
        assert settings["cloudiness"] == pytest.approx(70.0)
        assert settings["precipitation_deposits"] == pytest.approx(55.0)
        assert settings["wetness"] == pytest.approx(45.0)

    def test_no_rain_is_a_dry_clear_sky(self) -> None:
        settings = EnvironmentAction(rain=0.0).settings
        assert settings == {
            "precipitation": 0.0,
            "cloudiness": 0.0,
            "precipitation_deposits": 0.0,
            "wetness": 0.0,
        }

    def test_a_field_given_beside_rain_wins(self) -> None:
        settings = EnvironmentAction(rain=100.0, cloudiness=20.0).settings
        assert settings["cloudiness"] == pytest.approx(20.0)
        assert settings["wetness"] == pytest.approx(100.0)

    def test_rain_is_clamped_to_its_range(self) -> None:
        assert EnvironmentAction(rain=150.0).settings["precipitation"] == 100.0
        assert EnvironmentAction(rain=-5.0).settings["precipitation"] == 0.0

    def test_rain_and_precipitation_together_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="rain or precipitation"):
            EnvironmentAction(rain=50.0, precipitation=10.0)

    def test_rain_is_keyword_only(self) -> None:
        """A twelfth positional argument has nowhere to go: rain is keyword-only."""
        positional: list = [None] * 11
        with pytest.raises(TypeError):
            EnvironmentAction(*positional, 50.0)  # type: ignore[call-arg]


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
    def test_rain_scales_every_vehicles_wheels(self) -> None:
        world = _world_with(_Vehicle(1), _Vehicle(2, friction=2.0))
        EnvironmentAction(rain=100.0, tire_friction_quantile=0.0).execute(world)

        # rain=100: wetness 100, so the low end of a flooded road's range.
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.45] * 4)
        assert world.fleet[1].frictions == pytest.approx([2.0 * 0.45] * 4)

    def test_the_middle_of_the_range_by_default(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(rain=100.0).execute(world)
        assert world.fleet[0].frictions == pytest.approx([3.5 * 0.55] * 4)

    def test_a_dry_road_at_its_middle_leaves_the_vehicles_alone(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(rain=0.0).execute(world)
        assert world.fleet[0].applied == 0

    def test_weather_off_the_road_does_not_touch_the_grip(self) -> None:
        action = EnvironmentAction(fog_density=50.0, sun_altitude_angle=-10.0)
        assert not action.sets_tire_friction
        world = _world_with(_Vehicle(1))
        action.execute(world)
        world.get_actors.assert_not_called()

    def test_a_fixed_friction_wins_over_the_weather(self) -> None:
        world = _world_with(_Vehicle(1))
        EnvironmentAction(
            rain=100.0, tire_friction=0.3, tire_friction_quantile=1.0
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
        scenario.register_init(EnvironmentAction(rain=80.0))
        scenario.register_pre_tick(EnvironmentAction(tire_friction=0.5))
        assert scenario._post_tick_callbacks.count(follow_tire_friction) == 1

    def test_fog_is_not_followed(self) -> None:
        scenario = self._scenario()
        scenario.register_init(EnvironmentAction(fog_density=40.0))
        assert follow_tire_friction not in scenario._post_tick_callbacks


class TestVisibility:
    """How far drivers see follows the weather the action leaves."""

    @pytest.fixture(autouse=True)
    def _forget(self) -> None:
        from autoware_carla_scenario.visibility import forget_visibility

        forget_visibility()

    def _seen(self, world: MagicMock) -> float | None:
        from autoware_carla_scenario.visibility import world_visibility

        return world_visibility(world.id)

    def test_fog_shortens_it_to_the_middle_of_its_range(self) -> None:
        world = _world()
        world.id = 5
        EnvironmentAction(fog_density=100.0).execute(world)
        assert self._seen(world) == pytest.approx(27.5)

    def test_night_comes_from_the_sun(self) -> None:
        world = _world()
        world.id = 5
        EnvironmentAction(sun_altitude_angle=-30.0, visibility_quantile=0.0).execute(
            world
        )
        assert self._seen(world) == pytest.approx(50.0)

    def test_rain_counts_the_weather_the_world_already_has(self) -> None:
        world = _world(sun_altitude_angle=-30.0)
        world.id = 5
        EnvironmentAction(rain=100.0, visibility_quantile=1.0).execute(world)
        # Heavy rain at night: the rain's and the night's 150 m.
        assert self._seen(world) == pytest.approx(150.0)

    def test_a_fixed_visibility_wins_over_the_weather(self) -> None:
        world = _world()
        world.id = 5
        EnvironmentAction(
            fog_density=100.0, visibility_m=200.0, visibility_quantile=0.0
        ).execute(world)
        assert self._seen(world) == 200.0

    def test_the_visibility_alone_needs_no_weather(self) -> None:
        world = _world(sun_altitude_angle=-30.0)
        world.id = 5
        action = EnvironmentAction(visibility_quantile=1.0)
        assert action.sets_visibility and not action.sets_tire_friction
        action.execute(world)
        world.set_weather.assert_not_called()
        assert self._seen(world) == pytest.approx(150.0)

    def test_weather_that_does_not_hide_the_road_sets_none(self) -> None:
        world = _world()
        world.id = 5
        action = EnvironmentAction(cloudiness=80.0, wetness=40.0)
        assert not action.sets_visibility
        action.execute(world)
        assert self._seen(world) is None

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
            ({"visibility_quantile": -0.1}, r"\[0, 1\]"),
        ],
    )
    def test_out_of_range_is_refused(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            EnvironmentAction(**kwargs)
