"""The environment action: what it applies, and what it leaves alone.

The whole point of the action is the partial update -- a scenario that wants
rain must not silently reset the time of day -- so that is what most of these
tests are about.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import EnvironmentAction


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
