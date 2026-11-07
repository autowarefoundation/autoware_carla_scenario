"""How far drivers see: the range the conditions leave (the weather rules'),
the speed it allows, and the record a traffic backend reads it from."""

from __future__ import annotations

import pytest

from autoware_carla_scenario.visibility import (
    driver_quantile,
    forget_visibility,
    set_world_visibility,
    sight_speed,
    sight_speed_at,
    world_visibility,
)
from autoware_carla_scenario.weather_rules import weather_at, weather_range


def visibility_range(
    fog: float = 10000.0, rain: float = 0.0, time: str = "day"
) -> tuple[float, float]:
    return weather_range("visibility", _causes(fog, rain, time))


def visibility(
    fog: float = 10000.0, rain: float = 0.0, time: str = "day", quantile: float = 0.5
) -> float:
    return weather_at("visibility", _causes(fog, rain, time), quantile)


def _causes(fog: float, rain: float, time: str) -> dict:
    return {"fog_visibility_range": fog, "rainfall_rate": rain, "time_of_day": time}


@pytest.fixture(autouse=True)
def _forget() -> None:
    forget_visibility()


class TestVisibilityRange:
    def test_a_clear_day_sees_furthest(self) -> None:
        assert visibility_range() == pytest.approx((250.0, 400.0))
        assert visibility() == pytest.approx(325.0)

    def test_night_rain_and_fog_each_shorten_it(self) -> None:
        assert visibility_range(time="night") == pytest.approx((50.0, 150.0))
        assert visibility_range(rain=25.0) == pytest.approx((60.0, 150.0))
        assert visibility_range(fog=15.0) == pytest.approx((12.0, 18.0))

    def test_between_the_points_it_is_interpolated(self) -> None:
        assert visibility_range(fog=100.0) == pytest.approx((80.0, 120.0))
        assert visibility_range(time="dusk") == pytest.approx((150.0, 275.0))

    def test_the_shortest_bound_holds(self) -> None:
        # Light fog at night: the night's.
        assert visibility_range(fog=400.0, time="night") == pytest.approx((50.0, 150.0))
        assert visibility_range(fog=50.0, time="night") == pytest.approx((40.0, 60.0))

    def test_beyond_the_points_it_is_held(self) -> None:
        assert visibility_range(fog=5.0) == visibility_range(fog=15.0)
        assert visibility_range(rain=100.0) == visibility_range(rain=25.0)

    def test_the_quantile_picks_a_point_in_the_range(self) -> None:
        assert visibility(15.0, quantile=0.0) == pytest.approx(12.0)
        assert visibility(15.0, quantile=1.0) == pytest.approx(18.0)
        assert visibility(15.0, quantile=2.0) == pytest.approx(18.0)


class TestSightSpeed:
    def test_the_driver_stops_within_what_they_see(self) -> None:
        speed = sight_speed(50.0, decel=4.5, reaction_s=1.0)
        assert speed == pytest.approx(17.18, abs=0.01)
        assert speed * 1.0 + speed**2 / (2 * 4.5) == pytest.approx(50.0)

    def test_the_further_they_see_the_faster(self) -> None:
        speeds = [sight_speed(v, 4.5, 1.0) for v in (0.0, 15.0, 50.0, 150.0)]
        assert speeds[0] == 0.0
        assert speeds == sorted(speeds)

    def test_a_driver_who_cannot_brake_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            sight_speed(50.0, decel=0.0, reaction_s=1.0)


class TestSightSpeedAt:
    """The speed a driver keeps: the weather rules' sight_speed, which reads
    the visibility (a chain), never above the speed it stops from within."""

    def test_the_rules_give_it_below_the_stopping_speed(self) -> None:
        stop = sight_speed(50.0, 4.5, 1.0)
        # The rules' high end is the stopping speed (rounded to 0.1 m/s).
        assert sight_speed_at(50.0, 1.0) == pytest.approx(stop, abs=0.05)
        assert sight_speed_at(50.0, 0.0) == pytest.approx(0.75 * stop, abs=0.05)
        median = sight_speed_at(50.0)
        assert 0.75 * stop < median < stop
        assert sight_speed_at(50.0, 0.0) < median < sight_speed_at(50.0, 1.0)

    def test_a_car_that_brakes_harder_keeps_more_but_never_more_than_it_stops(
        self,
    ) -> None:
        soft = sight_speed_at(50.0, 0.5, decel=3.0, reaction_s=2.0)
        hard = sight_speed_at(50.0, 0.5, decel=7.5, reaction_s=0.5)
        assert soft < sight_speed_at(50.0, 0.5) < hard
        assert hard <= sight_speed(50.0, 7.5, 0.5)
        assert sight_speed_at(50.0, 1.0, decel=3.0, reaction_s=2.0) <= sight_speed(
            50.0, 3.0, 2.0
        )

    def test_the_rules_chain_from_the_conditions(self) -> None:
        from autoware_carla_scenario.weather_rules import default_rules

        rules = default_rules()
        assert rules.upstream("sight_speed") == ["visibility"]
        assert rules.order.index("visibility") < rules.order.index("sight_speed")
        # Thick fog: the visibility's median, then the speed kept in it.
        values = rules.sample({"fog_visibility_range": 15.0})
        assert values["visibility"] == pytest.approx(15.0)
        assert values["sight_speed"] == pytest.approx(
            sight_speed_at(values["visibility"])
        )


class TestDriverQuantile:
    def test_the_same_driver_draws_the_same_and_drivers_spread(self) -> None:
        assert driver_quantile("veh7") == driver_quantile("veh7")
        quantiles = [driver_quantile(str(i)) for i in range(1000)]
        assert all(0.0 < q < 1.0 for q in quantiles)
        assert sum(quantiles) / len(quantiles) == pytest.approx(0.5, abs=0.03)
        deciles = [
            sum(1 for q in quantiles if k / 10 <= q < (k + 1) / 10) for k in range(10)
        ]
        assert min(deciles) > 60
        # Keys a character apart land apart.
        assert abs(driver_quantile("1") - driver_quantile("2")) > 0.05


class TestWorldVisibility:
    def test_nothing_is_set_until_a_run_sets_it(self) -> None:
        assert world_visibility(1) is None

    def test_a_world_keeps_what_was_set_last(self) -> None:
        set_world_visibility(1, 80.0)
        set_world_visibility(1, 30.0)
        assert world_visibility(1) == 30.0

    def test_another_world_drops_the_last_ones(self) -> None:
        set_world_visibility(1, 80.0)
        set_world_visibility(2, 30.0)
        assert world_visibility(1) is None
        assert world_visibility(2) == 30.0

    def test_forgetting_clears_it(self) -> None:
        set_world_visibility(1, 80.0)
        forget_visibility()
        assert world_visibility(1) is None

    def test_no_visibility_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            set_world_visibility(1, 0.0)
