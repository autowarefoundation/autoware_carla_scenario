"""How far drivers see: the range the weather leaves, the speed it allows,
and the record a traffic backend reads it from."""

from __future__ import annotations

import pytest

from autoware_carla_scenario.visibility import (
    forget_visibility,
    set_world_visibility,
    sight_speed,
    visibility,
    visibility_range,
    world_visibility,
)


@pytest.fixture(autouse=True)
def _forget() -> None:
    forget_visibility()


class TestVisibilityRange:
    def test_a_clear_day_sees_furthest(self) -> None:
        assert visibility_range() == pytest.approx((250.0, 400.0))
        assert visibility() == pytest.approx(325.0)

    def test_night_rain_and_fog_each_shorten_it(self) -> None:
        assert visibility_range(sun_altitude_angle=-30.0) == pytest.approx(
            (50.0, 150.0)
        )
        assert visibility_range(precipitation=100.0) == pytest.approx((60.0, 150.0))
        assert visibility_range(fog_density=100.0) == pytest.approx((15.0, 40.0))

    def test_between_the_rows_it_is_interpolated(self) -> None:
        assert visibility_range(fog_density=50.0) == pytest.approx((80.0, 175.0))
        # Halfway through civil twilight.
        assert visibility_range(sun_altitude_angle=-3.0) == pytest.approx(
            (150.0, 275.0)
        )

    def test_the_shortest_bound_holds(self) -> None:
        # Light fog at night: the night's low end, the fog's high end.
        assert visibility_range(
            fog_density=30.0, sun_altitude_angle=-30.0
        ) == pytest.approx((50.0, 150.0))
        assert visibility_range(
            fog_density=70.0, sun_altitude_angle=-30.0
        ) == pytest.approx((40.0, 100.0))

    def test_out_of_range_weather_is_clamped(self) -> None:
        assert visibility_range(fog_density=150.0) == visibility_range(100.0)
        assert visibility_range(fog_density=-5.0) == visibility_range()

    def test_the_quantile_picks_a_point_in_the_range(self) -> None:
        assert visibility(100.0, 0.0, 90.0, 0.0) == pytest.approx(15.0)
        assert visibility(100.0, 0.0, 90.0, 1.0) == pytest.approx(40.0)
        assert visibility(100.0, 0.0, 90.0, 2.0) == pytest.approx(40.0)


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
