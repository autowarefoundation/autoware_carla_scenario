"""Conditions refuse a non-finite threshold, tolerance or size (#1327).

NaN fails every comparison, so ``x < 0`` lets it through and a condition
built with it never fires: a fail condition then never fails the run.
"""

from __future__ import annotations

from typing import Callable

import pytest

from autoware_carla_scenario.conditions import (
    AccelerationCondition,
    ComparisonRule,
    ElapsedTimeCondition,
    EntityDistanceCondition,
    EntityFilter,
    EntityPositionDistanceCondition,
    MetricCondition,
    PlaceClearCondition,
    RelativeSpeedCondition,
    RouteProgressCondition,
    ScalarComparisonRule,
    SpeedCondition,
    StandstillCondition,
    TemporaryStopCondition,
    TimeHeadwayCondition,
    TimeToCollisionCondition,
    WaypointCondition,
)
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, OpenDrivePose

NOT_FINITE = [float("nan"), float("inf"), float("-inf")]
_AT = CarlaWorldPose(x=0.0, y=0.0, z=0.0)
_METRIC = "min_distance_to_obstacle_m"


@pytest.mark.parametrize("value", NOT_FINITE)
def test_scalar_rule_refuses_a_value_that_is_not_finite(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        ScalarComparisonRule("s", ComparisonRule.GREATER_THAN, value)


@pytest.mark.parametrize("tolerance", [*NOT_FINITE, -1.0])
def test_scalar_rule_refuses_a_bad_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be non-negative and finite"):
        ScalarComparisonRule("s", ComparisonRule.EQUAL_TO, 1.0, tolerance)


def test_scalar_rule_takes_a_finite_value_and_tolerance() -> None:
    rule = ScalarComparisonRule("s", ComparisonRule.EQUAL_TO, 1.0, 0.0)
    assert rule.satisfied(1.0)


#: Each condition built with ``**kwargs`` for the parameter under test.
_TOLERANCE: dict[str, Callable[..., object]] = {
    "speed": lambda **kw: SpeedCondition(
        "ego", value=5.0, rule=ComparisonRule.LESS_THAN, label="t", **kw
    ),
    "acceleration": lambda **kw: AccelerationCondition("ego", 1.0, label="t", **kw),
    "relative_speed": lambda **kw: RelativeSpeedCondition(
        "ego", "npc", 1.0, label="t", **kw
    ),
    "entity_distance": lambda **kw: EntityDistanceCondition(
        "ego", "npc", 5.0, label="t", **kw
    ),
    "entity_position_distance": lambda **kw: EntityPositionDistanceCondition(
        "ego", _AT, 5.0, label="t", **kw
    ),
    "time_headway": lambda **kw: TimeHeadwayCondition(
        "ego", "npc", 2.0, label="t", **kw
    ),
    "time_to_collision": lambda **kw: TimeToCollisionCondition(
        "ego", "npc", 4.0, label="t", **kw
    ),
    "elapsed_time": lambda **kw: ElapsedTimeCondition(10.0, label="t", **kw),
    "route_progress": lambda **kw: RouteProgressCondition(label="t", **kw),
    "metric": lambda **kw: MetricCondition(_METRIC, label="t", **kw),
}


@pytest.mark.parametrize("tolerance", NOT_FINITE)
@pytest.mark.parametrize("name", sorted(_TOLERANCE))
def test_refuses_a_tolerance_that_is_not_finite(name: str, tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be non-negative and finite"):
        _TOLERANCE[name](tolerance=tolerance)


_VALUE: dict[str, Callable[[float], object]] = {
    "speed": lambda v: SpeedCondition(
        "ego", value=v, rule=ComparisonRule.LESS_THAN, label="t"
    ),
    "acceleration": lambda v: AccelerationCondition("ego", v, label="t"),
    "relative_speed": lambda v: RelativeSpeedCondition("ego", "npc", v, label="t"),
    "entity_distance": lambda v: EntityDistanceCondition("ego", "npc", v, label="t"),
    "entity_position_distance": lambda v: EntityPositionDistanceCondition(
        "ego", _AT, v, label="t"
    ),
    "time_headway": lambda v: TimeHeadwayCondition("ego", "npc", v, label="t"),
    "time_to_collision": lambda v: TimeToCollisionCondition("ego", "npc", v, label="t"),
    "route_progress": lambda v: RouteProgressCondition(value=v, label="t"),
    "metric": lambda v: MetricCondition(_METRIC, value=v, label="t"),
}


@pytest.mark.parametrize("value", NOT_FINITE)
@pytest.mark.parametrize("name", sorted(_VALUE))
def test_refuses_a_value_that_is_not_finite(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        _VALUE[name](value)


@pytest.mark.parametrize("seconds", NOT_FINITE)
def test_metric_refuses_max_ng_seconds_that_is_not_finite(seconds: float) -> None:
    """A NaN grace never runs out, so the metric never failed the run."""
    with pytest.raises(ValueError, match="max_ng_seconds must be non-negative and"):
        MetricCondition(
            _METRIC, ComparisonRule.LESS_THAN, 1.0, max_ng_seconds=seconds, label="t"
        )


@pytest.mark.parametrize("threshold", NOT_FINITE)
def test_standstill_refuses_a_speed_threshold_that_is_not_finite(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="speed_threshold must be non-negative and"):
        StandstillCondition("ego", 1.0, speed_threshold=threshold, label="t")


@pytest.mark.parametrize("parameter", ["s_margin", "speed_threshold"])
@pytest.mark.parametrize("number", NOT_FINITE)
def test_temporary_stop_refuses_a_parameter_that_is_not_finite(
    parameter: str, number: float
) -> None:
    od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
    with pytest.raises(ValueError, match=f"{parameter} must be .* and finite"):
        TemporaryStopCondition(
            "ego", stop_positions=[od], label="t", **{parameter: number}
        )


@pytest.mark.parametrize("distance", NOT_FINITE)
def test_waypoint_refuses_a_distance_that_is_not_finite(distance: float) -> None:
    with pytest.raises(ValueError, match="distance must be positive and finite"):
        WaypointCondition("ego", distance, label="t")


@pytest.mark.parametrize(
    ("parameter", "wording"),
    [
        ("length", "positive"),
        ("width", "positive"),
        ("margin", "non-negative"),
    ],
)
@pytest.mark.parametrize("number", NOT_FINITE)
def test_place_clear_refuses_a_size_that_is_not_finite(
    parameter: str, wording: str, number: float
) -> None:
    kwargs: dict[str, float] = {"length": 4.0, "width": 2.0, "margin": 0.3}
    kwargs[parameter] = number
    with pytest.raises(ValueError, match=f"{parameter} must be {wording} and finite"):
        PlaceClearCondition(_AT, label="t", **kwargs)


@pytest.mark.parametrize(
    "bound",
    [
        "min_distance_m",
        "max_distance_m",
        "min_ahead_m",
        "max_ahead_m",
        "min_left_m",
        "max_left_m",
    ],
)
@pytest.mark.parametrize("number", NOT_FINITE)
def test_entity_filter_refuses_a_bound_that_is_not_finite(
    bound: str, number: float
) -> None:
    """An absent bound is None; NaN slipped past `< 0` and `low > high`.

    A negative infinity distance is refused as negative, which it also is.
    """
    entities = EntityFilter(**{bound: number})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=f"{bound} must (be finite|not be neg)"):
        entities.validate()
