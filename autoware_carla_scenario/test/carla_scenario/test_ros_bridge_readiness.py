"""Unit tests for the ROS-free readiness aggregation logic."""

from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
    LOCALIZATION_STATE_INITIALIZED,
)
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
    OPERATION_MODE_AUTONOMOUS,
)
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import ROUTE_STATE_SET
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import ROUTE_STATE_UNSET
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
    ReadinessAggregator,
)


def _initialized_and_routed() -> ReadinessAggregator:
    agg = ReadinessAggregator()
    agg.update_localization(LOCALIZATION_STATE_INITIALIZED)
    agg.update_routing(ROUTE_STATE_SET)
    return agg


def test_a_route_is_only_offered_in_the_state_the_ad_api_takes_one():
    """``RoutingNode::on_set_route_points`` refuses every state but UNSET.

    Including its own starting UNKNOWN, which it answers with "The route is
    already set." -- so a bridge that asks before the mission planner has
    published a RouteState is refused against a stack that has no route.
    """
    agg = ReadinessAggregator()
    assert agg.route_acceptable is False  # UNKNOWN: the planner is not up yet
    agg.update_routing(ROUTE_STATE_SET)
    assert agg.route_acceptable is False
    agg.update_routing(ROUTE_STATE_UNSET)
    assert agg.route_acceptable is True


def test_not_ready_or_engageable_when_empty():
    agg = ReadinessAggregator()
    assert agg.can_engage is False
    assert agg.ready is False


def test_can_engage_needs_init_route_and_availability():
    agg = _initialized_and_routed()
    assert agg.can_engage is False  # autonomous not available yet

    agg.update_operation_mode(
        mode=1, is_control_enabled=False, is_autonomous_available=True
    )
    assert agg.can_engage is True
    assert agg.ready is False  # available != engaged


def test_ready_needs_autonomous_engaged_with_control():
    agg = _initialized_and_routed()
    # Autonomous mode but control not yet enabled -> not ready.
    agg.update_operation_mode(
        mode=OPERATION_MODE_AUTONOMOUS,
        is_control_enabled=False,
        is_autonomous_available=True,
    )
    assert agg.ready is False

    agg.update_operation_mode(
        mode=OPERATION_MODE_AUTONOMOUS,
        is_control_enabled=True,
        is_autonomous_available=True,
    )
    assert agg.ready is True


def test_missing_localization_blocks_ready():
    agg = ReadinessAggregator()
    agg.update_routing(ROUTE_STATE_SET)
    agg.update_operation_mode(
        mode=OPERATION_MODE_AUTONOMOUS,
        is_control_enabled=True,
        is_autonomous_available=True,
    )
    assert agg.ready is False
    assert agg.can_engage is False


def test_regressed_localization_clears_flags():
    agg = _initialized_and_routed()
    agg.update_operation_mode(
        mode=OPERATION_MODE_AUTONOMOUS,
        is_control_enabled=True,
        is_autonomous_available=True,
    )
    assert agg.ready is True

    agg.update_localization(2)  # INITIALIZING
    assert agg.ready is False


def test_localization_not_required_for_ground_truth_stacks():
    # localization:=false / E2E: readiness must not need INITIALIZED localization.
    agg = ReadinessAggregator(require_localization=False)
    agg.update_routing(ROUTE_STATE_SET)
    agg.update_operation_mode(
        mode=OPERATION_MODE_AUTONOMOUS,
        is_control_enabled=True,
        is_autonomous_available=True,
    )
    assert agg.localization_initialized is False  # never initialized
    assert agg.can_engage is True
    assert agg.ready is True


def test_the_enum_values_are_checked_against_the_ad_api() -> None:
    import pytest

    from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
        ENUM_VALUES,
        check_enum_values,
    )

    check_enum_values(dict(ENUM_VALUES))

    with pytest.raises(RuntimeError, match=r"RouteState.SET = 5 \(expected 2\)"):
        check_enum_values({**ENUM_VALUES, "RouteState.SET": 5})
    with pytest.raises(RuntimeError, match="RouteState.UNSET is missing"):
        check_enum_values(
            {k: v for k, v in ENUM_VALUES.items() if k != "RouteState.UNSET"}
        )


def test_the_package_imports_without_ros() -> None:
    """Only the node needs rclpy; the rest runs on the host for these tests."""
    import sys

    import autoware_carla_scenario.autoware_bridge.ros_bridge  # noqa: F401
    import autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api  # noqa: F401
    import autoware_carla_scenario.autoware_bridge.ros_bridge.client  # noqa: F401

    assert "rclpy" not in sys.modules
