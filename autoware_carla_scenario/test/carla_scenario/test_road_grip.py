"""The road's grip as a quantity of the run, which the traffic backends read."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from autoware_carla_scenario.road_grip import (
    forget_grip,
    grip_braking,
    set_world_grip,
    world_grip,
)


@pytest.fixture(autouse=True)
def _forget() -> Iterator[None]:
    forget_grip()
    yield
    forget_grip()


def test_a_grip_is_recorded_for_its_world_alone() -> None:
    assert world_grip(3) is None
    set_world_grip(3, 0.6)
    assert world_grip(3) == 0.6
    assert world_grip(None) is None
    # Another episode: the last one is not come back to.
    set_world_grip(4, 0.8)
    assert world_grip(3) is None and world_grip(4) == 0.8
    forget_grip()
    assert world_grip(4) is None


def test_no_grip_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        set_world_grip(1, 0.0)


def test_braking_scales_down_with_less_grip_and_never_up() -> None:
    assert grip_braking(4.5, None) == 4.5
    assert grip_braking(4.5, 0.5) == pytest.approx(2.25)
    assert grip_braking(4.5, 1.1) == 4.5
