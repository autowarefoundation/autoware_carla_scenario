"""Which OpenDRIVE lane a lane change ends on."""

from __future__ import annotations

import pytest

from autoware_carla_scenario.conditions import beside_lane_id


@pytest.mark.parametrize(
    ("lane_id", "side", "target"),
    [
        # Right of the reference line, driving along it: the centre is up.
        (-2, "left", -1),
        (-1, "right", -2),
        # Left of it, driving against it: the centre is down.
        (2, "left", 1),
        (1, "right", 2),
    ],
)
def test_a_change_ends_one_lane_over_towards_the_side_asked_for(
    lane_id: int, side: str, target: int
) -> None:
    assert beside_lane_id(lane_id, side) == target


def test_only_left_and_right_are_sides() -> None:
    with pytest.raises(ValueError):
        beside_lane_id(-1, "same")
