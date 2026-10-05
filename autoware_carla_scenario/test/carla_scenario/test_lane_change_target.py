"""Which OpenDRIVE lane a lane change ends on."""

from __future__ import annotations

import pytest

from autoware_carla_scenario import LaneChangeDirection
from autoware_carla_scenario.examples.lane_change import _target_lane_id


@pytest.mark.parametrize(
    ("lane_id", "direction", "target"),
    [
        # Right of the reference line, driving along it: the centre is up.
        (-2, LaneChangeDirection.LEFT, -1),
        (-1, LaneChangeDirection.RIGHT, -2),
        # Left of it, driving against it: the centre is down.
        (2, LaneChangeDirection.LEFT, 1),
        (1, LaneChangeDirection.RIGHT, 2),
    ],
)
def test_a_change_ends_one_lane_over_towards_the_side_asked_for(
    lane_id: int, direction: LaneChangeDirection, target: int
) -> None:
    assert _target_lane_id(lane_id, direction) == target
