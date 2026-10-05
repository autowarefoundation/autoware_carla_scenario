"""Which way out of a junction a turn takes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.traffic import TurnDirection
from autoware_carla_scenario.traffic.traffic_manager import _pick_branch


def _wp(yaw: float) -> SimpleNamespace:
    return SimpleNamespace(transform=SimpleNamespace(rotation=SimpleNamespace(yaw=yaw)))


# Heading east (yaw 0); CARLA's yaw grows clockwise seen from above, so a
# left turn ends at -90 and a right turn at +90.
_BRANCHES = {
    "left": [_wp(0.0), _wp(-88.0)],
    "straight": [_wp(0.0), _wp(3.0)],
    "right": [_wp(0.0), _wp(91.0)],
}


@pytest.mark.parametrize("direction", list(TurnDirection))
def test_each_direction_takes_its_own_way_out(direction: TurnDirection) -> None:
    picked = _pick_branch(_wp(0.0), list(_BRANCHES.values()), direction)
    assert picked is _BRANCHES[direction.value]
