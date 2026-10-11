"""Unit tests for the policy warm-up's pedestrian run-up."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.entity._policy_warmup import _CarriedWalker


def _walker(yaw: float) -> MagicMock:
    actor = MagicMock()
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x=10.0, y=20.0, z=1.0), carla.Rotation(yaw=yaw)
    )
    return actor


@pytest.mark.parametrize("yaw", [0.0, 30.0, 90.0, -135.0])
def test_a_walker_is_moved_back_along_its_heading(yaw: float) -> None:
    """At tick t it stands (ticks - t) steps of its velocity behind its pose."""
    actor = _walker(yaw)
    walker = _CarriedWalker(actor, 1.5, ticks=10, step_s=0.05)
    heading = math.radians(yaw)
    for tick in (0, 4, 10):
        walker.place(tick)
        placed = actor.set_transform.call_args.args[0]
        back_s = (10 - tick) * 0.05
        assert placed.location.x == pytest.approx(
            10.0 - 1.5 * math.cos(heading) * back_s
        )
        assert placed.location.y == pytest.approx(
            20.0 - 1.5 * math.sin(heading) * back_s
        )
        assert placed.location.z == pytest.approx(1.0)
        assert placed.rotation.yaw == pytest.approx(yaw)
