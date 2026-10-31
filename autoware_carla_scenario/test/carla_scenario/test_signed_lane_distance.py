"""Ahead of or behind an entity along the road: lane_offset and the signed
entity_distance it measures (a despawn behind the ego reads it)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    DistanceCoordinateSystem,
    EntityDistanceCondition,
)
from autoware_carla_scenario.coordinate import lane_distance
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose


def test_lane_offset_is_signed_by_the_way_the_source_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def gap(
        source: Any, travel_x: float, travel_y: float, target: Any, **kw: Any
    ) -> float:
        seen.append((travel_x, travel_y))
        return -42.0

    monkeypatch.setattr(lane_distance, "lane_gap", gap)
    standing = CarlaWorldPose(x=0.0, y=0.0, z=0.0, yaw=90.0)
    assert lane_distance.lane_offset(standing, CarlaWorldPose(0.0, -42.0, 0.0)) == -42.0
    ((x, y),) = seen
    assert x == pytest.approx(0.0, abs=1e-9) and y == pytest.approx(1.0)


def test_signed_is_a_lane_frame_distance() -> None:
    with pytest.raises(ValueError, match="lane frame"):
        EntityDistanceCondition("ego", "npc1", -30.0, signed=True, label="behind")


class _Actor:
    def __init__(self, x: float, yaw: float = 0.0) -> None:
        self.x, self.yaw = x, yaw

    def get_location(self) -> Any:
        return SimpleNamespace(x=self.x, y=0.0, z=0.0)

    def get_transform(self) -> Any:
        return SimpleNamespace(rotation=SimpleNamespace(yaw=self.yaw))


def test_a_target_far_enough_behind_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    import autoware_carla_scenario.conditions.composition.entity_distance as module

    actors = {"Ego": _Actor(0.0, yaw=0.0), "npc1": _Actor(-35.0)}
    monkeypatch.setattr(module, "find_actor_by_role_name", lambda w, n: actors[str(n)])
    # The target is looked up as present (not parked under the map, #922).
    monkeypatch.setattr(
        module, "find_present_actor_by_role_name", lambda w, n: actors[str(n)]
    )
    monkeypatch.setattr(
        module,
        "lane_offset",
        lambda source, target: target.x - source.x
        if math.cos(math.radians(source.yaw)) > 0
        else source.x - target.x,
    )
    behind = EntityDistanceCondition(
        "Ego",
        "npc1",
        -30.0,
        ComparisonRule.LESS_THAN_OR_EQUAL,
        coordinate_system=DistanceCoordinateSystem.LANE,
        signed=True,
        label="behind",
    )
    assert behind._measure(None) == -35.0  # type: ignore[arg-type]
    actors["npc1"] = _Actor(35.0)
    assert behind._measure(None) == 35.0  # type: ignore[arg-type]
