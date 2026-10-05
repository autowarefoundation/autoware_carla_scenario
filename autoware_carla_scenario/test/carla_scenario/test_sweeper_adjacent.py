"""The lanelet beside a pick, for a vehicle that drives alongside the ego."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.sweeper.bindings import AdjacentBinding, parse_binding


class _Graph:
    """A routing graph that knows only who is beside whom."""

    def __init__(self, left: dict[int, int], right: dict[int, int]) -> None:
        self._left, self._right = left, right

    def left(self, lanelet: SimpleNamespace) -> SimpleNamespace | None:
        found = self._left.get(lanelet.id)
        return None if found is None else SimpleNamespace(id=found)

    def right(self, lanelet: SimpleNamespace) -> SimpleNamespace | None:
        found = self._right.get(lanelet.id)
        return None if found is None else SimpleNamespace(id=found)


_MAP = SimpleNamespace(laneletLayer={i: SimpleNamespace(id=i) for i in (1, 2, 3)})
_GRAPH = _Graph(left={2: 1}, right={2: 3})


@pytest.mark.parametrize(("side", "beside"), [("left", 1), ("right", 3)])
def test_it_returns_the_lanelet_on_that_side(side: str, beside: int) -> None:
    result = AdjacentBinding(target_key="scenario.npc_lanelet_id", side=side).resolve(
        2, _MAP, _GRAPH
    )
    assert result.value == beside
    # It names another vehicle's lanelet, not where the ego starts.
    assert result.lanelet_id_override is None


def test_no_lane_on_that_side_raises_so_the_case_is_dropped() -> None:
    with pytest.raises(ValueError, match="no lane to change into on its left"):
        AdjacentBinding(target_key="k", side="left").resolve(1, _MAP, _GRAPH)


def test_it_is_reachable_from_the_yaml_form() -> None:
    binding = parse_binding(
        "scenario.npc_lanelet_id", {"type": "adjacent", "side": "right"}
    )
    assert binding == AdjacentBinding(
        target_key="scenario.npc_lanelet_id", side="right"
    )


def test_an_unknown_side_is_refused() -> None:
    with pytest.raises(ValueError, match="'left' or 'right'"):
        AdjacentBinding(target_key="k", side="up")
