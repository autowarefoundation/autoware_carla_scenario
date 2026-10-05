"""Picking a map's right turns, and reading each one's route off the graph.

What a logical right-turn scenario needs that a fixed one does not: a
constraint that says which way a junction lanelet turns, and a binding that
turns the pick into the lanelets the case drives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.sweeper.bindings import RouteThroughBinding
from autoware_carla_scenario.sweeper.constraints import (
    TurnDirectionConstraint,
    parse_constraint,
)
from autoware_carla_scenario.sweeper.expand import _override_value


def _lanelet(lanelet_id: int, **attributes: str) -> SimpleNamespace:
    """A stand-in for a lanelet2 lanelet: an ID and a tag map."""
    return SimpleNamespace(id=lanelet_id, attributes=dict(attributes))


class _Graph:
    """A routing graph over a fixed ``id -> following ids`` mapping."""

    def __init__(self, layer: dict[int, SimpleNamespace], edges: dict[int, list[int]]):
        self._layer = layer
        self._edges = edges

    def following(self, lanelet: SimpleNamespace) -> list[SimpleNamespace]:
        return [self._layer[i] for i in self._edges.get(lanelet.id, [])]


def _map(edges: dict[int, list[int]]) -> SimpleNamespace:
    ids = set(edges) | {i for following in edges.values() for i in following}
    layer = {i: _lanelet(i) for i in ids}
    return SimpleNamespace(laneletLayer=layer, _edges=edges)


class TestTurnDirectionConstraint:
    def test_it_matches_only_the_direction_it_names(self) -> None:
        right = TurnDirectionConstraint(value="right")
        assert right.evaluate(_lanelet(1, turn_direction="right"))
        assert not right.evaluate(_lanelet(2, turn_direction="left"))
        assert not right.evaluate(_lanelet(3, turn_direction="straight"))

    def test_a_lanelet_outside_a_junction_never_matches(self) -> None:
        # No turn_direction tag at all: an ordinary lane, not a manoeuvre.
        assert not TurnDirectionConstraint(value="right").evaluate(_lanelet(1))

    def test_an_unknown_direction_is_refused_when_it_is_written(self) -> None:
        # Caught at parse time, not silently matching nothing on every lanelet.
        with pytest.raises(ValueError, match="Unknown turn_direction"):
            TurnDirectionConstraint(value="sharp_right")

    def test_it_is_reachable_from_the_yaml_form(self) -> None:
        parsed = parse_constraint({"type": "turn_direction", "value": "right"})
        assert parsed == TurnDirectionConstraint(value="right")


class TestRouteThroughBinding:
    def test_it_returns_the_pick_and_what_follows_it(self) -> None:
        lanelet_map = _map({10: [20], 20: [30]})
        graph = _Graph(lanelet_map.laneletLayer, lanelet_map._edges)

        result = RouteThroughBinding(
            target_key="scenario.expected_route_lanelet_ids"
        ).resolve(10, lanelet_map, graph)

        assert result.value == [10, 20]
        # It describes the route, not where the ego starts.
        assert result.lanelet_id_override is None

    def test_depth_walks_further_down_the_graph(self) -> None:
        lanelet_map = _map({10: [20], 20: [30], 30: [40]})
        graph = _Graph(lanelet_map.laneletLayer, lanelet_map._edges)

        result = RouteThroughBinding(target_key="k", depth=3).resolve(
            10, lanelet_map, graph
        )

        assert result.value == [10, 20, 30, 40]

    def test_a_fork_resolves_the_same_way_every_time(self) -> None:
        lanelet_map = _map({10: [31, 22], 22: [], 31: []})
        graph = _Graph(lanelet_map.laneletLayer, lanelet_map._edges)

        result = RouteThroughBinding(target_key="k").resolve(10, lanelet_map, graph)

        # Lowest ID, whatever order the graph hands them back in -- the same
        # map has to expand to the same cases.
        assert result.value == [10, 22]

    def test_a_dead_end_raises_so_the_case_is_dropped(self) -> None:
        lanelet_map = _map({10: []})
        graph = _Graph(lanelet_map.laneletLayer, lanelet_map._edges)

        with pytest.raises(ValueError, match="no following"):
            RouteThroughBinding(target_key="k").resolve(10, lanelet_map, graph)

    def test_depth_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="depth must be >= 1"):
            RouteThroughBinding(target_key="k", depth=0)

    def test_last_only_gives_the_end_of_the_route_as_an_id(self) -> None:
        lanelet_map = _map({10: [20], 20: [30]})
        graph = _Graph(lanelet_map.laneletLayer, lanelet_map._edges)

        result = RouteThroughBinding(
            target_key="ego.goal_lanelet_id", depth=2, last_only=True
        ).resolve(10, lanelet_map, graph)

        assert result.value == 30


class TestOverrideValue:
    def test_a_list_renders_as_a_hydra_list_without_spaces(self) -> None:
        assert _override_value([502, 193]) == "[502,193]"

    def test_an_int_keeps_its_type(self) -> None:
        # ego.goal_lanelet_id does not take 1234.0.
        assert _override_value(1234) == "1234"

    def test_a_float_is_left_alone(self) -> None:
        assert _override_value(18.6) == "18.6"

    def test_a_bool_renders_the_way_hydra_reads_it(self) -> None:
        assert _override_value(True) == "true"
