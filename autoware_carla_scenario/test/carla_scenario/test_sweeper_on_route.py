"""``on_route``: a scenario tried only along a route, or anywhere when it has none."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from autoware_carla_scenario.sweeper.constraints import (
    AndConstraint,
    EqualsConstraint,
    NotConstraint,
    OnRouteConstraint,
    find_matching_lanelets,
    parse_constraint,
)


class _Layer:
    """A lanelet layer: iterated over, and looked up by id."""

    def __init__(self, ids: list[int]) -> None:
        self._lanelets = {i: SimpleNamespace(id=i) for i in ids}

    def __iter__(self) -> Any:
        return iter(self._lanelets.values())

    def __getitem__(self, lanelet_id: int) -> SimpleNamespace:
        return self._lanelets[lanelet_id]


_MAP: Any = SimpleNamespace(laneletLayer=_Layer([1, 2, 3, 4, 5]))


def _match(*constraints: Any) -> list[int]:
    return find_matching_lanelets(list(constraints), _MAP)


def test_only_the_routes_lanelets_match() -> None:
    assert _match(OnRouteConstraint((2, 4, 9))) == [2, 4]


def test_no_route_is_no_constraint() -> None:
    assert _match(OnRouteConstraint()) == [1, 2, 3, 4, 5]
    assert _match(OnRouteConstraint(()), NotConstraint(EqualsConstraint(3))) == [
        1,
        2,
        4,
        5,
    ]


def test_it_narrows_what_the_other_constraints_match() -> None:
    route = OnRouteConstraint((1, 2, 3))
    assert _match(AndConstraint((route, NotConstraint(EqualsConstraint(2))))) == [
        1,
        3,
    ]


def test_parsed_from_a_scenario_whose_route_is_left_empty() -> None:
    """The scenario names its route by reference and leaves it empty; a run gives one."""
    scenario = OmegaConf.create(
        {
            "sweep": {
                "route_lanelet_ids": [],
                "constraints": {
                    "ego.spawn_lanelet_id": [
                        {
                            "type": "on_route",
                            "lanelet_ids": "${sweep.route_lanelet_ids}",
                        }
                    ]
                },
            }
        }
    )

    def constraint() -> Any:
        sweep = OmegaConf.to_container(scenario.sweep, resolve=True)
        assert isinstance(sweep, dict)
        return parse_constraint(sweep["constraints"]["ego.spawn_lanelet_id"][0])

    assert _match(constraint()) == [1, 2, 3, 4, 5]
    scenario.sweep.route_lanelet_ids = [5, 3]
    assert constraint() == OnRouteConstraint((5, 3))
    assert _match(constraint()) == [3, 5]
