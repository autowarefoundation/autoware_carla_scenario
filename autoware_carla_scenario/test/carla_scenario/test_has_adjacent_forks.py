"""``has_adjacent``'s ``include_forks``: a turn pocket is its lane's neighbour (#1066).

A turn pocket that forks off the same predecessor as its through lane
overlaps that lane but shares no boundary with it, so the routing graph's
``left()`` / ``right()`` give neither a neighbour. On the Nishi-Shinjuku
fixture map, 160, 185, 213 and 556 are right-turn pockets beside 159, 184,
215 and 555; ``not has_adjacent: left`` took them for kerb lanes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.examples import run
from autoware_carla_scenario.sweeper.constraints import (
    HasAdjacentConstraint,
    create_routing_graph,
    find_matching_lanelets,
    parse_constraint,
)
from autoware_carla_scenario.sweeper.map_loader import load_lanelet2_map

_DATA = Path(__file__).resolve().parents[3] / "data"

#: Each right-turn pocket of the fixture map, with the through lane it forks
#: off beside (on the pocket's left: Nishi-Shinjuku drives on the left).
_POCKETS = {160: 159, 185: 184, 213: 215, 556: 555}


@pytest.fixture(scope="module")
def nishishinjuku() -> tuple[Any, Any]:
    lanelet_map = load_lanelet2_map(
        _DATA / "nishishinjuku.osm", _DATA / "nishishinjuku_carla.xodr"
    )
    return lanelet_map, create_routing_graph(lanelet_map)


def _ids(nishishinjuku: tuple[Any, Any], side: str, include_forks: bool) -> set[int]:
    lanelet_map, graph = nishishinjuku
    return HasAdjacentConstraint(side, include_forks=include_forks).find_matching_ids(
        lanelet_map, graph
    )


def test_the_routing_graph_gives_a_pocket_no_neighbour(
    nishishinjuku: tuple[Any, Any],
) -> None:
    lanelet_map, graph = nishishinjuku
    for pocket, lane in _POCKETS.items():
        assert graph.left(lanelet_map.laneletLayer[pocket]) is None
        assert graph.right(lanelet_map.laneletLayer[lane]) is None
    # So by default the pockets have no lane on their left.
    assert not set(_POCKETS) & _ids(nishishinjuku, "left", False)


def test_a_pocket_has_its_through_lane_on_its_left(
    nishishinjuku: tuple[Any, Any],
) -> None:
    left = _ids(nishishinjuku, "left", True)
    right = _ids(nishishinjuku, "right", True)
    assert 160 in left
    assert set(_POCKETS) <= left
    # And the through lane has the pocket on its right.
    assert set(_POCKETS.values()) <= right
    # The side is measured: a pocket has nothing on its right.
    assert not set(_POCKETS) & right


@pytest.mark.parametrize("side", ["left", "right"])
def test_forks_only_add_lanelets_the_routing_graph_gave_none(
    nishishinjuku: tuple[Any, Any], side: str
) -> None:
    lanelet_map, graph = nishishinjuku
    before = _ids(nishishinjuku, side, False)
    after = _ids(nishishinjuku, side, True)
    assert before <= after
    for added in after - before:
        lanelet = lanelet_map.laneletLayer[added]
        beside = graph.left(lanelet) if side == "left" else graph.right(lanelet)
        assert beside is None


def test_the_existing_adjacency_is_unchanged(nishishinjuku: tuple[Any, Any]) -> None:
    left = _ids(nishishinjuku, "left", False)
    right = _ids(nishishinjuku, "right", False)
    # 159, 184, 215 and 555 have a lane to change into on their left.
    assert set(_POCKETS.values()) <= left
    assert not set(_POCKETS.values()) & right
    lanelet_map, graph = nishishinjuku
    for side, ids in (("left", left), ("right", right)):
        expected = set()
        for lanelet in lanelet_map.laneletLayer:
            beside = graph.left(lanelet) if side == "left" else graph.right(lanelet)
            if beside is not None:
                expected.add(lanelet.id)
        assert ids == expected


def test_include_forks_parses_and_is_off_by_default() -> None:
    assert HasAdjacentConstraint("left").include_forks is False
    parsed = parse_constraint(
        {"type": "has_adjacent", "value": "left", "include_forks": True}
    )
    assert parsed == HasAdjacentConstraint("left", include_forks=True)


def test_pedestrian_dart_out_sweeps_no_pocket(nishishinjuku: tuple[Any, Any]) -> None:
    lanelet_map, graph = nishishinjuku
    cfg = run._compose_config("pedestrian_dart_out/pedestrian_dart_out", [])
    raw = OmegaConf.to_container(
        cfg.sweep.constraints["ego.spawn_lanelet_id"], resolve=True
    )
    assert isinstance(raw, list)
    matched = find_matching_lanelets(
        [parse_constraint(c) for c in raw], lanelet_map, graph
    )
    assert matched
    assert not set(_POCKETS) & set(matched)
    # The fixed case, a kerb lane, is still swept.
    assert int(cfg.ego.spawn_lanelet_id) in matched
