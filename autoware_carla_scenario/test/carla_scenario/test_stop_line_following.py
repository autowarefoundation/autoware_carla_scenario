"""Finding a stop line some lanelets ahead of where a vehicle starts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from lanelet2.core import LaneletMap

lanelet2 = pytest.importorskip("lanelet2")

from autoware_carla_scenario.utils.stop_line import (  # noqa: E402
    get_stop_line_linestrings_with_following,
)


def _road_with_stop_line_on(
    index: int, count: int = 3
) -> tuple[LaneletMap, list[int], int]:
    """*count* 10 m lanelets end to end, a traffic light's stop line at the
    start of lanelet *index*."""
    from lanelet2.core import (
        AttributeMap,
        Lanelet,
        LaneletMap,
        LineString3d,
        Point3d,
        TrafficLight,
        getId,
    )

    lanelet_map = LaneletMap()
    left = [Point3d(getId(), 10.0 * i, 1.75, 0.0) for i in range(count + 1)]
    right = [Point3d(getId(), 10.0 * i, -1.75, 0.0) for i in range(count + 1)]
    lanelets = [
        Lanelet(
            getId(),
            LineString3d(getId(), left[i : i + 2]),
            LineString3d(getId(), right[i : i + 2]),
        )
        for i in range(count)
    ]
    stop_line = LineString3d(
        getId(), [left[index], right[index]], AttributeMap({"type": "stop_line"})
    )
    light = LineString3d(
        getId(),
        [
            Point3d(getId(), 10.0 * index + 5, 1.0, 5.0),
            Point3d(getId(), 10.0 * index + 5, -1.0, 5.0),
        ],
        AttributeMap({"type": "traffic_light"}),
    )
    regulatory_element = TrafficLight(getId(), AttributeMap(), [light], stop_line)
    lanelets[index].addRegulatoryElement(regulatory_element)
    for lanelet in lanelets:
        lanelet_map.add(lanelet)
    return lanelet_map, [lanelet.id for lanelet in lanelets], stop_line.id


def test_one_step_finds_a_stop_line_on_the_next_lanelet() -> None:
    lanelet_map, ids, stop_line = _road_with_stop_line_on(1)
    found = get_stop_line_linestrings_with_following(lanelet_map, ids[0])
    assert [(owner, line.id) for owner, line in found] == [(ids[1], stop_line)]


def test_a_stop_line_two_lanelets_ahead_needs_depth_two() -> None:
    lanelet_map, ids, stop_line = _road_with_stop_line_on(2)
    assert get_stop_line_linestrings_with_following(lanelet_map, ids[0]) == []
    found = get_stop_line_linestrings_with_following(lanelet_map, ids[0], depth=2)
    assert [(owner, line.id) for owner, line in found] == [(ids[2], stop_line)]


# ---------------------------------------------------------------------------
# A lanelet the map does not hold (issue #1172)
#
# lanelet2 raises a RuntimeError ("Primitive with id ... does not exist") for
# a missing id, not a KeyError: the documented ValueError is raised by asking
# the layer first.
# ---------------------------------------------------------------------------

_MISSING = 987654321


def _map_manager(lanelet_map: LaneletMap, routing_graph: object) -> object:
    """A stand-in for the initialised MapManager, holding *lanelet_map*."""
    from types import SimpleNamespace

    return SimpleNamespace(lanelet_map=lanelet_map, routing_graph=routing_graph)


def test_a_missing_lanelet_raises_value_error() -> None:
    from autoware_carla_scenario.utils.stop_line import get_stop_line_linestrings

    lanelet_map, _, _ = _road_with_stop_line_on(1)
    with pytest.raises(ValueError, match=f"Lanelet ID {_MISSING} not found"):
        get_stop_line_linestrings(lanelet_map, _MISSING)


def test_a_missing_lanelet_raises_value_error_with_following() -> None:
    lanelet_map, _, _ = _road_with_stop_line_on(1)
    with pytest.raises(ValueError, match=f"Lanelet ID {_MISSING} not found"):
        get_stop_line_linestrings_with_following(lanelet_map, _MISSING)


@pytest.mark.parametrize(
    "wrapper",
    [
        "get_stop_line_linestrings",
        "get_stop_line_linestrings_with_following",
        "get_stop_line_poses",
        "get_stop_line_poses_with_following",
    ],
)
def test_the_wrappers_raise_value_error_for_a_missing_lanelet(
    wrapper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from autoware_carla_scenario.coordinate import stop_line
    from autoware_carla_scenario.coordinate.map_manager import MapManager

    lanelet_map, _, _ = _road_with_stop_line_on(1)
    mm = _map_manager(lanelet_map, MagicMock())
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    with pytest.raises(ValueError, match=f"Lanelet ID {_MISSING} not found"):
        getattr(stop_line, wrapper)(_MISSING)


# ---------------------------------------------------------------------------
# A routing graph the caller holds
# ---------------------------------------------------------------------------


def _no_graph_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make building a routing graph fail the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a routing graph was built")

    monkeypatch.setattr(lanelet2.routing, "RoutingGraph", refuse)


def test_a_given_routing_graph_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoware_carla_scenario.coordinate.map_manager import build_routing_graph

    lanelet_map, ids, stop_line = _road_with_stop_line_on(1)
    graph = build_routing_graph(
        lanelet_map, lanelet2.traffic_rules.Participants.Vehicle
    )
    _no_graph_built(monkeypatch)
    found = get_stop_line_linestrings_with_following(
        lanelet_map, ids[0], routing_graph=graph
    )
    assert [(owner, line.id) for owner, line in found] == [(ids[1], stop_line)]


def test_the_successors_are_those_of_the_given_routing_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    lanelet_map, ids, _ = _road_with_stop_line_on(1)
    graph = MagicMock()
    graph.following.return_value = []
    _no_graph_built(monkeypatch)
    assert (
        get_stop_line_linestrings_with_following(
            lanelet_map, ids[0], routing_graph=graph
        )
        == []
    )
    [(args, _)] = graph.following.call_args_list
    assert args[0].id == ids[0]


def test_the_wrapper_passes_the_map_manager_s_routing_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from autoware_carla_scenario.coordinate import stop_line
    from autoware_carla_scenario.coordinate.map_manager import MapManager

    lanelet_map, ids, _ = _road_with_stop_line_on(1)
    graph = MagicMock()
    graph.following.return_value = []
    mm = _map_manager(lanelet_map, graph)
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    _no_graph_built(monkeypatch)
    assert stop_line.get_stop_line_linestrings_with_following(ids[0]) == []
    graph.following.assert_called_once()


class _NoGraphMapManager:
    """A stand-in for the initialised MapManager whose routing graph must not
    be asked for: building it reads the whole map."""

    def __init__(self, lanelet_map: LaneletMap) -> None:
        self.lanelet_map = lanelet_map

    @property
    def routing_graph(self) -> object:
        raise AssertionError("the routing graph was asked for")


def test_the_wrapper_needs_no_graph_for_a_lanelet_s_own_stop_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.coordinate import stop_line
    from autoware_carla_scenario.coordinate.map_manager import MapManager

    lanelet_map, ids, line = _road_with_stop_line_on(0)
    mm = _NoGraphMapManager(lanelet_map)
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    found = stop_line.get_stop_line_linestrings_with_following(ids[0])
    assert [(owner, ls.id) for owner, ls in found] == [(ids[0], line)]


def test_the_wrapper_needs_no_graph_to_reject_a_missing_lanelet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.coordinate import stop_line
    from autoware_carla_scenario.coordinate.map_manager import MapManager

    lanelet_map, _, _ = _road_with_stop_line_on(0)
    mm = _NoGraphMapManager(lanelet_map)
    monkeypatch.setattr(MapManager, "get_instance", classmethod(lambda cls: mm))
    with pytest.raises(ValueError, match=f"Lanelet ID {_MISSING} not found"):
        stop_line.get_stop_line_linestrings_with_following(_MISSING)
