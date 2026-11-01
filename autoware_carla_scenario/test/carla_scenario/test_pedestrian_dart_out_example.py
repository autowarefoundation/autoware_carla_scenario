"""The pedestrian_dart_out example: its pedestrian waits at the kerb (#1064).

The kerb is on the side the road's traffic keeps to: the right in right-hand
traffic, the left in left-hand traffic. Nishi-Shinjuku, the fixture map and
the example's fixed case, drives on the left; a pedestrian placed on the right
of the ego's lane there stood in the oncoming lanelet, and the sweep picked the
lanes by the centre line.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.examples.configs import PedestrianDartOutConfig
from autoware_carla_scenario.examples.pedestrian_dart_out import (
    kerb_offset_m,
    kerb_pose,
)

_DATA = Path(__file__).resolve().parents[3] / "data"
XODR_PATH = _DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _DATA / "nishishinjuku.osm"


@pytest.mark.parametrize("distance", [3.0, -3.0])
def test_the_kerb_is_on_the_side_the_traffic_keeps_to(distance: float) -> None:
    # The sign is ignored: -3.0, the old right-hand value, is 3 m too.
    assert kerb_offset_m(distance, "RHT") == -3.0
    assert kerb_offset_m(distance, "LHT") == 3.0


def test_only_rht_and_lht_are_traffic_rules() -> None:
    with pytest.raises(ValueError, match="traffic_rule"):
        kerb_offset_m(3.0, "rht")


@pytest.mark.parametrize(("rule", "t", "heading"), [("RHT", -3.0, 1), ("LHT", 3.0, -1)])
def test_the_pedestrian_faces_the_road(rule: str, t: float, heading: int) -> None:
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    pose = kerb_pose(
        Lanelet2Pose(lanelet_id=20, s=5.0), PedestrianDartOutConfig(), rule
    )
    assert (pose.lanelet_id, pose.s, pose.t) == (20, 40.0, t)
    # A quarter turn towards the lane's centre.
    assert pose.heading == pytest.approx(heading * math.pi / 2)


def test_the_shipped_case_sweeps_the_kerb_lanes_of_a_left_hand_map() -> None:
    cfg = run._compose_config("pedestrian_dart_out/pedestrian_dart_out", [])
    assert cfg.scenario.pedestrian_offset_m == 3.0
    (both,) = cfg.sweep.constraints["ego.spawn_lanelet_id"]
    assert both.constraints[0] == {
        "type": "not",
        "constraint": {"type": "has_adjacent", "value": "left", "include_forks": True},
    }


@pytest.fixture(scope="module")
def nishishinjuku() -> Generator[None, None, None]:
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(XODR_PATH, OSM_PATH)
    yield
    # Release lanelet2/pyxodr objects while the C++ runtime is still valid.
    mm._routing_graph = None
    mm._lanelet_map = None
    mm._road_network = None
    mm._geo_origin = None
    mm._mgrs_offset = None
    MapManager.reset()


def _road_lanelets_at(pose: Any) -> list[int]:
    """The road lanelets of the loaded map a Lanelet2 pose stands in."""
    import lanelet2

    from autoware_carla_scenario.coordinate import to_carla_world

    mm = MapManager.get_instance()
    world = to_carla_world(pose)
    offset_x, offset_y = mm.mgrs_offset
    point = lanelet2.core.BasicPoint2d(world.x + offset_x, -world.y + offset_y)
    return [
        lanelet.id
        for _, lanelet in lanelet2.geometry.findWithin2d(
            mm.lanelet_map.laneletLayer, point, 0.0
        )
        if "subtype" in lanelet.attributes and lanelet.attributes["subtype"] == "road"
    ]


def test_the_fixed_case_puts_the_pedestrian_off_the_road(nishishinjuku: None) -> None:
    from dataclasses import replace

    from autoware_carla_scenario.coordinate import (
        Lanelet2Pose,
        road_traffic_rule,
        to_opendrive,
    )

    cfg = run._compose_config("pedestrian_dart_out/pedestrian_dart_out", [])
    spawn = Lanelet2Pose(
        lanelet_id=int(cfg.ego.spawn_lanelet_id), s=float(cfg.ego.spawn_s)
    )
    config = PedestrianDartOutConfig(
        pedestrian_ahead_m=float(cfg.scenario.pedestrian_ahead_m),
        pedestrian_offset_m=float(cfg.scenario.pedestrian_offset_m),
    )
    rule = road_traffic_rule(to_opendrive(spawn).road_id)
    assert rule == "LHT"

    kerb = kerb_pose(spawn, config, rule)
    assert kerb.t > 0.0
    assert _road_lanelets_at(kerb) == []
    # The other side is the road: the lane beside, by the centre line.
    assert _road_lanelets_at(replace(kerb, t=-kerb.t)) != []


def test_the_right_hand_kerb_of_the_old_case_was_the_oncoming_lane(
    nishishinjuku: None,
) -> None:
    """The case before #1064: lanelet 20, the pedestrian 3 m to its right."""
    from autoware_carla_scenario.coordinate import Lanelet2Pose

    old = Lanelet2Pose(lanelet_id=20, s=40.0, t=-3.0)
    assert _road_lanelets_at(old) == [28]
