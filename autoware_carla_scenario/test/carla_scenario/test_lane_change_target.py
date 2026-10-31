"""Which OpenDRIVE lane a lane change ends on."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from autoware_carla_scenario.conditions import beside_lane_id
from autoware_carla_scenario.coordinate.map_manager import MapManager

#: The nishishinjuku fixture map, kept at the repository root: every road of
#: it is left-hand traffic (``rule="LHT"``).
_DATA = Path(__file__).resolve().parents[3] / "data"
XODR_PATH = _DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _DATA / "nishishinjuku.osm"

#: A straight two-way street whose one road is right-hand traffic.
TWO_WAY = Path(__file__).resolve().parents[1] / "data" / "two_way_street"


@pytest.mark.parametrize(
    ("lane_id", "side", "target"),
    [
        # Right of the reference line, driving along it: the centre is up.
        (-2, "left", -1),
        (-1, "right", -2),
        # Left of it, driving against it: the centre is down.
        (2, "left", 1),
        (1, "right", 2),
        # Left of the innermost lane is across the centre line: lane 0 is
        # stepped over.
        (-1, "left", 1),
        (1, "left", -1),
    ],
)
def test_a_change_ends_one_lane_over_towards_the_side_asked_for(
    lane_id: int, side: str, target: int
) -> None:
    assert beside_lane_id(lane_id, side) == target
    assert beside_lane_id(lane_id, side, "RHT") == target


@pytest.mark.parametrize(
    ("lane_id", "side", "target"),
    [
        # Left of the reference line, driving along it: left is outwards.
        (2, "left", 3),
        (2, "right", 1),
        (1, "left", 2),
        # Right of it, driving against it.
        (-2, "left", -3),
        (-2, "right", -1),
        # Across the centre line, lane 0 stepped over.
        (1, "right", -1),
        (-1, "right", 1),
    ],
)
def test_left_hand_traffic_mirrors_the_sides(
    lane_id: int, side: str, target: int
) -> None:
    assert beside_lane_id(lane_id, side, "LHT") == target


def test_only_left_and_right_are_sides() -> None:
    with pytest.raises(ValueError):
        beside_lane_id(-1, "same")


def test_only_rht_and_lht_are_traffic_rules() -> None:
    with pytest.raises(ValueError):
        beside_lane_id(-1, "left", "rht")


def test_the_centre_lane_has_no_side() -> None:
    with pytest.raises(ValueError):
        beside_lane_id(0, "left")


def _load_map(xodr: Path, osm: Path) -> Generator[None, None, None]:
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(xodr, osm)
    yield
    # Release lanelet2/pyxodr objects while the C++ runtime is still valid.
    mm._routing_graph = None
    mm._lanelet_map = None
    mm._road_network = None
    mm._geo_origin = None
    mm._mgrs_offset = None
    MapManager.reset()


@pytest.fixture
def two_way_street() -> Generator[None, None, None]:
    yield from _load_map(TWO_WAY / "two_way_street.xodr", TWO_WAY / "lanelet2_map.osm")


def test_a_roads_stated_rule_is_read(two_way_street: None) -> None:
    from autoware_carla_scenario.coordinate import road_traffic_rule

    assert road_traffic_rule("0") == "RHT"
    with pytest.raises(KeyError):
        road_traffic_rule("no such road")


@pytest.mark.parametrize("rule", ["LHT", ""])
def test_a_road_without_a_stated_rule_is_right_hand_traffic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    """The street's road says LHT, or says nothing, which OpenDRIVE reads as RHT."""
    attribute = ' rule="RHT"'
    from types import SimpleNamespace

    from pyxodr.road_objects.network import RoadNetwork

    from autoware_carla_scenario.coordinate import road_traffic_rule

    text = (TWO_WAY / "two_way_street.xodr").read_text()
    assert text.count(attribute) == 1
    xodr = tmp_path / "street.xodr"
    xodr.write_text(text.replace(attribute, f' rule="{rule}"' if rule else ""))
    network = RoadNetwork(str(xodr))
    network.get_roads()
    monkeypatch.setattr(
        MapManager,
        "get_instance",
        classmethod(lambda cls: SimpleNamespace(road_network=network)),
    )
    assert road_traffic_rule("0") == (rule or "RHT")


@pytest.fixture(scope="module")
def loaded_map() -> Generator[None, None, None]:
    yield from _load_map(XODR_PATH, OSM_PATH)


class TestNishishinjuku:
    """On the fixture map, the lane beside is the routing graph's neighbour."""

    def test_its_roads_are_left_hand_traffic(self, loaded_map: None) -> None:
        from autoware_carla_scenario.coordinate import road_traffic_rule

        roads = MapManager.get_instance().road_network.road_ids_to_object
        assert {road_traffic_rule(road_id) for road_id in roads} == {"LHT"}

    def test_lanelet_183s_neighbours(self, loaded_map: None) -> None:
        """The issue's example: lanelet 183 is road 80 lane 2, 182 left, 184 right."""
        from autoware_carla_scenario.conditions import EntityLaneOfCondition
        from autoware_carla_scenario.coordinate import Lanelet2Pose

        lanes = {
            lane: EntityLaneOfCondition(
                "ego", Lanelet2Pose(lanelet_id=183, s=5.0), lane, label="x"
            ).get_details()
            for lane in ("same", "left", "right")
        }
        assert (lanes["same"]["road_id"], lanes["same"]["lane_id"]) == ("80", 2)
        assert (lanes["left"]["road_id"], lanes["left"]["lane_id"]) == ("80", 3)
        assert (lanes["right"]["road_id"], lanes["right"]["lane_id"]) == ("80", 1)

    def test_every_neighbour_on_the_same_road_is_the_lane_beside(
        self, loaded_map: None
    ) -> None:
        from autoware_carla_scenario.coordinate import (
            Lanelet2Pose,
            OpenDrivePose,
            lanelet_length,
            road_traffic_rule,
            to_opendrive,
        )

        mm = MapManager.get_instance()
        graph = mm.routing_graph
        roads = mm.road_network.road_ids_to_object

        def lane_of(lanelet_id: int) -> OpenDrivePose:
            length = lanelet_length(lanelet_id)
            return to_opendrive(Lanelet2Pose(lanelet_id=lanelet_id, s=length / 2))

        checked = 0
        mismatches = []
        for lanelet in mm.lanelet_map.laneletLayer:
            for side in ("left", "right"):
                neighbour = (
                    graph.left(lanelet) if side == "left" else graph.right(lanelet)
                )
                if neighbour is None:
                    continue
                here = lane_of(lanelet.id)
                there = lane_of(neighbour.id)
                if (
                    here.road_id != there.road_id
                    or here.lane_id == there.lane_id
                    or roads[here.road_id].road_xml.get("junction", "-1") != "-1"
                ):
                    continue
                checked += 1
                beside = beside_lane_id(
                    here.lane_id, side, road_traffic_rule(here.road_id)
                )
                if beside != there.lane_id:
                    mismatches.append(
                        (
                            lanelet.id,
                            side,
                            here.road_id,
                            here.lane_id,
                            beside,
                            there.lane_id,
                        )
                    )
        assert checked > 100
        assert mismatches == []
