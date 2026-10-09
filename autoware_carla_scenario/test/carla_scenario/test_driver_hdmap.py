"""The map as files: written once at scenario start, resolved by the policy.

The map is a roadgen-built road with one traffic light, written out as OpenDRIVE:
the same path a CARLA world's ``to_opendrive()`` takes, without a CARLA server.
What is pinned is the chain a light travels -- the OpenDRIVE signal id and lane
CARLA reports, to the Lanelet2 lanelet and regulatory element a policy reads.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock

import pytest

from carla_driver_interface.hdmap import MapFiles
from carla_driver_interface.protocol import (
    StopPoint,
    TrafficLight,
    TrafficLightState,
    Vec3,
    carla_driver_pb2,
)
from autoware_carla_scenario.driver.base import DriverClientConfig
from autoware_carla_scenario.driver.geometry import Pose
from autoware_carla_scenario.driver.renderer import RendererDataBuilder
from autoware_carla_scenario.utils.opendrive import sanitize_opendrive

from .test_driver_renderer import _ego, _light, _world

roadgen = pytest.importorskip("roadgen")

from autoware_carla_scenario.driver.hdmap_export import (  # noqa: E402
    MANIFEST_FILE,
    export_map,
    map_id_for,
)


@pytest.fixture(scope="module")
def opendrive(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Two roads through a junction; a light where the first one ends."""
    m = roadgen.Map()

    def lanes():  # noqa: ANN202
        return [
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="backward"),
        ]

    a = m.add_road(start=(0.0, 0.0, 0.0), end=(100.0, 0.0, 0.0), lanes=lanes())
    b = m.add_road(start=(120.0, 0.0, 0.0), end=(220.0, 0.0, 0.0), lanes=lanes())
    m.connect(a, b, junction=m.add_junction())
    light = m.add_traffic_light(a.lane(0), end="end")
    stop = m.add_stop_line(a.lane(0), end="end")
    m.add_traffic_light_rule([light], [a.lane(0)], stop_line=stop)
    path = tmp_path_factory.mktemp("xodr") / "town.xodr"
    m.export_opendrive(str(path))
    return path.read_text(encoding="utf-8")


def _signal_id(opendrive: str) -> str:
    return str(next(ElementTree.fromstring(opendrive).iter("signal")).get("id"))


def _light_on_road_0(opendrive: str) -> TrafficLight:
    """The light as CARLA reports it: its id, and a stop point at the first road's
    end on its right-hand lane (OpenDRIVE lane -1)."""
    return TrafficLight(
        opendrive_id=_signal_id(opendrive),
        state=TrafficLightState.TRAFFIC_LIGHT_STATE_RED,
        stop_points=[
            StopPoint(
                road_id=0,
                section_id=0,
                lane_id=-1,
                position_local=Vec3(x=99.0, y=-1.75, z=0.0),
            )
        ],
    )


def _relations(osm: Path) -> Dict[str, ElementTree.Element]:
    return {
        str(rel.get("id")): rel
        for rel in ElementTree.parse(osm).getroot().iter("relation")
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_map_is_written_once_under_its_content_id(
    opendrive: str, tmp_path: Path
) -> None:
    map_id = export_map(opendrive, "Carla/Maps/Town", tmp_path)
    assert map_id == map_id_for("Carla/Maps/Town", opendrive)
    assert map_id.startswith("Town-")
    directory = tmp_path / map_id
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    assert set(manifest["formats"]) == {"lanelet2"}
    assert (directory / manifest["formats"]["lanelet2"]["path"]).is_file()
    assert (directory / manifest["source"]).read_text() == opendrive

    written = (directory / MANIFEST_FILE).stat().st_mtime_ns
    assert export_map(opendrive, "Carla/Maps/Town", tmp_path) == map_id
    assert (directory / MANIFEST_FILE).stat().st_mtime_ns == written, "rewrote a set"
    assert [p.name for p in tmp_path.iterdir()] == [map_id], "left scratch behind"


def test_a_changed_map_gets_a_new_id() -> None:
    assert map_id_for("Town", "<OpenDRIVE/>") != map_id_for("Town", "<OpenDRIVE />")


def test_an_unknown_format_is_refused_before_anything_is_written(
    opendrive: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="unknown map format"):
        export_map(opendrive, "Town", tmp_path, formats=("lanelet3",))
    assert not any(tmp_path.iterdir())


def test_carla_departures_from_the_standard_are_rewritten() -> None:
    carla = (
        '<header><userData><vectorScene program="RoadRunner"/></userData></header>'
        '<roadMark sOffset="0" type="curb" weight="standard"/>'
        '<object id="1" type="-1" name="x"/>'
        '<cornerLocal u="0" v="0" z="0"/>'
        '<cornerRoad s="0" t="0" dz="0"/>'
    )
    fixed = sanitize_opendrive(carla)
    assert "userData" not in fixed
    assert 'color="standard"' in fixed
    assert 'type="none"' in fixed
    assert fixed.count('height="0"') == 2


# ---------------------------------------------------------------------------
# Reading what was written
# ---------------------------------------------------------------------------


def test_a_light_resolves_to_its_lanelet_and_regulatory_element(
    opendrive: str, tmp_path: Path
) -> None:
    map_files = MapFiles.open(tmp_path, export_map(opendrive, "Town", tmp_path))
    [stop] = map_files.stop_lines([_light_on_road_0(opendrive)])

    assert stop.traffic_light_id == _signal_id(opendrive)
    assert stop.position_local.tolist() == [99.0, -1.75, 0.0]
    assert len(stop.lane_ids) == 1 and stop.rule_ids and stop.light_ids

    # Cross-check against the Lanelet2 file itself: the lanelet is the one that
    # carries the regulatory element, and the element refers to the light.
    relations = _relations(map_files.path)
    lanelet = relations[stop.lane_ids[0]]
    regulatory = [
        m.get("ref")
        for m in lanelet.iter("member")
        if m.get("role") == "regulatory_element"
    ]
    assert set(stop.rule_ids) <= set(regulatory)
    refers = {
        m.get("ref")
        for m in relations[stop.rule_ids[0]].iter("member")
        if m.get("role") == "refers"
    }
    assert refers & set(stop.light_ids)


# ---------------------------------------------------------------------------
# What the runtime sends
# ---------------------------------------------------------------------------


def _carla_light(opendrive_id: str) -> MagicMock:
    light = _light("Red")
    light.get_opendrive_id.return_value = opendrive_id
    waypoint = light.get_stop_waypoints.return_value[0]
    waypoint.section_id = 0
    return light


def test_every_light_is_sent_with_its_stop_points_when_a_map_is_written() -> None:
    world = _world(actors=[_carla_light("943")])
    builder = RendererDataBuilder(
        world, _ego(), DriverClientConfig(), map_id="Town-abc"
    )
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity()))

    assert message.map_id == "Town-abc"
    [light] = message.traffic_lights
    assert light.opendrive_id == "943"
    assert light.state == carla_driver_pb2.TRAFFIC_LIGHT_STATE_RED
    [stop] = light.stop_points
    assert (stop.road_id, stop.section_id, stop.lane_id) == (1, 0, -1)
    # The stop waypoint as CARLA has it, mirrored into the local frame.
    assert (stop.position_local.x, stop.position_local.y) == (30.0, -0.0)


def test_without_a_map_no_lights_are_sent() -> None:
    builder = RendererDataBuilder(
        _world(actors=[_carla_light("943")]), _ego(), DriverClientConfig()
    )
    message = carla_driver_pb2.CarlaRendererData()
    message.ParseFromString(builder.build(1_000, Pose.identity()))
    assert message.map_id == "" and len(message.traffic_lights) == 0


def test_the_entity_writes_the_worlds_map(opendrive: str, tmp_path: Path) -> None:
    from autoware_carla_scenario.entity.carla_driver_entity import CarlaDriverEntity

    entity = CarlaDriverEntity(
        DriverClientConfig(map_dir=str(tmp_path)), client=MagicMock()
    )
    entity._map = SimpleNamespace(  # noqa: SLF001
        name="Carla/Maps/Town", to_opendrive=lambda: opendrive
    )
    map_id = entity._write_map()  # noqa: SLF001
    assert map_id == map_id_for("Carla/Maps/Town", opendrive)
    assert (tmp_path / map_id / MANIFEST_FILE).is_file()
