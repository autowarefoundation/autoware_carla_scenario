"""Background traffic regions on the fixture map: vehicle lanelets only (#975).

A crosswalk, walkway or road shoulder has no vehicle successor, so the default
sink ("no successor") matched every one of them and removed a vehicle as it
drove over a crosswalk; the default source placed vehicles across the road.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
import yaml

from autoware_carla_scenario import TrafficSinkAction, TrafficSourceAction
from autoware_carla_scenario.actions.background_traffic import LaneletRegion
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.traffic import TrafficBackend

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data"
_CONFIG = (
    _ROOT
    / "autoware_carla_scenario"
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
    / "config.yaml"
)
#: Subtypes no vehicle drives on.
_NOT_FOR_VEHICLES = {"crosswalk", "walkway", "road_shoulder", "stairs"}
#: A road lanelet a crosswalk crosses, which has a vehicle successor.
_THROUGH_ROAD = 31
#: A road lanelet with no successor: an end of the map's roads.
_DEAD_END = 124398


def _defaults() -> dict[str, Any]:
    return dict(yaml.safe_load(_CONFIG.read_text())["background_traffic"])


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """Load the fixture map once for this module."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield manager
    MapManager._instance = saved


@pytest.fixture(autouse=True)
def _module_map(loaded_map: MapManager) -> None:
    """Keep this module's map in place for each test (conftest restores it)."""
    MapManager._instance = loaded_map


def _subtype(manager: MapManager, lanelet_id: int) -> str:
    attributes = manager.lanelet_map.laneletLayer[lanelet_id].attributes
    return str(attributes["subtype"]) if "subtype" in attributes else ""


class _Backend(TrafficBackend):
    name = "fake"

    def __init__(self) -> None:
        self.vehicles: dict[str, tuple[float, float]] = {}
        self.removed: list[str] = []

    def spawn_background(
        self,
        world: Any,
        transform: Any,
        *,
        speed_kmh: float,
        blueprint: Optional[str] = None,
    ) -> Optional[str]:
        handle = f"bg{len(self.vehicles) + len(self.removed) + 1}"
        self.vehicles[handle] = (transform.location.x, transform.location.y)
        return handle

    def background_vehicles(self, world: Any) -> dict[str, tuple[float, float]]:
        return dict(self.vehicles)

    def remove_background(self, world: Any, handle: str) -> None:
        self.vehicles.pop(handle)
        self.removed.append(handle)


def _sink_with(x: float, y: float) -> tuple[TrafficSinkAction, _Backend]:
    sink = TrafficSinkAction(_defaults()["sink"]["constraints"])
    backend = _Backend()
    backend.vehicles["bg1"] = (x, y)
    sink.set_traffic_backend(backend)
    sink.execute(None)  # type: ignore[arg-type]
    return sink, backend


def _at(lanelet_id: int, s: float) -> tuple[float, float]:
    from autoware_carla_scenario.coordinate import Lanelet2Pose, to_carla_world

    pose = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
    return pose.x, pose.y


class TestDefaultSink:
    def test_it_holds_no_lanelet_vehicles_do_not_drive_on(
        self, loaded_map: MapManager
    ) -> None:
        ids = LaneletRegion(_defaults()["sink"]["constraints"]).lanelet_ids
        assert ids
        assert {_subtype(loaded_map, i) for i in ids}.isdisjoint(_NOT_FOR_VEHICLES)

    def test_a_vehicle_crossing_a_crosswalk_is_not_removed(self) -> None:
        sink, backend = _sink_with(*_at(_THROUGH_ROAD, 15.0))
        assert backend.removed == []
        assert sink.removed == 0

    def test_a_vehicle_on_a_dead_end_road_is_removed(
        self, loaded_map: MapManager
    ) -> None:
        from autoware_carla_scenario.coordinate import lanelet_length

        assert _subtype(loaded_map, _DEAD_END) == "road"
        lanelet = loaded_map.lanelet_map.laneletLayer[_DEAD_END]
        assert list(loaded_map.routing_graph.following(lanelet)) == []
        sink, backend = _sink_with(*_at(_DEAD_END, lanelet_length(_DEAD_END) / 2))
        assert backend.removed == ["bg1"]
        assert sink.removed == 1


class TestDefaultSource:
    def test_its_region_holds_no_lanelet_vehicles_do_not_drive_on(
        self, loaded_map: MapManager
    ) -> None:
        ids = LaneletRegion(_defaults()["source"]["constraints"]).lanelet_ids
        assert ids
        assert {_subtype(loaded_map, i) for i in ids}.isdisjoint(_NOT_FOR_VEHICLES)

    def test_a_seeded_source_places_none_off_the_road(
        self, loaded_map: MapManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import autoware_carla_scenario.coordinate as coordinate

        picked: list[int] = []
        real = coordinate.to_carla_world

        def recording(pose: Any) -> Any:
            picked.append(int(pose.lanelet_id))
            return real(pose)

        monkeypatch.setattr(coordinate, "to_carla_world", recording)
        source = _defaults()["source"]
        action = TrafficSourceAction(
            source["constraints"], initial_vehicles=40, min_gap_m=1.0, seed=975
        )
        backend = _Backend()
        action.set_traffic_backend(backend)
        action.execute(None)  # type: ignore[arg-type]
        assert len(backend.vehicles) == 40
        assert {_subtype(loaded_map, i) for i in picked}.isdisjoint(_NOT_FOR_VEHICLES)
