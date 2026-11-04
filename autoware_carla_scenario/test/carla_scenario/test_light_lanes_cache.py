"""``light_lanes._lanes_of`` reads a Lanelet2 map's lanes once per map (#1323).

The lanes are kept with the map they were read from and compared with ``is``:
keyed by ``id`` alone, a new map allocated at a freed map's address (a
scenario queue reloads the map at each start) was handed the old map's lanes.
"""

from __future__ import annotations

from typing import Any

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import pytest
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

from autoware_carla_scenario.utils import light_lanes

_LANE = {"type": "lanelet", "subtype": "road", "location": "urban"}
_ids = iter(range(1000, 100000))


def _map(count: int) -> Any:
    """*count* 10 m lanelets end to end along x."""
    lanelet_map = LaneletMap()
    for k in range(count):
        x0, x1 = 10.0 * k, 10.0 * (k + 1)
        left = LineString3d(
            next(_ids),
            [Point3d(next(_ids), x0, 3.5, 0.0), Point3d(next(_ids), x1, 3.5, 0.0)],
            AttributeMap({"type": "line_thin"}),
        )
        right = LineString3d(
            next(_ids),
            [Point3d(next(_ids), x0, 0.0, 0.0), Point3d(next(_ids), x1, 0.0, 0.0)],
            AttributeMap({"type": "line_thin"}),
        )
        lanelet_map.add(Lanelet(next(_ids), left, right, AttributeMap(_LANE)))
    return lanelet_map


class TestLanesFollowTheLoadedMap:
    @pytest.fixture(autouse=True)
    def _same_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(light_lanes, "_LANES", None)
        # Every map has the same id, as one allocated at a freed map's
        # address does: only identity tells them apart.
        monkeypatch.setattr(light_lanes, "id", lambda _obj: 1, raising=False)

    def test_a_new_map_gets_its_own_lanes(self) -> None:
        assert len(light_lanes._lanes_of(_map(3)).ids) == 3
        assert len(light_lanes._lanes_of(_map(1)).ids) == 1

    def test_the_same_map_reuses_its_lanes(self) -> None:
        lanelet_map = _map(2)
        assert light_lanes._lanes_of(lanelet_map) is light_lanes._lanes_of(lanelet_map)
