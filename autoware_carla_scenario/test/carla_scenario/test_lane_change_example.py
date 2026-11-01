"""The lane-change example, without CARLA.

``lane_change.py`` checks the failure outcome only for ``expect: fail``, so an
unknown ``expect`` has to be refused rather than checked as a success (#1074).
The target lane is tested in the Lanelet2 frame: an OpenDRIVE road and lane
read part of lanelet 136's centreline as the lane on its right (#1081).
"""

from __future__ import annotations

import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import autoware_lanelet2_extension_python.projection  # noqa: F401
import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario import EGO_ROLE_NAME, BaseCondition
from autoware_carla_scenario.coordinate import (
    Lanelet2Pose,
    lanelet_length,
    on_lanelet,
    to_carla_world,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.examples.configs import LaneChangeConfig
from autoware_carla_scenario.examples.lane_change import (
    lane_beside,
    lane_change_target,
)

#: The nishishinjuku fixture map, kept at the repository root.
_DATA = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.parametrize("pattern", ["lane_change/*", "lane_change_fail/*"])
def test_every_shipped_case_is_a_valid_config(pattern: str) -> None:
    cases = run._resolve_scenario_glob(pattern)
    assert cases
    for case in cases:
        cfg = run._compose_config(case, [])
        scenario = OmegaConf.to_container(cfg.scenario, resolve=True)
        assert isinstance(scenario, dict)
        config = LaneChangeConfig(**{str(k): v for k, v in scenario.items()})
        expected = "fail" if pattern.startswith("lane_change_fail") else "success"
        assert config.expect == expected


@pytest.mark.parametrize("expect", ["success", "fail"])
def test_a_known_outcome_is_accepted(expect: str) -> None:
    assert LaneChangeConfig(expect=expect).expect == expect


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"expect": "fial"}, r"expect must be 'success' or 'fail', not 'fial'"),
        ({"expect": "Fail"}, r"expect must be 'success' or 'fail'"),
        ({"expect": ""}, r"expect must be 'success' or 'fail'"),
        ({"direction": "up"}, r"direction must be 'left' or 'right', not 'up'"),
    ],
)
def test_an_unknown_value_is_refused(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LaneChangeConfig(**config)


# ---------------------------------------------------------------------------
# The target lane, in the Lanelet2 frame (#1081)
# ---------------------------------------------------------------------------

#: Road 62 of the fixture: part of lanelet 136's centreline reads as the
#: OpenDRIVE lane on its right.
SPAWN = 136


@pytest.fixture(scope="module")
def nishishinjuku() -> Iterator[None]:
    MapManager.reset()
    mm = MapManager.get_instance()
    mm.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield
    # Release lanelet2/pyxodr objects while the C++ runtime is still valid.
    mm._routing_graph = None
    mm._lanelet_map = None
    mm._road_network = None
    mm._geo_origin = None
    mm._mgrs_offset = None
    MapManager.reset()


class _Ego:
    """A world holding one ego, put on a lanelet by :meth:`on`."""

    def __init__(self) -> None:
        self.location = MagicMock(x=0.0, y=0.0, z=0.0)
        actor = MagicMock()
        actor.attributes = {"role_name": str(EGO_ROLE_NAME)}
        actor.id = 1
        actor.is_alive = True
        actor.get_location.return_value = self.location
        self.world = MagicMock()
        self.world.get_actors.return_value = [actor]

    def on(self, lanelet_id: int, s: float) -> Any:
        """The world with the ego centred on *lanelet_id* at *s*."""
        at = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
        assert on_lanelet(at, lanelet_id)
        self.location.x, self.location.y, self.location.z = at.x, at.y, at.z
        return self.world


def _along(lanelet_id: int, step: float = 1.0) -> list[float]:
    """Arc lengths along *lanelet_id*, *step* apart, its start to its end."""
    length = lanelet_length(lanelet_id)
    points = []
    s = 0.0
    while s <= length:
        points.append(s)
        s += step
    return points


def _first_result(condition: BaseCondition, lanelet_id: int) -> Any:
    """Drive the ego along *lanelet_id*'s centreline; the first result, if any."""
    ego = _Ego()
    for tick, s in enumerate(_along(lanelet_id)):
        result = condition.check(ego.on(lanelet_id, s), tick * 0.05)
        if result is not None:
            return result
    return None


def _setup(direction: str, expect: str) -> tuple[list[Any], list[Any]]:
    """``LaneChangeScenario.setup()``'s pass and fail conditions, CARLA stubbed."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario import EgoConfig, SpawnTransform
    from autoware_carla_scenario.examples.lane_change import LaneChangeScenario

    scenario = LaneChangeScenario(
        EgoConfig(
            spawn_location=SpawnTransform(carla.Transform(carla.Location())),
            vehicle_type="vehicle.mini.cooper",
        ),
        spawn_pose=Lanelet2Pose(lanelet_id=SPAWN, s=5.0),
        config=LaneChangeConfig(direction=direction, expect=expect),
    )
    scenario.set_client(MagicMock(), 8000)
    with unittest.mock.patch.object(
        scenario, "_setup_ego_spawn", return_value=MagicMock(road_id="62", lane_id=2)
    ):
        scenario.setup()
    return scenario._pass_conditions, scenario._fail_conditions


class TestLaneChangeTarget:
    def test_the_lane_beside_is_the_routing_graphs(self, nishishinjuku: None) -> None:
        graph = MapManager.get_instance().routing_graph
        layer = MapManager.get_instance().lanelet_map.laneletLayer
        right = graph.right(layer[SPAWN])
        assert right is not None
        assert lane_beside(SPAWN, "right") == int(right.id)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_staying_on_the_spawn_lanelet_is_not_a_lane_change(
        self, nishishinjuku: None, direction: str
    ) -> None:
        assert _first_result(lane_change_target(SPAWN, direction), SPAWN) is None

    def test_reaching_the_lane_beside_is(self, nishishinjuku: None) -> None:
        beside = lane_beside(SPAWN, "right")
        assert beside is not None
        result = _first_result(lane_change_target(SPAWN, "right"), beside)
        assert result is not None and result.passed

    def test_no_lane_beside_never_holds(self, nishishinjuku: None) -> None:
        # Lanelet 9 has no lane on its left in its direction
        # (``lane_change_fail/left``'s sweep picks it).
        assert lane_beside(9, "left") is None
        condition = lane_change_target(9, "left")
        assert condition.label == "ego_target_lane_sticky"
        assert _first_result(condition, 9) is None


class TestLaneChangeConditions:
    def test_success_does_not_pass_on_the_spawn_lanelet(
        self, nishishinjuku: None
    ) -> None:
        passes, fails = _setup("right", "success")
        assert [c.label for c in passes] == ["ego_target_lane_sticky"]
        assert [c.label for c in fails] == ["scenario_timeout"]
        assert _first_result(passes[0], SPAWN) is None

    def test_fail_does_not_fail_on_the_spawn_lanelet(self, nishishinjuku: None) -> None:
        passes, fails = _setup("right", "fail")
        assert [c.label for c in passes] == ["scenario_timeout"]
        assert [c.label for c in fails] == ["ego_target_lane_sticky"]
        assert _first_result(fails[0], SPAWN) is None
