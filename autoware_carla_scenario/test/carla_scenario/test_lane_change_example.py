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
    to_opendrive,
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


@pytest.mark.parametrize("case", ["lane_change_fail/left", "lane_change_fail/right"])
def test_a_refused_change_is_driven_long_enough_to_have_happened(case: str) -> None:
    """#2253: at 1 km/h for 1.5 s no lane change could have completed, so the
    refused cases passed whatever the ego did; they now drive at a speed and
    for as long as a change that is allowed does (``lane_change/*``)."""
    cfg = run._compose_config(case, [])
    allowed = run._compose_config(case.replace("lane_change_fail", "lane_change"), [])
    assert cfg.scenario.timeout_seconds >= allowed.scenario.timeout_seconds
    assert cfg.ego.initial_speed_kmh >= 20.0


@pytest.mark.parametrize("expect", ["success", "fail"])
def test_a_known_outcome_is_accepted(expect: str) -> None:
    assert LaneChangeConfig(expect=expect).expect == expect


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"expect": "fial"},
            r"^scenario\.expect must be 'success' or 'fail', not 'fial'",
        ),
        ({"expect": "Fail"}, r"^scenario\.expect must be 'success' or 'fail'"),
        ({"expect": ""}, r"^scenario\.expect must be 'success' or 'fail'"),
        (
            {"direction": "up"},
            r"^scenario\.direction must be 'left' or 'right', not 'up'",
        ),
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
    scenario = _set_up(direction, expect)
    return scenario._pass_conditions, scenario._fail_conditions


def _set_up(direction: str, expect: str) -> Any:
    """A ``LaneChangeScenario`` after ``setup()``, CARLA stubbed."""
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
    # The OpenDRIVE pose the spawn reads as, as _setup_ego_spawn() returns it
    # (road 62): a target derived from it fired on the spawn lanelet (#1081).
    spawn = to_opendrive(to_carla_world(Lanelet2Pose(lanelet_id=SPAWN, s=5.0)))
    with unittest.mock.patch.object(scenario, "_setup_ego_spawn", return_value=spawn):
        scenario.setup()
    return scenario


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

    def test_a_lane_beside_across_a_line_not_to_cross_counts(
        self, nishishinjuku: None
    ) -> None:
        # Lanelet 6 may not change into lanelet 5 on its left
        # (``lane_change_fail/left``'s default spawn): the fail case must still
        # notice the ego there.
        graph = MapManager.get_instance().routing_graph
        layer = MapManager.get_instance().lanelet_map.laneletLayer
        assert graph.left(layer[6]) is None
        assert lane_beside(6, "left") == 5
        assert _first_result(lane_change_target(6, "left"), 6) is None
        result = _first_result(lane_change_target(6, "left"), 5)
        assert result is not None and result.passed

    def test_a_lane_beside_on_the_right_across_a_line_not_to_cross_counts(
        self, nishishinjuku: None
    ) -> None:
        # Lanelet 3002160 may not change into lanelet 3002161 on its right:
        # the lane beside falls back to the adjacent lanelet on that side too.
        graph = MapManager.get_instance().routing_graph
        layer = MapManager.get_instance().lanelet_map.laneletLayer
        assert graph.right(layer[3002160]) is None
        assert lane_beside(3002160, "right") == 3002161

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
        assert [c.label for c in passes] == ["ego_kept_its_lane"]
        assert [c.label for c in fails] == [
            "ego_target_lane_sticky",
            "ego_left_its_lane",
            "scenario_timeout",
        ]
        # Driven along the spawn lanelet before the timeout: nothing fires.
        for condition in [*passes, *fails]:
            assert _first_result(condition, SPAWN) is None

    def test_fail_passes_at_its_timeout_in_its_own_lane(
        self, nishishinjuku: None
    ) -> None:
        """#2253: ``expect=fail`` passed on its timeout whatever the ego did;
        it passes only with the ego still in its own lane then."""
        (kept,), _fails = _setup("right", "fail")
        timeout = LaneChangeConfig().timeout_seconds
        ego = _Ego()
        assert kept.check(ego.on(SPAWN, 5.0), timeout - 0.05) is None
        result = kept.check(ego.on(SPAWN, 6.0), timeout)
        assert result is not None
        # A pass, said as one: not the "Timeout after ..." of a failure.
        assert result.passed
        assert not result.message.startswith("Timeout")
        # A lanelet that continues the spawn lanelet is its own lane too.
        mm = MapManager.get_instance()
        following = int(
            mm.routing_graph.following(mm.lanelet_map.laneletLayer[SPAWN], False)[0].id
        )
        assert kept.check(ego.on(following, 1.0), timeout + 0.05) is not None

    def test_fail_does_not_pass_at_its_timeout_off_its_own_lane(
        self, nishishinjuku: None
    ) -> None:
        passes, fails = _setup("right", "fail")
        (kept,) = passes
        target, _left_lane, timeout_condition = fails
        beside = lane_beside(SPAWN, "right")
        assert beside is not None
        timeout = LaneChangeConfig().timeout_seconds
        ego = _Ego()
        world = ego.on(beside, 5.0)
        assert kept.check(world, timeout) is None
        # The timeout fails it then, as the lane beside already had.
        assert timeout_condition.check(world, timeout) is not None
        assert target.check(world, timeout) is not None

    def test_fail_fails_once_the_ego_has_left_its_own_lane(
        self, nishishinjuku: None
    ) -> None:
        """Off its own lanes for ``LEAVING_LANE_SECONDS`` -- here into the lane
        beside, which the target check catches as well -- the run fails."""
        from autoware_carla_scenario.examples.lane_change import LEAVING_LANE_SECONDS

        _passes, fails = _setup("right", "fail")
        left_lane = fails[1]
        beside = lane_beside(SPAWN, "right")
        assert beside is not None
        ego = _Ego()
        assert left_lane.check(ego.on(SPAWN, 5.0), 0.0) is None
        assert left_lane.check(ego.on(beside, 5.0), 1.0) is None
        result = left_lane.check(ego.on(beside, 6.0), 1.0 + LEAVING_LANE_SECONDS)
        assert result is not None

    @pytest.mark.parametrize("expect", ["success", "fail"])
    def test_its_own_timeout_keeps_the_default_off(
        self, nishishinjuku: None, expect: str
    ) -> None:
        """Both register a timeout of their own that bounds the run -- the fail
        one; ``expect=fail``'s pass condition holds it only beside the ego
        being in its own lane -- so no default fail timeout is added, which
        would fail an ``expect=fail`` run before it could pass (#2382)."""
        from autoware_carla_scenario.scenario_runner import _register_default_timeout

        scenario = _set_up("right", expect)
        fails = list(scenario._fail_conditions)
        assert not _register_default_timeout(scenario, 1.0, "lane_change")
        assert scenario._fail_conditions == fails
