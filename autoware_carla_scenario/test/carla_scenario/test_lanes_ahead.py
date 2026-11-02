"""Staying on the lanes ahead: ``lanelets_ahead`` and the composition built on it.

"The ego left the lanes ahead of it" is not a condition of its own: it is an
``EntityLanePositionCondition`` per lanelet ``lanelets_ahead`` returns, under
an ``OrCondition``, under a logical ``NotCondition`` (``holds_when_pending``),
held by a ``PersistentCondition`` -- what ``cut_in``'s ``left_lanes_ahead``
builds.  These tests run it on a synthetic map, with the Lanelet2 and CARLA
frames aligned (CARLA ``(x, y)`` is Lanelet2 ``(x, -y)``)::

    y 7.0  +----- B1 -----+----- B2 -----+             (the lane on the left)
    y 3.5  +----- A1 -----+----- A2 -----+----- A3 ----+
    y 0.0  +--------------+--------------+\\
          x 0            50             100  F (forks off to the right,
                                                 a junction lanelet)
"""

from __future__ import annotations

import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# autoware_lanelet2_extension_python must be imported before lanelet2.
import autoware_lanelet2_extension_python.projection  # noqa: F401
import lanelet2.core
import lanelet2.routing
import lanelet2.traffic_rules
import pytest
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d

from autoware_carla_scenario import EGO_ROLE_NAME
from autoware_carla_scenario.conditions import (
    BaseCondition,
    CollisionCondition,
    NotCondition,
    PersistentCondition,
    ScenarioResult,
    StickyCondition,
    TimeoutCondition,
)
from autoware_carla_scenario.coordinate import (
    Lanelet2Pose,
    MapManager,
    OpenDrivePose,
    lanelets_ahead,
    on_lanelet,
    to_carla_world,
)
from autoware_carla_scenario.examples.configs import CutInConfig
from autoware_carla_scenario.examples.cut_in import (
    CUT_IN_ROLE,
    ego_lanes_ahead,
    lanes_ahead_before_junction,
    left_lanes_ahead,
    npc_in_lanes_ahead,
)

A1, A2, A3, B1, B2, F = 1, 2, 3, 11, 12, 21

_LANE = {"type": "lanelet", "subtype": "road", "location": "urban"}
#: A lanelet in a junction, as Autoware's maps tag one.
_TURN = {**_LANE, "turn_direction": "right"}
_ids = iter(range(1000, 100000))


def _point(x: float, y: float) -> Point3d:
    return Point3d(next(_ids), x, y, 0.0)


def _line(points: list[Point3d]) -> LineString3d:
    return LineString3d(next(_ids), points, AttributeMap({"type": "line_thin"}))


def _row(y: float) -> list[Point3d]:
    """Points at x = 0, 50, 100 along *y*, shared by the lanelets that meet there."""
    return [_point(x, y) for x in (0.0, 50.0, 100.0)]


def _map() -> Any:
    lanelet_map = LaneletMap()
    right, middle, left = _row(0.0), _row(3.5), _row(7.0)

    def add(
        lanelet_id: int,
        lb: list[Point3d],
        rb: list[Point3d],
        attributes: dict[str, str] = _LANE,
    ) -> None:
        lanelet_map.add(
            Lanelet(lanelet_id, _line(lb), _line(rb), AttributeMap(attributes))
        )

    add(A1, middle[0:2], right[0:2])
    add(A2, middle[1:3], right[1:3])
    add(B1, left[0:2], middle[0:2])
    add(B2, left[1:3], middle[1:3])
    # A3 straight on; F forks off to the right from the same end of A2.
    add(A3, [middle[2], _point(150.0, 3.5)], [right[2], _point(150.0, 0.0)])
    add(F, [middle[2], _point(140.0, -20.0)], [right[2], _point(143.5, -20.0)], _TURN)
    return lanelet_map


@pytest.fixture(scope="module")
def road() -> tuple[Any, Any]:
    lanelet_map = _map()
    rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )
    return lanelet_map, lanelet2.routing.RoutingGraph(lanelet_map, rules)


@pytest.fixture
def loaded(road: tuple[Any, Any]) -> Iterator[None]:
    """The synthetic map as the loaded one, Lanelet2 and CARLA frames aligned."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager._lanelet_map, manager._routing_graph = road
    manager._mgrs_offset = (0.0, 0.0)
    manager._z_offset = 0.0
    yield
    MapManager._instance = saved


# ---------------------------------------------------------------------------
# lanelets_ahead
# ---------------------------------------------------------------------------


class TestLaneletsAhead:
    def test_the_start_and_the_successors_it_reaches(self, loaded: None) -> None:
        assert lanelets_ahead(A1, 60.0) == [A1, A2]

    def test_every_branch_and_never_a_lane_beside(self, loaded: None) -> None:
        assert set(lanelets_ahead(A1, 300.0)) == {A1, A2, A3, F}
        assert lanelets_ahead(A1, 300.0)[:2] == [A1, A2]

    def test_a_short_reach_is_the_start_alone(self, loaded: None) -> None:
        assert lanelets_ahead(A1, 10.0) == [A1]

    def test_from_the_lane_beside(self, loaded: None) -> None:
        assert lanelets_ahead(B1, 300.0) == [B1, B2]


# ---------------------------------------------------------------------------
# NotCondition(holds_when_pending=True)
# ---------------------------------------------------------------------------


class _Fixed(BaseCondition):
    def __init__(self, result: ScenarioResult | None) -> None:
        super().__init__(label="child")
        self.result = result

    def check(self, world: Any, elapsed: float) -> ScenarioResult | None:
        return self.result


class TestLogicalNot:
    def test_holds_while_the_child_is_pending(self) -> None:
        result = NotCondition(_Fixed(None), holds_when_pending=True).check(
            MagicMock(), 1.5
        )
        assert result is not None and result.passed
        assert result.elapsed_seconds == 1.5

    @pytest.mark.parametrize("passed", [True, False])
    def test_pending_while_the_child_has_fired(self, passed: bool) -> None:
        child = _Fixed(ScenarioResult(passed=passed, message="", elapsed_seconds=0))
        assert (
            NotCondition(child, holds_when_pending=True).check(MagicMock(), 1) is None
        )

    def test_the_default_still_flips_the_verdict(self) -> None:
        assert NotCondition(_Fixed(None)).check(MagicMock(), 1) is None
        child = _Fixed(ScenarioResult(passed=True, message="", elapsed_seconds=0))
        result = NotCondition(child).check(MagicMock(), 1)
        assert result is not None and not result.passed

    def test_details_say_which_not(self) -> None:
        assert NotCondition(_Fixed(None), holds_when_pending=True).get_details()[
            "holds_when_pending"
        ]
        assert "holds_when_pending" not in NotCondition(_Fixed(None)).get_details()


# ---------------------------------------------------------------------------
# The composition: the ego left the lanes ahead of it
# ---------------------------------------------------------------------------


class _Ego:
    """A world holding one ego, moved by :meth:`at` (Lanelet2 frame)."""

    def __init__(self) -> None:
        self.location = MagicMock(x=0.0, y=0.0, z=0.0)
        actor = MagicMock()
        actor.attributes = {"role_name": str(EGO_ROLE_NAME)}
        actor.get_location.return_value = self.location
        self.world = MagicMock()
        self.world.get_actors.return_value = [actor]

    def at(self, x: float, y: float) -> Any:
        self.location.x, self.location.y = x, -y
        return self.world


def _run(
    condition: BaseCondition, path: list[tuple[float, float]], dt: float = 0.05
) -> ScenarioResult | None:
    """Drive the ego along *path*, one point a tick; the first result, if any."""
    ego = _Ego()
    for tick, (x, y) in enumerate(path):
        result = condition.check(ego.at(x, y), tick * dt)
        if result is not None:
            return result
    return None


@pytest.fixture
def left(loaded: None) -> Iterator[BaseCondition]:
    # Building a lanelet position resolves its OpenDRIVE lane, which the
    # synthetic map has none of; the lanelet test does not use it.
    with unittest.mock.patch(
        "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
        return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
    ):
        yield left_lanes_ahead(A1, 300.0, 0.2)


class TestLeftLanesAhead:
    def test_it_is_built_from_the_framework_conditions(
        self, left: BaseCondition
    ) -> None:
        assert isinstance(left, PersistentCondition)
        assert left.label == "ego_left_its_lane"
        details = left.get_details()
        assert details["required_duration_seconds"] == 0.2
        not_ = details["child"]
        assert not_["condition_type"] == "NotCondition"
        assert not_["holds_when_pending"] is True
        lanelets = [c["lanelet_id"] for c in not_["child"]["children"]]
        assert set(lanelets) == {A1, A2, A3, F}

    def test_staying_in_the_lane_is_not_leaving_it(self, left: BaseCondition) -> None:
        assert _run(left, [(5.0 + i, 1.75) for i in range(40)]) is None

    def test_across_a_seam_into_the_successor(self, left: BaseCondition) -> None:
        assert _run(left, [(40.0 + i, 1.75) for i in range(30)]) is None

    def test_down_a_fork_ahead(self, left: BaseCondition) -> None:
        # Along F's centreline, from (100, 1.75) to (141.75, -20).
        path = [(95.0 + i, 1.75) for i in range(5)] + [
            (100.0 + 41.75 * k / 10, 1.75 - 21.75 * k / 10) for k in range(1, 10)
        ]
        assert _run(left, path) is None

    def test_drifting_into_the_lane_beside(self, left: BaseCondition) -> None:
        # Into B1 on the fifth tick (t = 0.20 s), held there.
        path = [(10.0 + i, 1.75) for i in range(4)] + [
            (14.0 + i, 5.25) for i in range(20)
        ]
        result = _run(left, path)
        assert result is not None and result.passed
        # Off from t = 0.20 s; held for 0.2 s it fails at t = 0.40 s.
        assert result.elapsed_seconds == pytest.approx(0.40)
        assert "since 0.20s" in result.message

    def test_leaving_the_road(self, left: BaseCondition) -> None:
        path = [(10.0, 1.75)] + [(11.0, -5.0)] * 10
        result = _run(left, path)
        assert result is not None
        assert result.elapsed_seconds == pytest.approx(0.25)

    def test_a_tick_off_is_not_leaving(self, left: BaseCondition) -> None:
        path = [(10.0, 1.75), (11.0, 1.75), (12.0, 5.25), (13.0, 1.75)] * 5
        assert _run(left, path) is None

    def test_fewer_ticks_off_than_the_debounce(self, left: BaseCondition) -> None:
        # Three ticks (0.15 s) off, then back, again and again.
        path = ([(10.0, 1.75)] + [(10.0, 5.25)] * 3) * 5
        assert _run(left, path) is None

    def test_no_debounce_fails_on_the_first_tick_off(self, loaded: None) -> None:
        with unittest.mock.patch(
            "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
            return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
        ):
            condition = left_lanes_ahead(A1, 300.0, 0.0)
        assert isinstance(condition, NotCondition)
        result = _run(condition, [(10.0, 1.75), (10.0, 5.25)])
        assert result is not None and result.elapsed_seconds == pytest.approx(0.05)

    def test_one_lanelet_needs_no_or(self, loaded: None) -> None:
        with unittest.mock.patch(
            "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
            return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
        ):
            condition = left_lanes_ahead(A1, 10.0, 0.2)
        child = condition.get_details()["child"]["child"]
        assert child["condition_type"] == "EntityLanePositionCondition"
        assert _run(condition, [(10.0, 1.75)] * 10) is None


# ---------------------------------------------------------------------------
# cut_in's fail conditions
# ---------------------------------------------------------------------------


def _cut_in_conditions(
    config: CutInConfig,
) -> tuple[list[BaseCondition], list[BaseCondition]]:
    """``CutInScenario.setup()``'s pass and fail conditions, CARLA and the map stubbed."""
    from autoware_carla_scenario import EgoConfig, Lanelet2Pose, SpawnTransform
    from autoware_carla_scenario.examples import cut_in

    import typesafe_carla.carla as carla

    scenario = cut_in.CutInScenario(
        EgoConfig(
            spawn_location=SpawnTransform(carla.Transform(carla.Location())),
            vehicle_type="vehicle.mini.cooper",
        ),
        spawn_pose=Lanelet2Pose(lanelet_id=A1, s=5.0),
        config=config,
    )
    scenario.set_client(MagicMock(), 8000)
    stub = MagicMock()
    with (
        unittest.mock.patch.object(scenario, "_setup_ego_spawn", return_value=stub),
        unittest.mock.patch.object(cut_in, "snap_to_carla_road", return_value=stub),
        unittest.mock.patch.object(cut_in, "to_opendrive", return_value=stub),
        unittest.mock.patch.object(cut_in, "VehicleEntity"),
        unittest.mock.patch(
            "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
            return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
        ),
        unittest.mock.patch.object(
            cut_in, "lanelets_ahead", wraps=lanelets_ahead
        ) as ahead,
        # lanes_ahead_before_junction is the framework's, and calls its own
        # module's lanelets_ahead: counted with the example's.
        unittest.mock.patch(
            "autoware_carla_scenario.coordinate.transform.lanelets_ahead", ahead
        ),
    ):
        scenario.setup()
    # Once for the lanes the NPC cuts in on; for the ones the ego is to stay
    # on, once more and once for the lanes before the junction, whose
    # neighbours are left out of them.
    calls = 3 if config.fail_on_leaving_lane else 1
    assert (
        ahead.call_args_list == [unittest.mock.call(A1, config.lanes_ahead_m)] * calls
    )
    return scenario._pass_conditions, scenario._fail_conditions


def _cut_in_fail_conditions(config: CutInConfig) -> list[BaseCondition]:
    return _cut_in_conditions(config)[1]


class TestCutInFailConditions:
    def test_by_default_any_collision_and_leaving_the_lane_fail(
        self, loaded: None
    ) -> None:
        fails = _cut_in_fail_conditions(CutInConfig(npc_lanelet_id=B1))
        assert [c.label for c in fails] == [
            "ego_collided",
            "ego_left_its_lane",
            "scenario_timeout",
        ]
        collision = fails[0]
        assert isinstance(collision, CollisionCondition)
        assert "target" not in collision.get_details()
        left = fails[1]
        assert isinstance(left, PersistentCondition)
        assert left.get_details()["required_duration_seconds"] == 0.2
        assert isinstance(fails[2], TimeoutCondition)

    def test_both_can_be_switched_off(self, loaded: None) -> None:
        fails = _cut_in_fail_conditions(
            CutInConfig(
                npc_lanelet_id=B1,
                fail_on_leaving_lane=False,
                fail_on_any_collision=False,
            )
        )
        assert [c.label for c in fails] == ["hit_cut_in_vehicle", "scenario_timeout"]
        assert fails[0].get_details()["target"] == "npc1"

    def test_the_debounce_is_configurable(self, loaded: None) -> None:
        fails = _cut_in_fail_conditions(
            CutInConfig(npc_lanelet_id=B1, leaving_lane_seconds=0.5)
        )
        assert fails[1].get_details()["required_duration_seconds"] == 0.5


class TestCutInPassCondition:
    def test_the_npc_on_any_of_the_lanes_ahead_is_cut_in(self, loaded: None) -> None:
        passes, _ = _cut_in_conditions(CutInConfig(npc_lanelet_id=B1))
        assert [c.label for c in passes] == ["npc_cut_in_held"]
        sticky = passes[0].get_details()["child"]
        assert sticky["label"] == "npc_in_ego_lane_sticky"
        on_any = sticky["child"]
        assert on_any["condition_type"] == "OrCondition"
        # Not F: a junction lanelet, which the lane beside's overlap.
        assert {c["lanelet_id"] for c in on_any["children"]} == {A1, A2, A3}

    def test_lanes_ahead_stop_at_a_junction(self, loaded: None) -> None:
        assert set(lanelets_ahead(A1, 300.0)) == {A1, A2, A3, F}
        assert lanes_ahead_before_junction(A1, 300.0) == [
            c for c in lanelets_ahead(A1, 300.0) if c != F
        ]
        assert lanes_ahead_before_junction(A1, 60.0) == [A1, A2]

    def test_one_lanelet_needs_no_or(self, loaded: None) -> None:
        with unittest.mock.patch(
            "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
            return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
        ):
            condition = npc_in_lanes_ahead(A1, 10.0)
        assert isinstance(condition, StickyCondition)
        assert condition.label == "npc_in_ego_lane_sticky"
        child = condition.get_details()["child"]
        assert child["condition_type"] == "EntityLanePositionCondition"
        assert child["lanelet_id"] == A1


# ---------------------------------------------------------------------------
# cut_in's pass condition on the fixture map (#1078)
# ---------------------------------------------------------------------------

#: The nishishinjuku fixture map, kept at the repository root.
_DATA = Path(__file__).resolve().parents[3] / "data"

#: cut_in/left.yaml's fixed case: the ego on lanelet 20, the NPC beside it on
#: lanelet 19.
EGO_LANELET, NPC_LANELET = 20, 19


@pytest.fixture(scope="class")
def nishishinjuku() -> Iterator[None]:
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(_DATA / "nishishinjuku_carla.xodr", _DATA / "nishishinjuku.osm")
    yield
    # Release lanelet2/pyxodr objects while the C++ runtime is still valid.
    manager._routing_graph = None
    manager._lanelet_map = None
    manager._road_network = None
    MapManager._instance = saved


def _npc_at(pose: Lanelet2Pose) -> Any:
    """A world holding the cut-in NPC at *pose*."""
    where = to_carla_world(pose)
    actor = MagicMock()
    actor.attributes = {"role_name": str(CUT_IN_ROLE)}
    actor.get_location.return_value = MagicMock(x=where.x, y=where.y, z=where.z)
    world = MagicMock()
    world.get_actors.return_value = [actor]
    return world


@pytest.mark.usefixtures("nishishinjuku")
class TestCutInOnTheFixtureMap:
    """The NPC that finishes its lane change past the ego's spawn lanelet."""

    def _cut_in(self, pose: Lanelet2Pose, ego_lanelet: int = EGO_LANELET) -> bool:
        condition = npc_in_lanes_ahead(ego_lanelet, CutInConfig().lanes_ahead_m)
        return condition.check(_npc_at(pose), 3.0) is not None

    def test_on_the_spawn_lanelet(self) -> None:
        assert self._cut_in(Lanelet2Pose(lanelet_id=EGO_LANELET, s=30.0))

    def test_on_the_lanelet_after_it(self) -> None:
        successor = lanelets_ahead(EGO_LANELET, 1000.0)[1]
        assert successor != EGO_LANELET
        assert self._cut_in(Lanelet2Pose(lanelet_id=successor, s=10.0))

    def test_not_in_the_lane_beside(self) -> None:
        beside_next = lanelets_ahead(NPC_LANELET, 1000.0)[1]
        assert beside_next not in lanelets_ahead(EGO_LANELET, 1000.0)
        assert not self._cut_in(Lanelet2Pose(lanelet_id=NPC_LANELET, s=30.0))
        assert not self._cut_in(Lanelet2Pose(lanelet_id=beside_next, s=10.0))

    def test_not_the_lane_beside_crossing_a_junction(self) -> None:
        # The ego on 225, the NPC beside it on 226: driving straight on, the
        # NPC is on 360 in the junction, inside 359, which the ego's lane
        # turns onto.  It has not cut in.
        ego, npc_beside, npc_on, overlapped = 225, 226, 360, 359
        assert npc_on in lanelets_ahead(npc_beside, 1000.0)
        assert overlapped in lanelets_ahead(ego, 1000.0)
        pose = Lanelet2Pose(lanelet_id=npc_on, s=30.0)
        assert on_lanelet(to_carla_world(pose), overlapped)
        assert not self._cut_in(pose, ego_lanelet=ego)

    def test_not_past_a_junction(self) -> None:
        # The ego on 3002098, the NPC beside it on 3002099: past the
        # junction, 3002092 runs inside 3002020, which follows the ego's lane
        # through it.  The NPC there has not cut in.
        ego, npc_beside, npc_on, overlapped = 3002098, 3002099, 3002092, 3002020
        assert npc_on in lanelets_ahead(npc_beside, 1000.0)
        assert overlapped in lanelets_ahead(ego, 1000.0)
        pose = Lanelet2Pose(lanelet_id=npc_on, s=67.0)
        assert on_lanelet(to_carla_world(pose), overlapped)
        assert not self._cut_in(pose, ego_lanelet=ego)


def _ego_at(pose: Lanelet2Pose) -> Any:
    """A world holding the ego at *pose*."""
    where = to_carla_world(pose)
    actor = MagicMock()
    actor.attributes = {"role_name": str(EGO_ROLE_NAME)}
    actor.get_location.return_value = MagicMock(x=where.x, y=where.y, z=where.z)
    world = MagicMock()
    world.get_actors.return_value = [actor]
    return world


@pytest.mark.usefixtures("nishishinjuku")
class TestLeftLanesAheadOnTheFixtureMap:
    """The lanes ahead that loop round a block into the lane beside (#1080)."""

    #: cut_in/left's case on lanelet 133: its lane on the left is 132, which
    #: lanelets_ahead(133, 1000) reaches round the block.
    SPAWN, BESIDE = 133, 132

    def test_the_loop_reaches_the_lane_beside(self) -> None:
        assert self.BESIDE in lanelets_ahead(self.SPAWN, CutInConfig().lanes_ahead_m)

    def test_no_lanelet_beside_the_lane_is_the_egos(self) -> None:
        reach = CutInConfig().lanes_ahead_m
        kept = ego_lanes_ahead(self.SPAWN, reach)
        assert self.BESIDE not in kept
        own = lanes_ahead_before_junction(self.SPAWN, reach)
        assert set(own) <= set(kept)
        assert kept[0] == self.SPAWN

    def _left(self, lanelet: int) -> bool:
        condition = left_lanes_ahead(self.SPAWN, CutInConfig().lanes_ahead_m, 0.0)
        pose = Lanelet2Pose(lanelet_id=lanelet, s=60.0)
        return condition.check(_ego_at(pose), 1.0) is not None

    def test_swerving_into_the_lane_beside_fails(self) -> None:
        assert self._left(self.BESIDE)

    def test_staying_in_the_lane_does_not(self) -> None:
        assert not self._left(self.SPAWN)


@pytest.mark.usefixtures("nishishinjuku")
class TestATurnBesideTheLaneOnTheFixtureMap:
    """A turn through the junction ahead that ends beside the ego's lane is
    one of its lanes, not the lane beside it."""

    #: From lanelet 3002156, the right turn 3002162 -> 3002161 runs beside
    #: 3002160, which follows 3002156 before the junction.
    SPAWN, LANE, TURN = 3002156, 3002160, 3002161

    def test_the_turn_is_beside_the_lane(self) -> None:
        reach = CutInConfig().lanes_ahead_m
        assert self.LANE in lanes_ahead_before_junction(self.SPAWN, reach)
        assert self.TURN not in lanes_ahead_before_junction(self.SPAWN, reach)
        mm = MapManager.get_instance()
        lane = mm.lanelet_map.laneletLayer[self.LANE]
        assert int(mm.routing_graph.adjacentRight(lane).id) == self.TURN

    def test_the_turn_is_kept(self) -> None:
        assert self.TURN in ego_lanes_ahead(self.SPAWN, CutInConfig().lanes_ahead_m)

    def test_taking_the_turn_does_not_fail(self) -> None:
        condition = left_lanes_ahead(self.SPAWN, CutInConfig().lanes_ahead_m, 0.0)
        pose = Lanelet2Pose(lanelet_id=self.TURN, s=20.0)
        assert condition.check(_ego_at(pose), 1.0) is None


# ---------------------------------------------------------------------------
# The same composition in a scenario document
# ---------------------------------------------------------------------------


def _left_lanes_node(lanelets: list[int], seconds: float) -> Any:
    from autoware_carla_scenario.authoring.models import ConditionNode

    return ConditionNode(
        type="persistent",
        params={"duration": seconds},
        children=[
            ConditionNode(
                type="not",
                params={"holds_when_pending": True},
                children=[
                    ConditionNode(
                        type="any",
                        children=[
                            ConditionNode(
                                type="entity_lane_position",
                                params={"entity": "ego", "lanelet_id": lanelet},
                            )
                            for lanelet in lanelets
                        ],
                    )
                ],
            )
        ],
    )


class TestTheDocumentRecipe:
    def test_a_document_says_it_with_the_registered_cards(self, loaded: None) -> None:
        from autoware_carla_scenario.authoring.builders import instantiate_condition
        from autoware_carla_scenario.authoring.compiler import (
            BuildContext,
            compile_document,
        )
        from autoware_carla_scenario.authoring.starter import new_document

        document = new_document()
        document.assertions.fail_conditions = [_left_lanes_node([A1, A2, A3, F], 0.2)]
        compiled = compile_document(document)
        with unittest.mock.patch(
            "autoware_carla_scenario.conditions.composition.entity_lane_position.to_opendrive",
            return_value=OpenDrivePose(road_id="0", lane_id=-1, s=0.0),
        ):
            condition = instantiate_condition(
                compiled.fail_conditions[0], BuildContext(scenario=None)
            )
        assert isinstance(condition, PersistentCondition)
        assert _run(condition, [(40.0 + i, 1.75) for i in range(30)]) is None
        result = _run(condition, [(10.0, 1.75)] + [(10.0, 5.25)] * 10)
        assert result is not None and result.elapsed_seconds == pytest.approx(0.25)

    def test_the_not_card_defaults_to_flipping_the_verdict(self) -> None:
        from autoware_carla_scenario.authoring.builders import instantiate_condition
        from autoware_carla_scenario.authoring.compiler import (
            BuildContext,
            compile_document,
        )
        from autoware_carla_scenario.authoring.models import ConditionNode
        from autoware_carla_scenario.authoring.starter import new_document

        document = new_document()
        document.assertions.fail_conditions = [
            ConditionNode(
                type="not",
                children=[
                    ConditionNode(type="elapsed_time", params={"duration_seconds": 5})
                ],
            )
        ]
        compiled = compile_document(document)
        condition = instantiate_condition(
            compiled.fail_conditions[0], BuildContext(scenario=None)
        )
        assert isinstance(condition, NotCondition)
        assert "holds_when_pending" not in condition.get_details()
