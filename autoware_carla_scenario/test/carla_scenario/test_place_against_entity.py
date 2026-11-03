"""A place written against an entity (#1271): a ``RelativeLanePose``, or a
route pose without an anchor, given to ``PlaceClearCondition`` or
``EntityPositionDistanceCondition``.

It is placed against where that entity is now, afresh on every tick; while
the entity is not in the scene (absent, or parked under the map) the place is
nowhere and the condition does not hold, quietly.  A fixed place is resolved
once, as before.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.conditions import (
    EntityPositionDistanceCondition,
    PlaceClearCondition,
)
from autoware_carla_scenario.conditions import place_clear as place_module
from autoware_carla_scenario.conditions.entity_existence import (
    ReferenceAbsentError,
    ReferencePoses,
)
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, Lanelet2Pose
from autoware_carla_scenario.trajectory import resolve as resolve_module
from autoware_carla_scenario.trajectory.model import (
    MapPose,
    RelativeLanePose,
    RouteCrossingPose,
    RouteLanePose,
    RouteOppositePose,
    RouteRoadsidePose,
    placed_against_entity,
)

_AHEAD = RelativeLanePose(ds=20.0, entity_ref="Ego")
#: The heading (deg) of the lane every faked place lies on: not the
#: reference actor's (90).
_LANE_YAW = 30.0


def _actor(role: str, x: float, y: float = 0.0, z: float = 0.0) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(x, y, z)
    actor.get_transform.return_value = carla.Transform(
        carla.Location(x, y, z), carla.Rotation(yaw=90.0)
    )
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _fake_resolve(calls: list[Any]):
    """``resolve_position`` placing a relative pose ``ds`` metres along x
    from its reference, as the real one does through the reference it is
    given, on a lane heading :data:`_LANE_YAW`; with no ``yaw`` of its own
    it has none, as the real one does.  A fixed place at (100, 0)."""

    def lane_yaw(yaw: Optional[float]) -> Optional[float]:
        return None if yaw is None else _LANE_YAW + yaw

    def resolve(
        position: Any, reference: Optional[Any] = None
    ) -> tuple[float, float, Optional[float], Optional[float]]:
        calls.append(position)
        if isinstance(position, RelativeLanePose):
            if reference is None:
                raise ValueError("no reference pose was given to resolve it with")
            ref = reference(position.entity_ref)
            return (ref.x + position.ds, ref.y, ref.z, lane_yaw(position.yaw))
        if isinstance(position, RouteLanePose) and position.anchor is None:
            if reference is None:
                raise ValueError("no reference pose was given to resolve it with")
            ego = reference("Ego")
            return (ego.x + position.ds, ego.y, ego.z, lane_yaw(position.yaw))
        return (100.0, 0.0, 0.0, 0.0)

    return resolve


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    monkeypatch.setattr(resolve_module, "resolve_position", _fake_resolve(calls))
    return calls


@pytest.fixture
def looked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float, float]]:
    seen: list[tuple[float, float, float]] = []

    def look(
        world: Any, x: float, y: float, yaw: float, hl: float, hw: float, m: float
    ) -> int:
        seen.append((x, y, yaw))
        return -1

    monkeypatch.setattr(place_module, "_occupant", look)
    return seen


class TestPlacedAgainstEntity:
    @pytest.mark.parametrize(
        "position",
        [
            RelativeLanePose(ds=5.0, entity_ref="npc1"),
            RouteLanePose(ds=5.0),
            RouteOppositePose(ds=5.0),
            RouteRoadsidePose(ds=5.0),
        ],
    )
    def test_a_place_counted_from_an_entity_moves_with_it(self, position: Any) -> None:
        assert placed_against_entity(position)

    @pytest.mark.parametrize(
        "position",
        [
            CarlaWorldPose(x=1.0, y=2.0, z=0.0),
            Lanelet2Pose(3, 5.0),
            MapPose(x=1.0, y=2.0),
            RouteLanePose(ds=5.0, anchor="start"),
            RouteOppositePose(ds=5.0, anchor="start"),
            RouteRoadsidePose(ds=5.0, anchor="start"),
            RouteCrossingPose(approach="same", turn="right"),
        ],
    )
    def test_any_other_place_is_fixed(self, position: Any) -> None:
        assert not placed_against_entity(position)


class TestResolveAlongLane:
    """``resolve_along_lane`` resolves a pose with no yaw as the same pose
    facing along its lane (``yaw=0.0``); any other as it is."""

    @staticmethod
    def _passed(monkeypatch: pytest.MonkeyPatch, position: Any) -> Any:
        passed: list[Any] = []
        reference = object()

        def resolve(given: Any, ref: Any = None) -> tuple[float, float, None, float]:
            assert ref is reference
            passed.append(given)
            return (1.0, 2.0, None, 3.0)

        monkeypatch.setattr(resolve_module, "resolve_position", resolve)
        assert resolve_module.resolve_along_lane(position, reference) == (  # type: ignore[arg-type]
            1.0,
            2.0,
            None,
            3.0,
        )
        (given,) = passed
        return given

    @pytest.mark.parametrize(
        "position",
        [
            RelativeLanePose(ds=5.0, offset=0.5, d_lane=1, entity_ref="npc1"),
            RouteLanePose(ds=5.0, offset=0.5, d_lane=-1),
            RouteOppositePose(ds=5.0, lane=2, offset=0.5, anchor="start"),
            RouteRoadsidePose(ds=5.0, side="right", kerb_distance=1.0),
        ],
    )
    def test_no_yaw_becomes_along_the_lane(
        self, monkeypatch: pytest.MonkeyPatch, position: Any
    ) -> None:
        import dataclasses

        given = self._passed(monkeypatch, position)
        assert given == dataclasses.replace(position, yaw=0.0)

    @pytest.mark.parametrize(
        "position",
        [
            RelativeLanePose(ds=5.0, yaw=0.3, entity_ref="npc1"),
            RouteLanePose(ds=5.0, yaw=-0.2),
            MapPose(x=1.0, y=2.0),
            Lanelet2Pose(3, 5.0),
            RouteCrossingPose(approach="same", turn="right"),
        ],
    )
    def test_any_other_is_resolved_as_it_is(
        self, monkeypatch: pytest.MonkeyPatch, position: Any
    ) -> None:
        assert self._passed(monkeypatch, position) is position


class TestReferencePoses:
    def test_it_reads_where_the_entity_stands(self) -> None:
        pose = ReferencePoses(_world(_actor("Ego", 3.0, 4.0))).pose("Ego")
        assert (pose.x, pose.y, pose.yaw) == pytest.approx((3.0, 4.0, 90.0))

    def test_an_absent_entity_is_a_reference_absent_error(self) -> None:
        with pytest.raises(ReferenceAbsentError, match="'Ego' is not in the world"):
            ReferencePoses(_world()).pose("Ego")

    def test_a_hidden_entity_is_absent_only_when_asked(self) -> None:
        hidden = _world(_actor("npc1", 3.0, 4.0, -500.0))
        assert ReferencePoses(hidden).pose("npc1").z == pytest.approx(-500.0)
        with pytest.raises(ReferenceAbsentError):
            ReferencePoses(hidden, present_only=True).pose("npc1")

    def test_no_entity_is_not_an_absence(self) -> None:
        with pytest.raises(ValueError, match="relative to no entity") as raised:
            ReferencePoses(_world()).pose(None)
        assert not isinstance(raised.value, ReferenceAbsentError)


class TestPlaceClear:
    def test_a_relative_place_follows_its_reference(
        self, resolved: list[Any], looked: list[tuple[float, float, float]]
    ) -> None:
        condition = PlaceClearCondition(_AHEAD, 5.0, 2.0, label="ahead_of_ego")
        for tick, x in enumerate((0.0, 10.0, 25.0)):
            result = condition.check(_world(_actor("Ego", x)), 0.1 * tick)
            assert result is not None and result.passed
        # Along the lane it is placed on, not the reference's heading.
        assert looked == [(20.0, 0.0, 30.0), (30.0, 0.0, 30.0), (45.0, 0.0, 30.0)]
        assert len(resolved) == 3
        details = condition.get_details()
        assert (details["x"], details["y"]) == (45.0, 0.0)

    def test_a_route_pose_counted_from_the_ego_follows_it(
        self, resolved: list[Any], looked: list[tuple[float, float, float]]
    ) -> None:
        condition = PlaceClearCondition(RouteLanePose(ds=8.0), 5.0, 2.0, label="p")
        condition.check(_world(_actor("Ego", 0.0)), 0.0)
        condition.check(_world(_actor("Ego", 4.0)), 0.1)
        assert looked == [(8.0, 0.0, 30.0), (12.0, 0.0, 30.0)]

    @pytest.mark.parametrize("ego", [None, -500.0], ids=["absent", "hidden"])
    def test_an_absent_or_hidden_reference_holds_nothing_quietly(
        self,
        resolved: list[Any],
        looked: list[tuple[float, float, float]],
        caplog: pytest.LogCaptureFixture,
        ego: Optional[float],
    ) -> None:
        condition = PlaceClearCondition(_AHEAD, 5.0, 2.0, label="ahead_of_ego")
        world = _world() if ego is None else _world(_actor("Ego", 0.0, 0.0, ego))
        with caplog.at_level(logging.DEBUG, logger=place_module.__name__):
            assert condition.check(world, 0.0) is None
            assert condition.check(world, 0.1) is None
        assert looked == []
        assert [r for r in caplog.records if r.name == place_module.__name__] == []
        assert "x" not in condition.get_details()
        # Placed as soon as it is there.
        result = condition.check(_world(_actor("Ego", 5.0)), 0.2)
        assert result is not None and result.passed
        assert looked == [(25.0, 0.0, 30.0)]

    def test_a_yaw_of_its_own_is_kept(
        self, resolved: list[Any], looked: list[tuple[float, float, float]]
    ) -> None:
        pose = RelativeLanePose(ds=20.0, yaw=10.0, entity_ref="Ego")
        PlaceClearCondition(pose, 5.0, 2.0, label="p").check(
            _world(_actor("Ego", 0.0)), 0.0
        )
        assert looked == [(20.0, 0.0, 40.0)]
        assert resolved == [pose]  # resolved as written

    def test_the_place_is_nowhere_again_once_its_reference_goes(
        self, resolved: list[Any], looked: list[tuple[float, float, float]]
    ) -> None:
        condition = PlaceClearCondition(_AHEAD, 5.0, 2.0, label="ahead_of_ego")
        assert condition.check(_world(_actor("Ego", 0.0)), 0.0) is not None
        assert condition.check(_world(), 0.1) is None
        assert "x" not in condition.get_details()

    def test_a_relative_place_that_does_not_resolve_is_tried_again_said_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        looked: list[tuple[float, float, float]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        calls: list[Any] = []
        fake = _fake_resolve(calls)
        failing = {"on": True}

        def resolve(position: Any, reference: Any = None) -> Any:
            if failing["on"]:
                calls.append(position)
                raise ValueError("off the map")
            return fake(position, reference)

        monkeypatch.setattr(resolve_module, "resolve_position", resolve)
        condition = PlaceClearCondition(_AHEAD, 5.0, 2.0, label="ahead_of_ego")
        world = _world(_actor("Ego", 0.0))
        with caplog.at_level(logging.WARNING, logger=place_module.__name__):
            assert condition.check(world, 0.0) is None
            assert condition.check(world, 0.1) is None
        assert len(calls) == 2
        assert [r.name for r in caplog.records] == [place_module.__name__]
        failing["on"] = False
        assert condition.check(world, 0.2) is not None

    def test_a_fixed_place_is_resolved_once(
        self, resolved: list[Any], looked: list[tuple[float, float, float]]
    ) -> None:
        condition = PlaceClearCondition(Lanelet2Pose(3, 5.0), 5.0, 2.0, label="p")
        condition.check(_world(_actor("Ego", 0.0)), 0.0)
        condition.check(_world(_actor("Ego", 50.0)), 0.1)
        assert len(resolved) == 1
        assert looked == [(100.0, 0.0, 0.0)] * 2

    def test_through_the_real_resolver_an_absent_reference_is_nowhere(
        self,
        monkeypatch: pytest.MonkeyPatch,
        looked: list[tuple[float, float, float]],
    ) -> None:
        """Through ``resolve_position`` itself, its relative-lane lookups faked:
        the reference the condition hands it is read, and its absence is not
        an unplaceable place."""
        from autoware_carla_scenario.coordinate import map_manager, transform
        from autoware_carla_scenario.trajectory import relative_lane

        monkeypatch.setattr(
            relative_lane,
            "reference_lane_pose_of",
            lambda pose: Lanelet2Pose(1, float(pose.x)),
        )
        monkeypatch.setattr(
            relative_lane,
            "relative_lane_pose",
            lambda lanelet_map, graph, lane, relative: Lanelet2Pose(
                1, lane.s + relative.ds
            ),
        )
        monkeypatch.setattr(
            transform,
            "to_carla_world",
            lambda lane: CarlaWorldPose(x=lane.s, y=0.0, z=0.0, yaw=45.0),
        )
        monkeypatch.setattr(
            map_manager.MapManager, "get_instance", classmethod(lambda cls: MagicMock())
        )
        condition = PlaceClearCondition(_AHEAD, 5.0, 2.0, label="ahead_of_ego")
        assert condition.check(_world(), 0.0) is None
        assert condition.check(_world(_actor("Ego", 10.0)), 0.1) is not None
        assert condition.check(_world(_actor("Ego", 12.0)), 0.2) is not None
        # Facing along the lane it is placed on (45 deg), with no yaw given.
        assert looked == [(30.0, 0.0, 45.0), (32.0, 0.0, 45.0)]


class TestEntityPositionDistance:
    @staticmethod
    def _near(position: Any = _AHEAD) -> EntityPositionDistanceCondition:
        return EntityPositionDistanceCondition("npc1", position, 3.0, label="near")

    def test_a_relative_place_follows_its_reference(self, resolved: list[Any]) -> None:
        condition = self._near()
        # The place is 20 m ahead of the ego; npc1 stands still at x=30.
        assert (
            condition.check(_world(_actor("Ego", 0.0), _actor("npc1", 30.0)), 0.0)
            is None
        )
        result = condition.check(_world(_actor("Ego", 9.0), _actor("npc1", 30.0)), 0.1)
        assert result is not None and result.passed
        assert "1.00 m" in result.message
        assert math.isclose(condition.get_details()["target_x"], 29.0)
        assert (
            condition.check(_world(_actor("Ego", 20.0), _actor("npc1", 30.0)), 0.2)
            is None
        )
        assert len(resolved) == 3

    @pytest.mark.parametrize("ego", [None, -500.0], ids=["absent", "hidden"])
    def test_an_absent_or_hidden_reference_holds_nothing_quietly(
        self,
        resolved: list[Any],
        caplog: pytest.LogCaptureFixture,
        ego: Optional[float],
    ) -> None:
        from autoware_carla_scenario.conditions.composition import (
            entity_position_distance as distance_module,
        )

        condition = self._near()
        others = [_actor("npc1", 20.0)]
        if ego is not None:
            others.append(_actor("Ego", 0.0, 0.0, ego))
        with caplog.at_level(logging.DEBUG, logger=distance_module.__name__):
            assert condition.check(_world(*others), 0.0) is None
            assert condition.check(_world(*others), 0.1) is None
        assert [r for r in caplog.records if r.name == distance_module.__name__] == []
        assert condition.get_details()["position"] == "RelativeLanePose"
        result = condition.check(_world(_actor("Ego", 0.0), _actor("npc1", 20.0)), 0.2)
        assert result is not None and result.passed

    def test_a_fixed_place_is_resolved_once(self, resolved: list[Any]) -> None:
        condition = self._near(RouteLanePose(ds=5.0, anchor="start"))
        world = _world(_actor("Ego", 0.0), _actor("npc1", 99.0))
        assert condition.check(world, 0.0) is not None
        assert condition.check(world, 0.1) is not None
        assert len(resolved) == 1

    def test_a_reset_keeps_following(self, resolved: list[Any]) -> None:
        condition = self._near()
        condition.check(_world(_actor("Ego", 0.0), _actor("npc1", 20.0)), 0.0)
        condition.reset()
        result = condition.check(_world(_actor("Ego", 1.0), _actor("npc1", 30.0)), 0.0)
        assert result is None
        assert math.isclose(condition.get_details()["target_x"], 21.0)
