"""Unit tests for composition conditions (PersistentCondition, StandstillCondition, TemporaryStopCondition)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import (
    AndCondition,
    BaseCondition,
    EntityLanePositionCondition,
    OrCondition,
    PersistentCondition,
    ScenarioResult,
    StandstillCondition,
    TemporaryStopCondition,
)
from autoware_carla_scenario.conditions.comparison import (
    ComparisonRule,
    ScalarComparisonRule,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import (
    CarlaWorldPose,
    Lanelet2Pose,
    OpenDrivePose,
)

#: The converter's fixture map, which both packages' tests share.
_CONVERTER_TEST_DATA = (
    Path(__file__).resolve().parents[3]
    / "autoware_lanelet2_to_opendrive"
    / "test"
    / "data"
)
XODR_PATH = _CONVERTER_TEST_DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _CONVERTER_TEST_DATA / "nishishinjuku.osm"


@pytest.fixture
def without_a_map() -> Generator[None, None, None]:
    """Run a test with no MapManager, and give back the one that was there.

    Resolving an address with no map raises, which is what makes "no map was
    consulted" assertable.  The instance is restored afterwards because tests
    are handed to xdist workers individually: a worker may be holding a map
    another module's fixture loaded, and that fixture will not run again.
    """
    saved = MapManager._instance
    MapManager._instance = None
    yield
    MapManager._instance = saved


def _road_ids_of(condition: BaseCondition) -> list[str]:
    """Return the roads a condition tree ended up watching.

    Read off the summary the conditions publish themselves, so this does not
    have to know how the tree is nested.
    """

    def walk(node: object) -> Generator[str, None, None]:
        if isinstance(node, dict):
            if "road_id" in node:
                yield str(node["road_id"])
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    return list(walk(condition.to_summary_dict()))


# ---------------------------------------------------------------------------
# Test helper conditions
# ---------------------------------------------------------------------------


class AlwaysPassCondition(BaseCondition):
    """Test helper: always returns a passing result."""

    def __init__(self) -> None:
        super().__init__(label="always_pass")

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return ScenarioResult(
            passed=True, message="Always passes", elapsed_seconds=elapsed
        )


class AlwaysNoneCondition(BaseCondition):
    """Test helper: never triggers."""

    def __init__(self) -> None:
        super().__init__(label="always_none")

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return None


class AlwaysFailCondition(BaseCondition):
    """Test helper: always returns a failing result."""

    def __init__(self) -> None:
        super().__init__(label="always_fail")

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        return ScenarioResult(
            passed=False, message="Always fails", elapsed_seconds=elapsed
        )


class ToggleCondition(BaseCondition):
    """Test helper: alternates between pass and None based on a sequence."""

    def __init__(self, results: list[Optional[bool]]) -> None:
        super().__init__(label="toggle")
        self._results = results
        self._index = 0

    def check(self, world: object, elapsed: float) -> Optional[ScenarioResult]:
        if self._index >= len(self._results):
            return None
        val = self._results[self._index]
        self._index += 1
        if val is None:
            return None
        return ScenarioResult(
            passed=val, message=f"Toggle: {val}", elapsed_seconds=elapsed
        )


def _make_world_with_actor(
    role_name: str,
    x: float,
    y: float,
    z: float = 0.0,
    *,
    vx: float | None = None,
    vy: float | None = None,
    vz: float | None = None,
) -> MagicMock:
    """Return a MagicMock CARLA world that contains a single actor."""
    location = MagicMock()
    location.x = x
    location.y = y
    location.z = z

    actor = MagicMock()
    actor.attributes = {"role_name": role_name}
    actor.get_location.return_value = location

    if vx is not None:
        velocity = MagicMock()
        velocity.x = vx
        velocity.y = vy if vy is not None else 0.0
        velocity.z = vz if vz is not None else 0.0
        actor.get_velocity.return_value = velocity

    world = MagicMock()
    world.get_actors.return_value = [actor]
    return world


# ---------------------------------------------------------------------------
# PersistentCondition – unit tests
# ---------------------------------------------------------------------------


class TestPersistentCondition:
    def test_returns_none_before_duration(self) -> None:
        cond = PersistentCondition(AlwaysPassCondition(), duration=3.0)
        world = MagicMock()
        assert cond.check(world, elapsed=0.0) is None
        assert cond.check(world, elapsed=2.9) is None

    def test_returns_pass_at_duration(self) -> None:
        cond = PersistentCondition(AlwaysPassCondition(), duration=3.0)
        world = MagicMock()
        cond.check(world, elapsed=0.0)
        result = cond.check(world, elapsed=3.0)
        assert result is not None
        assert result.passed is True
        assert result.elapsed_seconds == pytest.approx(3.0)

    def test_returns_pass_beyond_duration(self) -> None:
        cond = PersistentCondition(AlwaysPassCondition(), duration=2.0)
        world = MagicMock()
        cond.check(world, elapsed=0.0)
        result = cond.check(world, elapsed=5.0)
        assert result is not None
        assert result.passed is True

    def test_timer_resets_on_none(self) -> None:
        """Timer resets when child returns None."""
        # Pass, Pass, None, Pass, Pass → needs restart
        toggle = ToggleCondition([True, True, None, True, True])
        cond = PersistentCondition(toggle, duration=1.5)
        world = MagicMock()

        cond.check(world, elapsed=0.0)  # True → start timer
        assert cond.check(world, elapsed=1.0) is None  # True, 1.0s < 1.5s
        cond.check(world, elapsed=1.2)  # None → reset
        cond.check(world, elapsed=2.0)  # True → restart timer
        assert cond.check(world, elapsed=3.0) is None  # True, but only 1.0s

    def test_timer_resets_on_fail(self) -> None:
        """Timer resets when child returns passed=False."""
        toggle = ToggleCondition([True, False, True, True])
        cond = PersistentCondition(toggle, duration=1.0)
        world = MagicMock()

        cond.check(world, elapsed=0.0)  # True → start
        cond.check(world, elapsed=0.5)  # False → reset
        cond.check(world, elapsed=1.0)  # True → restart
        assert cond.check(world, elapsed=1.5) is None  # True, only 0.5s since restart

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            PersistentCondition(AlwaysPassCondition(), duration=0.0)
        with pytest.raises(ValueError, match="duration must be positive"):
            PersistentCondition(AlwaysPassCondition(), duration=-1.0)

    def test_child_always_none(self) -> None:
        """Always returns None when child never triggers."""
        cond = PersistentCondition(AlwaysNoneCondition(), duration=1.0)
        world = MagicMock()
        assert cond.check(world, elapsed=0.0) is None
        assert cond.check(world, elapsed=100.0) is None

    def test_child_always_fails(self) -> None:
        """Always returns None when child always fails."""
        cond = PersistentCondition(AlwaysFailCondition(), duration=1.0)
        world = MagicMock()
        assert cond.check(world, elapsed=0.0) is None
        assert cond.check(world, elapsed=100.0) is None

    def test_message_contains_duration_info(self) -> None:
        cond = PersistentCondition(AlwaysPassCondition(), duration=2.0)
        world = MagicMock()
        cond.check(world, elapsed=0.0)
        result = cond.check(world, elapsed=2.0)
        assert result is not None
        assert "2.00s" in result.message

    def test_wraps_and_condition(self) -> None:
        """PersistentCondition can wrap a composite condition."""
        inner = AndCondition([AlwaysPassCondition(), AlwaysPassCondition()])
        cond = PersistentCondition(inner, duration=1.0)
        world = MagicMock()
        cond.check(world, elapsed=0.0)
        result = cond.check(world, elapsed=1.0)
        assert result is not None
        assert result.passed is True


# ---------------------------------------------------------------------------
# StandstillCondition (composition version) – unit tests
# ---------------------------------------------------------------------------


class TestStandstillConditionComposition:
    """Tests for the composition-based StandstillCondition.

    These match the original StandstillCondition test cases to ensure
    API compatibility.
    """

    def test_returns_none_before_duration(self) -> None:
        condition = StandstillCondition("ego", duration=3.0, label="test_standstill")
        world = _make_world_with_actor("ego", 0.0, 0.0, vx=0.0, vy=0.0)
        assert condition.check(world, elapsed=0.0) is None
        assert condition.check(world, elapsed=2.9) is None

    def test_returns_pass_after_duration(self) -> None:
        condition = StandstillCondition("ego", duration=3.0, label="test_standstill")
        world = _make_world_with_actor("ego", 0.0, 0.0, vx=0.0, vy=0.0)
        condition.check(world, elapsed=0.0)
        result = condition.check(world, elapsed=3.0)
        assert result is not None
        assert result.passed is True

    def test_timer_resets_when_moving(self) -> None:
        condition = StandstillCondition("ego", duration=2.0, label="test_standstill")
        world_stop = _make_world_with_actor("ego", 0.0, 0.0, vx=0.0, vy=0.0)
        world_move = _make_world_with_actor("ego", 0.0, 0.0, vx=5.0, vy=0.0)

        # Stand still from t=0 to t=1
        condition.check(world_stop, elapsed=0.0)
        assert condition.check(world_stop, elapsed=1.0) is None

        # Start moving at t=1.5 → timer resets
        assert condition.check(world_move, elapsed=1.5) is None

        # Stand still again from t=2
        condition.check(world_stop, elapsed=2.0)
        assert condition.check(world_stop, elapsed=3.0) is None  # only 1s
        result = condition.check(world_stop, elapsed=4.0)  # 2s standstill
        assert result is not None
        assert result.passed is True

    def test_speed_below_threshold_counts(self) -> None:
        condition = StandstillCondition(
            "ego", duration=1.0, speed_threshold=0.5, label="test_standstill"
        )
        # Speed = sqrt(0.3^2 + 0.3^2) ~ 0.42 < 0.5
        world = _make_world_with_actor("ego", 0.0, 0.0, vx=0.3, vy=0.3)
        condition.check(world, elapsed=0.0)
        result = condition.check(world, elapsed=1.0)
        assert result is not None
        assert result.passed is True

    def test_speed_above_threshold_no_trigger(self) -> None:
        condition = StandstillCondition(
            "ego", duration=1.0, speed_threshold=0.1, label="test_standstill"
        )
        world = _make_world_with_actor("ego", 0.0, 0.0, vx=1.0, vy=0.0)
        condition.check(world, elapsed=0.0)
        assert condition.check(world, elapsed=5.0) is None

    def test_entity_not_found_returns_none(self) -> None:
        condition = StandstillCondition("ego", duration=1.0, label="test_standstill")
        world = _make_world_with_actor("other", 0.0, 0.0, vx=0.0, vy=0.0)
        assert condition.check(world, elapsed=0.0) is None

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            StandstillCondition("ego", duration=0.0, label="test_standstill")
        with pytest.raises(ValueError, match="duration must be positive"):
            StandstillCondition("ego", duration=-1.0, label="test_standstill")

    def test_invalid_speed_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="speed_threshold must be non-negative"):
            StandstillCondition(
                "ego", duration=1.0, speed_threshold=-0.1, label="test_standstill"
            )

    def test_elapsed_seconds_in_result(self) -> None:
        condition = StandstillCondition("ego", duration=1.0, label="test_standstill")
        world = _make_world_with_actor("ego", 0.0, 0.0, vx=0.0, vy=0.0)
        condition.check(world, elapsed=10.0)
        result = condition.check(world, elapsed=11.0)
        assert result is not None
        assert result.elapsed_seconds == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# TemporaryStopCondition – unit tests
# ---------------------------------------------------------------------------


class TestTemporaryStopCondition:
    """Tests for TemporaryStopCondition.

    Uses mocked coordinate transforms and entity lookups.
    """

    @pytest.fixture(autouse=True)
    def _mock_road_helpers(self) -> Generator[None, None, None]:
        """Mock road helpers to avoid MapManager dependency in unit tests."""
        with patch.object(
            TemporaryStopCondition, "_get_road_length", return_value=100.0
        ), patch.object(TemporaryStopCondition, "_find_linked_roads", return_value=[]):
            yield

    def test_validation_empty_positions(self) -> None:
        with pytest.raises(ValueError, match="stop_positions must not be empty"):
            TemporaryStopCondition("ego", stop_positions=[], label="test_temp_stop")

    def test_validation_s_margin(self) -> None:
        od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
        with pytest.raises(ValueError, match="s_margin must be positive"):
            TemporaryStopCondition(
                "ego", stop_positions=[od], s_margin=0.0, label="test_temp_stop"
            )
        with pytest.raises(ValueError, match="s_margin must be positive"):
            TemporaryStopCondition(
                "ego", stop_positions=[od], s_margin=-1.0, label="test_temp_stop"
            )

    def test_validation_speed_threshold(self) -> None:
        od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
        with pytest.raises(ValueError, match="speed_threshold must be non-negative"):
            TemporaryStopCondition(
                "ego", stop_positions=[od], speed_threshold=-0.1, label="test_temp_stop"
            )

    def test_validation_stop_duration(self) -> None:
        od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
        with pytest.raises(ValueError, match="stop_duration must be positive"):
            TemporaryStopCondition(
                "ego", stop_positions=[od], stop_duration=0.0, label="test_temp_stop"
            )

    def test_single_opendrive_pose(self) -> None:
        """Single OpenDrivePose creates a PersistentCondition (not OrCondition)."""
        od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
        cond = TemporaryStopCondition(
            "ego", stop_positions=[od], label="test_temp_stop"
        )
        assert isinstance(cond._child, PersistentCondition)

    def test_multiple_opendrive_poses(self) -> None:
        """Multiple poses creates an OrCondition wrapping PersistentConditions."""
        od1 = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)
        od2 = OpenDrivePose(road_id="2", lane_id=-1, s=100.0)
        cond = TemporaryStopCondition(
            "ego", stop_positions=[od1, od2], label="test_temp_stop"
        )
        assert isinstance(cond._child, OrCondition)

    @patch("autoware_carla_scenario.conditions.composition.temporary_stop.to_opendrive")
    def test_lanelet2_pose_stays_in_its_own_frame(self, mock_to_od: MagicMock) -> None:
        """A Lanelet2Pose is matched on its lanelet, not converted to a road.

        Its s is the lanelet's own, which on a lane against the road's reference
        line runs the other way from the road's -- so converting it to a road
        and taking a window of road s put the window at the wrong end.
        """
        ll2 = Lanelet2Pose(lanelet_id=100, s=10.0, t=0.0)
        segment = EntityLanePositionCondition(
            "ego", OpenDrivePose(road_id="5", lane_id=-1, s=0.0), label="seg"
        )
        with patch.object(
            TemporaryStopCondition, "_build_lanelet_conditions", return_value=[segment]
        ) as build:
            cond = TemporaryStopCondition(
                "ego", stop_positions=[ll2], s_margin=3.0, label="test_temp_stop"
            )
        build.assert_called_once_with("ego", ll2, 3.0, label="test_temp_stop")
        mock_to_od.assert_not_called()
        assert isinstance(cond._child, PersistentCondition)

    @patch("autoware_carla_scenario.conditions.composition.temporary_stop.to_opendrive")
    def test_carla_world_pose_converted(self, mock_to_od: MagicMock) -> None:
        """CarlaWorldPose is converted via to_opendrive()."""
        mock_to_od.return_value = OpenDrivePose(road_id="7", lane_id=-1, s=40.0)
        cwp = CarlaWorldPose(x=10.0, y=20.0, z=0.0)
        cond = TemporaryStopCondition(
            "ego", stop_positions=[cwp], label="test_temp_stop"
        )
        mock_to_od.assert_called_once_with(cwp)
        assert isinstance(cond._child, PersistentCondition)

    def test_opendrive_pose_not_converted(self, without_a_map: None) -> None:
        """An OpenDrivePose reaches the road unchanged, consulting no map.

        This used to be asserted as "to_opendrive is never called", back when
        the identity case was a local short-circuit here.  It now lives inside
        `to_opendrive` itself, so the call happens and returns the pose as it
        stands.  With no map loaded, any real resolution would raise -- so
        reaching the right road is the guarantee, and it is asserted rather
        than the child's type, which the test above already covers.
        """
        od = OpenDrivePose(road_id="1", lane_id=-1, s=50.0)

        cond = TemporaryStopCondition(
            "ego", stop_positions=[od], label="test_temp_stop"
        )

        assert _road_ids_of(cond) == ["1"]


# ---------------------------------------------------------------------------
# TestEntityLanePositionAddress – a lanelet names a lane, not a road
# ---------------------------------------------------------------------------


class TestEntityLanePositionAddress:
    """The address an author writes must survive the trip into the runtime.

    An OpenDRIVE road is not a lane.  On this map lanelets 183 and 184 are lanes
    2 and 1 of the same road 80, so a resolution that kept only the road turned
    "the entity is on lanelet 183" into "the entity is anywhere on road 80" --
    true as well while it sits in the neighbouring lane, which let a cut-in
    scenario pass without the cut-in.
    """

    @pytest.fixture(scope="class")
    def loaded_map(self) -> Generator[None, None, None]:
        MapManager.reset()
        mm = MapManager.get_instance()
        mm.initialize(XODR_PATH, OSM_PATH)
        yield
        # Release lanelet2/pyxodr objects explicitly during teardown so they are
        # destroyed while the C++ runtime is still in a valid state, not during
        # Python interpreter shutdown (which can trigger std::terminate).
        mm._routing_graph = None
        mm._lanelet_map = None
        mm._road_network = None
        mm._geo_origin = None
        mm._mgrs_offset = None
        MapManager.reset()

    def test_neighbouring_lanelets_keep_their_own_lanes(self, loaded_map: None) -> None:
        # Both lanelets in one test rather than one each: what is being pinned
        # is that they *differ* while sharing a road, and loading the map costs
        # about two seconds -- which xdist would pay once per parametrised case,
        # since it hands them to different workers.
        addresses = {
            lanelet_id: EntityLanePositionCondition(
                "ego", Lanelet2Pose(lanelet_id=lanelet_id, s=0.0), label="ego_lane"
            ).get_details()
            for lanelet_id in (183, 184)
        }

        assert addresses[183]["road_id"] == addresses[184]["road_id"] == "80"
        assert addresses[183]["lane_id"] == 2
        assert addresses[184]["lane_id"] == 1

    def test_anywhere_on_road_names_no_lane(self) -> None:
        """The road-only case says so by name, rather than by omitting a lane."""
        condition = EntityLanePositionCondition.anywhere_on_road(
            "ego", "80", label="ego_road"
        )

        details = condition.get_details()
        assert details["road_id"] == "80"
        assert details["lane_id"] is None


#: A straight 100 m street, one lane each way, generated with roadgen (see its
#: generate.py): the backward lane runs against the road's reference line.
TWO_WAY = Path(__file__).resolve().parents[1] / "data" / "two_way_street"


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


def _within(field: str, lo: float, hi: float) -> list[ScalarComparisonRule]:
    return [
        ScalarComparisonRule(
            field=field, rule=ComparisonRule.GREATER_THAN_OR_EQUAL, value=lo
        ),
        ScalarComparisonRule(
            field=field, rule=ComparisonRule.LESS_THAN_OR_EQUAL, value=hi
        ),
    ]


def _world_at(pose: Lanelet2Pose, *, speed: float | None = None) -> MagicMock:
    from autoware_carla_scenario.coordinate import to_carla_world

    where = to_carla_world(pose)
    kwargs = {} if speed is None else {"vx": speed}
    return _make_world_with_actor("ego", where.x, where.y, where.z, **kwargs)


class TestPositionIsJudgedInTheFrameItWasGivenIn:
    """A lanelet's s and t are the lanelet's; a road's are the road's.

    On the two-way street the backward lanelet runs against the road's
    reference line: 10 m into the lanelet is road s 90.  A stretch written in
    lanelet s used to be compared with road s, so it matched at the other end
    of the lane -- which is how a scenario "stopped at the goal" 60 m short.
    """

    FORWARD = 1001003  # road 0, lane -1: along the reference line
    BACKWARD = 1001005  # road 0, lane 1: against it

    @pytest.fixture(scope="class")
    def loaded_map(self) -> Generator[None, None, None]:
        yield from _load_map(
            TWO_WAY / "two_way_street.xodr", TWO_WAY / "lanelet2_map.osm"
        )

    def test_each_lanelet_is_addressed_as_its_own_lane(self, loaded_map: None) -> None:
        """The converter's mapping puts each side of a two-way road on its side."""
        mapping = MapManager.get_instance().road_lanelet_mapping
        assert mapping is not None
        assert mapping.lanelet_to_road_and_lane == {
            self.FORWARD: (0, -1),
            self.BACKWARD: (0, 1),
        }

    def test_a_lanelet_stretch_is_read_in_lanelet_s(self, loaded_map: None) -> None:
        from autoware_carla_scenario.coordinate import to_carla_world, to_opendrive

        near_start = Lanelet2Pose(lanelet_id=self.BACKWARD, s=10.0)
        # The premise: the lanelet runs against the road, so its start is the
        # road's far end.
        assert to_opendrive(to_carla_world(near_start)).s == pytest.approx(
            90.0, abs=0.5
        )

        condition = EntityLanePositionCondition(
            "ego",
            Lanelet2Pose(lanelet_id=self.BACKWARD, s=0.0),
            _within("s", 5.0, 15.0),
            label="near_start",
        )
        assert condition.get_details()["frame"] == "lanelet2"
        result = condition.check(_world_at(near_start), elapsed=1.0)
        assert result is not None and result.passed
        assert f"lanelet {self.BACKWARD}" in result.message
        # Road s 5-15 is lanelet s 85-95: the other end of the lane, where the
        # stretch used to be found.
        far_end = Lanelet2Pose(lanelet_id=self.BACKWARD, s=90.0)
        assert condition.check(_world_at(far_end), elapsed=1.0) is None

    def test_the_lane_beside_it_is_not_the_lanelet(self, loaded_map: None) -> None:
        condition = EntityLanePositionCondition(
            "ego", Lanelet2Pose(lanelet_id=self.BACKWARD, s=0.0), label="lane"
        )
        beside = Lanelet2Pose(lanelet_id=self.FORWARD, s=50.0)
        assert condition.check(_world_at(beside), elapsed=1.0) is None
        here = Lanelet2Pose(lanelet_id=self.BACKWARD, s=50.0)
        assert condition.check(_world_at(here), elapsed=1.0) is not None

    def test_lanelet_t_is_left_of_the_lanelets_direction(
        self, loaded_map: None
    ) -> None:
        condition = EntityLanePositionCondition(
            "ego",
            Lanelet2Pose(lanelet_id=self.BACKWARD, s=0.0),
            _within("t", 0.5, 1.5),
            label="left",
        )
        left = Lanelet2Pose(lanelet_id=self.BACKWARD, s=30.0, t=1.0)
        right = Lanelet2Pose(lanelet_id=self.BACKWARD, s=30.0, t=-1.0)
        assert condition.check(_world_at(left), elapsed=1.0) is not None
        assert condition.check(_world_at(right), elapsed=1.0) is None

    def test_a_road_stretch_is_still_read_in_road_s(self, loaded_map: None) -> None:
        # The same place, addressed by road and lane: road s 90 is in [85, 95].
        condition = EntityLanePositionCondition(
            "ego",
            OpenDrivePose(road_id="0", lane_id=1, s=0.0),
            _within("s", 85.0, 95.0),
            label="road",
        )
        assert condition.get_details()["frame"] == "opendrive"
        near_start = Lanelet2Pose(lanelet_id=self.BACKWARD, s=10.0)
        result = condition.check(_world_at(near_start), elapsed=1.0)
        assert result is not None and "road '0'" in result.message

    def test_a_stop_on_a_lanelet_is_found_in_lanelet_s(self, loaded_map: None) -> None:
        condition = TemporaryStopCondition(
            "ego",
            stop_positions=[Lanelet2Pose(lanelet_id=self.BACKWARD, s=10.0)],
            s_margin=5.0,
            stop_duration=1.0,
            label="stop",
        )
        # Standing still 10 m into the lanelet, for longer than the duration.
        there = _world_at(Lanelet2Pose(lanelet_id=self.BACKWARD, s=10.0), speed=0.0)
        condition.check(there, elapsed=0.0)
        result = condition.check(there, elapsed=1.5)
        assert result is not None and result.passed
        # Standing still at road s 10 -- lanelet s 90 -- is not the stop.
        elsewhere = TemporaryStopCondition(
            "ego",
            stop_positions=[Lanelet2Pose(lanelet_id=self.BACKWARD, s=10.0)],
            s_margin=5.0,
            stop_duration=1.0,
            label="stop",
        )
        wrong = _world_at(Lanelet2Pose(lanelet_id=self.BACKWARD, s=90.0), speed=0.0)
        elsewhere.check(wrong, elapsed=0.0)
        assert elsewhere.check(wrong, elapsed=1.5) is None


class TestAStopMarginReachesAsFarAsItReaches:
    """A margin longer than the next lanelet carries on into the ones after it."""

    def test_the_margin_walks_on_through_short_lanelets(self) -> None:
        # 1 -> 2 (3 m) -> 3 (3 m) -> 4, and 1 <- 0 (10 m); 3 also loops back to 2.
        lengths = {0: 10.0, 1: 20.0, 2: 3.0, 3: 3.0, 4: 50.0}
        following = {1: [2], 2: [3], 3: [4, 2], 4: []}
        previous = {1: [0], 0: []}
        stretches: list[tuple[int, float, float]] = []

        def record(entity, lanelet_id, lo, hi, *, label):  # type: ignore[no-untyped-def]
            stretches.append((lanelet_id, lo, hi))
            return EntityLanePositionCondition(
                entity, OpenDrivePose(road_id="0", lane_id=-1, s=0.0), label=label
            )

        with patch.object(
            TemporaryStopCondition,
            "_get_lanelet_length",
            side_effect=lengths.__getitem__,
        ), patch.object(
            TemporaryStopCondition,
            "_find_linked_lanelets",
            side_effect=lambda lid, d: (following if d == "following" else previous)[
                lid
            ],
        ), patch.object(
            TemporaryStopCondition,
            "_make_lanelet_segment_condition",
            side_effect=record,
        ):
            TemporaryStopCondition(
                "ego",
                stop_positions=[Lanelet2Pose(lanelet_id=1, s=18.0)],
                s_margin=10.0,
                label="stop",
            )
        # Past the end: 8 m, of which lanelet 2 holds 3, lanelet 3 the next 3
        # and lanelet 4 the last 2 -- and the loop back to 2 is not walked again.
        # Before the start nothing runs over: 18 - 10 is still on lanelet 1.
        assert stretches == [
            (1, 8.0, 20.0),
            (2, 0.0, 3.0),
            (3, 0.0, 3.0),
            (4, 0.0, 2.0),
        ]


class TestAStopOnALaneletSpillsOntoTheLaneletsAroundIt:
    """What runs past a lanelet's end is taken from the lanelet after it."""

    @pytest.fixture(scope="class")
    def loaded_map(self) -> Generator[None, None, None]:
        yield from _load_map(XODR_PATH, OSM_PATH)

    def test_past_the_end_is_the_following_lanelet(self, loaded_map: None) -> None:
        from autoware_carla_scenario.coordinate import lanelet_length

        # On this map lanelet 183 is followed by 187.
        condition = TemporaryStopCondition(
            "ego",
            stop_positions=[Lanelet2Pose(lanelet_id=183, s=lanelet_length(183))],
            s_margin=5.0,
            label="stop",
        )
        watched: list[tuple[object, object]] = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                if "frame" in node:
                    watched.append((node["frame"], node["lanelet_id"]))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(condition.to_summary_dict())
        assert sorted(watched) == [("lanelet2", 183), ("lanelet2", 187)]


class TestAnOpenDriveAddressNeedsNoMap:
    """An address already in the runtime's frame is passed straight through.

    This is what keeps the constructor usable without a loaded map, which the
    unit tests above and every hand-written scenario written in OpenDRIVE rely
    on: only a Lanelet2 or CARLA address has to be resolved against a map.
    """

    def test_no_map_is_consulted(self, without_a_map: None) -> None:
        condition = EntityLanePositionCondition(
            "ego", OpenDrivePose(road_id="80", lane_id=2, s=0.0), label="ego_lane"
        )

        details = condition.get_details()
        assert details["road_id"] == "80"
        assert details["lane_id"] == 2
