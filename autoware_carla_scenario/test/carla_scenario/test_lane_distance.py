"""Distance along the road: the measurement OpenSCENARIO calls `lane`.

Driven against the converter's own nishishinjuku fixture rather than a
synthetic straight line, because a straight line is exactly the case where the
lane frame and the entity frame agree -- it would pass without measuring
anything the entity frame gets wrong.

Every expected value is built by placing a pose at a known ``s`` on a real road
and converting it out to CARLA world coordinates, so the assertion is a
round trip through the same projection the runtime uses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    DistanceCoordinateSystem,
    EntityDistanceCondition,
    RelativeDistanceType,
    TimeHeadwayCondition,
    TimeToCollisionCondition,
)

from autoware_carla_scenario.coordinate import lane_distance
from autoware_carla_scenario.coordinate.lane_distance import (
    lane_closing_speed,
    lane_gap,
    lane_separation,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, OpenDrivePose
from autoware_carla_scenario.coordinate.transform import to_carla_world

#: The converter's fixture map, which both packages' tests share.
_CONVERTER_TEST_DATA = (
    Path(__file__).resolve().parents[3]
    / "autoware_lanelet2_to_opendrive"
    / "test"
    / "data"
)
XODR_PATH = _CONVERTER_TEST_DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _CONVERTER_TEST_DATA / "nishishinjuku.osm"

#: A road long enough to put two vehicles far apart on, and curved enough that
#: the along-road distance and the straight line are not the same number.
ROAD = "144"


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """Load the fixture map once for this module."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(XODR_PATH, OSM_PATH)
    yield manager
    MapManager._instance = saved


def _at(s: float, *, road: str = ROAD) -> CarlaWorldPose:
    """Return the CARLA world pose of the road centre at arc length *s*."""
    return to_carla_world(OpenDrivePose(road_id=road, lane_id=-1, s=s, t=0.0))


def _towards(a: CarlaWorldPose, b: CarlaWorldPose) -> tuple[float, float]:
    """Return a velocity pointing from *a* to *b* (1 m/s)."""
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def _road_direction(s: float, *, road: str = ROAD) -> tuple[float, float]:
    """Return the road's own unit direction at arc length *s*.

    A vehicle's velocity has to be built from the tangent **at its own
    position**.  On a sharp bend that matters a great deal: road 147 turns
    137.7 degrees between ``s = 5`` and ``s = 25``, so borrowing one end's
    tangent for a car at the other end describes a car driving across the road
    rather than along it -- and then a test proves nothing about the thing it
    names.
    """
    return _towards(_at(s, road=road), _at(s + 1.0, road=road))


class TestSeparation:
    def test_it_is_the_arc_length_between_them(self, loaded_map: MapManager) -> None:
        behind, ahead = _at(10.0), _at(30.0)

        assert lane_separation(behind, ahead) == pytest.approx(20.0, abs=0.05)

    def test_it_is_unsigned_and_so_the_same_either_way(
        self, loaded_map: MapManager
    ) -> None:
        behind, ahead = _at(10.0), _at(30.0)

        assert lane_separation(ahead, behind) == pytest.approx(
            lane_separation(behind, ahead)
        )

    def test_neither_has_to_be_moving(self, loaded_map: MapManager) -> None:
        """A separation has no direction, so it needs no velocity to exist.

        This is the difference from a headway, and it is why the three
        conditions do not share one function.
        """
        assert lane_separation(_at(10.0), _at(30.0)) is not None

    def test_it_is_longer_than_the_straight_line_on_a_curve(
        self, loaded_map: MapManager
    ) -> None:
        """The whole reason the lane frame exists, on real road geometry.

        A chord is shorter than the arc it subtends, so a straight line between
        two vehicles on a curve under-reads the gap between them.  Measured
        over a long stretch of a real road so the difference is unmistakable
        rather than a rounding artifact.
        """
        behind, ahead = _at(0.0), _at(300.0)
        straight_line = math.hypot(ahead.x - behind.x, ahead.y - behind.y)

        along_road = lane_separation(behind, ahead)

        assert along_road is not None
        assert along_road == pytest.approx(300.0, abs=0.5)
        assert straight_line < along_road - 1.0


class TestGap:
    def test_a_target_ahead_reads_positive(self, loaded_map: MapManager) -> None:
        behind, ahead = _at(10.0), _at(30.0)
        vx, vy = _towards(behind, ahead)

        assert lane_gap(behind, vx, vy, ahead) == pytest.approx(20.0, abs=0.05)

    def test_a_target_behind_reads_negative(self, loaded_map: MapManager) -> None:
        """The sign is what lets a headway refuse a vehicle already overtaken.

        The overtaker is the one at ``s = 30`` still going the same way; the
        vehicle it passed is back at ``s = 10``.
        """
        passed, overtaker = _at(10.0), _at(30.0)
        vx, vy = _towards(passed, overtaker)

        assert lane_gap(overtaker, vx, vy, passed) == pytest.approx(-20.0, abs=0.05)

    def test_the_sign_follows_travel_and_not_the_map(
        self, loaded_map: MapManager
    ) -> None:
        """Driving the other way flips it, with nothing else changed.

        The direction along `s` is read from the vehicle's own motion rather
        than from `lane_id`, whose sign says which side of the reference line a
        lane is on -- a relationship to the direction of travel that is a
        convention, and one left-hand traffic inverts.  Japan is left-hand
        traffic, and this fixture is Nishi-Shinjuku.
        """
        behind, ahead = _at(10.0), _at(30.0)
        forward_x, forward_y = _towards(behind, ahead)

        forward = lane_gap(behind, forward_x, forward_y, ahead)
        reversed_ = lane_gap(behind, -forward_x, -forward_y, ahead)

        assert forward == pytest.approx(20.0, abs=0.05)
        assert reversed_ == pytest.approx(-20.0, abs=0.05)

    def test_a_vehicle_at_the_start_of_its_road_still_measures(
        self, loaded_map: MapManager
    ) -> None:
        """Regression: an earlier version sampled a point 2 m along the
        velocity and compared ``s``, which fell onto the neighbouring road for
        any vehicle within a probe's length of where its own road began --
        refusing to measure exactly where a junction approach is interesting.
        Reading the angle off the projection has nothing to fall off.
        """
        start, ahead = _at(0.5), _at(20.0)
        forward_x, forward_y = _towards(start, ahead)

        assert lane_gap(start, -forward_x, -forward_y, ahead) == pytest.approx(
            -19.5, abs=0.1
        )

    def test_a_stationary_source_has_no_gap(self, loaded_map: MapManager) -> None:
        assert lane_gap(_at(10.0), 0.0, 0.0, _at(30.0)) is None


class TestClosingSpeed:
    def test_a_still_target_closes_at_the_source_speed_along_the_road(
        self, loaded_map: MapManager
    ) -> None:
        behind, ahead = _at(10.0), _at(30.0)
        vx, vy = _towards(behind, ahead)

        closing = lane_closing_speed(behind, vx * 10.0, vy * 10.0, ahead, 0.0, 0.0)

        # Resolved onto the road, so it is the speed times the cosine between
        # the two -- at most the speed itself, and short of it by however much
        # the vehicle is cutting across the curve.
        assert closing is not None
        assert 0.0 < closing <= 10.0

    def test_a_target_keeping_pace_is_barely_closing(
        self, loaded_map: MapManager
    ) -> None:
        behind, ahead = _at(10.0), _at(30.0)
        vx, vy = _towards(behind, ahead)

        closing = lane_closing_speed(
            behind, vx * 10.0, vy * 10.0, ahead, vx * 10.0, vy * 10.0
        )

        assert closing == pytest.approx(0.0, abs=1.0)

    def test_a_still_source_is_closed_on_by_a_moving_target(
        self, loaded_map: MapManager
    ) -> None:
        """A parked car has a closing speed; what it does not have is a headway.

        Which way is "towards" comes from where the two are along the road, not
        from how the source is driving -- so a stationary source is not an
        unanswerable one, and the pair has a finite time to collision.
        """
        parked, approaching = _at(10.0), _at(30.0)
        # Driving back down the road, towards the parked car.
        vx, vy = _towards(approaching, parked)

        closing = lane_closing_speed(
            parked, 0.0, 0.0, approaching, vx * 10.0, vy * 10.0
        )

        assert closing is not None
        assert 0.0 < closing <= 10.0

    def test_a_faster_target_behind_is_closing_not_receding(
        self, loaded_map: MapManager
    ) -> None:
        """A rear-end collision closes just as surely as one in front."""
        overtaken, catching_up = _at(30.0), _at(10.0)
        forward_x, forward_y = _towards(catching_up, overtaken)

        closing = lane_closing_speed(
            overtaken,
            forward_x * 5.0,
            forward_y * 5.0,
            catching_up,
            forward_x * 15.0,
            forward_y * 15.0,
        )

        assert closing is not None
        assert closing > 0.0

    def test_two_level_along_the_road_have_no_direction_to_close_along(
        self, loaded_map: MapManager
    ) -> None:
        here = _at(10.0)
        vx, vy = _towards(here, _at(30.0))

        assert lane_closing_speed(here, vx, vy, here, 0.0, 0.0) is None

    def test_a_faster_target_is_opening_the_gap(self, loaded_map: MapManager) -> None:
        behind, ahead = _at(10.0), _at(30.0)
        vx, vy = _towards(behind, ahead)

        closing = lane_closing_speed(
            behind, vx * 5.0, vy * 5.0, ahead, vx * 15.0, vy * 15.0
        )

        assert closing is not None
        assert closing < 0.0


class TestWhenThereIsNoAnswer:
    def test_roads_no_chain_joins_are_refused_rather_than_approximated(
        self, loaded_map: MapManager
    ) -> None:
        """Across a junction, which needs a routing graph nothing holds yet.

        Falling back to the straight line here would be the silent swap of one
        measure for another that the whole lane frame exists to remove: a
        scenario would pass, and nothing would say which measure it passed on.
        """
        here = _at(10.0)
        elsewhere = _at(10.0, road="598")
        assert elsewhere.x != here.x or elsewhere.y != here.y

        assert lane_separation(here, elsewhere) is None

    def test_without_a_map_it_says_so_once_rather_than_every_tick(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A misconfiguration, not a fact about the scenario -- so it is logged.

        Logged *once*: a condition is evaluated every tick, so an unguarded
        warning would print at the tick rate for the whole run and bury
        everything else, while saying nothing new each time.
        """
        saved_instance = MapManager._instance
        saved_flag = lane_distance._warned_without_map
        MapManager._instance = None
        lane_distance._warned_without_map = False
        try:
            with caplog.at_level("WARNING"):
                for _ in range(5):
                    result = lane_separation(
                        CarlaWorldPose(x=0.0, y=0.0, z=0.0, yaw=0.0),
                        CarlaWorldPose(x=20.0, y=0.0, z=0.0, yaw=0.0),
                    )
        finally:
            MapManager._instance = saved_instance
            lane_distance._warned_without_map = saved_flag

        assert result is None
        assert caplog.text.count("needs a loaded map") == 1


# ---------------------------------------------------------------------------
# The three conditions, end to end
# ---------------------------------------------------------------------------

#: A tight bend in the fixture map, and what makes it worth naming: a leader
#: 20 m ahead **along the lane** from ``s = 5`` reads 0.30 m in the entity
#: frame, because at that curvature the chord to it is very nearly square to
#: the direction of travel.  Same geometry, same threshold, opposite answers.
BEND = "147"
BEND_START = 5.0
BEND_GAP = 20.0


def _actor(
    role: str,
    pose: CarlaWorldPose,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> MagicMock:
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(pose.x, pose.y, pose.z)
    actor.get_velocity.return_value = carla.Vector3D(velocity[0], velocity[1], 0.0)
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _on_the_bend(speed: float = 10.0) -> "tuple[MagicMock, float, float]":
    """Return a world with a follower on the bend and a leader 20 m up the lane.

    Also returns the follower's velocity, so a caller can reason about what the
    two frames should each make of it.
    """
    follower = _at(BEND_START, road=BEND)
    leader = _at(BEND_START + BEND_GAP, road=BEND)
    tangent = _at(BEND_START + 1.0, road=BEND)
    ux, uy = _towards(follower, tangent)
    return (
        _world(
            _actor("Ego", follower, velocity=(ux * speed, uy * speed)),
            _actor("npc1", leader),
        ),
        ux * speed,
        uy * speed,
    )


class TestTheConditionsInLaneFrame:
    """The parameter is wired through, and it changes the answer where it must.

    Each of these runs the same geometry through both frames.  A test that only
    checked the lane frame would pass just as well if the entity frame had
    quietly been giving the same number all along, which is the thing worth
    ruling out.
    """

    def test_headway_disagrees_with_the_entity_frame_on_a_bend(
        self, loaded_map: MapManager
    ) -> None:
        """The failure the whole issue is about, on real road geometry.

        The leader is 20 m up the lane, which at 10 m/s is a headway of about
        2.0 s -- a perfectly safe following distance.  The entity frame
        measures the chord to it as 0.30 m and calls that 0.03 s, so a "closer
        than 1 s" tailgating rule fires on a vehicle that is nowhere near
        tailgating.
        """
        world, _, _ = _on_the_bend(speed=10.0)

        entity_frame = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=1.0,
            rule=ComparisonRule.LESS_THAN,
            label="entity",
        )
        lane_frame = TimeHeadwayCondition(
            source="Ego",
            target="npc1",
            value=1.0,
            rule=ComparisonRule.LESS_THAN,
            coordinate_system=DistanceCoordinateSystem.LANE,
            label="lane",
        )

        assert entity_frame.check(world, 1.0) is not None, (
            "the entity frame is expected to be wrong here -- if it is not, "
            "this bend is no longer a bend and the test proves nothing"
        )
        assert lane_frame.check(world, 1.0) is None

    def test_distance_measures_along_the_road(self, loaded_map: MapManager) -> None:
        world, _, _ = _on_the_bend()

        condition = EntityDistanceCondition(
            source="Ego",
            target="npc1",
            value=BEND_GAP - 0.5,
            rule=ComparisonRule.GREATER_THAN,
            coordinate_system=DistanceCoordinateSystem.LANE,
            label="along",
        )

        assert condition.check(world, 1.0) is not None

    def test_ttc_resolves_both_halves_onto_the_road(
        self, loaded_map: MapManager
    ) -> None:
        """Gap and closing speed change frame together, or neither does.

        A lane-measured gap over a straight-line closing speed would count the
        follower's cornering as approach and make the TTC short for a reason
        that has nothing to do with the road.  20 m of lane at ~10 m/s is about
        2 s, so a "< 1 s" rule must stay quiet and a "< 5 s" rule must fire.
        """
        world, _, _ = _on_the_bend(speed=10.0)

        def ttc(threshold: float) -> TimeToCollisionCondition:
            return TimeToCollisionCondition(
                source="Ego",
                target="npc1",
                value=threshold,
                rule=ComparisonRule.LESS_THAN,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label=f"ttc_{threshold}",
            )

        assert ttc(1.0).check(world, 1.0) is None
        assert ttc(5.0).check(world, 1.0) is not None

    def test_lane_ttc_answers_for_the_same_pairs_as_the_entity_frame(
        self, loaded_map: MapManager
    ) -> None:
        """Regression for two pairs the lane frame used to go quiet on.

        The first version took the gap from `lane_gap`, which needs a moving
        source and signs by its travel.  That returned `None` for a stationary
        source being bore down on, and for a faster target closing from behind
        -- both of them collisions, and both of them the silent never-fires
        this coordinate system exists to remove.  Neither frame may refuse a
        pair the other answers for.

        Run on the gentle road rather than on the bend, because the claim is
        that the lane frame answers wherever the entity frame does -- which
        needs a stretch where the entity frame is itself trustworthy.  The bend
        is where the two are *supposed* to differ, and the test above covers
        that.  Each velocity still comes from the road's direction at that
        vehicle's own position.
        """
        behind, ahead = _at(10.0), _at(30.0)
        at_behind = _road_direction(10.0)
        at_ahead = _road_direction(30.0)

        def ttc_fires(world: MagicMock, system: DistanceCoordinateSystem) -> bool:
            condition = TimeToCollisionCondition(
                source="Ego",
                target="npc1",
                value=60.0,
                rule=ComparisonRule.LESS_THAN,
                coordinate_system=system,
                label="ttc",
            )
            return condition.check(world, 1.0) is not None

        # Parked, with a vehicle coming back down the road at it.
        stationary_source = _world(
            _actor("Ego", behind),
            _actor("npc1", ahead, velocity=(-at_ahead[0] * 10.0, -at_ahead[1] * 10.0)),
        )
        # Both going the same way, the one behind going three times faster.
        target_behind = _world(
            _actor("Ego", ahead, velocity=(at_ahead[0] * 5.0, at_ahead[1] * 5.0)),
            _actor("npc1", behind, velocity=(at_behind[0] * 15.0, at_behind[1] * 15.0)),
        )

        for name, world in (
            ("a stationary source being closed on", stationary_source),
            ("a faster target behind", target_behind),
        ):
            assert ttc_fires(
                world, DistanceCoordinateSystem.ENTITY
            ), f"{name}: the entity frame is expected to answer here"
            assert ttc_fires(world, DistanceCoordinateSystem.LANE), (
                f"{name}: the lane frame went quiet on a pair the entity "
                f"frame answers for"
            )

    def test_the_entity_frame_is_still_the_default(
        self, loaded_map: MapManager
    ) -> None:
        """A document written before this existed keeps the meaning it had."""
        for condition in (
            TimeHeadwayCondition(source="a", target="b", value=1.0, label="h"),
            EntityDistanceCondition(source="a", target="b", value=1.0, label="d"),
            TimeToCollisionCondition(source="a", target="b", value=1.0, label="t"),
        ):
            assert condition.get_details()["coordinate_system"] == "ENTITY"

    def test_ttc_to_a_place_measures_along_the_road(
        self, loaded_map: MapManager
    ) -> None:
        """A stop line on a bend, which is where the frame changes the answer.

        The place is 20 m up the lane at 10 m/s, so about 2 s away along the
        road.  The entity frame reads about **47 s**: the chord to the stop
        line is very nearly square to the direction of travel on this bend, so
        the component of the speed along it nearly vanishes and the quotient
        blows up.  A "TTC to the stop line under 4 s" rule therefore never
        fires there, on an approach that is two seconds out.

        A place does not move, so in the lane frame the closing speed reduces
        to the subject's own speed along the road -- which is what "time to the
        stop line" means.
        """
        world, _, _ = _on_the_bend(speed=10.0)
        stop_line = OpenDrivePose(
            road_id=BEND, lane_id=-1, s=BEND_START + BEND_GAP, t=0.0
        )

        lane_frame = TimeToCollisionCondition(
            source="Ego",
            position=stop_line,
            value=4.0,
            rule=ComparisonRule.LESS_THAN,
            coordinate_system=DistanceCoordinateSystem.LANE,
            label="lane",
        )
        entity_frame = TimeToCollisionCondition(
            source="Ego",
            position=stop_line,
            value=4.0,
            rule=ComparisonRule.LESS_THAN,
            label="entity",
        )

        assert lane_frame.check(world, 1.0) is not None
        assert entity_frame.check(world, 1.0) is None, (
            "the entity frame is expected to be wrong here -- if it is not, "
            "this bend is no longer a bend and the test proves nothing"
        )


class TestRefusedCombinations:
    def test_a_lane_distance_cannot_also_be_vertical(self) -> None:
        """A length along the road is one-dimensional; there is no z to add.

        Refused rather than resolved, because either reading loses something
        the author may have meant.
        """
        with pytest.raises(ValueError, match="no vertical component"):
            EntityDistanceCondition(
                source="a",
                target="b",
                value=1.0,
                vertical=True,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label="both",
            )

    def test_a_vertical_entity_distance_is_still_fine(self) -> None:
        condition = EntityDistanceCondition(
            source="a", target="b", value=1.0, vertical=True, label="v"
        )

        assert condition.get_details()["vertical"] is True

    def test_a_lane_distance_cannot_also_be_lateral(self) -> None:
        """The two frames arrived in separate PRs and meet here.

        ``distance_type`` picks a component of the straight line; the lane
        frame is along the road by construction, so there is no across-it to
        report and no reading of the pair that keeps both.
        """
        with pytest.raises(ValueError, match="no lateral component"):
            EntityDistanceCondition(
                source="a",
                target="b",
                value=1.0,
                distance_type=RelativeDistanceType.LATERAL,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label="both",
            )

    def test_a_longitudinal_lane_distance_is_accepted(self) -> None:
        """Saying it twice is redundant, not contradictory."""
        condition = EntityDistanceCondition(
            source="a",
            target="b",
            value=1.0,
            distance_type=RelativeDistanceType.LONGITUDINAL,
            coordinate_system=DistanceCoordinateSystem.LANE,
            label="along",
        )

        assert condition.get_details()["coordinate_system"] == "LANE"

    def test_a_lane_frame_ttc_cannot_be_edge_to_edge(self) -> None:
        """Same refusal, same reason, on the condition where it costs most.

        TTC is compared against a handful of seconds, so a vehicle length in
        the numerator is a large fraction of the answer -- which is exactly why
        measuring centre to centre and calling it bumper to bumper would be
        worth refusing rather than approximating.
        """
        with pytest.raises(ValueError, match="edge_to_edge is not available"):
            TimeToCollisionCondition(
                source="a",
                target="b",
                value=4.0,
                edge_to_edge=True,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label="both",
            )

    def test_an_entity_frame_ttc_may_still_be_edge_to_edge(self) -> None:
        condition = TimeToCollisionCondition(
            source="a", target="b", value=4.0, edge_to_edge=True, label="boxes"
        )

        assert condition.get_details()["edge_to_edge"] is True

    def test_a_lane_distance_cannot_be_edge_to_edge(self) -> None:
        """Bumper to bumper along a curve needs the boxes put on the road.

        Refused rather than quietly measured centre to centre: that is wrong by
        about a vehicle length, which at a close-quarters threshold is most of
        the threshold.
        """
        with pytest.raises(ValueError, match="edge_to_edge is not available"):
            EntityDistanceCondition(
                source="a",
                target="b",
                value=1.0,
                edge_to_edge=True,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label="both",
            )


# ---------------------------------------------------------------------------
# Across connected roads
# ---------------------------------------------------------------------------

#: A chain of three roads the fixture joins end to end, and their lengths.
#: The middle one is 16 m long, which is the point: an OpenDRIVE map is cut
#: into roads far shorter than a following distance, so a leader 20 m ahead is
#: routinely on the next road rather than this one.
CHAIN = ("32", "34", "36")


def _chain_lengths() -> "tuple[float, float, float]":
    lengths = tuple(lane_distance._road_length(road) for road in CHAIN)
    assert all(length is not None for length in lengths), CHAIN
    return lengths  # type: ignore[return-value]


class TestAcrossConnectedRoads:
    """The case measuring within one road answers ``None`` to.

    Not an edge case: on this fixture the median road is 33 m and three
    quarters are under 50 m, so this is where most following scenarios live.
    """

    def test_the_next_road_along_is_measured_not_refused(
        self, loaded_map: MapManager
    ) -> None:
        first, _, _ = CHAIN
        length, _, _ = _chain_lengths()
        behind = _at(length - 5.0, road=first)
        ahead = _at(7.0, road=CHAIN[1])

        assert lane_separation(behind, ahead) == pytest.approx(12.0, abs=0.1)

    def test_the_walk_carries_on_over_several_roads(
        self, loaded_map: MapManager
    ) -> None:
        first, middle, last = CHAIN
        first_length, middle_length, _ = _chain_lengths()
        behind = _at(first_length - 5.0, road=first)
        ahead = _at(3.0, road=last)

        assert lane_separation(behind, ahead) == pytest.approx(
            5.0 + middle_length + 3.0, abs=0.1
        )

    def test_a_separation_reads_the_same_from_either_end(
        self, loaded_map: MapManager
    ) -> None:
        """Which vehicle is asked must not change the answer.

        Worth pinning rather than assuming: a road-to-road link is usually
        written from one side only, because OpenDRIVE has a road adjoining a
        junction name the junction rather than the road beyond it.  On this
        fixture 367 of 490 road-to-road successors have no matching road
        predecessor, so a walk that trusted each road's own ``predecessor``
        would measure one way and refuse the other.
        """
        first, middle, _ = CHAIN
        first_length, _, _ = _chain_lengths()
        behind = _at(first_length - 5.0, road=first)
        ahead = _at(7.0, road=middle)

        there = lane_separation(behind, ahead)
        back = lane_separation(ahead, behind)
        assert there is not None and back is not None
        assert there == pytest.approx(back)

    def test_a_gap_over_the_join_keeps_its_sign(self, loaded_map: MapManager) -> None:
        first, middle, _ = CHAIN
        first_length, _, _ = _chain_lengths()
        behind = _at(first_length - 5.0, road=first)
        ahead = _at(7.0, road=middle)
        ux, uy = _towards(behind, _at(first_length - 4.0, road=first))

        assert lane_gap(behind, ux * 10, uy * 10, ahead) == pytest.approx(12.0, abs=0.1)
        # Same pair, driving the other way: the leader is now behind.
        assert lane_gap(behind, -ux * 10, -uy * 10, ahead) == pytest.approx(
            -12.0, abs=0.1
        )

    def test_a_closing_speed_over_the_join_is_the_along_road_difference(
        self, loaded_map: MapManager
    ) -> None:
        """Each road numbers ``s`` its own way, so the two speeds are put on
        the axis running from the source to the target before subtracting."""
        first, middle, _ = CHAIN
        first_length, _, _ = _chain_lengths()
        behind = _at(first_length - 5.0, road=first)
        ahead = _at(7.0, road=middle)
        ux, uy = _towards(behind, _at(first_length - 4.0, road=first))
        tx, ty = _towards(ahead, _at(8.0, road=middle))

        # Closing on a parked car at the follower's own along-road speed.
        assert lane_closing_speed(
            behind, ux * 10, uy * 10, ahead, 0.0, 0.0
        ) == pytest.approx(10.0, abs=0.1)
        # Keeping pace over the join: barely closing at all.
        assert lane_closing_speed(
            behind, ux * 10, uy * 10, ahead, tx * 10, ty * 10
        ) == pytest.approx(0.0, abs=0.1)

    def test_a_pair_further_apart_than_a_scenario_asks_about_is_refused(
        self, loaded_map: MapManager
    ) -> None:
        """The walk is bounded, so a map cannot be searched indefinitely."""
        first, _, _ = CHAIN
        assert lane_separation(_at(0.0, road=first), _at(200.0, road="2")) is None
