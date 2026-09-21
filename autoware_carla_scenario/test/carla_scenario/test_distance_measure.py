"""Relative distance type and freespace: what "20 m apart" is actually asking.

Both options exist because the default answers a question scenarios rarely
mean.  These tests are built around the two cases that motivated them: a car in
the next lane, and two vehicles close enough that their own length is most of
the threshold.
"""

from __future__ import annotations

import math
import re
from unittest.mock import MagicMock

import carla
import pytest

from autoware_carla_scenario import (
    ComparisonRule,
    EntityDistanceCondition,
    RelativeDistanceType,
)


def _actor(
    role: str,
    position: tuple[float, float, float],
    heading: tuple[float, float] = (1.0, 0.0),
    extent: tuple[float, float, float] | None = (2.5, 1.0, 0.75),
    box_offset: tuple[float, float, float] | None = None,
) -> MagicMock:
    """A fake actor; *extent* is the bounding box half-size, or None for no box."""
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(*heading, 0.0)
    actor.get_transform.return_value = transform
    if extent is None:
        actor.bounding_box = None
    else:
        box = MagicMock()
        box.extent = carla.Vector3D(*extent)
        box.location = carla.Location(*(box_offset or (0.0, 0.0, 0.0)))
        actor.bounding_box = box
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _measured(condition: EntityDistanceCondition, world: MagicMock) -> float:
    """Return the number the condition reports, by making the rule always true."""
    result = condition.check(world, 0.0)
    assert result is not None, "the condition did not fire"
    match = re.search(r"\(([0-9.]+) m\)", result.message)
    assert match is not None, result.message
    return float(match.group(1))


def _condition(**kwargs: object) -> EntityDistanceCondition:
    """A condition with a threshold high enough to always fire."""
    return EntityDistanceCondition(
        source="Ego",
        target="npc1",
        value=1e9,
        rule=ComparisonRule.LESS_THAN,
        label="measure",
        **kwargs,  # type: ignore[arg-type]
    )


class TestDistanceType:
    def test_a_car_in_the_next_lane_is_near_longitudinally_and_far_in_a_line(
        self,
    ) -> None:
        """The case the option exists for."""
        world = _world(
            _actor("Ego", (0, 0, 0), heading=(1.0, 0.0)),
            _actor("npc1", (2, 20, 0)),
        )

        straight = _measured(_condition(), world)
        longitudinal = _measured(
            _condition(distance_type=RelativeDistanceType.LONGITUDINAL), world
        )
        lateral = _measured(
            _condition(distance_type=RelativeDistanceType.LATERAL), world
        )

        assert math.isclose(straight, math.hypot(2, 20), abs_tol=0.01)
        assert math.isclose(longitudinal, 2.0, abs_tol=0.01)
        assert math.isclose(lateral, 20.0, abs_tol=0.01)

    def test_the_components_follow_the_subject_heading(self) -> None:
        """Turn the subject 90 degrees and longitudinal and lateral swap."""
        world = _world(
            _actor("Ego", (0, 0, 0), heading=(0.0, 1.0)),
            _actor("npc1", (2, 20, 0)),
        )
        longitudinal = _measured(
            _condition(distance_type=RelativeDistanceType.LONGITUDINAL), world
        )
        assert math.isclose(longitudinal, 20.0, abs_tol=0.01)

    def test_a_component_is_a_distance_not_a_signed_offset(self) -> None:
        """A vehicle 30 m behind is 30 m away, so `< 20` stays false.

        A signed reading would make a following-distance rule fire for the car
        that is nowhere near.
        """
        behind = _world(
            _actor("Ego", (0, 0, 0), heading=(1.0, 0.0)),
            _actor("npc1", (-30, 0, 0)),
        )
        condition = EntityDistanceCondition(
            source="Ego",
            target="npc1",
            value=20.0,
            rule=ComparisonRule.LESS_THAN,
            distance_type=RelativeDistanceType.LONGITUDINAL,
            label="following",
        )
        assert condition.check(behind, 0.0) is None


class TestFreespace:
    def test_the_gap_is_shorter_than_the_centre_distance_by_both_extents(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (10, 0, 0), extent=(2.5, 1.0, 0.75)),
        )
        centres = _measured(_condition(), world)
        gap = _measured(_condition(freespace=True), world)

        assert math.isclose(centres, 10.0, abs_tol=0.01)
        assert math.isclose(gap, 5.0, abs_tol=0.01)

    def test_the_narrow_axis_is_used_when_the_vehicles_are_side_by_side(self) -> None:
        """An oriented box, not a radius: a car is wider than it is long only across."""
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (0, 10, 0), extent=(2.5, 1.0, 0.75)),
        )
        gap = _measured(_condition(freespace=True), world)
        assert math.isclose(gap, 8.0, abs_tol=0.01)

    def test_overlapping_boxes_clamp_at_zero(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (3, 0, 0), extent=(2.5, 1.0, 0.75)),
        )
        assert _measured(_condition(freespace=True), world) == 0.0

    def test_a_diagonal_gap_is_the_real_box_to_box_distance(self) -> None:
        """Not the centre distance minus each box's reach along that line.

        Boxes of half-extent (2.5, 1.0) whose centres are (6, 6) apart are
        4.12 m apart: 1 m of clearance along x and 4 m along y.  Subtracting
        each box's radial reach gives 3.54 m, so a 4 m threshold would fire
        for a pair that is not that close.
        """
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (6, 6, 0), extent=(2.5, 1.0, 0.75)),
        )
        gap = _measured(_condition(freespace=True), world)
        assert math.isclose(gap, math.hypot(1.0, 4.0), abs_tol=0.01)

    def test_height_counts_when_vertical_is_asked_for(self) -> None:
        """The box has a roof, so a vertical freespace gap has to use it."""
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (0, 0, 10), extent=(2.5, 1.0, 0.75)),
        )
        gap = _measured(_condition(freespace=True, vertical=True), world)
        assert math.isclose(gap, 10.0 - 1.5, abs_tol=0.01)

    def test_the_box_offset_from_the_actor_origin_is_used(self) -> None:
        """A CARLA box sits near the actor's origin, not on it."""
        plain = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (10, 0, 0), extent=(2.5, 1.0, 0.75)),
        )
        # The same pair, but the target's box is a metre further forward.
        offset = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor(
                "npc1", (10, 0, 0), extent=(2.5, 1.0, 0.75), box_offset=(1.0, 0.0, 0.0)
            ),
        )
        assert math.isclose(
            _measured(_condition(freespace=True), offset)
            - _measured(_condition(freespace=True), plain),
            1.0,
            abs_tol=0.01,
        )

    def test_an_actor_without_a_bounding_box_contributes_nothing(self) -> None:
        """Degrade to the centre for that actor rather than refuse to measure."""
        world = _world(
            _actor("Ego", (0, 0, 0), extent=(2.5, 1.0, 0.75)),
            _actor("npc1", (10, 0, 0), extent=None),
        )
        assert math.isclose(
            _measured(_condition(freespace=True), world), 7.5, abs_tol=0.01
        )


class TestDefaultsAndGuards:
    def test_the_default_is_the_previous_behaviour(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0)), _actor("npc1", (3, 4, 0)))
        assert math.isclose(_measured(_condition(), world), 5.0, abs_tol=0.01)

    def test_vertical_with_a_directional_component_is_refused(self) -> None:
        """Ground-plane components plus height is a contradiction, not a refinement."""
        with pytest.raises(ValueError, match="ground-plane component"):
            EntityDistanceCondition(
                source="Ego",
                target="npc1",
                value=10.0,
                vertical=True,
                distance_type=RelativeDistanceType.LONGITUDINAL,
                label="contradiction",
            )

    def test_a_degenerate_heading_makes_a_component_unknown(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), heading=(0.0, 0.0)),
            _actor("npc1", (10, 0, 0)),
        )
        condition = EntityDistanceCondition(
            source="Ego",
            target="npc1",
            value=1e9,
            distance_type=RelativeDistanceType.LONGITUDINAL,
            label="unknown",
        )
        assert condition.check(world, 0.0) is None

    def test_details_report_both_choices(self) -> None:
        condition = _condition(
            distance_type=RelativeDistanceType.LATERAL, freespace=True
        )
        details = condition.get_details()
        assert details["distance_type"] == "LATERAL"
        assert details["freespace"] is True
