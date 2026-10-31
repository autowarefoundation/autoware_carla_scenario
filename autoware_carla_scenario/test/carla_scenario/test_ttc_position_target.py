"""TTC to a place, and TTC measured from the bumper.

The entity-to-entity form is unchanged and covered by
``test_relative_conditions.py``; what is new is a target that does not move and
a range that starts at the bounding box.
"""

from __future__ import annotations

import math
import re
from unittest.mock import MagicMock

import typesafe_carla.carla as carla
import pytest

from autoware_carla_scenario import (
    CarlaWorldPose,
    ComparisonRule,
    TimeToCollisionCondition,
)
from autoware_carla_scenario.conditions.composition.distance_measure import (
    RelativeDistanceType,
    separation,
)


def _actor(
    role: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extent: tuple[float, float, float] | None = (2.5, 1.0, 0.75),
    heading: tuple[float, float] = (1.0, 0.0),
    box_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MagicMock:
    """A fake actor; *box_offset* is ``bounding_box.location``, in its own frame."""
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(*heading, 0.0)
    actor.get_transform.return_value = transform
    if extent is None:
        actor.bounding_box = None
    else:
        box = MagicMock()
        box.extent = carla.Vector3D(*extent)
        box.location = carla.Location(*box_offset)
        box.rotation = carla.Rotation()
        actor.bounding_box = box
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


_STOP_LINE = CarlaWorldPose(x=100.0, y=0.0, z=0.0)


def _ttc(condition: TimeToCollisionCondition, world: MagicMock) -> float:
    result = condition.check(world, 0.0)
    assert result is not None, "the condition did not fire"
    match = re.search(r"\(([0-9.]+) s\)", result.message)
    assert match is not None, result.message
    return float(match.group(1))


def _to_stop_line(**kwargs: object) -> TimeToCollisionCondition:
    return TimeToCollisionCondition(
        source="Ego",
        position=_STOP_LINE,
        value=1e9,
        rule=ComparisonRule.LESS_THAN,
        label="ttc_to_stop_line",
        **kwargs,  # type: ignore[arg-type]
    )


class TestPositionTarget:
    def test_the_time_is_the_gap_over_the_subject_speed(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        assert math.isclose(_ttc(_to_stop_line(), world), 10.0, abs_tol=0.01)

    def test_driving_away_from_the_place_never_fires(self) -> None:
        """A place behind you has no time-to-collision, exactly as an entity does not."""
        world = _world(_actor("Ego", (0, 0, 0), velocity=(-10.0, 0.0, 0.0)))
        condition = TimeToCollisionCondition(
            source="Ego",
            position=_STOP_LINE,
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        assert condition.check(world, 0.0) is None

    def test_a_stationary_subject_never_fires(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0)))
        condition = TimeToCollisionCondition(
            source="Ego",
            position=_STOP_LINE,
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            label="ttc",
        )
        assert condition.check(world, 0.0) is None

    def test_only_the_component_towards_the_place_counts(self) -> None:
        """Crossing traffic closes on the conflict point more slowly than it drives."""
        world = _world(_actor("Ego", (0, 0, 0), velocity=(6.0, 8.0, 0.0)))
        # The place is due +x, so only the 6 m/s component closes on it.
        assert math.isclose(_ttc(_to_stop_line(), world), 100.0 / 6.0, abs_tol=0.01)

    def test_the_message_names_the_place(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        result = _to_stop_line().check(world, 0.0)
        assert result is not None
        assert "(100.0, 0.0)" in result.message


class TestFreespace:
    def test_the_range_starts_at_the_subject_bumper(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0), extent=(2.5, 1.0, 0.75))
        )
        assert math.isclose(
            _ttc(_to_stop_line(edge_to_edge=True), world), 9.75, abs_tol=0.01
        )

    def test_both_bumpers_count_for_an_entity_target(self) -> None:
        world = _world(
            _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)),
            _actor("npc1", (100, 0, 0)),
        )
        condition = TimeToCollisionCondition(
            source="Ego",
            target="npc1",
            value=1e9,
            rule=ComparisonRule.LESS_THAN,
            edge_to_edge=True,
            label="ttc",
        )
        assert math.isclose(_ttc(condition, world), 9.5, abs_tol=0.01)

    def test_edge_to_edge_is_off_by_default(self) -> None:
        world = _world(_actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0)))
        assert math.isclose(_ttc(_to_stop_line(), world), 10.0, abs_tol=0.01)


# A vehicle whose box sits 1.3 m forward of its origin, as CARLA's cars do.
_OFFSET = (1.3, 0.0, 0.7)
_EXTENT = (2.4, 1.0, 0.75)


def _between(
    source: MagicMock, target: MagicMock, **kwargs: object
) -> TimeToCollisionCondition:
    return TimeToCollisionCondition(
        source="Ego",
        target="npc1",
        label="ttc",
        **kwargs,  # type: ignore[arg-type]
    )


def _head_on(
    heading: tuple[float, float] = (1.0, 0.0),
) -> tuple[MagicMock, MagicMock]:
    """Ego at the origin and npc1 20 m ahead along *heading*, facing each other
    and closing at 20 m/s: 12.6 m between the bumpers, 0.63 s apart."""
    hx, hy = heading
    ego = _actor(
        "Ego",
        (0.0, 0.0, 0.0),
        velocity=(10.0 * hx, 10.0 * hy, 0.0),
        extent=_EXTENT,
        heading=heading,
        box_offset=_OFFSET,
    )
    npc = _actor(
        "npc1",
        (20.0 * hx, 20.0 * hy, 0.0),
        velocity=(-10.0 * hx, -10.0 * hy, 0.0),
        extent=_EXTENT,
        heading=(-hx, -hy),
        box_offset=_OFFSET,
    )
    return ego, npc


class TestFreespaceFromTheBoxCentre:
    """Edge to edge runs between the box centres, not the actor origins (#888)."""

    def test_head_on_fires_at_the_true_time_to_collision(self) -> None:
        ego, npc = _head_on()
        condition = _between(
            ego, npc, value=0.7, rule=ComparisonRule.LESS_THAN, edge_to_edge=True
        )
        assert math.isclose(_ttc(condition, _world(ego, npc)), 0.63, abs_tol=0.01)

    def test_the_range_is_the_separation(self) -> None:
        """The numerator is what `separation(edge_to_edge=True)` reports."""
        ego, npc = _head_on()
        gap = separation(ego, npc, edge_to_edge=True)
        longitudinal = separation(
            ego,
            npc,
            distance_type=RelativeDistanceType.LONGITUDINAL,
            edge_to_edge=True,
        )
        assert gap is not None and longitudinal is not None
        assert math.isclose(gap, 12.6, abs_tol=1e-4)
        assert math.isclose(longitudinal, 12.6, abs_tol=1e-4)
        condition = _between(ego, npc, value=1e9, edge_to_edge=True)
        assert math.isclose(_ttc(condition, _world(ego, npc)), gap / 20.0, abs_tol=0.01)

    def test_a_yawed_pair_is_measured_the_same(self) -> None:
        """Heading along +y, the offsets turn with the actors."""
        ego, npc = _head_on(heading=(0.0, 1.0))
        condition = _between(ego, npc, value=1e9, edge_to_edge=True)
        assert math.isclose(_ttc(condition, _world(ego, npc)), 0.63, abs_tol=0.01)

    def test_a_diagonal_pair_is_measured_the_same(self) -> None:
        h = math.sqrt(0.5)
        ego, npc = _head_on(heading=(h, h))
        condition = _between(ego, npc, value=1e9, edge_to_edge=True)
        # The box centres are on the diagonal, 17.4 m apart; each box reaches
        # its half length along it.
        assert math.isclose(_ttc(condition, _world(ego, npc)), 0.63, abs_tol=0.01)

    def test_a_place_target_takes_only_the_source_offset(self) -> None:
        world = _world(
            _actor(
                "Ego",
                (0.0, 0.0, 0.0),
                velocity=(10.0, 0.0, 0.0),
                extent=_EXTENT,
                box_offset=_OFFSET,
            )
        )
        # 100 m to the line, less the 1.3 m offset and the 2.4 m half length.
        assert math.isclose(
            _ttc(_to_stop_line(edge_to_edge=True), world), 9.63, abs_tol=0.01
        )

    def test_a_box_on_the_origin_is_unchanged(self) -> None:
        ego = _actor("Ego", (0, 0, 0), velocity=(10.0, 0.0, 0.0))
        npc = _actor("npc1", (100, 0, 0), heading=(-1.0, 0.0))
        condition = _between(ego, npc, value=1e9, edge_to_edge=True)
        assert math.isclose(_ttc(condition, _world(ego, npc)), 9.5, abs_tol=0.01)

    def test_centre_to_centre_still_runs_between_the_origins(self) -> None:
        """Without edge_to_edge the range is `separation(edge_to_edge=False)`'s."""
        ego, npc = _head_on()
        gap = separation(ego, npc)
        assert gap is not None and math.isclose(gap, 20.0)
        condition = _between(ego, npc, value=1e9)
        assert math.isclose(_ttc(condition, _world(ego, npc)), 1.0, abs_tol=0.01)


class TestConstruction:
    def test_neither_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TimeToCollisionCondition(source="Ego", value=4.0, label="neither")

    def test_both_targets_are_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TimeToCollisionCondition(
                source="Ego",
                target="npc1",
                position=_STOP_LINE,
                value=4.0,
                label="both",
            )


class TestDetails:
    def test_a_place_target_reports_where_it_resolved_to(self) -> None:
        details = _to_stop_line().get_details()
        assert math.isclose(details["target_x"], 100.0)
        assert "target" not in details

    def test_an_entity_target_reports_the_entity(self) -> None:
        condition = TimeToCollisionCondition(
            source="Ego", target="npc1", value=4.0, label="ttc"
        )
        details = condition.get_details()
        assert details["target"] == "npc1"
        assert "target_x" not in details
