"""A hidden entity is absent as the target of a pairwise condition (#922).

An entity spawned hidden is parked under the map (below
:data:`~autoware_carla_scenario.conditions.entity_existence.HIDDEN_BELOW_M`)
until it is brought in.  The guard of a composition condition already reads its
own entity that way; the second entity -- a target, a reference -- used to be
measured where it stood, horizontally, as if it were on the road right above.
So a TTC, a headway, a distance or a relative speed fired on an NPC that was
not in the scene yet, and only when it was named second.

Each condition is checked three ways: with the second entity parked under the
map it does not hold; with the same entity at road height it does (so the
first case is not quiet for some other reason); and with the two swapped it
gives the same answer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterator, Optional
from unittest.mock import MagicMock

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    ComparisonRule,
    DistanceCoordinateSystem,
    EntityDistanceCondition,
    RelativeSpeedCondition,
    SpeedCondition,
    SpeedCoordinateSystem,
    SpeedDirection,
    TimeHeadwayCondition,
    TimeToCollisionCondition,
)
from autoware_carla_scenario.conditions.base import BaseCondition
from autoware_carla_scenario.conditions.entity_existence import (
    HIDDEN_BELOW_M,
    find_present_actor_by_role_name,
)
from autoware_carla_scenario.coordinate.map_manager import MapManager
from autoware_carla_scenario.coordinate.poses import CarlaWorldPose, OpenDrivePose
from autoware_carla_scenario.coordinate.transform import to_carla_world

#: How far under its spawn a hidden entity is parked (as the spawn action does).
PARKED_BELOW_M = 500.0

#: The nishishinjuku fixture map, kept at the repository root.
_DATA = Path(__file__).resolve().parents[3] / "data"
XODR_PATH = _DATA / "nishishinjuku_carla.xodr"
OSM_PATH = _DATA / "nishishinjuku.osm"
#: A road with room for two vehicles 20 m apart (test_lane_distance.py).
ROAD = "144"


def _actor(
    role: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    heading: tuple[float, float] = (1.0, 0.0),
) -> MagicMock:
    """Return a fake CARLA actor at *position*, moving at *velocity*."""
    actor = MagicMock()
    actor.attributes = {"role_name": role}
    actor.get_location.return_value = carla.Location(*position)
    actor.get_velocity.return_value = carla.Vector3D(*velocity)
    transform = MagicMock()
    transform.get_forward_vector.return_value = carla.Vector3D(*heading, 0.0)
    actor.get_transform.return_value = transform
    return actor


def _world(*actors: MagicMock) -> MagicMock:
    world = MagicMock()
    world.get_actors.return_value = list(actors)
    return world


def _parked(z: float) -> float:
    """Return where an entity standing at height *z* is parked while hidden."""
    return z - PARKED_BELOW_M


#: Builds a condition with *source* and *target* as its two entities.
Pair = Callable[[str, str], BaseCondition]


def _fires(pair: Pair, source: str, target: str, world: MagicMock) -> bool:
    result: Optional[object] = pair(source, target).check(world, 1.0)
    return result is not None


def _assert_hidden_is_absent(
    pair: Pair,
    ego: Callable[[float], MagicMock],
    npc: Callable[[float], MagicMock],
    z: float = 0.3,
) -> None:
    """The three claims of this module, for one condition.

    *ego* and *npc* build the two actors at a given height; on the road both
    stand at *z*, and the npc is hidden at :func:`_parked` of it.
    """
    on_road = _world(ego(z), npc(z))
    hidden = _world(ego(z), npc(_parked(z)))

    # At road height the pair fires, whichever way round it is named.
    assert _fires(pair, "Ego", "npc1", on_road)
    assert _fires(pair, "npc1", "Ego", on_road)
    # Parked under the map, the npc is not there, as target or as source.
    assert not _fires(pair, "Ego", "npc1", hidden)
    assert not _fires(pair, "npc1", "Ego", hidden)


class TestTheHelper:
    def test_an_actor_on_the_road_is_present(self) -> None:
        actor = _actor("npc1", (0.0, 0.0, 0.3))
        assert find_present_actor_by_role_name(_world(actor), "npc1") is actor

    def test_an_actor_parked_under_the_map_is_absent(self) -> None:
        world = _world(_actor("npc1", (0.0, 0.0, HIDDEN_BELOW_M - 0.1)))
        assert find_present_actor_by_role_name(world, "npc1") is None

    def test_an_actor_not_in_the_world_is_absent(self) -> None:
        assert find_present_actor_by_role_name(_world(), "npc1") is None


class TestEntityFrame:
    """Head-on, 15 m apart, each closing at 10 m/s: every measure is defined
    for either order, so a swap may not change the answer."""

    @staticmethod
    def _ego(z: float) -> MagicMock:
        return _actor("Ego", (0.0, 0.0, z), (10.0, 0.0, 0.0), heading=(1.0, 0.0))

    @staticmethod
    def _npc(z: float) -> MagicMock:
        return _actor("npc1", (15.0, 0.0, z), (-10.0, 0.0, 0.0), heading=(-1.0, 0.0))

    def test_time_to_collision(self) -> None:
        # 15 m closing at 20 m/s: 0.75 s.
        _assert_hidden_is_absent(
            lambda s, t: TimeToCollisionCondition(
                source=s,
                target=t,
                value=2.0,
                rule=ComparisonRule.LESS_THAN,
                label="ttc",
            ),
            self._ego,
            self._npc,
        )

    def test_time_headway(self) -> None:
        # 15 m ahead at 10 m/s: 1.5 s, for either of the two.
        _assert_hidden_is_absent(
            lambda s, t: TimeHeadwayCondition(
                source=s,
                target=t,
                value=2.0,
                rule=ComparisonRule.LESS_THAN,
                label="thw",
            ),
            self._ego,
            self._npc,
        )

    def test_entity_distance(self) -> None:
        _assert_hidden_is_absent(
            lambda s, t: EntityDistanceCondition(
                source=s,
                target=t,
                value=20.0,
                rule=ComparisonRule.LESS_THAN,
                label="dist",
            ),
            self._ego,
            self._npc,
        )

    def test_relative_speed(self) -> None:
        # |v_subject| - |v_reference| = 10 - 10 = 0 m/s, either way round.
        _assert_hidden_is_absent(
            lambda s, t: RelativeSpeedCondition(
                entity_name=s,
                reference_entity_name=t,
                value=0.0,
                rule=ComparisonRule.EQUAL_TO,
                direction=SpeedDirection.MAGNITUDE,
                label="rel",
            ),
            self._ego,
            self._npc,
        )

    def test_the_parked_npc_still_reads_near_horizontally(self) -> None:
        """The case is the one the issue is about, not a far-away target:
        straight down from where it will appear, the npc is 15 m off."""
        npc = self._npc(_parked(0.3))
        location = npc.get_location()
        assert math.hypot(location.x, location.y) == pytest.approx(15.0)
        assert location.z < HIDDEN_BELOW_M


class TestSpeedInAnEntityFrame:
    """A speed taken along a hidden reference's axes has nothing to be
    taken along."""

    @staticmethod
    def _condition() -> SpeedCondition:
        return SpeedCondition(
            entity_name="Ego",
            value=5.0,
            rule=ComparisonRule.GREATER_THAN,
            direction=SpeedDirection.LONGITUDINAL,
            coordinate_system=SpeedCoordinateSystem.ENTITY,
            reference_entity_name="npc1",
            label="along_npc",
        )

    def test_a_reference_on_the_road_answers(self) -> None:
        world = _world(
            _actor("Ego", (0.0, 0.0, 0.3), (10.0, 0.0, 0.0)),
            _actor("npc1", (15.0, 0.0, 0.3)),
        )
        assert self._condition().check(world, 1.0) is not None

    def test_a_parked_reference_is_absent(self) -> None:
        world = _world(
            _actor("Ego", (0.0, 0.0, 0.3), (10.0, 0.0, 0.0)),
            _actor("npc1", (15.0, 0.0, _parked(0.3))),
        )
        assert self._condition().check(world, 1.0) is None


@pytest.fixture(scope="module")
def loaded_map() -> Iterator[MapManager]:
    """Load the fixture map once for this module."""
    saved = MapManager._instance
    MapManager.reset()
    manager = MapManager.get_instance()
    manager.initialize(XODR_PATH, OSM_PATH)
    yield manager
    MapManager._instance = saved


def _at(s: float) -> CarlaWorldPose:
    """Return the CARLA world pose of the centre of lane -1 of ROAD at *s*."""
    return to_carla_world(OpenDrivePose(road_id=ROAD, lane_id=-1, s=s, t=0.0))


def _road_direction(s: float) -> tuple[float, float]:
    """Return the road's unit direction at *s*."""
    a, b = _at(s), _at(s + 1.0)
    length = math.hypot(b.x - a.x, b.y - a.y)
    return (b.x - a.x) / length, (b.y - a.y) / length


class TestLaneFrame:
    def test_time_to_collision(self, loaded_map: MapManager) -> None:
        """The ego 20 m up the lane behind a standing npc, at 10 m/s: 2 s."""
        behind, ahead = _at(10.0), _at(30.0)
        ux, uy = _road_direction(10.0)

        def ego(_z: float) -> MagicMock:
            return _actor(
                "Ego", (behind.x, behind.y, behind.z), (ux * 10.0, uy * 10.0, 0.0)
            )

        def npc(z: float) -> MagicMock:
            # *z* is 0.3 on the road and parked below it when hidden: keep
            # the offset from the road's own height.
            return _actor("npc1", (ahead.x, ahead.y, ahead.z + z - 0.3))

        _assert_hidden_is_absent(
            lambda s, t: TimeToCollisionCondition(
                source=s,
                target=t,
                value=5.0,
                rule=ComparisonRule.LESS_THAN,
                coordinate_system=DistanceCoordinateSystem.LANE,
                label="ttc_lane",
            ),
            ego,
            npc,
        )
