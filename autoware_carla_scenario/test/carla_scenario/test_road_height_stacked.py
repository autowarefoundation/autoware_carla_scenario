"""The road height under a point where two roads are stacked (#1585).

CARLA projects a point onto the road nearest it in 3D, so a query at z = 0
reads whichever deck is nearest 0.  On the fixture map lanelet 132 runs over
lanelet 552: at (47.36, -79.03) the lower road is at about -8.1 and the
overpass at about -1.3, and a prop placed on the lower road was lifted onto
the bridge.  The fake map here has those two decks, and every caller of
:func:`~autoware_carla_scenario.trajectory.resolve.road_height` is checked to
read near the height it knows of the point.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    CarlaWorldPose,
    EntityPositionDistanceCondition,
    SpawnAction,
)
from autoware_carla_scenario.actions import spawn as spawn_module
from autoware_carla_scenario.trajectory.model import (
    MapPose,
    Trajectory,
    TrajectoryVertex,
)
from autoware_carla_scenario.trajectory.resolve import (
    resolve_trajectory,
    road_height,
)

#: The two decks over the same (x, y): the lower road and the overpass.
LOWER_Z = -8.14
UPPER_Z = -1.29
X, Y = 47.36, -79.03


class _StackedMap:
    """A CARLA map whose every point has two roads, one over the other: a
    waypoint is projected onto the deck nearest the queried location in 3D,
    as CARLA does."""

    def __init__(self) -> None:
        self.queried: list[float] = []

    def get_waypoint(self, location: Any, project_to_road: bool = True) -> Any:
        self.queried.append(float(location.z))
        deck = min((LOWER_Z, UPPER_Z), key=lambda z: abs(z - float(location.z)))
        return SimpleNamespace(
            transform=SimpleNamespace(
                location=SimpleNamespace(x=location.x, y=location.y, z=deck)
            )
        )


class _Prop:
    def __init__(self, blueprint: str, transform: Any) -> None:
        self.blueprint, self.transform = blueprint, transform

    def destroy(self) -> None:
        pass


class _World:
    id = 1

    def __init__(self) -> None:
        self.map = _StackedMap()
        self.spawned: list[_Prop] = []

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(filter=lambda name: [name])

    def get_map(self) -> Any:
        return self.map

    def get_actors(self) -> list[Any]:
        return []

    def try_spawn_actor(self, blueprint: str, transform: Any) -> Optional[_Prop]:
        prop = _Prop(blueprint, transform)
        self.spawned.append(prop)
        return prop


@pytest.fixture(autouse=True)
def _no_props_standing() -> Iterator[None]:
    spawn_module._STANDING.clear()
    yield
    spawn_module._STANDING.clear()


class TestRoadHeight:
    def test_a_zero_hint_reads_the_deck_nearest_zero(self) -> None:
        height = road_height(_StackedMap())  # type: ignore[arg-type]
        assert height(X, Y, 0.0) == pytest.approx(UPPER_Z)

    def test_a_hint_reads_the_road_nearest_it(self) -> None:
        stacked = _StackedMap()
        height = road_height(stacked)  # type: ignore[arg-type]
        assert height(X, Y, -8.47) == pytest.approx(LOWER_Z)
        assert height(X, Y, -1.51) == pytest.approx(UPPER_Z)
        assert stacked.queried == pytest.approx([-8.47, -1.51])


class TestTrajectory:
    @staticmethod
    def _resolve(position: Any) -> tuple[float, float, Optional[float], None]:
        # MapPose and CarlaWorldPose alike: x, y and z as written.
        return position.x, position.y, position.z, None

    def _heights(self, vertices: list[Any]) -> list[float]:
        trajectory = Trajectory(
            "t",
            [TrajectoryVertex(position, None) for position in vertices],
        )
        ground = road_height(_StackedMap())  # type: ignore[arg-type]
        return list(resolve_trajectory(trajectory, ground, resolve=self._resolve).zs)

    def test_a_vertex_without_a_height_stays_on_the_road_of_the_last_known(
        self,
    ) -> None:
        heights = self._heights(
            [CarlaWorldPose(X - 5.0, Y, -8.47), MapPose(X, Y), MapPose(X + 5.0, Y)]
        )
        assert heights == pytest.approx([-8.47, LOWER_Z, LOWER_Z])

    def test_before_the_first_known_height_it_reads_near_that_one(self) -> None:
        heights = self._heights([MapPose(X, Y), CarlaWorldPose(X + 5.0, Y, -8.47)])
        assert heights == pytest.approx([LOWER_Z, -8.47])

    def test_with_no_height_known_it_reads_near_zero(self) -> None:
        heights = self._heights([MapPose(X, Y), MapPose(X + 5.0, Y)])
        assert heights == pytest.approx([UPPER_Z, UPPER_Z])


class TestProp:
    def test_a_prop_on_the_lower_road_stands_on_it(self) -> None:
        world = _World()
        added = SpawnAction(name="cone", position=CarlaWorldPose(X, Y, -8.47))
        assert added.apply(world)  # type: ignore[arg-type]
        (cone,) = world.spawned
        # On the lower road, a little above it -- not on the overpass.
        assert cone.transform.location.z == pytest.approx(LOWER_Z + 0.05)

    def test_a_prop_on_the_overpass_stands_on_it(self) -> None:
        world = _World()
        added = SpawnAction(name="cone", position=CarlaWorldPose(X, Y, -1.51))
        assert added.apply(world)  # type: ignore[arg-type]
        assert world.spawned[0].transform.location.z == pytest.approx(UPPER_Z + 0.05)

    def test_a_shared_ground_is_read_near_the_prop(self) -> None:
        """The construction zone's path: the ground fetched once for every
        prop, each read near its own place's height."""
        world = _World()
        ground = road_height(world.map)  # type: ignore[arg-type]
        library = world.get_blueprint_library()
        for name, z in (("low", -8.47), ("high", -1.51)):
            SpawnAction(name=name, position=CarlaWorldPose(X, Y, z)).apply(
                world,  # type: ignore[arg-type]
                library,
                ground,
            )
        assert [p.transform.location.z for p in world.spawned] == pytest.approx(
            [LOWER_Z + 0.05, UPPER_Z + 0.05]
        )


class TestEntityPositionDistance:
    def test_the_place_is_on_the_road_the_entity_is_on(self) -> None:
        condition = EntityPositionDistanceCondition(
            entity_name="Ego",
            position=MapPose(x=X, y=Y),
            value=1.0,
            vertical=True,
            label="under_the_bridge",
        )
        actor = MagicMock()
        actor.attributes = {"role_name": "Ego"}
        actor.get_location.return_value = carla.Location(X, Y, -8.0)
        world = MagicMock()
        world.get_actors.return_value = [actor]
        with (
            patch(
                "autoware_carla_scenario.trajectory.resolve.resolve_position",
                return_value=(X, Y, None, None),
            ),
            patch(
                "autoware_carla_scenario.coordinate.map_manager.run_carla_map",
                return_value=_StackedMap(),
            ),
        ):
            distance = condition._measure(world)
        assert distance is not None
        assert math.isclose(distance, abs(LOWER_Z - -8.0), abs_tol=1e-6)
        assert math.isclose(condition.get_details()["target_z"], LOWER_Z)
