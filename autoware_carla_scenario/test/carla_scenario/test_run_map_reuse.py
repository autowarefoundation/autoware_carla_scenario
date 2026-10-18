"""A run reads the CARLA map its MapManager holds, not a fresh ``get_map()``.

Every ``world.get_map()`` builds a whole client-side ``carla.Map`` -- on a
large map hundreds of MB and seconds of work -- so the per-case paths of a run
take the one :class:`MapManager` built for the batch.  Without one they ask the
world, as they always have.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario.coordinate.map_manager import MapManager


def _spawn_point(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z))


def _held_map(name: str = "Carla/Maps/Odaiba") -> MagicMock:
    carla_map = MagicMock(name="held_map")
    carla_map.name = name
    carla_map.get_spawn_points.return_value = [_spawn_point(1.0, 2.0, 3.5)]
    carla_map.to_opendrive.return_value = "<OpenDRIVE/>"
    return carla_map


def _world() -> MagicMock:
    """A world whose get_map() fails the test if anything calls it."""
    world = MagicMock(name="world")
    world.get_map.side_effect = AssertionError("world.get_map() was called")
    return world


@pytest.fixture
def held_map() -> Generator[MagicMock, None, None]:
    """A MapManager holding a CARLA map, as a queue's start leaves it."""
    saved = MapManager._instance
    MapManager._instance = None
    carla_map = _held_map()
    MapManager.get_instance()._carla_map = carla_map
    yield carla_map
    MapManager._instance = saved


@pytest.fixture
def no_map_manager() -> Generator[None, None, None]:
    """No MapManager map, as a runner or tool without a queue has."""
    saved = MapManager._instance
    MapManager._instance = None
    yield
    MapManager._instance = saved


class TestWithAHeldMap:
    def test_the_spawn_point_z_comes_off_the_held_map(
        self, held_map: MagicMock
    ) -> None:
        from autoware_carla_scenario.coordinate.snap import (
            _z_from_nearest_spawn_point,
        )

        assert _z_from_nearest_spawn_point(1.0, 2.0, _world()) == 3.5

    def test_a_carla_pose_snaps_on_the_held_map(self, held_map: MagicMock) -> None:
        from autoware_carla_scenario.coordinate.poses import CarlaWorldPose
        from autoware_carla_scenario.coordinate.snap import (
            GroundProjectionConfig,
            _snap_carla_via_waypoint,
        )

        held_map.get_waypoint.return_value.transform.location = SimpleNamespace(
            x=1.0, y=2.0, z=0.0
        )
        with patch(
            "autoware_carla_scenario.coordinate.snap.refine_z_with_ground_projection",
            side_effect=lambda x, y, z, world, **kw: z,
        ):
            snapped = _snap_carla_via_waypoint(
                CarlaWorldPose(x=1.2, y=2.1, z=0.0),
                _world(),
                ground_projection=GroundProjectionConfig(),
            )

        assert (snapped.x, snapped.y, snapped.z) == (1.0, 2.0, 3.5)
        held_map.get_waypoint.assert_called_once()

    def test_an_opendrive_pose_snaps_on_the_held_map(self, held_map: MagicMock) -> None:
        from autoware_carla_scenario.coordinate.poses import OpenDrivePose
        from autoware_carla_scenario.coordinate.snap import (
            GroundProjectionConfig,
            _snap_opendrive_via_waypoint_xodr,
        )

        tf = held_map.get_waypoint_xodr.return_value.transform
        tf.location = SimpleNamespace(x=5.0, y=6.0, z=0.5)
        tf.rotation = SimpleNamespace(yaw=90.0)
        with patch(
            "autoware_carla_scenario.coordinate.snap.refine_z_with_ground_projection",
            side_effect=lambda x, y, z, world, **kw: z,
        ):
            snapped = _snap_opendrive_via_waypoint_xodr(
                OpenDrivePose(road_id="7", lane_id=-1, s=3.0),
                _world(),
                ground_projection=GroundProjectionConfig(),
            )

        assert (snapped.x, snapped.y, snapped.yaw) == (5.0, 6.0, 90.0)
        held_map.get_waypoint_xodr.assert_called_once_with(7, -1, 3.0)

    def test_the_runner_names_the_map_off_the_held_one(
        self, held_map: MagicMock
    ) -> None:
        from autoware_carla_scenario.scenario_runner import _map_name_of

        assert _map_name_of(_world()) == "Carla/Maps/Odaiba"

    def test_the_trajectory_names_the_held_map(
        self, held_map: MagicMock, tmp_path: Path
    ) -> None:
        import json

        from autoware_carla_scenario.trajectory_recorder import TrajectoryRecorder

        world = _world()
        world.id = 7
        world.get_actors.return_value.filter.return_value = []
        recorder = TrajectoryRecorder(tmp_path / "run.trajectory.jsonl")
        recorder.start(world)
        recorder.close()

        first = json.loads(
            (tmp_path / "run.trajectory.jsonl").read_text().splitlines()[0]
        )
        assert first == {
            "type": "world",
            "schema_version": 2,
            "map": "Carla/Maps/Odaiba",
            "episode": 7,
        }

    def test_the_renderer_walks_the_held_map(self, held_map: MagicMock) -> None:
        from autoware_carla_scenario.driver.renderer import RendererDataBuilder

        builder = RendererDataBuilder(_world(), MagicMock(), MagicMock())

        assert builder._map is held_map

    def test_the_metrics_read_the_held_map(self, held_map: MagicMock) -> None:
        from autoware_carla_scenario import metrics

        saved = metrics._CACHE.carla_map
        metrics._CACHE.carla_map = None
        try:
            assert metrics._map(_world()) is held_map
        finally:
            metrics._CACHE.carla_map = saved

    def test_the_sumo_backend_reads_the_held_map_opendrive(
        self, held_map: MagicMock
    ) -> None:
        from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

        context = SimpleNamespace(xodr_path=None, world=_world())

        assert (
            SumoTrafficBackend._opendrive(MagicMock(), context)  # type: ignore[arg-type]
            == "<OpenDRIVE/>"
        )

    def test_a_trajectory_resolves_on_the_held_map(self, held_map: MagicMock) -> None:
        from autoware_carla_scenario.actions.follow_trajectory import (
            _resolve_in_world,
        )

        with (
            patch(
                "autoware_carla_scenario.trajectory.resolve.road_height"
            ) as road_height,
            patch("autoware_carla_scenario.trajectory.resolve.resolve_trajectory"),
        ):
            _resolve_in_world(MagicMock(), _world(), MagicMock())

        road_height.assert_called_once_with(held_map)

    def test_the_z_offset_reads_the_map_built_for_it(
        self, no_map_manager: None
    ) -> None:
        """MapManager builds the map once, and the z offset reads that one."""
        manager = MapManager()  # the singleton: no_map_manager puts the old one back
        manager._lanelet_map = MagicMock()
        manager._mgrs_offset = (0.0, 0.0)
        manager._carla_map = _held_map()
        manager._carla_map.get_spawn_points.return_value = []

        assert manager._z_offset_from_spawn_points(_world()) is None


class TestWithoutAMapManager:
    """A runner or tool with no MapManager asks the world, as it always has."""

    def test_the_spawn_point_z_comes_off_the_world(self, no_map_manager: None) -> None:
        from autoware_carla_scenario.coordinate.snap import (
            _z_from_nearest_spawn_point,
        )

        world = MagicMock()
        world.get_map.return_value = _held_map()

        assert _z_from_nearest_spawn_point(1.0, 2.0, world) == 3.5
        world.get_map.assert_called_once_with()

    def test_the_runner_names_the_world_map(self, no_map_manager: None) -> None:
        from autoware_carla_scenario.scenario_runner import _map_name_of

        world = MagicMock()
        world.get_map.return_value = _held_map("Carla/Maps/Town10HD_Opt")

        assert _map_name_of(world) == "Carla/Maps/Town10HD_Opt"

    def test_a_runner_that_loaded_no_map_reloads_the_current_one(
        self, no_map_manager: None
    ) -> None:
        from autoware_carla_scenario import ScenarioRunner

        with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
            runner = ScenarioRunner(MagicMock())
        client = MagicMock()
        client.get_world.return_value.get_map.return_value = _held_map(
            "Carla/Maps/Town10HD_Opt"
        )
        runner._client = client

        runner.reset_world("case")

        client.load_world.assert_called_once_with("Town10HD_Opt")
