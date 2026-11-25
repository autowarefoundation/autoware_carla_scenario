"""Unit tests for the ego's debug markers, which the trajectory log records."""

from __future__ import annotations

from types import SimpleNamespace

from autoware_carla_scenario.debug_markers import DebugMarkers, DebugPolyline
from autoware_carla_scenario.entity.ego import EgoVehicle
from autoware_carla_scenario.scenario_runner import _ego_markers


def _markers(*points: tuple[float, float, float]) -> DebugMarkers:
    markers = DebugMarkers()
    markers.add_polyline(DebugPolyline(list(points)))
    return markers


def test_an_ego_shows_nothing_by_default() -> None:
    assert not EgoVehicle().debug_markers()


def test_the_runner_takes_what_the_ego_shows() -> None:
    markers = _markers((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    ego = SimpleNamespace(debug_markers=lambda: markers)
    assert _ego_markers(ego) is markers  # type: ignore[arg-type]


def test_an_ego_whose_markers_fail_does_not_fail_the_run() -> None:
    def _fail() -> DebugMarkers:
        raise RuntimeError("boom")

    assert _ego_markers(SimpleNamespace(debug_markers=_fail)) is None  # type: ignore[arg-type]
