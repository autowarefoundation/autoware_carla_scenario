"""A background vehicle goes on a driving lane, or nowhere (#976).

The TrafficManager backend used to spawn a background vehicle wherever it was
asked, crosswalks and footways included.  Now it snaps the spot to the nearest
``Driving`` lane, facing along it, and refuses one farther than
:data:`~autoware_carla_scenario.traffic.base.BACKGROUND_MAX_OFFSET_M` from
any, as SUMO does with the same value.  CARLA, its map and TraCI are faked.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
import typesafe_carla.carla as carla

import autoware_carla_scenario.traffic.base as base
import autoware_carla_scenario.traffic.sumo.backend as sumo_module
import autoware_carla_scenario.traffic.traffic_manager as tm_module
from autoware_carla_scenario.traffic.base import BACKGROUND_MAX_OFFSET_M
from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend

#: The fake lane: its centre line is x = 0, and it runs towards +y.
_LANE_YAW = 90.0


def _waypoint(
    x: float,
    y: float,
    yaw: float,
    lane_type: Any = carla.LaneType.Driving,
    left: Any = None,
    right: Any = None,
) -> Any:
    """A fake ``carla.Waypoint`` on a lane centred at *x*, heading *yaw*."""
    return SimpleNamespace(
        transform=carla.Transform(carla.Location(x, y, 1.25), carla.Rotation(yaw=yaw)),
        lane_type=lane_type,
        get_left_lane=lambda: left,
        get_right_lane=lambda: right,
    )


class _Lane:
    """A map with one driving lane, along x = 0; every query is kept."""

    def __init__(self, present: bool = True) -> None:
        self.present = present
        self.queries: List[dict] = []

    def get_waypoint(self, location: Any, **kwargs: Any) -> Optional[Any]:
        self.queries.append(kwargs)
        if not self.present:
            return None
        return _waypoint(0.0, location.y, _LANE_YAW)


class _TwoWayRoad(_Lane):
    """A two-way road whose centre line is x = 0: the nearest lane, centred
    at x = 1.6, runs towards -y; across the centre line, at x = -1.6, its
    left neighbour runs towards +y.  *beside* replaces that neighbour."""

    _UNSET: Any = object()

    def __init__(self, beside: Any = _UNSET) -> None:
        super().__init__()
        self.beside = beside

    def get_waypoint(self, location: Any, **kwargs: Any) -> Optional[Any]:
        self.queries.append(kwargs)
        beside = self.beside
        if beside is self._UNSET:
            beside = _waypoint(-1.6, location.y, _LANE_YAW)
        sidewalk = _waypoint(3.2, location.y, -_LANE_YAW, carla.LaneType.Sidewalk)
        return _waypoint(1.6, location.y, -_LANE_YAW, left=beside, right=sidewalk)


class _Actor:
    id = 3

    def __init__(self) -> None:
        self.velocity: Optional[tuple[float, float]] = None

    def set_target_velocity(self, velocity: Any) -> None:
        self.velocity = (velocity.x, velocity.y)


class _World:
    def __init__(self) -> None:
        self.spawned: List[Any] = []
        self.actor = _Actor()

    def get_blueprint_library(self) -> Any:
        blueprint = SimpleNamespace(has_attribute=lambda name: False)
        return SimpleNamespace(filter=lambda pattern: [blueprint])

    def try_spawn_actor(self, blueprint: Any, transform: Any) -> _Actor:
        self.spawned.append(transform)
        return self.actor


def _spot(x: float, yaw: float = 0.0) -> Any:
    """A spot *x* metres beside the lane's centre, facing *yaw*."""
    return carla.Transform(carla.Location(x, 5.0, 0.0), carla.Rotation(yaw=yaw))


def _tm_spawn(
    monkeypatch: pytest.MonkeyPatch, spot: Any, lane: Optional[_Lane] = None
) -> tuple[Optional[str], _World, _Lane]:
    road = lane or _Lane()
    monkeypatch.setattr(tm_module, "run_carla_map", lambda world: road)
    world = _World()
    handle = TrafficManagerBackend().spawn_background(world, spot, speed_kmh=36.0)
    return handle, world, road


class TestTrafficManager:
    def test_a_spot_10_m_from_any_lane_is_refused_and_said_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=tm_module.__name__):
            handle, world, _road = _tm_spawn(monkeypatch, _spot(10.0))
        assert handle is None
        assert world.spawned == []
        assert any(
            "refused" in r.getMessage() and r.levelno >= logging.INFO
            for r in caplog.records
        )

    def test_no_driving_lane_at_all_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=tm_module.__name__):
            handle, world, _road = _tm_spawn(
                monkeypatch, _spot(0.0), _Lane(present=False)
            )
        assert handle is None
        assert world.spawned == []
        assert any("refused" in r.getMessage() for r in caplog.records)

    def test_a_spot_1_m_from_a_lane_is_put_on_its_centre(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handle, world, road = _tm_spawn(monkeypatch, _spot(1.0))
        assert handle is not None
        (at,) = world.spawned
        assert (at.location.x, at.location.y) == pytest.approx((0.0, 5.0))
        # Lifted off the lane's surface, not the height it was asked at.
        assert at.location.z == pytest.approx(1.25 + 0.5)
        # Projected onto the nearest driving lane.
        (query,) = road.queries
        assert query["project_to_road"] is True
        assert query["lane_type"] == int(carla.LaneType.Driving)

    def test_the_offset_limit_is_the_shared_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inside, _world, _road = _tm_spawn(
            monkeypatch, _spot(BACKGROUND_MAX_OFFSET_M - 0.01)
        )
        outside, _world, _road = _tm_spawn(
            monkeypatch, _spot(BACKGROUND_MAX_OFFSET_M + 0.01)
        )
        assert inside is not None and outside is None

    @pytest.mark.parametrize("asked_yaw", [0.0, 45.0, 90.0, 150.0, 180.0])
    def test_it_faces_along_its_lane_at_any_yaw_within_90_degrees_of_it(
        self, monkeypatch: pytest.MonkeyPatch, asked_yaw: float
    ) -> None:
        handle, world, _road = _tm_spawn(monkeypatch, _spot(1.0, asked_yaw))
        assert handle is not None
        (at,) = world.spawned
        assert at.rotation.yaw == pytest.approx(_LANE_YAW)
        assert (at.rotation.pitch, at.rotation.roll) == (0.0, 0.0)
        # And it starts moving along the lane (+y), at its speed.
        assert world.actor.velocity is not None
        vx, vy = world.actor.velocity
        assert vx == pytest.approx(0.0, abs=1e-5)
        assert vy == pytest.approx(10.0, rel=1e-5)


class TestTrafficManagerDirection:
    """A spot near a two-way road's centre line goes on the lane running its
    way, never on the nearest one when that runs against it."""

    def test_the_lane_beside_running_its_way_is_taken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handle, world, _road = _tm_spawn(
            monkeypatch, _spot(0.2, _LANE_YAW), _TwoWayRoad()
        )
        assert handle is not None
        (at,) = world.spawned
        assert at.location.x == pytest.approx(-1.6)
        assert at.rotation.yaw == pytest.approx(_LANE_YAW)
        assert world.actor.velocity is not None
        _vx, vy = world.actor.velocity
        assert vy == pytest.approx(10.0, rel=1e-5)

    def test_the_nearest_lane_is_kept_when_it_runs_its_way(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handle, world, _road = _tm_spawn(
            monkeypatch, _spot(0.2, -_LANE_YAW), _TwoWayRoad()
        )
        assert handle is not None
        (at,) = world.spawned
        assert at.location.x == pytest.approx(1.6)
        assert at.rotation.yaw == pytest.approx(-_LANE_YAW)

    @pytest.mark.parametrize(
        "beside",
        [
            None,
            _waypoint(-1.6, 5.0, -_LANE_YAW),
            _waypoint(-1.6, 5.0, _LANE_YAW, carla.LaneType.Shoulder),
            _waypoint(-6.0, 5.0, _LANE_YAW),
        ],
        ids=["no-lane", "also-against", "not-driving", "too-far"],
    )
    def test_refused_when_no_lane_beside_runs_its_way(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        beside: Any,
    ) -> None:
        with caplog.at_level(logging.INFO, logger=tm_module.__name__):
            handle, world, _road = _tm_spawn(
                monkeypatch, _spot(0.2, _LANE_YAW), _TwoWayRoad(beside)
            )
        assert handle is None
        assert world.spawned == []
        assert any("refused" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# SUMO: the same limit
# ---------------------------------------------------------------------------


def _sumo_spawn(lane_offset_m: float) -> Optional[str]:
    """SUMO's background vehicle where its nearest lane is *lane_offset_m*
    away."""
    here = [0.0, 0.0]
    traci = SimpleNamespace(
        simulation=SimpleNamespace(
            convert2D=lambda edge, pos, lane: (here[0] + lane_offset_m, here[1])
        ),
        route=SimpleNamespace(add=lambda *args, **kwargs: None),
        vehicle=SimpleNamespace(
            add=lambda *args, **kwargs: None,
            setMaxSpeed=lambda *args: None,
        ),
    )
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend.__dict__.update(
        _running=True,
        _traci=traci,
        _background_count=0,
        _background=set(),
        _blueprints={},
        _blueprint_offsets={},
        _offset=(0.0, 0.0),
        _config=SimpleNamespace(blueprint="vehicle.x"),
    )

    def road_position(pose: Any, sumo_id: str) -> tuple[str, float, int]:
        here[:] = [pose.x, pose.y]
        return ("edge0", 5.0, 0)

    backend.__dict__["_road_position"] = road_position
    backend.__dict__["_wander"] = lambda edge, *args: [edge]
    return backend.spawn_background(None, _spot(0.0), speed_kmh=36.0)


class TestSumoSharesTheLimit:
    def test_sumo_reads_the_one_constant(self) -> None:
        assert sumo_module.BACKGROUND_MAX_OFFSET_M is base.BACKGROUND_MAX_OFFSET_M
        assert tm_module.BACKGROUND_MAX_OFFSET_M is base.BACKGROUND_MAX_OFFSET_M
        assert not hasattr(sumo_module, "_BACKGROUND_MAX_OFFSET_M")
        assert math.isfinite(BACKGROUND_MAX_OFFSET_M) and BACKGROUND_MAX_OFFSET_M > 0

    @pytest.mark.parametrize(
        ("offset_m", "placed"),
        [
            (1.0, True),
            (BACKGROUND_MAX_OFFSET_M - 0.01, True),
            (BACKGROUND_MAX_OFFSET_M + 0.01, False),
            (10.0, False),
        ],
    )
    def test_both_backends_draw_the_line_at_the_same_offset(
        self, monkeypatch: pytest.MonkeyPatch, offset_m: float, placed: bool
    ) -> None:
        tm_handle, _world, _road = _tm_spawn(monkeypatch, _spot(offset_m))
        assert (tm_handle is not None) is placed
        assert (_sumo_spawn(offset_m) is not None) is placed

    def test_sumo_reports_a_refusal_as_the_traffic_manager_does(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=sumo_module.__name__):
            assert _sumo_spawn(10.0) is None
        assert any(
            "refused" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        )
