"""Tests for the SUMO traffic backend (``traffic.backend=sumo``).

The first half needs neither SUMO nor CARLA.  The second runs the backend on a
real SUMO (libsumo, in-process) over a crossroads roadgen generates, with the
CARLA side faked: SUMO is the thing under test, not the simulator.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.traffic import (
    LaneChangeDirection,
    TrafficContext,
    TurnDirection,
    available_backends,
    build_backend,
)
from autoware_carla_scenario.traffic.sumo.config import SumoBackendConfig
from autoware_carla_scenario.traffic.sumo.geometry import (
    Pose2D,
    carla_to_sumo,
    sumo_to_carla,
)
from autoware_carla_scenario.traffic.sumo.network import sanitize_opendrive

_HAS_SUMO = all(
    importlib.util.find_spec(m) is not None for m in ("libsumo", "sumolib", "roadgen")
)
needs_sumo = pytest.mark.skipif(
    not _HAS_SUMO, reason="the `sumo` extra is not installed"
)


# ---------------------------------------------------------------------------
# Without SUMO
# ---------------------------------------------------------------------------


def test_sumo_is_a_built_in_backend_that_builds_without_sumo_loaded() -> None:
    assert "sumo" in available_backends()
    backend = build_backend(
        "sumo", {"ambient": {"enabled": False}, "traffic_light_authority": "none"}
    )
    assert backend.name == "sumo"
    assert backend.describe()["traffic_light_authority"] == "none"
    assert backend.describe()["scenario_vehicles"] == "traffic_manager"


def test_the_config_refuses_unknown_keys_and_light_modes() -> None:
    with pytest.raises(ValueError, match="Unknown SumoBackendConfig"):
        SumoBackendConfig.from_mapping({"warmup": 1})
    with pytest.raises(ValueError, match="Unknown AmbientTrafficConfig"):
        SumoBackendConfig.from_mapping({"ambient": {"vehicles_per_hr": 1}})
    with pytest.raises(ValueError, match="traffic_light_authority"):
        SumoBackendConfig.from_mapping({"traffic_light_authority": "both"})
    with pytest.raises(ValueError, match="scenario_vehicles"):
        SumoBackendConfig.from_mapping({"scenario_vehicles": "autoware"})
    config = SumoBackendConfig.from_mapping(
        {"sumo_args": ["--verbose", 1], "tm_port": None, "ambient": {"period_s": 2}}
    )
    assert config.sumo_args == ["--verbose", "1"]
    assert config.tm_port == SumoBackendConfig().tm_port
    assert config.ambient.period() == 2.0
    assert SumoBackendConfig().ambient.period() == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("yaw", "angle"), [(0.0, 90.0), (90.0, 180.0), (-90.0, 0.0), (180.0, 270.0)]
)
def test_carla_yaw_and_sumo_angle_name_the_same_heading(
    yaw: float, angle: float
) -> None:
    sumo = carla_to_sumo(Pose2D(10.0, 20.0, 1.0, yaw), length_m=0.0)
    assert sumo.heading_deg == pytest.approx(angle)
    # CARLA mirrors OpenDRIVE's (and so SUMO's) y axis.
    assert (sumo.x, sumo.y, sumo.z) == pytest.approx((10.0, -20.0, 1.0))


def test_sumo_holds_the_front_bumper_and_carla_the_centre() -> None:
    # Facing +x in both frames: the bumper is half a length ahead.
    sumo = carla_to_sumo(Pose2D(0.0, 0.0, 0.0, 0.0), length_m=4.0)
    assert (sumo.x, sumo.y) == pytest.approx((2.0, 0.0))
    # Facing CARLA +y (SUMO -y, south).
    sumo = carla_to_sumo(Pose2D(0.0, 0.0, 0.0, 90.0), length_m=4.0)
    assert (sumo.x, sumo.y) == pytest.approx((0.0, -2.0), abs=1e-9)


@pytest.mark.parametrize("yaw", [-170.0, -45.0, 0.0, 33.0, 120.0])
def test_the_two_conversions_undo_each_other(yaw: float) -> None:
    pose = Pose2D(-12.5, 40.25, 0.3, yaw)
    back = sumo_to_carla(carla_to_sumo(pose, 4.6, (3.0, -7.0)), 4.6, (3.0, -7.0))
    assert (back.x, back.y, back.z) == pytest.approx((pose.x, pose.y, pose.z))
    assert math.cos(math.radians(back.heading_deg - yaw)) == pytest.approx(1.0)


def test_sanitize_fixes_what_roadgen_refuses_in_carla_maps() -> None:
    text = (
        '<road><userData code="x"><vectorRoad/></userData><userData/>'
        '<roadMark sOffset="0" type="curb"/><roadMark color="white" type="solid"/>'
        '<object id="1" type="-1"/><cornerLocal u="0" v="0" z="0"/></road>'
    )
    out = sanitize_opendrive(text)
    assert "userData" not in out
    assert '<roadMark color="standard" sOffset="0"' in out
    assert '<roadMark color="white"' in out
    assert 'type="none"' in out and 'type="-1"' not in out
    assert '<cornerLocal height="0" u="0"' in out


# ---------------------------------------------------------------------------
# With SUMO, and a fake CARLA
# ---------------------------------------------------------------------------


class _Extent:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z = x, y, 0.8


class _Box:
    def __init__(self) -> None:
        self.extent = _Extent(2.3, 1.0)


class _FakeActor:
    _next_id = 100

    def __init__(self, transform: Any, role_name: str = "") -> None:
        _FakeActor._next_id += 1
        self.id = _FakeActor._next_id
        self.attributes = {"role_name": role_name}
        self.bounding_box = _Box()
        self.transform = transform
        self.constant_velocity: Any = None
        self.autopilot = False
        self.destroyed = False

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot = enabled

    def get_transform(self) -> Any:
        return self.transform

    def set_transform(self, transform: Any) -> None:
        self.transform = transform

    def get_velocity(self) -> Any:
        import carla

        return carla.Vector3D(0.0, 0.0, 0.0)

    def enable_constant_velocity(self, velocity: Any) -> None:
        self.constant_velocity = velocity

    def disable_constant_velocity(self) -> None:
        self.constant_velocity = None

    def destroy(self) -> bool:
        self.destroyed = True
        return True


class _Blueprint:
    def has_attribute(self, name: str) -> bool:
        return name == "role_name"

    def set_attribute(self, name: str, value: str) -> None:
        self.role_name = value


class _Library:
    def find(self, name: str) -> _Blueprint:
        return _Blueprint()


class _Actors(list):  # type: ignore[type-arg]
    def filter(self, pattern: str) -> list[Any]:
        return list(self) if pattern == "vehicle.*" else []


class _Stamp:
    def __init__(self, frame: int) -> None:
        self.frame = frame
        self.elapsed_seconds = 100.0 + frame * 0.05


class _Snapshot:
    def __init__(self, frame: int) -> None:
        self.timestamp = _Stamp(frame)


class _FakeWorld:
    def __init__(self, actors: list[_FakeActor]) -> None:
        self.actors = actors
        self.spawned: list[_FakeActor] = []
        self.frame = 0

    def get_snapshot(self) -> _Snapshot:
        self.frame += 1
        return _Snapshot(self.frame)

    def get_actors(self) -> _Actors:
        return _Actors(self.actors)

    def get_blueprint_library(self) -> _Library:
        return _Library()

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> _FakeActor:
        actor = _FakeActor(transform, getattr(blueprint, "role_name", ""))
        self.spawned.append(actor)
        return actor


class _Entity:
    def __init__(self, actor: _FakeActor, role_name: str) -> None:
        self.actor = actor
        self.role_name = role_name


@pytest.fixture(scope="module")
def crossroads(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A four-arm crossroads, two lanes each way, as OpenDRIVE.

    Arm 0 comes from the west, arm 1 from the east, 2 from OpenDRIVE's south
    and 3 from its north; each ends 15 m from the centre.
    """
    if not _HAS_SUMO:
        pytest.skip("the `sumo` extra is not installed")
    import roadgen

    road_map = roadgen.Map()

    def lanes() -> list[Any]:
        # Broken lines: SUMO lets a vehicle change across those only.
        return [
            roadgen.Lane(
                width=3.5, direction=d, left_marking="broken", right_marking="broken"
            )
            for d in ("backward", "backward", "forward", "forward")
        ]

    arms = [
        road_map.add_road(
            start=(x * 120.0, y * 120.0, 0.0),
            end=(x * 15.0, y * 15.0, 0.0),
            lanes=lanes(),
        )
        for x, y in ((-1, 0), (1, 0), (0, -1), (0, 1))
    ]
    junction = road_map.add_junction("x")
    for a, b in itertools.combinations(arms, 2):
        road_map.connect(a, b, junction=junction, ends=("end", "end"))
    path = tmp_path_factory.mktemp("sumo") / "crossroads.xodr"
    road_map.export_opendrive(str(path))
    return path


def _carla_transform(x: float, y: float, yaw: float) -> Any:
    import carla

    return carla.Transform(carla.Location(x, y, 0.0), carla.Rotation(yaw=yaw))


def _prepared(
    crossroads: Path, tmp_path: Path, **options: Any
) -> tuple[Any, _FakeWorld, _FakeActor, _FakeActor]:
    """A started backend with an NPC it drives and an ego driven elsewhere.

    Both are on arm 0, heading east (CARLA yaw 0) towards the junction: the NPC
    on the outer lane, the ego on the inner one 30 m behind.  CARLA's y is
    OpenDRIVE's mirrored, so the eastbound lanes (OpenDRIVE y < 0) are at
    CARLA y > 0.
    """
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    config = SumoBackendConfig(
        scenario_vehicles=options.pop("scenario_vehicles", "sumo"),
        ambient=AmbientTrafficConfig(
            enabled="period_s" in options, period_s=options.pop("period_s", None)
        ),
        traffic_light_authority="none",
        warmup_s=options.pop("warmup_s", 0.0),
        cache_dir=str(tmp_path / "cache"),
        **options,
    )
    backend = SumoTrafficBackend(config)
    backend.prepare(
        TrafficContext(
            xodr_path=crossroads,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    npc = _FakeActor(_carla_transform(-80.0, 5.25, 0.0), "npc")
    ego = _FakeActor(_carla_transform(-110.0, 1.75, 0.0), str(EGO_ROLE_NAME))
    world = _FakeWorld([npc, ego])
    backend.adopt(_Entity(npc, "npc"))
    backend.adopt(_Entity(ego, str(EGO_ROLE_NAME)))
    backend.start(world, skip_actor_ids={ego.id})
    return backend, world, npc, ego


@needs_sumo
def test_sumo_drives_the_npc_and_mirrors_the_ego(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, npc, ego = _prepared(crossroads, tmp_path)
    try:
        assert len(backend._driven) == 1 and len(backend._external) == 1
        for i in range(100):
            backend.tick(world, i * 0.05)
        # Driven east along its lane, with SUMO's speed as its velocity.
        x = npc.transform.location.x
        assert x > -80.0 + 5.0
        assert npc.transform.location.y == pytest.approx(5.25, abs=0.3)
        assert npc.transform.rotation.yaw == pytest.approx(0.0, abs=1.0)
        assert npc.constant_velocity is not None and npc.constant_velocity.x > 1.0
        # The ego, driven elsewhere, is where CARLA has it -- in SUMO too.
        ego_id = next(iter(backend._external))
        sx, sy = backend._traci.vehicle.getPosition(ego_id)
        assert (sx, sy) == pytest.approx((-110.0 + 2.3, -1.75), abs=0.3)
        assert ego.constant_velocity is None
    finally:
        backend.close()
    assert backend.describe()["backend"] == "sumo"


@needs_sumo
def test_manoeuvres_become_sumo_commands(crossroads: Path, tmp_path: Path) -> None:
    backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    entity = backend._driven[next(iter(backend._driven))]
    sumo_id = next(iter(backend._driven))
    try:
        backend.tick(world, 0.0)
        lane = backend._traci.vehicle.getLaneIndex(sumo_id)
        assert lane == 0  # the kerb lane
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        assert not backend.lane_change_finished(entity, world)
        for i in range(100):
            backend.tick(world, i * 0.05)
            if backend.lane_change_finished(entity, world):
                break
        assert backend.lane_change_finished(entity, world)
        assert backend._traci.vehicle.getLaneIndex(sumo_id) == 1
        # SUMO eases the heading back over a second, and the centre follows.
        for i in range(30):
            backend.tick(world, i * 0.05)
        assert npc.transform.location.y == pytest.approx(1.75, abs=0.3)

        backend.set_desired_speed(entity, world, 18.0)
        for i in range(60):
            backend.tick(world, i * 0.05)
        assert backend._traci.vehicle.getSpeed(sumo_id) == pytest.approx(5.0, abs=0.1)

        backend.turn_at_junction(entity, world, TurnDirection.LEFT)
        route = backend._traci.vehicle.getRoute(sumo_id)
        # From the west arm, left is OpenDRIVE's north arm (3), outbound.
        assert list(route[:2]) == ["0.fwd", "3.bwd"]
    finally:
        backend.close()


@needs_sumo
def test_ambient_traffic_is_mirrored_into_carla_and_cleaned_up(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, period_s=1.0, max_vehicles=3
    )
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
        alive = [a for a in world.spawned if not a.destroyed]
        assert 0 < len(alive) <= 3
        assert all(a.attributes["role_name"].startswith("sumo:") for a in alive)
        assert (tmp_path / "out" / "sumo" / "ambient.rou.xml").is_file()
    finally:
        backend.close()
    assert all(a.destroyed for a in world.spawned)
    assert backend._ambient == {} and backend._driven == {}


@needs_sumo
def test_by_default_the_scenarios_vehicles_stay_with_the_traffic_manager(
    crossroads: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend, world, npc, ego = _prepared(
        crossroads, tmp_path, scenario_vehicles="traffic_manager"
    )
    entity = next(e for e in backend._entities if e.actor is npc)
    try:
        # Published into SUMO rather than driven by it; the TrafficManager
        # drives the NPC, and nothing on this side the ego Autoware drives.
        assert backend._driven == {} and len(backend._external) == 2
        assert npc.autopilot and not ego.autopilot
        npc.transform = _carla_transform(-60.0, 5.25, 0.0)
        backend.tick(world, 0.0)
        npc_id = next(i for i, a in backend._external.items() if a is npc)
        sx, _sy = backend._traci.vehicle.getPosition(npc_id)
        assert sx == pytest.approx(-60.0 + 2.3, abs=0.3)
        assert npc.constant_velocity is None
        # Its manoeuvres go to the TrafficManager (which has no client here,
        # and says so instead of raising).
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        assert not backend.lane_change_finished(entity, world)
        assert backend.port == SumoBackendConfig().tm_port
    finally:
        backend.close()


class _Velocity:
    def __init__(self, x: float) -> None:
        self.x, self.y, self.z = x, 0.0, 0.0


@needs_sumo
def test_a_published_vehicle_keeps_its_speed_in_sumo_after_the_warm_up(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, npc, ego = _prepared(crossroads, tmp_path, warmup_s=2.0)
    ego_id = next(iter(backend._external))
    driven_id = next(iter(backend._driven))
    ego.get_velocity = lambda: _Velocity(5.0)  # type: ignore[method-assign]
    try:
        for i in range(30):
            loc = ego.transform.location
            ego.transform = _carla_transform(loc.x + 0.25, loc.y, 0.0)
            backend.tick(world, i * 0.05)
        # SUMO sees the published ego moving at its CARLA speed, and the
        # vehicle it drives was let go after the warm-up.
        assert backend._traci.vehicle.getSpeed(ego_id) == pytest.approx(5.0, abs=0.1)
        assert backend._traci.vehicle.getSpeed(driven_id) > 0.5
    finally:
        backend.close()


@needs_sumo
def test_fcd_output_records_how_sumo_time_maps_onto_carla_time(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, warmup_s=1.0, fcd_output=True
    )
    try:
        for i in range(10):
            backend.tick(world, i * 0.05)
    finally:
        backend.close()
    out = tmp_path / "out" / "sumo"
    assert (out / "fcd.xml").is_file()
    rows = (out / "clock.csv").read_text().splitlines()
    assert rows[0] == "sumo_time,carla_elapsed_seconds,carla_frame"
    sumo_t, carla_t, frame = (float(x) for x in rows[1].split(","))
    # The warm-up ran SUMO alone, so its clock leads CARLA's by that much.
    assert sumo_t == pytest.approx(1.05)
    assert (carla_t, frame) == (pytest.approx(100.05), 1)
    assert len(rows) == 11
