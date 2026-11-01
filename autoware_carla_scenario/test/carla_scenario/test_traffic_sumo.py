"""Tests for the SUMO traffic backend (``traffic.backend=sumo``).

The first half needs neither SUMO nor CARLA.  The second runs the backend on a
real SUMO (libsumo, in-process) over a crossroads roadgen generates, with the
CARLA side faked: SUMO is the thing under test, not the simulator.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import importlib.util
import logging
import itertools
import math
from pathlib import Path
from typing import Any

from types import SimpleNamespace

import pytest

from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.traffic import (
    DrivenEntity,
    LaneChangeDirection,
    SimulationFidelity,
    TrafficContext,
    TurnDirection,
    available_backends,
    build_backend,
)
from autoware_carla_scenario.traffic.fidelity import MesoscopicConfig
from autoware_carla_scenario.traffic.sumo.config import (
    DEFAULT_BICYCLE_BLUEPRINTS,
    AmbientTrafficConfig,
    BicycleConfig,
    PedestrianConfig,
    SumoBackendConfig,
)
from autoware_carla_scenario.traffic.sumo.geometry import (
    Pose2D,
    box_centre_to_origin,
    carla_to_sumo,
    carla_xy_to_sumo,
    origin_to_box_centre,
    sumo_to_carla,
    sumo_xy_to_carla,
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
    assert backend.describe()["scenario_pedestrians"] == "traffic_manager"
    assert backend.describe()["vehicle_control"] == "physics"


def test_the_config_refuses_unknown_keys_and_light_modes() -> None:
    with pytest.raises(ValueError, match="Unknown SumoBackendConfig"):
        SumoBackendConfig.from_mapping({"warmup": 1})
    with pytest.raises(ValueError, match="Unknown AmbientTrafficConfig"):
        SumoBackendConfig.from_mapping({"ambient": {"vehicles_per_hr": 1}})
    with pytest.raises(ValueError, match="traffic_light_authority"):
        SumoBackendConfig.from_mapping({"traffic_light_authority": "both"})
    with pytest.raises(ValueError, match="scenario_vehicles"):
        SumoBackendConfig.from_mapping({"scenario_vehicles": "autoware"})
    with pytest.raises(ValueError, match="scenario_pedestrians"):
        SumoBackendConfig.from_mapping({"scenario_pedestrians": "carla"})
    config = SumoBackendConfig.from_mapping(
        {"sumo_args": ["--verbose", 1], "tm_port": None, "ambient": {"period_s": 2}}
    )
    assert config.sumo_args == ["--verbose", "1"]
    assert config.tm_port == SumoBackendConfig().tm_port
    assert config.ambient.period() == 2.0
    assert SumoBackendConfig().ambient.period() == pytest.approx(4.0)


def test_sumo_adds_nobody_the_run_did_not_ask_for_by_default() -> None:
    config = SumoBackendConfig()
    assert not config.ambient.enabled
    assert not config.pedestrians.enabled
    assert config.scenario_vehicles == "traffic_manager"
    assert config.scenario_pedestrians == "traffic_manager"


def test_curve_speeds_are_on_by_default_and_must_be_positive() -> None:
    assert SumoBackendConfig().curve_lateral_acceleration == pytest.approx(3.0)
    assert (
        SumoBackendConfig.from_mapping(
            {"curve_lateral_acceleration": None}
        ).curve_lateral_acceleration
        is None
    )
    for bad in (0.0, -2.0):
        with pytest.raises(ValueError, match="curve_lateral_acceleration"):
            SumoBackendConfig.from_mapping({"curve_lateral_acceleration": bad})


def test_the_lane_change_options_are_checked() -> None:
    config = SumoBackendConfig()
    # The sublane model by default: no lane change in one step.
    assert config.lateral_resolution_m == pytest.approx(0.4)
    assert config.lane_change_duration_s is None
    assert config.step_length_s is None
    for name in ("lane_change_duration_s", "lateral_resolution_m", "step_length_s"):
        with pytest.raises(ValueError, match=name):
            SumoBackendConfig.from_mapping({name: 0.0})
    # SUMO takes one or the other.
    with pytest.raises(ValueError, match="lateral_resolution_m: null"):
        SumoBackendConfig.from_mapping({"lane_change_duration_s": 2.0})
    lane_based = SumoBackendConfig.from_mapping(
        {"lane_change_duration_s": 2.0, "lateral_resolution_m": None}
    )
    assert lane_based.lane_change_duration_s == pytest.approx(2.0)


def test_a_sumo_step_divides_the_carla_tick() -> None:
    from autoware_carla_scenario.traffic.sumo.signal_states import sumo_step

    assert sumo_step(0.05, None) == (0.05, 1)
    step, count = sumo_step(0.05, 0.01)
    assert count == 5
    assert step == pytest.approx(0.01)
    assert sumo_step(0.1, 0.1) == (0.1, 1)
    # 0.0125 divides the tick, but SUMO's clock would round it to 0.013.
    for bad in (0.03, 0.1, 0.0125):
        with pytest.raises(ValueError, match="step_length_s"):
            sumo_step(0.05, bad)


def test_a_lateral_acceleration_needs_the_sublane_model() -> None:
    from autoware_carla_scenario.traffic.parameters import (
        TrafficParameterError,
        VehicleParameters,
    )
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    with pytest.raises(
        TrafficParameterError,
        match="max_lateral_accel needs traffic.options.lateral_resolution_m",
    ):
        SumoTrafficBackend(
            SumoBackendConfig(lateral_resolution_m=None),
            vehicles=VehicleParameters(max_lateral_accel=0.5),
        )
    backend = SumoTrafficBackend(vehicles=VehicleParameters(max_lateral_accel=0.5))
    assert backend.vehicle_parameters.max_lateral_accel is not None


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


def test_a_carla_box_centre_is_its_origin_moved_by_the_turned_offset() -> None:
    # Facing CARLA +y: forward is +y, and the right (CARLA's +y of the actor's
    # own frame, left-handed) is -x.
    centre = origin_to_box_centre(Pose2D(10.0, 20.0, 0.5, 90.0), (1.3, 0.2))
    assert (centre.x, centre.y, centre.z) == pytest.approx((9.8, 21.3, 0.5))
    assert centre.heading_deg == 90.0
    # No offset, no move.
    pose = Pose2D(1.0, 2.0, 3.0, 45.0)
    assert origin_to_box_centre(pose, (0.0, 0.0)) == pose


@pytest.mark.parametrize("yaw", [-170.0, -45.0, 0.0, 33.0, 120.0])
def test_a_box_centre_and_its_origin_undo_each_other(yaw: float) -> None:
    pose = Pose2D(-12.5, 40.25, 0.3, yaw)
    back = origin_to_box_centre(box_centre_to_origin(pose, (1.3, -0.4)), (1.3, -0.4))
    assert (back.x, back.y, back.z) == pytest.approx((pose.x, pose.y, pose.z))
    assert math.hypot(
        box_centre_to_origin(pose, (1.3, -0.4)).x - pose.x,
        box_centre_to_origin(pose, (1.3, -0.4)).y - pose.y,
    ) == pytest.approx(math.hypot(1.3, 0.4))


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
    def __init__(self, location: Any = None) -> None:
        self.extent = _Extent(2.3, 1.0)
        if location is not None:
            # Where CARLA centres the box in the actor's own frame; a box
            # without one reads as centred on the origin.
            self.location = SimpleNamespace(x=location[0], y=location[1], z=location[2])


class _FakeActor:
    _next_id = 100

    def __init__(
        self,
        transform: Any,
        role_name: str = "",
        type_id: str = "vehicle.fake",
        box_location: Any = None,
    ) -> None:
        _FakeActor._next_id += 1
        self.id = _FakeActor._next_id
        self.type_id = type_id
        self.attributes = {"role_name": role_name}
        self.bounding_box = _Box(box_location)
        self.transform = transform
        self.constant_velocity: Any = None
        self.autopilot = False
        self.destroyed = False

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot = enabled

    def get_transform(self) -> Any:
        return self.transform

    @property
    def is_alive(self) -> bool:
        return not self.destroyed

    def get_location(self) -> Any:
        return self.transform.location

    def set_transform(self, transform: Any) -> None:
        self.transform = transform

    def get_velocity(self) -> Any:
        import typesafe_carla.carla as carla

        return carla.Vector3D(0.0, 0.0, 0.0)

    def get_angular_velocity(self) -> Any:
        import typesafe_carla.carla as carla

        return carla.Vector3D(0.0, 0.0, 0.0)

    def set_target_velocity(self, velocity: Any) -> None:
        self.target_velocity = velocity

    def get_physics_control(self) -> Any:
        raise RuntimeError("no physics in a fake")  # the backend falls back

    def apply_ackermann_controller_settings(self, settings: Any) -> None:
        self.controller_settings = settings

    def enable_constant_velocity(self, velocity: Any) -> None:
        self.constant_velocity = velocity

    def disable_constant_velocity(self) -> None:
        self.constant_velocity = None

    def apply_control(self, control: Any) -> None:
        self.control = control

    def destroy(self) -> bool:
        self.destroyed = True
        return True


class _Blueprint:
    def __init__(self, blueprint_id: str = "vehicle.fake") -> None:
        self.id = blueprint_id

    def has_attribute(self, name: str) -> bool:
        return name == "role_name"

    def set_attribute(self, name: str, value: str) -> None:
        self.role_name = value


class _Library:
    def __init__(self, missing: Any = ()) -> None:
        self.missing = set(missing)

    def find(self, name: str) -> _Blueprint:
        if name in self.missing:
            raise IndexError(f"blueprint '{name}' not found")
        return _Blueprint(name)


class _Actors(list):  # type: ignore[type-arg]
    def filter(self, pattern: str) -> list[Any]:
        return [a for a in self if fnmatch.fnmatch(a.type_id, pattern)]


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
        self.library_fetches = 0
        self.spawn_attempts = 0
        #: Blueprints the world does not have.
        self.missing: set[str] = set()
        #: The ``bounding_box.location`` every actor spawned here has.
        self.box_location: Any = None
        #: Where each actor was spawned, in spawn order.
        self.spawned_at: list[Any] = []

    def get_snapshot(self) -> _Snapshot:
        self.frame += 1
        return _Snapshot(self.frame)

    def get_actors(self) -> _Actors:
        return _Actors(self.actors)

    def get_map(self) -> Any:
        """A map no actor stands on: no road ahead to search for a dead end."""
        return SimpleNamespace(get_waypoint=lambda location: None)

    def get_blueprint_library(self) -> _Library:
        self.library_fetches += 1
        return _Library(self.missing)

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> _FakeActor:
        self.spawn_attempts += 1
        actor = _FakeActor(
            transform,
            getattr(blueprint, "role_name", ""),
            blueprint.id,
            box_location=self.box_location,
        )
        self.spawned.append(actor)
        self.spawned_at.append(transform)
        return actor


_CLIENT: dict[str, Any] = {}


class _BatchClient:
    """Keeps every command batch the backend sends."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def apply_batch(self, commands: list[Any]) -> None:
        self.batches.append(list(commands))


class _Entity:
    def __init__(self, actor: _FakeActor, role_name: str) -> None:
        self.actor = actor
        self.role_name = role_name


def _crossroads(directory: Path, *, sidewalks: bool, signals: bool = False) -> Path:
    """A four-arm crossroads, two lanes each way, as OpenDRIVE.

    Arm 0 comes from the west, arm 1 from the east, 2 from OpenDRIVE's south
    and 3 from its north; each ends 15 m from the centre.  With *sidewalks*,
    a 2 m footway runs outside each carriageway.  With *signals*, a traffic
    light controls the two lanes of each arm into the junction.
    """
    import roadgen

    road_map = roadgen.Map()

    def lanes() -> list[Any]:
        # Broken lines: SUMO lets a vehicle change across those only.
        driving = [
            roadgen.Lane(
                width=3.5, direction=d, left_marking="broken", right_marking="broken"
            )
            for d in ("backward", "backward", "forward", "forward")
        ]
        if not sidewalks:
            return driving
        return [
            roadgen.Lane(width=2.0, direction="backward", type_="sidewalk"),
            *driving,
            roadgen.Lane(width=2.0, direction="forward", type_="sidewalk"),
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
    if signals:
        first = 3 if sidewalks else 2  # the lanes into the junction
        for arm in arms:
            entering = [arm.lane(first), arm.lane(first + 1)]
            light = road_map.add_traffic_light(entering[0], end="end")
            stop = road_map.add_stop_line(entering[0], end="end")
            road_map.add_traffic_light_rule([light], entering, stop_line=stop)
    path = directory / "crossroads.xodr"
    road_map.export_opendrive(str(path))
    return path


@pytest.fixture(scope="module")
def crossroads(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The crossroads, without footways."""
    if not _HAS_SUMO:
        pytest.skip("the `sumo` extra is not installed")
    return _crossroads(tmp_path_factory.mktemp("sumo"), sidewalks=False)


@pytest.fixture(scope="module")
def signalled_crossroads(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The crossroads with footways and a traffic light on every arm."""
    if not _HAS_SUMO:
        pytest.skip("the `sumo` extra is not installed")
    return _crossroads(tmp_path_factory.mktemp("sumo"), sidewalks=True, signals=True)


@pytest.fixture(scope="module")
def crossroads_with_sidewalks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The crossroads, with a footway along every carriageway."""
    if not _HAS_SUMO:
        pytest.skip("the `sumo` extra is not installed")
    return _crossroads(tmp_path_factory.mktemp("sumo"), sidewalks=True)


def _carla_transform(x: float, y: float, yaw: float) -> Any:
    import typesafe_carla.carla as carla

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

    vehicles = options.pop("vehicles", None)
    box_location = options.pop("box_location", None)
    config = SumoBackendConfig(
        scenario_vehicles=options.pop("scenario_vehicles", "sumo"),
        ambient=AmbientTrafficConfig(
            enabled="period_s" in options, period_s=options.pop("period_s", None)
        ),
        traffic_light_authority="none",
        warmup_s=options.pop("warmup_s", 0.0),
        vehicle_control=options.pop("vehicle_control", "teleport"),
        cache_dir=str(tmp_path / "cache"),
        **options,
    )
    backend = SumoTrafficBackend(config, vehicles=vehicles)
    backend.prepare(
        TrafficContext(
            client=_CLIENT.get("client"),
            xodr_path=crossroads,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    npc = _FakeActor(
        _carla_transform(-80.0, 5.25, 0.0), "npc", box_location=box_location
    )
    ego = _FakeActor(
        _carla_transform(-110.0, 1.75, 0.0),
        str(EGO_ROLE_NAME),
        box_location=box_location,
    )
    world = _FakeWorld([npc, ego])
    world.box_location = box_location
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
def test_a_released_vehicle_is_published_rather_than_driven(
    crossroads: Path, tmp_path: Path
) -> None:
    """Taken back by an action, the NPC stays in SUMO -- as CARLA's, like the ego."""
    backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    sumo_id, entity = next(iter(backend._driven.items()))
    try:
        backend.tick(world, 0.0)
        backend.release(entity, world)
        assert sumo_id not in backend._driven
        assert backend._external[sumo_id] is npc
        assert npc.constant_velocity is None
        held = npc.transform.location.x
        for i in range(20):
            backend.tick(world, (i + 1) * 0.05)
        # SUMO no longer moves it: the pose it has is CARLA's own.
        assert npc.transform.location.x == pytest.approx(held)
        backend.release(entity, world)  # and saying it twice is harmless
    finally:
        backend.close()


@needs_sumo
def test_a_resumed_vehicle_is_driven_by_sumo_again(
    crossroads: Path, tmp_path: Path
) -> None:
    """Handed back, the NPC is SUMO's again, from where CARLA left it, with
    the modes and the speed it had."""
    backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    sumo_id, entity = next(iter(backend._driven.items()))
    tc = backend._traci
    try:
        backend.tick(world, 0.0)
        backend.set_desired_speed(entity, world, 18.0)
        modes = (
            tc.vehicle.getSpeedMode(sumo_id),
            tc.vehicle.getLaneChangeMode(sumo_id),
        )
        backend.release(entity, world)
        for i in range(5):
            backend.tick(world, (i + 1) * 0.05)
        held = npc.transform.location.x
        backend.resume(entity, world)
        assert backend._driven[sumo_id] is entity
        assert sumo_id not in backend._external
        assert (
            tc.vehicle.getSpeedMode(sumo_id),
            tc.vehicle.getLaneChangeMode(sumo_id),
        ) == modes
        for i in range(60):
            backend.tick(world, (i + 6) * 0.05)
        # SUMO moves it on from where it was held, at the speed it was set.
        assert npc.transform.location.x > held + 1.0
        assert tc.vehicle.getSpeed(sumo_id) == pytest.approx(18.0 / 3.6, abs=0.3)
        backend.resume(entity, world)  # and saying it twice is harmless
        assert backend._driven[sumo_id] is entity
    finally:
        backend.close()


@needs_sumo
def test_a_vehicle_published_from_the_start_is_not_resumed(
    crossroads: Path, tmp_path: Path
) -> None:
    """The ego, published from the start, stays CARLA's whatever it is told."""
    backend, world, _npc, ego = _prepared(crossroads, tmp_path)
    ego_id = next(iter(backend._external))
    try:
        stand_in = SimpleNamespace(actor=ego, role_name="Ego")
        backend.release(stand_in, world)
        backend.resume(stand_in, world)
        assert backend._external[ego_id] is ego
        assert ego_id not in backend._driven
    finally:
        backend.close()


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
@pytest.mark.parametrize("sublane", [False, True])
def test_a_lane_change_takes_as_long_as_its_lateral_speed_allows(
    crossroads: Path, tmp_path: Path, sublane: bool
) -> None:
    from autoware_carla_scenario.traffic.parameters import VehicleParameters

    options: dict[str, Any] = {} if sublane else {"lateral_resolution_m": None}
    backend, world, _npc, _ego = _prepared(
        crossroads,
        tmp_path,
        vehicles=VehicleParameters(max_lateral_speed_mps=0.5),
        **options,
    )
    entity = backend._driven[next(iter(backend._driven))]
    try:
        backend.tick(world, 0.0)
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        ticks = 0
        while ticks < 400 and not backend.lane_change_finished(entity, world):
            ticks += 1
            backend.tick(world, ticks * 0.05)
        assert backend.lane_change_finished(entity, world)
        # 3.5 m across at no more than 0.5 m/s: some 7 s, not the 3 s default.
        assert ticks * 0.05 > 5.0
    finally:
        backend.close()


@needs_sumo
def test_a_construction_zone_closes_and_reopens_the_lane(
    crossroads: Path, tmp_path: Path
) -> None:
    """TeraSim's full-lane construction zone: setDisallowed, then the lane's
    own permissions back (#402)."""
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    tc = backend._traci
    try:
        lane = backend.close_lane(world, _carla_transform(-60.0, 5.25, 0.0))
        assert lane is not None
        assert "passenger" in tc.lane.getDisallowed(lane)
        # Asked again for the same lane: the one closure.
        assert backend.close_lane(world, _carla_transform(-58.0, 5.25, 0.0)) == lane
        # The lane beside stays open: a closed lane is still matched as itself.
        assert "passenger" not in tc.lane.getDisallowed("0.fwd_1")
        backend.reopen_lane(world, lane)
        assert "passenger" not in tc.lane.getDisallowed(lane)
        backend.reopen_lane(world, lane)  # and saying it twice is harmless
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
        # The blueprint library is an RPC for every actor definition: fetched
        # on the ticks that spawn, at most one fetch each, not on every tick.
        assert 0 < world.library_fetches <= world.spawn_attempts
        assert (tmp_path / "out" / "sumo" / "ambient.rou.xml").is_file()
    finally:
        backend.close()
    assert all(a.destroyed for a in world.spawned)
    assert backend._ambient == {} and backend._driven == {}


@needs_sumo
def test_an_ambient_vehicle_can_be_taken_over_and_handed_back(
    crossroads: Path, tmp_path: Path
) -> None:
    """An adversity takes an ambient vehicle's mirror over (#417): SUMO stops
    following it, keeps it published from CARLA, and drives it again once
    handed back."""
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, period_s=1.0, max_vehicles=3
    )
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
        offered = backend.background_actors(world)
        assert offered and all(name.startswith("sumo:") for name in offered)
        name, actor = next(iter(offered.items()))
        sumo_id = name[len("sumo:") :]
        backend.release_actor(world, actor)
        assert sumo_id not in backend._ambient and backend._external[sumo_id] is actor
        assert name not in backend.background_actors(world)
        held = actor.transform.location.x
        for i in range(20):
            backend.tick(world, (200 + i) * 0.05)
        # Neither followed nor removed: the pose it has is CARLA's own.
        assert not actor.destroyed
        assert actor.transform.location.x == pytest.approx(held)
        backend.resume_actor(world, actor)
        assert sumo_id not in backend._external
        assert backend._ambient.get(sumo_id) is actor
        backend.resume_actor(world, actor)  # and saying it twice is harmless
    finally:
        backend.close()
    assert actor.destroyed


@needs_sumo
def test_a_taken_over_ambient_vehicle_is_cleaned_up_at_close(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, period_s=1.0, max_vehicles=3
    )
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
        _name, actor = next(iter(backend.background_actors(world).items()))
        backend.release_actor(world, actor)
    finally:
        backend.close()
    assert actor.destroyed and backend._handed_ambient == {}


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
def test_sumo_traffic_stops_for_carlas_pedestrians_on_the_road(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, npc, _ego = _prepared(crossroads_with_sidewalks, tmp_path)
    sumo_id = next(iter(backend._driven))
    tc = backend._traci
    # Two pedestrians across both eastbound lanes, 30 m ahead of the NPC, on
    # the road and nowhere near a crossing.
    walkers = [
        _FakeActor(_carla_transform(-50.0, y, 90.0), type_id="walker.pedestrian.0001")
        for y in (1.75, 5.25)
    ]
    for walker in walkers:
        walker.bounding_box.extent = _Extent(0.2, 0.2)
    world.actors.extend(walkers)
    try:
        for i in range(300):
            backend.tick(world, i * 0.05)
        persons = sorted(tc.person.getIDList())
        assert persons == sorted(f"carla_walker{w.id}" for w in walkers)
        # On the road lanes, not moved onto the footway.
        lanes = {tc.person.getLaneID(p) for p in persons}
        assert len(lanes) == 2
        assert all(backend._net.getLane(lane).allows("passenger") for lane in lanes)
        assert tc.vehicle.getSpeed(sumo_id) == pytest.approx(0.0, abs=0.05)
        stopped_at = tc.vehicle.getPosition(sumo_id)[0]
        assert -60.0 < stopped_at < -50.0

        # Gone from CARLA, gone from SUMO -- and the road is free again.
        world.actors[:] = [a for a in world.actors if a not in walkers]
        for i in range(100):
            backend.tick(world, 15.0 + i * 0.05)
        assert tc.person.getIDList() == ()
        assert tc.vehicle.getPosition(sumo_id)[0] > stopped_at + 5.0
    finally:
        backend.close()


@needs_sumo
def test_a_vehicle_inside_a_junction_joins_sumo_on_a_road_lane(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    # Lane 0 of an edge with a footway is the footway: a vehicle placed from
    # inside the junction starts on the lane its junction lane leads to.
    backend, world, _npc, _ego = _prepared(crossroads_with_sidewalks, tmp_path)
    tc = backend._traci
    # Inside the junction, on the junction lane from arm 0's outer lane into
    # arm 1 (SUMO's front bumper at x = -5: CARLA's centre half a car behind).
    car = _FakeActor(_carla_transform(-7.3, 5.25, 0.0), "late")
    junction_edge, _, junction_lane = tc.simulation.convertRoad(
        -5.0, -5.25, False, "passenger"
    )
    assert junction_edge.startswith(":")
    target = tc.lane.getLinks(f"{junction_edge}_{junction_lane}")[0][0]
    try:
        assert backend._add_vehicle("late", car)
        backend.tick(world, 0.0)
        assert tc.vehicle.getLaneID("late") == target
        assert backend._net.getLane(target).allows("passenger")
    finally:
        backend.close()


@needs_sumo
def test_background_vehicles_are_sumo_traffic_mirrored_into_carla(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=20.0
        )
        assert handle is not None
        for i in range(5):
            backend.tick(world, i * 0.05)
        assert handle in tc.vehicle.getIDList()
        assert tc.vehicle.getSpeed(handle) > 4.0
        # Mirrored into CARLA like the rest of SUMO's traffic.
        mirror = backend._ambient[handle]
        assert mirror in world.spawned
        (x, y) = backend.background_vehicles(world)[handle]
        assert -60.0 < x < -50.0 and y == pytest.approx(5.25, abs=1.0)

        backend.remove_background(world, handle)
        backend.tick(world, 0.3)
        assert handle not in tc.vehicle.getIDList()
        assert handle not in backend.background_vehicles(world)
        assert mirror.destroyed
    finally:
        backend.close()


@needs_sumo
def test_sumo_traffic_drives_no_faster_than_its_drivers_see(
    crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.visibility import (
        forget_visibility,
        set_world_visibility,
        sight_speed,
    )

    forget_visibility()
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    world.id = 11  # type: ignore[attr-defined]
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=60.0
        )
        assert handle is not None
        backend.tick(world, 0.0)
        assert tc.vehicle.getMaxSpeed(handle) == pytest.approx(60.0 / 3.6)

        # Fog: 20 m, so held to what a car stops from within 20 m.
        set_world_visibility(11, 20.0)
        backend.tick(world, 0.05)
        held = sight_speed(20.0, tc.vehicle.getDecel(handle), tc.vehicle.getTau(handle))
        assert held < 60.0 / 3.6
        assert tc.vehicle.getMaxSpeed(handle) == pytest.approx(held)
        # The scenario's own vehicle is the scenario's.
        (npc_id,) = backend._driven
        assert tc.vehicle.getMaxSpeed(npc_id) > held
        # A vehicle that departs in the fog is held to it too.
        later = backend.spawn_background(
            world, _carla_transform(-75.0, 1.75, 0.0), speed_kmh=60.0
        )
        assert later is not None
        backend.tick(world, 0.1)
        assert tc.vehicle.getMaxSpeed(later) == pytest.approx(held)
        assert backend.describe()["visibility_m"] == 20.0

        # The fog lifts: each gets its own speed back.
        forget_visibility()
        backend.tick(world, 0.15)
        assert tc.vehicle.getMaxSpeed(handle) == pytest.approx(60.0 / 3.6)
        assert tc.vehicle.getMaxSpeed(later) == pytest.approx(60.0 / 3.6)
    finally:
        backend.close()
        forget_visibility()


@needs_sumo
def test_sumo_traffic_can_ignore_the_visibility(
    crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.visibility import (
        forget_visibility,
        set_world_visibility,
    )
    from autoware_carla_scenario.traffic.parameters import VehicleParameters

    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, vehicles=VehicleParameters(follow_visibility=False)
    )
    world.id = 12  # type: ignore[attr-defined]
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=60.0
        )
        assert handle is not None
        set_world_visibility(12, 20.0)
        backend.tick(world, 0.0)
        backend.tick(world, 0.05)
        assert tc.vehicle.getMaxSpeed(handle) == pytest.approx(60.0 / 3.6)
    finally:
        backend.close()
        forget_visibility()


@needs_sumo
def test_sumo_drives_its_vehicles_with_the_traffic_parameters(
    crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import Spread, VehicleParameters
    from autoware_carla_scenario.visibility import (
        forget_visibility,
        set_world_visibility,
        sight_speed,
    )

    forget_visibility()
    parameters = VehicleParameters(
        speed_factor=Spread(0.5),
        min_gap_m=Spread(4.0),
        lane_changes=False,
        accel=Spread(1.5),
        decel=Spread(3.0),
        reaction_time_s=Spread(2.0, 0.5),
        imperfection=Spread(0.0),
    )
    backend, world, _npc, ego = _prepared(crossroads, tmp_path, vehicles=parameters)
    world.id = 13  # type: ignore[attr-defined]
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=60.0
        )
        assert handle is not None
        (npc_id,) = backend._driven
        (ego_id,) = backend._external
        backend.tick(world, 0.0)
        for sumo_id in (handle, npc_id):
            assert tc.vehicle.getMinGap(sumo_id) == pytest.approx(4.0)
            assert tc.vehicle.getAccel(sumo_id) == pytest.approx(1.5)
            assert tc.vehicle.getDecel(sumo_id) == pytest.approx(3.0)
            assert tc.vehicle.getImperfection(sumo_id) == pytest.approx(0.0)
            assert tc.vehicle.getLaneChangeMode(sumo_id) == 1537
            assert tc.vehicle.getSpeedFactor(sumo_id) == pytest.approx(0.5)
        # Each vehicle draws its own reaction time from the spread.
        taus = [tc.vehicle.getTau(i) for i in (handle, npc_id)]
        assert taus[0] != pytest.approx(taus[1])
        assert all(0.5 < t < 3.5 for t in taus)
        # Drawn once, as it departs: not again on the next step.
        backend.tick(world, 0.05)
        assert tc.vehicle.getTau(handle) == pytest.approx(taus[0])
        # What CARLA's side drives is only placed in SUMO.
        assert tc.vehicle.getTau(ego_id) == pytest.approx(1.0)
        assert backend.describe()["vehicles"]["reaction_time_s"] == {
            "mean": 2.0,
            "std": 0.5,
        }

        # The sight speed is worked out from the vehicle's own braking and
        # reaction.
        set_world_visibility(13, 20.0)
        backend.tick(world, 0.1)
        assert tc.vehicle.getMaxSpeed(handle) == pytest.approx(
            max(sight_speed(20.0, 3.0, taus[0]), 1.0)
        )
    finally:
        backend.close()
        forget_visibility()


@needs_sumo
@pytest.mark.parametrize("sublane", [False, True])
def test_sumo_gives_its_vehicles_their_lateral_speed_and_acceleration(
    crossroads: Path, tmp_path: Path, sublane: bool
) -> None:
    from autoware_carla_scenario.traffic.parameters import Spread, VehicleParameters

    parameters = VehicleParameters(
        max_lateral_speed_mps=Spread(0.6),
        max_lateral_accel=Spread(0.4) if sublane else None,
    )
    options: dict[str, Any] = (
        {} if sublane else {"lateral_resolution_m": None, "lane_change_duration_s": 2.0}
    )
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, vehicles=parameters, step_length_s=0.01, **options
    )
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=60.0
        )
        assert handle is not None
        (npc_id,) = backend._driven
        before = tc.simulation.getTime()
        backend.tick(world, 0.0)
        # Five SUMO steps of 0.01 s per 0.05 s tick.
        assert tc.simulation.getTime() - before == pytest.approx(0.05)
        assert tc.simulation.getDeltaT() == pytest.approx(0.01)
        for sumo_id in (handle, npc_id):
            assert tc.vehicle.getMaxSpeedLat(sumo_id) == pytest.approx(0.6)
            if sublane:
                assert float(
                    tc.vehicle.getParameter(sumo_id, "laneChangeModel.lcAccelLat")
                ) == pytest.approx(0.4)
    finally:
        backend.close()


def test_follow_visibility_is_a_traffic_parameter_now() -> None:
    with pytest.raises(ValueError, match="traffic.vehicles.follow_visibility"):
        SumoBackendConfig.from_mapping({"follow_visibility": False})


def test_sumo_refuses_the_traffic_managers_parameters() -> None:
    from autoware_carla_scenario.traffic.parameters import (
        Spread,
        TrafficParameterError,
        VehicleParameters,
    )
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    with pytest.raises(
        TrafficParameterError,
        match="ignore_lights can only be used with the 'traffic_manager' backend",
    ):
        SumoTrafficBackend(vehicles=VehicleParameters(ignore_lights=Spread(0.5)))


def test_sumo_hands_the_shared_parameters_to_its_traffic_manager() -> None:
    from autoware_carla_scenario.traffic.parameters import Spread, VehicleParameters
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(
        vehicles=VehicleParameters(
            speed_factor=Spread(0.9), reaction_time_s=Spread(1.5)
        )
    )
    assert backend.vehicle_parameters.reaction_time_s == Spread(1.5)
    # The TrafficManager driving the scenario's vehicles gets what it can do.
    assert backend._tm.vehicle_parameters == VehicleParameters(speed_factor=Spread(0.9))


@needs_sumo
def test_an_override_redraws_every_vehicle_sumo_drives(
    crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import Spread, VehicleParameters

    configured = VehicleParameters(reaction_time_s=Spread(1.0), decel=Spread(4.0))
    backend, world, npc, ego = _prepared(crossroads, tmp_path, vehicles=configured)
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=60.0
        )
        assert handle is not None
        backend.tick(world, 0.0)
        (npc_id,) = backend._driven
        (ego_id,) = backend._external
        assert tc.vehicle.getTau(handle) == pytest.approx(1.0)

        backend.override_vehicle_parameters(
            world, VehicleParameters(reaction_time_s=Spread(2.5))
        )
        for sumo_id in (handle, npc_id):
            assert tc.vehicle.getTau(sumo_id) == pytest.approx(2.5)
            assert tc.vehicle.getDecel(sumo_id) == pytest.approx(4.0)  # kept
        assert tc.vehicle.getTau(ego_id) == pytest.approx(1.0)  # CARLA's
        later = backend.spawn_background(
            world, _carla_transform(-75.0, 1.75, 0.0), speed_kmh=60.0
        )
        assert later is not None
        backend.tick(world, 0.05)
        assert tc.vehicle.getTau(later) == pytest.approx(2.5)

        # One vehicle alone: the others, and the run, keep theirs.
        backend.override_entity_parameters(
            world, VehicleParameters(reaction_time_s=Spread(0.5)), _Entity(npc, "npc")
        )
        assert tc.vehicle.getTau(npc_id) == pytest.approx(0.5)
        assert tc.vehicle.getTau(handle) == pytest.approx(2.5)
    finally:
        backend.close()
    assert backend.vehicle_parameters == configured


@needs_sumo
def test_an_override_redraws_every_pedestrian_sumo_walks(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import (
        PedestrianParameters,
        Spread,
    )

    backend, world, _ego = _walking(crossroads_with_sidewalks, tmp_path)
    tc = backend._traci
    try:
        for i in range(100):
            backend.tick(world, i * 0.05)
        assert tc.person.getIDList()
        backend.override_pedestrian_parameters(
            world,
            PedestrianParameters(walking_speed_mps=Spread(0.7), min_gap_m=Spread(0.9)),
        )
        for person in tc.person.getIDList():
            assert tc.person.getMaxSpeed(person) == pytest.approx(0.7)
            assert tc.person.getMinGap(person) == pytest.approx(0.9)
        before = set(tc.person.getIDList())
        for i in range(100, 200):
            backend.tick(world, i * 0.05)
        joined = set(tc.person.getIDList()) - before
        assert joined
        for person in joined:
            assert tc.person.getMaxSpeed(person) == pytest.approx(0.7)
    finally:
        backend.close()
    assert backend.pedestrian_parameters == PedestrianParameters()


@needs_sumo
def test_a_background_vehicle_off_the_road_is_refused(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    try:
        assert (
            backend.spawn_background(
                world, _carla_transform(-60.0, 500.0, 0.0), speed_kmh=20.0
            )
            is None
        )
    finally:
        backend.close()


@needs_sumo
def test_pedestrians_can_be_left_out_of_sumo(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _npc, _ego = _prepared(
        crossroads_with_sidewalks, tmp_path, publish_walkers=False
    )
    world.actors.append(
        _FakeActor(
            _carla_transform(-50.0, 1.75, 90.0), type_id="walker.pedestrian.0001"
        )
    )
    try:
        backend.tick(world, 0.0)
        assert backend._traci.person.getIDList() == ()
    finally:
        backend.close()


@needs_sumo
def test_a_pedestrian_with_no_footway_near_is_left_out(
    crossroads: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # SUMO stops altogether when a person is moved off an edge with no footway,
    # so on a network without any the pedestrian is not added at all.
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    world.actors.append(
        _FakeActor(
            _carla_transform(-50.0, 1.75, 90.0), type_id="walker.pedestrian.0001"
        )
    )
    try:
        with caplog.at_level("WARNING"):
            for i in range(3):
                backend.tick(world, i * 0.05)
        assert backend._traci.person.getIDList() == ()
        assert caplog.text.count("no footway") == 1
        assert backend._running
    finally:
        backend.close()


@needs_sumo
def test_turns_across_the_junction_are_held_to_their_curve_speed(
    crossroads: Path, tmp_path: Path
) -> None:
    import roadgen
    import sumolib

    from autoware_carla_scenario.traffic.sumo.network import build_network

    if "curve_lateral_acceleration" not in (roadgen.Map.export_sumo.__doc__ or ""):
        pytest.skip("roadgen without curve speeds")
    opendrive = crossroads.read_text()

    def slowest_turn(acceleration: float | None) -> float:
        net = sumolib.net.readNet(
            str(build_network(opendrive, tmp_path, acceleration).net_file),
            withInternal=True,
        )
        return min(
            lane.getSpeed()
            for edge in net.getEdges()
            if edge.getFunction() == "internal"
            for lane in edge.getLanes()
        )

    assert slowest_turn(2.0) < slowest_turn(None)


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


# ---------------------------------------------------------------------------
# The physics follower
# ---------------------------------------------------------------------------


def test_pure_pursuit_asks_to_turn_right_towards_a_point_on_the_right() -> None:
    from autoware_carla_scenario.traffic.sumo.physics_control import pursuit_curvature

    common = dict(x=0.0, y=0.0, yaw_deg=0.0)
    # CARLA is left-handed: +y is to the right of a car facing +x.
    right = pursuit_curvature(lookahead_x=10.0, lookahead_y=2.0, **common)  # type: ignore[arg-type]
    left = pursuit_curvature(lookahead_x=10.0, lookahead_y=-2.0, **common)  # type: ignore[arg-type]
    straight = pursuit_curvature(lookahead_x=10.0, lookahead_y=0.0, **common)  # type: ignore[arg-type]
    assert right > 0 > left
    assert straight == pytest.approx(0.0)
    # The arc through a point 10 m ahead and 2 m right: 2·sin(α)/d.
    assert right == pytest.approx(2 * (2 / 104**0.5) / 104**0.5)


def test_the_steer_command_inverts_carlas_square_law_and_corrects_the_rest() -> None:
    import math

    from autoware_carla_scenario.traffic.sumo.physics_control import steer_command

    # A curvature that needs 5.1° at the road wheel: 51° × steer² gives 0.316.
    curvature = math.tan(math.radians(5.1)) / 2.8
    exact = steer_command(
        curvature=curvature, actual_curvature=curvature, wheel_base=2.8, integral=0.0
    )
    assert exact.steer == pytest.approx(0.316, abs=1e-3) and exact.integral == 0.0
    # Turning less than asked: more steer, and the integral starts to build.
    short = steer_command(
        curvature=curvature, actual_curvature=0.0, wheel_base=2.8, integral=0.0
    )
    assert short.steer > exact.steer and short.integral > 0.0
    # Left is negative, and the command never leaves CARLA's [-1, 1].
    assert (
        steer_command(
            curvature=-curvature,
            actual_curvature=-curvature,
            wheel_base=2.8,
            integral=0.0,
        ).steer
        < 0
    )
    assert (
        steer_command(
            curvature=1.0, actual_curvature=-1.0, wheel_base=2.8, integral=5.0
        ).steer
        == 1.0
    )


def test_the_speed_controller_catches_up_holds_and_stops() -> None:
    from autoware_carla_scenario.traffic.sumo.physics_control import longitudinal

    base = dict(speed=5.0, sumo_speed=5.0, sumo_acceleration=0.0, integral=0.0)
    on_target = longitudinal(longitudinal_error=0.0, **base)  # type: ignore[arg-type]
    behind = longitudinal(longitudinal_error=2.0, **base)  # type: ignore[arg-type]
    ahead = longitudinal(longitudinal_error=-2.0, **base)  # type: ignore[arg-type]
    # 2 m behind its SUMO vehicle it aims 3 m/s faster; 2 m ahead, 3 m/s slower.
    assert behind.target_speed == pytest.approx(8.0) and behind.throttle > 0.0
    assert ahead.target_speed == pytest.approx(2.0) and ahead.brake > 0.0
    assert on_target.throttle == pytest.approx(
        0.0
    ) and on_target.brake == pytest.approx(0.0)
    # SUMO's own acceleration is fed forward.
    feedforward = longitudinal(
        speed=5.0,
        sumo_speed=5.0,
        sumo_acceleration=2.0,
        longitudinal_error=0.0,
        integral=0.0,
    )
    assert feedforward.throttle > 0.0
    # Stopped in SUMO and nearly stopped in CARLA: held on the brake.
    held = longitudinal(
        speed=0.1,
        sumo_speed=0.0,
        sumo_acceleration=0.0,
        longitudinal_error=0.0,
        integral=1.0,
    )
    assert (held.throttle, held.brake, held.integral) == (0.0, 1.0, 0.0)


@needs_sumo
def test_physics_mode_drives_sumos_vehicles_with_throttle_brake_and_steer(
    crossroads: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typesafe_carla.carla as carla

    # typesafe_carla's commands are opaque (no actor_id / control to read
    # back), so record what the backend builds them from.
    controls: list[SimpleNamespace] = []

    def apply_vehicle_control(actor_id: int, control: Any) -> SimpleNamespace:
        controls.append(SimpleNamespace(actor_id=actor_id, control=control))
        return controls[-1]

    monkeypatch.setattr(carla.command, "ApplyVehicleControl", apply_vehicle_control)
    client = _BatchClient()
    _CLIENT["client"] = client
    try:
        backend, world, npc, _ego = _prepared(
            crossroads, tmp_path, vehicle_control="physics"
        )
    finally:
        _CLIENT.clear()
    try:
        for i in range(20):
            backend.tick(world, i * 0.05)
        sent = [c for batch in client.batches for c in batch]
        assert controls and all(c in sent for c in controls)
        assert all(c.actor_id == npc.id for c in controls)
        # SUMO pulls away, so the car is given throttle -- and is not teleported.
        assert max(c.control.throttle for c in controls) > 0.0
        assert npc.constant_velocity is None
    finally:
        backend.close()


@needs_sumo
def test_teleport_mode_batches_the_pose_and_sends_a_speed_once(
    crossroads: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a client, each tick's pose goes into the batch, and the constant
    velocity -- which has no batch command -- only when SUMO's speed changed."""
    import typesafe_carla.carla as carla

    moves: list[SimpleNamespace] = []

    def apply_transform(actor_id: int, transform: Any) -> SimpleNamespace:
        moves.append(SimpleNamespace(actor_id=actor_id, transform=transform))
        return moves[-1]

    monkeypatch.setattr(carla.command, "ApplyTransform", apply_transform)
    client = _BatchClient()
    _CLIENT["client"] = client
    try:
        backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    finally:
        _CLIENT.clear()
    speeds: list[float] = []
    placed: list[Any] = []
    monkeypatch.setattr(npc, "enable_constant_velocity", lambda v: speeds.append(v.x))
    monkeypatch.setattr(npc, "set_transform", placed.append)
    sumo_id = next(iter(backend._driven))
    try:
        ticks = 100
        for i in range(ticks):
            if i == 20:  # SUMO brings it to a stop, where it then stands
                backend._traci.vehicle.setSpeed(sumo_id, 0.0)
            backend.tick(world, i * 0.05)
        sent = [c for batch in client.batches for c in batch]
        mine = [m for m in moves if m.actor_id == npc.id]
        # Not the first tick: on SUMO's road from that step, it is put where
        # its car is and followed from the next.
        assert len(mine) == ticks - 1 and all(m in sent for m in mine)
        assert placed == []  # never one RPC of its own
        # Sent again only when it differs from the one the actor has: once
        # for the standstill it ends in, however many ticks that lasts.
        assert all(a != b for a, b in zip(speeds, speeds[1:]))
        assert speeds[-1] == 0.0
        assert len(speeds) < ticks - 30
    finally:
        backend.close()


@needs_sumo
def test_feedback_moves_the_sumo_vehicle_to_where_its_car_really_is(
    crossroads: Path, tmp_path: Path
) -> None:
    client = _BatchClient()
    _CLIENT["client"] = client
    try:
        backend, world, npc, _ego = _prepared(
            crossroads, tmp_path, vehicle_control="physics", feedback_distance_m=2.0
        )
    finally:
        _CLIENT.clear()
    sumo_id = next(iter(backend._driven))
    try:
        for i in range(5):
            backend.tick(world, i * 0.05)
        # The car has fallen 4 m behind its SUMO vehicle, on the same lane.
        sx, _sy = backend._traci.vehicle.getPosition(sumo_id)
        npc.transform = _carla_transform(sx - 2.3 - 4.0, 5.25, 0.0)
        backend.tick(world, 0.3)
        # moveTo took effect at once: SUMO now has it where CARLA does.  (Its
        # lane position, which moveTo sets; getPosition is cached until the
        # next step.)  The arm starts at x = -120, so the bumper's x is
        # -120 + lane position.
        lane_pos = backend._traci.vehicle.getLanePosition(sumo_id)
        assert -120.0 + lane_pos == pytest.approx(sx - 4.0, abs=0.5)
        assert backend._feedback_count == 1
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# A CARLA actor's origin is not its box centre (#887)
# ---------------------------------------------------------------------------

#: A car's ``bounding_box.location``: its box sits ahead of its origin,
#: between the axles, and up off the road.
_BOX_LOCATION = (1.3, 0.0, 0.8)


def _box_front(actor: _FakeActor, transform: Any = None) -> tuple[float, float]:
    """The CARLA middle of *actor*'s box front, its origin at *transform*
    (where it stands, by default)."""
    tf = transform if transform is not None else actor.get_transform()
    location = getattr(actor.bounding_box, "location", None)
    forward = float(actor.bounding_box.extent.x) + (location.x if location else 0.0)
    right = location.y if location else 0.0
    yaw = math.radians(tf.rotation.yaw)
    return (
        tf.location.x + forward * math.cos(yaw) - right * math.sin(yaw),
        tf.location.y + forward * math.sin(yaw) + right * math.cos(yaw),
    )


def _sumo_bumper(backend: Any, sumo_id: str) -> tuple[float, float]:
    """Where SUMO has *sumo_id*'s front bumper, in CARLA's frame."""
    x, y = backend._traci.vehicle.getPosition(sumo_id)
    return sumo_xy_to_carla(x, y, backend._offset)


def _carla_yaw(backend: Any, sumo_id: str) -> float:
    return float(backend._traci.vehicle.getAngle(sumo_id)) - 90.0


@needs_sumo
@pytest.mark.parametrize("box_location", [None, _BOX_LOCATION])
def test_teleport_puts_the_box_front_at_sumos_front_bumper(
    crossroads: Path, tmp_path: Path, box_location: Any
) -> None:
    """The car's box -- not its origin -- is where SUMO's vehicle is, and the
    ego is in SUMO where its box front is; an actor whose box is centred on
    its origin is placed as before."""
    backend, world, npc, ego = _prepared(
        crossroads, tmp_path, box_location=box_location
    )
    sumo_id = next(iter(backend._driven))
    ego_id = next(iter(backend._external))
    tc = backend._traci
    try:
        for i in range(40):
            backend.tick(world, i * 0.05)
        speed = float(tc.vehicle.getSpeed(sumo_id))
        assert speed > 1.0
        # Put a tick's travel behind, which its constant velocity covers.
        fx, fy = _box_front(npc)
        yaw = math.radians(npc.transform.rotation.yaw)
        fx, fy = fx + math.cos(yaw) * speed * 0.05, fy + math.sin(yaw) * speed * 0.05
        assert (fx, fy) == pytest.approx(_sumo_bumper(backend, sumo_id), abs=0.02)
        # The ego, published, is in SUMO with its bumper at its box front.
        assert tc.vehicle.getPosition(ego_id) == pytest.approx(
            carla_xy_to_sumo(*_box_front(ego), backend._offset), abs=0.1
        )
    finally:
        backend.close()


@needs_sumo
@pytest.mark.parametrize("box_location", [None, _BOX_LOCATION])
def test_physics_mode_steers_origin_to_origin(
    crossroads: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    box_location: Any,
) -> None:
    """A car whose box front is at SUMO's front bumper is where it should be
    (no error to correct), and one put back on its SUMO vehicle is put with
    its box front there."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo import backend as backend_module

    errors: list[float] = []
    longitudinal = backend_module.longitudinal

    def record(**kwargs: Any) -> Any:
        errors.append(kwargs["longitudinal_error"])
        return longitudinal(**kwargs)

    moves: list[SimpleNamespace] = []

    def apply_transform(actor_id: int, transform: Any) -> SimpleNamespace:
        moves.append(SimpleNamespace(actor_id=actor_id, transform=transform))
        return moves[-1]

    monkeypatch.setattr(backend_module, "longitudinal", record)
    monkeypatch.setattr(carla.command, "ApplyTransform", apply_transform)
    client = _BatchClient()
    _CLIENT["client"] = client
    try:
        backend, world, npc, _ego = _prepared(
            crossroads, tmp_path, vehicle_control="physics", box_location=box_location
        )
    finally:
        _CLIENT.clear()
    sumo_id = next(iter(backend._driven))
    try:
        for i in range(10):
            backend.tick(world, i * 0.05)
        bx, by = _sumo_bumper(backend, sumo_id)
        yaw = _carla_yaw(backend, sumo_id)
        # Its origin where its box front lands on the bumper.
        reach = 2.3 + (box_location[0] if box_location else 0.0)
        heading = math.radians(yaw)
        ox, oy = bx - math.cos(heading) * reach, by - math.sin(heading) * reach
        npc.transform = _carla_transform(ox, oy, yaw)
        errors.clear()
        backend._drive(npc, sumo_id, backend._length(npc))
        assert errors and errors[-1] == pytest.approx(0.0, abs=1e-4)

        # Fallen 20 m behind: put back with its box front on the bumper.
        npc.transform = _carla_transform(
            ox - math.cos(heading) * 20.0, oy - math.sin(heading) * 20.0, yaw
        )
        moves.clear()
        backend._drive(npc, sumo_id, backend._length(npc))
        placed = [m.transform for m in moves if m.actor_id == npc.id]
        assert placed
        assert _box_front(npc, placed[-1]) == pytest.approx((bx, by), abs=1e-4)
    finally:
        backend._commands = []
        backend.close()


@needs_sumo
def test_a_mirror_is_spawned_with_its_box_where_sumos_vehicle_is(
    crossroads: Path, tmp_path: Path
) -> None:
    """The first mirror of a blueprint is spawned at a guess and put right at
    once; the next ones are spawned where they belong."""
    backend, world, _npc, _ego = _prepared(
        crossroads,
        tmp_path,
        period_s=1.0,
        max_vehicles=3,
        vehicle_control="physics",
        box_location=_BOX_LOCATION,
    )
    tc = backend._traci
    checked: set[int] = set()
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
            for sumo_id, mirror in backend._ambient.items():
                if mirror.id in checked:
                    continue
                # The tick it was spawned on: not steered yet.
                checked.add(mirror.id)
                bx, by = _sumo_bumper(backend, sumo_id)
                yaw = math.radians(_carla_yaw(backend, sumo_id))
                half = float(tc.vehicle.getLength(sumo_id)) / 2.0
                cx, cy = (
                    mirror.transform.location.x + 1.3 * math.cos(yaw),
                    mirror.transform.location.y + 1.3 * math.sin(yaw),
                )
                assert (cx + half * math.cos(yaw), cy + half * math.sin(yaw)) == (
                    pytest.approx((bx, by), abs=1e-4)
                )
        assert len(checked) >= 2
        # Spawned where it stands but for the first: its box was a guess.
        spawned = [(a.transform, at) for a, at in zip(world.spawned, world.spawned_at)]
        assert spawned[0][0].location.x != pytest.approx(spawned[0][1].location.x)
        for now, at in spawned[1:]:
            assert (now.location.x, now.location.y) == pytest.approx(
                (at.location.x, at.location.y)
            )
    finally:
        backend.close()


def test_a_spawn_puts_the_box_at_the_centre_it_is_given() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(SumoBackendConfig())
    world = _FakeWorld([])
    world.box_location = _BOX_LOCATION
    centre = Pose2D(10.0, 20.0, 0.0, 90.0)
    for _ in range(2):
        actor = backend._spawn_at_centre(world, _Blueprint(), "vehicle.fake", centre)
        tf = actor.get_transform()
        assert (tf.location.x, tf.location.y) == pytest.approx((10.0, 18.7))
        assert tf.location.z == pytest.approx(0.05)  # just above the road (#1003)
    # The first at the centre, as its blueprint has no box to read; the
    # second where it belongs straight away.
    assert world.spawned_at[0].location.y == pytest.approx(20.0)
    assert world.spawned_at[1].location.y == pytest.approx(18.7)
    # No offset: spawned at the centre, as before.
    plain = backend._spawn_at_centre(
        _FakeWorld([]), _Blueprint("vehicle.b"), "vehicle.b", centre
    )
    assert (plain.transform.location.x, plain.transform.location.y) == pytest.approx(
        (10.0, 20.0)
    )


class _KerbWorld(_FakeWorld):
    """Refuses a spawn whose wheels would touch the kerb: one below *floor*."""

    def __init__(self, floor: float) -> None:
        super().__init__([])
        self.floor = floor

    def try_spawn_actor(self, blueprint: _Blueprint, transform: Any) -> Any:
        if transform.location.z < self.floor:
            self.spawn_attempts += 1
            return None
        return super().try_spawn_actor(blueprint, transform)


def test_a_spawn_is_as_near_the_road_as_carla_takes_it() -> None:
    # Spawned high, a mirror falls onto the road and loses its speed (#1003):
    # the lowest lift CARLA takes, higher only when it refuses one.
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(SumoBackendConfig())
    centre = Pose2D(10.0, 20.0, 0.0, 90.0)
    world = _KerbWorld(floor=0.1)
    actor = backend._spawn_at_centre(world, _Blueprint(), "vehicle.fake", centre)
    assert actor.get_transform().location.z == pytest.approx(0.2)
    assert world.spawn_attempts == 2
    nowhere = _KerbWorld(floor=1.0)
    assert (
        backend._spawn_at_centre(nowhere, _Blueprint(), "vehicle.fake", centre) is None
    )
    assert nowhere.spawn_attempts == 3


@needs_sumo
@pytest.mark.parametrize("box_location", [None, _BOX_LOCATION])
def test_background_vehicles_are_where_their_mirrors_origins_are(
    crossroads: Path, tmp_path: Path, box_location: Any
) -> None:
    """At their CARLA origin, as the TrafficManager backend's: the box it
    turns into ends at SUMO's front bumper."""
    backend, world, _npc, _ego = _prepared(
        crossroads, tmp_path, box_location=box_location
    )
    tc = backend._traci
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=20.0
        )
        assert handle is not None
        for i in range(5):
            backend.tick(world, i * 0.05)
        assert handle in backend._ambient
        x, y = backend.background_vehicles(world)[handle]
        yaw = math.radians(_carla_yaw(backend, handle))
        reach = float(tc.vehicle.getLength(handle)) / 2.0 + (
            box_location[0] if box_location else 0.0
        )
        assert (x + reach * math.cos(yaw), y + reach * math.sin(yaw)) == (
            pytest.approx(_sumo_bumper(backend, handle), abs=1e-4)
        )
    finally:
        backend.close()


@needs_sumo
@pytest.mark.parametrize("box_offset", [(0.0, 0.0), (1.3, 0.0)])
def test_a_background_vehicle_is_spawned_with_its_origin_at_the_transform(
    crossroads: Path, tmp_path: Path, box_offset: tuple[float, float]
) -> None:
    """The transform is the vehicle's origin, as the TrafficManager backend
    spawns it and as background_vehicles gives it back: SUMO is asked for the
    bumper half a length ahead of the box centre its blueprint's offset puts
    ahead of that origin."""
    from autoware_carla_scenario.traffic.sumo import backend as backend_module

    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    asked: list[Pose2D] = []
    road_position = backend._road_position

    def record(pose: Pose2D, *args: Any, **kwargs: Any) -> Any:
        asked.append(pose)
        return road_position(pose, *args, **kwargs)

    backend._road_position = record
    backend._blueprint_offsets[backend._config.blueprint] = box_offset
    try:
        handle = backend.spawn_background(
            world, _carla_transform(-60.0, 5.25, 0.0), speed_kmh=20.0
        )
        assert handle is not None
        half = backend_module._BACKGROUND_LENGTH_M / 2.0
        assert carla_xy_to_sumo(-60.0 + box_offset[0] + half, 5.25) == pytest.approx(
            (asked[0].x, asked[0].y)
        )
    finally:
        backend.close()


@pytest.mark.parametrize("box_location", [None, _BOX_LOCATION])
def test_the_estimated_rear_axle_is_measured_from_the_origin(
    box_location: Any,
) -> None:
    """Without CARLA's wheels, the rear axle is half a wheelbase behind the
    box centre -- from the origin pure pursuit steers by, that is the box's
    offset further forward."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(SumoBackendConfig())
    actor = _FakeActor(_carla_transform(0.0, 0.0, 0.0), box_location=box_location)
    state = backend._geometry(actor)
    forward = box_location[0] if box_location else 0.0
    assert state["rear_offset"] == pytest.approx(forward - 0.29 * 4.6)
    assert state["wheel_base"] == pytest.approx(0.58 * 4.6)


def test_carla_light_states_reach_sumo_as_typesafe_carla_reports_them() -> None:
    """typesafe_carla's ``TrafficLight.get_state()`` gives a plain int."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    class _Lights:
        def __init__(self) -> None:
            self.states = {"tls": "rrrr"}

        def getRedYellowGreenState(self, tls: str) -> str:
            return self.states[tls]

        def setRedYellowGreenState(self, tls: str, state: str) -> None:
            self.states[tls] = state

    def light(state: Any) -> SimpleNamespace:
        return SimpleNamespace(get_state=lambda: int(state))

    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend._traci = SimpleNamespace(trafficlight=_Lights())
    backend._crossing_links = {}
    backend._bicycle_rides = {}
    backend._permissive_links = {}
    backend._signals = {
        ("tls", 0): light(carla.TrafficLightState.Green),
        ("tls", 1): light(carla.TrafficLightState.Yellow),
        ("tls", 3): light(carla.TrafficLightState.Off),  # no SUMO state: kept
    }
    backend._signals_to_sumo()
    assert backend._traci.trafficlight.states["tls"] == "Gyrr"


def test_a_crossing_walks_only_while_the_roads_it_crosses_are_red() -> None:
    """A crossing's link has no CARLA light: it is green while every vehicle
    link over a road it crosses is red, and red otherwise."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    states = {"tls": "rrrrGG"}
    lights = SimpleNamespace(
        getRedYellowGreenState=lambda tls: states[tls],
        setRedYellowGreenState=lambda tls, state: states.__setitem__(tls, state),
    )
    green = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Green))
    red = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Red))
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend._traci = SimpleNamespace(trafficlight=lights)
    # Links 0-1 drive over the road crossing 4 crosses, 2-3 over crossing 5's.
    backend._crossing_links = {"tls": [(4, [0, 1]), (5, [2, 3])]}
    backend._bicycle_rides = {}
    backend._permissive_links = {}
    backend._signals = {
        ("tls", 0): green,
        ("tls", 1): red,
        ("tls", 2): red,
        ("tls", 3): red,
    }
    backend._signals_to_sumo()
    assert states["tls"] == "GrrrrG"


def _permissive_backend(
    program: list[str], start: str, lights: dict[int, Any]
) -> tuple[Any, dict[str, str]]:
    """A backend whose tls ``tls`` runs *program*, shows *start*, and has
    *lights* switching its links, with its permissive links matched."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    states = {"tls": start}
    logic = SimpleNamespace(phases=[SimpleNamespace(state=s) for s in program])
    trafficlight = SimpleNamespace(
        getRedYellowGreenState=lambda tls: states[tls],
        setRedYellowGreenState=lambda tls, state: states.__setitem__(tls, state),
        getAllProgramLogics=lambda tls: [logic],
    )
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend._traci = SimpleNamespace(trafficlight=trafficlight)
    backend._crossing_links = {}
    backend._bicycle_rides = {}
    backend._signals = {("tls", index): light for index, light in lights.items()}
    backend._permissive_links = backend._match_permissive()
    return backend, states


def test_a_permissive_green_stays_one_after_a_red() -> None:
    """A link SUMO's program gives a permissive green (``g``) gets one on
    every CARLA Green, also after it has shown yellow and red."""
    import typesafe_carla.carla as carla

    shown = [carla.TrafficLightState.Green]
    light = SimpleNamespace(get_state=lambda: int(shown[0]))
    backend, states = _permissive_backend(
        ["Gg", "yy", "rr"], "Gg", {0: light, 1: light}
    )
    assert backend._permissive_links == {"tls": {1}}
    seen = []
    for state in (
        carla.TrafficLightState.Green,
        carla.TrafficLightState.Yellow,
        carla.TrafficLightState.Red,
        carla.TrafficLightState.Green,
    ):
        shown[0] = state
        backend._signals_to_sumo()
        seen.append(states["tls"])
    assert seen == ["Gg", "yy", "rr", "Gg"]


def test_a_permissive_green_comes_from_the_program_not_the_first_phase() -> None:
    """A link whose permissive green is in a later phase than the one SUMO
    starts in still gets ``g``; a link never ``g`` gets ``G``."""
    import typesafe_carla.carla as carla

    green = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Green))
    backend, states = _permissive_backend(
        ["Grr", "yrr", "rGg", "ryy"], "Grr", {0: green, 1: green, 2: green}
    )
    assert backend._permissive_links == {"tls": {2}}
    backend._signals_to_sumo()
    assert states["tls"] == "GGg"


def test_a_protected_then_permissive_turn_gets_the_permissive_green() -> None:
    """A link ``G`` in one phase and ``g`` in another yields on a CARLA
    Green: the Green does not say which phase it is."""
    import typesafe_carla.carla as carla

    green = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Green))
    backend, states = _permissive_backend(
        ["GrG", "yry", "GGg", "yyy", "rrr"], "GrG", {0: green, 2: green}
    )
    assert backend._permissive_links == {"tls": {2}}
    backend._signals_to_sumo()
    assert states["tls"] == "Grg"


def test_only_links_carlas_lights_switch_are_matched_as_permissive() -> None:
    import typesafe_carla.carla as carla

    green = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Green))
    backend, _states = _permissive_backend(["ggg", "rrr"], "ggg", {1: green})
    assert backend._permissive_links == {"tls": {1}}
    backend, _states = _permissive_backend(["GGr", "rrG"], "GGr", {0: green})
    assert backend._permissive_links == {}


def test_sumo_light_states_are_sent_to_carla_only_when_they_change() -> None:
    """``set_state()`` is an RPC: a light is set once per tick at most, and only
    when it shows something else -- also when a scenario switched it."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    class _Light:
        def __init__(self) -> None:
            self.state = int(carla.TrafficLightState.Red)
            self.sets = 0

        def get_state(self) -> int:
            return self.state

        def set_state(self, state: Any) -> None:
            self.sets += 1
            self.state = int(state)

    lights = SimpleNamespace(states={"tls": "GGr"})
    lights.getRedYellowGreenState = lambda tls: lights.states[tls]
    two_links, one_link = _Light(), _Light()
    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend._traci = SimpleNamespace(trafficlight=lights)
    backend._signals = {
        ("tls", 0): two_links,
        ("tls", 1): two_links,
        ("tls", 2): one_link,
    }
    backend._signals_to_carla()
    assert (two_links.sets, one_link.sets) == (1, 0)
    assert two_links.state == int(carla.TrafficLightState.Green)
    backend._signals_to_carla()  # nothing changed
    assert (two_links.sets, one_link.sets) == (1, 0)
    two_links.state = int(carla.TrafficLightState.Red)  # a scenario's action
    lights.states["tls"] = "GGy"
    backend._signals_to_carla()
    assert (two_links.sets, one_link.sets) == (2, 1)
    assert two_links.state == int(carla.TrafficLightState.Green)
    assert one_link.state == int(carla.TrafficLightState.Yellow)


class _FakeTrace:
    """roadgen.Trace's ``translate``, for a map whose signal 949 switches two links."""

    def translate(self, fmt: str, element: str, *, to: str) -> list[dict[str, Any]]:
        assert (fmt, to) == ("opendrive", "sumo")
        if element != "signal:949":
            return []  # as roadgen answers for a signal the map does not have
        return [
            {"ref": "tls:j_189", "role": "traffic_light"},
            {"ref": "tls:j_189/13", "role": "link"},
            {"ref": "tls:j_189/12", "role": "link"},
            {"ref": "tls:j_189/12", "role": "link"},
        ]


def test_the_signal_table_reads_a_lights_links_off_the_traces() -> None:
    from autoware_carla_scenario.traffic.sumo.network import SignalTable

    table = SignalTable(_FakeTrace())
    assert table.links("949") == [("j_189", 12), ("j_189", 13)]
    assert table.links("1") == []


def test_a_network_built_without_traces_has_no_signal_table(tmp_path: Path) -> None:
    from autoware_carla_scenario.traffic.sumo.network import SignalTable, SumoNetwork

    network = SumoNetwork(tmp_path / "network.net.xml", tmp_path / "network.safe")
    assert SignalTable.load(network) is None


def test_carla_lights_are_matched_to_sumo_links_by_their_opendrive_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from autoware_carla_scenario.traffic.sumo import backend as sumo_backend
    from autoware_carla_scenario.traffic.sumo.network import SignalTable, SumoNetwork

    lights = [
        SimpleNamespace(get_opendrive_id=lambda: "949"),
        SimpleNamespace(get_opendrive_id=lambda: "1"),
    ]
    world = SimpleNamespace(
        get_actors=lambda: SimpleNamespace(
            filter=lambda pattern: lights if pattern == "traffic.traffic_light" else []
        )
    )
    monkeypatch.setattr(
        SignalTable, "load", classmethod(lambda cls, network: cls(_FakeTrace()))
    )
    backend = sumo_backend.SumoTrafficBackend.__new__(sumo_backend.SumoTrafficBackend)
    backend._network = SumoNetwork(Path("network.net.xml"), Path("network.safe"))
    with caplog.at_level("WARNING"):
        matched = backend._match_signals(world)
    assert matched == {("j_189", 12): lights[0], ("j_189", 13): lights[0]}
    assert "light(s) 1" in caplog.text


@needs_sumo
def test_a_network_keeps_its_traces_whatever_its_map_is_called(
    crossroads: Path, tmp_path: Path
) -> None:
    """roadgen names its files after the map, and `map` is a name a map can have."""
    import re

    from autoware_carla_scenario.traffic.sumo.network import (
        SignalTable,
        build_network,
    )

    text = crossroads.read_text()
    header = re.search(r"<header\b[^>]*>", text)
    assert header is not None
    named = re.sub(r'\sname="[^"]*"', "", header.group(0)).replace(
        "<header", '<header name="map"', 1
    )
    network = build_network(text.replace(header.group(0), named, 1), tmp_path)
    assert all(path.is_file() for path in network.trace_files), sorted(
        path.name for path in network.net_file.parent.iterdir()
    )
    assert SignalTable.load(network) is not None


# ---------------------------------------------------------------------------
# Mesoscopic far traffic
# ---------------------------------------------------------------------------


class _LeavingEntity(DrivenEntity):
    """A scenario vehicle that lets the backend take its actor away."""

    def __init__(self, actor: _FakeActor, role_name: str) -> None:
        self.actor: Any = actor
        self.role_name = role_name

    def _replace_actor(self, actor: Any) -> bool:
        self.actor = actor
        return True


def _mesoscopic(
    crossroads: Path, tmp_path: Path, **options: Any
) -> tuple[Any, _FakeWorld, _LeavingEntity, _FakeActor]:
    """A started backend whose NPC (SUMO's, and free to leave CARLA) drives
    away east from an ego standing at the west end of arm 0."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    meso = {
        "enabled": True,
        "micro_radius_m": 40.0,
        "meso_radius_m": 50.0,
        "update_interval_s": 0.0,
        "scenario_vehicles": True,
        **options.pop("mesoscopic", {}),
    }
    ego_published = options.pop("ego_published", True)
    config = SumoBackendConfig(
        scenario_vehicles="sumo",
        ambient=AmbientTrafficConfig(
            enabled="period_s" in options, period_s=options.pop("period_s", None)
        ),
        traffic_light_authority="none",
        warmup_s=options.pop("warmup_s", 0.0),
        vehicle_control="teleport",
        cache_dir=str(tmp_path / "cache"),
        mesoscopic=MesoscopicConfig(**meso),
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
    npc = _FakeActor(_carla_transform(-80.0, 5.25, 0.0), "npc", "vehicle.npc")
    ego = _FakeActor(_carla_transform(-110.0, 1.75, 0.0), str(EGO_ROLE_NAME))
    world = _FakeWorld([npc, ego])
    entity = _LeavingEntity(npc, "npc")
    backend.adopt(entity)
    backend.adopt(_Entity(ego, str(EGO_ROLE_NAME)))
    backend.start(world, skip_actor_ids={ego.id} if ego_published else ())
    return backend, world, entity, ego


def _tick_until(backend: Any, world: _FakeWorld, done: Any, ticks: int = 600) -> None:
    for i in range(ticks):
        backend.tick(world, i * 0.05)
        if done():
            return
    raise AssertionError("never happened")


@needs_sumo
def test_a_far_vehicle_leaves_sumo_and_carla_and_comes_back_near(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, entity, ego = _mesoscopic(crossroads, tmp_path)
    tc = backend._traci
    npc = entity.actor
    (sumo_id,) = backend._driven
    try:
        assert entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC
        # Driving east, away from the ego: more than 50 m off, it goes.
        _tick_until(
            backend,
            world,
            lambda: entity.simulation_fidelity is SimulationFidelity.MESOSCOPIC,
        )
        assert entity.actor is None and npc.destroyed
        assert sumo_id not in tc.vehicle.getIDList() and sumo_id in backend._meso
        assert backend.fidelity_census() == {"microscopic": 1, "mesoscopic": 1}
        assert backend.fidelity_of(entity) is SimulationFidelity.MESOSCOPIC
        # The queue model drives it on, east.
        before = backend._meso_carla_pose(sumo_id, backend._now()).x
        for i in range(40):
            backend.tick(world, i * 0.05)
        after = backend._meso_carla_pose(sumo_id, backend._now())
        assert after.x > before + 5.0
        assert after.y == pytest.approx(5.25, abs=2.0)  # an eastbound lane

        # The ego comes near: it is back, in SUMO and as a new actor in CARLA.
        ego.transform = _carla_transform(after.x - 20.0, 1.75, 0.0)
        _tick_until(
            backend,
            world,
            lambda: entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC,
            ticks=5,
        )
        back = entity.actor
        assert back is not None and back in world.spawned
        assert back.attributes["role_name"] == "npc"
        assert backend._driven[sumo_id] is entity
        assert sumo_id not in backend._meso
        backend.tick(world, 0.0)
        assert sumo_id in tc.vehicle.getIDList()
        assert back.transform.location.x == pytest.approx(after.x, abs=15.0)
    finally:
        backend.close()


@needs_sumo
def test_a_mesoscopic_vehicle_takes_its_manoeuvres_as_far_as_it_can(
    crossroads: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend, world, entity, ego = _mesoscopic(crossroads, tmp_path)
    (sumo_id,) = backend._driven
    try:
        _tick_until(
            backend,
            world,
            lambda: entity.simulation_fidelity is SimulationFidelity.MESOSCOPIC,
        )
        backend.set_desired_speed(entity, world, 18.0)
        assert backend._meso.vehicle(sumo_id).speed_cap == pytest.approx(5.0)
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        assert "no lanes" in caplog.text
        assert not backend.lane_change_finished(entity, world)

        # An action that needs its actor has it at once, wherever it is.
        assert backend.require_microscopic(entity, world)
        assert entity.actor is not None
        assert entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC
        backend.tick(world, 0.0)
        assert backend._traci.vehicle.getSpeed(sumo_id) <= 5.0 + 0.1

        # Released, CARLA drives it: published, and never mesoscopic again.
        backend.release(entity, world)
        assert backend._external[sumo_id] is entity.actor
        for i in range(100):
            backend.tick(world, i * 0.05)
        assert entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC
    finally:
        backend.close()
    assert entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC


@needs_sumo
def test_an_ego_sumo_drives_is_a_focus_point_too(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, entity, ego = _mesoscopic(crossroads, tmp_path, ego_published=False)
    try:
        assert backend._external == {} and len(backend._driven) == 2
        ego_entity = next(e for e in backend._driven.values() if e.actor is ego)
        backend.set_desired_speed(ego_entity, world, 0.0)
        # The NPC, 30 m ahead of the ego, leaves once it is 50 m ahead.
        _tick_until(
            backend,
            world,
            lambda: entity.simulation_fidelity is SimulationFidelity.MESOSCOPIC,
        )
        # The ego itself never does: its entity keeps its actor.
        assert not ego.destroyed and len(backend._driven) == 1
    finally:
        backend.close()


@needs_sumo
def test_the_scenarios_vehicles_stay_in_carla_unless_let_go(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, entity, _ego = _mesoscopic(
        crossroads, tmp_path, mesoscopic={"scenario_vehicles": False}
    )
    try:
        for i in range(300):
            backend.tick(world, i * 0.05)
        assert entity.simulation_fidelity is SimulationFidelity.MICROSCOPIC
        assert entity.actor is not None and not entity.actor.destroyed
        assert len(backend._meso) == 0
    finally:
        backend.close()


@needs_sumo
def test_only_the_traffic_near_the_ego_is_in_sumo_and_carla(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _entity, ego = _mesoscopic(
        crossroads, tmp_path, period_s=0.5, warmup_s=20.0
    )
    tc = backend._traci
    try:
        for i in range(100):
            backend.tick(world, i * 0.05)
        census = backend.fidelity_census()
        assert census["mesoscopic"] > 0 and census["microscopic"] > 0
        ego_pose = backend._carla_pose_to_sumo(ego)
        ego_x, ego_y = ego_pose.x, ego_pose.y
        own = set(backend._driven) | set(backend._external)
        for sumo_id in tc.vehicle.getIDList():
            if sumo_id not in own:
                x, y = tc.vehicle.getPosition(sumo_id)
                # 50 m, and what one decision's worth of driving adds to it.
                assert math.hypot(x - ego_x, y - ego_y) < 55.0
        alive = [a for a in world.spawned if not a.destroyed]
        assert len(alive) == len([i for i in tc.vehicle.getIDList() if i not in own])
    finally:
        backend.close()
    assert backend._meso is None


# ---------------------------------------------------------------------------
# SUMO's own pedestrians
# ---------------------------------------------------------------------------


def test_the_pedestrian_options_are_checked() -> None:
    config = SumoBackendConfig.from_mapping(
        {"pedestrians": {"enabled": True, "period_s": 2.0, "radius_m": 40.0}}
    )
    assert config.pedestrians.enabled and config.pedestrians.period() == 2.0
    assert PedestrianConfig(persons_per_hour=1800).period() == pytest.approx(2.0)
    assert not SumoBackendConfig().pedestrians.enabled
    with pytest.raises(ValueError):
        PedestrianConfig(radius_m=0.0)


def _walking(
    crossroads: Path,
    tmp_path: Path,
    walkers: Any = None,
    **pedestrians: Any,
) -> tuple[Any, _FakeWorld, _FakeActor]:
    """A started backend generating SUMO pedestrians, with an ego published
    from the west arm; *walkers* are its pedestrian parameters."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    config = SumoBackendConfig(
        traffic_light_authority="none",
        warmup_s=pedestrians.pop("warmup_s", 0.0),
        vehicle_control="teleport",
        cache_dir=str(tmp_path / "cache"),
        ambient=AmbientTrafficConfig(enabled=False),
        pedestrians=PedestrianConfig(enabled=True, period_s=0.5, **pedestrians),
    )
    backend = SumoTrafficBackend(config, pedestrians=walkers)
    backend.prepare(
        TrafficContext(
            xodr_path=crossroads,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    ego = _FakeActor(_carla_transform(-100.0, 1.75, 0.0), str(EGO_ROLE_NAME))
    world = _FakeWorld([ego])
    backend.adopt(_Entity(ego, str(EGO_ROLE_NAME)))
    backend.start(world, skip_actor_ids={ego.id})
    return backend, world, ego


def _mirrors(world: _FakeWorld) -> list[_FakeActor]:
    return [
        a
        for a in world.spawned
        if a.attributes["role_name"].startswith("sumo:sumoped") and not a.destroyed
    ]


@needs_sumo
def test_sumos_pedestrians_walk_in_carla_as_walkers(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _ego = _walking(crossroads_with_sidewalks, tmp_path, warmup_s=20.0)
    tc = backend._traci
    try:
        assert (tmp_path / "out" / "sumo" / "pedestrians.rou.xml").is_file()
        for i in range(40):
            backend.tick(world, i * 0.05)
        mirrors = _mirrors(world)
        assert mirrors and len(mirrors) == len(backend._persons)
        for person, walker in backend._persons.items():
            x, y = tc.person.getPosition(person)
            location = walker.transform.location
            # Where SUMO has it (CARLA's y mirrored), a step behind at most.
            assert math.hypot(location.x - x, location.y + y) < 0.5
        walking = [w for w in mirrors if getattr(w, "control", None) is not None]
        assert any(w.control.speed > 0.5 for w in walking)

        # A mirror is CARLA's walker, but not one to publish back into SUMO.
        for walker in mirrors:
            walker.type_id = "walker.pedestrian.0001"
        world.actors.extend(mirrors)
        backend.tick(world, 2.0)
        assert backend._walkers == {}
    finally:
        backend.close()
    assert _mirrors(world) == []
    assert backend._persons == {}


@needs_sumo
def test_only_the_pedestrians_near_the_ego_are_in_carla(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, ego = _walking(
        crossroads_with_sidewalks, tmp_path, warmup_s=60.0, radius_m=40.0
    )
    tc = backend._traci
    try:
        for i in range(100):
            backend.tick(world, i * 0.05)
        assert backend._persons, "no pedestrian came near"
        assert len(backend._persons) < tc.person.getIDCount()
        focus = backend._carla_pose_to_sumo(ego)
        for person in backend._persons:
            x, y = tc.person.getPosition(person)
            assert math.hypot(x - focus.x, y - focus.y) <= 50.0 + 1.0
        limited, world2, _ = _walking(
            crossroads_with_sidewalks,
            tmp_path / "limited",
            warmup_s=60.0,
            max_persons=1,
        )
        try:
            for i in range(10):
                limited.tick(world2, i * 0.05)
            assert len(limited._persons) == 1
        finally:
            limited.close()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# The scenario's pedestrians
# ---------------------------------------------------------------------------


def _scenario_pedestrian(
    network: Path, tmp_path: Path, **options: Any
) -> tuple[Any, _FakeWorld, Any]:
    """A started backend with a scenario pedestrian at the south kerb of the
    west arm, facing across its eastbound lanes (CARLA yaw -90: SUMO north)."""
    from autoware_carla_scenario.entity.pedestrian_entity import (
        PedestrianEntity,
        PedestrianEntityConfig,
    )
    from autoware_carla_scenario.entity._spawn import SpawnTransform
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(
        SumoBackendConfig(
            traffic_light_authority="none",
            warmup_s=0.0,
            vehicle_control="teleport",
            cache_dir=str(tmp_path / "cache"),
            **options,
        )
    )
    backend.prepare(
        TrafficContext(
            xodr_path=network,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    walker = _FakeActor(
        _carla_transform(-50.0, 8.25, -90.0), "pedestrian1", "walker.pedestrian.0001"
    )
    walker.bounding_box.extent = _Extent(0.2, 0.2)
    entity = PedestrianEntity(
        PedestrianEntityConfig("pedestrian1", SpawnTransform(walker.transform))
    )
    entity._walker = walker  # type: ignore[assignment]
    entity.set_traffic_backend(backend)
    backend.adopt_pedestrian(entity)
    world = _FakeWorld([walker])
    backend.start(world)
    for i in range(3):
        backend.tick(world, i * 0.05)
    return backend, world, entity


@needs_sumo
def test_by_default_the_scenarios_pedestrians_walk_in_carla(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.entity.pedestrian_entity import walker_speed_command

    backend, world, entity = _scenario_pedestrian(crossroads_with_sidewalks, tmp_path)
    walker = entity.actor
    try:
        entity.walk_straight(1.5)
        # Walked straight ahead in CARLA, and published into SUMO.
        assert walker.control.speed == pytest.approx(walker_speed_command(1.5))
        assert backend._walked == {}
        assert list(backend._walkers) == [walker.id]
    finally:
        backend.close()


@needs_sumo
def test_sumo_walks_the_scenarios_pedestrian_straight_across_the_road(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, entity = _scenario_pedestrian(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    tc = backend._traci
    walker = entity.actor
    try:
        (person,) = backend._walked
        assert backend._walkers == {}  # walked, not published
        # Standing, the person stands where its walker does.
        assert tc.person.getPosition(person) == pytest.approx((-50.0, -8.25), abs=0.1)
        entity.walk_straight(1.5)
        assert list(backend._walking) == [person]
        lanes = set()
        for i in range(80):  # 4 s
            backend.tick(world, 1.0 + i * 0.05)
            lanes.add(tc.person.getLaneID(person))
            x, y = tc.person.getPosition(person)
            location = walker.transform.location
            # Where SUMO has it (CARLA's y mirrored), a step behind at most.
            assert math.hypot(location.x - x, location.y + y) < 0.1
            # CARLA 0.10's command for 1.5 m/s.
            assert walker.control.speed == pytest.approx(1.5 * 20.5)
        # Straight on across the eastbound lanes, at its speed, on the road
        # lanes SUMO's traffic drives (and yields to it on).
        assert tc.person.getPosition(person) == pytest.approx(
            (-50.0, -8.25 + 1.5 * 4.0), abs=0.1
        )
        assert any(backend._net.getLane(lane).allows("passenger") for lane in lanes)
        # Neither published nor mirrored back into CARLA.
        assert backend._walkers == {} and world.spawned == []
    finally:
        backend.close()
    assert backend._walked == {} and backend._pedestrian_entities == []


@needs_sumo
def test_sumo_walks_the_scenarios_pedestrian_the_way_it_is_told(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    """A walk given a yaw goes that way, not the way the walker faces: how a
    pedestrian turning back walks back to the kerb."""
    backend, world, entity = _scenario_pedestrian(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    tc = backend._traci
    walker = entity.actor
    try:
        (person,) = backend._walked
        entity.walk_straight(1.5)
        for i in range(20):  # 1 s out
            backend.tick(world, 1.0 + i * 0.05)
        out = tc.person.getPosition(person)
        assert out == pytest.approx((-50.0, -8.25 + 1.5), abs=0.1)
        # Back the way it came, though the walker still faces out.
        entity.walk_straight(1.5, yaw_deg=walker.transform.rotation.yaw + 180.0)
        for i in range(20):  # 1 s back
            backend.tick(world, 2.0 + i * 0.05)
        assert tc.person.getPosition(person) == pytest.approx((-50.0, -8.25), abs=0.1)
    finally:
        backend.close()


@needs_sumo
def test_sumo_traffic_stops_for_the_scenarios_pedestrian_sumo_walks(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.entity._spawn import SpawnTransform
    from autoware_carla_scenario.entity.pedestrian_entity import (
        PedestrianEntity,
        PedestrianEntityConfig,
    )

    backend, world, _npc, _ego = _prepared(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    sumo_id = next(iter(backend._driven))
    tc = backend._traci
    # A pedestrian 30 m ahead of the NPC SUMO drives, at the kerb, facing
    # across the NPC's lanes.
    walker = _FakeActor(
        _carla_transform(-50.0, 8.25, -90.0), "pedestrian1", "walker.pedestrian.0001"
    )
    walker.bounding_box.extent = _Extent(0.2, 0.2)
    entity = PedestrianEntity(
        PedestrianEntityConfig("pedestrian1", SpawnTransform(walker.transform))
    )
    entity._walker = walker  # type: ignore[assignment]
    entity.set_traffic_backend(backend)
    world.actors.append(walker)
    # Taken on as start() takes on the pedestrians adopted before it.
    backend._take_pedestrian(entity)
    try:
        backend.tick(world, 0.0)
        entity.walk_straight(1.0)
        # 3 s on, it stops in the middle of the NPC's lane (SUMO y -5.25).
        for i in range(1, 61):
            backend.tick(world, i * 0.05)
        entity.walk_straight(0.0)
        assert backend._traci.person.getLaneID(next(iter(backend._walked))) == (
            tc.vehicle.getLaneID(sumo_id)
        )
        for i in range(61, 300):
            backend.tick(world, i * 0.05)
        # The NPC stopped short of it.
        assert tc.vehicle.getSpeed(sumo_id) == pytest.approx(0.0, abs=0.05)
        assert -60.0 < tc.vehicle.getPosition(sumo_id)[0] < -50.0
    finally:
        backend.close()


@needs_sumo
def test_a_scenario_pedestrian_sumo_walks_stops_at_zero_speed(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, entity = _scenario_pedestrian(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    tc = backend._traci
    try:
        (person,) = backend._walked
        entity.walk_straight(1.5)
        for i in range(40):
            backend.tick(world, 1.0 + i * 0.05)
        moved = tc.person.getPosition(person)
        assert moved == pytest.approx((-50.0, -8.25 + 3.0), abs=0.1)
        entity.walk_straight(0.0)
        assert backend._walking == {}
        assert entity.actor.control.speed == 0.0
        for i in range(40):
            backend.tick(world, 3.0 + i * 0.05)
        assert tc.person.getPosition(person) == pytest.approx(moved, abs=0.1)
    finally:
        backend.close()


@needs_sumo
def test_a_released_scenario_pedestrian_is_published_and_walks_in_carla(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.entity.pedestrian_entity import walker_speed_command

    backend, world, entity = _scenario_pedestrian(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    walker = entity.actor
    try:
        (person,) = backend._walked
        entity.release_from_traffic(world)
        assert backend._walked == {} and backend._walkers == {walker.id: person}
        entity.walk_straight(1.5)
        assert walker.control.speed == pytest.approx(walker_speed_command(1.5))
        # Its pose goes into SUMO every tick, as any CARLA pedestrian's.
        walker.transform = _carla_transform(-50.0, 4.0, -90.0)
        backend.tick(world, 1.0)
        backend.tick(world, 1.05)
        x, y = backend._traci.person.getPosition(person)
        assert (x, y) == pytest.approx((-50.0, -4.0), abs=0.1)
    finally:
        backend.close()


@needs_sumo
def test_a_scenario_pedestrian_with_no_footway_near_walks_in_carla(
    crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.entity.pedestrian_entity import walker_speed_command

    backend, world, entity = _scenario_pedestrian(
        crossroads, tmp_path, scenario_pedestrians="sumo"
    )
    try:
        assert backend._walked == {}
        entity.walk_straight(1.5)
        assert entity.actor.control.speed == pytest.approx(walker_speed_command(1.5))
        assert backend._running
    finally:
        backend.close()


@needs_sumo
def test_a_scenario_pedestrian_gone_from_carla_leaves_sumo(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, entity = _scenario_pedestrian(
        crossroads_with_sidewalks, tmp_path, scenario_pedestrians="sumo"
    )
    try:
        entity.walk_straight(1.5)
        world.actors.clear()
        backend.tick(world, 1.0)
        backend.tick(world, 1.05)
        assert backend._walked == {} and backend._walking == {}
        assert backend._traci.person.getIDList() == ()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Pedestrian crossings
# ---------------------------------------------------------------------------


def _vehicle_links(net_file: Path) -> list[tuple[str, int, str, str]]:
    import sumolib

    net = sumolib.net.readNet(str(net_file), withInternal=True)
    return sorted(
        (tls.getID(), index, in_lane.getID(), out_lane.getID())
        for tls in net.getTrafficLights()
        for in_lane, out_lane, index in tls.getConnections()
        if out_lane.getEdge().getFunction() == ""
    )


@needs_sumo
def test_the_network_has_crossings_and_walking_areas(
    signalled_crossroads: Path, tmp_path: Path
) -> None:
    import sumolib

    from autoware_carla_scenario.traffic.sumo.network import (
        build_network,
        crossing_edges,
    )

    opendrive = signalled_crossroads.read_text()
    plain = build_network(opendrive, tmp_path / "cache")
    walking = build_network(opendrive, tmp_path / "cache", pedestrian_crossings=True)
    assert plain.net_file != walking.net_file  # cached apart
    assert crossing_edges(plain.net_file) == {}
    crossings = crossing_edges(walking.net_file)
    assert crossings and all(crossings.values())
    net = sumolib.net.readNet(str(walking.net_file), withInternal=True)
    assert any(e.getFunction() == "walkingarea" for e in net.getEdges())
    # The vehicle links keep the indices CARLA's lights are matched to.
    assert _vehicle_links(walking.net_file) == _vehicle_links(plain.net_file)


@needs_sumo
def test_each_signalled_crossing_waits_for_the_traffic_over_its_road(
    signalled_crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    backend = SumoTrafficBackend(
        SumoBackendConfig(
            warmup_s=0.0,
            vehicle_control="teleport",
            cache_dir=str(tmp_path / "cache"),
            ambient=AmbientTrafficConfig(enabled=False),
        )
    )
    backend.prepare(
        TrafficContext(
            xodr_path=signalled_crossroads,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    try:
        backend.start(_FakeWorld([]))
        ((tls, links),) = backend._crossing_links.items()
        assert backend._network is not None
        vehicle_count = len(_vehicle_links(backend._network.net_file))
        assert len(links) >= 4
        for index, foes in links:
            # After the vehicle links, and waiting for some of them only.
            assert index >= vehicle_count
            assert foes and all(foe < vehicle_count for foe in foes)
            assert len(foes) < vehicle_count
        assert tls in backend._traci.trafficlight.getIDList()
    finally:
        backend.close()
    assert backend._crossing_links == {}


@needs_sumo
def test_sumos_pedestrians_walk_with_the_pedestrian_parameters(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import PedestrianParameters, Spread

    backend, world, _ego = _walking(
        crossroads_with_sidewalks,
        tmp_path,
        PedestrianParameters(walking_speed_mps=Spread(1.2, 0.3), min_gap_m=Spread(0.6)),
    )
    tc = backend._traci
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
        people = tc.person.getIDList()
        assert len(people) >= 5
        speeds = [tc.person.getMaxSpeed(p) for p in people]
        # Each person walks at their own draw: a spread, not one speed.
        assert len({round(v, 3) for v in speeds}) > 1
        assert all(0.05 <= v for v in speeds)
        for person in people:
            assert tc.person.getMinGap(person) == pytest.approx(0.6)
        assert backend.describe()["pedestrians"]["walking_speed_mps"] == {
            "mean": 1.2,
            "std": 0.3,
        }
    finally:
        backend.close()


def _steps_on_crossings_at_red(
    signalled_crossroads: Path, tmp_path: Path, walkers: Any
) -> int:
    """Person-steps spent on a crossing while every light is held red."""
    from autoware_carla_scenario.traffic.sumo.network import crossing_edges

    backend, _world, _ego = _walking(signalled_crossroads, tmp_path, walkers)
    tc = backend._traci
    try:
        crossings = set(crossing_edges(backend._network.net_file))
        on = 0
        for _ in range(2400):  # 120 s
            for light in tc.trafficlight.getIDList():
                state = tc.trafficlight.getRedYellowGreenState(light)
                tc.trafficlight.setRedYellowGreenState(light, "r" * len(state))
            tc.simulationStep()
            on += sum(
                1 for p in tc.person.getIDList() if tc.person.getRoadID(p) in crossings
            )
        return on
    finally:
        backend.close()


@needs_sumo
def test_sumos_pedestrians_cross_on_red_only_when_told_to(
    signalled_crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import PedestrianParameters, Spread

    waiting = _steps_on_crossings_at_red(signalled_crossroads, tmp_path / "a", None)
    crossing = _steps_on_crossings_at_red(
        signalled_crossroads,
        tmp_path / "b",
        PedestrianParameters(red_light_violation_s=Spread(1000.0)),
    )
    # A crossing's own green may still come up between the vehicle lights SUMO
    # holds red (pedestrian links can be green), so "never" is too strong:
    # crossing on red is several times as common.
    assert crossing > 3 * max(waiting, 1)


@needs_sumo
def test_sumos_pedestrians_cross_roads_on_crossings(
    signalled_crossroads: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.sumo.network import crossing_edges

    backend, world, _ego = _walking(signalled_crossroads, tmp_path, warmup_s=10.0)
    tc = backend._traci
    try:
        crossings = set(crossing_edges(backend._network.net_file))
        seen: set[str] = set()
        for i in range(1200):
            backend.tick(world, i * 0.05)
            seen |= {tc.person.getRoadID(p) for p in tc.person.getIDList()}
            if seen & crossings:
                break
        assert seen & crossings, "no pedestrian walked over a crossing"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Bicycles
# ---------------------------------------------------------------------------


def test_the_bicycle_options_are_checked() -> None:
    assert not SumoBackendConfig().bicycles.enabled
    assert not SumoBackendConfig().cycle_on_sidewalks
    config = SumoBackendConfig.from_mapping(
        {
            "cycle_on_sidewalks": True,
            "bicycles": {
                "enabled": True,
                "period_s": 4.0,
                "blueprints": ["vehicle.bh.crossbike"],
            },
        }
    )
    bicycles = config.bicycles
    assert config.cycle_on_sidewalks and bicycles.enabled
    assert bicycles.period() == 4.0
    assert bicycles.blueprints == ["vehicle.bh.crossbike"]
    assert BicycleConfig(bicycles_per_hour=1800).period() == pytest.approx(2.0)
    assert BicycleConfig().blueprints == list(DEFAULT_BICYCLE_BLUEPRINTS)
    assert BicycleConfig().vehicle_control == "teleport"
    with pytest.raises(ValueError, match="Unknown BicycleConfig"):
        SumoBackendConfig.from_mapping({"bicycles": {"speed": 3}})
    with pytest.raises(ValueError, match="vehicle_control"):
        BicycleConfig(vehicle_control="pedal")
    # How they ride is traffic.bicycles, which every backend reads.
    with pytest.raises(ValueError, match="traffic.bicycles"):
        BicycleConfig.from_mapping({"max_speed_mps": 4.0})
    for name in ("radius_m", "max_distance_m"):
        with pytest.raises(ValueError, match=name):
            BicycleConfig.from_mapping({name: 0.0})


def test_a_bicycle_is_a_vehicle_whose_base_type_says_so() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import is_bicycle

    bike = _FakeActor(None, type_id="vehicle.bh.crossbike")
    bike.attributes["base_type"] = "Bicycle"
    assert is_bicycle(bike)
    assert not is_bicycle(_FakeActor(None))
    assert not is_bicycle(SimpleNamespace())


def test_a_bicycle_ride_goes_with_its_traffic_or_waits_for_it_to_stop() -> None:
    from autoware_carla_scenario.traffic.sumo.signal_states import ride_state

    assert ride_state("with", ["r", "G"]) == "g"
    assert ride_state("with", ["r", "g"]) == "g"
    assert ride_state("with", ["y", "r"]) == "y"
    assert ride_state("with", ["r", "r"]) == "r"
    assert ride_state("clear", ["r", "r"]) == "G"
    assert ride_state("clear", ["r", "y"]) == "r"


def _riding(
    road: Path,
    tmp_path: Path,
    *,
    vehicles: Any = None,
    riders: Any = None,
    warmup_s: float = 30.0,
    **bicycles: Any,
) -> tuple[Any, _FakeWorld, _FakeActor]:
    """A started backend generating SUMO bicycles on a network whose
    sidewalks they may ride, with an ego published from the west arm;
    *vehicles* are its vehicle parameters, *riders* its bicycle ones."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    config = SumoBackendConfig(
        traffic_light_authority="none",
        warmup_s=warmup_s,
        vehicle_control="physics",
        cache_dir=str(tmp_path / "cache"),
        cycle_on_sidewalks=True,
        ambient=AmbientTrafficConfig(enabled=False),
        bicycles=BicycleConfig(enabled=True, period_s=1.0, **bicycles),
    )
    backend = SumoTrafficBackend(config, vehicles=vehicles, bicycles=riders)
    backend.prepare(
        TrafficContext(
            xodr_path=road,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    ego = _FakeActor(_carla_transform(-100.0, 1.75, 0.0), str(EGO_ROLE_NAME))
    world = _FakeWorld([ego])
    backend.adopt(_Entity(ego, str(EGO_ROLE_NAME)))
    backend.start(world, skip_actor_ids={ego.id})
    return backend, world, ego


def _bikes(backend: Any) -> list[str]:
    """SUMO's own bicycles in SUMO now: not the published ego."""
    return [i for i in backend._traci.vehicle.getIDList() if i.startswith("sumobike")]


def _bicycle_mirrors(world: _FakeWorld) -> list[_FakeActor]:
    return [
        a
        for a in world.spawned
        if a.attributes["role_name"].startswith("sumo:sumobike") and not a.destroyed
    ]


@needs_sumo
def test_sumos_bicycles_ride_the_sidewalks_and_are_mirrored_as_bicycles(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _ego = _riding(crossroads_with_sidewalks, tmp_path)
    tc = backend._traci
    try:
        assert (tmp_path / "out" / "sumo" / "bicycles.rou.xml").is_file()
        on_sidewalks = set()
        for i in range(100):
            backend.tick(world, i * 0.05)
            for bike in _bikes(backend):
                assert tc.vehicle.getVehicleClass(bike) == "bicycle"
                if ".bike" in tc.vehicle.getRoadID(bike):
                    on_sidewalks.add(bike)
        # The network gave them the sidewalks, and they took them.
        assert on_sidewalks
        mirrors = _bicycle_mirrors(world)
        assert mirrors and len(mirrors) == len(backend._ambient)
        assert {a.type_id for a in mirrors} <= set(DEFAULT_BICYCLE_BLUEPRINTS)
        # Placed where SUMO has them each tick, not driven by the car physics
        # vehicle_control asks for, which would trip on the kerb.
        assert all(a.constant_velocity is not None for a in mirrors)
        assert backend.describe()["cycle_on_sidewalks"] is True
        assert backend.describe()["bicycle_control"] == "teleport"
    finally:
        backend.close()
    assert _bicycle_mirrors(world) == []


@needs_sumo
def test_bicycles_count_against_their_own_limit(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _ego = _riding(crossroads_with_sidewalks, tmp_path, max_bicycles=2)
    try:
        for i in range(100):
            backend.tick(world, i * 0.05)
            assert len(_bicycle_mirrors(world)) <= 2
        assert len(_bikes(backend)) > 2
        assert len(_bicycle_mirrors(world)) == 2
    finally:
        backend.close()


@needs_sumo
def test_sumos_bicycles_ride_with_the_bicycle_parameters(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import (
        BicycleParameters,
        Spread,
        VehicleParameters,
    )

    backend, world, _ego = _riding(
        crossroads_with_sidewalks,
        tmp_path,
        # The drivers' parameters are a car's: none of them reaches a rider.
        vehicles=VehicleParameters(accel=Spread(4.0), min_gap_m=Spread(3.0)),
        riders=BicycleParameters(max_speed_mps=Spread(4.0, 0.8), accel=0.8),
    )
    tc = backend._traci
    try:
        assert (tmp_path / "out" / "sumo" / "bicycle_type.add.xml").is_file()
        for i in range(100):
            backend.tick(world, i * 0.05)
        bikes = _bikes(backend)
        assert len(bikes) >= 5
        speeds = [tc.vehicle.getMaxSpeed(b) for b in bikes]
        # Each rider at their own draw: a spread, not one speed.
        assert len({round(v, 3) for v in speeds}) > 1
        assert all(0.5 <= v for v in speeds)
        for bike in bikes:
            assert tc.vehicle.getAccel(bike) == pytest.approx(0.8)
            assert tc.vehicle.getMinGap(bike) != pytest.approx(3.0)
        assert backend.describe()["bicycles"]["max_speed_mps"] == {
            "mean": 4.0,
            "std": 0.8,
        }
    finally:
        backend.close()


@needs_sumo
def test_a_world_without_bicycles_keeps_sumos_in_sumo(
    crossroads_with_sidewalks: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend, world, _ego = _riding(crossroads_with_sidewalks, tmp_path, warmup_s=0.0)
    assert isinstance(backend, SumoTrafficBackend)
    world.missing = set(DEFAULT_BICYCLE_BLUEPRINTS)
    try:
        with caplog.at_level("WARNING"):
            for i in range(100):
                backend.tick(world, i * 0.05)
        assert _bikes(backend)
        assert _bicycle_mirrors(world) == []
        assert caplog.text.count("none of the bicycle blueprints") == 1
    finally:
        backend.close()


@needs_sumo
def test_a_scenario_bicycle_joins_sumo_as_a_bicycle(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    backend = SumoTrafficBackend(
        SumoBackendConfig(
            scenario_vehicles="sumo",
            traffic_light_authority="none",
            warmup_s=0.0,
            vehicle_control="teleport",
            cache_dir=str(tmp_path / "cache"),
            cycle_on_sidewalks=True,
            ambient=AmbientTrafficConfig(enabled=False),
        )
    )
    backend.prepare(
        TrafficContext(
            xodr_path=crossroads_with_sidewalks,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    # On the eastbound sidewalk of the west arm (CARLA y > 0 is OpenDRIVE's
    # right): 7 m out, beyond the two 3.5 m lanes.
    bike = _FakeActor(
        _carla_transform(-80.0, 8.0, 0.0), "rider", "vehicle.bh.crossbike"
    )
    bike.attributes["base_type"] = "bicycle"
    world = _FakeWorld([bike])
    backend.adopt(_Entity(bike, "rider"))
    backend.start(world)
    tc = backend._traci
    try:
        ((sumo_id, _entity),) = backend._driven.items()
        assert tc.vehicle.getVehicleClass(sumo_id) == "bicycle"
        # Not yet inserted, so on no road: where its route starts.
        assert ".bike" in tc.vehicle.getRoute(sumo_id)[0]
        for i in range(200):
            backend.tick(world, i * 0.05)
        # Ridden on, where bicycles may go, and never off the end of its route.
        assert bike.transform.location.x > -70.0
        assert sumo_id in tc.vehicle.getIDList()
        assert all(
            backend._net.getEdge(edge).allows("bicycle")
            for edge in tc.vehicle.getRoute(sumo_id)
        )
    finally:
        backend.close()


@needs_sumo
def test_signalled_bicycle_rides_go_with_the_traffic_carlas_lights_let_go(
    signalled_crossroads: Path, tmp_path: Path
) -> None:
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import AmbientTrafficConfig

    backend = SumoTrafficBackend(
        SumoBackendConfig(
            warmup_s=0.0,
            vehicle_control="teleport",
            cache_dir=str(tmp_path / "cache"),
            cycle_on_sidewalks=True,
            ambient=AmbientTrafficConfig(enabled=False),
        )
    )
    backend.prepare(
        TrafficContext(
            xodr_path=signalled_crossroads,
            fixed_delta_seconds=0.05,
            random_seed=3,
            output_dir=tmp_path / "out",
        )
    )
    tc = backend._traci
    try:
        backend.start(_FakeWorld([]))
        ((tls, rides),) = backend._bicycle_rides.items()
        links = {
            index: (controlled[0][0], controlled[0][1])
            for index, controlled in enumerate(tc.trafficlight.getControlledLinks(tls))
        }
        onto = {
            index: companions
            for index, follows, companions in rides
            if follows == "with" and ".bike" in links[index][0]
        }
        assert len(onto) >= 4
        for index, companions in onto.items():
            in_lane, out_lane = links[index]
            road = in_lane.split(".bike", 1)[0]
            # The traffic of the carriageway beside its sidewalk, into the
            # same road.
            assert companions
            for companion in companions:
                assert links[companion][0].split(".")[0] == road.split(".")[0]
                assert (
                    links[companion][1].rsplit("_", 1)[0] == out_lane.rsplit("_", 1)[0]
                )
        # CARLA lets one companion go: its rides follow, giving way; the rest
        # wait with the traffic they go with.
        index, companions = next(iter(onto.items()))
        green = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Green))
        red = SimpleNamespace(get_state=lambda: int(carla.TrafficLightState.Red))
        vehicles = {c for cs in onto.values() for c in cs}
        backend._signals = {
            (tls, i): green if i == companions[0] else red for i in vehicles
        }
        backend._signals_to_sumo()
        now = tc.trafficlight.getRedYellowGreenState(tls)
        assert now[index] == "g"
        assert all(now[i] == "r" for i in onto if companions[0] not in onto[i])
    finally:
        backend.close()
    assert backend._bicycle_rides == {}


def test_sumo_refuses_nothing_of_the_bicycle_parameters() -> None:
    from autoware_carla_scenario.traffic import TrafficManagerBackend
    from autoware_carla_scenario.traffic.parameters import (
        BICYCLE_PARAMETER_BACKENDS,
        BicycleParameters,
        TrafficParameterError,
    )
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    assert SumoTrafficBackend().supported_bicycle_parameters() == set(
        BICYCLE_PARAMETER_BACKENDS
    )
    # The speed and the red light are on the types SUMO starts with.
    assert "max_speed_mps" not in SumoTrafficBackend().supported_bicycle_overrides()
    with pytest.raises(TrafficParameterError, match="traffic.bicycles.decel"):
        TrafficManagerBackend().set_bicycle_parameters(BicycleParameters(decel=4.0))


@needs_sumo
def test_an_override_redraws_every_bicycle_sumo_drives(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    from autoware_carla_scenario.traffic.parameters import BicycleParameters, Spread

    backend, world, _ego = _riding(
        crossroads_with_sidewalks, tmp_path, riders=BicycleParameters(decel=2.0)
    )
    tc = backend._traci
    try:
        backend.tick(world, 0.0)
        before = _bikes(backend)
        assert before
        backend.override_bicycle_parameters(
            world, BicycleParameters(decel=Spread(4.0), min_gap_m=0.3)
        )
        for bike in before:
            assert tc.vehicle.getDecel(bike) == pytest.approx(4.0)
            assert tc.vehicle.getMinGap(bike) == pytest.approx(0.3)
        # One that departs later draws the overrides too.
        for i in range(40):
            backend.tick(world, 0.05 * (i + 1))
        later = [b for b in _bikes(backend) if b not in before]
        assert later
        assert all(tc.vehicle.getDecel(b) == pytest.approx(4.0) for b in later)
        assert backend.bicycle_parameters.decel == Spread(4.0)
    finally:
        backend.close()
    # The next run starts from its config.
    assert backend.bicycle_parameters.decel == Spread(2.0)


@needs_sumo
def test_a_vehicle_parked_under_the_map_joins_sumo_when_it_comes_up(
    crossroads: Path, tmp_path: Path
) -> None:
    """Spawned hidden, an NPC stays out of SUMO -- its speed and route kept
    for later -- until a spawn brings it up; then SUMO drives it from there."""
    backend, world, _npc, _ego = _prepared(crossroads, tmp_path)
    hidden = _FakeActor(_carla_transform(-60.0, 5.25, 0.0), "late")
    hidden.transform.location.z = -500.0
    entity = _Entity(hidden, "late")
    try:
        backend.adopt(entity)
        backend._parked[hidden.id] = (hidden, entity)
        backend.set_desired_speed(entity, world, 18.0)
        backend.tick(world, 0.0)
        assert backend._id_of(entity) is None
        assert backend._waiting_speeds[id(entity)] == pytest.approx(5.0)
        hidden.transform = _carla_transform(-60.0, 5.25, 0.0)
        for i in range(60):
            backend.tick(world, (i + 1) * 0.05)
        sumo_id = backend._id_of(entity)
        assert sumo_id is not None and sumo_id in backend._traci.vehicle.getIDList()
        assert hidden.transform.location.x > -60.0 + 1.0
        assert backend._traci.vehicle.getSpeed(sumo_id) == pytest.approx(5.0, abs=0.3)
    finally:
        backend.close()


@needs_sumo
def test_a_published_vehicle_that_leaves_the_world_leaves_sumo(
    crossroads: Path, tmp_path: Path
) -> None:
    """The ego, published, destroyed (a despawn): out of SUMO, and the run
    goes on."""
    backend, world, _npc, ego = _prepared(crossroads, tmp_path)
    ego_id = next(iter(backend._external))
    try:
        backend.tick(world, 0.0)
        ego.transform.location.z = -500.0
        backend.tick(world, 0.05)
        assert ego_id not in backend._external
        assert ego_id not in backend._traci.vehicle.getIDList()
        assert ego.id in backend._parked  # back when it comes up
    finally:
        backend.close()


@needs_sumo
def test_a_vehicle_stopping_at_its_goal_stands_there(
    crossroads: Path, tmp_path: Path
) -> None:
    """finish_at stops the NPC with its middle at the last point, on the lane
    of its route there, and it stays standing."""
    import typesafe_carla.carla as carla

    backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    sumo_id, entity = next(iter(backend._driven.items()))
    try:
        backend.tick(world, 0.0)
        points = [carla.Location(x, 5.25, 0.0) for x in (-44.0, -42.0, -40.0)]
        assert backend.finish_at(entity, world, points)
        for i in range(400):
            backend.tick(world, (i + 1) * 0.05)
        assert npc.transform.location.x == pytest.approx(-40.0, abs=1.0)
        assert backend._traci.vehicle.getSpeed(sumo_id) == pytest.approx(0.0, abs=0.05)
    finally:
        backend.close()


@needs_sumo
def test_a_routes_points_are_matched_to_the_lanes_running_their_way(
    crossroads: Path, tmp_path: Path
) -> None:
    """A point between the two carriageways of arm 0 is on the eastbound
    lanes heading east and on the westbound heading west; a route's edges are
    kept once each, in order."""
    backend, _world, _npc, _ego = _prepared(crossroads, tmp_path)
    try:
        offset = backend._offset
        x, y = -60.0 + offset[0], 0.0 + offset[1]
        east = backend._lane_at(x, y, 0.0, "passenger")
        west = backend._lane_at(x, y, 180.0, "passenger")
        assert east is not None and west is not None
        assert east[0].getEdge().getID() != west[0].getEdge().getID()
        into = east[0].getEdge()
        out = next(iter(into.getOutgoing()))
        joined = backend._join_edges(
            [into.getID(), into.getID(), out.getID()], "passenger"
        )
        assert joined == [into.getID(), out.getID()]
    finally:
        backend.close()


@needs_sumo
def test_a_car_sumo_drives_stops_for_what_stands_in_its_way_in_carla(
    crossroads: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A car standing 6 m ahead of the NPC in CARLA -- one SUMO does not see
    there -- has it braked, and its SUMO vehicle held back with it; gone, both
    go on."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.utils import clearance

    controls: list[Any] = []

    def apply_vehicle_control(actor_id: int, control: Any) -> SimpleNamespace:
        controls.append(control)
        return SimpleNamespace(actor_id=actor_id, control=control)

    monkeypatch.setattr(carla.command, "ApplyVehicleControl", apply_vehicle_control)
    client = _BatchClient()
    _CLIENT["client"] = client
    try:
        backend, world, npc, _ego = _prepared(
            crossroads, tmp_path, vehicle_control="physics"
        )
    finally:
        _CLIENT.clear()
    npc.get_world = lambda: world  # type: ignore[attr-defined]
    standing: list[Any] = []

    def in_way(world_: Any, self_id: int, x: float, y: float, *rest: Any) -> Any:
        if not standing or rest[1] != "vehicle":
            return None
        found = clearance.Obstacle(7, standing[0], 5.25, 2.5, 0.0)
        found.ahead = standing[0] - x
        return found if 0.0 < found.ahead <= rest[2] + 2.5 else None

    monkeypatch.setattr(clearance, "first_box_in_way", in_way)
    sumo_id = next(iter(backend._driven))
    try:
        for i in range(40):
            backend.tick(world, i * 0.05)
        before = backend._traci.vehicle.getSpeed(sumo_id)
        standing.append(npc.transform.location.x + 6.0)
        controls.clear()
        backend.tick(world, 2.0)
        assert controls[-1].brake > 0.0 and controls[-1].throttle == 0.0
        assert sumo_id in backend._held_back
        # SUMO slows it from its next step on.
        backend.tick(world, 2.05)
        assert backend._traci.vehicle.getSpeed(sumo_id) < before
        standing.clear()
        backend.tick(world, 2.1)
        assert sumo_id not in backend._held_back
    finally:
        backend.close()


@needs_sumo
def test_a_vehicle_stopping_at_its_goal_is_driven_there_in_carla(
    crossroads: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """finish_at hands the NPC from SUMO to a final approach in CARLA, along
    the points given (off its lane's middle as they may be), published into
    SUMO meanwhile; told no more speeds, it is not SUMO's to drive again."""
    import typesafe_carla.carla as carla

    from autoware_carla_scenario.traffic import final_approach

    stepped: list[float] = []

    class _Approach:
        def __init__(
            self,
            actor: Any,
            start: Any,
            points: Any,
            stop: bool = True,
            speed: float = 4.0,
        ) -> None:
            self.points = points
            self.length = 4.0

        def step(self, dt: float, elapsed: float) -> None:
            stepped.append(elapsed)

    monkeypatch.setattr(final_approach, "FinalApproach", _Approach)
    backend, world, npc, _ego = _prepared(crossroads, tmp_path)
    sumo_id, entity = next(iter(backend._driven.items()))
    try:
        backend.tick(world, 0.0)
        points = [carla.Location(-76.0, 6.0, 0.0), carla.Location(-74.0, 6.2, 0.0)]
        assert backend.finish_at(entity, world, points)
        assert sumo_id not in backend._driven
        assert backend._external[sumo_id] is npc
        backend.set_desired_speed(entity, world, 0.0)  # said nothing of it
        backend.tick(world, 0.05)
        backend.tick(world, 0.1)
        assert stepped == [0.05, 0.1]
        assert sumo_id in backend._traci.vehicle.getIDList()
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Limits of 0 and the options' bounds (#959)
# ---------------------------------------------------------------------------


class _PositionsVehicle:
    """TraCI's vehicle domain, as far as ``_nearest`` and ``_ambient_ids``
    read it."""

    def __init__(self, at: dict[str, tuple[float, float]], bikes: set[str]) -> None:
        self._at = at
        self._bikes = bikes

    def getIDList(self) -> list[str]:  # noqa: N802 - TraCI's name
        return list(self._at)

    def getPosition(self, sumo_id: str) -> tuple[float, float]:  # noqa: N802
        return self._at[sumo_id]

    def getVehicleClass(self, sumo_id: str) -> str:  # noqa: N802
        return "bicycle" if sumo_id in self._bikes else "passenger"


def _measuring(config: Any = None, bikes: int = 0) -> Any:
    """A backend, never prepared, whose SUMO holds five cars 10 m apart east
    of the origin (and *bikes* bicycles 5 m further on each)."""
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(config or SumoBackendConfig())
    at = {f"car{i}": (10.0 * i, 0.0) for i in range(1, 6)}
    at.update({f"bike{i}": (10.0 * i + 5.0, 0.0) for i in range(1, bikes + 1)})
    backend._traci = SimpleNamespace(
        vehicle=_PositionsVehicle(at, {i for i in at if i.startswith("bike")})
    )
    return backend


_CARS = [f"car{i}" for i in range(1, 6)]
_ORIGIN = Pose2D(0.0, 0.0, 0.0, 90.0)


def test_a_limit_of_zero_keeps_no_vehicle() -> None:
    backend = _measuring()
    assert backend._nearest(_CARS, _ORIGIN, None, 0) == []
    assert backend._nearest(_CARS, _ORIGIN, 25.0, 0) == []
    # With no ego to measure from yet, still none: 0 is not "no limit".
    assert backend._nearest(_CARS, None, None, 0) == []
    assert backend._nearest(_CARS, None, 25.0, 0) == []


def test_a_radius_of_zero_keeps_no_vehicle() -> None:
    backend = _measuring()
    assert backend._nearest(_CARS, _ORIGIN, 0.0, None) == []
    assert backend._nearest(_CARS, _ORIGIN, 0.0, 3) == []
    assert backend._nearest(_CARS, None, 0.0, None) == []


def test_none_is_no_limit_and_no_radius() -> None:
    backend = _measuring()
    shuffled = ["car3", "car5", "car1", "car4", "car2"]
    assert backend._nearest(shuffled, _ORIGIN, None, None) == shuffled
    assert backend._nearest(shuffled, _ORIGIN, 25.0, None) == ["car1", "car2"]
    assert backend._nearest(shuffled, _ORIGIN, None, 2) == ["car1", "car2"]
    assert backend._nearest(shuffled, _ORIGIN, 35.0, 2) == ["car1", "car2"]
    # No ego yet: the first ones SUMO lists, up to the limit.
    assert backend._nearest(shuffled, None, None, 2) == ["car3", "car5"]
    assert backend._nearest(shuffled, None, None, None) == shuffled


def _ambient_ids(config: SumoBackendConfig) -> list[str]:
    backend = _measuring(config, bikes=3)
    backend._carla_pose_to_sumo = lambda _actor: _ORIGIN
    backend._ego_actor = object()
    return backend._ambient_ids()


def test_each_mirror_limit_of_zero_mirrors_none_of_its_kind() -> None:
    every = _CARS + ["bike1", "bike2", "bike3"]
    assert sorted(_ambient_ids(SumoBackendConfig())) == sorted(every)
    assert _ambient_ids(SumoBackendConfig(max_vehicles=0)) == [
        "bike1",
        "bike2",
        "bike3",
    ]
    assert (
        _ambient_ids(SumoBackendConfig(bicycles=BicycleConfig(max_bicycles=0))) == _CARS
    )
    assert (
        _ambient_ids(
            SumoBackendConfig(max_vehicles=0, bicycles=BicycleConfig(max_bicycles=0))
        )
        == []
    )
    # The bicycles' radius is the vehicles' when it is null, and its own else.
    assert _ambient_ids(SumoBackendConfig(spawn_radius_m=20.0)) == [
        "car1",
        "car2",
        "bike1",
    ]
    assert _ambient_ids(
        SumoBackendConfig(spawn_radius_m=20.0, bicycles=BicycleConfig(radius_m=30.0))
    ) == ["car1", "car2", "bike1", "bike2"]


def test_the_mirror_limits_take_whole_numbers_from_zero() -> None:
    assert SumoBackendConfig.from_mapping({"max_vehicles": 0}).max_vehicles == 0
    assert BicycleConfig.from_mapping({"max_bicycles": 0}).max_bicycles == 0
    assert PedestrianConfig.from_mapping({"max_persons": 0}).max_persons == 0
    for bad in (-1, -2, 2.5, 3.0, True, "3"):
        with pytest.raises(ValueError, match="^max_vehicles must be an integer"):
            SumoBackendConfig.from_mapping({"max_vehicles": bad})
        with pytest.raises(
            ValueError, match="^bicycles.max_bicycles must be an integer"
        ):
            SumoBackendConfig.from_mapping({"bicycles": {"max_bicycles": bad}})
        with pytest.raises(
            ValueError, match="^pedestrians.max_persons must be an integer"
        ):
            SumoBackendConfig.from_mapping({"pedestrians": {"max_persons": bad}})


def test_the_radii_must_be_positive_and_finite() -> None:
    assert (
        SumoBackendConfig.from_mapping({"spawn_radius_m": None}).spawn_radius_m is None
    )
    assert SumoBackendConfig(spawn_radius_m=50).spawn_radius_m == 50
    for bad in (0.0, -5.0, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="^spawn_radius_m must be positive"):
            SumoBackendConfig.from_mapping({"spawn_radius_m": bad})
        with pytest.raises(ValueError, match="^bicycles.radius_m must be positive"):
            BicycleConfig(radius_m=bad)
        with pytest.raises(ValueError, match="^pedestrians.radius_m must be positive"):
            PedestrianConfig(radius_m=bad)
    for bad in (0.0, -5.0, math.nan, -math.inf):
        with pytest.raises(
            ValueError, match="^bicycles.max_distance_m must be positive"
        ):
            BicycleConfig(max_distance_m=bad)
        with pytest.raises(
            ValueError, match="^pedestrians.max_distance_m must be positive"
        ):
            PedestrianConfig(max_distance_m=bad)
    # inf is no limit on how far a walk or a ride goes, as before.
    assert BicycleConfig(max_distance_m=math.inf).max_distance_m == math.inf
    assert PedestrianConfig(max_distance_m=math.inf).max_distance_m == math.inf


def test_the_options_refuse_negative_and_non_finite_numbers() -> None:
    positive = (
        "curve_lateral_acceleration",
        "lateral_resolution_m",
        "step_length_s",
    )
    not_negative = ("feedback_distance_m", "warmup_s")
    for bad in (-1.0, math.nan, math.inf):
        for key in positive:
            with pytest.raises(ValueError, match=f"^{key} must be positive and finite"):
                SumoBackendConfig.from_mapping({key: bad})
        with pytest.raises(ValueError, match="^lane_change_duration_s must be"):
            SumoBackendConfig.from_mapping(
                {"lane_change_duration_s": bad, "lateral_resolution_m": None}
            )
        for key in not_negative:
            with pytest.raises(ValueError, match=f"^{key} must be finite and not"):
                SumoBackendConfig.from_mapping({key: bad})
    for key in positive:
        with pytest.raises(ValueError, match=f"^{key} must be positive"):
            SumoBackendConfig.from_mapping({key: 0.0})
    # resync_distance_m: inf never puts a car back, as before; NaN, 0 and
    # below are refused.
    assert SumoBackendConfig(resync_distance_m=math.inf).resync_distance_m == math.inf
    for bad in (0.0, -1.0, math.nan, -math.inf):
        with pytest.raises(ValueError, match="^resync_distance_m must be positive"):
            SumoBackendConfig.from_mapping({"resync_distance_m": bad})
    # 0 is meaningful for these: never correct SUMO, and no warm-up.
    zero = SumoBackendConfig(feedback_distance_m=0.0, warmup_s=0.0)
    assert zero.feedback_distance_m == 0.0 and zero.warmup_s == 0.0


def test_the_tm_port_must_be_a_port() -> None:
    assert SumoBackendConfig.from_mapping({"tm_port": 8200}).tm_port == 8200
    for bad in (0, -1, 65536, 8000.0, True, "8000"):
        with pytest.raises(ValueError, match="^tm_port must be an integer"):
            SumoBackendConfig.from_mapping({"tm_port": bad})


def test_the_generated_demand_refuses_negative_and_non_finite_numbers() -> None:
    groups = (
        ("ambient", "vehicles_per_hour", AmbientTrafficConfig),
        ("pedestrians", "persons_per_hour", PedestrianConfig),
        ("bicycles", "bicycles_per_hour", BicycleConfig),
    )
    for group, rate, cls in groups:
        # No demand per hour is allowed: as good as none generated.
        assert cls.from_mapping({rate: 0}).period() > 1e9
        for bad in (-1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match=f"^{group}.{rate} must be finite"):
                cls.from_mapping({rate: bad})
        for bad in (0.0, -1.0, math.nan, math.inf):
            for key in ("period_s", "end_s"):
                with pytest.raises(
                    ValueError, match=f"^{group}.{key} must be positive"
                ):
                    SumoBackendConfig.from_mapping({group: {key: bad}})
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="^ambient.fringe_factor must be finite"):
            AmbientTrafficConfig(fringe_factor=bad)
        with pytest.raises(ValueError, match="^bicycles.min_distance_m must be finite"):
            BicycleConfig(min_distance_m=bad)


@needs_sumo
def test_max_vehicles_zero_mirrors_no_ambient_vehicle(
    crossroads: Path, tmp_path: Path
) -> None:
    backend, world, _npc, ego = _prepared(
        crossroads, tmp_path, period_s=1.0, max_vehicles=0
    )
    try:
        for i in range(200):
            backend.tick(world, i * 0.05)
        assert backend._ambient == {}
        # SUMO has its traffic all the same; none of it reaches CARLA.
        assert len(backend._traci.vehicle.getIDList()) > 2
        assert not any(
            a.attributes["role_name"].startswith("sumo:") for a in world.spawned
        )
    finally:
        backend.close()


@needs_sumo
def test_max_bicycles_zero_mirrors_no_bicycle(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _ego = _riding(crossroads_with_sidewalks, tmp_path, max_bicycles=0)
    try:
        for i in range(100):
            backend.tick(world, i * 0.05)
        assert _bikes(backend)
        assert _bicycle_mirrors(world) == []
    finally:
        backend.close()


@needs_sumo
def test_max_persons_zero_mirrors_no_pedestrian(
    crossroads_with_sidewalks: Path, tmp_path: Path
) -> None:
    backend, world, _ego = _walking(
        crossroads_with_sidewalks, tmp_path, warmup_s=60.0, max_persons=0
    )
    try:
        for i in range(20):
            backend.tick(world, i * 0.05)
        assert backend._traci.person.getIDCount() > 0
        assert backend._persons == {}
        assert _mirrors(world) == []
    finally:
        backend.close()


class _WalkerLibrary:
    """A blueprint library with some walkers, as a CARLA build has them."""

    def __init__(self, walkers: list[str]) -> None:
        self.walkers = walkers

    def find(self, name: str) -> _Blueprint:
        if name not in self.walkers:
            raise IndexError(
                f"tsc_blueprint_library_find: no blueprint with id '{name}'"
            )
        return _Blueprint(name)

    def filter(self, pattern: str) -> list[_Blueprint]:
        return [_Blueprint(w) for w in self.walkers if fnmatch.fnmatch(w, pattern)]


def _with_walker_blueprint(name: str) -> Any:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend.__new__(SumoTrafficBackend)
    backend._config = SimpleNamespace(  # type: ignore[assignment]
        pedestrians=SimpleNamespace(blueprint=name)
    )
    backend._walker_type = None
    return backend


def test_a_walker_blueprint_the_build_lacks_falls_back_to_one_it_has(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # CARLA 0.10 has none of 0.9's walker.pedestrian.0001 (#449): SUMO's
    # persons are mirrored as a walker it has, said once.
    library = _WalkerLibrary(["walker.pedestrian.0030", "walker.pedestrian.0015"])
    backend = _with_walker_blueprint("walker.pedestrian.0001")
    with caplog.at_level(logging.WARNING):
        assert backend._walker_blueprint(library) == "walker.pedestrian.0015"
        assert backend._walker_blueprint(library) == "walker.pedestrian.0015"
    assert sum("no walker blueprint" in r.getMessage() for r in caplog.records) == 1
    assert (
        _with_walker_blueprint("walker.pedestrian.0030")._walker_blueprint(library)
        == "walker.pedestrian.0030"
    )
    with pytest.raises(RuntimeError, match="no walker.pedestrian"):
        _with_walker_blueprint("walker.pedestrian.0001")._walker_blueprint(
            _WalkerLibrary([])
        )


def test_the_default_walker_blueprints_are_ones_carla_0_10_has() -> None:
    from autoware_carla_scenario import PedestrianEntityConfig
    from autoware_carla_scenario.traffic.sumo.config import PedestrianConfig

    assert PedestrianConfig().blueprint == "walker.pedestrian.0015"
    (walker_type,) = [
        f for f in dataclasses.fields(PedestrianEntityConfig) if f.name == "walker_type"
    ]
    assert walker_type.default == "walker.pedestrian.0015"
