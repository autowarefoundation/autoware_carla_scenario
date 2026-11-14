"""Tests for ``traffic.backend=hybrid``: the TrafficManager near the ego, a
mesoscopic model -- SUMO's own, or the queue model -- far from it.

CARLA and its TrafficManager are faked; the roads are a real SUMO network (a
grid of two-lane roads 200 m apart, from ``netgenerate``), and with
``engine: sumo`` the mesoscopic model is a real ``sumo --mesosim``.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario.traffic import (
    SimulationFidelity,
    TrafficContext,
    TrafficParameterError,
    VehicleParameters,
    available_backends,
    build_backend,
)

_HAS_SUMO = all(
    importlib.util.find_spec(m) is not None for m in ("libsumo", "sumolib", "traci")
)
needs_sumo = pytest.mark.skipif(
    not _HAS_SUMO, reason="the `sumo` extra is not installed"
)

STEP = 0.1


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _TM:
    """A TrafficManager that takes every call and keeps the desired speeds."""

    def __init__(self) -> None:
        self.desired: dict[int, float] = {}

    def set_desired_speed(self, actor: Any, kmh: float) -> None:
        self.desired[actor.id] = kmh

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _Client:
    def __init__(self) -> None:
        self.tm = _TM()

    def get_trafficmanager(self, port: int) -> _TM:
        return self.tm


class _Actor:
    def __init__(
        self, world: "_World", actor_id: int, type_id: str, transform: Any
    ) -> None:
        self.world = world
        self.id = actor_id
        self.type_id = type_id
        self.attributes: dict[str, str] = {}
        self.transform = transform
        self.velocity = carla.Vector3D(0.0, 0.0, 0.0)
        self.bounding_box = SimpleNamespace(
            extent=carla.Vector3D(2.3, 1.0, 0.8), location=carla.Location(0.0, 0.0, 0.0)
        )
        self.is_alive = True
        self.autopilot = False

    def get_transform(self) -> Any:
        return self.transform

    def get_location(self) -> Any:
        return self.transform.location

    def get_velocity(self) -> Any:
        return self.velocity

    def set_target_velocity(self, velocity: Any) -> None:
        self.velocity = velocity

    def set_autopilot(self, on: bool, port: int = 0) -> None:
        self.autopilot = on

    def destroy(self) -> bool:
        self.is_alive = False
        self.world.actors.pop(self.id, None)
        return True


class _Blueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id
        self.role_name = ""

    def has_attribute(self, name: str) -> bool:
        return name == "role_name"

    def set_attribute(self, name: str, value: str) -> None:
        self.role_name = value


class _World:
    def __init__(self) -> None:
        self.id = 1
        self.actors: dict[int, _Actor] = {}
        self.next_id = 100
        self.blocked = False

    def get_actors(self) -> Any:
        actors = list(self.actors.values())
        return SimpleNamespace(filter=lambda pattern: actors)

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(filter=lambda pattern: [_Blueprint(pattern)])

    def try_spawn_actor(
        self, blueprint: _Blueprint, transform: Any
    ) -> Optional[_Actor]:
        if self.blocked:
            return None
        self.next_id += 1
        actor = _Actor(self, self.next_id, blueprint.id, transform)
        actor.attributes["role_name"] = blueprint.role_name
        self.actors[actor.id] = actor
        return actor

    def place(
        self, x: float, y: float, yaw: float = 0.0, type_id: str = "vehicle.ego"
    ) -> _Actor:
        """A vehicle that is not background traffic: a focus point."""
        self.next_id += 1
        actor = _Actor(
            self,
            self.next_id,
            type_id,
            carla.Transform(carla.Location(x, y, 0.0), carla.Rotation(yaw=yaw)),
        )
        self.actors[actor.id] = actor
        return actor


def _on_lane(world: Any, transform: Any) -> Any:
    """The faked map's nearest driving lane: right where it is asked."""
    return carla.Transform(
        carla.Location(transform.location.x, transform.location.y, 0.0),
        carla.Rotation(yaw=transform.rotation.yaw),
    )


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 3 x 3 grid of junctions 200 m apart: edges ``A0B0``, ``B0C0``..."""
    from autoware_carla_scenario.traffic.sumo.network import sumo_home

    net = tmp_path_factory.mktemp("hybrid") / "grid.net.xml"
    subprocess.run(
        [
            str(sumo_home() / "bin" / "netgenerate"),
            "--grid",
            "--grid.number",
            "3",
            "--grid.length",
            "200",
            "--default.lanenumber",
            "2",
            "--tls.guess",
            "true",
            "--tls.guess.threshold",
            "0",
            "-o",
            str(net),
        ],
        check=True,
        capture_output=True,
    )
    return net


@pytest.fixture(params=["sumo", "queue"])
def engine(request: pytest.FixtureRequest) -> str:
    return str(request.param)


class _Run:
    """A hybrid backend on the grid, prepared and started, and its clock."""

    def __init__(self, backend: Any, world: _World, client: _Client) -> None:
        self.backend = backend
        self.world = world
        self.client = client
        self.now = 0.0

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / STEP)):
            self.now = round(self.now + STEP, 6)
            self.backend.tick(self.world, self.now)

    def spawn(self, x: float, y: float, yaw: float, speed_kmh: float = 36.0) -> str:
        handle = self.backend.spawn_background(
            self.world,
            carla.Transform(carla.Location(x, y, 0.0), carla.Rotation(yaw=yaw)),
            speed_kmh=speed_kmh,
            blueprint="vehicle.tesla.model3",
        )
        assert handle is not None
        return str(handle)


@pytest.fixture
def make(grid: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    monkeypatch.setattr(tm_module, "_background_lane", _on_lane)
    runs: list[_Run] = []

    def build(**mesoscopic: Any) -> _Run:
        world, client = _World(), _Client()
        backend = build_backend(
            "hybrid",
            {
                "net_path": str(grid),
                "mesoscopic": {"update_interval_s": 0.5, **mesoscopic},
            },
        )
        backend.prepare(
            TrafficContext(
                client=client,  # type: ignore[arg-type]
                world=world,  # type: ignore[arg-type]
                fixed_delta_seconds=STEP,
                random_seed=3,
                output_dir=tmp_path,
                file_stem="Hybrid",
            )
        )
        run = _Run(backend, world, client)
        runs.append(run)
        return run

    yield build
    for run in runs:
        run.backend.close()


# The grid in CARLA's frame: SUMO's y turned over (netgenerate puts no
# offset).  Edge A0B0 runs east along SUMO's y = 0, CARLA's y = 0, its
# lanes south of it -- CARLA's y positive -- and
# A0A1 runs north along x = 0, CARLA's y going negative.  Its two lanes are
# 3.2 m wide: the inner one centred at y = 1.6, the rightmost at 4.8.
_EAST = 0.0
_NORTH = -90.0


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_the_backend_is_registered_and_takes_the_traffic_managers_parameters() -> None:
    assert "hybrid" in available_backends()
    backend = build_backend(
        "hybrid", {}, VehicleParameters.from_mapping({"ignore_lights": {"mean": 0.1}})
    )
    assert backend.name == "hybrid"
    with pytest.raises(TrafficParameterError, match="'hybrid'"):
        build_backend(
            "hybrid",
            {},
            VehicleParameters.from_mapping({"reaction_time_s": {"mean": 1.0}}),
        )


def test_its_options_are_checked() -> None:
    from autoware_carla_scenario.traffic.sumo.hybrid import HybridBackendConfig

    config = HybridBackendConfig.from_mapping({})
    assert config.mesoscopic.enabled and config.mesoscopic.engine == "sumo"
    assert config.port == 8100
    assert not HybridBackendConfig.from_mapping(
        {"mesoscopic": {"enabled": False}}
    ).mesoscopic.enabled
    assert HybridBackendConfig.from_mapping({"port": 8101}).port == 8101
    with pytest.raises(ValueError, match="scenario_vehicles is not supported"):
        HybridBackendConfig.from_mapping({"mesoscopic": {"scenario_vehicles": True}})
    with pytest.raises(ValueError, match="bogus"):
        HybridBackendConfig.from_mapping({"bogus": 1})
    with pytest.raises(ValueError, match="mesoscopic.engine"):
        HybridBackendConfig.from_mapping({"mesoscopic": {"engine": "vissim"}})
    with pytest.raises(ValueError, match="port"):
        HybridBackendConfig.from_mapping({"port": 0})


def test_the_hydra_group_builds_it() -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import conf as conf_package
    from autoware_carla_scenario.traffic import TrafficConfig

    conf_dir = str(Path(conf_package.__file__).parent.resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["traffic=hybrid"])
    container = OmegaConf.to_container(cfg.traffic, resolve=True)
    assert isinstance(container, dict)
    traffic = TrafficConfig.from_mapping({str(k): v for k, v in container.items()})
    backend = build_backend(traffic.backend, traffic.options, traffic.vehicles)
    assert backend.name == "hybrid"
    assert backend.describe()["mesoscopic"]["engine"] == "sumo"


@needs_sumo
def test_the_roads_locate_a_vehicle_and_place_it_back(grid: Path) -> None:
    import sumolib

    from autoware_carla_scenario.traffic.sumo.mesosim import MesoRoads

    roads = MesoRoads(sumolib.net.readNet(str(grid)))
    assert "A0B0" in roads.links and roads.links["A0B0"].lanes == 2
    # SUMO's frame: on A0B0, 50 m along, heading east (SUMO's 90 degrees).
    found = roads.locate(50.0, -1.6, 90.0)
    assert found is not None
    edge, offset = found
    assert edge == "A0B0" and offset == pytest.approx(50.0 - 10.0, abs=10.0)
    pose = roads.pose(edge, offset)
    assert pose.heading_deg == pytest.approx(90.0, abs=1.0)
    # Facing west there is the other carriageway, B0A0.
    found = roads.locate(50.0, 1.6, 270.0)
    assert found is not None and found[0] == "B0A0"
    # Against the traffic of both, or far off every road: nowhere.
    assert roads.locate(50.0, -1.6, 0.0) is None
    assert roads.locate(100.0, 100.0, 90.0) is None
    route = roads.route_on(["A0B0"], __import__("random").Random(1), edges=4)
    assert route[0] == "A0B0" and len(route) == 5
    net = sumolib.net.readNet(str(grid))
    for a, b in zip(route, route[1:]):
        assert net.getEdge(b) in net.getEdge(a).getOutgoing()


@needs_sumo
def test_a_far_vehicle_leaves_carla_and_comes_back_near(make: Any, engine: str) -> None:
    run = make(engine=engine, micro_radius_m=60.0, meso_radius_m=80.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)  # at A0, the grid's corner
    backend.start(world, skip_actor_ids={ego.id})
    near = run.spawn(30.0, 1.6, _EAST)
    far = run.spawn(150.0, 1.6, _EAST)
    first = backend.background_actors(world)[far]
    assert first.autopilot
    run.run(1.0)

    # The far one is out of CARLA, still a background vehicle where the model
    # has it; the near one is not.
    assert set(backend.background_actors(world)) == {near}
    assert not first.is_alive and first.id not in world.actors
    where = backend.background_vehicles(world)
    assert set(where) == {near, far}
    x, y, z = where[far]
    # On its road's rightmost lane (the model has no lanes), 1.6 m or 4.8 m
    # south of the centre line.
    assert z is None and x == pytest.approx(150.0, abs=25.0) and 0.0 < y < 6.4
    census = backend.fidelity_census()
    assert census == {
        SimulationFidelity.MICROSCOPIC.value: 1,
        SimulationFidelity.MESOSCOPIC.value: 1,
    }
    assert backend.describe()["mesoscopic"]["to_mesoscopic"] == 1

    # The ego drives up to it: it comes back, as itself, on autopilot, at
    # the speed it was given.
    ego.transform = carla.Transform(
        carla.Location(x - 20.0, 1.6, 0.0), carla.Rotation(yaw=_EAST)
    )
    run.run(1.0)
    back = backend.background_actors(world)
    # (The near one, left behind, has gone the other way.)
    assert far in back and near not in back
    again = back[far]
    assert again is not first and again.is_alive and again.autopilot
    assert again.attributes["role_name"] == far
    assert again.type_id == "vehicle.tesla.model3"
    assert run.client.tm.desired[again.id] == pytest.approx(36.0)
    # Where the model had it, on its road, facing along it.
    assert 0.0 < again.transform.location.y < 6.4
    assert again.transform.rotation.yaw == pytest.approx(_EAST, abs=5.0)
    assert backend.describe()["mesoscopic"]["to_microscopic"] == 1


@needs_sumo
def test_out_of_carla_it_drives_on_along_the_roads(make: Any, engine: str) -> None:
    run = make(engine=engine, micro_radius_m=30.0, meso_radius_m=40.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, -400.0, _NORTH)  # by C2 at the far corner
    backend.start(world, skip_actor_ids={ego.id})
    handle = run.spawn(20.0, 1.6, _EAST)
    run.run(1.0)
    assert handle not in backend.background_actors(world)
    start = backend.background_vehicles(world)[handle]
    run.run(15.0)
    later = backend.background_vehicles(world)[handle]
    # Some 10 m/s for 15 s: well along, on whichever road it took.
    assert math.hypot(later[0] - start[0], later[1] - start[1]) > 50.0


@needs_sumo
def test_a_vehicle_with_no_room_in_carla_waits_out_of_it(make: Any) -> None:
    run = make(engine="queue", micro_radius_m=60.0, meso_radius_m=80.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)
    backend.start(world, skip_actor_ids={ego.id})
    handle = run.spawn(150.0, 1.6, _EAST, speed_kmh=5.0)
    run.run(1.0)
    assert handle not in backend.background_actors(world)
    world.blocked = True
    ego.transform = carla.Transform(carla.Location(130.0, 1.6, 0.0), carla.Rotation())
    run.run(1.0)
    assert handle not in backend.background_actors(world)
    assert handle in backend.background_vehicles(world)
    world.blocked = False
    run.run(1.0)
    assert handle in backend.background_actors(world)


@needs_sumo
def test_removing_a_vehicle_out_of_carla_removes_it(make: Any, engine: str) -> None:
    run = make(engine=engine, micro_radius_m=60.0, meso_radius_m=80.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)
    backend.start(world, skip_actor_ids={ego.id})
    handle = run.spawn(150.0, 1.6, _EAST)
    run.run(1.0)
    assert handle in backend.background_vehicles(world)
    backend.remove_background(world, handle)
    assert handle not in backend.background_vehicles(world)
    assert backend.fidelity_census()[SimulationFidelity.MESOSCOPIC.value] == 0
    backend.remove_background(world, handle)  # twice is harmless


@needs_sumo
def test_a_budget_keeps_only_the_nearest_in_carla(make: Any) -> None:
    run = make(engine="queue", micro_radius_m=500.0, max_microscopic=1)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)
    backend.start(world, skip_actor_ids={ego.id})
    near = run.spawn(30.0, 1.6, _EAST)
    run.spawn(120.0, 1.6, _EAST)
    run.run(1.0)
    assert set(backend.background_actors(world)) == {near}


@needs_sumo
def test_with_no_other_vehicle_nothing_changes_level(make: Any) -> None:
    run = make(engine="queue", micro_radius_m=30.0)
    backend, world = run.backend, run.world
    backend.start(world)
    handle = run.spawn(150.0, 1.6, _EAST)
    run.run(2.0)
    assert handle in backend.background_actors(world)


@needs_sumo
def test_mesoscopic_off_is_the_traffic_manager_alone(make: Any) -> None:
    run = make(enabled=False, micro_radius_m=30.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)
    backend.start(world, skip_actor_ids={ego.id})
    handle = run.spawn(150.0, 1.6, _EAST)
    run.run(2.0)
    assert handle in backend.background_actors(world)
    assert "mesoscopic" not in backend.describe()


@needs_sumo
def test_closing_forgets_the_vehicles_out_of_carla_and_stops_sumo(make: Any) -> None:
    run = make(engine="sumo", micro_radius_m=60.0, meso_radius_m=80.0)
    backend, world = run.backend, run.world
    ego = world.place(0.0, 1.6, _EAST)
    backend.start(world, skip_actor_ids={ego.id})
    run.spawn(150.0, 1.6, _EAST)
    run.run(1.0)
    process = backend._meso._process
    assert process.poll() is None
    backend.close()
    assert process.poll() is not None
    assert backend.background_vehicles(world) == {}
    backend.close()  # twice is harmless


def test_a_missing_network_is_refused(tmp_path: Path) -> None:
    from autoware_carla_scenario.traffic import TrafficBackendUnavailable

    if not _HAS_SUMO:
        pytest.skip("the `sumo` extra is not installed")
    backend = build_backend("hybrid", {"net_path": str(tmp_path / "none.net.xml")})
    with pytest.raises(TrafficBackendUnavailable, match="does not exist"):
        backend.prepare(TrafficContext(client=_Client(), fixed_delta_seconds=STEP))  # type: ignore[arg-type]
