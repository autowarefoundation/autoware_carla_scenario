"""The world reset between the cases of a batch: ``reload`` and ``soft`` (#282).

A reload loads the map again (~5 s a case on CARLA UE5).  A soft reset keeps
the episode and puts back what a run changed in it; it must leave nothing of
the previous case behind, and fall back to a reload whenever it cannot be sure
of that.  No CARLA here: the server is a fake that hands out worlds whose
actors, settings, weather, traffic lights and spectator the tests can read.
"""

from __future__ import annotations

import copy
import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional
from unittest.mock import MagicMock, patch

import pytest

from autoware_carla_scenario import ScenarioQueue, ScenarioRunner, ServerConfig
from autoware_carla_scenario.world_reset import (
    RESET_STRATEGIES,
    capture_baseline,
    check_world_reset,
    soft_reset_world,
)

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


# ---------------------------------------------------------------------------
# A fake CARLA server
# ---------------------------------------------------------------------------


class _Actor:
    def __init__(
        self,
        world: "_World",
        actor_id: int,
        type_id: str,
        parent: Optional["_Actor"] = None,
    ) -> None:
        self.world = world
        self.id = actor_id
        self.type_id = type_id
        self.parent = parent
        self.attributes: dict[str, str] = {}
        self.transform = "origin"
        #: A destroy the server refuses (returns False and the actor stays).
        self.stuck = False

    def destroy(self) -> bool:
        if self.stuck:
            return False
        self.world.events.append(("destroy", self.type_id))
        del self.world.actors[self.id]
        return True

    def get_transform(self) -> str:
        return self.transform

    def set_transform(self, transform: str) -> None:
        self.transform = transform


class _TrafficLight(_Actor):
    def __init__(self, world: "_World", actor_id: int) -> None:
        super().__init__(world, actor_id, "traffic.traffic_light")
        self.green = 10.0
        self.yellow = 3.0
        self.red = 2.0
        self.frozen = False

    def get_green_time(self) -> float:
        return self.green

    def get_yellow_time(self) -> float:
        return self.yellow

    def get_red_time(self) -> float:
        return self.red

    def set_green_time(self, value: float) -> None:
        self.green = value

    def set_yellow_time(self, value: float) -> None:
        self.yellow = value

    def set_red_time(self, value: float) -> None:
        self.red = value

    def is_frozen(self) -> bool:
        return self.frozen

    def freeze(self, frozen: bool) -> None:
        self.frozen = frozen


class _ActorList(list):
    def filter(self, pattern: str) -> "_ActorList":
        return _ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


class _World:
    """A freshly loaded world: the spectator, two traffic lights and a sign."""

    def __init__(self, server: "_Server", episode: int, map_name: str) -> None:
        self.server = server
        self.id = episode
        self.map_name = map_name
        self.events: list[tuple[str, str]] = []
        self.actors: dict[int, _Actor] = {}
        self.settings = SimpleNamespace(
            synchronous_mode=False, fixed_delta_seconds=None, substepping=True
        )
        self.weather = SimpleNamespace(cloudiness=0.0, fog_density=0.0)
        self.ticks = 0
        self._spectator = self._add(_Actor(self, 0, "spectator"))
        self._add(_TrafficLight(self, 0))
        self._add(_TrafficLight(self, 0))
        self._add(_Actor(self, 0, "traffic.stop"))

    def _add(self, actor: _Actor) -> _Actor:
        actor.id = self.server.next_id()
        self.actors[actor.id] = actor
        return actor

    def spawn(self, type_id: str, parent: Optional[_Actor] = None) -> _Actor:
        return self._add(_Actor(self, 0, type_id, parent))

    def get_actors(self) -> _ActorList:
        return _ActorList(self.actors.values())

    def lights(self) -> list[_TrafficLight]:
        return [a for a in self.actors.values() if isinstance(a, _TrafficLight)]

    def get_settings(self) -> SimpleNamespace:
        return copy.copy(self.settings)

    def apply_settings(self, settings: SimpleNamespace) -> None:
        self.events.append(("apply_settings", ""))
        self.settings = copy.copy(settings)

    def get_weather(self) -> SimpleNamespace:
        return copy.copy(self.weather)

    def set_weather(self, weather: SimpleNamespace) -> None:
        self.weather = copy.copy(weather)

    def get_spectator(self) -> _Actor:
        return self._spectator

    def reset_all_traffic_lights(self) -> None:
        self.events.append(("reset_all_traffic_lights", ""))

    def tick(self) -> int:
        self.ticks += 1
        return self.ticks

    def get_map(self) -> SimpleNamespace:
        return SimpleNamespace(name=f"Carla/Maps/{self.map_name}")


class _Server:
    """A CARLA client on a server other clients may load worlds on."""

    def __init__(self, maps: tuple[str, ...] = ("Town10HD_Opt", "Town01")) -> None:
        self.maps = maps
        self.loads: list[str] = []
        self.recorder_stops = 0
        self._ids = 0
        self._episode = 0
        self.world = _World(self, self._new_episode(), maps[0])

    def next_id(self) -> int:
        self._ids += 1
        return self._ids

    def _new_episode(self) -> int:
        self._episode += 1
        return self._episode

    def get_available_maps(self) -> list[str]:
        return [f"/Game/Carla/Maps/{name}" for name in self.maps]

    def load_world(self, name: str) -> _World:
        self.loads.append(name)
        self.foreign_load(name)
        return self.world

    def foreign_load(self, name: str) -> None:
        """Another client (or this one) loads *name*: a new episode."""
        self.world = _World(self, self._new_episode(), name)

    def get_world(self) -> _World:
        return self.world

    def stop_recorder(self) -> None:
        self.recorder_stops += 1


def _runner(server: _Server, **kwargs: object) -> ScenarioRunner:
    with patch("autoware_carla_scenario.scenario_runner.carla.Client"):
        runner = ScenarioRunner(MagicMock(), **kwargs)  # type: ignore[arg-type]
    runner._client = server  # type: ignore[assignment]
    return runner


def _play_a_case(world: _World) -> list[_Actor]:
    """What a case leaves in the world when its own cleanup misses it all."""
    ego = world.spawn("vehicle.lincoln.mkz")
    camera = world.spawn("sensor.camera.rgb", parent=ego)
    collision = world.spawn("sensor.other.collision", parent=ego)
    walker = world.spawn("walker.pedestrian.0001")
    controller = world.spawn("controller.ai.walker", parent=walker)
    prop = world.spawn("static.prop.trafficcone01")
    world.settings.synchronous_mode = True
    world.settings.fixed_delta_seconds = 0.05
    world.weather.cloudiness = 90.0
    world.weather.fog_density = 40.0
    for light in world.lights():
        light.frozen = True
        light.green = 99.0
    world.get_spectator().transform = "behind the ego"
    return [ego, camera, collision, walker, controller, prop]


@pytest.fixture()
def server() -> Iterator[_Server]:
    yield _Server()


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class TestReloadIsTheDefault:
    def test_the_default_reset_loads_the_map_again(self, server: _Server) -> None:
        runner = _runner(server)
        runner.load_map_by_name("Town10HD_Opt")
        _play_a_case(server.world)

        runner.reset_world("Case0")

        assert runner.world_reset == "reload"
        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]
        assert runner._baseline is None


class TestSoftResetOnTheSameMap:
    def test_no_reload_and_every_actor_of_the_case_destroyed(
        self, server: _Server
    ) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        world = server.world
        map_actors = set(world.actors)
        left = _play_a_case(world)

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt"]
        assert server.world is world
        assert runner._world is world
        assert set(world.actors) == map_actors
        assert all(actor.id not in world.actors for actor in left)

    def test_sensors_and_controllers_go_before_what_they_are_on(
        self, server: _Server
    ) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        _play_a_case(server.world)

        runner.reset_world("Case0")

        destroyed = [t for kind, t in server.world.events if kind == "destroy"]
        assert destroyed[:3] == [
            "sensor.camera.rgb",
            "sensor.other.collision",
            "controller.ai.walker",
        ]
        assert destroyed.index("walker.pedestrian.0001") < destroyed.index(
            "vehicle.lincoln.mkz"
        )

    def test_settings_weather_lights_spectator_and_recorder_are_put_back(
        self, server: _Server
    ) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        world = server.world
        _play_a_case(world)

        runner.reset_world("Case0")

        assert world.settings.synchronous_mode is False
        assert world.settings.fixed_delta_seconds is None
        assert world.weather.cloudiness == 0.0
        assert world.weather.fog_density == 0.0
        assert all(not light.frozen for light in world.lights())
        assert all(light.green == 10.0 for light in world.lights())
        assert ("reset_all_traffic_lights", "") in world.events
        assert world.get_spectator().transform == "origin"
        assert server.recorder_stops == 1
        # The settings go last: the world is ticked in step before them.
        assert world.events[-1] == ("apply_settings", "")

    def test_a_light_frozen_on_load_is_frozen_again(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        world = server.load_world("Town10HD_Opt")
        world.lights()[0].frozen = True
        server.loads.clear()
        runner._baseline = capture_baseline(world)  # type: ignore[arg-type]
        for light in world.lights():
            light.frozen = False

        runner.reset_world("Case0")

        assert server.loads == []
        assert [light.frozen for light in world.lights()] == [True, False]

    def test_soft_resets_follow_one_another_with_no_load(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")

        for case in range(5):
            _play_a_case(server.world)
            runner.reset_world(f"Case{case}")

        assert server.loads == ["Town10HD_Opt"]
        assert len(server.world.actors) == 4

    def test_a_map_actor_the_baseline_missed_is_kept(self, server: _Server) -> None:
        """A light the server listed only after the baseline was read is the
        map's, not a run's."""
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        late = server.world._add(_TrafficLight(server.world, 0))

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt"]
        assert late.id in server.world.actors

    def test_the_next_run_sets_up_on_the_soft_reset_world(
        self, server: _Server
    ) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        runner.reset_world("Case0")

        assert runner._world_for_next_run("Case1") is server.world
        assert server.loads == ["Town10HD_Opt"]


class TestTheReloadFallback:
    def test_a_different_map_is_reloaded(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        server.foreign_load("Town01")

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]
        assert server.world.map_name == "Town10HD_Opt"

    def test_the_same_map_loaded_by_someone_else_is_reloaded(
        self, server: _Server
    ) -> None:
        """Another episode of the same map holds another client's actors."""
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        server.foreign_load("Town10HD_Opt")

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]

    def test_a_failure_during_the_soft_reset_reloads(
        self, server: _Server, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        world = server.world
        _play_a_case(world)

        def broken(weather: object) -> None:
            raise RuntimeError("time-out of 10000ms while waiting for the simulator")

        world.set_weather = broken  # type: ignore[method-assign]

        with caplog.at_level("WARNING", logger="autoware_carla_scenario"):
            runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]
        assert server.world is not world
        assert any("soft reset failed" in r.getMessage() for r in caplog.records)
        # The reload read a new baseline: the next reset is soft again.
        runner.reset_world("Case1")
        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]

    def test_an_actor_that_survives_reloads(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        stuck = server.world.spawn("vehicle.tesla.model3")
        stuck.stuck = True

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]
        assert stuck.id not in server.world.actors

    def test_a_map_actor_gone_reloads(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        light = server.world.lights()[0]
        del server.world.actors[light.id]

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]

    def test_an_unreadable_baseline_reloads(self, server: _Server) -> None:
        with patch(
            "autoware_carla_scenario.scenario_runner.capture_baseline",
            side_effect=RuntimeError("no"),
        ):
            runner = _runner(server, world_reset="soft")
            runner.load_map_by_name("Town10HD_Opt")
        assert runner._baseline is None

        runner.reset_world("Case0")

        assert server.loads == ["Town10HD_Opt", "Town10HD_Opt"]
        assert runner._baseline is not None

    def test_a_failed_reload_leaves_no_baseline(self, server: _Server) -> None:
        runner = _runner(server, world_reset="soft")
        runner.load_map_by_name("Town10HD_Opt")
        server.foreign_load("Town01")
        server.load_world = MagicMock(side_effect=RuntimeError("std::exception"))  # type: ignore[method-assign]

        runner.reset_world("Case0")

        assert runner._baseline is None

    @pytest.mark.parametrize(
        ("reload_every", "expected_loads"),
        [(0, 0), (1, 6), (2, 3), (3, 2)],
    )
    def test_reload_every(
        self, server: _Server, reload_every: int, expected_loads: int
    ) -> None:
        runner = _runner(server, world_reset="soft", reload_every=reload_every)
        runner.load_map_by_name("Town10HD_Opt")

        for case in range(6):
            _play_a_case(server.world)
            runner.reset_world(f"Case{case}")

        assert len(server.loads) == 1 + expected_loads
        assert len(server.world.actors) == 4

    def test_a_runner_that_loaded_no_map_reloads_first_then_soft_resets(
        self, server: _Server
    ) -> None:
        runner = _runner(server, world_reset="soft")

        runner.reset_world("Case0")
        runner.reset_world("Case1")

        assert server.loads == ["Town10HD_Opt"]


# ---------------------------------------------------------------------------
# The soft reset itself
# ---------------------------------------------------------------------------


def test_the_soft_reset_refuses_another_episode(server: _Server) -> None:
    baseline = capture_baseline(server.world)  # type: ignore[arg-type]
    server.foreign_load("Town10HD_Opt")

    with pytest.raises(RuntimeError, match="not the episode"):
        soft_reset_world(server, server.world, baseline)  # type: ignore[arg-type]


def test_the_soft_reset_counts_what_it_destroyed(server: _Server) -> None:
    baseline = capture_baseline(server.world)  # type: ignore[arg-type]
    _play_a_case(server.world)

    assert soft_reset_world(server, server.world, baseline) == 6  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("strategy", RESET_STRATEGIES)
    def test_both_strategies_are_accepted(self, strategy: str) -> None:
        check_world_reset(strategy, 0)

    def test_an_unknown_strategy_is_refused_by_the_runner(self) -> None:
        with pytest.raises(ValueError, match="world_reset must be one of"):
            _runner(_Server(), world_reset="hard")

    def test_a_negative_reload_every_is_refused_by_the_runner(self) -> None:
        with pytest.raises(ValueError, match="reload_every must be >= 0"):
            _runner(_Server(), world_reset="soft", reload_every=-1)

    def test_the_queue_refuses_before_carla_starts(self) -> None:
        with pytest.raises(ValueError, match="world_reset must be one of"):
            ScenarioQueue(world_reset="Soft")
        with pytest.raises(ValueError, match="reload_every must be >= 0"):
            ScenarioQueue(reload_every=-2)

    def test_the_queue_hands_the_strategy_to_its_runner(self) -> None:
        with (
            patch("autoware_carla_scenario.scenario_queue.CarlaServerManager"),
            patch("autoware_carla_scenario.scenario_queue.ScenarioRunner") as runner,
        ):
            queue = ScenarioQueue(
                map_name="Town10HD_Opt", world_reset="soft", reload_every=4
            )
            queue.start()

        assert runner.call_args.kwargs["world_reset"] == "soft"
        assert runner.call_args.kwargs["reload_every"] == 4


class TestTheConfig:
    def test_server_config_defaults_to_reload(self) -> None:
        config = ServerConfig()
        assert config.world_reset == "reload"
        assert config.reload_every == 0

    def _composed(self, overrides: list[str]):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(_CONF.resolve()), version_base=None):
            return compose(config_name="config", overrides=overrides)

    def test_the_shipped_server_config_matches_the_defaults(self) -> None:
        from autoware_carla_scenario.examples.run import world_reset_settings

        cfg = self._composed([])
        assert cfg.server.world_reset == ServerConfig.world_reset
        assert cfg.server.reload_every == ServerConfig.reload_every
        assert world_reset_settings(cfg) == ("reload", 0)

    def test_soft_is_read_from_the_command_line(self) -> None:
        from autoware_carla_scenario.examples.run import world_reset_settings

        cfg = self._composed(["server.world_reset=soft", "server.reload_every=10"])
        assert world_reset_settings(cfg) == ("soft", 10)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ("server.world_reset=fast", "server.world_reset must be one of"),
            ("server.reload_every=-1", "server.reload_every must be >= 0"),
        ],
    )
    def test_a_bad_value_is_refused_naming_its_key(
        self, override: str, message: str
    ) -> None:
        from autoware_carla_scenario.examples.run import world_reset_settings

        cfg = self._composed([override])
        with pytest.raises(ValueError, match=message):
            world_reset_settings(cfg)

    def test_a_config_without_the_keys_reloads(self) -> None:
        from omegaconf import OmegaConf

        from autoware_carla_scenario.examples.run import world_reset_settings

        cfg = OmegaConf.create({"server": {"host": "localhost"}})
        assert world_reset_settings(cfg) == ("reload", 0)
