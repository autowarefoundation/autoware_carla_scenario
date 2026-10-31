"""What a traffic backend is, and what the default one does.

The manoeuvres themselves are pinned by ``test_driven``, which drives them
through an entity exactly as an action does.  These tests are about the other
half: the lifecycle the runner calls, the one-authority-per-vehicle rule, and
the promise every backend makes -- that an intent it cannot carry out is a
warning, not an exception thrown into the tick loop.

CARLA is faked.  Every call under test is a single RPC against a handle, so a
live server would only make the test slower.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional, Tuple

import pytest

from autoware_carla_scenario.entity import (
    SpawnPointIndex,
    VehicleEntity,
    VehicleEntityConfig,
)
from autoware_carla_scenario.traffic.driven import BackendDriven
from autoware_carla_scenario.traffic import (
    NullTrafficBackend,
    TrafficBackend,
    TrafficContext,
    build_backend,
)
from autoware_carla_scenario.traffic.base import LaneChangeDirection, TurnDirection
from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig
from autoware_carla_scenario.traffic.traffic_manager import TrafficManagerBackend


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Actor:
    def __init__(self, actor_id: int, speed_limit: float = 50.0) -> None:
        self.id = actor_id
        self.attributes: dict = {}
        self.autopilot: List[Tuple[bool, int]] = []
        self.speed_limit = speed_limit
        self.is_alive = True
        self.speed = 0.0
        self.location: Any = None
        self.controls: List[Any] = []
        self.destroyed = False

    def get_speed_limit(self) -> float:
        return self.speed_limit

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot.append((enabled, port))

    def get_velocity(self) -> Any:
        if self.destroyed:
            raise RuntimeError("destroyed")
        return SimpleNamespace(x=self.speed, y=0.0, z=0.0)

    def get_location(self) -> Any:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        return self.location or carla.Location(0.0, 0.0, 0.0)

    def apply_control(self, control: Any) -> None:
        self.controls.append(control)


class _ActorList:
    def __init__(self, actors: List[_Actor]) -> None:
        self._actors = actors
        self.filters: List[str] = []

    def filter(self, pattern: str) -> List[_Actor]:
        self.filters.append(pattern)
        return list(self._actors)


class _World:
    def __init__(self, actors: Optional[List[_Actor]] = None) -> None:
        self._actors = _ActorList(actors or [])

    def get_actors(self) -> _ActorList:
        return self._actors

    def get_map(self) -> Any:
        """A map no actor stands on: no road ahead to search."""
        return SimpleNamespace(get_waypoint=lambda location: None)


class _TrafficManager:
    """A TrafficManager that keeps a vehicle's speed as LibCarla's
    ``Parameters.cpp`` does: a percentage under the limit or an exact speed,
    either dropping the other, and the global percentage (LibCarla's default
    0) for a vehicle with neither."""

    def __init__(self) -> None:
        self.synchronous: List[bool] = []
        self.seeds: List[int] = []
        self.shut_downs = 0
        #: actor id -> every percentage under the limit it was given.
        self.speed_differences: dict = {}
        #: actor id -> every exact speed (km/h) it was given.
        self.desired_speeds: dict = {}
        #: Every global percentage set.
        self.global_differences: List[float] = []
        self.paths: List[Tuple[int, List[Any]]] = []
        self._percentage: dict = {}
        self._exact: dict = {}
        self._global = 0.0

    def vehicle_percentage_speed_difference(self, actor: Any, percent: float) -> None:
        self.speed_differences.setdefault(actor.id, []).append(percent)
        self._percentage[actor.id] = min(100.0, percent)
        self._exact.pop(actor.id, None)

    def set_desired_speed(self, actor: Any, speed: float) -> None:
        self.desired_speeds.setdefault(actor.id, []).append(speed)
        self._exact[actor.id] = max(0.0, speed)
        self._percentage.pop(actor.id, None)

    def global_percentage_speed_difference(self, percent: float) -> None:
        self.global_differences.append(percent)
        self._global = min(100.0, percent)

    def target_kmh(self, actor: Any) -> float:
        """The speed it drives at (``GetVehicleTargetVelocity``)."""
        if actor.id in self._percentage:
            percent = self._percentage[actor.id]
        elif actor.id in self._exact:
            return float(self._exact[actor.id])
        else:
            percent = self._global
        return float(actor.speed_limit * (1.0 - percent / 100.0))

    def set_path(self, vehicle: Any, path: List[Any]) -> None:
        self.paths.append((vehicle.id, list(path)))

    def set_synchronous_mode(self, enabled: bool) -> None:
        self.synchronous.append(enabled)

    def set_random_device_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def shut_down(self) -> None:
        self.shut_downs += 1


class _Client:
    def __init__(self, tm: Optional[_TrafficManager] = None) -> None:
        self.tm = tm or _TrafficManager()
        self.ports: List[int] = []

    def get_trafficmanager(self, port: int) -> _TrafficManager:
        self.ports.append(port)
        return self.tm


def _context(client: Any, world: Any = None, seed: int = 7) -> TrafficContext:
    return TrafficContext(client=client, world=world or _World(), random_seed=seed)


# ---------------------------------------------------------------------------
# The contract every backend keeps
# ---------------------------------------------------------------------------


def _backends() -> List[TrafficBackend]:
    """One of each built-in backend, for the tests that hold for all of them."""
    return [NullTrafficBackend(), TrafficManagerBackend(client=_Client())]


@pytest.mark.parametrize("backend", _backends(), ids=lambda b: b.name)
class TestTheContract:
    """Promises the runner relies on, from whichever backend it was given."""

    def test_the_lifecycle_runs_end_to_end(self, backend: TrafficBackend) -> None:
        world = _World([_Actor(1)])
        backend.prepare(_context(_Client(), world))
        backend.start(world)
        backend.tick(world, 0.05)
        backend.close()

    def test_closing_twice_is_closing_once(self, backend: TrafficBackend) -> None:
        """Teardown runs whatever happened during the run, including twice."""
        backend.prepare(_context(_Client()))
        backend.close()
        backend.close()

    def test_closing_without_preparing_is_safe(self, backend: TrafficBackend) -> None:
        """A run that failed before prepare() still reaches the finally block."""
        backend.close()

    def test_an_intent_never_raises(self, backend: TrafficBackend) -> None:
        """A backend that cannot do something warns; it does not end the run.

        A scenario is perfectly valid with a manoeuvre its traffic model has no
        answer for, and an exception here would come out of the tick loop.
        """
        # Not a DrivenEntity: a stand-in with nothing a backend reads.
        entity: Any = object()
        world = _World()
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        backend.turn_at_junction(entity, world, TurnDirection.RIGHT)
        backend.follow_route(entity, world, [])
        assert backend.lane_change_finished(entity, world) is False

    def test_it_names_itself_for_the_result_record(
        self, backend: TrafficBackend
    ) -> None:
        assert backend.describe()["backend"] == backend.name


# ---------------------------------------------------------------------------
# The TrafficManager backend
# ---------------------------------------------------------------------------


class TestTrafficManagerPrepare:
    def test_the_manager_is_put_in_step_and_seeded(self) -> None:
        """Both, before anything is spawned: a seed set later decides nothing."""
        client = _Client()
        backend = TrafficManagerBackend()
        backend.prepare(_context(client, seed=42))
        assert client.tm.synchronous == [True]
        assert client.tm.seeds == [42]

    def test_its_default_speed_is_set_not_assumed(self) -> None:
        """LibCarla's own default, 0 % under the limit, whatever the build."""
        from autoware_carla_scenario.traffic.traffic_manager import (
            TM_DEFAULT_SPEED_DIFFERENCE_PCT,
        )

        client = _Client()
        TrafficManagerBackend().prepare(_context(client))
        assert TM_DEFAULT_SPEED_DIFFERENCE_PCT == 0.0
        assert client.tm.global_differences == [0.0]

    def test_the_configured_port_is_the_one_reached(self) -> None:
        client = _Client()
        backend = TrafficManagerBackend(TrafficManagerBackendConfig(port=8123))
        backend.prepare(_context(client))
        assert client.ports == [8123]

    def test_without_a_client_nothing_is_sent_and_nothing_raises(self) -> None:
        TrafficManagerBackend().prepare(TrafficContext())


class TestTrafficManagerStart:
    def test_every_vehicle_is_handed_over(self) -> None:
        actors = [_Actor(1), _Actor(2)]
        world = _World(actors)
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=_Client()
        )
        backend.start(world)
        assert [a.autopilot for a in actors] == [[(True, 8123)], [(True, 8123)]]

    def test_only_vehicles(self) -> None:
        """Walkers and sensors are nobody's to autopilot."""
        world = _World([_Actor(1)])
        TrafficManagerBackend(client=_Client()).start(world)
        assert world.get_actors().filters == ["vehicle.*"]

    def test_a_named_actor_is_left_alone(self) -> None:
        """One authority per vehicle: an ego under external control is skipped.

        Passed as data rather than assumed, so the rule holds for a backend
        that has never heard of an Autoware ego.
        """
        ego, npc = _Actor(1), _Actor(2)
        backend = TrafficManagerBackend(client=_Client())
        backend.start(_World([ego, npc]), skip_actor_ids={ego.id})
        assert ego.autopilot == []
        assert npc.autopilot == [(True, backend.port)]


def _stand_in(actor: Any, role_name: Optional[str] = "npc1") -> Any:
    """An entity as a backend reads it: its actor and role name, nothing else."""
    if role_name is None:
        return SimpleNamespace(actor=actor)
    return SimpleNamespace(actor=actor, role_name=role_name)


class TestTrafficManagerRelease:
    """An action that drives a vehicle itself takes it back from the manager."""

    def test_released_after_start_it_comes_off_autopilot(self) -> None:
        npc = _Actor(2)
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=_Client()
        )
        backend.start(_World([npc]))
        backend.release(_stand_in(npc), _World())
        assert npc.autopilot == [(True, 8123), (False, 8123)]

    def test_released_before_start_it_is_never_handed_over(self) -> None:
        npc, other = _Actor(2), _Actor(3)
        backend = TrafficManagerBackend(client=_Client())
        backend.release(_stand_in(npc), _World())
        backend.start(_World([npc, other]))
        assert npc.autopilot == []
        assert other.autopilot == [(True, backend.port)]

    def test_a_new_run_forgets_what_was_released(self) -> None:
        npc = _Actor(2)
        backend = TrafficManagerBackend(client=_Client())
        backend.release(_stand_in(npc), _World())
        backend.close()
        backend.start(_World([npc]))
        assert npc.autopilot == [(True, backend.port)]


class TestTrafficManagerResume:
    """An action done with a vehicle hands it back to the manager."""

    def test_resumed_after_start_it_goes_back_on_autopilot(self) -> None:
        npc = _Actor(2)
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=_Client()
        )
        backend.start(_World([npc]))
        backend.release(_stand_in(npc), _World())
        backend.resume(_stand_in(npc), _World())
        assert npc.autopilot == [(True, 8123), (False, 8123), (True, 8123)]
        # Saying it twice is harmless.
        backend.resume(_stand_in(npc), _World())
        assert len(npc.autopilot) == 3

    def test_resumed_before_start_it_is_handed_over_at_start(self) -> None:
        npc = _Actor(2)
        backend = TrafficManagerBackend(client=_Client())
        backend.release(_stand_in(npc), _World())
        backend.resume(_stand_in(npc), _World())
        backend.start(_World([npc]))
        assert npc.autopilot == [(True, backend.port)]

    def test_a_vehicle_driven_elsewhere_is_never_put_on_autopilot(self) -> None:
        """An ego skipped at start stays its stack's, released and resumed or not."""
        ego = _Actor(1)
        backend = TrafficManagerBackend(client=_Client())
        backend.start(_World([ego]), skip_actor_ids={ego.id})
        backend.release(_stand_in(ego), _World())
        backend.resume(_stand_in(ego), _World())
        assert ego.autopilot == [(False, backend.port)]

    def test_a_vehicle_never_released_is_left_alone(self) -> None:
        npc = _Actor(2)
        backend = TrafficManagerBackend(client=_Client())
        backend.start(_World([npc]))
        backend.resume(_stand_in(npc), _World())
        assert npc.autopilot == [(True, backend.port)]


class TestTrafficManagerBackgroundVehicles:
    """An adversity takes a background vehicle over by its actor alone (#417)."""

    def test_the_background_vehicles_are_offered_by_role_name(self) -> None:
        live, gone = _Actor(2), _Actor(3)
        live.is_alive, gone.is_alive = True, False  # type: ignore[attr-defined]
        backend = TrafficManagerBackend(client=_Client())
        backend._background = {"background1": live, "background2": gone}  # type: ignore[dict-item]
        # The dead one is no agent of the scene, so never chosen.
        assert backend.background_actors(_World()) == backend._background
        assert NullTrafficBackend().background_actors(_World()) == {}

    def test_a_background_vehicle_comes_off_and_back_on_autopilot(self) -> None:
        npc = _Actor(2)
        npc.attributes = {"role_name": "background1"}
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=_Client()
        )
        backend.start(_World([npc]))
        backend.release_actor(_World(), npc)  # type: ignore[arg-type]
        backend.resume_actor(_World(), npc)  # type: ignore[arg-type]
        assert npc.autopilot == [(True, 8123), (False, 8123), (True, 8123)]
        assert NullTrafficBackend().release_actor(_World(), npc) is None  # type: ignore[arg-type,func-returns-value]


@pytest.mark.parametrize(
    ("intent", "entity_method"),
    [("release", "release_from_traffic"), ("resume", "resume_traffic")],
)
class TestTheHandOverIntents:
    """Release and resume go to the entity's own backend; the base does nothing."""

    def test_the_entity_asks_its_own_backend(
        self, intent: str, entity_method: str
    ) -> None:
        asked: List[Any] = []
        backend = NullTrafficBackend()
        setattr(backend, intent, lambda entity, world: asked.append(entity))
        entity = VehicleEntity(
            VehicleEntityConfig(role_name="npc1", spawn_location=SpawnPointIndex(0))
        )
        entity.set_traffic_backend(backend)
        getattr(entity, entity_method)(_World())
        assert asked == [entity]

    def test_the_base_backend_does_it_quietly(
        self, intent: str, entity_method: str
    ) -> None:
        getattr(NullTrafficBackend(), intent)(
            _stand_in(_Actor(1), role_name=None), _World()
        )


class TestVehicleEntityDestroy:
    def test_it_holds_no_map_past_its_run(self) -> None:
        """A finished run's scenarios stay in the queue, and their entities with
        them: a map kept for a lane change would stay too, one a case."""
        entity = VehicleEntity(
            VehicleEntityConfig(role_name="npc1", spawn_location=SpawnPointIndex(0))
        )
        entity._lane_change_target = (1, 2)
        entity._lane_change_map = object()  # type: ignore[assignment]
        entity.destroy()
        assert entity._lane_change_map is None
        assert entity._lane_change_target is None


class TestTrafficManagerClose:
    def test_the_manager_is_shut_down_once(self) -> None:
        """A fresh manager per run: shutting down resets its map cache."""
        client = _Client()
        backend = TrafficManagerBackend(client=client)
        backend.prepare(_context(client))
        backend.close()
        backend.close()
        assert client.tm.shut_downs == 1

    def test_a_failing_shutdown_does_not_fail_the_run(self) -> None:
        """The result is already decided by the time teardown runs."""

        class _Angry(_TrafficManager):
            def shut_down(self) -> None:
                raise RuntimeError("CARLA said no")

        client = _Client(_Angry())
        backend = TrafficManagerBackend(client=client)
        backend.prepare(_context(client))
        backend.close()

    def test_it_reports_what_drove_the_run(self) -> None:
        client = _Client()
        backend = TrafficManagerBackend(TrafficManagerBackendConfig(port=8123))
        backend.prepare(_context(client, seed=11))
        assert backend.describe() == {
            "backend": "traffic_manager",
            "port": 8123,
            "random_seed": 11,
            "follow_visibility": False,
            "visibility_m": None,
        }


class TestTrafficManagerVisibility:
    """The TrafficManager models no sight: it logs the plan and does nothing."""

    @pytest.fixture(autouse=True)
    def _forget(self) -> Iterator[None]:
        from autoware_carla_scenario.visibility import forget_visibility

        forget_visibility()
        yield
        forget_visibility()

    def test_a_visibility_is_logged_once_and_nothing_is_done(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario.visibility import set_world_visibility

        actor = _Actor(1)
        world = _World([actor])
        world.id = 4  # type: ignore[attr-defined]
        client = _Client()
        backend = TrafficManagerBackend(client=client)
        backend.prepare(_context(client, world))
        backend.start(world)
        calls = (list(actor.autopilot), list(client.ports))

        set_world_visibility(4, 50.0)
        with caplog.at_level(logging.INFO):
            backend.tick(world, 0.05)
            backend.tick(world, 0.10)
        planned = [r for r in caplog.records if "drivers see 50 m" in r.getMessage()]
        assert len(planned) == 1
        assert "at most 62 km/h" in planned[0].getMessage()
        # Nothing reached CARLA: no autopilot change, no TrafficManager call.
        assert (actor.autopilot, client.ports) == calls

        caplog.clear()
        set_world_visibility(4, 20.0)
        with caplog.at_level(logging.INFO):
            backend.tick(world, 0.15)
        assert any("drivers see 20 m" in r.getMessage() for r in caplog.records)
        backend.close()

    def _following(
        self, *actors: Any, speed_factor: Optional[float] = None
    ) -> Tuple[Any, Any, Any]:
        from autoware_carla_scenario.traffic.parameters import (
            Spread,
            VehicleParameters,
        )

        world = _World(list(actors))
        world.id = 4  # type: ignore[attr-defined]
        client = _Client()
        backend = TrafficManagerBackend(
            client=client,
            vehicles=VehicleParameters(
                follow_visibility=True,
                speed_factor=None if speed_factor is None else Spread(speed_factor),
            ),
        )
        backend.prepare(_context(client, world))
        backend.start(world)
        return backend, world, client.tm

    def test_with_follow_visibility_its_traffic_is_held_to_what_it_sees(
        self,
    ) -> None:
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            set_world_visibility,
            sight_speed,
        )
        from autoware_carla_scenario.traffic.traffic_manager import (
            SIGHT_UPDATE_S,
            TM_DEFAULT_SPEED_DIFFERENCE_PCT,
        )

        town = _Actor(1, speed_limit=50.0)
        highway = _Actor(2, speed_limit=120.0)
        backend, world, tm = self._following(town, highway)
        backend.tick(world, 0.05)
        assert tm.speed_differences == {}  # nothing set, nothing held

        set_world_visibility(4, 50.0)
        backend.tick(world, 0.10)
        limit = sight_speed(50.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        # 62 km/h: the town's 50 is under it, kept at the TrafficManager's
        # own percentage; the highway's 120 is held to it.
        assert 1 not in tm.speed_differences
        assert tm.speed_differences[2] == [pytest.approx(100 * (1 - limit / 120.0))]
        assert backend.describe()["follow_visibility"] is True

        # The limit changes along the road: worked out again each update.
        highway.speed_limit = 100.0
        backend.tick(world, 0.15)
        assert len(tm.speed_differences[2]) == 1  # not due yet
        backend.tick(world, 0.10 + SIGHT_UPDATE_S)
        assert tm.speed_differences[2][-1] == pytest.approx(100 * (1 - limit / 100.0))

        # The fog lifts: each held vehicle gets its own back.
        from autoware_carla_scenario.visibility import forget_visibility

        forget_visibility()
        backend.tick(world, 0.2 + SIGHT_UPDATE_S)
        assert tm.speed_differences[2][-1] == TM_DEFAULT_SPEED_DIFFERENCE_PCT
        assert tm.target_kmh(highway) == pytest.approx(100.0)
        assert 1 not in tm.speed_differences
        backend.close()

    def test_a_destroyed_vehicle_is_forgotten(self) -> None:
        from autoware_carla_scenario.visibility import set_world_visibility

        gone = _Actor(2, speed_limit=120.0)
        backend, world, tm = self._following(gone)
        set_world_visibility(4, 50.0)
        backend.tick(world, 0.05)
        assert 2 in backend._held
        gone.is_alive = False
        set_world_visibility(4, 40.0)
        backend.tick(world, 0.10)
        assert backend._autopiloted == {} and backend._held == {}
        backend.close()

    def test_a_vehicle_is_never_held_faster_than_its_own_speed_factor(
        self,
    ) -> None:
        from autoware_carla_scenario.visibility import set_world_visibility

        # 0.3 of the limit: 36 km/h on a 120 road, under what 300 m allows.
        highway = _Actor(2, speed_limit=120.0)
        backend, world, tm = self._following(highway, speed_factor=0.3)
        assert tm.speed_differences[2] == [pytest.approx(70.0)]
        set_world_visibility(4, 300.0)
        backend.tick(world, 0.05)
        assert tm.speed_differences[2] == [pytest.approx(70.0)]
        backend.close()

    def test_turning_follow_visibility_off_gives_the_traffic_its_speed_back(
        self,
    ) -> None:
        from autoware_carla_scenario.traffic.parameters import VehicleParameters
        from autoware_carla_scenario.traffic.traffic_manager import (
            TM_DEFAULT_SPEED_DIFFERENCE_PCT,
        )
        from autoware_carla_scenario.visibility import set_world_visibility

        highway = _Actor(2, speed_limit=120.0)
        backend, world, tm = self._following(highway)
        set_world_visibility(4, 30.0)
        backend.tick(world, 0.05)
        assert tm.speed_differences[2][-1] > TM_DEFAULT_SPEED_DIFFERENCE_PCT
        backend.override_vehicle_parameters(
            world, VehicleParameters(follow_visibility=False)
        )
        backend.tick(world, 0.10)
        assert tm.speed_differences[2][-1] == TM_DEFAULT_SPEED_DIFFERENCE_PCT
        assert tm.target_kmh(highway) == pytest.approx(120.0)
        backend.close()

    def test_a_sight_speed_just_under_the_limit_holds(self) -> None:
        """62 km/h on an 80 road, 77 % of it: held, and given the limit back."""
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        road = _Actor(2, speed_limit=80.0)
        backend, world, tm = self._following(road)
        assert tm.target_kmh(road) == pytest.approx(80.0)
        set_world_visibility(4, 50.0)
        backend.tick(world, 0.05)
        limit = sight_speed(50.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        assert tm.target_kmh(road) == pytest.approx(limit)
        forget_visibility()
        backend.tick(world, 0.10)
        assert tm.speed_differences[2][-1] == 0.0
        assert tm.target_kmh(road) == pytest.approx(80.0)
        backend.close()

    def test_a_speed_factor_is_given_back_after_the_hold(self) -> None:
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        highway = _Actor(2, speed_limit=120.0)
        backend, world, tm = self._following(highway, speed_factor=0.9)
        assert tm.target_kmh(highway) == pytest.approx(108.0)
        set_world_visibility(4, 50.0)
        backend.tick(world, 0.05)
        limit = sight_speed(50.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        assert tm.target_kmh(highway) == pytest.approx(limit)
        forget_visibility()
        backend.tick(world, 0.10)
        assert tm.speed_differences[2][-1] == pytest.approx(10.0)
        assert tm.target_kmh(highway) == pytest.approx(108.0)
        backend.close()

    @pytest.mark.parametrize("own_kmh", [40.0, 20.0, 0.0])
    def test_a_speed_of_its_own_is_held_under_and_given_back(
        self, own_kmh: float
    ) -> None:
        """A scenario's ``set_desired_speed`` survives the fog: a stop stays
        a stop, and the speed comes back when the fog lifts."""
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        car = _Actor(2, speed_limit=50.0)
        backend, world, tm = self._following(car)
        entity: Any = SimpleNamespace(actor=car, role_name="car")
        backend.set_desired_speed(entity, world, own_kmh)
        set_world_visibility(4, 20.0)  # 35 km/h
        backend.tick(world, 0.05)
        limit = sight_speed(20.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        assert tm.target_kmh(car) == pytest.approx(min(own_kmh, limit))
        assert 2 not in tm.speed_differences
        assert max(tm.desired_speeds[2]) == own_kmh  # never raised
        forget_visibility()
        backend.tick(world, 0.10)
        assert tm.target_kmh(car) == pytest.approx(own_kmh)
        assert 2 not in tm.speed_differences
        backend.close()

    def test_a_speed_set_while_held_is_held_too(self) -> None:
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        car = _Actor(2, speed_limit=50.0)
        backend, world, tm = self._following(car)
        entity: Any = SimpleNamespace(actor=car, role_name="car")
        backend.set_desired_speed(entity, world, 40.0)
        set_world_visibility(4, 20.0)
        backend.tick(world, 0.05)
        limit = sight_speed(20.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        assert tm.target_kmh(car) == pytest.approx(limit)
        backend.set_desired_speed(entity, world, 0.0)  # a stop: it wins
        assert tm.target_kmh(car) == 0.0
        backend.tick(world, 0.10)
        assert tm.target_kmh(car) == 0.0
        backend.set_desired_speed(entity, world, 60.0)  # held under the sight
        assert tm.target_kmh(car) == pytest.approx(limit)
        forget_visibility()
        backend.tick(world, 0.15)
        assert tm.target_kmh(car) == pytest.approx(60.0)
        backend.close()

    def test_a_vehicle_resumed_in_the_fog_is_held_at_once(self) -> None:
        """A speed set while it was released is held as it is resumed, not
        at the next update."""
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        car = _Actor(2, speed_limit=50.0)
        backend, world, tm = self._following(car)
        entity: Any = SimpleNamespace(actor=car, role_name="car")
        set_world_visibility(4, 20.0)
        backend.tick(world, 0.05)
        limit = sight_speed(20.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        assert tm.target_kmh(car) == pytest.approx(limit)
        backend.release(entity, world)
        backend.set_desired_speed(entity, world, 60.0)  # the scenario's to drive
        assert tm.target_kmh(car) == pytest.approx(60.0)
        backend.resume(entity, world)
        assert tm.target_kmh(car) == pytest.approx(limit)
        forget_visibility()
        backend.tick(world, 0.10)
        assert tm.target_kmh(car) == pytest.approx(60.0)
        backend.close()

    def test_a_speed_factor_drawn_while_held_is_held_at_once(self) -> None:
        """A new draw drops the speed it was given (in the TrafficManager as
        here): held from the draw, and given the draw back."""
        from autoware_carla_scenario.traffic.parameters import (
            Spread,
            VehicleParameters,
        )
        from autoware_carla_scenario.visibility import (
            DEFAULT_DECEL,
            DEFAULT_REACTION_S,
            forget_visibility,
            set_world_visibility,
            sight_speed,
        )

        car = _Actor(2, speed_limit=50.0)
        backend, world, tm = self._following(car)
        entity: Any = SimpleNamespace(actor=car, role_name="car")
        backend.set_desired_speed(entity, world, 40.0)
        set_world_visibility(4, 20.0)
        backend.tick(world, 0.05)
        limit = sight_speed(20.0, DEFAULT_DECEL, DEFAULT_REACTION_S) * 3.6
        backend.override_entity_parameters(
            world, VehicleParameters(speed_factor=Spread(0.9)), entity
        )
        assert tm.target_kmh(car) == pytest.approx(limit)
        forget_visibility()
        backend.tick(world, 0.10)
        assert tm.target_kmh(car) == pytest.approx(45.0)
        backend.close()

    def test_a_world_with_no_visibility_logs_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        world = _World()
        world.id = 4  # type: ignore[attr-defined]
        backend = TrafficManagerBackend(client=_Client())
        with caplog.at_level(logging.INFO):
            backend.tick(world, 0.05)
        assert not any("drivers see" in r.getMessage() for r in caplog.records)


class _Blueprint:
    def has_attribute(self, name: str) -> bool:
        return False


class _BackgroundActor:
    def __init__(self) -> None:
        self.id = 100
        self.destroyed = 0

    def set_target_velocity(self, velocity: Any) -> None:
        pass

    def get_location(self) -> Any:
        return SimpleNamespace(x=1.0, y=2.0)

    def destroy(self) -> None:
        self.destroyed += 1


class _BackgroundWorld:
    def __init__(self, blueprints: List[_Blueprint]) -> None:
        self.blueprints = blueprints
        self.spawned: List[_BackgroundActor] = []

    def get_blueprint_library(self) -> Any:
        return SimpleNamespace(filter=lambda pattern: list(self.blueprints))

    def try_spawn_actor(self, blueprint: Any, transform: Any) -> _BackgroundActor:
        self.spawned.append(_BackgroundActor())
        return self.spawned[-1]


def _road_transform() -> Any:
    import typesafe_carla.carla as carla  # noqa: PLC0415

    return carla.Transform(carla.Location(1.0, 2.0, 0.0), carla.Rotation())


class TestTrafficManagerBackground:
    """Background vehicles: spawned, listed by handle, removed once."""

    def test_no_blueprint_spawns_nothing(self) -> None:
        world: Any = _BackgroundWorld([])
        backend = TrafficManagerBackend()
        assert (
            backend.spawn_background(world, _road_transform(), speed_kmh=30.0) is None
        )
        assert world.spawned == []

    def test_a_vehicle_is_removed_once_and_an_unknown_handle_is_ignored(
        self,
    ) -> None:
        world: Any = _BackgroundWorld([_Blueprint()])
        backend = TrafficManagerBackend()
        handle = backend.spawn_background(world, _road_transform(), speed_kmh=30.0)
        assert handle is not None
        assert backend.background_vehicles(world) == {handle: (1.0, 2.0)}
        backend.remove_background(world, "no-such-handle")
        backend.remove_background(world, handle)
        backend.remove_background(world, handle)
        assert world.spawned[0].destroyed == 1
        assert backend.background_vehicles(world) == {}


# ---------------------------------------------------------------------------
# What the TrafficManager may not destroy
# ---------------------------------------------------------------------------


class _Waypoint:
    """A lane *left* metres long, ending without a successor when *dead*."""

    road_id, section_id, lane_id = 1, 0, -1

    def __init__(self, left: float, dead: bool) -> None:
        self.left, self.dead = left, dead
        self.s = 1000.0 - left

    def next(self, step: float) -> List["_Waypoint"]:
        if step <= self.left:
            return [_Waypoint(self.left - step, self.dead)]
        return [] if self.dead else [_Waypoint(1000.0, False)]


def _road(monkeypatch: pytest.MonkeyPatch, left: float, dead: bool) -> List[Any]:
    """Every vehicle stands *left* metres before its lane ends; returns the
    locations the road is looked up at."""
    import autoware_carla_scenario.traffic.traffic_manager as tm_module

    looked_up: List[Any] = []

    def get_waypoint(location: Any) -> _Waypoint:
        looked_up.append(location)
        return _Waypoint(left, dead)

    road = SimpleNamespace(get_waypoint=get_waypoint)
    monkeypatch.setattr(tm_module, "run_carla_map", lambda world: road)
    return looked_up


def _on_autopilot(*actors: _Actor) -> Tuple[TrafficManagerBackend, _Client]:
    client = _Client()
    backend = TrafficManagerBackend(
        TrafficManagerBackendConfig(port=8123), client=client
    )
    backend.start(_World(list(actors)))
    return backend, client


def _stand(backend: TrafficManagerBackend, until: float, since: float = 0.0) -> None:
    """Tick *backend* every half second of the run from *since* to *until*."""
    elapsed = since
    while elapsed <= until:
        backend.tick(_World(), elapsed)
        elapsed += 0.5


class TestTheTrafficManagerKeepsScenarioVehicles:
    """The TrafficManager destroys a vehicle standing 90 s or meeting a dead
    end; the scenario's own are kept from it, background vehicles are not."""

    def test_a_standing_vehicle_is_re_registered_before_it_is_destroyed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from autoware_carla_scenario.traffic.traffic_manager import (
            _TM_BLOCKED_MARGIN_S,
            _TM_BLOCKED_TIME_THRESHOLD_S,
        )

        _road(monkeypatch, 1000.0, dead=False)
        ego = _Actor(1)
        ego.attributes = {"role_name": "Ego"}
        backend, _ = _on_autopilot(ego)
        refresh = _TM_BLOCKED_TIME_THRESHOLD_S - _TM_BLOCKED_MARGIN_S
        _stand(backend, refresh - 1.0)
        assert ego.autopilot == [(True, 8123)]
        with caplog.at_level(logging.INFO):
            _stand(backend, 2 * refresh + 1.0, since=refresh - 0.5)
        assert ego.autopilot == [(True, 8123)] + [(False, 8123), (True, 8123)] * 2
        kept = [
            r for r in caplog.records if "off autopilot and put back" in r.getMessage()
        ]
        assert len(kept) == 1  # Once a vehicle, not each time.
        assert kept[0].getMessage().startswith("Ego: stood 60 s")

    def test_a_moving_vehicle_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _road(monkeypatch, 1000.0, dead=False)
        npc = _Actor(2)
        npc.speed = 1.0
        backend, _ = _on_autopilot(npc)
        _stand(backend, 200.0)
        assert npc.autopilot == [(True, 8123)]

    def test_the_rest_of_its_path_is_given_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        _road(monkeypatch, 1000.0, dead=False)
        npc = _Actor(2)
        npc.location = carla.Location(11.0, 0.0, 0.0)
        backend, client = _on_autopilot(npc)
        path = [carla.Location(float(x), 0.0, 0.0) for x in (5, 10, 15, 500)]
        backend._remember_path(npc, path)  # type: ignore[arg-type]
        _stand(backend, 61.0)
        assert client.tm.paths == [(2, path[2:])]

    def test_a_background_vehicle_is_left_to_the_traffic_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _road(monkeypatch, 3.0, dead=True)
        car = _Actor(3)
        backend = TrafficManagerBackend(
            TrafficManagerBackendConfig(port=8123), client=_Client()
        )
        backend._background = {"background1": car}  # type: ignore[dict-item]
        backend.start(_World([car]))
        _stand(backend, 200.0)
        assert car.autopilot == [(True, 8123)]
        assert car.controls == []

    def test_a_vehicle_is_held_before_a_dead_end(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _road(monkeypatch, 20.0, dead=True)
        npc = _Actor(2)
        npc.attributes = {"role_name": "npc14"}
        backend, _ = _on_autopilot(npc)
        with caplog.at_level(logging.INFO):
            _stand(backend, 200.0)
        assert npc.autopilot == [(True, 8123), (False, 8123)]
        assert [(c.throttle, c.brake) for c in npc.controls] == [(0.0, 1.0)]
        held = [r for r in caplog.records if "its road ends" in r.getMessage()]
        assert len(held) == 1 and held[0].getMessage().startswith("npc14:")

    def test_a_dead_end_beyond_the_horizon_is_no_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _road(monkeypatch, 200.0, dead=True)
        npc = _Actor(2)
        backend, _ = _on_autopilot(npc)
        _stand(backend, 1.0)
        assert npc.autopilot == [(True, 8123)]

    def test_a_held_vehicle_given_a_route_drives_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typesafe_carla.carla as carla  # noqa: PLC0415

        _road(monkeypatch, 20.0, dead=True)
        npc = _Actor(2)
        backend, _ = _on_autopilot(npc)
        _stand(backend, 1.0)
        backend._remember_path(npc, [carla.Location(300.0, 0.0, 0.0)])  # type: ignore[arg-type]
        backend._drive_on(npc)  # type: ignore[arg-type]
        _stand(backend, 3.0, since=1.5)
        # Back on autopilot, and not held again while far from the route's end.
        assert npc.autopilot == [(True, 8123), (False, 8123), (True, 8123)]

    def test_a_released_vehicle_is_the_scenarios(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _road(monkeypatch, 20.0, dead=True)
        npc = _Actor(2)
        backend, _ = _on_autopilot(npc)
        backend.release(_stand_in(npc), _World())
        _stand(backend, 200.0)
        assert npc.autopilot == [(True, 8123), (False, 8123)]
        assert npc.controls == []

    def test_a_destroyed_vehicle_is_forgotten(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _road(monkeypatch, 1000.0, dead=False)
        npc = _Actor(2)
        backend, _ = _on_autopilot(npc)
        npc.is_alive = False
        _stand(backend, 1.0)
        assert backend._kept == {}

    def test_a_vehicle_standing_where_the_road_went_on_is_not_walked_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        looked_up = _road(monkeypatch, 1000.0, dead=False)
        backend, _ = _on_autopilot(_Actor(2))
        _stand(backend, 30.0)
        assert len(looked_up) == 1


# ---------------------------------------------------------------------------
# The no-traffic backend
# ---------------------------------------------------------------------------


class TestNullBackend:
    def test_nothing_is_autopiloted(self) -> None:
        """The point of it: only what something else drives moves."""
        actors = [_Actor(1), _Actor(2)]
        NullTrafficBackend().start(_World(actors))
        assert [a.autopilot for a in actors] == [[], []]

    def test_it_says_what_it_is_leaving_standing(self, caplog) -> None:
        """An ego with the default `ego.entity=autopilot` is driven by the
        traffic backend, so under this one it never moves.  Without this the run
        dies on its timeout with nothing in the log saying why.
        """
        with caplog.at_level(logging.WARNING):
            NullTrafficBackend().start(_World([_Actor(1), _Actor(2)]))
        assert "2 vehicle(s) are left standing" in caplog.text

    def test_a_vehicle_driven_elsewhere_is_not_counted(self, caplog) -> None:
        """An Autoware ego is not standing; something else drives it."""
        ego = _Actor(1)
        with caplog.at_level(logging.WARNING):
            NullTrafficBackend().start(_World([ego]), skip_actor_ids={ego.id})
        assert "left standing" not in caplog.text

    def test_it_takes_no_options(self) -> None:
        """A typo in an options block is a refusal, not a silently ignored key."""
        with pytest.raises(ValueError, match="takes no options"):
            build_backend("none", {"port": 8100})


# ---------------------------------------------------------------------------
# What an entity does with a backend
# ---------------------------------------------------------------------------


class _Recording(TrafficBackend):
    """A backend that only remembers what it was asked for."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []

    def change_lane(self, entity: Any, world: Any, direction: Any) -> None:
        self.calls.append(("change_lane", direction))

    def lane_change_finished(self, entity: Any, world: Any) -> bool:
        self.calls.append(("lane_change_finished", None))
        return True

    def turn_at_junction(
        self, entity: Any, world: Any, direction: Any, **kwargs: Any
    ) -> None:
        self.calls.append(("turn_at_junction", direction))

    def follow_route(self, entity: Any, world: Any, path: Any) -> None:
        self.calls.append(("follow_route", path))


class _Vehicle(BackendDriven):
    def __init__(self, role_name: str = "npc1") -> None:
        self.actor = _Actor(1)
        #: What `register_entity` files this vehicle under.
        self.role_name = role_name


class _EmptyIsFalsey(TrafficBackend):
    """A backend that reports how many vehicles it is running -- none, yet.

    Not a contrived case: a backend wrapping a traffic simulator has an obvious
    ``__len__``, and it is empty exactly when a run starts.
    """

    name = "falsey"

    def __len__(self) -> int:
        return 0


class TestAnExplicitBackendIsNeverReplaced:
    """Selection tests for ``None``, not for truthiness.

    A backend is a third party's object.  One that was explicitly chosen has to
    drive the run whatever it reports about itself, or its lifecycle is never
    invoked and the TrafficManager silently takes over.
    """

    def test_the_runner_keeps_it(self) -> None:
        from unittest.mock import MagicMock  # noqa: PLC0415

        from autoware_carla_scenario import ScenarioRunner  # noqa: PLC0415

        backend = _EmptyIsFalsey()
        assert ScenarioRunner(MagicMock(), traffic_backend=backend).traffic_backend is (
            backend
        )

    def test_an_entity_keeps_it(self) -> None:
        vehicle, backend = _Vehicle(), _EmptyIsFalsey()
        vehicle.set_traffic_backend(backend)
        vehicle.set_client(_Client(), tm_port=8123)
        assert vehicle._resolve_backend() is backend


class TestWhatAScenarioInjects:
    """`register_entity` hands over one thing, and it is the backend.

    `set_client` names a particular traffic model, which is the one thing on
    `BackendDriven` that does.  Nothing on the live path calls it any more, and
    these tests are what keeps that true: a run that selected a backend must not
    also be quietly handing its NPCs a TrafficManager's ingredients.
    """

    @staticmethod
    def _scenario():
        import typesafe_carla.carla as carla  # noqa: PLC0415

        from autoware_carla_scenario import BaseScenario, EgoConfig  # noqa: PLC0415
        from autoware_carla_scenario.entity import SpawnTransform  # noqa: PLC0415

        class _Scenario(BaseScenario):
            def setup(self) -> None: ...

            def is_done(self) -> bool:
                return True

        return _Scenario(
            EgoConfig(
                spawn_location=SpawnTransform(
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
                ),
                vehicle_type="vehicle.mini.cooper",
            )
        )

    def test_a_backend_is_what_an_npc_is_given(self) -> None:
        scenario, backend, npc = self._scenario(), _Recording(), _Vehicle()
        scenario.set_client(_Client())
        scenario.set_traffic_backend(backend)

        scenario.register_entity(npc)

        assert npc._resolve_backend() is backend
        # And no TrafficManager ingredients on the side.
        assert npc._tm_client is None
        assert npc._fallback_backend is None

    def test_a_scenario_with_no_backend_still_drives_its_npc(self) -> None:
        """The path a scenario driven outside a runner takes, unchanged."""
        scenario, npc = self._scenario(), _Vehicle()
        scenario.set_client(_Client(), tm_port=8123)

        scenario.register_entity(npc)

        backend = npc._resolve_backend()
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 8123


class TestEntityDelegation:
    def test_every_intent_reaches_the_backend(self) -> None:
        vehicle, backend = _Vehicle(), _Recording()
        vehicle.set_traffic_backend(backend)
        world = _World()

        vehicle.change_lane(world, LaneChangeDirection.LEFT)
        vehicle.turn_at_junction(world, TurnDirection.RIGHT)
        vehicle.follow_route(world, [])
        assert vehicle.lane_change_finished(world) is True
        assert [name for name, _ in backend.calls] == [
            "change_lane",
            "turn_at_junction",
            "follow_route",
            "lane_change_finished",
        ]

    def test_an_injected_backend_wins_over_a_client(self) -> None:
        """A run that selected a traffic model means that model to drive.

        The client is only the fallback's ingredient, and it may arrive after
        the backend does -- ``ScenarioRunner`` injects both.
        """
        vehicle, backend = _Vehicle(), _Recording()
        vehicle.set_traffic_backend(backend)
        vehicle.set_client(_Client(), tm_port=8123)

        vehicle.change_lane(_World(), LaneChangeDirection.RIGHT)
        assert [name for name, _ in backend.calls] == ["change_lane"]

    def test_a_client_alone_still_means_the_traffic_manager(self) -> None:
        """The path an entity built outside a run takes, unchanged."""
        vehicle = _Vehicle()
        vehicle.set_client(_Client(), tm_port=8123)
        backend = vehicle._resolve_backend()
        assert isinstance(backend, TrafficManagerBackend)
        assert backend.port == 8123

    def test_the_backend_is_built_once_per_client_not_once_per_manoeuvre(
        self,
    ) -> None:
        """The client arrives exactly once, so the backend it stands for does too."""
        vehicle = _Vehicle()
        vehicle.set_client(_Client(), tm_port=8123)
        assert vehicle._resolve_backend() is vehicle._resolve_backend()

    def test_neither_is_not_a_crash(self) -> None:
        """An entity with no client still answers what needs no CARLA call.

        Whether a lane change has settled is arithmetic on the entity's own
        state, and that is what such an entity did before the seam existed.
        """
        vehicle = _Vehicle()
        vehicle.change_lane(_World(), LaneChangeDirection.LEFT)
        assert vehicle.lane_change_finished(_World()) is False
