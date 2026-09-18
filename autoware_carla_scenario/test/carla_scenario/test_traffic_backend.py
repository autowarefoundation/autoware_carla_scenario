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
from typing import Any, List, Optional, Tuple

import pytest

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
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self.autopilot: List[Tuple[bool, int]] = []

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot.append((enabled, port))


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


class _TrafficManager:
    def __init__(self) -> None:
        self.synchronous: List[bool] = []
        self.seeds: List[int] = []
        self.shut_downs = 0

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
        entity = object()
        world = _World()
        backend.change_lane(entity, world, LaneChangeDirection.LEFT)
        backend.turn_at_junction(entity, world, TurnDirection.RIGHT)
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
        }


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
        import carla  # noqa: PLC0415

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
        assert vehicle.lane_change_finished(world) is True
        assert [name for name, _ in backend.calls] == [
            "change_lane",
            "turn_at_junction",
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
