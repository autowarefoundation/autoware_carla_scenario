"""Background traffic sources and sinks, against a fake backend and world.

What they place and when is decided here; whether SUMO or the TrafficManager
then drives the vehicles is the backends' business, tested with them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import carla
import pytest

from autoware_carla_scenario import (
    BaseScenario,
    EgoConfig,
    ElapsedTimeCondition,
    SpawnTransform,
    TrafficSinkAction,
    TrafficSourceAction,
)
from autoware_carla_scenario.actions import background_traffic
from autoware_carla_scenario.traffic import TrafficBackend

_NOT_JUNCTION = [{"type": "not", "constraint": {"type": "is_junction"}}]


class _Backend(TrafficBackend):
    """Places a vehicle wherever it is asked to and remembers where."""

    name = "fake"

    def __init__(self) -> None:
        self.vehicles: dict[str, tuple[float, float]] = {}
        self.removed: list[str] = []

    def spawn_background(
        self,
        world: Any,
        transform: Any,
        *,
        speed_kmh: float,
        blueprint: Optional[str] = None,
    ) -> Optional[str]:
        handle = f"bg{len(self.vehicles) + len(self.removed) + 1}"
        self.vehicles[handle] = (transform.location.x, transform.location.y)
        return handle

    def background_vehicles(self, world: Any) -> dict[str, tuple[float, float]]:
        return dict(self.vehicles)

    def remove_background(self, world: Any, handle: str) -> None:
        self.vehicles.pop(handle)
        self.removed.append(handle)


class _World:
    """A world whose clock the test sets, with no vehicles of its own."""

    def __init__(self) -> None:
        self.time = 0.0

    def get_snapshot(self) -> Any:
        return SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=self.time))

    def get_actors(self) -> Any:
        return SimpleNamespace(filter=lambda pattern: [])


@pytest.fixture(autouse=True)
def _map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three 100 m lanelets side by side at y = 0, 10 and 20: lanelet n at y = 10 (n - 1)."""
    import autoware_carla_scenario.coordinate as coordinate

    monkeypatch.setattr(
        background_traffic.LaneletRegion,
        "lanelet_ids",
        property(lambda self: [1, 2, 3]),
    )
    monkeypatch.setattr(coordinate, "lanelet_length", lambda lanelet_id: 100.0)

    def to_carla_world(pose: Any) -> Any:
        x, y = pose.s, 10.0 * (pose.lanelet_id - 1)
        return SimpleNamespace(
            x=x,
            y=y,
            to_carla_transform=lambda: carla.Transform(carla.Location(x=x, y=y, z=0.0)),
        )

    monkeypatch.setattr(coordinate, "to_carla_world", to_carla_world)
    # On the region: lanelets 1 and 2 only, i.e. y < 15.
    monkeypatch.setattr(
        background_traffic.LaneletRegion, "contains", lambda self, x, y: y < 15.0
    )


def _run(action: Any, world: _World, seconds: float, dt: float = 0.05) -> None:
    """Tick *action* as the scenario's pre-tick loop would, for *seconds*."""
    steps = round(seconds / dt)
    for _ in range(steps):
        action.tick(world, world.time)
        world.time += dt


class TestSource:
    def test_initial_vehicles_are_placed_on_the_first_tick(self) -> None:
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=5)
        source.set_traffic_backend(backend)
        source.tick(world, 0.0)
        assert len(backend.vehicles) == 5

    def test_registered_for_initialization_it_places_them_once_and_stops(self) -> None:
        """What register_init does: one tick, at elapsed 0, never again."""
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(
            _NOT_JUNCTION, initial_vehicles=4, vehicles_per_minute=600.0
        )
        scenario = _scenario()
        scenario.register_init(source)
        scenario.set_traffic_backend(backend)
        scenario.run_init(world)
        assert len(backend.vehicles) == 4
        # The tick loop never sees it, so the rate never applies.
        assert source not in scenario._pre_tick_actions

    def test_the_rate_adds_vehicles_in_simulated_time(self) -> None:
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(
            _NOT_JUNCTION, vehicles_per_minute=60.0, min_gap_m=0.0
        )
        source.set_traffic_backend(backend)
        _run(source, world, 10.0)
        # One a second for ten seconds, give or take the first tick.
        assert 9 <= len(backend.vehicles) <= 10

    def test_until_ends_the_source(self) -> None:
        """Only during the first 3 s: nothing is added once until fires."""
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(
            _NOT_JUNCTION,
            initial_vehicles=2,
            vehicles_per_minute=60.0,
            min_gap_m=0.0,
            until=ElapsedTimeCondition(3.0, label="first_3s"),
        )
        source.set_traffic_backend(backend)
        # The tick at 3.0 s is the last it acts on: it acts, then sees until.
        _run(source, world, 3.1)
        at_three = len(backend.vehicles)
        _run(source, world, 10.0)
        assert len(backend.vehicles) == at_three
        assert 4 <= at_three <= 5

    def test_max_vehicles_caps_what_is_on_the_road_at_once(self) -> None:
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(
            _NOT_JUNCTION,
            initial_vehicles=10,
            vehicles_per_minute=600.0,
            max_vehicles=6,
            min_gap_m=0.0,
        )
        source.set_traffic_backend(backend)
        _run(source, world, 5.0)
        assert len(backend.vehicles) == 6
        # One leaves; its place is taken again.
        backend.remove_background(world, next(iter(backend.vehicles)))
        _run(source, world, 1.0)
        assert len(backend.vehicles) == 6

    def test_new_vehicles_keep_their_distance(self) -> None:
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=12, min_gap_m=20.0)
        source.set_traffic_backend(backend)
        source.tick(world, 0.0)
        points = list(backend.vehicles.values())
        assert points
        for i, a in enumerate(points):
            for b in points[i + 1 :]:
                assert ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= 20.0

    def test_the_same_seed_places_the_same_vehicles(self) -> None:
        def placed(seed: int) -> list[tuple[float, float]]:
            backend = _Backend()
            source = TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=5, seed=seed)
            source.set_traffic_backend(backend)
            source.tick(_World(), 0.0)
            return list(backend.vehicles.values())

        assert placed(7) == placed(7)
        assert placed(7) != placed(8)

    def test_without_a_backend_it_warns_and_does_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        source = TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=3)
        with caplog.at_level("WARNING"):
            source.tick(_World(), 0.0)
        assert "no traffic backend" in caplog.text

    def test_counts_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError):
            TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=-1)
        with pytest.raises(ValueError):
            TrafficSourceAction(_NOT_JUNCTION, vehicles_per_minute=-1.0)


class TestSink:
    def test_it_removes_the_vehicles_on_its_lanelets_only(self) -> None:
        backend, world = _Backend(), _World()
        backend.vehicles = {"a": (5.0, 0.0), "b": (5.0, 10.0), "c": (5.0, 20.0)}
        sink = TrafficSinkAction(_NOT_JUNCTION)
        sink.set_traffic_backend(backend)
        _run(sink, world, 0.2)
        assert set(backend.vehicles) == {"c"}
        assert sink.removed == 2

    def test_it_keeps_removing_while_it_runs_and_stops_with_until(self) -> None:
        backend, world = _Backend(), _World()
        sink = TrafficSinkAction(
            _NOT_JUNCTION, until=ElapsedTimeCondition(2.0, label="first_2s")
        )
        sink.set_traffic_backend(backend)
        _run(sink, world, 1.0)
        backend.vehicles["late"] = (5.0, 0.0)
        _run(sink, world, 0.5)
        assert "late" not in backend.vehicles
        _run(sink, world, 1.0)  # past 2 s: the sink has ended
        backend.vehicles["after"] = (5.0, 0.0)
        _run(sink, world, 1.0)
        assert "after" in backend.vehicles


def _scenario() -> BaseScenario:
    class _Scenario(BaseScenario):
        def setup(self) -> None: ...

        def is_done(self) -> bool:
            return True

    return _Scenario(
        EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            ),
            vehicle_type="vehicle.mini.cooper",
        )
    )


class TestScenarioHandsOverTheBackend:
    def test_an_action_registered_before_the_backend_gets_it(self) -> None:
        scenario, backend = _scenario(), _Backend()
        source = TrafficSourceAction(_NOT_JUNCTION)
        scenario.register_pre_tick(source)
        scenario.set_traffic_backend(backend)
        assert source._backend is backend

    def test_an_action_registered_after_the_backend_gets_it(self) -> None:
        scenario, backend = _scenario(), _Backend()
        scenario.set_traffic_backend(backend)
        sink = TrafficSinkAction(_NOT_JUNCTION)
        scenario.register_post_tick(sink)
        assert sink._backend is backend


class TestHydraSwitch:
    """``background_traffic`` in the shipped config turns into actions on any scenario."""

    @staticmethod
    def _registered(overrides: list[str]) -> BaseScenario:
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.examples import conf as conf_package
        from autoware_carla_scenario.examples.run import add_background_traffic

        conf_dir = str(Path(conf_package.__file__).parent.resolve())
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)
        scenario = _scenario()
        add_background_traffic(cfg, scenario)
        return scenario

    def test_it_is_off_by_default(self) -> None:
        scenario = self._registered([])
        assert not scenario._init_actions and not scenario._pre_tick_actions

    def test_enabled_it_places_vehicles_at_initialization_and_keeps_adding(
        self,
    ) -> None:
        scenario = self._registered(["background_traffic.enabled=true"])
        (initial,) = scenario._init_actions
        assert isinstance(initial, TrafficSourceAction)
        kinds = sorted(type(a).__name__ for a in scenario._pre_tick_actions)
        assert kinds == ["TrafficSinkAction", "TrafficSourceAction"]

    def test_a_rate_of_zero_keeps_it_to_initialization(self) -> None:
        scenario = self._registered(
            [
                "background_traffic.enabled=true",
                "background_traffic.source.vehicles_per_minute=0",
                "background_traffic.sink.constraints=null",
            ]
        )
        assert len(scenario._init_actions) == 1
        assert scenario._pre_tick_actions == []

    def test_stop_after_seconds_becomes_the_source_end_condition(self) -> None:
        scenario = self._registered(
            [
                "background_traffic.enabled=true",
                "background_traffic.source.stop_after_seconds=12.5",
            ]
        )
        (source,) = [
            a for a in scenario._pre_tick_actions if isinstance(a, TrafficSourceAction)
        ]
        assert isinstance(source._until, ElapsedTimeCondition)


class TestConstraintsFromText:
    """How the scenario editor hands a source or sink its constraints."""

    def test_a_yaml_list_is_read_as_the_constraints(self) -> None:
        text = "- type: not\n  constraint:\n    type: is_junction\n"
        assert background_traffic.constraints_from_text(text) == _NOT_JUNCTION

    def test_a_single_mapping_is_a_list_of_one(self) -> None:
        text = "type: lanelet_length\nrule: greater_than_or_equal\nvalue: 20.0\n"
        assert background_traffic.constraints_from_text(text) == [
            {"type": "lanelet_length", "rule": "greater_than_or_equal", "value": 20.0}
        ]

    def test_an_unknown_constraint_is_refused_when_read(self) -> None:
        with pytest.raises(Exception):
            background_traffic.constraints_from_text("- type: no_such_constraint\n")

    def test_empty_text_is_refused(self) -> None:
        with pytest.raises(ValueError):
            background_traffic.constraints_from_text("  ")

    @pytest.mark.parametrize("type_id", ["traffic_source", "traffic_sink"])
    def test_the_editor_defaults_are_valid_constraints(self, type_id: str) -> None:
        from autoware_carla_scenario.authoring.registry import get_action_spec

        spec = get_action_spec(type_id)
        assert spec is not None
        (field,) = [f for f in spec.fields if f.name == "constraints"]
        assert background_traffic.constraints_from_text(field.default)
