"""Background traffic sources and sinks, against a fake backend and world.

What they place and when is decided here; whether SUMO or the TrafficManager
then drives the vehicles is the backends' business, tested with them.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import typesafe_carla.carla as carla
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

# Imported here, before the autouse ``_map`` fixture fakes
# ``coordinate.to_carla_world`` and ``coordinate.lanelet_length``: the
# Hydra tests import ``examples.run``, which pulls in the example scenarios
# and ``placement``, and a module first imported while the fakes are in place
# binds them by name for the rest of the process (``from
# autoware_carla_scenario import to_carla_world``, which the package's lazy
# ``__getattr__`` also caches) -- monkeypatch puts back only the attribute it
# set (#1916).
from autoware_carla_scenario.examples import run as _run_module  # noqa: F401
from autoware_carla_scenario.traffic import TrafficBackend

_NOT_JUNCTION = [{"type": "not", "constraint": {"type": "is_junction"}}]


class _Backend(TrafficBackend):
    """Places a vehicle wherever it is asked to and remembers where."""

    name = "fake"

    def __init__(self) -> None:
        self.vehicles: dict[str, tuple[float, float, Optional[float]]] = {}
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
        self.vehicles[handle] = (
            transform.location.x,
            transform.location.y,
            transform.location.z,
        )
        return handle

    def background_vehicles(
        self, world: Any
    ) -> dict[str, tuple[float, float, Optional[float]]]:
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


def _modules_holding(fakes: tuple[object, ...]) -> list[str]:
    """``module.name`` of every module attribute that is one of *fakes*."""
    held = []
    for module_name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            if any(value is fake for fake in fakes):
                held.append(f"{module_name}.{name}")
    return held


@pytest.fixture(autouse=True)
def _map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Three 100 m lanelets side by side at y = 0, 10 and 20: lanelet n at y = 10 (n - 1).

    Once the test is done the fakes must be gone from every module: one a
    module bound while they were in place would stand in for the real
    function in every later test of the process (#1916).
    """
    import autoware_carla_scenario.coordinate as coordinate

    monkeypatch.setattr(
        background_traffic.LaneletRegion,
        "lanelet_ids",
        property(lambda self: [1, 2, 3]),
    )

    def lanelet_length(lanelet_id: int) -> float:
        return 100.0

    monkeypatch.setattr(coordinate, "lanelet_length", lanelet_length)

    def to_carla_world(pose: Any) -> Any:
        x, y = pose.s, 10.0 * (pose.lanelet_id - 1)
        return SimpleNamespace(
            x=x,
            y=y,
            z=0.0,
            to_carla_transform=lambda: carla.Transform(carla.Location(x=x, y=y, z=0.0)),
        )

    monkeypatch.setattr(coordinate, "to_carla_world", to_carla_world)
    # On the region: lanelets 1 and 2 only, i.e. y < 15.
    monkeypatch.setattr(
        background_traffic.LaneletRegion,
        "contains",
        lambda self, x, y, z=None: y < 15.0,
    )
    yield
    monkeypatch.undo()
    leaked = _modules_holding((to_carla_world, lanelet_length))
    assert not leaked, f"the faked map leaked into {leaked}"


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

    def test_a_rerun_places_the_initial_vehicles_again_where_the_first_did(
        self,
    ) -> None:
        """A source registered in a scenario's __init__ is the same object on
        the next run: reset, it places its initial vehicles again, drawn
        from its seed as the first run's were (#928)."""
        source = TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=3, seed=11)
        placed = []
        for _ in range(2):
            backend = _Backend()
            source.reset()
            source.set_traffic_backend(backend)
            source.tick(_World(), 0.0)
            placed.append(sorted(backend.vehicles.values()))
        assert len(placed[0]) == 3
        assert placed[1] == placed[0]

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
        def placed(seed: int) -> list[tuple[float, float, Optional[float]]]:
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

    @pytest.mark.parametrize("key", ["initial_vehicles", "max_vehicles", "seed"])
    @pytest.mark.parametrize("value", [2.5, math.nan, math.inf, True, -1, "2"])
    def test_counts_and_seed_must_be_whole_numbers(self, key: str, value: Any) -> None:
        # #1492: 2.5, NaN or True passed `< 0` and reached range() mid-run.
        with pytest.raises(ValueError, match=key):
            TrafficSourceAction(_NOT_JUNCTION, **{key: value})

    def test_whole_float_counts_are_read_as_ints(self) -> None:
        # Typed Any: the point is a float where an int is declared.
        two: Any = 2.0
        three: Any = 3.0
        four: Any = 4.0
        source = TrafficSourceAction(
            _NOT_JUNCTION, initial_vehicles=two, max_vehicles=three, seed=four
        )
        assert source._initial == 2 and type(source._initial) is int  # noqa: SLF001
        assert source._max == 3 and type(source._max) is int  # noqa: SLF001
        assert source._seed == 4 and type(source._seed) is int  # noqa: SLF001
        world, backend = _World(), _Backend()
        source.set_traffic_backend(backend)
        source.tick(world, 0.0)
        assert source.spawned == 2
        assert len(backend.vehicles) == 2

    def test_a_whole_float_limit_caps_the_vehicles(self) -> None:
        limit: Any = 4.0
        source = TrafficSourceAction(
            _NOT_JUNCTION, initial_vehicles=6, max_vehicles=limit, min_gap_m=0.0
        )
        world, backend = _World(), _Backend()
        source.set_traffic_backend(backend)
        source.tick(world, 0.0)
        assert len(backend.vehicles) == 4

    @pytest.mark.parametrize("rate", [math.nan, math.inf, -math.inf])
    def test_a_rate_must_be_finite(self, rate: float) -> None:
        # #1321: NaN or infinity failed int() on the second tick, mid-run.
        with pytest.raises(ValueError, match="vehicles_per_minute"):
            TrafficSourceAction(_NOT_JUNCTION, vehicles_per_minute=rate)

    @pytest.mark.parametrize("gap", [math.nan, math.inf, -1.0])
    def test_a_gap_must_be_finite_and_not_negative(self, gap: float) -> None:
        # A NaN gap let every spot through: no distance is below it.
        with pytest.raises(ValueError, match="min_gap_m"):
            TrafficSourceAction(_NOT_JUNCTION, min_gap_m=gap)

    def test_a_zero_rate_and_gap_are_taken(self) -> None:
        TrafficSourceAction(_NOT_JUNCTION, vehicles_per_minute=0.0, min_gap_m=0.0)

    def test_a_vehicle_parked_under_the_map_takes_no_room(self) -> None:
        # #853: an entity parked 500 m under the map until its spawn.
        def vehicle(x: float, y: float, z: float) -> Any:
            return SimpleNamespace(get_location=lambda: carla.Location(x=x, y=y, z=z))

        world = _World()
        world.get_actors = lambda: SimpleNamespace(  # type: ignore[method-assign]
            filter=lambda pattern: [vehicle(1.0, 2.0, 0.3), vehicle(5.0, 6.0, -499.7)]
        )
        assert background_traffic._vehicle_positions(world) == [
            (1.0, 2.0, pytest.approx(0.3))
        ]


class TestSink:
    def test_it_removes_the_vehicles_on_its_lanelets_only(self) -> None:
        backend, world = _Backend(), _World()
        backend.vehicles = {
            "a": (5.0, 0.0, 0.0),
            "b": (5.0, 10.0, 0.0),
            "c": (5.0, 20.0, 0.0),
        }
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
        backend.vehicles["late"] = (5.0, 0.0, 0.0)
        _run(sink, world, 0.5)
        assert "late" not in backend.vehicles
        _run(sink, world, 1.0)  # past 2 s: the sink has ended
        backend.vehicles["after"] = (5.0, 0.0, 0.0)
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

    @staticmethod
    def _composed(overrides: list[str]) -> Any:
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from autoware_carla_scenario.examples import conf as conf_package

        conf_dir = str(Path(conf_package.__file__).parent.resolve())
        with initialize_config_dir(config_dir=conf_dir, version_base=None):
            return compose(
                config_name="config",
                overrides=["background_traffic.enabled=true", *overrides],
            )

    @pytest.mark.parametrize(
        ("override", "key"),
        [
            # What int() alone read as 1, 1, 2, 1 and 7 (#1445).
            ("background_traffic.seed=1.9", "background_traffic.seed"),
            ("background_traffic.seed=true", "background_traffic.seed"),
            ("background_traffic.seed=-5", "background_traffic.seed"),
            ("background_traffic.seed=abc", "background_traffic.seed"),
            # Above 2**31 - 1, what CARLA and SUMO take (#1560).
            ("background_traffic.seed=2147483648", "background_traffic.seed"),
            (
                "background_traffic.source.initial_vehicles=2.9",
                "background_traffic.source.initial_vehicles",
            ),
            (
                "background_traffic.source.initial_vehicles=true",
                "background_traffic.source.initial_vehicles",
            ),
            (
                "background_traffic.source.initial_vehicles=-1",
                "background_traffic.source.initial_vehicles",
            ),
            (
                "background_traffic.source.max_vehicles=1.5",
                "background_traffic.source.max_vehicles",
            ),
            (
                "background_traffic.source.max_vehicles=false",
                "background_traffic.source.max_vehicles",
            ),
            (
                "background_traffic.source.max_vehicles=-2",
                "background_traffic.source.max_vehicles",
            ),
        ],
    )
    def test_a_count_or_seed_that_is_no_whole_number_is_refused_up_front(
        self, override: str, key: str
    ) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed([override])
        with pytest.raises(ValueError, match=rf"^{key} must "):
            run.add_background_traffic(cfg, _scenario())
        # Before anything is built, as the run's other settings are.
        with pytest.raises(run.RunConfigError, match=rf"^{key} must "):
            run.check_run_config(cfg)

    def test_whole_floats_are_those_numbers(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed(
            [
                "background_traffic.seed=7.0",
                "background_traffic.source.initial_vehicles=3.0",
                "background_traffic.source.max_vehicles=4.0",
                "background_traffic.source.vehicles_per_minute=0",
            ]
        )
        settings = run.background_traffic_settings(cfg)
        assert settings is not None
        assert (settings.seed, settings.initial_vehicles, settings.max_vehicles) == (
            7,
            3,
            4,
        )
        assert isinstance(settings.initial_vehicles, int)
        scenario = _scenario()
        run.add_background_traffic(cfg, scenario)
        (initial,) = scenario._init_actions
        assert isinstance(initial, TrafficSourceAction)
        assert (initial._initial, initial._max, initial._seed) == (3, 4, 7)

    @pytest.mark.parametrize(
        ("key", "message"),
        [
            ("speed_kmh", "is not a background speed"),
            ("min_gap_m", "must be finite and not negative"),
            ("vehicles_per_minute", "must be finite and not negative"),
        ],
    )
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            (10**400, "1000"),
            # More digits than str() turns into text (#1660).
            (10**5000, "an integer of about 10\\*\\*"),
            (-(10**5000), "an integer of about -10\\*\\*"),
        ],
        ids=["huge", "too-long-for-str", "-too-long-for-str"],
    )
    def test_an_amount_too_big_for_a_float_is_refused_by_its_key(
        self, key: str, message: str, value: int, shown: str
    ) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed([])
        cfg.background_traffic.source[key] = value
        prefix = rf"^background_traffic\.source\.{key}: {shown}.* {message}"
        with pytest.raises(ValueError, match=prefix):
            run.background_traffic_settings(cfg)
        with pytest.raises(run.RunConfigError, match=prefix):
            run.check_run_config(cfg)

    @pytest.mark.parametrize("key", ["speed_kmh", "min_gap_m", "vehicles_per_minute"])
    def test_a_list_holding_such_an_int_is_not_a_number(self, key: str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed([])
        cfg.background_traffic.source[key] = [10**5000]
        with pytest.raises(
            ValueError,
            match=rf"^background_traffic\.source\.{key}: a list holding an integer",
        ):
            run.background_traffic_settings(cfg)

    @pytest.mark.parametrize("block", ["source", "sink"])
    @pytest.mark.parametrize(
        ("value", "shown"),
        [
            ("abc", "'abc'"),
            (True, "True"),
            (0, "0"),
            (10**400, "1000"),
            (10**5000, "an integer of about 10\\*\\*"),
        ],
        ids=["text", "bool", "zero", "huge", "too-long-for-str"],
    )
    def test_a_stop_after_seconds_that_is_no_duration_is_refused_by_its_key(
        self, block: str, value: object, shown: str
    ) -> None:
        """#1660: refused, naming the key, where it was refused without it."""
        from autoware_carla_scenario.examples import run

        cfg = self._composed([])
        cfg.background_traffic[block].stop_after_seconds = value
        with pytest.raises(
            ValueError,
            match=rf"^background_traffic\.{block}\.stop_after_seconds: "
            rf"{shown}.* is not a duration",
        ):
            run.add_background_traffic(cfg, _scenario())

    @pytest.mark.parametrize("value", [12.5, 3, "7.5"])
    def test_a_stop_after_seconds_is_read_as_before(self, value: float | str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed([])
        cfg.background_traffic.source.stop_after_seconds = value
        cfg.background_traffic.sink.stop_after_seconds = value
        scenario = _scenario()
        run.add_background_traffic(cfg, scenario)
        ends = [a._until for a in scenario._pre_tick_actions]
        assert len(ends) == 2
        durations = set()
        for end in ends:
            assert isinstance(end, ElapsedTimeCondition)
            durations.add(end.duration_seconds)
        assert durations == {float(value)}

    @pytest.mark.parametrize("off", ["'false'", "no", "0", "'0'"])
    def test_enabled_spelled_off_as_text_is_off(self, off: str) -> None:
        """``${oc.env:VAR,false}`` gives text: ``"false"`` is off, where
        ``bool("false")`` turned the traffic on (#1753)."""
        from autoware_carla_scenario.examples import run

        cfg = self._composed(
            [f"background_traffic.enabled={off}", "background_traffic.seed=1.9"]
        )
        assert run.background_traffic_settings(cfg) is None
        scenario = _scenario()
        run.add_background_traffic(cfg, scenario)
        assert not scenario._init_actions and not scenario._pre_tick_actions
        run.check_run_config(cfg)

    @pytest.mark.parametrize("on", ["'true'", "yes", "1", "' On '"])
    def test_enabled_spelled_on_as_text_is_on(self, on: str) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed([f"background_traffic.enabled={on}"])
        assert run.background_traffic_settings(cfg) is not None

    @pytest.mark.parametrize(
        ("bad", "shown"), [("ture", "'ture'"), ("2", "2"), ("0.0", "0.0")]
    )
    def test_enabled_that_spells_neither_is_refused_up_front(
        self, bad: str, shown: str
    ) -> None:
        import re

        from autoware_carla_scenario.examples import run

        cfg = self._composed([f"background_traffic.enabled={bad}"])
        message = (
            rf"^background_traffic\.enabled must be true or false, "
            rf"got {re.escape(shown)}$"
        )
        with pytest.raises(ValueError, match=message):
            run.add_background_traffic(cfg, _scenario())
        with pytest.raises(run.RunConfigError, match=message):
            run.check_run_config(cfg)

    def test_a_disabled_block_is_not_read(self) -> None:
        from autoware_carla_scenario.examples import run

        cfg = self._composed(
            ["background_traffic.enabled=false", "background_traffic.seed=1.9"]
        )
        assert run.background_traffic_settings(cfg) is None
        run.check_run_config(cfg)


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


class TestMinGapDistance:
    """The source measures its gap with ``math.hypot`` of the differences
    (Codon's ``math`` has no ``dist``): the same number as ``math.dist``."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ((0.0, 0.0), (0.0, 0.0)),
            ((0.0, 0.0), (3.0, 4.0)),
            ((-3.0, 4.0), (0.0, 0.0)),
            ((1.5, -2.25), (-7.125, 9.0)),
            ((1e-300, 1e-300), (-1e-300, -1e-300)),
            ((1e300, 1e300), (-1e300, -1e300)),
            ((0.1, 0.2), (0.3, 0.7)),
            ((5.0, 5.0), (5.0, 20.0)),
        ],
    )
    def test_hypot_equals_dist(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> None:
        import math

        assert math.hypot(a[0] - b[0], a[1] - b[1]) == math.dist(a, b)

    def test_a_vehicle_exactly_min_gap_away_does_not_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import autoware_carla_scenario.coordinate as coordinate

        pose = SimpleNamespace(
            x=0.0,
            y=0.0,
            z=0.0,
            to_carla_transform=lambda: carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0)
            ),
        )
        monkeypatch.setattr(coordinate, "to_carla_world", lambda _pose: pose)
        backend, world = _Backend(), _World()
        source = TrafficSourceAction(_NOT_JUNCTION, min_gap_m=15.0)
        # 15 m away (9, 12): far enough; 14.9 m away: too close.
        assert source._spawn_one(world, backend, [(9.0, 12.0, 0.0)]) is not None
        assert source._spawn_one(world, backend, [(0.0, 14.9, 0.0)]) is None


class TestMinGapLevels:
    """#1599: a vehicle on another level of road, near in plan, takes no room
    from a spawn; one whose height is unknown is placed in plan alone."""

    @pytest.fixture
    def source(self, monkeypatch: pytest.MonkeyPatch) -> TrafficSourceAction:
        import autoware_carla_scenario.coordinate as coordinate

        pose = SimpleNamespace(
            x=0.0,
            y=0.0,
            z=10.0,
            to_carla_transform=lambda: carla.Transform(
                carla.Location(x=0.0, y=0.0, z=10.0)
            ),
        )
        monkeypatch.setattr(coordinate, "to_carla_world", lambda _pose: pose)
        return TrafficSourceAction(_NOT_JUNCTION, initial_vehicles=1, min_gap_m=15.0)

    @pytest.mark.parametrize("z", [3.2, 16.8, 10.0 - 2.01, 12.01])
    def test_a_vehicle_on_another_level_does_not_block(
        self, source: TrafficSourceAction, z: float
    ) -> None:
        # 6.8 m under, as lanelet 552 runs under 132 on the fixture map.
        assert source._spawn_one(_World(), _Backend(), [(1.0, 1.0, z)]) is not None

    @pytest.mark.parametrize("z", [10.0, 10.5, 8.0, 12.0])
    def test_a_vehicle_on_the_same_level_blocks(
        self, source: TrafficSourceAction, z: float
    ) -> None:
        assert source._spawn_one(_World(), _Backend(), [(1.0, 1.0, z)]) is None

    def test_a_vehicle_of_unknown_height_blocks_in_plan(
        self, source: TrafficSourceAction
    ) -> None:
        # A vehicle SUMO's queue model holds has no height.
        assert source._spawn_one(_World(), _Backend(), [(1.0, 1.0, None)]) is None
        assert source._spawn_one(_World(), _Backend(), [(20.0, 1.0, None)]) is not None

    def test_execute_keeps_the_backends_heights(
        self, source: TrafficSourceAction
    ) -> None:
        backend = _Backend()
        backend.vehicles = {"under": (1.0, 1.0, 3.2), "flat": (40.0, 0.0, None)}
        source.set_traffic_backend(backend)
        source.tick(_World(), 0.0)
        assert len(backend.vehicles) == 3

    def test_a_world_vehicle_on_another_level_does_not_block(
        self, source: TrafficSourceAction
    ) -> None:
        world = _World()
        under = SimpleNamespace(
            get_location=lambda: carla.Location(x=1.0, y=1.0, z=3.2)
        )
        world.get_actors = lambda: SimpleNamespace(  # type: ignore[method-assign]
            filter=lambda pattern: [under]
        )
        backend = _Backend()
        source.set_traffic_backend(backend)
        source.tick(world, 0.0)
        assert len(backend.vehicles) == 1
