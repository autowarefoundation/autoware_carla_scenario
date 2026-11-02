"""The traffic parameters every backend shares, and the refusal of the ones a
backend cannot honour.

CARLA is faked, as in ``test_traffic_backend``: what is under test is which
TrafficManager calls the parameters become.  What SUMO makes of them is
pinned in ``test_traffic_sumo``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import pytest

from autoware_carla_scenario.traffic import (
    Spread,
    PEDESTRIAN_PARAMETER_BACKENDS,
    PedestrianParameters,
    VEHICLE_PARAMETER_BACKENDS,
    NullTrafficBackend,
    TrafficConfig,
    TrafficContext,
    TrafficManagerBackend,
    TrafficParameterError,
    VehicleParameters,
    build_backend,
)


class _TrafficManager:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any) -> None:
            self.calls.append((name, args))

        return record

    def called(self, name: str) -> List[Tuple[Any, ...]]:
        return [args for called, args in self.calls if called == name]


class _Client:
    def __init__(self) -> None:
        self.tm = _TrafficManager()

    def get_trafficmanager(self, port: int) -> _TrafficManager:
        return self.tm


class _Actor:
    def __init__(self, actor_id: int) -> None:
        self.id = actor_id
        self.attributes: dict = {}
        self.autopilot: List[bool] = []

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot.append(enabled)


class _World:
    def __init__(self, actors: List[_Actor]) -> None:
        self._actors = actors

    def get_actors(self) -> "_World":
        return self

    def filter(self, pattern: str) -> List[_Actor]:
        return list(self._actors)


# ---------------------------------------------------------------------------
# The type
# ---------------------------------------------------------------------------


def test_every_field_names_the_backends_that_support_it() -> None:
    assert set(VEHICLE_PARAMETER_BACKENDS) == set(
        VehicleParameters.__dataclass_fields__  # type: ignore[attr-defined]
    )


def test_nothing_set_is_every_default() -> None:
    assert VehicleParameters().set_fields() == []
    assert VehicleParameters().describe() == {}


@pytest.mark.parametrize(
    "field, value",
    [
        ("speed_factor", 0.0),
        ("min_gap_m", -1.0),
        ("ignore_lights", 1.5),
        ("random_lane_changes", -0.1),
        ("decel", 0.0),
        ("reaction_time_s", -0.5),
        ("imperfection", 2.0),
        ("max_lateral_speed_mps", 0.0),
        ("max_lateral_accel", -1.0),
    ],
)
def test_an_impossible_value_is_refused(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        VehicleParameters(**{field: Spread(value)})  # type: ignore[arg-type]


def test_a_mapping_with_an_unknown_key_is_refused() -> None:
    with pytest.raises(ValueError, match="reaction_time"):
        VehicleParameters.from_mapping({"reaction_time": 1.0})


def test_the_traffic_config_reads_its_vehicles_and_pedestrians_nodes() -> None:
    config = TrafficConfig.from_mapping(
        {
            "backend": "sumo",
            "options": {},
            "vehicles": {"speed_factor": 0.8, "follow_visibility": None},
            "pedestrians": {"walking_speed_mps": 1.2},
        }
    )
    assert config.vehicles == VehicleParameters(speed_factor=Spread(0.8))
    assert config.pedestrians == PedestrianParameters(walking_speed_mps=Spread(1.2))
    empty = TrafficConfig.from_mapping({})
    assert empty.vehicles == VehicleParameters()
    assert empty.pedestrians == PedestrianParameters()


def test_the_old_parameters_node_points_to_the_new_ones() -> None:
    with pytest.raises(ValueError, match="traffic.vehicles.*traffic.pedestrians"):
        TrafficConfig.from_mapping({"parameters": {"speed_factor": 0.8}})


def test_restricting_keeps_only_what_is_supported() -> None:
    parameters = VehicleParameters(
        speed_factor=Spread(0.8), reaction_time_s=Spread(1.5)
    )
    assert parameters.restricted_to({"speed_factor"}) == VehicleParameters(
        speed_factor=Spread(0.8)
    )


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_a_field_another_backend_supports_is_refused_naming_it() -> None:
    parameters = VehicleParameters(
        speed_factor=Spread(0.9), reaction_time_s=Spread(1.5), decel=Spread(4.0)
    )
    with pytest.raises(TrafficParameterError) as raised:
        TrafficManagerBackend(vehicles=parameters)
    message = str(raised.value)
    assert "'traffic_manager' traffic backend does not support" in message
    assert (
        "traffic.vehicles.reaction_time_s can only be used with the 'sumo' backend"
        in message
    )
    assert "traffic.vehicles.decel can only be used with" in message
    # A field it does support is not named.
    assert "speed_factor" not in message


def test_a_backend_that_reads_no_parameters_refuses_every_one() -> None:
    with pytest.raises(
        TrafficParameterError,
        match="speed_factor can only be used with the 'traffic_manager' or "
        "'sumo' backends",
    ):
        NullTrafficBackend().set_vehicle_parameters(
            VehicleParameters(speed_factor=Spread(1.0))
        )
    # Nothing set is nothing refused.
    NullTrafficBackend().set_vehicle_parameters(VehicleParameters())


def test_building_a_backend_checks_its_parameters() -> None:
    with pytest.raises(TrafficParameterError, match="'sumo' backend"):
        build_backend(
            "traffic_manager", {}, VehicleParameters(reaction_time_s=Spread(1.0))
        )
    backend = build_backend(
        "traffic_manager", {}, VehicleParameters(min_gap_m=Spread(3.0))
    )
    assert backend.vehicle_parameters.min_gap_m == Spread(3.0)


def test_each_backend_supports_what_the_table_says() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend
    from autoware_carla_scenario.traffic.sumo.config import SumoBackendConfig

    assert TrafficManagerBackend().supported_vehicle_parameters() == {
        "speed_factor",
        "min_gap_m",
        "lane_changes",
        "ignore_lights",
        "ignore_signs",
        "ignore_vehicles",
        "ignore_walkers",
        "random_lane_changes",
        "follow_visibility",
    }
    assert SumoTrafficBackend().supported_vehicle_parameters() == {
        "speed_factor",
        "min_gap_m",
        "lane_changes",
        "accel",
        "decel",
        "reaction_time_s",
        "imperfection",
        "max_lateral_speed_mps",
        "max_lateral_accel",
        "follow_visibility",
    }
    # A lateral acceleration only with SUMO's sublane model, its default.
    lane_based = SumoTrafficBackend(SumoBackendConfig(lateral_resolution_m=None))
    assert lane_based.supported_vehicle_parameters() == (
        SumoTrafficBackend().supported_vehicle_parameters() - {"max_lateral_accel"}
    )
    assert NullTrafficBackend().supported_vehicle_parameters() == set()


def test_every_hand_written_field_list_names_every_field() -> None:
    """Codon has no field reflection, so the methods spell the fields out."""
    fields = set(VehicleParameters.__dataclass_fields__)  # type: ignore[attr-defined]
    everything = VehicleParameters(
        speed_factor=Spread(1.0),
        min_gap_m=Spread(1.0),
        lane_changes=True,
        ignore_lights=Spread(0.5),
        ignore_signs=Spread(0.5),
        ignore_vehicles=Spread(0.5),
        ignore_walkers=Spread(0.5),
        random_lane_changes=Spread(0.5),
        accel=Spread(1.0),
        decel=Spread(1.0),
        reaction_time_s=Spread(1.0),
        imperfection=Spread(0.5),
        max_lateral_speed_mps=Spread(1.0),
        max_lateral_accel=Spread(1.0),
        follow_visibility=True,
    )
    assert set(everything.set_fields()) == fields
    assert set(everything.describe()) == fields
    assert everything.restricted_to(fields) == everything


# ---------------------------------------------------------------------------
# What the TrafficManager makes of them
# ---------------------------------------------------------------------------


def test_the_traffic_manager_sets_the_speed_and_gap_on_each_vehicle() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client,
        vehicles=VehicleParameters(speed_factor=Spread(0.8), min_gap_m=Spread(3.5)),
    )
    npc = _Actor(1)
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(_World([npc]))
    (difference,) = client.tm.called("vehicle_percentage_speed_difference")
    assert difference[0] is npc and difference[1] == pytest.approx(20.0)
    assert client.tm.called("distance_to_leading_vehicle") == [(npc, 3.5)]
    assert backend.describe()["vehicles"] == {
        "speed_factor": {"mean": 0.8, "std": 0.0},
        "min_gap_m": {"mean": 3.5, "std": 0.0},
    }


def test_the_traffic_manager_sets_each_vehicles_parameters_on_autopilot() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client,
        vehicles=VehicleParameters(
            lane_changes=False,
            ignore_lights=Spread(0.25),
            ignore_signs=Spread(0.5),
            ignore_vehicles=Spread(0.0),
            ignore_walkers=Spread(1.0),
            random_lane_changes=Spread(0.1),
        ),
    )
    npc, ego = _Actor(1), _Actor(2)
    world = _World([npc, ego])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(world, skip_actor_ids={ego.id})
    tm = client.tm
    assert tm.called("auto_lane_change") == [(npc, False)]
    assert tm.called("ignore_lights_percentage") == [(npc, 25.0)]
    assert tm.called("ignore_signs_percentage") == [(npc, 50.0)]
    assert tm.called("ignore_vehicles_percentage") == [(npc, 0.0)]
    assert tm.called("ignore_walkers_percentage") == [(npc, 100.0)]
    (left,) = tm.called("random_left_lanechange_percentage")
    (right,) = tm.called("random_right_lanechange_percentage")
    assert left[1] == pytest.approx(10.0) and right[1] == pytest.approx(10.0)
    assert ego.autopilot == []


def test_the_traffic_manager_without_parameters_sets_nothing_more() -> None:
    client = _Client()
    backend = TrafficManagerBackend(client=client)
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(_World([_Actor(1)]))
    # The global percentage is LibCarla's own default (0), set, not assumed.
    assert {name for name, _ in client.tm.calls} == {
        "set_synchronous_mode",
        "set_random_device_seed",
        "global_percentage_speed_difference",
    }
    assert client.tm.called("global_percentage_speed_difference") == [(0.0,)]
    assert "vehicles" not in backend.describe()


class _Entity:
    def __init__(self, actor: _Actor) -> None:
        self.actor = actor
        self.role_name = "npc"


def _released_before_start(
    vehicles: VehicleParameters,
) -> Tuple[_Client, TrafficManagerBackend, _World, _Actor, _Entity]:
    """A vehicle the scenario drives from its init (a trajectory), so not put
    on autopilot at start."""
    client = _Client()
    backend = TrafficManagerBackend(client=client, vehicles=vehicles)
    npc = _Actor(42)
    world = _World([npc])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.release_actor(world, npc)  # type: ignore[arg-type]
    backend.start(world)  # type: ignore[arg-type]
    return client, backend, world, npc, _Entity(npc)


def _speed_differences(client: _Client) -> List[float]:
    return [args[1] for args in client.tm.called("vehicle_percentage_speed_difference")]


def test_an_entity_override_before_its_first_autopilot_is_kept() -> None:
    """#1090: the first draw on autopilot draws the override, not the run's."""
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.8), min_gap_m=Spread(3.5))
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3)),
        entity,  # type: ignore[arg-type]
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_differences(client) == [pytest.approx(-30.0)]
    # The fields it does not override are the run's.
    assert client.tm.called("distance_to_leading_vehicle") == [(npc, 3.5)]
    assert backend._speed_differences[42] == pytest.approx(-30.0)


def test_entity_overrides_layer_in_order() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.8), min_gap_m=Spread(3.5))
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3), min_gap_m=Spread(6.0)),
        entity,  # type: ignore[arg-type]
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(0.5)),
        entity,  # type: ignore[arg-type]
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_differences(client) == [pytest.approx(50.0)]
    assert client.tm.called("distance_to_leading_vehicle") == [(npc, 6.0)]


def test_an_entity_override_of_a_vehicle_on_autopilot_draws_at_once() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(speed_factor=Spread(0.8))
    )
    npc = _Actor(1)
    world = _World([npc])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(world)  # type: ignore[arg-type]
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3)),
        _Entity(npc),  # type: ignore[arg-type]
    )
    assert _speed_differences(client) == [pytest.approx(20.0), pytest.approx(-30.0)]


def test_a_later_override_of_every_vehicle_wins_over_an_entity_override() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.8))
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3), min_gap_m=Spread(6.0)),
        entity,  # type: ignore[arg-type]
    )
    backend.override_vehicle_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(0.9)),
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_differences(client) == [pytest.approx(10.0)]
    assert client.tm.called("distance_to_leading_vehicle") == [(npc, 6.0)]


def test_entity_overrides_are_forgotten_when_the_run_closes() -> None:
    """The next run (whose actor may have the same id) draws its own."""
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.8))
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3)),
        entity,  # type: ignore[arg-type]
    )
    backend.close()
    assert backend._entity_overrides == {}
    client.tm.calls.clear()
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(_World([_Actor(42)]))  # type: ignore[arg-type]
    assert _speed_differences(client) == [pytest.approx(20.0)]


# ---------------------------------------------------------------------------
# A vehicle's first draw keeps what it was told before it (#1096)
# ---------------------------------------------------------------------------


def _speed_calls(client: _Client) -> List[Tuple[str, Any]]:
    return [
        (name, args[1])
        for name, args in client.tm.calls
        if name in ("set_desired_speed", "vehicle_percentage_speed_difference")
    ]


class _SpawningWorld(_World):
    """Enough of a world to spawn one background vehicle in."""

    def __init__(self, actor: _Actor) -> None:
        super().__init__([actor])
        self.actor = actor

    def get_blueprint_library(self) -> Any:
        from types import SimpleNamespace  # noqa: PLC0415

        return SimpleNamespace(
            filter=lambda pattern: [SimpleNamespace(has_attribute=lambda n: False)]
        )

    def try_spawn_actor(self, blueprint: Any, transform: Any) -> _Actor:
        return self.actor

    def get_map(self) -> Any:
        from types import SimpleNamespace  # noqa: PLC0415

        import typesafe_carla.carla as carla  # noqa: PLC0415

        return SimpleNamespace(
            get_waypoint=lambda location, **kwargs: SimpleNamespace(
                transform=carla.Transform(location, carla.Rotation())
            )
        )


def test_a_background_vehicle_spawned_before_start_keeps_its_exact_speed() -> None:
    """The background cap of #967 survives the draw at start."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(speed_factor=Spread(1.0))
    )
    backend.prepare(TrafficContext(client=client, random_seed=1))
    client.tm.calls.clear()
    npc = _Actor(7)
    npc.set_target_velocity = lambda velocity: None  # type: ignore[attr-defined]
    world = _SpawningWorld(npc)
    spot = carla.Transform(carla.Location(x=10.0), carla.Rotation())
    backend.spawn_background(world, spot, speed_kmh=20.0)  # type: ignore[arg-type]
    backend.start(world)  # type: ignore[arg-type]
    assert _speed_calls(client) == [("set_desired_speed", 20.0)]
    assert backend._exact_kmh == {7: 20.0}
    assert backend._speed_differences == {}


def test_a_speed_set_before_start_is_kept_over_the_speed_factor() -> None:
    """An init-phase SetSpeed, as a SetSpeed after the draw is."""
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(speed_factor=Spread(0.9))
    )
    npc = _Actor(3)
    world = _World([npc])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    client.tm.calls.clear()
    backend.set_desired_speed(_Entity(npc), world, 0.0)  # type: ignore[arg-type]
    backend.start(world)  # type: ignore[arg-type]
    assert _speed_calls(client) == [("set_desired_speed", 0.0)]
    assert backend._exact_kmh == {3: 0.0}


def test_a_speed_set_while_released_is_kept_on_resume() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.9))
    )
    backend.set_desired_speed(entity, world, 15.0)  # type: ignore[arg-type]
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_calls(client) == [("set_desired_speed", 15.0)]
    assert backend._exact_kmh == {42: 15.0}
    # Its speed factor was drawn all the same: a later override still draws.
    backend.override_vehicle_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(0.8)),
    )
    assert _speed_calls(client)[-1] == (
        "vehicle_percentage_speed_difference",
        pytest.approx(20.0),
    )


def test_an_entity_override_after_a_speed_wins_at_the_first_draw() -> None:
    """The one given last wins: an override of ``speed_factor`` given after
    the speed, both before the first draw, is sent by it."""
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.9))
    )
    backend.set_desired_speed(entity, world, 15.0)  # type: ignore[arg-type]
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3)),
        entity,  # type: ignore[arg-type]
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_calls(client) == [
        ("set_desired_speed", 15.0),
        ("vehicle_percentage_speed_difference", pytest.approx(-30.0)),
    ]
    assert backend._exact_kmh == {}


def test_a_run_override_after_a_speed_wins_at_the_first_draw() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.9))
    )
    backend.set_desired_speed(entity, world, 15.0)  # type: ignore[arg-type]
    backend.override_vehicle_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(0.8)),
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_calls(client)[-1] == (
        "vehicle_percentage_speed_difference",
        pytest.approx(20.0),
    )
    assert backend._exact_kmh == {}


def test_a_speed_after_an_entity_override_wins_at_the_first_draw() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(speed_factor=Spread(0.9))
    )
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(speed_factor=Spread(1.3)),
        entity,  # type: ignore[arg-type]
    )
    backend.set_desired_speed(entity, world, 15.0)  # type: ignore[arg-type]
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert _speed_calls(client) == [("set_desired_speed", 15.0)]
    assert backend._exact_kmh == {42: 15.0}


def _stub_route_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``follow_route`` plan a route of three points with no map."""
    from types import SimpleNamespace  # noqa: PLC0415

    import autoware_carla_scenario.coordinate.map_manager as map_manager  # noqa: PLC0415
    import autoware_carla_scenario.coordinate.transform as transform  # noqa: PLC0415
    import autoware_carla_scenario.odd.route_plan as route_plan  # noqa: PLC0415
    import autoware_carla_scenario.traffic.traffic_manager as traffic_manager  # noqa: PLC0415

    monkeypatch.setattr(route_plan, "route_poses", lambda *a, **k: [0, 1, 2])
    monkeypatch.setattr(transform, "to_carla_location", lambda pose: pose)
    monkeypatch.setattr(
        map_manager.MapManager,
        "get_instance",
        staticmethod(lambda: SimpleNamespace(lanelet_map=None, routing_graph=None)),
    )
    monkeypatch.setattr(traffic_manager, "run_carla_map", lambda world: None)
    monkeypatch.setattr(
        traffic_manager, "_clear_of_junctions", lambda m, points: list(points)
    )
    monkeypatch.setattr(traffic_manager, "_in_junction", lambda m, p: False)
    monkeypatch.setattr(traffic_manager, "_at_junction_mouth", lambda m, p: False)
    monkeypatch.setattr(traffic_manager, "_lane_change_marks", lambda path: [])


def test_a_route_given_before_start_keeps_lane_changes_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``follow_route`` turns the TrafficManager's lane changes off; the
    first draw of ``lane_changes: true`` does not turn them back on."""
    from types import SimpleNamespace  # noqa: PLC0415

    _stub_route_planning(monkeypatch)
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(lane_changes=True)
    )
    npc, other = _Actor(5), _Actor(6)
    world = _World([npc, other])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    client.tm.calls.clear()
    backend.follow_route(
        _Entity(npc),  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        [SimpleNamespace(), SimpleNamespace()],  # type: ignore[list-item]
    )
    backend.start(world)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == [(npc, False), (other, True)]


@pytest.mark.parametrize(
    ("lane_changes", "after"), [(True, True), (False, False), (None, True)]
)
def test_the_end_of_a_route_gives_lane_changes_back(
    monkeypatch: pytest.MonkeyPatch, lane_changes: Optional[bool], after: bool
) -> None:
    """``end_route`` undoes ``follow_route``'s lane-change lock (#1100): the
    run's ``lane_changes`` again, ``false`` kept off, and unset the
    TrafficManager's own default, which is on."""
    from types import SimpleNamespace  # noqa: PLC0415

    _stub_route_planning(monkeypatch)
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(lane_changes=lane_changes)
    )
    npc = _Actor(5)
    world = _World([npc])
    entity = _Entity(npc)
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(world)  # type: ignore[arg-type]
    backend.follow_route(
        entity,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        [SimpleNamespace(), SimpleNamespace()],  # type: ignore[list-item]
    )
    assert backend._lane_locked == {5}
    client.tm.calls.clear()

    backend.end_route(entity, world)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == [(npc, after)]
    assert backend._lane_locked == set()
    # Safe to repeat: nothing more to give back.
    backend.end_route(entity, world)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == [(npc, after)]


def test_the_end_of_a_route_follows_the_entity_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vehicle whose own override turned its lane changes off keeps them
    off when its route ends, whatever the run's parameters say."""
    from types import SimpleNamespace  # noqa: PLC0415

    _stub_route_planning(monkeypatch)
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(lane_changes=True)
    )
    npc = _Actor(5)
    world = _World([npc])
    entity = _Entity(npc)
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(world)  # type: ignore[arg-type]
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(lane_changes=False),
        entity,  # type: ignore[arg-type]
    )
    backend.follow_route(
        entity,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        [SimpleNamespace(), SimpleNamespace()],  # type: ignore[list-item]
    )
    client.tm.calls.clear()
    backend.end_route(entity, world)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == [(npc, False)]


def test_the_end_of_no_route_sends_nothing() -> None:
    """A vehicle never given a route has no lane-change lock to give back."""
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(lane_changes=True)
    )
    npc = _Actor(5)
    world = _World([npc])
    backend.prepare(TrafficContext(client=client, random_seed=1))
    backend.start(world)  # type: ignore[arg-type]
    client.tm.calls.clear()
    backend.end_route(_Entity(npc), world)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == []


def test_a_lane_changes_override_after_a_route_wins_at_the_first_draw() -> None:
    client, backend, world, npc, entity = _released_before_start(
        VehicleParameters(lane_changes=True)
    )
    backend._lane_locked.add(42)  # as follow_route leaves it
    backend.override_entity_parameters(
        world,  # type: ignore[arg-type]
        VehicleParameters(lane_changes=True),
        entity,  # type: ignore[arg-type]
    )
    backend.resume_actor(world, npc)  # type: ignore[arg-type]
    assert client.tm.called("auto_lane_change") == [(npc, True)]


# ---------------------------------------------------------------------------
# Pedestrians
# ---------------------------------------------------------------------------


def test_every_pedestrian_field_names_the_backends_that_support_it() -> None:
    assert set(PEDESTRIAN_PARAMETER_BACKENDS) == set(
        PedestrianParameters.__dataclass_fields__  # type: ignore[attr-defined]
    )
    everything = PedestrianParameters(
        walking_speed_mps=Spread(1.0),
        min_gap_m=Spread(0.5),
        red_light_violation_s=Spread(10.0),
    )
    assert set(everything.set_fields()) == set(PEDESTRIAN_PARAMETER_BACKENDS)
    assert set(everything.describe()) == set(PEDESTRIAN_PARAMETER_BACKENDS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("walking_speed_mps", 0.0),
        ("min_gap_m", -1.0),
        ("red_light_violation_s", -1.0),
    ],
)
def test_an_impossible_pedestrian_value_is_refused(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        PedestrianParameters(**{field: Spread(value)})  # type: ignore[arg-type]


def test_the_traffic_manager_refuses_pedestrian_parameters() -> None:
    """It drives no pedestrians, so it can honour none of them."""
    assert TrafficManagerBackend().supported_pedestrian_parameters() == set()
    with pytest.raises(
        TrafficParameterError,
        match="traffic.pedestrians.walking_speed_mps can only be used with the "
        "'sumo' backend",
    ):
        build_backend(
            "traffic_manager",
            {},
            pedestrians=PedestrianParameters(walking_speed_mps=Spread(1.2)),
        )


def test_the_pedestrian_parameters_are_recorded() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    backend = SumoTrafficBackend(
        pedestrians=PedestrianParameters(red_light_violation_s=Spread(30.0))
    )
    assert backend.describe()["pedestrians"] == {
        "red_light_violation_s": {"mean": 30.0, "std": 0.0}
    }


def test_every_hand_written_bicycle_field_list_names_every_field() -> None:
    """Codon has no field reflection, so the methods spell the fields out."""
    from autoware_carla_scenario.traffic.parameters import (
        BICYCLE_BOUNDS,
        BICYCLE_PARAMETER_BACKENDS,
        BicycleParameters,
    )

    fields = set(BicycleParameters.__dataclass_fields__)  # type: ignore[attr-defined]
    assert fields == set(BICYCLE_BOUNDS) == set(BICYCLE_PARAMETER_BACKENDS)
    everything = BicycleParameters(
        **{name: 1.0 if name != "imperfection" else 0.5 for name in fields}
    )
    assert set(everything.set_fields()) == fields
    assert set(everything.describe()) == fields
    assert everything.overridden_by(BicycleParameters()) == everything
    assert BicycleParameters(decel=2.0).overridden_by(
        BicycleParameters(accel=1.0, decel=3.0)
    ) == BicycleParameters(accel=1.0, decel=3.0)


def test_a_bicycle_parameter_out_of_its_bounds_is_refused() -> None:
    from autoware_carla_scenario.traffic.parameters import BicycleParameters

    with pytest.raises(ValueError, match="imperfection"):
        BicycleParameters(imperfection=1.5)
    with pytest.raises(ValueError, match="max_speed_mps"):
        BicycleParameters.from_mapping({"max_speed_mps": 0.1})


def test_the_traffic_config_reads_the_bicycle_parameters() -> None:
    from autoware_carla_scenario.traffic.config import TrafficConfig
    from autoware_carla_scenario.traffic.parameters import BicycleParameters

    config = TrafficConfig.from_mapping(
        {
            "backend": "sumo",
            "bicycles": {"max_speed_mps": {"mean": 4.0, "std": 0.5}, "decel": None},
        }
    )
    assert config.bicycles == BicycleParameters(max_speed_mps=Spread(4.0, 0.5))
    backend = build_backend("sumo", {}, bicycles=config.bicycles)
    assert backend.describe()["bicycles"] == {
        "max_speed_mps": {"mean": 4.0, "std": 0.5}
    }
    with pytest.raises(TrafficParameterError, match="traffic.bicycles"):
        build_backend("traffic_manager", {}, bicycles=config.bicycles)
