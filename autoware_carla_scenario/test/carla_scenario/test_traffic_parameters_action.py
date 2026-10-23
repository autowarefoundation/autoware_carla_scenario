"""Changing the traffic's parameters during a run
(:class:`~autoware_carla_scenario.TrafficParametersAction`).

The TrafficManager is faked, as in ``test_traffic_parameters``: what is under
test is which calls an override becomes.  SUMO's side runs on libsumo in
``test_traffic_sumo``.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Tuple

import pytest

from autoware_carla_scenario import TrafficParametersAction
from autoware_carla_scenario.entity.registry import (
    clear_entities,
    register_vehicle_entity,
)
from autoware_carla_scenario.traffic import (
    NullTrafficBackend,
    PedestrianParameters,
    Spread,
    TrafficContext,
    TrafficManagerBackend,
    TrafficParameterError,
    VehicleParameters,
)
from autoware_carla_scenario.traffic.parameters import (
    record_run_parameters,
    run_parameters,
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

    def set_autopilot(self, enabled: bool, port: int) -> None:
        pass


class _World:
    def __init__(self, actors: List[_Actor]) -> None:
        self.actors = actors

    def get_actors(self) -> "_World":
        return self

    def filter(self, pattern: str) -> List[_Actor]:
        return list(self.actors)


class _Entity:
    def __init__(self, actor: _Actor) -> None:
        self.actor = actor


@pytest.fixture(autouse=True)
def _forget() -> Iterator[None]:
    yield
    clear_entities()
    record_run_parameters(VehicleParameters(), PedestrianParameters())


def _started(
    vehicles: VehicleParameters, count: int = 3
) -> tuple[TrafficManagerBackend, _Client, _World]:
    client = _Client()
    backend = TrafficManagerBackend(client=client, vehicles=vehicles)
    world = _World([_Actor(i) for i in range(1, count + 1)])
    backend.prepare(TrafficContext(client=client, random_seed=0))
    backend.start(world)
    return backend, client, world


def _speed_calls(client: _Client) -> list[tuple[int, float]]:
    return [
        (args[0].id, args[1])
        for args in client.tm.called("vehicle_percentage_speed_difference")
    ]


# ---------------------------------------------------------------------------
# What the action takes
# ---------------------------------------------------------------------------


def test_an_action_that_changes_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="sets no vehicle, pedestrian or bicycle"):
        TrafficParametersAction(VehicleParameters())


def test_an_entity_takes_vehicle_parameters_only() -> None:
    with pytest.raises(ValueError, match="an entity is a vehicle"):
        TrafficParametersAction(
            pedestrians=PedestrianParameters(walking_speed_mps=Spread(1.0)),
            entity_name="npc1",
        )


def test_an_entity_takes_no_bicycle_parameters() -> None:
    from autoware_carla_scenario.traffic.parameters import BicycleParameters

    with pytest.raises(ValueError, match="an entity is a vehicle"):
        TrafficParametersAction(
            bicycles=BicycleParameters(decel=Spread(3.0)), entity_name="npc1"
        )


def test_sumo_changes_a_bicycle_parameter_but_not_its_speed_during_a_run() -> None:
    from autoware_carla_scenario.traffic.parameters import BicycleParameters
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    sumo = SumoTrafficBackend()
    TrafficParametersAction(
        bicycles=BicycleParameters(decel=Spread(3.0))
    ).set_traffic_backend(sumo)
    with pytest.raises(
        TrafficParameterError,
        match="max_speed_mps can only be set before the run, in its config",
    ):
        TrafficParametersAction(
            bicycles=BicycleParameters(max_speed_mps=Spread(4.0))
        ).set_traffic_backend(sumo)
    with pytest.raises(TrafficParameterError, match="traffic.bicycles"):
        TrafficParametersAction(
            bicycles=BicycleParameters(decel=Spread(3.0))
        ).set_traffic_backend(TrafficManagerBackend())


def test_a_field_the_backend_cannot_change_during_a_run_is_refused() -> None:
    action = TrafficParametersAction(
        vehicles=VehicleParameters(reaction_time_s=Spread(1.5))
    )
    with pytest.raises(
        TrafficParameterError,
        match="'traffic_manager' traffic backend cannot change during a run: "
        "traffic.vehicles.reaction_time_s can only be used with the 'sumo'",
    ):
        action.set_traffic_backend(TrafficManagerBackend())
    walking = TrafficParametersAction(
        pedestrians=PedestrianParameters(walking_speed_mps=Spread(1.0))
    )
    with pytest.raises(TrafficParameterError, match="traffic.pedestrians"):
        walking.set_traffic_backend(TrafficManagerBackend())
    with pytest.raises(TrafficParameterError, match="'none'"):
        TrafficParametersAction(
            VehicleParameters(speed_factor=Spread(1.0))
        ).set_traffic_backend(NullTrafficBackend())


def test_sumo_cannot_change_the_red_light_rule_during_a_run() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    sumo = SumoTrafficBackend()
    TrafficParametersAction(
        pedestrians=PedestrianParameters(walking_speed_mps=Spread(1.0))
    ).set_traffic_backend(sumo)
    with pytest.raises(
        TrafficParameterError,
        match="red_light_violation_s can only be set before the run, in its config",
    ):
        TrafficParametersAction(
            pedestrians=PedestrianParameters(red_light_violation_s=Spread(5.0))
        ).set_traffic_backend(sumo)


# ---------------------------------------------------------------------------
# The whole run's traffic
# ---------------------------------------------------------------------------


def test_every_vehicle_draws_the_changed_fields_again() -> None:
    backend, client, world = _started(
        VehicleParameters(speed_factor=Spread(0.9), min_gap_m=Spread(3.0))
    )
    assert len(_speed_calls(client)) == 3
    client.tm.calls.clear()

    action = TrafficParametersAction(VehicleParameters(speed_factor=Spread(0.5)))
    action.set_traffic_backend(backend)
    action.execute(world)  # type: ignore[arg-type]

    # Each vehicle drew the speed again; the gap it kept.
    assert sorted(_speed_calls(client)) == [
        (1, pytest.approx(50.0)),
        (2, pytest.approx(50.0)),
        (3, pytest.approx(50.0)),
    ]
    assert client.tm.called("distance_to_leading_vehicle") == []
    # The run's parameters are merged field by field.
    assert backend.vehicle_parameters == VehicleParameters(
        speed_factor=Spread(0.5), min_gap_m=Spread(3.0)
    )
    # ... which is what the ODD's probes read from now on.
    recorded = run_parameters()
    assert recorded is not None and recorded[0] == backend.vehicle_parameters


def test_a_vehicle_that_joins_later_draws_from_the_new_parameters() -> None:
    backend, client, world = _started(VehicleParameters(speed_factor=Spread(0.9)))
    action = TrafficParametersAction(VehicleParameters(speed_factor=Spread(0.6)))
    action.set_traffic_backend(backend)
    action.execute(world)  # type: ignore[arg-type]
    client.tm.calls.clear()

    late = _Actor(9)
    world.actors.append(late)
    backend._autopilot(late)  # type: ignore[arg-type]
    assert _speed_calls(client) == [(9, pytest.approx(40.0))]


def test_the_next_run_starts_from_its_own_parameters() -> None:
    configured = VehicleParameters(speed_factor=Spread(0.9))
    backend, _client, world = _started(configured)
    action = TrafficParametersAction(VehicleParameters(speed_factor=Spread(0.5)))
    action.set_traffic_backend(backend)
    action.execute(world)  # type: ignore[arg-type]
    assert backend.vehicle_parameters != configured

    backend.close()
    assert backend.vehicle_parameters == configured


# ---------------------------------------------------------------------------
# One vehicle
# ---------------------------------------------------------------------------


def test_an_entity_draws_alone_and_the_run_keeps_its_parameters() -> None:
    configured = VehicleParameters(speed_factor=Spread(0.9))
    backend, client, world = _started(configured)
    register_vehicle_entity("npc2", _Entity(world.actors[1]))  # type: ignore[arg-type]
    client.tm.calls.clear()

    action = TrafficParametersAction(
        VehicleParameters(speed_factor=Spread(1.3), ignore_lights=Spread(1.0)),
        entity_name="npc2",
    )
    action.set_traffic_backend(backend)
    action.execute(world)  # type: ignore[arg-type]

    assert _speed_calls(client) == [(2, pytest.approx(-30.0))]
    (lights,) = client.tm.called("ignore_lights_percentage")
    assert lights[0].id == 2 and lights[1] == pytest.approx(100.0)
    assert backend.vehicle_parameters == configured


def test_an_entity_not_in_the_run_is_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend, client, world = _started(VehicleParameters(speed_factor=Spread(0.9)))
    client.tm.calls.clear()
    action = TrafficParametersAction(
        VehicleParameters(speed_factor=Spread(1.3)), entity_name="nobody"
    )
    action.set_traffic_backend(backend)
    with caplog.at_level("WARNING"):
        action.execute(world)  # type: ignore[arg-type]
    assert "'nobody' not found" in caplog.text
    assert client.tm.calls == []


# ---------------------------------------------------------------------------
# From the scenario editor
# ---------------------------------------------------------------------------


def test_the_editors_fields_become_parameters() -> None:
    from autoware_carla_scenario.traffic.authoring import (
        authored_pedestrian_parameters,
        authored_vehicle_parameters,
    )

    assert authored_vehicle_parameters() is None
    assert authored_vehicle_parameters(
        speed_factor_mean=0.8, speed_factor_std=0.1, lane_changes=False
    ) == VehicleParameters(speed_factor=Spread(0.8, 0.1), lane_changes=False)
    assert authored_pedestrian_parameters(
        pedestrian_min_gap_m_mean=0.5
    ) == PedestrianParameters(min_gap_m=Spread(0.5))
    from autoware_carla_scenario.traffic.authoring import (
        authored_bicycle_parameters,
    )
    from autoware_carla_scenario.traffic.parameters import BicycleParameters

    assert authored_bicycle_parameters() is None
    assert authored_bicycle_parameters(
        bicycle_decel_mean=3.5, bicycle_decel_std=0.4
    ) == BicycleParameters(decel=Spread(3.5, 0.4))
    with pytest.raises(ValueError, match="a std needs a mean"):
        authored_vehicle_parameters(decel_std=1.0)


# ---------------------------------------------------------------------------
# A value or a spread
# ---------------------------------------------------------------------------


def test_a_plain_value_is_a_spread_of_no_width() -> None:
    assert VehicleParameters(
        speed_factor=0.9, min_gap_m=3, reaction_time_s=Spread(1.5, 0.3)
    ) == VehicleParameters(
        speed_factor=Spread(0.9),
        min_gap_m=Spread(3.0),
        reaction_time_s=Spread(1.5, 0.3),
    )
    assert PedestrianParameters(walking_speed_mps=1.2) == PedestrianParameters(
        walking_speed_mps=Spread(1.2)
    )
    with pytest.raises(ValueError, match="ignore_lights must be within"):
        VehicleParameters(ignore_lights=1.5)


def test_an_action_sets_a_value_every_vehicle_takes() -> None:
    backend, client, world = _started(VehicleParameters(speed_factor=Spread(0.9, 0.1)))
    client.tm.calls.clear()
    action = TrafficParametersAction(VehicleParameters(speed_factor=0.5))
    action.set_traffic_backend(backend)
    action.execute(world)  # type: ignore[arg-type]
    assert sorted(_speed_calls(client)) == [
        (1, pytest.approx(50.0)),
        (2, pytest.approx(50.0)),
        (3, pytest.approx(50.0)),
    ]
