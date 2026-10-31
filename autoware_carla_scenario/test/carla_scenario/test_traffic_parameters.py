"""The traffic parameters every backend shares, and the refusal of the ones a
backend cannot honour.

CARLA is faked, as in ``test_traffic_backend``: what is under test is which
TrafficManager calls the parameters become.  What SUMO makes of them is
pinned in ``test_traffic_sumo``.
"""

from __future__ import annotations

from typing import Any, List, Tuple

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
