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
    PARAMETER_BACKENDS,
    NullTrafficBackend,
    TrafficConfig,
    TrafficContext,
    TrafficManagerBackend,
    TrafficParameterError,
    TrafficParameters,
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
    assert set(PARAMETER_BACKENDS) == set(
        TrafficParameters.__dataclass_fields__  # type: ignore[attr-defined]
    )


def test_nothing_set_is_every_default() -> None:
    assert TrafficParameters().set_fields() == []
    assert TrafficParameters().describe() == {}


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
    ],
)
def test_an_impossible_value_is_refused(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        TrafficParameters(**{field: value})  # type: ignore[arg-type]


def test_a_mapping_with_an_unknown_key_is_refused() -> None:
    with pytest.raises(ValueError, match="reaction_time"):
        TrafficParameters.from_mapping({"reaction_time": 1.0})


def test_the_traffic_config_reads_its_parameters_node() -> None:
    config = TrafficConfig.from_mapping(
        {
            "backend": "sumo",
            "options": {},
            "parameters": {"speed_factor": 0.8, "follow_visibility": None},
        }
    )
    assert config.parameters == TrafficParameters(speed_factor=0.8)
    assert TrafficConfig.from_mapping({}).parameters == TrafficParameters()


def test_restricting_keeps_only_what_is_supported() -> None:
    parameters = TrafficParameters(speed_factor=0.8, reaction_time_s=1.5)
    assert parameters.restricted_to({"speed_factor"}) == TrafficParameters(
        speed_factor=0.8
    )


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_a_field_another_backend_supports_is_refused_naming_it() -> None:
    parameters = TrafficParameters(
        speed_factor=0.9, reaction_time_s=1.5, follow_visibility=True
    )
    with pytest.raises(TrafficParameterError) as raised:
        TrafficManagerBackend(parameters=parameters)
    message = str(raised.value)
    assert "'traffic_manager' traffic backend does not support" in message
    assert (
        "traffic.parameters.reaction_time_s can only be used with the 'sumo' backend"
        in message
    )
    assert "traffic.parameters.follow_visibility can only be used with" in message
    # A field it does support is not named.
    assert "speed_factor" not in message


def test_a_backend_that_reads_no_parameters_refuses_every_one() -> None:
    with pytest.raises(
        TrafficParameterError,
        match="speed_factor can only be used with the 'traffic_manager' or "
        "'sumo' backends",
    ):
        NullTrafficBackend().set_parameters(TrafficParameters(speed_factor=1.0))
    # Nothing set is nothing refused.
    NullTrafficBackend().set_parameters(TrafficParameters())


def test_building_a_backend_checks_its_parameters() -> None:
    with pytest.raises(TrafficParameterError, match="'sumo' backend"):
        build_backend("traffic_manager", {}, TrafficParameters(follow_visibility=False))
    backend = build_backend("traffic_manager", {}, TrafficParameters(min_gap_m=3.0))
    assert backend.parameters.min_gap_m == 3.0


def test_each_backend_supports_what_the_table_says() -> None:
    from autoware_carla_scenario.traffic.sumo.backend import SumoTrafficBackend

    assert TrafficManagerBackend().supported_parameters() == {
        "speed_factor",
        "min_gap_m",
        "lane_changes",
        "ignore_lights",
        "ignore_signs",
        "ignore_vehicles",
        "ignore_walkers",
        "random_lane_changes",
    }
    assert SumoTrafficBackend().supported_parameters() == {
        "speed_factor",
        "min_gap_m",
        "lane_changes",
        "accel",
        "decel",
        "reaction_time_s",
        "imperfection",
        "follow_visibility",
    }
    assert NullTrafficBackend().supported_parameters() == set()


def test_every_hand_written_field_list_names_every_field() -> None:
    """Codon has no field reflection, so the methods spell the fields out."""
    fields = set(TrafficParameters.__dataclass_fields__)  # type: ignore[attr-defined]
    everything = TrafficParameters(
        speed_factor=1.0,
        min_gap_m=1.0,
        lane_changes=True,
        ignore_lights=0.5,
        ignore_signs=0.5,
        ignore_vehicles=0.5,
        ignore_walkers=0.5,
        random_lane_changes=0.5,
        accel=1.0,
        decel=1.0,
        reaction_time_s=1.0,
        imperfection=0.5,
        follow_visibility=True,
    )
    assert set(everything.set_fields()) == fields
    assert set(everything.describe()) == fields
    assert everything.restricted_to(fields) == everything


# ---------------------------------------------------------------------------
# What the TrafficManager makes of them
# ---------------------------------------------------------------------------


def test_the_traffic_manager_sets_its_global_parameters_when_prepared() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client,
        parameters=TrafficParameters(speed_factor=0.8, min_gap_m=3.5),
    )
    backend.prepare(TrafficContext(client=client, random_seed=1))
    (difference,) = client.tm.called("global_percentage_speed_difference")
    assert difference[0] == pytest.approx(20.0)
    assert client.tm.called("set_global_distance_to_leading_vehicle") == [(3.5,)]
    assert backend.describe()["parameters"] == {"speed_factor": 0.8, "min_gap_m": 3.5}


def test_the_traffic_manager_sets_each_vehicles_parameters_on_autopilot() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client,
        parameters=TrafficParameters(
            lane_changes=False,
            ignore_lights=0.25,
            ignore_signs=0.5,
            ignore_vehicles=0.0,
            ignore_walkers=1.0,
            random_lane_changes=0.1,
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
    assert {name for name, _ in client.tm.calls} == {
        "set_synchronous_mode",
        "set_random_device_seed",
    }
    assert "parameters" not in backend.describe()
