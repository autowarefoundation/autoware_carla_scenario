"""Traffic parameters drawn from an ODD, and drawn again by each road user.

An ODD bounds the mean and the spread of a traffic parameter
(:mod:`autoware_carla_scenario.odd.traffic_probes`); a sweep draws both per
case; every vehicle (or pedestrian) of the run then draws its own value from
that spread (:class:`~autoware_carla_scenario.traffic.parameters.Spread`).
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from autoware_carla_scenario.odd import OddAttribute, OddDefinition, OddSampler
from autoware_carla_scenario.odd.backend import BUILT_IN_PROBES, DEFAULT_KNOBS
from autoware_carla_scenario.odd.traffic_probes import (
    PARAMETER_RANGES,
    TRAFFIC_PROBES,
)
from autoware_carla_scenario.traffic import (
    PEDESTRIAN_PARAMETER_BACKENDS,
    VEHICLE_PARAMETER_BACKENDS,
    PedestrianParameters,
    TrafficContext,
    TrafficManagerBackend,
    VehicleParameters,
)
from autoware_carla_scenario.traffic.parameters import (
    VEHICLE_BOUNDS,
    Spread,
    record_run_parameters,
)


def _value(overrides: list[str], key: str) -> float:
    for override in overrides:
        name, _, value = override.partition("=")
        if name == key:
            return float(value)
    raise AssertionError(f"{key} not in {overrides}")


# ---------------------------------------------------------------------------
# The spread
# ---------------------------------------------------------------------------


def test_a_spread_draws_a_normal_cut_to_its_bounds() -> None:
    rng = random.Random(0)
    draws = [Spread(0.9, 0.2).draw(rng, (0.0, 1.0)) for _ in range(2000)]
    assert all(0.0 <= d <= 1.0 for d in draws)
    # Cut at 1.0, half a std above the mean: a truncated normal's mean is
    # 0.9 - 0.2 * pdf(0.5) / cdf(0.5) = 0.798.
    assert statistics.mean(draws) == pytest.approx(0.798, abs=0.015)
    assert statistics.stdev(draws) > 0.1


def test_a_spread_of_no_width_is_its_mean() -> None:
    assert Spread(1.2).draw(random.Random(0), (0.0, 5.0)) == 1.2


def test_a_number_or_a_mean_and_std_is_read_as_a_spread() -> None:
    parameters = VehicleParameters.from_mapping(
        {
            "speed_factor": {"mean": 0.9, "std": 0.1},
            "min_gap_m": 2.5,
            "decel": {"mean": None, "std": None},  # the Hydra groups' "unset"
        }
    )
    assert parameters.speed_factor == Spread(0.9, 0.1)
    assert parameters.min_gap_m == Spread(2.5)
    assert parameters.decel is None
    assert parameters.describe() == {
        "speed_factor": {"mean": 0.9, "std": 0.1},
        "min_gap_m": {"mean": 2.5, "std": 0.0},
    }


@pytest.mark.parametrize(
    "mapping, match",
    [
        ({"speed_factor": {"std": 0.1}}, "a std needs a mean"),
        ({"speed_factor": {"mean": 1.0, "sd": 0.1}}, "unknown key"),
        ({"speed_factor": {"mean": 1.0, "std": -0.1}}, "std must not be negative"),
        ({"ignore_lights": {"mean": 1.5}}, "ignore_lights must be within"),
    ],
)
def test_an_impossible_spread_is_refused(mapping: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        VehicleParameters.from_mapping(mapping)


# ---------------------------------------------------------------------------
# Each vehicle draws its own
# ---------------------------------------------------------------------------


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
        self._actors = actors

    def get_actors(self) -> "_World":
        return self

    def filter(self, pattern: str) -> List[_Actor]:
        return list(self._actors)


def _speed_differences(seed: int) -> list[float]:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client,
        vehicles=VehicleParameters(
            speed_factor=Spread(0.9, 0.1), ignore_lights=Spread(0.2, 0.1)
        ),
    )
    backend.prepare(TrafficContext(client=client, random_seed=seed))
    backend.start(_World([_Actor(i) for i in range(50)]))
    assert len(client.tm.called("ignore_lights_percentage")) == 50
    return [args[1] for args in client.tm.called("vehicle_percentage_speed_difference")]


def test_each_vehicle_draws_its_own_value_from_the_spread() -> None:
    differences = _speed_differences(seed=3)
    factors = [1.0 - d / 100.0 for d in differences]
    assert len(set(factors)) == 50
    assert 0.85 < statistics.mean(factors) < 0.95
    assert 0.05 < statistics.stdev(factors) < 0.15
    # The same seed, the same drivers.
    assert _speed_differences(seed=3) == differences
    assert _speed_differences(seed=4) != differences


def test_a_vehicle_keeps_its_draws_when_resumed() -> None:
    client = _Client()
    backend = TrafficManagerBackend(
        client=client, vehicles=VehicleParameters(speed_factor=Spread(0.9, 0.1))
    )
    actor = _Actor(1)
    backend.prepare(TrafficContext(client=client, random_seed=0))
    backend.start(_World([actor]))
    backend.release_actor(_World([actor]), actor)  # type: ignore[arg-type]
    backend.resume_actor(_World([actor]), actor)  # type: ignore[arg-type]
    assert len(client.tm.called("vehicle_percentage_speed_difference")) == 1


# ---------------------------------------------------------------------------
# The ODD's probes and knobs
# ---------------------------------------------------------------------------


def test_every_numeric_parameter_has_a_mean_and_a_std_probe() -> None:
    numeric = {("vehicles", f) for f in VEHICLE_BOUNDS} | {
        ("pedestrians", f) for f in PEDESTRIAN_PARAMETER_BACKENDS
    }
    assert {key for key in PARAMETER_RANGES if key[0] != "bicycles"} == numeric
    assert {f for _, f in PARAMETER_RANGES if _ == "vehicles"} <= set(
        VEHICLE_PARAMETER_BACKENDS
    )
    for space, field in numeric:
        prefix = "vehicle" if space == "vehicles" else "pedestrian"
        for statistic in ("mean", "std"):
            name = f"{prefix}_{field}_{statistic}"
            probe, _unit = BUILT_IN_PROBES[name]
            knob = DEFAULT_KNOBS[probe]
            assert knob.key == f"traffic.{space}.{field}.{statistic}"


def test_the_probes_read_what_the_run_set() -> None:
    record_run_parameters(
        VehicleParameters(speed_factor=Spread(0.9, 0.1)),
        PedestrianParameters(walking_speed_mps=Spread(1.2, 0.2)),
    )
    try:
        assert TRAFFIC_PROBES["vehicle_speed_factor_mean"][0](None) == 0.9
        assert TRAFFIC_PROBES["vehicle_speed_factor_std"][0](None) == 0.1
        assert TRAFFIC_PROBES["pedestrian_walking_speed_mps_std"][0](None) == 0.2
        # Not set: a missing value, as an ODD's open world reads it.
        assert TRAFFIC_PROBES["vehicle_decel_mean"][0](None) is None
    finally:
        record_run_parameters(VehicleParameters(), PedestrianParameters())


def _behaviour_odd() -> OddDefinition:
    return OddDefinition(
        "behaviour",
        [
            OddAttribute(
                "behaviour.speed_factor",
                TRAFFIC_PROBES["vehicle_speed_factor_mean"][0],
                buckets=[0.8, 0.9, 1.0],
            ),
            OddAttribute(
                "behaviour.speed_spread",
                TRAFFIC_PROBES["vehicle_speed_factor_std"][0],
                buckets=[0.0, 0.1, 0.2],
            ),
            OddAttribute(
                "behaviour.walking_speed",
                TRAFFIC_PROBES["pedestrian_walking_speed_mps_mean"][0],
                unit="m/s",
                buckets=[1.0, 1.5],
            ),
        ],
    )


def test_an_odd_draws_each_cases_mean_and_spread() -> None:
    cases = OddSampler(_behaviour_odd(), seed=0).sample(20)
    means = [_value(c.overrides, "traffic.vehicles.speed_factor.mean") for c in cases]
    stds = [_value(c.overrides, "traffic.vehicles.speed_factor.std") for c in cases]
    walking = [
        _value(c.overrides, "traffic.pedestrians.walking_speed_mps.mean") for c in cases
    ]
    assert all(0.8 <= m <= 1.0 for m in means) and len(set(means)) > 1
    assert all(0.0 <= s <= 0.2 for s in stds) and len(set(stds)) > 1
    assert all(1.0 <= w <= 1.5 for w in walking)


def test_a_drawn_case_reaches_the_backend(tmp_path: Path) -> None:
    from hydra import compose, initialize_config_dir

    from autoware_carla_scenario.examples import conf as conf_package
    from autoware_carla_scenario.examples.run import build_traffic_backend

    (case,) = OddSampler(_behaviour_odd(), seed=1).sample(1)
    overrides = [
        o
        for o in case.overrides
        if o.startswith("traffic.") and not o.startswith("traffic.pedestrians")
    ]
    conf_dir = str(Path(conf_package.__file__).parent.resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    backend = build_traffic_backend(cfg)
    spread = backend.vehicle_parameters.speed_factor
    assert spread == Spread(
        _value(overrides, "traffic.vehicles.speed_factor.mean"),
        _value(overrides, "traffic.vehicles.speed_factor.std"),
    )


def test_every_bicycle_parameter_has_a_probe_and_a_knob() -> None:
    from autoware_carla_scenario.traffic.parameters import BICYCLE_BOUNDS

    assert {f for space, f in PARAMETER_RANGES if space == "bicycles"} == set(
        BICYCLE_BOUNDS
    )
    for field in BICYCLE_BOUNDS:
        for statistic in ("mean", "std"):
            probe, _unit = BUILT_IN_PROBES[f"bicycle_{field}_{statistic}"]
            knob = DEFAULT_KNOBS[probe]
            assert knob.key == f"traffic.bicycles.{field}.{statistic}"


def test_the_bicycle_probes_read_what_the_run_set() -> None:
    from autoware_carla_scenario.traffic.parameters import BicycleParameters

    record_run_parameters(VehicleParameters(), PedestrianParameters())
    assert TRAFFIC_PROBES["bicycle_max_speed_mps_mean"][0](None) is None
    record_run_parameters(
        VehicleParameters(),
        PedestrianParameters(),
        BicycleParameters(max_speed_mps=Spread(4.0, 0.5)),
    )
    try:
        assert TRAFFIC_PROBES["bicycle_max_speed_mps_mean"][0](None) == 4.0
        assert TRAFFIC_PROBES["bicycle_max_speed_mps_std"][0](None) == 0.5
        assert TRAFFIC_PROBES["bicycle_accel_mean"][0](None) is None
    finally:
        record_run_parameters(VehicleParameters(), PedestrianParameters())
    assert TRAFFIC_PROBES["bicycle_max_speed_mps_mean"][0](None) is None


def test_a_drawn_bicycle_case_writes_the_bicycle_parameters() -> None:
    from autoware_carla_scenario.traffic.config import TrafficConfig

    probe, unit = BUILT_IN_PROBES["bicycle_max_speed_mps_mean"]
    attribute = OddAttribute("riders.speed", probe, unit=unit, buckets=[2.0, 5.0, 8.0])
    sampler = OddSampler(OddDefinition("riders", [attribute]), seed=3)
    (case,) = sampler.sample(1)
    speed = _value(case.overrides, "traffic.bicycles.max_speed_mps.mean")
    assert 2.0 <= speed <= 8.0
    config = TrafficConfig.from_mapping(
        {"backend": "sumo", "bicycles": {"max_speed_mps": {"mean": speed, "std": None}}}
    )
    assert config.bicycles.max_speed_mps == Spread(speed)
