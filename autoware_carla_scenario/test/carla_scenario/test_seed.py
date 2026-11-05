"""The run's seed: one number that every random draw of a run follows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import pytest
import typesafe_carla.carla as carla
from omegaconf import OmegaConf

from autoware_carla_scenario import (
    AdversarialTrafficAction,
    AdversityFeasibleCondition,
    AdversityKind,
    AdversityLocation,
    BaseScenario,
    EgoConfig,
    JaywalkAction,
    NadeAdversity,
    SpawnTransform,
    TrafficSourceAction,
)
from autoware_carla_scenario.driver import DriverLidarConfig
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.seed import (
    DEFAULT_SEED,
    resolve_seed,
    run_seed,
    set_run_seed,
)
from autoware_carla_scenario.sensor.carla_lidar import CarlaLidarSensor
from autoware_carla_scenario.sweeper.expand import expand_config

_NOT_JUNCTION = [{"type": "not", "constraint": {"type": "is_junction"}}]


@pytest.fixture(autouse=True)
def _restore_run_seed() -> Iterator[None]:
    previous = set_run_seed(DEFAULT_SEED)
    yield
    set_run_seed(previous)


# ---------------------------------------------------------------------------
# The seed itself
# ---------------------------------------------------------------------------


def test_a_seed_left_unset_is_the_runs() -> None:
    assert resolve_seed(None) == DEFAULT_SEED
    set_run_seed(42)
    assert run_seed() == 42
    assert resolve_seed(None) == 42
    assert resolve_seed(7) == 7


def test_setting_the_seed_returns_the_one_it_replaces() -> None:
    assert set_run_seed(3) == DEFAULT_SEED
    assert set_run_seed(4) == 3


def test_a_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        set_run_seed(-1)


# ---------------------------------------------------------------------------
# What draws follows it
# ---------------------------------------------------------------------------


def _draws(rng: object) -> list[float]:
    return [rng.random() for _ in range(5)]  # type: ignore[attr-defined]


def test_actions_and_conditions_with_no_seed_draw_from_the_runs() -> None:
    set_run_seed(11)
    unseeded = [
        TrafficSourceAction(_NOT_JUNCTION)._rng,  # noqa: SLF001
        AdversityFeasibleCondition(
            "npc", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="f"
        )._rng,  # noqa: SLF001
        JaywalkAction("p1", AdversityLocation.ANY)._feasible._rng,  # noqa: SLF001
    ]
    set_run_seed(0)
    seeded = [
        TrafficSourceAction(_NOT_JUNCTION, seed=11)._rng,  # noqa: SLF001
        AdversityFeasibleCondition(
            "npc", AdversityKind.JAYWALKING, AdversityLocation.ANY, label="f", seed=11
        )._rng,  # noqa: SLF001
        JaywalkAction("p1", AdversityLocation.ANY, seed=11)._feasible._rng,  # noqa: SLF001
    ]
    for a, b in zip(unseeded, seeded):
        assert _draws(a) == _draws(b)


def test_an_explicit_seed_wins_over_the_runs() -> None:
    set_run_seed(11)
    explicit = TrafficSourceAction(_NOT_JUNCTION, seed=5)._rng  # noqa: SLF001
    set_run_seed(0)
    again = TrafficSourceAction(_NOT_JUNCTION, seed=5)._rng  # noqa: SLF001
    assert _draws(explicit) == _draws(again)


def test_adversarial_traffic_samples_from_the_runs_seed() -> None:
    adversities = [NadeAdversity(AdversityKind.JAYWALKING, AdversityLocation.ANY, 0.5)]
    set_run_seed(9)
    unseeded = AdversarialTrafficAction(adversities)._sampler  # noqa: SLF001
    seeded = AdversarialTrafficAction(adversities, seed=9)._sampler  # noqa: SLF001
    assert _draws(unseeded._draws) == _draws(seeded._draws)  # noqa: SLF001


def test_the_lidar_weather_noise_takes_the_runs_seed_when_it_attaches() -> None:
    lidar = CarlaLidarSensor(DriverLidarConfig())
    set_run_seed(23)
    blueprint = SimpleNamespace(
        set_attribute=lambda name, value: None, has_attribute=lambda name: True
    )
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda bp, transform, attach_to: SimpleNamespace(
            listen=lambda cb: None
        ),
    )
    lidar.attach(world, SimpleNamespace(type_id="vehicle.lincoln.mkz"))
    assert lidar._noise_seed == 23  # noqa: SLF001
    assert CarlaLidarSensor(DriverLidarConfig(noise_seed=4))._noise_seed == 4  # noqa: SLF001


# ---------------------------------------------------------------------------
# The config's seed
# ---------------------------------------------------------------------------


def test_the_config_seed_is_read_from_the_top_level() -> None:
    assert run.config_seed(OmegaConf.create({"seed": 12})) == 12
    assert run.config_seed(OmegaConf.create({"seed": None})) is None
    assert run.config_seed(OmegaConf.create({})) is None


def test_a_whole_float_config_seed_is_that_seed() -> None:
    assert run.config_seed(OmegaConf.create({"seed": 2.0})) == 2
    assert run.config_seed(OmegaConf.create({"seed": "7"})) == 7


# What int() alone took (1.9 and true as seed 1), or refused only once the
# scenario was built (-1), with a traceback (#1437).
_NOT_SEEDS = [1.9, True, False, "abc", "1.5", "", -1, float("nan"), float("inf")]


@pytest.mark.parametrize("value", _NOT_SEEDS)
def test_a_config_seed_that_is_no_seed_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match=r"^seed must "):
        run.config_seed(OmegaConf.create({"seed": value}))


@pytest.mark.parametrize("value", _NOT_SEEDS)
def test_expanding_a_config_refuses_a_seed_that_is_no_seed(value: object) -> None:
    # A config without a sweep too: every case would run with it.
    with pytest.raises(ValueError, match=r"^seed must "):
        expand_config(OmegaConf.create({"seed": value}))
    with pytest.raises(ValueError, match=r"^seed must "):
        _odd_cases({"seed": value, "sweep": {"odd_sample": {"count": 2}}})


def _shipped(overrides: list[str]) -> object:
    from .test_video_removed import _shipped as compose

    return compose(overrides)


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("seed=1.9", "seed"),
        ("seed=true", "seed"),
        ("seed=abc", "seed"),
        ("seed=-1", "seed"),
        ("+sweep.odd_sample.seed=1.5", "sweep.odd_sample.seed"),
        ("+sweep.odd_sample.seed=-3", "sweep.odd_sample.seed"),
        ("+sweep.odd_sample.count=2.7", "sweep.odd_sample.count"),
        ("+sweep.odd_sample.count=0", "sweep.odd_sample.count"),
        ("+sweep.odd_sample.count=true", "sweep.odd_sample.count"),
    ],
)
def test_a_run_is_refused_for_a_seed_or_count_before_anything_is_built(
    override: str, key: str
) -> None:
    # A single run (run_scenario), the lanelet sweeper (before forking) and
    # a local or SLURM batch all refuse it in check_batch_configs, naming
    # the key, rather than once build_scenario sets the seed.
    cfg = _shipped([override])
    with pytest.raises(run.RunConfigError, match=rf"^{key} must "):
        run.check_run_config(cfg)  # type: ignore[arg-type]
    good = _shipped([])
    with pytest.raises(run.RunConfigError, match=rf"^bad: {key} must "):
        run.check_batch_configs([good, cfg], ["good", "bad"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "overrides",
    [
        ["seed=2.0"],
        ["seed=null"],
        ["+sweep.odd_sample.seed=4", "+sweep.odd_sample.count=3.0"],
        ["+sweep.odd_sample.seed=null", "+sweep.odd_sample.count=null"],
    ],
)
def test_a_run_with_whole_seeds_and_counts_is_accepted(overrides: list[str]) -> None:
    run.check_run_config(_shipped(overrides))  # type: ignore[arg-type]


class _Scenario(BaseScenario):
    def setup(self) -> None:
        pass

    def is_done(self) -> bool:
        return True


def test_background_traffic_with_no_seed_follows_the_scenarios() -> None:
    def placed_seeds(background: dict[str, object], random_seed: int) -> list[object]:
        ego = EgoConfig(
            spawn_location=SpawnTransform(
                carla.Transform(carla.Location(x=0, y=0, z=0))
            )
        )
        scenario = _Scenario(ego, random_seed=random_seed)
        cfg = OmegaConf.create({"background_traffic": background})
        run.add_background_traffic(cfg, scenario)
        return [
            action._rng.getstate()  # noqa: SLF001
            for action in scenario._init_actions + scenario._pre_tick_actions  # noqa: SLF001
            if isinstance(action, TrafficSourceAction)
        ]

    background: dict[str, object] = {
        "enabled": True,
        "seed": None,
        "source": {
            "constraints": _NOT_JUNCTION,
            "initial_vehicles": 2,
            "vehicles_per_minute": 1.0,
        },
        "sink": {},
    }
    followed = placed_seeds(background, 31)
    assert followed == placed_seeds({**background, "seed": 31}, 0)
    assert followed != placed_seeds(background, 32)


def _odd_cases(cfg: dict[str, object]) -> list[list[str]]:
    return expand_config(OmegaConf.create({"odd": "default", **cfg}))


def test_a_sweep_draws_its_odd_cases_from_the_runs_seed() -> None:
    sweep = {"odd_sample": {"count": 4}}
    from_run = _odd_cases({"seed": 5, "sweep": sweep})
    assert from_run == _odd_cases({"sweep": {"odd_sample": {"count": 4, "seed": 5}}})
    assert from_run != _odd_cases({"seed": 6, "sweep": sweep})
    # The sample's own seed wins.
    assert (
        _odd_cases({"seed": 6, "sweep": {"odd_sample": {"count": 4, "seed": 5}}})
        == from_run
    )


# A component's own seed, as its config dataclass holds it (#1445).
@pytest.mark.parametrize("value", [1.9, True, False, "7", "abc"])
def test_check_seed_refuses_what_is_no_whole_number(value: object) -> None:
    from autoware_carla_scenario.seed import check_seed

    with pytest.raises(ValueError, match=r"^x\.seed must be a whole number"):
        check_seed("x.seed", value)  # type: ignore[arg-type]


def test_check_seed_refuses_a_negative_or_non_finite_seed() -> None:
    from autoware_carla_scenario.seed import check_seed

    with pytest.raises(ValueError, match=r"^x\.seed must not be negative"):
        check_seed("x.seed", -1)
    with pytest.raises(ValueError, match=r"^x\.seed must be finite"):
        check_seed("x.seed", float("nan"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(960, 960), (960.0, 960), (1.2e6, 1_200_000), (-3.0, -3), (2**62, 2**62)],
)
def test_check_whole_number_returns_an_int(value: object, expected: int) -> None:
    from autoware_carla_scenario.seed import check_whole_number

    number = check_whole_number("x.n", value)  # type: ignore[arg-type]
    assert type(number) is int
    assert number == expected


def test_check_whole_number_takes_a_numpy_integer() -> None:
    import numpy as np

    from autoware_carla_scenario.seed import check_whole_number

    for value in (np.int64(64), np.int32(64), np.uint16(64), np.float64(64.0)):
        number = check_whole_number("x.n", value)  # type: ignore[arg-type]
        assert type(number) is int
        assert number == 64


@pytest.mark.parametrize("value", [960.5, 64.5, True, False, "960", None, [1]])
def test_check_whole_number_refuses_what_is_no_whole_number(value: object) -> None:
    from autoware_carla_scenario.seed import check_whole_number

    with pytest.raises(ValueError, match=r"^x\.n must be a whole number"):
        check_whole_number("x.n", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_check_whole_number_refuses_what_is_not_finite(value: float) -> None:
    from autoware_carla_scenario.seed import check_whole_number

    with pytest.raises(ValueError, match=r"^x\.n must be finite"):
        check_whole_number("x.n", value)  # type: ignore[arg-type]


def test_check_seed_takes_a_whole_number() -> None:
    from autoware_carla_scenario.seed import check_seed

    assert check_seed("x.seed", 0) == 0
    assert check_seed("x.seed", 2.0) == 2  # type: ignore[arg-type]
    assert check_seed("x.seed", 2**62) == 2**62
