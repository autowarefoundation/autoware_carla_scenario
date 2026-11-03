"""``detect-no-3d-model`` searches as far as the runner does (issue #1192).

The tool used to cast its ray 5 m below the z estimate while the runner's
shipped ``entity.ground_projection_ray_distance_lower`` is 10 m, so it could
flag lanelets a run would project onto -- and sweeps exclude what it flags.
Its defaults and ``EntityConfig``'s must be the runner's composed ``entity.*``
values.  ``GroundProjectionConfig``'s API default is deliberately not: see
its docstring.  None of this needs CARLA.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from autoware_carla_scenario.coordinate.snap import GroundProjectionConfig
from autoware_carla_scenario.scenario_config import EntityConfig
from autoware_carla_scenario.tools import detect_no_3d_model_lanelets as tool

_CONF = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoware_carla_scenario"
    / "examples"
    / "conf"
)


@pytest.fixture(scope="module")
def runner_entity() -> tuple[float, float]:
    """The runner's composed ``(upper, lower)`` ground-projection range."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(_CONF.resolve()), version_base=None):
        cfg = compose(config_name="config")
    return (
        float(cfg.entity.ground_projection_ray_distance_upper),
        float(cfg.entity.ground_projection_ray_distance_lower),
    )


def test_cli_defaults_are_the_runners(runner_entity: tuple[float, float]) -> None:
    args = tool._build_parser().parse_args(["map.yaml"])
    assert (args.ray_upper, args.ray_lower) == runner_entity


def test_function_defaults_are_the_runners(
    runner_entity: tuple[float, float],
) -> None:
    params = inspect.signature(tool.detect_no_3d_model_lanelets).parameters
    assert (
        params["ray_distance_upper"].default,
        params["ray_distance_lower"].default,
    ) == runner_entity


def test_entity_config_defaults_are_the_shipped_yaml(
    runner_entity: tuple[float, float],
) -> None:
    entity = EntityConfig()
    assert (
        entity.ground_projection_ray_distance_upper,
        entity.ground_projection_ray_distance_lower,
    ) == runner_entity


def test_ground_projection_config_default_stays_shallower(
    runner_entity: tuple[float, float],
) -> None:
    # Callers that pass no config (trajectory runs, walker placement) keep
    # their estimate on a miss; a deeper default could drop them onto a road
    # under a bridge instead.
    config = GroundProjectionConfig()
    assert config.ray_distance_upper == runner_entity[0]
    assert config.ray_distance_lower == 5.0 < runner_entity[1]


def test_explicit_ray_options_override_the_defaults() -> None:
    args = tool._build_parser().parse_args(
        ["map.yaml", "--ray-upper", "7.5", "--ray-lower", "20"]
    )
    assert (args.ray_upper, args.ray_lower) == (7.5, 20.0)
