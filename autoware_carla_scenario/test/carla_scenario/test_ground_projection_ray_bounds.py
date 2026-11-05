"""A ground-projection ray that cannot hit is refused up front (issue #1416).

The ray starts ``upper`` metres above the z estimate and searches
``upper + lower`` metres down.  A ray of no length, or a non-finite one, hits
nothing (and a negative range on one side means nothing): ``detect-no-3d-model``
would flag every lanelet and
write them all to the map's YAML, and a run would leave every pose at its
estimate.  ``GroundProjectionConfig``, the runner's ``entity.*`` keys and the
tool's ``--ray-upper``/``--ray-lower`` (and ``--timeout``) refuse them.  None
of this needs CARLA.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoware_carla_scenario.coordinate.snap import (
    GroundProjectionConfig,
    check_ray_distances,
)
from autoware_carla_scenario.tools import detect_no_3d_model_lanelets as tool

#: ``(upper, lower, the name the message gives)`` of rays that cannot hit.
_BAD_RAYS = [
    (-1.0, 10.0, "upper"),
    (5.0, -0.5, "lower"),
    (math.nan, 10.0, "upper"),
    (5.0, math.inf, "lower"),
    (-math.inf, 10.0, "upper"),
    (0.0, 0.0, "must be positive"),
]


@pytest.mark.parametrize(("upper", "lower", "match"), _BAD_RAYS)
def test_ground_projection_config_refuses_a_ray_that_cannot_hit(
    upper: float, lower: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GroundProjectionConfig(ray_distance_upper=upper, ray_distance_lower=lower)


@pytest.mark.parametrize(("upper", "lower"), [(5.0, 5.0), (0.0, 10.0), (3.0, 0.0)])
def test_ground_projection_config_accepts_a_ray_of_some_length(
    upper: float, lower: float
) -> None:
    config = GroundProjectionConfig(ray_distance_upper=upper, ray_distance_lower=lower)
    assert (config.ray_distance_upper, config.ray_distance_lower) == (upper, lower)


def test_the_default_ground_projection_config_is_accepted() -> None:
    GroundProjectionConfig()


def test_the_message_names_the_value_as_the_caller_calls_it() -> None:
    with pytest.raises(ValueError, match=r"^--ray-lower must be non-negative"):
        check_ray_distances(5.0, -1.0, "--ray-upper", "--ray-lower")


def test_the_detection_refuses_a_ray_that_cannot_hit() -> None:
    # Before the world or the map is touched.
    with pytest.raises(ValueError, match="ray_distance_upper"):
        tool.detect_no_3d_model_lanelets(
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            (0.0, 0.0),
            0.0,
            ray_distance_upper=-1.0,
        )


# ---------------------------------------------------------------------------
# detect-no-3d-model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--ray-upper", "-1"], "--ray-upper must be non-negative"),
        (["--ray-lower", "-1"], "--ray-lower must be non-negative"),
        (["--ray-upper", "nan"], "--ray-upper must be non-negative and finite"),
        (["--ray-lower", "inf"], "--ray-lower must be non-negative and finite"),
        (["--ray-upper", "0", "--ray-lower", "0"], "must be positive"),
        (["--timeout", "0"], "--timeout must be positive"),
        (["--timeout", "-3"], "--timeout must be positive"),
        (["--timeout", "nan"], "--timeout must be positive and finite"),
        (["--timeout", "inf"], "--timeout must be positive and finite"),
    ],
)
def test_the_tool_refuses_before_connecting_or_touching_the_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    match: str,
) -> None:
    def no_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("connected to CARLA")

    def no_map(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("read the map config")

    monkeypatch.setattr(tool, "carla", SimpleNamespace(Client=no_client))
    monkeypatch.setattr(tool, "_resolve_map_config", no_map)
    text = "map:\n  name: M\n  no_3d_model_lanelet_ids: [1]\n"
    config = tmp_path / "map.yaml"
    config.write_text(text)

    with pytest.raises(SystemExit) as exited:
        tool.main([str(config), *argv])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "detect-no-3d-model: error:" in err
    assert match in err
    assert config.read_text() == text


def test_the_tool_accepts_a_ray_starting_at_the_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Reached(Exception):
        pass

    def stop(*args: Any, **kwargs: Any) -> None:
        raise _Reached

    monkeypatch.setattr(tool, "_resolve_map_config", stop)
    config = tmp_path / "map.yaml"
    config.write_text("map:\n  name: M\n")
    with pytest.raises(_Reached):
        tool.main([str(config), "--ray-upper", "0"])


# ---------------------------------------------------------------------------
# The runner's entity.ground_projection_ray_distance_*
# ---------------------------------------------------------------------------


def _shipped(overrides: list[str]) -> Any:
    from .test_video_removed import _shipped as compose

    return compose(overrides)


def test_the_shipped_entity_values_are_accepted() -> None:
    from autoware_carla_scenario.examples import run

    projection = run.build_ground_projection(_shipped([]))
    assert (projection.ray_distance_upper, projection.ray_distance_lower) == (
        5.0,
        10.0,
    )
    run.check_run_config(_shipped([]))


@pytest.mark.parametrize(
    ("override", "key"),
    [
        (
            "entity.ground_projection_ray_distance_lower=-1",
            "entity.ground_projection_ray_distance_lower must be non-negative",
        ),
        (
            "entity.ground_projection_ray_distance_upper=nan",
            "entity.ground_projection_ray_distance_upper must be non-negative",
        ),
        (
            "entity.ground_projection_ray_distance_upper=abc",
            "entity.ground_projection_ray_distance_upper must be a number",
        ),
        pytest.param(
            f"entity.ground_projection_ray_distance_lower={10**400}",
            "entity.ground_projection_ray_distance_lower must be non-negative",
            id="too-big-for-a-float",  # #1652
        ),
    ],
)
def test_a_run_config_with_a_ray_that_cannot_hit_is_refused(
    override: str, key: str
) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([override])
    with pytest.raises(ValueError, match=key):
        run.build_ground_projection(cfg)
    with pytest.raises(run.RunConfigError, match=key):
        run.check_run_config(cfg)


def test_a_ray_of_no_length_is_refused_naming_both_keys() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped(
        [
            "entity.ground_projection_ray_distance_upper=0",
            "entity.ground_projection_ray_distance_lower=0",
        ]
    )
    with pytest.raises(
        run.RunConfigError,
        match=(
            r"entity\.ground_projection_ray_distance_upper \+ "
            r"entity\.ground_projection_ray_distance_lower must be positive"
        ),
    ):
        run.check_run_config(cfg)


def test_a_run_is_refused_before_the_scenario_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.sweeper.isolation import SWEEPER_OWNS_RETRIES_ENV

    monkeypatch.delenv(SWEEPER_OWNS_RETRIES_ENV, raising=False)
    cfg = _shipped(["entity.ground_projection_ray_distance_lower=-1"])
    monkeypatch.setattr(
        run, "build_scenario", lambda *_, **__: pytest.fail("built the scenario")
    )
    hydra_main = run._hydra_main.__wrapped__  # type: ignore[attr-defined]
    with pytest.raises(
        run.RunConfigError, match="entity.ground_projection_ray_distance_lower"
    ):
        hydra_main(cfg)


def test_a_batch_is_refused_before_any_scenario_is_built(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.examples import run

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    monkeypatch.setattr(
        run, "build_scenario", lambda *_, **__: pytest.fail("built the scenario")
    )
    with pytest.raises(SystemExit) as exited:
        run.run_batch(
            ["lane_change/left"],
            ["entity.ground_projection_ray_distance_upper=-2"],
        )
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "entity.ground_projection_ray_distance_upper must be non-negative" in out
