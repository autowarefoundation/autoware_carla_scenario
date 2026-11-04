"""``server.cooldown_seconds``, ``server.max_tick_rate_hz`` and
``server.fog_density_scale`` are refused before anything starts unless they
are finite numbers in range (#1439), as the ports are (``test_port_settings.py``).

They were read with ``float()`` alone, so a NaN or negative value passed the
config check: the queue read a NaN or negative cooldown as none, the runner a
NaN or non-positive tick rate as uncapped, and a NaN fog scale reached the
server's command line when no LiDAR refused it."""

from __future__ import annotations

from typing import Any

import pytest


def _shipped(overrides: list[str]) -> Any:
    from .test_video_removed import _shipped as compose

    return compose(overrides)


def test_the_shipped_settings_and_valid_ones_are_accepted() -> None:
    from autoware_carla_scenario.examples import run

    run.check_run_config(_shipped([]))
    cfg = _shipped(
        [
            "server.cooldown_seconds=0",
            "server.max_tick_rate_hz=20",
            "server.fog_density_scale=0.01",
        ]
    )
    run.check_run_config(cfg)
    assert run._cooldown_seconds(cfg) == 0.0
    assert run._max_tick_rate_hz(cfg) == 20.0
    assert run.server_extra_args(cfg) != []
    run.check_run_config(_shipped(["server.cooldown_seconds=1.5"]))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        *[
            ("cooldown_seconds", v)
            for v in ["-1", "-0.5", "nan", "inf", "true", "'2.0'", "null"]
        ],
        *[
            ("max_tick_rate_hz", v)
            for v in ["0", "0.0", "-20", "nan", "inf", "-inf", "true", "'20'"]
        ],
        *[
            ("fog_density_scale", v)
            for v in ["0", "-0.01", "nan", "inf", "false", "'0.01'"]
        ],
    ],
)
def test_a_run_config_with_a_bad_server_number_is_refused(key: str, value: str) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"server.{key}={value}"])
    with pytest.raises(
        run.RunConfigError, match=rf"^server\.{key} must be a finite number"
    ):
        run.check_run_config(cfg)


@pytest.mark.parametrize("key", ["max_tick_rate_hz", "fog_density_scale"])
def test_null_keeps_the_server_s_own(key: str) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"server.{key}=null"])
    run.check_run_config(cfg)
    assert run._max_tick_rate_hz(cfg) is None
    assert run.server_extra_args(cfg) == []


def test_a_nan_fog_scale_never_reaches_the_server_command_line() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped(["server.fog_density_scale=nan"])
    assert not (cfg.get("driver") or {}).get("lidars")
    with pytest.raises(ValueError, match=r"server\.fog_density_scale"):
        run.server_extra_args(cfg)


def test_a_batch_is_refused_for_a_bad_server_number_before_any_case_runs() -> None:
    from autoware_carla_scenario.examples import run

    good = _shipped([])
    bad = _shipped(["server.cooldown_seconds=nan"])
    with pytest.raises(run.RunConfigError, match=r"^bad: server\.cooldown_seconds"):
        run.check_batch_configs([good, bad], ["good", "bad"])
