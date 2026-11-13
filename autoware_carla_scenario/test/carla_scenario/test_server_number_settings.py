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
        # Too big for a float (#1652).
        *[
            pytest.param(key, str(10**400), id=f"{key}-huge")
            for key in ("cooldown_seconds", "max_tick_rate_hz")
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


# ``server.cooldown_max_retries`` and ``server.reload_every`` are whole
# numbers, 0 or more: int() alone ran 1.9 and true as 1 retry or reload (#1445).
@pytest.mark.parametrize("key", ["cooldown_max_retries", "reload_every"])
@pytest.mark.parametrize("value", ["1.9", "2.5", "true", "false", "abc", "nan"])
def test_a_server_count_that_is_no_whole_number_is_refused(
    key: str, value: str
) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"server.{key}={value}"])
    with pytest.raises(
        run.RunConfigError, match=rf"^server\.{key} must be a whole number"
    ):
        run.check_run_config(cfg)


@pytest.mark.parametrize("key", ["cooldown_max_retries", "reload_every"])
def test_a_negative_server_count_is_refused(key: str) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"server.{key}=-1"])
    with pytest.raises(run.RunConfigError, match=rf"^server\.{key} must be >= 0"):
        run.check_run_config(cfg)


def test_a_null_cooldown_max_retries_is_refused() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped(["server.cooldown_max_retries=null"])
    with pytest.raises(
        run.RunConfigError, match=r"^server\.cooldown_max_retries must be a whole"
    ):
        run.check_run_config(cfg)


def test_whole_float_server_counts_are_those_counts() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped(
        [
            "server.cooldown_max_retries=2.0",
            "server.world_reset=soft",
            "server.reload_every=3.0",
        ]
    )
    run.check_run_config(cfg)
    retries = run.cooldown_max_retries(cfg)
    assert retries == 2 and isinstance(retries, int)
    assert run._queue_retries(cfg) == 2
    world_reset, reload_every = run.world_reset_settings(cfg)
    assert (world_reset, reload_every) == ("soft", 3)
    assert isinstance(reload_every, int)
    # null reloads only when a soft reset cannot be done, as 0 does.
    assert run.world_reset_settings(_shipped(["server.reload_every=null"]))[1] == 0


def test_the_sweeper_refuses_a_fractional_cooldown_max_retries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Before any map is fetched or case forked, as a negative one is."""
    from unittest.mock import MagicMock, patch

    from omegaconf import OmegaConf

    from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as mod

    sweeper = mod.LaneletConstraintSweeper()
    sweeper.config = OmegaConf.create(
        {"sweep": {}, "server": {"cooldown_max_retries": 1.9}}
    )
    sweeper.launcher = MagicMock()
    with (
        patch.object(mod, "load_sweep_map") as load,
        patch.object(mod, "expand_sweep", return_value=[("a=1",)]),
        pytest.raises(SystemExit) as exited,
    ):
        sweeper.sweep([])
    assert exited.value.code == 2
    assert "cooldown_max_retries must be a whole number" in capsys.readouterr().err
    load.assert_not_called()


@pytest.mark.parametrize(
    ("override", "shown"),
    [
        # Read as the default reload before #2143.
        ("server.world_reset=''", "''"),
        ("server.world_reset=false", "False"),
        ("server.world_reset=0", "0"),
        ("server.world_reset=[]", "[]"),
        ("server.world_reset=sotf", "'sotf'"),
    ],
)
def test_world_reset_refuses_anything_but_a_strategy(override: str, shown: str) -> None:
    """Only an absent or null ``server.world_reset`` is ``reload`` (#2143)."""
    from autoware_carla_scenario.examples import run

    with pytest.raises(ValueError) as refused:
        run.world_reset_settings(_shipped([override]))
    assert str(refused.value) == (
        f"server.world_reset must be one of ['reload', 'soft'], got {shown}."
    )


def test_world_reset_reads_a_strategy_and_null_as_before() -> None:
    from autoware_carla_scenario.examples import run

    assert run.world_reset_settings(_shipped([]))[0] == "reload"
    assert run.world_reset_settings(_shipped(["server.world_reset=null"]))[0] == (
        "reload"
    )
    assert run.world_reset_settings(_shipped(["server.world_reset=soft"]))[0] == "soft"
    assert run.world_reset_settings(_shipped(["server.world_reset='reload'"]))[0] == (
        "reload"
    )
