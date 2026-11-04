"""The ports a run connects on -- ``server.port``, ``traffic_manager.port`` and
``traffic.options.port`` -- are refused before anything starts unless they
are TCP ports (#1423): 0, 70000 or ``true`` used to pass the config check and
fail only at ``client.get_trafficmanager(port)`` with CARLA up."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.traffic import build_traffic_manager
from autoware_carla_scenario.traffic.config import TrafficManagerBackendConfig

_NOT_PORTS = [0, -1, 65536, 70000, True, False, 8100.0, "8100"]


def _shipped(overrides: list[str]) -> Any:
    from .test_video_removed import _shipped as compose

    return compose(overrides)


@pytest.mark.parametrize("port", [1, 2000, 8100, 65535])
def test_the_traffic_manager_options_take_a_port(port: int) -> None:
    assert TrafficManagerBackendConfig(port=port).port == port
    assert TrafficManagerBackendConfig.from_mapping({"port": port}).port == port


def test_the_traffic_manager_options_default_to_a_port() -> None:
    TrafficManagerBackendConfig()
    TrafficManagerBackendConfig.from_mapping({})


@pytest.mark.parametrize("port", _NOT_PORTS)
def test_the_traffic_manager_options_refuse_what_is_not_a_port(port: Any) -> None:
    match = r"^traffic\.options\.port \(traffic_manager\.port\) must be an integer"
    with pytest.raises(ValueError, match=match):
        TrafficManagerBackendConfig(port=port)
    with pytest.raises(ValueError, match=match):
        build_traffic_manager({"port": port})


def test_the_shipped_ports_are_accepted() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([])
    assert (cfg.server.port, cfg.traffic_manager.port) == (2000, 8100)
    run.check_run_config(cfg)
    run.check_run_config(_shipped(["traffic=sumo"]))
    run.check_run_config(_shipped(["traffic=none"]))


@pytest.mark.parametrize("key", ["server.port", "traffic_manager.port"])
@pytest.mark.parametrize("value", ["0", "70000", "-1", "true", "8100.0", "'8100'"])
def test_a_run_config_with_a_port_that_is_not_one_is_refused(
    key: str, value: str
) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"{key}={value}"])
    with pytest.raises(run.RunConfigError, match=rf"^{key} must be an integer"):
        run.check_run_config(cfg)


@pytest.mark.parametrize("traffic", ["traffic_manager", "sumo", "none"])
def test_the_traffic_manager_port_is_refused_whichever_backend_runs(
    traffic: str,
) -> None:
    # The queue hands traffic_manager.port to the scenario's entities even
    # when the traffic group's backend reads no port of its own.
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"traffic={traffic}", "traffic_manager.port=70000"])
    with pytest.raises(run.RunConfigError, match="traffic_manager.port"):
        run.check_run_config(cfg)


@pytest.mark.parametrize("value", ["0", "70000", "true"])
def test_a_run_config_with_a_traffic_options_port_that_is_not_one_is_refused(
    value: str,
) -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped([f"traffic.options.port={value}"])
    with pytest.raises(run.RunConfigError, match=r"traffic\.options\.port"):
        run.check_run_config(cfg)


def test_a_legacy_config_without_the_traffic_group_is_refused_too() -> None:
    from autoware_carla_scenario.examples import run

    cfg = _shipped(["traffic_manager.port=true"])
    OmegaConf.set_struct(cfg, False)
    del cfg["traffic"]
    with pytest.raises(run.RunConfigError, match="traffic_manager.port"):
        run.check_run_config(cfg)


def test_a_batch_is_refused_for_a_port_before_any_case_runs() -> None:
    from autoware_carla_scenario.examples import run

    good = _shipped([])
    bad = _shipped(["server.port=70000"])
    with pytest.raises(run.RunConfigError, match="server.port"):
        run.check_batch_configs([good, bad], ["good", "bad"])
