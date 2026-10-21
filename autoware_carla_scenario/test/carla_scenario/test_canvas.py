"""Sensors rendered by carla_canvas (``canvas.config``): the run points
typesafe_carla's sensor plugin at carla_canvas before anything listens."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.canvas import (
    CONFIG_ENV,
    INSTALL_HINT,
    PLUGIN_ENV,
    CanvasUnavailableError,
    use_canvas,
)
from autoware_carla_scenario.examples.run import apply_canvas, canvas_config


@pytest.fixture
def canvas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """carla_canvas installed, its plugin at a known path; the environment clean."""
    plugin = tmp_path / "libcarla_canvas_plugin.so"
    plugin.write_bytes(b"")
    module = types.ModuleType("carla_canvas.plugin")
    module.plugin_path = lambda: plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carla_canvas", types.ModuleType("carla_canvas"))
    monkeypatch.setitem(sys.modules, "carla_canvas.plugin", module)
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    monkeypatch.delenv(PLUGIN_ENV, raising=False)
    return plugin


def test_no_config_leaves_the_sensors_to_carla(canvas: Path) -> None:
    import os

    assert use_canvas(None) is None
    assert use_canvas("") is None
    assert CONFIG_ENV not in os.environ
    assert PLUGIN_ENV not in os.environ


def test_a_config_points_typesafe_carla_at_carla_canvas(
    canvas: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    config = tmp_path / "lidarverse.yaml"
    config.write_text("renderer: {}\n")
    monkeypatch.chdir(tmp_path)
    assert use_canvas("lidarverse.yaml") == config
    assert os.environ[CONFIG_ENV] == str(config)
    assert os.environ[PLUGIN_ENV] == str(canvas)


def test_a_missing_file_or_client_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(CanvasUnavailableError, match="not a file"):
        use_canvas(tmp_path / "nowhere.yaml")
    config = tmp_path / "c.yaml"
    config.write_text("")
    monkeypatch.setitem(sys.modules, "carla_canvas.plugin", None)
    with pytest.raises(CanvasUnavailableError) as raised:
        use_canvas(config)
    assert INSTALL_HINT in str(raised.value)


def test_a_batch_names_one_config(canvas: Path, tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("")

    def cfg(path: object) -> Any:
        return OmegaConf.create({"canvas": {"config": path}})

    assert canvas_config(cfg(None), OmegaConf.create({})) is None
    assert canvas_config(cfg(str(config)), cfg(str(config))) == str(config)
    with pytest.raises(ValueError, match="2 canvas.config"):
        canvas_config(cfg(str(config)), cfg("other.yaml"))
    assert apply_canvas(cfg(str(config))) == config
