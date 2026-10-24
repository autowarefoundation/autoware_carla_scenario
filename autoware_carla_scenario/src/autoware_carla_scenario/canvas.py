"""Sensors rendered by carla_canvas instead of the CARLA server (``canvas.config``).

[carla_canvas](https://github.com/hakuturu583/carla_canvas) swaps how CARLA
renders cameras and LiDARs behind the CARLA API: a configuration file names a
renderer service (3D Gaussian splatting, a LiDAR world model such as
LidarVerse, ...) and the sensor types it renders, and typesafe_carla's sensor
plugin hands those sensors' measurements over from the service instead of
from the server.  Nothing else about a run changes: the scenario, the driver
and the measurements they read are the same.

``canvas.config=path/to/canvas.yaml`` points the run at such a file:
:func:`use_canvas` sets ``CARLA_CANVAS_CONFIG`` and typesafe_carla's
``TYPESAFE_CARLA_SENSOR_PLUGIN`` before any sensor listens, which is when
typesafe_carla loads the plugin (once per process).  The renderer service is
the configuration's: started by the plugin when nothing answers at its
``renderer.endpoint`` and the file gives a ``renderer.command``, or run by
hand (``carla-canvas serve --config ...``).

Needs carla_canvas's client (``pip install 'carla-canvas[client]>=0.2'``), which
is not an extra of this package: it needs protobuf 7, and the vendored
alpasim modules are regenerated on protobuf 4.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "CONFIG_ENV",
    "INSTALL_HINT",
    "PLUGIN_ENV",
    "CanvasUnavailableError",
    "use_canvas",
]

logger = logging.getLogger(__name__)

#: carla_canvas's configuration file.
CONFIG_ENV = "CARLA_CANVAS_CONFIG"
#: The sensor plugin typesafe_carla loads.
PLUGIN_ENV = "TYPESAFE_CARLA_SENSOR_PLUGIN"
#: How to install what this module needs.
INSTALL_HINT = "pip install 'carla-canvas[client]>=0.2'"


class CanvasUnavailableError(RuntimeError):
    """``canvas.config`` is set, but carla_canvas's client is not installed,
    or the file is not there."""

    def __init__(self, message: str) -> None:
        # Explicit for the static check, which cannot build a subclass of an
        # exception without one (docs/typecheck.md).
        super().__init__(message)


def use_canvas(config: Union[str, Path, None]) -> Optional[Path]:
    """Have the sensors *config* names rendered by carla_canvas; its path.

    ``None`` or ``""`` leaves the run's sensors to CARLA (and returns
    ``None``).  Call it before any sensor listens: typesafe_carla loads its
    sensor plugin then, once for the process.

    Raises:
        CanvasUnavailableError: The file does not exist, or carla_canvas is
            not installed.
    """
    if config is None or str(config) == "":
        return None
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        raise CanvasUnavailableError(f"canvas.config {path} is not a file")
    try:
        from carla_canvas.plugin import plugin_path  # noqa: PLC0415
    except ImportError as exc:
        raise CanvasUnavailableError(
            "canvas.config is set, but carla_canvas, which renders its sensors, "
            f"is not installed: {INSTALL_HINT} (or leave canvas.config unset)"
        ) from exc
    plugin = str(plugin_path())
    previous = os.environ.get(PLUGIN_ENV)
    if previous and previous != plugin:
        logger.warning(
            "%s named %s; carla_canvas's plugin %s is used instead",
            PLUGIN_ENV,
            previous,
            plugin,
        )
    os.environ[CONFIG_ENV] = str(path)
    os.environ[PLUGIN_ENV] = plugin
    logger.info("Sensors rendered by carla_canvas: %s (plugin %s)", path, plugin)
    return path
