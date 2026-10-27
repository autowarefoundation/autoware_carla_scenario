"""HILS: stream a scenario run's ego to devices under test (the ``hils`` extra).

A separate process, ``scenario-hils CONFIG`` (:mod:`.cli`), runs on the HILS
PC beside the ECUs. It connects to the run's CARLA server through
typesafe_carla as a client of its own, follows the ego the runner spawns
(:mod:`.bridge`), and sends what the ECUs would see through
[hilspy](https://github.com/hakuturu583/hilspy) channels: periodic messages
filled from the ego's signals (:mod:`.signals`: CAN frames from a DBC, a
GNSS receiver's SBF / NMEA / UBX), and what sensors on the ego measure
(:mod:`.measurements`: LiDAR packets of a real sensor model, camera frames
into a V4L2 device), the sensors rendered by CARLA or by carla_canvas.  The
whole set-up is one YAML file (:mod:`.config`); docs/hils.md describes it.
"""

from __future__ import annotations

from .config import (
    DeviceSpec,
    FieldSource,
    HilsConfig,
    HilsConfigError,
    MessageSpec,
    SensorSpec,
    load_config,
    parse_config,
)
from .signals import SIGNALS, EgoState, ego_signals

__all__ = [
    "SIGNALS",
    "DeviceSpec",
    "EgoState",
    "FieldSource",
    "HilsConfig",
    "HilsConfigError",
    "MessageSpec",
    "SensorSpec",
    "ego_signals",
    "load_config",
    "parse_config",
]
