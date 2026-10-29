"""The socket-free half of :mod:`.protocol`: its constants and message lines.

:mod:`.protocol` describes the messages between a run and the HILS bridge and
re-exports everything here; this module holds what needs no socket, so the
library check compiles it (docs/typecheck.md) and a checked module, such as
:mod:`~autoware_carla_scenario.actions.communication_loss`, can import it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "COMMAND_FIELDS",
    "COMMAND_MODES",
    "DEFAULT_PORT",
    "LINK_DIRECTIONS",
    "PROTOCOL_VERSION",
    "RUNNER_MESSAGES",
    "encode",
    "parse_address",
]

#: Bumped on an incompatible change; both ends check it.
PROTOCOL_VERSION = 1

#: The port the bridge listens on unless told otherwise.
DEFAULT_PORT = 7700

#: How an ECU's command drives the ego:
#:
#: * ``"control"`` -- CARLA's ``VehicleControl``: pedals, steering wheel, gear.
#: * ``"ackermann"`` -- CARLA's ``VehicleAckermannControl``: a steering angle
#:   and a target speed / acceleration, which CARLA's controller tracks.
COMMAND_MODES: Tuple[str, ...] = ("control", "ackermann")

#: Each mode's fields and what they mean (CARLA's conventions).
COMMAND_FIELDS: Dict[str, Dict[str, str]] = {
    "control": {
        "throttle": "[0, 1]",
        "brake": "[0, 1]",
        "steer": "[-1, 1], positive to the right",
        "hand_brake": "0 or 1",
        "reverse": "0 or 1",
        "manual_gear_shift": "0 or 1",
        "gear": "the gear, with manual_gear_shift",
    },
    "ackermann": {
        "steer": "rad, the front wheels' angle, positive to the right",
        "steer_speed": "rad/s; 0 steers at once",
        "speed": "m/s, the target speed",
        "acceleration": "m/s^2",
        "jerk": "m/s^3",
    },
}


#: The messages a run sends the bridge (the table of :mod:`.protocol`).
RUNNER_MESSAGES: Tuple[str, ...] = ("start", "end", "disconnect", "reconnect")

#: Which way of a device's link a ``disconnect`` stops: what the simulation
#: sends the ECU, what the ECU sends back, or both.
LINK_DIRECTIONS: Tuple[str, ...] = ("to_ecu", "from_ecu", "both")


def encode(message: Mapping[str, Any]) -> bytes:
    """One message, as the line that carries it."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def parse_address(address: str, default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """``host:port`` (or ``host``) as ``(host, port)``."""
    host, sep, port = address.rpartition(":")
    if not sep:
        return address, default_port
    return host or "0.0.0.0", int(port)
