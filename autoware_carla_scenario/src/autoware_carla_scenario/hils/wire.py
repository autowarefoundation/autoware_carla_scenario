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
    "format_address",
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


def _parse_port(text: str, address: str) -> int:
    """The port *text* of *address*: a whole number in 0..65535 (0 picks one)."""
    if not text:
        raise ValueError(f"address {address!r}: the port after ':' is empty")
    for c in text:
        if c < "0" or c > "9":
            raise ValueError(
                f"address {address!r}: the port {text!r} is not a whole number"
            )
    port = int(text)
    if port > 65535:
        raise ValueError(f"address {address!r}: the port {port} is not within 0-65535")
    return port


def parse_address(address: str, default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """*address* as ``(host, port)``.

    The forms taken:

    * ``host:port`` and ``host`` (*default_port*); an empty host
      (``:7700``) is ``0.0.0.0``.
    * ``[host]:port`` and ``[host]``, for an IPv6 address: ``[::1]:7700``.
    * An IPv6 address without brackets (more than one ``:``), such as ``::1``,
      is a host alone, on *default_port*: ``fe80::1:7700`` is that whole
      address, not port 7700 of ``fe80::1``. Give a port with brackets.

    The host comes back without brackets, as :func:`socket.create_server` and
    :func:`socket.create_connection` take it.

    Raises:
        ValueError: The port is not a whole number in 0..65535, or the
            brackets are unbalanced or empty.
    """
    if address.startswith("["):
        end = address.find("]")
        if end < 0:
            raise ValueError(f"address {address!r}: '[' is never closed by ']'")
        host = address[1:end]
        if not host:
            raise ValueError(f"address {address!r}: no host between '[' and ']'")
        if "[" in host:
            raise ValueError(f"address {address!r}: a '[' inside the brackets")
        rest = address[end + 1 :]
        if not rest:
            return host, default_port
        if not rest.startswith(":"):
            raise ValueError(
                f"address {address!r}: expected ':port' after ']', got {rest!r}"
            )
        return host, _parse_port(rest[1:], address)
    if "[" in address or "]" in address:
        raise ValueError(
            f"address {address!r}: brackets go around the host only: [host]:port"
        )
    colons = address.count(":")
    if colons == 0:
        return address, default_port
    if colons > 1:
        # An IPv6 address without brackets: all of it is the host.
        return address, default_port
    host, _, port = address.partition(":")
    return host or "0.0.0.0", _parse_port(port, address)


def format_address(host: str, port: int) -> str:
    """``host:port``, with an IPv6 *host* in brackets (``[::1]:7700``).

    The inverse of :func:`parse_address`.
    """
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"
