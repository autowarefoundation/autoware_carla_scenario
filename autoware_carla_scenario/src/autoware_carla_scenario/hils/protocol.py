"""The link between a scenario run and the resident HILS bridge.

The bridge (``scenario-hils``) stays up on the HILS PC; a run with
``ego.entity=hils`` connects to it over TCP for the length of a scenario.  The
messages are JSON objects, one per line, each with a ``type``:

========================  ===================================================
runner -> bridge
``start``                 a scenario starts: ``scenario`` (its name), ``ego_id``
                          (the ego's actor id).  The bridge runs its
                          ``scenario_start`` steps, then answers ``ready``.
``end``                   the scenario ends: the bridge runs its
                          ``scenario_end`` steps, then answers ``ended``.
``disconnect``            stop data (fault injection): ``id`` (names the
                          cut), ``device`` (its name; ``null`` for every
                          device), ``messages`` (names; ``null`` for
                          everything) and ``direction`` (one of
                          :data:`LINK_DIRECTIONS`), until ``reconnect`` with
                          the ``id`` or the scenario's end.  The bridge
                          answers ``error`` for a device it does not have.
``reconnect``             ``id``: carry what that cut stopped again.
bridge -> runner
``hello``                 on connecting: ``protocol`` (:data:`PROTOCOL_VERSION`).
``ready``                 the ECUs are initialized; the scenario may start.
``error``                 ``message``: initialization failed, or the bridge
                          refused the request.
``command``               the ECUs' latest command: ``mode`` (one of
                          :data:`COMMAND_MODES`) and its fields
                          (:data:`COMMAND_FIELDS`), whenever an ECU sends one.
``ended``                 the ECUs are finalized.
========================  ===================================================

Only the standard library is needed here, so a run reaches the bridge without
hilspy installed.
"""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "COMMAND_FIELDS",
    "COMMAND_MODES",
    "DEFAULT_PORT",
    "LINK_DIRECTIONS",
    "PROTOCOL_VERSION",
    "LineReader",
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


class LineReader:
    """Reads the messages arriving on a socket, a line at a time.

    Args:
        sock: The connected socket.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = b""
        self.closed = False

    def read(self, timeout: Optional[float] = 0.0) -> List[Dict[str, Any]]:
        """The messages that arrived; waits up to *timeout* s for the first.

        ``None`` waits indefinitely.  :attr:`closed` turns ``True`` once the
        peer closes.

        Raises:
            ValueError: A line is not a JSON object.
        """
        messages = self._take()
        if messages or self.closed:
            return messages
        self._sock.settimeout(timeout)
        while True:
            try:
                chunk = self._sock.recv(65536)
            except (socket.timeout, BlockingIOError):
                return self._take()
            except OSError:
                self.closed = True
                return self._take()
            if not chunk:
                self.closed = True
                return self._take()
            self._buffer += chunk
            messages = self._take()
            if messages:
                return messages
            # A partial line: wait for its end without blocking the caller long.
            self._sock.settimeout(0.0 if timeout == 0.0 else timeout)

    def _take(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if not line.strip():
                continue
            message = json.loads(line)
            if not isinstance(message, dict) or "type" not in message:
                raise ValueError(f"not a message: {line[:200]!r}")
            messages.append(message)
        return messages
