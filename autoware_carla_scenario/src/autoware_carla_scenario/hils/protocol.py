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
hilspy installed.  The constants and :func:`encode` / :func:`parse_address`
live in :mod:`.wire`, which needs no socket (so the static check compiles
it); they are re-exported here.
"""

from __future__ import annotations

import json
import math
import socket
from typing import Any, Dict, List, Optional

from .wire import (
    COMMAND_FIELDS,
    COMMAND_MODES,
    DEFAULT_PORT,
    LINK_DIRECTIONS,
    PROTOCOL_VERSION,
    RUNNER_MESSAGES,
    encode,
    parse_address,
)

__all__ = [
    "COMMAND_FIELDS",
    "COMMAND_MODES",
    "DEFAULT_PORT",
    "LINK_DIRECTIONS",
    "RUNNER_MESSAGES",
    "PROTOCOL_VERSION",
    "LineReader",
    "encode",
    "parse_address",
]

#: The largest magnitude a command field may have: CARLA's controls are 32-bit
#: floats, which turn a larger (finite) double into an infinity.
COMMAND_VALUE_LIMIT = 3.4028234663852886e38


def command_value(value: Any) -> Optional[float]:
    """A command field's *value* as a float, or ``None`` when it is not a
    number CARLA can take: not finite (NaN, an infinity), or beyond
    :data:`COMMAND_VALUE_LIMIT` (an integer too large for a float included)."""
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number) or abs(number) > COMMAND_VALUE_LIMIT:
        return None
    return number


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
