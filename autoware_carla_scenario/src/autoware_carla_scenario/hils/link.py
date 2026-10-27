"""The bridge's end of the link with scenario runs (:mod:`.protocol`).

:class:`RunnerLink` listens for runs and serves one at a time: what a run asks
(``start``, ``end``, ``disconnect``, ``reconnect``) and its going away become
events on a queue the bridge's loop takes them from, so lifecycle steps run on
the bridge's own thread, in order with its ticks; :meth:`RunnerLink.send`
answers (from any thread: the ECUs' commands are forwarded from the threads
receiving them).
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

from .protocol import (
    PROTOCOL_VERSION,
    RUNNER_MESSAGES,
    LineReader,
    encode,
    parse_address,
)

__all__ = ["RunnerLink"]

logger = logging.getLogger(__name__)

#: An event for the bridge: ``("connect" | "start" | "end" | "disconnect" |
#: "reconnect" | "gone", message)``.
Event = Tuple[str, Dict[str, Any]]

#: How often the threads look at whether the link is closing, seconds.
_POLL_S = 0.2


class RunnerLink:
    """Listens for scenario runs at *address* (``host:port``; port 0 picks one).

    Args:
        address: Where to listen.
        events: Where the runs' requests go.
    """

    def __init__(self, address: str, events: "queue.Queue[Event]") -> None:
        host, port = parse_address(address)
        self._server = socket.create_server((host, port), reuse_port=False)
        self._server.settimeout(_POLL_S)
        self._events = events
        self._conn: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="hils-runner", daemon=True
        )

    @property
    def address(self) -> Tuple[str, int]:
        """Where the link listens."""
        host, port = self._server.getsockname()[:2]
        return str(host), int(port)

    @property
    def connected(self) -> bool:
        """Whether a run is connected."""
        return self._conn is not None

    def start(self) -> None:
        """Start serving runs."""
        self._thread.start()
        logger.info("waiting for scenario runs on %s:%d", *self.address)

    def send(self, message: Mapping[str, Any]) -> bool:
        """Send *message* to the connected run; whether there was one to send to."""
        data = encode(message)
        with self._lock:
            conn = self._conn
            if conn is None:
                return False
            try:
                conn.sendall(data)
            except OSError as exc:
                logger.debug("the run's link failed: %s", exc)
                return False
        return True

    def close(self) -> None:
        """Stop serving; a connected run sees the link close."""
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._server.close()

    def _serve(self) -> None:
        while not self._closing.is_set():
            try:
                conn, peer = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            logger.info("scenario run connected from %s:%d", *peer[:2])
            with self._lock:
                self._conn = conn
            self.send({"type": "hello", "protocol": PROTOCOL_VERSION})
            self._events.put(("connect", {}))
            try:
                self._read(conn)
            finally:
                with self._lock:
                    self._conn = None
                conn.close()
                logger.info("scenario run disconnected")
                self._events.put(("gone", {}))

    def _read(self, conn: socket.socket) -> None:
        reader = LineReader(conn)
        while not self._closing.is_set() and not reader.closed:
            try:
                messages = reader.read(timeout=_POLL_S)
            except ValueError as exc:
                logger.warning("a run sent what is not a message: %s", exc)
                return
            for message in messages:
                kind = str(message.get("type"))
                if kind in RUNNER_MESSAGES:
                    self._events.put((kind, message))
                else:
                    logger.warning("a run sent an unknown message %r", kind)
