"""The bridge's end of the link with scenario runs (:mod:`.protocol`).

:class:`RunnerLink` listens for runs and serves one at a time: what a run asks
(``start``, ``end``, ``disconnect``, ``reconnect``) and its going away become
events on a queue the bridge's loop takes them from, so lifecycle steps run on
the bridge's own thread, in order with its ticks; :meth:`RunnerLink.send`
answers (from any thread: the ECUs' commands are forwarded from the threads
receiving them).

Threads and the connection: the serving thread alone reads it (a
:class:`~.protocol.LineReader`, which waits with a select and leaves the
socket's timeout alone) and alone closes it; senders write it one at a time,
under the link's lock, bounded by the socket's own timeout
(:data:`SEND_TIMEOUT_S`).  A send that fails may have written part of its line,
so the link shuts the connection down rather than send after it: the serving
thread then sees the run gone (#1792).
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
    format_address,
    parse_address,
)

__all__ = ["SEND_TIMEOUT_S", "RunnerLink"]

logger = logging.getLogger(__name__)

#: An event for the bridge: ``("connect" | "start" | "end" | "disconnect" |
#: "reconnect" | "gone", message)``.
Event = Tuple[str, Dict[str, Any]]

#: How often the threads look at whether the link is closing, seconds.
_POLL_S = 0.2

#: How long a send may wait for a run that does not read, seconds; past it the
#: run counts as gone.  The connection's own timeout: the reader does not set
#: one.
SEND_TIMEOUT_S = 10.0


class RunnerLink:
    """Listens for scenario runs at *address* (``host:port``; port 0 picks one).

    An IPv6 address is written in brackets to give a port (``[::1]:7700``);
    :func:`~.wire.parse_address` lists the forms. ``[::]`` also takes IPv4
    connections where the OS supports dual-stack sockets
    (:func:`socket.has_dualstack_ipv6`); elsewhere it is IPv6 only, and
    ``0.0.0.0`` listens on IPv4.

    Args:
        address: Where to listen.
        events: Where the runs' requests go.
    """

    def __init__(self, address: str, events: "queue.Queue[Event]") -> None:
        host, port = parse_address(address)
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        # ``[::]`` is every address: IPv4 too, where the OS can serve both on
        # one socket (else it is IPv6 only, and ``0.0.0.0`` is for IPv4).
        dualstack = host == "::" and socket.has_dualstack_ipv6()
        self._server = socket.create_server(
            (host, port), family=family, dualstack_ipv6=dualstack, reuse_port=False
        )
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
        logger.info("waiting for scenario runs on %s", format_address(*self.address))

    def send(self, message: Mapping[str, Any]) -> bool:
        """Send *message* to the connected run; whether it went.

        A send that fails (the run gone, or not reading for
        :data:`SEND_TIMEOUT_S`) may have left part of its line on the
        connection, so it ends the connection: nothing more is sent on it, and
        the serving thread reports the run gone.
        """
        data = encode(message)
        with self._lock:
            conn = self._conn
            if conn is None:
                return False
            try:
                conn.sendall(data)
            except OSError as exc:
                logger.warning("the link with the scenario run failed: %s", exc)
                # Not closed here: the serving thread may be reading it, and
                # closes it itself once the shutdown wakes it.
                self._conn = None
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
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
            try:
                self._serve_one(conn, peer)
            except Exception:
                # Whatever one run does, the next is still served.
                logger.exception("the link with a scenario run failed")

    def _serve_one(self, conn: socket.socket, peer: Any) -> None:
        try:
            logger.info(
                "scenario run connected from %s",
                format_address(str(peer[0]), int(peer[1])),
            )
            # Blocking, but no send waits past SEND_TIMEOUT_S; the reader
            # waits with a select and leaves this alone.
            conn.settimeout(SEND_TIMEOUT_S)
            with self._lock:
                self._conn = conn
            self.send({"type": "hello", "protocol": PROTOCOL_VERSION})
            self._events.put(("connect", {}))
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
