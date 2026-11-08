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
hilspy installed.  The constants and :func:`encode`, :func:`parse_address`,
:func:`format_address` and :func:`command_value` live in :mod:`.wire`, which
needs no socket (so the static check compiles it); they are re-exported here.
"""

from __future__ import annotations

import json
import selectors
import socket
import time
from typing import Any, Dict, List, NoReturn, Optional

from .wire import (
    COMMAND_FIELDS,
    COMMAND_MODES,
    COMMAND_VALUE_LIMIT,
    DEFAULT_PORT,
    LINK_DIRECTIONS,
    PROTOCOL_VERSION,
    RUNNER_MESSAGES,
    command_value,
    encode,
    format_address,
    parse_address,
)

__all__ = [
    "COMMAND_FIELDS",
    "COMMAND_MODES",
    "COMMAND_VALUE_LIMIT",
    "DEFAULT_PORT",
    "LINK_DIRECTIONS",
    "MAX_LINE_BYTES",
    "RUNNER_MESSAGES",
    "PROTOCOL_VERSION",
    "LineReader",
    "command_value",
    "encode",
    "format_address",
    "parse_address",
]

#: The longest line :class:`LineReader` buffers, bytes: a peer that sends more
#: without a newline is refused rather than buffered without bound.
MAX_LINE_BYTES = 1 << 20

#: How much of a refused line a :class:`LineReader` error quotes, bytes.
_PREVIEW_BYTES = 200


def _preview(line: bytes) -> str:
    """The start of *line*, for an error message."""
    if len(line) > _PREVIEW_BYTES:
        return f"{line[:_PREVIEW_BYTES]!r}... ({len(line)} bytes)"
    return repr(line)


def _parse(line: bytes) -> Dict[str, Any]:
    """The message on *line*.

    Raises:
        ValueError: The line is not a JSON object with a ``type``.
    """
    try:
        message = json.loads(line)
    except Exception as exc:
        # Not only ValueError: a deeply nested line is a RecursionError.
        raise ValueError(
            f"not a message: {_preview(line)} ({type(exc).__name__})"
        ) from exc
    if not isinstance(message, dict) or "type" not in message:
        raise ValueError(f"not a message: {_preview(line)}")
    return message


class LineReader:
    """Reads the messages arriving on a socket, a line at a time.

    A line longer than :data:`MAX_LINE_BYTES` is refused; the stream cannot be
    read on after it, so the reader counts as :attr:`closed` then.  A line that
    is not a message is refused and dropped, and the reader reads on; the
    messages that arrived before it are returned first, and the call after
    raises its error.

    Args:
        sock: The connected socket.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()
        # How much of the buffer is known to hold no newline: only what arrives
        # after it is searched, so a long line is not rescanned per chunk.
        self._scanned = 0
        #: A line's refusal held back for the next read, as the messages that
        #: preceded it were returned first.
        self._refused: Optional[ValueError] = None
        #: Whether nothing more can be read: the peer closed, or a line was
        #: too long to read on after it.
        self._eof = False

    @property
    def closed(self) -> bool:
        """Whether the stream is over: the peer closed it (or a line was too
        long to read on after).  Not before a refusal held back for the next
        read has been raised, so that a caller reading until closed still
        hears why the peer's last line was refused."""
        return self._eof and self._refused is None

    def read(self, timeout: Optional[float] = 0.0) -> List[Dict[str, Any]]:
        """The messages that arrived; waits up to *timeout* s for the first.

        ``None`` waits indefinitely.  A peer that keeps sending without ending
        a line does not hold the call past *timeout*.  :attr:`closed` turns
        ``True`` once the peer closes.

        The wait does not touch the socket's own timeout
        (:meth:`socket.socket.settimeout`), which stays the one its sends
        are bounded by, even while another thread sends.

        Raises:
            ValueError: A line is not a JSON object with a ``type`` (raised
                by the call after the one that returned the messages before
                it, if any), or is longer than :data:`MAX_LINE_BYTES`.
        """
        messages = self._take()
        if messages or self.closed:
            return messages
        deadline = None if timeout is None else time.monotonic() + timeout
        wait = timeout
        # The wait is a select, not the socket's timeout: that one is the
        # socket's own, for its sends, which another thread may be making
        # (#1792).  A readable socket's recv returns at once.
        with selectors.DefaultSelector() as selector:
            try:
                selector.register(self._sock, selectors.EVENT_READ)
            except (OSError, ValueError):
                # Closed (or shut down and closed) under the reader.
                self._eof = True
                return self._take()
            while True:
                try:
                    ready = selector.select(wait)
                except (OSError, ValueError):
                    self._eof = True
                    return self._take()
                if not ready:
                    return self._take()
                try:
                    chunk = self._sock.recv(65536)
                except (socket.timeout, BlockingIOError, InterruptedError):
                    return self._take()
                except OSError:
                    self._eof = True
                    return self._take()
                if not chunk:
                    self._eof = True
                    return self._take()
                self._buffer += chunk
                messages = self._take()
                if messages:
                    return messages
                # A partial line: wait for its end, but no longer than asked.
                # (A zero timeout reads on while more has already arrived.)
                if deadline is not None and timeout != 0.0:
                    wait = deadline - time.monotonic()
                    if wait <= 0.0:
                        return messages

    def _take(self) -> List[Dict[str, Any]]:
        refused, self._refused = self._refused, None
        if refused is not None:
            raise refused
        messages: List[Dict[str, Any]] = []
        while True:
            end = self._buffer.find(b"\n", self._scanned)
            if end < 0:
                self._scanned = len(self._buffer)
                if self._scanned > MAX_LINE_BYTES:
                    self._overlong(self._scanned)
                return messages
            if end > MAX_LINE_BYTES:
                self._overlong(end)
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            self._scanned = 0
            if not line.strip():
                continue
            try:
                messages.append(_parse(line))
            except ValueError as exc:
                if not messages:
                    raise
                # The messages before the bad line are not lost with it (an
                # ``error`` or ``ended`` among them says why the peer stops):
                # they go first, and the next read raises.
                self._refused = exc
                return messages

    def _overlong(self, length: int) -> NoReturn:
        preview = bytes(self._buffer[:_PREVIEW_BYTES])
        self._buffer = bytearray()
        self._scanned = 0
        self._eof = True
        raise ValueError(
            f"not a message: a line longer than {MAX_LINE_BYTES} bytes "
            f"({length} bytes so far, starting {preview!r})"
        )
