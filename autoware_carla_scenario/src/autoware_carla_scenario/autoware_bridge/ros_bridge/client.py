"""gRPC client for the scenario library's ``AutowareBridge`` server.

This is the Autoware side of the (deliberately tiny) contract: pull the mission
once it is available (``GetMission``) and push a single readiness flag
(``ReportReadiness``).  It opens an ``insecure_channel`` to a server the interface
node launches and dials, and is kept free of ``rclpy`` so it can be unit tested
against an in-process server.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, Optional

from .._proto import autoware_bridge_pb2 as pb2
from .._proto import autoware_bridge_pb2_grpc as pb2_grpc
import grpc

logger = logging.getLogger(__name__)

__all__ = ["ScenarioBridgeClient", "Mission", "TurnIndicatorForwarder"]


@dataclass(frozen=True)
class Mission:
    """The scenario's mission, as map-frame ``Pose`` wire messages.

    Attributes:
        initial_pose: Where Autoware initializes localization.
        goal: Where Autoware plans the route to.
    """

    initial_pose: pb2.Pose
    goal: pb2.Pose


class ScenarioBridgeClient:
    """Talks to the scenario library's ``AutowareBridge`` server.

    Args:
        address: ``host:port`` of the scenario-library server to dial.
        channel: Pre-built channel to use instead of dialling *address*.
            Intended for tests, which run the server in-process.
    """

    def __init__(
        self,
        address: str = "localhost:50052",
        *,
        channel: Optional[grpc.Channel] = None,
    ) -> None:
        self._address = address
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(address)
        self._stub = pb2_grpc.AutowareBridgeStub(self._channel)

    def wait_for_ready(self, timeout: float) -> bool:
        """Block until the channel connects, returning ``True`` on success.

        Args:
            timeout: Seconds to wait for the server to become reachable.
        """
        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            logger.warning(
                "AutowareBridge server at %s not reachable within %.1fs",
                self._address,
                timeout,
            )
            return False

    def get_mission(self, timeout: Optional[float] = None) -> Optional[Mission]:
        """Return the mission once the scenario has one, else ``None``.

        Non-fatal: while the scenario has not configured a mission yet the server
        answers ``available=False`` and this returns ``None``, so the caller can
        keep polling.

        Args:
            timeout: Per-RPC timeout in seconds.

        Raises:
            grpc.RpcError: If the server cannot be reached.
        """
        response = self._stub.GetMission(pb2.GetMissionRequest(), timeout=timeout)
        if not response.available:
            return None
        return Mission(initial_pose=response.initial_pose, goal=response.goal)

    def report_readiness(self, ready: bool, timeout: Optional[float] = None) -> None:
        """Push the readiness flag to the scenario framework.

        Args:
            ready: ``True`` once Autoware is initialized, routed, engaged, driving.
            timeout: Per-RPC timeout in seconds.

        Raises:
            grpc.RpcError: If the server cannot be reached.
        """
        self._stub.ReportReadiness(
            pb2.ReportReadinessRequest(ready=ready), timeout=timeout
        )

    def report_turn_indicators(
        self, command: int, timeout: Optional[float] = None
    ) -> None:
        """Push Autoware's turn-indicator command to the scenario framework.

        Args:
            command: An ``autoware_vehicle_msgs/TurnIndicatorsCommand`` value.
            timeout: Per-RPC timeout in seconds.

        Raises:
            grpc.RpcError: If the server cannot be reached.
        """
        self._stub.ReportTurnIndicators(
            pb2.ReportTurnIndicatorsRequest(command=command), timeout=timeout
        )

    def close(self) -> None:
        """Close the channel.  Idempotent; a no-op for an injected channel."""
        if self._owns_channel and self._channel is not None:
            self._channel.close()
            self._channel = None  # type: ignore[assignment]


class TurnIndicatorForwarder:
    """Forwards Autoware's turn-indicator command to the scenario.

    Autoware publishes ``TurnIndicatorsCommand`` every control cycle.  A change is
    reported at once; the command in force is reported again every
    ``resend_s``, so a scenario that started over -- a retried attempt, a fresh
    ego, a bridge restarted with nothing kept -- learns it without waiting for
    Autoware to change its mind.  ``NO_COMMAND`` is not forwarded.

    One report at a time: a message arriving while one is in flight is skipped,
    not queued behind it, so a server that stops answering holds one executor
    thread, not one per message.

    Args:
        client: The bridge client to report through.
        timeout: Per-RPC timeout in seconds.
        resend_s: Seconds after which an unchanged command is reported again.
        clock: Monotonic seconds; tests substitute their own.
    """

    #: ``TurnIndicatorsCommand.NO_COMMAND``.
    NO_COMMAND = 0

    def __init__(
        self,
        client: ScenarioBridgeClient,
        timeout: Optional[float],
        *,
        resend_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._resend_s = resend_s
        self._clock = clock
        self._sent: Optional[int] = None
        self._sent_at = 0.0
        self._lock = threading.Lock()

    def observe(self, command: int) -> Optional[bool]:
        """Take one published command.

        Returns:
            ``True`` if it was reported as a change, ``False`` if reported again
            unchanged, ``None`` if nothing was sent.

        Raises:
            grpc.RpcError: If reporting it failed; a later message retries.
        """
        if command == self.NO_COMMAND:
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            now = self._clock()
            changed = command != self._sent
            if not changed and now - self._sent_at < self._resend_s:
                return None
            self._client.report_turn_indicators(command, timeout=self._timeout)
            self._sent, self._sent_at = command, now
            return changed
        finally:
            self._lock.release()
