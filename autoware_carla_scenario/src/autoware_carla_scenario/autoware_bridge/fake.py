"""In-memory fake :class:`AutowareBridge` for unit tests.

:class:`FakeAutowareBridge` records calls and reports readiness after a
configurable number of polls, so an :class:`AutowareEgoEntity` can be driven to
"ready" without a live Autoware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .base import AutowareBridge, BridgePose


@dataclass
class FakeAutowareBridge(AutowareBridge):
    """Deterministic fake bridge for tests.

    Args:
        ready_after: Number of :meth:`is_ready` polls (after :meth:`configure`)
            before it returns ``True``.  ``0`` means ready on the first poll.
    """

    ready_after: int = 0
    #: What :attr:`client_connected` says.
    connected: bool = True

    #: Ordered log of method names invoked, for assertions.
    calls: List[str] = field(default_factory=list)
    #: The initial pose passed to :meth:`configure`, if any.
    configured_initial_pose: Optional[BridgePose] = None
    #: The goal passed to :meth:`configure`, or the last :meth:`reroute`, if any.
    configured_goal: Optional[BridgePose] = None
    #: The waypoints passed to :meth:`configure` or the last :meth:`reroute`, in order.
    configured_waypoints: List[BridgePose] = field(default_factory=list)
    #: Bumped by :meth:`configure` and :meth:`reroute`, as the gRPC server's
    #: ``mission_version`` is: 0 before the first mission.
    mission_version: int = 0
    #: Whether :meth:`close` has been called.
    closed: bool = False
    #: The turn-indicator command :meth:`take_turn_indicators` hands over next.
    turn_indicators: Optional[int] = None

    _ready_polls: int = 0

    def configure(
        self,
        initial_pose: BridgePose,
        goal: BridgePose,
        waypoints: Optional[list[BridgePose]] = None,
    ) -> None:
        self.calls.append("configure")
        self.configured_initial_pose = initial_pose
        self.configured_goal = goal
        self.configured_waypoints = list(waypoints) if waypoints is not None else []
        self.mission_version += 1

    def reroute(
        self,
        goal: BridgePose,
        waypoints: Optional[list[BridgePose]] = None,
    ) -> None:
        if self.configured_initial_pose is None:
            raise RuntimeError("reroute() before configure(): there is no mission")
        self.calls.append("reroute")
        self.configured_goal = goal
        self.configured_waypoints = list(waypoints) if waypoints is not None else []
        self.mission_version += 1

    def is_ready(self) -> bool:
        self.calls.append("is_ready")
        if self.configured_initial_pose is None:
            return False
        ready = self._ready_polls >= self.ready_after
        self._ready_polls += 1
        return ready

    def report_turn_indicators(self, command: int) -> None:
        """Queue *command* as if the stack had just reported it."""
        self.turn_indicators = command

    def take_turn_indicators(self) -> Optional[int]:
        command = self.turn_indicators
        self.turn_indicators = None
        return command

    @property
    def client_connected(self) -> bool:
        return self.connected

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True
