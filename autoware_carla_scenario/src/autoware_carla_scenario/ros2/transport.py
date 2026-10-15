"""The DDS transport the ROS 2 ego entity talks through.

:class:`Ros2EgoTransport` is all the entity sees: it publishes the ego's state and
hands over the commands that arrived since the last poll.  The Cyclone DDS
implementation lives in :mod:`.cyclone` and imports ``cyclonedds`` (the ``ros2``
extra); :class:`FakeRos2Transport` stands in for it in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from .commands import (
    Command,
    ControlCommand,
    EgoState,
    SteerAccelCommand,
    TrajectoryCommand,
)
from .config import Ros2EgoConfig

__all__ = ["CameraFrame", "FakeRos2Transport", "Received", "Ros2EgoTransport"]

#: A command and the name of the source it came from.  ``Tuple[str, Command]``,
#: with the union spelled out: Codon 0.19 does not resolve an alias imported
#: from another module inside an alias of this one (docs/typecheck.md).
Received = Tuple[str, Union[SteerAccelCommand, ControlCommand, TrajectoryCommand]]


@dataclass(frozen=True)
class CameraFrame:
    """One frame of a camera the entity publishes.

    Attributes:
        camera: Index of the camera in :attr:`~.config.Ros2EgoConfig.cameras`.
        stamp_us: Simulation time the frame was rendered at, microseconds.
        width: Image width, pixels.
        height: Image height, pixels.
        bgra: The pixels, row by row, four bytes each (B, G, R, A), as CARLA
            renders them.
    """

    camera: int
    stamp_us: int
    width: int
    height: int
    bgra: bytes


class Ros2EgoTransport(ABC):
    """Moves the ego's state out to ROS 2 and its commands in.

    The lifecycle is :meth:`open` once when the scenario starts, then each tick
    :meth:`publish_state`, :meth:`poll` and :meth:`poll_turn_indicators`, and
    :meth:`close` at teardown.
    :meth:`publish_image` is called from a CARLA sensor thread, between
    :meth:`open` and :meth:`close`.

    Args:
        config: Topics, domain and command sources.
    """

    _config: Ros2EgoConfig

    def __init__(self, config: Ros2EgoConfig) -> None:
        self._config = config

    @abstractmethod
    def open(self) -> None:
        """Join the DDS domain and create the publishers and subscribers."""
        ...

    @abstractmethod
    def publish_state(self, state: EgoState) -> None:
        """Publish the ego's state on every configured state topic."""
        ...

    @abstractmethod
    def publish_image(self, frame: CameraFrame) -> None:
        """Publish *frame* on its camera's topic."""
        ...

    @abstractmethod
    def poll(self) -> List[Received]:
        """Return the commands that arrived since the last poll, by source.

        A source's commands come in the order they arrived.
        """
        ...

    @abstractmethod
    def poll_turn_indicators(self) -> Optional[int]:
        """The newest ``TurnIndicatorsCommand`` value since the last poll, if any.

        Kept apart from :meth:`poll`: the indicators are not a command that
        drives, so they take no part in choosing the source that does.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Leave the domain.  Must be idempotent."""
        ...


class FakeRos2Transport(Ros2EgoTransport):
    """An in-memory transport: records what is published, replays what is queued.

    Queue commands with :meth:`deliver` and turn indicators with
    :meth:`deliver_turn_indicators`; the next poll returns them.
    """

    opened: int
    closed: int
    published: List[EgoState]
    images: List[CameraFrame]
    _queue: List[Received]
    _turn_indicators: Optional[int]

    def __init__(self, config: Ros2EgoConfig) -> None:
        super().__init__(config)
        self.opened = 0
        self.closed = 0
        self.published = []
        self.images = []
        self._queue = []
        self._turn_indicators = None

    def deliver(self, source_name: str, command: Command) -> None:
        """Queue *command* as if *source_name* had just published it."""
        self._queue.append((source_name, command))

    def deliver_turn_indicators(self, command: int) -> None:
        """Queue a ``TurnIndicatorsCommand`` value, as if just published."""
        self._turn_indicators = command

    def open(self) -> None:
        self.opened += 1

    def publish_state(self, state: EgoState) -> None:
        self.published.append(state)

    def publish_image(self, frame: CameraFrame) -> None:
        self.images.append(frame)

    def poll(self) -> List[Received]:
        received, self._queue = self._queue, []
        return received

    def poll_turn_indicators(self) -> Optional[int]:
        command, self._turn_indicators = self._turn_indicators, None
        return command

    def close(self) -> None:
        self.closed += 1
