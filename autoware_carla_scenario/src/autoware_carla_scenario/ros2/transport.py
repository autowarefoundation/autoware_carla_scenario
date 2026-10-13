"""The DDS transport the ROS 2 ego entity talks through.

:class:`Ros2EgoTransport` is all the entity sees: it publishes the ego's state and
hands over the commands that arrived since the last poll.  The Cyclone DDS
implementation lives in :mod:`.cyclone` and imports ``cyclonedds`` (the ``ros2``
extra); :class:`FakeRos2Transport` stands in for it in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from .commands import Command, EgoState
from .config import Ros2EgoConfig

__all__ = ["FakeRos2Transport", "Received", "Ros2EgoTransport"]

#: A command and the name of the source it came from.
Received = Tuple[str, Command]


class Ros2EgoTransport(ABC):
    """Moves the ego's state out to ROS 2 and its commands in.

    The lifecycle is :meth:`open` once when the scenario starts, then each tick
    :meth:`publish_state` and :meth:`poll`, and :meth:`close` at teardown.

    Args:
        config: Topics, domain and command sources.
    """

    _config: Ros2EgoConfig

    def __init__(self, config: Ros2EgoConfig) -> None:
        self._config = config

    @property
    def config(self) -> Ros2EgoConfig:
        """Return the configuration."""
        return self._config

    @abstractmethod
    def open(self) -> None:
        """Join the DDS domain and create the publishers and subscribers."""
        ...

    @abstractmethod
    def publish_state(self, state: EgoState) -> None:
        """Publish the ego's state on every configured state topic."""
        ...

    @abstractmethod
    def poll(self) -> List[Received]:
        """Return the commands that arrived since the last poll, oldest first."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Leave the domain.  Must be idempotent."""
        ...


class FakeRos2Transport(Ros2EgoTransport):
    """An in-memory transport: records what is published, replays what is queued.

    Queue commands with :meth:`deliver`; the next :meth:`poll` returns them.
    """

    opened: int
    closed: int
    published: List[EgoState]
    _queue: List[Received]

    def __init__(self, config: Ros2EgoConfig) -> None:
        super().__init__(config)
        self.opened = 0
        self.closed = 0
        self.published = []
        self._queue = []

    def deliver(self, source_name: str, command: Command) -> None:
        """Queue *command* as if *source_name* had just published it."""
        self._queue.append((source_name, command))

    def open(self) -> None:
        self.opened += 1

    def publish_state(self, state: EgoState) -> None:
        self.published.append(state)

    def poll(self) -> List[Received]:
        received, self._queue = self._queue, []
        return received

    def close(self) -> None:
        self.closed += 1
