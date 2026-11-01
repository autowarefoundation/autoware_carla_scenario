"""Backends a run needs started before it: declared, found, started, stopped.

Some parts of a scenario run on more than CARLA: a pedestrian whose motion a
model generates (``SmartPedestrianControllerAction`` and its ARDY container),
a renderer served by a process of its own.  Such a part says so with
:func:`~autoware_carla_scenario.prepare.declare.need_prepare` on its class::

    @need_prepare(ArdyBackend)
    class SmartPedestrianControllerAction(BaseAction): ...

and the queue (:class:`~autoware_carla_scenario.scenario_queue.ScenarioQueue`)
has a :class:`BackendPreparer` start what the scenarios it runs need before
the first of them starts: what a declarative scenario's actions and a
scenario class declare is read before CARLA starts; what a scenario's
``setup()`` registers is checked again before its run ticks, and started
then if it was not.  Before starting anything the preparer adds up what the
backends need (:class:`Resources`) and refuses, with
:class:`InsufficientResourcesError`, a host that has less to give; the
backends run until the queue stops.

This module holds the contract only (checked by Codon, as the queue and the
runner that use it are); the decorator, the preparer and the backends live
in its submodules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from ..actions.base import BaseAction
    from ..scenario_base import BaseScenario

__all__ = [
    "Backend",
    "BackendError",
    "BackendPreparer",
    "InsufficientResourcesError",
    "Resources",
]


class BackendError(RuntimeError):
    """A backend that cannot be started, or stopped answering."""


class InsufficientResourcesError(BackendError):
    """The host has less than the backends a run needs would take."""


@dataclass(frozen=True)
class Resources:
    """What a backend takes of the host while it runs.

    Attributes:
        gpu_memory_mb: Memory on one GPU, MiB (0: no GPU).
        memory_mb: Host memory, MiB.
        gpu: The GPU (``nvidia-smi`` index) it must run on; ``None``: any.
    """

    gpu_memory_mb: int = 0
    memory_mb: int = 0
    gpu: Optional[int] = None


@runtime_checkable
class Backend(Protocol):
    """A service a run needs started before it, and stopped after."""

    @property
    def name(self) -> str:
        """Its name: one backend of a name runs per preparer."""
        ...

    def resources(self) -> Resources:
        """What it takes of the host while it runs."""
        ...

    def start(self, log_dir: Optional[Path] = None, gpu: Optional[int] = None) -> None:
        """Start it and return once it answers.

        Args:
            log_dir: Where its output goes; ``None`` keeps its own.
            gpu: The GPU (``nvidia-smi`` index) it was placed on, for one
                that needs a GPU.

        Raises:
            BackendError: If it could not be started or never answered.
        """
        ...

    def address(self) -> str:
        """``host:port`` it answers on, once started."""
        ...

    def close(self) -> None:
        """Stop it and whatever it started.  Idempotent; never raises."""
        ...


@runtime_checkable
class BackendPreparer(Protocol):
    """Starts the backends the scenarios and actions of a queue need."""

    def prepare_scenarios(
        self, scenarios: Sequence["BaseScenario"], log_dir: Optional[Path] = None
    ) -> None:
        """Start what *scenarios* declare they need, before CARLA starts.

        What a declarative scenario's actions and a scenario's class declare
        is known before the scenario is set up.

        Raises:
            InsufficientResourcesError: If the host cannot run them.
            BackendError: If one could not be started.
        """
        ...

    def prepare_actions(
        self, actions: Sequence["BaseAction"], log_dir: Optional[Path] = None
    ) -> None:
        """Start what *actions* (a set-up scenario's) need and is not running.

        Raises:
            InsufficientResourcesError: If the host cannot run them.
            BackendError: If one could not be started.
        """
        ...

    def close(self) -> None:
        """Stop every backend started.  Idempotent; never raises."""
        ...
