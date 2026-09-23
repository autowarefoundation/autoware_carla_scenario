"""Where a running scenario's signal controllers can be found by name.

The same shape as :mod:`autoware_carla_scenario.entity.registry`, and for the
same reason: an action and a condition written in a document name a controller
in a string, and both have to reach the one object the scenario is running.
"""

from __future__ import annotations

import logging
from typing import Optional

from .controller import SignalController

logger = logging.getLogger(__name__)

__all__ = [
    "clear_signal_controllers",
    "find_signal_controller",
    "register_signal_controller",
    "registered_signal_controllers",
]

_CONTROLLERS: "dict[str, SignalController]" = {}


def register_signal_controller(controller: SignalController) -> None:
    """Publish *controller* under its own name, replacing any of that name."""
    if controller.name in _CONTROLLERS:
        logger.warning(
            "A signal controller named '%s' was already registered; replacing it",
            controller.name,
        )
    _CONTROLLERS[controller.name] = controller


def find_signal_controller(name: str) -> Optional[SignalController]:
    """Return the controller called *name*, or ``None``."""
    return _CONTROLLERS.get(name)


def registered_signal_controllers() -> "tuple[SignalController, ...]":
    """Return every registered controller, in registration order."""
    return tuple(_CONTROLLERS.values())


def clear_signal_controllers() -> None:
    """Forget every controller.

    Called between scenarios of a queue: a controller is the *run's*, and one
    left behind would keep driving lights for the next scenario, which would
    look like that scenario's own junction misbehaving.
    """
    _CONTROLLERS.clear()
