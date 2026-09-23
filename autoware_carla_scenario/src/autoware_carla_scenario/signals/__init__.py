"""Traffic signal controllers: a junction's cycle, declared and run.

The scenario document declares controllers on its :class:`MapRef`, because a
junction's cycle is a property of the road network rather than of the
storyboard -- which is where OpenSCENARIO puts it too, under
``RoadNetwork/TrafficSignals``.  This package is the runtime half.
"""

from .controller import (
    STATE_NAMES,
    Phase,
    PhaseState,
    SignalController,
    build_controllers,
)
from .registry import (
    clear_signal_controllers,
    find_signal_controller,
    register_signal_controller,
    registered_signal_controllers,
)

__all__ = [
    "STATE_NAMES",
    "Phase",
    "PhaseState",
    "SignalController",
    "build_controllers",
    "clear_signal_controllers",
    "find_signal_controller",
    "register_signal_controller",
    "registered_signal_controllers",
]
