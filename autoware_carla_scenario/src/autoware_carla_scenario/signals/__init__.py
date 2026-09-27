"""Traffic signal controllers: a junction's cycle, declared and run.

The scenario document declares a junction in two halves on its :class:`MapRef`.
``signal_groups`` is the road -- which signals move together, which movements
cross -- and is the same for every scenario that runs there.
``traffic_signal_controllers`` is the timing, which is this scenario's to
choose; OpenSCENARIO agrees, keeping its controllers under
``RoadNetwork/TrafficSignals``, a section of the scenario file rather than of
the map.

This package is the runtime half.  It deals only in signals: the groups are
resolved away by :func:`build_controllers`.
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
