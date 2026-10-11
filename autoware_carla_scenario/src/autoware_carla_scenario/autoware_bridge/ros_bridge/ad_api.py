"""Autoware AD API names and the readiness aggregation logic.

The topic/service names are the stable AD API contract (``autoware_adapi_specs``).
:class:`ReadinessAggregator` is kept free of ROS 2 -- it operates on the plain
enum values, written out here so the module imports without ``rclpy`` or
``autoware_adapi_v1_msgs`` and the readiness/engage logic is unit tested without
an Autoware stack.  The node checks them against the message classes when it
starts (:func:`check_enum_values`), so a value that drifted fails there.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- AD API interface names (see autoware_adapi_specs) -----------------------

#: Service: initialize localization at a pose (empty pose = GNSS auto-init).
LOCALIZATION_INITIALIZE_SERVICE = "/api/localization/initialize"
#: Topic: LocalizationInitializationState.
LOCALIZATION_INITIALIZATION_STATE_TOPIC = "/api/localization/initialization_state"
#: Service: set a route from a goal pose (+ optional waypoints).
ROUTING_SET_ROUTE_POINTS_SERVICE = "/api/routing/set_route_points"
#: Topic: RouteState.
ROUTING_STATE_TOPIC = "/api/routing/state"
#: Topic: OperationModeState.
OPERATION_MODE_STATE_TOPIC = "/api/operation_mode/state"
#: Service: change the operation mode to autonomous.
OPERATION_MODE_CHANGE_TO_AUTONOMOUS_SERVICE = "/api/operation_mode/change_to_autonomous"

# -- Enum values (autoware_adapi_v1_msgs; see check_enum_values) --------------

#: LocalizationInitializationState.INITIALIZED.
LOCALIZATION_STATE_INITIALIZED = 3
#: RouteState.SET.
ROUTE_STATE_SET = 2
#: RouteState.UNSET -- the only state the AD API accepts a route in.
ROUTE_STATE_UNSET = 1
#: RouteState.UNKNOWN -- what the AD API reports before the planner is up.
ROUTE_STATE_UNKNOWN = 0
#: OperationModeState.AUTONOMOUS.
OPERATION_MODE_AUTONOMOUS = 2


#: The values above, by their names in ``autoware_adapi_v1_msgs``.
ENUM_VALUES: dict[str, int] = {
    "LocalizationInitializationState.INITIALIZED": LOCALIZATION_STATE_INITIALIZED,
    "RouteState.UNKNOWN": ROUTE_STATE_UNKNOWN,
    "RouteState.UNSET": ROUTE_STATE_UNSET,
    "RouteState.SET": ROUTE_STATE_SET,
    "OperationModeState.AUTONOMOUS": OPERATION_MODE_AUTONOMOUS,
}


def check_enum_values(defined: dict[str, int]) -> None:
    """Check the values above against what ``autoware_adapi_v1_msgs`` defines.

    *defined* maps each name in :data:`ENUM_VALUES` to the message class's
    constant, as the node reads it.

    Raises:
        RuntimeError: If the AD API defines one differently, or not at all.
    """
    wrong: list[str] = []
    for name, expected in ENUM_VALUES.items():
        if name not in defined:
            wrong.append(f"{name} is missing")
        elif defined[name] != expected:
            wrong.append(f"{name} = {defined[name]} (expected {expected})")
    if wrong:
        raise RuntimeError("autoware_adapi_v1_msgs defines " + ", ".join(wrong))


@dataclass
class ReadinessAggregator:
    """Folds the three AD API states into the engage decision and readiness flag.

    Feed it the plain enum values from the AD API message callbacks; it exposes:

    * :attr:`can_engage` - localization initialized, route set, and autonomous
      mode available: the precondition for calling ``change_to_autonomous``.
    * :attr:`ready` - localization initialized, route set, and Autoware now in
      AUTONOMOUS with control enabled: the flag pushed to the scenario framework.

    ``require_localization`` folds in the localization state.  Set it ``False`` for
    stacks that do not localize through the AD API -- e.g. a CARLA ground-truth /
    E2E-planner setup (``localization:=false``), where ``/localization/kinematic_state``
    is published directly and ``LocalizationInitializationState`` never reaches
    INITIALIZED.  It defaults to ``True`` to match the mainline flow.
    """

    require_localization: bool = True
    localization_initialized: bool = False
    #: The last observed ``RouteState.state``; UNKNOWN until one is received.
    route_state: int = ROUTE_STATE_UNKNOWN
    autonomous_available: bool = False
    autonomous_engaged: bool = False

    def update_localization(self, state: int) -> None:
        """Update from ``LocalizationInitializationState.state``."""
        self.localization_initialized = state == LOCALIZATION_STATE_INITIALIZED

    def update_routing(self, state: int) -> None:
        """Update from ``RouteState.state``."""
        self.route_state = state

    @property
    def route_set(self) -> bool:
        return self.route_state == ROUTE_STATE_SET

    @property
    def route_acceptable(self) -> bool:
        """Whether the AD API would take a route now.

        ``RoutingNode::on_set_route_points`` refuses anything but UNSET, and it
        starts at UNKNOWN -- so before the mission planner has published its
        first ``RouteState`` the API answers "The route is already set." for a
        stack that has no route at all.
        """
        return self.route_state == ROUTE_STATE_UNSET

    def update_operation_mode(
        self, mode: int, is_control_enabled: bool, is_autonomous_available: bool
    ) -> None:
        """Update from ``OperationModeState`` fields."""
        self.autonomous_available = is_autonomous_available
        self.autonomous_engaged = (
            mode == OPERATION_MODE_AUTONOMOUS and is_control_enabled
        )

    @property
    def _localization_ok(self) -> bool:
        """Whether localization is satisfied (always ``True`` when not required)."""
        return self.localization_initialized or not self.require_localization

    @property
    def can_engage(self) -> bool:
        """Whether ``change_to_autonomous`` may be called now."""
        return self._localization_ok and self.route_set and self.autonomous_available

    @property
    def ready(self) -> bool:
        """Whether Autoware is initialized, routed, engaged, and driving."""
        return self._localization_ok and self.route_set and self.autonomous_engaged
