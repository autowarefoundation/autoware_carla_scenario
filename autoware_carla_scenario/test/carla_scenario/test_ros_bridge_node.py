"""Tests for the ``scenario_bridge`` node's routing, without ROS 2.

:mod:`autoware_carla_scenario.autoware_bridge.ros_bridge.node` imports ``rclpy``
and Autoware's message packages, which the host does not have.  These tests put
small stand-ins for them in ``sys.modules`` -- just what the node touches: a
``Node`` that hands out recording service clients and timers, and message
classes with the fields it fills -- and import the node against them.  The
scenario side is the framework's real gRPC server on a loopback port, so the
mission travels the wire exactly as in a run.

AD API service calls are answered by hand (:meth:`_Future.answer`), which is
what lets a test say when Autoware accepted or rejected a request.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from autoware_carla_scenario.autoware_bridge import (
    AutowareBridgeConfig,
    BridgePose,
    GrpcAutowareBridgeServer,
)
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
    OPERATION_MODE_AUTONOMOUS,
    ROUTE_STATE_SET,
    ROUTE_STATE_UNSET,
    ROUTING_CHANGE_ROUTE_POINTS_SERVICE,
    ROUTING_SET_ROUTE_POINTS_SERVICE,
)

_NODE_MODULE = "autoware_carla_scenario.autoware_bridge.ros_bridge.node"

# ---------------------------------------------------------------------------
# Stand-ins for rclpy and the message packages
# ---------------------------------------------------------------------------


class _Future:
    """An ``rclpy`` future the test answers."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[Any], None]] = []
        self._done = False
        self._cancelled = False
        self._response: Any = None

    def add_done_callback(self, callback: Callable[[Any], None]) -> None:
        self._callbacks.append(callback)

    def done(self) -> bool:
        return self._done

    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = self._done = True

    def result(self) -> Any:
        return self._response

    def answer(self, success: bool, message: str = "") -> None:
        self._response = SimpleNamespace(
            status=SimpleNamespace(success=success, code=0, message=message)
        )
        self._done = True
        for callback in self._callbacks:
            callback(self)


class _ServiceClient:
    def __init__(self, srv_type: Any, name: str) -> None:
        self.srv_type = srv_type
        self.name = name
        self.calls: list[tuple[Any, _Future]] = []

    def service_is_ready(self) -> bool:
        return True

    def call_async(self, request: Any) -> _Future:
        future = _Future()
        self.calls.append((request, future))
        return future


class _Timer:
    def __init__(self, period: float, callback: Callable[[], None]) -> None:
        self.period = period
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _Logger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _log(self, level: str, message: str, **_kwargs: Any) -> None:
        self.records.append((level, message))

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)


#: ROS parameters the next node is given, by name, over its declared defaults.
_PARAMETERS: dict[str, Any] = {}


class _Node:
    """``rclpy.node.Node``, as much of it as the node uses."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.clients: dict[str, _ServiceClient] = {}
        self.timers: list[_Timer] = []
        self.logger = _Logger()

    def declare_parameter(self, name: str, default: Any) -> Any:
        value = _PARAMETERS.get(name, default)
        typed = SimpleNamespace(
            string_value=value,
            bool_value=value,
            double_value=value,
            integer_value=value,
        )
        return SimpleNamespace(get_parameter_value=lambda: typed)

    def create_client(self, srv_type: Any, name: str, **_kwargs: Any) -> _ServiceClient:
        client = _ServiceClient(srv_type, name)
        self.clients[name] = client
        return client

    def create_subscription(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def create_timer(
        self, period: float, callback: Callable[[], None], **_kwargs: Any
    ) -> _Timer:
        timer = _Timer(period, callback)
        self.timers.append(timer)
        return timer

    def get_logger(self) -> _Logger:
        return self.logger

    def get_clock(self) -> Any:
        now = SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=0, nanosec=0))
        return SimpleNamespace(now=lambda: now)

    def destroy_node(self) -> bool:
        return True


def _point() -> SimpleNamespace:
    return SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _Pose:
    def __init__(self) -> None:
        self.position = _point()
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


class _PoseWithCovarianceStamped:
    def __init__(self) -> None:
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = SimpleNamespace(pose=_Pose(), covariance=[0.0] * 36)


class _SetRoutePoints:
    class Request:
        def __init__(self) -> None:
            self.header = SimpleNamespace(frame_id="", stamp=None)
            self.option = SimpleNamespace(allow_goal_modification=False)
            self.goal = _Pose()
            self.waypoints: list[_Pose] = []


class _InitializeLocalization:
    class Request:
        def __init__(self) -> None:
            self.pose: list[Any] = []


class _ChangeOperationMode:
    class Request:
        pass


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _fake_ros_modules() -> dict[str, types.ModuleType]:
    """The modules the node imports, as stand-ins."""
    states = {
        "LocalizationInitializationState": SimpleNamespace(INITIALIZED=3),
        "OperationModeState": SimpleNamespace(AUTONOMOUS=OPERATION_MODE_AUTONOMOUS),
        "RouteState": SimpleNamespace(UNKNOWN=0, UNSET=ROUTE_STATE_UNSET, SET=2),
    }
    qos = _module(
        "rclpy.qos",
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
        QoSDurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL="transient_local"),
        QoSHistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        QoSReliabilityPolicy=SimpleNamespace(RELIABLE="reliable"),
    )
    return {
        "rclpy": _module("rclpy", init=None, ok=lambda: False, shutdown=None),
        "rclpy.callback_groups": _module(
            "rclpy.callback_groups", ReentrantCallbackGroup=object
        ),
        "rclpy.executors": _module("rclpy.executors", MultiThreadedExecutor=object),
        "rclpy.node": _module("rclpy.node", Node=_Node),
        "rclpy.qos": qos,
        "autoware_adapi_v1_msgs": _module("autoware_adapi_v1_msgs"),
        "autoware_adapi_v1_msgs.msg": _module("autoware_adapi_v1_msgs.msg", **states),
        "autoware_adapi_v1_msgs.srv": _module(
            "autoware_adapi_v1_msgs.srv",
            ChangeOperationMode=_ChangeOperationMode,
            InitializeLocalization=_InitializeLocalization,
            SetRoutePoints=_SetRoutePoints,
        ),
        "autoware_vehicle_msgs": _module("autoware_vehicle_msgs"),
        "autoware_vehicle_msgs.msg": _module(
            "autoware_vehicle_msgs.msg", TurnIndicatorsCommand=object
        ),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module(
            "geometry_msgs.msg",
            Pose=_Pose,
            PoseWithCovarianceStamped=_PoseWithCovarianceStamped,
        ),
    }


@pytest.fixture
def node_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """The node module, imported against the stand-ins; removed afterwards."""
    for name, module in _fake_ros_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, _NODE_MODULE, raising=False)
    module = importlib.import_module(_NODE_MODULE)
    try:
        yield module
    finally:
        # Not left for a test that checks the package imports without rclpy.
        sys.modules.pop(_NODE_MODULE, None)
        package = sys.modules["autoware_carla_scenario.autoware_bridge.ros_bridge"]
        if getattr(package, "node", None) is module:
            delattr(package, "node")


@pytest.fixture
def bridge() -> Iterator[GrpcAutowareBridgeServer]:
    server = GrpcAutowareBridgeServer(AutowareBridgeConfig(address="localhost:0"))
    try:
        yield server
    finally:
        server.close()


_INITIAL = BridgePose.from_yaw(1.0, 2.0, 0.0, 0.0)
_GOAL = BridgePose.from_yaw(90.0, 0.0, 0.0, 0.0)
_WAYPOINTS = [BridgePose.from_yaw(x, 0.0, 0.0, 0.0) for x in (30.0, 60.0)]


def _node(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    **parameters: Any,
) -> Any:
    """A node dialling *bridge*, localizing outside the AD API to keep it short."""
    _PARAMETERS.clear()
    _PARAMETERS.update(
        bridge_address=bridge.client_address,
        initialize_localization=False,
        require_localization_initialized=False,
        **parameters,
    )
    try:
        node = node_module.ScenarioBridgeNode()
    finally:
        _PARAMETERS.clear()
    return node


def _route_calls(node: Any) -> list[tuple[Any, _Future]]:
    return node.clients[ROUTING_SET_ROUTE_POINTS_SERVICE].calls


def _change_calls(node: Any) -> list[tuple[Any, _Future]]:
    return node.clients[ROUTING_CHANGE_ROUTE_POINTS_SERVICE].calls


def _xs(request: Any) -> tuple[list[float], float]:
    """The waypoints' and the goal's x, which is what tells these routes apart."""
    return [pose.position.x for pose in request.waypoints], request.goal.position.x


def _route_state(state: int) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _operation_mode(mode: int, engaged: bool) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        is_autoware_control_enabled=engaged,
        is_autonomous_mode_available=True,
    )


def _drive_to_ready(node: Any) -> None:
    """Route, engage and report ready, as Autoware would answer."""
    node._tick()  # noqa: SLF001 - the mission
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    _route_calls(node)[-1][1].answer(True)
    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    node._on_operation_mode_state(_operation_mode(1, False))  # noqa: SLF001
    engage = node.clients["/api/operation_mode/change_to_autonomous"].calls
    engage[-1][1].answer(True)
    node._on_operation_mode_state(  # noqa: SLF001
        _operation_mode(OPERATION_MODE_AUTONOMOUS, True)
    )
    node._tick()  # noqa: SLF001 - reports readiness


def _watch(node: Any) -> None:
    """One tick of the route watch the node started once it was ready."""
    watches = [t for t in node.timers if t.callback == node._watch_route]  # noqa: SLF001
    assert len(watches) == 1, "the route watch is running"
    watches[0].callback()


# ---------------------------------------------------------------------------
# #872: the waypoints reach set_route_points
# ---------------------------------------------------------------------------


def test_the_route_asked_for_passes_through_the_scenarios_waypoints(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL, _WAYPOINTS)
    node = _node(node_module, bridge)

    node._tick()  # noqa: SLF001
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001

    ((request, _future),) = _route_calls(node)
    assert _xs(request) == ([30.0, 60.0], 90.0)
    assert request.header.frame_id == "map"


def test_a_mission_without_waypoints_asks_for_none(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)

    node._tick()  # noqa: SLF001
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001

    ((request, _future),) = _route_calls(node)
    assert _xs(request) == ([], 90.0)


# ---------------------------------------------------------------------------
# #873: a re-route reaches change_route_points
# ---------------------------------------------------------------------------


def test_the_node_watches_for_a_re_route_once_ready(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge, route_watch_period_s=2.0)

    _drive_to_ready(node)

    assert bridge.is_ready()
    assert node.timers[0].cancelled  # the startup tick
    assert [t.period for t in node.timers if not t.cancelled] == [2.0]


def test_a_re_route_changes_autowares_route(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _drive_to_ready(node)

    _watch(node)  # nothing new yet
    assert _change_calls(node) == []

    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0), _WAYPOINTS)
    _watch(node)

    ((request, future),) = _change_calls(node)
    assert _xs(request) == ([30.0, 60.0], 120.0)
    assert (
        len(_route_calls(node)) == 1
    )  # set once, at the start, and only changed since

    _watch(node)  # in flight: not sent again
    future.answer(True)
    _watch(node)  # the route Autoware has: not sent again
    assert len(_change_calls(node)) == 1

    bridge.reroute(_GOAL)
    _watch(node)
    assert [_xs(call[0]) for call in _change_calls(node)] == [
        ([30.0, 60.0], 120.0),
        ([], 90.0),
    ]


def test_no_route_change_before_readiness(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """The startup routes whatever the mission is; only after it is a change a change."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    node._tick()  # noqa: SLF001
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    _route_calls(node)[-1][1].answer(True)
    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001

    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0))
    for _ in range(3):
        node._tick()  # noqa: SLF001 - still starting: not ready
    assert _change_calls(node) == []
    assert not [t for t in node.timers if t.callback == node._watch_route]  # noqa: SLF001

    node._on_operation_mode_state(_operation_mode(1, False))  # noqa: SLF001
    node.clients["/api/operation_mode/change_to_autonomous"].calls[-1][1].answer(True)
    node._on_operation_mode_state(  # noqa: SLF001
        _operation_mode(OPERATION_MODE_AUTONOMOUS, True)
    )
    node._tick()  # noqa: SLF001 - ready now
    _watch(node)

    ((request, _future),) = _change_calls(node)
    assert _xs(request) == ([], 120.0)


def test_a_change_waits_for_the_route_to_be_set(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _drive_to_ready(node)
    node._on_route_state(_route_state(3))  # noqa: SLF001 - CHANGING

    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0))
    _watch(node)
    assert _change_calls(node) == []

    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    _watch(node)
    assert len(_change_calls(node)) == 1


def test_a_rejected_change_is_retried_then_given_up_on(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge, reroute_max_attempts=2)
    _drive_to_ready(node)
    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0))

    _watch(node)
    _change_calls(node)[-1][1].answer(False, "reroute is not safe")
    _watch(node)  # retried
    _change_calls(node)[-1][1].answer(False, "reroute is not safe")
    _watch(node)  # given up on
    _watch(node)

    assert len(_change_calls(node)) == 2
    errors = [m for level, m in node.logger.records if level == "error"]
    assert len(errors) == 1 and "giving up" in errors[0]

    # A later re-route is tried afresh.
    bridge.reroute(BridgePose.from_yaw(150.0, 0.0, 0.0, 0.0))
    _watch(node)
    assert _xs(_change_calls(node)[-1][0]) == ([], 150.0)
