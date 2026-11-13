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
import logging
import sys
import time
import types
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import grpc
import pytest

from autoware_carla_scenario.autoware_bridge import (
    AutowareBridgeConfig,
    BridgePose,
    GrpcAutowareBridgeServer,
)
from autoware_carla_scenario.constants import EGO_ROLE_NAME
from autoware_carla_scenario.autoware_bridge.ros_bridge.ad_api import (
    OPERATION_MODE_AUTONOMOUS,
    ROUTE_STATE_UNKNOWN,
    ROUTE_STATE_ARRIVED,
    ROUTE_STATE_CHANGING,
    ROUTE_STATE_SET,
    ROUTE_STATE_UNSET,
    ROUTING_CHANGE_ROUTE_POINTS_SERVICE,
    ROUTING_CLEAR_ROUTE_SERVICE,
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
        #: What service_is_ready() says: whether the AD API serves it.
        self.ready = True

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: Any) -> _Future:
        future = _Future()
        self.calls.append((request, future))
        return future


class _Timer:
    def __init__(
        self, period: float, callback: Callable[[], None], clock: Any = None
    ) -> None:
        self.period = period
        self.callback = callback
        self.clock = clock
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
        self,
        period: float,
        callback: Callable[[], None],
        clock: Any = None,
        **_kwargs: Any,
    ) -> _Timer:
        timer = _Timer(period, callback, clock)
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


class _ClearRoute:
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
        "RouteState": SimpleNamespace(
            UNKNOWN=0,
            UNSET=ROUTE_STATE_UNSET,
            SET=ROUTE_STATE_SET,
            ARRIVED=ROUTE_STATE_ARRIVED,
            CHANGING=ROUTE_STATE_CHANGING,
        ),
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
        "rclpy.clock": _module(
            "rclpy.clock",
            Clock=lambda clock_type=None: SimpleNamespace(clock_type=clock_type),
            ClockType=SimpleNamespace(STEADY_TIME="steady", ROS_TIME="ros"),
        ),
        "rclpy.node": _module("rclpy.node", Node=_Node),
        "rclpy.qos": qos,
        "autoware_adapi_v1_msgs": _module("autoware_adapi_v1_msgs"),
        "autoware_adapi_v1_msgs.msg": _module("autoware_adapi_v1_msgs.msg", **states),
        "autoware_adapi_v1_msgs.srv": _module(
            "autoware_adapi_v1_msgs.srv",
            ChangeOperationMode=_ChangeOperationMode,
            ClearRoute=_ClearRoute,
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
        {
            "bridge_address": bridge.client_address,
            "initialize_localization": False,
            "require_localization_initialized": False,
            **parameters,
        }
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


#: How many times, and for how long, :func:`_repeat_until` steps the node.
#: Each step's gRPC call is bounded by the node's own timeout (1 s for
#: GetMission, 5 s for ReportReadiness), so this only ends a test whose node
#: never gets there.
_STEPS = 20
_STEPS_TIMEOUT_S = 30.0


def _repeat_until(step: Callable[[], None], predicate: Callable[[], Any]) -> None:
    """Run *step* until *predicate* holds: at least once, at most ``_STEPS`` times.

    A step's GetMission and ReportReadiness are blocking calls to the real
    server, and on a loaded machine one can time out -- the first, which also
    connects the channel, most of all.  The node retries it on its next tick
    (or route watch), so a test that waits on the round trip steps the node
    until it has landed, as its timer would, instead of once (#2148).
    Whatever a test says did *not* happen is unaffected: the node never
    reaches the bridge any sooner for being stepped more.
    """
    deadline = time.monotonic() + _STEPS_TIMEOUT_S
    for _ in range(_STEPS):
        step()
        if predicate():
            return
        if time.monotonic() > deadline:
            break
    raise AssertionError(f"not reached after {_STEPS} steps: {predicate}")


def _tick_until(node: Any, predicate: Callable[[], Any]) -> None:
    """Tick *node* until *predicate* holds (see :func:`_repeat_until`)."""
    _repeat_until(node._tick, predicate)  # noqa: SLF001


def _watching(node: Any) -> bool:
    """Whether the node has reported ready and started its route watch."""
    return any(t.callback == node._watch_route for t in node.timers)  # noqa: SLF001


def _drive_to_ready(node: Any) -> None:
    """Route, engage and report ready, as Autoware would answer."""
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001 - AD API up
    _tick_until(node, lambda: _route_calls(node))  # the mission
    _route_calls(node)[-1][1].answer(True)
    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    node._on_operation_mode_state(_operation_mode(1, False))  # noqa: SLF001
    engage = node.clients["/api/operation_mode/change_to_autonomous"].calls
    engage[-1][1].answer(True)
    node._on_operation_mode_state(  # noqa: SLF001
        _operation_mode(OPERATION_MODE_AUTONOMOUS, True)
    )
    _tick_until(node, lambda: _watching(node))  # reports readiness


def _watch(node: Any) -> None:
    """One tick of the route watch the node started once it was ready."""
    watches = [t for t in node.timers if t.callback == node._watch_route]  # noqa: SLF001
    assert len(watches) == 1, "the route watch is running"
    watches[0].callback()


def _watch_until(node: Any, predicate: Callable[[], Any]) -> None:
    """Route-watch *node* until *predicate* holds: a re-route is seen only once
    its GetMission lands (see :func:`_repeat_until`)."""
    _repeat_until(lambda: _watch(node), predicate)


# ---------------------------------------------------------------------------
# #872: the waypoints reach set_route_points
# ---------------------------------------------------------------------------


def test_the_route_asked_for_passes_through_the_scenarios_waypoints(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL, _WAYPOINTS)
    node = _node(node_module, bridge)
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001 - AD API up

    _tick_until(node, lambda: _route_calls(node))

    ((request, _future),) = _route_calls(node)
    assert _xs(request) == ([30.0, 60.0], 90.0)
    assert request.header.frame_id == "map"


def test_a_mission_without_waypoints_asks_for_none(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001 - AD API up

    _tick_until(node, lambda: _route_calls(node))

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
    _watch_until(node, lambda: _change_calls(node))

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
    _watch_until(node, lambda: len(_change_calls(node)) == 2)
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
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001 - AD API up
    _tick_until(node, lambda: _route_calls(node))
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
    _tick_until(node, lambda: _watching(node))  # ready now
    _watch_until(node, lambda: _change_calls(node))

    ((request, _future),) = _change_calls(node)
    assert _xs(request) == ([], 120.0)


def test_a_change_waits_for_the_route_to_be_set(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _drive_to_ready(node)
    node._on_route_state(_route_state(ROUTE_STATE_CHANGING))  # noqa: SLF001

    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0))
    _watch(node)
    assert _change_calls(node) == []

    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    _watch_until(node, lambda: _change_calls(node))
    assert len(_change_calls(node)) == 1


def test_a_rejected_change_is_retried_then_given_up_on(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge, reroute_max_attempts=2)
    _drive_to_ready(node)
    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0))

    _watch_until(node, lambda: _change_calls(node))
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
    _watch_until(node, lambda: len(_change_calls(node)) == 3)
    assert _xs(_change_calls(node)[-1][0]) == ([], 150.0)


def test_a_re_route_after_arrival_clears_the_route_sets_it_and_engages_again(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """ARRIVED takes neither routing service: the route is cleared first."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _drive_to_ready(node)
    engage = node.clients["/api/operation_mode/change_to_autonomous"].calls
    node._on_route_state(_route_state(ROUTE_STATE_ARRIVED))  # noqa: SLF001
    node._on_operation_mode_state(_operation_mode(1, False))  # noqa: SLF001 - STOP

    bridge.reroute(BridgePose.from_yaw(120.0, 0.0, 0.0, 0.0), _WAYPOINTS)
    _watch_until(node, lambda: node.clients[ROUTING_CLEAR_ROUTE_SERVICE].calls)

    assert _change_calls(node) == []
    clears = node.clients[ROUTING_CLEAR_ROUTE_SERVICE].calls
    assert len(clears) == 1
    _watch(node)  # in flight: not sent again
    clears[-1][1].answer(True)
    _watch(node)  # still reads ARRIVED: cleared, not cleared again
    assert len(clears) == 1
    assert len(_route_calls(node)) == 1

    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    assert len(_route_calls(node)) == 2
    assert _xs(_route_calls(node)[-1][0]) == ([30.0, 60.0], 120.0)
    _route_calls(node)[-1][1].answer(True)
    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    assert len(engage) == 2
    engage[-1][1].answer(True)

    _watch(node)  # the route Autoware has now: nothing more
    assert _change_calls(node) == []
    assert len(clears) == 1
    assert len(_route_calls(node)) == 2


def test_the_node_checks_every_route_state_it_relies_on(
    node_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() checks each of ad_api's values against the messages, none missing."""

    class _Started(Exception):
        pass

    def _init(**_kwargs: Any) -> None:
        raise _Started

    monkeypatch.setattr(node_module.rclpy, "init", _init)
    with pytest.raises(_Started):
        node_module.main()


# ---------------------------------------------------------------------------
# #2049: the mission is asked for only while the AD API is up
# ---------------------------------------------------------------------------


def _set_services_ready(node: Any, ready: bool) -> None:
    for client in node.clients.values():
        client.ready = ready


def test_the_node_does_not_ask_for_the_mission_before_the_ad_api_is_up(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """Started beside ros2 launch, the node would otherwise reach the bridge
    seconds in, and the scenario would stop waiting for Autoware to boot."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _set_services_ready(node, False)

    for _ in range(5):
        node._tick()  # noqa: SLF001
    assert not bridge.client_connected
    assert bridge.seconds_since_mission_poll() is None
    waiting = [m for _level, m in node.logger.records if "Waiting for" in m]
    assert waiting and ROUTING_SET_ROUTE_POINTS_SERVICE in waiting[-1]

    # The services are there, but the mission planner has not said a word.
    _set_services_ready(node, True)
    node._on_route_state(_route_state(ROUTE_STATE_UNKNOWN))  # noqa: SLF001
    node._tick()  # noqa: SLF001
    assert not bridge.client_connected

    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    _tick_until(node, lambda: bridge.client_connected)
    assert bridge.client_connected
    assert len(_route_calls(node)) == 1


def test_the_node_waits_only_for_the_services_it_calls(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """Localizing outside the AD API, the node needs no localization service."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    node.clients["/api/localization/initialize"].ready = False
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001

    _tick_until(node, lambda: bridge.client_connected)

    assert bridge.client_connected


def test_the_node_stops_asking_while_the_ad_api_is_down(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """Its silence is how the scenario knows the stack is gone."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    _drive_to_ready(node)
    _watch(node)
    asked = bridge.seconds_since_mission_poll()
    assert asked is not None

    node.clients[ROUTING_SET_ROUTE_POINTS_SERVICE].ready = False
    time.sleep(0.05)
    _watch(node)
    _watch(node)
    silent = bridge.seconds_since_mission_poll()
    assert silent is not None and silent >= asked + 0.05
    errors = [m for level, m in node.logger.records if level == "error"]
    assert len(errors) == 1 and "went away" in errors[0]

    node.clients[ROUTING_SET_ROUTE_POINTS_SERVICE].ready = True

    def _asked_again() -> bool:
        since = bridge.seconds_since_mission_poll()
        return since is not None and since < silent

    _watch_until(node, _asked_again)
    back = bridge.seconds_since_mission_poll()
    assert back is not None and back < silent


def test_a_stack_whose_ad_api_is_not_up_stays_in_the_paced_boot_phase(
    node_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issue's probe, with the node: it polls from its first tick, but the
    scenario keeps pacing the world under boot_timeout_s until the AD API is up,
    and only then counts ready_timeout_ticks."""
    from .test_autoware_ego_entity import (
        _FakeActor,
        _FakeLauncher,
        _FakeWorld,
        _launched_entity,
        _place,
    )

    # Where the ego is in the map frame does not matter here: no map is loaded.
    monkeypatch.setattr(
        "autoware_carla_scenario.entity.bridged_ego.to_map_frame",
        lambda _pose: _INITIAL,
    )
    bridge = GrpcAutowareBridgeServer(
        AutowareBridgeConfig(address="localhost:0"), autostart=False
    )
    entity = _launched_entity(
        bridge,
        _FakeLauncher(),
        ready_timeout_ticks=5,
        boot_timeout_s=60.0,
        boot_tick_period_s=0.01,
    )
    world = _FakeWorld([_FakeActor(42, str(EGO_ROLE_NAME))])
    _place(entity, world)
    node = None
    try:
        entity.on_scenario_start(world)
        node = _node(node_module, bridge)
        _set_services_ready(node, False)

        started = time.monotonic()
        for _ in range(20):
            node._tick()  # noqa: SLF001
            entity.on_tick(world, 0.0)
        paced_s = time.monotonic() - started

        assert not entity.termination_requested, entity.termination_reason
        assert not bridge.client_connected
        assert paced_s >= 20 * 0.01

        _set_services_ready(node, True)
        node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
        _tick_until(node, lambda: bridge.client_connected)
        for _ in range(5):
            entity.on_tick(world, 0.0)
        assert entity.termination_reason == "The Autoware stack not ready after 5 ticks"
    finally:
        if node is not None:
            node.destroy_node()
        entity.on_scenario_end(world)


#: LocalizationInitializationState values the pose initializer publishes.
_UNINITIALIZED = 1
_INITIALIZED = 3


def test_initializing_localization_does_not_wait_for_a_route_state(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """The mission planner reports UNKNOWN until localization is initialized,
    from the mission's initial pose: waiting for its state would deadlock."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(
        node_module,
        bridge,
        initialize_localization=True,
        require_localization_initialized=True,
    )
    initialize = node.clients["/api/localization/initialize"].calls

    node._on_route_state(_route_state(ROUTE_STATE_UNKNOWN))  # noqa: SLF001
    node._tick()  # noqa: SLF001 - the pose initializer is not up yet
    assert not bridge.client_connected
    waiting = [m for _level, m in node.logger.records if "Waiting for" in m]
    assert waiting and "localization state" in waiting[-1]

    node._on_localization_state(SimpleNamespace(state=_UNINITIALIZED))  # noqa: SLF001
    _tick_until(node, lambda: initialize)
    # It has the mission, but has not reported in: it cannot be routed yet.
    assert bridge.seconds_since_mission_poll() is not None
    assert not bridge.client_connected
    assert len(initialize) == 1
    assert _route_calls(node) == []  # the planner is still UNKNOWN

    initialize[-1][1].answer(True)
    node._on_localization_state(SimpleNamespace(state=_INITIALIZED))  # noqa: SLF001
    assert _route_calls(node) == []
    _tick_until(node, lambda: bridge.client_connected)  # reports in
    assert bridge.client_connected and not bridge.is_ready()
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    assert len(_route_calls(node)) == 1

    _route_calls(node)[-1][1].answer(True)
    node._on_route_state(_route_state(ROUTE_STATE_SET))  # noqa: SLF001
    node._on_operation_mode_state(_operation_mode(1, False))  # noqa: SLF001
    node.clients["/api/operation_mode/change_to_autonomous"].calls[-1][1].answer(True)
    node._on_operation_mode_state(  # noqa: SLF001
        _operation_mode(OPERATION_MODE_AUTONOMOUS, True)
    )
    _tick_until(node, bridge.is_ready)
    assert bridge.is_ready()


def test_the_node_polls_on_the_steady_clock(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """Liveness is wall-clock time: the polls must not follow a slow /clock."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge, use_sim_time=True)
    _drive_to_ready(node)

    assert len(node.timers) == 2  # the startup tick, then the route watch
    assert all(timer.clock.clock_type == "steady" for timer in node.timers)


# ---------------------------------------------------------------------------
# #2051: the stack has booted once it can be routed, not once it asked for
# the mission
# ---------------------------------------------------------------------------


class _RpcError(grpc.RpcError):
    def code(self) -> str:
        return "UNAVAILABLE"


def _readiness_reports(
    node: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[bool], list[bool]]:
    """Record each ReportReadiness the node sends, and each that reaches the bridge.

    Whether a call lands within one tick is up to the machine (the node
    retries one that fails on its next tick, #2148), so a test waits for it
    to land and checks what landed -- but one that says the node does not
    report checks what it sent, failed attempts included.
    """
    sent: list[bool] = []
    landed: list[bool] = []
    send = node._client.report_readiness  # noqa: SLF001

    def _report(ready: bool, timeout: Any = None) -> None:
        sent.append(ready)
        send(ready, timeout=timeout)
        landed.append(ready)

    monkeypatch.setattr(node._client, "report_readiness", _report)  # noqa: SLF001
    return sent, landed


def test_the_node_reports_in_once_localization_is_initialized(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pose initializer is up, NDT is not: each initialize is rejected and
    retried, and the node, which has the mission, does not report in."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(
        node_module,
        bridge,
        initialize_localization=True,
        require_localization_initialized=True,
    )
    sent, reports = _readiness_reports(node, monkeypatch)
    initialize = node.clients["/api/localization/initialize"].calls
    node._on_localization_state(SimpleNamespace(state=_UNINITIALIZED))  # noqa: SLF001

    _tick_until(node, lambda: initialize)  # the mission
    initialize[-1][1].answer(False, "ndt_align not ready")
    for _ in range(4):
        node._tick()  # noqa: SLF001
        initialize[-1][1].answer(False, "ndt_align not ready")
    assert len(initialize) == 5
    assert bridge.seconds_since_mission_poll() is not None
    assert sent == []
    assert not bridge.client_connected

    node._tick()  # noqa: SLF001
    initialize[-1][1].answer(True)
    node._on_localization_state(SimpleNamespace(state=_INITIALIZED))  # noqa: SLF001
    _tick_until(node, lambda: reports)
    assert reports == [False]
    assert bridge.client_connected and not bridge.is_ready()

    # Once only: not even tried again.
    tried = len(sent)
    node._tick()  # noqa: SLF001
    node._tick()  # noqa: SLF001
    assert reports == [False] and len(sent) == tried


def test_a_gnss_initialized_stack_reports_in_without_an_initialize(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """automatic_pose_initializer localizes from GNSS before the node asks."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(
        node_module,
        bridge,
        initialize_localization=True,
        require_localization_initialized=True,
    )
    sent, reports = _readiness_reports(node, monkeypatch)
    node._on_localization_state(SimpleNamespace(state=_INITIALIZED))  # noqa: SLF001

    _tick_until(node, lambda: reports)

    assert node.clients["/api/localization/initialize"].calls == []
    assert reports == [False]
    assert bridge.client_connected


def test_a_gnss_initialization_while_the_nodes_is_pending_reports_in(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNSS gets there first, while the node's own initialize is unanswered."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(
        node_module,
        bridge,
        initialize_localization=True,
        require_localization_initialized=True,
    )
    sent, reports = _readiness_reports(node, monkeypatch)
    initialize = node.clients["/api/localization/initialize"].calls
    node._on_localization_state(SimpleNamespace(state=_UNINITIALIZED))  # noqa: SLF001
    _tick_until(node, lambda: initialize)
    assert len(initialize) == 1
    assert sent == []

    node._on_localization_state(SimpleNamespace(state=_INITIALIZED))  # noqa: SLF001
    _tick_until(node, lambda: reports)

    assert reports == [False]


@pytest.mark.parametrize("auto_engage", [True, False])
def test_a_node_that_does_not_initialize_localization_reports_in_with_the_mission(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
    auto_engage: bool,
) -> None:
    """Its gate already waited for the mission planner: it can be routed."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge, auto_engage=auto_engage)
    sent, reports = _readiness_reports(node, monkeypatch)
    node._on_route_state(_route_state(ROUTE_STATE_UNKNOWN))  # noqa: SLF001
    node._tick()  # noqa: SLF001
    assert sent == [] and bridge.seconds_since_mission_poll() is None

    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    _tick_until(node, lambda: reports)

    assert reports == [False]
    assert bridge.client_connected and not bridge.is_ready()
    assert len(_route_calls(node)) == 1


@pytest.mark.parametrize("auto_engage", [True, False])
def test_a_node_without_auto_engage_reports_in_once_localized(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
    auto_engage: bool,
) -> None:
    """Engaging is after the boot: it plays no part in reporting in."""
    bridge.configure(_INITIAL, _GOAL)
    node = _node(
        node_module,
        bridge,
        initialize_localization=True,
        require_localization_initialized=True,
        auto_engage=auto_engage,
    )
    sent, reports = _readiness_reports(node, monkeypatch)
    node.clients["/api/operation_mode/change_to_autonomous"].ready = auto_engage
    initialize = node.clients["/api/localization/initialize"].calls
    node._on_localization_state(SimpleNamespace(state=_UNINITIALIZED))  # noqa: SLF001
    _tick_until(node, lambda: initialize)
    initialize[-1][1].answer(True)
    node._on_localization_state(SimpleNamespace(state=_INITIALIZED))  # noqa: SLF001

    _tick_until(node, lambda: reports)

    assert reports == [False]


def test_a_failed_report_in_is_retried(
    node_module: types.ModuleType,
    bridge: GrpcAutowareBridgeServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge.configure(_INITIAL, _GOAL)
    node = _node(node_module, bridge)
    send = node._client.report_readiness  # noqa: SLF001
    attempts: list[bool] = []

    def _report(ready: bool, timeout: Any = None) -> None:
        attempts.append(ready)
        if len(attempts) == 1:
            raise _RpcError()
        send(ready, timeout=timeout)

    monkeypatch.setattr(node._client, "report_readiness", _report)  # noqa: SLF001
    node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
    _tick_until(node, lambda: attempts)  # the mission, then the failed report
    assert attempts == [False]
    assert not bridge.client_connected
    assert [m for level, m in node.logger.records if level == "warning"]

    node._tick()  # noqa: SLF001
    assert attempts == [False, False]  # retried on the next tick
    _tick_until(node, lambda: bridge.client_connected)

    # More than once if the retry failed too, and only ever "booted".
    assert set(attempts) == {False}
    assert bridge.client_connected


def test_the_framework_knows_the_nodes_poll_defaults(
    node_module: types.ModuleType, bridge: GrpcAutowareBridgeServer
) -> None:
    """The liveness check compares with these when bridge_parameters omit them."""
    from autoware_carla_scenario.autoware_stack.bridge_node import (
        MISSION_POLL_TIMEOUT_S,
        ROUTE_WATCH_PERIOD_S,
    )

    node = _node(node_module, bridge)
    assert node._route_watch_period_s == ROUTE_WATCH_PERIOD_S  # noqa: SLF001
    assert node._mission_poll_timeout_s == MISSION_POLL_TIMEOUT_S  # noqa: SLF001


@pytest.mark.parametrize("initialize_localization", [True, False])
def test_ticks_stay_paced_until_the_stack_can_be_routed(
    node_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    initialize_localization: bool,
) -> None:
    """The issue's case, end to end over the real server: the pose
    initializer is up but NDT is not, so the node has the mission and keeps
    retrying the initialize. The scenario keeps pacing the world under
    boot_timeout_s, and counts ready_timeout_ticks only once localization is
    INITIALIZED."""
    from .test_autoware_ego_entity import (
        _FakeActor,
        _FakeLauncher,
        _FakeWorld,
        _launched_entity,
        _place,
    )

    monkeypatch.setattr(
        "autoware_carla_scenario.entity.bridged_ego.to_map_frame",
        lambda _pose: _INITIAL,
    )
    bridge = GrpcAutowareBridgeServer(
        AutowareBridgeConfig(address="localhost:0"), autostart=False
    )
    entity = _launched_entity(
        bridge,
        _FakeLauncher(),
        ready_timeout_ticks=5,
        boot_timeout_s=60.0,
        boot_tick_period_s=0.05,
    )
    world = _FakeWorld([_FakeActor(42, str(EGO_ROLE_NAME))])
    _place(entity, world)
    caplog.set_level(logging.INFO, logger="autoware_carla_scenario.entity.bridged_ego")
    node = None
    try:
        entity.on_scenario_start(world)
        node = _node(
            node_module,
            bridge,
            initialize_localization=initialize_localization,
            require_localization_initialized=initialize_localization,
        )
        initialize = node.clients["/api/localization/initialize"].calls
        if initialize_localization:
            # The pose initializer is up; the mission planner waits for it.
            node._on_localization_state(  # noqa: SLF001
                SimpleNamespace(state=_UNINITIALIZED)
            )
        node._on_route_state(_route_state(ROUTE_STATE_UNKNOWN))  # noqa: SLF001
        if initialize_localization:
            # The mission, which the first initialize waits for.
            _tick_until(node, lambda: initialize)
            initialize[-1][1].answer(False, "ndt_align not ready")

        started = time.monotonic()
        for _ in range(20):
            node._tick()  # noqa: SLF001
            if initialize and not initialize[-1][1].done():
                initialize[-1][1].answer(False, "ndt_align not ready")
            entity.on_tick(world, 0.0)
        paced_s = time.monotonic() - started

        assert not entity.termination_requested, entity.termination_reason
        assert not bridge.client_connected
        assert paced_s >= 20 * 0.05
        took = [r for r in caplog.records if "took the mission" in r.getMessage()]
        if initialize_localization:
            # It has the mission, and the scenario says so, once.
            assert bridge.seconds_since_mission_poll() is not None
            assert len(initialize) == 21
            assert len(took) == 1
        else:
            assert bridge.seconds_since_mission_poll() is None
            assert took == []

        # The map has loaded: NDT aligns, or the mission planner is up.
        if initialize_localization:
            node._tick()  # noqa: SLF001
            initialize[-1][1].answer(True)
            node._on_localization_state(  # noqa: SLF001
                SimpleNamespace(state=_INITIALIZED)
            )
        else:
            node._on_route_state(_route_state(ROUTE_STATE_UNSET))  # noqa: SLF001
        _tick_until(node, lambda: bridge.client_connected)

        started = time.monotonic()
        for _ in range(5):
            entity.on_tick(world, 0.0)
        assert time.monotonic() - started < 5 * 0.05
        assert entity.termination_reason == "The Autoware stack not ready after 5 ticks"
    finally:
        if node is not None:
            node.destroy_node()
        entity.on_scenario_end(world)
