"""Round-trip tests for ScenarioBridgeClient against an in-process server.

ROS 2-free: a stub server plays the scenario-library role on a real loopback
port, so these exercise the generated stubs and the client end to end without a
running Autoware stack.
"""

from concurrent import futures

from autoware_carla_scenario.autoware_bridge.ros_bridge.client import (
    ScenarioBridgeClient,
    TurnIndicatorForwarder,
)
from autoware_carla_scenario.autoware_bridge._proto import autoware_bridge_pb2 as pb2
from autoware_carla_scenario.autoware_bridge._proto import (
    autoware_bridge_pb2_grpc as pb2_grpc,
)
import grpc
import pytest


class _StubServer(pb2_grpc.AutowareBridgeServicer):
    """Records readiness reports and hands out a mission once one is set."""

    def __init__(self):
        self.mission = None  # (initial_pose, goal) or None
        self.reported = []
        self.turn_indicators = []

    def GetMission(self, request, context):  # noqa: N802 - gRPC signature
        if self.mission is None:
            return pb2.GetMissionResponse(available=False)
        initial, goal = self.mission
        return pb2.GetMissionResponse(available=True, initial_pose=initial, goal=goal)

    def ReportReadiness(self, request, context):  # noqa: N802 - gRPC signature
        self.reported.append(request.ready)
        return pb2.ReportReadinessResponse()

    def ReportTurnIndicators(self, request, context):  # noqa: N802 - gRPC signature
        self.turn_indicators.append(request.command)
        return pb2.ReportTurnIndicatorsResponse()


@pytest.fixture
def server():
    stub = _StubServer()
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb2_grpc.add_AutowareBridgeServicer_to_server(stub, grpc_server)
    stub.port = grpc_server.add_insecure_port("localhost:0")
    grpc_server.start()
    try:
        yield stub
    finally:
        grpc_server.stop(grace=None)


def _client(server):
    channel = grpc.insecure_channel(f"localhost:{server.port}")
    return ScenarioBridgeClient(channel=channel)


def _pose(x, y, z):
    return pb2.Pose(
        position=pb2.Vector3(x=x, y=y, z=z),
        rotation=pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    )


def test_get_mission_is_none_until_available(server):
    client = _client(server)
    assert client.get_mission(timeout=5.0) is None


def test_get_mission_returns_configured_mission(server):
    server.mission = (_pose(1.0, 2.0, 3.0), _pose(10.0, 20.0, 30.0))
    client = _client(server)

    mission = client.get_mission(timeout=5.0)

    assert mission is not None
    assert mission.initial_pose.position.x == pytest.approx(1.0)
    assert mission.goal.position.x == pytest.approx(10.0)
    assert mission.goal.position.z == pytest.approx(30.0)


def test_report_readiness_reaches_the_server(server):
    client = _client(server)
    client.report_readiness(True, timeout=5.0)
    assert server.reported == [True]


def test_close_is_a_no_op_for_injected_channel(server):
    client = _client(server)
    client.close()  # must not raise
    client.close()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_turn_indicators_are_forwarded_when_they_change(server):
    forwarder = TurnIndicatorForwarder(_client(server), 5.0, clock=_Clock())
    # Autoware publishes every control cycle; only the changes go over.
    reported = [forwarder.observe(command) for command in (0, 2, 2, 2, 0, 1, 1, 3)]
    assert server.turn_indicators == [2, 1, 3]
    assert reported == [None, True, None, None, None, True, None, True]


def test_the_command_in_force_is_sent_again_now_and_then(server):
    """So a scenario that started over (a retried attempt) learns it."""
    clock = _Clock()
    forwarder = TurnIndicatorForwarder(_client(server), 5.0, resend_s=1.0, clock=clock)
    forwarder.observe(2)
    clock.now = 0.5
    assert forwarder.observe(2) is None
    clock.now = 1.0
    assert forwarder.observe(2) is False  # reported again, unchanged
    assert server.turn_indicators == [2, 2]


def test_a_failed_report_is_sent_again(server):
    channel = grpc.insecure_channel("localhost:1")  # nobody there
    forwarder = TurnIndicatorForwarder(
        ScenarioBridgeClient(channel=channel), 0.2, clock=_Clock()
    )
    with pytest.raises(grpc.RpcError):
        forwarder.observe(2)
    # Not recorded as sent: a working client gets it with the next message.
    forwarder._client = _client(server)  # noqa: SLF001
    assert forwarder.observe(2) is True
    assert server.turn_indicators == [2]


def test_a_message_during_a_report_in_flight_is_skipped(server):
    forwarder = TurnIndicatorForwarder(_client(server), 5.0, clock=_Clock())
    forwarder._lock.acquire()  # noqa: SLF001 - a report in flight
    try:
        assert forwarder.observe(2) is None
    finally:
        forwarder._lock.release()  # noqa: SLF001
    assert server.turn_indicators == []


def test_the_mission_carries_the_framework_servers_waypoints_and_version():
    """The scenario's waypoints reach the node's Mission, in order (#872)."""
    from autoware_carla_scenario.autoware_bridge import (
        AutowareBridgeConfig,
        BridgePose,
        GrpcAutowareBridgeServer,
    )

    bridge = GrpcAutowareBridgeServer(AutowareBridgeConfig(address="localhost:0"))
    client = ScenarioBridgeClient(
        channel=grpc.insecure_channel(f"localhost:{bridge.port}")
    )
    try:
        bridge.configure(
            BridgePose.from_yaw(1.0, 2.0, 0.0, 0.0),
            BridgePose.from_yaw(9.0, 0.0, 0.0, 0.0),
            [BridgePose.from_yaw(x, 0.0, 0.0, 0.0) for x in (3.0, 5.0, 7.0)],
        )
        mission = client.get_mission(timeout=5.0)
        assert mission is not None
        assert [p.position.x for p in mission.waypoints] == [3.0, 5.0, 7.0]
        assert mission.goal.position.x == pytest.approx(9.0)

        bridge.reroute(BridgePose.from_yaw(20.0, 0.0, 0.0, 0.0))
        rerouted = client.get_mission(timeout=5.0)
        assert rerouted is not None
        assert rerouted.version > mission.version
        assert rerouted.waypoints == ()
        assert rerouted.goal.position.x == pytest.approx(20.0)
        assert rerouted.initial_pose == mission.initial_pose
    finally:
        bridge.close()


def test_a_server_without_mission_versions_reads_as_version_zero(server):
    server.mission = (_pose(1.0, 2.0, 3.0), _pose(10.0, 20.0, 30.0))
    mission = _client(server).get_mission(timeout=5.0)
    assert mission is not None
    assert mission.version == 0
    assert mission.waypoints == ()
