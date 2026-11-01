"""gRPC client for the alpasim ``egodriver.EgodriverService`` contract.

This is the *runtime* half of the conversation: the scenario framework renders
observations from CARLA and asks a driver policy -- running as a separate gRPC server,
for example ``carla-driver-interface serve``, or served by the run itself
(``driver.policy``) -- what to do next. ``driver.transport`` says how it is reached:
over TCP, over a Unix domain socket, or (a policy the run serves) by calling its
servicer directly, with the same messages and bytes as on the wire.

The generated stubs come from the alpasim protos vendored in ``carla-driver-interface``,
so the messages are wire compatible with an upstream alpasim driver as well; see
``carla_driver_interface/proto/README.md``.
"""

from __future__ import annotations

import atexit
import functools
import importlib
import logging
import os
import shutil
import tempfile
from typing import Any, Optional, Sequence

import grpc
import numpy as np
from google.protobuf.message import DecodeError
from numpy.typing import NDArray

from carla_driver_interface.contract import (
    CONTRACT_REVISION,
    cameras_to_revision,
    contract_metadata,
    negotiate,
)
import carla_driver_interface
from carla_driver_interface import server as _cdi_server
from carla_driver_interface.driver import BaseDriver
from carla_driver_interface.geometry import body_to_optical
from carla_driver_interface.policies import load_policy
from carla_driver_interface.server import build_server
from carla_driver_interface.service import EgodriverServicer
from carla_driver_interface.protocol import (
    EGODRIVER_SERVICE_FULL_NAME,
    MAX_MESSAGE_BYTES,
    add_CarlaVehicleServiceServicer_to_server,
    add_EgodriverServiceServicer_to_server,
    carla_driver_pb2,
    carla_driver_pb2_grpc,
    channel_options,
    common_pb2,
    egodriver_pb2,
    egodriver_pb2_grpc,
    sensorsim_pb2,
)

from .base import (
    BaseEgoDriverClient,
    DriveOutcome,
    DriverCameraLens,
    DriverClientConfig,
    EgoObservation,
)
from ..seed import resolve_seed
from .geometry import Trajectory, waypoints_to_proto
from .observation import sensor_pose_in_rig


logger = logging.getLogger(__name__)

# The wire constants live with the contract; re-exported for existing callers.
__all__ = [
    "EGODRIVER_SERVICE_FULL_NAME",
    "MAX_MESSAGE_BYTES",
    "EgoDriverGrpcClient",
    "channel_options",
]


class EgoDriverGrpcClient(BaseEgoDriverClient):
    """Talks to a driver policy over ``egodriver.EgodriverService``.

    Args:
        config: Connection and cadence settings.
        channel: Pre-built channel to use instead of dialling
            :attr:`DriverClientConfig.address`.  Intended for tests, which run the
            policy in-process.

    Raises:
        RuntimeError: ``config.transport`` serves ``config.policy`` in a way the
            installed carla-driver-interface cannot (``unix``, ``direct``: 2.2+).
    """

    def __init__(
        self,
        config: DriverClientConfig,
        channel: Optional[grpc.Channel] = None,
    ) -> None:
        super().__init__(config)
        if channel is None and config.policy is not None and config.transport != "tcp":
            # Now, while the run sets up, rather than at its first scenario.
            _require_in_process_support(config.transport)
        self._owns_channel = channel is None
        self._channel: Optional[grpc.Channel] = channel
        self._stub: Optional[egodriver_pb2_grpc.EgodriverServiceStub] = None
        self._vehicle: Optional[carla_driver_pb2_grpc.CarlaVehicleServiceStub] = None
        #: Whether the policy serves ``CarlaVehicleService``; ``False`` once it
        #: answered UNIMPLEMENTED (carla-driver-interface < 2.1), so it is not
        #: asked again.
        self._serves_vehicle_commands = True
        self._session_uuid: Optional[str] = None
        #: Where the policy is, for the log: an address, or the in-process servicer.
        self._target = "an injected channel" if channel is not None else config.address

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def session_uuid(self) -> Optional[str]:
        """Return the active session id, or ``None`` before :meth:`start_session`."""
        return self._session_uuid

    def _connect(self) -> egodriver_pb2_grpc.EgodriverServiceStub:
        """Return the stub, opening the configured transport on first use."""
        if self._stub is not None:
            return self._stub
        if self._channel is None:
            self._channel, self._target = _open_channel(self._config)
        self._stub = egodriver_pb2_grpc.EgodriverServiceStub(self._channel)
        self._vehicle = carla_driver_pb2_grpc.CarlaVehicleServiceStub(self._channel)
        return self._stub

    def _camera_specs(self, rear_axle_offset_m: float = 0.0) -> list:
        """Return the ``AvailableCamera`` entries describing the configured rig.

        As contract revision 2 declares them: ``rig_to_camera`` is the camera's
        optical frame in the rig (x right, y down, z along the optical axis).

        Args:
            rear_axle_offset_m: Where the rig origin (the rear axle) is from the
                actor origin the cameras are mounted relative to; the mount is
                moved by it, as the LiDAR's and the T4 calibration's are.
        """
        cameras = []
        for camera in self._config.cameras:
            spec = sensorsim_pb2.CameraSpec(
                logical_id=camera.logical_id,
                resolution_h=camera.image_height,
                resolution_w=camera.image_width,
            )
            if camera.lens is None:
                sensor = camera.to_sensor_config()
                spec.opencv_pinhole_param.CopyFrom(
                    sensorsim_pb2.OpenCVPinholeCameraParam(
                        principal_point_x=sensor.cx,
                        principal_point_y=sensor.cy,
                        focal_length_x=sensor.fx,
                        focal_length_y=sensor.fy,
                    )
                )
            else:
                spec.ftheta_param.CopyFrom(_ftheta_param(camera.lens))
            cameras.append(
                sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                    intrinsics=spec,
                    # Without this the policy sees the protobuf-default (identity,
                    # origin) pose and reads every frame from the wrong viewpoint,
                    # not even the configured (1.5, 0, 1.6) mount.  The mount is
                    # relative to the actor origin, where CARLA attaches the
                    # camera, so it moves to the rig origin (the rear axle).
                    rig_to_camera=body_to_optical(
                        sensor_pose_in_rig(
                            camera.position_x,
                            camera.position_y,
                            camera.position_z,
                            camera.roll,
                            camera.pitch,
                            camera.yaw,
                            rear_axle_offset_m,
                        )
                    ).to_proto(),
                    logical_id=camera.logical_id,
                )
            )
        return cameras

    # ------------------------------------------------------------------
    # BaseEgoDriverClient interface
    # ------------------------------------------------------------------

    def start_session(
        self, session_uuid: str, scene_id: str, *, rear_axle_offset_m: float = 0.0
    ) -> None:
        """Open a rollout session with the policy.

        Also calls ``get_version`` first so that an unreachable or mismatched policy
        fails immediately with a clear message rather than midway through the scenario.
        Its answer says which contract revision the policy speaks
        (``carla_driver_interface.contract``); the cameras are declared in that
        revision, so a policy on carla-driver-interface 1.x still reads them right.

        Args:
            session_uuid: Identifier correlating every subsequent call.
            scene_id: Human-readable scene name for the policy's debug output.
            rear_axle_offset_m: Where the rig origin (the rear axle) is from the
                actor origin, along the actor's x axis (negative behind); the
                cameras' ``rig_to_camera`` is moved by it.

        Raises:
            grpc.RpcError: If the policy cannot be reached.
            carla_driver_interface.contract.ContractError: If the policy speaks a
                revision this release cannot translate.
        """
        stub = self._connect()

        version, revision = negotiate(stub, timeout=self._config.timeout_s)
        logger.info(
            "Connected to %s at %s (version_id=%r git_hash=%r contract revision %d)",
            EGODRIVER_SERVICE_FULL_NAME,
            self._target,
            version.version_id,
            version.git_hash,
            revision,
        )

        request = egodriver_pb2.DriveSessionRequest(
            session_uuid=session_uuid,
            random_seed=resolve_seed(self._config.random_seed),
            debug_info=egodriver_pb2.DriveSessionRequest.DebugInfo(scene_id=scene_id),
            rollout_spec=egodriver_pb2.DriveSessionRequest.RolloutSpec(
                vehicle=egodriver_pb2.DriveSessionRequest.RolloutSpec.VehicleDefinition(
                    available_cameras=cameras_to_revision(
                        self._camera_specs(rear_axle_offset_m),
                        CONTRACT_REVISION,
                        revision,
                    )
                )
            ),
        )
        stub.start_session(
            request,
            timeout=self._config.timeout_s,
            metadata=contract_metadata(revision),
        )
        self._session_uuid = session_uuid
        logger.info("Driver session started: uuid=%s scene=%s", session_uuid, scene_id)

    def submit_route(
        self, timestamp_us: int, waypoints_in_rig: NDArray[np.float64]
    ) -> None:
        """Send the route the ego should follow, in the rig frame."""
        stub = self._require_session()
        route = egodriver_pb2.Route(
            timestamp_us=timestamp_us,
            waypoints=waypoints_to_proto(np.asarray(waypoints_in_rig).reshape(-1, 3)),
        )
        stub.submit_route(
            egodriver_pb2.RouteRequest(session_uuid=self._session_uuid, route=route),
            timeout=self._config.timeout_s,
        )

    def submit_image_observation(
        self,
        logical_id: str,
        frame_start_us: int,
        frame_end_us: int,
        image_bytes: bytes,
    ) -> None:
        """Send one encoded camera frame."""
        stub = self._require_session()
        stub.submit_image_observation(
            egodriver_pb2.RolloutCameraImage(
                session_uuid=self._session_uuid,
                camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                    frame_start_us=frame_start_us,
                    frame_end_us=frame_end_us,
                    image_bytes=image_bytes,
                    logical_id=logical_id,
                ),
            ),
            timeout=self._config.timeout_s,
        )

    def submit_egomotion_observation(
        self, observations: Sequence[EgoObservation]
    ) -> None:
        """Send the ego's estimated poses and dynamic states since the last step."""
        if not observations:
            return
        stub = self._require_session()
        trajectory = Trajectory(
            [observation.timestamp_us for observation in observations],
            [observation.pose for observation in observations],
        ).to_proto()
        states = [
            common_pb2.DynamicState(
                linear_velocity=_vec3(observation.linear_velocity),
                angular_velocity=_vec3(observation.angular_velocity),
                linear_acceleration=_vec3(observation.linear_acceleration),
            )
            for observation in observations
        ]
        stub.submit_egomotion_observation(
            egodriver_pb2.RolloutEgoTrajectory(
                session_uuid=self._session_uuid,
                trajectory=trajectory,
                dynamic_states=states,
            ),
            timeout=self._config.timeout_s,
        )

    def submit_recording_ground_truth(
        self, timestamp_us: int, trajectory_in_rig: Trajectory
    ) -> None:
        """Send the reference trajectory, in the rig frame."""
        stub = self._require_session()
        stub.submit_recording_ground_truth(
            egodriver_pb2.GroundTruthRequest(
                session_uuid=self._session_uuid,
                ground_truth=egodriver_pb2.GroundTruth(
                    timestamp_us=timestamp_us,
                    trajectory=trajectory_in_rig.to_proto(),
                ),
            ),
            timeout=self._config.timeout_s,
        )

    def drive(
        self,
        time_now_us: int,
        time_query_us: int,
        renderer_data: bytes = b"",
    ) -> DriveOutcome:
        """Ask the policy for a plan.

        Returns:
            The plan in the **local** frame, plus the policy's termination flag,
            its vehicle command (``CarlaVehicleService.get_vehicle_command``,
            asked right after) and whatever diagnostics it chose to surface.
        """
        stub = self._require_session()
        response = stub.drive(
            egodriver_pb2.DriveRequest(
                session_uuid=self._session_uuid,
                time_now_us=time_now_us,
                time_query_us=time_query_us,
                renderer_data=renderer_data,
            ),
            timeout=self._config.timeout_s,
        )
        raw_debug = bytes(response.debug_info.unstructured_debug_info)
        outcome = DriveOutcome(
            trajectory=Trajectory.from_proto(response.trajectory),
            terminate_session=bool(response.terminate_session),
            debug_info=raw_debug,
        )
        _decode_debug_info(raw_debug, outcome)
        outcome.turn_indicator = self._turn_indicator()
        return outcome

    def _turn_indicator(self) -> int:
        """The turn indicator the policy decided with its latest drive.

        0 (no command) from a policy that does not serve ``CarlaVehicleService``.
        """
        if not self._serves_vehicle_commands or self._vehicle is None:
            return 0
        try:
            command = self._vehicle.get_vehicle_command(
                carla_driver_pb2.VehicleCommandRequest(session_uuid=self._session_uuid),
                timeout=self._config.timeout_s,
            )
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.UNIMPLEMENTED:
                # The plan already arrived: an indicator is not worth losing it.
                logger.warning(
                    "get_vehicle_command failed (%s); no turn indicator this step",
                    error.code(),
                )
                return 0
            logger.info(
                "The driver policy serves no CarlaVehicleService "
                "(carla-driver-interface < 2.1); it commands no turn indicators"
            )
            self._serves_vehicle_commands = False
            return 0
        return int(command.turn_indicator)

    def close_session(self) -> None:
        """Close the session and the channel.  Safe to call more than once."""
        if self._stub is not None and self._session_uuid is not None:
            try:
                self._stub.close_session(
                    egodriver_pb2.DriveSessionCloseRequest(
                        session_uuid=self._session_uuid
                    ),
                    timeout=self._config.timeout_s,
                )
            except grpc.RpcError:
                logger.warning("close_session RPC failed", exc_info=True)
        self._session_uuid = None
        self._stub = None
        self._vehicle = None
        self._serves_vehicle_commands = True
        if self._channel is not None and self._owns_channel:
            self._channel.close()
        self._channel = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_session(self) -> egodriver_pb2_grpc.EgodriverServiceStub:
        """Return the stub, asserting a session is open.

        Raises:
            RuntimeError: If :meth:`start_session` has not been called.
        """
        if self._stub is None or self._session_uuid is None:
            raise RuntimeError("No driver session is open. Call start_session() first.")
        return self._stub


def _decode_debug_info(raw: bytes, outcome: DriveOutcome) -> None:
    """Fill *outcome*'s diagnostics from a ``CarlaDriveDebugInfo`` payload.

    The field is deliberately unstructured in the alpasim contract, so anything
    that fails to parse is left alone rather than treated as an error -- a policy
    is free to put its own encoding there.
    """
    if not raw:
        return
    message = carla_driver_pb2.CarlaDriveDebugInfo()
    try:
        message.ParseFromString(raw)
    except DecodeError:
        logger.debug("Driver debug info is not a CarlaDriveDebugInfo", exc_info=True)
        return
    outcome.policy_name = str(message.policy_name)
    outcome.inference_seconds = float(message.inference_seconds)
    outcome.debug_scalars = dict(message.scalars)


def _vec3(values: NDArray[np.float64]) -> common_pb2.Vec3:
    """Return *values* as a protobuf ``Vec3``."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return common_pb2.Vec3(x=array[0], y=array[1], z=array[2])


#: The carla-driver-interface release the in-process transports first ship in.
_IN_PROCESS_RELEASE = "2.2"


def _open_channel(config: DriverClientConfig) -> tuple[grpc.Channel, str]:
    """A channel to the configured policy over ``config.transport``, and where it is.

    Raises:
        RuntimeError: The transport needs a newer carla-driver-interface.
    """
    if config.policy is None:
        # A policy in a process of its own, at a TCP or a unix: address.
        return (
            grpc.insecure_channel(config.address, options=channel_options()),
            config.address,
        )
    if config.transport == "direct":
        return _in_process_channel(config.policy), f"policy {config.policy!r} (direct)"
    if config.transport == "unix":
        target = _serve_on_unix_socket(config.policy)
    else:
        target = f"localhost:{_serve_in_process(config.policy)}"
    return grpc.insecure_channel(target, options=channel_options()), target


def _require_in_process_support(transport: str) -> None:
    """Refuse *transport* on a carla-driver-interface that cannot serve it, saying why."""
    try:
        importlib.import_module("carla_driver_interface.inprocess")
    except ImportError:
        supported = False
    else:
        supported = hasattr(_cdi_server, "bind_address")
    if not supported:
        raise RuntimeError(
            f"driver.transport={transport} with driver.policy needs "
            f"carla-driver-interface>={_IN_PROCESS_RELEASE}, which serves a policy "
            "on a Unix socket and in-process; this environment has "
            f"{carla_driver_interface.__version__}. Upgrade it, or use "
            "driver.transport=tcp."
        )


@functools.lru_cache(maxsize=None)
def _policy(spec: str) -> BaseDriver:
    """The *spec* policy, loaded once per run whatever transport reaches it.

    Every scenario's session goes to the same policy, which the protocol resets
    between sessions, so a model is loaded once, not per scenario.
    """
    return load_policy(spec)


@functools.lru_cache(maxsize=None)
def _serve_in_process(spec: str) -> int:
    """Serve the *spec* policy from this process and return its port. Once per run."""
    server, port = build_server(_policy(spec), port=0, host="localhost")
    server.start()
    atexit.register(server.stop, None)
    logger.info("Serving policy %r in this process on port %d", spec, port)
    return port


#: The longest path a Unix socket address may carry (``sun_path``, less its NUL).
_MAX_SOCKET_PATH = 107


@functools.lru_cache(maxsize=None)
def _serve_on_unix_socket(spec: str) -> str:
    """Serve the *spec* policy from this process on a Unix socket; its address. Once per run.

    In a directory of its own (``0700``, under ``$TMPDIR``), so runs sharing a node
    -- SLURM tasks, say -- never share a socket, and nothing else can dial it.
    """
    _require_in_process_support("unix")
    directory = tempfile.mkdtemp(prefix="acs-policy-")
    if len(os.path.join(directory, "policy.sock")) > _MAX_SOCKET_PATH:
        shutil.rmtree(directory, ignore_errors=True)
        directory = tempfile.mkdtemp(prefix="acs-policy-", dir="/tmp")
    address = f"unix://{os.path.join(directory, 'policy.sock')}"
    server, _ = build_server(_policy(spec), host=address)
    server.start()
    # Stopped before the directory goes: atexit runs in reverse order.
    atexit.register(shutil.rmtree, directory, True)
    atexit.register(server.stop, None)
    logger.info("Serving policy %r in this process on %s", spec, address)
    return address


@functools.lru_cache(maxsize=None)
def _in_process_servicer(spec: str) -> EgodriverServicer:
    """The servicer a server for *spec* would run, for the direct transport. Once per run."""
    _require_in_process_support("direct")
    logger.info("Serving policy %r in this process, called directly", spec)
    return EgodriverServicer(_policy(spec))


def _in_process_channel(spec: str) -> grpc.Channel:
    """A fresh channel to *spec*'s one servicer, carrying both services a server would.

    One per client: closing it at the end of a scenario leaves the servicer, and the
    policy, to the next.
    """
    from carla_driver_interface.inprocess import InProcessChannel  # noqa: PLC0415

    servicer = _in_process_servicer(spec)
    channel = InProcessChannel(channel_options())
    add_EgodriverServiceServicer_to_server(servicer, channel)
    add_CarlaVehicleServiceServicer_to_server(servicer, channel)
    return channel


def _ftheta_param(lens: DriverCameraLens) -> Any:
    """alpasim's ``FthetaCameraParam`` of *lens*: the camera's frames are drawn
    through it, not through the pinhole of its ``fov``."""
    c, d, e = lens.linear_cde
    return sensorsim_pb2.FthetaCameraParam(
        principal_point_x=lens.principal_point[0],
        principal_point_y=lens.principal_point[1],
        reference_poly=(
            sensorsim_pb2.FthetaCameraParam.PolynomialType.PIXELDIST_TO_ANGLE
            if lens.reference_poly == "pixeldist_to_angle"
            else sensorsim_pb2.FthetaCameraParam.PolynomialType.ANGLE_TO_PIXELDIST
        ),
        pixeldist_to_angle_poly=list(lens.pixeldist_to_angle_poly),
        angle_to_pixeldist_poly=list(lens.angle_to_pixeldist_poly),
        max_angle=lens.max_angle,
        linear_cde=sensorsim_pb2.LinearCde(linear_c=c, linear_d=d, linear_e=e),
    )
