"""gRPC client for the alpasim ``egodriver.EgodriverService`` contract.

This is the *runtime* half of the conversation: the scenario framework renders
observations from CARLA and asks a driver policy -- running as a separate gRPC server,
for example ``carla-driver-interface serve`` -- what to do next.

The generated stubs come from the alpasim protos vendored in ``carla-driver-interface``,
so the messages are wire compatible with an upstream alpasim driver as well; see
``carla_driver_interface/proto/README.md``.
"""

from __future__ import annotations

import atexit
import functools
import logging
from typing import Optional, Sequence

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
from carla_driver_interface.geometry import body_to_optical
from carla_driver_interface.policies import load_policy
from carla_driver_interface.server import build_server
from carla_driver_interface.protocol import (
    EGODRIVER_SERVICE_FULL_NAME,
    MAX_MESSAGE_BYTES,
    carla_driver_pb2,
    channel_options,
    common_pb2,
    egodriver_pb2,
    egodriver_pb2_grpc,
    sensorsim_pb2,
)

from .base import BaseEgoDriverClient, DriveOutcome, DriverClientConfig, EgoObservation
from .geometry import Trajectory, waypoints_to_proto
from .observation import camera_extrinsics_to_rig


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
    """

    def __init__(
        self,
        config: DriverClientConfig,
        channel: Optional[grpc.Channel] = None,
    ) -> None:
        super().__init__(config)
        self._owns_channel = channel is None
        self._channel: Optional[grpc.Channel] = channel
        self._stub: Optional[egodriver_pb2_grpc.EgodriverServiceStub] = None
        self._session_uuid: Optional[str] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def session_uuid(self) -> Optional[str]:
        """Return the active session id, or ``None`` before :meth:`start_session`."""
        return self._session_uuid

    def _connect(self) -> egodriver_pb2_grpc.EgodriverServiceStub:
        """Return the stub, dialling the configured address on first use."""
        if self._stub is not None:
            return self._stub
        if self._channel is None:
            target = (
                self._config.address
                if self._config.policy is None
                else f"localhost:{_serve_in_process(self._config.policy)}"
            )
            self._channel = grpc.insecure_channel(target, options=channel_options())
        self._stub = egodriver_pb2_grpc.EgodriverServiceStub(self._channel)
        return self._stub

    def _camera_specs(self) -> list:
        """Return the ``AvailableCamera`` entries describing the configured rig.

        As contract revision 2 declares them: ``rig_to_camera`` is the camera's
        optical frame in the rig (x right, y down, z along the optical axis).
        """
        cameras = []
        for camera in self._config.cameras:
            sensor = camera.to_sensor_config()
            spec = sensorsim_pb2.CameraSpec(
                opencv_pinhole_param=sensorsim_pb2.OpenCVPinholeCameraParam(
                    principal_point_x=sensor.cx,
                    principal_point_y=sensor.cy,
                    focal_length_x=sensor.fx,
                    focal_length_y=sensor.fy,
                ),
                logical_id=camera.logical_id,
                resolution_h=camera.image_height,
                resolution_w=camera.image_width,
            )
            cameras.append(
                sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                    intrinsics=spec,
                    # Without this the policy sees the protobuf-default (identity,
                    # origin) pose and reads every frame from the wrong viewpoint,
                    # not even the configured (1.5, 0, 1.6) mount.
                    rig_to_camera=body_to_optical(
                        camera_extrinsics_to_rig(
                            camera.position_x,
                            camera.position_y,
                            camera.position_z,
                            camera.roll,
                            camera.pitch,
                            camera.yaw,
                        )
                    ).to_proto(),
                    logical_id=camera.logical_id,
                )
            )
        return cameras

    # ------------------------------------------------------------------
    # BaseEgoDriverClient interface
    # ------------------------------------------------------------------

    def start_session(self, session_uuid: str, scene_id: str) -> None:
        """Open a rollout session with the policy.

        Also calls ``get_version`` first so that an unreachable or mismatched policy
        fails immediately with a clear message rather than midway through the scenario.
        Its answer says which contract revision the policy speaks
        (``carla_driver_interface.contract``); the cameras are declared in that
        revision, so a policy on carla-driver-interface 1.x still reads them right.

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
            self._config.address,
            version.version_id,
            version.git_hash,
            revision,
        )

        request = egodriver_pb2.DriveSessionRequest(
            session_uuid=session_uuid,
            random_seed=self._config.random_seed,
            debug_info=egodriver_pb2.DriveSessionRequest.DebugInfo(scene_id=scene_id),
            rollout_spec=egodriver_pb2.DriveSessionRequest.RolloutSpec(
                vehicle=egodriver_pb2.DriveSessionRequest.RolloutSpec.VehicleDefinition(
                    available_cameras=cameras_to_revision(
                        self._camera_specs(), CONTRACT_REVISION, revision
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
            The plan in the **local** frame, plus the policy's termination flag
            and whatever diagnostics it chose to surface.
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
        return outcome

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


@functools.lru_cache(maxsize=None)
def _serve_in_process(spec: str) -> int:
    """Serve the *spec* policy from this process and return its port.

    Once per run: every scenario's session goes to the same policy, which the
    protocol resets between sessions, so a model is loaded once, not per scenario.
    """
    server, port = build_server(load_policy(spec), port=0, host="localhost")
    server.start()
    atexit.register(server.stop, None)
    logger.info("Serving policy %r in this process on port %d", spec, port)
    return port
