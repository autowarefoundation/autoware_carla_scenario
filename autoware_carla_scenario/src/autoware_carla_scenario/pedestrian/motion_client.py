"""The client of a pedestrian motion generator (the ``PedestrianMotion`` contract).

``proto/pedestrian_motion/v0/pedestrian_motion.proto`` is the contract: the
generator places the joints of a body frame by frame, in its CONTRACT FRAME
(right-handed, z up, metres: CARLA's world with its y negated).
:class:`PedestrianMotionClient` speaks it, and :func:`to_contract` /
:func:`to_carla` move a ground point and a heading between that frame and
CARLA's.  Those two and the contract's values (:class:`TextSpan`,
:class:`RootWaypoint`, :class:`GeneratedFrame`, ...) are defined in
:mod:`~autoware_carla_scenario.pedestrian.contract`, which needs no grpc,
and re-exported here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import grpc

from ._proto import pedestrian_motion_pb2 as pb
from ._proto import pedestrian_motion_pb2_grpc as pb_grpc
from .contract import (
    GeneratedFrame,
    MotionGeneratorInfo,
    PoseKeyframe,
    RootWaypoint,
    TextSpan,
    to_carla,
    to_contract,
)

__all__ = [
    "CONTRACT_VERSION",
    "GeneratedFrame",
    "MotionGeneratorInfo",
    "PedestrianMotionClient",
    "PoseKeyframe",
    "RootWaypoint",
    "TextSpan",
    "to_carla",
    "to_contract",
]

#: The contract version this client speaks (the proto's header).
CONTRACT_VERSION = "0.1.0"


class PedestrianMotionClient:
    """Speaks ``pedestrian_motion.v0.PedestrianMotion`` to one generator.

    Args:
        address: ``host:port`` of the generator.
        timeout_s: How long one call may take.  A Step generates the next
            couple of seconds of motion on the GPU; the simulation waits for
            it.
        channel: A channel to use instead of dialling *address* (tests).
    """

    def __init__(
        self,
        address: str,
        timeout_s: float = 60.0,
        channel: Optional[grpc.Channel] = None,
    ) -> None:
        self._address = address
        self._timeout_s = timeout_s
        self._owns_channel = channel is None
        self._channel = (
            channel if channel is not None else grpc.insecure_channel(address)
        )
        self._stub = pb_grpc.PedestrianMotionStub(self._channel)
        self._joint_names: tuple[str, ...] = ()

    def negotiate(self) -> MotionGeneratorInfo:
        """Agree on the contract; what the generator produces.

        Raises:
            RuntimeError: If the generator speaks another contract version,
                or answers a frame rate that is not finite and positive.
            grpc.RpcError: If it does not answer.
        """
        reply = self._stub.Negotiate(
            pb.NegotiateRequest(contract_version=CONTRACT_VERSION),
            timeout=self._timeout_s,
        )
        if reply.contract_version != CONTRACT_VERSION:
            raise RuntimeError(
                f"the pedestrian motion generator at {self._address} speaks contract "
                f"{reply.contract_version or '(none)'}, not {CONTRACT_VERSION}: use "
                "a backend image of the contract this autoware_carla_scenario speaks"
            )
        fps = float(reply.fps)
        # Every frame number is a time times the rate: an unset (0), NaN or
        # infinite rate would place every frame at 0 or fail int().
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(
                f"the pedestrian motion generator at {self._address} answers a "
                f"frame rate of {fps!r}, not a finite positive one"
            )
        self._joint_names = tuple(reply.joint_names)
        return MotionGeneratorInfo(
            backend=reply.backend,
            backend_version=reply.backend_version,
            model=reply.model,
            fps=fps,
            horizon_frames=int(reply.horizon_frames),
            joint_names=self._joint_names,
        )

    def open_session(
        self,
        session_id: str,
        root: tuple[float, float],
        heading: float,
        *,
        seed: int = 0,
        texts: Sequence[TextSpan] = (),
        waypoints: Sequence[RootWaypoint] = (),
        keyframes: Sequence[PoseKeyframe] = (),
    ) -> None:
        """Open (or reopen) a session at a pose, with its conditions."""
        request = pb.OpenSessionRequest(
            session_id=session_id,
            root=pb.Vec3(x=root[0], y=root[1], z=0.0),
            heading=heading,
            seed=seed,
        )
        for span in texts:
            request.texts.add(start_frame=span.start_frame, text=span.text)
        for point in waypoints:
            request.waypoints.add(
                frame=point.frame,
                x=point.x,
                y=point.y,
                has_heading=point.heading is not None,
                heading=0.0 if point.heading is None else point.heading,
            )
        for key in keyframes:
            message = request.keyframes.add(frame=key.frame)
            for joint in key.joints:
                message.joints.add(x=joint[0], y=joint[1], z=joint[2])
        self._stub.OpenSession(request, timeout=self._timeout_s)

    def step(self, session_id: str, keep_until_frame: int) -> list[GeneratedFrame]:
        """The motion after *keep_until_frame* (-1: from the start)."""
        reply = self._stub.Step(
            pb.StepRequest(session_id=session_id, keep_until_frame=keep_until_frame),
            timeout=self._timeout_s,
        )
        names = self._joint_names
        return [
            GeneratedFrame(
                frame=int(f.frame),
                root=(f.root.x, f.root.y, f.root.z),
                heading=float(f.heading),
                joints={name: (j.x, j.y, j.z) for name, j in zip(names, f.joints)},
            )
            for f in reply.frames
        ]

    def close_session(self, session_id: str) -> None:
        """Forget a session; an unreachable generator is not an error."""
        try:
            self._stub.CloseSession(
                pb.CloseSessionRequest(session_id=session_id), timeout=5.0
            )
        except grpc.RpcError:
            pass

    def close(self) -> None:
        """Close the channel, if this client opened it."""
        if self._owns_channel:
            self._channel.close()
