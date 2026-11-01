"""The client of a pedestrian motion generator (the ``PedestrianMotion`` contract).

``proto/pedestrian_motion/v0/pedestrian_motion.proto`` is the contract: the
generator places the joints of a body frame by frame, in its CONTRACT FRAME
(right-handed, z up, metres: CARLA's world with its y negated).
:class:`PedestrianMotionClient` speaks it, and :func:`to_contract` /
:func:`to_carla` move a ground point and a heading between that frame and
CARLA's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import grpc

from ._proto import pedestrian_motion_pb2 as pb
from ._proto import pedestrian_motion_pb2_grpc as pb_grpc
from .skeleton import Vector

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


def to_contract(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    """A CARLA ground point and yaw as the contract's ``(x, y, heading)``."""
    return x, -y, -math.radians(yaw_deg)


def to_carla(x: float, y: float, heading: float) -> tuple[float, float, float]:
    """A contract ground point and heading as CARLA's ``(x, y, yaw_deg)``."""
    return x, -y, -math.degrees(heading)


@dataclass(frozen=True)
class MotionGeneratorInfo:
    """What a generator answered :meth:`PedestrianMotionClient.negotiate`."""

    backend: str
    backend_version: str
    model: str
    fps: float
    horizon_frames: int
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class TextSpan:
    """The text the motion follows from *start_frame* on."""

    start_frame: int
    text: str


@dataclass(frozen=True)
class RootWaypoint:
    """Where the root is at *frame* (contract frame), and how it faces."""

    frame: int
    x: float
    y: float
    heading: Optional[float] = None


@dataclass(frozen=True)
class PoseKeyframe:
    """The whole body at *frame*: each joint of the generator's
    :attr:`MotionGeneratorInfo.joint_names`, in order (contract frame)."""

    frame: int
    joints: tuple[Vector, ...]


@dataclass(frozen=True)
class GeneratedFrame:
    """One frame of a generated motion (contract frame).

    Attributes:
        frame: Its number from the session's start.
        root: The root's ground point.
        heading: The way the body faces.
        joints: Each joint's position, by name.
    """

    frame: int
    root: Vector
    heading: float
    joints: dict[str, Vector]


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
            RuntimeError: If the generator speaks another contract version.
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
        self._joint_names = tuple(reply.joint_names)
        return MotionGeneratorInfo(
            backend=reply.backend,
            backend_version=reply.backend_version,
            model=reply.model,
            fps=float(reply.fps),
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
