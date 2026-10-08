"""Policy-facing types: what a driving policy implements and what it is given.

:class:`BaseDriver` is what a policy implements. Everything gRPC -- session
bookkeeping, frame retention, ego history, rig/local conversion -- lives in
:mod:`autoware_carla_egodriver.service`, so a policy only sees numpy and plain Python.

The observation set mirrors alpasim's ``EgodriverService`` one for one: images,
egomotion, route and recorded ground truth arrive through separate calls before every
:meth:`BaseDriver.drive`, as the alpasim runtime sequences them.
"""

from __future__ import annotations

import io
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .geometry import Pose, Trajectory
from .protocol import AvailableCamera, CarlaRendererData, DynamicState

__all__ = ["BaseDriver", "CameraFrame", "DriveContext", "DriveResult", "SessionState"]


@dataclass(frozen=True)
class CameraFrame:
    """One encoded camera image, as it arrived over the wire.

    ``image_bytes`` stays encoded (PNG or JPEG): a policy that only needs the front
    camera should not pay to decode the others. :meth:`as_array` decodes.
    """

    logical_id: str
    frame_start_us: int
    frame_end_us: int
    image_bytes: bytes

    def as_array(self) -> NDArray[np.uint8]:
        """Decode to an ``(H, W, 3)`` uint8 RGB array."""
        from PIL import Image

        with Image.open(io.BytesIO(self.image_bytes)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)


@dataclass
class SessionState:
    """Per-rollout state the servicer maintains on the policy's behalf."""

    uuid: str
    seed: int
    scene_id: str
    cameras: Dict[str, AvailableCamera]

    #: Bounded history per camera, oldest first; ``[]`` for every declared camera
    #: before its first frame. :meth:`latest_frame` gives the newest.
    frame_history: Dict[str, List[CameraFrame]] = field(default_factory=dict)
    #: Estimated ego trajectory in the ``local`` frame (``local -> rig_est``).
    ego_trajectory: Trajectory = field(default_factory=Trajectory.empty)
    #: Dynamic states in the rig frame, aligned 1:1 with ``ego_trajectory``.
    dynamic_states: List[DynamicState] = field(default_factory=list)
    #: Latest route waypoints in the rig frame, ``(N, 3)``; empty when unset.
    route_waypoints_in_rig: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    route_timestamp_us: int = 0
    #: Latest recorded ground truth in the rig frame, when the runtime sends one.
    ground_truth_in_rig: Optional[Trajectory] = None
    drive_count: int = 0

    #: Guards mutation from concurrent gRPC handler threads.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def latest_frame(self, logical_id: str) -> Optional[CameraFrame]:
        """Newest frame from one camera, or ``None`` before the first arrives."""
        frames = self.frame_history.get(logical_id)
        return frames[-1] if frames else None

    def latest_pose(self) -> Optional[Pose]:
        """Newest estimated ego pose (``local -> rig_est``), if any."""
        return self.ego_trajectory.last_pose

    def latest_dynamic_state(self) -> Optional[DynamicState]:
        return self.dynamic_states[-1] if self.dynamic_states else None

    def speed_mps(self) -> float:
        """Speed from the newest dynamic state, in m/s; 0 before the first."""
        state = self.latest_dynamic_state()
        if state is None:
            return 0.0
        v = state.linear_velocity
        return float(np.linalg.norm([v.x, v.y, v.z]))


@dataclass(frozen=True)
class DriveContext:
    """Everything a policy gets for one :meth:`BaseDriver.drive` call."""

    session: SessionState
    #: Planning time; the pose at this instant anchors the returned plan.
    time_now_us: int
    #: The instant the runtime will next step the vehicle to.
    time_query_us: int
    #: CARLA ground truth (lights, other vehicles, speed limit) when the runtime
    #: sends it; ``None`` under an upstream alpasim runtime.
    renderer_data: Optional[CarlaRendererData]


@dataclass
class DriveResult:
    """What a policy returns.

    ``trajectory_in_rig`` is the plan in the rig frame at ``time_now_us``; the
    servicer converts it into the ``local`` frame the response carries.
    """

    trajectory_in_rig: Trajectory
    #: End the rollout now (``DriveResponse.terminate_session``).
    terminate_session: bool = False
    #: Sent back in ``CarlaDriveDebugInfo.scalars``.
    debug_scalars: Dict[str, float] = field(default_factory=dict)
    #: Alternative plans considered, in the rig frame.
    sampled_trajectories_in_rig: List[Trajectory] = field(default_factory=list)


class BaseDriver(ABC):
    """Base class for a driving policy: implement :meth:`drive`.

    The observation hooks are optional; :class:`SessionState` already records
    everything a simple policy needs.
    """

    #: Reported through ``get_version`` and ``CarlaDriveDebugInfo.policy_name``.
    name: str = "base"

    #: Frames per camera kept in ``SessionState.frame_history``.
    frame_history_length: int = 1

    def on_session_start(self, session: SessionState) -> None:  # noqa: B027
        """Called once per rollout, before any observation."""

    def on_session_close(self, session: SessionState) -> None:  # noqa: B027
        """Called once per rollout, after the last :meth:`drive`."""

    def on_image(self, session: SessionState, frame: CameraFrame) -> None:  # noqa: B027
        """Called for every camera frame, after it lands in ``session``."""

    def on_egomotion(self, session: SessionState, new_poses: Trajectory) -> None:  # noqa: B027
        """Called for every egomotion batch, after it lands in ``session``."""

    def on_route(self, session: SessionState) -> None:  # noqa: B027
        """Called whenever a new route lands in ``session``."""

    def on_ground_truth(self, session: SessionState) -> None:  # noqa: B027
        """Called whenever recorded ground truth lands in ``session``."""

    @abstractmethod
    def drive(self, ctx: DriveContext) -> DriveResult:
        """Produce a plan for ``ctx.session`` at ``ctx.time_now_us``."""
