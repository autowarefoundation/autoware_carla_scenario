"""TeraSim's static adversities: a stalled object and collision objects (#401).

TeraSim's ``StalledObjectAdversity`` and ``CollisionAdversity`` are not
sampled; they are scheduled.  From ``start_time`` to ``end_time`` (scenario
seconds) a vehicle stands still at a given pose -- one for a stalled object,
two for the aftermath of a collision -- and is gone outside that window.

An entity cannot be spawned once a run has started, so the vehicle is one the
scenario spawns in ``setup`` (out of the world, ``spawn.hidden``, or not:
the action takes it out from its first tick).  :class:`StalledObjectAction`
is a :class:`~._trajectory_run.TrajectoryRun` of a trajectory that
stays at the pose from ``start_time`` to ``end_time``, in ``POSITION`` mode
with ``hidden_outside_trajectory``: parked under the map with its physics off
before and after, held at the pose, stopped, in between.
:func:`collision_objects` is two of them.
"""

from __future__ import annotations

from typing import Optional, Union

from ..conditions.base import BaseCondition
from ..conditions.trajectory_time import TrajectoryTimeCondition
from ..coordinate.poses import CarlaWorldPose, Lanelet2Pose
from ..entity_role import EntityRole
from ..trajectory.model import (
    MapPose,
    ReferenceContext,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTiming,
    TrajectoryVertex,
)
from .base import TickTiming
from ._trajectory_run import TrajectoryRun

__all__ = ["FOREVER_S", "StalledObjectAction", "collision_objects"]

#: The end of a stalled object with no ``end_time``: TeraSim's ``-1``, never.
FOREVER_S = 1.0e9


def _held(
    label: str,
    pose: Union[Lanelet2Pose, MapPose, CarlaWorldPose],
    start_time: float,
    end_time: float,
) -> Trajectory:
    """A trajectory that stays at *pose* from *start_time* to *end_time*."""
    # Typed locals: the vertex takes a BaseCondition, which Codon does not
    # upcast a TrajectoryTimeCondition to inside the call.
    appears: BaseCondition = TrajectoryTimeCondition(start_time)
    removed: BaseCondition = TrajectoryTimeCondition(end_time)
    return Trajectory(
        label, [TrajectoryVertex(pose, appears), TrajectoryVertex(pose, removed)]
    )


class StalledObjectAction(TrajectoryRun):
    """A vehicle stalled at *pose* from *start_time* until *end_time*.

    Args:
        entity_name: ``role_name`` of the vehicle; spawned by the scenario.
        pose: Where it stands: a Lanelet2 pose, or an xy+yaw one
            (:class:`~..trajectory.model.MapPose` with its ``yaw``, or a
            CARLA world pose).
        start_time: When it appears (s from the scenario's start).
        end_time: When it is removed; ``None`` never (:data:`FOREVER_S`).
        condition: Starts the action, and so takes the vehicle out of the
            world until *start_time*; ``None`` at once.
        timing: Tick phase.
        label: Human-readable identifier.
        overrides: Take over the vehicle's movement from another action
            holding it (:attr:`~BaseAction.overrides`).

    Raises:
        ValueError: A :class:`MapPose` without a ``yaw`` (a pose that stays
            put has no direction of its own), a negative *start_time*, or an
            *end_time* not after it.
    """

    _pose: Union[Lanelet2Pose, MapPose, CarlaWorldPose]

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        pose: Union[Lanelet2Pose, MapPose, CarlaWorldPose],
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "stalled_object",
        overrides: bool = False,
    ) -> None:
        if isinstance(pose, MapPose) and pose.yaw is None:
            raise ValueError(f"{label}: a stalled object's MapPose needs its yaw")
        start = float(start_time)
        end = FOREVER_S
        if end_time is not None:
            end = float(end_time)
        if start < 0.0:
            raise ValueError(f"{label}: start_time must not be negative")
        if not start < end <= FOREVER_S:
            raise ValueError(
                f"{label}: end_time must be after start_time (and before "
                f"{FOREVER_S:g} s, never)"
            )
        super().__init__(
            entity_name,
            _held(label, pose, start, end),
            TrajectoryTiming(ReferenceContext.ABSOLUTE),
            TrajectoryFollowingMode.POSITION,
            condition=condition,
            timing=timing,
            label=label,
            overrides=overrides,
            hidden_outside_trajectory=True,
        )
        self._pose = pose

    @property
    def pose(self) -> Union[Lanelet2Pose, MapPose, CarlaWorldPose]:
        """Where the vehicle stands."""
        return self._pose

    @property
    def start_time(self) -> float:
        """When it appears (s)."""
        return self._vertex_time(0)

    @property
    def end_time(self) -> Optional[float]:
        """When it is removed (s), or ``None`` for never."""
        end = self._vertex_time(-1)
        if end >= FOREVER_S:
            return None
        return end

    def _vertex_time(self, index: int) -> float:
        """The time of the held trajectory's vertex *index*."""
        time = self.trajectory.vertices[index].time
        assert time is not None  # noqa: S101 -- both vertices carry one
        return time


def collision_objects(
    first: Union[EntityRole, str],
    second: Union[EntityRole, str],
    first_pose: Union[Lanelet2Pose, MapPose, CarlaWorldPose],
    second_pose: Union[Lanelet2Pose, MapPose, CarlaWorldPose],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    *,
    label: str = "collision",
) -> list[StalledObjectAction]:
    """TeraSim's collision objects: two vehicles stalled where they collided.

    Two :class:`StalledObjectAction` of the same window, labelled
    ``<label>_1`` and ``<label>_2``; register both.
    """
    return [
        StalledObjectAction(
            first, first_pose, start_time, end_time, label=f"{label}_1"
        ),
        StalledObjectAction(
            second, second_pose, start_time, end_time, label=f"{label}_2"
        ),
    ]
