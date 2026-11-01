"""Drive a planned adversarial command, then hand the vehicle back (#391).

What the adversarial actions share once a trigger has fired and its command
is planned (:func:`~..adversity.command.plan_command`): the path is driven
by a :class:`~._trajectory_run.TrajectoryRun` on the path's own
clock -- ``FOLLOW`` (the vehicle's own motion, by default) or ``POSITION``
(TeraSim's ``moveToXY`` teleport) -- which takes the vehicle back from its
traffic backend first.  When the path is done a vehicle is handed back to
the backend (:meth:`~..traffic.driven.BackendDriven.resume_traffic`) still
moving (``stop_at_end=False``) and rejoins traffic from there, as TeraSim's
controller turns SUMO's checks back on; a pedestrian is left standing where
the path ended.

A traffic backend's background vehicle has no entity (#417): its actor is
driven directly (:class:`_ActorFollow`), taken from the backend with
:meth:`~..traffic.base.TrafficBackend.release_actor` and handed back with
:meth:`~..traffic.base.TrafficBackend.resume_actor`.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from ..action_state import ActionState
from ..adversity_activity import ActiveAdversity, begin_adversity, end_adversity
from ..adversity.command import AdversarialCommand, plan_command
from ..adversity.geometry import PredictedState
from ..adversity.observation import Scene
from ..conditions.base import BaseCondition
from ..conditions.trajectory_time import TrajectoryTimeCondition
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..traffic.base import TrafficBackend
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

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["AdversarialExecution", "drivable", "path_trajectory", "start_execution"]

logger = logging.getLogger(__name__)


def path_trajectory(
    name: str,
    path: list[PredictedState],
    box_ahead: float = 0.0,
    box_left: float = 0.0,
) -> Trajectory:
    """*path* as a timed trajectory in the map frame, its times from 0.

    The path is planned for the centre of the vehicle's bounding box (as an
    :class:`~..adversity.observation.AgentSnapshot` has it); the trajectory
    is driven by the actor's origin, which the follow action puts on it.  So
    each pose is the origin's: the centre less the box's offset, *box_ahead*
    along the state's heading and *box_left* to its left (#889).
    """
    if len(path) < 2:
        raise ValueError("a path to drive needs at least two states")
    start = path[0].t
    vertices: list[TrajectoryVertex] = []
    for p in path:
        # A typed local: the vertex takes a BaseCondition, which Codon does not
        # upcast a TrajectoryTimeCondition to inside the call.
        at: BaseCondition = TrajectoryTimeCondition(p.t - start)
        cos_yaw = math.cos(p.yaw)
        sin_yaw = math.sin(p.yaw)
        x = p.x - box_ahead * cos_yaw + box_left * sin_yaw
        y = p.y - box_ahead * sin_yaw - box_left * cos_yaw
        vertices.append(TrajectoryVertex(MapPose(x, y, p.yaw), at))
    return Trajectory(name, vertices)


class _ActorFollow(TrajectoryRun):
    """A follow action on a background vehicle's *actor*, which no entity
    stands for: taken from *backend* when its run begins."""

    _actor: "carla.Actor"
    _backend: TrafficBackend

    def __init__(
        self,
        name: str,
        actor: "carla.Actor",
        backend: TrafficBackend,
        trajectory: Trajectory,
        following_mode: TrajectoryFollowingMode,
        label: str,
    ) -> None:
        super().__init__(
            name,
            trajectory,
            time_reference=TrajectoryTiming(ReferenceContext.RELATIVE),
            following_mode=following_mode,
            timing=TickTiming.PRE_TICK,
            label=label,
            stop_at_end=False,
        )
        self._actor = actor
        self._backend = backend

    def _drive_target(self, world: "carla.World") -> None:
        """Move the actor one tick; take it from its backend first."""
        if self._resolved is None:
            self._resolved = self._resolve(world)
            if self._resolved is None:
                return
        if not self._begun:
            self._begin(world, self._actor)
            if not self._released:
                self._backend.release_actor(world, self._actor)
                self._released = True
        self._advance(world, self._actor)


class AdversarialExecution:
    """One planned command, driven on *entity_name* from its first tick.

    Args:
        entity_name: ``role_name`` of the vehicle or pedestrian.
        path: The planned path of its bounding box's centre, in the Lanelet2
            map frame.
        following_mode: ``FOLLOW`` (default) or ``POSITION``.
        label: For the logs and the follow action's label.
        actor: For a background vehicle no entity stands for, its actor...
        backend: ...and the traffic backend that drives it otherwise.
        box_ahead: How far the box's centre is ahead of the actor's origin
            (m), which is put on the path less it: :func:`path_trajectory`.
        box_left: How far it is left of the origin (m).
    """

    entity_name: str
    _follow: TrajectoryRun
    _handed_back: bool
    _actor: Optional["carla.Actor"]
    _backend: Optional[TrafficBackend]
    _kind: str
    _responding_to: str
    _begun: bool

    def __init__(
        self,
        entity_name: str,
        path: list[PredictedState],
        following_mode: TrajectoryFollowingMode = TrajectoryFollowingMode.FOLLOW,
        label: str = "adversity",
        actor: Optional["carla.Actor"] = None,
        backend: Optional[TrafficBackend] = None,
        box_ahead: float = 0.0,
        box_left: float = 0.0,
        kind: str = "",
        responding_to: str = "",
    ) -> None:
        self.entity_name = str(entity_name)
        trajectory = path_trajectory(label, path, box_ahead, box_left)
        follow: TrajectoryRun = TrajectoryRun(
            self.entity_name,
            trajectory,
            time_reference=TrajectoryTiming(ReferenceContext.RELATIVE),
            following_mode=following_mode,
            timing=TickTiming.PRE_TICK,
            label=label,
            stop_at_end=False,
        )
        if actor is not None and backend is not None:
            # Typed locals: Codon unwraps the Optionals here.
            vehicle: "carla.Actor" = actor
            driver: TrafficBackend = backend
            direct: TrajectoryRun = _ActorFollow(
                self.entity_name, vehicle, driver, trajectory, following_mode, label
            )
            follow = direct
        # The run is the adversarial action's, which holds the entity's
        # movement for it: it takes no control domain of its own.
        follow._claims_control = False
        self._follow = follow
        self._actor = actor
        self._backend = backend
        self._handed_back = False
        # What it carries out, for the conditions asking what is under way
        # (adversity_activity): noted when it first drives, until it hands
        # the road user back.
        self._kind = kind
        self._responding_to = responding_to
        self._begun = False

    @property
    def done(self) -> bool:
        """Whether the path is driven and the vehicle handed back."""
        return self._handed_back

    def tick(self, world: carla.World, elapsed: float) -> None:
        """Drive one tick; once the path is done, hand the vehicle back."""
        if self._handed_back:
            return
        if not self._begun and self._kind:
            begin_adversity(
                ActiveAdversity(
                    self.entity_name,
                    self._kind,
                    self._follow.label,
                    self._responding_to,
                    elapsed,
                )
            )
            self._begun = True
        self._follow.tick(world, elapsed)
        if self._follow.state is not ActionState.COMPLETE:
            return
        self.hand_back(world)

    def hand_back(self, world: carla.World) -> None:
        """Give the vehicle back to its driver: the path is done, or the run
        was stopped (an action overrode it)."""
        if self._handed_back:
            return
        self._handed_back = True
        self._end()
        actor = self._actor
        backend = self._backend
        if actor is not None and backend is not None:
            driver: TrafficBackend = backend
            released: "carla.Actor" = actor
            driver.resume_actor(world, released)
        else:
            vehicle = find_vehicle_entity(self.entity_name)
            if vehicle is None:
                return
            vehicle.resume_traffic(world)
        logger.info(
            "%s: '%s' handed back to traffic", self._follow.label, self.entity_name
        )

    def close(self) -> None:
        """Close the follow action's conditions; what it carried out is over."""
        self._end()
        self._follow.close()

    def _end(self) -> None:
        if self._begun:
            self._begun = False
            end_adversity(self.entity_name)


def drivable(name: str) -> bool:
    """Whether an adversity can be driven on *name*: a registered vehicle or
    pedestrian, what the follow action finds."""
    return find_vehicle_entity(name) is not None or (
        find_pedestrian_entity(name) is not None
    )


def start_execution(
    scene: Scene,
    name: str,
    command: AdversarialCommand,
    following_mode: TrajectoryFollowingMode,
    label: str,
    actor: Optional["carla.Actor"] = None,
    backend: Optional[TrafficBackend] = None,
    kind: str = "",
    responding_to: str = "",
) -> Optional[AdversarialExecution]:
    """*command* planned for *name* on *scene*, ready to drive -- a background
    vehicle's *actor* through its *backend* when given; None, with a warning,
    when it cannot be planned (off the lanes, no lane to change to).  *kind*
    (an ``AdversityKind`` value) is what it carries out, and *responding_to*
    whose adversity a victim's response is (adversity_activity)."""
    observed = scene.vehicle(name)
    try:
        path = plan_command(scene.features, observed, command)
    except ValueError as error:
        logger.warning("%s: %s; nothing done", label, error)
        return None
    # The path is the box centre's, the actor's origin is driven (#889).
    return AdversarialExecution(
        name,
        path,
        following_mode,
        label,
        actor,
        backend,
        observed.agent.box_ahead,
        observed.agent.box_left,
        kind,
        responding_to,
    )
