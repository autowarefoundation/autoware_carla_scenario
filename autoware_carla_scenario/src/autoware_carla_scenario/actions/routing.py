"""Send an entity along a route: through its waypoints, on time, to a goal.

The one action that moves a road user from place to place.  It says *where*
-- a goal, and the waypoints on the way, each of which may carry *when*: an
arrival time or a departure condition -- and the :class:`RoutingMode` says
*who drives*:

* ``driver`` (the default): the entity's own driver.  An Autoware ego is
  given a mission; a vehicle a traffic backend drives (the TrafficManager,
  SUMO) the route as a path to follow
  (:meth:`~autoware_carla_scenario.traffic.driven.BackendDriven.route_to`);
  a pedestrian is walked along its route on the walkable lanelets
  (:meth:`~autoware_carla_scenario.entity.pedestrian_entity.PedestrianEntity.route_to`).
  The legs between waypoints are Lanelet2 routes.  Waypoint times are met by
  speed alone (:mod:`._route_pacing`); an Autoware ego takes no speed, so it
  is only given the route, with a warning that its timing cannot be met.
* ``follow``: the action drives the entity itself, along straight legs
  between the waypoints, with the controller the driver interface uses -- a
  pedestrian is steered along them -- and the waypoints' times kept by that
  controller (:class:`~._trajectory_run.TrajectoryRun`).
* ``position``: the entity is placed on the legs every tick: a kinematic
  replay that keeps the times exactly, whatever the vehicle could do.

**When it runs is the caller's choice, with one hard constraint.**  The ego
needs a goal at least once during initialization: the runner waits for it to
report ready *before* the tick loop starts, and an Autoware ego only becomes
ready once it has localized, routed and engaged, so a first goal delivered from
inside the loop would never arrive.  That one is registered with
:meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_init`.
Re-routing part-way through a run, or routing another entity once something
has happened, is an ordinary tick-loop action.
"""

from __future__ import annotations

import bisect
import enum
import logging
import math
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ..conditions.base import BaseCondition
from ..coordinate import CarlaWorldPose, GroundProjectionConfig, Lanelet2Pose
from ..entity.pedestrian_entity import DEFAULT_WALKING_SPEED_MPS, PedestrianEntity
from ..entity.registry import find_pedestrian_entity, find_vehicle_entity
from ..entity_role import EntityRole
from ..traffic.driven import BackendDriven
from ..traffic.final_approach import lane_changes
from ..trajectory.model import (
    ResolvedTrajectory,
    Trajectory,
    TrajectoryFollowingMode,
    TrajectoryTiming,
    TrajectoryVertex,
)
from ..waypoint_arrival import WaypointArrival
from ._route_pacing import PaceMark, RoutePacer
from ._road_threat import DEFAULT_TURN_BACK_WITHIN_M
from ._trajectory_run import ARRIVAL_TOLERANCE_M, TrajectoryRun, _entity_actor
from ..control_domain import MOVEMENT, ControlDomain
from .base import TickTiming

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..driver.control import ControlConfig

logger = logging.getLogger(__name__)

__all__ = ["RoutingAction", "RoutingMode"]

#: Metres between the points of the path a vehicle's route is paced on.
_PACE_STEP_M = 2.0
#: The speed (m/s) a vehicle with no speed limit to read may go at.
_DEFAULT_LIMIT_MPS = 50.0 / 3.6
#: The fastest (m/s) CARLA walks a pedestrian.
_WALKER_LIMIT_MPS = 2.0
#: The farthest (m) a pedestrian's written-out waypoint may be from the one
#: before for the waypoints to be walked as written (a recorded walk).
_WALK_AS_WRITTEN_M = 10.0
#: How far (m) off its planned route a vehicle is said to have left it.
_OFF_PATH_M = 8.0
#: Metres between the samples of a track matched to the lanes.
_MATCH_STEP_M = 2.0
#: A speed command that changes less than this (m/s) is not sent again.
_SPEED_RESEND_MPS = 0.05
#: An actor this far down (m) is parked under the map, out of the scene.
_HIDDEN_BELOW_M: float = 100.0
#: How far (m) off its path an entity may stand for its run to start where
#: it stands, the path's nearest point.
_ON_PATH_M: float = 6.0

#: How near (m) its goal a stop_at_goal vehicle is driven the rest of the way
#: itself, to stop exactly where it is to (:meth:`RoutingAction._finish_exactly`).
_FINISH_M: float = 15.0
#: The spacing (m) of the points it is driven along.
_FINISH_STEP_M: float = 2.0
#: Waypoints further apart (m) than this are joined along the lanes, not
#: straight (_finish_points).
_CHORD_M: float = 4.0
#: How far (m) before such a change a stop_at_goal vehicle is driven itself.
_BEFORE_LANE_CHANGE_M: float = 5.0
#: A waypoint this far (m) off its lanes' middle leaves them; and how far
#: before its goal (m) one may, for it to be driven itself from before.
_OFF_LANE_M: float = 0.75
_OFF_LANE_WITHIN_M: float = 40.0


def _finish_points(
    written: ResolvedTrajectory,
    planned: Optional[ResolvedTrajectory],
    x: float,
    y: float,
) -> list[tuple[float, float, float]]:
    """The points, every :data:`_FINISH_STEP_M` m from ``(x, y)``, a
    stop_at_goal vehicle drives the rest of its route along (its goal not
    among them): its trajectory as written -- off its lanes' middle as the
    recording was (vehicle_96, moved over early to pass vehicle_88) -- but
    where two of its waypoints are more than :data:`_CHORD_M` apart, along
    its lanes (*planned*) between them, moved over from how far off them the
    one lies to how far the other does: straight, the line cut the corner its
    lanes turn (vehicle_41, onto the kerb)."""
    out: list[tuple[float, float, float]] = []
    here = written.project(x, y)
    marks = [written.vertex_distance(i) for i in range(len(written.xs))]
    along = here + _FINISH_STEP_M
    near = 0.0
    while along < written.length:
        index = max(bisect.bisect_right(marks, along) - 1, 0)
        index = min(index, len(marks) - 2)
        span = marks[index + 1] - marks[index]
        sample = written.at_distance(along)
        if planned is None or span <= _CHORD_M:
            out.append((sample.x, sample.y, sample.z))
            along += _FINISH_STEP_M
            continue
        lanes: ResolvedTrajectory = planned
        ax, ay = written.xs[index], written.ys[index]
        bx, by = written.xs[index + 1], written.ys[index + 1]
        a_s = lanes.project(ax, ay, near=near)
        b_s = lanes.project(bx, by, near=a_s)
        near = a_s
        share = (along - marks[index]) / span
        on = a_s + (b_s - a_s) * share
        side = (
            _side(lanes, a_s, ax, ay) * (1.0 - share)
            + _side(lanes, b_s, bx, by) * share
        )
        px, py, pz = _beside(lanes, on, side)
        out.append((px, py, pz))
        along += _FINISH_STEP_M
    return out


def _kerb_free(
    world: "carla.World",
    actor: "carla.Actor",
    points: list[tuple[float, float, float]],
    x: float,
    y: float,
) -> list[tuple[float, float, float]]:
    """*points* -- *actor*'s path from ``(x, y)`` -- moved off the sidewalks
    its corners would stand on (:func:`~..utils.clearance.off_the_kerb`), as
    they are with no map to look at."""
    from ..coordinate.map_manager import run_carla_map  # noqa: PLC0415
    from ..utils.clearance import off_the_kerb  # noqa: PLC0415

    try:
        carla_map = run_carla_map(world)
    except (AttributeError, RuntimeError):
        return points
    extent = actor.bounding_box.extent
    return off_the_kerb(carla_map, points, float(extent.x), float(extent.y), x, y)


def _direction(path: ResolvedTrajectory, s: float) -> tuple[float, float]:
    """The unit direction of *path* at *s*; ``(0, 0)`` where it has none."""
    ahead = path.at_distance(min(s + 0.5, path.length))
    behind = path.at_distance(max(s - 0.5, 0.0))
    hx, hy = ahead.x - behind.x, ahead.y - behind.y
    norm = math.hypot(hx, hy)
    if norm < 1e-6:
        return 0.0, 0.0
    return hx / norm, hy / norm


def _side(path: ResolvedTrajectory, s: float, x: float, y: float) -> float:
    """How far (m) ``(x, y)`` lies left of *path* at *s* (negative: right)."""
    at = path.at_distance(s)
    hx, hy = _direction(path, s)
    return hx * (y - at.y) - hy * (x - at.x)


def _beside(
    path: ResolvedTrajectory, s: float, side: float
) -> tuple[float, float, float]:
    """The point *side* m left of *path* at *s*."""
    at = path.at_distance(s)
    hx, hy = _direction(path, s)
    return at.x - hy * side, at.y + hx * side, at.z


def _finish_from(
    path: ResolvedTrajectory, written: Optional[ResolvedTrajectory] = None
) -> float:
    """How far (m) before *path*'s end a stop_at_goal vehicle is driven the
    rest of the way itself: :data:`_FINISH_M`, or from before the route's last
    change to the lane beside -- which its driver, keeping to its lane, would
    leave to the last metres (vehicle_134, crossing behind a lorry)."""
    changes = lane_changes(list(path.xs), list(path.ys))
    last = changes[-1][0] if changes else -1
    finish = _FINISH_M
    if last >= 0:
        before = path.length - path.vertex_distance(last) + _BEFORE_LANE_CHANGE_M
        finish = max(finish, before)
    if written is not None:
        # And from before where its trajectory as written leaves its lanes'
        # middle (vehicle_96, moved over early to pass a bus's corner, which
        # its driver -- keeping to the middle -- ran into).
        off = _leaves_lanes(path, written)
        if off is not None and path.length - off <= _OFF_LANE_WITHIN_M:
            finish = max(finish, path.length - off + _BEFORE_LANE_CHANGE_M)
    return finish


def _leaves_lanes(
    path: ResolvedTrajectory, written: ResolvedTrajectory
) -> Optional[float]:
    """How far (m) along *path* the first of *written*'s waypoints more than
    :data:`_OFF_LANE_M` off it lies, or ``None``."""
    near = 0.0
    for x, y in zip(written.xs, written.ys):
        near = path.project(x, y, near=near)
        if abs(_side(path, near, x, y)) > _OFF_LANE_M:
            return near
    return None


class RoutingMode(enum.Enum):
    """Who moves the entity along its route (see :mod:`.routing`)."""

    #: The entity's own driver: TrafficManager / SUMO, the pedestrian route
    #: walker, Autoware; legs between waypoints routed on the Lanelet2 map.
    DRIVER = "driver"
    #: The action's own controller, along straight legs between waypoints.
    FOLLOW = "follow"
    #: Placed on the straight legs every tick (a kinematic replay).
    POSITION = "position"


class RoutingAction(TrajectoryRun):
    """Send an entity through waypoints to a goal (see :mod:`.routing`).

    Names the entity rather than holding it, as every other action does: the
    lookup happens when the action runs, so an NPC registered during ``setup()``
    or an ego swapped in after the action was built is still found.

    **The route** is either a *goal* with Lanelet2 *waypoints* -- the
    destination an ego is given -- or a *trajectory*, whose vertices are the
    waypoints and whose last vertex is the goal: map poses, Lanelet2 poses,
    poses relative to an entity's lane or to the scenario's route, each with
    an optional departure condition (:class:`~autoware_carla_scenario.TrajectoryVertex`):
    a ``TrajectoryTimeCondition`` is the waypoint's arrival time (on the
    clock *time_reference* sets; ignored without one), any other condition
    is waited for before the entity goes on.

    It routes an entity already in the scene: one spawned out of the world
    is brought in by a :class:`~.spawn.SpawnAction` first (the routing card
    triggered on the spawn card completing).  A run starts where the entity
    stands, at the nearest point of its path within :data:`_ON_PATH_M` (past
    *initial_distance_offset*): a spawn moved along its lane to a clear spot
    starts its route there.

    **In ``driver`` mode** the route starts where the entity is and goes
    through every vertex after it.  With nothing to time it is handed over once and the run
    ends at once for a vehicle; a pedestrian's run ends when it arrives.  A
    route with times or conditions keeps running until the entity reaches
    the goal, the action setting its driver's speed every tick.  ``follow``
    and ``position`` move the entity along the trajectory itself, its first
    vertex the start, until it reaches the end (see
    :class:`~._trajectory_run.TrajectoryRun` for waypoint conditions,
    hiding and the end of a run).

    Every mode logs when each timed waypoint was reached against when it was
    due (:meth:`take_waypoint_arrivals`, the trajectory log's
    ``waypoint_arrival`` records).

    Args:
        entity_name: Role name of the entity to route.
        goal: Lanelet2 pose the entity is routed to; or give *trajectory*.
        condition: Condition gating the action.
        timing: Which tick phase to run on, when registered on the loop.
        label: Action label.
        once: Whether the action fires at most once.
        overrides: Take over a control domain of the entity another action
            holds (:attr:`~BaseAction.overrides`).
        initial_pose: CARLA world pose the entity is expected to start at, used
            to initialize localization (``driver`` mode).  ``None`` reads the
            attached actor instead.
        ground_projection: Settings used to snap the goal to the road surface.
        waypoints: Lanelet2 poses the route passes through, with *goal*.
        walking_speed: How fast a pedestrian walks the route in ``driver``
            mode where *speed* does not say, in m/s.
        turn_back_ttc_s: For a pedestrian stepped over a road stretch: a
            vehicle at most this (s) from it, along the road, is a threat --
            it waits at the kerb for it, and turns back from it within
            *turn_back_within_m* of the kerb.  ``None`` (default): not by time
            to collision.  The names, meanings and test of
            :class:`~.walk_straight.WalkStraightAction`'s.
        turn_back_distance_m: Likewise, a vehicle at most this far (m) along
            the road.  ``None`` (default): not by distance.  With neither, a
            pedestrian waits at the kerb only for the light.
        turn_back_within_m: How far onto the road, from the kerb, it may be
            and still turn back (m).
        cross_factor: The probability (0..1) that a pedestrian walked by
            CARLA's walker controller (``driver`` mode, a map with a
            navigation mesh) may cross a road away from a crossing: 1 for a
            recorded walk that did, 0 keeps it to crossings; ``None`` is
            CARLA's default (0).  Beside *walking_speed*: both are how this
            pedestrian walks this route.
        trajectory: The waypoints, the last one the goal, instead of *goal*
            and *waypoints*.
        mode: Who moves the entity (:class:`RoutingMode`).
        time_reference: How the waypoints' times map onto the scenario clock;
            ``None`` ignores them.
        initial_distance_offset: Start this many metres along the trajectory.
        until: Overrides what ends the run.
        control_config: Gains for ``follow`` mode.
        speed: The speed (m/s) the entity goes at where no time paces it;
            ``None`` takes its own (``follow``/``position``: the speed it has
            when the run starts; ``driver``: its driver's).
        stop_at_end: Brake a vehicle to a stop where a ``follow`` or
            ``position`` run ends (default).
        stop_at_goal: In ``driver`` mode, stop a vehicle where its route ends
            and hold it there (its driver asked for no speed) rather than let
            its driver drive on at its own speed (the default): a road user
            replayed from a recording waits where it was last seen going.

    Raises:
        ValueError: Without a goal or a trajectory, with both, with
            waypoints beside a trajectory, a walking speed that is not
            positive, or what :class:`~._trajectory_run.TrajectoryRun` refuses.
    """

    _stop_at_goal: bool
    _finishing: bool
    _finish_m: Optional[float]
    _goal: Optional[Lanelet2Pose]
    _from_where_it_stands: bool
    _initial_pose: Optional[CarlaWorldPose]
    _ground_projection: Optional[GroundProjectionConfig]
    _waypoints: tuple[Lanelet2Pose, ...]
    _walking_speed: float
    _cross_factor: Optional[float]
    _turn_back_ttc_s: Optional[float]
    _turn_back_distance_m: Optional[float]
    _turn_back_within_m: float
    _mode: RoutingMode
    _walking: bool
    _stepping: bool
    _pacer: Optional[RoutePacer]
    _path: Optional[ResolvedTrajectory]
    _here: float
    _sent: float
    _arrivals: list[WaypointArrival]
    _stop_vertices: list[int]
    _mark_points: list[tuple[int, float, float]]
    _mark_lanes: list[tuple[int, Lanelet2Pose]]
    _start_pose: Optional[CarlaWorldPose]
    _warned_off: bool
    _route_start: Optional[Lanelet2Pose]

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        goal: Optional[Lanelet2Pose] = None,
        condition: Optional[BaseCondition] = None,
        timing: TickTiming = TickTiming.PRE_TICK,
        *,
        label: str = "routing",
        once: bool = True,
        overrides: bool = False,
        initial_pose: Optional[CarlaWorldPose] = None,
        ground_projection: Optional[GroundProjectionConfig] = None,
        waypoints: Sequence[Lanelet2Pose] = (),
        walking_speed: float = DEFAULT_WALKING_SPEED_MPS,
        cross_factor: Optional[float] = None,
        turn_back_ttc_s: Optional[float] = None,
        turn_back_distance_m: Optional[float] = None,
        turn_back_within_m: float = DEFAULT_TURN_BACK_WITHIN_M,
        trajectory: Optional[Trajectory] = None,
        mode: RoutingMode = RoutingMode.DRIVER,
        time_reference: Optional[TrajectoryTiming] = None,
        initial_distance_offset: float = 0.0,
        until: Optional[BaseCondition] = None,
        control_config: Optional["ControlConfig"] = None,
        speed: Optional[float] = None,
        stop_at_end: bool = True,
        stop_at_goal: bool = False,
    ) -> None:
        if not walking_speed > 0.0:
            raise ValueError("walking_speed must be positive")
        if turn_back_ttc_s is not None and turn_back_ttc_s < 0.0:
            raise ValueError("turn_back_ttc_s must not be negative")
        if turn_back_distance_m is not None and turn_back_distance_m < 0.0:
            raise ValueError("turn_back_distance_m must not be negative")
        if turn_back_within_m < 0.0:
            raise ValueError("turn_back_within_m must not be negative")
        if cross_factor is not None:
            share: float = cross_factor
            if not 0.0 <= share <= 1.0:
                raise ValueError("cross_factor is a probability, 0 to 1")
        if (goal is None) == (trajectory is None):
            raise ValueError(
                f"{label}: give a route a goal (with its waypoints) or a "
                "trajectory, one of the two"
            )
        stops = tuple(waypoints)
        route: Trajectory = _goal_route(label, stops, goal, trajectory)
        if trajectory is not None and stops:
            raise ValueError(
                f"{label}: a trajectory's vertices are its waypoints; give no "
                "waypoints beside it"
            )
        following = TrajectoryFollowingMode.FOLLOW
        if mode is RoutingMode.POSITION:
            following = TrajectoryFollowingMode.POSITION
        super().__init__(
            entity_name,
            route,
            time_reference,
            following,
            initial_distance_offset,
            condition,
            timing,
            label=label,
            once=once,
            overrides=overrides,
            until=until,
            control_config=control_config,
            speed=speed,
            stop_at_end=stop_at_end,
        )
        self._goal = goal
        self._stop_at_goal = bool(stop_at_goal)
        self._finishing = False
        self._finish_m = None
        self._from_where_it_stands = False
        self._initial_pose = initial_pose
        self._ground_projection = ground_projection
        self._waypoints = stops
        self._walking_speed = float(walking_speed)
        self._cross_factor = cross_factor
        self._turn_back_ttc_s = turn_back_ttc_s
        self._turn_back_distance_m = turn_back_distance_m
        self._turn_back_within_m = float(turn_back_within_m)
        self._mode = mode
        self._walking = False
        self._stepping = False
        self._pacer = None
        self._path = None
        self._here = 0.0
        self._sent = -1.0
        self._arrivals = []
        self._stop_vertices = []
        self._mark_points = []
        self._mark_lanes = []
        self._start_pose = None
        self._warned_off = False
        self._route_start = None

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def waypoints(self) -> tuple[Lanelet2Pose, ...]:
        """The Lanelet2 poses the route given with a goal passes through."""
        return self._waypoints

    @property
    def goal(self) -> Optional[Lanelet2Pose]:
        """The Lanelet2 pose the entity is routed to; ``None`` for a trajectory."""
        return self._goal

    @property
    def entity_name(self) -> Union[EntityRole, str]:
        """Role name of the entity this routes."""
        return self._entity_name

    @property
    def mode(self) -> RoutingMode:
        """Who moves the entity."""
        return self._mode

    @property
    def walking(self) -> bool:
        """Whether a pedestrian this routed in ``driver`` mode is still walking."""
        return self._walking

    @property
    def needs_tick_loop(self) -> bool:
        """Anything but a vehicle handed an untimed route keeps acting."""
        if self._mode is not RoutingMode.DRIVER:
            return True
        if self._goal is None:
            return True  # a route written out is followed to its end
        if self._timed():
            return True
        return self._walks_pedestrian()

    def _walks_pedestrian(self) -> bool:
        """Whether the entity is a pedestrian, which ``driver`` mode walks."""
        return (
            find_vehicle_entity(self._entity_name) is None
            and find_pedestrian_entity(self._entity_name) is not None
        )

    def take_waypoint_arrivals(self) -> list[WaypointArrival]:
        """The timed waypoints reached since the last call, for the run's log."""
        arrivals = self._arrivals
        self._arrivals = []
        pacer = self._pacer
        if pacer is not None:
            paced: RoutePacer = pacer
            arrivals.extend(paced.take_arrivals())
        return arrivals

    # ------------------------------------------------------------------
    # BaseAction / TrajectoryRun interface
    # ------------------------------------------------------------------

    def _control_claims(self) -> list[tuple[str, ControlDomain]]:
        """``follow``/``position``: the entity's movement.  ``driver``: its
        lateral domain (the route its driver takes), and its longitudinal one
        too where the action sets the pace -- a speed, a timed waypoint, a
        stop at the goal -- or walks a pedestrian."""
        entity = str(self._entity_name)
        if (
            self._mode is not RoutingMode.DRIVER
            or self._given_speed is not None
            or self._stop_at_goal
            or self._timed()
            or self._walks_pedestrian()
        ):
            return [(entity, domain) for domain in MOVEMENT]
        return [(entity, ControlDomain.LATERAL)]

    def _reissues_by_default(self) -> bool:
        """``follow``/``position`` move the entity every tick; ``driver`` mode
        while it paces or walks the entity, and until its run has begun: one
        that could not begin on its trigger's tick (waiting for where it
        appears to clear, for its actor or its references) tries again on
        the next."""
        if self._mode is not RoutingMode.DRIVER:
            return True
        return self._stepping or not self._begun

    def execute(self, world: "carla.World") -> None:
        """Route the entity, or move it one tick along its route."""
        from ..action_state import ActionState  # noqa: PLC0415

        if self.state is ActionState.STANDBY:
            self._keep(self._pacer)
            self._pacer = None
            self._path = None
            self._stepping = False
            self._walking = False
            self._sent = -1.0
            self._warned_off = False
            self._route_start = None
        super().execute(world)

    def _drive_target(self, world: "carla.World") -> None:
        """Move what this action drives one tick, as its mode says."""
        if self._mode is not RoutingMode.DRIVER:
            super()._drive_target(world)
            self._observe_run()
            return
        vehicle = find_vehicle_entity(self._entity_name)
        if vehicle is not None:
            driven: BackendDriven = vehicle
            self._drive_route(world, driven)
            return
        pedestrian = find_pedestrian_entity(self._entity_name)
        if pedestrian is None:
            self._stepping = False
            self._walking = False
            self._entity_absent()
            return
        walker: PedestrianEntity = pedestrian
        self._walk_route(world, walker)

    def _gate_opened(self, index: int) -> None:
        """Note when a ``follow``/``position`` waypoint's condition held."""
        pacer = self._pacer
        if pacer is not None:
            paced: RoutePacer = pacer
            paced.note_open(index, self._elapsed)

    # ------------------------------------------------------------------
    # follow / position: arrivals on the trajectory
    # ------------------------------------------------------------------

    def _observe_run(self) -> None:
        """Note the waypoints a ``follow``/``position`` run has reached."""
        resolved = self._resolved
        if not self._begun or resolved is None:
            return
        path: ResolvedTrajectory = resolved
        if self._pacer is None:
            marks: list[PaceMark] = []
            for index, vertex in enumerate(self._trajectory.vertices):
                marks.append(
                    PaceMark(
                        index,
                        path.vertex_distance(index),
                        self._due(vertex),
                        vertex.gate,
                    )
                )
            self._pacer = RoutePacer(
                str(self._entity_name),
                self._mode.value,
                marks,
                ARRIVAL_TOLERANCE_M,
            )
        actor = _entity_actor(str(self._entity_name))
        pacer = self._pacer
        if actor is None or pacer is None:
            return
        found: "carla.Actor" = actor
        paced: RoutePacer = pacer
        location = found.get_transform().location
        self._here = path.project(location.x, location.y, near=self._here)
        paced.observe(self._here, self._elapsed)

    # ------------------------------------------------------------------
    # driver: a vehicle
    # ------------------------------------------------------------------

    def _drive_route(self, world: "carla.World", entity: BackendDriven) -> None:
        """One tick of ``driver`` mode for a vehicle."""
        actor: Optional["carla.Actor"] = None
        if hasattr(entity, "actor"):
            actor = entity.actor
        if self._begun:
            if actor is not None:
                moving: "carla.Actor" = actor
                self._pace_vehicle(world, entity, moving)
            return
        external = hasattr(entity, "use_autopilot") and entity.use_autopilot is False
        if not self._begin_route(world, actor):
            return
        stops = self._stops(False)
        if not external and self._resolved is not None:
            stops = self._lane_stops(stops)
        if actor is not None and not external:
            # Spawned out of the world and kept from its backend until it was
            # brought in: back on it, to drive the route it is about to be
            # given (nothing for one its backend drives already).
            entity.resume_traffic(world)
        initial = self._initial_pose
        if initial is None and self._resolved is not None:
            initial = self._start_pose
        route_start = self._route_start
        if stops and route_start is not None:
            begin: Lanelet2Pose = route_start
            logger.info(
                "%s: '%s' routed from %d@%.1f through %s",
                self.label,
                str(self._entity_name),
                begin.lanelet_id,
                begin.s,
                ", ".join(f"{p.lanelet_id}@{p.s:.1f}" for p in stops),
            )
            whole: list[Lanelet2Pose] = [begin]
            whole.extend(stops)
            entity.follow_route(world, whole)
        elif stops:
            self._hand_route(world, entity, stops, initial)
        if external:
            if self._timed():
                logger.warning(
                    "%s: '%s' is driven by an external stack, which takes no "
                    "speed: it is given the route, and its waypoints' times "
                    "and conditions cannot be met",
                    self.label,
                    str(self._entity_name),
                )
            self._finished = True
            return
        start = self._start_pose
        # A route written out is followed to its end, timed or not: what
        # ends the run (and so what a despawn after it waits for) is arrival.
        follows = self._timed() or self._goal is None
        if actor is not None and start is not None and stops and follows:
            vehicle: "carla.Actor" = actor
            here: CarlaWorldPose = start
            self._plan_vehicle_pacing(here, stops, vehicle)
        if self._pacer is None and self._given_speed is not None:
            cruise: float = self._given_speed
            entity.set_desired_speed(world, cruise * 3.6)
        if self._pacer is None:
            self._finished = True
            return
        self._stepping = True

    def _hand_route(
        self,
        world: "carla.World",
        entity: BackendDriven,
        stops: list[Lanelet2Pose],
        initial: Optional[CarlaWorldPose],
    ) -> None:
        """Give *entity* the route through *stops*, the last the goal."""
        goal = stops[-1]
        via = stops[:-1]
        if TYPE_CHECKING:
            # Codon calls a method subclasses override (ExternallyDrivenEgo's,
            # BridgedEgoEntity's) through a dispatch thunk, which takes no
            # keyword argument (docs/typecheck.md): the check calls
            # BackendDriven's own, and each override through its class.
            BackendDriven.route_to(
                entity,
                world,
                goal,
                initial_pose=initial,
                ground_projection=self._ground_projection,
                waypoints=via,
            )
        else:
            entity.route_to(
                world,
                goal,
                initial_pose=initial,
                ground_projection=self._ground_projection,
                waypoints=via,
            )

    def _plan_vehicle_pacing(
        self, start: CarlaWorldPose, stops: list[Lanelet2Pose], actor: "carla.Actor"
    ) -> None:
        """Lay out the path the backend drives *stops* on, to pace it along."""
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..coordinate.transform import to_carla_world  # noqa: PLC0415
        from ..odd.route_plan import PlannedRoute, RouteError, route_poses  # noqa: PLC0415
        from ..trajectory.relative_lane import reference_lane_pose_of  # noqa: PLC0415

        mm = MapManager.get_instance()
        route_start = self._route_start
        if route_start is not None:
            begin: Lanelet2Pose = route_start
        else:
            begin = reference_lane_pose_of(start)
        route = PlannedRoute(
            start=begin,
            goal=stops[-1],
            via=tuple(stops[:-1]),
        )
        try:
            poses = route_poses(route, mm.lanelet_map, mm.routing_graph, _PACE_STEP_M)
        except RouteError as error:
            logger.warning(
                "%s: '%s' cannot be paced to its waypoints' times: %s",
                self.label,
                str(self._entity_name),
                error,
            )
            return
        xs, ys, zs = [start.x], [start.y], [start.z]
        yaws: list[Optional[float]] = [None]
        for pose in poses:
            point = to_carla_world(pose)
            xs.append(point.x)
            ys.append(point.y)
            zs.append(point.z)
            yaws.append(None)
        path = ResolvedTrajectory(xs, ys, zs, yaws)
        self._start_pacing(path, actor, ARRIVAL_TOLERANCE_M)

    def _pace_vehicle(
        self, world: "carla.World", entity: BackendDriven, actor: "carla.Actor"
    ) -> None:
        """Set the vehicle's speed for its next timed waypoint."""
        path = self._path
        pacer = self._pacer
        if path is None or pacer is None:
            return
        planned: ResolvedTrajectory = path
        paced: RoutePacer = pacer
        try:
            location = actor.get_transform().location
        except RuntimeError:
            # Gone from the world (a traffic backend removed it): nothing
            # left to pace.
            logger.warning(
                "%s: '%s' is gone; its route ends here",
                self.label,
                str(self._entity_name),
            )
            self._finished = True
            self._stepping = False
            return
        self._here = planned.project(location.x, location.y, near=self._here)
        paced.observe(self._here, self._elapsed)
        on_path = planned.at_distance(self._here)
        off = math.hypot(on_path.x - location.x, on_path.y - location.y)
        if off > _OFF_PATH_M and not self._warned_off:
            self._warned_off = True
            logger.warning(
                "%s: '%s' is %.0f m off its route, %.0f of %.0f m along it",
                self.label,
                str(self._entity_name),
                off,
                self._here,
                planned.length,
            )
        limit = _speed_limit(actor)
        cruise = limit
        if self._given_speed is not None:
            given: float = self._given_speed
            cruise = min(given, limit)
        if (
            self._stop_at_goal
            and not self._finishing
            and planned.length - self._here <= self._finish_distance(planned)
        ):
            self._finish_exactly(world, entity, location)
        if self._here >= planned.length - ARRIVAL_TOLERANCE_M:
            logger.info(
                "%s: '%s' reached the end of its route",
                self.label,
                str(self._entity_name),
            )
            if self._stop_at_goal:
                # Stopped where it arrived, and held there.
                entity.set_desired_speed(world, 0.0)
            elif paced.paced or self._given_speed is not None:
                # Arrived: the driver's own speed again from here on.
                entity.set_desired_speed(world, cruise * 3.6)
            self._finished = True
            self._stepping = False
            return
        if not paced.paced and self._given_speed is None:
            return  # nothing to time: the driver's own speed
        speed = paced.speed(
            world,
            self._here,
            self._elapsed,
            cruise,
            limit,
            (location.x, location.y),
        )
        if abs(speed - self._sent) >= _SPEED_RESEND_MPS or (
            speed == 0.0 and self._sent != 0.0
        ):
            entity.set_desired_speed(world, speed * 3.6)
            self._sent = speed

    def _finish_distance(self, planned: ResolvedTrajectory) -> float:
        """:func:`_finish_from` for *planned*, worked out once."""
        if self._finish_m is None:
            self._finish_m = _finish_from(planned, self._resolved)
        finish: float = self._finish_m
        return finish

    def _finish_exactly(
        self, world: "carla.World", entity: BackendDriven, location: "carla.Location"
    ) -> None:
        """Have the backend drive the rest of the route -- the trajectory as
        written, its last vertex where it is to stop, a lane's width off the
        middle as it may be -- itself and stop there (stop_at_goal)."""
        import typesafe_carla.carla as carla  # noqa: PLC0415

        self._finishing = True
        resolved = self._resolved
        if resolved is None:
            return
        written: ResolvedTrajectory = resolved
        end = written.end()
        here = written.project(location.x, location.y)
        placed = _finish_points(written, self._path, location.x, location.y)
        if hasattr(entity, "actor") and entity.actor is not None:
            moving: "carla.Actor" = entity.actor
            placed = _kerb_free(world, moving, placed, location.x, location.y)
        points = [carla.Location(x=x, y=y, z=z) for x, y, z in placed]
        points.append(carla.Location(x=end.x, y=end.y, z=end.z))
        if entity.finish_at(world, points):
            logger.info(
                "%s: '%s' is driven the last %.0f m to where it is to stop",
                self.label,
                str(self._entity_name),
                written.length - here,
            )

    # ------------------------------------------------------------------
    # driver: a pedestrian
    # ------------------------------------------------------------------

    def _walk_route(self, world: "carla.World", walker: PedestrianEntity) -> None:
        """One tick of ``driver`` mode for a pedestrian."""
        actor: Optional["carla.Actor"] = None
        if hasattr(walker, "actor"):
            actor = walker.actor
        if not self._begun:
            if not self._begin_route(world, actor):
                return
            stops = self._stops(True)
            if not stops:
                self._finished = True
                return
            initial = self._initial_pose
            if initial is None and self._resolved is not None:
                initial = self._start_pose
            walker.set_turn_back(
                self._turn_back_distance_m,
                self._turn_back_ttc_s,
                self._turn_back_within_m,
            )
            track = self._walked_track()
            if track is not None:
                # A recorded walk, its points a few metres apart: walked as it
                # was, not re-routed between them on the walkable lanelets
                # (which sends a pedestrian at a junction's corner down a
                # walkway the recording never took).
                walked: ResolvedTrajectory = track
                walker.walk_along(walked, self._cruise_walk(), self._cross_factor)
            else:
                walker.route_to(
                    world,
                    stops[-1],
                    initial_pose=initial,
                    ground_projection=self._ground_projection,
                    waypoints=stops[:-1],
                    walking_speed=self._cruise_walk(),
                    cross_factor=self._cross_factor,
                )
            if actor is not None and self._timed():
                walk = walker.walk_path
                if walk is not None:
                    planned: ResolvedTrajectory = walk
                    standing: "carla.Actor" = actor
                    self._start_pacing(planned, standing, 1.0)
            self._walking = walker.walking
            self._stepping = self._walking
            if not self._walking:
                self._finished = True
            return
        speed: Optional[float] = None
        path = self._path
        pacer = self._pacer
        if path is not None and pacer is not None and actor is not None:
            planned_walk: ResolvedTrajectory = path
            paced: RoutePacer = pacer
            moving: "carla.Actor" = actor
            location = moving.get_transform().location
            self._here = planned_walk.project(location.x, location.y, near=self._here)
            paced.observe(self._here, self._elapsed)
            speed = paced.speed(
                world,
                self._here,
                self._elapsed,
                self._cruise_walk(),
                _WALKER_LIMIT_MPS,
                (location.x, location.y),
            )
        if actor is not None:
            if TYPE_CHECKING:
                pass  # utils.clearance is not compiled (docs/typecheck.md)
            else:
                from ._trajectory_run import _vehicle_before_walker  # noqa: PLC0415

                if _vehicle_before_walker(actor):
                    # A vehicle in its way (a car stopped where a cyclist rides
                    # on): it waits for it rather than walk into it.
                    speed = 0.0
        if speed is None:
            walker.walk_on()
        else:
            walker.walk_on(speed)
        self._walking = walker.walking
        if not self._walking:
            self._finished = True
            self._stepping = False

    def _walked_track(self) -> Optional[ResolvedTrajectory]:
        """The written-out waypoints from where the run starts, as a walk, when
        none is more than :data:`_WALK_AS_WRITTEN_M` from the one before;
        ``None`` for waypoints to route between."""
        resolved = self._resolved
        start = self._start_pose
        if self._goal is not None or resolved is None or start is None:
            return None
        path: ResolvedTrajectory = resolved
        here: CarlaWorldPose = start
        xs, ys, zs = [here.x], [here.y], [here.z]
        yaws: list[Optional[float]] = [None]
        for index in self._stop_vertices:
            sample = path.at_vertex(index)
            if math.hypot(sample.x - xs[-1], sample.y - ys[-1]) > _WALK_AS_WRITTEN_M:
                return None
            if math.hypot(sample.x - xs[-1], sample.y - ys[-1]) < 0.05:
                continue
            xs.append(sample.x)
            ys.append(sample.y)
            zs.append(sample.z)
            yaws.append(None)
        if len(xs) < 2:
            return None
        return ResolvedTrajectory(xs, ys, zs, yaws)

    def _cruise_walk(self) -> float:
        """The speed a pedestrian walks its route at where nothing paces it."""
        if self._given_speed is not None:
            given: float = self._given_speed
            if given > 0.0:
                return given
        return self._walking_speed

    # ------------------------------------------------------------------
    # driver: shared
    # ------------------------------------------------------------------

    def _timed(self) -> bool:
        """Whether a waypoint has a time (with a reference) or a condition."""
        for vertex in self._trajectory.vertices:
            if vertex.gate is not None:
                return True
            if vertex.time is not None and self._time_reference is not None:
                return True
        return False

    def _due(self, vertex: TrajectoryVertex) -> Optional[float]:
        """The scenario time *vertex* is due at, or ``None``."""
        time = vertex.time
        if time is None or self._time_reference is None:
            return None
        at: float = time
        return self._scenario_time_of(at)

    def _begin_route(
        self, world: "carla.World", actor: Optional["carla.Actor"]
    ) -> bool:
        """Start a ``driver`` run: resolve the waypoints, bring the entity in,
        note where it starts (:attr:`_start_pose`).  ``False`` to try again
        next tick: an entity with no actor yet, or references not there yet.
        """
        needs_world = self._needs_world()
        if actor is None and needs_world:
            logger.warning("%s: '%s' has no actor", self.label, str(self._entity_name))
            return False
        if self._resolved is None and needs_world:
            self._resolved = self._resolve(world)
            if self._resolved is None:
                return False
        self._begun = True
        self._finished = False
        self._start_time = self._elapsed
        self._clock_shift = 0.0
        self._here = 0.0
        self._start_pose = self._initial_pose
        if actor is None:
            return True
        found: "carla.Actor" = actor
        standing = found.get_transform()
        resolved = self._resolved
        if resolved is not None and standing.location.z > -_HIDDEN_BELOW_M:
            path: ResolvedTrajectory = resolved
            # Where it stands: a spawn may have moved it along its lane.
            here = path.project(standing.location.x, standing.location.y)
            on = path.at_distance(here)
            gap = math.hypot(standing.location.x - on.x, standing.location.y - on.y)
            if gap <= _ON_PATH_M and here > self._initial_distance_offset:
                self._initial_distance_offset = here
            self._from_where_it_stands = gap <= _ON_PATH_M
        if self._start_pose is None:
            self._start_pose = CarlaWorldPose.from_carla_transform(standing)
        return True

    def _needs_world(self) -> bool:
        """Whether the waypoints have to be placed in the world first."""
        if self._initial_distance_offset > 0.0:
            return True
        for vertex in self._trajectory.vertices:
            if not isinstance(vertex.position, Lanelet2Pose):
                return True
        return False

    def _stops(self, walker: bool) -> list[Lanelet2Pose]:
        """The waypoints of a ``driver`` route, as Lanelet2 poses.

        A vertex before *initial_distance_offset* is passed already, and the
        one the entity appears on is where it stands.  Their indices go to
        :attr:`_stop_vertices`, in the same order.
        """
        resolved = self._resolved
        offset = self._initial_distance_offset
        stops: list[Lanelet2Pose] = []
        self._stop_vertices = []
        self._mark_points = []
        self._mark_lanes = []
        goal = self._goal
        if goal is not None and resolved is None:
            # As given: the goal stands for both ends of a route without
            # waypoints, and is not to be passed twice.
            end: Lanelet2Pose = goal
            count = len(self._trajectory.vertices)
            stops.extend(self._waypoints)
            stops.append(end)
            first = count - len(stops)
            self._stop_vertices = list(range(first, count))
            self._mark_lanes = list(zip(self._stop_vertices, stops))
            return stops
        for index, vertex in enumerate(self._trajectory.vertices):
            if resolved is not None:
                placed: ResolvedTrajectory = resolved
                along = placed.vertex_distance(index)
                if along < offset - 1e-6:
                    continue
                if self._from_where_it_stands and along <= offset + 0.5:
                    continue  # the vertex it stands on
            position = vertex.position
            if isinstance(position, Lanelet2Pose):
                # Named as the class: Codon does not narrow a union by
                # isinstance() (docs/typecheck.md).
                lane: Lanelet2Pose = position
                stops.append(lane)
                self._mark_lanes.append((index, lane))
            elif resolved is not None:
                world_path: ResolvedTrajectory = resolved
                sample = world_path.at_vertex(index)
                self._mark_points.append((index, sample.x, sample.y))
                stops.append(
                    _lane_pose(
                        CarlaWorldPose(
                            x=sample.x, y=sample.y, z=sample.z, yaw=sample.yaw
                        ),
                        walker,
                    )
                )
            else:
                continue
            self._stop_vertices.append(index)
        return stops

    def _lane_stops(self, stops: list[Lanelet2Pose]) -> list[Lanelet2Pose]:
        """The stops a vehicle's route takes to keep to the lanes its written
        waypoints were driven on: the track between them matched to the
        lanes (:func:`~autoware_carla_scenario.traffic.route_match.match_track`),
        from where the run starts.  A track that cannot be matched falls back
        to *stops*, less those no lane route reaches (:meth:`_drivable`),
        with a warning that says why."""
        resolved = self._resolved
        if resolved is None:
            return stops
        path: ResolvedTrajectory = resolved
        samples: list[tuple[float, float, float]] = []
        along = self._initial_distance_offset
        while along < path.length + _MATCH_STEP_M:
            sample = path.at_distance(min(along, path.length))
            samples.append((sample.x, sample.y, sample.yaw))
            along += _MATCH_STEP_M
        matched, problem = _matched_lanes(samples)
        if matched:
            # Started where the track starts, on its lane, rather than on
            # whichever lane is under the vehicle (the opposite one, a
            # junction branch), which a route would have to loop back from.
            self._route_start = matched[0]
            return matched[1:] or matched
        logger.warning(
            "%s: the track of '%s' was not matched to lanes (%s); its "
            "waypoints are routed one by one instead",
            self.label,
            str(self._entity_name),
            problem,
        )
        return self._drivable(stops)

    def _drivable(self, stops: list[Lanelet2Pose]) -> list[Lanelet2Pose]:
        """*stops* less the ones no lane route reaches from the stop before.

        A waypoint written off the lanes (at the roadside, inside a junction)
        is placed on the lanelet under it, which may be one the route cannot
        drive to next -- the opposite lane, a crossing branch.  Such a
        waypoint is left out, with :attr:`_stop_vertices`, rather than lose
        the whole route; its time or condition goes with it.
        """
        from ..coordinate.map_manager import MapManager  # noqa: PLC0415
        from ..trajectory.relative_lane import reference_lane_pose_of  # noqa: PLC0415

        start = self._start_pose
        if start is None or not stops:
            return stops
        here: CarlaWorldPose = start
        mm = MapManager.get_instance()
        layer = mm.lanelet_map.laneletLayer
        graph = mm.routing_graph
        last = reference_lane_pose_of(here)
        kept: list[Lanelet2Pose] = []
        indices: list[int] = []
        for stop, index in zip(stops, self._stop_vertices):
            reachable = stop.lanelet_id == last.lanelet_id and stop.s >= last.s - 0.5
            if not reachable and stop.lanelet_id != last.lanelet_id:
                path = graph.shortestPath(
                    layer[int(last.lanelet_id)], layer[int(stop.lanelet_id)]
                )
                reachable = path is not None
            if not reachable:
                continue
            kept.append(stop)
            indices.append(index)
            last = stop
        if len(kept) < len(stops):
            logger.warning(
                "%s: %d of the %d waypoints of '%s' are on no lane route from "
                "the one before; left out",
                self.label,
                len(stops) - len(kept),
                len(stops),
                str(self._entity_name),
            )
        self._stop_vertices = indices
        return kept

    def _start_pacing(
        self, path: ResolvedTrajectory, actor: "carla.Actor", tolerance: float
    ) -> None:
        """Pace the entity along *path*, on which it passes its waypoints
        (:attr:`_mark_points`)."""
        vertices = self._trajectory.vertices
        marks: list[PaceMark] = []
        near = 0.0
        previous_x, previous_y = path.xs[0], path.ys[0]
        points = list(self._mark_points)
        for at, lane in self._mark_lanes:
            points.append(_mark_point(at, lane))
        points.sort(key=_vertex_of)
        for index, x, y in points:
            # As far on as the straight line there could take the path.
            window = math.hypot(x - previous_x, y - previous_y) * 2.0 + 30.0
            near = path.project(x, y, near=near, window=window)
            previous_x, previous_y = x, y
            vertex = vertices[index]
            marks.append(PaceMark(index, near, self._due(vertex), vertex.gate))
        velocity = actor.get_velocity()
        self._path = path
        self._pacer = RoutePacer(
            str(self._entity_name),
            self._mode.value,
            marks,
            tolerance,
            math.hypot(velocity.x, velocity.y),
        )

    def _keep(self, pacer: Optional[RoutePacer]) -> None:
        """Keep the arrivals a run's *pacer* noted that were not taken yet."""
        if pacer is not None:
            done: RoutePacer = pacer
            self._arrivals.extend(done.take_arrivals())


def _goal_route(
    label: str,
    waypoints: tuple[Lanelet2Pose, ...],
    goal: Optional[Lanelet2Pose],
    trajectory: Optional[Trajectory],
) -> Trajectory:
    """The trajectory a route given as *waypoints* and a *goal* stands for,
    or *trajectory* itself."""
    if trajectory is not None:
        given: Trajectory = trajectory
        return given
    assert goal is not None
    end: Lanelet2Pose = goal
    vertices: list[TrajectoryVertex] = [TrajectoryVertex(pose) for pose in waypoints]
    vertices.append(TrajectoryVertex(end))
    if len(vertices) < 2:
        # A trajectory needs two vertices; the goal stands for both ends.
        vertices.insert(0, TrajectoryVertex(end))
    return Trajectory(label, vertices)


def _vertex_of(point: tuple[int, float, float]) -> int:
    return point[0]


def _mark_point(index: int, pose: Lanelet2Pose) -> tuple[int, float, float]:
    """Vertex *index* at Lanelet2 *pose*, as a CARLA world point."""
    from ..coordinate.transform import to_carla_world  # noqa: PLC0415

    point = to_carla_world(pose)
    return (index, point.x, point.y)


def _matched_lanes(
    samples: list[tuple[float, float, float]],
) -> tuple[list[Lanelet2Pose], str]:
    """The lane stops a track of CARLA ``(x, y, yaw)`` samples runs along,
    or none and why (:func:`~autoware_carla_scenario.traffic.route_match.match_track`)."""
    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..traffic.route_match import TrackMatchError, match_track  # noqa: PLC0415

    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    frame = [(x + offset_x, -y + offset_y, -math.radians(yaw)) for x, y, yaw in samples]
    try:
        return match_track(frame, mm.lanelet_map, mm.routing_graph), ""
    except TrackMatchError as error:
        return [], str(error)


def _lane_pose(pose: CarlaWorldPose, walker: bool) -> Lanelet2Pose:
    """*pose* on the lanelet under it: a walkable one for a *walker*."""
    from ..trajectory.relative_lane import reference_lane_pose_of  # noqa: PLC0415

    if walker:
        from ..entity.pedestrian_entity import _on_walkable_lanelet  # noqa: PLC0415

        on = _on_walkable_lanelet(pose)
        if on is not None:
            walkable: Lanelet2Pose = on
            return walkable
    return reference_lane_pose_of(pose)


def _speed_limit(actor: "carla.Actor") -> float:
    """The speed limit (m/s) where *actor* drives, or a town default."""
    from ..utils.vehicles import as_vehicle  # noqa: PLC0415

    try:
        limit = float(as_vehicle(actor).get_speed_limit()) / 3.6
    except (AttributeError, RuntimeError):
        return _DEFAULT_LIMIT_MPS
    if limit <= 0.0:
        return _DEFAULT_LIMIT_MPS
    return limit
