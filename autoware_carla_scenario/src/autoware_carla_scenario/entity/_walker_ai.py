"""Walk a routed pedestrian with CARLA's own walker controller (#521).

A pedestrian a ``RoutingAction`` sends along its route in ``driver`` mode is
walked by CARLA's ``controller.ai.walker`` where the map has a pedestrian
navigation mesh: the controller keeps to pavements, crosses at crossings
(waiting at the kerb as CARLA's walkers do) and walks round what stands in
its way, which a walker steered straight at its path does not.  It is given
the route a few targets at a time (:data:`TARGET_SPACING_M` apart along the
walk, the next once it is :data:`REACHED_M` short of the last along the
walk, or that near it), and its top speed is the walk's pace.

**Crossing a road away from a crossing** is a walker's own:
``cross_factor`` (0..1, ``None`` CARLA's default, 0) is the probability that
its routes may use the road anywhere.  CARLA keeps one factor for the world,
and reads it only when a walker's controller starts (``Navigation::AddWalker``,
on the client, which draws the walker's road filter from it then): so the
factor is set to the walker's own just before its controller starts, and
back to CARLA's default straight after -- deterministic for 0 and 1, the
draw CARLA makes for a share in between.  Everything runs on the tick loop's
thread, so no two walkers' factors meet.

**Holding.**  A walker held by its route's timing stays on its controller
with a top speed of 0: taken off it, it would be put back on the navigation
mesh's nearest point when it walked on -- a jump, for one that stood off it.
CARLA gives a walker that has stood 4 s a random new target
(``Navigation::UpdateCrowd``, "unblocking"), so every walker's current
target is sent again every :data:`RETARGET_S` seconds and at once when it
walks on.

**A target out of the controller's reach is moved.**  CARLA routes a walker
that may not use the road (its road filter) only to a point within 2 m of a
walkable polygon off the road (``Navigation::GetAgentRoute``'s pick extent);
for any other it finds no route and the walker walks to a random point of
the map instead (``WalkerManager::SetWalkerRoute``), again each time the
target is sent.  A walk recorded on the road is given a cross factor of 1 and
its road stretches are stepped (below); as a safety net for one that is not
(cross factor under a half: the scenario and the map disagree, which is
warned of), a target on a road lanelet and on no walkway or crossing is
moved to the nearest point of a walkway or crossing.  Standing at the end of
its walk so moved, the walker is held there.

**Road stretches are stepped.**  CARLA's controller does not take a walker
across a road where it was recorded crossing it, whatever its cross factor:
the road costs ten times the pavement in its routing, so any crossing within
that is taken instead, and stepping off a kerb starts a stop-and-check that
re-sending a target restarts.  So for a walker that may cross (cross factor
at least a half) the walk is read along its line, a point a metre, against
the Lanelet2 map: a stretch on a road lanelet and on no walkway or crossing
is walked with the framework's own stepping (``drive`` answers ``False`` and
the caller steps it), off the controller.  It is handed back to the
controller only once it stands on a walkway or a crossing again, at least
:data:`HAND_BACK_PAST_M` past the stretch's end along the walk -- never if
the walk ends on the road.  Its position is compared over the tick its
controller starts again, and a jump of more than :data:`SNAP_M` is said.
While stepped it is no crowd agent: walkers on the controller do not steer
round it, nor it round them, so it gives way instead (:func:`give_way`): it
stands while a walker -- or a vehicle -- stands within :data:`GIVE_WAY_M`
ahead of it on its walk, for at most :data:`GIVE_WAY_MAX_S` at a time (two
stepped walkers facing each other would otherwise wait for ever); it does not
stop in the road for traffic coming.

**At the kerb** it checks before it steps off, up to :data:`KERB_WAIT_MAX_S`:
it waits while the traffic light nearest it (the one whose stop waypoint is
nearest, ``WalkerManager::GetTrafficLightAffecting``, any distance) is green
or yellow for the vehicles -- the light rule CARLA's own walkers keep
(``WalkerEventStopAndCheck``) -- and, given turn-back thresholds, while a
vehicle is a threat by the test :class:`~..actions.walk_straight.WalkStraightAction`
turns back by (:func:`~..actions._road_threat.road_threat`: along the road,
by distance or by time to collision).  Once off the kerb it keeps watching:
a threat while it is within its turn-back distance of the kerb turns it back
to just behind the kerb, where it checks again; further out it carries
on.

A map without a navigation mesh -- ``get_random_location_from_navigation()``
finds nothing -- is said once, and its walkers are stepped along the route as
before (:func:`~.pedestrian_entity.steer_walker`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from ..kinematics import Vector3
from ..kinematics.walker_motion import WalkerVelocities, simulated_seconds

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..trajectory.model import ResolvedTrajectory, TrajectorySample

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_CROSS_FACTOR",
    "REACHED_M",
    "RETARGET_S",
    "TARGET_SPACING_M",
    "drive",
    "give_way",
    "killed_by_crowd",
    "has_navmesh",
    "release",
    "road_stretches",
]

#: CARLA's own pedestrian cross factor (``Navigation::_probability_crossing``).
CARLA_CROSS_FACTOR: float = 0.0
#: Metres between the targets a walker is sent to along its walk.
TARGET_SPACING_M: float = 10.0
#: How near (m) a target counts as reached: that far short of it along the
#: walk, or that far from it in a straight line.
REACHED_M: float = 2.0
#: How often (s of driving) a walker's current target is sent again.
RETARGET_S: float = 2.0
#: How many draws decide a map has no navigation mesh (a draw may miss on
#: one that has).
_NAVMESH_PROBES = 20
#: How near (m) its walk a walker has to stand for its controller to start.
_START_NEAR_M = 3.0
#: A walker this far down (m) is out of the world, not yet brought in.
_HIDDEN_BELOW_M = -100.0
#: Below this (m/s) a walker is held, off its controller.
_HOLD_MPS = 0.05
#: The share of a walker's cross factor from which its road stretches are
#: stepped.
STEP_CROSSING_FROM: float = 0.5
#: How far (m) past a road stretch's end, along the walk, a stepped walker is
#: handed back to its controller (on a walkway or a crossing).
HAND_BACK_PAST_M: float = 1.0
#: How far (m) a walk is sampled apart to find its road stretches.
_SAMPLE_M: float = 1.0
#: A jump (m) over the tick a controller starts that is said.
SNAP_M: float = 0.5
#: How far (m) into a walkway a target moved off the road is put.
_INTO_WALK_M: float = 0.5
#: How many lanelets nearest a target are searched for a walkway.
_NEAREST_LANELETS = 40
#: How far ahead (m) along its walk a stepped walker gives way to a walker.
GIVE_WAY_M: float = 1.5
#: How far to each side (m) of its walk a walker it gives way to stands
#: (their own size added).
_GIVE_WAY_SIDE_M: float = 0.4
#: The longest (s) a stepped walker gives way at a time.
GIVE_WAY_MAX_S: float = 4.0
#: Below this speed (m/s) a vehicle in a stepped walker's way stands.
_STANDING_MPS: float = 0.5
#: The longest (s) a walker waits at the kerb before it steps off
#: (``WalkerEventStopAndCheck(60)``).
KERB_WAIT_MAX_S: float = 60.0
#: How far (m) beyond what a threat reaches a vehicle is not looked at
#: (to spare the lane search for the far ones).
_THREAT_MARGIN_M: float = 10.0
#: The fastest (m/s) a vehicle is taken to close on a walker at, to bound
#: how far out a time to collision reaches.
_THREAT_SPEED_MPS: float = 20.0
#: How near (m) the place behind the kerb a turned-back walker has reached.
_BACK_REACHED_M: float = 0.3
#: How long (s) each stepped walker has given way, by actor id.
_GIVING_WAY: dict[int, float] = {}
#: The tick, for the retargeting clock where the world does not say.
_DEFAULT_DT_S = 0.05

#: Whether each episode's map has a navigation mesh, by world id.
_NAVMESH: dict[int, bool] = {}
#: The walkers' velocities, measured from where they stand: a walker on its
#: controller is placed, and its ``get_velocity()`` reads 0.
_WALKING = WalkerVelocities()


class _Driven:
    """A walker on its controller: where it is sent, and how fast."""

    controller: "carla.WalkerAIController"
    targets: list[tuple[float, "carla.Location"]]
    cross_factor: Optional[float]
    index: int
    speed: float
    running: bool
    since_sent: float
    stretches: list[tuple[float, float]]
    stepping: bool
    started_at: Optional[tuple[float, float]]
    kerb_waited: float
    kerb_light: Optional["carla.TrafficLight"]
    kerb_light_read: bool
    kerb_said: str
    turn_back: Optional[tuple[Optional[float], Optional[float], float]]
    kerb_here: float
    turning_back: bool
    back_to: Optional["TrajectorySample"]
    waiting_back: bool

    def __init__(
        self,
        controller: "carla.WalkerAIController",
        targets: list[tuple[float, "carla.Location"]],
        cross_factor: Optional[float],
    ) -> None:
        self.controller = controller
        self.targets = targets
        self.cross_factor = cross_factor
        self.index = 0
        self.speed = -1.0
        self.running = False
        self.since_sent = 0.0
        #: ``(start, end)`` along the walk of its road stretches, stepped.
        self.stretches: list[tuple[float, float]] = []
        self.stepping = False
        #: Where the walker stood as its controller last started, to compare.
        self.started_at: Optional[tuple[float, float]] = None
        #: How long (s) it has waited at the kerb of the stretch ahead, and
        #: the light it checks there (read once, as CARLA does), if any.
        self.kerb_waited = 0.0
        self.kerb_light = None
        self.kerb_light_read = False
        self.kerb_said = ""
        #: ``(distance_m, ttc_s, within_m)`` it waits and turns back by, or
        #: ``None``: neither.
        self.turn_back: Optional[tuple[Optional[float], Optional[float], float]] = None
        #: Where along its walk it stepped off the kerb (negative: past
        #: turning back, or no kerb), turning back to, or waiting behind it.
        self.kerb_here = -1.0
        self.turning_back = False
        self.back_to = None
        self.waiting_back = False


#: Each driven walker, by actor id.
_DRIVEN: dict[int, _Driven] = {}


def has_navmesh(world: "carla.World") -> bool:
    """Whether *world*'s map has a pedestrian navigation mesh (asked once per
    episode; a map without one is said once)."""
    key = int(world.id) if hasattr(world, "id") else 0
    if key not in _NAVMESH:
        found = False
        for _ in range(_NAVMESH_PROBES):
            if world.get_random_location_from_navigation() is not None:
                found = True
                break
        _NAVMESH[key] = found
        if not found:
            logger.warning(
                "The map has no pedestrian navigation mesh: routed walkers are "
                "stepped along their routes, not walked by CARLA's walker "
                "controller (build one when importing the map)"
            )
    return _NAVMESH[key]


def _start(world: "carla.World", driven: _Driven) -> None:
    """Put the walker on its controller, with its own cross factor."""
    factor = driven.cross_factor
    if factor is not None:
        # frand() <= factor: -1 is never, 1 is always.
        world.set_pedestrians_cross_factor(-1.0 if factor <= 0.0 else float(factor))
    try:
        driven.controller.start()
    finally:
        if factor is not None:
            world.set_pedestrians_cross_factor(CARLA_CROSS_FACTOR)
    driven.running = True
    driven.speed = -1.0
    driven.since_sent = RETARGET_S  # sent at once


def drive(
    walker: "carla.Actor",
    path: "ResolvedTrajectory",
    here: float,
    speed: float,
    location: Optional["carla.Location"] = None,
    cross_factor: Optional[float] = None,
    turn_back: Optional[tuple[Optional[float], Optional[float], float]] = None,
) -> bool:
    """One tick of *walker* (``here`` m along *path*, a ResolvedTrajectory, at
    *location*) on CARLA's walker controller at *speed* m/s, crossing roads
    away from crossings with probability *cross_factor*, waiting at a kerb and
    turning back by *turn_back* (``(distance_m, ttc_s, within_m)``); ``False``
    when the map has no navigation mesh, or while it is stepped, for the
    caller to step it."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    world = walker.get_world()
    if not has_navmesh(world):
        return False
    if location is None:
        location = walker.get_location()
    if int(walker.id) not in _DRIVEN:
        # Not before the walker stands where its walk is: one just brought in
        # (set where it appears) still reads its old place until the world
        # ticks, and a controller started then registers it there -- under
        # the map, at its spawn -- and walks it from wherever that is.
        on = path.at_distance(here)
        if (
            location.z < _HIDDEN_BELOW_M
            or math.hypot(location.x - on.x, location.y - on.y) > _START_NEAR_M
        ):
            return True
        targets: list[tuple[float, "carla.Location"]] = []
        along = TARGET_SPACING_M
        while along < path.length:
            sample = path.at_distance(along)
            targets.append((along, carla.Location(x=sample.x, y=sample.y, z=sample.z)))
            along += TARGET_SPACING_M
        end = path.at_distance(path.length)
        targets.append((path.length, carla.Location(x=end.x, y=end.y, z=end.z)))
        if cross_factor is None or cross_factor < STEP_CROSSING_FROM:
            moved = _onto_walkways(targets)
            if moved:
                # Its walk is on the road where it may not be: the document
                # (its cross factor) and the map disagree.
                logger.warning(
                    "walker %d: %d of its %d targets on the road, where its "
                    "cross factor (%s) keeps it off -- the scenario and the map "
                    "disagree; sent to the nearest walkway instead",
                    int(walker.id),
                    moved,
                    len(targets),
                    "default" if cross_factor is None else "%.2f" % float(cross_factor),
                )
        blueprint = world.get_blueprint_library().find("controller.ai.walker")
        spawned = world.spawn_actor(blueprint, carla.Transform(), attach_to=walker)
        if TYPE_CHECKING:
            controller = spawned.as_walker_ai_controller()
        else:
            controller = spawned
        created = _Driven(controller, targets, cross_factor)
        if cross_factor is not None and cross_factor >= STEP_CROSSING_FROM:
            try:
                created.stretches = road_stretches(path)
            except RuntimeError:  # no Lanelet2 map to read the walk against
                created.stretches = []
            if created.stretches:
                logger.info(
                    "walker %d crosses a road away from a crossing: %s m of "
                    "its walk stepped",
                    int(walker.id),
                    ", ".join("%.0f-%.0f" % stretch for stretch in created.stretches),
                )
        _DRIVEN[int(walker.id)] = created
        logger.debug(
            "walker %d: on CARLA's controller (cross factor %s), %d targets to "
            "(%.1f, %.1f)",
            int(walker.id),
            "default" if cross_factor is None else "%.2f" % float(cross_factor),
            len(targets),
            targets[-1][1].x,
            targets[-1][1].y,
        )
    driven = _DRIVEN[int(walker.id)]
    driven.turn_back = turn_back
    _check_snap(walker, driven, location)
    stepped = _stepped(driven, here, location)
    if stepped and not driven.stepping and driven.running:
        waiting = _kerb_wait(walker, driven, path, here, location)
        if waiting:
            # Stands at the kerb on its controller until the way is clear.
            if driven.speed != 0.0:
                driven.controller.set_max_speed(0.0)
                driven.speed = 0.0
            return True
    if stepped and not driven.stepping:
        driven.stepping = True
        # Off the kerb it watches for a threat to turn back from; one that
        # set off on the road has no kerb behind it.
        driven.kerb_here = here if driven.running else -1.0
        driven.kerb_waited = 0.0
        driven.kerb_light_read = False
        driven.kerb_said = ""
        logger.info(
            "walker %d: stepped over the road from %.1f m", int(walker.id), here
        )
        if driven.running:
            _stop(driven)
            _own_steps(walker)
    elif not stepped and driven.stepping:
        driven.stepping = False
        driven.since_sent = RETARGET_S
        logger.info("walker %d: back on its controller at %.1f m", int(walker.id), here)
    while driven.index < len(driven.targets) - 1:
        along, target = driven.targets[driven.index]
        near = math.hypot(location.x - target.x, location.y - target.y) <= REACHED_M
        if here < along - REACHED_M and not near:
            break
        driven.index += 1
        driven.since_sent = RETARGET_S
    if driven.stepping and driven.turn_back is not None:
        if _turns_back(walker, driven, path, here, location, speed):
            return True  # walking back to the kerb, or waiting behind it
    if driven.stepping:
        return False  # the caller steps it, on the road
    wanted = max(speed, 0.0)
    last = driven.targets[-1][1]
    if (
        driven.index == len(driven.targets) - 1
        and math.hypot(location.x - last.x, location.y - last.y) <= REACHED_M
    ):
        wanted = 0.0  # at its walk's end, as near as its controller goes
    if wanted < _HOLD_MPS:
        wanted = 0.0  # held: on its controller, at a standstill
    if driven.speed < _HOLD_MPS <= wanted:
        driven.since_sent = RETARGET_S  # walking on: its target, at once
    if not driven.running:
        if _kind_at_or_none(location.x, location.y) == "road":
            # Not started standing on the road, where the navigation mesh
            # would put it back on its nearest walkable point (a jump): stepped
            # along its walk until it stands on a walkway or a crossing.
            return False
        driven.started_at = (location.x, location.y)
        _start(world, driven)
    driven.since_sent += _tick_seconds(world)
    if driven.since_sent >= RETARGET_S:
        driven.controller.go_to_location(driven.targets[driven.index][1])
        driven.since_sent = 0.0
    if abs(wanted - driven.speed) > 0.05:
        driven.controller.set_max_speed(wanted)
        driven.speed = wanted
    return True


def _threat(
    walker: "carla.Actor",
    location: "carla.Location",
    turn_back: Optional[tuple[Optional[float], Optional[float], float]],
    standing: bool,
) -> str:
    """What threatens *walker* at *location* by its turn-back thresholds
    (:func:`~..actions._road_threat.road_threat`), or ``""``; *standing*: as
    if it stood (at the kerb), not walking at its velocity."""
    if turn_back is None:
        return ""
    from ..actions._road_threat import road_threat  # noqa: PLC0415

    distance_m, ttc_s, _within = turn_back
    reach = _THREAT_MARGIN_M
    if distance_m is not None:
        reach += distance_m
    if ttc_s is not None:
        reach = max(reach, ttc_s * _THREAT_SPEED_MPS + _THREAT_MARGIN_M)
    walking_x = walking_y = 0.0
    if not standing:
        velocity = _walking(walker, location)
        if velocity is not None:
            walking_x, walking_y = float(velocity.x), float(velocity.y)
    try:
        found = road_threat(
            walker.get_world(), location, walking_x, walking_y, distance_m, ttc_s, reach
        )
    except (AttributeError, RuntimeError):
        return ""  # a world with no vehicles to ask (a test double)
    return "" if found is None else found


def _walking(walker: "carla.Actor", location: "carla.Location") -> Optional[Vector3]:
    """*walker*'s velocity at *location*, measured from where it stood when
    last asked (:class:`~..kinematics.walker_motion.WalkerVelocities`), or
    ``None`` where it cannot be read."""
    try:
        return _WALKING.walker(walker, simulated_seconds(walker.get_world()), location)
    except (AttributeError, RuntimeError):
        return None


def _turns_back(
    walker: "carla.Actor",
    driven: _Driven,
    path: "ResolvedTrajectory",
    here: float,
    location: "carla.Location",
    speed: float,
) -> bool:
    """One tick of a stepped walker watching for a threat off the kerb
    (``turn_back``): whether it walks back to the kerb, or waits behind it,
    this tick -- else the caller steps it on.  Within ``within_m`` of the
    kerb a threat turns it back to :data:`~..actions._road_threat.STEP_BACK_M`
    behind it (no further back than where its walk began), where it checks at
    the kerb again; further out it carries on, as a
    :class:`~..actions.walk_straight.WalkStraightAction` pedestrian does."""
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from ..actions._road_threat import STEP_BACK_M  # noqa: PLC0415
    from .pedestrian_entity import walker_control_speed  # noqa: PLC0415

    turn_back = driven.turn_back
    assert turn_back is not None
    _distance, _ttc, within_m = turn_back
    # Where it stands each tick off the kerb, for its velocity when a threat
    # is looked for (asked again this tick, the same).
    _walking(walker, location)
    if driven.waiting_back:
        if _kerb_wait(walker, driven, path, here, location):
            walker.apply_control(carla.WalkerControl(speed=0.0))
            return True
        driven.waiting_back = False
        driven.kerb_here = here  # off the kerb again
        driven.kerb_waited = 0.0
        driven.kerb_light_read = False
        driven.kerb_said = ""
        return False
    if driven.turning_back:
        target = driven.back_to
        assert target is not None
        dx, dy = target.x - location.x, target.y - location.y
        gap = math.hypot(dx, dy)
        if gap <= _BACK_REACHED_M:
            walker.apply_control(carla.WalkerControl(speed=0.0))
            driven.turning_back = False
            driven.waiting_back = True
            logger.info("walker %d: is back behind the kerb", int(walker.id))
            return True
        walker.apply_control(
            carla.WalkerControl(
                direction=carla.Vector3D(x=dx / gap, y=dy / gap, z=0.0),
                speed=walker_control_speed(max(speed, 1.0)),
            )
        )
        return True
    if driven.kerb_here < 0.0:
        return False
    out_m = here - driven.kerb_here
    if out_m > within_m:
        driven.kerb_here = -1.0  # too far out to turn back: it carries on
        return False
    threat = _threat(walker, location, turn_back, standing=False)
    if not threat:
        return False
    back = path.at_distance(max(0.0, driven.kerb_here - STEP_BACK_M))
    driven.back_to = back
    driven.turning_back = True
    logger.info(
        "walker %d: turns back %.2f m onto the road (%s)",
        int(walker.id),
        max(out_m, 0.0),
        threat,
    )
    return _turns_back(walker, driven, path, here, location, speed)


def _kerb_wait(
    walker: "carla.Actor",
    driven: _Driven,
    path: "ResolvedTrajectory",
    here: float,
    location: "carla.Location",
) -> bool:
    """Whether *walker*, at the kerb of the road stretch ahead (``here`` m
    along *path*, at *location*), waits this tick before it steps off: the
    light nearest it green or yellow for the vehicles, or a vehicle a threat
    by its turn-back thresholds -- for at most :data:`KERB_WAIT_MAX_S`."""
    world = walker.get_world()
    if driven.kerb_waited >= KERB_WAIT_MAX_S:
        if driven.kerb_said != "timeout":
            driven.kerb_said = "timeout"
            logger.info(
                "walker %d: steps off the kerb after %.0f s waiting",
                int(walker.id),
                driven.kerb_waited,
            )
        return False
    if not driven.kerb_light_read:
        driven.kerb_light_read = True
        driven.kerb_light = _nearest_light(world, location)
    why = ""
    light = driven.kerb_light
    if light is not None:
        try:
            state = str(light.get_state())
        except RuntimeError:
            state = ""
        if state.endswith("Green") or state.endswith("Yellow"):
            why = "the light (%d) %s for the vehicles" % (
                int(light.id),
                state.rsplit(".", 1)[-1],
            )
    if not why:
        threat = _threat(walker, location, driven.turn_back, standing=True)
        if threat:
            why = threat
    if not why:
        if driven.kerb_waited > 0.0:
            logger.info(
                "walker %d: steps off the kerb after %.1f s waiting",
                int(walker.id),
                driven.kerb_waited,
            )
        return False
    # Said once per light or vehicle waited for, not as its distance changes.
    said = why.split(" m away")[0].split(" s away")[0]
    said = said.rsplit(" ", 1)[0] if said != why else said
    if said != driven.kerb_said:
        driven.kerb_said = said
        logger.info("walker %d: waits at the kerb for %s", int(walker.id), why)
    driven.kerb_waited += _tick_seconds(world)
    return True


#: Each episode's traffic lights' stop waypoints, by world id.
_LIGHTS: dict[int, list[tuple["carla.TrafficLight", float, float, float]]] = {}


def _nearest_light(
    world: "carla.World", location: "carla.Location"
) -> Optional["carla.TrafficLight"]:
    """The traffic light whose stop waypoint is nearest *location*, at any
    distance (``WalkerManager::GetTrafficLightAffecting``, max_distance -1),
    or ``None`` on a map with none."""
    key = int(world.id) if hasattr(world, "id") else 0
    if key not in _LIGHTS:
        stops: list[tuple["carla.TrafficLight", float, float, float]] = []
        try:
            for actor in world.get_actors().filter("traffic.traffic_light"):
                if TYPE_CHECKING:
                    light = actor.as_traffic_light()
                else:
                    light = actor
                for waypoint in light.get_stop_waypoints():
                    at = waypoint.transform.location
                    stops.append((light, at.x, at.y, at.z))
        except (AttributeError, RuntimeError):
            stops = []
        _LIGHTS.clear()
        _LIGHTS[key] = stops
    best: Optional["carla.TrafficLight"] = None
    nearest = float("inf")
    for light, x, y, z in _LIGHTS[key]:
        d = (location.x - x) ** 2 + (location.y - y) ** 2 + (location.z - z) ** 2
        if d < nearest:
            nearest, best = d, light
    return best


def give_way(
    walker: "carla.Actor",
    path: "ResolvedTrajectory",
    here: float,
    location: "carla.Location",
) -> bool:
    """Whether *walker*, stepped (``here`` m along *path*, at *location*),
    is to stand this tick for a walker in its way: one within
    :data:`GIVE_WAY_M` ahead along its walk, for at most
    :data:`GIVE_WAY_MAX_S` at a time."""
    from ..utils.clearance import first_in_way  # noqa: PLC0415

    key = int(walker.id)
    world = walker.get_world()
    ahead = path.at_distance(min(here + 1.0, path.length))
    heading = math.degrees(math.atan2(ahead.y - location.y, ahead.x - location.x))
    found = first_in_way(
        world,
        key,
        location.x,
        location.y,
        heading,
        "walker",
        GIVE_WAY_M,
        _GIVE_WAY_SIDE_M,
    )
    what = "walker"
    if found is None:
        # A vehicle standing in its way (its width and the margin across);
        # traffic coming is not stopped for in the middle of the road.
        try:
            half = float(walker.bounding_box.extent.y) + _GIVE_WAY_SIDE_M
        except AttributeError:
            half = _GIVE_WAY_SIDE_M
        found = first_in_way(
            world, key, location.x, location.y, heading, "vehicle", GIVE_WAY_M, half
        )
        if found is not None and found.speed >= _STANDING_MPS:
            found = None  # coming: not waited for in the middle of the road
        what = "vehicle"
    if found is None:
        _GIVING_WAY.pop(key, None)
        return False
    waited = _GIVING_WAY.get(key, 0.0)
    if waited >= GIVE_WAY_MAX_S:
        return False  # walks on; waits again once the way has been clear
    if waited == 0.0:
        logger.info(
            "walker %d: gives way to %s %d, %.1f m along its walk",
            key,
            what,
            int(found.id),
            here,
        )
    _GIVING_WAY[key] = waited + _tick_seconds(world)
    return True


def road_stretches(path: "ResolvedTrajectory") -> list[tuple[float, float]]:
    """``(start, end)`` along *path* (m) of its stretches on a road lanelet
    and on no walkway or crossing: where it crosses a road away from one."""
    stretches: list[tuple[float, float]] = []
    start: Optional[float] = None
    along = 0.0
    while True:
        point = path.at_distance(min(along, path.length))
        if _kind_at(point.x, point.y) == "road":
            if start is None:
                start = max(0.0, along - _SAMPLE_M / 2.0)
        elif start is not None:
            stretches.append((float(start), along - _SAMPLE_M / 2.0))
            start = None
        if along >= path.length:
            break
        along += _SAMPLE_M
    if start is not None:
        stretches.append((float(start), path.length))
    return stretches


def _stepped(driven: _Driven, here: float, location: "carla.Location") -> bool:
    """Whether the walker, *here* m along its walk, is to be stepped: in a
    road stretch (from just before it), or past one and not yet on a
    walkway or a crossing :data:`HAND_BACK_PAST_M` beyond it."""
    for start, end in driven.stretches:
        if here < start - 0.5:
            return False
        if here <= end:
            return True
        if driven.stepping and not (
            here >= end + HAND_BACK_PAST_M
            and _kind_at(location.x, location.y) == "walk"
        ):
            # Past the stretch: back to the controller only standing on a
            # walkway or a crossing, a metre on from the kerb.
            return True
    return False


def _own_steps(walker: "carla.Actor") -> None:
    """Let *walker* take steps again (``apply_control``) off its controller.
    Starting the controller set the walker's physics and its collisions off
    (``WalkerAIController::Start``) and stopping it sets neither on again: a
    walker moved by CARLA's navigation ignores steps until its physics is on,
    and with its collisions off it falls through the ground."""
    try:
        walker.set_simulate_physics(True)
    except (AttributeError, RuntimeError):
        pass
    try:
        walker.set_collisions(True)
    except (AttributeError, RuntimeError):
        pass


def _check_snap(
    walker: "carla.Actor", driven: _Driven, location: "carla.Location"
) -> None:
    """Say a jump over the tick the walker's controller started."""
    before = driven.started_at
    if before is None:
        return
    driven.started_at = None
    jump = math.hypot(location.x - before[0], location.y - before[1])
    if jump > SNAP_M:
        logger.warning(
            "walker %d jumped %.1f m as its controller started at (%.1f, %.1f)",
            int(walker.id),
            jump,
            before[0],
            before[1],
        )


def _onto_walkways(targets: list[tuple[float, "carla.Location"]]) -> int:
    """Move each of *targets* on a road lanelet (and on no walkway or
    crossing) to the nearest walkway or crossing; how many were moved."""
    moved = 0
    for _along, target in targets:
        try:
            if _kind_at(target.x, target.y) != "road":
                continue
            onto = _nearest_walk(target.x, target.y)
        except RuntimeError:  # no Lanelet2 map to read
            return moved
        if onto is None:
            continue
        target.x, target.y = onto
        moved += 1
    return moved


def _nearest_walk(x: float, y: float) -> Optional[tuple[float, float]]:
    """The point of a walkway or crossing nearest CARLA ``(x, y)``,
    :data:`_INTO_WALK_M` inside it where that is still on one, or ``None``
    with none near."""
    import lanelet2.core  # noqa: PLC0415
    import lanelet2.geometry  # noqa: PLC0415

    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..route.geometry import MapFeatures  # noqa: PLC0415

    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    px, py = x + offset_x, -y + offset_y
    point = lanelet2.core.BasicPoint2d(px, py)
    best: Optional[tuple[float, float, float]] = None
    for distance, lanelet in lanelet2.geometry.findNearest(
        mm.lanelet_map.laneletLayer, point, _NEAREST_LANELETS
    ):
        if not MapFeatures.is_walkable(lanelet):
            continue
        if best is not None and distance >= best[0]:
            break
        corners = [(c.x, c.y) for c in lanelet.polygon2d()]
        for i, (ax, ay) in enumerate(corners):
            bx, by = corners[(i + 1) % len(corners)]
            dx, dy = bx - ax, by - ay
            span = dx * dx + dy * dy
            u = 0.0 if span <= 0.0 else ((px - ax) * dx + (py - ay) * dy) / span
            u = min(1.0, max(0.0, u))
            qx, qy = ax + u * dx, ay + u * dy
            gap = math.hypot(px - qx, py - qy)
            if best is None or gap < best[0]:
                best = (gap, qx, qy)
    if best is None:
        return None
    gap, qx, qy = best
    if gap > 1e-6:
        ix = qx + (qx - px) / gap * _INTO_WALK_M
        iy = qy + (qy - py) / gap * _INTO_WALK_M
        if _kind_at(ix - offset_x, -(iy - offset_y)) == "walk":
            qx, qy = ix, iy
    return (qx - offset_x, -(qy - offset_y))


def _kind_at_or_none(x: float, y: float) -> str:
    """:func:`_kind_at`, or ``""`` without a Lanelet2 map to read."""
    try:
        return _kind_at(x, y)
    except RuntimeError:
        return ""


def _kind_at(x: float, y: float) -> str:
    """``"walk"`` on a walkway or a crossing, ``"road"`` on a road lanelet
    only, ``""`` on none: what the Lanelet2 map has at CARLA ``(x, y)``."""
    import lanelet2.core  # noqa: PLC0415
    import lanelet2.geometry  # noqa: PLC0415

    from ..coordinate.map_manager import MapManager  # noqa: PLC0415
    from ..route.geometry import MapFeatures  # noqa: PLC0415

    mm = MapManager.get_instance()
    offset_x, offset_y = mm.mgrs_offset
    point = lanelet2.core.BasicPoint2d(x + offset_x, -y + offset_y)
    kind = ""
    for _distance, lanelet in lanelet2.geometry.findWithin2d(
        mm.lanelet_map.laneletLayer, point, 0.0
    ):
        if MapFeatures.is_walkable(lanelet):
            return "walk"
        if MapFeatures.is_road(lanelet):
            kind = "road"
    return kind


def killed_by_crowd(walker_id: int) -> bool:
    """Whether CARLA's crowd has taken *walker_id* for run over: a vehicle
    came within its avoidance radius (``dtCrowd`` marks the agent dead with
    no contact -- a walker on its controller has its collisions off), and
    ``WalkerNavigation`` then destroys its controller and starts the walker's
    death life span, which removes it some ten seconds later."""
    driven = _DRIVEN.get(int(walker_id))
    if driven is None or not driven.running:
        return False
    try:
        return not bool(driven.controller.is_alive)
    except (AttributeError, RuntimeError):
        return False


def release(walker_id: int) -> None:
    """Stop and destroy *walker_id*'s controller, if it has one."""
    _GIVING_WAY.pop(int(walker_id), None)
    _WALKING.forget(int(walker_id))
    driven = _DRIVEN.pop(int(walker_id), None)
    if driven is None:
        return
    if driven.running:
        _stop(driven)
    try:
        driven.controller.destroy()
    except RuntimeError:
        pass


def _stop(driven: _Driven) -> None:
    """Take the walker off its controller."""
    driven.running = False
    try:
        driven.controller.stop()
    except RuntimeError:
        pass


def _tick_seconds(world: "carla.World") -> float:
    try:
        return float(world.get_snapshot().timestamp.delta_seconds)
    except (AttributeError, RuntimeError):
        return _DEFAULT_DT_S
