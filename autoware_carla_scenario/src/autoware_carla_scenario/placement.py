"""Place a road user so a scenario measure reads a given value at the start.

A scenario parameter usually is a raw setting -- "the NPC starts 10 m further
along its lanelet than the ego along its own" -- and an ODD-sampled sweep
writes the value it drew for a measure into it through a control's ``scale``
and ``offset``.  Where the raw setting and the measure are not the same
quantity that mapping is a guess: the ego's and the NPC's lanelets do not run
alike on a curve (the outer one is longer, and neighbouring lanelets need not
start or end side by side), and the measure is bumper to bumper, so the arc
length along one is not the gap
:data:`~autoware_carla_scenario.measures.VEHICLE_AHEAD_GAP_M` reads, and no
constant offset is right everywhere.

A measure that the initial placement alone determines can instead be the
parameter: the scenario takes "the measure should read *v* at the start"
and solves for the placement, evaluating the measure's own definition -- the
same code, not a copy of it -- on the poses and the vehicles it would spawn.
:func:`solve_along` is the solver, a 1-D root find along a lanelet;
:func:`place_at_vehicle_ahead_gap` applies it to the gap to a vehicle ahead,
with the vehicles' bumpers read off their blueprints before either is
spawned (:func:`vehicle_bumpers`)::

    ego = self.ego_config.spawn_location.value      # after _setup_ego_spawn()
    npc_pose = place_at_vehicle_ahead_gap(
        ego, cfg.npc_lanelet_id, cfg.npc_gap_m,
        ego_bumpers=vehicle_bumpers(world, self.ego_config.vehicle_type),
        npc_bumpers=vehicle_bumpers(world, cfg.npc_vehicle_type),
    )

:func:`place_at_stop_line_distance` places the ego itself the same way, so
:data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M` reads a given
distance to the stop line ahead.

A control that writes such a parameter declares ``target: measure``
(:mod:`autoware_carla_scenario.odd.sampler`, docs/odd.md, "Controls").
"""

from __future__ import annotations

from typing import Callable, Optional

import typesafe_carla.carla as carla

from .constants import BUMPER_PROBE_ROLE_NAME
from .coordinate.poses import Lanelet2Pose
from .coordinate.transform import lanelet_length, to_carla_world
from .coordinate.map_manager import MapManager
from .measures import (
    AHEAD_RANGE_M,
    STOP_LINE_DISTANCE_M,
    VEHICLE_AHEAD_GAP_M,
    Bumpers,
    bumpers_of_box,
    stop_line_distance_between,
    stop_line_lane,
    vehicle_ahead_gap_between,
)

__all__ = [
    "place_at_stop_line_distance",
    "place_at_vehicle_ahead_gap",
    "solve_along",
    "vehicle_bumpers",
]

#: How far apart :func:`solve_along` reads the measure, in metres of ``s``,
#: before it narrows a crossing down.
SCAN_STEP_M = 0.5
#: How close to the target :func:`solve_along` settles, in the measure's unit.
TOLERANCE = 1e-3
#: How far either side of where the lane's length puts it
#: :func:`place_at_stop_line_distance` looks for the ego's place, in metres.
_STOP_LINE_SEARCH_M = 5.0
#: How far into its own lanelet :func:`place_at_stop_line_distance` looks at a
#: spawn the measure places on the lanelet next to it, in metres.
_NUDGE_M = 0.5
#: The fewest metres of ``s`` :func:`solve_along` narrows a crossing to.
_MIN_INTERVAL_M = 1e-7
#: How many times the tolerance the best point of a narrowed crossing may miss
#: the target by and still be taken: one that misses by more sat on a jump of
#: the measure, not on a crossing (:func:`_narrow`).
_LOOSE_TOLERANCE_FACTOR = 10.0


def _narrow(
    read_at: Callable[[float], Optional[float]],
    target: float,
    low: float,
    low_value: float,
    high: float,
    tolerance: float,
) -> Optional[float]:
    """Bisect ``[low, high]``, across which ``read_at`` crosses *target*.

    ``None`` when a point inside reads nothing, or when the bisection ends on
    no point that reads within :data:`_LOOSE_TOLERANCE_FACTOR` times
    *tolerance* of *target*: the measure jumps over the target there (the
    vehicle ahead changes, a stop line is passed) rather than crossing it,
    and :func:`solve_along` keeps scanning for a real crossing.
    """
    below = low_value < target
    best = low
    best_error = abs(low_value - target)
    while high - low > _MIN_INTERVAL_M:
        mid = 0.5 * (low + high)
        value = read_at(mid)
        if value is None:
            return None
        error = abs(value - target)
        if error < best_error:
            best, best_error = mid, error
        if error <= tolerance:
            return mid
        if (value < target) == below:
            low = mid
        else:
            high = mid
    if best_error > _LOOSE_TOLERANCE_FACTOR * tolerance:
        return None
    return best


def solve_along(
    read_at: Callable[[float], Optional[float]],
    target: float,
    low: float,
    high: float,
    *,
    step: float = SCAN_STEP_M,
    tolerance: float = TOLERANCE,
    what: str = "the measure",
) -> float:
    """The smallest ``s`` in ``[low, high]`` at which ``read_at(s)`` is *target*.

    ``read_at`` reads the measure for a road user placed at ``s`` -- along a
    lanelet, say -- and ``None`` where the measure would not see it.  It is
    read every *step* from *low* to *high*; the first two neighbouring reads
    on either side of *target* are narrowed down by bisection until it reads
    within *tolerance* of it.  The measure needs only to be continuous in
    ``s``, not monotonic: on a curve the first crossing is the one nearest
    *low*.

    Raises:
        ValueError: When no ``s`` in ``[low, high]`` reads *target*.  The
            message names *what* and the values it does read there.
    """
    if step <= 0.0:
        raise ValueError(f"solve_along(): step must be positive, got {step}")
    if high < low:
        raise ValueError(f"solve_along(): empty interval [{low}, {high}]")
    previous_s = low
    previous: Optional[float] = None
    lowest = float("inf")
    highest = float("-inf")
    # The first pair of reads that straddle the target with no crossing
    # between them, for the error: "reads 5 to 50" alone does not say why 20
    # was not found.
    jumped = False
    jump_low = 0.0
    jump_high = 0.0
    count = int((high - low) / step)
    for i in range(count + 2):
        s = min(low + i * step, high)
        if i > 0 and s <= previous_s:
            break
        value = read_at(s)
        if value is not None:
            lowest = min(lowest, value)
            highest = max(highest, value)
            if abs(value - target) <= tolerance:
                return s
            if previous is not None and (previous < target) != (value < target):
                found = _narrow(read_at, target, previous_s, previous, s, tolerance)
                if found is not None:
                    return found
                if not jumped:
                    jumped, jump_low, jump_high = True, previous_s, s
        previous_s, previous = s, value
    if lowest > highest:
        raise ValueError(
            f"{what} cannot read {target:g}: it reads nothing for s in "
            f"[{low:g}, {high:g}]"
        )
    passes = ""
    if jumped:
        passes = (
            f", passing {target:g} only where it jumps or reads nothing "
            f"(first between s = {jump_low:.2f} and {jump_high:.2f})"
        )
    raise ValueError(
        f"{what} cannot read {target:g}: for s in [{low:g}, {high:g}] it reads "
        f"{lowest:.2f} to {highest:.2f}{passes}"
    )


#: Where :func:`vehicle_bumpers` spawns it: this high above the map's origin,
#: clear of everything on the road, and higher by as much again if that fails.
_PROBE_HEIGHT_M = 500.0
#: Blueprint id -> its bumpers, as measured.
_BUMPERS: dict[str, Bumpers] = {}


def _destroy_probes(world: carla.World, probes: list[carla.Actor], what: str) -> None:
    """Tick *world* once, then destroy every one of *probes*.

    An actor the world has not ticked since it was spawned is one CARLA's
    client does not know yet, and ``destroy()`` does nothing to it
    (:func:`vehicle_bumpers`), so the world is ticked first: this client's
    tick in synchronous mode, the server's waited for otherwise.  Every probe
    is destroyed, or tried, before a failure is reported.  No probes, no tick.

    Raises:
        RuntimeError: When a probe cannot be destroyed even after the tick;
            *what* names the caller.  It is then left in the world.
    """
    if len(probes) == 0:
        return
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()
    left: list[int] = []
    for probe in probes:
        if not probe.destroy():
            left.append(int(probe.id))
    if len(left) > 0:
        raise RuntimeError(
            f"{what}: the probe vehicle(s) {left} could not be destroyed; "
            "left in the world until the next run clears them"
        )


def _destroy_probe(world: carla.World, probe: carla.Actor) -> None:
    """Destroy *probe*, ticking the world once first if CARLA does not know
    it yet (:func:`vehicle_bumpers`, :func:`_destroy_probes`)."""
    if probe.destroy():
        return
    _destroy_probes(world, [probe], "vehicle_bumpers()")


def vehicle_bumpers(world: carla.World, vehicle_type: str) -> Bumpers:
    """The bumpers of a vehicle of blueprint *vehicle_type*, before one is
    spawned where it will drive.

    CARLA gives a blueprint no size: a vehicle's bounding box is known only
    once it is spawned.  A vehicle of the blueprint already in the world --
    an ego kept from an earlier scenario, a vehicle another one spawned -- is
    read as it is: its bounding box taken as is, scaled or modded as that
    actor may be, not the blueprint's stock one.  Otherwise one is spawned
    high above the map (tried at :data:`_PROBE_HEIGHT_M`, then twice and
    three times as high), read and destroyed straight away; the
    vehicle-ahead measures do not count it, by its role name
    (:data:`~autoware_carla_scenario.constants.BUMPER_PROBE_ROLE_NAME`) or by
    its height.  An actor the world has not ticked since it was spawned is
    one CARLA's client does not know yet, and ``destroy()`` does nothing to it
    ("attempting to destroy an actor that is already dead"): the probe would
    stay, falling from on high, for every vehicle loop of the run to find;
    once the TrafficManager is given it, the next synchronous tick never
    returns.  So when ``destroy()`` says it did nothing, the world is ticked
    once (waited for, off synchronous mode) and the probe destroyed then.  In
    synchronous mode that tick is this client's, whoever else ticks the world:
    call it where the caller owns the ticking, as a scenario's ``setup()``
    does.
    The result is remembered by blueprint id for the rest of the process.
    Placing a vehicle for a bumper-to-bumper gap needs its size where it is
    placed, and this way it is placed once, where the gap reads right, rather
    than spawned and then moved.

    Raises:
        ValueError: When the blueprint is not a vehicle CARLA has, or no
            vehicle of it can be spawned to measure.
        RuntimeError: When the probe cannot be destroyed even after a tick;
            it is then left in the world.
    """
    if vehicle_type in _BUMPERS:
        return _BUMPERS[vehicle_type]
    for existing in world.get_actors():
        if existing.type_id == vehicle_type:
            found = bumpers_of_box(existing.bounding_box)
            _BUMPERS[vehicle_type] = found
            return found
    library = world.get_blueprint_library()
    known = {bp.id for bp in library.filter("vehicle.*")}
    if vehicle_type not in known:
        raise ValueError(
            f"vehicle_bumpers(): {vehicle_type!r} is not a vehicle blueprint"
        )
    blueprint = library.find(vehicle_type)
    blueprint.set_attribute("role_name", BUMPER_PROBE_ROLE_NAME)
    for attempt in range(1, 4):
        at = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=_PROBE_HEIGHT_M * attempt),
            carla.Rotation(),
        )
        actor = world.try_spawn_actor(blueprint, at)
        if actor is None:
            continue
        spawned: carla.Actor = actor
        try:
            bumpers = bumpers_of_box(spawned.bounding_box)
        except Exception:
            # The error being raised is the one to report; the probe is
            # destroyed as far as it can be.
            try:
                _destroy_probe(world, spawned)
            except Exception:
                pass
            raise
        _destroy_probe(world, spawned)
        _BUMPERS[vehicle_type] = bumpers
        return bumpers
    raise ValueError(
        f"vehicle_bumpers(): no {vehicle_type} could be spawned to measure"
    )


def place_at_vehicle_ahead_gap(
    ego: carla.Transform,
    lanelet_id: int,
    gap_m: float,
    *,
    ego_bumpers: Bumpers,
    npc_bumpers: Bumpers,
) -> Lanelet2Pose:
    """The pose on *lanelet_id* at which a vehicle is *gap_m* ahead of the ego.

    "Ahead" is what :data:`~autoware_carla_scenario.measures.VEHICLE_AHEAD_GAP_M`
    reads, by the same code
    (:func:`~autoware_carla_scenario.measures.vehicle_ahead_gap_between`):
    bumper to bumper along the ego's lane on the Lanelet2 map, from an ego at
    *ego* (its spawn transform) with *ego_bumpers*, to a vehicle with
    *npc_bumpers* on the centreline of *lanelet_id*, heading along it, as a
    :class:`~autoware_carla_scenario.Lanelet2Pose` spawns
    (:func:`~autoware_carla_scenario.snap_to_carla_road` takes x, y and the
    heading from the lanelet).  The NPC's ``s`` is solved for on its own
    lanelet, reading the gap where it would stand, so the measure reads
    *gap_m* at the start however the two lanelets are cut, how long they are
    and how they curve.  The bumpers come from :func:`vehicle_bumpers`.

    Raises:
        ValueError: When no point of the lanelet is *gap_m* ahead -- it is too
            short, turns away, or is not on the ego's lane or beside it --
            saying what gaps it does reach.
    """
    length = lanelet_length(lanelet_id)

    def read_at(s: float) -> Optional[float]:
        at = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
        other = carla.Transform(
            carla.Location(x=at.x, y=at.y, z=at.z), carla.Rotation(yaw=at.yaw)
        )
        return vehicle_ahead_gap_between(ego, ego_bumpers, other, npc_bumpers)

    s = solve_along(
        read_at,
        gap_m,
        0.0,
        length,
        what=f"{VEHICLE_AHEAD_GAP_M} on lanelet {lanelet_id} (length {length:.1f} m)",
    )
    return Lanelet2Pose(lanelet_id=lanelet_id, s=s)


def _lanelets_before(lanelet_id: int, reach_m: float) -> list[int]:
    """The lanelets before *lanelet_id*, nearest first, until *reach_m*
    metres of them: the first of the routing graph's ``previous`` of each,
    by the lowest id, so the same map always gives the same lane."""
    mm = MapManager.get_instance()
    layer = mm.lanelet_map.laneletLayer
    graph = mm.routing_graph
    behind: list[int] = []
    current_id = lanelet_id
    walked = 0.0
    while walked < reach_m:
        before = sorted(int(ll.id) for ll in graph.previous(layer[current_id]))
        if not before or before[0] in behind or before[0] == lanelet_id:
            break
        current_id = before[0]
        behind.append(current_id)
        walked += lanelet_length(current_id)
    return behind


def _pose_on(
    lane: list[int], starts: list[float], lengths: list[float], u: float
) -> Lanelet2Pose:
    """The pose *u* metres along *lane*, whose lanelets start at *starts*."""
    i = len(lane) - 1
    while i > 0 and u < starts[i]:
        i -= 1
    return Lanelet2Pose(lanelet_id=lane[i], s=min(max(u - starts[i], 0.0), lengths[i]))


def place_at_stop_line_distance(
    spawn: Lanelet2Pose, distance_m: float, *, ego_bumpers: Bumpers
) -> Lanelet2Pose:
    """The pose, on the lane through *spawn*, at which an ego with
    *ego_bumpers* is *distance_m* before the stop line ahead.

    "Before" is what :data:`~autoware_carla_scenario.measures.STOP_LINE_DISTANCE_M`
    reads, by the same code
    (:func:`~autoware_carla_scenario.measures.stop_line_distance_between`):
    along the ego's lane from its front bumper.  The ego is moved along the
    lane *spawn* is on: on towards the stop line the measure reads from
    *spawn*, along the branch it is on where the lane forks
    (:func:`~autoware_carla_scenario.measures.stop_line_lane`), or back onto
    the lanelets before *spawn*'s, up to
    :data:`~autoware_carla_scenario.measures.AHEAD_RANGE_M`.  So *spawn* need
    only be somewhere within that range before the stop line.

    Raises:
        ValueError: When no point of that lane is *distance_m* before a stop
            line, saying what distances it does reach.
    """
    ahead: list[int] = []
    # Where the spawn lanelet meets the next one, or a branch overlaps its
    # sibling, the lane the measure finds may start on the other lanelet: the
    # spawn is looked at a little way into its own lanelet too.
    length = lanelet_length(spawn.lanelet_id)
    for s in (spawn.s, min(spawn.s + _NUDGE_M, length), max(spawn.s - _NUDGE_M, 0.0)):
        at = to_carla_world(Lanelet2Pose(lanelet_id=spawn.lanelet_id, s=s))
        ahead = stop_line_lane(
            carla.Transform(
                carla.Location(x=at.x, y=at.y, z=at.z), carla.Rotation(yaw=at.yaw)
            ),
            ego_bumpers,
        )
        if spawn.lanelet_id in ahead:
            break
    if not ahead:
        raise ValueError(
            f"{STOP_LINE_DISTANCE_M}: no stop line within {AHEAD_RANGE_M:g} m "
            f"ahead of lanelet {spawn.lanelet_id} at s={spawn.s:.2f}"
        )
    if spawn.lanelet_id not in ahead:
        raise ValueError(
            f"{STOP_LINE_DISTANCE_M}: the stop line ahead of lanelet "
            f"{spawn.lanelet_id} at s={spawn.s:.2f} is read along lanelets "
            f"{ahead}, a lane the spawn's lanelet is not on"
        )
    # The lane the measure reads the stop line along -- its branch, where the
    # lane forks -- from the spawn's lanelet, and the lane back from that.
    ahead = ahead[ahead.index(spawn.lanelet_id) :]
    lane = [*reversed(_lanelets_before(spawn.lanelet_id, AHEAD_RANGE_M)), *ahead]
    lengths = [lanelet_length(lanelet_id) for lanelet_id in lane]
    starts: list[float] = []
    total = 0.0
    for length in lengths:
        starts.append(total)
        total += length

    def read_at(u: float) -> Optional[float]:
        at = to_carla_world(_pose_on(lane, starts, lengths, u))
        ego = carla.Transform(
            carla.Location(x=at.x, y=at.y, z=at.z), carla.Rotation(yaw=at.yaw)
        )
        return stop_line_distance_between(ego, ego_bumpers)

    # The stop line is the one ahead of *spawn*; along the lane the distance
    # falls metre for metre, so the answer is near where that says, and is
    # looked for only there -- not before a stop line further back.
    u_spawn = starts[lane.index(spawn.lanelet_id)] + spawn.s
    gap = read_at(u_spawn)
    if gap is None:
        raise ValueError(
            f"{STOP_LINE_DISTANCE_M}: no stop line within {AHEAD_RANGE_M:g} m "
            f"ahead of lanelet {spawn.lanelet_id} at s={spawn.s:.2f}"
        )
    guess = u_spawn + gap - distance_m
    low = max(0.0, guess - _STOP_LINE_SEARCH_M)
    high = min(total, guess + _STOP_LINE_SEARCH_M)
    if high < low:
        raise ValueError(
            f"{STOP_LINE_DISTANCE_M} cannot read {distance_m:g}: the lane back "
            f"from lanelet {spawn.lanelet_id} (lanelets {lane}) reaches only "
            f"{u_spawn + gap:.1f} m before the stop line"
        )
    u = solve_along(
        read_at,
        distance_m,
        low,
        high,
        what=(
            f"{STOP_LINE_DISTANCE_M} on the lane through lanelet "
            f"{spawn.lanelet_id} (lanelets {lane})"
        ),
    )
    return _pose_on(lane, starts, lengths, u)
