"""Place a road user so a scenario measure reads a given value at the start.

A scenario parameter usually is a raw setting -- "the NPC starts 10 m further
along its lanelet than the ego along its own" -- and an ODD-sampled sweep
writes the value it drew for a measure into it through a control's ``scale``
and ``offset``.  Where the raw setting and the measure are not the same
quantity that mapping is a guess: the ego's and the NPC's lanelets do not run
alike on a curve (the outer one is longer), so the arc length along one is
not the gap :data:`~autoware_carla_scenario.measures.VEHICLE_AHEAD_GAP_M`
reads, and no constant offset is right everywhere.

A measure that the initial placement alone determines can instead be the
parameter: the scenario takes "the measure should read *v* at the start"
and solves for the placement, evaluating the measure's own definition on the
poses it would spawn at.  :func:`solve_along` is the solver, a 1-D root find
along a lanelet; :func:`place_at_vehicle_ahead_gap` applies it to the gap to
a vehicle ahead::

    ego = self.ego_config.spawn_location.value      # after _setup_ego_spawn()
    npc_pose = place_at_vehicle_ahead_gap(ego, cfg.npc_lanelet_id, cfg.npc_gap_m)

A control that writes such a parameter declares ``target: measure``
(:mod:`autoware_carla_scenario.odd.sampler`, docs/odd.md, "Controls").
"""

from __future__ import annotations

from typing import Callable, Optional

import typesafe_carla.carla as carla

from .coordinate.poses import Lanelet2Pose
from .coordinate.transform import lanelet_length, to_carla_world
from .measures import VEHICLE_AHEAD_GAP_M, vehicle_ahead_gap_between

__all__ = ["place_at_vehicle_ahead_gap", "solve_along"]

#: How far apart :func:`solve_along` reads the measure, in metres of ``s``,
#: before it narrows a crossing down.
SCAN_STEP_M = 0.5
#: How close to the target :func:`solve_along` settles, in the measure's unit.
TOLERANCE = 1e-3
#: The fewest metres of ``s`` :func:`solve_along` narrows a crossing to.
_MIN_INTERVAL_M = 1e-7


def _narrow(
    read_at: Callable[[float], Optional[float]],
    target: float,
    low: float,
    low_value: float,
    high: float,
    tolerance: float,
) -> Optional[float]:
    """Bisect ``[low, high]``, across which ``read_at`` crosses *target*.

    ``None`` when a point inside reads nothing.
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
        previous_s, previous = s, value
    if lowest > highest:
        raise ValueError(
            f"{what} cannot read {target:g}: it reads nothing for s in "
            f"[{low:g}, {high:g}]"
        )
    raise ValueError(
        f"{what} cannot read {target:g}: for s in [{low:g}, {high:g}] it reads "
        f"{lowest:.2f} to {highest:.2f}"
    )


def place_at_vehicle_ahead_gap(
    ego: carla.Transform, lanelet_id: int, gap_m: float
) -> Lanelet2Pose:
    """The pose on *lanelet_id* at which a vehicle is *gap_m* ahead of the ego.

    "Ahead" is what :data:`~autoware_carla_scenario.measures.VEHICLE_AHEAD_GAP_M`
    reads (:func:`~autoware_carla_scenario.measures.vehicle_ahead_gap_between`):
    centre to centre along the ego's heading, from an ego at *ego* (its spawn
    transform) to a vehicle on the centreline of *lanelet_id*, placed as a
    :class:`~autoware_carla_scenario.Lanelet2Pose` spawns
    (:func:`~autoware_carla_scenario.snap_to_carla_road` takes x and y from
    the lanelet).  So the measure reads *gap_m* at the start, wherever the two
    lanelets are and however differently they curve.

    Raises:
        ValueError: When no point of the lanelet is *gap_m* ahead -- it is too
            short, or turns away -- saying what gaps it does reach.
    """
    length = lanelet_length(lanelet_id)

    def read_at(s: float) -> Optional[float]:
        at = to_carla_world(Lanelet2Pose(lanelet_id=lanelet_id, s=s))
        return vehicle_ahead_gap_between(ego, carla.Location(x=at.x, y=at.y, z=at.z))

    s = solve_along(
        read_at,
        gap_m,
        0.0,
        length,
        what=f"{VEHICLE_AHEAD_GAP_M} on lanelet {lanelet_id} (length {length:.1f} m)",
    )
    return Lanelet2Pose(lanelet_id=lanelet_id, s=s)
