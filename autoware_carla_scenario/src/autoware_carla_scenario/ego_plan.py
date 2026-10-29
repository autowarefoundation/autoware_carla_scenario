"""A driver's plan as the run records it: the trajectory, timed.

:func:`~autoware_carla_scenario.debug_markers.plan_markers` draws a plan as a
line, which is all a viewer needs to show it and all the ``markers`` record
keeps.  Reading how the plan behaves -- whether the next plan agrees with
this one about the same moment ahead, how fast the steering it implies moves,
when the policy changed its mind -- needs more: when each point is planned
for, the heading there, and what the policy said with it.  :class:`EgoPlan`
holds that, and :class:`PlanMarkers` -- the plan's markers, carrying it --
brings it to the trajectory log the way the markers come, so the run's own
loop needs nothing new: a ``plan`` record is written beside the ``markers``
record of every plan (docs/trajectory_log.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .debug_markers import DebugMarkers, plan_polyline, world_point

if TYPE_CHECKING:
    from .driver.geometry import Trajectory

__all__ = ["EgoPlan", "PlanMarkers", "PlanPoint", "ego_plan_markers"]

#: ``(dt, x, y, z, yaw)``: seconds after the plan was made, CARLA world
#: position in metres, and CARLA yaw in degrees.
PlanPoint = Tuple[float, float, float, float, float]

_US_PER_S = 1_000_000.0


@dataclass(frozen=True)
class EgoPlan:
    """One plan of a driver, in CARLA's world frame.

    Attributes:
        step: Which plan of the session it is, from 1.
        planned_at_us: When it was made, on the driver's clock (a driver
            policy's: microseconds since its session began; an Autoware ego's:
            its command's stamp).  Plans of one run share the clock, so two
            plans are ``planned_at_us`` apart.
        points: The planned poses of the rig's origin (``PlanPoint``), in
            order.  ``dt`` is the point's timestamp less ``planned_at_us``.
        policy_name: The policy, when its answer said.
        inference_seconds: How long the policy took, when it said.
        turn_indicator: The turn indicators it commanded with the plan
            (``DriveOutcome.turn_indicator``: 0 no command, 1 off, 2 left,
            3 right).
        debug_scalars: The diagnostics it surfaced with the plan.
    """

    step: int
    planned_at_us: int
    points: List[PlanPoint]
    policy_name: str
    inference_seconds: float
    turn_indicator: int
    debug_scalars: Dict[str, float]


class PlanMarkers(DebugMarkers):
    """A plan's debug markers, carrying the plan they draw (:attr:`plan`)."""

    plan: EgoPlan

    def __init__(self, plan: EgoPlan) -> None:
        super().__init__()
        self.plan = plan

    def __bool__(self) -> bool:
        # A plan is recorded even with nothing to draw (an empty trajectory):
        # its markers clear the drawing, and its turn indicator command and
        # diagnostics still count.
        return True


def ego_plan_markers(
    trajectory: "Trajectory",
    planned_at_us: int,
    step: int,
    *,
    policy_name: str = "",
    inference_seconds: float = 0.0,
    turn_indicator: int = 0,
    debug_scalars: Optional[Dict[str, float]] = None,
) -> PlanMarkers:
    """The markers of a driver's *trajectory* (in the right-handed local frame
    it plans in), its *step*-th plan, made at *planned_at_us* on its clock,
    with what the driver said with it.

    The markers draw it as
    :func:`~autoware_carla_scenario.debug_markers.plan_markers` does; the plan
    keeps each point's time and heading in CARLA's world (the yaw flipped
    back with ``y``).  A point with a coordinate that is not a finite number
    is left out of both: neither can be followed nor written as JSON.
    """
    points: List[PlanPoint] = []
    for timestamp_us, pose in zip(trajectory.timestamps_us, trajectory.poses):
        x, y, z = world_point(pose)
        point = (
            (timestamp_us - planned_at_us) / _US_PER_S,
            x,
            y,
            z,
            -math.degrees(pose.yaw),
        )
        if all(math.isfinite(v) for v in point):
            points.append(point)
    markers = PlanMarkers(
        EgoPlan(
            step=step,
            planned_at_us=planned_at_us,
            points=points,
            policy_name=policy_name,
            inference_seconds=inference_seconds,
            turn_indicator=turn_indicator,
            debug_scalars={} if debug_scalars is None else dict(debug_scalars),
        )
    )
    markers.add_polyline(plan_polyline([(x, y, z) for _, x, y, z, _ in points]))
    return markers
