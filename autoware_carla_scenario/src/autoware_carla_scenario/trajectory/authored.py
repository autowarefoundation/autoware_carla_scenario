"""The typed builders of a *Set Goal* card: goal, via lanelets, timing, path.

What :mod:`.authoring` builds from values already typed (``str``, numbers, a
sequence of lanelet ids): a ``goal`` card's goal (:func:`authored_goal`) and
via lanelets (:func:`authored_waypoints`), the time reference a card's
``time_domain`` names (:func:`authored_timing`), and a path along the
centrelines of a route of lanelets (``_lanelet_path``), with the card's
choices (:data:`PATH_SOURCES`, :data:`TIME_DOMAINS`).

Split out of ``trajectory.authoring``, which keeps the parsers of document
values of any type (YAML rows, editor fields) and re-exports these, so that
the library check compiles it (``docs/typecheck.md``, "Checking the framework
itself").  What needs the loaded map is imported inside the function that
uses it, so the editor process -- which never imports CARLA or lanelet2 --
can import this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

from .model import (
    ReferenceContext,
    Trajectory,
    TrajectoryTiming,
    TrajectoryVertex,
)

if TYPE_CHECKING:
    from ..coordinate.poses import Lanelet2Pose

__all__ = [
    "LANELET_PATH_STEP_M",
    "PATH_SOURCES",
    "TIME_DOMAINS",
    "authored_goal",
    "authored_timing",
    "authored_waypoints",
]

#: Where a document's route comes from: ``goal`` names a goal and via
#: lanelets; the others write the waypoints out as a trajectory.
PATH_SOURCES: tuple[str, ...] = (
    "goal",
    "vertices",
    "lanelets",
    "relative_lane",
    "route",
)

#: A document's time reference: ``none`` ignores the vertex times.
TIME_DOMAINS: tuple[str, ...] = ("none", "relative", "absolute")

#: How far apart (m) the vertices of a lanelet path are.
LANELET_PATH_STEP_M = 2.0


def _lanelet_path(
    name: str,
    lanelet_ids: list[int],
    speed_kmh: Optional[float],
    lateral_offset_m: float,
) -> Trajectory:
    """A path along the centrelines of *lanelet_ids*, timed at *speed_kmh*."""
    if not lanelet_ids:
        raise ValueError(f"{name}: a lanelet path needs at least one lanelet")
    from ..conditions.base import BaseCondition  # noqa: PLC0415
    from ..conditions.trajectory_time import (  # noqa: PLC0415
        TrajectoryTimeCondition,
    )
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415
    from ..coordinate.transform import lanelet_length  # noqa: PLC0415

    speed_ms = (speed_kmh or 0.0) / 3.6
    vertices: list[TrajectoryVertex] = []
    travelled = 0.0
    for position, lanelet_id in enumerate(lanelet_ids):
        length = lanelet_length(int(lanelet_id))
        steps = max(1, math.ceil(length / LANELET_PATH_STEP_M))
        # A lanelet's start is the previous one's end: taken once.
        first = 0 if position == 0 else 1
        for step in range(first, steps + 1):
            s = length * step / steps
            pose = Lanelet2Pose(lanelet_id=int(lanelet_id), s=s, t=lateral_offset_m)
            if speed_ms > 0.0:
                # A typed local: the vertex takes a BaseCondition, which Codon
                # does not upcast a TrajectoryTimeCondition to inside the
                # call, nor to an Optional of it.
                at: BaseCondition = TrajectoryTimeCondition((travelled + s) / speed_ms)
                vertices.append(TrajectoryVertex(pose, at))
            else:
                vertices.append(TrajectoryVertex(pose))
        travelled += length
    return Trajectory(name, vertices)


def authored_goal(
    path_source: str, lanelet_id: Optional[int] = None, s: Optional[float] = None
) -> "Optional[Lanelet2Pose]":
    """A ``goal`` card's goal, *s* metres along lanelet *lanelet_id*; ``None``
    for a card that writes its waypoints out."""
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    if path_source != "goal":
        return None
    return Lanelet2Pose(lanelet_id=int(lanelet_id or 0), s=float(s or 0.0))


def authored_waypoints(
    path_source: str, lanelet_ids: Optional[Sequence[int]] = None
) -> "list[Lanelet2Pose]":
    """A ``goal`` card's via lanelets, each a pose at its start; none for a
    card that writes its waypoints out."""
    from ..coordinate.poses import Lanelet2Pose  # noqa: PLC0415

    if path_source != "goal" or lanelet_ids is None:
        return []
    return [Lanelet2Pose(lanelet_id=int(i), s=0.0) for i in lanelet_ids]


def authored_timing(
    time_domain: str, scale: Optional[float] = 1.0, offset: Optional[float] = 0.0
) -> Optional[TrajectoryTiming]:
    """The time reference a card's ``time_domain`` names, or ``None`` for none."""
    if time_domain == "none":
        return None
    return TrajectoryTiming(
        domain=ReferenceContext(time_domain),
        scale=1.0 if scale is None else float(scale),
        offset=0.0 if offset is None else float(offset),
    )
