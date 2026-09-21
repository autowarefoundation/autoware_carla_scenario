"""How far apart two actors are, and what "apart" is being asked about.

Three measurements share this module because the answer to "are they 20 m
apart?" depends on two choices that a bare Euclidean distance quietly makes for
the author:

* **along which axis** -- the straight line between the two, or the gap along
  the road, or the offset across it;
* **between which points** -- centre to centre, or bumper to bumper.

Both are load-bearing.  A car in the next lane is not "20 m away" in any sense
a following-distance scenario means, and centre-to-centre is wrong by about a
vehicle length at exactly the ranges a close-quarters scenario is written
about.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Optional

from ...kinematics import Vector3
from .base import entity_axes

if TYPE_CHECKING:
    import carla

__all__ = ["RelativeDistanceType", "separation"]


class RelativeDistanceType(enum.Enum):
    """Which component of the separation is being measured.

    Mirrors OpenSCENARIO's ``RelativeDistanceType`` for the three values that
    have a meaning without a road reference.  ``LONGITUDINAL`` and ``LATERAL``
    are taken in the **source** entity's frame -- the entity coordinate system,
    in OpenSCENARIO's terms.  Its ``lane`` coordinate system is not offered:
    that needs the lanelet routing graph, and approximating it with the
    source's heading would be right on a straight road and wrong exactly where
    it matters.

    Every value is a *distance*: non-negative, whichever side the target is on.
    A signed reading would make ``< 20`` true for a vehicle 30 m behind, which
    is not what a following-distance rule is asking.

    Attributes:
        EUCLIDEAN: Straight-line distance.
        LONGITUDINAL: Distance along the source's heading.
        LATERAL: Distance across it.
    """

    EUCLIDEAN = "euclidean"
    LONGITUDINAL = "longitudinal"
    LATERAL = "lateral"


def _half_extent_along(actor: "carla.Actor", direction: Vector3) -> float:
    """Return how far *actor*'s bounding box reaches along *direction*.

    The support function of an oriented box: the box's half-extents projected
    onto the direction, summed.  Using the oriented box rather than a radius
    matters because a car is three times longer than it is wide, so a circle
    around it is wrong by a metre in one axis or the other whichever radius is
    picked.

    An actor with no bounding box contributes nothing, which makes freespace
    degrade to centre-to-centre for that actor rather than fail.
    """
    box = getattr(actor, "bounding_box", None)
    if box is None:
        return 0.0
    axes = entity_axes(actor)
    if axes is None:
        return 0.0
    forward, left = axes
    extent = box.extent
    return abs(direction.dot(forward)) * extent.x + abs(direction.dot(left)) * extent.y


def separation(
    source: "carla.Actor",
    target: "carla.Actor",
    *,
    distance_type: RelativeDistanceType = RelativeDistanceType.EUCLIDEAN,
    freespace: bool = False,
    vertical: bool = False,
) -> Optional[float]:
    """Return the distance from *source* to *target*, or ``None`` if unknowable.

    Args:
        source: The actor measured from; its heading defines the axes for a
            directional *distance_type*.
        target: The actor measured to.
        distance_type: Which component to measure.
        freespace: Measure between bounding boxes rather than between centres,
            clamped at zero once they overlap.
        vertical: Include the height difference.  Only meaningful for
            :attr:`RelativeDistanceType.EUCLIDEAN`; the directional components
            are ground-plane by construction.

    Returns:
        The distance in metres, or ``None`` when a directional component was
        asked for and the source's heading is degenerate -- which is "cannot
        tell" and must not be reported as a zero component.
    """
    src = source.get_location()
    tgt = target.get_location()
    delta = Vector3(
        tgt.x - src.x,
        tgt.y - src.y,
        (tgt.z - src.z) if vertical else 0.0,
    )

    if distance_type is RelativeDistanceType.EUCLIDEAN:
        distance = delta.magnitude()
        if not freespace:
            return distance
        if distance == 0.0:
            return 0.0
        direction = delta / distance
        return max(
            0.0,
            distance
            - _half_extent_along(source, direction)
            - _half_extent_along(target, direction),
        )

    axes = entity_axes(source)
    if axes is None:
        return None
    forward, left = axes
    axis = forward if distance_type is RelativeDistanceType.LONGITUDINAL else left

    distance = abs(delta.dot(axis))
    if not freespace:
        return distance
    return max(
        0.0,
        distance - _half_extent_along(source, axis) - _half_extent_along(target, axis),
    )
