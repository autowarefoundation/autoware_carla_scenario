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
import math
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

    The box's own rotation is taken to be the actor's.  That holds for every
    vehicle and walker blueprint CARLA ships -- their boxes are axis-aligned
    with the actor -- and it is what lets the axes be read once, from the
    transform, rather than composed per box.

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
    return (
        abs(direction.dot(forward)) * extent.x
        + abs(direction.dot(left)) * extent.y
        + abs(direction.z) * extent.z
    )


def _box_centre(actor: "carla.Actor") -> Vector3:
    """Return the world-space centre of *actor*'s bounding box.

    A CARLA bounding box is positioned relative to the actor's origin rather
    than on it -- for a vehicle the box sits a little above and, on some
    models, slightly forward of the origin -- so measuring from the actor's
    location alone puts the box in the wrong place by that offset.
    """
    location = actor.get_location()
    centre = Vector3(location.x, location.y, location.z)
    box = getattr(actor, "bounding_box", None)
    axes = entity_axes(actor)
    if box is None or axes is None:
        return centre
    forward, left = axes
    offset = box.location
    # `left` is the negation of CARLA's own y, which is the axis a box offset
    # is written in, so the y term is subtracted rather than added.
    return Vector3(
        centre.x + forward.x * offset.x - left.x * offset.y,
        centre.y + forward.y * offset.x - left.y * offset.y,
        centre.z + offset.z,
    )


#: World axes, for the per-axis gap of the Euclidean measurement.
_WORLD_X = Vector3(1.0, 0.0, 0.0)
_WORLD_Y = Vector3(0.0, 1.0, 0.0)
_WORLD_Z = Vector3(0.0, 0.0, 1.0)


def _euclidean_gap(
    source: "carla.Actor",
    target: "carla.Actor",
    delta: Vector3,
    vertical: bool,
) -> float:
    """Return the closest distance between the two boxes.

    Measured per world axis and recombined, rather than by subtracting each
    box's reach along the centre-to-centre line.  The radial subtraction is
    wrong whenever the two are separated along more than one axis: for boxes
    of half-extent ``(2.5, 1.0)`` whose centres are ``(6, 6)`` apart it gives
    3.54 m where the true gap is 4.12 m, so a threshold between the two fires
    for a pair that is not that close.

    Each box is taken as its world-aligned enclosure, which is exact while the
    two actors are axis-aligned and conservative -- never reporting more room
    than there is -- when they are not.
    """
    axes = [(_WORLD_X, delta.x), (_WORLD_Y, delta.y)]
    if vertical:
        axes.append((_WORLD_Z, delta.z))

    total = 0.0
    for axis, offset in axes:
        reach = _half_extent_along(source, axis) + _half_extent_along(target, axis)
        gap = max(0.0, abs(offset) - reach)
        total += gap * gap
    return math.sqrt(total)


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
    if freespace:
        src = _box_centre(source)
        tgt = _box_centre(target)
    else:
        src_location = source.get_location()
        tgt_location = target.get_location()
        src = Vector3(src_location.x, src_location.y, src_location.z)
        tgt = Vector3(tgt_location.x, tgt_location.y, tgt_location.z)

    delta = Vector3(
        tgt.x - src.x,
        tgt.y - src.y,
        (tgt.z - src.z) if vertical else 0.0,
    )

    if distance_type is RelativeDistanceType.EUCLIDEAN:
        if not freespace:
            return delta.magnitude()
        return _euclidean_gap(source, target, delta, vertical)

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
