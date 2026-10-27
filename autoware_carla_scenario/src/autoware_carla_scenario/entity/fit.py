"""The vehicle blueprint nearest a size, chosen on the server the run is on.

An entity with ``vehicle_type: auto`` and a ``size`` (a recorded box, say) is
played by the vehicle of the running CARLA whose bounding box is nearest that
size (:func:`fitted_vehicle`).  A blueprint's box is not among its attributes
and an actor cannot be scaled, so each candidate is spawned once, out of
sight, to read its box, then destroyed; the boxes are kept for the world
(:func:`vehicle_sizes`).  The choice is made against whatever the server
has -- its version, its assets -- rather than a table written for one.

Candidates are the blueprints whose ``base_type`` is a car, a van, a truck
or a bus (no bicycles or motorcycles), less :data:`FIT_EXCLUDED`: the
emergency vehicles, which CARLA 0.10 marks with no ``special_type``.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Optional

__all__ = ["FIT_BASE_TYPES", "FIT_EXCLUDED", "fitted_vehicle", "vehicle_sizes"]

logger = logging.getLogger(__name__)

#: The blueprints' ``base_type`` a fitted vehicle may have.
FIT_BASE_TYPES = ("car", "van", "truck", "bus")
#: Blueprint ids (glob patterns) never chosen: emergency vehicles.
FIT_EXCLUDED = ("*firetruck*", "*ambulance*", "*cop*", "*police*")
#: Where a candidate is spawned to read its box: far below any map.
_OUT_OF_SIGHT = (0.0, 0.0, -2000.0)

#: The boxes read, by the world's id: ``{blueprint: (length, width)}``.
_SIZES: dict[int, dict[str, tuple[float, float]]] = {}


def _candidate(blueprint: Any) -> bool:
    if any(fnmatch.fnmatch(blueprint.id, pattern) for pattern in FIT_EXCLUDED):
        return False
    if not blueprint.has_attribute("base_type"):
        return True  # an older server: no base type to go by
    return blueprint.get_attribute("base_type").as_str() in FIT_BASE_TYPES


def vehicle_sizes(world: Any) -> dict[str, tuple[float, float]]:
    """``{blueprint: (length, width)}`` (m) of every candidate vehicle of
    *world*, read once per world by spawning each out of sight."""
    key = int(world.id)
    known = _SIZES.get(key)
    if known is not None:
        return known
    import typesafe_carla.carla as carla  # noqa: PLC0415

    sizes: dict[str, tuple[float, float]] = {}
    x, y, z = _OUT_OF_SIGHT
    for n, blueprint in enumerate(world.get_blueprint_library().filter("vehicle.*")):
        if not _candidate(blueprint):
            continue
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "size_probe")
        spot = carla.Transform(carla.Location(x=x + 20.0 * n, y=y, z=z))
        actor = world.try_spawn_actor(blueprint, spot)
        if actor is None:
            logger.warning(
                "vehicle sizes: could not spawn %s to read its box", blueprint.id
            )
            continue
        try:
            extent = actor.bounding_box.extent
            sizes[blueprint.id] = (2.0 * float(extent.x), 2.0 * float(extent.y))
        finally:
            actor.destroy()
    logger.info("vehicle sizes: read the boxes of %d vehicles", len(sizes))
    _SIZES[key] = sizes
    return sizes


def fitted_vehicle(world: Any, length: float, width: float) -> Optional[str]:
    """The candidate vehicle of *world* whose box is nearest *length* x
    *width* (m): the least sum of the squared relative differences of the
    two.  ``None`` when *world* has none, or the size is not one."""
    if length <= 0.0 or width <= 0.0:
        return None
    sizes = vehicle_sizes(world)
    if not sizes:
        return None

    def gap(name: str) -> float:
        size_l, size_w = sizes[name]
        return ((size_l - length) / length) ** 2 + ((size_w - width) / width) ** 2

    return min(sorted(sizes), key=gap)
