"""The vehicle blueprint nearest a size, chosen on the server the run is on.

An entity with ``vehicle_type: auto`` and a ``size`` (a recorded box, say) is
played by the vehicle of the running CARLA whose bounding box is nearest that
size (:func:`fitted_vehicle`).  A blueprint's box is not among its attributes
and an actor cannot be scaled, so each candidate is spawned once, out of
sight, to read its box, then destroyed after one tick; the boxes are kept
for the world (:func:`vehicle_sizes`).  The choice is made against whatever the server
has -- its version, its assets -- rather than a table written for one.

Candidates are the blueprints whose ``base_type`` is a car, a van, a truck
or a bus (no bicycles or motorcycles), less :data:`FIT_EXCLUDED`: the
emergency vehicles, which CARLA 0.10 marks with no ``special_type``.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

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


def _candidate(blueprint: carla.ActorBlueprint) -> bool:
    if any(fnmatch.fnmatch(blueprint.id, pattern) for pattern in FIT_EXCLUDED):
        return False
    if not blueprint.has_attribute("base_type"):
        return True  # an older server: no base type to go by
    return blueprint.get_attribute("base_type").as_str() in FIT_BASE_TYPES


def vehicle_sizes(world: carla.World) -> dict[str, tuple[float, float]]:
    """``{blueprint: (length, width)}`` (m) of every candidate vehicle of
    *world*, read once per world by spawning each out of sight.

    Every candidate is spawned and its box read, then the world is ticked
    once and every probe destroyed: CARLA's client does not know an actor
    that has not been ticked since it was spawned, and ``destroy()`` does
    nothing to it (``placement._destroy_probes``, #231).  A probe left in the
    world is a vehicle the TrafficManager is later given, and the next
    synchronous tick never returns (#940).  The probes carry
    :data:`~autoware_carla_scenario.constants.BUMPER_PROBE_ROLE_NAME`, which
    the vehicle-ahead measures skip, as they skip a vehicle that far below
    the ego.  In synchronous mode the tick is this client's: call this where
    the caller owns the ticking, as a scenario's ``setup()`` does.

    Raises:
        RuntimeError: When a probe cannot be destroyed even after the tick.
    """
    key = int(world.id)
    known = _SIZES.get(key)
    if known is not None:
        return known
    import typesafe_carla.carla as carla  # noqa: PLC0415

    from ..constants import BUMPER_PROBE_ROLE_NAME  # noqa: PLC0415
    from ..placement import _destroy_probes  # noqa: PLC0415

    sizes: dict[str, tuple[float, float]] = {}
    probes: list[carla.Actor] = []
    x, y, z = _OUT_OF_SIGHT
    try:
        for n, blueprint in enumerate(
            world.get_blueprint_library().filter("vehicle.*")
        ):
            if not _candidate(blueprint):
                continue
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", BUMPER_PROBE_ROLE_NAME)
            spot = carla.Transform(carla.Location(x=x + 20.0 * n, y=y, z=z))
            actor = world.try_spawn_actor(blueprint, spot)
            if actor is None:
                logger.warning(
                    "vehicle sizes: could not spawn %s to read its box", blueprint.id
                )
                continue
            probe: carla.Actor = actor
            probes.append(probe)
            extent = probe.bounding_box.extent
            sizes[blueprint.id] = (2.0 * float(extent.x), 2.0 * float(extent.y))
    except Exception:
        # The error being raised is the one to report; the probes spawned so
        # far are destroyed as far as they can be.
        try:
            _destroy_probes(world, probes, "vehicle_sizes()")
        except Exception:
            pass
        raise
    # Kept before the probes go: a probe left behind fails this setup, and
    # the next one does not spawn them all again.
    _SIZES[key] = sizes
    logger.info("vehicle sizes: read the boxes of %d vehicles", len(sizes))
    _destroy_probes(world, probes, "vehicle_sizes()")
    return sizes


def fitted_vehicle(world: carla.World, length: float, width: float) -> Optional[str]:
    """The candidate vehicle of *world* whose box is nearest *length* x
    *width* (m): the least sum of the squared relative differences of the
    two.  ``None`` when *world* has none, or the size is not one."""
    if length <= 0.0 or width <= 0.0:
        return None
    sizes = vehicle_sizes(world)
    if len(sizes) == 0:
        return None

    def gap(name: str) -> float:
        size_l, size_w = sizes[name]
        return ((size_l - length) / length) ** 2 + ((size_w - width) / width) ** 2

    return min(sorted(sizes), key=gap)
