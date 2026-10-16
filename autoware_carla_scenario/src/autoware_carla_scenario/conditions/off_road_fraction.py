"""Fail condition on the fraction of an entity's box that is off the driving lanes."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import typesafe_carla.carla as carla

from ..constants import EGO_ROLE_NAME
from ..coordinate.map_manager import run_carla_map
from ..entity_role import EntityRole
from .base import BaseCondition, ScenarioResult, find_actor_by_role_name

__all__ = ["OffRoadFractionCondition"]

logger = logging.getLogger(__name__)

#: Sample points along the box's length (front to rear edge, both included).
_ALONG = 5
#: Sample points across the box's width (left to right edge, both included).
_ACROSS = 3


class OffRoadFractionCondition(BaseCondition):
    """Fail condition that triggers when enough of an entity's box is off the road.

    Every check samples a 5 x 3 grid of points over the entity's bounding
    box, seen from above: 5 evenly spaced along its length and 3 across its
    width, the box's edges and corners included (so 15 points: the 4
    corners, the middle of each edge, the centre and points in between).
    A point is *off* when no driving lane is under it -- when
    ``carla.Map.get_waypoint(point, project_to_road=False,
    lane_type=carla.LaneType.Driving)`` returns ``None``, queried at the
    height of the entity's transform, so that where roads cross at different
    heights the lane is the one the entity is on.  The condition fails once
    ``off / 15 >= fraction``.

    The measure is the fraction of sample points, not of the box's area: one
    corner off is 1/15, a whole side of the box (5 points) 1/3.

    This is the pass/fail counterpart of the ``offroad`` metric
    (:mod:`autoware_carla_scenario.metrics`, AlpaSim's), which is 1 as soon
    as any *one* corner of the ego's box is on no driving lane.  That is a
    measurement, and a strict one: a corner over a gap between lanes at a
    junction entry, or a small offset between the OpenDRIVE lanes and where
    the vehicle drives, counts though the vehicle is in its lane.  Judging a
    run by it fails such runs; this condition gives the judgement a
    tolerance instead.

    Args:
        fraction: The fraction of sample points, in ``(0, 1]``, that must be
            off the driving lanes for the condition to fail.  ``1/15`` (or
            anything up to it) fails on one point off, as the metric does on
            one corner; ``1.0`` only when the whole box is off.
        entity_name: The ``role_name`` of the actor to watch.  ``None`` (the
            default) watches the ego.
        label: Human-readable identifier for this condition.

    Raises:
        ValueError: If *fraction* is not in ``(0, 1]``.
    """

    # Declared for the static check (docs/typecheck.md).
    _fraction: float
    _entity_name: str
    _carla_map: Optional[carla.Map]
    _last_fraction: Optional[float]

    def __init__(
        self,
        fraction: float,
        entity_name: Union[EntityRole, str, None] = None,
        *,
        label: str,
    ) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        super().__init__(label=label)
        self._fraction = float(fraction)
        # Kept as its string, the one thing every use of it reads.
        self._entity_name = str(EGO_ROLE_NAME if entity_name is None else entity_name)
        self._carla_map: Optional[carla.Map] = None
        self._last_fraction: Optional[float] = None

    @property
    def off_road_fraction(self) -> Optional[float]:
        """The fraction of sample points off the driving lanes when last
        checked, or ``None`` before the entity was first found."""
        return self._last_fraction

    def get_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "entity_name": self._entity_name,
            "fraction": self._fraction,
        }
        if self._last_fraction is not None:
            details["off_road_fraction"] = self._last_fraction
        return details

    def _off_points(self, actor: carla.Actor, carla_map: carla.Map) -> int:
        """How many of the grid's points over *actor*'s box are on no driving lane."""
        transform = actor.get_transform()
        box = actor.bounding_box
        yaw = math.radians(transform.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        # CARLA's frame is left-handed: right of forward is (-sin, cos).
        right_x = -forward_y
        right_y = forward_x
        centre_x = (
            transform.location.x + box.location.x * forward_x + box.location.y * right_x
        )
        centre_y = (
            transform.location.y + box.location.x * forward_y + box.location.y * right_y
        )
        # The entity's height, so the lane found is the one it is on where
        # roads cross at different heights.
        z = transform.location.z
        driving = int(carla.LaneType.Driving)

        off = 0
        for i in range(_ALONG):
            along = box.extent.x * (2.0 * i / (_ALONG - 1) - 1.0)
            for j in range(_ACROSS):
                across = box.extent.y * (2.0 * j / (_ACROSS - 1) - 1.0)
                point = carla.Location(
                    x=centre_x + along * forward_x + across * right_x,
                    y=centre_y + along * forward_y + across * right_y,
                    z=z,
                )
                waypoint = carla_map.get_waypoint(
                    point, project_to_road=False, lane_type=driving
                )
                if waypoint is None:
                    off += 1
        return off

    def check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a fail result if enough of the entity's box is off the road.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=False`` when at least
            *fraction* of the sample points are off the driving lanes,
            ``None`` otherwise, and while the entity is not found.
        """
        actor = find_actor_by_role_name(world, self._entity_name)
        if actor is None:
            return None
        held = self._carla_map
        if held is None:
            # The run's: get_map() builds a whole new map on every call, and
            # one fetched here would stay with the scenario.
            held = run_carla_map(world)
            self._carla_map = held
        # Annotated, so Codon unwraps the Optional (docs/typecheck.md).
        carla_map: carla.Map = held

        off_fraction = self._off_points(actor, carla_map) / (_ALONG * _ACROSS)
        self._last_fraction = off_fraction
        if off_fraction < self._fraction:
            return None
        message = (
            f"{off_fraction:.0%} of the box of '{self._entity_name}' is off the "
            f"driving lanes (limit {self._fraction:.0%}) at {elapsed:.2f}s"
        )
        logger.info(message)
        return ScenarioResult(passed=False, message=message, elapsed_seconds=elapsed)
