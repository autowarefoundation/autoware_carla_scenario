"""Temporary stop condition for detecting standstill at specified positions.

A stop position is matched in the frame it was given in.  One given as a
lanelet is a stretch of that lanelet, in its own ``s``; one given in OpenDRIVE
(or as a CARLA world position) is a stretch of the OpenDRIVE road, in the road's
``s``.  When the margin around it runs past the end of the lanelet or road,
additional :class:`EntityLanePositionCondition` instances are created for the
lanelets or roads before and after it and combined via :class:`OrCondition`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from ...coordinate.map_manager import MapManager
from ...coordinate.poses import AnyPose, Lanelet2Pose, OpenDrivePose
from ...coordinate.transform import lanelet_length, to_opendrive
from ..and_condition import AndCondition
from ..base import BaseCondition, ScenarioResult
from ..comparison import ComparisonRule, ScalarComparisonRule
from ...entity_role import EntityRole
from ..or_condition import OrCondition
from ..persistent import PersistentCondition
from .base import CompositionCondition
from .entity_lane_position import EntityLanePositionCondition
from .speed import SpeedCondition

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)


class TemporaryStopCondition(CompositionCondition):
    """Pass when entity temporarily stops at any of the given positions.

    For each stop position, constructs a composite condition:
    ``PersistentCondition(AndCondition(position_cond, SpeedCondition))``.

    When the margin around a stop position spans across OpenDRIVE road
    boundaries, multiple :class:`EntityLanePositionCondition` instances
    (one per road segment) are wrapped in an :class:`OrCondition`.

    When multiple stop positions are given, they are combined with
    :class:`OrCondition` — stopping at any one position is sufficient.

    An :class:`EntityExistenceCondition` guard ensures the entity is present
    before the inner conditions are evaluated.

    Args:
        entity_name: The ``role_name`` attribute of the actor to track.
        stop_positions: One or more poses where a stop is expected.  A
            :class:`Lanelet2Pose` is matched on its lanelet, in the lanelet's own
            ``s`` (from its start along its direction of travel); any other pose
            is converted to an :class:`OpenDrivePose` and matched on its road,
            in the road's ``s``, in any lane.
        s_margin: Arc-length margin (m) around each stop position.
            The entity must be within ``[s - s_margin, s + s_margin]``, in the
            frame of the position.
        speed_threshold: Maximum speed (m/s) considered as stopped.
        stop_duration: Minimum consecutive seconds the entity must remain
            stopped at the position.

    Raises:
        ValueError: If *stop_positions* is empty, *s_margin* is not positive,
            *speed_threshold* is negative, or *stop_duration* is not positive.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        stop_positions: Sequence[AnyPose],
        s_margin: float = 5.0,
        speed_threshold: float = 0.1,
        stop_duration: float = 1.0,
        *,
        label: str,
    ) -> None:
        if not stop_positions:
            raise ValueError("stop_positions must not be empty")
        if s_margin <= 0:
            raise ValueError("s_margin must be positive")
        if speed_threshold < 0:
            raise ValueError("speed_threshold must be non-negative")
        if stop_duration <= 0:
            raise ValueError("stop_duration must be positive")

        persistent_conditions: list[PersistentCondition] = []
        for pose in stop_positions:
            # In the frame the position was given in: a lanelet is a lane, with
            # an s of its own that runs the other way from the road's on every
            # lane against the road's reference line.
            if isinstance(pose, Lanelet2Pose):
                position_conds = self._build_lanelet_conditions(
                    entity_name, pose, s_margin, label=label
                )
            else:
                # Build position conditions spanning multiple roads if needed
                position_conds = self._build_position_conditions(
                    entity_name, to_opendrive(pose), s_margin, label=label
                )
            if len(position_conds) == 1:
                position_cond: BaseCondition = position_conds[0]
            else:
                position_cond = OrCondition(position_conds)

            speed_cond = SpeedCondition(
                entity_name=entity_name,
                value=speed_threshold,
                rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                label=f"{label}_speed",
            )
            and_cond = AndCondition([position_cond, speed_cond])
            persistent = PersistentCondition(and_cond, duration=stop_duration)
            persistent_conditions.append(persistent)

        if len(persistent_conditions) == 1:
            child: BaseCondition = persistent_conditions[0]
        else:
            child = OrCondition(persistent_conditions)

        super().__init__(child=child, entity_name=entity_name, label=label)

    # ------------------------------------------------------------------
    # Margin / road-boundary helpers
    # ------------------------------------------------------------------

    @classmethod
    def _build_position_conditions(
        cls,
        entity_name: Union[EntityRole, str],
        od_pose: OpenDrivePose,
        s_margin: float,
        *,
        label: str,
    ) -> list[EntityLanePositionCondition]:
        """Build EntityLanePositionConditions, splitting across roads when needed.

        If the margin range ``[s - s_margin, s + s_margin]`` stays within the
        road, a single condition is returned.  When the range overflows past
        ``s < 0`` (road start) or ``s > road_length`` (road end), additional
        conditions for predecessor / successor roads are appended.
        """
        road_length = cls._get_road_length(od_pose.road_id)
        s_min = od_pose.s - s_margin
        s_max = od_pose.s + s_margin

        conditions: list[EntityLanePositionCondition] = []

        # --- Main road (clamped to valid range) ---
        clamped_min = max(0.0, s_min)
        clamped_max = min(road_length, s_max)
        conditions.append(
            cls._make_road_segment_condition(
                entity_name, od_pose.road_id, clamped_min, clamped_max, label=label
            )
        )
        logger.info(
            "Position condition: road='%s' s=[%.1f, %.1f] (road_length=%.1f)",
            od_pose.road_id,
            clamped_min,
            clamped_max,
            road_length,
        )

        # --- Overflow before road start (s - margin < 0) ---
        if s_min < 0:
            overflow = abs(s_min)
            for pred_id, contact in cls._find_linked_roads(
                od_pose.road_id, "predecessor"
            ):
                pred_length = cls._get_road_length(pred_id)
                if contact == "end":
                    lo = max(0.0, pred_length - overflow)
                    hi = pred_length
                else:
                    lo = 0.0
                    hi = min(overflow, pred_length)
                conditions.append(
                    cls._make_road_segment_condition(
                        entity_name, pred_id, lo, hi, label=label
                    )
                )
                logger.info(
                    "  + predecessor road='%s' s=[%.1f, %.1f] (contact=%s, length=%.1f)",
                    pred_id,
                    lo,
                    hi,
                    contact,
                    pred_length,
                )

        # --- Overflow beyond road end (s + margin > road_length) ---
        if s_max > road_length:
            overflow = s_max - road_length
            for succ_id, contact in cls._find_linked_roads(
                od_pose.road_id, "successor"
            ):
                succ_length = cls._get_road_length(succ_id)
                if contact == "start":
                    lo = 0.0
                    hi = min(overflow, succ_length)
                else:
                    lo = max(0.0, succ_length - overflow)
                    hi = succ_length
                conditions.append(
                    cls._make_road_segment_condition(
                        entity_name, succ_id, lo, hi, label=label
                    )
                )
                logger.info(
                    "  + successor road='%s' s=[%.1f, %.1f] (contact=%s, length=%.1f)",
                    succ_id,
                    lo,
                    hi,
                    contact,
                    succ_length,
                )

        return conditions

    @classmethod
    def _build_lanelet_conditions(
        cls,
        entity_name: Union[EntityRole, str],
        pose: Lanelet2Pose,
        s_margin: float,
        *,
        label: str,
    ) -> list[EntityLanePositionCondition]:
        """The Lanelet2 counterpart of :meth:`_build_position_conditions`.

        ``[s - s_margin, s + s_margin]`` on the lanelet, in its own ``s``; what
        runs past its start is taken from the end of each lanelet before it,
        and what runs past its end from the start of each lanelet after it
        (the routing graph's previous and following lanelets).
        """
        length = cls._get_lanelet_length(pose.lanelet_id)
        s_min = pose.s - s_margin
        s_max = pose.s + s_margin

        conditions = [
            cls._make_lanelet_segment_condition(
                entity_name,
                pose.lanelet_id,
                max(0.0, s_min),
                min(length, s_max),
                label=label,
            )
        ]
        logger.info(
            "Position condition: lanelet=%d s=[%.1f, %.1f] (length=%.1f)",
            pose.lanelet_id,
            max(0.0, s_min),
            min(length, s_max),
            length,
        )
        if s_min < 0:
            overflow = -s_min
            for previous in cls._find_linked_lanelets(pose.lanelet_id, "previous"):
                previous_length = cls._get_lanelet_length(previous)
                lo, hi = max(0.0, previous_length - overflow), previous_length
                conditions.append(
                    cls._make_lanelet_segment_condition(
                        entity_name, previous, lo, hi, label=label
                    )
                )
                logger.info("  + previous lanelet=%d s=[%.1f, %.1f]", previous, lo, hi)
        if s_max > length:
            overflow = s_max - length
            for following in cls._find_linked_lanelets(pose.lanelet_id, "following"):
                lo, hi = 0.0, min(overflow, cls._get_lanelet_length(following))
                conditions.append(
                    cls._make_lanelet_segment_condition(
                        entity_name, following, lo, hi, label=label
                    )
                )
                logger.info(
                    "  + following lanelet=%d s=[%.1f, %.1f]", following, lo, hi
                )
        return conditions

    @staticmethod
    def _make_lanelet_segment_condition(
        entity_name: Union[EntityRole, str],
        lanelet_id: int,
        s_lo: float,
        s_hi: float,
        *,
        label: str,
    ) -> EntityLanePositionCondition:
        """Create an EntityLanePositionCondition for a stretch of one lanelet."""
        return EntityLanePositionCondition(
            entity_name,
            Lanelet2Pose(lanelet_id=lanelet_id, s=0.0),
            rules=[
                ScalarComparisonRule(
                    field="s",
                    rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                    value=s_lo,
                ),
                ScalarComparisonRule(
                    field="s",
                    rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                    value=s_hi,
                ),
            ],
            label=label,
        )

    @staticmethod
    def _get_lanelet_length(lanelet_id: int) -> float:
        """The length of a lanelet's centerline."""
        return lanelet_length(lanelet_id)

    @staticmethod
    def _find_linked_lanelets(lanelet_id: int, direction: str) -> list[int]:
        """The lanelets the routing graph puts before or after *lanelet_id*.

        Args:
            lanelet_id: The lanelet to query.
            direction: ``"previous"`` or ``"following"``.
        """
        mm = MapManager.get_instance()
        lanelet = mm.lanelet_map.laneletLayer[lanelet_id]
        graph = mm.routing_graph
        linked = (
            graph.previous(lanelet)
            if direction == "previous"
            else graph.following(lanelet)
        )
        return [other.id for other in linked]

    @staticmethod
    def _make_road_segment_condition(
        entity_name: Union[EntityRole, str],
        road_id: str,
        s_lo: float,
        s_hi: float,
        *,
        label: str,
    ) -> EntityLanePositionCondition:
        """Create an EntityLanePositionCondition for a road segment.

        The lane is deliberately not part of it: a stop is at an ``s`` along the
        road, and which lane the vehicle waits in is not being asserted.
        """
        return EntityLanePositionCondition.anywhere_on_road(
            entity_name,
            road_id,
            rules=[
                ScalarComparisonRule(
                    field="s",
                    rule=ComparisonRule.GREATER_THAN_OR_EQUAL,
                    value=s_lo,
                ),
                ScalarComparisonRule(
                    field="s",
                    rule=ComparisonRule.LESS_THAN_OR_EQUAL,
                    value=s_hi,
                ),
            ],
            label=label,
        )

    @staticmethod
    def _get_road_length(road_id: str) -> float:
        """Compute the total arc length of an OpenDRIVE road's reference line."""
        mm = MapManager.get_instance()
        road = mm.road_network.road_ids_to_object[road_id]
        ref_line: np.ndarray = road.reference_line
        if len(ref_line) < 2:
            return 0.0
        deltas = np.diff(ref_line, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    @staticmethod
    def _find_linked_roads(road_id: str, direction: str) -> list[tuple[str, str]]:
        """Find predecessor or successor roads with their contact points.

        Parses the ``<link>`` XML element of the given road to extract
        ``<predecessor>`` or ``<successor>`` entries whose ``elementType``
        is ``"road"`` (junctions are excluded).

        Args:
            road_id: The OpenDRIVE road ID to query.
            direction: ``"predecessor"`` or ``"successor"``.

        Returns:
            List of ``(linked_road_id, contact_point)`` tuples.
            ``contact_point`` is ``"start"`` or ``"end"``.
        """
        mm = MapManager.get_instance()
        road = mm.road_network.road_ids_to_object[road_id]
        results: list[tuple[str, str]] = []
        try:
            link_xml = road.road_xml.find("link")
            if link_xml is None:
                return results
            default_contact = "end" if direction == "predecessor" else "start"
            for elem_xml in link_xml.findall(direction):
                elem_type = elem_xml.attrib.get("elementType", "")
                if elem_type != "road":
                    continue
                elem_id = elem_xml.attrib.get("elementId", "")
                if elem_id not in mm.road_network.road_ids_to_object:
                    continue
                contact = elem_xml.attrib.get("contactPoint", default_contact)
                results.append((elem_id, contact))
        except Exception:
            logger.debug(
                "Failed to find %s roads for road %s",
                direction,
                road_id,
                exc_info=True,
            )
        return results

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result once the child condition fires.

        The entity is guaranteed to exist by the
        :class:`EntityExistenceCondition` guard, and the child
        (``OrCondition`` / ``PersistentCondition``) has already passed
        when this method is called.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=True``.
        """
        return ScenarioResult(
            passed=True,
            message=(
                f"Entity '{self._entity_name}' has temporarily stopped"
                f" at a target position at {elapsed:.2f}s"
            ),
            elapsed_seconds=elapsed,
        )
