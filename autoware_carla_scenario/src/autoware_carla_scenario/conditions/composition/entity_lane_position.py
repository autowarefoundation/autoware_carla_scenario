"""Entity-lane-position scenario pass condition."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from ...coordinate.poses import AnyPose, CarlaWorldPose, OpenDrivePose
from ...coordinate.transform import project_onto_road, to_opendrive
from ...entity_role import EntityRole
from ..base import ScenarioResult, find_actor_by_role_name
from ..comparison import ScalarComparisonRule
from .base import CompositionCondition

if TYPE_CHECKING:
    import carla

logger = logging.getLogger(__name__)

_VALID_FIELDS = frozenset({"s", "t"})


class EntityLanePositionCondition(CompositionCondition):
    """Pass condition that triggers when a named entity is on a specified OpenDRIVE road and lane.

    On every call to :meth:`check`, the entity's CARLA world position is converted
    to an :class:`OpenDrivePose` using the coordinate transformation system.  The
    condition triggers when the resulting ``road_id`` and ``lane_id`` match the
    specified values, and all optional comparison *rules* on ``s`` / ``t`` are
    satisfied.

    The lane is named by a *position* in whichever frame the author thinks in.
    A scenario is written in Lanelet2 -- that is the map an author reads and the
    frame the constraint sweeper resolves spawns in -- while the runtime works
    in OpenDRIVE, so the address is normalised once here, at construction, by
    :func:`~autoware_carla_scenario.coordinate.transform.to_opendrive`.  Doing
    it here rather than in each caller is deliberate: an OpenDRIVE road is *not*
    a lane, and a caller that resolved only the road turned "the entity is on
    lanelet 183" into "the entity is anywhere on road 80" -- which is also true
    while it sits in the neighbouring lane 184, so a cut-in scenario passed
    without the cut-in.

    A condition that means *any* lane of a road is built with
    :meth:`anywhere_on_road` instead.  When comparison *rules* are also
    specified, :func:`project_onto_road` is used to obtain accurate ``s``/``t``
    values on the confirmed road.

    An :class:`EntityExistenceCondition` guard ensures the entity is present
    before the position check runs.

    .. note::
        :class:`~autoware_carla_scenario.coordinate.map_manager.MapManager` must
        be initialised before :meth:`check`, and before *this constructor* when
        *position* is anything but an :class:`OpenDrivePose` -- resolving a
        lanelet to a road and lane reads the map.  Conditions are built inside
        the scenario's ``setup()``, against a live world, so that holds there;
        an :class:`OpenDrivePose` needs no map and can be built anywhere.

    Args:
        entity_name: The ``role_name`` attribute of the actor to track.
        position: Where the entity must be, as a Lanelet2, OpenDRIVE or CARLA
            world pose.  Only the lane it names is used -- say *where along it*
            with *rules*, which are what the runtime compares ``s``/``t``
            against.
        rules: Optional comparison rules applied to the ``s`` and/or ``t``
            coordinates of the resolved :class:`OpenDrivePose`.
        label: Human-readable name for the condition.

    Raises:
        ValueError: If any rule has a ``field`` other than ``'s'`` or ``'t'``.
    """

    def __init__(
        self,
        entity_name: Union[EntityRole, str],
        position: AnyPose,
        rules: Optional[list[ScalarComparisonRule]] = None,
        *,
        label: str,
    ) -> None:
        address = to_opendrive(position)
        super().__init__(entity_name=entity_name, label=label)
        self._road_id = address.road_id
        self._lane_id: Optional[int] = address.lane_id
        self._rules: list[ScalarComparisonRule] = rules or []

        for rule in self._rules:
            if rule.field not in _VALID_FIELDS:
                raise ValueError(
                    f"ScalarComparisonRule field must be 's' or 't', got '{rule.field}'"
                )

    @classmethod
    def anywhere_on_road(
        cls,
        entity_name: Union[EntityRole, str],
        road_id: str,
        rules: Optional[list[ScalarComparisonRule]] = None,
        *,
        label: str,
    ) -> "EntityLanePositionCondition":
        """Build a condition satisfied on any lane of *road_id*.

        A road is rarely a place a scenario means -- it carries several lanes,
        going both ways -- so reach for this only where the lane genuinely does
        not matter.  Stopping short of a stop line is the case that motivated
        it: the stop is at an ``s`` along the road, and which lane the vehicle
        waits in is not being asserted.

        It exists as its own constructor rather than as a lane the caller
        leaves out, so that "I could not work out the lane" -- the mistake this
        class is built to prevent -- cannot be spelled the same way as "the
        lane does not matter here".

        Args:
            entity_name: The ``role_name`` attribute of the actor to track.
            road_id: The OpenDRIVE road the entity must be on.
            rules: Optional comparison rules on ``s`` and/or ``t``.
            label: Human-readable name for the condition.
        """
        # An address always names a lane, so one is supplied and then dropped:
        # the road is what this condition matches on.  Going through __init__
        # rather than around it keeps one construction path.
        condition = cls(
            entity_name,
            OpenDrivePose(road_id=road_id, lane_id=0, s=0.0),
            rules,
            label=label,
        )
        condition._lane_id = None
        return condition

    def get_details(self) -> dict[str, Any]:
        details = super().get_details()
        details.update(
            {
                "road_id": self._road_id,
                "lane_id": self._lane_id,
                "rules": [r.to_dict() for r in self._rules],
            }
        )
        return details

    def _check(self, world: "carla.World", elapsed: float) -> Optional[ScenarioResult]:
        """Return a pass result if the named entity is on the specified road and lane.

        The entity is guaranteed to exist by the
        :class:`EntityExistenceCondition` guard.

        Args:
            world: The CARLA world instance.
            elapsed: Elapsed time in seconds since the scenario started.

        Returns:
            :class:`ScenarioResult` with ``passed=True`` if the entity is on the
            specified road and lane and all rules are satisfied, ``None`` otherwise.
        """
        assert self._entity_name is not None
        entity = find_actor_by_role_name(world, self._entity_name)
        if entity is None:
            return None

        loc = entity.get_location()
        carla_pose = CarlaWorldPose(x=loc.x, y=loc.y, z=loc.z)

        # Always use to_opendrive() first — it finds the nearest road and
        # therefore acts as the authoritative "is the entity on this road?" check.
        od_pose = to_opendrive(carla_pose)

        if od_pose.road_id != self._road_id:
            logger.debug(
                "EntityLanePositionCondition: '%s' on road='%s' lane=%d "
                "(want road='%s') at (%.1f, %.1f, %.1f) t=%.2fs",
                self._entity_name,
                od_pose.road_id,
                od_pose.lane_id,
                self._road_id,
                loc.x,
                loc.y,
                loc.z,
                elapsed,
            )
            return None

        if self._lane_id is not None and od_pose.lane_id != self._lane_id:
            logger.debug(
                "EntityLanePositionCondition: '%s' on road='%s' lane=%d "
                "(want lane=%d) at t=%.2fs",
                self._entity_name,
                od_pose.road_id,
                od_pose.lane_id,
                self._lane_id,
                elapsed,
            )
            return None

        # Without a lane to pin it -- anywhere_on_road() -- and with rules to
        # satisfy, use project_onto_road() for accurate s/t on the specific road
        # (to_opendrive() may pick a slightly different nearest point when roads
        # are close together).
        if self._lane_id is None and self._rules:
            od_pose = project_onto_road(carla_pose, self._road_id)

        # Evaluate s/t comparison rules.
        field_values = {"s": od_pose.s, "t": od_pose.t}
        for rule in self._rules:
            actual = field_values[rule.field]
            if not rule.satisfied(actual):
                logger.debug(
                    "EntityLanePositionCondition: '%s' rule %s %s %.3f "
                    "not satisfied (actual %.3f) at t=%.2fs",
                    self._entity_name,
                    rule.field,
                    rule.rule.name,
                    rule.value,
                    actual,
                    elapsed,
                )
                return None

        lane_desc = "(any lane)" if self._lane_id is None else f"lane {self._lane_id}"
        msg = (
            f"Entity '{self._entity_name}' is on road '{self._road_id}'"
            f" {lane_desc} (s={od_pose.s:.2f}, t={od_pose.t:.2f}) at {elapsed:.2f}s"
        )
        logger.info(
            "EntityLanePositionCondition: MATCHED — '%s' on road='%s' %s"
            " (s=%.2f, t=%.2f)",
            self._entity_name,
            self._road_id,
            lane_desc,
            od_pose.s,
            od_pose.t,
        )

        return ScenarioResult(
            passed=True,
            message=msg,
            elapsed_seconds=elapsed,
        )
