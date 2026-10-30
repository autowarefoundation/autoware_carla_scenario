"""Why a run failed: the category of the fail condition that ended it.

A fail condition's label is the scenario author's (``hit_pedestrian``,
``ego_collided``), and two scenarios name the same failure apart.  A
**failure category** (:class:`FailureCategory`) names it the same in every
scenario, so a batch, a sweep or two versions of a driver can be compared by
what went wrong:

* a scenario gives one with
  :meth:`~autoware_carla_scenario.scenario_base.BaseScenario.register_fail_condition`
  (``category=``);
* without one, the condition says (:meth:`BaseCondition.failure_category
  <autoware_carla_scenario.conditions.BaseCondition.failure_category>`):
  a collision is :attr:`~FailureCategory.COLLISION`, a timeout
  :attr:`~FailureCategory.PROGRESS`, a metric condition what its metric
  reads (a collision metric :attr:`~FailureCategory.COLLISION`).
  A wrapper of one condition (sticky, persistent) says what it wraps; a
  combination of several (AND, OR, NOT) says nothing, and the run's failure is
  :attr:`~FailureCategory.OTHER`.

A run that failed records it as :class:`Failure` in its result
(``"failure"`` in the result JSON and ``batch_results.json``).  A timeout is
told apart further by how the ego ended it (:func:`progress_detail`): never
moving, standing still, or moving without reaching what the scenario asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Mapping, Optional  # noqa: UP035

__all__ = [
    "Failure",
    "FailureCategory",
    "NEVER_MOVED",
    "MOVING",
    "STALLED",
    "progress_detail",
]


class FailureCategory(Enum):
    """What went wrong in a run that failed.

    Written in results by its lower-case name (``"collision"``).

    Attributes:
        COLLISION: The ego hit something.
        TRAFFIC_RULE: It broke a traffic rule: ran a red light, drove against
            the traffic, too fast.
        SIGNALLING: It did not signal what it did (no turn indicator).
        LANE_KEEPING: It left its lane or the road.
        STABILITY: It did not drive steadily: sawed at the steering, weaved,
            flicked its indicators.
        SAFETY_MARGIN: It came too close: time to collision, headway, distance.
        PROGRESS: It did not get where the scenario asked in time.
        DRIVER: The driver ended the run itself.
        INFRASTRUCTURE: The run could not go on: the ego vanished, or the run
            raised.
        OTHER: None of these, or a condition that does not say.
    """

    COLLISION = auto()
    TRAFFIC_RULE = auto()
    SIGNALLING = auto()
    LANE_KEEPING = auto()
    STABILITY = auto()
    SAFETY_MARGIN = auto()
    PROGRESS = auto()
    DRIVER = auto()
    INFRASTRUCTURE = auto()
    OTHER = auto()


#: :func:`progress_detail`: the ego never moved.
NEVER_MOVED = "never_moved"
#: :func:`progress_detail`: it moved, then stood still until the end.
STALLED = "stalled"
#: :func:`progress_detail`: it was moving at the end.
MOVING = "moving"

#: How long the ego must have stood still for a timeout to be ``stalled`` (s).
_STALLED_AFTER_S = 3.0


def progress_detail(moved: bool, still_s: float) -> str:
    """How the ego ended a run that ran out of time.

    Args:
        moved: Whether it ever moved
            (:func:`~autoware_carla_scenario.metrics.ego_progress`).
        still_s: How long it had stood still at the end, in seconds.

    Returns:
        :data:`NEVER_MOVED`, :data:`STALLED` (still for at least 3 s) or
        :data:`MOVING`.
    """
    if not moved:
        return NEVER_MOVED
    if still_s >= _STALLED_AFTER_S:
        return STALLED
    return MOVING


@dataclass
class Failure:
    """Why a run failed.

    Attributes:
        condition: The fail condition that ended the run, as its status is
            labelled (``"fail[0](hit_pedestrian)"``); ``""`` when no condition
            did (the driver ended it, or it raised).
        category: Its category.
        at: When it failed, in simulated seconds.
        message: What the condition said.
        detail: More about it: for :attr:`~FailureCategory.PROGRESS`, how the
            ego ended the run (:func:`progress_detail`); otherwise ``None``.
        also_triggered: The other fail conditions that held when it ended, by
            their status labels, in the order they were registered.
    """

    condition: str
    category: FailureCategory
    at: float
    message: str
    detail: Optional[str] = None
    also_triggered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict, the category by its lower-case name."""
        return {
            "condition": self.condition,
            "category": self.category.name.lower(),
            "at": self.at,
            "message": self.message,
            "detail": self.detail,
            "also_triggered": list(self.also_triggered),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Failure:
        """The failure :meth:`to_dict` wrote (a result read back from JSON).

        Lenient, as results of other versions are read back together: a
        category this version does not know is :attr:`~FailureCategory.OTHER`,
        and a missing field empty.
        """
        detail = data.get("detail")
        try:
            category = FailureCategory[str(data.get("category", "")).upper()]
        except KeyError:
            category = FailureCategory.OTHER
        return Failure(
            condition=str(data.get("condition", "")),
            category=category,
            at=float(data.get("at") or 0.0),
            message=str(data.get("message", "")),
            detail=None if detail is None else str(detail),
            also_triggered=[str(label) for label in data.get("also_triggered") or []],
        )
