"""What a driver policy is told of a road user, where a scenario overrides it.

:class:`~autoware_carla_scenario.actions.PerceptionFaultInjectionAction` sets
a :class:`PerceptionFault` on one CARLA actor: the category, the box's size,
or whether it is seen at all, as a perception stack that got that one road
user wrong would report it.  Every other road user is reported as it is.

The faults are read where a policy is told of the other road users -- the
``renderer_data`` actors and the live T4Scene's boxes -- and nowhere else: the
trajectory log, the T4 dataset, the metrics and the conditions keep CARLA's
ground truth.  They are forgotten as each run starts
(:func:`reset_perception_faults`).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .number_settings import finite_number_setting
from .value_text import shown

__all__ = [
    "PerceivedCategory",
    "PerceptionFault",
    "perception_fault",
    "perception_faults",
    "renderer_type_id",
    "reset_perception_faults",
    "set_perception_fault",
    "t4_label",
]


class PerceivedCategory(enum.Enum):
    """The category a road user is reported as: the T4Scene contract's
    ``ObjectCategory``, and ``ACTUAL`` for the one it has."""

    ACTUAL = "actual"
    UNKNOWN = "unknown"
    VEHICLE = "vehicle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    TRAILER = "trailer"
    ANIMAL = "animal"
    HAZARD = "hazard"
    OVER_DRIVABLE = "over_drivable"
    UNDER_DRIVABLE = "under_drivable"


def t4_label(category: PerceivedCategory) -> int:
    """The T4Scene ``ObjectCategory`` value of *category* (``ACTUAL`` has none)."""
    if category is PerceivedCategory.ACTUAL:
        raise ValueError("ACTUAL is the road user's own category, not a label")
    return _T4_LABELS[category]


def renderer_type_id(category: PerceivedCategory) -> str:
    """The ``type_id`` the ``renderer_data`` actors report *category* with.

    A policy tells road users apart by its prefix (``vehicle.``, ``walker.``,
    ``static.prop.``), so the prefix is the category's and the rest names the
    fault: ``walker.perception_fault.pedestrian``.
    """
    if category is PerceivedCategory.ACTUAL:
        raise ValueError("ACTUAL is the road user's own category, not a type_id")
    if category is PerceivedCategory.PEDESTRIAN:
        prefix = "walker"
    elif category in _PROPS:
        prefix = "static.prop"
    else:
        prefix = "vehicle"
    return f"{prefix}.perception_fault.{category.value}"


_T4_LABELS = {
    PerceivedCategory.UNKNOWN: 0,
    PerceivedCategory.VEHICLE: 1,
    PerceivedCategory.CAR: 2,
    PerceivedCategory.TRUCK: 3,
    PerceivedCategory.BUS: 4,
    PerceivedCategory.MOTORCYCLE: 5,
    PerceivedCategory.BICYCLE: 6,
    PerceivedCategory.PEDESTRIAN: 7,
    PerceivedCategory.TRAILER: 8,
    PerceivedCategory.ANIMAL: 9,
    PerceivedCategory.HAZARD: 10,
    PerceivedCategory.OVER_DRIVABLE: 11,
    PerceivedCategory.UNDER_DRIVABLE: 12,
}

#: Reported as props: neither a vehicle nor a walker.
_PROPS = frozenset(
    {
        PerceivedCategory.UNKNOWN,
        PerceivedCategory.ANIMAL,
        PerceivedCategory.HAZARD,
        PerceivedCategory.OVER_DRIVABLE,
        PerceivedCategory.UNDER_DRIVABLE,
    }
)


@dataclass(frozen=True)
class PerceptionFault:
    """How one road user is reported, where it differs from what it is.

    Attributes:
        category: The category reported; ``ACTUAL`` keeps its own.  A
            member's name (``"CAR"``, as a scenario file or the editor
            spells it) or value (``"car"``) is read as that member.
        length_m: The box's length (along the road user's heading), m; 0
            keeps its own.
        width_m: The box's width, m; 0 keeps its own.
        height_m: The box's height, m; 0 keeps its own.
        visible: ``False`` leaves the road user out altogether.

    Raises:
        ValueError: If *category* is no :class:`PerceivedCategory` nor the
            name of one, a size is no number (a bool included), negative or
            not finite, or *visible* is no bool.
    """

    category: PerceivedCategory = PerceivedCategory.ACTUAL
    length_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    visible: bool = True

    def __post_init__(self) -> None:
        length_m = _check_size("length_m", self.length_m)
        width_m = _check_size("width_m", self.width_m)
        height_m = _check_size("height_m", self.height_m)
        if TYPE_CHECKING:
            # The check types every field as it is annotated (#2621).
            pass
        else:
            # A category spelt as text (``"car"``) would raise as the
            # renderer and the T4Scene read it, every tick, and ``"actual"``
            # would be stored as a fault that changes nothing; a size of
            # ``True`` would read as 1 m, and ``visible="false"`` as True.
            object.__setattr__(self, "category", _category(self.category))
            object.__setattr__(self, "length_m", length_m)
            object.__setattr__(self, "width_m", width_m)
            object.__setattr__(self, "height_m", height_m)
            if not isinstance(self.visible, bool):
                raise ValueError(
                    "PerceptionFault: visible must be True or False, "
                    f"got {shown(self.visible)}"
                )

    @property
    def is_none(self) -> bool:
        """Whether the road user is reported as it is."""
        return self == PerceptionFault()

    def size(
        self, length_m: float, width_m: float, height_m: float
    ) -> tuple[float, float, float]:
        """The box's ``(length, width, height)`` reported for one whose own is
        the arguments."""
        return (
            self.length_m or length_m,
            self.width_m or width_m,
            self.height_m or height_m,
        )


def _check_size(name: str, value: float) -> float:
    """*value*, a box size, as a ``float``; refused if it is no number (a
    bool included), negative or not finite."""
    size = finite_number_setting(f"PerceptionFault: {name}", value)
    if size < 0.0:
        raise ValueError(
            f"PerceptionFault: {name} must be finite and not negative "
            f"(0 keeps the road user's own), got {shown(value)}"
        )
    return size


def _category(category: PerceivedCategory) -> PerceivedCategory:
    """*category*, or the member a string names, by its name (``"CAR"``) or
    its value (``"car"``); refused if it is neither."""
    if TYPE_CHECKING:
        # The check types *category* as the member it is annotated with.
        return category
    else:
        if isinstance(category, PerceivedCategory):
            return category
        if isinstance(category, str):
            for member in PerceivedCategory:
                if category in (member.name, member.value):
                    return member
        names = ", ".join(member.name for member in PerceivedCategory)
        raise ValueError(
            "PerceptionFault: category must be a PerceivedCategory or the name "
            f"of one ({names}), got {shown(category)}"
        )


_FAULTS: dict[int, PerceptionFault] = {}


def set_perception_fault(actor_id: int, fault: Optional[PerceptionFault]) -> None:
    """Report the actor *actor_id* as *fault* says from now on; ``None``, or a
    fault that changes nothing, reports it as it is again."""
    if fault is None or fault.is_none:
        _FAULTS.pop(int(actor_id), None)
    else:
        _FAULTS[int(actor_id)] = fault


def perception_fault(actor_id: int) -> Optional[PerceptionFault]:
    """The fault the actor *actor_id* is reported with, or ``None``."""
    return _FAULTS.get(int(actor_id))


def perception_faults() -> dict[int, PerceptionFault]:
    """Every fault now set, by actor id (a copy)."""
    return dict(_FAULTS)


def reset_perception_faults() -> None:
    """Forget every fault: each run starts with every road user as it is."""
    _FAULTS.clear()
