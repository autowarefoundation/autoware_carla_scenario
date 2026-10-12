"""Cover items: what is sampled, when, and how a value is put in a bucket (openodd's).

The model follows ``cover()`` in ASAM OpenSCENARIO DSL (section 7.5). An
item's ``event`` is a :class:`SamplingEvent` or a
:class:`~autoware_carla_scenario.BaseCondition`: the item is sampled every
time the condition becomes satisfied (named ``condition:<label>`` in reports).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from openodd.items import (
    ABOVE_RANGE,
    BELOW_RANGE,
    COVER_MEASURES,
    HOLDS,
    CoverGroup,
    CoverItem,
    CrossItem,
    SamplingEvent,
    duplicates,
    event_name,
    value_label,
)

if TYPE_CHECKING:
    from ..conditions import BaseCondition

__all__ = [
    "ABOVE_RANGE",
    "BELOW_RANGE",
    "COVER_MEASURES",
    "HOLDS",
    "CoverGroup",
    "CoverItem",
    "CrossItem",
    "Event",
    "SamplingEvent",
    "duplicates",
    "event_name",
    "value_label",
]

#: When an item is sampled: a built-in event, or a condition of the scenario.
Event = Union[SamplingEvent, "BaseCondition"]
