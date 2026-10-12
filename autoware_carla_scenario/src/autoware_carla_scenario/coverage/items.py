"""Cover items, after ``cover()`` in ASAM OpenSCENARIO DSL -- openodd_py's.

:class:`CoverItem`, :class:`CrossItem` and :class:`SamplingEvent` are
openodd_py's (``openodd.items``); this module keeps the import path a
scenario's :meth:`~autoware_carla_scenario.BaseScenario.register_cover` and
the static check's model of it use.  An item is sampled on a
:class:`SamplingEvent` or, given a
:class:`~autoware_carla_scenario.BaseCondition` as its ``event``, every time
that condition becomes satisfied.
"""

from openodd.items import (
    ABOVE_RANGE,
    BELOW_RANGE,
    COVER_MEASURES,
    HOLDS,
    CoverGroup,
    CoverItem,
    CrossItem,
    Event,
    SamplingEvent,
    duplicates,
    event_name,
    value_label,
)

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
