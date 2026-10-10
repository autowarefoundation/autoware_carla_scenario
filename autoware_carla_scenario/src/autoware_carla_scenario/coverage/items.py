"""Cover items: what is sampled, when, and how a value is put in a bucket.

The model follows ``cover()`` in ASAM OpenSCENARIO DSL (section 7.5 of the
language reference) and keeps its vocabulary, so a reader who knows the DSL
recognises the arguments:

* ``expression`` -- what is sampled.  Here a callable that reads the world.
* ``unit`` -- the unit the expression returns its value in.  Informational:
  nothing is converted.
* ``range`` + ``every`` -- equal buckets over a range of interest.
* ``buckets`` -- explicit bucket boundaries; ``N`` values make ``N - 1``
  buckets.
* ``values`` -- one bucket per value, which is how the DSL treats an enum or a
  bool.
* ``ignore`` -- samples for which it returns true are left out.
* ``event`` -- when the item is sampled.  The DSL samples at the end of the
  scenario unless told otherwise, and so does this.
* ``target`` -- how many hits make a bucket covered.
* cross coverage -- the Cartesian product of items sampled on the same event
  (``cover(name, items: [a, b])`` in the DSL).

A value outside every bucket is not dropped silently: it is counted as out of
range (below or above a numeric item's buckets) or unexpected (not one of a
categorical item's values), so a report shows the run reached somewhere the
model did not describe.
"""

from __future__ import annotations

import bisect
import enum
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Union

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..conditions import BaseCondition

__all__ = [
    "duplicates",
    "BELOW_RANGE",
    "ABOVE_RANGE",
    "CoverGroup",
    "CoverItem",
    "CrossItem",
    "SamplingEvent",
    "value_label",
]

#: Pseudo-bucket a numeric sample below the first bucket is counted in.
BELOW_RANGE = "<below>"
#: Pseudo-bucket a numeric sample above the last bucket is counted in.
ABOVE_RANGE = "<above>"


class SamplingEvent(enum.Enum):
    """When a cover item is sampled during a run.

    ``END`` is the default, as in OpenSCENARIO DSL.  Pass a
    :class:`~autoware_carla_scenario.BaseCondition` as the ``event`` instead to
    sample every time that condition becomes satisfied.
    """

    #: Once, when the scenario clock starts.
    START = "start"
    #: Once, when the tick loop ends -- passed, failed or done.
    END = "end"
    #: After every world tick.  A bucket's hits are then ticks spent in it.
    TICK = "tick"


class CoverGroup(enum.Enum):
    """Which report a cover item belongs to.

    Kept apart because they answer different questions: ODD coverage is which
    operating conditions the runs exercised, whatever the scenario was;
    scenario coverage is which values of a scenario's own parameters they hit.
    """

    ODD = "odd"
    SCENARIO = "scenario"


Event = Union[SamplingEvent, "BaseCondition"]


def duplicates(names: Iterable[str]) -> list[str]:
    """The names that occur more than once, sorted."""
    return sorted(n for n, count in Counter(names).items() if count > 1)


def value_label(value: Any) -> str:
    """The bucket label of a categorical value: an enum by name, a bool in lowercase."""
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _number_label(x: float) -> str:
    return f"{x:g}"


def event_name(event: Event) -> str:
    """How *event* is written in a report."""
    if isinstance(event, SamplingEvent):
        return event.value
    return f"condition:{getattr(event, 'label', type(event).__name__)}"


@dataclass
class CoverItem:
    """One cover item.  Built by :meth:`BaseScenario.register_cover`.

    Exactly one of *values*, *buckets* and *range* (with *every*) defines the
    buckets.
    """

    name: str
    expression: Callable[["carla.World"], Any]
    unit: str = ""
    range: Optional[tuple[float, float]] = None
    every: Optional[float] = None
    buckets: Optional[Sequence[float]] = None
    values: Optional[Iterable[Any]] = None
    ignore: Optional[Callable[[Any], bool]] = None
    event: Event = SamplingEvent.END
    text: str = ""
    target: int = 1
    group: CoverGroup = CoverGroup.SCENARIO

    #: Bucket edges of a numeric item, ascending; empty for a categorical one.
    edges: list[float] = field(init=False, default_factory=list)
    #: Bucket labels, in order.
    labels: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cover: name must not be empty")
        if self.target < 1:
            raise ValueError(f"cover({self.name}): target must be at least 1")
        given = [
            k
            for k, v in (
                ("values", self.values),
                ("buckets", self.buckets),
                ("range", self.range),
            )
            if v is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"cover({self.name}): give exactly one of values, buckets or "
                f"range (with every); got {given or 'none'}"
            )
        if self.every is not None and self.range is None:
            raise ValueError(f"cover({self.name}): every needs a range")
        if self.values is not None:
            self.values = list(self.values)
            self.labels = [value_label(v) for v in self.values]
            if not self.labels:
                raise ValueError(f"cover({self.name}): values must not be empty")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"cover({self.name}): values must be distinct")
            return
        if self.range is not None:
            if self.every is None:
                raise ValueError(f"cover({self.name}): range needs every")
            low, high = (float(x) for x in self.range)
            if not low < high:
                raise ValueError(f"cover({self.name}): range must be (low, high)")
            if not self.every > 0:
                raise ValueError(f"cover({self.name}): every must be positive")
            count = max(1, math.ceil((high - low) / self.every - 1e-9))
            self.edges = [low + i * self.every for i in range(count)] + [high]
        else:
            assert self.buckets is not None
            self.edges = [float(x) for x in self.buckets]
            if len(self.edges) < 2:
                raise ValueError(f"cover({self.name}): buckets needs two edges or more")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError(f"cover({self.name}): bucket edges must ascend")
        last = len(self.edges) - 2
        self.labels = [
            f"[{_number_label(a)}, {_number_label(b)}{']' if i == last else ')'}"
            for i, (a, b) in enumerate(zip(self.edges, self.edges[1:]))
        ]

    @property
    def numeric(self) -> bool:
        """Whether the buckets are ranges rather than values."""
        return bool(self.edges)

    def bucket_of(self, value: Any) -> str:
        """The label of the bucket *value* falls in.

        Returns :data:`BELOW_RANGE` or :data:`ABOVE_RANGE` for a numeric value
        outside the buckets, and the value's own label for a categorical value
        that is not one of :attr:`values` (which is then not a label in
        :attr:`labels`).  Each bucket holds its lower edge; the last holds its
        upper edge as well.
        """
        if not self.numeric:
            return value_label(value)
        x = float(value)
        if math.isnan(x) or x < self.edges[0]:
            return BELOW_RANGE
        if x > self.edges[-1]:
            return ABOVE_RANGE
        index = bisect.bisect_right(self.edges, x) - 1
        return self.labels[min(index, len(self.labels) - 1)]

    def describe(self) -> dict[str, Any]:
        """The item's definition, as written to a coverage file."""
        return {
            "name": self.name,
            "group": self.group.value,
            "text": self.text,
            "unit": self.unit,
            "event": event_name(self.event),
            "kind": "numeric" if self.numeric else "categorical",
            "buckets": list(self.labels),
            "target": self.target,
        }


@dataclass
class CrossItem:
    """Cross coverage of cover items sampled on the same event."""

    name: str
    items: list[CoverItem]
    text: str = ""
    target: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cross: name must not be empty")
        if len(self.items) < 2:
            raise ValueError(f"cross({self.name}): cross at least two items")
        if self.target < 1:
            raise ValueError(f"cross({self.name}): target must be at least 1")
        first = self.items[0]
        for item in self.items[1:]:
            if item.event is not first.event:
                raise ValueError(
                    f"cross({self.name}): {item.name} is sampled on "
                    f"{event_name(item.event)} and {first.name} on "
                    f"{event_name(first.event)}; only items sampled on the "
                    "same event can be crossed"
                )
        if len({i.group for i in self.items}) != 1:
            raise ValueError(f"cross({self.name}): items are in different groups")

    @property
    def event(self) -> Event:
        return self.items[0].event

    @property
    def group(self) -> CoverGroup:
        return self.items[0].group

    def describe(self) -> dict[str, Any]:
        """The cross's definition, as written to a coverage file."""
        return {
            "name": self.name,
            "group": self.group.value,
            "text": self.text,
            "event": event_name(self.event),
            "items": [i.name for i in self.items],
            "target": self.target,
        }
