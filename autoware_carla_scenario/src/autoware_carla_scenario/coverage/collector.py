"""Sample cover items during a run and write what they hit.

:class:`CoverageCollector` is driven by the scenario runner: :meth:`start` when
the clock starts, :meth:`tick` after every world tick, :meth:`end` when the
tick loop is over.  Like the trajectory recorder it is best effort -- an
expression that raises is logged once and counted as no sample, and nothing
here can fail a run.  An expression that returns ``None`` has nothing to say
(the ego is gone, the simulator does not report the weather) and is not a
sample either.

With an ODD, every tick also samples the ODD's attributes, once each
(:meth:`OddDefinition.sample`).  Those values fill the ODD's cover items and
decide whether the tick was inside the ODD.

The coverage file (``{Scenario}_coverage.json``) holds, per bucket of one
run, its hits and, for items sampled every tick, its exposure: the seconds
spent in it, the metres the ego drove in it, and how many times it was
entered.  ``scenario-coverage`` merges any number of them into a report.
"""

from __future__ import annotations

import itertools
import json
import math
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .items import (
    ABOVE_RANGE,
    BELOW_RANGE,
    COVER_MEASURES,
    HOLDS,
    CoverItem,
    CrossItem,
    Event,
    SamplingEvent,
    duplicates,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

    from ..odd.model import OddDefinition, OddVerdict

__all__ = ["COVERAGE_SCHEMA", "CoverageCollector", "CROSS_SEPARATOR"]

logger = logging.getLogger(__name__)

#: Written to every coverage file, so a reader can refuse one it does not know.
COVERAGE_SCHEMA = "autoware_carla_scenario.coverage/1"

#: Joins the bucket labels of a cross-coverage cell.
CROSS_SEPARATOR = " / "

#: The measures a bucket keeps, in the order a stay is recorded.
_MEASURES = COVER_MEASURES

#: Faster than this between two ticks, the ego was moved, not driven (a
#: respawn, a teleport): the step adds no distance.
MAX_EGO_SPEED_MPS = 100.0

#: Out-of-ODD intervals kept per run; a run that leaves the ODD more often is
#: still counted in full, only the intervals past this are not listed.
MAX_OUT_INTERVALS = 100


def _situation_value(holds: Any) -> Optional[str]:
    """A situation's sample: it held, it rested on missing values, or nothing."""
    if holds is True:
        return HOLDS
    if holds is None:
        return UNKNOWN_SITUATION  # counted apart, and the stay is over
    return None  # failed, or inactive


#: A situation whose verdict rested on missing values.
UNKNOWN_SITUATION = "unknown"


class _OddMonitor:
    """Whether each tick was inside the ODD, and which modules ruled it out.

    A tick is ``inside``, ``outside``, or ``assumed``: inside only because
    values were missing, which OpenODD's missing-value semantics count as
    inside.
    """

    def __init__(self, odd: "OddDefinition") -> None:
        self.odd = odd
        self.ticks = {"inside": 0, "assumed": 0, "outside": 0}
        self.seconds = {"inside": 0.0, "assumed": 0.0, "outside": 0.0}
        self.modules = {
            m.name: {"failed_ticks": 0, "missing_ticks": 0} for m in odd.modules
        }
        self.out_intervals: list[list[float]] = []
        #: Whether the excursion going on is the last interval listed.
        self._recording = False

    def record(self, values: dict[str, Any], start: float, end: float) -> "OddVerdict":
        """Judge one tick; the verdict, for the situations."""
        verdict = self.odd.evaluate(values)
        if not verdict.inside:
            key = "outside"
        else:
            key = "assumed" if verdict.assumed else "inside"
        self.ticks[key] += 1
        self.seconds[key] += max(0.0, end - start)
        for name, holds in verdict.modules.items():
            if holds is False:
                self.modules[name]["failed_ticks"] += 1
            elif holds is None:
                self.modules[name]["missing_ticks"] += 1
        outside = not verdict.inside
        if outside and self._recording:
            self.out_intervals[-1][1] = end
        elif outside and len(self.out_intervals) < MAX_OUT_INTERVALS:
            self.out_intervals.append([start, end])
            self._recording = True
        if not outside:
            self._recording = False
        return verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.odd.describe(),
            "ticks": dict(self.ticks),
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            "module_ticks": {k: dict(v) for k, v in self.modules.items()},
            "out_intervals": [
                [round(a, 3), round(b, 3)] for a, b in self.out_intervals
            ],
        }


class _Exposure:
    """Per bucket: hits, and for samples taken every tick, time, distance, entries.

    A sample on a tick stands for the step since the previous tick: its
    seconds and the metres the ego drove.  A stay is a run of consecutive
    ticks in one bucket; entering it again after a tick elsewhere, or after a
    tick with no sample, is a new entry.  A one-shot sample is an entry of no
    duration.
    """

    def __init__(self, labels: list[str], min_stay: Optional[float] = None) -> None:
        self.hits: dict[str, int] = {label: 0 for label in labels}
        self.seconds: dict[str, float] = {label: 0.0 for label in labels}
        self.meters: dict[str, float] = {label: 0.0 for label in labels}
        self.entries: dict[str, int] = {label: 0 for label in labels}
        #: The bucket of the stay going on, if any.
        self._stay: Optional[str] = None
        #: With a minimum stay, the measures of the stays that lasted long
        #: enough, and the stay going on: [bucket, hits, seconds, meters].
        self.min_stay = min_stay
        self._counted: dict[str, dict[str, float]] = {
            m: {label: 0 for label in labels} for m in _MEASURES
        }
        self._pending: Optional[list[Any]] = None

    def add(self, label: str, step: Optional["_Step"]) -> None:
        self.hits[label] += 1
        if step is None:
            self.entries[label] += 1
            return
        if label != self._stay:
            self.entries[label] += 1
            self._settle()
            self._pending = [label, 0, 0.0, 0.0]
        self._stay = label
        self.seconds[label] += step.seconds
        self.meters[label] += step.meters
        if self._pending is not None:
            self._pending[1] += 1
            self._pending[2] += step.seconds
            self._pending[3] += step.meters

    def gap(self) -> None:
        """A tick without a sample in any bucket: the stay is over."""
        self._settle()
        self._stay = None

    def _settle(self) -> None:
        """Count the stay that just ended, if it lasted long enough."""
        pending, self._pending = self._pending, None
        if pending is not None and self._long_enough(pending):
            self._count(self._counted, pending)

    def _long_enough(self, pending: list[Any]) -> bool:
        return self.min_stay is not None and pending[2] >= self.min_stay - 1e-9

    @staticmethod
    def _count(into: dict[str, dict[str, float]], pending: list[Any]) -> None:
        label = pending[0]
        for measure, amount in zip(_MEASURES, (pending[1], pending[2], pending[3], 1)):
            into[measure][label] += amount

    def counted(self) -> dict[str, dict[str, float]]:
        """The measures of the stays at least ``min_stay`` long, the last included."""
        out = {m: dict(v) for m, v in self._counted.items()}
        if self._pending is not None and self._long_enough(self._pending):
            self._count(out, self._pending)
        return out

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hits": dict(self.hits),
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            "meters": {k: round(v, 3) for k, v in self.meters.items()},
            "entries": dict(self.entries),
        }
        if self.min_stay is not None:
            out["counted"] = {
                m: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}
                for m, d in self.counted().items()
            }
        return out


class _Step:
    """The step one tick stands for: its duration, and the ego's distance."""

    __slots__ = ("seconds", "meters")

    def __init__(self, seconds: float, meters: float) -> None:
        self.seconds = seconds
        self.meters = meters


class _ItemHits:
    def __init__(self, item: CoverItem, outside: list[str]) -> None:
        self.item = item
        #: Buckets outside the run's ODD: reported, but not coverage targets.
        self.outside = outside
        self.exposure = _Exposure(item.labels, item.min_stay)
        self.hits = self.exposure.hits
        self.out_of_range: dict[str, int] = {}
        self.ignored = 0
        self.samples = 0
        self.failed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.item.describe(),
            "outside_odd": list(self.outside),
            "samples": self.samples,
            "ignored": self.ignored,
            **self.exposure.to_dict(),
            "out_of_range": dict(self.out_of_range),
        }


class _CrossHits:
    def __init__(self, cross: CrossItem, outside: dict[str, list[str]]) -> None:
        self.cross = cross
        cells = list(itertools.product(*(i.labels for i in cross.items)))
        self.exposure = _Exposure(
            [CROSS_SEPARATOR.join(c) for c in cells], cross.min_stay
        )
        self.hits = self.exposure.hits
        self.outside = [
            CROSS_SEPARATOR.join(cell)
            for cell in cells
            if any(
                label in outside.get(item.name, ())
                for item, label in zip(cross.items, cell)
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.cross.describe(),
            "buckets": list(self.hits),
            "outside_odd": list(self.outside),
            **self.exposure.to_dict(),
        }


class CoverageCollector:
    """Samples a run's cover items on their events and counts bucket hits."""

    def __init__(
        self,
        items: list[CoverItem],
        crosses: Optional[list[CrossItem]] = None,
        odd: Optional["OddDefinition"] = None,
    ) -> None:
        """Collect *items* and *crosses*, and the attributes of *odd* if given.

        With an *odd*, its attributes' cover items come first, and every tick
        is also judged inside or outside it.
        """
        self._odd = _OddMonitor(odd) if odd is not None else None
        # ODD item -> the attribute whose sampled value fills it.
        self._odd_items: dict[str, str] = {}
        # Situation item -> the module whose verdict fills it.
        self._situation_items: dict[str, str] = {}
        outside: dict[str, list[str]] = {}
        if odd is not None:
            for attribute in odd.attributes:
                if attribute.item is not None:
                    self._odd_items[attribute.item.name] = attribute.name
                    outside[attribute.item.name] = odd.outside_buckets(attribute)
            situations = odd.situations()
            self._situation_items = {
                m.item.name: m.name for m in situations if m.item is not None
            }
            items = [
                *odd.cover_items(),
                *(m.item for m in situations if m.item is not None),
                *items,
            ]
        names = [i.name for i in items] + [c.name for c in crosses or ()]
        if duplicates(names):
            raise ValueError(
                f"coverage: names used more than once: {duplicates(names)}"
            )
        self._items = [_ItemHits(i, outside.get(i.name, [])) for i in items]
        self._crosses = [_CrossHits(c, outside) for c in crosses or ()]
        # Items and crosses by the event they are sampled on.
        self._items_on: dict[int, list[_ItemHits]] = {}
        for entry in self._items:
            self._items_on.setdefault(id(entry.item.event), []).append(entry)
        self._crosses_on: dict[int, list[_CrossHits]] = {}
        for cross in self._crosses:
            self._crosses_on.setdefault(id(cross.cross.event), []).append(cross)
        # Conditions an item is sampled on, each with whether it was satisfied
        # on the previous tick: an item is sampled when one becomes satisfied,
        # not on every tick it stays so.
        self._conditions: dict[int, list[Any]] = {
            id(e.item.event): [e.item.event, False]
            for e in self._items
            if not isinstance(e.item.event, SamplingEvent)
        }
        self._failed_conditions: set[int] = set()
        self._last_elapsed = 0.0
        self._last_position: Optional[tuple[float, float, float]] = None
        # Whether anything is sampled every tick, and so measures distance.
        tick = id(SamplingEvent.TICK)
        self._timed = tick in self._items_on or tick in self._crosses_on

    def start(self, world: "carla.World", elapsed: float) -> None:
        """Sample the items on :attr:`SamplingEvent.START`."""
        self._last_elapsed = elapsed
        self._last_position = self._ego_position(world) if self._timed else None
        self._sample_event(world, SamplingEvent.START)

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Sample the items on :attr:`SamplingEvent.TICK` and on conditions that fired."""
        values: dict[str, Any] = {}
        if self._odd is not None:
            attributes = self._odd.odd.sample(world)
            values = {item: attributes[a] for item, a in self._odd_items.items()}
            verdict = self._odd.record(attributes, self._last_elapsed, elapsed)
            for item, module in self._situation_items.items():
                values[item] = _situation_value(verdict.modules.get(module))
        step = self._step(world, elapsed)
        self._sample_event(world, SamplingEvent.TICK, values, step)
        for key, entry in self._conditions.items():
            condition, was_satisfied = entry
            try:
                satisfied = condition.check(world, elapsed) is not None
            except Exception:
                if key not in self._failed_conditions:
                    self._failed_conditions.add(key)
                    logger.warning(
                        "coverage: sampling condition %r raised; it stays "
                        "unsatisfied (logged once)",
                        getattr(condition, "label", condition),
                        exc_info=True,
                    )
                satisfied = False
            entry[1] = satisfied
            if satisfied and not was_satisfied:
                self._sample_event(world, condition)

    def end(self, world: "carla.World", elapsed: float) -> None:
        """Sample the items on :attr:`SamplingEvent.END`."""
        self._sample_event(world, SamplingEvent.END)

    def to_dict(self, scenario: str) -> dict[str, Any]:
        """One run's coverage, in the coverage-file format."""
        return {
            "schema": COVERAGE_SCHEMA,
            "scenario": scenario,
            "items": [e.to_dict() for e in self._items],
            "crosses": [c.to_dict() for c in self._crosses],
            "odd": self._odd.to_dict() if self._odd is not None else None,
        }

    def write(self, path: Path, scenario: str) -> None:
        """Write the coverage file; a failure is logged, never raised."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.to_dict(scenario), indent=2), encoding="utf-8"
            )
        except Exception:
            logger.warning("coverage: could not write %s", path, exc_info=True)

    # ------------------------------------------------------------------

    def _step(self, world: "carla.World", elapsed: float) -> _Step:
        """The step since the previous tick; remembers this tick's time and place."""
        seconds = max(0.0, elapsed - self._last_elapsed)
        # A clock that went back does not move the start of the next step.
        self._last_elapsed = max(elapsed, self._last_elapsed)
        if not self._timed:
            return _Step(seconds, 0.0)  # nothing measures distance: no lookup
        position = self._ego_position(world)
        meters = 0.0
        if position is not None and self._last_position is not None:
            meters = math.dist(position, self._last_position)
            # NaN (a physics blow-up) or a jump (a respawn) is not driving.
            if not math.isfinite(meters) or meters > MAX_EGO_SPEED_MPS * max(
                seconds, 0.05
            ):
                meters = 0.0
        self._last_position = position
        return _Step(seconds, meters)

    @staticmethod
    def _ego_position(world: "carla.World") -> Optional[tuple[float, float, float]]:
        from ..odd.probes import ego_position  # noqa: PLC0415 - odd imports coverage

        try:
            return ego_position(world)
        except Exception:
            return None

    def _sample_event(
        self,
        world: "carla.World",
        event: Event,
        values: Optional[dict[str, Any]] = None,
        step: Optional[_Step] = None,
    ) -> None:
        """Sample the items on *event*; *values* holds those sampled already.

        *step* is the step a tick stands for; ``None`` for a one-shot event.
        """
        entries = self._items_on.get(id(event), ())
        if not entries:
            return
        buckets: dict[str, Optional[str]] = {}
        for entry in entries:
            name = entry.item.name
            if values is not None and name in values:
                value = values[name]
            else:
                value = self._evaluate(entry, world)
            label = self._count(entry, value)
            buckets[name] = label
            if label is None:
                entry.exposure.gap()
            else:
                entry.exposure.add(label, step)
        for cross in self._crosses_on.get(id(event), ()):
            cell = [buckets.get(i.name) for i in cross.cross.items]
            key = CROSS_SEPARATOR.join(cell) if None not in cell else None  # type: ignore[arg-type]
            if key is not None and key in cross.hits:
                cross.exposure.add(key, step)
            else:
                cross.exposure.gap()

    def _evaluate(self, entry: _ItemHits, world: "carla.World") -> Any:
        try:
            return entry.item.expression(world)
        except Exception:
            self._fail_once(entry, "expression raised")
            return None

    def _count(self, entry: _ItemHits, value: Any) -> Optional[str]:
        """Count *value* for one item; the bucket it hit, or ``None`` when it hit none."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None  # NaN (say, 0/0) says nothing either
        item = entry.item
        try:
            if item.ignore is not None and item.ignore(value):
                entry.ignored += 1
                return None
            label = item.bucket_of(value)
        except Exception:
            self._fail_once(entry, f"cannot bucket {value!r}")
            return None
        entry.samples += 1
        if label in entry.hits:
            return label
        if label in (BELOW_RANGE, ABOVE_RANGE) or not item.numeric:
            entry.out_of_range[label] = entry.out_of_range.get(label, 0) + 1
        return None

    @staticmethod
    def _fail_once(entry: _ItemHits, what: str) -> None:
        if entry.failed:
            logger.debug("coverage: %s: %s", entry.item.name, what, exc_info=True)
            return
        entry.failed = True
        logger.warning(
            "coverage: %s: %s; counted as no sample (logged once)",
            entry.item.name,
            what,
            exc_info=True,
        )
