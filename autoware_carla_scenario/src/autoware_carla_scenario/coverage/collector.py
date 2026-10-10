"""Sample cover items during a run and write what they hit.

:class:`CoverageCollector` is driven by the scenario runner: :meth:`start` when
the clock starts, :meth:`tick` after every world tick, :meth:`end` when the
tick loop is over.  Like the trajectory recorder it is best effort -- an
expression that raises is logged once and counted as no sample, and nothing
here can fail a run.  An expression that returns ``None`` has nothing to say
(the ego is gone, the simulator does not report the weather) and is not a
sample either.

The coverage file (``{Scenario}_coverage.json``) holds hit counts per bucket
for one run; ``scenario-coverage`` merges any number of them into a report.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .items import (
    ABOVE_RANGE,
    BELOW_RANGE,
    CoverItem,
    CrossItem,
    Event,
    SamplingEvent,
)

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["COVERAGE_SCHEMA", "CoverageCollector", "CROSS_SEPARATOR"]

logger = logging.getLogger(__name__)

#: Written to every coverage file, so a reader can refuse one it does not know.
COVERAGE_SCHEMA = "autoware_carla_scenario.coverage/1"

#: Joins the bucket labels of a cross-coverage cell.
CROSS_SEPARATOR = " / "

_NO_SAMPLE = object()


class _ItemHits:
    def __init__(self, item: CoverItem) -> None:
        self.item = item
        self.hits: dict[str, int] = {label: 0 for label in item.labels}
        self.out_of_range: dict[str, int] = {}
        self.ignored = 0
        self.samples = 0
        self.failed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.item.describe(),
            "samples": self.samples,
            "ignored": self.ignored,
            "hits": dict(self.hits),
            "out_of_range": dict(self.out_of_range),
        }


class _CrossHits:
    def __init__(self, cross: CrossItem) -> None:
        self.cross = cross
        self.hits: dict[str, int] = {
            CROSS_SEPARATOR.join(cell): 0
            for cell in itertools.product(*(i.labels for i in cross.items))
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.cross.describe(), "hits": dict(self.hits)}


class CoverageCollector:
    """Samples a run's cover items on their events and counts bucket hits."""

    def __init__(
        self,
        items: list[CoverItem],
        crosses: Optional[list[CrossItem]] = None,
    ) -> None:
        names = [i.name for i in items] + [c.name for c in crosses or ()]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"coverage: names used more than once: {duplicates}")
        self._items = [_ItemHits(i) for i in items]
        self._crosses = [_CrossHits(c) for c in crosses or ()]
        # Conditions an item or a cross is sampled on, and whether each was
        # satisfied on the previous tick: an item is sampled when one becomes
        # satisfied, not on every tick it stays so.
        self._conditions: dict[int, Any] = {}
        self._was_satisfied: dict[int, bool] = {}
        for entry in self._items:
            event = entry.item.event
            if not isinstance(event, SamplingEvent):
                self._conditions[id(event)] = event
                self._was_satisfied[id(event)] = False

    @property
    def empty(self) -> bool:
        return not self._items

    def start(self, world: "carla.World", elapsed: float) -> None:
        """Sample the items on :attr:`SamplingEvent.START`."""
        self._sample_event(world, SamplingEvent.START)

    def tick(self, world: "carla.World", elapsed: float) -> None:
        """Sample the items on :attr:`SamplingEvent.TICK` and on conditions that fired."""
        self._sample_event(world, SamplingEvent.TICK)
        for key, condition in self._conditions.items():
            try:
                satisfied = condition.check(world, elapsed) is not None
            except Exception:
                logger.warning(
                    "coverage: sampling condition %r raised; it stays unsatisfied",
                    getattr(condition, "label", condition),
                    exc_info=True,
                )
                satisfied = False
            rising = satisfied and not self._was_satisfied[key]
            self._was_satisfied[key] = satisfied
            if rising:
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

    def _sample_event(self, world: "carla.World", event: Event) -> None:
        buckets: dict[str, Optional[str]] = {}
        for entry in self._items:
            if entry.item.event is event:
                buckets[entry.item.name] = self._sample(entry, world)
        if not buckets:
            return
        for cross in self._crosses:
            if cross.cross.event is not event:
                continue
            cell = [buckets.get(i.name) for i in cross.cross.items]
            if all(label is not None for label in cell):
                key = CROSS_SEPARATOR.join(cell)  # type: ignore[arg-type]
                if key in cross.hits:
                    cross.hits[key] += 1

    def _sample(self, entry: _ItemHits, world: "carla.World") -> Optional[str]:
        """Sample one item; the bucket it hit, or ``None`` when it hit none."""
        item = entry.item
        value = self._evaluate(entry, world)
        if value is _NO_SAMPLE or value is None:
            return None
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
            entry.hits[label] += 1
            return label
        if label in (BELOW_RANGE, ABOVE_RANGE) or not item.numeric:
            entry.out_of_range[label] = entry.out_of_range.get(label, 0) + 1
        return None

    def _evaluate(self, entry: _ItemHits, world: "carla.World") -> Any:
        try:
            return entry.item.expression(world)
        except Exception:
            self._fail_once(entry, "expression raised")
            return _NO_SAMPLE

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
