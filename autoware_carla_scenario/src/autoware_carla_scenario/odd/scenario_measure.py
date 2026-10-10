"""Map an ODD attribute onto a scenario measure.

An ODD's taxonomy is its own; the quantities a scenario sets up are measured
under keys the framework fixes (:mod:`autoware_carla_scenario.measures`).
:func:`scenario_measure` is the mapping between the two: the probe of an
attribute that reads the running scenario's measure *key*::

    OddAttribute(
        "dynamic.vehicle_ahead_gap",
        scenario_measure(VEHICLE_AHEAD_GAP_M),
        unit="m",
        buckets=[0, 10, 20, 30, 50, 100],
    )

or, in a binding file, ``vehicle_ahead_gap: {measure: vehicle_ahead_gap_m}``.

The ODD names a measure key, never a scenario; a scenario names its measures
and its controls, never the ODD.  The sampler joins them on the key: an
attribute measured by key *k* is drawn by the running scenario's control of
*k* (:mod:`autoware_carla_scenario.odd.sampler`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..measures import BUILT_IN_MEASURES, read_measure

if TYPE_CHECKING:
    import typesafe_carla.carla as carla

__all__ = ["ScenarioMeasure", "scenario_measure"]


class ScenarioMeasure:
    """The probe that reads the running scenario's measure :attr:`key`."""

    def __init__(self, key: str) -> None:
        if not key:
            raise ValueError("scenario_measure(): key must not be empty")
        self.key = key

    @property
    def unit(self) -> str:
        """The unit of a built-in measure; ``""`` for a scenario's own."""
        built_in = BUILT_IN_MEASURES.get(self.key)
        return "" if built_in is None else built_in.unit

    def __call__(self, world: "carla.World") -> Any:
        return read_measure(self.key, world)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScenarioMeasure) and other.key == self.key

    def __hash__(self) -> int:
        return hash(("scenario_measure", self.key))

    def __repr__(self) -> str:
        return f"scenario_measure({self.key!r})"


def scenario_measure(key: str) -> ScenarioMeasure:
    """The probe that reads the running scenario's measure *key*.

    Outside a run (a route check, a test) it reads the built-in measure, if
    *key* is one.
    """
    return ScenarioMeasure(key)
