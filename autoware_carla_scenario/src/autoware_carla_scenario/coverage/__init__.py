"""ODD and scenario coverage, after ``cover()`` in ASAM OpenSCENARIO DSL.

A scenario declares cover items with
:meth:`~autoware_carla_scenario.BaseScenario.register_cover` and crosses them
with :meth:`~autoware_carla_scenario.BaseScenario.register_cross`; the runner
adds the ODD items (:func:`odd_cover_items`) to every run, samples everything
on its event and writes ``{Scenario}_coverage.json``.  ``scenario-coverage``
merges those files into a report.  See ``docs/coverage.md``.
"""

from .collector import COVERAGE_SCHEMA, CoverageCollector
from .items import CoverGroup, CoverItem, CrossItem, SamplingEvent
from .odd import OddProbe, odd_cover_items
from .report import CoverageReport, merge_coverage

__all__ = [
    "COVERAGE_SCHEMA",
    "CoverGroup",
    "CoverItem",
    "CoverageCollector",
    "CoverageReport",
    "CrossItem",
    "OddProbe",
    "SamplingEvent",
    "merge_coverage",
    "odd_cover_items",
]
