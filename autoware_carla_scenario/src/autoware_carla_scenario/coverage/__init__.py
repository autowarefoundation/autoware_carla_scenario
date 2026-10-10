"""ODD and scenario coverage, after ``cover()`` in ASAM OpenSCENARIO DSL.

A scenario declares cover items with
:meth:`~autoware_carla_scenario.BaseScenario.register_cover` and crosses them
with :meth:`~autoware_carla_scenario.BaseScenario.register_cross`.  The runner
adds the attributes of the run's ODD (:mod:`autoware_carla_scenario.odd`) to
every run, samples everything on its event and writes
``{Scenario}_coverage.json``.  ``scenario-coverage`` merges those files into a
report.  See ``docs/coverage.md``.
"""

from .collector import COVERAGE_SCHEMA, CoverageCollector
from .items import CoverGroup, CoverItem, CrossItem, SamplingEvent
from .report import CoverageReport, merge_coverage

__all__ = [
    "COVERAGE_SCHEMA",
    "CoverGroup",
    "CoverItem",
    "CoverageCollector",
    "CoverageReport",
    "CrossItem",
    "SamplingEvent",
    "merge_coverage",
]
