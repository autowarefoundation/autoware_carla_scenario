"""ODD and scenario coverage, after ``cover()`` in ASAM OpenSCENARIO DSL.

Coverage is openodd_py's (``openodd``): the cover items, the collector, the
coverage file, the merged report and the COD export.  A scenario declares
cover items with :meth:`~autoware_carla_scenario.BaseScenario.register_cover`
and crosses them with
:meth:`~autoware_carla_scenario.BaseScenario.register_cross`; the runner adds
the attributes of the run's ODD (:mod:`autoware_carla_scenario.odd`) and
collects everything with openodd's :class:`CoverageCollector`, which asks
:class:`~autoware_carla_scenario.odd.CarlaBackend` where the ego is, and
writes ``{Scenario}_coverage.json``.  ``scenario-coverage`` (:func:`main`)
merges those files into a report.  See ``docs/coverage.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from openodd import (
    COVERAGE_SCHEMA,
    CoverageCollector,
    CoverageReport,
    export_cod,
    load_and_merge,
    merge_coverage,
)

from .items import CoverGroup, CoverItem, CrossItem, SamplingEvent

__all__ = [
    "COVERAGE_SCHEMA",
    "CoverGroup",
    "CoverItem",
    "CoverageCollector",
    "CoverageReport",
    "CrossItem",
    "SamplingEvent",
    "export_cod",
    "load_and_merge",
    "main",
    "merge_coverage",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``scenario-coverage``: openodd_py's ``openodd-coverage``, under this name."""
    from openodd.report import main as openodd_main  # noqa: PLC0415

    return openodd_main(argv, prog="scenario-coverage")
