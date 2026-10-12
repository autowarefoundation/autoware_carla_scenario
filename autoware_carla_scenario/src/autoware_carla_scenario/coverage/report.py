"""Merge coverage files and grade them (openodd's merge).

A bucket is covered when its hits -- or the seconds, metres or entries the
item's ``cover_by`` names -- reach its ``target``; an item's grade is its
covered buckets over those inside the ODD; a group's grade is the mean of its
items'. An item defined differently in different runs is reported once per
definition (``name#2``). See ``docs/coverage.md``.

``scenario-coverage`` is openodd's ``openodd-coverage``: the report, and with
``--export-cod`` each run's ODD samples as an OpenODD COD (:mod:`.cod`).
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from openodd.cod import export_cods
from openodd.report import main as _main
from openodd.report import (
    COVERAGE_FILE_GLOB,
    COVERAGE_SCHEMAS,
    CoverageReport,
    MergedEntry,
    find_coverage_files,
    load_and_merge,
    load_coverage_files,
    merge_coverage,
)

__all__ = [
    "COVERAGE_FILE_GLOB",
    "COVERAGE_SCHEMAS",
    "CoverageReport",
    "MergedEntry",
    "export_cods",
    "find_coverage_files",
    "load_and_merge",
    "load_coverage_files",
    "main",
    "merge_coverage",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``scenario-coverage``: openodd's ``openodd-coverage``, under this name."""
    return _main(argv, prog="scenario-coverage")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
