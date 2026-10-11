"""Merge coverage files and grade them (openodd's merge).

A bucket is covered when its hits -- or the seconds, metres or entries the
item's ``cover_by`` names -- reach its ``target``; an item's grade is its
covered buckets over those inside the ODD; a group's grade is the mean of its
items'. An item defined differently in different runs is reported once per
definition (``name#2``). See ``docs/coverage.md``.

``scenario-coverage`` is openodd's report, and can also export each run's ODD
samples as an OpenODD COD (``--export-cod``, :mod:`.cod`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

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

logger = logging.getLogger(__name__)


def export_cods(
    documents: Sequence[tuple[Path, dict[str, Any]]], out_dir: Path
) -> list[Any]:
    """Export every run's ODD samples as an OpenODD COD under *out_dir*."""
    from .cod import export_cod  # noqa: PLC0415

    exported = []
    used: set[str] = set()
    for path, doc in documents:
        stem = path.name.removesuffix(".json").removesuffix("_coverage")
        unique, n = stem, 1
        while unique in used:  # runs of one scenario in different directories
            n += 1
            unique = f"{stem}-{n}"
        used.add(unique)
        result = export_cod(doc, out_dir, unique)
        if result is None:
            logger.warning("coverage: %s has no ODD samples to export", path)
        else:
            exported.append(result)
    return exported


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``scenario-coverage``: merge coverage files into a report."""
    parser = argparse.ArgumentParser(
        prog="scenario-coverage",
        description=(
            "Merge the *_coverage.json files a scenario run writes and report "
            "ODD and scenario coverage: grade per item, and the buckets no run hit."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="coverage files, or directories to search for them (e.g. outputs/)",
    )
    parser.add_argument(
        "--json", type=Path, help="also write the merged report as JSON here"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="write the Markdown report here instead of standard output",
    )
    parser.add_argument(
        "--export-cod",
        type=Path,
        metavar="DIR",
        help=(
            "also write each run's ODD samples as an ASAM OpenODD COD table, "
            "with its manifest and taxonomy, into DIR"
        ),
    )
    parser.add_argument(
        "--max-holes",
        type=int,
        default=20,
        help="holes listed per item in the summary table (default: 20)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    documents = load_coverage_files(args.paths)
    report = merge_coverage([doc for _, doc in documents])
    if args.export_cod is not None:
        exported = export_cods(documents, args.export_cod)
        print(  # noqa: T201
            f"Exported {len(exported)} COD table(s) to {args.export_cod}",
            file=sys.stderr,
        )
        if not exported:
            return 1  # asked for, and nothing had samples to export
    if report.runs == 0:
        print("No coverage files found.", file=sys.stderr)
        return 1
    markdown = report.to_markdown(max_holes=args.max_holes)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
