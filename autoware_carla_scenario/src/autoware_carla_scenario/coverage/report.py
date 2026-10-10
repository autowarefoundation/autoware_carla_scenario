"""Merge coverage files and grade them.

OpenSCENARIO DSL says what to collect, not how runs are merged or graded, so
this follows the convention coverage tools inherit from hardware verification:

* a bucket is covered when its hits reach the item's ``target``;
* an item's grade is its covered buckets over all its buckets;
* a cross's grade is its covered cells over all its cells;
* a group's grade is the mean of its items' and crosses' grades.

Two coverage files describe the same item when the name, the bucket labels and
the event agree.  An item defined differently in different runs is reported
once per definition (with a ``#2`` suffix and so on), never merged, because
adding hits from different buckets would be meaningless.

Besides hits, a merged bucket counts the runs that hit it: on an item sampled
every tick, hits are ticks, and a bucket hit for many ticks in one run is
still only one situation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .collector import COVERAGE_SCHEMA, CROSS_SEPARATOR

__all__ = [
    "CoverageReport",
    "MergedEntry",
    "OddExposure",
    "find_coverage_files",
    "main",
    "merge_coverage",
]

logger = logging.getLogger(__name__)

COVERAGE_FILE_GLOB = "*_coverage.json"


@dataclass
class MergedEntry:
    """An item or a cross merged over runs."""

    name: str
    group: str
    kind: str  # "numeric", "categorical" or "cross"
    event: str
    text: str
    target: int
    buckets: list[str]
    unit: str = ""
    items: list[str] = field(default_factory=list)  # a cross's items
    hits: dict[str, int] = field(default_factory=dict)
    runs: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    samples: int = 0
    ignored: int = 0
    #: Buckets outside the ODD: reported, but not coverage targets.
    outside: list[str] = field(default_factory=list)

    @property
    def targets(self) -> list[str]:
        """The buckets coverage aims at: all but those outside the ODD."""
        excluded = set(self.outside)
        return [b for b in self.buckets if b not in excluded]

    @property
    def covered(self) -> list[str]:
        return [b for b in self.targets if self.hits.get(b, 0) >= self.target]

    @property
    def holes(self) -> list[str]:
        return [b for b in self.targets if self.hits.get(b, 0) < self.target]

    @property
    def grade(self) -> float:
        """Covered targets over targets; 1.0 when the ODD leaves no target."""
        targets = self.targets
        return len(self.covered) / len(targets) if targets else 1.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "group": self.group,
            "kind": self.kind,
            "event": self.event,
            "text": self.text,
            "target": self.target,
            "grade": self.grade,
            "buckets": [
                {
                    "bucket": b,
                    "hits": self.hits.get(b, 0),
                    "runs": self.runs.get(b, 0),
                    "covered": self.hits.get(b, 0) >= self.target,
                    "outside_odd": b in self.outside,
                }
                for b in self.buckets
            ],
        }
        if self.kind == "cross":
            out["items"] = list(self.items)
        else:
            out.update(
                unit=self.unit,
                samples=self.samples,
                ignored=self.ignored,
                out_of_range=dict(self.out_of_range),
            )
        return out


@dataclass
class OddExposure:
    """How long the runs of one ODD spent inside and outside it."""

    name: str
    text: str = ""
    runs: int = 0
    #: Runs that left the ODD at least once.
    runs_outside: int = 0
    ticks: dict[str, int] = field(
        default_factory=lambda: {"inside": 0, "outside": 0, "unknown": 0}
    )
    seconds: dict[str, float] = field(
        default_factory=lambda: {"inside": 0.0, "outside": 0.0, "unknown": 0.0}
    )
    #: Module -> ticks it failed (ruled the ODD out) and ticks it was unknown.
    module_ticks: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Attributes the ODD monitors but has no buckets for.
    unmeasured: list[str] = field(default_factory=list)
    #: (scenario, outside seconds, out intervals) of the runs that left it.
    excursions: list[tuple[str, float, list[list[float]]]] = field(default_factory=list)

    def add(self, scenario: str, raw: dict[str, Any]) -> None:
        self.runs += 1
        self.text = self.text or raw.get("text", "")
        for key in self.ticks:
            self.ticks[key] += int(raw.get("ticks", {}).get(key, 0))
            self.seconds[key] += float(raw.get("seconds", {}).get(key, 0.0))
        for module, counts in raw.get("module_ticks", {}).items():
            mine = self.module_ticks.setdefault(
                module, {"failed_ticks": 0, "unknown_ticks": 0}
            )
            for key in mine:
                mine[key] += int(counts.get(key, 0))
        for name in raw.get("unmeasured", ()):
            if name not in self.unmeasured:
                self.unmeasured.append(name)
        outside_seconds = float(raw.get("seconds", {}).get("outside", 0.0))
        if int(raw.get("ticks", {}).get("outside", 0)) > 0:
            self.runs_outside += 1
            self.excursions.append(
                (scenario, outside_seconds, list(raw.get("out_intervals", ())))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "runs": self.runs,
            "runs_outside": self.runs_outside,
            "ticks": dict(self.ticks),
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            "module_ticks": {k: dict(v) for k, v in self.module_ticks.items()},
            "unmeasured": list(self.unmeasured),
            "excursions": [
                {"scenario": s, "outside_seconds": round(t, 3), "intervals": i}
                for s, t, i in self.excursions
            ],
        }


@dataclass
class CoverageReport:
    """Coverage merged over a number of runs."""

    runs: int = 0
    scenarios: dict[str, int] = field(default_factory=dict)
    entries: list[MergedEntry] = field(default_factory=list)
    #: ODD name -> how the runs measured against it fared.
    odds: dict[str, OddExposure] = field(default_factory=dict)

    def groups(self) -> list[str]:
        order = {"odd": 0, "scenario": 1}
        return sorted(
            {e.group for e in self.entries}, key=lambda g: (order.get(g, 9), g)
        )

    def group_grade(self, group: str) -> float:
        grades = [e.grade for e in self.entries if e.group == group]
        return sum(grades) / len(grades) if grades else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "scenarios": dict(self.scenarios),
            "odds": {k: v.to_dict() for k, v in self.odds.items()},
            "groups": {
                g: {
                    "grade": self.group_grade(g),
                    "entries": [e.to_dict() for e in self.entries if e.group == g],
                }
                for g in self.groups()
            },
        }

    def to_markdown(self, *, max_holes: int = 20) -> str:
        lines = [
            "# Coverage report",
            "",
            f"Runs merged: {self.runs}",
            "",
        ]
        for scenario, count in sorted(self.scenarios.items()):
            lines.append(f"- {scenario}: {count}")
        for exposure in self.odds.values():
            lines += self._odd_markdown(exposure)
        for group in self.groups():
            entries = [e for e in self.entries if e.group == group]
            title = (
                "ODD coverage" if group == "odd" else f"{group.capitalize()} coverage"
            )
            lines += [
                "",
                f"## {title}: {self.group_grade(group):.1%}",
                "",
                "| Item | Event | Grade | Covered | Holes |",
                "|---|---|---|---|---|",
            ]
            for e in entries:
                holes = e.holes
                shown = ", ".join(holes[:max_holes])
                if len(holes) > max_holes:
                    shown += f", ... ({len(holes) - max_holes} more)"
                label = f"{e.name} ({' x '.join(e.items)})" if e.items else e.name
                lines.append(
                    f"| {label} | {e.event} | {e.grade:.0%} "
                    f"| {len(e.covered)}/{len(e.targets)} | {shown or '-'} |"
                )
            for e in entries:
                if e.kind == "cross":
                    continue
                lines += [
                    "",
                    f"### {e.name}" + (f" [{e.unit}]" if e.unit else ""),
                    "",
                ]
                if e.text:
                    lines += [e.text, ""]
                lines += ["| Bucket | Hits | Runs |", "|---|---|---|"]
                for b in e.buckets:
                    if b in e.outside:
                        mark = " (outside ODD" + (
                            ", reached)" if e.hits.get(b) else ")"
                        )
                    else:
                        mark = "" if e.hits.get(b, 0) >= e.target else " (hole)"
                    lines.append(
                        f"| {b}{mark} | {e.hits.get(b, 0)} | {e.runs.get(b, 0)} |"
                    )
                for b, n in sorted(e.out_of_range.items()):
                    lines.append(f"| {b} (outside the buckets) | {n} | - |")
                if e.samples == 0:
                    lines += ["", "_No samples: the value was never available._"]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _odd_markdown(exposure: OddExposure) -> list[str]:
        total = sum(exposure.seconds.values())

        def share(key: str) -> str:
            seconds = exposure.seconds[key]
            pct = f" ({seconds / total:.1%})" if total else ""
            return f"{seconds:.1f} s{pct}"

        lines = [
            "",
            f"## ODD: {exposure.name}",
            "",
        ]
        if exposure.text:
            lines += [exposure.text, ""]
        lines += [
            f"- Runs: {exposure.runs}, of which left the ODD: {exposure.runs_outside}",
            f"- Inside: {share('inside')}",
            f"- Outside: {share('outside')}",
            f"- Unknown: {share('unknown')}",
        ]
        if exposure.unmeasured:
            lines.append(
                "- Monitored but not covered (no buckets): "
                + ", ".join(exposure.unmeasured)
            )
        failing = [
            (name, counts)
            for name, counts in exposure.module_ticks.items()
            if counts["failed_ticks"] or counts["unknown_ticks"]
        ]
        if failing:
            lines += ["", "| Module | Failed ticks | Unknown ticks |", "|---|---|---|"]
            for name, counts in failing:
                lines.append(
                    f"| {name} | {counts['failed_ticks']} | {counts['unknown_ticks']} |"
                )
        if exposure.excursions:
            lines += [
                "",
                "| Scenario | Outside | First intervals [s] |",
                "|---|---|---|",
            ]
            worst = sorted(exposure.excursions, key=lambda e: -e[1])[:10]
            for scenario, seconds, intervals in worst:
                shown = ", ".join(f"{a:g}-{b:g}" for a, b in intervals[:3])
                lines.append(f"| {scenario} | {seconds:.1f} s | {shown} |")
        return lines


def find_coverage_files(paths: Iterable[Path]) -> list[Path]:
    """Coverage files named by *paths*: files as given, directories searched."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob(COVERAGE_FILE_GLOB)))
        elif path.is_file():
            found.append(path)
        else:
            logger.warning("coverage: %s does not exist", path)
    return found


def _key(entry: dict[str, Any], kind: str) -> tuple[Any, ...]:
    return (
        entry["name"],
        kind,
        entry.get("event", ""),
        tuple(entry.get("buckets", ())),
        tuple(entry.get("items", ())),
        tuple(entry.get("outside_odd", ())),
    )


def _cross_outside(
    cross: dict[str, Any], outside_by_item: dict[str, set[str]]
) -> list[str]:
    """Cells of *cross* with a bucket outside the ODD in any of its items."""
    items = list(cross.get("items", ()))
    out = []
    for cell in cross.get("hits", {}):
        labels = cell.split(CROSS_SEPARATOR)
        if len(labels) == len(items) and any(
            label in outside_by_item.get(item, ()) for item, label in zip(items, labels)
        ):
            out.append(cell)
    return out


def merge_coverage(documents: Sequence[dict[str, Any]]) -> CoverageReport:
    """Merge coverage documents (the contents of coverage files) into a report."""
    report = CoverageReport()
    merged: dict[tuple[Any, ...], MergedEntry] = {}
    by_name: dict[str, int] = {}
    for doc in documents:
        if doc.get("schema") != COVERAGE_SCHEMA:
            raise ValueError(
                f"not a coverage file of schema {COVERAGE_SCHEMA}: {doc.get('schema')!r}"
            )
        report.runs += 1
        scenario = str(doc.get("scenario", ""))
        report.scenarios[scenario] = report.scenarios.get(scenario, 0) + 1
        odd = doc.get("odd")
        if odd:
            name = str(odd.get("name", ""))
            report.odds.setdefault(name, OddExposure(name)).add(scenario, odd)
        outside_by_item = {
            e["name"]: set(e.get("outside_odd", ())) for e in doc.get("items", ())
        }
        entries = [(e, e["kind"]) for e in doc.get("items", ())]
        entries += [
            ({**c, "outside_odd": _cross_outside(c, outside_by_item)}, "cross")
            for c in doc.get("crosses", ())
        ]
        for raw, kind in entries:
            key = _key(raw, kind)
            entry = merged.get(key)
            if entry is None:
                count = by_name.get(raw["name"], 0) + 1
                by_name[raw["name"]] = count
                if count > 1:
                    logger.warning(
                        "coverage: %s is defined differently in different runs; "
                        "reported separately as %s#%d",
                        raw["name"],
                        raw["name"],
                        count,
                    )
                entry = MergedEntry(
                    name=raw["name"] if count == 1 else f"{raw['name']}#{count}",
                    group=raw.get("group", "scenario"),
                    kind=kind,
                    event=raw.get("event", ""),
                    text=raw.get("text", ""),
                    target=int(raw.get("target", 1)),
                    buckets=list(raw.get("buckets") or raw.get("hits", {}).keys()),
                    unit=raw.get("unit", ""),
                    items=list(raw.get("items", ())),
                    outside=list(raw.get("outside_odd", ())),
                )
                merged[key] = entry
                report.entries.append(entry)
            for bucket, hits in raw.get("hits", {}).items():
                entry.hits[bucket] = entry.hits.get(bucket, 0) + int(hits)
                if hits:
                    entry.runs[bucket] = entry.runs.get(bucket, 0) + 1
            for bucket, hits in raw.get("out_of_range", {}).items():
                entry.out_of_range[bucket] = entry.out_of_range.get(bucket, 0) + int(
                    hits
                )
            entry.samples += int(raw.get("samples", 0))
            entry.ignored += int(raw.get("ignored", 0))
    if not merged and report.runs == 0:
        logger.warning("coverage: nothing to merge")
    return report


def load_and_merge(paths: Iterable[Path]) -> CoverageReport:
    """Find the coverage files under *paths* and merge them."""
    documents = []
    for path in find_coverage_files(paths):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("coverage: skipping unreadable %s", path, exc_info=True)
    return merge_coverage(documents)


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
        "--max-holes",
        type=int,
        default=20,
        help="holes listed per item in the summary table (default: 20)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    report = load_and_merge(args.paths)
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
