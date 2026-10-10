"""Export a run's ODD samples as an ASAM OpenODD 1.0 current operational domain.

The coverage file keeps, per tick, the time, the ego's position and every ODD
attribute's value (``odd.samples``).  This turns them into OpenODD's exchange
format for what was observed: a COD table (section 8.3), a manifest linking
its columns to the taxonomy (8.3.4), and the taxonomy itself, which has to
travel with a COD (6.1.4.5).

* ``TEMPORAL_EXTENT`` is the run's start time (UTC) plus the elapsed time.
* ``SPATIAL_EXTENT`` is ``"latitude longitude"``, from the map's OpenDRIVE
  geoReference.  A tick without an ego position is left out: the column must
  not be empty.
* A number's column is ``name;unit``; a categorical holds its literal; a
  boolean ``true`` or ``false``; a missing value is empty, which OpenODD reads
  as unknown (8.3.2).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

__all__ = ["CodExport", "export_cod"]

_LITERAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CodExport:
    """The files written for one run."""

    def __init__(self, cod: Path, manifest: Path, taxonomy: Path, rows: int) -> None:
        self.cod = cod
        self.manifest = manifest
        self.taxonomy = taxonomy
        #: Rows written; ticks without an ego position are not.
        self.rows = rows


def _kind(meta: dict[str, Any], column: list[Any]) -> str:
    """``boolean``, ``integer``, ``categorical`` or ``number``."""
    values = meta.get("values")
    seen = [v for v in column if v is not None]
    if values is not None and set(values) <= {"false", "true"}:
        return "boolean"
    if all(isinstance(v, bool) for v in seen) and seen and values is None:
        return "boolean"
    if values is not None:
        if all(re.fullmatch(r"-?\d+", str(v)) for v in values):
            return "integer"
        return "categorical"
    if seen and all(isinstance(v, str) for v in seen):
        return "categorical"
    return "number"


def _unit_type(unit: str) -> Optional[str]:
    from ..odd.units import Units  # noqa: PLC0415 - odd imports coverage

    return Units().unit_type(unit) if unit else None


def _taxonomy(attributes: list[dict[str, Any]], kinds: list[str]) -> dict[str, Any]:
    """An OpenODD YAML taxonomy of the attributes, nested by their dotted names."""
    root: dict[str, Any] = {}
    for meta, kind in zip(attributes, kinds):
        *parents, leaf = str(meta["name"]).split(".")
        node = root
        for part in parents:
            node = node.setdefault(part, {})
        if kind == "boolean":
            node[leaf] = "boolean"
        elif kind == "integer":
            node[leaf] = "integer count"
        elif kind == "categorical":
            literals = [str(v) for v in meta.get("values") or ()]
            node[leaf] = [v for v in literals if _LITERAL.match(v)] or literals
        else:
            unit_type = _unit_type(str(meta.get("unit", ""))) or "count"
            node[leaf] = f"float {unit_type}"
    return {"TAXONOMY": root}


def _cell(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "boolean":
        return "true" if value in (True, "true") else "false"
    return str(value)


def export_cod(doc: dict[str, Any], out_dir: Path, stem: str) -> Optional[CodExport]:
    """Write *doc*'s ODD samples (a coverage document) as a COD under *out_dir*.

    Returns ``None`` when the document has no samples (no ODD, or a coverage
    file from before samples were recorded).
    """
    odd = doc.get("odd") or {}
    samples = odd.get("samples")
    if not samples or not samples.get("rows"):
        return None
    attributes: list[dict[str, Any]] = list(samples.get("attributes", ()))
    rows: list[list[Any]] = list(samples["rows"])
    columns = [[row[3 + i] for row in rows] for i in range(len(attributes))]
    kinds = [_kind(meta, column) for meta, column in zip(attributes, columns)]
    started = _start(odd.get("started_at"))

    out_dir.mkdir(parents=True, exist_ok=True)
    cod = out_dir / f"{stem}_cod.csv"
    manifest = out_dir / f"{stem}_cod_manifest.csv"
    taxonomy = out_dir / f"{stem}_taxonomy.yml"

    headers = [
        f"{meta['name']};{meta['unit']}"
        if kind == "number" and meta.get("unit")
        else str(meta["name"])
        for meta, kind in zip(attributes, kinds)
    ]
    written = 0
    with cod.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["TEMPORAL_EXTENT", "SPATIAL_EXTENT", *headers])
        for row in rows:
            elapsed, lat, lon = row[0], row[1], row[2]
            if lat is None or lon is None:
                continue
            when = started + timedelta(seconds=float(elapsed))
            writer.writerow(
                [
                    when.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    f"{lat:.7f} {lon:.7f}",
                    *(_cell(v, k) for v, k in zip(row[3:], kinds)),
                ]
            )
            written += 1

    leaves = [str(meta["name"]).rsplit(".", 1)[-1] for meta in attributes]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "FIELD_ID",
                "FILE_NAME",
                "TYPE",
                "COLUMN",
                "TAXONOMY_ID",
                "UNIT",
                "COMMENT",
            ]
        )
        writer.writerow(
            ["001", taxonomy.name, "taxonomy", "", "", "", "Taxonomy of the ODD"]
        )
        for n, (meta, header, leaf, kind) in enumerate(
            zip(attributes, headers, leaves, kinds), start=2
        ):
            unique = leaves.count(leaf) == 1
            writer.writerow(
                [
                    f"{n:03d}",
                    cod.name,
                    "cod",
                    header,
                    leaf if unique else meta["name"],
                    meta.get("unit", "") if kind == "number" else "",
                    meta["name"],
                ]
            )

    taxonomy.write_text(
        yaml.safe_dump(_taxonomy(attributes, kinds), sort_keys=False),
        encoding="utf-8",
    )
    return CodExport(cod, manifest, taxonomy, written)


def _start(started_at: Any) -> datetime:
    """The run's start time; the epoch when it was not recorded."""
    if isinstance(started_at, str):
        try:
            return datetime.fromisoformat(started_at)
        except ValueError:
            pass
    return datetime(1970, 1, 1)
