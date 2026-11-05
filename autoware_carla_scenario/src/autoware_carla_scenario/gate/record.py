"""What one run of a logical scenario leaves for its gate: ``{stem}_gate.json``.

A logical scenario's gate (:class:`~autoware_carla_scenario.authoring.models.GateSpec`)
is judged over all its runs, after they have run (:mod:`.evaluate`,
``scenario-gate``).  Each run therefore writes down, beside its result, what
the gate is judged on and what it needs to tell its runs apart:

* the logical scenario (its document's id, title and digest) and the gate it
  states, so the record is judged against the gate the run had;
* the case -- the map and the route match the run drove, and how many matches
  the expansion found there, from which the share of the cases run follows;
* the outcome -- passed or not, and why not;
* the files beside it -- the result, the coverage file the coverage criteria
  are merged from, the trajectory log;
* the ego's drive in the route's frame, a row per tick: the time, the route s,
  the signed lateral offset from the route's reference line (left positive)
  and the speed -- the trajectory-consistency criteria compare these across
  runs without the map.

Free of CARLA and of pydantic: the runner writes it, the evaluator reads it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "GATE_RECORD_SCHEMA",
    "GATE_RECORD_SUFFIX",
    "EGO_COLUMNS",
    "GateCase",
    "GateRecord",
    "read_gate_record",
    "write_gate_record",
]

#: The format of a gate record; :func:`read_gate_record` refuses any other.
GATE_RECORD_SCHEMA = "autoware_carla_scenario.gate_record/1"

#: A run writes its gate record as ``{stem}{GATE_RECORD_SUFFIX}``
#: (``run_outputs.RUN_FILE_SUFFIXES`` holds it too).
GATE_RECORD_SUFFIX = "_gate.json"

#: The columns of the ego's rows: seconds since the first row, route s (m),
#: lateral offset from the route's reference line (m, left positive), speed
#: (m/s).
EGO_COLUMNS: tuple[str, ...] = ("t", "s", "lateral_m", "speed_mps")


@dataclass(frozen=True)
class GateCase:
    """The concrete case a run drove: one route match on one map.

    Attributes:
        map: The map the run was on.
        index: The match's index in its search.
        lanelet_ids: The route's lanelets, in driving order; with :attr:`map`,
            what tells two cases apart.
        case_count: How many cases the search expanded to on this map; ``0``
            when the run was not told.
        segment_kinds: Each segment's kind (``lane`` / ``junction``).
        segment_ends: The route s each segment ends at; the first starts at 0.
    """

    map: str
    index: int
    lanelet_ids: tuple[int, ...]
    case_count: int = 0
    segment_kinds: tuple[str, ...] = ()
    segment_ends: tuple[float, ...] = ()

    @property
    def key(self) -> tuple[str, tuple[int, ...]]:
        """What tells this case apart from the logical scenario's others."""
        return self.map, self.lanelet_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map,
            "index": self.index,
            "lanelet_ids": list(self.lanelet_ids),
            "case_count": self.case_count,
            "segment_kinds": list(self.segment_kinds),
            "segment_ends": [round(v, 4) for v in self.segment_ends],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GateCase":
        return cls(
            map=str(raw.get("map") or ""),
            index=int(raw.get("index") or 0),
            lanelet_ids=tuple(int(v) for v in raw.get("lanelet_ids") or ()),
            case_count=int(raw.get("case_count") or 0),
            segment_kinds=tuple(str(v) for v in raw.get("segment_kinds") or ()),
            segment_ends=tuple(float(v) for v in raw.get("segment_ends") or ()),
        )


@dataclass
class GateRecord:
    """One run of a logical scenario, as its gate reads it.

    Attributes:
        scenario: The logical scenario's id (its document's).
        title: Its title.
        digest: The SHA-256 of the document the run ran (``ui`` left out):
            records of one id with different digests ran different versions.
        gate: The gate the document stated, as it stores it.
        case: The case the run drove.
        passed: Whether the run passed.
        message: The result's message.
        elapsed_seconds: How long the run ran (simulated).
        failure_category: Why it failed (``failure.category``), or ``None``.
        files: The run's other outputs by role (``result``, ``coverage``,
            ``trajectory``), as names beside the record.
        ego: The ego's rows, :data:`EGO_COLUMNS` each.
        path: Where the record was read from (not written).
    """

    scenario: str
    title: str
    digest: str
    gate: dict[str, Any]
    case: GateCase
    passed: bool
    message: str = ""
    elapsed_seconds: float = 0.0
    failure_category: Optional[str] = None
    files: dict[str, str] = field(default_factory=dict)
    ego: list[tuple[float, float, float, float]] = field(default_factory=list)
    path: Optional[Path] = None

    def file(self, role: str) -> Optional[Path]:
        """The run's output *role* (``coverage``, ...), beside the record."""
        name = self.files.get(role)
        if not name or self.path is None:
            return None
        return self.path.parent / name

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_RECORD_SCHEMA,
            "scenario": self.scenario,
            "title": self.title,
            "digest": self.digest,
            "gate": dict(self.gate),
            "case": self.case.to_dict(),
            "outcome": {
                "passed": self.passed,
                "message": self.message,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
                "failure_category": self.failure_category,
            },
            "files": dict(self.files),
            "ego": {
                "columns": list(EGO_COLUMNS),
                "rows": [[round(v, 4) for v in row] for row in self.ego],
            },
        }

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], path: Optional[Path] = None
    ) -> "GateRecord":
        """The record *raw* holds.

        Raises:
            ValueError: If *raw* is not a gate record of this format.
        """
        schema = raw.get("schema")
        if schema != GATE_RECORD_SCHEMA:
            raise ValueError(
                f"not a gate record (schema {schema!r}, expected "
                f"{GATE_RECORD_SCHEMA!r})"
            )
        outcome = raw.get("outcome") or {}
        ego = raw.get("ego") or {}
        columns = list(ego.get("columns") or EGO_COLUMNS)
        if columns != list(EGO_COLUMNS):
            raise ValueError(f"ego columns {columns}, expected {list(EGO_COLUMNS)}")
        rows: list[tuple[float, float, float, float]] = []
        for row in ego.get("rows") or ():
            values = tuple(float(v) for v in row)
            if len(values) != len(EGO_COLUMNS) or not all(
                math.isfinite(v) for v in values
            ):
                raise ValueError(f"malformed ego row {row!r}")
            rows.append((values[0], values[1], values[2], values[3]))
        category = outcome.get("failure_category")
        return cls(
            scenario=str(raw.get("scenario") or ""),
            title=str(raw.get("title") or ""),
            digest=str(raw.get("digest") or ""),
            gate=dict(raw.get("gate") or {}),
            case=GateCase.from_dict(raw.get("case") or {}),
            passed=bool(outcome.get("passed")),
            message=str(outcome.get("message") or ""),
            elapsed_seconds=float(outcome.get("elapsed_seconds") or 0.0),
            failure_category=None if category is None else str(category),
            files={str(k): str(v) for k, v in (raw.get("files") or {}).items()},
            ego=rows,
            path=path,
        )


def write_gate_record(path: Path, record: GateRecord) -> None:
    """Write *record* to *path* (its directory made if need be)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact: a row a tick, and one per line would double the file.
    path.write_text(
        json.dumps(record.to_dict(), separators=(",", ":")), encoding="utf-8"
    )


def read_gate_record(path: Path) -> GateRecord:
    """The gate record in *path*.

    Raises:
        ValueError: If the file is not JSON, or not a gate record.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot be read ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a gate record (not a JSON object)")
    try:
        return GateRecord.from_dict(raw, path)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
