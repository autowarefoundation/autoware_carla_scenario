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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional  # noqa: UP035

from ..atomic_file import write_text_atomically

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

#: The largest route s, lateral offset (m) and speed (m/s) an ego row may
#: hold, either sign.  No route, lane or vehicle comes near it; a record past
#: it is corrupt, and its values would overflow the spread the gate measures.
EGO_VALUE_LIMIT = 1e7


def _object(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """The JSON object *raw* holds at *key*; empty when it holds none.

    Raises:
        ValueError: If *key* holds something other than an object.
    """
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} is not a JSON object ({type(value).__name__})")
    return value


def _array(raw: Mapping[str, Any], key: str) -> list[Any]:
    """The JSON array *raw* holds at *key*; empty when it holds none.

    Raises:
        ValueError: If *key* holds something other than an array.
    """
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key!r} is not a JSON array ({type(value).__name__})")
    return value


@dataclass(frozen=True)
class GateCase:
    """The concrete case a run drove: one route match on one map.

    Attributes:
        map: The map the run was on.
        index: The match's index in its search.
        lanelet_ids: The route's lanelets, in driving order.
        case_count: How many cases the search expanded to on this map; ``0``
            when the run was not told.
        segment_kinds: Each segment's kind (``lane`` / ``junction``).
        segment_ends: The route s each segment ends at; the first starts at 0.
        segment_lanelets: How many of :attr:`lanelet_ids` each segment holds,
            in order: where the segments split.  The route search keeps two
            matches over the same lanelets that split differently apart
            (their trims follow from the split), so this tells them apart
            too; empty in a record written before it was recorded.
    """

    map: str
    index: int
    lanelet_ids: tuple[int, ...]
    case_count: int = 0
    segment_kinds: tuple[str, ...] = ()
    segment_ends: tuple[float, ...] = ()
    segment_lanelets: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        """What tells this case apart from the logical scenario's others: the
        map, the route's lanelets and where its segments split, as the route
        search tells its matches apart.  Integers only, so the same match in
        two runs of a sweep (a baseline and a candidate) has the same key.  A
        record with no split recorded keys by its map and lanelets alone."""
        return self.map, self.lanelet_ids, self.segment_lanelets

    def to_dict(self) -> dict[str, Any]:
        return {
            "map": self.map,
            "index": self.index,
            "lanelet_ids": list(self.lanelet_ids),
            "case_count": self.case_count,
            "segment_kinds": list(self.segment_kinds),
            "segment_ends": [round(v, 4) for v in self.segment_ends],
            "segment_lanelets": list(self.segment_lanelets),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GateCase":
        """The case *raw* holds.

        Raises:
            ValueError: If *raw* is not a JSON object, or one of its lists is
                not a JSON array.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"the case is not a JSON object ({type(raw).__name__})")
        return cls(
            map=str(raw.get("map") or ""),
            index=int(raw.get("index") or 0),
            lanelet_ids=tuple(int(v) for v in _array(raw, "lanelet_ids")),
            case_count=int(raw.get("case_count") or 0),
            segment_kinds=tuple(str(v) for v in _array(raw, "segment_kinds")),
            segment_ends=tuple(float(v) for v in _array(raw, "segment_ends")),
            segment_lanelets=tuple(int(v) for v in _array(raw, "segment_lanelets")),
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
            ValueError: If *raw* is not a gate record of this format: another
                schema, no ``gate`` object, or a part that is not of its JSON
                type (``case``, ``outcome``, ``files`` and ``ego`` objects,
                the ego's rows arrays), or an outcome whose ``passed`` is not
                a JSON boolean (missing, ``"false"``, ``0``, ...: counting it
                as a pass or a failure would guess the run's result).
        """
        if not isinstance(raw, dict):
            raise ValueError("not a gate record (not a JSON object)")
        schema = raw.get("schema")
        if schema != GATE_RECORD_SCHEMA:
            raise ValueError(
                f"not a gate record (schema {schema!r}, expected "
                f"{GATE_RECORD_SCHEMA!r})"
            )
        if raw.get("gate") is None:
            raise ValueError("not a gate record (no 'gate' object)")
        gate = _object(raw, "gate")
        case = _object(raw, "case")
        outcome = _object(raw, "outcome")
        files = _object(raw, "files")
        ego = _object(raw, "ego")
        columns = _array(ego, "columns") or list(EGO_COLUMNS)
        if columns != list(EGO_COLUMNS):
            raise ValueError(f"ego columns {columns}, expected {list(EGO_COLUMNS)}")
        rows: list[tuple[float, float, float, float]] = []
        for row in _array(ego, "rows"):
            if not isinstance(row, list):
                raise ValueError(f"malformed ego row {row!r}")
            values = tuple(float(v) for v in row)
            if len(values) != len(EGO_COLUMNS) or not all(
                math.isfinite(v) for v in values
            ):
                raise ValueError(f"malformed ego row {row!r}")
            if any(abs(v) > EGO_VALUE_LIMIT for v in values[1:]):
                raise ValueError(
                    f"ego row {row!r} beyond {EGO_VALUE_LIMIT:g} (route s, "
                    "lateral offset or speed)"
                )
            rows.append((values[0], values[1], values[2], values[3]))
        passed = outcome.get("passed")
        if not isinstance(passed, bool):
            # bool("false") and bool([0]) are True: a run is passed only by a
            # JSON true, as to_dict() writes it, never by a truthy value.
            raise ValueError(
                f"outcome 'passed' is not a JSON boolean ({passed!r}, "
                f"{type(passed).__name__})"
            )
        category = outcome.get("failure_category")
        return cls(
            scenario=str(raw.get("scenario") or ""),
            title=str(raw.get("title") or ""),
            digest=str(raw.get("digest") or ""),
            gate=dict(gate),
            case=GateCase.from_dict(case),
            passed=passed,
            message=str(outcome.get("message") or ""),
            elapsed_seconds=float(outcome.get("elapsed_seconds") or 0.0),
            failure_category=None if category is None else str(category),
            files={str(k): str(v) for k, v in files.items()},
            ego=rows,
            path=path,
        )


def write_gate_record(path: Path, record: GateRecord) -> None:
    """Write *record* to *path* (its directory made if need be), in one
    rename (:func:`~autoware_carla_scenario.atomic_file.write_text_atomically`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact: a row a tick, and one per line would double the file.  In one
    # rename: a run killed while writing it leaves no truncated record, which
    # the gate would refuse as unreadable (#1956).
    write_text_atomically(path, json.dumps(record.to_dict(), separators=(",", ":")))


def read_gate_record(path: Path) -> GateRecord:
    """The gate record in *path*.

    Raises:
        ValueError: If the file cannot be read, is not JSON, or is not a gate
            record -- malformed in any way, never another exception.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # A string with a lone surrogate ("\ud800") parses, but cannot be
        # written out again (stdout, the reports): refuse it here.
        json.dumps(raw, ensure_ascii=False).encode("utf-8")
    except (OSError, RecursionError, ValueError) as exc:
        # ValueError: not JSON, not UTF-8 (either way), or an integer literal
        # past Python's digit limit; RecursionError: nested too deeply.
        raise ValueError(f"{path}: cannot be read ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a gate record (not a JSON object)")
    try:
        return GateRecord.from_dict(raw, path)
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        # OverflowError: an infinite number (1e999, Infinity) as an int.
        raise ValueError(f"{path}: {exc}") from exc
