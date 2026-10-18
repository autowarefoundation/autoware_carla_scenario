"""A batch's summary: the table it prints, and ``batch_results.json``.

Free of heavy imports (CARLA, Hydra), so that ``scenario-slurm collect``
(docs/slurm.md) reports a batch that SLURM ran exactly as a batch run in one
process reports itself (``examples/run.py``).  A result is anything with a
:class:`~autoware_carla_scenario.conditions.ScenarioResult`'s attributes.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Protocol

__all__ = ["print_summary", "write_batch_result_json"]


class _Status(Protocol):
    @property
    def label(self) -> str: ...

    @property
    def satisfied(self) -> bool: ...

    @property
    def message(self) -> str: ...


class _Result(Protocol):
    @property
    def passed(self) -> bool: ...

    @property
    def message(self) -> str: ...

    @property
    def elapsed_seconds(self) -> float: ...

    @property
    def condition_statuses(self) -> Sequence[_Status]: ...

    def to_dict(self) -> dict[str, Any]: ...


def write_batch_result_json(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path,
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
) -> Path:
    """Write a machine-readable JSON summary to *output_dir* and return the path.

    *overrides*, one list per result, are what each case ran with beyond the
    command line's (its sweep case).  A case in *errored* (indices into
    *results*) raised rather than ran, and its entry says so with
    ``"error": true``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_results.json"
    json_results = [
        {
            "scenario": name,
            **({"overrides": overrides[i]} if overrides is not None else {}),
            **result.to_dict(),
            **({"error": True} if i in errored else {}),
        }
        for i, (name, result) in enumerate(zip(names, results))
    ]
    json_path.write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_path


def print_summary(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path = Path("scenario_outputs"),
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
) -> bool:
    """Print a formatted result table and return ``True`` if all passed.

    A case in *errored* (indices into *results*) raised on every attempt
    rather than ran: it is tagged ``ERROR``, not ``FAIL``, and counts as not
    passed.  A machine-readable JSON file is also written to *output_dir*.
    """
    sep = "=" * 60
    thin = "-" * 60
    print(f"\n{sep}")  # noqa: T201
    print("Batch Scenario Results")  # noqa: T201
    print(sep)  # noqa: T201
    for index, (name, result) in enumerate(zip(names, results)):
        tag = "ERROR" if index in errored else "PASS" if result.passed else "FAIL"
        print(f"  [{tag}] {name} ({result.elapsed_seconds:.1f}s)")  # noqa: T201
        if not result.passed:
            for line in result.message.splitlines():
                print(f"         {line}")  # noqa: T201
        if result.condition_statuses:
            max_label_len = max(len(cs.label) for cs in result.condition_statuses)
            for cs in result.condition_statuses:
                mark = "OK" if cs.satisfied else "NG"
                padded = cs.label.ljust(max_label_len)
                print(f"    [{mark}] {padded} : {cs.message}")  # noqa: T201

    json_path = write_batch_result_json(
        names, results, output_dir, overrides, errored=errored
    )
    print(thin)  # noqa: T201
    print(f"Result JSON: {json_path}")  # noqa: T201

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(thin)  # noqa: T201
    errors = f", {len(errored)} errored" if errored else ""
    print(f"{passed}/{total} scenarios passed{errors}")  # noqa: T201
    print(sep)  # noqa: T201
    return passed == total
