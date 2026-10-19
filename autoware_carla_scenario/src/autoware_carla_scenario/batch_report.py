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
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .rerun import RerunConversion

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
    conversions: list[RerunConversion] | None = None,
) -> Path:
    """Write a machine-readable JSON summary to *output_dir* and return the path.

    *overrides*, one list per result, are what each case ran with beyond the
    command line's (its sweep case).  A case in *errored* (indices into
    *results*) raised rather than ran, and its entry says so with
    ``"error": true``.  A case whose Rerun recording (``rerun_recording``)
    failed to convert -- one of *conversions* -- says why in ``rerun_error``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_results.json"
    rerun_errors = {
        str(conversion.recording): conversion.error
        for conversion in conversions or []
        if not conversion.ok
    }
    json_results = []
    for i, (name, result) in enumerate(zip(names, results)):
        entry = result.to_dict()
        recording = entry.get("rerun_recording")
        json_results.append(
            {
                "scenario": name,
                **({"overrides": overrides[i]} if overrides is not None else {}),
                **entry,
                **({"error": True} if i in errored else {}),
                **(
                    {"rerun_error": rerun_errors[recording]}
                    if recording in rerun_errors
                    else {}
                ),
            }
        )
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
    conversions: list[RerunConversion] | None = None,
) -> bool:
    """Print a formatted result table and return ``True`` if all passed.

    A case in *errored* (indices into *results*) raised on every attempt
    rather than ran: it is tagged ``ERROR``, not ``FAIL``, and counts as not
    passed.  A machine-readable JSON file is also written to *output_dir*.
    *conversions* are the batch's Rerun recordings (``rerun.enabled``): each
    case's is listed under it, and every failure in the summary.  They do
    not change whether the batch passed.
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
        recording = result.to_dict().get("rerun_recording")
        if recording is not None:
            print(f"    Rerun: {recording}")  # noqa: T201

    json_path = write_batch_result_json(
        names,
        results,
        output_dir,
        overrides,
        errored=errored,
        conversions=conversions,
    )
    print(thin)  # noqa: T201
    print(f"Result JSON: {json_path}")  # noqa: T201
    if conversions is not None:
        from .rerun import report  # noqa: PLC0415

        print(report(conversions))  # noqa: T201

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(thin)  # noqa: T201
    errors = f", {len(errored)} errored" if errored else ""
    print(f"{passed}/{total} scenarios passed{errors}")  # noqa: T201
    print(sep)  # noqa: T201
    return passed == total
