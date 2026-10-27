"""A batch's summary: the table it prints, and ``batch_results.json``.

Free of heavy imports (CARLA, Hydra), so that ``scenario-slurm collect``
(docs/slurm.md) reports a batch that SLURM ran exactly as a batch run in one
process reports itself (``examples/run.py``).  A result is anything with a
:class:`~autoware_carla_scenario.conditions.ScenarioResult`'s attributes
(``rerun_recording`` optionally: a result without it has no recording).
"""

from __future__ import annotations

import json
from pathlib import Path

# typing's, not collections.abc's: Codon has no collections.abc (the library
# check compiles this module, docs/typecheck.md).
from typing import Any, Collection, Mapping, Protocol, Sequence  # noqa: UP035

__all__ = ["print_summary", "write_batch_result_json"]


# The Protocols' bodies are typed stubs, not `...`, for the library check:
# Codon types a `...` body as returning None.  Nothing instantiates or calls
# them.
class _Status(Protocol):
    @property
    def label(self) -> str:
        return ""

    @property
    def satisfied(self) -> bool:
        return False

    @property
    def message(self) -> str:
        return ""


class _Result(Protocol):
    @property
    def passed(self) -> bool:
        return False

    @property
    def message(self) -> str:
        return ""

    @property
    def elapsed_seconds(self) -> float:
        return 0.0

    @property
    def condition_statuses(self) -> Sequence[_Status]:
        return list[_Status]()

    @property
    def rerun_recording(self) -> str | None:
        """Optional: a result without it has no recording (see _recording)."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return {}


def _recording(result: _Result) -> str | None:
    """The Rerun recording of *result*'s run, if one was made."""
    try:
        recording = result.rerun_recording
    except AttributeError:
        return None
    return None if recording is None else str(recording)


#: The indent of a member of an entry of ``batch_results.json``.
_MEMBER_INDENT = "    "


def _member(key: str, value_json: str) -> str:
    """The line ``"key": value`` of an entry, *value_json* re-indented to it.

    *value_json* is ``json.dumps(value, indent=2, ensure_ascii=False)``.
    """
    value = value_json.replace("\n", "\n" + _MEMBER_INDENT)
    return f"{_MEMBER_INDENT}{json.dumps(key, ensure_ascii=False)}: {value}"


def _result_members(result: _Result) -> list[str]:
    """*result*'s ``to_dict()`` members at an entry's indent, as one item.

    The members' lines already end in their commas, so they are one item of
    the entry's members (none for an empty dict).
    """
    text = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    # "{}" for an empty dict; otherwise "{", the members' lines, "}".
    lines = text.split("\n")[1:-1]
    if not lines:
        return []
    return ["\n".join("  " + line for line in lines)]


def write_batch_result_json(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path,
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
    rerun_errors: Mapping[int, str] | None = None,
) -> Path:
    """Write a machine-readable JSON summary to *output_dir* and return the path.

    *overrides*, one list per result, are what each case ran with beyond the
    command line's (its sweep case).  A case in *errored* (indices into
    *results*) raised rather than ran, and its entry says so with
    ``"error": true``.  A case in *rerun_errors* (index -> why) has a Rerun
    recording (``rerun_recording``) that failed to convert, and its entry
    says why in ``rerun_error``.

    The file is what ``json.dumps(entries, indent=2, ensure_ascii=False)``
    gives, laid out as text a member at a time, since an entry's values (a
    condition's details) are of any type.
    """
    failed_recordings: dict[int, str] = {}
    if rerun_errors is not None:
        failed_recordings = dict(rerun_errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_results.json"
    entries: list[str] = []
    for i, (name, result) in enumerate(zip(names, results)):
        members = [_member("scenario", json.dumps(name, ensure_ascii=False))]
        if overrides is not None:
            members.append(
                _member(
                    "overrides",
                    json.dumps(overrides[i], indent=2, ensure_ascii=False),
                )
            )
        members.extend(_result_members(result))
        if i in errored:
            members.append(_member("error", "true"))
        if i in failed_recordings:
            members.append(
                _member(
                    "rerun_error",
                    json.dumps(failed_recordings[i], ensure_ascii=False),
                )
            )
        entries.append("  {\n" + ",\n".join(members) + "\n  }")
    text = "[\n" + ",\n".join(entries) + "\n]" if entries else "[]"
    json_path.write_text(text, encoding="utf-8")
    return json_path


def print_summary(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path = Path("scenario_outputs"),
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
    rerun_errors: Mapping[int, str] | None = None,
) -> bool:
    """Print a formatted result table and return ``True`` if all passed.

    A case in *errored* (indices into *results*) raised on every attempt
    rather than ran: it is tagged ``ERROR``, not ``FAIL``, and counts as not
    passed.  A machine-readable JSON file is also written to *output_dir*.
    A case's Rerun recording (``rerun.enabled``) is listed under it, and when
    any case has one, how many were written and each that failed
    (*rerun_errors*, index -> why).  They do not change whether the batch
    passed.
    """
    failed_recordings: dict[int, str] = {}
    if rerun_errors is not None:
        failed_recordings = dict(rerun_errors)
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
        if _recording(result) is not None:
            print(f"    Rerun: {_recording(result)}")  # noqa: T201

    json_path = write_batch_result_json(
        names,
        results,
        output_dir,
        overrides,
        errored=errored,
        rerun_errors=failed_recordings,
    )
    print(thin)  # noqa: T201
    print(f"Result JSON: {json_path}")  # noqa: T201
    recorded = [i for i, result in enumerate(results) if _recording(result) is not None]
    if recorded:
        written = sum(1 for i in recorded if i not in failed_recordings)
        print(  # noqa: T201
            f"Rerun recordings: {written}/{len(recorded)} written "
            "(open one with: rerun <file.rrd>)"
        )
        for i in sorted(failed_recordings):
            print(f"  [FAILED] {names[i]}: {failed_recordings[i]}")  # noqa: T201

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(thin)  # noqa: T201
    errors = f", {len(errored)} errored" if errored else ""
    print(f"{passed}/{total} scenarios passed{errors}")  # noqa: T201
    print(sep)  # noqa: T201
    return passed == total
