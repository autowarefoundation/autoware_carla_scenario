"""A batch's summary: the table it prints, and ``batch_results.json``.

Free of heavy imports (CARLA, Hydra), so that ``scenario-slurm collect``
(docs/slurm.md) reports a batch that SLURM ran exactly as a batch run in one
process reports itself (``examples/run.py``).  A result is anything with a
:class:`~autoware_carla_scenario.conditions.ScenarioResult`'s attributes
(``rerun_recording`` and ``failure`` optionally: a result without them has no
recording, and says nothing of why it failed).
"""

from __future__ import annotations

import json
from pathlib import Path

# typing's, not collections.abc's: Codon has no collections.abc (the library
# check compiles this module, docs/typecheck.md).
from typing import Any, Collection, Mapping, Protocol, Sequence  # noqa: UP035

from .atomic_file import write_text_atomically
from .failure import Failure

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

    @property
    def failure(self) -> Failure | None:
        """Optional: a result without it says nothing of why (see _category)."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return dict[str, Any]()


def _recording(result: _Result) -> str | None:
    """The Rerun recording of *result*'s run, if one was made."""
    try:
        recording = result.rerun_recording
    except AttributeError:
        return None
    return None if recording is None else str(recording)


def _category(result: _Result) -> str | None:
    """Why *result*'s run failed, as ``category`` or ``category: detail``
    (:mod:`autoware_carla_scenario.failure`), if it says."""
    try:
        failure = result.failure
    except (AttributeError, TypeError, ValueError):
        # A result without one, or with one this version cannot read.
        return None
    if failure is None:
        return None
    name = failure.category.name.lower()
    return name if failure.detail is None else f"{name}: {failure.detail}"


def _member(member_json: str) -> str:
    """A member of an entry, laid out at an entry's depth.

    *member_json* is ``json.dumps({key: value}, indent=2,
    ensure_ascii=False)``, a dict of the one member (made by the caller:
    the library check cannot call a function with a value of any type).  The
    result is what ``json.dumps(entries, indent=2, ensure_ascii=False)``
    gives for the member: its lines between the braces, indented one level
    further.
    """
    return "\n".join("  " + line for line in member_json.split("\n")[1:-1])


#: The message of a pending entry (:func:`write_batch_result_json`).
PENDING_MESSAGE = "No result yet (the batch has not run this case)"


def _pending_entry(
    name: str,
    overrides: list[list[str]] | None,
    index: int,
    stems: list[str] | None = None,
) -> str:
    """The entry of case *index*, *name*, which the batch has not run: as
    ``json.dumps`` lays out ``{"scenario": name, "overrides": ...,
    "stem": ..., "passed": None, "message": PENDING_MESSAGE,
    "elapsed_seconds": None, "pending": True}`` at an entry's depth."""
    members = [_member(json.dumps({"scenario": name}, indent=2, ensure_ascii=False))]
    if overrides is not None:
        members.append(
            _member(
                json.dumps(
                    {"overrides": overrides[index]}, indent=2, ensure_ascii=False
                )
            )
        )
    if stems is not None:
        members.append(
            _member(json.dumps({"stem": stems[index]}, indent=2, ensure_ascii=False))
        )
    members.append('    "passed": null')
    members.append(
        _member(json.dumps({"message": PENDING_MESSAGE}, indent=2, ensure_ascii=False))
    )
    members.append('    "elapsed_seconds": null')
    members.append('    "pending": true')
    return "  {\n" + ",\n".join(members) + "\n  }"


def write_batch_result_json(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path,
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
    rerun_errors: Mapping[int, str] | None = None,
    pending: bool = False,
    stems: list[str] | None = None,
) -> Path:
    """Write a machine-readable JSON summary to *output_dir* and return the path.

    *overrides*, one list per result, are what each case ran with beyond the
    command line's (its sweep case).  A case in *errored* (indices into
    *results*) raised rather than ran, and its entry says so with
    ``"error": true``.  A case in *rerun_errors* (index -> why) has a Rerun
    recording (``rerun_recording``) that failed to convert, and its entry
    says why in ``rerun_error``.

    With *pending*, each name past *results* (a case the batch has not run,
    or never will: it was cut short) is written too, as SLURM's cases without
    an outcome are (``ui.cases``): ``"passed": null``, ``"pending": true``,
    its *overrides* when they are given.  Without it those names are left
    out.

    *stems*, one per name, are the file stems each case's outputs are named
    by (its result, gate record, coverage file, ...), written as each
    entry's ``stem``, after its *overrides*: a batch resumed into its own
    directory (``--resume-into``) removes the outputs of the cases it runs
    again by them, and those of no other case.

    The file is what ``json.dumps(entries, indent=2, ensure_ascii=False)``
    gives, laid out as text a member at a time, since an entry's values (a
    condition's details) are of any type.

    It is replaced in one rename
    (:func:`~autoware_carla_scenario.atomic_file.write_text_atomically`): a
    batch rewrites it as each case ends (``examples/run.py``), and the viewer,
    reading it meanwhile, finds the cases run so far, whole, never half a
    file (#1566).
    """
    failed_recordings: dict[int, str] = {}
    if rerun_errors is not None:
        failed_recordings = dict(rerun_errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_results.json"
    entries: list[str] = []
    for i, (name, result) in enumerate(zip(names, results)):
        # The entry was the dict merge {"scenario": name, "overrides": ...,
        # **result.to_dict(), "error": True, "rerun_error": ...}.  A key of
        # the result's dict that the entry also sets is written once, as the
        # merge kept it: in its first place, with its last value.
        own: dict[str, str] = {
            "scenario": _member(
                json.dumps({"scenario": name}, indent=2, ensure_ascii=False)
            )
        }
        if overrides is not None:
            own["overrides"] = _member(
                json.dumps({"overrides": overrides[i]}, indent=2, ensure_ascii=False)
            )
        if stems is not None:
            own["stem"] = _member(
                json.dumps({"stem": stems[i]}, indent=2, ensure_ascii=False)
            )
        if i in errored:
            own["error"] = _member(
                json.dumps({"error": True}, indent=2, ensure_ascii=False)
            )
        if i in failed_recordings:
            own["rerun_error"] = _member(
                json.dumps(
                    {"rerun_error": failed_recordings[i]}, indent=2, ensure_ascii=False
                )
            )
        data = result.to_dict()
        members: list[str] = []
        for key in ("scenario", "overrides"):
            if key in own and key in data:
                members.append(
                    _member(json.dumps({key: data[key]}, indent=2, ensure_ascii=False))
                )
            elif key in own:
                members.append(own[key])
        # The stem the batch gave the case, whatever the result says.
        if "stem" in own:
            members.append(own["stem"])
        for key, value in data.items():
            if key in ("scenario", "overrides", "stem") and key in own:
                continue
            if key in ("error", "rerun_error") and key in own:
                members.append(own[key])
            else:
                members.append(
                    _member(json.dumps({key: value}, indent=2, ensure_ascii=False))
                )
        for key in ("error", "rerun_error"):
            if key in own and key not in data:
                members.append(own[key])
        entries.append("  {\n" + ",\n".join(members) + "\n  }")
    if pending:
        for i in range(len(results), len(names)):
            entries.append(_pending_entry(names[i], overrides, i, stems))
    text = "[\n" + ",\n".join(entries) + "\n]" if entries else "[]"
    write_text_atomically(json_path, text)
    return json_path


def print_summary(
    names: list[str],
    results: Sequence[_Result],
    output_dir: Path = Path("scenario_outputs"),
    overrides: list[list[str]] | None = None,
    errored: Collection[int] = (),
    rerun_errors: Mapping[int, str] | None = None,
    stems: list[str] | None = None,
) -> bool:
    """Print a formatted result table and return ``True`` if all passed.

    A case in *errored* (indices into *results*) raised on every attempt
    rather than ran: it is tagged ``ERROR``, not ``FAIL``, and counts as not
    passed.  A machine-readable JSON file is also written to *output_dir*,
    each entry with its file stem when *stems* are given
    (:func:`write_batch_result_json`).
    A case's Rerun recording (``rerun.enabled``) is listed under it, and when
    any case has one, how many were written and each that failed
    (*rerun_errors*, index -> why).  They do not change whether the batch
    passed.

    Raises:
        OSError: The JSON file cannot be written; raised once the rest of
            the summary is printed.
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
            category = _category(result)
            if category is not None:
                print(f"         [{category}]")  # noqa: T201
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

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    # A JSON file that cannot be written (an unwritable or read-only output
    # directory) raises once the rest of the summary is printed, so the
    # verdict is not lost (#1740).
    try:
        json_path = write_batch_result_json(
            names,
            results,
            output_dir,
            overrides,
            errored=errored,
            rerun_errors=failed_recordings,
            stems=stems,
        )
        print(thin)  # noqa: T201
        print(f"Result JSON: {json_path}")  # noqa: T201
    finally:
        recorded = [
            i for i, result in enumerate(results) if _recording(result) is not None
        ]
        if recorded:
            written = sum(1 for i in recorded if i not in failed_recordings)
            print(  # noqa: T201
                f"Rerun recordings: {written}/{len(recorded)} written "
                "(open one with: rerun <file.rrd>)"
            )
            for i in sorted(failed_recordings):
                print(f"  [FAILED] {names[i]}: {failed_recordings[i]}")  # noqa: T201

        print(thin)  # noqa: T201
        errors = f", {len(errored)} errored" if errored else ""
        print(f"{passed}/{total} scenarios passed{errors}")  # noqa: T201
        print(sep)  # noqa: T201
    return passed == total
