"""The marker a ``--multirun`` job leaves when it is given up: ``case_errored.json``.

A case that raises, times out or crashes writes no result and so no gate
record (:mod:`.record`).  A batch lists such a case in its own summary
(``batch_results.json``, a SLURM batch's outcomes), but a ``--multirun``
sweep has no summary: its jobs are directories of their own.  So a sweep
writes this marker into the job's directory when it gives a job up for any
reason but a failed verdict -- the lanelet sweeper once every attempt has
failed, ``examples/run.py`` when a case of Hydra's basic sweeper raises --
and ``scenario-gate`` refuses a set of runs holding one
(:func:`~autoware_carla_scenario.gate.evaluate.unrecorded_cases`).

A job is also given the marker before each attempt, saying it did not finish
(``"unfinished": true``, :func:`write_case_unfinished`), and the marker is
removed once the job ends with a verdict, passed or failed, or replaced by
the one saying why it was given up.  So a job whose process died with it --
CARLA segfaulting the sweep, the OOM killer, ``SIGKILL``, a Ctrl-C -- leaves
the marker too (#1833), and a job run again in the same directory
(``hydra.sweep.dir=`` naming it again) decides afresh, and a retry that
passes leaves none.  Its name ends in neither a run output's suffix
(``run_outputs.RUN_SUFFIXES``) nor ``_result.json``: the viewer, the retry's
clean-up and the coverage merge do not take it for a run's file.

Free of heavy imports: the sweeper's parent process writes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional  # noqa: UP035

from ..atomic_file import write_text_atomically

__all__ = [
    "CASE_ERRORED",
    "CASE_ERRORED_SCHEMA",
    "find_case_errored",
    "read_case_errored",
    "UNFINISHED_REASON",
    "remove_case_errored",
    "write_case_errored",
    "write_case_unfinished",
]

#: The marker's file name, in the job's directory.
CASE_ERRORED = "case_errored.json"

#: The format of the marker.
CASE_ERRORED_SCHEMA = "autoware_carla_scenario.case_errored/1"

#: The reason of the marker a job is given before each attempt
#: (:func:`write_case_unfinished`): one left behind says the job's process
#: died before the job ended.
UNFINISHED_REASON = "did not finish (its process died, or the sweep was stopped)"


def write_case_errored(
    directory: Path,
    *,
    overrides: list[str],
    exit_code: Optional[int],
    attempts: int,
    reason: str,
    unfinished: bool = False,
) -> Path:
    """Write the marker of a job given up into *directory*, in one rename,
    and return its path.

    *exit_code* is the job's last exit status (``None`` when it timed out or
    has not ended, a negative number for the signal it died of); *reason*
    says why in words.  *unfinished* marks the marker a job is given before
    an attempt (:func:`write_case_unfinished`) with ``"unfinished": true``;
    the others have no such key.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CASE_ERRORED
    # The object `json.dumps(..., indent=2, ensure_ascii=False)` writes, a
    # member at a time: each value is of its own type, which one dict of them
    # all could not be in Codon (docs/typecheck.md).  The overrides, the one
    # member that can nest, are laid out with json.dump(), which takes json's
    # encoder written in Python on every version (as write_json does): it
    # raises RecursionError a thousand levels down.  json.dumps() takes the C
    # encoder for indent=2 from Python 3.13 on, which on 3.14 goes as deep as
    # the stack allows, laying out an indent that grows with each level: an
    # override nested a hundred thousand deep took gigabytes before it gave up.
    overrides_chunks = _Chunks()
    json.dump(
        {"overrides": list(overrides)},
        overrides_chunks,
        indent=2,
        ensure_ascii=False,
    )
    members = [
        _member(
            json.dumps({"schema": CASE_ERRORED_SCHEMA}, indent=2, ensure_ascii=False)
        ),
        _member(overrides_chunks.text()),
        _member(json.dumps({"exit_code": exit_code}, indent=2, ensure_ascii=False)),
        _member(json.dumps({"attempts": attempts}, indent=2, ensure_ascii=False)),
        _member(json.dumps({"reason": reason}, indent=2, ensure_ascii=False)),
    ]
    if unfinished:
        members.append(
            _member(json.dumps({"unfinished": True}, indent=2, ensure_ascii=False))
        )
    write_text_atomically(path, "{\n" + ",\n".join(members) + "\n}")
    return path


class _Chunks:
    """What ``json.dump()`` writes to: the chunks it writes, joined by
    :meth:`text`."""

    _parts: list[str]

    def __init__(self) -> None:
        self._parts = list[str]()

    def write(self, chunk: str) -> int:
        self._parts.append(chunk)
        return len(chunk)

    def text(self) -> str:
        return "".join(self._parts)


def _member(single: str) -> str:
    """The one member of *single*, ``json.dumps({key: value}, indent=2,
    ensure_ascii=False)``, as it is laid out in any object of members at that
    depth: the text between the braces (``"{\\n"`` and ``"\\n}"``)."""
    return single[2:-2]


def write_case_unfinished(
    directory: Path, *, overrides: list[str], attempt: int
) -> Path:
    """Write the marker a job is given before its attempt *attempt* (from 1)
    into *directory*, replacing any marker there, and return its path.

    It says the job did not finish (:data:`UNFINISHED_REASON`); the sweep
    removes it once the job ends with a verdict
    (:func:`remove_case_errored`), or replaces it with the marker of a job
    given up.  Left behind, it is the trace of a job whose process died with
    it (#1833).
    """
    return write_case_errored(
        directory,
        overrides=overrides,
        exit_code=None,
        attempts=attempt,
        reason=UNFINISHED_REASON,
        unfinished=True,
    )


def remove_case_errored(directory: Path) -> bool:
    """Remove the marker from *directory*; whether there was one."""
    path = directory / CASE_ERRORED
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def read_case_errored(path: Path) -> Optional[dict[str, Any]]:
    """The marker at *path* as a JSON object; ``None`` when it cannot be read
    or is not one (it still marks its job as given up)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_case_errored(path: Path) -> list[Path]:
    """Every marker under the directory *path*, sorted; none for a file."""
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob(CASE_ERRORED) if p.is_file())
