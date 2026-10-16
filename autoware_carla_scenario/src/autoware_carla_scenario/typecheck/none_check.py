"""The None check: an Optional used without a test for None (#239).

Codon 0.19 unwraps an ``Optional[T]`` implicitly wherever a ``T`` is needed
(arithmetic, an attribute, an argument), so the compile accepts
``cfg.min_speed_kmh / 3.6`` for a ``float | None`` field, and Python raises
``TypeError`` when the field is ``None``.  Codon has no strict-optional mode,
so the scenario check runs mypy over the scenario's own modules, the ones it
compiles, and keeps only what mypy reports of a value that may be ``None``
used where it may not be: an attribute of it (``union-attr``), an operator
(``operator``), an argument (``arg-type``), an assignment (``assignment``),
a return (``return-value``), an index (``index``) or an overloaded call
(``call-overload``), each message naming ``None``.  Every other mypy error
is Codon's to report, or no business of the check's.

mypy runs as its own process (``python -m mypy``), with no configuration file
(the scenario package's own mypy settings do not apply), following imports
silently: it reads the framework and the scenario package's other modules for
their types and reports only in the modules given.  The bodies of functions
without annotations are checked too (``--check-untyped-defs``).  Its cache is kept in the
user's cache directory, so only the first check reads the framework.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NoneFinding", "find_none_misuses"]

#: The mypy error codes of a value that may be None used where it may not.
NONE_CODES = frozenset(
    {
        "union-attr",
        "operator",
        "arg-type",
        "assignment",
        "return-value",
        "index",
        "call-overload",
    }
)

_LINE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?:(?P<col>\d+):)? "
    r"(?P<severity>error|note): (?P<message>.*?)(?:  \[(?P<code>[a-z-]+)\])?$"
)


@dataclass(frozen=True)
class NoneFinding:
    """What mypy reports of a value that may be None, at its place."""

    path: str
    line: int
    column: int | None
    message: str
    code: str
    notes: tuple[str, ...] = ()


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    version = f"py{sys.version_info.major}{sys.version_info.minor}"
    return Path(base) / "autoware_carla_scenario" / "mypy" / version


def mypy_available() -> bool:
    """Whether mypy can be run (a dependency of the framework)."""
    try:
        return importlib.util.find_spec("mypy") is not None
    except (ImportError, ValueError):
        return False


def _own_path(reported: str, paths: list[Path]) -> str:
    """The module of *paths* mypy reports an error of at *reported*.

    mypy replays the errors of a module its cache holds unchanged with the
    path the module had when it was cached, so a module of the same name and
    contents elsewhere (a test's temporary package, another checkout) is
    reported at that path: the module given whose path ends the same way,
    the longest way, is meant.
    """
    if any(str(p) == reported for p in paths):
        return reported
    parts = Path(reported).parts

    def shared(path: Path) -> int:
        n = 0
        for mine, theirs in zip(reversed(path.parts), reversed(parts)):
            if mine != theirs:
                break
            n += 1
        return n

    best = max(paths, key=shared)
    return str(best) if shared(best) > 0 else reported


def find_none_misuses(
    paths: list[Path], timeout: float = 600.0
) -> list[NoneFinding] | None:
    """What mypy reports of a ``None`` misused in the modules *paths*.

    ``None`` when there is no verdict: mypy is not installed, it ran out of
    time, or it failed without a report (a crash, a syntax error the compile
    reports anyway).
    """
    if not paths or not mypy_available():
        return None
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file=",
        "--follow-imports=silent",
        "--ignore-missing-imports",
        "--strict-optional",
        # The body of a function without annotations too: the check compiles
        # every function of the scenario's modules.
        "--check-untyped-defs",
        "--show-column-numbers",
        "--show-error-codes",
        "--no-pretty",
        "--no-error-summary",
        "--hide-error-context",
        "--no-color-output",
        f"--cache-dir={_cache_dir()}",
        *(str(p) for p in paths),
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode not in (0, 1):
        return None  # 2: mypy could not check them (a crash, bad arguments)
    findings: list[NoneFinding] = []
    current: NoneFinding | None = None
    reported = ""
    for raw in proc.stdout.splitlines():
        match = _LINE.match(raw.strip())
        if match is None:
            continue
        if match["severity"] == "note":
            if current is not None and match["path"] == reported:
                current = NoneFinding(
                    current.path,
                    current.line,
                    current.column,
                    current.message,
                    current.code,
                    (*current.notes, match["message"]),
                )
                findings[-1] = current
            continue
        current = None
        code = match["code"] or ""
        if code not in NONE_CODES or "None" not in match["message"]:
            continue
        reported = match["path"]
        current = NoneFinding(
            _own_path(reported, paths),
            int(match["line"]),
            int(match["col"]) if match["col"] else None,
            match["message"],
            code,
        )
        findings.append(current)
    return findings
