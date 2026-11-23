"""What a crash report holds and how large it gets: the settings and their parsers.

The environment variables that turn crash reports off, place them, send them
and choose what they hold (:data:`CRASH_REPORT_ENV`, :data:`ITEMS_ENV`, ...),
the items a report can hold (:data:`ITEMS`), the :class:`Settings` read from
them and the parsers of their values (a list of items, a size, a count), the
cut of a file to the size limit (:func:`_truncate`), a name made safe for a
file (:func:`_safe_name`), the environment variables a report keeps
(:data:`_ENV_PATTERNS`), and the command and script that run a crashed case
again (:func:`repro_command`, :func:`repro_script`, #2364).

Split out of :mod:`.crash_report`, which re-exports every name moved here and
keeps what reads the environment (``read_settings()``) and writes and sends
the reports (sentry_sdk, faulthandler), so that the library check compiles it
(``docs/typecheck.md``, "Checking the framework itself").
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CRASH_REPORT_DIR_ENV",
    "CRASH_REPORT_ENV",
    "DSN_ENV",
    "ITEMS",
    "ITEMS_ENV",
    "LOG_LINES_ENV",
    "MAX_FILE_SIZE_ENV",
    "MAX_REPORTS_ENV",
    "SEND_ITEMS_ENV",
    "Settings",
]

#: Set to ``0``, ``false``, ``no`` or ``off`` to turn crash reports off.
CRASH_REPORT_ENV = "SCENARIO_CRASH_REPORT"
#: Where reports go before a run's output directory is known.
CRASH_REPORT_DIR_ENV = "SCENARIO_CRASH_REPORT_DIR"
#: The project reports are also sent to (a Sentry DSN).  Nothing is sent
#: without it: not the SDK's own ``SENTRY_DSN``, which a shell may set for
#: another project.
DSN_ENV = "SCENARIO_CRASH_REPORT_DSN"

#: What a report holds besides the exception, ``traceback.txt`` and
#: ``context.json`` (always): the frames' local variables (``locals``),
#: ``log.txt`` and the log breadcrumbs (``log``), ``config.yaml``
#: (``config``), the environment variables in ``context.json`` and
#: ``repro.sh`` (``environment``), the installed packages (``packages``) and
#: ``repro.sh`` itself (``repro``).
ITEMS: tuple[str, ...] = ("locals", "log", "config", "environment", "packages", "repro")
#: What reports hold (:data:`ITEMS`): a comma-separated list, ``all``,
#: ``none``, or ``all,-locals`` (all but those named with ``-``).
ITEMS_ENV = "SCENARIO_CRASH_REPORT_ITEMS"
#: What of a report is sent, with a DSN: the same form; only what reports
#: hold can be sent.  What :data:`ITEMS_ENV` says when unset.
SEND_ITEMS_ENV = "SCENARIO_CRASH_REPORT_SEND_ITEMS"
#: The largest file of a report other than ``event.json`` (``1MiB``,
#: ``512KB``, a number of bytes; ``0``: no limit): a longer one is cut,
#: ``log.txt`` keeping its end.
MAX_FILE_SIZE_ENV = "SCENARIO_CRASH_REPORT_MAX_FILE_SIZE"
#: How many reports a process writes at most (``0``: no limit).
MAX_REPORTS_ENV = "SCENARIO_CRASH_REPORT_MAX_REPORTS"
#: How many log records ``log.txt`` keeps.
LOG_LINES_ENV = "SCENARIO_CRASH_REPORT_LOG_LINES"

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gib": 1024**3,
}


@dataclass(frozen=True)
class Settings:
    """What a crash report holds, what of it is sent, and how large it gets.

    Attributes:
        items: What reports hold (:data:`ITEMS`).
        send_items: What of them is sent, with a DSN (within :attr:`items`).
        max_file_size: The largest ``log.txt``, ``traceback.txt`` and
            ``config.yaml`` of a report, in bytes; ``None``: no limit
            (``context.json`` and ``repro.sh`` are never cut).  1 MiB:
            ``log.txt`` at its default length fits, a config is a few
            kilobytes.
        max_reports: How many reports a process writes at most; ``None``: no
            limit.  20: a batch whose every case fails on every attempt does
            not fill the disk, nor a Sentry project's quota.
        log_lines: How many log records ``log.txt`` keeps.  5000: a few
            minutes of a run's log, the start of the case included.
    """

    items: frozenset[str] = frozenset(ITEMS)
    send_items: frozenset[str] = frozenset(ITEMS)
    max_file_size: int | None = 1024**2
    max_reports: int | None = 20
    log_lines: int = 5000


def _parse_items(name: str, text: str) -> frozenset[str]:
    chosen: set[str] = set()
    for word in (w.strip().lower() for w in text.split(",")):
        if not word:
            continue
        if word == "all":
            chosen.update(ITEMS)
        elif word == "none":
            chosen.clear()
        elif word.startswith("-") and word[1:] in ITEMS:
            chosen.discard(word[1:])
        elif word in ITEMS:
            chosen.add(word)
        else:
            raise ValueError(
                f"{name}: unknown item {word!r} (one of {', '.join(ITEMS)}, "
                "all, none, or -<item>)"
            )
    return frozenset(chosen)


def _parse_size(name: str, text: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*", text.lower())
    if match is None or match.group(2) not in _SIZE_UNITS:
        raise ValueError(f"{name}: {text!r} is not a size (1MiB, 512KB, 1048576)")
    size = int(float(match.group(1)) * _SIZE_UNITS[match.group(2)])
    if size == 0:
        return None
    return size


def _parse_count(name: str, text: str, *, zero_is_none: bool) -> int | None:
    try:
        count = int(text.strip())
    except ValueError:
        count = -1
    if count < 0 or (count == 0 and not zero_is_none):
        raise ValueError(f"{name}: {text!r} is not a whole number of at least 1")
    if count == 0:
        return None
    return count


def _truncate(data: bytes, limit: int | None, *, keep_end: bool = False) -> bytes:
    """*data*, cut to at most *limit* bytes with a line saying so."""
    if limit is None or len(data) <= limit:
        return data
    note = f"[... {len(data) - limit} bytes cut (crash report size limit) ...]\n"
    marker = note.encode()
    if keep_end:
        room = max(limit - len(marker), 0)
        return marker + data[len(data) - room :]
    room = max(limit - len(marker) - 1, 0)
    return data[:room] + b"\n" + marker


#: The environment variables ``context.json`` and ``repro.sh`` keep: those
#: that change what a run does.  One whose name looks like a secret is left
#: out (``slurm.settings.SENSITIVE_NAME_PATTERNS``).
_ENV_PATTERNS = (
    "CARLA*",
    "SCENARIO*",
    "SWEEP_*",
    "HYDRA_*",
    "TYPESAFE_CARLA*",
    "AUTOWARE*",
    "ROS_*",
    "RMW_*",
    "CYCLONEDDS*",
    "SUMO_HOME",
    "SLURM_*",
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "file"


def repro_command(argv: list[str], overrides: Optional[list[str]]) -> list[str]:
    """The command that runs the same case again: the `scenario` CLI with
    the case's *overrides*, or, before they are known, the process's *argv*."""
    if overrides is None:
        return list(argv)
    # Overrides are the `scenario` CLI's, whatever runs them here (a SLURM
    # task's worker, a script calling main()).
    return ["scenario"] + overrides


def repro_overrides(scenario: str, overrides: list[str]) -> list[str]:
    """What ``repro.sh`` of a crash report of a batch's case runs
    ``scenario`` with: its *scenario* and *overrides*, without the batch's
    launcher (``launcher=``, ``launcher.*``), so that it runs again here, as
    a single run (``slurm.worker.repro_overrides()``, #2386)."""
    kept = [f"scenario={scenario}"]
    for o in overrides:
        if not o.lstrip("+~").startswith("launcher"):
            kept.append(o)
    return kept


def repro_script(
    argv: list[str], cwd: str, env: dict[str, str], command: list[str]
) -> bytes:
    """``repro.sh``: from *cwd*, with the kept environment *env*, run
    *command* (:func:`repro_command`), naming the original command line
    *argv* in a comment: each of its lines a comment line, so that an
    argument holding a newline does not end the comment."""
    command_line = "\n# ".join(shlex.join(argv).split("\n"))
    lines = [
        "#!/bin/sh",
        "# Runs the case this crash report comes from again.",
        f"# Command line: {command_line}",
        f"cd {shlex.quote(cwd)} || exit 1",
    ]
    for name, value in env.items():
        lines.append(f"export {name}={shlex.quote(value)}")
    lines.append(f'exec {shlex.join(command)} "$@"')
    return ("\n".join(lines) + "\n").encode()
