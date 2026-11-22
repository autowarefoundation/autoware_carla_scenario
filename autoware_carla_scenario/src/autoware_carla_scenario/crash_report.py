"""Crash reports: what a run that raised or crashed leaves to reproduce it.

The ``scenario`` CLI installs this as it starts (:func:`reporting`).  It is
built on the Sentry SDK (``sentry-sdk``): an error is captured as a Sentry
event -- the exception with every frame and its local variables, the recent
log records as breadcrumbs, the installed packages -- and written, with what
it takes to run the same thing again, as a **crash report**: a directory of
its own under ``crash_reports/`` in the run's output directory::

    crash_reports/20261122T101500Z-RuntimeError-1a2b3c4d/
        event.json       # the Sentry event, as it would be sent
        traceback.txt    # the traceback, chained exceptions included
        log.txt          # the last log records of the process
        context.json     # command line, overrides, versions, environment
        config.yaml      # the run's composed config (a Hydra run only)
        repro.sh         # the command that runs the same case again

What is captured:

* a run that raised (a single run, or a case of a ``--multirun`` that Hydra's
  basic sweeper goes on from);
* every failed attempt the scenario queue retries
  (:data:`CAPTURED_LOGGERS`): a batch goes on from a case that raised and
  keeps only a line of it in its result;
* a SLURM task that raised, and its cases' failed attempts
  (:mod:`~autoware_carla_scenario.slurm.worker`), in the batch's directory;
* anything else ``scenario`` raises before it exits.

A refusal of the configuration (a bad value, a scenario the static check
refuses) is the user's to fix, not a crash, and is not reported.

A **native crash** (a ``SIGSEGV`` in the CARLA client, an abort) ends the
process before Python can write anything.  So, once the run's output
directory is known, the report is prepared ahead: ``crash_reports/
native-<time>-<host>-<pid>/`` holds ``context.json``, ``config.yaml`` and
``repro.sh`` from the start, and :mod:`faulthandler` writes the Python stack
of every thread to its ``faulthandler.log`` if the process dies of a fatal
signal.  The directory is removed when the run ends normally; one left with
an empty ``faulthandler.log`` is the trace of a process killed without a
chance to write (``SIGKILL``, the OOM killer).

Sending: only with ``SCENARIO_CRASH_REPORT_DSN`` set (:data:`DSN_ENV`) is
every report also sent to that project, its files as attachments.  The Sentry
SDK's own ``SENTRY_DSN`` and ``SENTRY_SPOTLIGHT`` are ignored: a shell that
sets them for another project sends nothing of this one.  Any server that speaks Sentry's protocol takes it; the one
docs/crash_reports.md recommends is GlitchTip, an open-source (MIT) server
hosted by the user.  Without a DSN nothing leaves the machine.  The event
holds local variables and the config, so set it only for a project you trust
with them.

Environment variables:

* ``SCENARIO_CRASH_REPORT=0`` (or ``false``, ``no``, ``off``) turns all of
  this off;
* ``SCENARIO_CRASH_REPORT_DIR`` names the directory reports go to before a
  run's output directory is known (``outputs/crash_reports`` otherwise), and
  prepares native-crash reports there from the start;
* ``SCENARIO_CRASH_REPORT_ITEMS`` and ``SCENARIO_CRASH_REPORT_SEND_ITEMS``
  choose what a report holds and what of it is sent (:data:`ITEMS`), and
  ``SCENARIO_CRASH_REPORT_MAX_FILE_SIZE``, ``SCENARIO_CRASH_REPORT_MAX_REPORTS``
  and ``SCENARIO_CRASH_REPORT_LOG_LINES`` bound its size
  (:class:`Settings`, :func:`read_settings`).
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import faulthandler
import functools
import json
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import sentry_sdk
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.integrations.argv import ArgvIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.modules import ModulesIntegration
from sentry_sdk.transport import HttpTransport, Transport

from .config_flags import FALSE_TEXT
from .slurm.settings import snapshot_environment

__all__ = [
    "CAPTURED_LOGGERS",
    "CRASH_REPORTS_DIRNAME",
    "CRASH_REPORT_DIR_ENV",
    "CRASH_REPORT_ENV",
    "DSN_ENV",
    "ITEMS",
    "Settings",
    "capture",
    "enabled",
    "install",
    "installed",
    "read_settings",
    "reporting",
    "run_context",
    "set_output_dir",
    "uninstall",
    "wants",
]

logger = logging.getLogger(__name__)

#: Set to ``0``, ``false``, ``no`` or ``off`` to turn crash reports off.
CRASH_REPORT_ENV = "SCENARIO_CRASH_REPORT"
#: Where reports go before a run's output directory is known.
CRASH_REPORT_DIR_ENV = "SCENARIO_CRASH_REPORT_DIR"
#: The project reports are also sent to (a Sentry DSN).  Nothing is sent
#: without it: not the SDK's own ``SENTRY_DSN``, which a shell may set for
#: another project.
DSN_ENV = "SCENARIO_CRASH_REPORT_DSN"
#: The directory, in a run's output directory, its reports go to.
CRASH_REPORTS_DIRNAME = "crash_reports"
#: Loggers whose records with an exception attached at ``WARNING`` or above
#: are captured: the queue's failed attempts, which it retries or (in a
#: batch) turns into a failed result.
CAPTURED_LOGGERS: tuple[str, ...] = ("autoware_carla_scenario.scenario_queue",)

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

#: The attribute marking an exception reported (:func:`capture`).
_REPORTED = "_autoware_carla_scenario_crash_reported"
#: Where reports go when nothing else says.
_DEFAULT_REPORT_DIR = Path("outputs") / CRASH_REPORTS_DIRNAME
#: Each item's file of a report.
_ITEM_FILES = {"config.yaml": "config", "log.txt": "log", "repro.sh": "repro"}
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
        max_file_size: The largest file of a report other than
            ``event.json``, in bytes; ``None``: no limit.  1 MiB: ``log.txt``
            at its default length fits, a config is a few kilobytes.
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
    return size or None


def _parse_count(name: str, text: str, *, zero_is_none: bool) -> int | None:
    try:
        count = int(text.strip())
    except ValueError:
        count = -1
    if count < 0 or (count == 0 and not zero_is_none):
        raise ValueError(f"{name}: {text!r} is not a whole number of at least 1")
    return count or None


def read_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """The crash reports' :class:`Settings` from the environment (*environ*,
    ``os.environ`` when ``None``).

    A variable that cannot be read is warned about and its default kept: a
    crash report never stops a run.
    """
    env = os.environ if environ is None else environ
    values: dict[str, Any] = {}

    def read(name: str, field_name: str, parse: Any) -> None:
        text = env.get(name)
        if text is None or not text.strip():
            return
        try:
            values[field_name] = parse(name, text)
        except ValueError as exc:
            logger.warning("%s; using the default", exc)

    read(ITEMS_ENV, "items", _parse_items)
    read(SEND_ITEMS_ENV, "send_items", _parse_items)
    read(MAX_FILE_SIZE_ENV, "max_file_size", _parse_size)
    read(
        MAX_REPORTS_ENV,
        "max_reports",
        lambda n, t: _parse_count(n, t, zero_is_none=True),
    )
    read(
        LOG_LINES_ENV,
        "log_lines",
        lambda n, t: _parse_count(n, t, zero_is_none=False),
    )
    settings = dataclasses.replace(Settings(), **values)
    # What is sent is what reports hold unless said, and never more.
    send = values.get("send_items", settings.items)
    return dataclasses.replace(settings, send_items=send & settings.items)


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


def enabled() -> bool:
    """Whether crash reports are on (:data:`CRASH_REPORT_ENV`)."""
    return os.environ.get(CRASH_REPORT_ENV, "").strip().lower() not in FALSE_TEXT


@functools.cache
def _package_version() -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("autoware-carla-scenario")
    except Exception:  # noqa: BLE001 - a source checkout without metadata
        return "unknown"


@functools.cache
def _platform() -> str:
    return platform.platform()


def _kept_environment() -> dict[str, str]:
    return snapshot_environment(_ENV_PATTERNS)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "file"


@dataclass
class _RunContext:
    """What a report says about the run it comes from."""

    #: Where its reports go.
    report_dir: Path
    #: The run's output directory, when it has one.
    output_dir: Path | None = None
    #: Its Hydra overrides (a ``--multirun`` case's own), when it has them.
    overrides: list[str] | None = None
    #: Its composed config, as YAML.
    config_yaml: str | None = None


class _LogBuffer(logging.Handler):
    """Keeps the last *lines* log records, formatted."""

    def __init__(self, lines: int) -> None:
        super().__init__(level=logging.NOTSET)
        self.lines: collections.deque[str] = collections.deque(maxlen=lines)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must not raise
            self.handleError(record)

    def text(self) -> bytes:
        return ("\n".join(self.lines) + "\n").encode("utf-8", "replace")


class _CaptureHandler(logging.Handler):
    """Captures the exception of a record of :data:`CAPTURED_LOGGERS`."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, Exception):
            capture(exc, message=record.getMessage())


class _LocalTransport(Transport):
    """Writes each event, with its attachments, as a report directory, and
    hands it on to Sentry when a DSN is set."""

    def __init__(self, reporter: _Reporter) -> None:
        super().__init__(None)
        self._reporter = reporter
        #: Sends to Sentry: set once the client's options are known.
        self.upstream: Transport | None = None

    def capture_envelope(self, envelope: Envelope) -> None:
        try:
            self._reporter.write_envelope(envelope)
        except Exception:  # noqa: BLE001 - a report must never end the run
            logger.warning("Cannot write the crash report", exc_info=True)
        if self.upstream is not None:
            self.upstream.capture_envelope(
                _for_sending(envelope, self._reporter.settings)
            )

    def flush(self, timeout: float, callback: Any | None = None) -> None:
        if self.upstream is not None:
            self.upstream.flush(timeout, callback)

    def kill(self) -> None:
        if self.upstream is not None:
            self.upstream.kill()


def _for_sending(envelope: Envelope, settings: Settings) -> Envelope:
    """*envelope* with only what :attr:`Settings.send_items` sends."""
    left_out = settings.items - settings.send_items
    if not left_out:
        return envelope
    sent = Envelope(headers=dict(envelope.headers))
    for item in envelope.items:
        event = item.get_event()
        if event is not None:
            sent.add_event(_event_for_sending(event, left_out))
            continue
        name = str(item.headers.get("filename") or "")
        if item.type != "attachment" or not name:
            sent.add_item(item)
            continue
        if _ITEM_FILES.get(name) in left_out:
            continue
        data = item.get_bytes()
        if "environment" in left_out and name == "context.json":
            context = json.loads(data)
            context.pop("environment", None)
            data = json.dumps(context, indent=2).encode()
        elif "environment" in left_out and name == "repro.sh":
            data = b"".join(
                line
                for line in data.splitlines(keepends=True)
                if not line.startswith(b"export ")
            )
        sent.add_item(Item(payload=data, headers=dict(item.headers)))
    return sent


def _event_for_sending(event: Any, left_out: frozenset[str]) -> Any:
    """*event* without the items *left_out* hold in it (a copy)."""
    event = copy.deepcopy(event)
    if "locals" in left_out:
        for key in ("exception", "threads"):
            for value in (event.get(key) or {}).get("values") or []:
                for frame in (value.get("stacktrace") or {}).get("frames") or []:
                    frame.pop("vars", None)
    if "log" in left_out:
        event.pop("breadcrumbs", None)
    if "packages" in left_out:
        event.pop("modules", None)
    return event


@dataclass
class _NativeReport:
    """The report a native crash would leave (:meth:`_Reporter.arm_native`)."""

    directory: Path
    log: IO[str]
    pid: int = field(default_factory=os.getpid)


class _Reporter:
    """The installed crash reporter (one per process)."""

    def __init__(self, argv: Sequence[str], settings: Settings) -> None:
        self.argv = list(argv)
        self.settings = settings
        self.cwd = Path.cwd()
        self.started = time.time()
        env_dir = os.environ.get(CRASH_REPORT_DIR_ENV)
        self.context = _RunContext(
            report_dir=Path(env_dir) if env_dir else _DEFAULT_REPORT_DIR
        )
        self.log_buffer = _LogBuffer(settings.log_lines)
        self.capture_handler = _CaptureHandler()
        self.lock = threading.Lock()
        #: The report directory of each event written, by event id.
        self.written: dict[str, Path] = {}
        #: How many reports were captured (:attr:`Settings.max_reports`).
        self.count = 0
        self.native: _NativeReport | None = None

    # -- what a report holds -------------------------------------------------

    def repro_command(self) -> list[str]:
        """The command that runs the same case again."""
        overrides = self.context.overrides
        if overrides is None:
            return list(self.argv)
        # Overrides are the `scenario` CLI's, whatever runs them here (a SLURM
        # task's worker, a script calling main()).
        return ["scenario", *overrides]

    def context_dict(self) -> dict[str, Any]:
        ctx = self.context
        return {
            "argv": self.argv,
            "cwd": str(self.cwd),
            "repro": self.repro_command(),
            "overrides": ctx.overrides,
            "output_dir": str(ctx.output_dir) if ctx.output_dir else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "pid": os.getpid(),
            "autoware_carla_scenario": _package_version(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": _platform(),
            "machine": platform.machine(),
            **(
                {"environment": _kept_environment()}
                if "environment" in self.settings.items
                else {}
            ),
        }

    def repro_script(self) -> bytes:
        env = _kept_environment() if "environment" in self.settings.items else {}
        lines = [
            "#!/bin/sh",
            "# Runs the case this crash report comes from again.",
            f"# Command line: {shlex.join(self.argv)}",
            f"cd {shlex.quote(str(self.cwd))} || exit 1",
        ]
        lines += [f"export {name}={shlex.quote(value)}" for name, value in env.items()]
        lines.append(f'exec {shlex.join(self.repro_command())} "$@"')
        return ("\n".join(lines) + "\n").encode()

    def context_files(self) -> dict[str, bytes]:
        """The files of a report that say how to run it again."""
        items = self.settings.items
        limit = self.settings.max_file_size
        files = {"context.json": json.dumps(self.context_dict(), indent=2).encode()}
        if "repro" in items:
            files["repro.sh"] = self.repro_script()
        if self.context.config_yaml is not None and "config" in items:
            files["config.yaml"] = self.context.config_yaml.encode()
        return {name: _truncate(data, limit) for name, data in files.items()}

    def log_text(self) -> bytes:
        """``log.txt``, cut to :attr:`Settings.max_file_size` from its start."""
        return _truncate(
            self.log_buffer.text(), self.settings.max_file_size, keep_end=True
        )

    # -- writing -------------------------------------------------------------

    def write_envelope(self, envelope: Envelope) -> None:
        event = envelope.get_event()
        if event is None:
            return
        event_id = str(event.get("event_id") or "")
        exc_type = "error"
        values = (event.get("exception") or {}).get("values") or []
        if values:
            exc_type = str(values[-1].get("type") or exc_type)
        directory = self._new_report_dir(
            f"{_timestamp()}-{_safe_name(exc_type)}-{event_id[:8]}"
        )
        _write(directory / "event.json", json.dumps(event, indent=2, default=str))
        for item in envelope.items:
            name = item.headers.get("filename")
            if item.type == "attachment" and name:
                _write(directory / _safe_name(str(name)), item.get_bytes())
        self.written[event_id] = directory
        logger.error("Crash report written to %s", directory)

    def _new_report_dir(self, name: str) -> Path:
        base = self.context.report_dir
        directory = base / name
        suffix = 1
        while directory.exists():
            suffix += 1
            directory = base / f"{name}-{suffix}"
        directory.mkdir(parents=True)
        return directory

    # -- native crashes ------------------------------------------------------

    def arm_native(self) -> None:
        """Prepare the report a native crash would leave, in the report
        directory, and point :mod:`faulthandler` at it.  One prepared there
        already has its files rewritten for the context, not made again:
        each case of a batch is one, on a filesystem its tasks share."""
        native = self.native
        if (
            native is not None
            and native.pid == os.getpid()
            and native.directory.parent == self.context.report_dir
        ):
            self._write_native_files(native.directory)
            return
        self.disarm_native()
        # The host too: the tasks of a SLURM batch on several nodes share the
        # batch's directory, and their process ids can collide.
        host = _safe_name(socket.gethostname())
        directory = (
            self.context.report_dir / f"native-{_timestamp()}-{host}-{os.getpid()}"
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._write_native_files(directory)
            log = open(directory / "faulthandler.log", "w")  # noqa: SIM115
        except OSError:
            logger.warning("Cannot prepare the native crash report", exc_info=True)
            faulthandler.enable(all_threads=True)
            return
        faulthandler.enable(file=log, all_threads=True)
        self.native = _NativeReport(directory, log)

    def _write_native_files(self, directory: Path) -> None:
        files = self.context_files()
        for name in ("config.yaml", "repro.sh"):
            if name not in files:
                (directory / name).unlink(missing_ok=True)
        for name, data in files.items():
            _write(directory / name, data)

    def disarm_native(self) -> None:
        """Remove the report :meth:`arm_native` prepared, unless a fatal
        signal wrote to it; :mod:`faulthandler` writes to stderr again."""
        native, self.native = self.native, None
        if native is None:
            return
        faulthandler.enable(all_threads=True)
        try:
            native.log.close()
        except OSError:
            pass
        if native.pid != os.getpid():
            # A forked child's copy: the directory is its parent's.
            return
        log_path = native.directory / "faulthandler.log"
        try:
            if log_path.exists() and log_path.stat().st_size > 0:
                return
            shutil.rmtree(native.directory, ignore_errors=True)
            native.directory.parent.rmdir()  # crash_reports/, when empty
        except OSError:
            pass


#: The installed reporter, if any.
_REPORTER: list[_Reporter | None] = [None]


def installed() -> bool:
    """Whether a crash reporter is installed in this process."""
    return _REPORTER[0] is not None


def install(argv: Sequence[str] | None = None) -> bool:
    """Install the crash reporter for this process.

    Args:
        argv: The command line the process was started with
            (``sys.argv`` when ``None``), before anything takes arguments
            off it: what ``repro.sh`` runs outside a Hydra run.

    Returns:
        Whether it is installed: ``False`` when :func:`enabled` says it is
        off, or when it already is (the reporter installed first stays).
    """
    if not enabled() or installed():
        return False
    settings = read_settings()
    reporter = _Reporter(sys.argv if argv is None else argv, settings)
    dsn = os.environ.get(DSN_ENV) or None
    transport = _LocalTransport(reporter)
    integrations: list[Any] = [DedupeIntegration(), ArgvIntegration()]
    if "log" in settings.items:
        # Log records become breadcrumbs, never events of their own: the
        # framework logs errors it handles.
        integrations.append(LoggingIntegration(level=logging.INFO, event_level=None))
    if "packages" in settings.items:
        integrations.append(ModulesIntegration())
    sentry_sdk.init(
        # Never None: the SDK would read SENTRY_DSN then.
        dsn=dsn or "",
        spotlight=False,
        transport=transport,
        release=os.environ.get("SENTRY_RELEASE")
        or f"autoware-carla-scenario@{_package_version()}",
        default_integrations=False,
        auto_enabling_integrations=False,
        integrations=integrations,
        include_local_variables="locals" in settings.items,
        max_breadcrumbs=200 if "log" in settings.items else 0,
    )
    # Not the breadcrumbs of an earlier reporter in this process.
    for scope in (sentry_sdk.get_global_scope(), sentry_sdk.get_isolation_scope()):
        scope.clear_breadcrumbs()
    if dsn:
        transport.upstream = HttpTransport(sentry_sdk.get_client().options)
    if "log" in settings.items:
        # It sees what reaches the root logger: INFO and up once the CLI's
        # logging.basicConfig() has run.
        logging.getLogger().addHandler(reporter.log_buffer)
    for name in CAPTURED_LOGGERS:
        logging.getLogger(name).addHandler(reporter.capture_handler)
    _REPORTER[0] = reporter
    if os.environ.get(CRASH_REPORT_DIR_ENV):
        reporter.arm_native()
    else:
        faulthandler.enable(all_threads=True)
    return True


def uninstall() -> None:
    """Uninstall the crash reporter: flush what is being sent, remove the
    native-crash report a normal end leaves unused.  Nothing when none is
    installed."""
    reporter = _REPORTER[0]
    if reporter is None:
        return
    _REPORTER[0] = None
    reporter.disarm_native()
    faulthandler.disable()
    logging.getLogger().removeHandler(reporter.log_buffer)
    for name in CAPTURED_LOGGERS:
        logging.getLogger(name).removeHandler(reporter.capture_handler)
    client = sentry_sdk.get_client()
    try:
        client.flush(timeout=5.0)
    finally:
        client.close()


@contextmanager
def reporting(
    argv: Sequence[str] | None = None,
    *,
    ignore: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Install the crash reporter for the ``with`` block (:func:`install`),
    capture what the block raises (an :class:`Exception`, not ``SystemExit``
    or ``KeyboardInterrupt``, nor one of *ignore*: a refusal the user is to
    fix) and uninstall it when the block ends."""
    if not install(argv):
        yield
        return
    try:
        yield
    except Exception as exc:
        if not isinstance(exc, ignore):
            capture(exc)
        raise
    finally:
        uninstall()


def wants(item: str) -> bool:
    """Whether a reporter is installed and its reports hold *item*
    (:data:`ITEMS`): what a caller need not compute otherwise (the config's
    YAML, for ``config``)."""
    reporter = _REPORTER[0]
    return reporter is not None and item in reporter.settings.items


def set_output_dir(output_dir: Path) -> None:
    """Send the process's reports to ``<output_dir>/crash_reports/`` from now
    on (a batch's), and prepare a native-crash report there.  Nothing when no
    reporter is installed."""
    reporter = _REPORTER[0]
    if reporter is None:
        return
    reporter.context = _RunContext(report_dir=Path(output_dir) / CRASH_REPORTS_DIRNAME)
    reporter.arm_native()


@contextmanager
def run_context(
    output_dir: Path,
    *,
    overrides: Sequence[str] | None = None,
    config_yaml: str | None = None,
) -> Iterator[None]:
    """Reports captured in the ``with`` block (:func:`capture`) are a run's:
    in ``<output_dir>/crash_reports/``, with its *overrides* (what
    ``repro.sh`` runs ``scenario`` with) and its composed config
    (*config_yaml*).  A native-crash report is prepared there for the block.
    Nothing when no reporter is installed; the block's caller captures what
    it raises."""
    reporter = _REPORTER[0]
    if reporter is None:
        yield
        return
    previous = reporter.context
    # A native-crash report prepared before (a batch's) is again afterwards.
    rearm = reporter.native is not None
    reporter.context = _RunContext(
        report_dir=Path(output_dir) / CRASH_REPORTS_DIRNAME,
        output_dir=Path(output_dir),
        overrides=None if overrides is None else [str(o) for o in overrides],
        config_yaml=config_yaml,
    )
    reporter.arm_native()
    try:
        yield
    finally:
        reporter.context = previous
        if rearm and _REPORTER[0] is reporter:
            reporter.arm_native()
        else:
            reporter.disarm_native()


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def capture(exc: BaseException, *, message: str | None = None) -> Path | None:
    """Write a crash report of *exc* (and send it, with a DSN).

    Once per exception: *exc* is not reported again, nor is an exception
    raised from a reported one (``raise ... from``, or while handling it).

    Args:
        exc: What was raised.
        message: What was logged with it, if anything.

    Returns:
        The report's directory, or ``None`` when no reporter is installed,
        *exc* was reported already or the report could not be written.
        Never raises.
    """
    reporter = _REPORTER[0]
    if reporter is None:
        return None
    with reporter.lock:
        # Marked on the exception itself, not kept by id: an id is reused
        # once the exception is gone, and keeping it would keep its frames.
        if any(getattr(e, _REPORTED, False) for e in _exception_chain(exc)):
            return None
        try:
            setattr(exc, _REPORTED, True)
        except (AttributeError, TypeError):
            pass  # an exception without attributes: reported each time
        limit = reporter.settings.max_reports
        if limit is not None and reporter.count >= limit:
            if reporter.count == limit:
                logger.warning(
                    "%d crash reports written: no more from this process (%s)",
                    limit,
                    MAX_REPORTS_ENV,
                )
            reporter.count += 1
            return None
        reporter.count += 1
    try:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        files = reporter.context_files()
        files["traceback.txt"] = _truncate(
            trace.encode("utf-8", "replace"),
            reporter.settings.max_file_size,
            keep_end=True,
        )
        with sentry_sdk.new_scope() as scope:
            for name, data in files.items():
                scope.add_attachment(bytes=data, filename=name)
            if "log" in reporter.settings.items:
                scope.add_attachment(bytes=reporter.log_text, filename="log.txt")
            ctx = reporter.context
            scope.set_context(
                "run",
                {
                    "output_dir": str(ctx.output_dir) if ctx.output_dir else None,
                    "overrides": ctx.overrides,
                    "repro": shlex.join(reporter.repro_command()),
                },
            )
            if message:
                scope.set_extra("message", message)
            event_id = sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001 - a report must never end the run
        logger.warning("Cannot capture the crash report", exc_info=True)
        return None
    if event_id is None:
        return None
    return reporter.written.get(event_id)


def _write(path: Path, data: str | bytes) -> None:
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    path.write_bytes(data)
    if path.name.endswith(".sh"):
        path.chmod(0o755)
