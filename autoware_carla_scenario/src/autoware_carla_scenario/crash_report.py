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
* anything else ``scenario`` raises before it exits.

A refusal of the configuration (a bad value, a scenario the static check
refuses) is the user's to fix, not a crash, and is not reported.

A **native crash** (a ``SIGSEGV`` in the CARLA client, an abort) ends the
process before Python can write anything.  So, once the run's output
directory is known, the report is prepared ahead: ``crash_reports/
native-<time>-<pid>/`` holds ``context.json``, ``config.yaml`` and
``repro.sh`` from the start, and :mod:`faulthandler` writes the Python stack
of every thread to its ``faulthandler.log`` if the process dies of a fatal
signal.  The directory is removed when the run ends normally; one left with
an empty ``faulthandler.log`` is the trace of a process killed without a
chance to write (``SIGKILL``, the OOM killer).

Sending: with ``SENTRY_DSN`` set (and the usual ``SENTRY_ENVIRONMENT``,
``SENTRY_RELEASE``), every report is also sent to that Sentry project, its
files as attachments.  Without it nothing leaves the machine.  The event
holds local variables and the config, so set it only for a project you trust
with them.

Environment variables:

* ``SCENARIO_CRASH_REPORT=0`` (or ``false``, ``no``, ``off``) turns all of
  this off;
* ``SCENARIO_CRASH_REPORT_DIR`` names the directory reports go to before a
  run's output directory is known (``outputs/crash_reports`` otherwise), and
  prepares native-crash reports there from the start.
"""

from __future__ import annotations

import collections
import faulthandler
import json
import logging
import os
import platform
import re
import shlex
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.integrations.argv import ArgvIntegration
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.modules import ModulesIntegration
from sentry_sdk.transport import HttpTransport, Transport

__all__ = [
    "CAPTURED_LOGGERS",
    "CRASH_REPORTS_DIRNAME",
    "CRASH_REPORT_DIR_ENV",
    "CRASH_REPORT_ENV",
    "DSN_ENV",
    "capture",
    "enabled",
    "install",
    "installed",
    "reporting",
    "run_context",
    "set_report_dir",
    "uninstall",
]

logger = logging.getLogger(__name__)

#: Set to ``0``, ``false``, ``no`` or ``off`` to turn crash reports off.
CRASH_REPORT_ENV = "SCENARIO_CRASH_REPORT"
#: Where reports go before a run's output directory is known.
CRASH_REPORT_DIR_ENV = "SCENARIO_CRASH_REPORT_DIR"
#: The Sentry project reports are also sent to (the Sentry SDK's own name).
DSN_ENV = "SENTRY_DSN"
#: The directory, in a run's output directory, its reports go to.
CRASH_REPORTS_DIRNAME = "crash_reports"
#: Loggers whose records with an exception attached at ``WARNING`` or above
#: are captured: the queue's failed attempts, which it retries or (in a
#: batch) turns into a failed result.
CAPTURED_LOGGERS: tuple[str, ...] = ("autoware_carla_scenario.scenario_queue",)

#: Where reports go when nothing else says.
_DEFAULT_REPORT_DIR = Path("outputs") / CRASH_REPORTS_DIRNAME
#: How many log records ``log.txt`` keeps.
_LOG_LINES = 5000
#: The environment variables ``context.json`` and ``repro.sh`` keep, by
#: prefix: those that change what a run does.  Any whose name looks like a
#: secret is left out (:data:`_SECRET_NAME`).
_ENV_PREFIXES = (
    "CARLA",
    "SCENARIO",
    "SWEEP_",
    "HYDRA_",
    "TYPESAFE_CARLA",
    "AUTOWARE",
    "ROS_",
    "RMW_",
    "CYCLONEDDS",
    "SUMO_HOME",
    "SLURM_",
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
)
_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|DSN", re.I)
_FALSE = {"0", "false", "no", "off"}


def enabled() -> bool:
    """Whether crash reports are on (:data:`CRASH_REPORT_ENV`)."""
    return os.environ.get(CRASH_REPORT_ENV, "").strip().lower() not in _FALSE


def _package_version() -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("autoware-carla-scenario")
    except Exception:  # noqa: BLE001 - a source checkout without metadata
        return "unknown"


def _kept_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_ENV_PREFIXES) and not _SECRET_NAME.search(name)
    }


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
    #: Whether a native-crash report is prepared in :attr:`report_dir`.
    native: bool = False


class _LogBuffer(logging.Handler):
    """Keeps the last :data:`_LOG_LINES` log records, formatted."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.lines: collections.deque[str] = collections.deque(maxlen=_LOG_LINES)
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
            self.upstream.capture_envelope(envelope)

    def flush(self, timeout: float, callback: Any | None = None) -> None:
        if self.upstream is not None:
            self.upstream.flush(timeout, callback)

    def kill(self) -> None:
        if self.upstream is not None:
            self.upstream.kill()


@dataclass
class _NativeReport:
    """The report a native crash would leave (:meth:`_Reporter.arm_native`)."""

    directory: Path
    log: IO[str]
    pid: int = field(default_factory=os.getpid)


class _Reporter:
    """The installed crash reporter (one per process)."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        self.cwd = Path.cwd()
        self.started = time.time()
        env_dir = os.environ.get(CRASH_REPORT_DIR_ENV)
        self.default_dir = Path(env_dir) if env_dir else _DEFAULT_REPORT_DIR
        self.context = _RunContext(report_dir=self.default_dir)
        self.log_buffer = _LogBuffer()
        self.capture_handler = _CaptureHandler()
        self.lock = threading.Lock()
        #: ids of the exceptions reported, so one is reported once, and an
        #: exception raised from one (the queue's "failed after N attempts")
        #: is not reported again.
        self.reported: set[int] = set()
        #: The report directory of each event written, by event id.
        self.written: dict[str, Path] = {}
        self.native: _NativeReport | None = None

    # -- what a report holds -------------------------------------------------

    def repro_command(self) -> list[str]:
        """The command that runs the same case again."""
        overrides = self.context.overrides
        if overrides is None:
            return list(self.argv)
        program = self.argv[0] if self.argv else "scenario"
        if Path(program).name in ("run.py", "__main__.py", "-c", "-m"):
            program = "scenario"
        return [program, *overrides]

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
            "platform": platform.platform(),
            "machine": platform.machine(),
            "environment": _kept_environment(),
        }

    def repro_script(self) -> bytes:
        env = _kept_environment()
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
        files = {
            "context.json": json.dumps(self.context_dict(), indent=2).encode(),
            "repro.sh": self.repro_script(),
        }
        if self.context.config_yaml is not None:
            files["config.yaml"] = self.context.config_yaml.encode()
        return files

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
        directory, and point :mod:`faulthandler` at it."""
        self.disarm_native()
        directory = self.context.report_dir / f"native-{_timestamp()}-{os.getpid()}"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            for name, data in self.context_files().items():
                _write(directory / name, data)
            log = open(directory / "faulthandler.log", "w")  # noqa: SIM115
        except OSError:
            logger.warning("Cannot prepare the native crash report", exc_info=True)
            faulthandler.enable(all_threads=True)
            return
        faulthandler.enable(file=log, all_threads=True)
        self.native = _NativeReport(directory, log)

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
    reporter = _Reporter(sys.argv if argv is None else argv)
    dsn = os.environ.get(DSN_ENV) or None
    transport = _LocalTransport(reporter)
    sentry_sdk.init(
        dsn=dsn,
        transport=transport,
        release=os.environ.get("SENTRY_RELEASE")
        or f"autoware-carla-scenario@{_package_version()}",
        default_integrations=False,
        auto_enabling_integrations=False,
        integrations=[
            # Log records become breadcrumbs, never events of their own: the
            # framework logs errors it handles.
            LoggingIntegration(level=logging.INFO, event_level=None),
            DedupeIntegration(),
            ModulesIntegration(),
            ArgvIntegration(),
        ],
        include_local_variables=True,
        max_breadcrumbs=200,
    )
    if dsn:
        transport.upstream = HttpTransport(sentry_sdk.get_client().options)
    logging.getLogger().addHandler(reporter.log_buffer)
    if logging.getLogger().level > logging.INFO or (
        logging.getLogger().level == logging.NOTSET
    ):
        # The buffer sees what reaches the root logger: INFO and up, as
        # logging.basicConfig() sets later anyway.
        logging.getLogger().setLevel(logging.INFO)
    for name in CAPTURED_LOGGERS:
        logging.getLogger(name).addHandler(reporter.capture_handler)
    _REPORTER[0] = reporter
    if os.environ.get(CRASH_REPORT_DIR_ENV):
        reporter.context.native = True
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


def set_report_dir(directory: Path) -> None:
    """Send the process's reports to *directory* from now on (a batch's
    ``crash_reports/``), and prepare a native-crash report there."""
    reporter = _REPORTER[0]
    if reporter is None:
        return
    reporter.context = _RunContext(report_dir=directory, native=True)
    reporter.arm_native()


@contextmanager
def run_context(
    output_dir: Path,
    *,
    overrides: Sequence[str] | None = None,
    config_yaml: str | None = None,
) -> Iterator[None]:
    """Report what the ``with`` block raises as a run's: in
    ``<output_dir>/crash_reports/``, with its *overrides* (what ``repro.sh``
    runs) and its composed config (*config_yaml*).  A native-crash report is
    prepared there for the block.  Nothing when no reporter is installed."""
    reporter = _REPORTER[0]
    if reporter is None:
        yield
        return
    previous = reporter.context
    reporter.context = _RunContext(
        report_dir=Path(output_dir) / CRASH_REPORTS_DIRNAME,
        output_dir=Path(output_dir),
        overrides=None if overrides is None else [str(o) for o in overrides],
        config_yaml=config_yaml,
        native=True,
    )
    reporter.arm_native()
    try:
        yield
    finally:
        reporter.disarm_native()
        reporter.context = previous
        if _REPORTER[0] is reporter and previous.native:
            reporter.arm_native()


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
        if any(id(e) in reporter.reported for e in _exception_chain(exc)):
            return None
        reporter.reported.add(id(exc))
    try:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        files = reporter.context_files()
        files["traceback.txt"] = trace.encode("utf-8", "replace")
        with sentry_sdk.new_scope() as scope:
            for name, data in files.items():
                scope.add_attachment(bytes=data, filename=name)
            scope.add_attachment(bytes=reporter.log_buffer.text, filename="log.txt")
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
