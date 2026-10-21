"""Rerun recordings of a run's trajectory logs (the ``rerun`` extra).

Each run writes ``{stem}.trajectory.jsonl``; the separate
``autoware-carla-scenario-rerun`` package converts it into ``{stem}.rrd``, a
recording for the Rerun viewer.  With ``rerun.enabled=true`` the runner hands
each log to a :class:`RerunConverter` as soon as the log is closed: the
conversion runs on a background thread (the converter releases the GIL) while
the next case runs, and the run waits for it at the end and reports every
failure with the case it belongs to.

:func:`convert_outputs` converts the logs of a finished run instead::

    from autoware_carla_scenario.rerun import convert_outputs
    convert_outputs("outputs/2026-10-18/15-58-45")

or, from a shell, ``python -m autoware_carla_scenario.rerun <output dir>``.
Neither needs CARLA.  Both need the extra:
``pip install 'autoware-carla-scenario[rerun]'``.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union

__all__ = [
    "INSTALL_HINT",
    "RECORDING_SUFFIX",
    "RerunConversion",
    "RerunConverter",
    "RerunUnavailableError",
    "convert_outputs",
    "main",
    "recording_path",
    "report",
    "require_converter",
]

logger = logging.getLogger(__name__)

#: How to install what this module needs.
INSTALL_HINT = "pip install 'autoware-carla-scenario[rerun]'"

#: The suffix of the log a recording is made from, and of the recording.
TRAJECTORY_SUFFIX = ".trajectory.jsonl"
RECORDING_SUFFIX = ".rrd"

#: The package that does the conversion.
_CONVERTER_MODULE = "autoware_carla_scenario_rerun"


class RerunUnavailableError(RuntimeError):
    """``rerun.enabled`` is set, but the converter is not installed."""


def require_converter() -> ModuleType:
    """Return the converter package, or raise :class:`RerunUnavailableError`
    saying how to install it.

    Called before a run starts, so a run asked for recordings it cannot make
    is refused before it spends any time on CARLA.
    """
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(_CONVERTER_MODULE)
    except ImportError as exc:
        raise RerunUnavailableError(
            "rerun.enabled is set, but autoware-carla-scenario-rerun, which "
            f"makes the Rerun recordings, is not installed: {INSTALL_HINT} "
            "(or set rerun.enabled=false)"
        ) from exc


def _label(trajectory: Path) -> str:
    """The run *trajectory* is the log of: its file stem (``X`` of
    ``X.trajectory.jsonl``)."""
    name = trajectory.name
    if name.endswith(TRAJECTORY_SUFFIX):
        return name[: -len(TRAJECTORY_SUFFIX)]
    return trajectory.stem


def recording_path(trajectory: Union[str, Path]) -> Path:
    """The recording made of *trajectory*: ``X.rrd`` beside
    ``X.trajectory.jsonl``, where the converter writes it by default."""
    path = Path(trajectory)
    return path.with_name(_label(path) + RECORDING_SUFFIX)


@dataclass(frozen=True)
class RerunConversion:
    """How converting one trajectory log went.

    Attributes:
        label: The run the log belongs to (its file stem, ``CutInScenario-3``).
        trajectory: The log.
        recording: The recording made of it, or that would have been.
        error: Why the conversion failed (``ExceptionType: message``), or
            ``None`` when it did not.
        stats: What the converter returned (``Stats``): ticks, actors, the
            record types and fields it skipped, ...; ``None`` on failure.
    """

    label: str
    trajectory: Path
    recording: Path
    error: Optional[str] = None
    stats: Optional[dict[str, Any]] = field(default=None, compare=False)

    @property
    def ok(self) -> bool:
        """Whether the recording was written."""
        return self.error is None


@dataclass
class _Pending:
    label: str
    trajectory: Path
    recording: Path
    future: "Future[Any]"


class RerunConverter:
    """Converts trajectory logs into Rerun recordings, one at a time, on a
    background thread.

    :meth:`submit` returns at once; :meth:`wait` waits for every log submitted
    so far and returns how each went, in the order they were submitted.  A
    log that fails does not stop the others: its :class:`RerunConversion`
    carries the error, and :meth:`wait` logs it with the run's label.

    Use it as a context manager, or call :meth:`close`, so its thread ends.
    """

    _module: Any
    _strict: bool
    _executor: Optional[ThreadPoolExecutor]
    _pending: list[_Pending]
    _done: list[RerunConversion]
    _lidar_max_points: Optional[int]
    _lidar_sdk: Any
    _lidar: Any

    def __init__(
        self,
        *,
        strict: bool = False,
        converter: Any = None,
        lidar_max_points: Optional[int] = None,
        lidar_sdk: Any = None,
    ) -> None:
        """Args:
        strict: Refuse a log with a record type or a field the converter
            does not know (it raises, and the conversion fails), rather
            than skip it with a warning.
        converter: What converts a log (``convert(jsonl, out_rrd, *,
            strict)``); the ``autoware_carla_scenario_rerun`` package when
            ``None``.
        lidar_max_points: Also record each run's LiDAR sweeps
            (``rerun.lidar``, :mod:`.rerun_lidar`), at most this many points
            a sweep (0: all); ``None`` records none.
        lidar_sdk: The Rerun SDK the sweeps are written with; ``rerun``
            when ``None``.

        Raises:
            RerunUnavailableError: *converter* is ``None`` and the package
                is not installed.
            LidarRerunUnavailableError: LiDAR sweeps are asked for and the
                Rerun SDK is not installed.
        """
        self._module = converter if converter is not None else require_converter()
        self._strict = strict
        self._lidar_max_points = lidar_max_points
        self._lidar_sdk = None
        if lidar_max_points is not None:
            from .rerun_lidar import require_sdk  # noqa: PLC0415

            self._lidar_sdk = lidar_sdk if lidar_sdk is not None else require_sdk()
        self._lidar = None
        # One worker: a log converts in milliseconds, and a single thread
        # keeps the conversions in the order the runs ended.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rerun-convert"
        )
        self._pending = []
        self._done = []

    @property
    def strict(self) -> bool:
        """Whether a log with what the converter does not know fails."""
        return self._strict

    @property
    def records_lidar(self) -> bool:
        """Whether each run's LiDAR sweeps are recorded too (``rerun.lidar``)."""
        return self._lidar_max_points is not None

    def start_lidar(self, trajectory: Path, label: str, start_frame: int) -> None:
        """Record the LiDAR sweeps of the run *trajectory* is the log of, from
        *start_frame* (the scenario clock's first), into ``{label}.lidar.rrd``
        beside it -- when :attr:`records_lidar`; otherwise nothing.

        The recording is the active one (:func:`.rerun_lidar.active_recorder`)
        until :meth:`stop_lidar`.
        """
        if self._lidar_max_points is None:
            return
        self.stop_lidar()
        from .rerun_lidar import LidarRecorder, lidar_recording_path  # noqa: PLC0415

        self._lidar = LidarRecorder(
            lidar_recording_path(Path(trajectory).parent, label),
            label,
            start_frame,
            self._lidar_max_points,
            sdk=self._lidar_sdk,
        ).start()

    def stop_lidar(self) -> None:
        """Close the run's LiDAR recording, if one is open."""
        recorder, self._lidar = self._lidar, None
        if recorder is not None:
            try:
                recorder.close()
            except Exception:  # noqa: BLE001 - a lost recording is not a lost run
                logger.exception("closing the LiDAR recording %s failed", recorder.path)

    def submit(self, trajectory: Path, label: str) -> Path:
        """Convert *trajectory* in the background; return the recording it
        is written to (:func:`recording_path`)."""
        if self._executor is None:
            raise RuntimeError("RerunConverter is closed")
        recording = recording_path(trajectory)
        lidar: Optional[Path] = None
        if self._lidar_max_points is not None:
            from .rerun_lidar import lidar_recording_path  # noqa: PLC0415

            lidar = lidar_recording_path(Path(trajectory).parent, _label(trajectory))
        future = self._executor.submit(self._convert, trajectory, recording, lidar)
        self._pending.append(_Pending(label, trajectory, recording, future))
        logger.debug("[%s] Converting %s into %s", label, trajectory, recording)
        return recording

    def _convert(
        self, trajectory: Path, recording: Path, lidar: Optional[Path] = None
    ) -> Any:
        stats = self._module.convert(
            str(trajectory), str(recording), strict=self._strict
        )
        if lidar is not None and lidar.is_file():
            from .rerun_lidar import merge_into  # noqa: PLC0415

            try:
                merge_into(recording, lidar)
            except Exception:  # noqa: BLE001 - the sweeps stay in their own file
                logger.exception(
                    "merging the LiDAR sweeps %s into %s failed", lidar, recording
                )
        return stats

    def wait(self) -> list[RerunConversion]:
        """Wait for every log submitted so far, and return how each one --
        those an earlier call returned too -- went, in submission order.

        Each failure is logged as an error naming its run.
        """
        pending, self._pending = self._pending, []
        for item in pending:
            try:
                stats = item.future.result()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                conversion = RerunConversion(
                    item.label,
                    item.trajectory,
                    item.recording,
                    error=f"{type(exc).__name__}: {exc}",
                )
                logger.error(
                    "[%s] Rerun recording of %s failed: %s",
                    item.label,
                    item.trajectory,
                    conversion.error,
                )
            else:
                conversion = RerunConversion(
                    item.label,
                    item.trajectory,
                    item.recording,
                    stats=dict(stats) if isinstance(stats, dict) else None,
                )
                logger.info(
                    "[%s] Rerun recording written to %s", item.label, item.recording
                )
            self._done.append(conversion)
        return list(self._done)

    def close(self) -> list[RerunConversion]:
        """:meth:`wait`, then end the background thread."""
        self.stop_lidar()
        conversions = self.wait()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        return conversions

    def __enter__(self) -> "RerunConverter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def report(conversions: list[RerunConversion]) -> str:
    """One line on *conversions*, then one per failure, for a run's summary."""
    failed = [c for c in conversions if not c.ok]
    lines = [
        f"Rerun recordings: {len(conversions) - len(failed)}/{len(conversions)} "
        "written (open one with: rerun <file.rrd>)"
    ]
    lines.extend(f"  [FAILED] {c.label}: {c.error}" for c in failed)
    return "\n".join(lines)


def convert_outputs(
    output_dir: Union[str, Path],
    *,
    strict: bool = False,
    jobs: Optional[int] = None,
) -> list[RerunConversion]:
    """Convert every trajectory log under *output_dir* (recursively) into a
    recording beside it, *jobs* at a time (all cores when ``None``); or, when
    *output_dir* is a log itself, that log.

    For a run made with ``rerun.enabled=false``, or by a version without it.
    A log that fails does not stop the others; each failure is logged.

    Raises:
        RerunUnavailableError: The converter is not installed.
    """
    module = require_converter()
    path = Path(output_dir)
    if path.is_file():
        with RerunConverter(strict=strict, converter=module) as converter:
            converter.submit(path, _label(path))
            return converter.close()
    entries = module.convert_dir(str(path), jobs=jobs, strict=strict)
    conversions: list[RerunConversion] = []
    for entry in entries:
        trajectory = Path(entry["input"])
        label = _label(trajectory)
        if entry.get("ok"):
            conversions.append(
                RerunConversion(
                    label,
                    trajectory,
                    Path(entry["output"]),
                    stats=dict(entry),
                )
            )
        else:
            error = (
                f"{entry.get('error_type', 'ConversionError')}: {entry.get('error')}"
            )
            logger.error(
                "[%s] Rerun recording of %s failed: %s", label, trajectory, error
            )
            conversions.append(
                RerunConversion(
                    label, trajectory, recording_path(trajectory), error=error
                )
            )
    return conversions


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m autoware_carla_scenario.rerun PATH...``: convert the
    trajectory logs of finished runs (each PATH an output directory or a
    log); exit 1 if any failed, 2 without the converter."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m autoware_carla_scenario.rerun",
        description="Convert the trajectory logs under each PATH (an output "
        "directory, or a *.trajectory.jsonl) into Rerun recordings (.rrd) "
        "beside them.",
    )
    parser.add_argument("path", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail a log with a record type or field the converter does not know",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="logs at a time (all cores by default)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        conversions = [
            conversion
            for path in args.path
            for conversion in convert_outputs(path, strict=args.strict, jobs=args.jobs)
        ]
    except RerunUnavailableError as exc:
        print(exc, file=sys.stderr)  # noqa: T201
        return 2
    print(report(conversions))  # noqa: T201
    return 0 if all(c.ok for c in conversions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
