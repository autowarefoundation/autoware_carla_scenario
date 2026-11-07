"""What one run leaves under its file stem: the single list of them.

A run names everything it writes after one stem (``CutInScenario``, then
``CutInScenario-1`` ...): its result, trajectory log, coverage file and so on
are files, its T4 dataset (``{stem}.t4``) and SUMO's outputs
(``{stem}.sumo``) directories.  Three places need the
whole list, and all read it from here:

* ``scenario_runner._run_file_stem``, which takes the first stem none of them
  exists under, so a run never writes over an earlier one's outputs;
* :func:`remove_run_outputs`, which the SLURM worker calls before it runs a
  case under the stem its manifest gives it, and ``scenario_queue`` before it
  retries a failed attempt under the stem that attempt had, so a case run
  again leaves none of its last attempt's behind -- and the T4 writer, which
  refuses a directory already holding a dataset, records the new one;
* :func:`run_output_stems`, its inverse, which the Hydra sweeper calls to
  learn which stems its case runs took in a directory -- and, before a job's
  first attempt, which runs an earlier sweep left in its job directory, which
  it removes with :func:`remove_run_outputs` (#1956).

The modules that write each output keep their own name for its suffix
(``rerun.RECORDING_SUFFIX``, ``t4_settings.DATASET_SUFFIX``,
``traffic.sumo.backend.OUTPUT_SUFFIX`` ...);
``test_run_outputs.py`` keeps them equal to the ones here.  Those of a run's
environment and lock and of its T4Scene and map are defined here, and
``environment`` and ``t4_scene.convert`` re-export them under the names they
had (#1713); so is the metrics' scratch recording, which ``rerun_metrics``
re-exports, and the ODD archive, openodd's ``ODD_ARCHIVE_SUFFIX``, which
``ui.case_paths`` reads from here (#1748).  The path rule of
a trajectory log and its recordings lives here too (:func:`trajectory_stem`,
:func:`recording_path`, :func:`sensors_recording_path`), so that the checked
``ui.web_viewer_paths`` shares it with ``rerun`` and ``rerun_lidar``, which
re-export it under the names they had.

Free of imports beyond ``pathlib``, ``shutil`` and the gate's record (whose
:data:`~autoware_carla_scenario.gate.record.GATE_RECORD_SUFFIX` it lists, and
which imports only the standard library and the package's ``atomic_file``):
the SLURM worker and the library check read it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .gate.record import GATE_RECORD_SUFFIX

__all__ = [
    "COVERAGE_SUFFIX",
    "ENVIRONMENT_SUFFIX",
    "LOCK_SUFFIX",
    "METRICS_SUFFIX",
    "ODD_ARCHIVE_SUFFIX",
    "RECORDING_SUFFIX",
    "RESULT_SUFFIX",
    "RUN_DIR_SUFFIXES",
    "RUN_FILE_SUFFIXES",
    "RUN_SUFFIXES",
    "SENSORS_SUFFIX",
    "T4SCENE_MAP_SUFFIX",
    "T4SCENE_SUFFIX",
    "TRAJECTORY_SUFFIX",
    "recording_path",
    "run_output_paths",
    "run_outputs_exist",
    "remove_run_outputs",
    "run_output_stems",
    "sensors_recording_path",
    "trajectory_stem",
]

#: A run's result: ``{stem}_result.json``.
RESULT_SUFFIX = "_result.json"
#: Its ODD coverage file: ``{stem}_coverage.json``.
COVERAGE_SUFFIX = "_coverage.json"
#: A run's trajectory log: ``{stem}.trajectory.jsonl``.
TRAJECTORY_SUFFIX = ".trajectory.jsonl"
#: The Rerun recording made of it: ``{stem}.rrd``.
RECORDING_SUFFIX = ".rrd"
#: The sensors recorded during the run: ``{stem}.sensors.rrd``.
SENSORS_SUFFIX = ".sensors.rrd"
#: The scratch recording a run's metrics are written to before they are merged
#: into its recording: ``{stem}.metrics.rrd``.
METRICS_SUFFIX = ".metrics.rrd"
#: The run's ODD as the files it was read from: ``{stem}_odd.tar.gz``
#: (openodd's ``ODD_ARCHIVE_SUFFIX``, which this module cannot import).
ODD_ARCHIVE_SUFFIX = "_odd.tar.gz"
#: The run's environment: ``{stem}_environment.json``.
ENVIRONMENT_SUFFIX = "_environment.json"
#: The lock of that environment: ``{stem}_requirements.lock``.
LOCK_SUFFIX = "_requirements.lock"
#: The T4Scene of the trajectory log: ``{stem}.t4scene.pb``.
T4SCENE_SUFFIX = ".t4scene.pb"
#: The map of that T4Scene: ``{stem}.t4scene.map.pb``.
T4SCENE_MAP_SUFFIX = ".t4scene.map.pb"

#: The files a run writes as ``{stem}{suffix}``.
RUN_FILE_SUFFIXES: tuple[str, ...] = (
    RESULT_SUFFIX,
    TRAJECTORY_SUFFIX,
    COVERAGE_SUFFIX,
    # A logical scenario's gate record (docs/gate.md).
    GATE_RECORD_SUFFIX,
    ODD_ARCHIVE_SUFFIX,
    ENVIRONMENT_SUFFIX,
    LOCK_SUFFIX,
    # CARLA's own recorder (its replay log).
    ".log",
    # The Rerun recording of the trajectory log (rerun.enabled).
    RECORDING_SUFFIX,
    # The sensors recorded during the run, merged into the recording and
    # removed when it is written; left behind by a run that did not finish.
    SENSORS_SUFFIX,
    # The metrics' scratch recording, merged into the recording and removed;
    # left behind by a run killed while it was merged (a SLURM timeout).
    METRICS_SUFFIX,
    # The T4Scene of the trajectory log and its map (t4scene.enabled).
    T4SCENE_SUFFIX,
    T4SCENE_MAP_SUFFIX,
)

#: The directories a run writes as ``{stem}{suffix}``.
RUN_DIR_SUFFIXES: tuple[str, ...] = (
    # Its T4 dataset (t4.enabled), its version ``0`` inside.
    ".t4",
    # The SUMO backend's log, generated demand, FCD and clock (traffic=sumo).
    ".sumo",
)

#: Every suffix of a run output, longest first, so that ``X.sensors.rrd``
#: and ``X.metrics.rrd`` read as stem ``X``, not ``X.sensors`` or ``X.metrics``.
RUN_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        RUN_FILE_SUFFIXES + RUN_DIR_SUFFIXES,
        key=lambda suffix: len(suffix),
        reverse=True,
    )
)


def trajectory_stem(trajectory: Path) -> str:
    """The run *trajectory* is the log of: its file stem (``X`` of
    ``X.trajectory.jsonl``)."""
    name = trajectory.name
    if name.endswith(TRAJECTORY_SUFFIX):
        return name[: -len(TRAJECTORY_SUFFIX)]
    return trajectory.stem


def recording_path(trajectory: Path) -> Path:
    """The recording made of *trajectory*: ``X.rrd`` beside
    ``X.trajectory.jsonl``, where the converter writes it by default."""
    path = Path(trajectory)
    return path.with_name(trajectory_stem(path) + RECORDING_SUFFIX)


def sensors_recording_path(directory: Path, stem: str) -> Path:
    """``{directory}/{stem}.sensors.rrd``: the sensors a run named *stem*
    recorded."""
    return Path(directory) / f"{stem}{SENSORS_SUFFIX}"


def run_output_paths(directory: Path, stem: str) -> list[Path]:
    """Every path a run named *stem* writes in *directory*: the files of
    :data:`RUN_FILE_SUFFIXES`, then the directories of
    :data:`RUN_DIR_SUFFIXES`."""
    paths: list[Path] = []
    for suffix in RUN_FILE_SUFFIXES:
        paths.append(directory / f"{stem}{suffix}")
    for suffix in RUN_DIR_SUFFIXES:
        paths.append(directory / f"{stem}{suffix}")
    return paths


def run_outputs_exist(directory: Path, stem: str) -> bool:
    """Whether a run named *stem* left anything in *directory*."""
    for path in run_output_paths(directory, stem):
        if path.exists():
            return True
    return False


def remove_run_outputs(directory: Path, stem: str) -> None:
    """Remove what a run named *stem* left in *directory*: every path of
    :func:`run_output_paths`, its T4 dataset directory included -- the T4
    writer refuses a directory already holding a dataset, so a case run again
    under *stem* would otherwise record none and keep the last attempt's.  A
    symlink is unlinked, never what it points to; the outputs of any other
    stem are left alone."""
    for path in run_output_paths(directory, stem):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def run_output_stems(directory: Path, job_log: str | None) -> set[str]:
    """The stems of the runs that left an output in *directory*: the inverse
    of :func:`run_output_paths`, what ``scenario_runner._run_file_stem``
    skips over.

    *job_log* is the file name of Hydra's job log (``<job name>.log``), which
    is not a run's CARLA recorder log though it ends as one does.
    """
    try:
        names = [path.name for path in directory.iterdir()]
    except OSError:
        return set[str]()
    stems: set[str] = set()
    for name in names:
        if job_log is not None and name == job_log:
            continue
        for suffix in RUN_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                stems.add(name[: -len(suffix)])
                break
    return stems
