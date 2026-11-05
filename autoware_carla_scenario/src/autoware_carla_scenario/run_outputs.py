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
  learn which stems its case runs took in a directory.

The modules that write each output keep their own name for its suffix
(``rerun.RECORDING_SUFFIX``, ``t4_settings.DATASET_SUFFIX``,
``traffic.sumo.backend.OUTPUT_SUFFIX`` ...);
``test_run_outputs.py`` keeps them equal to the ones here.

Free of imports beyond ``pathlib`` and ``shutil``: the SLURM worker and the
library check read it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "RUN_DIR_SUFFIXES",
    "RUN_FILE_SUFFIXES",
    "RUN_SUFFIXES",
    "run_output_paths",
    "run_outputs_exist",
    "remove_run_outputs",
    "run_output_stems",
]

#: The files a run writes as ``{stem}{suffix}``.
RUN_FILE_SUFFIXES: tuple[str, ...] = (
    "_result.json",
    ".trajectory.jsonl",
    "_coverage.json",
    # A logical scenario's gate record (docs/gate.md).
    "_gate.json",
    "_odd.tar.gz",
    "_environment.json",
    "_requirements.lock",
    # CARLA's own recorder (its replay log).
    ".log",
    # The Rerun recording of the trajectory log (rerun.enabled).
    ".rrd",
    # The sensors recorded during the run, merged into the recording and
    # removed when it is written; left behind by a run that did not finish.
    ".sensors.rrd",
    # The T4Scene of the trajectory log and its map (t4scene.enabled).
    ".t4scene.pb",
    ".t4scene.map.pb",
)

#: The directories a run writes as ``{stem}{suffix}``.
RUN_DIR_SUFFIXES: tuple[str, ...] = (
    # Its T4 dataset (t4.enabled), its version ``0`` inside.
    ".t4",
    # The SUMO backend's log, generated demand, FCD and clock (traffic=sumo).
    ".sumo",
)

#: Every suffix of a run output, longest first, so that ``X.sensors.rrd``
#: reads as stem ``X``, not ``X.sensors``.
RUN_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        RUN_FILE_SUFFIXES + RUN_DIR_SUFFIXES,
        key=lambda suffix: len(suffix),
        reverse=True,
    )
)


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
