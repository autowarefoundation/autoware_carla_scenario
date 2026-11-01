"""What one run leaves under its file stem: the single list of them.

A run names everything it writes after one stem (``CutInScenario``, then
``CutInScenario-1`` ...): its result, trajectory log, coverage file and so on
are files, its T4 dataset (``{stem}.t4``) a directory.  Two places need the
whole list, and both read it from here:

* ``scenario_runner._run_file_stem``, which takes the first stem none of them
  exists under, so a run never writes over an earlier one's outputs;
* the SLURM worker (``slurm.worker._remove_outputs``), which runs a case under
  the stem its manifest gives it and removes that stem's outputs first, so a
  case run again leaves none of its last attempt's behind -- and the T4 writer,
  which refuses a directory already holding a dataset, records the new one.

The modules that write each output keep their own name for its suffix
(``rerun.RECORDING_SUFFIX``, ``t4_settings.DATASET_SUFFIX`` ...);
``test_run_outputs.py`` keeps them equal to the ones here.

Free of imports beyond ``pathlib``: the SLURM worker and the library check
read it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "RUN_DIR_SUFFIXES",
    "RUN_FILE_SUFFIXES",
    "run_output_paths",
    "run_outputs_exist",
]

#: The files a run writes as ``{stem}{suffix}``.
RUN_FILE_SUFFIXES: tuple[str, ...] = (
    "_result.json",
    ".trajectory.jsonl",
    "_coverage.json",
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

#: The directories a run writes as ``{stem}{suffix}``: its T4 dataset
#: (t4.enabled), its version ``0`` inside.
RUN_DIR_SUFFIXES: tuple[str, ...] = (".t4",)


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
