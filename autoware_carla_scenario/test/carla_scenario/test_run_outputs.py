"""What a run leaves under its stem, and a SLURM case run again clearing it (#945)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoware_carla_scenario import environment, rerun, rerun_lidar, t4_settings
from autoware_carla_scenario.run_outputs import (
    RUN_DIR_SUFFIXES,
    RUN_FILE_SUFFIXES,
    run_output_paths,
    run_outputs_exist,
)
from autoware_carla_scenario.scenario_runner import _run_file_stem
from autoware_carla_scenario.slurm import manifest
from autoware_carla_scenario.slurm.worker import _remove_outputs
from autoware_carla_scenario.t4_scene import convert

STEM = "CutInScenario"


def _write_outputs(directory: Path, stem: str) -> None:
    """Every output a run named *stem* writes, a dataset in its directory."""
    for suffix in RUN_FILE_SUFFIXES:
        (directory / f"{stem}{suffix}").write_text("old")
    for suffix in RUN_DIR_SUFFIXES:
        version = directory / f"{stem}{suffix}" / "0"
        version.mkdir(parents=True)
        (version / "annotation").mkdir()
        (version / "annotation" / "scene.json").write_text("[]")


def test_the_suffixes_are_the_writers_own() -> None:
    # Each writer keeps its own name for its suffix; the list must hold them.
    assert set(RUN_FILE_SUFFIXES) >= {
        rerun.TRAJECTORY_SUFFIX,
        rerun.RECORDING_SUFFIX,
        rerun_lidar.SENSORS_SUFFIX,
        convert.SCENE_SUFFIX,
        convert.MAP_SUFFIX,
        environment.ENVIRONMENT_SUFFIX,
        environment.LOCK_SUFFIX,
    }
    assert RUN_DIR_SUFFIXES == (t4_settings.DATASET_SUFFIX,)
    assert not set(RUN_FILE_SUFFIXES) & set(RUN_DIR_SUFFIXES)


def test_the_manifest_re_exports_the_list() -> None:
    assert manifest.RUN_FILE_SUFFIXES is RUN_FILE_SUFFIXES


def test_run_output_paths_names_files_then_directories(tmp_path: Path) -> None:
    paths = run_output_paths(tmp_path, STEM)
    assert paths == [
        tmp_path / f"{STEM}{suffix}" for suffix in RUN_FILE_SUFFIXES + RUN_DIR_SUFFIXES
    ]
    assert not run_outputs_exist(tmp_path, STEM)
    _write_outputs(tmp_path, STEM)
    assert run_outputs_exist(tmp_path, STEM)


def test_remove_outputs_leaves_no_output_of_the_stem(tmp_path: Path) -> None:
    _write_outputs(tmp_path, STEM)
    _write_outputs(tmp_path, f"{STEM}-1")
    (tmp_path / "batch_results.json").write_text("[]")

    _remove_outputs(tmp_path, STEM)

    assert not any(path.exists() for path in run_output_paths(tmp_path, STEM))
    assert not (tmp_path / f"{STEM}.t4").exists()
    # Another case's outputs, and the batch's own files, are kept.
    assert all(path.exists() for path in run_output_paths(tmp_path, f"{STEM}-1"))
    assert (tmp_path / "batch_results.json").exists()


def test_remove_outputs_of_a_stem_with_none_is_a_no_op(tmp_path: Path) -> None:
    _remove_outputs(tmp_path, STEM)
    assert list(tmp_path.iterdir()) == []


def test_remove_outputs_unlinks_a_symlinked_dataset_not_its_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "kept").write_text("x")
    (tmp_path / f"{STEM}.t4").symlink_to(target, target_is_directory=True)
    _remove_outputs(tmp_path, STEM)
    assert not (tmp_path / f"{STEM}.t4").exists()
    assert (target / "kept").exists()


@pytest.mark.parametrize("path", run_output_paths(Path("."), STEM), ids=str)
def test_every_output_the_stem_counts_is_cleared_by_the_worker(
    tmp_path: Path, path: Path
) -> None:
    # Whatever takes a stem from a local run (_run_file_stem) is what a SLURM
    # case run again under that stem removes first: else it would run beside
    # (or, for a T4 dataset, fail to replace) its last attempt's output.
    output = tmp_path / path.name
    if path.name.endswith(RUN_DIR_SUFFIXES):
        output.mkdir()
    else:
        output.write_text("old")
    assert _run_file_stem(tmp_path, STEM) == f"{STEM}-1"

    _remove_outputs(tmp_path, STEM)

    assert not output.exists()
    assert _run_file_stem(tmp_path, STEM) == STEM


def test_a_rerun_writes_its_t4_dataset_again(tmp_path: Path) -> None:
    writer_module = pytest.importorskip("autoware_carla_scenario_t4")
    from autoware_carla_scenario.t4 import _VERSION

    def run_case(frames: int) -> dict[str, int]:
        writer = writer_module.T4Writer(
            t4_settings.dataset_path(tmp_path, STEM) / _VERSION, scene_name=STEM
        )
        writer.add_sensor(t4_settings.LIDAR_CHANNEL, "lidar", (0, 0, 0), (1, 0, 0, 0))
        for i in range(frames):
            writer.begin_frame_carla(
                i * 100_000, (i, 0, 0, 0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)
            )
            writer.add_lidar(t4_settings.LIDAR_CHANNEL, np.zeros((4, 4), np.float32))
            writer.end_frame()
        return writer.finish()

    assert run_case(5)["samples"] == 5
    (tmp_path / f"{STEM}_result.json").write_text('{"passed": false}')
    # Without the clearing, the writer refuses the directory.
    with pytest.raises(ValueError, match="already holds a T4 dataset"):
        run_case(2)

    _remove_outputs(tmp_path, STEM)
    assert run_case(2)["samples"] == 2
    assert not (tmp_path / f"{STEM}_result.json").exists()
