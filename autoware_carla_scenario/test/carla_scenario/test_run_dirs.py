"""run_dirs: the Hydra argv helpers and the run directory reservation that
examples.run and ui.runner share (#1653)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from autoware_carla_scenario import run_dirs
from autoware_carla_scenario.examples import run
from autoware_carla_scenario.ui import runner

_NOW = datetime(2026, 3, 13, 12, 0, 0)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("hydra.mode=MULTIRUN", True),
        ("+hydra.mode=MULTIRUN", True),
        ("++hydra.mode=MULTIRUN", True),
        ("hydra.mode=multirun", False),
        ("hydra.mode=RUN", False),
        ("--multirun", False),
    ],
)
def test_the_multirun_mode_override(arg: str, expected: bool) -> None:
    assert run_dirs.is_multirun_mode_override(arg) is expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["scenario", "--help"], True),
        (["scenario", "-h"], True),
        (["scenario", "--cfg", "job"], True),
        (["scenario", "--cfg=job"], True),
        (["scenario", "-sc", "install=bash"], True),
        (["scenario", "scenario=cut_in/left"], False),
        # The program name is not an argument.
        (["--help"], False),
        (["scenario", "help=1"], False),
    ],
)
def test_asks_for_no_job(argv: list[str], expected: bool) -> None:
    assert run_dirs.asks_for_no_job(argv) is expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["hydra.run.dir=mine"], True),
        (["+hydra.run.dir=mine"], True),
        (["++hydra.run.dir=mine"], True),
        (["~hydra.run.dir"], True),
        (["hydra.run.dir.x=mine"], False),
        (["hydra.sweep.dir=mine"], False),
        ([], False),
    ],
)
def test_sets_hydra_key(argv: list[str], expected: bool) -> None:
    assert run_dirs.sets_hydra_key(argv, "hydra.run.dir") is expected


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 10, 12, 9, 5, 3),
        datetime(1000, 1, 1),
        datetime(9999, 12, 31, 23, 59, 59, 999999),
    ],
)
@pytest.mark.parametrize("root", ["outputs", "multirun"])
def test_the_timestamped_dir_is_what_strftime_makes(root: str, now: datetime) -> None:
    # What examples.run made with strftime.
    expected = f"{root}/{now.strftime('%Y-%m-%d')}/{now.strftime('%H-%M-%S')}"
    assert run_dirs.timestamped_dir(root, now) == expected


def test_a_timestamped_dir_is_never_shared(tmp_path: Path) -> None:
    made = [
        run_dirs.reserve_timestamped_dir(tmp_path, "outputs", _NOW) for _ in range(3)
    ]
    assert made == [
        "outputs/2026-03-13/12-00-00",
        "outputs/2026-03-13/12-00-00-1",
        "outputs/2026-03-13/12-00-00-2",
    ]
    assert all((tmp_path / name).is_dir() for name in made)
    # A file in the way takes the name too.
    (tmp_path / "outputs/2026-03-13/12-00-00-3").write_text("")
    assert run_dirs.reserve_timestamped_dir(tmp_path, "outputs", _NOW) == (
        "outputs/2026-03-13/12-00-00-4"
    )


def test_a_dangling_link_takes_the_name_unless_told_otherwise(tmp_path: Path) -> None:
    day = tmp_path / "multirun/2026-03-13"
    day.mkdir(parents=True)
    os.symlink(tmp_path / "nowhere", day / "12-00-00")
    # examples.run's FileExistsError: the next name.
    assert run_dirs.reserve_timestamped_dir(tmp_path, "multirun", _NOW) == (
        "multirun/2026-03-13/12-00-00-1"
    )
    # ui.runner's copy raised there, and still does.
    with pytest.raises(FileExistsError):
        runner._reserve_multirun_dir(tmp_path, _NOW)


def test_any_other_failure_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith("12-00-00"):
            raise PermissionError(self)
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(PermissionError):
        run_dirs.reserve_timestamped_dir(tmp_path, "outputs", _NOW)


def test_the_old_names_are_the_shared_ones() -> None:
    assert run._is_multirun_mode_override is run_dirs.is_multirun_mode_override
    assert run._multirun_requested_in_argv is run_dirs.multirun_requested_in_argv
    assert run._sets_hydra_key is run_dirs.sets_hydra_key
    assert runner._multirun_dir is run_dirs.multirun_dir
