"""What the ``scenario`` command refuses before anything runs (#2089).

- ``launcher=slurm`` without ``scenario=``: a SLURM batch needs scenarios to
  run; @hydra.main used to run the default scenario here instead.
- ``--resume-from`` where nothing reads it: a single run or Hydra's basic
  sweeper used to run every case.
- ``ego.entity``, ``typecheck`` and the ego's lanelet ids: the pre-run check
  (:func:`check_run_config`, :func:`check_batch_configs`) refuses them, so a
  lanelet sweep and a glob batch refuse them before any case runs, and a
  fractional lanelet id is refused rather than truncated.
- A missing Rerun SDK is a :class:`RerunUnavailableError`, so the CLI exits 2.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

_CONCRETE = "intersection_passing/straight_3npc"


def _shipped(overrides: list[str]) -> Any:
    from .test_video_removed import _shipped as compose

    return compose(overrides)


def _main(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], **stubs: Any
) -> tuple[int | str | None, list[Any]]:
    """Run ``scenario`` with *argv*, with Hydra's entry point and the batch
    stubbed out: returns the exit code (``None`` when it returned) and what
    the stubs were called with."""
    from autoware_carla_scenario.examples import run

    calls: list[Any] = []

    def hydra_main() -> None:
        calls.append(("hydra", list(sys.argv)))

    def run_batch(names: list[str], overrides: list[str], **kwargs: Any) -> None:
        calls.append(("batch", names, kwargs))

    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_hydra_main", hydra_main)
    monkeypatch.setattr(run, "run_batch", run_batch)
    monkeypatch.setattr(run, "_reserve_hydra_output_dir", lambda argv: (argv, None))
    for name, value in stubs.items():
        monkeypatch.setattr(run, name, value)
    monkeypatch.delenv("SWEEP_RESUME_FROM", raising=False)
    monkeypatch.setattr(sys, "argv", ["scenario", *argv])
    try:
        run.main()
    except SystemExit as exc:
        return exc.code, calls
    return None, calls


# ---------------------------------------------------------------------------
# launcher=slurm without scenario=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["launcher=slurm"],
        ["launcher=slurm", "--no-wait"],
        ["launcher.type=slurm"],
        ["--multirun", "launcher=slurm"],
    ],
)
def test_launcher_slurm_without_a_scenario_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, argv: list[str]
) -> None:
    code, calls = _main(monkeypatch, argv)
    assert code == 2
    assert calls == []
    assert "launcher=slurm runs a batch" in capsys.readouterr().err


def test_launcher_slurm_with_a_scenario_runs_a_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, calls = _main(monkeypatch, [f"scenario={_CONCRETE}", "launcher=slurm"])
    assert code is None
    assert [call[:2] for call in calls] == [("batch", [_CONCRETE])]


def test_launcher_slurm_help_still_reaches_hydra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, calls = _main(monkeypatch, ["launcher=slurm", "--help"])
    assert code is None
    assert [call[0] for call in calls] == ["hydra"]


# ---------------------------------------------------------------------------
# --resume-from where nothing reads it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        # A single run.
        [f"scenario={_CONCRETE}"],
        [],
        # Hydra's basic sweeper.
        ["--multirun", f"scenario={_CONCRETE}", "seed=1,2"],
        # The lanelet sweeper without --multirun: a single run.
        [f"scenario={_CONCRETE}", "hydra/sweeper=lanelet_constraint"],
    ],
)
def test_resume_from_is_refused_where_nothing_reads_it(
    monkeypatch: pytest.MonkeyPatch, capsys: Any, argv: list[str]
) -> None:
    code, calls = _main(monkeypatch, [*argv, "--resume-from", "3"])
    assert code == 2
    assert calls == []
    assert capsys.readouterr().err.startswith("Error: --resume-from resumes a batch")


@pytest.mark.parametrize(
    "argv",
    [
        ["--multirun", f"scenario={_CONCRETE}", "hydra/sweeper=lanelet_constraint"],
        ["-m", f"scenario={_CONCRETE}", "+hydra/sweeper=lanelet_constraint"],
        [
            "hydra.mode=MULTIRUN",
            f"scenario={_CONCRETE}",
            "hydra/sweeper=lanelet_constraint",
        ],
    ],
)
def test_resume_from_reaches_the_lanelet_sweeper(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    import os

    seen: list[str | None] = []

    def hydra_main() -> None:
        seen.append(os.environ.get("SWEEP_RESUME_FROM"))

    code, _ = _main(monkeypatch, [*argv, "--resume-from", "3"], _hydra_main=hydra_main)
    assert code is None
    assert seen == ["3"]


@pytest.mark.parametrize(
    "argv",
    [
        ["scenario=intersection_passing/*"],
        ["--multirun", "scenario=intersection_passing/*"],
        [f"scenario={_CONCRETE}", "launcher=slurm"],
    ],
)
def test_resume_from_reaches_a_batch(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    code, calls = _main(monkeypatch, [*argv, "--resume-from", "3"])
    assert code is None
    assert len(calls) == 1 and calls[0][0] == "batch"
    assert calls[0][2]["resume_from"] == 3


def test_a_single_run_without_resume_from_reaches_hydra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, calls = _main(monkeypatch, [f"scenario={_CONCRETE}"])
    assert code is None
    assert [call[0] for call in calls] == ["hydra"]


# ---------------------------------------------------------------------------
# ego.entity, typecheck and the ego's lanelet ids in the pre-run check
# ---------------------------------------------------------------------------


def test_the_shipped_config_passes_the_new_checks() -> None:
    from autoware_carla_scenario.examples import run

    run.check_run_config(_shipped([]))
    run.check_run_config(
        _shipped(
            [
                "ego.spawn_lanelet_id=242.0",
                "ego.goal_lanelet_id='243'",
                "++ego.waypoint_lanelet_ids=[1,2.0]",
                "typecheck=off",
                "ego.entity=carla_driver",
            ]
        )
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("ego.entity=bogus", r"Unknown ego\.entity: 'bogus'"),
        ("typecheck=bogus", r"typecheck must be one of auto, required, off"),
        ("ego.spawn_lanelet_id=242.7", r"ego\.spawn_lanelet_id must be a whole number"),
        ("ego.spawn_lanelet_id=abc", r"ego\.spawn_lanelet_id must be a whole number"),
        ("ego.spawn_lanelet_id=true", r"ego\.spawn_lanelet_id must be a whole number"),
        ("ego.goal_lanelet_id=abc", r"ego\.goal_lanelet_id must be a whole number"),
        ("ego.goal_lanelet_id=243.5", r"ego\.goal_lanelet_id must be a whole number"),
        (
            "ego.waypoint_lanelet_ids=[1,abc]",
            r"ego\.waypoint_lanelet_ids\[1\] must be a whole number",
        ),
        (
            "ego.waypoint_lanelet_ids=[1.5]",
            r"ego\.waypoint_lanelet_ids\[0\] must be a whole number",
        ),
        (
            "ego.waypoint_lanelet_ids=[null]",
            r"ego\.waypoint_lanelet_ids\[0\] must be a lanelet id",
        ),
        ("ego.waypoint_lanelet_ids=abc", r"ego\.waypoint_lanelet_ids must be a list"),
    ],
)
def test_the_run_check_refuses(
    override: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.examples import run

    monkeypatch.delenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", raising=False)
    cfg = _shipped([f"++{override}"])
    with pytest.raises(run.RunConfigError, match=message):
        run.check_run_config(cfg)
    # And a batch, naming the case refused, before any case runs.
    with pytest.raises(run.RunConfigError, match=rf"^bad: {message}"):
        run.check_batch_configs([_shipped([]), cfg], ["good", "bad"])


def test_a_lanelet_sweep_refuses_them_before_any_case(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from autoware_carla_scenario.sweeper import lanelet_constraint_sweeper as sweeper

    cfg = _shipped(["ego.entity=bogus"])
    with pytest.raises(SystemExit) as exited:
        sweeper._refuse_before_forking(cfg)
    assert exited.value.code == 2
    assert "Unknown ego.entity: 'bogus'" in capsys.readouterr().err


def test_a_glob_batch_refuses_them_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from autoware_carla_scenario.examples import run

    def build(*_: Any, **__: Any) -> Any:
        raise AssertionError("nothing is built")

    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "build_scenario", build)
    monkeypatch.setattr(run, "_batch_queue", build)
    monkeypatch.delenv("SWEEP_RESUME_FROM", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scenario", "scenario=intersection_passing/*", "ego.goal_lanelet_id=abc"],
    )
    with pytest.raises(SystemExit) as exited:
        run.main()
    assert exited.value.code == 1
    assert "ego.goal_lanelet_id must be a whole number" in capsys.readouterr().err


def test_the_build_refuses_a_fractional_lanelet_id_rather_than_truncate() -> None:
    from autoware_carla_scenario.examples import run

    with pytest.raises(ValueError, match=r"ego\.spawn_lanelet_id"):
        run.build_spawn_pose(_shipped(["ego.spawn_lanelet_id=242.7"]))
    with pytest.raises(ValueError, match=r"ego\.goal_lanelet_id"):
        run.build_goal_pose(_shipped(["ego.goal_lanelet_id=243.5"]))
    with pytest.raises(ValueError, match=r"ego\.waypoint_lanelet_ids\[0\]"):
        run.build_waypoint_poses(_shipped(["++ego.waypoint_lanelet_ids=[7.5]"]))
    with pytest.raises(ValueError, match=r"ego\.spawn_lanelet_id"):
        run.build_planned_route(_shipped(["ego.spawn_lanelet_id=242.7"]))
    # A whole number in any spelling is read as itself.
    assert (
        run.build_spawn_pose(_shipped(["ego.spawn_lanelet_id=242.0"])).lanelet_id == 242
    )
    goal = run.build_goal_pose(_shipped(["ego.goal_lanelet_id='243'"]))
    assert goal is not None and goal.lanelet_id == 243
    assert [
        pose.lanelet_id
        for pose in run.build_waypoint_poses(
            _shipped(["++ego.waypoint_lanelet_ids=[1,2.0]"])
        )
    ] == [1, 2]


def test_build_ego_entity_reads_the_same_names() -> None:
    from autoware_carla_scenario.examples import run

    assert run.build_ego_entity(_shipped([])) is None
    with pytest.raises(ValueError, match=r"Unknown ego\.entity: 'bogus'"):
        run.build_ego_entity(_shipped(["ego.entity=bogus"]))


# ---------------------------------------------------------------------------
# A missing Rerun SDK
# ---------------------------------------------------------------------------


def test_a_missing_rerun_sdk_is_a_rerun_unavailable_error() -> None:
    from autoware_carla_scenario.rerun import RerunUnavailableError
    from autoware_carla_scenario.rerun_lidar import LidarRerunUnavailableError

    assert issubclass(LidarRerunUnavailableError, RerunUnavailableError)
    assert issubclass(LidarRerunUnavailableError, RuntimeError)


def test_a_missing_rerun_sdk_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from autoware_carla_scenario.rerun_lidar import LidarRerunUnavailableError

    def run_batch(*_: Any, **__: Any) -> None:
        raise LidarRerunUnavailableError("no Rerun SDK")

    code, _ = _main(
        monkeypatch, ["scenario=intersection_passing/*"], run_batch=run_batch
    )
    assert code == 2
    assert "no Rerun SDK" in capsys.readouterr().err
