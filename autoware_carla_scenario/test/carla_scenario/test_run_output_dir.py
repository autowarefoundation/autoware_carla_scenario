"""A run through Hydra, single or ``--multirun``, writes to a directory of its
own: two runs started in the same second never share one (#1359)."""

from __future__ import annotations

import datetime as datetime_module
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.examples import run


class _Frozen(datetime_module.datetime):
    @classmethod
    def now(cls, tz: Any = None) -> _Frozen:  # type: ignore[override]
        return cls(2026, 3, 13, 12, 0, 0)


@pytest.fixture
def frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every run starts at 2026-03-13 12:00:00, in *tmp_path*."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(datetime_module, "datetime", _Frozen)
    return tmp_path


def test_two_single_runs_in_one_second_get_their_own_directories(
    frozen: Path,
) -> None:
    argv = ["scenario", "scenario=cut_in/left"]
    first, first_dir = run._reserve_hydra_output_dir(argv)
    second, second_dir = run._reserve_hydra_output_dir(argv)
    assert first == [*argv, "hydra.run.dir=outputs/2026-03-13/12-00-00"]
    assert second == [*argv, "hydra.run.dir=outputs/2026-03-13/12-00-00-1"]
    assert first_dir == Path("outputs/2026-03-13/12-00-00")
    assert second_dir == Path("outputs/2026-03-13/12-00-00-1")
    assert (frozen / "outputs/2026-03-13/12-00-00").is_dir()
    assert (frozen / "outputs/2026-03-13/12-00-00-1").is_dir()
    # A batch started in the same second takes the next one.
    assert run._make_batch_output_dir() == frozen / "outputs/2026-03-13/12-00-00-2"


@pytest.mark.parametrize("flag", ["--multirun", "-m", "hydra.mode=MULTIRUN"])
def test_two_sweeps_in_one_second_get_their_own_directories(
    frozen: Path, flag: str
) -> None:
    argv = ["scenario", "scenario=cut_in/left", flag]
    first, _ = run._reserve_hydra_output_dir(argv)
    second, _ = run._reserve_hydra_output_dir(argv)
    assert first[-1] == "hydra.sweep.dir=multirun/2026-03-13/12-00-00"
    assert second[-1] == "hydra.sweep.dir=multirun/2026-03-13/12-00-00-1"
    assert not (frozen / "outputs").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["scenario", "hydra.run.dir=mine"],
        ["scenario", "+hydra.run.dir=mine"],
        ["scenario", "++hydra.run.dir=mine"],
        # What the viewer's grouped runs pass.
        ["scenario", "hydra.run.dir=multirun/2026-03-13/12-00-00/0"],
        ["scenario", "--multirun", "hydra.sweep.dir=mine"],
        ["scenario", "--cfg", "job"],
        ["scenario", "--cfg=job"],
        ["scenario", "--help"],
        ["scenario", "--hydra-help"],
        ["scenario", "--info"],
    ],
)
def test_a_directory_the_user_names_is_kept(frozen: Path, argv: list[str]) -> None:
    assert run._reserve_hydra_output_dir(argv) == (argv, None)
    assert not (frozen / "outputs").exists()
    assert not (frozen / "multirun").exists()


def test_the_other_mode_s_directory_does_not_count(frozen: Path) -> None:
    # A sweep directory named for a single run does not choose its run dir.
    argv = ["scenario", "hydra.sweep.dir=mine"]
    reserved, _ = run._reserve_hydra_output_dir(argv)
    assert reserved[-1] == "hydra.run.dir=outputs/2026-03-13/12-00-00"


def test_an_unused_directory_is_removed_and_a_used_one_kept(frozen: Path) -> None:
    _, unused = run._reserve_hydra_output_dir(["scenario"])
    _, used = run._reserve_hydra_output_dir(["scenario"])
    assert unused is not None
    assert used is not None
    (used / "Demo_result.json").write_text("{}")
    run._remove_if_empty(unused)
    run._remove_if_empty(used)
    assert not unused.exists()
    assert used.is_dir()


#: ``scenario`` through Hydra, with the run of a scenario faked and the clock
#: frozen at 12:00:00: it prints the directory Hydra ran it in.
_APP = """
import datetime as datetime_module
import sys
from pathlib import Path
from types import SimpleNamespace

from hydra.core.hydra_config import HydraConfig

from autoware_carla_scenario.examples import run


class Frozen(datetime_module.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 3, 13, 12, 0, 0)


datetime_module.datetime = Frozen


def fake_run_scenario(cfg, **_):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"ran in {output_dir.relative_to(Path.cwd()).as_posix()}", flush=True)
    return SimpleNamespace(passed=True)


run.run_scenario = fake_run_scenario
sys.argv = ["scenario", *sys.argv[1:]]
run.main()
"""


def _scenario(tmp_path: Path, *args: str) -> list[str]:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(_APP), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [
        line.removeprefix("ran in ")
        for line in proc.stdout.splitlines()
        if line.startswith("ran in ")
    ]


def test_scenario_runs_in_one_second_do_not_share_a_directory(tmp_path: Path) -> None:
    assert _scenario(tmp_path) == ["outputs/2026-03-13/12-00-00"]
    assert _scenario(tmp_path) == ["outputs/2026-03-13/12-00-00-1"]
    # Hydra wrote its own record in each.
    for name in ("12-00-00", "12-00-00-1"):
        assert (tmp_path / "outputs/2026-03-13" / name / ".hydra").is_dir()
    # A directory the run names is Hydra's, as always.
    assert _scenario(tmp_path, "hydra.run.dir=mine") == ["mine"]


def test_scenario_sweeps_in_one_second_do_not_share_a_directory(
    tmp_path: Path,
) -> None:
    sweep = ("--multirun", "seed=1,2")
    assert _scenario(tmp_path, *sweep) == [
        "multirun/2026-03-13/12-00-00/0",
        "multirun/2026-03-13/12-00-00/1",
    ]
    assert _scenario(tmp_path, *sweep) == [
        "multirun/2026-03-13/12-00-00-1/0",
        "multirun/2026-03-13/12-00-00-1/1",
    ]


def test_a_run_hydra_refuses_leaves_no_directory(tmp_path: Path) -> None:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(_APP), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), "no_such_key_anywhere=1"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode != 0
    assert list((tmp_path / "outputs" / "2026-03-13").iterdir()) == []


# ---------------------------------------------------------------------------
# A batch writes to the directory hydra.run.dir= (hydra.sweep.dir=) names (#2294)
# ---------------------------------------------------------------------------


def _dispatched(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> tuple[int | str | None, list[tuple[list[str], dict[str, Any]]]]:
    """``scenario`` with *argv*, run_batch stubbed: the exit code and each
    call's overrides and keyword arguments."""
    from .test_run_refusals import _main

    batches: list[tuple[list[str], dict[str, Any]]] = []

    def run_batch(_names: list[str], overrides: list[str], **kwargs: Any) -> None:
        batches.append((list(overrides), kwargs))

    code, _ = _main(monkeypatch, argv, run_batch=run_batch)
    return code, batches


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["scenario=lane_change/*", "hydra.run.dir=mine"], Path("mine")),
        (["scenario=lane_change/*", "++hydra.run.dir=mine"], Path("mine")),
        (
            ["--multirun", "scenario=lane_change/*", "hydra.sweep.dir=sweep"],
            Path("sweep"),
        ),
        # The other mode's key is ignored, as Hydra ignores it.
        (["scenario=lane_change/*", "hydra.sweep.dir=sweep"], None),
        (["--multirun", "scenario=lane_change/*", "hydra.run.dir=mine"], None),
        (["scenario=lane_change/*"], None),
    ],
)
def test_a_batch_reads_its_output_directory_off_the_command_line(
    frozen: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: Path | None,
) -> None:
    code, batches = _dispatched(monkeypatch, [*argv, "seed=3"])
    assert code is None
    [(overrides, kwargs)] = batches
    assert kwargs.get("output_dir") == expected
    # Neither key is a case's override.
    assert overrides == ["seed=3"]


def test_a_batch_resolves_a_now_interpolation_in_its_directory(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, batches = _dispatched(
        monkeypatch, ["scenario=lane_change/*", "hydra.run.dir=out/${now:%Y}"]
    )
    assert code is None
    [(_, kwargs)] = batches
    out = kwargs["output_dir"]
    assert out.parent == Path("out") and out.name.isdigit(), out


def test_a_batch_runs_in_the_directory_it_names(
    frozen: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    seen: list[Path] = []

    class _Reached(Exception):
        pass

    def fake_queue(_cfg: Any, _paths: Any, output_dir: Path, **_: Any) -> None:
        seen.append(output_dir)
        raise _Reached

    monkeypatch.setattr(run, "build_scenario", lambda cfg, **_: (None, object()))
    monkeypatch.setattr(run, "resolve_map_paths", lambda _m: SimpleNamespace())
    monkeypatch.setattr(run, "_batch_queue", fake_queue)
    monkeypatch.setattr(
        run, "_make_batch_output_dir", lambda: pytest.fail("reserved a directory")
    )
    (frozen / "mine").mkdir()  # an empty one is taken as it is
    for name in ("mine", "new/deeper"):
        with pytest.raises(_Reached):
            run.run_batch(["lane_change/left"], [], output_dir=Path(name))
        assert seen[-1] == (frozen / name).resolve()
        assert seen[-1].is_dir()


@pytest.mark.parametrize("resume", [False, True])
def test_a_batch_refuses_a_directory_that_is_not_its_own(
    frozen: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resume: bool,
) -> None:
    monkeypatch.setattr(
        run, "_compose_config", lambda *_: pytest.fail("composed a config")
    )
    target = frozen / "mine"
    if resume:
        target.mkdir()
        extra: dict[str, Any] = {"resume_into": frozen / "old"}
    else:
        target.mkdir()
        (target / "batch_results.json").write_text("{}")
        extra = {}
    with pytest.raises(SystemExit) as raised:
        run.run_batch(["lane_change/left"], [], output_dir=target, **extra)
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: "), err
    assert ("--resume-into" in err) if resume else ("not empty" in err), err
    assert len(err.strip().splitlines()) == 1, err
