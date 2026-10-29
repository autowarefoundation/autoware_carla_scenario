"""slurm.scripts, split out of slurm.submit so the library check compiles it
(#732): submit still exports every name it did, and the job scripts and
sbatch command lines it builds are, byte for byte, the ones it built before.

``data/slurm_scripts/golden.json`` holds what submit built before the split,
for the settings below; regenerate it only for a change meant to alter them::

    uv run python -m autoware_carla_scenario.test.carla_scenario.test_slurm_scripts
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import scripts, submit
from autoware_carla_scenario.slurm.manifest import BatchLayout
from autoware_carla_scenario.slurm.settings import SlurmSettings

_GOLDEN = Path(__file__).resolve().parent / "data" / "slurm_scripts" / "golden.json"

#: What submit exported before the split (its ``__all__``) and the names of
#: it the tests and the other modules import.
_SUBMIT_NAMES = {
    "ACTIVE_STATES",
    "CANCEL_WAIT_SECONDS",
    "CLI_MODULE",
    "REMOTE_EARLY_OUTPUT_DIR",
    "WORKER_MODULE",
    "_collect_command",
    "_expand_ranges",
    "_run_sbatch",
    "_shell_path",
    "_wait_until_gone",
    "after_submit",
    "callable_path",
    "cancel_jobs",
    "collect_script",
    "follow",
    "job_script",
    "relative_batch_dir",
    "sbatch_command",
    "squeue_states",
    "submit_batch",
    "submit_cases",
}

_PASSTHROUGH_ENV = {
    "PATH": "/opt/venv/bin:/usr/bin",
    "LD_LIBRARY_PATH": "/opt/carla lib:/usr/lib",
    "PYTHONPATH": "",
    "VIRTUAL_ENV": "/opt/venv",
    "CARLA_ROOT": "/opt/carla's root",
    "ONEPLANNER_FLAGS": '--a="b c" $HOME `x`',
}

#: name -> (launcher settings, manifest, batch directory)
_CASES: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {
    "local-defaults": (
        {},
        {"python": "/usr/bin/python3", "cwd": "/work", "env": {}},
        "/work/outputs/2026-10-19/12-00-00",
    ),
    "local-every-sbatch-option": (
        {
            "partition": "main",
            "account": "acct",
            "qos": "high",
            "time": "01:00:00",
            "mem": "12G",
            "cpus_per_task": 8,
            "gres": "shard:1",
            "array_parallelism": 2,
            "job_name": "my job",
            "extra_sbatch_args": ["--nice=10", "--exclusive"],
            "export": "NONE",
            "signal_grace_seconds": 30,
            "collect_mem": "4G",
            "headless": False,
            "python": "~/venv/bin/python",
            "setup": ["module load cuda", "source /venv/bin/activate"],
            "env": {"ONEPLANNER_X": "set by launcher.env", "B": "two words"},
        },
        {"python": "/usr/bin/python3", "cwd": "/work dir", "env": {}},
        "/work dir/outputs/2026-10-19/12-00-00",
    ),
    "local-env-passthrough": (
        {"python": "~", "collect_mem": None, "cases_per_task": 3},
        {"python": "/opt/venv/bin/python", "cwd": "/w", "env": _PASSTHROUGH_ENV},
        "/w/out/d/t",
    ),
    "local-home-python": (
        {"python": "~/", "array_parallelism": 1},
        {"python": "/x", "cwd": "/w'q", "env": {"VIRTUAL_ENV": "/v"}},
        "relative/d/t",
    ),
    "remote-relative": (
        {
            "remote_cwd": "workspace/OnePlanner",
            "python": ".venv/bin/python",
            "setup": ["source ~/workspace/opt/env.sh"],
            "env": {"ONEPLANNER_X": "x"},
        },
        {
            "python": "/submit/host/.venv/bin/python",
            "cwd": "/home/alice/mnt/node/OnePlanner",
            "batch_dir_relative": "outputs/2026-10-19/12-00-00",
            "env": {},
        },
        "/home/alice/mnt/node/OnePlanner/outputs/2026-10-19/12-00-00",
    ),
    "remote-home-and-passthrough": (
        {
            "remote_cwd": "~/work space",
            "python": "~/py/bin/python",
            "array_parallelism": 4,
            "partition": "gpu",
            "account": "a",
            "qos": "q",
            "mem": "8G",
            "collect_mem": None,
            "env_passthrough": ["ONEPLANNER_*"],
            "extra_sbatch_args": ["--nice=1"],
            "headless": False,
        },
        {
            "python": "/submit/python",
            "cwd": "/home/alice/mnt",
            "batch_dir_relative": "out dir/2026-10-19/12-00-00",
            "env": {"ONEPLANNER_FLAGS": '--a="b c"', "ONEPLANNER_Y": "y"},
        },
        "/home/alice/mnt/out dir/2026-10-19/12-00-00",
    ),
    "remote-absolute": (
        {"remote_cwd": "/scratch/alice", "python": "python3", "job_name": "n"},
        {
            "python": "/submit/python",
            "cwd": "/mnt/scratch",
            "batch_dir_relative": "o/d/t",
            "env": {},
        },
        "/mnt/scratch/o/d/t",
    ),
    "remote-home": (
        {"remote_cwd": "~", "python": "~/bin/python", "export": "ALL"},
        {
            "python": "/submit/python",
            "cwd": "/mnt/home",
            "batch_dir_relative": "o/d/t",
            "env": {"PATH": "/bin"},
        },
        "/mnt/home/o/d/t",
    ),
    "remote-without-relative-batch": (
        {"remote_cwd": "work", "python": "python3"},
        {"python": "/submit/python", "cwd": "/mnt/work", "env": {}},
        "/mnt/work/o/d/t",
    ),
}

_SHELL_PATHS = [
    "~",
    "~/",
    "~//",
    "~/venv/bin/python",
    "~//a b",
    "/opt/py/bin/python",
    ".venv/bin/python",
    "python3",
    "it's",
    "",
]

_RANGES = ["0-3,5", "0-3,5%2", "", "7", "1-2,,4-4", "10-12%1", "3,1"]

_RELATIVE_DIRS = [
    ("outputs/a/b", "/w"),
    ("/w/outputs/x", "/w"),
    ("/w/a/../b", "/w"),
    ("/elsewhere/x", "/w"),
    ("/w", "/w"),
    (".", "/w"),
]

#: squeue -h -r -o "%i %T" output for job 7, and its return code and stderr.
_SQUEUE = [
    ("7_0 RUNNING\n7_1 PENDING\n7_[2-4,6%2] PENDING\n8_9 RUNNING\n", 0, ""),
    ("7_[0-1] PENDING\n\nlonely\n7_x RUNNING\n7_3 COMPLETING extra\n", 0, ""),
    ("", 1, "slurm_load_jobs error: Invalid job id specified"),
    ("", 1, "squeue: error: connection refused"),
]


def _outcome(build: Any) -> Any:
    try:
        return build()
    except (ValueError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _squeue(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int, stderr: str
) -> dict[int, str] | None:
    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        assert command[:5] == ["squeue", "-h", "-r", "-j", "7"]
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(submit.subprocess, "run", run)
    return submit.squeue_states("7")


def _render(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Everything the golden file holds, built by slurm.submit."""
    out: dict[str, Any] = {}
    for name, (raw, manifest, batch) in _CASES.items():
        settings = SlurmSettings.from_mapping(raw)
        layout = BatchLayout(Path(batch))
        script = layout.job_script(3)
        collect = layout.root / "jobs" / "3-collect.sh"
        out[name] = {
            "job_script": {
                str(submission): _outcome(
                    lambda s=submission: submit.job_script(
                        layout, manifest, settings, s
                    )
                )
                for submission in (0, 3)
            },
            "sbatch_command": {
                str(tasks): submit.sbatch_command(layout, settings, script, tasks)
                for tasks in (1, 7)
            },
            "collect_script": _outcome(
                lambda: submit.collect_script(manifest, settings)
            ),
            "collect_command": _outcome(
                lambda: submit._collect_command(layout, manifest, settings, "41")
            ),
            "collect_command_with_script": _outcome(
                lambda: submit._collect_command(
                    layout, manifest, settings, "41_2", collect
                )
            ),
        }
    out["_shell_path"] = {path: submit._shell_path(path) for path in _SHELL_PATHS}
    out["_expand_ranges"] = {spec: submit._expand_ranges(spec) for spec in _RANGES}
    out["relative_batch_dir"] = [
        _outcome(lambda b=batch, c=cwd: submit.relative_batch_dir(Path(b), c))
        for batch, cwd in _RELATIVE_DIRS
    ]
    out["squeue_states"] = [_squeue(monkeypatch, *case) for case in _SQUEUE]
    # JSON's own types: tuples as lists, int keys as str.
    return json.loads(json.dumps(out))


def _golden() -> dict[str, Any]:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", sorted(_golden()))
def test_submit_builds_what_it_built_before_the_split(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _render(monkeypatch)[key] == _golden()[key]


def test_the_golden_file_covers_every_case() -> None:
    assert set(_golden()) == {
        *_CASES,
        "_shell_path",
        "_expand_ranges",
        "relative_batch_dir",
        "squeue_states",
    }


def test_submit_still_exports_every_name() -> None:
    for name in _SUBMIT_NAMES:
        assert hasattr(submit, name), name
    assert set(submit.__all__) >= {
        "callable_path",
        "collect_script",
        "follow",
        "job_script",
        "relative_batch_dir",
        "sbatch_command",
        "squeue_states",
        "submit_batch",
        "submit_cases",
    }


@pytest.mark.parametrize(
    "name",
    [
        "CLI_MODULE",
        "REMOTE_EARLY_OUTPUT_DIR",
        "WORKER_MODULE",
        "_expand_ranges",
        "_shell_path",
        "relative_batch_dir",
        "sbatch_command",
    ],
)
def test_submit_re_exports_the_scripts_module(name: str) -> None:
    assert getattr(submit, name) is getattr(scripts, name)


def test_the_batch_layout_is_the_checked_batch_paths() -> None:
    """manifest.BatchLayout keeps its readers, and its paths are scripts'."""
    layout = BatchLayout(Path("/b"))
    assert isinstance(layout, scripts.BatchPaths)
    assert layout == BatchLayout(Path("/b"))
    assert layout != scripts.BatchPaths(Path("/b"))
    assert repr(layout) == "BatchLayout(batch_dir=PosixPath('/b'))"
    assert hash(layout) == hash(BatchLayout(Path("/b")))
    assert layout.root == Path("/b/slurm")
    assert layout.manifest == Path("/b/slurm/manifest.json")
    assert layout.logs == Path("/b/slurm/logs")
    assert layout.job_script(2) == Path("/b/slurm/jobs/2.sh")
    assert layout.submission(2) == Path("/b/slurm/submissions/2.json")
    assert layout.task(1, 2) == Path("/b/slurm/tasks/1-2.json")
    assert layout.task_port(1, 2) == Path("/b/slurm/tasks/1-2.port")
    assert layout.result(4) == Path("/b/slurm/results/4.json")
    assert layout.start_lock == Path("/b/slurm/start.lock")
    with pytest.raises(AttributeError):
        layout.batch_dir = Path("/c")  # type: ignore[misc]
    with pytest.raises(AttributeError):
        layout.other = 1  # type: ignore[attr-defined]


if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as patch:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(_render(patch), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
