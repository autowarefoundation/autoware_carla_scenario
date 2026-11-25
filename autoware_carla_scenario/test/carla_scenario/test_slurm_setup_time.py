"""#2483: a task runs ``launcher.setup`` in the batch's working directory and
free to read unset variables; ``launcher.time`` is text, never YAML's base-60
number; ``launcher.signal_grace_seconds`` is at most what ``--signal`` takes.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from autoware_carla_scenario.slurm import submit
from autoware_carla_scenario.slurm.scripts import (
    BatchPaths,
    ScriptInputs,
    collect_script,
    job_script,
)
from autoware_carla_scenario.slurm.settings import (
    MAX_SIGNAL_GRACE_SECONDS,
    SlurmSettings,
)

_BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(_BASH is None, reason="no bash")

#: A venv's activate script that reads an unset variable, as ROS's, colcon's
#: and Lmod's scripts do.
_ACTIVATE = """\
if [ -n "$COLCON_TRACE" ]; then echo trace; fi
export ACTIVATED_FROM="$PWD"
"""

#: The "interpreter": records where it ran and what setup left it.
_PYTHON = """\
#!/bin/sh
printf '%s\\n%s\\n' "$PWD" "${ACTIVATED_FROM:-none}" > "$RECORD"
"""


def _project(tmp_path: Path) -> tuple[Path, Path]:
    """A submitting directory (a name to quote) with a venv, and a fake python."""
    project = tmp_path / "proj dir $x"
    activate = project / ".venv" / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(_ACTIVATE, encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    return project, python


def _inputs(project: Path, python: Path) -> ScriptInputs:
    return ScriptInputs(
        python=str(python),
        cwd=str(project),
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        batch_dir_relative=None,
    )


@needs_bash
def test_the_job_script_parses(tmp_path: Path) -> None:
    project, python = _project(tmp_path)
    settings = SlurmSettings.from_mapping({"setup": ["source .venv/bin/activate"]})
    layout = BatchPaths(project / "outputs" / "o't")
    for text in (
        job_script(layout, _inputs(project, python), settings, 0),
        collect_script(_inputs(project, python), settings, layout),
    ):
        script = tmp_path / "job.sh"
        script.write_text(text, encoding="utf-8")
        subprocess.run([str(_BASH), "-n", str(script)], check=True)


def test_setup_runs_after_the_cd_to_the_batch_directory(tmp_path: Path) -> None:
    project, python = _project(tmp_path)
    settings = SlurmSettings.from_mapping({"setup": ["source .venv/bin/activate"]})
    layout = BatchPaths(project / "outputs" / "t")
    lines = job_script(layout, _inputs(project, python), settings, 0).splitlines()
    cd = lines.index(f"cd {shlex.quote(str(project))} || exit 1")
    setup = lines.index("source .venv/bin/activate")
    assert cd < setup
    assert lines[setup - 1] == "set +u" and lines[setup + 1] == "set -u"


@needs_bash
def test_a_task_sbatched_from_elsewhere_runs_setup_in_the_batch_directory(
    tmp_path: Path,
) -> None:
    """A resubmit from the batch's directory (or the viewer's) started the
    job there, so a relative ``source .venv/bin/activate`` found nothing (or
    another venv); and the activate script's unset variable ended the job
    under ``set -u`` before the worker started."""
    project, python = _project(tmp_path)
    settings = SlurmSettings.from_mapping(
        {"setup": ["source .venv/bin/activate"], "python": str(python)}
    )
    layout = BatchPaths(project / "outputs" / "t")
    script = tmp_path / "job.sh"
    script.write_text(
        job_script(layout, _inputs(project, python), settings, 0), encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    record = tmp_path / "record"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SLURM_ARRAY_TASK_ID": "0",
        "RECORD": str(record),
    }
    done = subprocess.run(
        [str(_BASH), str(script)],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        str(project),
        str(project),
    ]


def test_a_resubmit_cds_to_the_manifest_cwd_not_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manifest = {"python": "/usr/bin/python3", "cwd": "/work", "env": {}}
    layout = submit.BatchLayout(Path("/work/outputs/d/t"))
    settings = SlurmSettings.from_mapping({"setup": ["source .venv/bin/activate"]})
    lines = submit.job_script(layout, manifest, settings, 1).splitlines()
    assert lines.index("cd /work || exit 1") < lines.index("source .venv/bin/activate")
    assert str(elsewhere) not in "\n".join(lines)


def test_a_remote_task_still_runs_setup_in_its_remote_directory() -> None:
    settings = SlurmSettings.from_mapping(
        {
            "remote_cwd": "workspace/p",
            "python": ".venv/bin/python",
            "setup": ["source .venv/bin/activate"],
        }
    )
    manifest = ScriptInputs("/x/python", "/submit", {}, "outputs/d/t")
    lines = job_script(BatchPaths(Path("outputs/d/t")), manifest, settings, 0)
    lines_list = lines.splitlines()
    remote_cd = next(
        i for i, line in enumerate(lines_list) if line.startswith('cd "$HOME"/')
    )
    setup = lines_list.index("source .venv/bin/activate")
    assert remote_cd < setup
    # The submit host's directory is no path on the node.
    assert "cd /submit" not in lines


def test_the_collect_script_wraps_setup_in_set_plus_u(tmp_path: Path) -> None:
    project, python = _project(tmp_path)
    settings = SlurmSettings.from_mapping({"setup": ["source x"]})
    layout = BatchPaths(project / "outputs" / "t")
    lines = collect_script(_inputs(project, python), settings, layout).splitlines()
    setup = lines.index("source x")
    assert lines[setup - 1] == "set +u" and lines[setup + 1] == "set -u"


# ── launcher.time ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("yaml", "seconds"), [("12:00:00", 43200), ("120:00", 7200), ("90", 90)]
)
def test_an_unquoted_yaml_time_is_refused(yaml: str, seconds: int) -> None:
    launcher: Any = OmegaConf.to_container(OmegaConf.create(f"time: {yaml}\n"))
    assert launcher["time"] == seconds  # YAML 1.1's base-60 reading
    with pytest.raises(ValueError, match=r"launcher\.time must be text") as caught:
        SlurmSettings.from_mapping(launcher)
    assert 'time: "12:00:00"' in str(caught.value)


@pytest.mark.parametrize("value", [43200, 90, 0, 1.5, -1])
def test_a_number_time_is_refused(value: Any) -> None:
    with pytest.raises(ValueError, match=r"launcher\.time must be text"):
        SlurmSettings.from_mapping({"time": value})


@pytest.mark.parametrize("value", ["12:00:00", "120:00", "90", "2-00", "UNLIMITED"])
def test_a_quoted_time_is_taken_as_written(value: str) -> None:
    launcher: Any = OmegaConf.to_container(OmegaConf.create(f'time: "{value}"\n'))
    settings = SlurmSettings.from_mapping(launcher)
    assert settings.time == value
    assert f"--time={value}" in submit.sbatch_command(
        submit.BatchLayout(Path("/b")), settings, Path("/b/j.sh"), 1
    )


@pytest.mark.parametrize("value", [None, ""])
def test_a_null_or_blank_time_is_unset(value: Any) -> None:
    assert SlurmSettings.from_mapping({"time": value}).time is None


def test_a_bool_time_is_still_refused_as_before() -> None:
    with pytest.raises(ValueError, match=r"launcher\.time must be text, got True"):
        SlurmSettings.from_mapping({"time": True})


def test_the_shipped_launcher_config_reads() -> None:
    import autoware_carla_scenario

    conf = (
        Path(autoware_carla_scenario.__file__).parent
        / "examples"
        / "conf"
        / "launcher"
        / "slurm.yaml"
    )
    launcher: Any = OmegaConf.to_container(OmegaConf.load(conf))
    assert SlurmSettings.from_mapping(launcher).time is None


# ── launcher.signal_grace_seconds ──────────────────────────────────────────


@pytest.mark.parametrize("value", [0, 120, MAX_SIGNAL_GRACE_SECONDS, "65535"])
def test_signal_grace_up_to_65535_is_taken(value: Any) -> None:
    settings = SlurmSettings.from_mapping({"signal_grace_seconds": value})
    assert settings.signal_grace_seconds == int(value)


@pytest.mark.parametrize("value", [65536, "65536", 10**9])
def test_signal_grace_past_65535_is_refused(value: Any) -> None:
    with pytest.raises(
        ValueError, match=r"launcher\.signal_grace_seconds must be at most 65535"
    ):
        SlurmSettings.from_mapping({"signal_grace_seconds": value})
