"""What a SLURM batch leaves in its output directory, and reads back.

A batch submitted with ``launcher=slurm`` writes its cases' outputs where a
local batch does (``outputs/<date>/<time>/``), under the same names, and keeps
what the launcher itself needs in ``slurm/`` beside them::

    slurm/manifest.json         every case: its config, overrides, file stem
    slurm/submissions/<k>.json  submission k: its job, which cases each task runs
    slurm/jobs/<k>.sh           the job script submission k runs
    slurm/tasks/<k>-<t>.json    task t of submission k: host, ports, progress
    slurm/results/<i>.json      case i's outcome, written as soon as it ran
    slurm/logs/                 each task's stdout and stderr
    slurm/start.lock            when the last task started (start_stagger_seconds)

Plain JSON, written atomically (a reader never sees half a file), and free of
heavy imports: the ``scenario-slurm`` commands only read these.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_VERSION",
    "RUN_FILE_SUFFIXES",
    "BatchLayout",
    "read_json",
    "sequential_stems",
    "write_json",
]

#: Bumped when the manifest changes in a way an older worker cannot read.
MANIFEST_VERSION = 1

#: The files a run writes under its stem (``scenario_runner``): what is
#: removed before a case runs again, so a rerun never leaves a stale one.
RUN_FILE_SUFFIXES: tuple[str, ...] = (
    "_result.json",
    ".trajectory.jsonl",
    "_coverage.json",
    "_odd.tar.gz",
    "_environment.json",
    "_requirements.lock",
    ".log",
)


def write_json(path: Path, data: Any) -> None:
    """Write *data* to *path* as JSON, atomically (rename over the old file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Read the JSON file *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def sequential_stems(scenario_classes: list[str]) -> list[str]:
    """The file stem each case of a batch is named by, in a local batch's way.

    A local batch names a run's files by its scenario class, and the next run
    of the same class ``{class}-1``, ``{class}-2`` and so on, in the order they
    ran (``scenario_runner._run_file_stem``).  A case keeps that name whichever
    task runs it, and when.
    """
    seen: dict[str, int] = {}
    stems: list[str] = []
    for name in scenario_classes:
        count = seen.get(name, 0)
        stems.append(name if count == 0 else f"{name}-{count}")
        seen[name] = count + 1
    return stems


@dataclass(frozen=True)
class BatchLayout:
    """Where the launcher's files of the batch in *batch_dir* are."""

    batch_dir: Path

    @property
    def root(self) -> Path:
        return self.batch_dir / "slurm"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def submission(self, index: int) -> Path:
        return self.root / "submissions" / f"{index}.json"

    def job_script(self, index: int) -> Path:
        return self.root / "jobs" / f"{index}.sh"

    def task(self, submission: int, task: int) -> Path:
        return self.root / "tasks" / f"{submission}-{task}.json"

    def task_port(self, submission: int, task: int) -> Path:
        """The CARLA RPC port task *task* holds (the job's leak guard reads it)."""
        return self.root / "tasks" / f"{submission}-{task}.port"

    @property
    def start_lock(self) -> Path:
        """When the batch's last task started (``launcher.start_stagger_seconds``)."""
        return self.root / "start.lock"

    def result(self, case: int) -> Path:
        return self.root / "results" / f"{case}.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def submissions(self) -> list[int]:
        """The indices of the batch's submissions, in order."""
        directory = self.root / "submissions"
        if not directory.is_dir():
            return []
        return sorted(int(p.stem) for p in directory.glob("*.json") if p.stem.isdigit())

    def next_submission(self) -> int:
        existing = self.submissions()
        return existing[-1] + 1 if existing else 0

    def read_manifest(self) -> dict[str, Any]:
        """The batch's manifest.

        Raises:
            FileNotFoundError: *batch_dir* holds no SLURM batch.
            ValueError: The manifest is of a version this framework cannot read.
        """
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"{self.batch_dir} holds no SLURM batch (no {self.manifest})"
            )
        manifest = read_json(self.manifest)
        version = manifest.get("version")
        if version != MANIFEST_VERSION:
            raise ValueError(
                f"{self.manifest} is a version {version} manifest; this "
                f"autoware_carla_scenario reads version {MANIFEST_VERSION}"
            )
        return manifest

    def read_submission(self, index: int) -> dict[str, Any]:
        return read_json(self.submission(index))

    def read_result(self, case: int) -> dict[str, Any] | None:
        path = self.result(case)
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None

    def read_task(self, submission: int, task: int) -> dict[str, Any] | None:
        path = self.task(submission, task)
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
