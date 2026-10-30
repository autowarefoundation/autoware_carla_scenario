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

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scripts import BatchPaths

__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_VERSIONS",
    "RUN_FILE_SUFFIXES",
    "BatchLayout",
    "config_files",
    "file_sha256",
    "resubmit_target",
    "read_json",
    "sequential_stems",
    "write_json",
]

#: Bumped when the manifest changes in a way an older worker cannot read.
#: Version 2 is a batch submitted with ``launcher.remote_cwd``: its jobs find
#: the batch at ``batch_dir_relative`` under their working directory, not at
#: ``batch_dir`` (the submit host's path), which an older framework would use.
#: A batch without ``remote_cwd`` is still written as version 1.
MANIFEST_VERSION = 2

#: The versions this framework reads.
MANIFEST_VERSIONS: tuple[int, ...] = (1, 2)

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
    # The Rerun recording of the trajectory log (rerun.enabled).
    ".rrd",
)


def file_sha256(path: Path) -> str:
    """The SHA-256 of the file *path*, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_key(prefix: str, key: Any) -> str:
    """The name of *key* under *prefix* in a config, as the worker reports a
    difference (``a.b``, ``a.list[2]``)."""
    if isinstance(key, int):
        return f"{prefix}[{key}]"
    return f"{prefix}.{key}" if prefix else str(key)


def config_files(
    config: Any, cache: dict[str, str] | None = None, prefix: str = ""
) -> dict[str, str]:
    """Every string of the resolved *config* that is the absolute path of a
    file here, by its key: the file's SHA-256.

    A task on a node whose paths differ from this host's (another mount,
    another home, the package installed elsewhere) resolves such a value to
    another path; it is the same config when the file there has the same
    content (``worker._check_same_config``).  *cache* holds the digests of
    the paths already read, across cases.
    """
    cache = {} if cache is None else cache
    found: dict[str, str] = {}
    if isinstance(config, dict):
        items: Any = config.items()
    elif isinstance(config, list):
        items = enumerate(config)
    else:
        if (
            isinstance(config, str)
            and config.startswith("/")
            and len(config) < 4096
            and "\n" not in config
            and "\0" not in config
        ):
            if config not in cache:
                try:
                    path = Path(config)
                    cache[config] = file_sha256(path) if path.is_file() else ""
                except OSError:
                    cache[config] = ""
            if cache[config]:
                found[prefix or "<root>"] = cache[config]
        return found
    for key, value in items:
        found.update(config_files(value, cache, config_key(prefix, key)))
    return found


def resubmit_target(layout: "BatchLayout", manifest: dict[str, Any]) -> str:
    """The batch's directory as ``scenario-slurm`` on the submit host takes
    it: a ``remote_cwd`` batch collected on a node names the submit host's
    path (the manifest's ``batch_dir``), not the node's."""
    if manifest.get("batch_dir_relative") and manifest.get("batch_dir"):
        return str(manifest["batch_dir"])
    return str(layout.batch_dir)


def write_json(path: Path, data: Any) -> None:
    """Write *data* to *path* as JSON, atomically (rename over the old file).

    The file is the user's alone (``0600``, as ``mkstemp`` creates it): the
    manifest holds the environment the batch was submitted with.
    """
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
class BatchLayout(BatchPaths):
    """Where the launcher's files of the batch in *batch_dir* are, and what
    they hold.

    The paths are :class:`~.scripts.BatchPaths`', which the library check
    compiles (#732); this reads the files.
    """

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
        if version not in MANIFEST_VERSIONS:
            raise ValueError(
                f"{self.manifest} is a version {version} manifest; this "
                "autoware_carla_scenario reads version(s) "
                + ", ".join(str(v) for v in MANIFEST_VERSIONS)
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
