"""What a SLURM batch leaves in its output directory, and reads back.

A batch submitted with ``launcher=slurm`` writes its cases' outputs where a
local batch does (``outputs/<date>/<time>/``), under the same names, and keeps
what the launcher itself needs in ``slurm/`` beside them::

    slurm/manifest.json         every case: its config, overrides, file stem;
                                which cases decide the batch-level settings
    slurm/submissions/<k>.json  submission k: its job, which cases each task runs
                                (created exclusively: its index is reserved)
    slurm/jobs/<k>.sh           the job script submission k runs
    slurm/tasks/<k>-<t>.json    task t of submission k: host, ports, progress
    slurm/results/<i>.json      case i's outcome, written as soon as it ran
    slurm/logs/                 each task's stdout and stderr
    slurm/start.lock            when the last task started (start_stagger_seconds)

Plain JSON, written atomically (a reader never sees half a file), and free of
heavy imports: the ``scenario-slurm`` commands only read these.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..run_outputs import RUN_FILE_SUFFIXES
from .scripts import BatchPaths
from .stems import config_key, sequential_stems

__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_VERSIONS",
    "RUN_FILE_SUFFIXES",
    "SBATCH_IN_FLIGHT_SECONDS",
    "BatchLayout",
    "config_files",
    "config_key",
    "file_sha256",
    "resubmit_target",
    "read_json",
    "read_submission_record",
    "sbatch_in_flight",
    "sequential_stems",
    "write_json",
]

logger = logging.getLogger(__name__)

#: Bumped when the manifest changes in a way an older worker cannot read.
#: Version 2 is a batch submitted with ``launcher.remote_cwd``: its jobs find
#: the batch at ``batch_dir_relative`` under their working directory, not at
#: ``batch_dir`` (the submit host's path), which an older framework would use.
#: A batch without ``remote_cwd`` is still written as version 1.
#:
#: ``batch_settings`` (which cases decide the batch's queue, Rerun converter,
#: canvas and metrics, #1225) needs no new version: a worker that predates it
#: ignores it, and one reading a manifest without it decides from the cases of
#: its own task, as workers did before.
MANIFEST_VERSION = 2

#: The versions this framework reads.
MANIFEST_VERSIONS: tuple[int, ...] = (1, 2)


def file_sha256(path: Path) -> str:
    """The SHA-256 of the file *path*, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def read_submission_record(path: Path) -> dict[str, Any] | None:
    """The submission record *path*, or ``None`` when there is none to read.

    A record that cannot be read as one -- empty, not JSON, or JSON that is
    not an object (outside corruption, or an ``O_CREAT | O_EXCL`` submitter
    that died mid-write, :func:`_create_exclusive`) -- is ``None`` with a
    warning naming the file (#1356).  A missing one (removed since it was
    listed: its sbatch failed) is ``None`` without one.
    """
    try:
        record = read_json(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record (%s)", path, exc
        )
        return None
    if not isinstance(record, dict):
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record "
            "(it holds JSON that is not an object)",
            path,
        )
        return None
    return record


#: How long a submission record whose ``job_id`` is still ``None`` counts as
#: in flight (its submitter running sbatch) from its ``submitted_at``.  Past
#: this, it is taken for one a crashed or killed submitter left behind, which
#: would otherwise refuse every resubmit of the batch (#1352).
SBATCH_IN_FLIGHT_SECONDS = 600.0


def sbatch_in_flight(submission: dict[str, Any], now: float | None = None) -> bool:
    """Whether *submission* is being submitted: its index is reserved, its
    sbatch has not returned yet (no ``job_id``), and it was reserved less than
    :data:`SBATCH_IN_FLIGHT_SECONDS` ago."""
    if submission.get("job_id"):
        return False
    started = submission.get("submitted_at")
    if not isinstance(started, (int, float)):
        return False
    now = time.time() if now is None else now
    return now - float(started) < SBATCH_IN_FLIGHT_SECONDS


def _create_exclusive(path: Path, data: Any) -> bool:
    """Create *path* holding *data* as JSON, unless it exists; whether it was
    created.

    The JSON is written to a temporary file first and hard-linked to *path*:
    ``link()`` fails if *path* exists, also on NFS (where ``flock()`` may not
    work and ``O_EXCL`` did not before NFSv3), and a reader never sees the
    file empty.  On NFS a link whose reply was lost is retried by the client
    and reports EEXIST although it was made: the temporary file's link count
    tells.  On a file system without hard links, ``O_CREAT | O_EXCL``.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return os.stat(tmp).st_nlink == 2
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
            try:
                created = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return False
            with os.fdopen(created, "w", encoding="utf-8") as stream:
                stream.write(text)
        return True
    finally:
        Path(tmp).unlink(missing_ok=True)


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
        """The index after the batch's last submission.  Not reserved: two
        submitters may both be told it (:meth:`reserve_submission`)."""
        existing = self.submissions()
        return existing[-1] + 1 if existing else 0

    def reserve_submission(self, record: dict[str, Any]) -> int:
        """Create the batch's next submission record from *record* (its
        ``index`` set), atomically; that index.

        Two submitters of one batch at once -- two viewer tabs, the viewer
        and ``scenario-slurm resubmit`` -- each get an index of their own
        (#1352): the record is created exclusively, and an index another
        submitter took is passed over.  :func:`write_json` then fills it in.
        """
        (self.root / "submissions").mkdir(parents=True, exist_ok=True)
        index = self.next_submission()
        while True:
            record["index"] = index
            if _create_exclusive(self.submission(index), record):
                return index
            index = max(index + 1, self.next_submission())

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

    def submission_record(self, index: int) -> dict[str, Any] | None:
        """Submission *index*'s record, or ``None`` when it is missing or
        cannot be read as one (:func:`read_submission_record`)."""
        return read_submission_record(self.submission(index))

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
