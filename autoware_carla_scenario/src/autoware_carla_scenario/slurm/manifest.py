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

The records are written as ``dict[str, Any]``, and read into a
:class:`Manifest` (its cases each a :class:`ManifestCase`), a
:class:`SubmissionRecord`, a :class:`CaseOutcome` and a :class:`TaskStatus`:
their members as their JSON text (:class:`.records.JsonMembers`), each read
when it is asked for.  The library check (docs/typecheck.md) compiles this
module, and Codon cannot hold a JSON document's values of mixed types beyond
the function that parses it: the version a manifest names is checked as the
JSON text of its value, and what creates a submission record exclusively is
given the record's JSON text.  The readers the check leaves out
(:mod:`.submit`, :mod:`.worker`, the gate, the viewer) read the manifest as
they read the ``dict`` it was (:meth:`Manifest.get`, ``manifest[key]``);
a :class:`Manifest` holds as JSON text only the members the checked readers
read of it, so that a manifest of many cases costs little more than that
``dict``.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..failure import Failure
from ..run_outputs import RUN_FILE_SUFFIXES
from .records import JsonMembers, parse_object
from .scripts import BatchPaths, file_sha256
from .stems import config_key, sequential_stems

__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_VERSIONS",
    "RUN_FILE_SUFFIXES",
    "SBATCH_IN_FLIGHT_SECONDS",
    "BatchLayout",
    "CaseOutcome",
    "Manifest",
    "ManifestCase",
    "SubmissionRecord",
    "TaskStatus",
    "config_files",
    "config_key",
    "file_sha256",
    "manifest_of_text",
    "resubmit_target",
    "read_json",
    "read_submission_record",
    "readable_result",
    "recorded_verdict",
    "no_job_note",
    "sbatch_in_flight",
    "sbatch_timed_out",
    "sequential_stems",
    "submission_record_of",
    "verdict_of_document",
    "write_json",
    "write_json_text",
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


def resubmit_target(layout: "BatchLayout", manifest: "Manifest") -> str:
    """The batch's directory as ``scenario-slurm`` on the submit host takes
    it: a ``remote_cwd`` batch collected on a node names the submit host's
    path (the manifest's ``batch_dir``), not the node's."""
    if manifest.is_true("batch_dir_relative") and manifest.is_true("batch_dir"):
        return manifest.text("batch_dir")
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


def write_json_text(path: Path, text: str) -> None:
    """Write *text*, a JSON document's text as :func:`write_json` writes it
    (:meth:`.records.JsonMembers.object_text`), to *path* as
    :func:`write_json` writes it: atomically, the user's alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    """Read the JSON file *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_submission_record(path: Path) -> SubmissionRecord | None:
    """The submission record *path*, or ``None`` when there is none to read.

    A record that cannot be read as one -- empty, not JSON, or JSON that is
    not an object (outside corruption, or an ``O_CREAT | O_EXCL`` submitter
    that died mid-write, :func:`_create_exclusive`) -- is ``None`` with a
    warning naming the file (#1356).  A missing one (removed since it was
    listed: its sbatch failed) is ``None`` without one.
    """
    # Parsed here, not in a helper: Python 3.10 counts the calls on the stack
    # with the levels of nesting json.loads() reads, and read_json() read
    # the record as deep as this reads it.
    try:
        members = parse_object(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record (%s)", path, exc
        )
        return None
    if members is None:
        logger.warning(
            "Ignoring %s: it cannot be read as a submission record "
            "(it holds JSON that is not an object)",
            path,
        )
        return None
    return submission_record_of(members)


#: How long a submission record whose ``job_id`` is still ``None`` counts as
#: in flight (its submitter running sbatch) from its ``submitted_at``.  Past
#: this, it is taken for one a crashed or killed submitter left behind, which
#: would otherwise refuse every resubmit of the batch (#1352).
SBATCH_IN_FLIGHT_SECONDS = 600.0


def sbatch_in_flight(submission: SubmissionRecord, now: float | None = None) -> bool:
    """Whether *submission* is being submitted: its index is reserved, its
    sbatch has not returned yet (no ``job_id``), and it was reserved less than
    :data:`SBATCH_IN_FLIGHT_SECONDS` ago.

    The clock is read only for a record with no job id and a number as its
    ``submitted_at``; an integer no float holds raises
    :class:`OverflowError`, after it is read."""
    if submission.has_job:
        return False
    if not submission.started_is_number():
        return False
    clock = time.time() if now is None else now
    return clock - submission.started_seconds() < SBATCH_IN_FLIGHT_SECONDS


def sbatch_timed_out(submission: SubmissionRecord) -> bool:
    """Whether *submission* has no job id because its sbatch timed out
    (``sbatch_timed_out``, which ``submit_cases`` records, #1791): SLURM may
    have taken the job all the same, under a job id nobody was told.

    Such a record still counts as in flight (:func:`sbatch_in_flight`) for
    :data:`SBATCH_IN_FLIGHT_SECONDS`, as one whose sbatch has not returned."""
    return not submission.has_job and submission.says_timed_out


def no_job_note(submission: SubmissionRecord, now: float | None = None) -> str:
    """Why *submission*, which has no ``job_id``, has none (#1791), as
    ``status``, ``wait``, ``resubmit`` and the viewer say it: its sbatch timed
    out (the job may exist), it is still in flight, or the record is stale (its
    submitter stopped before sbatch returned)."""
    if sbatch_timed_out(submission):
        at = submission.timed_out_at_text()
        return (
            "no job id: sbatch timed out"
            + ("" if at is None else f" at {at}")
            + "; the job may exist all the same: check `squeue --me`"
        )
    if sbatch_in_flight(submission, now):
        return "no job id yet (sbatch in flight)"
    return (
        "no job id: its submitter stopped before sbatch returned, more than "
        f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s ago (a stale record)"
    )


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
    return _link_exclusive(path, fd, tmp, text)


def _link_exclusive(path: Path, fd: int, tmp: str, text: str) -> bool:
    """Create *path* holding *text*, unless it exists, through the temporary
    file *tmp* (open as *fd*), which is removed; whether it was created
    (:func:`_create_exclusive`)."""
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
                # 0o600: Codon has no octal literal.
                created = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
            except FileExistsError:
                return False
            with os.fdopen(created, "w", encoding="utf-8") as stream:
                stream.write(text)
        return True
    finally:
        Path(tmp).unlink(missing_ok=True)


def _value_json(members: JsonMembers, key: str) -> str:
    """The JSON text of the value of the member *key* of *members*, ``null``
    when there is none: what ``dict.get(key)`` gave of the object."""
    return members.value_text(key) if members.has(key) else "null"


def submission_record_of(members: JsonMembers) -> SubmissionRecord:
    """The submission record of *members*, the members of a submission's
    JSON object (:class:`SubmissionRecord`)."""
    has_job = bool(json.loads(_value_json(members, "job_id")))
    started = json.loads(_value_json(members, "submitted_at"))
    started_is_number = isinstance(started, int) or isinstance(started, float)
    timed_out = json.loads(_value_json(members, "sbatch_timed_out"))
    says_timed_out = isinstance(timed_out, bool) and bool(timed_out)
    return SubmissionRecord(members, has_job, started_is_number, says_timed_out)


class SubmissionRecord:
    """A submission's record (``submissions/<k>.json``,
    :meth:`BatchLayout.submission_record`): the record's members, each read
    when it is asked for, as the record read as a ``dict`` gave it.

    The record is what its submitter wrote (``slurm/submit.py``'s
    ``submit_cases``): the submission's ``index``, when it was ``created``
    and ``submitted_at`` (``time.time()``, missing from a record older than
    #1352), its ``cases`` and the ``chunks`` its tasks run, its ``launcher``
    settings, its ``job_id`` and ``collect_job_id`` (``null`` until sbatch
    returns), its ``sbatch`` command line, and either that its sbatch timed
    out (``sbatch_timed_out``, ``sbatch_timed_out_at``, #1791) or the
    ``collect_sbatch`` of the job collecting it and why that failed
    (``collect_error``) -- or anything a hand or another version left there.

    :meth:`get` and ``record[key]`` read a member as ``dict.get(key)`` and
    ``dict[key]`` did, for the readers that pass its value on; the typed
    accessors read what the checked readers ask of it.

    Made by :func:`submission_record_of`, which reads whether it has a job
    id, a number as ``submitted_at`` and says its sbatch timed out when the
    record is read, as deep in the stack as the record was read as a whole:
    asking them, as ``sbatch_in_flight()`` and ``no_job_note()`` do, reads
    nothing more, as the dict's ``get()`` did not, wherever they are asked
    (Python 3.10 counts the calls on the stack with the levels of nesting
    ``json.loads()`` reads, and 3.12 counts a constructor's, called from C,
    against the C stack ``json.loads()`` reads on).
    """

    members: JsonMembers
    _has_job: bool
    _started_is_number: bool
    _says_timed_out: bool

    def __init__(
        self,
        members: JsonMembers,
        has_job: bool,
        started_is_number: bool,
        says_timed_out: bool,
    ) -> None:
        self.members = members
        self._has_job = has_job
        self._started_is_number = started_is_number
        self._says_timed_out = says_timed_out

    def has(self, key: str) -> bool:
        """Whether the record has the member *key*."""
        return self.members.has(key)

    def get(self, key: str) -> Any:
        """The value of the member *key*, ``None`` when there is none: the
        record's ``dict.get(key)``."""
        return json.loads(_value_json(self.members, key))

    def __getitem__(self, key: str) -> Any:
        """The value of the member *key*: the record's ``dict[key]``, a
        :class:`KeyError` when there is none."""
        return self.members.value(key)

    @property
    def has_job(self) -> bool:
        """Whether the record has a job id: a true ``job_id`` (of any type,
        as ``bool()`` reads it)."""
        return self._has_job

    def job_text(self) -> str:
        """The record's ``job_id`` as text: ``f"{record.get('job_id')}"``
        (``"None"`` when it has none)."""
        return str(json.loads(_value_json(self.members, "job_id")))

    def is_true(self, key: str) -> bool:
        """Whether the member *key* is true, as ``bool()`` reads its value
        (false when there is none): ``bool(record.get(key))``."""
        return bool(json.loads(_value_json(self.members, key)))

    def text(self, key: str) -> str:
        """The member *key* as text: ``str(record[key])``, a
        :class:`KeyError` when there is none."""
        return str(self.members.value(key))

    def size_of(self, key: str) -> int:
        """``len(record.get(key) or [])``: 0 when the record has no true
        *key*, the :class:`TypeError` CPython raises for a true value with
        no length (a number, ``true``)."""
        value = json.loads(_value_json(self.members, key))
        if not bool(value):
            return 0
        return len(value)

    def chunk_list(self) -> list[list[int]]:
        """The record's ``chunks``, as ``enumerate(record.get("chunks") or
        [])`` gave them: none when they are false, the :class:`TypeError`
        CPython raises for chunks that cannot be iterated (a number,
        ``true``).  Each chunk is the value the record holds (a list of the
        cases its task runs in a record its submitter wrote; an object's key
        or a string's character in place of one), which Python alone hands
        on: its length and its cases are read as they were."""
        chunks = json.loads(_value_json(self.members, "chunks"))
        if not bool(chunks):
            return []
        if TYPE_CHECKING:
            return [[int(case) for case in chunk] for chunk in chunks]
        else:
            return list(chunks)

    def chunk_cases(self) -> list[int]:
        """The cases of the record's ``chunks``, chunk after chunk, as
        ``[case for chunk in record["chunks"] for case in chunk]`` gave
        them: a :class:`KeyError` when it has none, the :class:`TypeError`
        CPython raises for chunks, or a chunk, that cannot be iterated (a
        number, ``true`` or ``null``).  Each case is the value the record
        holds (an int in a record its submitter wrote; an object's key or a
        string's character, and any value a hand left, in place of one),
        which Python alone hands on."""
        chunks = self.members.value("chunks")
        if TYPE_CHECKING:
            return [int(case) for chunk in chunks for case in chunk]
        else:
            return [case for chunk in chunks for case in chunk]

    def started_is_number(self) -> bool:
        """Whether the record's ``submitted_at`` is a number: an int (a
        boolean among them) or a float, as ``isinstance(value, (int,
        float))`` read it."""
        return self._started_is_number

    def started_seconds(self) -> float:
        """The record's ``submitted_at`` as ``float()`` reads it (an
        :class:`OverflowError` for an integer no float holds), when it is a
        number (:meth:`started_is_number`)."""
        return float(json.loads(_value_json(self.members, "submitted_at")))

    @property
    def says_timed_out(self) -> bool:
        """Whether the record says its sbatch timed out: its
        ``sbatch_timed_out`` is ``true`` (``is True``: no other true value)."""
        return self._says_timed_out

    def timed_out_at_text(self) -> str | None:
        """When the record says its sbatch timed out, as text: a true
        ``sbatch_timed_out_at`` (of any type, as ``bool()`` reads it), as
        an f-string gives it."""
        at = json.loads(_value_json(self.members, "sbatch_timed_out_at"))
        if bool(at):
            return str(at)
        return None

    def task_of(self, case: int) -> int | None:
        """The first of the record's tasks that runs *case*: the index in its
        ``chunks`` (none when they are false) of the first chunk holding it
        (``case in chunk``), or ``None``.  Raises what that raises: a
        :class:`TypeError` for ``chunks`` that cannot be iterated or a chunk
        that cannot hold a number (a string, a number, ``null``)."""
        chunks = json.loads(_value_json(self.members, "chunks"))
        if not bool(chunks):
            return None
        task = 0
        for chunk in chunks:
            if isinstance(chunk, list):
                # ``case in chunk``: an element equal to it, as ``==`` reads
                # a number (``1.0``, ``true`` for 1); never another value.
                for item in chunk:
                    if isinstance(item, bool):
                        if int(bool(item)) == case:
                            return task
                    elif isinstance(item, int):
                        if int(item) == case:
                            return task
                    elif isinstance(item, float):
                        number = float(item)
                        if (
                            math.isfinite(number)
                            and number == math.floor(number)
                            and int(number) == case
                        ):
                            return task
            elif not isinstance(chunk, dict):
                # A string, a number, true or null: the TypeError CPython
                # raises for ``case in chunk``, whose text differs between
                # its versions.  (An object holds no number: its keys are
                # strings.)
                if TYPE_CHECKING:
                    pass
                else:
                    if case in chunk:
                        return task
            task += 1
        return None


#: The members of a manifest :class:`Manifest` holds as their JSON text: what
#: the checked readers read of it (:meth:`.BatchLayout.read_manifest`'s
#: version, :func:`resubmit_target`, :mod:`.collect`).  The others (its
#: ``cases`` among them, read through :meth:`Manifest.cases`) are read as
#: the readers the check leaves out read them (:meth:`Manifest.get`).
_MANIFEST_MEMBERS: tuple[str, ...] = (
    "version",
    "expand",
    "batch_dir",
    "batch_dir_relative",
)

#: The members of a case :class:`ManifestCase` holds as their JSON text: what
#: the checked readers read of it (:mod:`.collect`).
_CASE_MEMBERS: tuple[str, ...] = ("index", "label", "stem", "case_overrides")


def manifest_of_text(text: str) -> Manifest | None:
    """The manifest of the JSON text *text*, as ``json.loads()`` reads it
    (a key named twice keeps its first place and its last value), or
    ``None`` when *text* is JSON that is not an object.

    Raises what ``json.loads()`` raises for text that is not JSON (a
    :class:`ValueError`, a :class:`RecursionError` for text nested too deep
    for it), and nothing else.  The members the checked readers read
    (``_MANIFEST_MEMBERS``, and ``_CASE_MEMBERS`` of each case) are
    written back here with the C encoder, as :class:`.records.JsonMembers`
    holds a value, so that nothing of the manifest is read again deeper in
    the stack than ``json.loads()`` read it (Python 3.10 counts the calls on
    the stack with the levels of nesting it reads); nothing else is written
    again, so the manifest costs little more than the document
    ``json.loads()`` read.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        return None
    members = JsonMembers()
    keys: list[str] = []
    cases: list[ManifestCase] = []
    # The shape of ``cases`` when it is a value Python cannot iterate.
    cases_shape = ""
    for key, value in data.items():
        keys.append(str(key))
        if str(key) in _MANIFEST_MEMBERS:
            members.put(str(key), json.dumps(value, ensure_ascii=False))
        if str(key) != "cases":
            continue
        cases = []
        cases_shape = ""
        if isinstance(value, list) or isinstance(value, dict) or isinstance(value, str):
            # A case of each item ``for case in manifest["cases"]`` gave:
            # a list's items, an object's keys, a string's characters.
            for item in value:
                case = JsonMembers()
                shape = ""
                if isinstance(item, dict):
                    for name, field in item.items():
                        if str(name) in _CASE_MEMBERS:
                            case.put(str(name), json.dumps(field, ensure_ascii=False))
                elif isinstance(item, list):
                    shape = "[]"
                elif isinstance(item, str):
                    shape = '""'
                elif isinstance(item, bool):
                    shape = "false"
                elif isinstance(item, int):
                    shape = "0"
                elif isinstance(item, float):
                    shape = "0.0"
                else:
                    shape = "null"
                cases.append(ManifestCase(case, shape))
        else:
            # A number, a boolean or null: the value itself, whose truth
            # ``manifest.get("cases") or []`` reads.
            cases_shape = json.dumps(value, ensure_ascii=False)
    manifest = Manifest(members, keys, cases, cases_shape)
    if not TYPE_CHECKING:
        # The manifest as json.loads() read it, for the readers the check
        # leaves out (Manifest.get(), manifest[key]).
        manifest._document = data
    return manifest


def _not_held(what: str, key: str) -> ValueError:
    """The error of asking a manifest or a case for the text of a member it
    does not hold (``_MANIFEST_MEMBERS``, ``_CASE_MEMBERS``)."""
    return ValueError(
        f"a {what} read by read_manifest() holds no {key!r} as JSON text: "
        "only what the checked readers read of it"
    )


class ManifestCase:
    """A case of a batch's manifest (:meth:`Manifest.cases`): the members
    the checked readers read of it (``_CASE_MEMBERS``: its ``index``,
    ``label``, file ``stem`` and ``case_overrides``), each read when it is
    asked for, as the case read as a ``dict`` gave it.  The rest of the case
    (its resolved ``config``, the ``files`` it names, ...) is read through
    ``manifest["cases"]`` (:meth:`Manifest.get`).

    The case is what its submitter wrote (``slurm/submit.py``), or anything
    a hand or another version left there: a ``stem`` is missing from a
    manifest older than #1957.  A case that is not an object (a hand's,
    another version's) is held as a value of its type (*shape*: ``[]``,
    ``""``, ``0``, ``0.0``, ``false`` or ``null``, empty for an object), so
    that reading a member of it raises what CPython raises for its type
    (``'int' object is not subscriptable``; a list, a string or a number in
    its place raises alike whatever it holds).
    """

    members: JsonMembers
    _shape: str

    def __init__(self, members: JsonMembers, shape: str = "") -> None:
        self.members = members
        self._shape = shape

    def _value(self, key: str) -> Any:
        """The value of the member *key*, one of ``_CASE_MEMBERS``: the
        case's ``case[key]``, a :class:`KeyError` when there is none (a
        :class:`TypeError` for a case that is not an object)."""
        if key not in _CASE_MEMBERS:
            raise _not_held("case", key)
        if self._shape != "":
            return json.loads(self._shape)[key]
        return self.members.value(key)

    def _get(self, key: str) -> Any:
        """The value of the member *key*, one of ``_CASE_MEMBERS``,
        ``None`` when there is none: the case's ``case.get(key)`` (an
        :class:`AttributeError` for a case that is not an object)."""
        if key not in _CASE_MEMBERS:
            raise _not_held("case", key)
        if self._shape != "":
            return json.loads(self._shape).get(key)
        return json.loads(_value_json(self.members, key))

    def value_json(self, key: str) -> str:
        """The JSON text of the member *key*, one of ``_CASE_MEMBERS``,
        ``null`` when there is none: the case's ``case.get(key)`` (an
        :class:`AttributeError` for a case that is not an object)."""
        if key not in _CASE_MEMBERS:
            raise _not_held("case", key)
        if self._shape != "":
            json.loads(self._shape).get(key)
        return _value_json(self.members, key)

    def index(self) -> int:
        """The case's ``index`` as ``int(case["index"])`` reads it: a
        :class:`KeyError` when it has none, a :class:`TypeError` or a
        :class:`ValueError` (an :class:`OverflowError` for an infinity) for
        a value ``int()`` cannot read."""
        return int(self._value("index"))

    def index_value(self) -> int:
        """The case's ``index`` as the manifest holds it: ``case["index"]``,
        an int in a manifest its submitter wrote (and any value a hand left,
        which Python alone hands on)."""
        value = self._value("index")
        if TYPE_CHECKING:
            return int(value)
        else:
            return value

    def label(self) -> str:
        """The case's ``label`` as text: ``str(case["label"])``."""
        return str(self._value("label"))

    def stem(self) -> str:
        """The case's file stem as text, empty when it has none (a manifest
        older than #1957): ``str(case.get("stem") or "")``."""
        stem = self._get("stem")
        if bool(stem):
            return str(stem)
        return ""

    def case_overrides(self) -> list[str]:
        """The overrides that make the case, as ``list(case["case_overrides"])``
        reads them: a list's items (the strings its submitter wrote, and
        any value a hand left, which Python alone hands on), an object's
        keys, a string's characters; a :class:`TypeError` for a number or
        ``null``."""
        value = self._value("case_overrides")
        if TYPE_CHECKING:
            return [str(item) for item in value]
        else:
            return list(value)


class Manifest:
    """A batch's manifest (``slurm/manifest.json``,
    :meth:`BatchLayout.read_manifest`): the members the checked readers read
    of it, each read when it is asked for, and its cases
    (:class:`ManifestCase`).

    The manifest is what its submitter wrote (``slurm/submit.py``): its
    ``version``, when it was ``created``, its ``batch_dir`` (and, with
    ``launcher.remote_cwd``, its ``batch_dir_relative``), the ``cwd``,
    ``python`` and ``argv`` that submitted it, its ``scenario_names``,
    ``overrides`` and whether it expands them (``expand``), its
    ``build_scenario_fn``, whether it was ``typechecked``, its ``launcher``
    settings, the ``env`` its jobs set again, which cases decide its
    ``batch_settings`` (missing from a manifest older than #1225), and its
    ``cases`` -- or anything a hand or another version left there.

    :meth:`get`, ``manifest[key]`` and ``key in manifest`` read it as the
    ``dict`` it was read as before, for the readers the library check leaves
    out (:mod:`.submit`, :mod:`.worker`, :mod:`.cli`, the gate, the viewer):
    the very values ``json.loads()`` read.  The typed accessors read what the
    checked readers (:mod:`.collect`, :func:`resubmit_target`) ask of it,
    which the manifest holds as JSON text: ``_MANIFEST_MEMBERS``, and
    ``_CASE_MEMBERS`` of each case.

    Made by :func:`manifest_of_text`, which reads its cases when the
    manifest is read, as deep in the stack as ``json.loads()`` read it.
    """

    #: The members of ``_MANIFEST_MEMBERS`` it has.
    members: JsonMembers
    _cases: list[ManifestCase]
    #: ``cases`` as its JSON text when Python cannot iterate it (a number,
    #: a boolean or ``null``); empty when it can.
    _cases_shape: str
    #: The name of each of its members, in its place.
    _keys: list[str]
    #: The manifest as ``json.loads()`` read it, for :meth:`get` and
    #: ``manifest[key]``: Python alone holds it (manifest_of_text()).
    _document: dict[str, Any]

    def __init__(
        self,
        members: JsonMembers,
        keys: list[str],
        cases: list[ManifestCase],
        cases_shape: str = "",
    ) -> None:
        self.members = members
        self._keys = keys
        self._cases = cases
        self._cases_shape = cases_shape
        self._document = {}

    def keys(self) -> list[str]:
        """The name of each of the manifest's members, in its place:
        ``list(manifest)`` of the manifest read as a ``dict``."""
        return list(self._keys)

    def has(self, key: str) -> bool:
        """Whether the manifest has the member *key*."""
        return key in self._keys

    def __contains__(self, key: str) -> bool:
        """``key in manifest``: whether it has the member *key*."""
        return key in self._keys

    def get(self, key: str, default: Any = None) -> Any:
        """The value of the member *key*, *default* when there is none: the
        manifest's ``dict.get(key, default)``, the value ``json.loads()``
        read."""
        return self._document.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """The value of the member *key*: the manifest's ``dict[key]``, the
        value ``json.loads()`` read; a :class:`KeyError` when there is
        none."""
        return self._document[key]

    def _held_text(self, key: str) -> str:
        """The JSON text of the member *key*, ``null`` when there is none;
        a :class:`ValueError` for a member it has that it does not hold as
        text (one not of ``_MANIFEST_MEMBERS``)."""
        if key not in _MANIFEST_MEMBERS and self.has(key):
            raise _not_held("manifest", key)
        return _value_json(self.members, key)

    def version_text(self) -> str:
        """The JSON text of the manifest's ``version``, ``null`` when it has
        none (:meth:`BatchLayout._check_version`)."""
        return self._held_text("version")

    def is_true(self, key: str) -> bool:
        """Whether the member *key* (one of ``_MANIFEST_MEMBERS``) is
        true, as ``bool()`` reads its value (false when there is none):
        ``bool(manifest.get(key))``."""
        return bool(json.loads(self._held_text(key)))

    def text(self, key: str) -> str:
        """The member *key* (one of ``_MANIFEST_MEMBERS``) as text:
        ``str(manifest[key])``, a :class:`KeyError` when there is none."""
        if key not in _MANIFEST_MEMBERS and self.has(key):
            raise _not_held("manifest", key)
        return str(self.members.value(key))

    def listed_cases(self) -> list[ManifestCase]:
        """The manifest's cases, as ``for case in manifest.get("cases") or
        []`` gave them: none when it has none or false ones, the
        :class:`TypeError` CPython raises for true ``cases`` it cannot
        iterate (a number, ``true``)."""
        if not self.has("cases"):
            return []
        if self._cases_shape != "":
            cases = json.loads(self._cases_shape)
            if not bool(cases):
                return []
            for _ in cases:
                pass
        return self._cases

    def cases(self) -> list[ManifestCase]:
        """The manifest's cases, as ``for case in manifest["cases"]`` gave
        them: a :class:`KeyError` when it has none, the
        :class:`TypeError` CPython raises for ``cases`` it cannot iterate
        (a number, ``true`` or ``null``)."""
        if not self.has("cases"):
            raise KeyError("cases")
        if self._cases_shape != "":
            for _ in json.loads(self._cases_shape):
                pass
        return self._cases


class CaseOutcome:
    """A case's outcome as its task recorded it (``results/<i>.json``,
    :meth:`BatchLayout.read_result`): the record's members, each read when
    it is asked for, as the record read as a ``dict`` gave it.

    The record is what the task wrote (``slurm/worker.py``): the case's
    ``index``, ``label`` and ``stem``, its ``result``
    (``ScenarioResult.to_dict()``), whether it ``errored``, the
    ``submission``, ``task``, ``job_id`` and ``host`` that ran it, its
    ``wall_seconds`` and when it ``finished``, and why its Rerun recording
    failed (``rerun_error``), if it did -- or anything a hand or another
    version left there.
    """

    members: JsonMembers

    def __init__(self, members: JsonMembers) -> None:
        self.members = members

    def has(self, key: str) -> bool:
        """Whether the record has the member *key*."""
        return self.members.has(key)

    def get(self, key: str) -> Any:
        """The value of the member *key*, ``None`` when there is none: the
        record's ``dict.get(key)``."""
        return json.loads(_value_json(self.members, key))

    @property
    def errored(self) -> bool:
        """Whether the record says the case raised rather than ran: a true
        ``errored`` (of any type, as ``bool()`` reads it)."""
        return bool(json.loads(_value_json(self.members, "errored")))

    def names_submission(self, submission: int) -> bool:
        """Whether the record's ``submission`` is *submission*: the
        record's ``get("submission") == submission``, which a number equal
        to it (``1.0``, ``true`` for 1) is too; never another value."""
        named = json.loads(_value_json(self.members, "submission"))
        if isinstance(named, bool):
            return int(bool(named)) == submission
        if isinstance(named, int):
            return int(named) == submission
        if isinstance(named, float):
            number = float(named)
            return (
                math.isfinite(number)
                and number == math.floor(number)
                and int(number) == submission
            )
        return False

    @property
    def rerun_error(self) -> str | None:
        """Why the case's Rerun recording failed, if the record says: a true
        ``rerun_error`` (of any type, as ``bool()`` reads it), as ``str()``
        gives it."""
        why = json.loads(_value_json(self.members, "rerun_error"))
        if bool(why):
            return str(why)
        return None

    def result(self) -> JsonMembers | None:
        """The members of the record's ``result``, or ``None`` when it has
        none or one that is not an object."""
        if not self.members.has("result"):
            return None
        result = parse_object(self.members.value_text("result"))
        if result is None:
            return None
        return result


class TaskStatus:
    """A task's status as it last wrote it (``tasks/<k>-<t>.json``,
    :meth:`BatchLayout.read_task`): the record's members, each read when it
    is asked for, as the record read as a ``dict`` gave it.

    The record is what the task writes (``slurm/worker.py``'s ``_Task``):
    its ``submission``, ``task``, ``job_id``, ``array_job_id``, ``host`` and
    ``pid``, its ``state``, when it ``started``, its ``cases``, those
    ``done`` and the ``current`` one, and, as its run adds them, when it was
    ``updated``, its ``ros_domains`` and ``ports``, when it ``finished``, its
    ``error`` and ``traceback`` and its ``peak_*_rss_mb`` -- or anything a
    hand or another version left there.  An empty record
    (``TaskStatus(JsonMembers())``) is what its readers take for a task that
    has none.
    """

    members: JsonMembers

    def __init__(self, members: JsonMembers) -> None:
        self.members = members

    def has(self, key: str) -> bool:
        """Whether the record has the member *key*."""
        return self.members.has(key)

    def get(self, key: str) -> Any:
        """The value of the member *key*, ``None`` when there is none: the
        record's ``dict.get(key)``."""
        return json.loads(_value_json(self.members, key))

    def state_text(self, default: str) -> str:
        """The task's ``state`` as text, *default* when the record has none:
        ``f"{status.get('state', default)}"`` of the record read as a
        ``dict``."""
        if not self.members.has("state"):
            return default
        return str(json.loads(self.members.value_text("state")))

    def joined(self, prefix: str, key: str) -> str:
        """*prefix* and the member *key* when it is true, else empty:
        ``prefix + status.get(key) if status.get(key) else ""``, the
        :class:`TypeError` CPython raises for a true value that is not a
        string."""
        value = json.loads(_value_json(self.members, key))
        if not bool(value):
            return ""
        if TYPE_CHECKING:
            return prefix + str(value)
        else:
            return prefix + value

    def port_text(self) -> str:
        """The task's RPC port as ``status`` shows it: ``" port " +
        str(ports["rpc"])`` when ``ports = status.get("ports") or {}`` has a
        true ``rpc``, else empty; the :class:`AttributeError` CPython raises
        for true ``ports`` that are not an object."""
        ports = json.loads(_value_json(self.members, "ports"))
        if not bool(ports):
            return ""
        if not bool(ports.get("rpc")):
            return ""
        return " port " + str(ports["rpc"])

    def size_of(self, key: str) -> int:
        """``len(status.get(key) or [])``: 0 when the record has no true
        *key*, the :class:`TypeError` CPython raises for a true value with
        no length (a number, ``true``)."""
        value = json.loads(_value_json(self.members, key))
        if not bool(value):
            return 0
        return len(value)

    def true_text(self, key: str) -> str | None:
        """The member *key* as text when it is true (``f"{value}"``), else
        ``None``."""
        value = json.loads(_value_json(self.members, key))
        if bool(value):
            return str(value)
        return None

    @property
    def error(self) -> str | None:
        """Why the task failed, if the record says: a true ``error`` (of any
        type, as ``bool()`` reads it), as ``str()`` gives it."""
        why = json.loads(_value_json(self.members, "error"))
        if bool(why):
            return str(why)
        return None

    def state_in(self, states: frozenset[str]) -> bool:
        """Whether the task's ``state`` is one of *states*: the record's
        ``get("state", "?") in states``, which raises the
        :class:`TypeError` CPython raises for a ``state`` no set holds (a
        list, an object)."""
        state = json.loads(_value_json(self.members, "state"))
        if isinstance(state, str):
            return str(state) in states
        if isinstance(state, list) or isinstance(state, dict):
            if TYPE_CHECKING:
                # The error CPython raises for a value no set holds, whose
                # text differs between its versions.
                return False
            else:
                return state in states
        return False

    def is_at(self, case: int) -> bool:
        """Whether the task's ``current`` case is *case*: the record's
        ``get("current") == case``, which a number equal to it (``1.0``,
        ``true`` for 1) is too."""
        current = json.loads(_value_json(self.members, "current"))
        if isinstance(current, bool):
            return int(bool(current)) == case
        if isinstance(current, int):
            return int(current) == case
        if isinstance(current, float):
            number = float(current)
            return (
                math.isfinite(number)
                and number == math.floor(number)
                and int(number) == case
            )
        return False


def _real_number(value_json: str) -> bool:
    """Whether the JSON text *value_json* is a number of seconds."""
    value = json.loads(value_json)
    # ``True`` is an int, but no number of seconds; nor is an integer
    # literal too long for a float (``float()`` overflows on it).
    if isinstance(value, bool):
        return False
    if not (isinstance(value, int) or isinstance(value, float)):
        return False
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _readable_status(status_json: str) -> bool:
    """Whether the JSON text *status_json* is a condition's status that can
    be reported: an object with a string ``label`` and ``message`` and a
    boolean ``satisfied``."""
    status = json.loads(status_json)
    if not isinstance(status, dict):
        return False
    label = status.get("label")
    satisfied = status.get("satisfied")
    message = status.get("message")
    return (
        isinstance(label, str)
        and isinstance(satisfied, bool)
        and isinstance(message, str)
    )


def readable_result(result: JsonMembers) -> bool:
    """Whether *result*, the members of a recorded ``result``
    (``ScenarioResult.to_dict()``), can be reported (#1726): its ``message``
    is a string, its ``elapsed_seconds`` a number (not a boolean), its
    ``condition_statuses``, if any, a list of objects each with a string
    ``label`` and ``message`` and a boolean ``satisfied``, and its
    ``failure``, if any, an object.  What ``to_dict()`` leaves out
    (``rerun_recording``, ``failure``) may be missing; ``passed`` is
    :func:`recorded_verdict`'s to check."""
    message = json.loads(_value_json(result, "message"))
    if not isinstance(message, str):
        return False
    if not _real_number(_value_json(result, "elapsed_seconds")):
        return False
    statuses_json = _value_json(result, "condition_statuses")
    if statuses_json != "null":
        statuses = json.loads(statuses_json)
        if not isinstance(statuses, list):
            return False
        for status in statuses:
            if not _readable_status(json.dumps(status)):
                return False
    failure_json = _value_json(result, "failure")
    if failure_json == "null":
        return True
    failure = json.loads(failure_json)
    if not isinstance(failure, dict):
        return False
    try:
        Failure.from_dict(failure)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return False
    return True


def recorded_verdict(record: CaseOutcome | None) -> bool | None:
    """Whether the outcome a task recorded for a case (``results/<i>.json``,
    read by :meth:`BatchLayout.read_result`) passed: its ``result``'s
    ``passed`` when ``result`` is an object and ``passed`` a boolean, else
    ``None`` -- a record that holds no verdict, which every reader takes as
    not passed (#1718), never ``"false"`` or ``1`` as a pass.

    A ``result`` that cannot be reported (:func:`readable_result`: its
    ``message``, ``elapsed_seconds``, ``condition_statuses`` or ``failure``
    missing or of the wrong type) holds no verdict either (#1726), so that the counts of
    ``scenario-slurm status`` and ``wait``, the viewer, ``collect`` and
    ``resubmit`` all agree that it did not pass."""
    if record is None:
        return None
    # Annotated: Codon 0.19 refuses the call's value for the local without.
    members: JsonMembers | None = record.result()
    if members is None:
        return None
    return _verdict_of_result(members)


def _verdict_of_result(result: JsonMembers) -> bool | None:
    """:func:`recorded_verdict` of the members of a recorded ``result``."""
    passed = json.loads(_value_json(result, "passed"))
    if not isinstance(passed, bool):
        return None
    verdict = bool(passed)
    if readable_result(result):
        return verdict
    return None


def verdict_of_document(document: Any) -> bool | None:
    """:func:`recorded_verdict` of a case's outcome as ``json.loads()``
    reads it (``ui.cases`` reads the file itself): ``None`` for a document
    that is not an object, or whose ``result`` is not one.

    Each member of the ``result`` is written back as
    :class:`.records.JsonMembers` holds a value, with the C encoder, and is
    not read again as part of the whole: what ``json.loads()`` read, nested
    however deep, is read here too, on Python 3.10 (which counts the calls
    on the stack with the levels of nesting) as on 3.12."""
    if not isinstance(document, dict):
        return None
    result = document.get("result")
    if not isinstance(result, dict):
        return None
    members = JsonMembers()
    for key, value in result.items():
        members.set_text(key, json.dumps(value, ensure_ascii=False))
    return _verdict_of_result(members)


@dataclass(frozen=True)
class BatchLayout(BatchPaths):
    """Where the launcher's files of the batch in *batch_dir* are, and what
    they hold.

    The paths, and which submissions the batch holds (:meth:`submissions`,
    :meth:`next_submission`), are :class:`~.scripts.BatchPaths`', which the
    library check compiles (#732, #1648); this reads the files.
    """

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

    def read_manifest(self) -> Manifest:
        """The batch's manifest (:class:`Manifest`).

        Raises:
            FileNotFoundError: *batch_dir* holds no SLURM batch.
            ValueError: The manifest is not JSON, not a JSON object, or of a
                version this framework cannot read (#1630).
            OSError: The manifest cannot be read.
        """
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"{self.batch_dir} holds no SLURM batch (no {self.manifest})"
            )
        # Parsed in a helper, as read_json() parsed it: Python 3.10 counts
        # the calls on the stack with the levels of nesting json.loads()
        # reads.
        manifest = manifest_of_text(self.manifest.read_text(encoding="utf-8"))
        if manifest is None:
            raise ValueError(
                f"{self.manifest} is not a SLURM batch manifest "
                "(it holds JSON that is not an object)"
            )
        # The version as JSON text, which json.loads() reads back as the
        # value the manifest holds.
        self._check_version(manifest.version_text())
        return manifest

    def _check_version(self, text: str) -> None:
        """Refuse a manifest whose ``version``, the JSON text *text* of its
        value (``null`` when it has none), is not one this framework reads
        (:meth:`read_manifest`).

        Raises:
            ValueError: That version is not one of :data:`MANIFEST_VERSIONS`.
        """
        version = json.loads(text)
        # ``True == 1``: a bool is no version, though ``in`` would match it.
        # A value read from JSON equals a version only as a number: an int
        # (as itself) or a float (as the float of the version).
        known = False
        if isinstance(version, bool):
            known = False
        elif isinstance(version, int):
            known = int(version) in MANIFEST_VERSIONS
        elif isinstance(version, float):
            known = float(version) in [float(v) for v in MANIFEST_VERSIONS]
        if not known:
            raise ValueError(
                f"{self.manifest} is a version {str(version)} manifest; this "
                "autoware_carla_scenario reads version(s) "
                + ", ".join(str(v) for v in MANIFEST_VERSIONS)
            )

    def read_submission(self, index: int) -> SubmissionRecord | None:
        """Submission *index*'s record, or ``None`` when it holds JSON that
        is not an object; raises what reading it as JSON raises
        (:class:`FileNotFoundError` when it is missing, another
        :class:`OSError`, a :class:`ValueError` for what is not JSON).

        Parsed here, as :func:`read_submission_record` parses it."""
        members = parse_object(self.submission(index).read_text(encoding="utf-8"))
        if members is None:
            return None
        return submission_record_of(members)

    def submission_record(self, index: int) -> SubmissionRecord | None:
        """Submission *index*'s record, or ``None`` when it is missing or
        cannot be read as one (:func:`read_submission_record`)."""
        return read_submission_record(self.submission(index))

    def read_result(self, case: int) -> CaseOutcome | None:
        """Case *case*'s outcome, or ``None`` when there is none to read:
        missing, cannot be read, not JSON (not UTF-8 among them), or JSON
        that is not an object (#1630)."""
        path = self.result(case)
        if not path.is_file():
            return None
        try:
            members = parse_object(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if members is None:
            return None
        return CaseOutcome(members)

    def read_task(self, submission: int, task: int) -> TaskStatus | None:
        """Task *task* of submission *submission*'s status, or ``None`` when
        there is none to read: missing, cannot be read, not JSON (not UTF-8
        among them), or JSON that is not an object (#1630)."""
        path = self.task(submission, task)
        if not path.is_file():
            return None
        try:
            members = parse_object(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if members is None:
            return None
        return TaskStatus(members)
