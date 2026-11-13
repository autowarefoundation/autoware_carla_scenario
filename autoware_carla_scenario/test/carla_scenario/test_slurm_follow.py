"""What ``scenario-slurm wait`` (``submit.follow``), a submitter's last word
(``submit.after_submit``) and ``resubmit --cases`` make of a SLURM batch.

Pinned against the code that followed a submission and picked the cases to
resubmit before they moved into the library check (the ``_old_*`` oracles
below, copied from ``slurm/submit.py`` and ``slurm/cli.py`` as they were),
over generated batches: normal ones, records of every JSON type, keys named
twice, chunks missing, of every type and holding values of every type,
outcomes naming the submission as a number of every type (``true`` and
``1.0`` among them), cases of every shape, labels and ``--cases`` entries of
every kind, files missing, empty, not JSON or not objects, and values nested
deep.  ``squeue``, ``scancel``, ``sbatch`` and ``scontrol`` are fakes on
``PATH`` (squeue answering each job in turn, so that a submission is followed
for a few polls), and the clocks are fixed: ``time.time()`` never moves, and
``time.monotonic()`` moves only as far as the fake ``time.sleep()`` sleeps,
which may raise ``KeyboardInterrupt`` at a chosen call.  For each call (the
command run through scenario-slurm's main, or the function called), the exit
status or what it raises, stdout, stderr, the log, the commands the fakes ran,
the clocks' reads and sleeps in their order, and the files of the batch are
compared.
"""

from __future__ import annotations

import inspect
import json
import logging
import random
import shlex
import shutil
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.slurm import cli, submit
from autoware_carla_scenario.slurm import queue as queue_mod
from autoware_carla_scenario.slurm.collect import collect, unfinished_cases
from autoware_carla_scenario.slurm.manifest import (
    MANIFEST_VERSION,
    SBATCH_IN_FLIGHT_SECONDS,
    BatchLayout,
    recorded_verdict,
    sbatch_in_flight,
    sbatch_timed_out,
)
from autoware_carla_scenario.slurm.queue import (
    ACTIVE_STATES,
    CANCEL_WAIT_SECONDS,
    _wait_until_gone,
    cancel_jobs,
    squeue_states,
)
from autoware_carla_scenario.slurm.scripts import CLI_MODULE
from autoware_carla_scenario.slurm.settings import SlurmSettings
from autoware_carla_scenario.slurm.status import busy_reason

from .test_slurm_batch_readers import _write, nested, write_batch

#: The fixed wall clock.
_NOW = 1_800_000_000.25

# -- The oracles: the code as it was ------------------------------------------------

_busy = busy_reason
_refuse_if_busy = cli._refuse_if_busy
submit_cases = submit.submit_cases


def _layout(path: str) -> BatchLayout:
    return BatchLayout(Path(path).expanduser().resolve())


def _old_after_submit(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* to its end and collect (``wait``), or say
    how to; the exit status."""
    submission = layout.submission_record(index)
    if submission is None:
        print(  # noqa: T201
            f"Error: the submission record {layout.submission(index)} cannot be "
            f"read; see the batch with: scenario-slurm status {layout.batch_dir}",
            file=sys.stderr,
        )
        return 1
    if settings.wait:
        return _old_follow(layout, index, settings)
    if submission.get("collect_error"):
        return 1  # submit_cases said what failed, and how to collect
    cli = f"{shlex.quote(sys.executable)} -m {CLI_MODULE}"
    target = shlex.quote(str(layout.batch_dir))
    print(f"Submitted; not waiting (--no-wait). Job {submission['job_id']}.")  # noqa: T201
    if submission.get("collect_job_id"):
        print(  # noqa: T201
            f"Collect job {submission['collect_job_id']} writes "
            f"{layout.batch_dir / 'batch_results.json'} once the array ends."
        )
    print(f"  follow:   scenario-slurm wait {target}")  # noqa: T201
    print(f"  status:   scenario-slurm status {target}")  # noqa: T201
    print(f"  collect:  scenario-slurm collect {target}")  # noqa: T201
    print(f"  (or {cli} <command> {target})")  # noqa: T201
    return 0


def _old_follow(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """Follow submission *index* until SLURM lists none of its tasks, then
    collect the batch; the exit status (0 when every case passed).

    A submission with no job yet (its sbatch has not returned, timed out, or
    its submitter died before it did) is neither followed nor collected
    (#1417, #1791): ``squeue -j None`` would read as a job that ended."""
    from autoware_carla_scenario.slurm.collect import collect  # noqa: PLC0415

    submission = layout.submission_record(index)
    if submission is None:
        print(  # noqa: T201
            f"Error: the submission record {layout.submission(index)} is missing "
            "or cannot be read; collect the batch with: "
            f"scenario-slurm collect {layout.batch_dir}",
            file=sys.stderr,
        )
        return 1
    if not submission.get("job_id"):
        if sbatch_timed_out(submission):
            at = submission.get("sbatch_timed_out_at")
            # Only while the record still counts as in flight is a resubmit
            # refused (#1796); after that a plain resubmit is taken.
            refused = (
                f" (refused until {SBATCH_IN_FLIGHT_SECONDS:.0f} s after the "
                "record was made, unless with --force)"
                if sbatch_in_flight(submission)
                else ""
            )
            print(  # noqa: T201
                f"Error: submission {index} of this batch has no job id to "
                "follow: its sbatch timed out"
                + (f" at {at}" if at else "")
                + ", and SLURM may have taken the job all the same. If "
                "`squeue --me` lists it, collect the batch once it ends: "
                f"scenario-slurm collect {layout.batch_dir}; if not, run its "
                f"cases again: scenario-slurm resubmit {layout.batch_dir}" + refused,
                file=sys.stderr,
            )
        elif sbatch_in_flight(submission):
            print(  # noqa: T201
                f"Error: submission {index} of this batch is being submitted "
                "(sbatch has not returned yet), by another resubmit or the "
                "viewer; try again once it has: "
                f"scenario-slurm wait {layout.batch_dir}",
                file=sys.stderr,
            )
        else:
            print(  # noqa: T201
                f"Error: submission {index} of this batch has no job: its "
                "submitter stopped before sbatch returned, more than "
                f"{SBATCH_IN_FLIGHT_SECONDS:.0f} s ago, so the record is stale. "
                f"See the batch with: scenario-slurm status {layout.batch_dir}",
                file=sys.stderr,
            )
        return 1
    job = str(submission["job_id"])
    cases = [case for chunk in submission["chunks"] for case in chunk]
    last = ""
    unknown_since: float | None = None
    try:
        while True:
            states = squeue_states(job)
            done = passed = 0
            for case in cases:
                record = layout.read_result(case)
                if record is not None and record.get("submission") == index:
                    done += 1
                    passed += recorded_verdict(record) is True
            running = sum(1 for s in (states or {}).values() if s == "RUNNING")
            pending = sum(1 for s in (states or {}).values() if s == "PENDING")
            line = (
                f"[slurm] job {job}: {done}/{len(cases)} case(s) done "
                f"({passed} passed); {running} task(s) running, {pending} pending"
            )
            if line != last:
                print(line, flush=True)  # noqa: T201
                last = line
            if states is not None and not any(
                state in ACTIVE_STATES for state in states.values()
            ):
                break
            if states is None:
                # squeue failed: give it a few polls before giving up on it.
                unknown_since = unknown_since or time.monotonic()
                if time.monotonic() - unknown_since > 10 * settings.poll_seconds:
                    print(  # noqa: T201
                        "[slurm] squeue keeps failing; stopped following. Collect "
                        f"later with: scenario-slurm collect {layout.batch_dir}",
                        flush=True,
                    )
                    return 1
            else:
                unknown_since = None
            time.sleep(settings.poll_seconds)
    except KeyboardInterrupt:
        if settings.cancel_on_interrupt:
            cancel_jobs([job])
            print(  # noqa: T201
                f"\n[slurm] interrupted: cancelled job {job}. Run the cases that "
                f"did not finish with: scenario-slurm resubmit {layout.batch_dir}",
                flush=True,
            )
            _wait_until_gone(job, CANCEL_WAIT_SECONDS, settings.poll_seconds)
        else:
            print(  # noqa: T201
                f"\n[slurm] stopped following job {job}, which goes on. Follow it "
                f"again with: scenario-slurm wait {layout.batch_dir}",
                flush=True,
            )
            return 130
    return collect(layout)


def _old_resubmit(args: Any) -> int:
    """``scenario-slurm resubmit`` as it was."""
    layout = _layout(args.batch_dir)
    manifest = layout.read_manifest()
    reason = None if args.force else _busy(layout)
    if reason is not None:
        print(f"Error: {reason}", file=sys.stderr)  # noqa: T201
        return 1
    if args.cases:
        labels = {str(c["label"]): int(c["index"]) for c in manifest["cases"]}
        # A case named twice (by its label and its index, say) runs once.
        picked: dict[int, None] = {}
        for item in args.cases.split(","):
            item = item.strip()
            if item in labels:
                picked[labels[item]] = None
            elif (
                item.isascii()
                and item.isdigit()
                and 0 <= int(item) < len(manifest["cases"])
            ):
                picked[int(item)] = None
            else:
                print(f"Error: no case {item!r} in the batch", file=sys.stderr)  # noqa: T201
                return 1
        chosen = sorted(picked)
    else:
        chosen = unfinished_cases(layout, failed=args.failed)
    if not chosen:
        print("Nothing to resubmit: every case has a verdict.")  # noqa: T201
        # A dry run never collects: it writes nothing (#1768).
        return collect(layout) if args.wait and not args.dry_run else 0
    settings = SlurmSettings.from_mapping(manifest["launcher"]).replace(
        array_parallelism=args.array_parallelism,
        cases_per_task=args.cases_per_task,
        wait=args.wait,
    )
    labels_of = [manifest["cases"][i]["label"] for i in chosen]
    print(f"Resubmitting {len(chosen)} case(s): {', '.join(labels_of)}")  # noqa: T201
    if args.dry_run:
        return 0
    index = submit_cases(
        layout,
        manifest,
        settings,
        sorted(chosen),
        before_sbatch=None if args.force else _refuse_if_busy(layout),
    )
    return _old_after_submit(layout, index, settings)


@contextmanager
def old_following(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``wait``, ``after_submit`` and ``resubmit`` as they were (the
    oracles), called where the code calls them."""
    with monkeypatch.context() as patch:
        patch.setattr(submit, "follow", _old_follow)
        patch.setattr(submit, "after_submit", _old_after_submit)
        patch.setattr(cli, "after_submit", _old_after_submit)
        patch.setattr(cli, "_resubmit", _old_resubmit)
        yield


# -- The fakes: SLURM's tools and the clocks -----------------------------------------

#: squeue answers each job with the next of its answers (the last one once
#: they run out): ``"fail"``, ``"invalid"`` or the lines it prints.
_SQUEUE = """\
import json, os, sys
job = sys.argv[sys.argv.index("-j") + 1]
base = os.environ["FAKE_SLURM_DIR"]
with open(os.path.join(base, "calls.log"), "a", encoding="utf-8") as log:
    log.write(json.dumps(["squeue", *sys.argv[1:]]) + "\\n")
answers = json.load(open(os.path.join(base, "squeue.json"), encoding="utf-8"))
seen = json.load(open(os.path.join(base, "asked.json"), encoding="utf-8")) if (
    os.path.exists(os.path.join(base, "asked.json"))
) else {}
turn = seen.get(job, 0)
seen[job] = turn + 1
json.dump(seen, open(os.path.join(base, "asked.json"), "w", encoding="utf-8"))
mine = answers.get(job, ["invalid"])
answer = mine[min(turn, len(mine) - 1)]
if answer == "fail":
    sys.stderr.write("slurm_load_jobs error: Socket timed out\\n")
    sys.exit(1)
if answer == "invalid":
    sys.stderr.write("slurm_load_jobs error: Invalid job id specified\\n")
    sys.exit(1)
sys.stdout.write(answer)
"""

_LOGGED = """\
import json, os, sys
base = os.environ["FAKE_SLURM_DIR"]
name = os.path.basename(sys.argv[0])
with open(os.path.join(base, "calls.log"), "a", encoding="utf-8") as log:
    log.write(json.dumps([name, *sys.argv[1:]]) + "\\n")
if os.path.exists(os.path.join(base, name + ".fail")):
    sys.stderr.write(name + ": refused\\n")
    sys.exit(1)
if name == "sbatch":
    count = os.path.join(base, "sbatch.count")
    number = int(open(count).read()) if os.path.exists(count) else 0
    open(count, "w").write(str(number + 1))
    sys.stdout.write(f"{5000 + number};cluster\\n")
"""


def install_fakes(directory: Path) -> dict[str, str]:
    """Write fake ``squeue``, ``scancel``, ``sbatch`` and ``scontrol`` to
    *directory*; the environment that puts them first on ``PATH``."""
    bin_dir = directory / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("squeue", _SQUEUE),
        ("scancel", _LOGGED),
        ("sbatch", _LOGGED),
        ("scontrol", _LOGGED),
    ):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable} -SI\n{body}", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    # scontrol fails: MaxArraySize is SLURM's default.
    (directory / "scontrol.fail").write_text("", encoding="utf-8")
    import os  # noqa: PLC0415

    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_SLURM_DIR": str(directory),
    }


class FakeTime:
    """The clocks follow, the queue and the oracles read: the wall clock
    fixed, the monotonic one moved only by :meth:`sleep`, which raises
    ``KeyboardInterrupt`` at its *interrupt_at*-th call (1 the first)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []
        self.now = 1000.0
        self.sleeps = 0
        self.interrupt_at: set[int] = set()

    def reset(self, interrupt_at: set[int] | None = None) -> None:
        self.events = []
        self.now = 1000.0
        self.sleeps = 0
        self.interrupt_at = set(interrupt_at or ())

    def time(self) -> float:
        self.events.append(("time", _NOW))
        return _NOW

    def monotonic(self) -> float:
        self.events.append(("monotonic", self.now))
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.events.append(("sleep", seconds))
        if self.sleeps in self.interrupt_at:
            raise KeyboardInterrupt
        self.now += seconds


class Slurm:
    """The fakes of a test: where they log, what squeue answers, the clocks."""

    def __init__(self, directory: Path, clock: FakeTime) -> None:
        self.directory = directory
        self.clock = clock

    def answer(self, squeue: dict[str, list[str]]) -> None:
        (self.directory / "squeue.json").write_text(
            json.dumps(squeue), encoding="utf-8"
        )

    def reset(self, interrupt_at: set[int] | None = None) -> None:
        for name in ("calls.log", "asked.json", "sbatch.count"):
            (self.directory / name).unlink(missing_ok=True)
        self.clock.reset(interrupt_at)

    def calls(self) -> list[Any]:
        path = self.directory / "calls.log"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]


def _patch_clock(patch: pytest.MonkeyPatch, clock: FakeTime) -> None:
    """Hand *clock* to every module that follows a submission: submit, the
    queue, this module's oracles and wherever ``submit.follow`` is."""
    modules = {submit, queue_mod, sys.modules[__name__]}
    module = inspect.getmodule(submit.follow)
    if module is not None:
        modules.add(module)
    for module in modules:
        patch.setattr(module, "time", clock)


@pytest.fixture
def slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Slurm:
    env = install_fakes(tmp_path / "fake-slurm")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(time, "time", lambda: _NOW)
    monkeypatch.setattr(submit, "_now", lambda: "2026-11-13T00:00:00+00:00")
    clock = FakeTime()
    _patch_clock(monkeypatch, clock)
    return Slurm(tmp_path / "fake-slurm", clock)


# -- The corpus ---------------------------------------------------------------------

#: What an outcome may name as its submission.
_SUBMISSIONS = [
    "0",
    "1",
    "2",
    "true",
    "false",
    "1.0",
    "0.0",
    "-0.0",
    "1.5",
    "NaN",
    "Infinity",
    "1e400",
    "123456789012345678901234567890",
    '"0"',
    '"1"',
    "null",
    "[0]",
    "{}",
    "[]",
]


def _result_record(rng: random.Random, index: Any) -> str:
    members: list[tuple[str, str]] = []
    if rng.random() < 0.9:
        members.append(("submission", rng.choice(_SUBMISSIONS)))
    if rng.random() < 0.1:
        members.append(("submission", rng.choice(_SUBMISSIONS)))  # named twice
    passed = rng.choice(["true", "false", '"true"', "1"])
    result = (
        '{"passed": ' + passed + ', "message": "ok", "elapsed_seconds": 1.5}'
        if rng.random() < 0.9
        else rng.choice(["null", "3", '"x"', '{"passed": true}'])
    )
    members.append(("result", result))
    members.append(("errored", rng.choice(["false", "true"])))
    members.append(("index", json.dumps(index)))
    return "{" + ", ".join(f"{json.dumps(k)}: {v}" for k, v in members) + "}"


def write_follow_batch(
    batch_dir: Path, rng: random.Random
) -> tuple[dict[str, list[str]], list[str]]:
    """A batch of *rng*'s making (:func:`write_batch`'s, with outcomes that
    name a submission); squeue's answers to its jobs, each in turn, and the
    ``--cases`` values to resubmit."""
    first = write_batch(batch_dir, rng)
    root = batch_dir / "slurm"
    manifest = root / "manifest.json"
    if manifest.is_file() and rng.random() < 0.9:
        # What a job script reads of it, so that a resubmit gets to sbatch.
        text = manifest.read_text(encoding="utf-8", errors="surrogatepass")
        if text.startswith("{"):
            text = '{"python": "python3", "cwd": "/work", ' + text[1:]
            _write(manifest, text)
    submissions = root / "submissions"
    if rng.random() < 0.3:
        # One more, with no job id: its sbatch timed out, at a time or none.
        at = rng.choice(["null", '""', "0", '"2026-11-12T00:10:00"', "[]"])
        started = rng.choice([repr(_NOW - 10), repr(_NOW - 5000), "null"])
        _write(
            submissions / f"{len(BatchLayout(batch_dir).submissions())}.json",
            '{"job_id": null, "sbatch_timed_out": true, '
            f'"sbatch_timed_out_at": {at}, "submitted_at": {started}, '
            '"chunks": [[0]]}',
        )
    results = root / "results"
    shutil.rmtree(results, ignore_errors=True)
    for name in [*range(9), "a", "b", "x", "1.0", "True", "None"]:
        if rng.random() < 0.35:
            continue
        content: str | bytes = _result_record(rng, name)
        if rng.random() < 0.05:
            content = rng.choice([b"", b"{", b"[]", b"3", b"\xff"])
        _write(results / f"{name}.json", content)
    squeue: dict[str, list[str]] = {}
    for job, answer in first.items():
        if answer in ("fail", "invalid"):
            turns = [answer] * rng.choice([1, 3, 30])
        else:
            turns = [answer] * rng.choice([1, 2, 3])
            if rng.random() < 0.5:
                # Tasks running and pending, as followed.
                lines = [
                    f"{job}_{task} {rng.choice(['RUNNING', 'PENDING'])}\n"
                    for task in range(rng.randint(1, 3))
                ]
                turns.insert(0, "".join(lines))
        if rng.random() < 0.2:
            turns.append("fail")
        # Each ends: gone from the queue, or squeue failing from then on.
        turns.append(rng.choice(["", "", "", "invalid", "fail"]))
        squeue[job] = turns
    labels: list[str] = []
    try:
        cases = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
            "cases"
        ]
        for case in cases:
            try:
                labels.append(str(case["label"]))
            except Exception:  # noqa: BLE001, S112
                continue
    except Exception:  # noqa: BLE001, S110
        pass
    tokens = [
        *labels,
        *labels,
        "0",
        "1",
        "2",
        "7",
        "99",
        "-1",
        "01",
        "+1",
        " 1 ",
        "1.0",
        "",
        "x",
        "٣",  # an Arabic-Indic digit: isdigit(), not ASCII
        "²",  # a superscript two: isdigit(), not ASCII, int() refuses
        "True",
        "None",
    ]
    specs = []
    for _ in range(4):
        count = rng.choice([1, 1, 2, 3])
        specs.append(",".join(rng.choice(tokens) for _ in range(count)))
    specs.append(",".join(labels))
    return squeue, specs


# -- Running a call ------------------------------------------------------------------


def _text(value: Any) -> str:
    try:
        return json.dumps(value, indent=1, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return repr(value)


def _outcome(call: Callable[[], Any]) -> tuple:
    try:
        return ("returns", _text(call()))
    except Exception as exc:  # noqa: BLE001
        return ("raises", type(exc).__name__, str(exc))


class Seen:
    """What a call returns or raises, prints, logs, runs, reads of the
    clocks and writes; the batch put back as it was after each call."""

    def __init__(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
        slurm: Slurm,
        batch_dir: Path,
        snapshot: Path,
    ) -> None:
        self.capsys = capsys
        self.caplog = caplog
        self.slurm = slurm
        self.batch_dir = batch_dir
        self.snapshot = snapshot
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(batch_dir, snapshot, symlinks=True)

    def files(self) -> list[tuple[str, bytes | None]]:
        return sorted(
            (
                str(p.relative_to(self.batch_dir)),
                p.read_bytes() if p.is_file() else None,
            )
            for p in self.batch_dir.rglob("*")
        )

    def restore(self) -> None:
        shutil.rmtree(self.batch_dir)
        shutil.copytree(self.snapshot, self.batch_dir, symlinks=True)

    def __call__(
        self, call: Callable[[], Any], interrupt_at: set[int] | None = None
    ) -> tuple:
        self.capsys.readouterr()
        self.caplog.clear()
        self.slurm.reset(interrupt_at)
        try:
            with self.caplog.at_level(logging.DEBUG):
                outcome = _outcome(call)
            captured = self.capsys.readouterr()
            logs = [(r.name, r.levelno, r.getMessage()) for r in self.caplog.records]
            return (
                outcome,
                captured.out,
                captured.err,
                logs,
                self.slurm.calls(),
                list(self.slurm.clock.events),
                self.files(),
            )
        finally:
            self.restore()


#: A call, and the sleeps at which it is interrupted.
Call = tuple[Callable[[], Any], set[int]]


def _follow(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """``submit.follow``, looked up when called (the oracle, or not)."""
    return submit.follow(layout, index, settings)


def _after_submit(layout: BatchLayout, index: int, settings: SlurmSettings) -> int:
    """``submit.after_submit``, looked up when called (the oracle, or not)."""
    return submit.after_submit(layout, index, settings)


def _calls(batch_dir: Path, rng: random.Random, specs: list[str]) -> dict[str, Call]:
    """Each call of the batch: following each submission (and one past the
    last), after_submit of each, ``wait`` and ``resubmit --cases``."""
    layout = BatchLayout(batch_dir)
    batch = str(batch_dir)
    count = len(layout.submissions())
    calls: dict[str, Call] = {}
    for index in range(count + 1):
        poll = rng.choice([0.1, 1.0, 15.0])
        cancel = rng.random() < 0.6
        interrupt = rng.choice([set(), set(), {1}, {2}, {1, 3}])
        settings = SlurmSettings(poll_seconds=poll, cancel_on_interrupt=cancel)
        calls[f"follow-{index}"] = (
            partial(_follow, layout, index, settings),
            interrupt,
        )
        wait = rng.random() < 0.3
        after = SlurmSettings(wait=wait, poll_seconds=poll)
        calls[f"after-submit-{index}"] = (
            partial(_after_submit, layout, index, after),
            set(),
        )
    calls["wait"] = (lambda: cli.main(["wait", batch, "--poll-seconds", "0.5"]), set())
    calls["wait-0"] = (
        lambda: cli.main(["wait", batch, "--submission", "0", "--poll-seconds", "2"]),
        {2},
    )
    for number, spec in enumerate(specs):
        extra = rng.choice(
            [
                ["--dry-run"],
                ["--dry-run", "--force"],
                ["--force", "--no-wait"],
                ["--no-wait"],
                ["--force", "--wait"],
            ]
        )
        calls[f"resubmit-{number}"] = (
            partial(cli.main, ["resubmit", batch, f"--cases={spec}", *extra]),
            set(),
        )
    return calls


def _compare(
    batch_dir: Path,
    calls: dict[str, Call],
    seen: Seen,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, (call, interrupt_at) in calls.items():
        with old_following(monkeypatch):
            expected = seen(call, interrupt_at)
        actual = seen(call, interrupt_at)
        assert actual == expected, name


@pytest.mark.parametrize("seed", range(16))
def test_every_call_matches_the_oracle(
    tmp_path: Path,
    slurm: Slurm,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    rng = random.Random(seed)
    for number in range(4):
        batch_dir = tmp_path / f"batch-{number}"
        squeue, specs = write_follow_batch(batch_dir, rng)
        slurm.answer(squeue)
        seen = Seen(capsys, caplog, slurm, batch_dir, tmp_path / "snapshot")
        _compare(batch_dir, _calls(batch_dir, rng, specs), seen, monkeypatch)


def test_the_corpus_covers_what_the_calls_tell_apart(
    tmp_path: Path,
    slurm: Slurm,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corpus reaches every branch worth pinning."""
    texts: list[str] = []
    raised: set[str] = set()
    with old_following(monkeypatch):
        for seed in range(16):
            rng = random.Random(seed)
            for number in range(4):
                batch_dir = tmp_path / f"s{seed}" / f"batch-{number}"
                squeue, specs = write_follow_batch(batch_dir, rng)
                slurm.answer(squeue)
                seen = Seen(capsys, caplog, slurm, batch_dir, tmp_path / "snapshot")
                for call, interrupt_at in _calls(batch_dir, rng, specs).values():
                    outcome, out, err, logs, ran, events, _ = seen(call, interrupt_at)
                    if outcome[0] == "raises":
                        raised.add(outcome[1])
                    texts.append(
                        str(outcome)
                        + out
                        + err
                        + "".join(m for _, _, m in logs)
                        + json.dumps(ran)
                        + json.dumps(events)
                    )
    joined = "\n".join(texts)
    missing: list[str] = []
    for phrase in (
        "case(s) done (0 passed)",
        "case(s) done (1 passed)",
        "1 task(s) running",
        "1 pending",
        "squeue keeps failing",
        "interrupted: cancelled job",
        "stopped following job",
        "is still in the queue",
        "has no job id to follow: its sbatch timed out at",
        "has no job id to follow: its sbatch timed out,",
        "(refused until",
        "is being submitted",
        "so the record is stale",
        "is missing or cannot be read",
        "cannot be read; see the batch",
        "Submitted; not waiting (--no-wait). Job",
        "Collect job",
        "no case ",
        "Resubmitting",
        "Nothing to resubmit",
        "SLURM job 50",
        '"scancel"',
        '"sbatch"',
        '"sleep"',
        '"monotonic"',
    ):
        if phrase not in joined:
            missing.append(phrase)
    assert not missing, missing
    assert {"TypeError", "KeyError"} <= raised, raised


# -- Values nested deep --------------------------------------------------------------


def _deep_batch(batch_dir: Path, where: str, depth: int) -> None:
    root = batch_dir / "slurm"
    value = nested(depth)
    cases = [{"index": 0, "label": "a"}, {"index": 1, "label": "b"}]
    text = json.dumps({"version": MANIFEST_VERSION, "launcher": {}, "cases": cases})
    if where == "label":
        text = text.replace('"label": "a"', f'"label": {value}')
    _write(root / "manifest.json", text)
    members = {
        "job_id": '"1000"',
        "chunks": "[[0, 1]]",
        "cases": "[0, 1]",
        "collect_job_id": '"2000"',
    }
    if where == "job_id":
        members["job_id"] = value
    elif where == "chunks":
        members["chunks"] = f"[[0, 1], {value}]"
    elif where == "chunk":
        members["chunks"] = f"[[0, 1, {value}]]"
    elif where == "collect_error":
        members["collect_error"] = value
    elif where == "collect_job_id":
        members["collect_job_id"] = value
    elif where == "submission":
        members["deep"] = value
    text = "{" + ", ".join(f'"{k}": {v}' for k, v in members.items()) + "}"
    _write(root / "submissions" / "0.json", text)
    outcome = (
        '{"submission": 0, "errored": false, "result": {"passed": true, '
        '"message": "ok", "elapsed_seconds": 1.0}}'
    )
    if where == "result":
        outcome = outcome.replace('"submission": 0', f'"submission": {value}')
    _write(root / "results" / "0.json", outcome)
    _write(root / "results" / "1.json", outcome)


_DEEP = [
    "label",
    "job_id",
    "chunks",
    "chunk",
    "collect_error",
    "collect_job_id",
    "submission",
    "result",
]


def deep_calls(batch_dir: Path) -> dict[str, Call]:
    layout = BatchLayout(batch_dir)
    batch = str(batch_dir)
    return {
        "follow": (lambda: submit.follow(layout, 0, SlurmSettings()), set()),
        "after-submit": (
            lambda: submit.after_submit(layout, 0, SlurmSettings(wait=False)),
            set(),
        ),
        "wait": (lambda: cli.main(["wait", batch]), set()),
        "cases": (
            lambda: cli.main(["resubmit", batch, "--cases", "a,1,b", "--dry-run"]),
            set(),
        ),
        "cases-forced": (
            lambda: cli.main(["resubmit", batch, "--cases", "0", "--force", "--wait"]),
            set(),
        ),
    }


@pytest.mark.parametrize("where", _DEEP)
@pytest.mark.parametrize("depth", [800, 5000, 100_000])
def test_a_batch_nested_deep(
    tmp_path: Path,
    slurm: Slurm,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
    depth: int,
) -> None:
    """A value nested deep is read wherever it was, and what it raises
    where it was not is raised alike."""
    batch_dir = tmp_path / "batch"
    _deep_batch(batch_dir, where, depth)
    slurm.answer({"1000": ["1000_0 RUNNING\n", ""], "5000": [""]})
    seen = Seen(capsys, caplog, slurm, batch_dir, tmp_path / "snapshot")
    _compare(batch_dir, deep_calls(batch_dir), seen, monkeypatch)


# -- Word for word -------------------------------------------------------------------


def test_following_a_submission_word_for_word(
    tmp_path: Path, slurm: Slurm, capsys: pytest.CaptureFixture[str]
) -> None:
    batch_dir = tmp_path / "batch"
    root = batch_dir / "slurm"
    cases = [{"index": i, "label": f"c{i}"} for i in range(3)]
    _write(
        root / "manifest.json",
        json.dumps({"version": MANIFEST_VERSION, "launcher": {}, "cases": cases}),
    )
    _write(
        root / "submissions" / "0.json",
        json.dumps({"job_id": 1000, "chunks": [[0, 1], [2]], "cases": [0, 1, 2]}),
    )
    passing = {
        "submission": True,
        "errored": False,
        "result": {"passed": True, "message": "ok", "elapsed_seconds": 1.0},
    }
    _write(root / "results" / "0.json", json.dumps({**passing, "submission": 0.0}))
    _write(root / "results" / "1.json", json.dumps({**passing, "submission": 1}))
    slurm.answer({"1000": ["1000_0 RUNNING\n1000_1 PENDING\n", "fail", ""]})
    settings = SlurmSettings(poll_seconds=2.0)
    assert submit.follow(BatchLayout(batch_dir), 0, settings) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[:2] == [
        "[slurm] job 1000: 1/3 case(s) done (1 passed); 1 task(s) running, 1 pending",
        "[slurm] job 1000: 1/3 case(s) done (1 passed); 0 task(s) running, 0 pending",
    ]
    assert slurm.clock.events == [
        ("sleep", 2.0),
        ("monotonic", 1002.0),
        ("monotonic", 1002.0),
        ("sleep", 2.0),
    ]
    assert submit.follow(BatchLayout(batch_dir), 1, settings) == 1
    assert capsys.readouterr().err == (
        f"Error: the submission record {root / 'submissions' / '1.json'} is "
        "missing or cannot be read; collect the batch with: "
        f"scenario-slurm collect {batch_dir}\n"
    )


def test_resubmit_cases_word_for_word(
    tmp_path: Path, slurm: Slurm, capsys: pytest.CaptureFixture[str]
) -> None:
    batch_dir = tmp_path / "batch"
    root = batch_dir / "slurm"
    cases = [{"index": i, "label": f"c{i}"} for i in range(3)]
    _write(
        root / "manifest.json",
        json.dumps({"version": MANIFEST_VERSION, "launcher": {}, "cases": cases}),
    )
    batch = str(batch_dir)
    assert cli.main(["resubmit", batch, "--dry-run", "--cases", "c2, 0,2,c0"]) == 0
    assert capsys.readouterr().out == "Resubmitting 2 case(s): c0, c2\n"
    assert cli.main(["resubmit", batch, "--dry-run", "--cases", "c1,3"]) == 1
    assert capsys.readouterr().err == "Error: no case '3' in the batch\n"
    assert cli.main(["resubmit", batch, "--dry-run", "--cases", "١"]) == 1
    assert capsys.readouterr().err == "Error: no case '١' in the batch\n"
