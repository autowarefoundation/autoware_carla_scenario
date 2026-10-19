"""A stand-in for SLURM's sbatch, squeue, scancel and scontrol, for the tests.

``fake_slurm.py <command> <args>``: an array job's tasks run as local
subprocesses, at most ``%K`` at once, each in a process group of its own (what
``scancel`` signals), with the variables SLURM gives a task.  The state lives in
``$FAKE_SLURM_STATE``: ``jobs/<id>/<task>.json`` per task, ``sbatch.log`` (one
JSON line per submission, its argv).

Only what the launcher uses: ``sbatch --parsable [--array=a-b%K]
[--dependency=afterany:J] [--output=P] [--wrap=CMD | script]``, ``squeue -h -r
-j J -o "%i %T"``, ``scancel J...``, ``scontrol show job J``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("FAKE_SLURM_STATE", "/tmp/fake_slurm"))
#: Seconds between SIGTERM and SIGKILL on scancel (SLURM's KillWait).
KILL_WAIT_S = float(os.environ.get("FAKE_SLURM_KILL_WAIT", "5"))
ACTIVE = ("PENDING", "RUNNING")


def _job_dir(job: str) -> Path:
    return STATE / "jobs" / str(job)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    for _ in range(50):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.02)
    return {}


def _tasks(job: str) -> dict[int, dict]:
    directory = _job_dir(job)
    if not directory.is_dir():
        return {}
    return {int(p.stem): _read(p) for p in directory.glob("*.json") if p.stem.isdigit()}


def _next_job_id() -> str:
    counter = STATE / "counter"
    STATE.mkdir(parents=True, exist_ok=True)
    import fcntl

    with open(STATE / "counter.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = int(counter.read_text()) + 1 if counter.exists() else 1000
        counter.write_text(str(value))
    return str(value)


def sbatch(args: list[str]) -> int:
    array = None
    dependency = None
    output = "slurm-%j.out"
    wrap = None
    script = None
    rest = []
    for arg in args:
        if arg.startswith("--array="):
            array = arg.split("=", 1)[1]
        elif arg.startswith("--dependency="):
            dependency = arg.split("=", 1)[1]
        elif arg.startswith("--output="):
            output = arg.split("=", 1)[1]
        elif arg.startswith("--wrap="):
            wrap = arg.split("=", 1)[1]
        elif arg.startswith("-"):
            rest.append(arg)
        else:
            script = arg
    if script is None and wrap is None:
        print("sbatch: error: no script", file=sys.stderr)
        return 1
    job = _next_job_id()
    with open(STATE / "sbatch.log", "a") as log:
        log.write(json.dumps({"job": job, "argv": args}) + "\n")
    if array is not None:
        spec, _, limit = array.partition("%")
        first, _, last = spec.partition("-")
        tasks = list(range(int(first), int(last or first) + 1))
        parallel = int(limit) if limit else len(tasks)
    else:
        tasks, parallel = [0], 1
    for task in tasks:
        _write(_job_dir(job) / f"{task}.json", {"state": "PENDING"})
    plan = {
        "array": array is not None,
        "tasks": tasks,
        "parallel": parallel,
        "dependency": dependency,
        "output": output,
        "command": ["bash", "-c", wrap] if wrap is not None else ["bash", script],
    }
    _write(_job_dir(job) / "plan.json", plan)
    subprocess.Popen(
        [sys.executable, __file__, "_run", job],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(job)
    return 0


def _run(job: str) -> int:
    plan = _read(_job_dir(job) / "plan.json")
    dependency = plan.get("dependency")
    if dependency:
        waited = dependency.split(":", 1)[1]
        while any(t.get("state") in ACTIVE for t in _tasks(waited).values()):
            time.sleep(0.2)
    running: dict[int, subprocess.Popen] = {}
    pending = list(plan["tasks"])
    while pending or running:
        for task, proc in list(running.items()):
            code = proc.poll()
            if code is not None:
                state = _read(_job_dir(job) / f"{task}.json")
                if state.get("state") != "CANCELLED":
                    state["state"] = "COMPLETED" if code == 0 else "FAILED"
                state["exit_code"] = code
                _write(_job_dir(job) / f"{task}.json", state)
                del running[task]
        while pending and len(running) < plan["parallel"]:
            task = pending.pop(0)
            if _read(_job_dir(job) / f"{task}.json").get("state") == "CANCELLED":
                continue
            task_id = str(int(job) * 100 + task)
            env = dict(os.environ)
            env.update(
                {
                    "SLURM_JOB_ID": task_id if plan["array"] else job,
                    "SLURM_ARRAY_JOB_ID": job,
                    "SLURM_ARRAY_TASK_ID": str(task),
                }
            )
            if not plan["array"]:
                env.pop("SLURM_ARRAY_TASK_ID")
                env.pop("SLURM_ARRAY_JOB_ID")
            out = (
                plan["output"]
                .replace("%A", job)
                .replace("%a", str(task))
                .replace("%j", task_id if plan["array"] else job)
            )
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            stream = open(out, "ab")
            proc = subprocess.Popen(
                plan["command"],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            running[task] = proc
            _write(
                _job_dir(job) / f"{task}.json",
                {"state": "RUNNING", "pid": proc.pid, "pgid": proc.pid},
            )
        time.sleep(0.1)
    return 0


def squeue(args: list[str]) -> int:
    job = args[args.index("-j") + 1]
    tasks = _tasks(job)
    if not tasks:
        print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
        return 1
    plan = _read(_job_dir(job) / "plan.json")
    for task, state in sorted(tasks.items()):
        if state.get("state") in ACTIVE:
            name = f"{job}_{task}" if plan.get("array") else job
            print(f"{name} {state['state']}")
    return 0


def scancel(args: list[str]) -> int:
    for job in args:
        if job.startswith("-"):
            continue
        for task, state in _tasks(job).items():
            if state.get("state") == "PENDING":
                _write(_job_dir(job) / f"{task}.json", {"state": "CANCELLED"})
            elif state.get("state") == "RUNNING":
                state["state"] = "CANCELLED"
                _write(_job_dir(job) / f"{task}.json", state)
                try:
                    os.killpg(state["pgid"], signal.SIGTERM)
                except ProcessLookupError:
                    continue
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import os,signal,sys,time\n"
                        f"time.sleep({KILL_WAIT_S})\n"
                        "try: os.killpg(int(sys.argv[1]), signal.SIGKILL)\n"
                        "except ProcessLookupError: pass",
                        str(state["pgid"]),
                    ],
                    start_new_session=True,
                )
    return 0


def scontrol(args: list[str]) -> int:
    job = args[-1]
    for task, state in sorted(_tasks(job).items()):
        print(
            f"JobId={job}_{task} ArrayTaskId={task}\n   JobState={state.get('state')}"
        )
    return 0


def main() -> int:
    command, args = sys.argv[1], sys.argv[2:]
    if command == "_run":
        return _run(args[0])
    return {
        "sbatch": sbatch,
        "squeue": squeue,
        "scancel": scancel,
        "scontrol": scontrol,
    }[command](args)


if __name__ == "__main__":
    sys.exit(main())
