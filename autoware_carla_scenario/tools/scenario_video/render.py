"""Record example scenarios as side-by-side CARLA / sumo-gui videos.

    python render.py <cases.txt> <work_dir> [--only NAME ...] [--min-free-gb 6]

Needs a CARLA server on localhost:2000 with the map's town loaded or loadable,
the ``sumo`` extra, ffmpeg, and a display for sumo-gui.  Each line of the cases
file is ``name|scenario|ego spawn lanelet|title|extra overrides``: the scenario
is expanded on Town10HD_Opt (``scenario-expand``), the case whose ego spawns on
that lanelet is run with ``traffic=sumo``, and

* ``chase.py`` films the ego and logs CARLA's lights,
* SUMO writes its FCD and, through ``tls_add.xml``, its signal states,
* ``sumo_replay.py`` replays both in sumo-gui, frame for frame,
* ``compose.py`` lays the two side by side into ``<work_dir>/<name>.mp4``,
* ``check_run.py`` compares the signals and measures walking speeds.

One run at a time, and none started while less than ``--min-free-gb`` of memory
is available.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
SCENARIO = str(Path(PYTHON).with_name("scenario"))
EXPAND = str(Path(PYTHON).with_name("scenario-expand"))
MAP = "map=town10hd_opt"


def _available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _wait_for_memory(minimum: float) -> None:
    while (free := _available_gb()) < minimum:
        print(f"  waiting: {free:.1f} GB available, want {minimum:.1f}")
        time.sleep(30)


def _case_overrides(scenario: str, spawn: str) -> list[str]:
    """The override list of the case of *scenario* whose ego spawns on *spawn*."""
    out = subprocess.run(
        [EXPAND, f"scenario={scenario}", MAP],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cases = json.loads(out)["cases"]
    for case in cases:
        if case and case[0] == f"ego.spawn_lanelet_id={spawn}":
            return case
    raise SystemExit(f"{scenario}: no case spawns the ego on lanelet {spawn}")


def _run_scenario(
    scenario: str, overrides: list[str], extra: list[str], tls_add: Path, log_path: Path
) -> None:
    with log_path.open("w") as log:
        subprocess.run(
            [
                SCENARIO,
                f"scenario={scenario}",
                *overrides,
                *extra,
                MAP,
                "traffic=sumo",
                "traffic.options.fcd_output=true",
                f"traffic.options.sumo_args=[--additional-files,{tls_add}]",
                "typecheck=off",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )


def _stop(chase: subprocess.Popen) -> None:
    """Let the chase finish (it exits once the ego is gone), else end it."""
    try:
        chase.wait(timeout=60)
    except subprocess.TimeoutExpired:
        chase.terminate()  # chase.py removes its camera on SIGTERM
        try:
            chase.wait(timeout=15)
        except subprocess.TimeoutExpired:
            chase.kill()


def _render(
    name: str, scenario: str, spawn: str, title: str, extra: list[str], work: Path
) -> tuple[str, bool]:
    """Record one clip; returns its summary line and whether every step passed."""
    run_dir = work / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    tls_add = run_dir / "tls_add.xml"
    tls_states = run_dir / "tls_states.xml"
    tls_add.write_text(
        f'<additional>\n    <timedEvent type="SaveTLSStates" dest="{tls_states}"/>\n</additional>\n'
    )
    overrides = _case_overrides(scenario, spawn)
    (run_dir / "args.txt").write_text(
        " ".join([f"scenario={scenario}", *overrides, *extra]) + "\n"
    )
    chase = subprocess.Popen(
        [PYTHON, str(HERE / "chase.py"), str(run_dir / "carla")],
        stdout=(run_dir / "chase.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        _run_scenario(scenario, overrides, extra, tls_add, run_dir / "run.log")
    except subprocess.TimeoutExpired:
        return f"{name}: ERROR, the scenario run timed out", False
    finally:
        _stop(chase)
    text = (run_dir / "run.log").read_text()
    result = (re.findall(r"Result: ([A-Z]+)", text) or ["UNKNOWN"])[-1]
    json_paths = re.findall(r"Result JSON: (\S+)", text)
    nets = re.findall(r"SUMO started on (\S+)", text)
    if not json_paths or not nets:
        return (
            f"{name}: ERROR, the run logged no result or SUMO network (see run.log)",
            False,
        )
    out_dir = Path(json_paths[-1]).parent
    net = nets[-1]
    sumo_out = out_dir / "sumo"
    with (run_dir / "replay.log").open("w") as log:
        replay = subprocess.run(
            [
                PYTHON,
                str(HERE / "sumo_replay.py"),
                net,
                str(sumo_out / "fcd.xml"),
                str(sumo_out / "clock.csv"),
                str(tls_states),
                str(run_dir / "carla"),
                str(run_dir / "sumo"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    compose = subprocess.run(
        [
            PYTHON,
            str(HERE / "compose.py"),
            str(run_dir / "carla"),
            str(run_dir / "sumo"),
            str(work / f"{name}.mp4"),
            title,
            result,
        ],
        stdout=(run_dir / "compose.log").open("w"),
        stderr=subprocess.STDOUT,
        check=False,
    )
    check = subprocess.run(
        [
            PYTHON,
            str(HERE / "check_run.py"),
            net,
            str(run_dir / "carla" / "lights.csv"),
            str(tls_states),
            str(sumo_out / "clock.csv"),
            str(sumo_out / "fcd.xml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (run_dir / "check.txt").write_text(check.stdout + check.stderr)
    frames = len(list((run_dir / "carla").glob("carla_*.png")))
    failed = [
        step
        for step, rc in (
            ("replay", replay.returncode),
            ("compose", compose.returncode),
            ("check", check.returncode),
        )
        if rc != 0
    ]
    status = f"FAILED: {', '.join(failed)}" if failed else "ok"
    summary = f"{name}: {result}, {frames} frames, {status} | " + " | ".join(
        check.stdout.strip().splitlines()
    )
    return summary, result == "PASSED" and not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    failures = []
    for line in args.cases.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, scenario, spawn, title, extra = (line.split("|") + [""])[:5]
        if args.only and name not in args.only:
            continue
        _wait_for_memory(args.min_free_gb)
        summary, ok = _render(name, scenario, spawn, title, extra.split(), args.work)
        if not ok:
            failures.append(name)
        print(summary, flush=True)
        with (args.work / "summary.txt").open("a") as f:
            f.write(summary + "\n")
    if failures:
        raise SystemExit(f"not every clip came out clean: {', '.join(failures)}")


if __name__ == "__main__":
    main()
