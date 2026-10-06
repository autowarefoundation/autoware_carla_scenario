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


def _render(
    name: str, scenario: str, spawn: str, title: str, extra: list[str], work: Path
) -> str:
    run_dir = work / name
    subprocess.run(["rm", "-rf", str(run_dir)], check=True)
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
    with (run_dir / "run.log").open("w") as log:
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
    try:
        chase.wait(timeout=60)
    except subprocess.TimeoutExpired:
        chase.kill()
    text = (run_dir / "run.log").read_text()
    result = (re.findall(r"Result: ([A-Z]+)", text) or ["UNKNOWN"])[-1]
    out_dir = Path(re.findall(r"Result JSON: (\S+)", text)[-1]).parent
    net = re.findall(r"SUMO started on (\S+)", text)[-1]
    sumo_out = out_dir / "sumo"
    with (run_dir / "replay.log").open("w") as log:
        subprocess.run(
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
    subprocess.run(
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
    return f"{name}: {result}, {frames} frames | " + " | ".join(
        check.stdout.strip().splitlines()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--min-free-gb", type=float, default=6.0)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    for line in args.cases.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, scenario, spawn, title, extra = (line.split("|") + [""])[:5]
        if args.only and name not in args.only:
            continue
        _wait_for_memory(args.min_free_gb)
        summary = _render(name, scenario, spawn, title, extra.split(), args.work)
        print(summary, flush=True)
        with (args.work / "summary.txt").open("a") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
