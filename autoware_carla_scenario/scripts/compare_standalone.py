"""Run scenario configs with the Python runner and as standalone binaries, and compare.

    uv run python autoware_carla_scenario/scripts/compare_standalone.py \\
        lane_change/left intersection_passing/straight ...

Needs a reachable CARLA server (``--host``/``--port``; started from
``$CARLA_EXECUTABLE`` when none is running and left running for both sides),
and the bundles ``scenario-build --out <dist>`` wrote. For each config it runs
``uv run scenario`` and ``<dist>/<config>/bin/<config>`` one after the other,
reads both result JSONs, and prints a table: the verdict, the elapsed time and
the message of each. The exit status is 0 when every verdict agreed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_RESULT_LINE = re.compile(r"^Result JSON: (.+)$", re.MULTILINE)


def _run(cmd: list[str], log: Path, timeout: float) -> tuple[int, str, float]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        out, rc = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = f"{exc.stdout or ''}{exc.stderr or ''}\nTIMED OUT after {timeout}s"
        rc = -1
    log.write_text(out)
    return rc, out, time.monotonic() - started


def _result(out: str, cwd: Path) -> dict | None:
    found = _RESULT_LINE.findall(out)
    if not found:
        return None
    path = Path(found[-1].strip())
    path = path if path.is_absolute() else cwd / path
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("compare_outputs"))
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="per run, wall seconds"
    )
    args = parser.parse_args()

    from autoware_carla_scenario.server import CarlaServerManager

    server = CarlaServerManager(host=args.host, port=args.port)
    server.start()  # reuses a running server; else launches $CARLA_EXECUTABLE
    rows = []
    try:
        for config in args.configs:
            stem = config.replace("/", "_")
            logs = args.out / stem
            logs.mkdir(parents=True, exist_ok=True)
            py_rc, py_out, py_wall = _run(
                [
                    "uv",
                    "run",
                    "scenario",
                    f"scenario={config}",
                    f"server.host={args.host}",
                    f"server.port={args.port}",
                ],
                logs / "python.log",
                args.timeout,
            )
            exe = args.dist / stem / "bin" / stem
            bin_rc, bin_out, bin_wall = _run(
                [
                    str(exe),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--output-dir",
                    str(logs / "binary"),
                ],
                logs / "binary.log",
                args.timeout,
            )
            rows.append(
                (
                    config,
                    _result(py_out, Path.cwd()),
                    py_rc,
                    py_wall,
                    _result(bin_out, Path.cwd()),
                    bin_rc,
                    bin_wall,
                )
            )
            print(f"done: {config} (python rc={py_rc}, binary rc={bin_rc})", flush=True)
    finally:
        server.stop()

    def verdict(r: dict | None, rc: int) -> str:
        return "ERROR" if r is None else ("PASS" if r["passed"] else "FAIL")

    disagree = 0
    print("\n| config | python | binary | sim time py / bin | wall py / bin | agree |")
    print("|---|---|---|---|---|---|")
    for config, pr, prc, pw, br, brc, bw in rows:
        pv, bv = verdict(pr, prc), verdict(br, brc)
        ok = pv == bv and pv != "ERROR"
        disagree += not ok
        sim = (
            f"{pr['elapsed_seconds']:.2f} / {br['elapsed_seconds']:.2f}"
            if pr and br
            else "-"
        )
        print(
            f"| {config} | {pv} | {bv} | {sim} | {pw:.0f}s / {bw:.0f}s | {'yes' if ok else 'NO'} |"
        )
    print("\nmessages:")
    for config, pr, _, _, br, _, _ in rows:
        print(
            f"- {config}\n    python: {pr and pr['message']}\n    binary: {br and br['message']}"
        )
    print(f"\nlogs: {args.out}/<config>/{{python,binary}}.log")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
