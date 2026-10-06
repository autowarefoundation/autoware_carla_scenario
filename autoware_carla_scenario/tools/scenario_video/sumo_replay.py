"""Replay a run's SUMO side in sumo-gui, one screenshot per CARLA frame.

    python sumo_replay.py <net.xml> <fcd.xml> <clock.csv> <tls_states.xml> \
        <carla_frames_dir> <out_dir>

The run's FCD (``traffic.options.fcd_output=true``) holds every vehicle and
person, but not the signals; ``tls_states.xml`` is SUMO's ``SaveTLSStates``
output of the same run, and each step the replay sets every signal to the state
SUMO had then -- which, with ``traffic_light_authority: carla``, is CARLA's.
Without it, sumo-gui would run the network's own fixed-time programs.

Each screenshot is named after the CARLA elapsed time of the frame it matches
(through ``clock.csv``).  The run's own vehicles are red, background traffic
from a traffic source light blue, SUMO's ambient traffic keeps its colours,
persons are yellow, and the view follows the ego.
"""

from __future__ import annotations

import bisect
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import sumo
import sumolib
import traci


def _signal_states(path: Path) -> dict[str, list[tuple[float, str]]]:
    """Every signal's states over the run, by time."""
    states: dict[str, list[tuple[float, str]]] = defaultdict(list)
    pattern = re.compile(r'<tlsState time="([\d.]+)" id="([^"]+)"[^>]*state="([^"]+)"')
    for match in pattern.finditer(path.read_text()):
        states[match.group(2)].append((float(match.group(1)), match.group(3)))
    return states


def main() -> None:
    net, fcd, clock, tls_file, frames, out_dir = sys.argv[1:7]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        (float(r["carla_elapsed_seconds"]), float(r["sumo_time"]))
        for r in csv.DictReader(open(clock))
    ]
    carla_times = [r[0] for r in rows]
    wanted: dict[float, float] = {}
    for frame in sorted(Path(frames).glob("carla_*.png")):
        t = float(frame.stem.split("_")[1])
        j = bisect.bisect_left(carla_times, t)
        i = min(
            (k for k in (j - 1, j) if 0 <= k < len(carla_times)),
            key=lambda k: abs(carla_times[k] - t),
        )
        wanted[round(rows[i][1], 2)] = t
    first = min(wanted) - 1.0
    signals = _signal_states(Path(tls_file))
    signal_times = {tls: [t for t, _ in seq] for tls, seq in signals.items()}

    binary = str(Path(sumo.__file__).parent / "bin" / "sumo-gui")
    traci.start(
        [
            binary,
            "-n",
            net,
            "--start",
            "--quit-on-end",
            "--step-length",
            "0.05",
            "--begin",
            f"{first:.2f}",
            "--window-size",
            "1100,640",
            "--delay",
            "0",
            "--no-warnings",
            "--collision.action",
            "none",
        ]
    )
    network = sumolib.net.readNet(net)
    edges = [e.getID() for e in network.getEdges() if e.getFunction() == ""]
    footway = next(
        (
            e.getID()
            for e in network.getEdges()
            if e.getFunction() == "" and e.allows("pedestrian")
        ),
        edges[0],
    )
    traci.route.add("r0", [edges[0]])
    vehicles: set[str] = set()
    persons: set[str] = set()
    ego = None
    for _event, element in ET.iterparse(fcd, events=("end",)):
        if element.tag != "timestep":
            continue
        t = float(element.get("time"))
        if t < first:
            element.clear()
            continue
        here_v = {v.get("id"): v for v in element.findall("vehicle")}
        here_p = {p.get("id"): p for p in element.findall("person")}
        for vid in vehicles - set(here_v):
            traci.vehicle.remove(vid)
            vehicles.discard(vid)
        for pid in persons - set(here_p):
            traci.person.remove(pid)
            persons.discard(pid)
        for vid in here_v:
            if vid in vehicles:
                continue
            traci.vehicle.add(vid, "r0", departPos="0")
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setLaneChangeMode(vid, 0)
            if vid.startswith("bg"):
                traci.vehicle.setColor(vid, (90, 170, 255, 255))
            elif not vid.startswith("sumo"):
                traci.vehicle.setColor(vid, (230, 40, 40, 255))
                if vid.lower().startswith("ego"):
                    ego = vid
            vehicles.add(vid)
        for pid, person in here_p.items():
            if pid in persons:
                continue
            edge = person.get("edge") or footway
            if edge.startswith(":"):
                edge = footway
            traci.person.add(pid, edge, 0.0)
            traci.person.appendWaitingStage(pid, 1e7)
            traci.person.setColor(pid, (255, 200, 0, 255))
            traci.person.setWidth(pid, 0.8)
            traci.person.setLength(pid, 0.8)
            persons.add(pid)
        for tls, seq in signals.items():
            k = bisect.bisect_right(signal_times[tls], t) - 1
            if k >= 0:
                traci.trafficlight.setRedYellowGreenState(tls, seq[k][1])
        traci.simulationStep()
        for vid, v in here_v.items():
            try:
                traci.vehicle.moveToXY(
                    vid,
                    "",
                    -1,
                    float(v.get("x")),
                    float(v.get("y")),
                    float(v.get("angle")),
                    2,
                )
            except traci.TraCIException:
                pass
        for pid, p in here_p.items():
            try:
                traci.person.moveToXY(
                    pid,
                    "",
                    float(p.get("x")),
                    float(p.get("y")),
                    float(p.get("angle")),
                    6,
                )
            except traci.TraCIException as exc:
                print("person", pid, exc)
        key = round(t, 2)
        if key in wanted:
            if ego:
                traci.gui.setSchema("View #0", "real world")
                traci.gui.trackVehicle("View #0", ego)
                traci.gui.setZoom("View #0", 450)
            traci.gui.screenshot("View #0", str(out / f"sumo_{wanted[key]:010.3f}.png"))
        element.clear()
    traci.simulationStep()
    traci.close()
    print("screenshots requested", len(wanted))


if __name__ == "__main__":
    main()
