"""Check a recorded run: SUMO's signals against CARLA's lights, and walking speed.

    python check_run.py <net.xml> <lights.csv> <tls_states.xml> <clock.csv> <fcd.xml>

Each CARLA light state ``chase.py`` logged is compared, through the roadgen
traces the SUMO backend matches lights with, with SUMO's state of every signal
link that light switches at that moment.  Each person's speed is taken from the
FCD over the steps it moves.  Prints a summary; exits non-zero on a mismatch.
"""

from __future__ import annotations

import bisect
import csv
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from autoware_carla_scenario.traffic.sumo.network import SignalTable, SumoNetwork
from sumo_replay import signal_states

_EXPECTED = {"Red": "r", "Yellow": "yY", "Green": "Gg"}


def main() -> int:
    net, lights, tls_file, clock, fcd = (Path(a) for a in sys.argv[1:6])
    table = SignalTable.load(SumoNetwork(net, net.with_suffix(".safe")))
    if table is None:
        print("signals: no roadgen traces beside the network")
        return 1
    rows = [
        (float(r["carla_elapsed_seconds"]), float(r["sumo_time"]))
        for r in csv.DictReader(clock.open())
    ]
    carla_times = [r[0] for r in rows]
    states = signal_states(tls_file)
    agree = 0
    disagree: Counter[tuple[str, str, str]] = Counter()
    for t, signal, state in csv.reader(lights.open()):
        expected = _EXPECTED.get(state)
        if expected is None:
            continue
        i = bisect.bisect_left(carla_times, float(t))
        if i >= len(rows):
            continue
        sumo_t = rows[i][1]
        for tls, index in table.links(signal):
            seq = states.get(tls, [])
            k = bisect.bisect_right([s for s, _ in seq], sumo_t) - 1
            if k < 0:
                continue
            got = seq[k][1][index]
            if got in expected:
                agree += 1
            else:
                disagree[(signal, state, got)] += 1
    total = agree + sum(disagree.values())
    print(f"signals: {agree}/{total} link samples agree")
    if total == 0:
        # Nothing compared is not a pass: no lights logged, none matched to a
        # link, or no SUMO state recorded at those times.
        print("signals: no link sample to compare")
    for (signal, state, got), n in disagree.most_common(5):
        print(f"  light {signal} {state} but SUMO {got!r}: {n}")

    tracks: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for _event, element in ET.iterparse(fcd, events=("end",)):
        if element.tag == "timestep":
            t = float(element.get("time"))
            for person in element.findall("person"):
                tracks[person.get("id")].append(
                    (t, float(person.get("x")), float(person.get("y")))
                )
            element.clear()
    for pid, track in tracks.items():
        speeds = [
            math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
            for (t0, x0, y0), (t1, x1, y1) in zip(track, track[1:])
            if t1 > t0
        ]
        moving = sorted(s for s in speeds if s > 0.3)
        if moving:
            print(
                f"person {pid}: median {moving[len(moving) // 2]:.2f} m/s while moving, max {moving[-1]:.2f} m/s"
            )
        else:
            print(f"person {pid}: did not move")
    return 0 if total and not disagree else 1


if __name__ == "__main__":
    sys.exit(main())
