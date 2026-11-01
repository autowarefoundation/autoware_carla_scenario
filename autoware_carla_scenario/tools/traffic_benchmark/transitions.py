#!/usr/bin/env python
"""How SUMO's vehicles come into CARLA and leave it, from a run's trajectory log (#449).

    python transitions.py <run.trajectory.jsonl> [--after 1.0] [--worst 5]

For each SUMO vehicle (``sumo:*``) that appears after ``--after`` seconds
(coming back from the queue model, or in from SUMO's edge of the radius):

* how far from the ego it appears (a vehicle appearing within sight pops);
* how near the nearest other vehicle is then (one appearing on top of
  another);
* its speed then, and its largest acceleration over the second after (a jolt
  when it drives off).

And for each one that leaves CARLA, how far from the ego it is when it goes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _speed(row: list[float]) -> float:
    return math.hypot(row[7], row[8])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--after", type=float, default=1.0)
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    roles: dict[int, str] = {}
    ticks: list[tuple[float, dict[int, list[float]]]] = []
    gone_at: dict[int, float] = {}
    for line in args.log.read_text().splitlines():
        record = json.loads(line)
        kind = record.get("type")
        if kind == "actor":
            roles[record["id"]] = record.get("role") or ""
        elif kind == "tick":
            ticks.append((record["t"], {row[0]: row for row in record["actors"]}))
        elif kind == "actor_gone":
            gone_at[record["id"]] = ticks[-1][0] if ticks else 0.0
    ego = next(i for i, r in roles.items() if r == "Ego")

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for index, (_, actors) in enumerate(ticks):
        for actor_id in actors:
            first.setdefault(actor_id, index)
            last[actor_id] = index

    appear = []
    for actor_id, index in first.items():
        role = roles.get(actor_id, "")
        t, actors = ticks[index]
        if not role.startswith("sumo:") or t < args.after or ego not in actors:
            continue
        here = actors[actor_id]
        e = actors[ego]
        to_ego = math.hypot(here[1] - e[1], here[2] - e[2])
        others = [
            math.hypot(here[1] - row[1], here[2] - row[2])
            for other, row in actors.items()
            if other != actor_id
        ]
        nearest = min(others) if others else math.inf
        speeds = [
            _speed(ticks[j][1][actor_id])
            for j in range(index, min(index + 21, len(ticks)))
            if actor_id in ticks[j][1]
        ]
        dt = ticks[1][0] - ticks[0][0] if len(ticks) > 1 else 0.05
        accel = max((abs(b - a) / dt for a, b in zip(speeds, speeds[1:])), default=0.0)
        appear.append((role, t, to_ego, nearest, speeds[0], accel))

    leave = []
    for actor_id, index in last.items():
        role = roles.get(actor_id, "")
        if not role.startswith("sumo:") or actor_id not in gone_at:
            continue
        t, actors = ticks[index]
        if ego not in actors:
            continue
        here, e = actors[actor_id], actors[ego]
        leave.append((role, t, math.hypot(here[1] - e[1], here[2] - e[2])))

    def stats(values: list[float]) -> str:
        if not values:
            return "none"
        ordered = sorted(values)
        return (
            f"n={len(values)} min={ordered[0]:.1f} median={statistics.median(values):.1f} "
            f"max={ordered[-1]:.1f}"
        )

    print(f"appeared after {args.after:.0f} s: {len(appear)}")
    print("  distance to the ego (m):     ", stats([a[2] for a in appear]))
    print("  nearest other vehicle (m):   ", stats([a[3] for a in appear]))
    print("  speed on appearing (m/s):    ", stats([a[4] for a in appear]))
    print("  largest accel in 1 s (m/s2): ", stats([a[5] for a in appear]))
    print(f"left CARLA: {len(leave)}")
    print("  distance to the ego (m):     ", stats([g[2] for g in leave]))
    print(f"closest to another on appearing (worst {args.worst}):")
    for role, t, to_ego, nearest, speed, accel in sorted(appear, key=lambda a: a[3])[
        : args.worst
    ]:
        print(
            f"  {t:6.2f}s {role:14s} {to_ego:6.1f} m from the ego, {nearest:4.1f} m from another, {speed:4.1f} m/s, accel {accel:5.1f}"
        )
    print(f"hardest accel after appearing (worst {args.worst}):")
    for role, t, to_ego, nearest, speed, accel in sorted(appear, key=lambda a: -a[5])[
        : args.worst
    ]:
        print(
            f"  {t:6.2f}s {role:14s} {to_ego:6.1f} m from the ego, {nearest:4.1f} m from another, {speed:4.1f} m/s, accel {accel:5.1f}"
        )


if __name__ == "__main__":
    main()
