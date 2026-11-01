#!/usr/bin/env python
"""Measure the SUMO backend's level of detail on a real CARLA map (#449).

    python benchmark.py <out_dir> [--configs all_mirrored,...] [--seconds 120]
        [--vehicles-per-hour 3000] [-- <more scenario overrides>]

Runs the ego driving on Town10HD_Opt (``drive.py``) with ``traffic=sumo`` and
heavy ambient demand, once per configuration, each in a process of its own:

* ``all_mirrored``: every SUMO vehicle a CARLA actor;
* ``spawn_radius``: only those within ``spawn_radius_m`` (150 m) of the ego;
* ``meso_physics``: mesoscopic far traffic (default radii), CARLA actors
  driven by their physics;
* ``meso_teleport``: the same, CARLA actors placed where SUMO has them.

Every tick it times ``world.tick()`` and the backend's tick, and every
second it reads SUMO's vehicle count, CARLA's vehicle actors and the backend's
fidelity census. Each run writes ``<config>/ticks.csv`` and its run outputs;
``summary.json`` and the table printed at the end compare them, over the
ticks after ``--settle`` seconds.

Needs a CARLA server it may launch (``CARLA_EXECUTABLE``) or reach, and the
``sumo`` extra.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIGS: dict[str, list[str]] = {
    "all_mirrored": [],
    "spawn_radius": ["traffic.options.spawn_radius_m=150.0"],
    "meso_physics": [
        "traffic.options.mesoscopic.enabled=true",
        "traffic.options.vehicle_control=physics",
    ],
    "meso_teleport": [
        "traffic.options.mesoscopic.enabled=true",
        "traffic.options.vehicle_control=teleport",
    ],
    # Town10HD_Opt is small: within the default 200 m most of it is near the
    # ego, so the queue model holds little. A radius that leaves it more.
    # SUMO's pedestrians in CARLA, crossing at CARLA's lights: what
    # pedestrians.json reports (see _PedestrianObserver).
    "pedestrians": [
        "traffic.options.pedestrians.enabled=true",
        "traffic.options.pedestrians.persons_per_hour=2400",
        # The default (walker.pedestrian.0001) is not in CARLA 0.10.
        "traffic.options.pedestrians.blueprint=walker.pedestrian.0015",
        "traffic.options.pedestrian_crossings=true",
        "traffic.options.spawn_radius_m=150.0",
    ],
    # A scenario vehicle SUMO drives, let leave CARLA too.
    "meso_scenario_vehicle": [
        "traffic.options.mesoscopic.enabled=true",
        "traffic.options.mesoscopic.micro_radius_m=80.0",
        "traffic.options.mesoscopic.scenario_vehicles=true",
        "traffic.options.scenario_vehicles=sumo",
        "scenario.npc_ahead_m=20.0",
        "scenario.npc_speed_kmh=50.0",
        # The ego stays put, so npc1 drives out of the radius (and maybe back).
        "scenario.ego_speed_kmh=0.0",
    ],
    "meso_near": [
        "traffic.options.mesoscopic.enabled=true",
        "traffic.options.mesoscopic.micro_radius_m=80.0",
        "traffic.options.vehicle_control=physics",
    ],
}


def _overrides(seconds: float, vehicles_per_hour: int, extra: list[str]) -> list[str]:
    return [
        "scenario=traffic_benchmark/drive",
        "map=town10hd_opt",
        "traffic=sumo",
        "traffic.options.ambient.enabled=true",
        f"traffic.options.ambient.vehicles_per_hour={vehicles_per_hour}",
        f"scenario.seconds={seconds}",
        "typecheck=off",
        *extra,
    ]


class _PedestrianObserver:
    """Reads SUMO through the backend's TraCI every tick: each person stepping
    onto a crossing and whether its signal let it, each crossing green while a
    vehicle link from or to a road it crosses is green too, and how many
    walkers CARLA holds. Written to pedestrians.json as it goes."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.ready = False
        self.crossings: dict[str, tuple[str, list[int], list[int]]] = {}
        self.where: dict[str, str] = {}
        self.entries = 0
        self.red_entries: list[str] = []
        self.conflict_ticks: dict[str, int] = {}
        self.green_ticks: dict[str, int] = {}
        self.ticks = 0
        self.walkers: list[int] = []
        self.persons: list[int] = []

    def _prepare(self, backend) -> None:
        import xml.etree.ElementTree as ET  # noqa: PLC0415

        traci = backend._traci
        crossed: dict[str, set[str]] = {}
        for edge in ET.parse(backend._network.net_file).getroot().iter("edge"):
            if edge.get("function") == "crossing":
                crossed[edge.get("id")] = set((edge.get("crossingEdges") or "").split())
        for tls in traci.trafficlight.getIDList():
            links = traci.trafficlight.getControlledLinks(tls)
            for crossing, roads in crossed.items():
                walk = [
                    i
                    for i, link in enumerate(links)
                    if any(
                        lane.rsplit("_", 1)[0] == crossing for lane in link[0] if lane
                    )
                ]
                if not walk:
                    continue
                drive = [
                    i
                    for i, link in enumerate(links)
                    if any(
                        lane
                        and not lane.startswith(":")
                        and (
                            lane.rsplit("_", 1)[0] in roads
                            or link[0][1].rsplit("_", 1)[0] in roads
                        )
                        for lane in link[0][:1]
                    )
                ]
                self.crossings[crossing] = (tls, walk, drive)
        self.ready = True

    def tick(self, backend, world) -> None:
        if not getattr(backend, "_running", False):
            return
        if not self.ready:
            self._prepare(backend)
        traci = backend._traci
        self.ticks += 1
        states = {}
        for crossing, (tls, walk, drive) in self.crossings.items():
            state = states.get(tls)
            if state is None:
                state = states[tls] = traci.trafficlight.getRedYellowGreenState(tls)
            green = any(state[i] in "Gg" for i in walk)
            if green:
                self.green_ticks[crossing] = self.green_ticks.get(crossing, 0) + 1
                if any(state[i] in "Gg" for i in drive):
                    self.conflict_ticks[crossing] = (
                        self.conflict_ticks.get(crossing, 0) + 1
                    )
        for person in traci.person.getIDList():
            road = traci.person.getRoadID(person)
            before = self.where.get(person)
            self.where[person] = road
            if road in self.crossings and before != road:
                self.entries += 1
                tls, walk, _ = self.crossings[road]
                state = states.get(tls) or traci.trafficlight.getRedYellowGreenState(
                    tls
                )
                if all(state[i] in "rs" for i in walk):
                    self.red_entries.append(f"{person} onto {road}")
        if self.ticks % 20 == 0:
            self.persons.append(len(traci.person.getIDList()))
            self.walkers.append(len(world.get_actors().filter("walker.*")))
            self.write()

    def write(self) -> None:
        (self.out / "pedestrians.json").write_text(
            json.dumps(
                {
                    "signalled_crossings": len(self.crossings),
                    "crossing_entries": self.entries,
                    "entries_on_red": len(self.red_entries),
                    "red_entries": self.red_entries[:20],
                    "crossing_green_ticks": sum(self.green_ticks.values()),
                    "green_with_crossed_road_green_ticks": sum(
                        self.conflict_ticks.values()
                    ),
                    "conflicting_crossings": sorted(self.conflict_ticks),
                    "sumo_persons_mean": statistics.fmean(self.persons)
                    if self.persons
                    else 0,
                    "carla_walkers_mean": statistics.fmean(self.walkers)
                    if self.walkers
                    else 0,
                },
                indent=2,
            )
        )


def _measure(out: Path, overrides: list[str]) -> None:
    """In this process: time the run's ticks, then run it."""
    sys.path.insert(0, str(HERE))
    from drive import DriveConfig, DriveScenario  # noqa: PLC0415

    import typesafe_carla.carla as carla  # noqa: PLC0415

    from autoware_carla_scenario import register_conf_dir, register_scenario  # noqa: PLC0415
    from autoware_carla_scenario.traffic.sumo.backend import (  # noqa: PLC0415
        SumoTrafficBackend,
    )

    register_conf_dir(HERE / "conf")
    register_scenario("traffic_benchmark", DriveScenario, DriveConfig)

    out.mkdir(parents=True, exist_ok=True)
    rows = csv.writer((out / "ticks.csv").open("w", newline=""))
    rows.writerow(
        [
            "elapsed",
            "world_ms",
            "backend_ms",
            "sumo_vehicles",
            "meso_vehicles",
            "carla_vehicles",
        ]
    )
    state = {"world_ms": 0.0, "next_sample": 0.0}
    observer = (
        _PedestrianObserver(out)
        if any(
            o.startswith("traffic.options.pedestrians.enabled=true") for o in overrides
        )
        else None
    )

    world_tick = carla.World.tick

    def timed_world_tick(self, *args, **kwargs):
        start = time.perf_counter()
        result = world_tick(self, *args, **kwargs)
        state["world_ms"] = (time.perf_counter() - start) * 1000.0
        return result

    backend_tick = SumoTrafficBackend.tick

    def timed_backend_tick(self, world, elapsed):
        start = time.perf_counter()
        backend_tick(self, world, elapsed)
        backend_ms = (time.perf_counter() - start) * 1000.0
        if observer is not None:
            observer.tick(self, world)
        sumo = meso = carla_vehicles = ""
        if elapsed >= state["next_sample"]:
            state["next_sample"] = elapsed + 1.0
            census = self.fidelity_census()
            sumo = census.get("microscopic", "")
            meso = census.get("mesoscopic", "")
            carla_vehicles = sum(
                1
                for a in world.get_actors().filter("vehicle.*")
                if a.get_location().z > -100.0
            )
        rows.writerow(
            [
                f"{elapsed:.2f}",
                f"{state['world_ms']:.2f}",
                f"{backend_ms:.2f}",
                sumo,
                meso,
                carla_vehicles,
            ]
        )

    carla.World.tick = timed_world_tick
    SumoTrafficBackend.tick = timed_backend_tick

    from autoware_carla_scenario.examples import run  # noqa: PLC0415

    sys.argv = ["scenario", f"hydra.run.dir={out}", *overrides]
    run.main()


def _summarize(out: Path, settle: float) -> dict[str, float]:
    world, backend, sumo, meso, carla_n = [], [], [], [], []
    with (out / "ticks.csv").open() as f:
        for row in csv.DictReader(f):
            if float(row["elapsed"]) < settle:
                continue
            world.append(float(row["world_ms"]))
            backend.append(float(row["backend_ms"]))
            if row["sumo_vehicles"] != "":
                sumo.append(float(row["sumo_vehicles"]))
                meso.append(float(row["meso_vehicles"] or 0))
                carla_n.append(float(row["carla_vehicles"]))
    total = [w + b for w, b in zip(world, backend)]

    def p95(values: list[float]) -> float:
        return sorted(values)[int(0.95 * (len(values) - 1))] if values else 0.0

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    return {
        "ticks": len(total),
        "tick_ms_mean": mean(total),
        "tick_ms_p95": p95(total),
        "world_ms_mean": mean(world),
        "backend_ms_mean": mean(backend),
        "sumo_vehicles_mean": mean(sumo),
        "meso_vehicles_mean": mean(meso),
        "carla_vehicles_mean": mean(carla_n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--settle", type=float, default=20.0)
    parser.add_argument("--vehicles-per-hour", type=int, default=3000)
    parser.add_argument("--measure", help=argparse.SUPPRESS)
    args, extra = parser.parse_known_args()
    extra = [e for e in extra if e != "--"]
    if args.measure:
        _measure(args.out, json.loads(extra[0]))
        return
    summary: dict[str, dict[str, float]] = {}
    for name in args.configs.split(","):
        overrides = _overrides(
            args.seconds, args.vehicles_per_hour, [*CONFIGS[name], *extra]
        )
        print(f"== {name}: {' '.join(overrides)}", flush=True)
        done = subprocess.run(
            [
                sys.executable,
                __file__,
                str(args.out / name),
                "--measure",
                name,
                "--",
                json.dumps(overrides),
            ],
            check=False,
        )
        if done.returncode != 0:
            print(f"   {name}: the run exited with {done.returncode}", flush=True)
        if (args.out / name / "ticks.csv").exists():
            summary[name] = _summarize(args.out / name, args.settle)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    keys = [
        "ticks",
        "tick_ms_mean",
        "tick_ms_p95",
        "world_ms_mean",
        "backend_ms_mean",
        "sumo_vehicles_mean",
        "meso_vehicles_mean",
        "carla_vehicles_mean",
    ]
    print("config".ljust(16) + "".join(k.rjust(20) for k in keys))
    for name, values in summary.items():
        print(name.ljust(16) + "".join(f"{values[k]:20.1f}" for k in keys))


if __name__ == "__main__":
    main()
