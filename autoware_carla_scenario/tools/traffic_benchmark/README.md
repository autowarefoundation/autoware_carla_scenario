# Traffic benchmark

These tools measure the SUMO backend's level of detail on a real CARLA map
(#449). The results are in
[Traffic Backends](../../docs/traffic_backends.md#measured-on-town10hd_opt).

```bash
python benchmark.py <out_dir> [--configs all_mirrored,spawn_radius,...] \
    [--seconds 120] [--settle 20] [--vehicles-per-hour 4000] [-- <overrides>]
python transitions.py <out_dir>/<config>/.../DriveScenario.trajectory.jsonl
```

`benchmark.py` runs the ego driving on Town10HD_Opt (`drive.py`) with
`traffic=sumo` and heavy ambient demand. Each configuration runs in a process
of its own:

| Config | What it sets |
| --- | --- |
| `all_mirrored` | every SUMO vehicle a CARLA actor |
| `spawn_radius` | `spawn_radius_m=150` |
| `meso_physics`, `meso_teleport` | `mesoscopic` with the default radii, under each `vehicle_control` |
| `meso_near` | `mesoscopic` with an 80 m micro radius |
| `meso_scenario_vehicle` | the same, with a SUMO-driven scenario vehicle (npc1) let leave CARLA and the ego held still |
| `pedestrians` | SUMO's pedestrians and crossings |

**What it records:**

- **Every tick**, `<config>/ticks.csv` gets the wall time of `world.tick()` and
  of the backend's tick.
- **Every second**, it also gets SUMO's vehicles, the queue model's and
  CARLA's.
- **At the end**, `summary.json` compares the configurations.
- **With pedestrians**, `pedestrians.json` counts:
  - the persons who stepped onto a signalled crossing, and on red;
  - the ticks a crossing was green while a road it crosses was green too;
  - SUMO's persons against CARLA's walkers.

`transitions.py` reads a run's trajectory log and reports how SUMO's
vehicles come into CARLA and leave it:

- how far from the ego they appear and leave;
- how near another vehicle they appear;
- how hard they accelerate in the first second.

The run needs a CARLA server it may launch (`CARLA_EXECUTABLE`) and the `sumo`
extra.
