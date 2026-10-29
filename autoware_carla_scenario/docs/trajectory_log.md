# Trajectory Log

Every run writes `{stem}.trajectory.jsonl` next to its `_result.json`: the
vehicles, pedestrians and traffic lights of the world, once per `world.tick()`,
and what the run made of them -- the map it ran on, the ego's plan, its
conditions tick by tick and its verdict -- one JSON object per line.
`TrajectoryRecorder` (`autoware_carla_scenario/trajectory_recorder.py`) writes
it, and the carla-hub run report replays it. The file holds everything needed
to show the run again without CARLA, the ego's plan and the path it drove
included: the `autoware_carla_scenario_rerun` package (the `rerun` extra;
`rerun.enabled=true` runs it on each log as its run ends) makes a Rerun
recording of it, which is how a run is seen ([Seeing a run](usage.md#seeing-a-run)).
`scenario-t4scene` writes it as a T4Scene, the scene exchange format of
tier4/e2e-devkit ([T4 Scene](t4_scene.md)).

Its format is published as a JSON Schema of one line, the contract for readers
in any language: `schemas/trajectory.schema.json` in the package and in the
wheel (JSON Schema 2020-12, tuples as `prefixItems`). From Python, find it with
`importlib.resources.files("autoware_carla_scenario") / "schemas" /
"trajectory.schema.json"`, which needs no CARLA client. Code that already
imports the recorder can use `trajectory_recorder.SCHEMA_PATH`.
`trajectory_recorder.schema.json`, next to the recorder, is the same schema in
draft-07 (`trajectory_recorder.DRAFT7_SCHEMA_PATH`), for the generators that
cannot read 2020-12 tuples.

## Versions

The format has a version, `schema_version` in the `world` line: **2**. Version
2 added the records `map`, `warmup`, `actor_gone`, `markers`, `condition`,
`action` and `result`, and `schema_version` itself; a file of version 1 has no
`schema_version` and none of those records, and still follows the schema.
The records marked (2) in the table below that are not in that list
(`rear_axle`, `plan`, `adversity`, `waypoint_arrival`, `collision`, `walker_fallen`,
`spawn`, `despawn`) came later within version 2: a reader of version 2 that does not
know one skips it.
Every change keeps the records there are: a new version adds records (and
optional fields), so a reader that skips the `type`s it does not know reads
any later file.

## Records

Each line is one of nineteen records, told apart by `type`. A file starts with one
`world` line and one `map` line, then one `traffic_light` line per light and one
`warmup` line. After that come the ticks: each `tick` line is preceded by an
`actor` line for every vehicle or walker appearing in it for the first time
(an ego's followed by a `rear_axle` line) and
an `actor_gone` line for every one that has gone since the last, and followed
by a `markers` line when the ego's plan changed on it (and a `plan` line after
it, the plan timed), a `condition` line for
each condition that changed and an `action` line for each action that moved on. A run that reached a verdict ends with `result`.

```json
{"type":"world","schema_version":2,"map":"Carla/Maps/Town10HD_Opt","episode":7}
{"type":"map","name":"Town10HD_Opt","source":"git+https://huggingface.co/datasets/AutowareFoundation/carla-ue5-maps@c834d5c#autoware_maps/Town10HD_Opt","lanelet2":{"path":"/.../lanelet2_map.osm","sha256":"5d88...","bytes":1234567},"opendrive":{"path":"/.../Town10HD_Opt.xodr","sha256":"9a1f...","bytes":2345678},"projector_info":{"projector_type":"MGRS","vertical_datum":"WGS84","mgrs_grid":"54SUE"},"projector_type":null,"mgrs_offset":[81655.0,50137.5],"z_offset":0.0}
{"type":"traffic_light","id":9,"opendrive_id":"42","location":[10.0,20.0,0.0],"boxes":[[10.0,20.0,5.0,0.2,0.3,0.6,180.0]],"stops":[[5.0,18.0,0.0,-90.0]]}
{"type":"warmup","ticks":66,"start_frame":34,"end_frame":100}
{"type":"actor","id":1,"type_id":"vehicle.lincoln.mkz","role":"ego","extent":[2.45,1.07,0.77],"center":[0.0,0.0,0.7]}
{"type":"tick","frame":100,"t":0.0,"actors":[[1,0.0,1.0,0.0,0.0,0.0,90.0,0.0,0.0,0.0]],"control":{"1":[0.5,0.0,0.0,1]},"lights":{"9":"Red"}}
{"type":"markers","frame":100,"t":0.0,"actor":1,"polylines":[{"points":[[0.0,1.0,0.3],[5.0,1.0,0.3]],"color":[0,255,0],"thickness":0.1,"point_size":0.1}]}
{"type":"plan","frame":100,"t":0.0,"actor":1,"step":1,"planned_at_us":3050000,"policy":"resworld","inference_seconds":0.031,"turn_indicator":1,"debug_scalars":{},"points":[[0.1,0.0,1.0,0.0,90.0],[0.2,5.0,1.0,0.0,90.0]]}
{"type":"condition","frame":101,"t":0.05,"role":"pass","index":0,"label":"arrived","satisfied":false}
{"type":"action","frame":101,"t":0.05,"phase":"post_tick","index":0,"label":"cut_in","action_type":"LaneChangeAction","state":"startTransition"}
{"type":"actor_gone","id":3,"frame":180,"t":4.0}
{"type":"result","passed":true,"message":"Ego reached the goal","elapsed_seconds":12.3,"conditions":[{"label":"pass[0](arrived)","role":"pass","condition_type":"ReachPositionCondition","satisfied":true,"message":"..."}]}
```

| Record | Written | Fields |
|---|---|---|
| `world` | once, first | `map` (CARLA map name), `episode` (`world.id`) |
| `traffic_light` | once per light, when recording starts | `id`, `opendrive_id`, `location` `[x,y,z]`, `boxes` `[[x,y,z,ex,ey,ez,yaw]]` (one per light head), `stops` `[[x,y,z,yaw]]` (stop-line waypoints) |
| `actor` | before the first tick a vehicle/walker appears in | `id`, `type_id` (blueprint), `role` (`role_name`), `extent` (bounding box half size), `center` (box centre offset from the actor origin, actor frame) |
| `tick` | after each `world.tick()` | `frame` (CARLA frame), `t` (scenario seconds), `actors` `[[id,x,y,z,roll,pitch,yaw,vx,vy,vz]]`, `control` `{id: [throttle,steer,brake,gear]}` (egos only), `lights` `{id: state}` (changes only) |
| `map` (2) | once, after `world` | `name` (short map name), `source` (`map.source`, else `map.directory`, else null), `lanelet2` / `opendrive` (`{path, sha256, bytes}` of the file read, or null), `projector_info` (`map_projector_info.yaml` as read, or null), `projector_type` (the config's override, or null), `mgrs_offset` `[ox,oy]` and `z_offset` (the loaded Lanelet2 map's, or null) |
| `warmup` (2) | once, before the first tick | `ticks` (frames of the CARLA recording before the scenario clock started: the warm-up), `start_frame` (the frame the recorder started at, null without a recording), `end_frame` (the frame of the first tick, t = 0) |
| `rear_axle` (2) | right after an ego's `actor` line | `id`, `offset` (where its rear axle -- Autoware's `base_link`, the driver policy's rig origin -- is from the actor origin along its x axis, metres, negative behind: the rig origin a driver policy's entity uses, else from its wheels, else half its box behind) |
| `actor_gone` (2) | before the first tick a vehicle/walker is missing from | `id`, and the `frame` and `t` of that tick |
| `markers` (2) | after a tick on which the ego's debug markers changed | `frame`, `t`, `actor` (the ego, or null), `polylines` `[{points [[x,y,z]], color [r,g,b], thickness, point_size}]`; they hold until the next `markers` line, and none (`[]`) clears them |
| `plan` (2) | right after the `markers` line of a driver's plan (a driver policy's, `CarlaDriverEntity`, or an Autoware ego's trajectory command, `Ros2EgoEntity`; `ego_plan.PlanMarkers`), once per plan, even an empty one | `frame`, `t`, `actor` (the ego, or null), `step` (the plan's number in the driver's session, from 1), `planned_at_us` (when it was made, on the driver's clock -- a policy's microseconds since its session began, an Autoware command's stamp -- not `t`), `policy`, `inference_seconds` (0 when not said, null for NaN or infinity), `turn_indicator` (0 no command, 1 off, 2 left, 3 right), `debug_scalars` `{name: value}` (null for NaN or infinity), `points` `[[dt,x,y,z,yaw]]` (the rig's planned poses, `dt` seconds after the plan was made; world frame, not lifted as the markers are; a pose that is not finite left out) |
| `condition` (2) | when a condition is first checked, and when it changes | `frame`, `t`, `role` (`pass`/`fail`), `index` (in its list), `label`, `satisfied` |
| `action` (2) | on the first tick of the loop, and when the action's state changes | `frame`, `t`, `phase` (`pre_tick`/`post_tick`), `index` (in its list), `label`, `action_type` (its class), `state` (`standbyState`, `startTransition`, `runningState`, `endTransition`, `completeState`: OpenSCENARIO's, `ActionState`) |
| `adversity` (2) | on the tick NADE samples an adversity (`AdversarialTrafficAction`) | `frame`, `t`, `label` (the action's), `agent`, `kind` (`AdversityKind`), `location` (`AdversityLocation`), `probability` (`p`, cut for an unavoidable one), `is_probability` (`q`), `weight` (the run's, once sampled), `avoidable`, `victims`, `response` (`avoid`, `accept` or empty) |
| `waypoint_arrival` (2) | when a routed entity has reached a waypoint with a time or a departure condition (`RoutingAction`, any mode), and its target is known | `frame`, `t`, `label` (the action's), `entity`, `vertex` (0-based), `kind` (`time`, or `condition`: the target is when it first held), `mode` (`driver`, `follow`, `position`), `target_t`, `actual_t`, `error_s` (`actual_t - target_t`) |
| `collision` (2) | when a vehicle other than the ego begins a contact with a vehicle or a walker, and when it ends (no event from the pair for a second) (`npc_collisions`) | `frame`, `t`, `actors` `[role, role]` (type ids where an actor has no role), `ids` `[id, id]`, `impulse` (N*s; at the end, the strongest), `ego` (one side is the ego), `phase` (`start`/`end`), `duration` (s, at the end; null at the start) |
| `walker_fallen` (2) | once per walker seen fallen over (tipped beyond 50 deg, or dropped more than half its height between two looks a quarter second apart while barely moving sideways) | `frame`, `t`, `id`, `role`, `drop` (m), `tilt` (deg) |
| `spawn` (2) | on the tick a `SpawnAction` brought an entity into the scene (one spawned hidden, as a rule; not for a prop) | `frame`, `t`, `label` (the action's), `entity` (role), `id` (its actor's, or null) |
| `despawn` (2) | on the tick a `DespawnAction` removed an entity, or the run a walker that lay fallen 5 s | `frame`, `t`, `label` (the action's), `entity` (role), `id` (its actor's, or null), `reason` (`action`/`fallen`) |
| `result` (2) | last, when the run reached a verdict | `passed`, `message`, `elapsed_seconds`, `conditions` `[{label, role, condition_type, satisfied, message}]` (_result.json's, without the details) |

Fixed-length arrays are tuples: the schema gives each position its meaning, and
generated types keep them as tuples where the target language has them.

## Frames, units and time

- Positions and rotations are in CARLA's left-handed world frame (x east,
  y south, z up; see [Coordinate frames](driver_interface.md#coordinate-frames)). Positions are in metres, angles in degrees, and velocities
  are world-frame m/s. Every float is rounded to 3 decimals.
- `t` is simulated time since the scenario clock started, the time pass/fail
  conditions see. It is not wall time. Warm-up ticks are not recorded: the
  `warmup` line says how many there were, so a frame of the CARLA recording,
  which starts with them, is `frame - warmup.start_frame`.
- Ticks are 20 Hz (`FIXED_DELTA_SECONDS`).
- A `plan` line has two clocks: `t`, the tick it is written on, and
  `planned_at_us`, the driver's own (a policy's from its session's start, an
  Autoware ego's command stamps), which its points' `dt` count from. A plan is written on the tick it is made on, except
  one made before the first recorded tick (while the run waits for the ego to
  be ready), written on that tick: compare plans by `planned_at_us`, not by `t`.

To move a pose into the Lanelet2 / Autoware `map` frame from Python, build a
`CarlaWorldPose` and call `coordinate.transform.to_map_frame` (see
[Coordinate transforms](api.md#coordinate-transforms-autoware_carla_scenariocoordinate)).
In other languages, apply the same conversion with the map's MGRS offset
`(ox, oy)` and elevation offset `z_offset` (`MapManager.mgrs_offset` and
`MapManager.z_offset`):

```
x_map   =  x + ox
y_map   = -y + oy
z_map   =  z + z_offset
yaw_map = -radians(yaw)      (pitch negated likewise; roll unchanged)
vy_map  = -vy
```

A file of version 2 carries these offsets in its `map` line (`mgrs_offset`,
`z_offset`), with the Lanelet2 and OpenDRIVE files the run read and their
SHA-256, so a reader can load the very map the run was on.

## What a reader has to handle

- **Light states are deltas.** `lights` holds only the lights that changed
  since the last written tick (all of them in the first tick), and is absent
  when none changed. Carry each light's state forward.
- **Ticks can be missing.** A tick without any vehicle or walker is not
  written, so `frame` and `t` can skip.
- **Entities brought in late are marked.** An entity spawned hidden is in
  `actors` from the start, parked out of sight under the map;
  a `spawn` line says when a `SpawnAction` put it on its place, as a
  `despawn` line says when one was taken out.
- **Actors that go are marked.** A vehicle or walker that is destroyed or
  despawned stops appearing in `actors`, and from version 2 an `actor_gone`
  line says so before the first tick it is missing from.
- **Markers, conditions and actions are deltas too.** A `markers` line holds
  until the next one, and a `plan` line until the next `plan` line (markers
  that carry no plan clear the drawing, not the plan); a `condition` line until the next of the same `role` and
  `index`; an `action` line until the next of the same `phase` and `index`
  (`startTransition` and `endTransition` hold for one tick).
  The run stops checking conditions at the first that decides it, so the last
  tick may have no line for the ones after it.
- **Episode ids are 64-bit.** `episode` is a random uint64, so it is often past
  2^53: read it as an unsigned 64-bit integer (it only tells episodes apart).
- **Map keys are strings.** `control` and `lights` are keyed by actor id as a
  decimal string (`"1"`), a JSON object key.
- **Integers are plain JSON numbers.** Nothing is quoted: unlike the
  protobuf JSON mapping, which writes 64-bit integers as strings, the file and
  its schema keep them numbers. The schema caps `frame` (uint64 in LibCarla)
  at 2^53 - 1 and actor ids (uint32) at 2^32 - 1; `episode`, a random uint64,
  is capped at 2^64 - 1 (see above). Any reader,
  including JavaScript's double-only `JSON.parse`, holds them exactly; a value
  past the cap fails validation rather than silently losing precision.
- **Only egos have controls.** `control` carries actors whose `role` is
  `ego` or `hero` (any case). `steer` is normalised to [-1, 1]; it is not a
  wheel angle.

## Reading it from another language

Generate types from the schema with the generator for your language:

| Language | Generator | Notes |
|---|---|---|
| TypeScript | [json-schema-to-typescript](https://github.com/bcherny/json-schema-to-typescript) | A discriminated union on `type`, with tuple types |
| Python | [datamodel-code-generator](https://github.com/datamodel-code-generator/datamodel-code-generator) | Pydantic models; tuples become length-checked lists |
| Rust | [typify](https://github.com/oxidecomputer/typify) | Reads 2020-12; an enum on `type` (the schema's `discriminator`), tuples as tuples |
| C++, Go, Rust, C#, ... | [quicktype](https://github.com/glideapps/quicktype) | Reads the draft-07 copy; merges the records into one type with optional fields; tuples become arrays |

```bash
npx json-schema-to-typescript -i schemas/trajectory.schema.json -o trajectory.d.ts
uvx --from datamodel-code-generator datamodel-codegen \
  --input schemas/trajectory.schema.json --input-file-type jsonschema \
  --output-model-type pydantic_v2.BaseModel --output trajectory_log.py
```

Any JSON Schema 2020-12 validator checks a file line by line, for example
`jsonschema` in Python:

```python
import json
import jsonschema
from importlib.resources import files

schema = files("autoware_carla_scenario") / "schemas" / "trajectory.schema.json"
validator = jsonschema.Draft202012Validator(json.loads(schema.read_text()))
with open("CutInScenario.trajectory.jsonl") as f:
    for line in f:
        validator.validate(json.loads(line))
```

The top level is a `oneOf` of the records, each fixing `type` with a `const`,
and names the field in an OpenAPI-style `discriminator` (`propertyName: type`,
with the mapping from each `type` to its record), which validators ignore and
generators that know it use for a tagged union. `x-schema-version` is the
format version the schema describes. The draft-07 copy says the same with
`definitions` and `items` arrays.

## Changing the format

A change to a record in `trajectory_recorder.py` needs the same change in
`schemas/trajectory.schema.json`, and the draft-07 copy written again from it:

```bash
uv run python autoware_carla_scenario/scripts/trajectory_schema_draft7.py
```

The recorder's tests check every line they record against both schemas, which
forbid fields they do not list (`additionalProperties: false`), and that the
copy is the one the script writes, so they fail until both have the change. A
change that a reader of the last version could not skip over is a new
`schema_version` (`SCHEMA_VERSION`, and `x-schema-version` in the schema).
