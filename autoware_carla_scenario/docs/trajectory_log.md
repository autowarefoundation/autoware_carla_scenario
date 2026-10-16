# Trajectory Log

Every run writes `{stem}.trajectory.jsonl` next to its `_result.json`: the
vehicles, pedestrians and traffic lights of the world, one JSON object per line,
once per `world.tick()`. `TrajectoryRecorder`
(`autoware_carla_scenario/trajectory_recorder.py`) writes it, and the carla-hub
run report replays it.

Its format is published as a JSON Schema,
`autoware_carla_scenario/schemas/trajectory.schema.json` in the package.
The schema ships in the wheel, and Python can locate it as
`autoware_carla_scenario.trajectory_recorder.SCHEMA_PATH`. It describes one line
and is the contract for readers in any language. The recorder's tests check
every line it writes against the schema, so the schema and the recorder cannot
drift apart.

## Records

Each line is one of four records, told apart by `type`. A file starts with one
`world` line, then one `traffic_light` line per light. After that come `tick`
lines, and each tick is preceded by an `actor` line for every vehicle or walker
appearing in it for the first time.

```json
{"type":"world","map":"Carla/Maps/Town10HD_Opt","episode":7}
{"type":"traffic_light","id":9,"opendrive_id":"42","location":[10.0,20.0,0.0],"boxes":[[10.0,20.0,5.0,0.2,0.3,0.6,180.0]],"stops":[[5.0,18.0,0.0,-90.0]]}
{"type":"actor","id":1,"type_id":"vehicle.lincoln.mkz","role":"ego","extent":[2.45,1.07,0.77],"center":[0.0,0.0,0.7]}
{"type":"tick","frame":100,"t":0.0,"actors":[[1,0.0,1.0,0.0,0.0,0.0,90.0,0.0,0.0,0.0]],"control":{"1":[0.5,0.0,0.0,1]},"lights":{"9":"Red"}}
```

| Record | Written | Fields |
|---|---|---|
| `world` | once, first | `map` (CARLA map name), `episode` (`world.id`) |
| `traffic_light` | once per light, when recording starts | `id`, `opendrive_id`, `location` `[x,y,z]`, `boxes` `[[x,y,z,ex,ey,ez,yaw]]` (one per light head), `stops` `[[x,y,z,yaw]]` (stop-line waypoints) |
| `actor` | before the first tick a vehicle/walker appears in | `id`, `type_id` (blueprint), `role` (`role_name`), `extent` (bounding box half size), `center` (box centre offset from the actor origin, actor frame) |
| `tick` | after each `world.tick()` | `frame` (CARLA frame), `t` (scenario seconds), `actors` `[[id,x,y,z,roll,pitch,yaw,vx,vy,vz]]`, `control` `{id: [throttle,steer,brake,gear]}` (egos only), `lights` `{id: state}` (changes only) |

Fixed-length arrays are tuples: the schema gives each position its meaning, and
generated types keep them as tuples where the target language has them.

## Frames, units and time

- Positions and rotations are in CARLA's left-handed world frame (x east,
  y south, z up). Positions are in metres, angles in degrees, and velocities
  are world-frame m/s. Every float is rounded to 3 decimals.
- `t` is simulated time since the scenario clock started, the time pass/fail
  conditions see. It is not wall time. Warm-up ticks are not recorded.
- Ticks are 20 Hz (`FIXED_DELTA_SECONDS`).

To move a pose into the Lanelet2 / Autoware `map` frame, use the map's MGRS
offset `(ox, oy)` and elevation offset `z_offset` (`coordinate/transform.py`):

```
x_map   =  x + ox
y_map   = -y + oy
z_map   =  z + z_offset
yaw_map = -radians(yaw)      (pitch negated likewise; roll unchanged)
vy_map  = -vy
```

The file does not carry these offsets. They come from the map.

## What a reader has to handle

- **Light states are deltas.** `lights` holds only the lights that changed
  since the last written tick (all of them in the first tick), and is absent
  when none changed. Carry each light's state forward.
- **Ticks can be missing.** A tick without any vehicle or walker is not
  written, so `frame` and `t` can skip.
- **Actors are not marked as destroyed.** A destroyed actor stops appearing in
  `actors`.
- **Map keys are strings.** `control` and `lights` are keyed by actor id as a
  decimal string (`"1"`), a JSON object key.
- **Integers are plain JSON numbers.** Nothing is quoted: unlike the
  protobuf JSON mapping, which writes 64-bit integers as strings, the file and
  its schema keep them numbers. The schema caps `episode` and `frame` (uint64
  in LibCarla) at 2^53 - 1 and actor ids (uint32) at 2^32 - 1. Any reader,
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
| Python | [datamodel-code-generator](https://github.com/koxudaxi/datamodel-code-generator) | Pydantic models; tuples become length-checked lists |
| C++, Go, Rust, C#, ... | [quicktype](https://github.com/glideapps/quicktype) | Merges the four records into one type with optional fields; tuples become arrays |

```bash
npx json-schema-to-typescript -i trajectory.schema.json -o trajectory.d.ts
uvx --from datamodel-code-generator datamodel-codegen \
  --input trajectory.schema.json --input-file-type jsonschema \
  --output-model-type pydantic_v2.BaseModel --output trajectory_log.py
```

Any JSON Schema draft-07 validator checks a file line by line, for example
`jsonschema` in Python:

```python
import json
import jsonschema
from autoware_carla_scenario.trajectory_recorder import SCHEMA_PATH

validator = jsonschema.Draft7Validator(json.loads(SCHEMA_PATH.read_text()))
with open("CutInScenario.trajectory.jsonl") as f:
    for line in f:
        validator.validate(json.loads(line))
```

The schema uses draft-07 rather than 2020-12 because code generators support it
more widely. quicktype, for one, cannot read 2020-12 tuples (`prefixItems`).

## Changing the format

A change to a record in `trajectory_recorder.py` needs the same change in the
schema. The schema forbids fields it does not list (`additionalProperties:
false`), so the recorder tests fail until the schema has the change.
