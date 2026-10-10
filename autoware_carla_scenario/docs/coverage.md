# Coverage

A passing scenario says the ego handled the situations it was put in. Coverage
says which situations those were, and which ones no run has reached yet.

The framework measures two kinds, and reports them apart:

| Kind | Question | Declared by |
|---|---|---|
| **ODD coverage** | Which operating conditions (road, weather, traffic) did the runs exercise? | The framework, for every run |
| **Scenario coverage** | Which values of a scenario's own parameters (gap, speed at cut-in, ...) did the runs hit? | Each scenario, with `register_cover()` |

Both follow `cover()` in
[ASAM OpenSCENARIO DSL](https://publications.pages.asam.net/standards/ASAM_OpenSCENARIO/ASAM_OpenSCENARIO_DSL/latest/language-reference/coverage_main.html)
and keep its vocabulary.

## How it works

1. A **cover item** names a value and divides its possible values into
   **buckets**.
2. On the item's **sampling event**, the runner reads the value and counts a
   **hit** in the bucket it falls in.
3. Each run writes its hits to `{ScenarioName}_coverage.json` next to
   `{ScenarioName}_result.json`. A batch that runs one scenario class more
   than once writes `{ScenarioName}-1_coverage.json` and so on.
4. `scenario-coverage` merges any number of those files. It grades each item
   and lists its **holes**: the buckets no run has hit.

A bucket is covered once its hits reach the item's `target` (one by default).
An item's grade is its covered buckets over all its buckets. A group's grade
(ODD or scenario) is the mean of its items' grades. The DSL standardises what
is collected, not how it is graded; this is the convention coverage tools
share.

## Declaring scenario coverage

Declare cover items in `setup()`:

```python
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    EGO_ROLE_NAME,
    BaseScenario,
    SamplingEvent,
    TurnDirection,
    find_actor_by_role_name,
)


def ego_speed_kmh(world: carla.World) -> float | None:
    ego = find_actor_by_role_name(world, EGO_ROLE_NAME)
    if ego is None:
        return None  # no sample
    v = ego.get_velocity()
    return (v.x**2 + v.y**2 + v.z**2) ** 0.5 * 3.6


class CutInScenario(BaseScenario):
    def setup(self) -> None:
        ...
        # Equal buckets: 0-10, 10-20, ..., 50-60 km/h, sampled when the
        # cut-in starts (the condition becomes satisfied).
        self.register_cover(
            "speed_at_cut_in",
            ego_speed_kmh,
            unit="km/h",
            range=(0.0, 60.0),
            every=10.0,
            event=cut_in_started,  # any BaseCondition
        )
        # Explicit bucket edges: 4 edges make 3 buckets.
        self.register_cover(
            "gap_at_cut_in",
            gap_m,
            unit="m",
            buckets=[0.0, 5.0, 15.0, 40.0],
            event=cut_in_started,
        )
        # One bucket per value (an enum class, [False, True], strings).
        self.register_cover(
            "turn",
            lambda world: self._config.turn,
            values=[TurnDirection.LEFT, TurnDirection.RIGHT],
        )
        # Every combination of speed and gap buckets.
        self.register_cross("speed_x_gap", ["speed_at_cut_in", "gap_at_cut_in"])
```

### `register_cover()` arguments

| Argument | DSL | Meaning |
|---|---|---|
| `name` | name | Unique within the scenario. |
| `expression` | `expression:` | `Callable[[carla.World], value]`. Returning `None` takes no sample. |
| `unit` | `unit:` | The unit the expression returns, for the report. Nothing is converted. |
| `range`, `every` | `range:`, `every:` | Equal buckets of width `every` over `(low, high)`. |
| `buckets` | `buckets:` | Explicit edges; `N` edges make `N - 1` buckets. |
| `values` | enum / bool item | One bucket per value. |
| `ignore` | `ignore:` | Called with each value; a sample it returns true for is left out. |
| `event` | `event:` | When to sample (below). Default `SamplingEvent.END`, as in the DSL. |
| `text` | `text:` | A description for the report. |
| `target` | `target:` | What a bucket needs to be covered, in `cover_by` (hits by default). |
| `cover_by` | - | What `target` counts: `"hits"`, or `"seconds"`, `"meters"`, `"entries"` (below). |
| `min_stay` | - | Stays in a bucket shorter than this many seconds do not count (below). |

Give exactly one of `range` (with `every`), `buckets` and `values`. Each
numeric bucket holds its lower edge, and the last bucket its upper edge too.
A value outside every bucket is not lost: the report counts it as outside the
buckets, which shows a run reached somewhere the item does not describe.

### Sampling events

| `event` | Sampled |
|---|---|
| `SamplingEvent.END` (default) | Once, when the tick loop ends (pass, fail or done). |
| `SamplingEvent.START` | Once, when the scenario clock starts. |
| `SamplingEvent.TICK` | After every tick. Hits are then ticks, not runs. |
| a `BaseCondition` | Each time the condition becomes satisfied. |

Choose the event deliberately. The default samples the value at the end of
the run. That is right for a parameter the scenario was configured with, and
wrong for something that changes during the run, such as speed. To sample at
a moment that matters (the start of a cut-in), pass a condition that marks
it. A condition used as an event is checked once per tick by the coverage
collector, so give it its own instance rather than one that is also a pass or
fail condition.

`register_cross()` crosses items sampled on the **same** event: a cell is hit
when all the items hit those buckets on the same sample. The number of cells
is the product of the items' bucket counts, so cross only items that interact.

## ODD coverage

Every run is measured against an **ODD**, picked with the `odd` config key
(`odd=default` unless told otherwise). Every tick, the ODD's attributes are
sampled. Their buckets are the ODD coverage, and the ODD's modules decide
whether the tick was inside the ODD. [ODD](odd.md) explains how to write one
in Python or OpenODD YAML.

The built-in `default` ODD has no modules, so every condition is inside it.
Its attributes are a first subset of the
[ISO 34503](https://www.iso.org/standard/78952.html) taxonomy, from each of its
three top-level categories:

| Item | Buckets | Read from |
|---|---|---|
| `odd.scenery.junction` | false, true | The ego's CARLA waypoint |
| `odd.scenery.location` ² | urban, nonurban, private | Lanelet2 `location` tag of the ego's lanelet |
| `odd.scenery.road_type` ² | road, highway, road_shoulder, play_street, parking | Lanelet2 `subtype` tag of the ego's lanelet |
| `odd.scenery.speed_limit` ² | 0-30-40-50-60-80-100-130 km/h | Lanelet2 `speed_limit` tag, else `vehicle.get_speed_limit()` |
| `odd.scenery.lane_count` ² | 1, 2, 3, 4 | Driving lanes in the ego's direction, outside junctions |
| `odd.environment.illumination` | day, low_sun, twilight, night | Sun altitude: day from 15°, low sun from 0°, civil twilight to -6° |
| `odd.environment.rain` | none, light, moderate, heavy | CARLA precipitation (0-100): 1, 30, 70 |
| `odd.environment.fog` | none, light, moderate, heavy | CARLA fog density (0-100): 1, 30, 70 |
| `odd.dynamic.ego_speed` ³ | 0-5, then 10 km/h wide centred on 10, 20, ... 120 (5-15, 15-25, ...) | Ego velocity |
| `odd.dynamic.traffic_density` | none, low (1-2), medium (3-5), high (6+) | Other vehicles within 50 m |
| `odd.dynamic.pedestrian_nearby` | false, true | A walker within 50 m |
| `odd.dynamic.vehicle_ahead_gap` | 0-10-20-30-50-100 m | Scenario measure `vehicle_ahead_gap_m`: the nearest vehicle ahead in the ego's lane or a lane beside it |
| `odd.dynamic.vehicle_ahead_relative_speed` | -30, -15, -5, 5, 15, 30 km/h | Scenario measure `vehicle_ahead_relative_speed_kph` |
| `odd.dynamic.crossing_pedestrian_gap` | 0-10-20-30-50 m | Scenario measure `crossing_pedestrian_gap_m`: the nearest pedestrian ahead that is moving |
| `odd.dynamic.crossing_pedestrian_speed` | 0.5, 1, 1.5, 2.5, 4 m/s | Scenario measure `crossing_pedestrian_speed_ms` |

² Covered by a stay of 2 s or more (`cover_by="entries", min_stay=2`).
Clipping a section while merging or changing lanes does not cover it.

³ Covered by a stay of 3 s or more (`cover_by="entries", min_stay=3`). An ego
accelerating from a stop to 60 km/h passes through every bucket below 60
without driving at any of those speeds. Holding a bucket for 3 s means
driving at roughly that speed, not passing through it. The buckets are
centred on the multiples of 10 km/h, where speed limits are, so an ego
cruising at 50 km/h stays in `[45, 55)` instead of flickering between two
buckets whose edge is at 50.

A reading the simulator does not support returns nothing. The item then has
no samples, the report says so, and the run is not affected. Examples are the
weather on a CARLA build without weather, and the Lanelet2 tags in a run
without a Lanelet2 map.

A bucket an ODD's module rules out (`nonurban`, when the ODD is urban roads
only) is **outside the ODD**. It is listed, and its hits are shown, but it is
not a coverage target: the grade counts only the buckets inside the ODD.

CARLA's weather values are 0-100 intensities, not physical quantities such as
mm/h of rain. The rain and fog buckets are therefore named levels.

Because ODD items are sampled every tick, their hits count ticks. The report
also counts, per bucket, how many **runs** hit it. A bucket held for many
ticks in a single run is still only one situation.

### Coverage criteria: `cover_by` and `min_stay`

By default a bucket is covered after `target` hits, one by default: a single
tick is enough. That lets an ego that clips a three-lane section for one frame
while merging cover `lane_count=3`. Two parameters say what covering means
instead:

- `cover_by` picks the measure `target` counts: `"hits"` (the default),
  `"seconds"` spent in the bucket, `"meters"` the ego drove in it, or
  `"entries"` into it (see exposure, below).
- `min_stay` (seconds) leaves out stays shorter than it: their hits, seconds,
  metres and the entry itself do not count towards `target`.

```python
self.register_cover(
    "gap", ego_gap, buckets=[0, 5, 10, 30], event=SamplingEvent.TICK,
    cover_by="meters", target=50, min_stay=1.0,   # 50 m per bucket, stays of 1 s or more
)
```

`"seconds"`, `"meters"` and `min_stay` need an item sampled on
`SamplingEvent.TICK`: a one-shot sample has no duration or distance.
`"entries"` works on any event. ODD attributes take the same three
parameters (`OddAttribute(..., cover_by="meters", target=200, min_stay=2)`,
or the same keys in a binding file). Crosses take them too.

The coverage file keeps the raw measures and, with `min_stay`, the counted
ones under `counted`. The report grades on the counted measure and shows a
hole's progress (`(hole: 120/200 m)`). Items whose criteria differ between
runs are reported apart, like items whose buckets differ.

### Exposure: seconds, meters, entries

Ticks are a poor measure of how much of a situation a run saw: a stopped ego
keeps adding them. So every item sampled on `TICK` (every ODD item, and any
scenario item or cross with `event=SamplingEvent.TICK`) also records, per
bucket:

| Measure | Meaning |
|---|---|
| `seconds` | simulated time spent in the bucket |
| `meters` | distance the ego drove while the item was in the bucket |
| `entries` | how many times the item entered the bucket; a run of consecutive ticks in it counts once |

A tick stands for the step since the previous tick: its duration, and the
ego's displacement over it. A tick with no sample (a missing value, an
ignored one) ends the stay, so the next sample in the same bucket is a new
entry. A step faster than 100 m/s is a respawn or a teleport, not driving,
and adds no distance; without an ego, no distance is measured. A one-shot item
(`START`, `END`, a condition) has no duration: each hit is an entry, and its
seconds and meters stay 0.

### Situation coverage

ODD modules covered as situations (`docs/odd.md`, Situations) report in a
group of their own between ODD and scenario coverage. Each is an item
`odd.situation.<module>` with one bucket, `holds`, sampled every tick from
the module's verdict, and graded with its `target`, `cover_by` and
`min_stay`. Ticks where the verdict rested on missing values are listed as
`unknown`, outside the bucket.

## The report

```bash
# Every *_coverage.json under outputs/ and multirun/
scenario-coverage outputs/ multirun/

# Also write the merged report as JSON, and the Markdown to a file
scenario-coverage outputs/ --json coverage.json --markdown coverage.md
```

For each ODD the runs were measured against, the Markdown report first says
how long they spent inside it, inside only because values were missing, and
outside it. It
then lists the modules that ruled ticks out, and the runs that left the ODD,
with when. After that comes a summary table per group: item, event, grade,
covered buckets (of those inside the ODD) and holes. Below it, each item gets its buckets with hits and
runs, and an item sampled every tick its seconds, meters and entries too.
Exposure sums over runs; a coverage file written before it was recorded makes
the item's sums unknown (`?` in Markdown, `null` in JSON) rather than short. Runs merge per item: two coverage files describe the same item when the
name, event and buckets agree. An item whose definition changed between runs
is reported once per definition (`name#2`, ...), because adding up hits of
different buckets would mean nothing.

## Exporting the observed conditions as an OpenODD COD

Besides the aggregates, the coverage file keeps every tick's ODD values
(`odd.samples`): the elapsed time, the ego's latitude and longitude (from the
map's OpenDRIVE geoReference), and each attribute's value, `null` when it was
missing. `odd.started_at` is the run's start time in UTC. A run still writes
one file; ten minutes at 20 Hz add about 1 MB.

OpenODD's exchange format for what was observed is the **current operational
domain** (COD) table (ASAM OpenODD 1.0, section 8.3). Write one per run with:

```bash
scenario-coverage outputs/ --export-cod cod/
```

Each run gives three files:

| File | Contents |
|---|---|
| `<run>_cod.csv` | `TEMPORAL_EXTENT` (start time + elapsed, `"YYYY-MM-DD HH:MM:SS.mmm"`), `SPATIAL_EXTENT` (`"lat lon"`), then one column per attribute: `name;unit` for numbers, the literal for categoricals, `true`/`false` for booleans, empty when missing (OpenODD reads it as unknown) |
| `<run>_cod_manifest.csv` | The manifest (8.3.4): which column is which taxonomy concept, in which unit |
| `<run>_taxonomy.yml` | An OpenODD YAML taxonomy of the attributes, which a COD has to travel with (6.1.4.5) |

A tick without an ego position is left out, since `SPATIAL_EXTENT` may not be
empty. OpenODD wants every number to carry a unit: give numeric attributes a
`unit`, or their column has none. A name such as `env.rain.duration` beside an
attribute `env.rain` is written as OpenODD writes a measure of an element: the
key `rain.duration` under `env`. Times are UTC. Coverage files written before samples were recorded are skipped with a
warning. Scenario cover items are not taxonomy concepts and stay in the
coverage file only.

## Coverage file format

```json
{
  "schema": "autoware_carla_scenario.coverage/1",
  "scenario": "CutInScenario",
  "items": [
    {
      "name": "gap_at_cut_in",
      "group": "scenario",
      "text": "",
      "unit": "m",
      "event": "condition:cut_in_started",
      "kind": "numeric",
      "buckets": ["[0, 5)", "[5, 15)", "[15, 40]"],
      "outside_odd": [],
      "target": 1,
      "samples": 1,
      "ignored": 0,
      "hits": {"[0, 5)": 0, "[5, 15)": 1, "[15, 40]": 0},
      "seconds": {"[0, 5)": 0.0, "[5, 15)": 0.0, "[15, 40]": 0.0},
      "meters": {"[0, 5)": 0.0, "[5, 15)": 0.0, "[15, 40]": 0.0},
      "entries": {"[0, 5)": 0, "[5, 15)": 1, "[15, 40]": 0},
      "out_of_range": {}
    }
  ],
  "crosses": [
    {
      "name": "speed_x_gap",
      "group": "scenario",
      "text": "",
      "event": "condition:cut_in_started",
      "items": ["speed_at_cut_in", "gap_at_cut_in"],
      "target": 1,
      "buckets": ["[0, 10) / [0, 5)", "..."],
      "outside_odd": [],
      "hits": {"[0, 10) / [0, 5)": 0, "...": 0},
      "seconds": {"...": 0.0},
      "meters": {"...": 0.0},
      "entries": {"...": 0}
    }
  ],
  "odd": {
    "name": "urban",
    "text": "Urban roads up to 60 km/h",
    "roots": ["root"],
    "modules": [{"name": "roads", "include_and": ["scenery.location in [urban]"], "...": "..."}],
    "unmeasured": [],
    "ticks": {"inside": 512, "assumed": 8, "outside": 40},
    "seconds": {"inside": 25.6, "assumed": 0.4, "outside": 2.0},
    "module_ticks": {"roads": {"failed_ticks": 40, "missing_ticks": 8}},
    "out_intervals": [[12.3, 14.3]]
  }
}
```

## Not yet supported

- **`record()`.** KPIs such as minimum TTC are not collected.
- **`sample()`.** There is no value captured at one event and reported at
  another. Sample on the event itself instead.
