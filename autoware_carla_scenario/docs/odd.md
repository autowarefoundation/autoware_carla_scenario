# ODD

An **Operational Design Domain** (ODD) states the conditions a system is
built to drive in: urban roads, up to 60 km/h, no heavy rain, and so on. Every
run is measured against one. The runner samples the ODD's attributes on every
tick. Their buckets are the run's **ODD coverage**: which conditions it
reached (see [Coverage](coverage.md)). The ODD's modules say which ticks were
**outside** the ODD.

An ODD is written in one of two ways. Both build the same objects, so they
are measured, judged and reported the same way:

- **Python**: openodd's model ([Writing an ODD in Python](https://github.com/hakuturu583/openodd_py/blob/main/docs/python_odd.md)),
  with this framework's probes. It is checked with Codon before it is used,
  like a scenario ([Static Check](typecheck.md)).
- **[ASAM OpenODD](https://www.asam.net/standards/detail/openodd/) 1.0 YAML**
  (the standard's chapter 10). It is read into the same objects. A **binding
  file** next to it names the probes that measure its concepts.

```
 OpenODD YAML + binding file ──(load_odd_binding)──┐
                                                    ├──▶ OddDefinition ──▶ coverage + inside/outside per tick
 Python (checked by Codon) ─────────────────────────┘
```

OpenODD also maps its model to OpenSCENARIO DSL (chapter 9, model to DSL only)
and to tables (chapter 8). Neither is read here.

The reading, judging, versioning, sampling and coverage are
[openodd](https://github.com/hakuturu583/openodd_py) (`openodd-py`, a Rust
core with a Python API), which knows nothing of CARLA. This framework is its
**backend**: `autoware_carla_scenario.odd.backend.CarlaBackend`, a subclass
of `openodd.Backend`, gives the built-in probes a binding file names,
`measure:` keys read from the running scenario, the knobs that set the
weather and the sun, and where the ego is (for coverage). Importing `autoware_carla_scenario.odd`
makes it openodd's default backend, so `openodd.load_odd_binding()` and
`autoware_carla_scenario.odd.load_odd_binding()` read the same.

### Versions of an ODD

OpenODD 1.0 defines no version of an ODD; openodd builds one on what it does
define ([its versioning profile](https://github.com/hakuturu583/openodd_py/blob/main/docs/versioning.md)):

- the **revision** is the SHA-256 of the files the ODD was read from -- the
  binding file, the OpenODD files it names and every file they `IMPORT` -- by
  name and content. The same files are the same revision on any machine and
  from any repository or commit; any change to any of them is a new one;
- the **version** is what the root modules (those under `ODD:`) declare in
  their `METADATA` (`version: 2.1.0`);
- the **structure** is the digest of the ODD as built (modules, buckets, the
  buckets outside it), for an ODD written in Python, which has no files.

All three are written to every coverage file (the ODD's `revision`,
`version`, `structure`), so coverage can be told apart by the ODD it was
measured against. `OddDefinition.archive()` packs the files into a
deterministic `.tar.gz` that `openodd.load_odd_archive()` reads back exactly;
a run writes it beside its coverage file (`{ScenarioName}_odd.tar.gz`).

## Concepts

The model and its semantics follow OpenODD 1.0 (chapters 6 and 7).

| Concept | What it is |
|---|---|
| **Attribute** | A taxonomy concept, measured on every tick: a **probe** reads it from the world, and its **buckets** make it a cover item. |
| **Condition** | A test on one attribute (`speed_limit.between(0, 60)`), a group of conditions (`all_of`, `any_of`), another module's verdict (`module_holds`), or a missing value (`is_unknown`). |
| **Module** | A named rule: `INCLUDE AND (NOT EXCLUDE)`. It has at most one include section and at most one exclude section. |
| **Label** | A name several modules declare. It holds when any active module declaring it holds. |
| **Root** | An entry point. Each root candidate (`roots=`; by default every module) that no other module refers to, by name or through a label, is a root. The ODD holds when its active roots hold. |

**Missing values.** OpenODD's *missing-value semantics* apply. A value the
probe could not read does not, by itself, put a situation outside the ODD
(open world). Examples are a simulator without weather, and a run without a
Lanelet2 map. When a value is required, the ODD says so with
`attr.is_unknown()` in an exclude section (`x: unknown` in YAML). A tick that
is inside only because values were missing is reported as such.

**Inactive modules.** An inactive module is ignored. A condition referring to
it, either way, drops out of its section, as if it were not written. A
section with nothing left in it counts as absent: an include section then
holds, and an exclude section does not. The standard says such conditions
are "automatically satisfied"; taken literally, that would make an exclude
section that refers to a switched-off hazard exclude everything, so this
reads "ignored" as the neutral element.

Modules follow ISO 34503's "default" definition mode, as OpenODD requires.
Whatever no module rules out is inside the ODD.

## Writing an ODD in Python

An ODD is a function that takes nothing and returns an `OddDefinition`. The
model, the registry and the reading of a name are openodd's (its
[Writing an ODD in Python](https://github.com/hakuturu583/openodd_py/blob/main/docs/python_odd.md));
this framework re-exports them, gives them its probes, and checks the ODD
with Codon:

```python
from __future__ import annotations

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    OddAttribute,
    OddDefinition,
    OddModule,
    module_holds,
    register_odd,
)
from autoware_carla_scenario.odd import (
    INTENSITY_LEVELS,
    lanelet_location,
    rain,
    speed_limit_kph,
)


def ego_yaw_rate(world: carla.World) -> float | None:
    ...  # any function of the world is a probe; None means unknown


def urban_odd() -> OddDefinition:
    location = OddAttribute(
        "scenery.location", lanelet_location, values=["urban", "nonurban", "private"]
    )
    speed_limit = OddAttribute(
        "scenery.speed_limit", speed_limit_kph, unit="km/h",
        buckets=[0, 30, 40, 50, 60, 80, 100],
    )
    weather = OddAttribute("environment.rain", rain, values=INTENSITY_LEVELS)
    yaw_rate = OddAttribute(
        "dynamic.yaw_rate", ego_yaw_rate, unit="deg/s", range=(-30.0, 30.0), every=10.0
    )
    return OddDefinition(
        "urban",
        [location, speed_limit, weather, yaw_rate],
        [
            OddModule(
                "urban_roads",
                include_and=[location.is_in(["urban"]), speed_limit.between(0, 60)],
                text="Urban roads up to 60 km/h",
            ),
            OddModule("weather", exclude_or=[weather.is_in(["heavy"])], labels=["fair"]),
            OddModule(
                "root", include_and=[module_holds("urban_roads"), module_holds("fair")]
            ),
        ],
        roots=["root"],
        text="Urban roads, fair weather",
    )


register_odd("urban", urban_odd)
```

### Attributes

`OddAttribute(name, probe, *, unit, range, every, buckets, values, text)`
takes the bucket arguments of
[`register_cover()`](coverage.md#register_cover-arguments). Give at most one
of `values`, `buckets` and `range` (with `every`). An attribute with none of
them is **monitored but not covered**: conditions may test it, and the report
lists it so the blind spot is visible. Its cover item is named
`"odd." + name`.


### Conditions

| Method | Holds when |
|---|---|
| `attr.is_in(values)` | the value is one of `values` (compared by label: an enum by name, a bool as `true`/`false`) |
| `attr.equals(value)` | the value equals `value` |
| `attr.between(low, high)` | `low <= value <= high` |
| `attr.at_least(x)` / `attr.at_most(x)` | `value >= x` / `value <= x` |
| `attr.greater_than(x)` / `attr.less_than(x)` | `value > x` / `value < x` |
| `attr.is_unknown()` | the value is missing |
| `all_of([...])` / `any_of([...])` | all of / any of the conditions hold |
| `module_holds(name, holds=True)` | the module (or label) named `name` holds, or does not |

Numbers are in the attribute's `unit`.

### Modules

`OddModule(name, *, include_and, include_or, exclude_and, exclude_or, labels,
active, text, situation, target, cover_by, min_stay)`. Give at most one of
`include_and` / `include_or`, and at most one of `exclude_and` / `exclude_or`.

### Situations

Attribute coverage counts each attribute on its own: three lanes and 60 km/h
can both be covered by runs that never drove 60 km/h on a three-lane road. A
module already says such a combination, so a module can be **covered as a
situation**:

```python
OddModule(
    "three_lane_cruise",
    include_and=[lanes.equals(3), speed.between(55, 65)],
    situation=True,
    cover_by="meters", target=200, min_stay=3.0,   # 200 m, counting stays of 3 s or more
)
```

Every tick, the module's verdict (which the ODD works out anyway) is a sample
of the item `odd.situation.three_lane_cruise`, which has one bucket, `holds`.
A tick where the module holds is a sample in it. A tick where it fails, or is
inactive, ends the stay. A tick where it rests on missing values ends the stay
too, and is counted apart as `unknown`. `target`, `cover_by` and `min_stay`
work as for attributes (`docs/coverage.md`). The report lists situations in a
group of their own, **Situation coverage**.

A situation is what the runs should drive, not a bound of the ODD: with
`roots` left out, a situation nothing refers to is not a root candidate. To
make it a bound too, refer to it from a root, or name it in `roots`.

### Built-in probes

These live in `autoware_carla_scenario.odd`:

| Probe | Returns | Read from |
|---|---|---|
| `ego_speed_kph` | float, km/h | Ego velocity |
| `speed_limit_kph` | float, km/h | Lanelet2 `speed_limit` tag, else CARLA |
| `lanelet_speed_limit_kph` | float, km/h | Lanelet2 `speed_limit` tag |
| `lanelet_location` | str | Lanelet2 `location` tag (`urban`, `nonurban`, ...) |
| `lanelet_subtype` | str | Lanelet2 `subtype` tag (`road`, `highway`, ...) |
| `in_junction` | bool | CARLA waypoint |
| `lane_count` | int | Driving lanes in the ego's direction (CARLA, outside junctions) |
| `illumination` | `ILLUMINATION_LEVELS` | Sun altitude |
| `rain`, `fog` | `INTENSITY_LEVELS` | CARLA weather (0-100) |
| `visibility` | float, m | How far drivers see, as the run's weather left it ([Visibility follows the weather](#visibility-follows-the-weather)); nothing when the run set none |
| `traffic_density` | `TRAFFIC_DENSITY_LEVELS` | Other vehicles within `NEARBY_RADIUS_M` |
| `pedestrian_nearby` | bool | A walker within `NEARBY_RADIUS_M` |
| `vehicle_<field>_mean`, `vehicle_<field>_std`, `pedestrian_<field>_mean`, `pedestrian_<field>_std` | float | The mean and std of a traffic parameter's spread, as the run set it ([Traffic parameters from the ODD](#traffic-parameters-from-the-odd)); nothing when the run set none |
The Lanelet2 probes need the run to have a Lanelet2 map (`map.lanelet2_path`).
Without one they return nothing.

The probes whose value the map alone decides (`speed_limit_kph`,
`lanelet_speed_limit_kph`, `lanelet_location`, `lanelet_subtype`,
`in_junction`, `lane_count`) can also read it off a lanelet, which is how a
planned route is checked before a run
([Checking a planned route](#checking-a-planned-route)) and how a sweep picks
the road a drawn case is on ([The road a case is on](#the-road-a-case-is-on)).

What they read off a lanelet is public too, for code that has to judge a
lanelet exactly as the ODD does: `lanelet_in_junction(lanelet)` (a
`turn_direction` tag, `scenery.junction`), `lanelet_road_type(lanelet)` (the
`subtype` tag, `scenery.road_type`) and `lanelet_speed_limit(lanelet)` (the
`speed_limit` tag, km/h). The adversities' location gate reads them
(`autoware_carla_scenario.adversity.location`).

Import them from `autoware_carla_scenario.odd` itself
(`from autoware_carla_scenario.odd import rain`). The Codon model that checks
an ODD has no `probes` submodule.

### Scenario measures

The road users a scenario stages -- a vehicle ahead in the next lane, a
pedestrian running out -- are not read by ODD probes: they are **measures**
of the scenario, under keys the framework fixes
(`autoware_carla_scenario.measures`). Every `BaseScenario` has these, read
generically around the ego:

| Key | Unit | Version | Reads |
|---|---|---|---|
| `vehicle_ahead_gap_m` | m | 2 | The free road to the nearest vehicle ahead on the ego's lane or a lane beside it: its rear bumper less the ego's front bumper, along the ego's lane on the Lanelet2 map (up to `AHEAD_RANGE_M`); a vehicle about to cut in is measured from before it changes lanes |
| `vehicle_ahead_relative_speed_kph` | km/h | 2 | That vehicle's speed along the ego's lane less the ego's |
| `crossing_pedestrian_gap_m` | m | How far ahead the nearest pedestrian ahead that is moving was when it set off, held while it crosses |
| `crossing_pedestrian_speed_ms` | m/s | That pedestrian's speed |
| `stop_line_distance_m` | m | 1 | Along the ego's lane, from its front bumper to the nearest stop line ahead -- a traffic light's, a traffic sign's or a road marking's, as a `stop_line_offset` binding finds them -- up to `AHEAD_RANGE_M`, on any branch of the lane where it forks (the measure does not know the route); nothing without a lane to read along |

The pedestrian measures read in the ego's frame: ahead along its heading,
within `PEDESTRIAN_LATERAL_M` of its centre line.

#### The vehicle ahead

The vehicle-ahead measures read along the road, as a driver would say "the
car in front is 15 m ahead", and from the map:

- **The ego's lane** is the lanelet the ego is on -- of those under its
  origin (within `ON_LANELET_M`), the one whose centreline runs closest to its
  heading, within `EGO_LANE_HEADING_DEG` -- with every lanelet following it up
  to `AHEAD_RANGE_M` ahead and the lanelets behind it. Where the lane
  branches, **every branch** is the ego's lane: the measure does not know the
  route, so a vehicle ahead on either branch counts. Behind, the lane runs back
  one way only, through the first of the lanelets before each (Lanelet2's
  `previous(...)[0]`), where lanes merge; a lanelet joining the ego's lane
  is not its lane, nor beside it. Its arc length `s` runs along their
  centrelines, on from one to the next.
- **A vehicle counts** when its origin is on one of those lanelets, or on a
  lanelet beside one of them -- its left or right neighbour in the Lanelet2
  routing graph, whether a lane change is allowed across the line or not --
  and further along the ego's lane than the ego's origin. A lane two over, an
  oncoming lane or a vehicle off the lanelets does not count, however near.
- **The gap** is `s` of the vehicle's rear bumper less `s` of the ego's front
  bumper. A bumper is the middle of the end of the actor's bounding box,
  along its own heading (`Bumpers`, `bumpers_of`), and it is projected onto
  the ego's lane: onto the lanelet the vehicle is on, or the one it is beside,
  and those before and after it. So the gap does not depend on where the
  lanelets beside each other are cut or how long they are -- the outer lane of
  a bend is longer than the inner one, and the same `s` on the two is not
  side by side -- and on a bend it is the distance along the road, not a
  straight line. A vehicle on the ego's own lane that overlaps it reads a
  negative gap; one in a lane beside that overlaps it (its rear bumper behind
  the ego's front bumper) is alongside, not ahead, and does not count, so it
  never hides the vehicle that is ahead. One beside and ahead of the ego
  counts, which is how a vehicle about to cut in is measured before it
  changes lanes.
- **The relative speed** is the vehicle's velocity along the lane's tangent at
  its rear bumper less the ego's along the tangent at its front bumper: two
  vehicles keeping pace round a bend read 0 though their headings differ.

Without a lane to measure along -- no Lanelet2 map loaded, or the ego on no
lanelet that runs its way -- the measures fall back to the **geometric**
definition: a vehicle ahead of the ego's origin within three
`LANE_HALF_WIDTH_M` of its centre line counts, and the gap is bumper to bumper
along the ego's heading. The run logs a warning the first time, and the
coverage file says how many frames each definition read
(`measures.vehicle_ahead_gap_m.definitions`: `along_lane`, `geometric`;
`vehicle_ahead_definitions()` in Python). The geometric definition, too,
does not count a vehicle alongside in the next lane.

A reading costs about 3 ms with ten vehicles around the ego on CARLA
Town04_Opt (it was some 170 ms).
The ego's lane is found once and kept while its lanelet is still the ego's
-- the ego on it, running its way, and no lanelet overlapping it (found by
geometry, however the map cuts a fork's branches or an exit into lanelets)
running closer to the ego's heading -- and the whole
map is searched only when it is not (or, off the lanelets, once the ego has
moved 2 m or turned 5°); a vehicle is placed by
the lane's own lanelets, not by searching the map; and a projection reads
only the stretch of a centreline near the point. The Lanelet2 routing graph
the lane is read from -- some 25 s to build on Town04_Opt -- is built in the
run's setup whenever the ODD maps a vehicle-ahead measure that the
scenario reads with the built-in one -- not for one it replaced with
`register_measure` (`prepare_measures`) -- with every lanelet's bounding box,
not on the first tick. A failed reading reads nothing and is warned of once a run.

#### Measure versions

A measure's **version** (`Measure.version`, `measure_version(key)`) says
which definition it reads with, and is raised whenever what it reads changes
meaning. The vehicle-ahead measures are version 2. Version 1 (before
autoware-carla-scenario 12.0) read the gap centre to centre along the ego's
heading -- 4.75 m more than version 2 for a Lincoln MKZ behind a Mini on
CARLA 0.10 -- and counted a vehicle within three half-lanes of the ego's
centre line, whatever the map.

Coverage keeps the versions apart. openodd's merge treats two coverage items
as the same when their name, buckets, event and criteria agree; it does not
read their description. So the cover item of an ODD attribute mapped onto a
measure past version 1 is written under its version,
`odd.dynamic.vehicle_ahead_gap@v2`: merged with files written before, the
report lists `odd.dynamic.vehicle_ahead_gap` (version 1) and
`odd.dynamic.vehicle_ahead_gap@v2` (version 2) apart, never one bucket's hits
added to the other's. The mark is `@v<n>`, not `#<n>`: `#<n>` is the suffix
openodd's merge gives an item it meets defined differently (version 1 files
whose buckets changed merge as `odd.dynamic.vehicle_ahead_gap#2`), and
openodd's sampler adds up an item's `#<n>` definitions. A coverage-driven
sweep (`strategy: coverage`) reads only the current version: `OddSampler`
hands openodd's sampler the `@v2` items under the attribute's own item name
and leaves the older ones out (`current_measure_coverage`), so a
`coverage_from` that also holds version 1's files is safe. The 12.0.0
release wrote version 2 as `#2`; those files read as version 1, and their
hits are left out of a coverage-driven sweep on purpose: since 12.0.0 a
vehicle alongside in the next lane no longer counts as ahead, which changed
what version 2 reads, not only its name. Each
coverage file also lists the
measures its ODD reads under `measures`, with their version, unit and text.
A scenario that replaces a built-in measure (`register_measure`) promises the
same quantity, so its replacement carries the built-in's version.

Each reads nothing (missing) while there is no such road user. A scenario
that knows better replaces one -- it knows which actor *is* its cut-in
vehicle -- or adds one of its own:

```python
class MyCutIn(BaseScenario):
    def setup(self) -> None:
        ...
        self.register_measure(VEHICLE_AHEAD_GAP_M, self._gap_to_my_npc)
```

An ODD **maps** its taxonomy onto measures: in Python with
`scenario_measure(key)` as the attribute's probe, in a binding file with
`measure:` in place of `probe:`:

```python
OddAttribute(
    "dynamic.vehicle_ahead_gap",
    scenario_measure(VEHICLE_AHEAD_GAP_M),
    unit="m",
    buckets=[0, 10, 20, 30, 50, 100],
)
```

```yaml
probes:
  headway: {measure: vehicle_ahead_gap_m, buckets: [0, 10, 20, 30]}
```

The attribute then reads the running scenario's measure, its replacement
included, in the measure's unit: an attribute mapped onto a built-in measure
takes its unit, and one given that is not it is refused.  The keys are in the
Codon model (`from autoware_carla_scenario.measures import VEHICLE_AHEAD_GAP_M`),
so a scenario or a Python ODD naming them is checked like any other
([Static Check](typecheck.md)): `autoware_carla_scenario.measures` is modelled
whole, `Measure`, `BUILT_IN_MEASURES`, `read_measure()`, `check_measure_key()`
and `measured_scenario()` included. A `Measure` is frozen, as a `Metric` is:
assigning to one of its fields fails the check, as it fails the run. A
scenario may build its own (`Measure("my_measure", read, "m")`); the check
calls its `read` with a world and refuses one that returns anything but a
`float` or an `Optional[float]`. A metric's keys are modelled the same way
([Metrics](metrics.md)): `autoware_carla_scenario.metrics` is modelled whole, `Metric` and `BUILT_IN_METRICS` included. The ODD names a measure key, never a scenario; the scenario names
its measures, never the ODD. The default ODD maps
`dynamic.vehicle_ahead_gap`, `dynamic.vehicle_ahead_relative_speed`,
`dynamic.crossing_pedestrian_gap` and `dynamic.crossing_pedestrian_speed`
onto the four built-in measures; the two of the vehicle ahead count stays of
1 s or more (`min_stay`), so a gap the ego only closes through is not
covered.

#### Measures and metrics

A measure reads a condition the ego drives in: what the scenario sets up.
How the ego *drives* -- whether it collides, leaves the road, drives against
the traffic, how close it comes to other road users -- is an outcome, read by a
**metric** ([Metrics](metrics.md)). An ODD cannot map onto a metric:
`scenario_measure("offroad")`, or `measure: offroad` in a binding file, is
refused. Were it allowed, the ego's own driving would decide whether a run is
inside the domain it is tested in, and an ODD excluding `offroad` would report
a run that left the road as outside the ODD rather than as failed. A run
watches a metric with `MetricCondition` instead.

## Writing an ODD in OpenODD YAML

The OpenODD files are plain OpenODD 1.0. The example below follows the
standard's own examples (Code 170, 171 and 192):

```yaml
# taxonomy.yml
TAXONOMY:
    environment_conditions:
        rainfall_rate: float precipitation_rate
        rainfall_level:                       # literals defined by ranges: ordered
            no_rain:
                rainfall_rate: "< 0.1 mm/h"
            light_rain:
                rainfall_rate: "[0.1 .. 2.5] mm/h"
            heavy_rain:
                rainfall_rate: "> 2.5 mm/h"
        wind_speed: float velocity
    scenery:
        road_type: [road, highway, road_shoulder]
        lane_count: integer count
```

```yaml
# odd.yml
IMPORT:
    - taxonomy.yml

ODD:
    odd1:
        TITLE: The baseline ODD
        INCLUDE_AND:
            low_speed_roads: true
            needs_wind: true
        EXCLUDE_OR:
            bad_weather: true

MODULES:
    low_speed_roads:
        INCLUDE_AND:
            road_type: [road]
            lane_count: "< 3"
    bad_weather_1:
        LABEL: bad_weather
        INCLUDE_OR:
            rainfall_level: ">= heavy_rain"
            wind_speed: "> 50 km/h"
    needs_wind:
        EXCLUDE_OR:
            wind_speed: unknown               # the wind speed is required
```

The binding file is this framework's own. It says which probe measures each
concept, and its buckets:

```yaml
# urban.yaml
openodd: [odd.yml]                 # relative to this file
name: urban                        # default: this file's stem
text: Urban roads, fair weather
probes:
  road_type: {probe: lanelet_subtype}
  rainfall_rate: {probe: my_package.probes:rain_rate_mm_h, unit: mm/h}
  wind_speed: {probe: my_package.probes:wind_mps, unit: m/s}
  lane_count: {probe: lane_count, values: [1, 2, 3, 4], cover_by: meters, target: 200, min_stay: 2}
```

Run with `odd=path/to/urban.yaml`. An OpenODD file on its own can be named
too, but then nothing is measured: every concept is missing.

### Taxonomies and modules from git

An `openodd` entry can name a file in a git repository at a revision, so a
taxonomy or a module library kept elsewhere is used as it was at that
revision:

```yaml
# urban.yaml
openodd:
  - git: https://example.com/odd/taxonomy.git
    rev: 'v1.2.0'                  # a tag, a branch, or a commit
    path: taxonomy.yml             # a list for several files
  - git: git@example.com:odd/modules.git
    rev: '3f2a9c1e0b7d4a6c8e5f1a2b3c4d5e6f7a8b9c0d'
    path: odd/urban.yml
  - local_overrides.yml            # relative to this file
probes:
  ...
```

- An `IMPORT` is looked for next to the importing file first, and then in
  the other sources: the directory of each file named, and the root of each
  repository. The standard makes file names unique within one transmission,
  so `odd/urban.yml` in one repository can `IMPORT: [taxonomy.yml]` from
  another. A name found in two sources is refused.
- `git fetch` does the fetching, so `url` is anything it takes, with the
  credentials it is configured with. Prompts are turned off, and the `ext::`
  transport is refused.
- Repositories and checkouts are cached in
  `$AUTOWARE_CARLA_SCENARIO_ODD_CACHE` when it is set, else openodd's
  `$OPENODD_CACHE` (default `~/.cache/openodd`). A full commit id checked out
  once is read from the cache without the network; a tag or a branch is
  fetched on every load, to learn which commit it names now. Pin a commit for
  runs that must be reproducible.
- Quote `rev`: unquoted, a commit id of digits and one `e` (`1630255e36`)
  or a long one of digits only is read as a number, and looked for as one
  (`1.630255e+36`).
- Fetching needs `git`, which the scenario image does not have. A run there
  can still read a source pinned to a full commit id, from a cache filled
  beforehand (`scenario-odd show urban.yaml` with the same cache directory)
  and mounted at `$AUTOWARE_CARLA_SCENARIO_ODD_CACHE`.
- The commit each source was read at is written to the coverage file (the
  ODD's `sources`), and `scenario-coverage` lists it, warning when the merged
  runs read different commits of a source.

From Python, pass `GitSource(url, rev, path)` to `load_openodd()`.

### What is read

**`IMPORT`**: other files, relative to the importing one, or else in the
other sources (see above). A cycle is refused.

Every file read with an ODD makes up one transmission, in the standard's
words, and the standard makes two things unique within one. Both are
refused rather than guessed at:

- two different files of the same name, wherever they come from;
- a concept defined in two files, even the same way. A file can still add
  concepts to a container another file started (`weather:` in both, with
  different concepts under it).

**`TAXONOMY`**: nested mappings are records and containers. A leaf is one of:

- `integer|long|float|double <unit type>`: a number;
- `boolean`;
- a list of literals: a categorical;
- a mapping of literals to expressions on other concepts: a categorical
  defined by them. Its value is the literal whose expressions hold; on a
  shared range endpoint, the literal written first wins. Defined by ranges,
  its literals are **ordered** in the order written, so `"< heavy_rain"`
  means the literals before it;
- the id of a categorical: the same literals.

A reference to a record (a user-defined type) and a `shapefile` are not
followed.

**`MODULES`** and **`ODD`**: modules. Those under `ODD` are the root
candidates: the ones among them no other module refers to are the roots. The
standard's examples also put referenced modules there. Without an `ODD`
section, every module no other module refers to is a root.

A module's keys are:

- `TITLE`, `DESCRIPTION`;
- `ACTIVE`;
- `LABEL` / `LABELS` (one name or a list);
- `METADATA` (ignored, here and inside a section);
- one `INCLUDE_AND` or `INCLUDE_OR`, and one `EXCLUDE_AND` or `EXCLUDE_OR`.

A section nests one level of the other operator: `OR:` in an `AND` section,
`AND:` in an `OR` one.

A section maps a concept to an expression, or a module or label to
`true`/`false`. A concept is named by its id (`wind_speed`), or by as much of
its path as makes it unique.

| Expression | Meaning |
|---|---|
| `low`, `3`, `true`, `"0 mm/h"` | equal |
| `[a, b]` | one of the literals |
| `"> x unit"`, `">= x"`, `"< x"`, `"<= x"` | a bound (`>` excludes `x`, `>=` includes it) |
| `"[low .. high] unit"`, `"[low, high] unit"` | an inclusive range |
| `"< literal"`, `"[a .. b]"` | a bound or range over an ordered categorical |
| `unknown` (`none`, `null`, `undefined`, or YAML `null`) | the value is missing, unless the categorical has a literal of that name |

A number's unit must measure the concept's unit type. It is converted into
the probe's unit, so `"> 50 km/h"` against a probe in m/s is `> 13.89`. The
units known are listed below; a document adds more with OpenODD's
`conversion:` block.

| Unit type | Units |
|---|---|
| length | m, km, cm, mm, mi, in, ft |
| velocity | m/s, km/h (kph), mph |
| acceleration | m/s^2, g |
| time | s, ms (msec), min, h |
| angle | deg, rad |
| precipitation_rate | mm/h (mm/hr) |
| temperature | C, K, F |
| fraction | % |
| bandwidth | bps, kbps, Mbps, Gbps |
| frequency | Hz, kHz |
| illuminance | lx |
| count, occurrence | count, 1/h, occ/hr |

The following are refused, each with the module it appears in:

- a concept that is not in the taxonomy;
- a literal a categorical does not have;
- a unit of the wrong type;
- a second include or exclude section;
- deeper nesting;
- a module or label id containing `unknown`.

The YAML is read with YAML 1.2 booleans: only `true` and `false` are
booleans, so literals such as `off` or `NO` stay literals. A key written
twice is refused. As the standard notes, an unquoted expression starting
with `>` (`speed: > 50 km/h`) is a YAML block scalar, so quote it.

Not read: `COD` / `OD` records, numeric terms (`1.75*ego_width`), `$`
parameters, and condition-level metadata.

### The binding file

| Key | Meaning |
|---|---|
| `openodd` | The OpenODD files, relative to the binding file |
| `name`, `text` | The ODD's name and description |
| `probes` | Concept → `probe` (built-in name or `package.module:function`) or `measure` (a scenario measure's key, see [Scenario measures](#scenario-measures)), `unit` (built-in probes and measures know theirs), buckets (`values`, `buckets`, or `range` + `every`), `text`, and the coverage criteria `target`, `cover_by`, `min_stay` |
| `situations` | Module → its criteria (`target`, `cover_by`, `min_stay`, all optional): the modules to cover as situations (see Situations above) |

```yaml
situations:
  three_lane_cruise: {cover_by: meters, target: 200, min_stay: 3}
  urban_junction_rain: {}          # one stay of any length
```

Situations live in the binding file, not in the OpenODD documents: OpenODD has
no notion of coverage, and a construct of its own in an OpenODD document would
not be compliant. A module named under `situations` and not under the `ODD`
section is not a root candidate.

When no buckets are given:

- a categorical gets one bucket per literal, and a boolean gets two;
- a number gets buckets at the **thresholds the modules test it against**.
  `"<= 60 km/h"` makes `[-inf, 60)` and `[60, inf]`, so every boundary the ODD
  draws is tested from both sides. Each bucket holds its lower edge, so
  `[60, inf]` holds 60, which `<= 60` lets in. That bucket is partly inside, so
  it stays a coverage target. With `"< 60 km/h"`, it would be outside.

A categorical defined by expressions is measured when what its expressions
read is. A concept with no probe is always missing, and is reported as
monitored but not covered.

## Picking the ODD of a run

The `odd` config key, or `ScenarioQueue(odd=...)` /
`ScenarioRunner(odd=...)`, takes one of:

| Value | ODD |
|---|---|
| `default` | The built-in one ([Coverage](coverage.md#odd-coverage)): every attribute, no modules. It is openodd's bundled `default` ODD ([its files](https://github.com/hakuturu583/openodd_py/tree/main/python/openodd/odds/default)), measured by this framework's probes, so it has a revision like any ODD read from files |
| a name | One registered with `register_odd(name, builder)` in code, or by a package through the `autoware_carla_scenario.odds` entry point group (or openodd's `openodd.odds`) |
| `path/to/urban.yaml` | A binding file (or an OpenODD file on its own, measuring nothing) |
| `package.module:function` | A builder to import and call |

```bash
uv run scenario scenario=intersection_passing/left_turn odd=path/to/urban.yaml
uv run scenario scenario=intersection_passing/left_turn odd=my_package.odds:urban_odd
```

A package that ships its ODD registers it through the
`autoware_carla_scenario.odds` entry point group, or openodd's `openodd.odds`
when the ODD is not CARLA's alone (both are read; `openodd.resolve_odd()`
resolves names as `resolve_odd()` does, with files measured by its `backend`). The entry point names a
function that takes nothing and calls `register_odd`, the convention the
scenario and traffic-backend groups use. A private taxonomy can then plug in
without the framework knowing it:

```toml
[project.entry-points."autoware_carla_scenario.odds"]
my_package = "my_package.odds:register"
```

```python
def register() -> None:
    register_odd("urban", urban_odd)
```

## Sampling scenarios from the ODD

Coverage says which situations of the ODD the runs so far have driven. The
sampler draws the settings of the next runs from the ODD: only combinations
the ODD admits, and, if asked, aimed at what is still uncovered. It is a
sweep, so it is run by `scenario-expand` and by the lanelet sweeper, and with
`autoware.launcher` each case gets a fresh Autoware ([Autoware as the
Ego](autoware.md)):

```bash
uv run scenario --multirun hydra/sweeper=lanelet_constraint \
  scenario=intersection_passing/left_turn map=town10hd_opt \
  +sweep.odd_sample.count=20 +sweep.odd_sample.strategy=coverage \
  '+sweep.odd_sample.coverage_from=[outputs]'
```

A glob batch with `--multirun` draws each matched scenario's cases the same
way and runs them all in one queue ([Batch Execution](usage.md#running-every-sweep-case-in-a-batch)).
Scenarios that draw from one ODD go on from each other's draws, in the order
the batch lists them: with `strategy: coverage` a later scenario counts the
earlier ones' cases as covered and aims at the buckets and situations they
left (see [How a case is drawn](#how-a-case-is-drawn)); with
`strategy: uniform` it draws from a random stream of its own rather than
repeat their values. Here `cut_in/right`'s 3 cases aim at what
`cut_in/left`'s 3 left uncovered:

```bash
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt \
  +sweep.odd_sample.count=3 +sweep.odd_sample.strategy=coverage
```

The same seed, ODD, knobs, coverage and scenarios, in the same order, give
the same cases, on any machine (openodd's sampler draws from ChaCha8 seeded
with the seed's SHA-256). Two ODDs are one when they are built alike (their
`structure`), however the command line names them (a registered name, a YAML
path); a scenario on another ODD starts afresh.

or in the scenario's YAML:

```yaml
sweep:
  odd_sample:
    count: 20              # cases; one per lanelet case when left out
    seed: null             # the run's (`seed`) when null; same seed, ODD, knobs and coverage: same cases
    strategy: coverage     # uniform | coverage | weighted (weighted when `weights` is given)
    coverage_from: [outputs]   # earlier runs' *_coverage.json (files or directories)
    weights: null          # module -> share of the cases, or a YAML file of them; see "Sampling by shares of the ODD"
    odd: null              # the ODD to draw from; the run's (`odd`) by default
    knobs: {}              # by ODD attribute, replacing the scenario's controls; see below
    flaky: null            # look for flaky cases; see "Finding flaky cases" below
    tire_friction: true    # draw each case's grip in the range its weather allows; see "Tire friction follows the weather"
    visibility: true       # draw how far each case's drivers see in the range its weather allows; see "Visibility follows the weather"
```

Each case is a list of Hydra overrides. A sweep that also has `constraints`
gives each drawn case the next lanelet case in turn; one without draws from
the ODD alone, and needs no map to expand. The last override,
`++odd_case=...`, is the case itself, encoded: the run records it, with its
outcome, in its coverage file (see [Finding flaky cases](#finding-flaky-cases)).

### Who knows what

Three things meet in a sampled case, and none of them knows the others:

| | Says | Knows |
|---|---|---|
| **The ODD** | which conditions the system is for, and which scenario measure each of its attributes is (its mapping) | its taxonomy and the measure keys; no scenario |
| **A scenario** | its measures (`BaseScenario`'s, or its own replacements) and its **controls**: which of its parameters sets which measure, and what it can stage | the measure keys; not the ODD's taxonomy or bounds |
| **The framework** | the measure keys, and the weather and sun every scenario can be given | neither |

An ODD tightened, loosened or written in another taxonomy needs no scenario
changed; a scenario added needs no ODD changed. The sampler joins them on the
measure key: an attribute the ODD maps onto measure *k* is drawn with the
running scenario's control of *k*. Attributes nothing sets are left open: a
combination is admitted when the ODD can hold for some value of them.
Controls of measures the ODD does not map are unused -- what the ODD does not
speak of, it does not bound.

### Controls

A **control** (an `OddKnob`) ties a measure to the config key that sets it.
A scenario declares its own in its config, next to the parameters they set,
keyed by measure:

```yaml
# scenario/cut_in/left.yaml
controls:
  vehicle_ahead_gap_m:
    key: scenario.npc_gap_m
    target: measure                    # the gap the measure reads at the start
    range: [2.0, 30.0]                 # what the scenario can stage
  vehicle_ahead_relative_speed_kph:
    key: scenario.npc_initial_speed_kmh
    offset: ${ego.initial_speed_kmh}   # its speed less the ego's
    range: [-15.0, 15.0]
    limits: [0.0, null]                # a speed: never below zero, however slow the ego
```

| Field | Meaning |
|---|---|
| `key` | The config key the value is written to |
| `target` | `setting` (the default): the key is a raw setting, written through `scale` and `offset`. `measure`: the key takes the measure's value at the start itself, written unchanged -- no `scale`, `offset`, `integer` or `values` (see below) |
| `range` | Numeric: `[low, high]` in the measure's unit, the values the scenario can bring about. A bucket is drawn only where it meets the range (by more than an edge), and only that part of it |
| `values` | Per bucket label. Categorical: the value to write, or `[low, high)` to draw it from (`[x, x]` is `x`); a bucket without one is not drawn. Numeric: a range to draw from instead of the bucket's interval -- which an unbounded bucket (`-inf`/`inf` edges) needs to be drawn at all |
| `scale`, `offset` | A numeric value drawn in the unit is written as `value * scale + offset` (`scale` not zero) |
| `integer` | Write an integer. A rounded numeric value that falls in another bucket is drawn again |
| `limits` | Numeric: `[low, high]` the **written** value must lie in, `null` for an open end -- what the key takes, whatever `offset` comes to. A bucket only values outside it would reach is not drawn |

#### Checking controls

`scenario-check` validates `controls:` of every config it checks, and
`scenario-expand`, a sweep and a `--multirun` batch refuse invalid ones, each
problem as one line with its config path, not a traceback:

```text
cut_in/left: controls.vehicle_ahead_gap_m.range: expected [low, high] with low <= high, got [30, 2]
cut_in/left: controls.vehicle_ahead_gap_m.rnge: unknown field (did you mean 'range'?)
cut_in/left: controls.vehicle_ahead_gap_m: expected a key or a mapping, got 5
cut_in/left: controls.vehicle_ahead_gap_m.key: 'scenario.npc_gapm' is not a key of the config (did you mean 'scenario.npc_gap_m'?)
```

Errors (exit status 1; the config is not compiled): a control that is not a key
or a mapping, an unknown field, a missing or unknown `key` (it must be a key
of the composed config), a `target` that is neither `setting` nor `measure`,
`scale`/`offset`/`integer`/`values` under `target: measure`, a `range` or
`limits` that is not `[low, high]` with `low <= high`.

A measure name that is not a built-in one is only a **warning**
(`controls.vehicle_ahead_gap: not a built-in measure (did you mean
'vehicle_ahead_gap_m'?)`): a scenario may register measures of its own at run
time (`register_measure`), which a check of the config cannot see, so it
cannot tell a typo from such a measure.

`{measure: key}` is short for `{measure: {key: key}}`. The ODD's attribute
and the measure share a unit: the binding takes the measure's.

#### Raw settings and measure-targeted controls

A raw setting (`target: setting`) is a scenario parameter the drawn value is
mapped onto by `scale` and `offset`, and the mapping is the author's claim
that the parameter brings the measure about. It holds where the two are the
same quantity by definition. It is a guess where they are not:
`npc_ahead_m` is the NPC's `s` along its lanelet less the ego's along its
own, while `vehicle_ahead_gap_m` is bumper to bumper along the ego's lane.
Neighbouring lanelets do not run alike on a curve -- the outer one is longer,
and their `s` do not line up -- and the bumpers depend on the vehicles, so no
constant offset is right wherever the scenario is placed (on CARLA 0.10
Town04_Opt one guessed +11 m was measured 2.2 m off, and the right offset
there is right nowhere else).

A measure that the **initial placement alone determines** can be the
parameter instead. The scenario takes "the measure should read *v* at the
start" and solves for the placement that makes it so, evaluating the
measure's own definition on the poses it will spawn at; the control says
`target: measure`, and the value drawn is written unchanged:

```python
from autoware_carla_scenario import place_at_vehicle_ahead_gap, vehicle_bumpers

# The NPC's s on its lanelet at which vehicle_ahead_gap_m reads cfg.npc_gap_m,
# seen from where the ego spawns -- whatever the two lanelets' shapes.
ego_at = to_carla_world(self._spawn_pose).to_carla_transform()
npc_pose = place_at_vehicle_ahead_gap(
    ego_at,
    cfg.npc_lanelet_id,
    cfg.npc_gap_m,
    ego_bumpers=vehicle_bumpers(world, self.ego_config.vehicle_type),
    npc_bumpers=vehicle_bumpers(world, cfg.npc_vehicle_type),
)
```

`place_at_vehicle_ahead_gap` reads the gap with the measure's own code
(`measures.vehicle_ahead_gap_between`, which the measure reads every vehicle
with) for the NPC at each `s` of its lanelet, as a `Lanelet2Pose` spawns
there -- on the centreline, heading along it -- and narrows the first
crossing of *v* down by bisection (`solve_along`, a 1-D root find that takes
any `s -> measure` function, for a measure of a scenario's own). It solves
on the NPC's own lanelet and projects the NPC onto the ego's lane, so it does
not matter where the lanelets beside each other start or how long they are.
A gap the lanelet cannot reach -- too short, turning away, or out of the
lanes the measure counts -- is refused with the gaps it does reach, so a
drawn case that cannot be staged fails at setup rather than running another
gap.

The gap is bumper to bumper, so the placement needs both vehicles' sizes
before either is spawned. CARLA gives a blueprint no size: `vehicle_bumpers`
reads the bounding box of a vehicle of the blueprint already in the world
when there is one, and otherwise spawns one high above the map (higher again
if that spawn is refused), reads its bounding box and destroys it before the
world ticks again -- once per blueprint per process. The vehicle-ahead
measures do not count a vehicle that far above the ego. The NPC is then spawned once, where the gap reads right, rather
than spawned, measured and moved -- a move that would race its initial
speed and the first tick.

Which to use:

| The measure is | Control |
|---|---|
| decided by where and how road users start, and the parameter is that measure by definition (`npc_initial_speed_kmh` less the ego's is the relative speed) | a raw setting, with the `offset` that makes it the measure |
| decided by where road users start, but the parameter is another quantity (an arc length for a gap) | `target: measure`, with the scenario solving its placement |
| decided by what the run then drives (a pedestrian's gap when it sets off, anything a controller or traffic changes) | a raw setting; how far the measure drifts from what was drawn is read off the runs, in openodd's case plot (its drift/calibration view), not solved at setup |

The relative speed of `cut_in` is the first kind: both vehicles start at
their speeds along their lanes, and the measure is the NPC's velocity along
the ego's lane at the NPC less the ego's along it at the ego, so its
`offset: ${ego.initial_speed_kmh}` writes it exactly: on a bend the NPC
heads along its lane, which runs alongside the ego's lane where it is. (A
lane beside a bend is shorter or longer than the ego's, so an NPC on it
covers the ego's lane a few per cent slower or faster -- 3.5 m off a 100 m
radius, 3.5 % -- which the measure does not count: it reads speeds, not how
fast `s` changes.)

The weather and the sun are set for every scenario, through the
`environment` config, which sets them before the run starts
(`EnvironmentAction`); the built-in probes reading them have controls of
their own:

| Probe | Key | Ranges |
|---|---|---|
| `rain` | `environment.rain` | `none` 0, `light` 1-30, `moderate` 30-70, `heavy` 70-100 |
| `fog` | `environment.fog_density` | as `rain` |
| `illumination` | `environment.sun_altitude_angle` | `day` 15-90, `low_sun` 0-15, `twilight` -6-0, `night` -90 to -6 |
| `visibility` | `environment.visibility_m` | 15-400 m, whatever the weather |
| `road_wetness` | `environment.wetness` | as `rain` |
| `tire_friction` | `environment.tire_friction` | 0.3-1.1 of the dry road's grip, whatever the weather |
| `wind` | `environment.wind_intensity` | as `rain` |
| `cloudiness` | `environment.cloudiness` | as `rain` |

`road_wetness` reads the weather's `wetness` alone, which its knob sets, so a
drawn value reads back as its bucket; the standing water `rain` brings is the
rain's, and the grip follows the larger of the two. `tire_friction` reads the
grip the run's weather left the vehicles' wheels (a share of the dry
road's, 1.0) -- `None` when nothing set it -- and its knob names it outright, as
`visibility`'s names the visibility. `wind` and `cloudiness` read CARLA's
0-100 intensities as levels, as `rain` and `fog` do.

`environment.rain` sets the precipitation and, with it, the cloudiness,
standing water and wetness of CARLA's rain presets (`EnvironmentAction`'s
`rain`): precipitation alone draws only the falling drops, and a camera sees a
dry road under a clear sky. The probe reads the precipitation, so a drawn value
still reads back as its bucket. `rain` and `precipitation` cannot both be set
(the run fails at its start), and the `none` bucket, `rain=0`, is a dry road
under a clear sky whatever the map's own weather.

Fog is drawn as `fog_density` alone. How thick that looks is the server's: a
CARLA 0.10 server on the UE 5.8 engine (the `ue58-dev` tree) multiplies it by `carla.Weather.FogDensityScale`,
whose default 0.001 leaves even `fog_density=100` a faint haze on the horizon.
Start it with `-ExecCmds="carla.Weather.FogDensityScale 0.01"` for fog a
camera sees, or set `server.fog_density_scale=0.01`
([LiDAR](driver_interface.md#lidar)).

The `rain` and `fog` knobs set the LiDAR noise as well: each `driver.lidars`
sweep carries the world's precipitation and fog density
([LiDAR](driver_interface.md#lidar)).

A scenario that sets the weather itself (its own `EnvironmentAction`) has
the last word over the config.

### Tire friction follows the weather

CARLA draws rain but does not drive on it: a vehicle grips a soaked road as
it grips a dry one. So an `EnvironmentAction` that sets the road's water --
`rain`, `wetness` or `precipitation_deposits` -- also scales every vehicle's
wheel friction (`friction_force_multiplier`) to a share of the dry road's
grip, within the range the resulting weather leaves
(`tire_friction_range`). The default [weather rules](#weather-rules) give it
by the larger of `wetness` and `precipitation_deposits`:

| Road water | Grip, share of the dry road's | What `rain` gives it |
|---|---|---|
| 0 (dry) | 0.90-1.10 | `rain=0` |
| 50 (wet) | 0.65-0.85 | `rain=30` (standing water 50) |
| 100 (flooded) | 0.45-0.65 | `rain=100` |

Between the rows the bounds are interpolated (the rules' terms overlap so
that they read as this table). Measured tire-road friction
is about 0.8-1.0 on dry asphalt, 0.5-0.7 on wet and 0.3-0.5 under a water
film; each range spans what tires, their wear and the asphalt make of the
same water, and a dry road's is centred on CARLA's own grip, so a dry run
drives as it did before.

Where in the range a run's grip is, is `environment.tire_friction_quantile`
(0 its low end, 1 its high end; the middle when not set). A
`sweep.odd_sample` draws it for every case (`environment.tire_friction_quantile=0.412`
among the case's overrides, recorded in its coverage file), from the sweep's
`seed` and the case, so the same sweep draws the same grip. The weather comes
from the ODD and the grip from the range that weather allows: a heavy-rain
case never runs on a dry road's grip. `tire_friction: false` in the
`odd_sample` leaves every case at the middle of its range, and
`environment.tire_friction` fixes the grip whatever the weather (it wins over
the quantile, and a case that names one draws no quantile; the ODD's
`tire_friction` probe draws it through that knob).

The vehicles in the world when the action fires are scaled then; a scenario
with such an action -- the config's, or its own -- also runs
`follow_tire_friction` after every tick, which gives the vehicles spawned
later (background traffic) the same grip. A scenario that sets the weather on
the tick loop sets the grip with it: a factor applied again replaces the last
one rather than compounding it.

The grip is CARLA's vehicle physics alone: no traffic backend reads it.

The example scenarios declare these:

| Scenario | Measure | Parameter |
|---|---|---|
| `cut_in/*` | `vehicle_ahead_gap_m` | `npc_gap_m` (`target: measure`), 2-30 m |
| `cut_in/*` | `vehicle_ahead_relative_speed_kph` | `npc_initial_speed_kmh` less the ego's, -15 to 15 km/h |
| `pedestrian_dart_out` | `crossing_pedestrian_gap_m` | `trigger_distance_m`, 8-30 m |
| `pedestrian_dart_out` | `crossing_pedestrian_speed_ms` | `walk_speed_ms`, 0.8-3 m/s |

`traffic_light_compliance` declares none, but has what a control needs:
`stop_line_distance_m` (`target: measure`: the ego is moved along the lane its
spawn pose is on until `stop_line_distance_m` reads it,
`place_at_stop_line_distance`), and its own measure `red_light_duration_s`,
the seconds every light is held red (`light_switch_delay_seconds`, a raw
setting). With the delay drawn, `timeout_after_green_seconds` counts the
fail-safe timeout from the green light instead of from the start. A drawn
distance does not decide the verdict by itself: the run is judged by where
the ego stops against that same line -- not past it on red, standing in the
stop zone (`stop_zone_min_m`-`stop_zone_max_m` before it) by the green -- so
an ego started far back may roll up to the line on red.

`sweep.odd_sample.knobs` takes the same fields keyed by the ODD's attributes,
and replaces anything above, attribute by attribute, for that sweep. It is for
the command line -- one sweep against one ODD -- not for a scenario's config,
which would then name the ODD's taxonomy: a scenario says `controls`. Controls set where a run starts;
the measures read what it then drives, which is what coverage counts.

### Visibility follows the weather

CARLA draws the weather but its traffic never looks through it: the
TrafficManager's vehicles know where every actor is, fog or not. A driver
does not, and drives so as to stop within the road they see. So how far
drivers see is a quantity of the run, in metres, which the weather bounds
(`visibility_range`, in `autoware_carla_scenario.visibility`). By the default
[weather rules](#weather-rules), each of the fog, the rain and the light
gives a range, and the shortest holds.

| Weather | Visibility, m |
|---|---|
| clear day | 250-400 |
| `fog_density` 30 / 70 / 100 | 120-250 / 40-100 / 15-40 |
| `precipitation` 30 / 70 / 100 | 180-320 / 100-220 / 60-150 |
| night (`sun_altitude_angle` -6 and below) | 50-150 (a dipped beam to a main beam or a lit street) |

Between the rows the bounds are interpolated; through civil twilight (0 to
-6 degrees) the day's range shortens to the night's. An `EnvironmentAction`
that sets the fog, the rain (`rain` or `precipitation`) or the sun records
the visibility of the weather it leaves for its world, at
`environment.visibility_quantile` in that range (0 its low end, 1 its high
end; the middle when not set); `environment.visibility_m` fixes it whatever
the weather. A `sweep.odd_sample` draws the quantile for every case, as it
draws the grip, so the ODD's `rain`, `fog` and `illumination` knobs are what
shorten it: a night case's drivers see 50-150 m, a heavy-fog case's 15-40 m.
`visibility: false` in the `odd_sample` leaves every case at the middle of
its range. An ODD that names the visibility itself (the `visibility` probe)
sets it through its own knob, `environment.visibility_m`, and a case with one
keeps it. A run that sets no weather records none, and its traffic sees as
it always did; each run starts with none, whatever the last left.

What follows it is the traffic backend's (`traffic.vehicles.follow_visibility`):
the `sumo` backend holds its own
traffic to the speed it stops from within it, by default
([Visibility](traffic_backends.md#the-sumo-backend)); the `traffic_manager`
backend holds every vehicle it drives to it with `follow_visibility: true`,
under its road's speed limit
([Weather and the TrafficManager](traffic_backends.md#weather-and-the-trafficmanager)),
and otherwise only logs the speed it would hold traffic to.

### Weather rules

The ranges of the grip and of the visibility are fuzzy rules, not code:
`weather_rules.yaml` in the package (`autoware_carla_scenario.weather_rules`),
the CARLA backend's own rules (`CarlaBackend.rules`), which openodd reads and
infers in its Rust core (Sugeno, zero order;
[its rules docs](https://github.com/hakuturu583/openodd_py/blob/main/docs/rules.md)
have the file format in full). A **quantity** (`tire_friction`, `visibility`)
is one or more **rule bases**; each reads some of the weather's fields, sorts
each into fuzzy terms, and its rules give the `low` and `high` ends of the
range per combination of terms:

```yaml
version: 1
quantities:
  tire_friction:
    rule_bases:
      water:
        inputs:
          water:
            universe: [0, 100]
            terms:
              dry: {trapezoid: [0, 0, 0, 50]}
              wet: {triangle: [0, 50, 100]}
              flooded: {trapezoid: [50, 100, 100, 100]}
        rules:
          - if: (water IS dry)
            then: {low: 0.90, high: 1.10}
          - if: (water IS wet)
            then: {low: 0.65, high: 0.85}
          - if: (water IS flooded)
            then: {low: 0.45, high: 0.65}
```

- A rule base's range is its rules' ends, averaged by how strongly each rule
  fires. Triangles that peak where their neighbours end split every value
  between two terms, so one-term rules read linearly between the peaks: the
  default rules give exactly the tables above.
- An antecedent is clauses `(input IS term)` joined by `AND`, `OR` and
  `NOT`, so a rule may read more than one input --
  `(water IS wet) AND (sun_altitude_angle IS night)`.
- The inputs a rule base may read are the weather's `cloudiness`,
  `precipitation`, `precipitation_deposits`, `wetness`, `wind_intensity`,
  `fog_density`, `fog_distance`, `sun_altitude_angle` and
  `sun_azimuth_angle`, and `water`, the larger of `wetness` and
  `precipitation_deposits` (`weather_rules.WEATHER_INPUTS`). A value outside
  an input's `universe` is clamped to it. An `EnvironmentAction` hands the
  rules every field of the weather it leaves; `tire_friction_range()` and
  `visibility_range()` hand only their own arguments, so an input they do not
  take needs a `default`.
- With more than one rule base, `combine` (`min`, the default, or `max`)
  joins their ranges end by end: the visibility's fog, rain and light are
  three rule bases, and the shortest bound holds.

A rule file is checked when it is read: every term and rule well formed,
every input's universe covered by its terms, and some rule firing wherever
the inputs can be (`openodd.RulesError` says where).

**A run's rules.** `environment.rules` names a rule file -- a path, absolute
or relative to the directory the run starts in, or an `https://` URL --
merged over the default rules: it names only what it changes (a rule base's
`rules` without its `inputs`, say), and may `extends` other files, by path,
URL (pinned with `{url, sha256}`) or git source. A sweep sets it like any
other key (`environment.rules=/data/grip_measured.yaml`); every run sets the
rules it names, or the default ones, so one run's do not reach the next.

**Rules by vehicle.** Tyres differ by vehicle; the weather does not:

```yaml
environment:
  rules:
    default: weather/measured.yaml          # every vehicle no entry names
    vehicles:                               # the first entry a vehicle matches
      ego: weather/ego_measured.yaml        # the ego (role_name "Ego")
      "vehicle.mini.*": weather/mini.yaml   # a blueprint pattern
```

Each vehicle's file is read over the run's default rules. An
`EnvironmentAction` that sets the road's water gives each vehicle the grip of
its own rules, all at the same `tire_friction_quantile` (a case draws one),
and a vehicle spawned later its own too; the ODD's `tire_friction` probe
reads the ego's. `tire_friction`, the grip named outright, is every
vehicle's. How far drivers see is the default rules'.

`weather_rules.weather_rule_files()` lists every rule file a run follows,
with the SHA-256 of each.

### Traffic parameters from the ODD

How the traffic behaves -- how fast drivers go against the limit, how they
react, how often they run a red light, how fast pedestrians walk -- is the
run's traffic parameters ([Traffic parameters](traffic_backends.md#traffic-parameters)).
Each numeric one is a spread, a mean and a standard deviation, that every
vehicle or pedestrian draws its own value from (a normal distribution cut to
the parameter's bounds). An ODD bounds the spread: the mean and the std of
each are built-in probes, with knobs, so an ODD that names them makes a sweep
draw both per case, and every road user of the case then draws its own value
from them.

| Probe | Knob (written) | Knob range |
|---|---|---|
| `vehicle_<field>_mean` | `traffic.vehicles.<field>.mean` | the field's mean range below |
| `vehicle_<field>_std` | `traffic.vehicles.<field>.std` | the field's std range below |
| `pedestrian_<field>_mean` | `traffic.pedestrians.<field>.mean` | likewise |
| `pedestrian_<field>_std` | `traffic.pedestrians.<field>.std` | likewise |
| `bicycle_<field>_mean` | `traffic.bicycles.<field>.mean` | likewise (`sumo` only) |
| `bicycle_<field>_std` | `traffic.bicycles.<field>.std` | likewise (`sumo` only) |

| Parameter | Unit | Mean range | Std range |
|---|---|---|---|
| `vehicles.speed_factor` | | 0.5-1.5 | 0-0.3 |
| `vehicles.min_gap_m` | m | 0.5-10 | 0-3 |
| `vehicles.ignore_lights`, `ignore_signs`, `ignore_vehicles`, `ignore_walkers`, `random_lane_changes` | | 0-1 | 0-0.5 |
| `vehicles.accel` | m/s^2 | 0.5-5 | 0-1.5 |
| `vehicles.decel` | m/s^2 | 1-9 | 0-2 |
| `vehicles.reaction_time_s` | s | 0.3-3 | 0-1 |
| `vehicles.imperfection` | | 0-1 | 0-0.5 |
| `vehicles.max_lateral_speed_mps` | m/s | 0.2-2 | 0-0.5 |
| `vehicles.max_lateral_accel` | m/s^2 | 0.2-3 | 0-0.5 |
| `pedestrians.walking_speed_mps` | m/s | 0.5-2.5 | 0-0.5 |
| `pedestrians.min_gap_m` | m | 0.1-2 | 0-0.5 |
| `pedestrians.red_light_violation_s` | s | 0-60 | 0-30 |
| `bicycles.max_speed_mps` | m/s | 2-8 | 0-1.5 |
| `bicycles.accel` | m/s^2 | 0.5-2 | 0-0.5 |
| `bicycles.decel` | m/s^2 | 1-4 | 0-1 |
| `bicycles.min_gap_m` | m | 0.2-2 | 0-0.5 |
| `bicycles.reaction_time_s` | s | 0.3-2 | 0-0.5 |
| `bicycles.imperfection` | | 0-1 | 0-0.3 |
| `bicycles.max_lateral_speed_mps` | m/s | 0.2-2 | 0-0.5 |
| `bicycles.red_light_violation_s` | s | 0-30 | 0-10 |

The bicycles' are `traffic.bicycles` (`BicycleParameters`,
[Traffic parameters](traffic_backends.md#traffic-parameters)): only SUMO
simulates bicycles, so an ODD that draws them is an ODD for the `sumo`
backend.

A bucket of the ODD's attribute narrows the range, as for any knob. In a
binding file:

```yaml
# binding.yaml
behaviour.speed_factor: {probe: vehicle_speed_factor_mean}
behaviour.speed_spread: {probe: vehicle_speed_factor_std}
behaviour.reaction_time: {probe: vehicle_reaction_time_s_mean}
behaviour.walking_speed: {probe: pedestrian_walking_speed_mps_mean}
```

A case drawn with `behaviour.speed_factor` in `[0.8, 0.9)` and
`behaviour.speed_spread` in `[0.05, 0.1)` writes
`traffic.vehicles.speed_factor.mean` and `.std`, and each vehicle of its run
draws its own speed factor from that normal distribution. The probes read
what the run set, every tick, so the run's coverage records the mean and std
it drove with; a run that set none reads a missing value.

A parameter the run's backend does not support is still refused: an ODD that
draws a reaction time is an ODD for the `sumo` backend, and a case of it run
on the `traffic_manager` backend fails before it starts
([Traffic parameters](traffic_backends.md#traffic-parameters)).

### The road a case is on

A sweep with cases of its own -- constraints, or a route search -- draws the
attributes the Lanelet2 map decides as well: whether the case is in a
junction, how many lanes, the speed limit, the location and road type
tags. Each is read off every case's lanelet (its first
`*lanelet_id` override: the constraints' pick, or the route's spawn) with the
probe's lanelet form, and gets a knob that reaches the buckets those lanelets
are in. The knobs write no setting a run reads (`_odd_map.*`, taken off
again); instead each drawn case is put on a case whose lanelet is in every
bucket drawn, the one run least so far among those that are. So a lanelet
case may run more than once, or not at all, even without a `count`: the
buckets decide, not the list of cases. Each run records its case
(`++odd_case`) with its own lanelet's buckets and values, not the ones
drawn.

Not every combination of buckets is on one lanelet -- no junction where the
limit is 30, say -- and each knob draws its attribute on its own, so 20
cases are drawn for each one kept and those no lanelet fits are passed over.
If too few fit, the rest go on the lanelet in most of their buckets, with a
warning. So `strategy: coverage` aims at the road as at everything else: the
buckets of junction, lane count or speed limit that earlier runs left uncovered
come first, as far as the sweep's cases reach them.

A sweep that gives an attribute a knob itself (`odd_sample.knobs`) keeps it.
The flaky search (`odd_sample.flaky`) draws its cases as it chooses them, none
passed over; each goes on the lanelet in most of its buckets.

### How a case is drawn

1. Each knob's attribute gets one of its buckets, leaving out the buckets
   outside the ODD (the ones `scenario-odd show` lists as out).
2. The combination is kept only if the ODD's verdict on those buckets is not
   *outside* -- so modules that rule out a combination (no heavy rain at
   night, say) are honoured, not just single buckets.
3. A value is drawn inside each bucket (a numeric bucket's interval, or the
   knob's range) and checked against the ODD again, as a value, with the
   attributes an OpenODD ODD derives from others worked out: a condition finer
   than a bucket (`between(0, 45)` across a `[30, 60)` bucket) holds for what
   is drawn. What is written is the value exactly as drawn, and a numeric one
   is read back to check it is still in its bucket.

`strategy: uniform` draws every admitted bucket alike. `strategy: coverage`
reads `coverage_from` and draws each attribute from its **least covered**
buckets first, counting the cases already drawn in the batch -- by earlier
scenarios on the same ODD too, in a glob batch -- as covered, so a batch
works through the holes before it repeats one; it falls back to any
bucket (the less covered the likelier) only when the ODD admits no
combination of the least covered. **Situations** that are still holes (or
that no earlier run measured) are aimed at first, one case each; inactive
ones are not, nor are those an earlier case of the batch was aimed at. A
situation the knobs alone decide -- every attribute it tests has a knob, and it refers to no other module -- holds in its case. One that also tests what no knob sets -- a speed, a road -- gets a case
under which it *can* hold; whether it does is up to the drive. A situation
the ODD and the knobs leave no case for is given up on, with a warning.

`coverage_from` paths are relative to the directory the sweep is started
from. Coverage is read when the sweep is expanded: to resume a sweep
(`sweep.resume_from`) into the same cases, read the same coverage -- point
`coverage_from` at the earlier runs, not at the sweep's own output.

From Python:

```python
from pathlib import Path

from autoware_carla_scenario.coverage.report import load_and_merge
from autoware_carla_scenario.odd import OddKnob, OddSampler, resolve_odd

sampler = OddSampler(
    resolve_odd("path/to/urban.yaml"),
    controls={"vehicle_ahead_gap_m": OddKnob("scenario.npc_gap_m", range=(2.0, 30.0))},
    seed=0,
    strategy="coverage",
    coverage=load_and_merge([Path("outputs")]),
)
for case in sampler.sample(20):
    print(case.overrides, case.buckets, case.situation)
```

To go on from cases drawn before -- another scenario's, on the same ODD --
pass them as `drawn`; `OddDraws` keeps them per ODD, as a batch does:

```python
from autoware_carla_scenario.odd import OddDraws

draws = OddDraws()
draws.add(sampler.odd, sampler.sample(20))
later = OddSampler(sampler.odd, seed=0, strategy="coverage", drawn=draws.of(sampler.odd))
```

### Sampling by shares of the ODD

`weights` gives chosen parts of the ODD a share of the cases, and draws the
rest as `strategy: uniform` would (it implies `strategy: weighted`). The
parts are the ODD's **modules**, named in its OpenODD files, of one condition
or of several -- so the sweep names no attribute of the taxonomy, and a module
redefined in the ODD needs no change to the sweep:

```yaml
# regions.yaml: parts of the ODD, in plain OpenODD
IMPORT: [urban.yaml]
MODULES:
  heavy_rain:
    INCLUDE_AND:
      environment.rain: heavy
  rainy_night_merge:
    INCLUDE_AND:
      environment.rain: heavy
      environment.illumination: night
      scenery.road_type: merge
```

```yaml
# shares/rainy.yaml
heavy_rain: 0.3            # 30% of the cases in heavy rain
rainy_night_merge: 0.1     # 10% in heavy rain at night in a merge
```

```bash
uv run scenario --multirun 'scenario=cut_in/*' odd=path/to/urban.binding.yaml \
  +sweep.odd_sample.count=50 +sweep.odd_sample.weights=shares/rainy.yaml
# or inline, for one module
  '+sweep.odd_sample.weights={heavy_rain: 0.5}'
```

The binding file names the regions file with the ODD's
(`openodd: [urban.yaml, regions.yaml]`), and the ODD names its roots under
`ODD:`: a module that is a root -- a document with no `ODD:` section takes
every module that is not a situation as one -- would bound the ODD, and is
refused a share (every case is in it). A file of shares is relative to the
directory the sweep is started from.

Each module holds its share of the cases, however the modules overlap: with
`heavy_rain: 0.3` and `night: 0.3`, 30% of the cases are in heavy rain and
30% at night, those in both counting for both. A share is a fraction in
`[0, 1]`, not a weight relative to the others. Inside and outside the modules
the cases are drawn as the uniform strategy draws them (of all the ways to
hold the shares, the one closest to it), rounded to whole cases counting the
cases drawn before -- by the earlier scenarios on the ODD in a glob batch too,
so ten scenarios drawing 1 case each with `heavy_rain: 0.3` draw 3 in heavy
rain between them.

Refused: a name that is no module, a root, a share outside `[0, 1]`,
`weights` with another strategy; a module whose truth the knobs do not
decide (it tests an attribute nothing draws, such as the ego's speed); shares
that cannot hold together inside the ODD (`a: 0.7` and `b: 0.6` for modules
that never overlap). With `flaky`, the part of each batch the search does not
explore holds the shares. A sweep with `constraints` whose map decides some
attributes keeps, of more cases drawn, those on a lanelet of the sweep, so
its cases hold the shares only roughly. The shares are the sampler's, not OpenODD's, which
has no construct for a distribution (openodd's
[weighted sampling](https://github.com/hakuturu583/openodd_py/blob/main/docs/weighted_sampling.md)).

### Finding flaky cases

Every run a sweep draws from the ODD records its case in its coverage file:
the values it ran with, why it was drawn, and whether it passed (openodd's
`CoverageCollector.record_case` and `record_outcome`; the runner calls them,
so a scenario needs nothing of its own). The merged report then lists the
runs, flakiest first, with their flakiness assessed over all the runs of
their scenario (`scenario-coverage`, section *Flaky cases*).

With `flaky`, the sweep also looks for them. openodd's `FlakySearch` learns
where the runs in `coverage_from` passed and failed (Gaussian-process
classification), and draws part of each batch where a small change would flip
the outcome -- after the sampler has covered every bucket, the rest of the
batch still sweeping the ODD (openodd's
[docs/flaky.md](https://github.com/hakuturu583/openodd_py/blob/main/docs/flaky.md)):

```yaml
sweep:
  odd_sample:
    count: 20
    strategy: coverage
    coverage_from: [outputs]       # the earlier sweeps' runs: what the search learns from
    flaky:
      explore_fraction: 0.8        # of each batch, once every bucket is covered
      scenario: CutInScenario      # learn from this scenario's runs only (all, when left out)
```

`flaky: true` takes the defaults. Its other fields are `FlakySearch`'s:
`pool`, `coverage_depth`, `threshold`, `epsilon`, `perturbations`, `draws`
and `directions`. Only runs drawn in the same space -- the same attributes,
ranges and labels -- with an outcome are learnt from; a sweep with no such
run draws all its cases with the sampler's own policy. Each case chosen for
its instability records the flakiness predicted for it, beside the one the
report assesses. Run the sweeps one after another, each reading the earlier
ones' outputs, and every sweep looks where the last left the outcome unsure.

## Checking an ODD

```bash
uv run scenario-odd check my_package.odds:urban_odd   # compile with Codon
uv run scenario-odd check path/to/urban.yaml          # read the binding and OpenODD files
uv run scenario-odd show path/to/urban.yaml           # attributes, buckets in and out
uv run scenario-odd list                              # ODDs known by name
```

`check` compiles a Python ODD with Codon, the same way a scenario is
compiled. It catches the following:

- a misspelt name;
- a condition of the wrong type (`between("0", 60)`);
- a probe that does not take a world.

The runner makes the same check before a run, under the `typecheck` config
key. OpenODD YAML is checked as it is read (see [What is read](#what-is-read)).

## Checking a planned route

Before a scenario runs, its ego route can be checked against the ODD on the
Lanelet2 map alone, with no CARLA server:

```bash
uv run scenario-odd route path/to/urban.yaml intersection_passing/left_turn
uv run scenario-odd route urban 'intersection_passing/*' lane_change/left map=nishishinjuku
uv run scenario-odd route urban 'lane_change/*' --json
```

Each scenario config is composed as a run composes it. Name it as
`scenario=` does (a glob works too); arguments with an `=` are Hydra
overrides for all of them. The route is planned from the ego's spawn pose
(`ego.spawn_lanelet_id`, `ego.spawn_s`), through `ego.waypoint_lanelet_ids`,
to the goal (`ego.goal_lanelet_id`, `ego.goal_s`), as the shortest path in
the map's routing graph, leg by leg. A scenario that derives its goal in
`setup()` from `scenario.expected_route_lanelet_ids`, as
`intersection_passing` does, gets that goal here too (a configured goal
wins). A goal derived any other way is only known once the scenario runs:
that route is reported as not planned.

The ODD's attributes are then read lanelet by lanelet along the route, and
each lanelet gets one of three verdicts:

| Verdict | Meaning |
|---|---|
| inside | The ODD holds there, whatever happens at run time. |
| outside | The ODD fails there, whatever happens at run time. Reported with the lanelet ids and the modules that fail. |
| undecided | It depends on what only the run can tell (weather, traffic, speed), or on a value the map does not have. At run time a missing value counts as inside, so this is "inside only by assumption". |

The map decides an attribute when its probe has a lanelet form,
`probe.on_lanelet(lanelet, lanelet_map, routing_graph)`. It returns the
value, `None` where the run would read none, or `UNDECIDED` where only the
run can tell (for example `speed_limit_kph` on a lanelet with no
`speed_limit` tag, where the run falls back to CARLA's limit). The built-in
probes listed above have one; a custom probe opts in by having an
`on_lanelet` attribute (a function attribute, or a method of a callable
object). An attribute whose probe has none is **open**: it could take any
value, and the ODD's own three-valued evaluation decides, exactly as it does
for buckets outside the ODD. An OpenODD attribute derived from others is
worked out from their values on the lanelet, and is open while one of them
is.

What the lanelet forms read, and where that differs from the run:

| Probe | On a lanelet | At run time |
|---|---|---|
| `in_junction` | The lanelet has a `turn_direction` tag (Autoware tags every lanelet in an intersection with one; the sweeper's `is_junction` reads it the same way) | CARLA waypoint `is_junction` |
| `lane_count` | The lanelet and every same-direction neighbour, walked left and right in the routing graph (`left`/`right`, else `adjacentLeft`/`adjacentRight`, so lanes behind a solid line count too); nothing in a junction | CARLA OpenDRIVE waypoints: driving lanes of the same sign |
| `speed_limit_kph` | The `speed_limit` tag, else (no tag, or not a number) undecided | The tag, else CARLA's `get_speed_limit()` |
| `lanelet_location`, `lanelet_subtype`, `lanelet_speed_limit_kph` | The tag | The tag of the lanelet the ego is on |

Route lane counts come from Lanelet2 and run-time lane counts come from
CARLA's OpenDRIVE waypoints. The two maps are converted from each other but
not always lane for lane, so they can disagree; so can the two junction
criteria.

The report also gives the **expected coverage**: the metres the route spends
in each bucket of each attribute (`<undecided>` and `<missing>` collect the
metres where the map cannot or does not say). The start lanelet counts from
the spawn pose, the goal lanelet up to the goal pose, and a lane change
splits the lanelets' length between the two sides. Over several scenarios
the metres are summed, and the report lists the buckets of each map-decided
attribute that **no planned route reaches**, leaving out buckets outside the
ODD (they are not targets). An attribute the map decides in general but left
undecided on some lanelet of a route (metres in `<undecided>`) is reported as
**undetermined** instead: the run may reach any of its buckets there, so none
is called unreached.

The exit status is 0 when no route leaves the ODD, 1 when one does, and 2
when a route could not be planned (no goal, no path, a lanelet the map does
not have, no Lanelet2 map) or the ODD could not be read.

From Python:

```python
from autoware_carla_scenario.odd import (
    PlannedRoute, combine_route_coverage, plan_route_coverage,
)
from autoware_carla_scenario.coordinate import Lanelet2Pose

route = PlannedRoute(start=Lanelet2Pose(203, 25.0), goal=Lanelet2Pose(207, 10.0))
coverage = plan_route_coverage("urban", route, lanelet_map=lanelet_map)
coverage.leaves_odd, coverage.outside(), coverage.expected_m
combine_route_coverage("urban", [coverage, ...]).unreached
```

`plan_route_coverage` also takes a scenario (its spawn pose, waypoints and
goal) or a list of lanelet ids. Without `lanelet_map` it uses the map the
runner has loaded.

At start-up, once `setup()` has settled the goal, the runner plans the
route the same way and logs a warning when it leaves the ODD. It is a
warning only. It is skipped for an ODD with no modules, and anything that
keeps the route from being planned is logged at debug level; the run goes
on regardless.

## What the report says

For each ODD the merged runs were measured against, `scenario-coverage` says
the following ([Coverage](coverage.md#the-report)):

- how long the runs were inside it, inside only because values were missing,
  and outside it;
- which modules ruled ticks out, and how often, and how often their verdict
  rested on missing values;
- which runs left it, and when (the first intervals);
- which attributes are monitored but not covered.

A bucket is **outside the ODD** when the ODD fails for every value in it,
whatever the other attributes are. It is reported, but it is not a coverage
target. An example is `nonurban` under `location.is_in(["urban"])`.

To decide this, the ODD's own evaluation runs with the bucket in place of the
attribute's value and every other attribute left open. Labels, inactive
modules, groups and module references therefore count exactly as they do on
a tick. A condition that ties attributes together (`any_of` across two of
them, or a junction and a speed) rules out combinations, never a single
bucket. A bucket only partly inside (`[60, 90]` under `<= 60`) stays a
target.
