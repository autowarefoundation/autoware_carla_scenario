# ODD

An **Operational Design Domain** (ODD) states the conditions a system is
built to drive in: urban roads, up to 60 km/h, no heavy rain, and so on. Every
run is measured against one. The runner samples the ODD's attributes on every
tick. Their buckets are the run's **ODD coverage**: which conditions it
reached (see [Coverage](coverage.md)). The ODD's modules say which ticks were
**outside** the ODD.

An ODD is written in one of two ways. Both build the same objects, so they
are measured, judged and reported the same way:

- **Python**: the canonical form. It is checked with Codon before it is used,
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

An ODD is a function that takes nothing and returns an `OddDefinition`:

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
| `traffic_density` | `TRAFFIC_DENSITY_LEVELS` | Other vehicles within `NEARBY_RADIUS_M` |
| `pedestrian_nearby` | bool | A walker within `NEARBY_RADIUS_M` |
The Lanelet2 probes need the run to have a Lanelet2 map (`map.lanelet2_path`).
Without one they return nothing.

The probes whose value the map alone decides (`speed_limit_kph`,
`lanelet_speed_limit_kph`, `lanelet_location`, `lanelet_subtype`,
`in_junction`, `lane_count`) can also read it off a lanelet, which is how a
planned route is checked before a run
([Checking a planned route](#checking-a-planned-route)).

Import them from `autoware_carla_scenario.odd` itself
(`from autoware_carla_scenario.odd import rain`). The Codon model that checks
an ODD has no `probes` submodule.

### Scenario measures

The road users a scenario stages -- a vehicle ahead in the next lane, a
pedestrian running out -- are not read by ODD probes: they are **measures**
of the scenario, under keys the framework fixes
(`autoware_carla_scenario.measures`). Every `BaseScenario` has these, read
generically in the ego's frame:

| Key | Unit | Reads |
|---|---|---|
| `vehicle_ahead_gap_m` | m | The nearest vehicle ahead in the ego's lane or a lane beside it, centre to centre along the ego's heading (up to `AHEAD_RANGE_M`); a vehicle about to cut in is measured from before it changes lanes |
| `vehicle_ahead_relative_speed_kph` | km/h | That vehicle's speed along the ego's heading less the ego's |
| `crossing_pedestrian_gap_m` | m | How far ahead the nearest pedestrian ahead that is moving was when it set off, held while it crosses |
| `crossing_pedestrian_speed_ms` | m/s | That pedestrian's speed |

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
so a scenario or a Python ODD naming them is checked like any other. The ODD names a measure key, never a scenario; the scenario names
its measures, never the ODD. The default ODD maps
`dynamic.vehicle_ahead_gap`, `dynamic.vehicle_ahead_relative_speed`,
`dynamic.crossing_pedestrian_gap` and `dynamic.crossing_pedestrian_speed`
onto the four built-in measures; the two of the vehicle ahead count stays of
1 s or more (`min_stay`), so a gap the ego only closes through is not
covered.



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
    rev: v1.2.0                    # a tag, a branch, or a commit
    path: taxonomy.yml             # a list for several files
  - git: git@example.com:odd/modules.git
    rev: 3f2a9c1e0b7d4a6c8e5f1a2b3c4d5e6f7a8b9c0d
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
  `$AUTOWARE_CARLA_SCENARIO_ODD_CACHE` (default
  `~/.cache/autoware_carla_scenario/openodd`). A full commit id checked out
  once is read from the cache without the network; a tag or a branch is
  fetched on every load, to learn which commit it names now. Pin a commit for
  runs that must be reproducible.
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
| `default` | The built-in one ([Coverage](coverage.md#odd-coverage)): every attribute, no modules |
| a name | One registered with `register_odd(name, builder)` in code, or by a package through the `autoware_carla_scenario.odds` entry point group |
| `path/to/urban.yaml` | A binding file (or an OpenODD file on its own, measuring nothing) |
| `package.module:function` | A builder to import and call |

```bash
uv run scenario scenario=intersection_passing/left_turn odd=path/to/urban.yaml
uv run scenario scenario=intersection_passing/left_turn odd=my_package.odds:urban_odd
```

A package that ships its ODD registers it through the
`autoware_carla_scenario.odds` entry point group. The entry point names a
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
way and runs them all in one queue ([Batch Execution](usage.md#running-every-sweep-case-in-a-batch));
each scenario draws its own, so two scenarios on one ODD and seed are given
the same settings:

```bash
uv run scenario --multirun 'scenario=cut_in/*' map=town10hd_opt \
  +sweep.odd_sample.count=20 +sweep.odd_sample.strategy=coverage
```

or in the scenario's YAML:

```yaml
sweep:
  odd_sample:
    count: 20              # cases; one per lanelet case when left out
    seed: 0                # same seed, ODD, knobs and coverage: same cases
    strategy: coverage     # uniform | coverage
    coverage_from: [outputs]   # earlier runs' *_coverage.json (files or directories)
    odd: null              # the ODD to draw from; the run's (`odd`) by default
    knobs: {}              # by ODD attribute, replacing the scenario's controls; see below
```

Each case is a list of Hydra overrides. A sweep that also has `constraints`
gives each drawn case the next lanelet case in turn; one without draws from
the ODD alone, and needs no map to expand.

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
    key: scenario.npc_ahead_m
    range: [6.0, 30.0]                 # what the scenario can stage
  vehicle_ahead_relative_speed_kph:
    key: scenario.npc_initial_speed_kmh
    offset: ${ego.initial_speed_kmh}   # its speed less the ego's
    range: [-15.0, 15.0]
    limits: [0.0, null]                # a speed: never below zero, however slow the ego
```

| Field | Meaning |
|---|---|
| `key` | The config key the value is written to |
| `range` | Numeric: `[low, high]` in the measure's unit, the values the scenario can bring about. A bucket is drawn only where it meets the range (by more than an edge), and only that part of it |
| `values` | Per bucket label. Categorical: the value to write, or `[low, high)` to draw it from (`[x, x]` is `x`); a bucket without one is not drawn. Numeric: a range to draw from instead of the bucket's interval -- which an unbounded bucket (`-inf`/`inf` edges) needs to be drawn at all |
| `scale`, `offset` | A numeric value drawn in the unit is written as `value * scale + offset` (`scale` not zero) |
| `integer` | Write an integer. A rounded numeric value that falls in another bucket is drawn again |
| `limits` | Numeric: `[low, high]` the **written** value must lie in, `null` for an open end -- what the key takes, whatever `offset` comes to. A bucket only values outside it would reach is not drawn |

`{measure: key}` is short for `{measure: {key: key}}`. The ODD's attribute
and the measure share a unit: the binding takes the measure's.

The weather and the sun are set for every scenario, through the
`environment` config, which sets them before the run starts
(`EnvironmentAction`); the built-in probes reading them have controls of
their own:

| Probe | Key | Ranges |
|---|---|---|
| `rain` | `environment.precipitation` | `none` 0, `light` 1-30, `moderate` 30-70, `heavy` 70-100 |
| `fog` | `environment.fog_density` | as `rain` |
| `illumination` | `environment.sun_altitude_angle` | `day` 15-90, `low_sun` 0-15, `twilight` -6-0, `night` -90 to -6 |

A scenario that sets the weather itself (its own `EnvironmentAction`) has
the last word over the config.

The example scenarios declare these:

| Scenario | Measure | Parameter |
|---|---|---|
| `cut_in/*` | `vehicle_ahead_gap_m` | `npc_ahead_m`, 6-30 m |
| `cut_in/*` | `vehicle_ahead_relative_speed_kph` | `npc_initial_speed_kmh` less the ego's, -15 to 15 km/h |
| `pedestrian_dart_out` | `crossing_pedestrian_gap_m` | `trigger_distance_m`, 8-30 m |
| `pedestrian_dart_out` | `crossing_pedestrian_speed_ms` | `walk_speed_ms`, 0.8-3 m/s |

`sweep.odd_sample.knobs` takes the same fields keyed by the ODD's attributes,
and replaces anything above, attribute by attribute, for that sweep. It is for
the command line -- one sweep against one ODD -- not for a scenario's config,
which would then name the ODD's taxonomy: a scenario says `controls`. Controls set where a run starts;
the measures read what it then drives, which is what coverage counts.

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
buckets first, counting the cases already drawn in the batch as covered, so
a batch works through the holes before it repeats one; it falls back to any
bucket (the less covered the likelier) only when the ODD admits no
combination of the least covered. **Situations** that are still holes (or
that no earlier run measured) are aimed at first, one case each; inactive
ones are not. A situation the knobs alone decide -- every attribute it tests
has a knob, and it refers to no other module -- holds in its case. One that also tests what no knob sets -- a speed, a road -- gets a case
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
    controls={"vehicle_ahead_gap_m": OddKnob("scenario.npc_ahead_m", range=(6.0, 30.0))},
    seed=0,
    strategy="coverage",
    coverage=load_and_merge([Path("outputs")]),
)
for case in sampler.sample(20):
    print(case.overrides, case.buckets, case.situation)
```

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
