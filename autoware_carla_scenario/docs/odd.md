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
active, text)`. Give at most one of `include_and` / `include_or`, and at most
one of `exclude_and` / `exclude_or`.

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

Import them from `autoware_carla_scenario.odd` itself
(`from autoware_carla_scenario.odd import rain`). The Codon model that checks
an ODD has no `probes` submodule.


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
  lane_count: {probe: lane_count, values: [1, 2, 3, 4]}
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
| `probes` | Concept → `probe` (built-in name or `package.module:function`), `unit` (built-in probes know theirs), buckets (`values`, `buckets`, or `range` + `every`), `text` |

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
