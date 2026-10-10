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
- **[ASAM OpenODD](https://www.asam.net/standards/detail/openodd/) YAML**: read
  into the same objects.

```
 OpenODD YAML ──(load_openodd)──┐
                                ├──▶ OddDefinition ──▶ coverage + inside/outside per tick
 Python (checked by Codon) ─────┘
```

## Concepts

The names follow OpenODD 1.0.

| Concept | What it is |
|---|---|
| **Attribute** | A leaf of the taxonomy, measured on every tick: a **probe** reads it from the world, and its **buckets** make it a cover item. |
| **Condition** | A test on one attribute (`speed_limit.between(0, 60)`), a group of conditions (`all_of`, `any_of`), or another module's verdict (`module_holds`). |
| **Module** | A named rule. It holds when its *include* conditions hold and its *exclude* conditions do not. |
| **Root** | The module whose verdict is the ODD's. Without one, the ODD holds when every active module does. |

Evaluation is three-valued. An attribute whose probe has nothing to say is
**unknown**, not false. Examples are a simulator without weather, or a run
without a Lanelet2 map. A tick whose verdict is unknown is counted apart:
neither inside nor outside the ODD.

The model is **permissive**: whatever no module rules out is inside the ODD.
OpenODD 1.0 does not standardise permissive and restrictive definitions.

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
        root="root",
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
| `all_of([...])` / `any_of([...])` | all of / any of the conditions hold |
| `module_holds(name, holds=True)` | the module (or label) named `name` holds, or does not |

Numbers are in the attribute's `unit`.

### Modules

`OddModule(name, *, include_and, include_or, exclude_and, exclude_or, labels,
active, text)`:

- The module holds when all of `include_and` hold, one of `include_or` holds,
  and neither all of `exclude_and` nor one of `exclude_or` does.
- A **label** holds when any module that declares it holds.
- An **inactive** module is evaluated and reported, but the ODD does not
  need it.

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

## Writing an ODD in OpenODD YAML

```yaml
ODD:
  name: urban
  root: root_odd
  text: Urban roads, fair weather

TAXONOMY:
  road:
    location: [urban, nonurban, private]
    speed_limit: int speed
    junction: boolean
  env:
    visibility: float distance
    weather:
      rain: [none, light, moderate, heavy]
      fog:
        severity:            # derived: the first literal whose conditions hold
          none:  {env.visibility: ">= 500 m"}
          light: {env.visibility: "[200 .. 500] m"}
          dense: {env.visibility: "< 200 m"}
  ego:
    speed: float velocity

COVERAGE:                    # probes and buckets (an extension of OpenODD)
  road.location: {probe: lanelet_location}
  road.speed_limit: {probe: speed_limit_kph}
  road.junction: {probe: in_junction}
  env.weather.rain: {probe: rain}
  ego.speed: {probe: ego_speed_kph, range: [0, 120], every: 10}

MODULES:
  urban_only:
    TITLE: Urban roads up to 60 km/h
    INCLUDE_AND:
      road.location: [urban]
      road.speed_limit: "<= 60 km/h"
  weather:
    LABEL: fair_weather
    EXCLUDE_OR:
      env.weather.rain: [heavy]
      env.weather.fog.severity: [dense]
  slow_at_junctions:
    INCLUDE_OR:
      road.junction: false
      AND:
        road.junction: true
        ego.speed: "<= 30 km/h"
  root_odd:
    INCLUDE_AND:
      urban_only: true
      fair_weather: true
      slow_at_junctions: true
```

### What is read

**`TAXONOMY`** leaves can be:

- `float <unit type>` or `int <unit type>`;
- `boolean`;
- `string`;
- a list of literals;
- a mapping of literals to conditions (a derived enumeration).

A reference to another record (`"vehicle/pose"`) is not followed.

**`MODULES`** keys:

- `INCLUDE_AND`, `INCLUDE_OR`, `EXCLUDE_AND`, `EXCLUDE_OR`;
- `TITLE`, `DESCRIPTION`;
- `LABEL` / `LABELS`;
- `ACTIVE`.

A section maps one of these to a value:

- a taxonomy path to an expression;
- a module or label name to `true`/`false`;
- `AND`/`OR` to a nested section.

**Expressions**:

| Expression | Meaning |
|---|---|
| `"[low .. high] unit"` | inclusive range |
| `">= x unit"`, `"> x"`, `"<= x"`, `"< x"`, `"== x"` | comparison; `>` excludes `x`, `>=` includes it |
| `[a, b]` | one of the literals |
| `literal`, `3`, `true` | equal to it |

A number with a unit is converted into the probe's unit, so `"<= 16.7 m/s"`
against a probe in km/h is `<= 60.12`. The units known are:

| Quantity | Units |
|---|---|
| Length | m, km, cm, mm |
| Speed | m/s, km/h (kph), mph |
| Time | s, ms, min, h |
| Angle | deg, rad |
| Precipitation rate | mm/h |
| Percentage | % |

**`COVERAGE`** and **`ODD`** are this framework's extensions, carrying what
measuring needs.

`COVERAGE` binds a taxonomy path to a probe, and gives its buckets:

- `probe`: a built-in probe name, or `package.module:function`.
- `unit`: the unit the probe returns. Built-in probes know their own.
- Buckets: `values`, `buckets`, or `range` with `every`.
- `text`: a description for the report.

When no buckets are given:

- an enumeration gets one bucket per literal, and a boolean gets two;
- a number gets buckets at the **thresholds the modules test it against**.
  `"<= 60 km/h"` makes `[-inf, 60)` and `[60, inf]`, so every boundary the ODD
  draws is tested from both sides. Each bucket holds its lower edge, so the
  value 60 itself lands in `[60, inf]`, which `<= 60` marks as outside. Give
  explicit `buckets` where the edge value matters.

An attribute with no probe is unknown on every tick, and is reported as
unmeasured.

`ODD` gives the name, the root module and a description.

Several files can be merged. For example, keep the taxonomy, the modules and
the bindings in one file each, and load them together:
`load_openodd("taxonomy.yaml", "odd.yaml")`.

OpenODD 1.0's normative YAML mapping was not available while this reader was
written. It follows the public examples of the standard's YAML. A document
that relies on more than the above may need adapting.

## Picking the ODD of a run

The `odd` config key, or `ScenarioQueue(odd=...)` /
`ScenarioRunner(odd=...)`, takes one of:

| Value | ODD |
|---|---|
| `default` | The built-in one ([Coverage](coverage.md#odd-coverage)): every attribute, no modules |
| a name | One registered with `register_odd(name, builder)` in code, or by a package through the `autoware_carla_scenario.odds` entry point group |
| `path/to/odd.yaml` | An OpenODD YAML file |
| `package.module:function` | A builder to import and call |

```bash
uv run scenario scenario=intersection_passing/left_turn odd=path/to/urban.yaml
uv run scenario scenario=intersection_passing/left_turn odd=my_package.odds:urban_odd
```

A package that ships its ODD registers the builder under the
`autoware_carla_scenario.odds` entry point group. A private taxonomy can then
plug in without the framework knowing it:

```toml
[project.entry-points."autoware_carla_scenario.odds"]
urban = "my_package.odds:urban_odd"
```

## Checking an ODD

```bash
uv run scenario-odd check my_package.odds:urban_odd   # compile with Codon
uv run scenario-odd check path/to/urban.yaml          # read the YAML
uv run scenario-odd show path/to/urban.yaml           # attributes, buckets in and out
uv run scenario-odd list                              # ODDs known by name
```

`check` compiles a Python ODD with Codon, the same way a scenario is
compiled. It catches the following:

- a misspelt name;
- a condition of the wrong type (`between("0", 60)`);
- a probe that does not take a world.

The runner makes the same check before a run, under the `typecheck` config
key. YAML is checked as it is read: a path that is not in the taxonomy, or a
unit that cannot be converted, is refused with the module it is in.

## What the report says

For each ODD the merged runs were measured against, `scenario-coverage` says
the following ([Coverage](coverage.md#the-report)):

- how long the runs were inside it, outside it, and unknown;
- which modules ruled ticks out, and how often;
- which runs left it, and when (the first intervals);
- which attributes are monitored but not covered.

A bucket that a module rules out on its own is **outside the ODD**. It is
reported, but it is not a coverage target. An example is `nonurban` under
`location.is_in(["urban"])`. Some conditions tie several attributes together
(`any_of` across two of them, or a junction and a speed). Such a condition
rules out combinations, never a single bucket, so it leaves the buckets as
targets.
