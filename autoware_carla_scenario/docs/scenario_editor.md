# Scenario Editor

The Scenario Editor is a web UI for authoring scenarios declaratively and
exporting them as reproducible **Scenario Packages**. It runs as its own
application:

```bash
uv run scenario-editor          # http://localhost:9100
```

It is deliberately **separate from the Scenario Result Viewer**
(`uv run viewer`). Neither mounts the other, and running one does not start the
other; integrating the two is a later phase.

## What it edits

The editor never edits Python. Its canonical representation is the **Scenario
IR** -- a `ScenarioDocument` stored as YAML -- which the framework compiles into
the *existing* runtime primitives:

```
ScenarioDocument -> ScenarioCompiler -> BaseAction / BaseCondition -> BaseScenario
```

There is no editor-only runtime. Every action a document names is built by
`autoware_carla_scenario.actions`, every condition by
`autoware_carla_scenario.conditions`, and every spawn constraint is evaluated by
the same `sweeper.constraints` engine a `--multirun` sweep uses.

## The canvas

The main view is a **swimlane DAG**: one lane per actor, with the horizontal
axis reading as *scenario progression*, not time. Actions that act on the
environment rather than a vehicle get a **World** lane of their own.

```
Ego
[Spawn] ------------ Drive -----------------------------
NPC1
[Spawn] - Follow ---------------- <> Lane Change Left
                                       ^
                                 +-----+-----+
                                 |    ALL    |
                            NPC1 -> Ego   NPC1 -> Ego
                            Distance      TTC
                            < 20 m        < 4 s
PASS
  NPC1 enters the ego lane
FAIL
  Collision
  Timeout 30 s
```

Two rules make it readable:

* **The distance between cards means nothing.** Column position is
  `ui.column_hint`, which is presentation only and never reaches the runtime.
* **There is no separate event lane.** A condition is a *trigger*, drawn under
  the action it fires, so cause and effect are next to each other instead of
  being correlated across the screen.

Every condition reads as `subject -> target | metric | rule value`, e.g.
`NPC1 -> Ego | Distance | < 20 m` or `Ego -> Lanelet 183 | Position | inside`.
Compose them with `ALL`, `ANY`, `NOT`, `Sticky` and `Persistent`, which map onto
`AndCondition`, `OrCondition`, `NotCondition`, `StickyCondition` and
`PersistentCondition`.

## Entity spawn

An entity spawns in one of two modes.

**Fixed** pins a lanelet and an offset:

| Field | Value |
| --- | --- |
| Spawn mode | Fixed |
| Lanelet ID | 183 |
| Offset | 12.5 m |

**Constraint search** hands the lanelet choice to the existing
lanelet-constraint sweeper. The constraint tree is edited as a tree and is
serialised straight into `sweep.constraints`:

```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: and
        constraints:
          - type: has_adjacent
            value: left
          - type: lanelet_length
            rule: greater_than_or_equal
            value: 10.0
          - type: not
            constraint: {type: is_junction}
          - type: not
            constraint:
              type: in_set
              values: ${map.no_3d_model_lanelet_ids}
```

**Preview matches** evaluates that tree against the real Lanelet2 map and
reports, for example, `133 matched of 979 lanelets`. Loading a map is opt-in and
cached; if the map files are missing the count is unavailable but the constraint
editor keeps working.

### The map

The drawing comes from
[`simple_lanelet2`](https://github.com/hakuturu583/simple_lanelet2)'s wasm map
viewer — the same project that provides the `lanelet2` Python API this framework
runs on. The editor serves the scenario's `.osm` at
`/draft/<id>/map.osm`, the viewer parses and renders it in the browser, and the
editor drives it through two calls: `setHighlight()` with the matched IDs, and a
`select` listener that turns **clicking a lanelet into setting the spawn
lanelet**.

Constraint evaluation stays on the server, in the framework's own sweeper — the
viewer only draws. That split is deliberate: a second constraint engine in
JavaScript could disagree with the sweep the scenario will actually run.

The module is loaded from the project's GitHub Pages build, because the wasm is
built rather than committed. It is optional: when it cannot be fetched the panel
falls back to a server-rendered SVG and the match count is unaffected, so an
air-gapped editor still works. Point `SCENARIO_EDITOR_MAP_VIEWER` at the output
of `simple_lanelet2`'s `tools/build_web.sh` to serve it yourself.

### Derived offsets

The longitudinal offset can be **Fixed** or **Derived** from the map. A derived
offset is a `sweep.bindings` entry -- currently `StopLineOffsetBinding`:

| Field | Value |
| --- | --- |
| Position on matched lanelet | Derived |
| Derived from | Before stop line |
| Distance | 15 m |

```yaml
sweep:
  bindings:
    ego.spawn_s:
      type: stop_line_offset
      offset: 15.0
```

The sweeper enumerates a single target key per run, so one entity's spawn can be
searched per scenario; the editor warns when a document asks for more.

The ego reaches the runner through the framework's own `ego.spawn_lanelet_id` /
`ego.spawn_s` keys. Other entities get a declared
`scenario.spawn_overrides.<entity>` sub-tree so they are addressable by exactly
the same plain `key=value` overrides.

## Metadata-driven GUI

Actions, conditions, constraints and bindings keep growing, so the templates
render *metadata*, never a `type` switch. Adding a primitive is a Python-side
change in `autoware_carla_scenario.authoring.registry`:

```python
register_action_spec(
    ActionSpec(
        type_id="lane_change",
        title="Lane Change",
        category="Vehicle / Motion",
        builder="build_lane_change_action",
        visual_kind="instant",
        fields=(FieldSpec("direction", "Direction", "select", "left", _DIRECTIONS),),
    )
)
```

plus a matching factory in `authoring.builders` that returns the framework's own
class. The inspector, the canvas, validation and the compiler all pick it up
with no template edit. `test_authoring_registry.py` fails if a spec names a
builder that does not exist, or a visual that names a field the primitive lacks.

## Save Draft vs Export Package

**Save Draft** writes the working document to `scenario_drafts/<id>.yaml`.

**Export Package** produces a directory another machine can copy and run:

```
cut_in_scenario/
|-- pyproject.toml              # dependencies, pinned exactly
|-- uv.lock                     # the resolved graph (tracked in git)
|-- .python-version             # e.g. 3.10.20 -- the exact patch version
|-- README.md
|-- conf/scenario/cut_in.yaml   # Hydra config
|-- scenario/
|   |-- document.yaml           # the Scenario IR
|   `-- manifest.yaml           # what this package was generated from
|-- src/cut_in_scenario/        # register() + the DeclarativeScenario binding
`-- tests/test_scenario.py      # loads, validates and compiles -- no CARLA needed
```

Run it with:

```bash
uv sync --locked
uv run scenario scenario=cut_in map=nishishinjuku
```

### Reproducibility

| What | Pinned by |
| --- | --- |
| `autoware-carla-scenario` | exact version, or an exact commit SHA |
| `autoware-lanelet2-to-opendrive` | the same way as the framework |
| Python | `.python-version`, exact patch version |
| uv | `[tool.uv] required-version`, when uv's version could be read |
| Everything else | `uv.lock` |

Both workspace projects are declared because the framework imports the
converter at module scope (`coordinate.road_lanelet_mapping`) without declaring
it as a dependency -- inside the workspace it is always installed alongside, so
the omission only shows up in a package that depends on the framework alone.

A **branch is never emitted** -- `main`, `master` and `HEAD` all move, so a
package pinned to one stops being the package that was tested as soon as
somebody pushes. When no immutable pin can be determined the export fails rather
than guessing. A local-path dependency is only produced by an explicit
development export, which says so in its README and manifest.

Set `SCENARIO_EXPORT_FRAMEWORK_VERSION` to pin a published release instead of
the current checkout's commit.

### Export is atomic

The exporter builds into a temporary directory and runs `uv lock`, then
`uv sync --locked`, then the generated package's own tests. **If dependency
locking or verification fails, nothing is written** -- a package whose
dependencies never resolved is not a successful export.

The manifest records only values that were actually observed:

```yaml
format_version: 1
scenario: {id: cut_in, title: Cut in, document_version: 1, package: cut-in-scenario}
runtime: {python: 3.10.20, uv: 0.9.7, requires_python: '>=3.10,<3.11'}
autoware_carla_scenario:
  source: git
  repository: https://github.com/tier4/autoware_lanelet2_to_opendrive
  commit: 0123456789abcdef0123456789abcdef01234567
  subdirectory: autoware_carla_scenario
files: {document: scenario/document.yaml, hydra_config: conf/scenario/cut_in.yaml}
notes: []
```

Anything that could not be determined is `null` with a note, never a plausible
guess.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCENARIO_EDITOR_DRAFTS` | `./scenario_drafts` | Where drafts are stored |
| `SCENARIO_EDITOR_EXPORT_DIR` | `./scenario_packages` | Default export destination |
| `SCENARIO_EDITOR_HOST` | `0.0.0.0` | Bind address |
| `SCENARIO_EDITOR_PORT` | `9100` | Bind port (the result viewer uses 9000) |
| `SCENARIO_EDITOR_MAP_VIEWER` | GitHub Pages build | URL of `simple_lanelet2`'s `viewer.js` |

## Running an authored scenario

An exported package plugs in through the same entry point any scenario package
uses (see [Architecture](architecture.md)):

```bash
cd cut_in_scenario
uv sync --locked
uv run scenario scenario=cut_in map=nishishinjuku

# Sweep every lanelet the spawn constraints match:
uv run scenario --multirun scenario=cut_in map=nishishinjuku \
    hydra/sweeper=lanelet_constraint
```

The CARLA client is an optional extra and is **not** locked, because the client
wheel is not published to PyPI; install it into the same environment before
running against a live server.

## Offline

Only two things in the editor come from the network, and neither is load-bearing
for editing:

| | |
| --- | --- |
| htmx | every interaction; the editor needs it |
| `simple_lanelet2`'s map viewer | the map drawing only — falls back to a server-rendered SVG |

The stylesheet is served by the app itself rather than by a CDN, so a blocked
egress rule cannot take the layout with it.
