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

The main view is a **swimlane DAG drawn as a DAW arrangement**: one track per
actor, with the horizontal axis reading as *scenario progression*, not time.
Actions that act on the environment rather than a vehicle get a **World** track
of their own.

```
STEP >   | 1          | 2                 | 3
---------+------------+-------------------+------------------
Ego      | [SPAWN]    | Drive             |
---------+------------+-------------------+------------------
NPC1     | [SPAWN]    | Follow            | # Lane Change Left
         |            |                   |         ^
         |            |                   |   +-----+-----+
         |            |                   |   |    ALL    |
         |            |                   |  NPC1->Ego  NPC1->Ego
         |            |                   |  Distance   TTC
         |            |                   |  < 20 m     < 4 s
---------+------------+-------------------+------------------
PASS     | NPC1 enters the ego lane
FAIL     | Collision  | Timeout 30 s
```

The sequencer furniture is what makes the direction readable: track headers
down the left, a numbered ruler across the top, a bar line before every slot
and every other slot shaded. The ruler counts **steps and not seconds**, which
is the whole reason it is a step ruler -- see the first rule below.

### What a step is

Nothing in Python has a "step". The runtime arms **every** action from the
first tick and ticks them all every frame:

```python
for action in scenario._pre_tick_actions:
    action.tick(world, elapsed)
```

An action fires when its own trigger says so, not when its column comes up, and
an action with no trigger gets `AlwaysTrueCondition` -- so it fires on tick 1
however far right it is drawn. `column_hint` is read by nothing outside this
editor: `compiler.py`, `declarative.py` and the exporters contain no reference
to `ui` at all.

So a step is not a point in time. It is **a set of actions nothing orders**,
and the canvas draws it that way:

* **Several actions can share one step**, stacked in the same column. That is
  the honest picture of two actions that are both armed and neither waiting on
  the other -- spreading them across two columns would draw a sequence the
  runtime does not have.
* **An action may not share a step with anything its trigger waits on.** Within
  a step nothing is ordered, so a dependency has to be visibly earlier. Adding
  an `action_state` trigger pushes the dependent right on the spot, moving a
  dependency right carries its dependents with it, and a move that would break
  the rule is repaired rather than refused -- the card lands as close to where
  it was aimed as its dependencies allow. `validator.py` reports a hand-edited
  `ui` block that says otherwise.

That makes the step axis mean exactly one thing: **left of** is "already
finished, and something here is waiting on it". It still says nothing about
*how much* earlier.

Three rules make it readable:

* **The distance between cards means nothing.** Column position is
  `ui.column_hint`, which is presentation only and never reaches the runtime.
  A clip in step 3 may only start after one in step 2 if something says so;
  how much later is not on screen because the document does not know.
* **There is no separate event lane.** A condition is a *trigger*, drawn under
  the action it fires and joined to it by a solid line, so cause and effect are
  next to each other instead of being correlated across the screen.
* **One actor reacting to another is a reference, not a coincidence.**
  "Swerve once NPC1 has cut in" is a `Lane Change` clip on the ego track whose
  trigger is an **Action state** condition naming NPC1's cut-in and the state
  `completeState`. That reference lives in the document
  (`params: {action: a_..., state: completeState}`), so it is a fact the
  scenario contains rather than something the canvas infers.

The canvas draws that reference as a **dashed line in the causing actor's track
colour**, running from the action to the condition waiting on it. The cut-in
example reads as one chain across two tracks:

```
NPC1   ... [Cut in] ......
                     :         <- dashed: Cut in -> the condition waiting on it
Ego    ..............:...  [Evade right]
                          ^ Cut in · NPC1 | Action | completeState
                            solid: fires this action
```

Both line types are named in the legend above the canvas, because a dash
pattern is not self-explanatory. Crucially the dashed line is drawn **only**
from `data-caused-by`, the document's own reference -- never from where two
cards happen to sit. Moving a clip can therefore neither invent a causal link
nor erase one, and a condition that names no action (a distance check, say) is
never drawn as caused by anything.

### Action states

The states are ASAM OpenSCENARIO 1.2's `StoryboardElementState`, applied to a
single action:

| State | Meaning |
| --- | --- |
| `standbyState` | Instantiated, waiting for its start trigger |
| `startTransition` | The trigger fired and `execute()` ran -- held for one tick |
| `runningState` | The work is under way |
| `endTransition` | The completion criteria were met -- held for one tick |
| `completeState` | Finished (a repeating action returns to `standbyState`) |

`completeState` is the one that makes "after the cut-in" verifiable. A forced
lane change stays in `runningState` until the vehicle has actually **settled
onto the next lane** -- a different lane id, within
`LANE_CHANGE_CENTER_TOLERANCE_M` of its centre and
`LANE_CHANGE_HEADING_TOLERANCE_DEG` of its heading. A lane id change alone is
not enough: a car whose centre has just crossed the boundary is still diagonal,
and a reaction triggered on that would fire mid-manoeuvre.

There is deliberately **no failure state and no action-level timeout**, because
OpenSCENARIO has neither. A manoeuvre that never happens stays in
`runningState` forever, so `completeState` cannot be reached by a lane change
that did not occur; ending such a run is the scenario timeout's job, which is
already a FAIL condition. The two transition states are each held for exactly
one tick, so a condition watching `startTransition` sees it.

Which is also why a card can be moved into an **empty** step rather than only
swapped with its neighbour: a reaction has to be placeable to the right of the
cause on another track, and its own track is usually empty in between.

Every condition reads as `subject -> target | metric | rule value`, e.g.
`NPC1 -> Ego | Distance | < 20 m` or `Ego -> Lanelet 183 | Position | inside`.

A target that is not an actor is **named for the coordinate system it is in**.
This framework speaks both Lanelet2 and OpenDRIVE and their ids are written the
same way, so `Ego -> 183` would say nothing about which `183` is meant.

### Position: pick a frame, not a mixture

There are two position conditions, one per coordinate system, because the two
need different fields:

| | Address | Reads as |
| --- | --- | --- |
| **Position (Lanelet2)** | a lanelet | `NPC1 -> Lanelet 183 \| Position \| inside` |
| **Position (OpenDRIVE)** | a road, optionally a lane | `NPC1 -> Road 80 \| Position \| lane 2 \| inside` |

A **lanelet already names one lane**, so the Lanelet2 condition resolves it to
an OpenDRIVE road *and* lane and there is nothing further to pin. An
**OpenDRIVE road does not**: a road carries several lanes, so road and lane
together are what address a place uniquely, and a road left without a lane says
so on the card (`any lane`) rather than looking like a precise address.

### Lanelet2 ids are picked, never typed

Every field naming a Lanelet2 lanelet is edited **only on the map**. The number beside it is a
readout, not an input: nobody knows lanelet ids by heart, so typing one is
guesswork the map answers exactly, and making the map the single editor leaves
one path a value can arrive by instead of two to keep in step. **Edit** opens
the viewer over the whole window -- the inspector column is 384px wide, and a
city map that size cannot be picked from -- and clicking a lanelet fills the
field and closes it. This comes from the field's declared kind (`lanelet`), so
any field naming a lanelet gets the picker without touching a template.

There is no "type it instead" fallback for a viewer that will not load, for the
same reason the SVG map fallback was removed: the page fetches htmx from a CDN
too, so an editor that cannot reach the network has no working controls at all.

Three fields still take typed ids, each because their value is **not only** a
lanelet id, and a test in `test_authoring_registry.py` fails if a new one
appears without being listed there with its reason:

| Field | Why it is not picked |
| --- | --- |
| `equals.value` (spawn constraint) | `any` matches every lanelet -- a sentinel the sweeper parses, not an id |
| `in_set.values` (spawn constraint) | accepts `${map.no_3d_model_lanelet_ids}`, a reference resolved at sweep time |
| `traffic_signal` regulatory element ids | the viewer's `regulatory` layer reports the id of the *linestring* that draws a sign, not of the regulatory element -- a picker there would save a confidently wrong number |

Two implementation notes, both of which cost an afternoon to find:

* The picker is **moved to `<body>`** before it is shown. The inspector sits
  inside a `position: sticky` wrapper, and sticky creates a stacking context
  whatever its own `z-index` is -- so a `z-index` left in place ranks only
  against its siblings, and the canvas lanes (`z-index: 2`) paint straight over
  a modal asking for 80.
* A **set** of lanelets works the same way, toggling on each click, and saves
  when the picker is closed -- sending the form on every toggle would re-render
  the inspector and tear the map down mid-selection.
* Which viewer layers a click may land on is part of the field's declaration
  (`FieldSpec.picks`). The layers overlap: a click on a road usually lands on
  the direction arrow, and a hair to the side lands on a `bound`, **which
  reports the id of a linestring, not of either lanelet it separates**. Only
  the layers whose id is the lanelet's own are accepted.
* `/static` is served with `Cache-Control: no-cache`. `StaticFiles` sends an
  `ETag` but no `Cache-Control`, which lets a browser invent a freshness
  lifetime and keep running an old `editor.js` **without asking** -- an editor
  visibly missing a feature the server is already serving. The `ETag` still
  makes the check a 304.

They are separate conditions rather than one with a coordinate-system switch
because offering a lanelet and a lane on the same condition is what let the two
contradict each other: on the nishishinjuku map lanelets 183 and 184 are lanes
2 and 1 of road 80, so pinning lanelet 183 and lane 1 asked for a place that
does not exist.
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
built rather than committed. Point `SCENARIO_EDITOR_MAP_VIEWER` at the output of
`simple_lanelet2`'s `tools/build_web.sh` to serve it yourself. When it cannot be
fetched the panel says so; the match count and the matched-ID list come from the
server and are unaffected.

It is the **only** renderer. A server-rendered SVG used to sit behind it as an
offline fallback, but the page loads htmx from a CDN and every control here is an
`hx-` attribute, so an editor that cannot reach the network does not work at all
— the fallback bought no offline capability while costing a second drawing of the
same map on screen at once.

The viewer has a single highlight channel — one outline colour, no second class
— so **what is outlined follows the spawn mode**: a constraint search outlines
its matches, a fixed spawn outlines the pinned lanelet. The caption under the map
names which of the two it is, rather than showing colour swatches the viewer does
not use.

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

**Export Package** produces a `.zip` **the browser downloads**, holding a
directory another machine can unpack and run:

```
cut_in.zip
`-- cut_in_scenario/
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

There is no destination field. The editor is routinely used from another machine
on the LAN, where a path typed into it would name a directory on the host running
the server -- not one the person exporting can reach. The package is built in a
temporary directory, zipped, and handed back; the build tree is removed, so the
only thing that outlives the request is the archive, and re-exporting never has
to overwrite a half-written one.

The response is still the **report** -- warnings, the tool log, whether the
package's own tests passed -- with the download link in it. Making the response
the file itself would throw away the very things an export is checked for.

Unpack and run it with:

```bash
unzip cut_in.zip && cd cut_in_scenario
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
| `SCENARIO_EDITOR_EXPORT_DIR` | `./scenario_packages` | Where an export's `.zip` is staged until the browser fetches it |
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
| `simple_lanelet2`'s map viewer | the map drawing only — the panel says so when it cannot be fetched |

The stylesheet is served by the app itself rather than by a CDN, so a blocked
egress rule cannot take the layout with it.
