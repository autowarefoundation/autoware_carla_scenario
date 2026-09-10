# Scenario Editor

The Scenario Editor is a web UI for authoring scenarios declaratively and
exporting them as self-contained **wheelhouses**. It runs as its own
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

## Which map: the Map library

A scenario's map is named by a **source URI** pointing at a git repository, and
the editor fetches and caches it. **Map library** in the header browses a
repository -- HuggingFace, GitHub, your own -- and lists every map in it; the
Scenario inspector shows, per scenario, whether the map is downloaded and
whether its revision is pinned.

See [HD Maps from a Git Repository](maps.md). Two things are worth knowing here:
a map Autoware's own setup already downloaded is used as it sits, and OpenDRIVE
is not needed to author a scenario -- lanelets are.

## Where it happens: the places panel

One map carries every lanelet the scenario names -- each entity's spawn, the
ego's goal, and the lanelet every condition watches. The canvas says *what*
happens and in what order; it has no way to say *where*, and until this panel
existed the answer was a scatter of ids across lane heads and condition cards
that only meant something with the map open in another window.

It sits at the top of the right-hand rail, over the inspector:

```
+------------------------------------------+---------------------------+
| Arrangement (the timeline, full width)    | Places      < 1/133 >     |
| ACTOR | Init | 1  | 2  | 3  | 4  | 5      |   (o) NPC1 spawn BOUND    |
| Ego   |      | [] |    |    |    |        |   (o) Ego goal            |
| NPC1  |      | [] | [] |    |    |        |   (o) Ego spawn           |
| Env   |      |    | [] |    |    |        | [Ego spawn][Ego goal][..] |
| PASS  | ...                               +---------------------------+
| FAIL  | ...                               | Inspector                 |
+------------------------------------------+---------------------------+
```

Three things had to be true at once -- the map, the timeline and the inspector
all on screen without scrolling -- and this is the arrangement that gets there
without taking width from the thing that needs it. The timeline's axis *is*
width: a card further right is later in the story, so narrowing it costs steps
you can see. The map is roughly square and the inspector is a column, so both
fit the rail; and "where" then sits directly over "what", which is what makes
clicking a place and reading its object one movement of the eye.

The whole editor is one CSS grid (`.ed-stage`) for that reason. `#editor-body`
stays the single htmx swap target and simply does not draw a box -- `display:
contents` lifts the timeline and the inspector into that grid -- so the map can
sit beside either of them while every edit still re-renders one element. Moving
the panel is a change to one block of `editor.css`, not to a template.

Which glyph a place gets is the **field's own declaration**, not a guess from
whatever holds it: `FieldSpec.place` is `spawn`, `goal` or `watched`, so the
`Set Goal` action's lanelet draws as a destination exactly as the ego's own goal
does. A primitive that registers a new lanelet field is drawn without the map
learning anything about it.

Each place is a **pin in its actor's track colour** -- the same hue that actor's
lane head and every condition naming it already wear -- with a glyph for what
the place is: a target for a spawn, a flag for a goal, an eye for a lanelet
something watches. The viewer has one highlight channel, so the outline says
"this scenario touches here" and the pins say which of them is which.

Under the map the same places are chips, in document order. Clicking one opens
the object that named it in the inspector *and* points the map at it; clicking a
place on the map does the same, because the overview draws places the document
already names and the only thing a click there can mean is "show me what named
this". Nothing is edited here -- a lanelet is still set where it is written, in
the picker its own field opens.

A label reads away from its dot, except on the right half of the map, where it
reads back towards the middle: that edge is also where the viewer keeps its own
zoom buttons, and a label running under them is a label nobody can read.

### An abstract scenario draws one bound pattern

A scenario whose lanelet is a constraint search does not name a place: it names
a **set**, and the sweeper runs it once per member. Outlining all 133 matches
draws the *search*, and none of those 133 runs looks like that picture. So one
match is **bound** -- through the same `swept_slot` the exported config sweeps
and the same constraint engine the sweep evaluates -- and the panel says which
of how many is on screen. The arrows step through the runs a `--multirun` sweep
would perform, one concrete scenario at a time.

Everything else keeps the id stored beside it, because that is exactly what a
run that does not sweep uses: the sweeper enumerates a single target key, so at
most one lanelet in a document is ever bound.

Binding means parsing the map on the server, which is measured in seconds, so it
is a button (**Bind a pattern**) rather than something every page load pays for.
Once the map is cached every later render binds without being asked, including
the panel's own refresh after an edit.

### Where a pin goes

The labels are the server's -- rendered with the panel, beside the list that
repeats them -- and only the *coordinates* are the viewer's. `editor.js` asks it
through its public API: `focusOn(id)` centres the view on a primitive, so
focusing one and reading `getView()` back reports that lanelet's centre in the
map's own coordinates, and the view is then put back where it was. The
alternative was projecting the `.osm` a second time in JavaScript, which is a
second answer to "where is this" that can disagree with the drawing it is laid
over. Each lanelet is measured once and kept on the frame, which is the frame
`reuseMap()` parks across a re-render, so an edit re-places the pins without
asking the map anything.

The panel frames itself on the places rather than on the whole city when the map
loads, and a pan or zoom after that is the person's: a re-render carries it
across, exactly as the picker's does. Tilted into the viewer's 3D view the pins
hide themselves, because `getView()` then reports drawing coordinates and a
point on a hill and the ground behind it are drawn in the same place.

This is a second parse of the same `.osm` while a picker is open, which is the
cost the inspector's own map was removed to avoid. It is a different drawing
rather than the same one twice: the picker answers "which lanelet is this
field", and the overview answers "where is this scenario" -- a question nothing
else on the page answers at all.

## The canvas

The main view is a **swimlane DAG drawn as a DAW arrangement**: one track per
actor, with the horizontal axis reading as *scenario progression*, not time.
Under the actors sits the **Environment** track, for actions no vehicle
performs.

```
STEP >      | 1          | 2                 | 3
------------+------------+-------------------+------------------
Ego         | Drive      |                   |
------------+------------+-------------------+------------------
NPC1        | Follow     |                   | # Lane Change Left
            |            |                   |         ^
            |            |                   |   +-----+-----+
            |            |                   |   |    ALL    |
            |            |                   |  NPC1->Ego  NPC1->Ego
            |            |                   |  Distance   TTC
            |            |                   |  < 20 m     < 4 s
------------+------------+-------------------+------------------
Environment |            | Set Traffic Signal|
------------+------------+-------------------+------------------
PASS        | NPC1 enters the ego lane
FAIL        | Collision  | Timeout 30 s
```

### Which track an action belongs to

The **action type** decides, never the author. `ActionSpec.scope` is either
`actor` or `environment`, and an environment action -- setting traffic lights,
for instance -- is drawn in the Environment track whatever its `actor` field
says. Nothing reads that field when such an action is built, so honouring it
put the card in some vehicle's lane and claimed a relationship the run does not
have. The editor no longer offers the field for one, refuses it on the way in,
and drops a stale value on the next save; each track's *+ action* menu offers
only the actions that belong in it.

The Environment track is drawn even when it is empty, because it is where you
go to add the first one. An action that merely has no actor *yet* is drawn
there too: it is invalid, and a card nothing draws cannot be corrected.

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

A step number means the same column on every lane, so there is no leading
column an actor track has and the Environment or verdict lanes do not. A spawn is not
a step -- it is the state a track starts in -- so it is stated in the lane head
rather than given a slot on the ruler.

**An action with no trigger fires on the first tick, wherever it is drawn.**
`BaseAction` defaults to `AlwaysTrueCondition`, so a card in step 5 with an
empty trigger runs immediately and its position is simply wrong about it. The
canvas writes "fires immediately" under the card, and the validator warns --
never errors, because a position is presentation.

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

Either can be narrowed to **a stretch of that lane** with `s from` / `s to` and
`t from` / `t to`. The runtime has always accepted these comparisons; the editor
used to discard them. They are the abstract way to say "in the last 20 m before
the junction": unlike an absolute region, `s` and `t` are measured along and
across the lane, so the condition keeps its meaning wherever a constraint sweep
puts the entity.

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

## Fixed or searched: every lanelet, one question

**Edit** on any lanelet field opens the map, and the panel beside it asks the
same thing wherever you opened it from — **Fixed** or **Constraint search**:

| Where a lanelet is named | Slot | Hydra key a sweep writes |
| --- | --- | --- |
| An entity's spawn | `ego.spawn`, `npc1.spawn` | `ego.spawn_lanelet_id`, `scenario.spawn_overrides.npc1.lanelet_id` |
| The ego's goal | `ego.goal` | `ego.goal_lanelet_id` |
| A `lanelet` parameter of an action or a condition | `<node id>.<field>` | `scenario.param_overrides.<node id>.<field>` |

A constraint is a statement about the map, so it is written with the map it
searches in view, and the matches are outlined on that same map as they are
counted. That is why the choice is in the picker and not in the 384px inspector
column, which only states which mode is in force and offers Edit.

A **set** of lanelets — `stop_lanelets`, the map's exclusion list — is picked by
hand and gets no such panel: a search names *one* lanelet per run, because that
is what the sweeper enumerates, so offering the choice there would promise that
a search fills the whole set.

**Fixed** pins the lanelet the field shows. For a spawn that is a lanelet and an
offset:

| Field | Value | Where |
| --- | --- | --- |
| Lanelet ID | 183 | the picker, by clicking the map |
| Offset | 12.5 m | the picker, under the same heading |

**Constraint search** hands the choice to the existing lanelet-constraint
sweeper. The tree is edited in the picker, each node with its own parameters —
the inspector column is behind the map while it is open — with the match count
beside it outlining what it found on that same map, and is serialised straight
into `sweep.constraints` under whichever key addresses the slot:

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

The id the field shows stays in the document under a search: it is the default a
run that does not sweep falls back to, and the value the exported config
declares so that Hydra's struct mode accepts the override at all.

**One search per run.** The sweeper enumerates a single target key, so a
document may search for one lanelet; a second search is a validation *warning*
rather than an error — the scenario still runs, with everything else pinned to
its default — and names which slot is being driven.

**Preview matches** evaluates that tree against the real Lanelet2 map and
reports, for example, `133 matched of 979 lanelets`. It is counted when the
picker is opened, not on every render: each lanelet field on the inspector has a
picker in the page at once, and counting means parsing a city. Loading a map is
opt-in and cached; if the map files are missing the count is unavailable but the
constraint editor keeps working.

### The map

The drawing comes from
[`simple_lanelet2`](https://github.com/hakuturu583/simple_lanelet2)'s wasm map
viewer — the same project that provides the `lanelet2` Python API this framework
runs on. The editor serves the scenario's `.osm` at
`/draft/<id>/map.osm`, the viewer parses and renders it in the browser, and the
editor drives it through two calls: `setHighlight()` with the matched IDs, and a
`select` listener that turns **clicking a lanelet into setting whichever lanelet
field the picker was opened from**.

Constraint evaluation stays on the server, in the framework's own sweeper — the
viewer only draws. That split is deliberate: a second constraint engine in
JavaScript could disagree with the sweep the scenario will actually run.

The module is loaded from the project's GitHub Pages build, because the wasm is
built rather than committed. Point `SCENARIO_EDITOR_MAP_VIEWER` at the output of
`simple_lanelet2`'s `tools/build_web.sh` to serve it yourself. When it cannot be
fetched the panel says so; the match count and the matched-ID list come from the
server and are unaffected.

### The map is parsed once

htmx replaces the whole inspector on every edit, so the server sends a
brand-new, empty map frame each time. Mounting that frame refetched the `.osm`
and parsed it again in wasm — one fetch per edit, for a map that had not
changed.

The frame that already holds the parsed scene carries a `data-viewer-key`, and
`reuseMap()` in `editor.js` swaps it back in over the fresh one, copying across
only what actually differs: which lanelets are outlined. The whole frame moves
rather than the canvas inside it — the viewer keeps a reference to the element
it was constructed with and observes it for resizes, so lifting the canvas out
would leave it measuring a node that is no longer on the page. Panning and
zooming survive an edit as a consequence, which re-mounting had been silently
throwing away.

That is what makes the picker editable. An edit in its side panel — the fixed or
searched choice, the constraints, a spawn's offset along the lanelet — posts like
every other control and swaps the whole editor body, which builds a fresh copy
of the modal while the open one hangs off `<body>`. The fresh copy is swapped in
behind the person using it, and the key carries the parsed map across, pan and
zoom included.

It is the **only** renderer. A server-rendered SVG used to sit behind it as an
offline fallback, but the page loads htmx from a CDN and every control here is an
`hx-` attribute, so an editor that cannot reach the network does not work at all
— the fallback bought no offline capability while costing a second drawing of the
same map on screen at once.

The viewer has a single highlight channel — one outline colour, no second class
— so **what is outlined follows the mode**: a constraint search outlines its
matches, a pinned lanelet outlines itself. The caption under the map names which
of the two it is, rather than showing colour swatches the viewer does not use.

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

## Ego: who drives, and where to

The ego's inspector opens with **Driven by** — the stack that drives it, exported
as `ego.entity`:

| Driven by | `ego.entity` | Goal |
| --- | --- | --- |
| TrafficManager (CARLA autopilot) | `autopilot` | optional — it reads none |
| Autoware | `autoware` | **required** — it plans its route to the goal and will not move without one |

(The framework's third value, `carla_driver`, needs a `driver` config group the
editor does not author; a run can still select it from the command line.)

The **Goal** section sits beside the spawn, because the two are the ends of the
same thing: where the run starts, and where the ego is meant to get to. A goal is
picked from the map like a spawn is, and may be **searched for** like one — the
sweep then writes `ego.goal_lanelet_id`. An Autoware ego without one is a validation
error; an ego the TrafficManager drives may be given none — a cut-in or a
red-light run is about what happens on the way — and **Clear goal** puts it back
to that.

A goal is not a card. A card happens *during* a run, and a goal is what the ego
needs before one can start: Autoware localizes at the spawn, plans a route to
the goal, and only then engages. The framework asks the same of a hand-written
scenario — `BaseScenario.require_goal()`, asked by `register_route_to_goal` and
by `ScenarioRunner` once `setup()` returns — so the editor stores the goal on the
ego and exports it as the framework's own keys:

```yaml
ego:
  spawn_lanelet_id: 183
  spawn_s: 0.0
  entity: autoware
  goal_lanelet_id: 265
  goal_s: 12.5
```

The goal keys are written only when a goal is set, leaving the `ego` group's own
`null` in place otherwise.

The **Set Goal** card still exists, for changing a destination mid-run or for a
vehicle that is not the ego. One owned by the ego and left in the initialization
phase is warned about: the ego's own goal is already delivered there, so the card
would send a second destination in the same phase. Only a vehicle that plans its own route reads a
goal at all, so a goal stored on any other entity is a validation error rather
than something quietly dropped at export.

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

### The map path is a document field

`/draft/<id>/map.osm` serves the file the document names, and that name is
typed by whoever is using the editor -- which binds `0.0.0.0` by default. Handed
straight to a `FileResponse` it was an arbitrary local file read for anyone who
could reach the port. A path is accepted only when it resolves **inside**
`SCENARIO_EDITOR_MAP_ROOTS` (the working directory unless set) and names a
`.osm`, so what the route can serve is bounded by where the editor was started
rather than by what the process can read.

That is a bound, not authentication: the editor still has none, so it belongs on
a network you trust.

## Save Draft vs Export Wheelhouse

**Save Draft** writes the working document to `scenario_drafts/<id>.yaml`.

**Export Wheelhouse** produces a `.zip` **the browser downloads**, holding every
wheel the scenario needs:

```
cut_in-wheelhouse.zip
`-- cut_in_scenario_wheelhouse/
    |-- cut_in_scenario-0.1.0-py3-none-any.whl     # the scenario itself
    |-- autoware_carla_scenario-*.whl              # the framework, at the pinned commit
    |-- autoware_lanelet2_to_opendrive-*.whl
    |-- carla-0.10.0-cp310-cp310-linux_x86_64.whl  # not published to any index
    |-- ... every transitive dependency, ~70 wheels
    |-- requirements.txt                           # the whole set, pinned
    `-- README.md                                  # how to install it
```

Install it with pip and nothing else -- no `uv`, no `git`, no network, no
resolution:

```bash
unzip cut_in-wheelhouse.zip
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links cut_in_scenario_wheelhouse cut-in-scenario
.venv/bin/scenario scenario=cut_in map=nishishinjuku
```

That is the point of the format. Autoware's `scenario_bridge` installs a
scenario into a venv built from `python3-venv` and `python3-pip` -- the two
things rosdep can resolve -- and has neither uv nor, on a vehicle, a network
route. A uv project needs all three; a wheelhouse needs none.

The scenario's document and Hydra config travel **inside** its wheel, so an
installed scenario is self-contained: there is no directory that has to be kept
beside it.

### What the wheelhouse is built from

The exporter still generates the uv project it always did -- `pyproject.toml`
with the framework pinned to an exact commit, `uv.lock`, the document, the Hydra
config, the package's own tests -- and still runs `uv sync --locked` and those
tests against it. That project is now a **build input**: the wheels are resolved
from its lockfile and the project itself does not leave the server.

The CARLA client is vendored into it (`carla_wheels/`, reached by a relative
`[tool.uv] find-links`) and requested through the framework's `carla` extra,
because a scenario that cannot `import carla` cannot run and the client is on no
index. When no client wheel can be found -- a framework installed from a wheel
rather than run out of its repository -- the export says so in its warnings and
leaves the client out.

There is no destination field. The editor is routinely used from another machine
on the LAN, where a path typed into it would name a directory on the host running
the server -- not one the person exporting can reach. Everything is built in a
temporary directory, the wheelhouse is zipped, and the build tree is removed, so
the only thing that outlives the request is the archive, and re-exporting never
has to overwrite a half-written one.

The response is still the **report** -- warnings, the tool log, whether the
package's own tests passed -- with the download link in it. Making the response
the file itself would throw away the very things an export is checked for.

### One platform, one interpreter

A wheelhouse is resolved *by* an interpreter *for* a platform. The wheels the
editor builds are the ones Python 3.10 on the exporting machine selected, and the
CARLA client in particular is a compiled CPython-3.10-only extension. Installing
them under a different Python or on a different platform fails on the first wheel
with no matching tag; re-export on a matching machine instead.

It is also large -- the client, OpenCV and the lanelet2 bindings come to most of
160 MB. That is the cost of not needing a network at install time.

### Reproducibility

| What | Pinned by |
| --- | --- |
| `autoware-carla-scenario` | exact version, or an exact commit SHA |
| `autoware-lanelet2-to-opendrive` | the same way as the framework |
| `carla` | the wheel vendored in the repository, at the version the framework's `carla` extra names |
| Python | `.python-version`, exact patch version |
| uv | `[tool.uv] required-version`, when uv's version could be read |
| Everything else | `uv.lock`, and then the wheels built from it |

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
the current checkout's commit, and `SCENARIO_EXPORT_CARLA_WHEELS` to point the
export at a directory of client wheels other than the repository's own.

### Export is atomic

The exporter builds into a temporary directory and runs `uv lock`, then
`uv sync --locked`, then the generated package's own tests, then the wheelhouse.
**If locking, verification or the wheelhouse fails, nothing is written** -- a
package whose dependencies never resolved, or that no environment can install,
is not a successful export.

The manifest records only values that were actually observed:

```yaml
format_version: 2
scenario: {id: cut_in, title: Cut in, document_version: 1, package: cut-in-scenario}
runtime: {python: 3.10.20, uv: 0.12.0, requires_python: '>=3.10,<3.11'}
wheelhouse:
  directory: cut_in_scenario_wheelhouse
  install: pip install --no-index --find-links cut_in_scenario_wheelhouse cut-in-scenario
  wheels: 70
  python: '3.10.20'
autoware_carla_scenario:
  source: git
  repository: https://github.com/tier4/autoware_lanelet2_to_opendrive
  commit: 0123456789abcdef0123456789abcdef01234567
  subdirectory: autoware_carla_scenario
  extras: [carla]
files:
  document: src/cut_in_scenario/document.yaml
  hydra_config: src/cut_in_scenario/conf/scenario/cut_in.yaml
notes: []
```

Anything that could not be determined is `null` with a note, never a plausible
guess.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCENARIO_EDITOR_DRAFTS` | `./scenario_drafts` | Where drafts are stored |
| `SCENARIO_EDITOR_EXPORT_DIR` | `./scenario_packages` | Where an export's wheelhouse `.zip` is staged until the browser fetches it |
| `SCENARIO_EDITOR_MAP_ROOTS` | the working directory | Directories a Lanelet2 map may be read from, `:`-separated |
| `SCENARIO_EDITOR_HOST` | `0.0.0.0` | Bind address |
| `SCENARIO_EDITOR_PORT` | `9100` | Bind port (the result viewer uses 9000) |
| `SCENARIO_EDITOR_MAP_VIEWER` | GitHub Pages build | URL of `simple_lanelet2`'s `viewer.js` |

## Running an authored scenario

An exported scenario plugs in through the same entry point any scenario package
uses (see [Architecture](architecture.md)) -- installed from its wheelhouse, the
`scenario` command is on the venv's `PATH`:

```bash
.venv/bin/scenario scenario=cut_in map=nishishinjuku

# Sweep every lanelet the spawn constraints match:
.venv/bin/scenario --multirun scenario=cut_in map=nishishinjuku \
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
