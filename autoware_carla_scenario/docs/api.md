# API Reference

This page lists the public surface of `autoware_carla_scenario`. All
symbols exported via the top-level package (`autoware_carla_scenario`)
are stable; symbols only reachable through deeper subpackages should be
considered internal unless explicitly noted.

## Top-level package

```python
from autoware_carla_scenario import (
    BaseScenario, EgoConfig, ScenarioQueue, ScenarioRunner,
    CarlaServerManager, CarlaScenarioFixture,
    # Conditions, actions, kinematics, coordinate poses ...
)
```

The full list of re-exports is defined in `autoware_carla_scenario.__all__`
and includes the conditions, actions, kinematics, coordinate, entity,
and sensor types described below.

## Scenario orchestration

| Symbol | Module | Purpose |
|--------|--------|---------|
| `BaseScenario` | `scenario_base` | Abstract base for user scenarios. Subclasses implement `setup()` and `is_done()`. |
| `EgoConfig` | `scenario_base` | `VehicleEntityConfig` subclass that fixes `role_name` to `EGO_ROLE_NAME` and carries both ends of the run: the spawn and the ego's `goal_pose`. `None` means "not named yet" — a scenario may derive the goal in `setup()`, and one that ends setup with none is refused. |
| `ScenarioRunner` | `scenario_runner` | Executes a single `BaseScenario` against a CARLA world (sync mode tick loop, recording, cleanup). |
| `ScenarioQueue` | `scenario_queue` | Context manager that owns a `CarlaServerManager` and runs registered scenarios sequentially with cooldown / retry. |
| `CarlaServerManager` | `server` | Starts, reuses, and stops the CARLA UE5 process. Reads `CARLA_EXECUTABLE`, else launches the CARLA `scenario-setup` installed. |
| `CarlaScenarioFixture` | `pytest_fixtures` | Helper that registers a scenario into a queue at import time and exposes a session-scoped pytest fixture for its `ScenarioResult`. |
| `EGO_ROLE_NAME` | `constants` | Reserved CARLA `role_name` used for the ego actor. |
| `EntityRole` | `entity_role` | Validated `role_name` wrapper for CARLA actors. Factories: `EntityRole.ego()`, `EntityRole.npc(n)`. |

`CameraRecorder` (re-exported from the top-level package, source in
`autoware_carla_scenario.camera_recorder`) is the two-pass video
renderer driven by the native CARLA recorder + an RGB camera sensor.
It is used internally by `ScenarioRunner` and can also be instantiated
directly.

## Coverage (`autoware_carla_scenario.coverage`)

See [Coverage](coverage.md).

| Symbol | Purpose |
|--------|---------|
| `BaseScenario.register_cover(name, expression, *, unit, range, every, buckets, values, ignore, event, text, target, cover_by, min_stay)` | Declare a cover item, after `cover()` in OpenSCENARIO DSL. |
| `BaseScenario.register_cross(name, items, *, text, target, cover_by, min_stay)` | Cross coverage of cover items sampled on the same event. |
| `SamplingEvent` | When an item is sampled: `START`, `END` (default) or `TICK`. Re-exported from the top-level package. |
| `CoverItem`, `CrossItem` | The item definitions `register_cover()` / `register_cross()` build. |
| `BaseScenario.register_measure(key, read, *, unit, text)`, `.measure(key, world)`, `.measures` | A scenario's measures: what it sets up, under fixed keys (`autoware_carla_scenario.measures`: `VEHICLE_AHEAD_GAP_M`, `VEHICLE_AHEAD_RELATIVE_SPEED_KPH`, `CROSSING_PEDESTRIAN_GAP_M`, `CROSSING_PEDESTRIAN_SPEED_MS`). Replace a built-in one or add one of its own. See [Scenario measures](odd.md#scenario-measures). |
| `CoverageCollector` | Samples items on their events during a run; written as `{Scenario}_coverage.json`. |
| `merge_coverage(documents)`, `CoverageReport` | Merge coverage files and grade them. |
| `coverage.cod.export_cod(document, out_dir, stem)` | Write a run's ODD samples as an ASAM OpenODD COD table, manifest and taxonomy (`scenario-coverage --export-cod`). |

## ODD (`autoware_carla_scenario.odd`)

See [ODD](odd.md).

| Symbol | Purpose |
|--------|---------|
| `OddAttribute(name, probe, *, unit, range, every, buckets, values, text, target, cover_by, min_stay)` | A measured taxonomy concept. Conditions: `is_in`, `equals`, `between`, `at_least`, `at_most`, `greater_than`, `less_than`, `is_unknown`. |
| `OddModule(name, *, include_and, include_or, exclude_and, exclude_or, labels, active, text, situation, target, cover_by, min_stay)` | A named rule; with `situation=True`, also covered as a situation. `OddDefinition.situations()` lists those. |
| `OddDefinition(name, attributes, modules, *, roots, text)` | The ODD: judges values with OpenODD's semantics (roots, labels, inactive modules, missing values), and marks buckets outside it. |
| `all_of`, `any_of`, `module_holds` | Group conditions; refer to another module or label. |
| `default_odd()` | The built-in ODD: every attribute, no modules. |
| `register_odd(name, builder)`, `resolve_odd(spec)` | Name an ODD; turn a name, `.yaml` path or `module:function` into one. Packages register through the `autoware_carla_scenario.odds` entry point group (a function calling `register_odd`). |
| `load_odd_file(path)`, `load_odd_binding(path)`, `load_openodd(*sources, bindings, name, text, situations)` | Read ASAM OpenODD 1.0 YAML (`IMPORT`, `TAXONOMY`, `MODULES`, `ODD`), with a binding file naming the probes. |
| `GitSource(url, rev, path)`, `GitSourceError` | An OpenODD file in a git repository at a revision, for `load_openodd()` or a binding file's `openodd` list. |
| `ego_speed_kph`, `speed_limit_kph`, `lanelet_location`, `lanelet_subtype`, `lanelet_speed_limit_kph`, `in_junction`, `lane_count`, `illumination`, `rain`, `fog`, `traffic_density`, `pedestrian_nearby` | Built-in probes. |
| `scenario_measure(key)`, `ScenarioMeasure` | The probe mapping an attribute onto the running scenario's measure *key* (`measure:` in a binding file). See [Scenario measures](odd.md#scenario-measures). |
| `typecheck.typecheck_odd(builder)` | Compile a Python ODD with Codon. |
| `plan_route_coverage(odd, scenario_or_route, *, lanelet_map, routing_graph, name)` | Check a planned route against the ODD on the Lanelet2 map: the verdict per lanelet (inside, outside, undecided) and the expected metres per bucket. Takes a scenario, a `PlannedRoute(start, goal, via, name)` or lanelet ids. Raises `RouteError` when the route cannot be planned. See [Checking a planned route](odd.md#checking-a-planned-route). |
| `combine_route_coverage(odd, routes)` | The expected coverage of several routes, and the buckets no route reaches. |
| `plan_route(route, lanelet_map, routing_graph)` | The lanelets of a `PlannedRoute`, with the metres driven on each. |
| `UNDECIDED` | What a probe's `on_lanelet` returns for a value only the run can tell. |
| `OddSampler(odd, knobs, *, controls, seed, strategy, coverage, max_tries)` | Draws concrete cases inside the ODD: `.sample(count)` gives `OddSample(index, overrides, buckets, values, situation)`. `strategy="coverage"` draws the least covered buckets and the uncovered situations first. See [Sampling scenarios from the ODD](odd.md#sampling-scenarios-from-the-odd). |
| `OddKnob(key, *, values, scale, offset, integer, range)`, `knobs_from_mapping(raw)`, `DEFAULT_KNOBS` | A control: the config key that sets a measure (or, as a sweep knob, an attribute), and what it can stage. Scenarios declare theirs under `controls` in their config, by measure key. |

## Conditions (`autoware_carla_scenario.conditions`)

All conditions inherit from `BaseCondition` and return a
`ScenarioResult` from `check(world, elapsed)` once triggered (or
`None` when not yet triggered).

| Symbol | Description |
|--------|-------------|
| `BaseCondition` | Abstract base class. Subclasses override `_check()`. |
| `ScenarioResult` | Pass / fail outcome with message, elapsed time, and per-condition statuses. |
| `ConditionStatus` | Per-condition leaf record used for reporting. |
| `AlwaysTrueCondition` | Default trigger for actions. |
| `AndCondition`, `OrCondition`, `NotCondition` | Logical combinators. |
| `StickyCondition`, `PersistentCondition` | Latch / persist a child condition's truth value. |
| `TimeoutCondition`, `ElapsedTimeCondition` | Time-based triggers. |
| `EntityDistanceCondition`, `TimeToCollisionCondition` | Relative conditions between two entities: separation, and time to collision along the line joining them. |
| `CollisionCondition`, `EntityExistenceCondition` | Safety checks. |
| `TrafficSignalCondition` | Traffic-light state check. |
| `RouteProgressCondition` | An entity's (default: the ego's) distance along the scenario's route, from an anchor (`junction:0:entry`, ...), compared with a rule; robust to lane changes. See [Logical Scenarios from Routes](logical_scenarios.md). |
| `ComparisonRule`, `ScalarComparisonRule`, `compare` | Numeric comparison primitives. `compare(actual, rule, value, tolerance)` is the underlying helper. |
| `find_actor_by_role_name`, `find_actor_in_list` | Helpers for locating CARLA actors by role. `find_actor_in_list` is reachable via `autoware_carla_scenario.conditions`. |

### Composition conditions (`autoware_carla_scenario.conditions.composition`)

These build on `CompositionCondition`, which composes a child condition
tree internally:

- `EntityLanePositionCondition`
- `SpeedCondition`, `SpeedDirection`, `SpeedCoordinateSystem`
- `StandstillCondition`
- `TemporaryStopCondition`
- `WaypointCondition`, `WaypointCheckType`

## Actions (`autoware_carla_scenario.actions`)

| Symbol | Description |
|--------|-------------|
| `BaseAction` | Abstract base. Owns a trigger `BaseCondition`, a `TickTiming`, and an `execute(world)` side effect. |
| `TickTiming` | Enum: `PRE_TICK` / `POST_TICK`. |
| `TurnAction`, `TurnDirection` | Steer the ego through left / right turns via the CARLA TrafficManager route hints. |
| `LaneChangeAction`, `LaneChangeDirection` | Trigger a TrafficManager lane change. |
| `TrafficSignalAction`, `TrafficLightTarget` | Set traffic-light states (e.g. all RED, all GREEN, or a specific actor). |
| `FollowTrajectoryAction` | Move a vehicle or pedestrian along a `Trajectory` (OpenSCENARIO `FollowTrajectoryAction`); `speed` paces the segments no time does; `held_vertex` names the vertex a waypoint condition is holding it at; `appear_on_start=True` brings a hidden entity in on its first vertex, moving at `speed`, when a run starts. See [Trajectories and Recorded-Scene Replay](trajectory.md). |

## Trajectories (`autoware_carla_scenario.trajectory`)

See [Trajectories and Recorded-Scene Replay](trajectory.md).

| Symbol | Description |
|--------|-------------|
| `Trajectory`, `TrajectoryVertex` | A named polyline and its vertices: a position and the condition the entity departs it on (`advance`) -- a time, `TrajectoryTimeCondition(t)` (read back as `vertex.time`), or any other condition ([waypoint conditions](trajectory.md#waypoint-conditions); `Trajectory.is_gated`, `Trajectory.gated({index: condition})`). |
| `TrajectoryTimeCondition` | A vertex's time: depart when the trajectory's clock reaches it. |
| `MapPose` | An absolute pose in Autoware's `map` frame, as a recording states it. |
| `RelativeLanePose` | A pose relative to an entity in lane coordinates (`ds`, `offset`, `d_lane`, `yaw`, `entity_ref`; OpenSCENARIO `RelativeLanePosition`), placed when the action starts. |
| `RouteLanePose`, `RouteOppositePose`, `RouteCrossingPose`, `RouteCrosswalkPose`, `RouteRoadsidePose` | Poses placed against the scenario's route (a [logical scenario](logical_scenarios.md)) when the action starts: along the route and across lanes, on the opposite road, on a lanelet entering a route junction from the left / right / opposite, on a junction's crosswalk, at the roadside. |
| `TrajectoryTiming`, `ReferenceContext` | How vertex times map onto the scenario clock (`τ * scale + offset`, from the scenario or the action start). |
| `TrajectoryFollowingMode` | `POSITION` (kinematic replay) or `FOLLOW` (a controller tracks it). |

Recorded scenes (T4) are transcribed into documents outside the framework, by
the separate `scene_to_scenario_transpiler` package (`scenario-import-t4`).

## Sensors (`autoware_carla_scenario.sensor`)

| Symbol | Description |
|--------|-------------|
| `CameraSensorBase`, `CameraSensorConfig` | Provider-agnostic camera sensor interface. |
| `CarlaCameraSensor`, `CarlaCameraSensorConfig` | CARLA RGB camera implementation, used by the video recorder and the driver rig. |

## External driver (`autoware_carla_scenario.driver`)

Connects the ego to a driving policy served over alpasim's
`egodriver.EgodriverService`. See [External Driver Interface](driver_interface.md).

| Symbol | Description |
|--------|-------------|
| `BaseEgoDriverClient` | Transport-agnostic client interface: session lifecycle, observation submission, `drive`. |
| `EgoDriverGrpcClient` | gRPC implementation against the vendored alpasim protos. |
| `DriverClientConfig`, `DriverCameraConfig` | Connection settings, policy cadence, and the camera rig streamed to the policy. |
| `ControlConfig`, `TrajectoryFollower`, `VehicleCommand` | Pure-pursuit + PID tracking that turns a plan into `carla.VehicleControl`. |
| `Pose`, `Trajectory` | Rigid-transform primitives with protobuf conversions (right-handed frame). |
| `EgoObservation`, `DriveOutcome` | The state sent to the policy and the plan it returns. |
| `RendererDataBuilder` | Collects CARLA ground truth (governing traffic light, other vehicles, speed limit) into the `renderer_data` extension payload. |

## Traffic backends (`autoware_carla_scenario.traffic`)

Owns the vehicles a scenario did not author, and answers the manoeuvre intents
asked of the ones it did. See [Traffic Backends](traffic_backends.md).

| Symbol | Description |
|--------|-------------|
| `TrafficBackend` | The interface: `prepare` / `adopt` / `start` / `tick` / `close`, plus the manoeuvre intents. |
| `TrafficContext` | The run a backend joins: client, world, map name, OpenDRIVE, step, seed, output directory. |
| `TrafficManagerBackend` | CARLA's TrafficManager — the default backend. |
| `NullTrafficBackend` | No traffic model at all (`traffic=none`). |
| `TrafficConfig`, `TrafficManagerBackendConfig` | Which backend drives a run, and the TrafficManager's own options. |
| `register_backend`, `build_backend`, `available_backends` | The registry a third-party backend joins through the `autoware_carla_scenario.traffic_backends` entry point. |
| `TrafficBackendError`, `TrafficBackendUnavailable` | A backend that cannot do what the run needs; the second names a simulator that is not installed. |

## Coordinate transforms (`autoware_carla_scenario.coordinate`)

| Symbol | Description |
|--------|-------------|
| `Lanelet2Pose`, `OpenDrivePose`, `CarlaWorldPose`, `AnyPose` | Frame-tagged pose dataclasses. |
| `CoordinateFrame`, `FrameMismatchError`, `frame_of` | Coordinate-frame tagging and validation. |
| `MapManager` | Singleton owning the loaded `LaneletMap`, `pyxodr` road network, MGRS offset, and z offset. |
| `to_carla_world`, `to_carla_location`, `to_lanelet2`, `to_opendrive` | Pose conversion entry points (overload by input frame). |
| `project_onto_road` | Project a `CarlaWorldPose` onto a specified OpenDRIVE road. |
| `snap_to_carla_road` | Ray-cast a pose onto the rendered CARLA ground surface. |
| `GroundProjectionConfig` | Ray-cast tuning for `snap_to_carla_road`. |
| `get_stop_line_poses`, `get_stop_line_poses_with_following` | Resolve stop-line `Lanelet2Pose`s for a lanelet (optionally including following lanelets). |

## Entities (`autoware_carla_scenario.entity`)

| Symbol | Description |
|--------|-------------|
| `VehicleEntity`, `VehicleEntityConfig` | Generic vehicle actor with retry-aware spawn. |
| `EgoVehicle` | Subclass with the fixed `EGO_ROLE_NAME`, plus the `on_scenario_start` / `on_tick` / `on_scenario_end` lifecycle hooks `ScenarioRunner` calls. |
| `AutowareEntity` | Opts out of TrafficManager autopilot and leaves the actor for an external stack. |
| `CarlaDriverEntity` | Drives the ego from an external policy's plan over the `egodriver` gRPC contract. See [External Driver Interface](driver_interface.md). |
| `SpawnLocation` (Protocol) | Tag interface implemented by spawn-point providers. |
| `SpawnTransform` | Spawn at an explicit `carla.Transform`. |
| `SpawnPointIndex` | Spawn at the N-th map spawn point. |

## Kinematics (`autoware_carla_scenario.kinematics`)

Frame-aware velocity / acceleration types with affine-space arithmetic
(absolute - absolute = relative; absolute + relative = absolute).

| Symbol | Description |
|--------|-------------|
| `Vector3` | Frame-tagged 3-vector. |
| `CoordinateFrame`, `FrameMismatchError`, `frame_of` | Re-exported from `coordinate.frames`. |
| `AbsoluteVelocity`, `RelativeVelocity`, `FrenetVelocity` | Velocity types. |
| `AbsoluteAcceleration`, `RelativeAcceleration`, `FrenetAcceleration` | Acceleration types. |

## Lanelet constraint sweeper (`autoware_carla_scenario.sweeper`)

Powers the Hydra `lanelet_constraint` sweeper plugin. Can also be used
directly from Python:

| Symbol | Description |
|--------|-------------|
| `LaneletConstraintSweeper` | The Hydra `Sweeper` implementation (also re-exposed via `hydra_plugins.autoware_scenario_sweeper`). |
| `Constraint` (Protocol) | Base interface. |
| `EqualsConstraint`, `InSetConstraint`, `LaneletLengthConstraint`, `HasStopLineConstraint`, `HasTrafficLightStopLineConstraint`, `HasAdjacentConstraint`, `IsJunctionConstraint`, `PreviousOfConstraint`, `FollowingOfConstraint` | Atomic constraints. |
| `AndConstraint`, `OrConstraint`, `NotConstraint` | Combinators. |
| `parse_constraint`, `find_matching_lanelets` | YAML-to-`Constraint` parsing and lanelet matching. The corresponding YAML `type:` keys for the atomics above are `equals`, `in_set`, `lanelet_length`, `has_stop_line`, `has_traffic_light_stop_line`, `has_adjacent`, `is_junction`, `previous_of`, `following_of`. |
| `Binding` (Protocol), `StopLineOffsetBinding`, `parse_binding` | Per-match parameter derivation (e.g. compute `ego.spawn_s` from a stop-line offset). |
| `load_lanelet2_map` | Lightweight Lanelet2 loader used outside of CARLA. |

A scenario whose `sweep` holds a `route` search instead of `constraints` is
expanded over the routes of the map that match it (`expand_route`, one case per
match); see [Logical Scenarios from Routes](logical_scenarios.md).

## Logical scenarios (`autoware_carla_scenario.route`)

See [Logical Scenarios from Routes](logical_scenarios.md).

| Symbol | Description |
|--------|-------------|
| `parse_route_search` -> `RouteSearchSpec` | Read a route search (`LaneSegmentSpec` / `JunctionSegmentSpec` segments, `Range` bounds, ego placement, `max_matches`, `seed`). No map needed. |
| `RouteMatch`, `RouteSegmentMatch` | A concrete route: lanelet ids, start / end s, each segment's route-s span and lanelets; `anchor_s(anchor)`, `to_config()` / `from_config()` (the `scenario.route.*` keys). |
| `parse_anchor` | `start`, `end`, `segment:K:start|end`, `junction:K:entry|exit`. |
| `route.search.find_route_matches` | Every route of a Lanelet2 map matching a search, sorted (or shuffled by `seed`). |
| `route.frame.RouteFrame`, `ego_placement` | A match on its map: `locate(s)`, `point(s)`, `project(x, y)`; where the ego spawns and its goal is. |
| `route.positions.resolve_route_pose` | Place a route pose against a frame and the ego's route s. |
| `set_scenario_route`, `scenario_route`, `scenario_route_frame`, `clear_scenario_route` | The route the running scenario is about. |
| `route.geometry.MapFeatures` | The per-lanelet questions a search asks (lanes beside, opposite lane, junction members and approaches, crosswalks), cached. |

The plugin is registered with Hydra under
`hydra/sweeper=lanelet_constraint`; see
`src/hydra_plugins/autoware_scenario_sweeper/`.

## Result viewer (`autoware_carla_scenario.ui`)

Used internally by the `viewer` CLI. The web app is the supported
surface, but the helper modules are importable for tooling:

| Symbol | Description |
|--------|-------------|
| `ui.app` | FastAPI application object and route handlers. |
| `ui.scanner` | Discover sessions / scenarios under `outputs/` and `multirun/`, build condition trees. |
| `ui.runner` | Background `subprocess.run(["uv", "run", "scenario", ...])` orchestration with thread-safe progress. |
| `ui.sweep_resolver.resolve_sweep` | Resolve a sweep without launching CARLA. |
| `ui.models` | Pydantic models (`SessionSummary`, `SessionItem`, `ConditionNode`, `ScenarioResultView`, `RunProgress`). |

## Scenario authoring (`autoware_carla_scenario.authoring`)

The declarative authoring layer behind the [Scenario Editor](scenario_editor.md).
None of it imports CARLA or lanelet2, so a document can be loaded, validated,
compiled and exported anywhere.

| Symbol | Description |
|--------|-------------|
| `ScenarioDocument` | The Scenario IR: entities, actions, assertions, an optional `route` search (a [logical scenario](logical_scenarios.md)), and a `ui` block that is presentation only. |
| `RouteSearch`, `LaneSegment`, `JunctionSegment`, `LengthRange`, `CountRange` | The route search as the IR states it; `RouteSearch.to_sweep_dict()` is what the route search reads. |
| `Entity`, `SpawnSpec`, `SValue`, `BindingRef`, `GoalSpec`, `EgoDriver` | Actors, how they spawn (fixed lanelet, or a constraint search with an optionally derived offset), and — for the ego alone — which stack drives it and where it is sent. |
| `LaneletChoice`, `LaneletSlot`, `ScenarioDocument.lanelet_slots` | Fixed or searched, and one view over every place a document names a lanelet — a spawn, a goal, an action's or a condition's `lanelet` parameter — so the picker, the validator and the Hydra config read one answer. |
| `ActionNode`, `ConditionNode`, `ConstraintNode` | Recursive IR nodes; a node's meaning comes from its registry spec, not from a `type` switch. |
| `ActionSpec`, `ConditionSpec`, `ConstraintSpec`, `BindingSpec`, `FieldSpec`, `ConditionVisual` | Metadata describing how a primitive is presented, edited and built. |
| `register_action_spec`, `register_condition_spec`, `register_constraint_spec`, `register_binding_spec` | Add a primitive; the GUI, validation and the compiler pick it up with no template change. |
| `validate_document` -> `ValidationReport` | Metadata-driven validation; errors block export, warnings do not. |
| `compile_document` -> `CompiledScenario` | Resolve entity ids to CARLA roles and type-check parameters, without importing CARLA. |
| `authoring.builders` | Factories that turn a compiled plan into the framework's own `BaseAction` / `BaseCondition`. Imports CARLA lazily. |
| `build_scenario_config`, `dump_scenario_config` | Render a document as the framework's Hydra scenario config, including a `sweep` section the existing sweeper understands. |
| `export_package` -> `ExportResult` | Write a reproducible Scenario Package and the wheelhouse built from it; raises `PackageExportError` (leaving nothing behind) when locking, verification or the wheelhouse fails. |
| `build_wheelhouse` -> `Wheelhouse` | Resolve a locked package's whole dependency graph into wheels that `pip install --no-index --find-links` can install with no uv, git or network; raises `WheelhouseError`. |
| `DraftStore`, `Draft`, `load_document`, `save_document` | YAML persistence for editor drafts and exported documents. |
| `new_document`, `blank_document` | Starter documents. |

## Declarative runtime (`autoware_carla_scenario.declarative`)

| Symbol | Description |
|--------|-------------|
| `DeclarativeScenario` | A `BaseScenario` whose content comes from a `ScenarioDocument`. Registers the same pre/post-tick actions and pass/fail conditions a hand-written scenario would. Imports CARLA. |
| `DeclarativeScenarioConfig` | Hydra config group: `document_path`, `timeout_seconds`, `spawn_overrides` (the addressable per-entity spawn keys a sweep drives), `param_overrides`, and `route` (a logical scenario's route match, as `scenario-expand` writes it). |

## Scenario editor (`autoware_carla_scenario.editor`)

Used by the `scenario-editor` CLI. A separate application from `ui`; neither
imports the other.

| Symbol | Description |
|--------|-------------|
| `editor.app.create_app` | Build the FastAPI application (draft and export directories are arguments). |
| `editor.routes` | HTML-partial routes driven by htmx. |
| `editor.service.EditorService` | Every document mutation the editor performs, testable without a web client. |
| `editor.map_preview.evaluate_slot` | Evaluate one lanelet field's constraints with the framework's own sweeper, and decide which lanelets the viewer outlines. |
| `editor.map_preview.lanelet2_source` | The `.osm` the editor serves to `simple_lanelet2`'s wasm map viewer. |
| `editor.app.map_viewer_url` | Where that viewer is loaded from (`SCENARIO_EDITOR_MAP_VIEWER`). |
| `editor.forms.parse_params` | Turn a form submission into typed IR parameters, driven by `FieldSpec` metadata. |

## Utilities (`autoware_carla_scenario.utils`)

| Symbol | Description |
|--------|-------------|
| `find_nearest_traffic_light` | Find the nearest CARLA traffic-light actor for a Lanelet2 traffic-light id. |
| `get_signal_ids_for_controller` | Map an OpenDRIVE controller id to its signal ids. |
| `lanelet2_traffic_light_id_to_opendrive_controller_id` | ID translation between Lanelet2 regulatory elements and OpenDRIVE controllers. |
| `get_stop_line_linestrings` | Collect Lanelet2 stop-line `LineString3d` objects for a lanelet. |

## CLI entry points

Defined in `pyproject.toml`:

| Command | Module |
|---------|--------|
| `scenario` | `autoware_carla_scenario.examples.run:main` |
| `detect-no-3d-model` | `autoware_carla_scenario.tools.detect_no_3d_model_lanelets:main` |
| `viewer` | `autoware_carla_scenario.ui:main` |
| `scenario-editor` | `autoware_carla_scenario.editor:main` |
| `scenario-new` | `autoware_carla_scenario.scaffold.generator:main` |
| `scenario-coverage` | `autoware_carla_scenario.coverage.report:main` |
| `scenario-odd` | `autoware_carla_scenario.odd.cli:main` |

The `scenario` command also exposes Python-level helpers in
`autoware_carla_scenario.examples.run` for downstream packages:

| Symbol | Description |
|--------|-------------|
| `register_scenario(name, scenario_cls, config_cls)` | Register a built-in-style scenario class under a Hydra `scenario.name`. |
| `register_scenario_builder(name, builder)` | Register a custom builder when the constructor signature differs. |
| `get_scenario_registry()` | Return a copy of the registry. |
| `build_ego_and_spawn(cfg)` | Build `(EgoConfig, Lanelet2Pose, GroundProjectionConfig)` from a resolved Hydra config. |
| `build_scenario(cfg, *, build_scenario_fn=None)` | Look up the registered builder and instantiate the scenario. |
| `run_scenario(cfg, ...)` / `run_scenario_with_queue(...)` / `run_batch(...)` / `main()` | Programmatic execution paths used by Hydra and the glob batch dispatcher. |
