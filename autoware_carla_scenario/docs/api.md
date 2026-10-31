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
| `CarlaServerManager` | `server` | Starts, reuses, and stops the CARLA UE5 process. Reads `CARLA_EXECUTABLE`, else launches the CARLA `scenario-setup` installed. A server it launched is stopped when the launching process dies, even by SIGKILL; one it reused never is. |
| `CarlaScenarioFixture` | `pytest_fixtures` | Helper that registers a scenario into a queue at import time and exposes a session-scoped pytest fixture for its `ScenarioResult`. |
| `EGO_ROLE_NAME` | `constants` | Reserved CARLA `role_name` used for the ego actor. |
| `EntityRole` | `entity_role` | Validated `role_name` wrapper for CARLA actors. Factories: `EntityRole.ego()`, `EntityRole.npc(n)`. |

`CameraRecorder` (the video renderer) was removed in 15.0.0 with the video
itself: a run's trajectory log is what a run is seen from
([Seeing a run](usage.md#seeing-a-run)).

## Coverage (`autoware_carla_scenario.coverage`)

See [Coverage](coverage.md).

| Symbol | Purpose |
|--------|---------|
| `BaseScenario.register_cover(name, expression, *, unit, range, every, buckets, values, ignore, event, text, target, cover_by, min_stay)` | Declare a cover item, after `cover()` in OpenSCENARIO DSL. |
| `BaseScenario.register_cross(name, items, *, text, target, cover_by, min_stay)` | Cross coverage of cover items sampled on the same event. |
| `SamplingEvent` | When an item is sampled: `START`, `END` (default) or `TICK`. Re-exported from the top-level package. |
| `CoverItem`, `CrossItem` | The item definitions `register_cover()` / `register_cross()` build. |
| `autoware_carla_scenario.metrics`: `BUILT_IN_METRICS`, `Metric`, `read_metric(key, world)`, `COLLISION_ANY`, `COLLISION_AT_FAULT`, `OFFROAD`, `WRONG_LANE`, `MIN_DISTANCE_TO_OBSTACLE_M`, ... | How the ego drives, read off the world per frame: AlpaSim's metrics. An outcome, which no ODD maps onto. See [Metrics](metrics.md). |
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
| `default_odd()` | The built-in ODD: openodd's bundled `default` (every attribute, no modules), measured in CARLA. |
| `register_odd(name, builder)`, `resolve_odd(spec)` | Name an ODD; turn a name, `.yaml` path or `module:function` into one. Packages register through the `autoware_carla_scenario.odds` entry point group (a function calling `register_odd`). |
| `load_odd_file(path)`, `load_odd_binding(path)`, `load_openodd(*sources, bindings, name, text, situations)` | Read ASAM OpenODD 1.0 YAML (`IMPORT`, `TAXONOMY`, `MODULES`, `ODD`), with a binding file naming the probes. |
| `GitSource(url, rev, path)`, `GitSourceError` | An OpenODD file in a git repository at a revision, for `load_openodd()` or a binding file's `openodd` list. |
| `ego_speed_kph`, `speed_limit_kph`, `lanelet_location`, `lanelet_subtype`, `lanelet_speed_limit_kph`, `in_junction`, `lane_count`, `illumination`, `rain`, `fog`, `traffic_density`, `pedestrian_nearby` | Built-in probes. |
| `lanelet_in_junction(lanelet)`, `lanelet_road_type(lanelet)`, `lanelet_speed_limit(lanelet)` | What the Lanelet2 probes read off one lanelet: its junction, road type (`subtype`) and speed limit (km/h). |
| `scenario_measure(key)`, `ScenarioMeasure` | The probe mapping an attribute onto the running scenario's measure *key* (`measure:` in a binding file). See [Scenario measures](odd.md#scenario-measures). |
| `typecheck.typecheck_odd(builder)` | Compile a Python ODD with Codon. |
| `plan_route_coverage(odd, scenario_or_route, *, lanelet_map, routing_graph, name)` | Check a planned route against the ODD on the Lanelet2 map: the verdict per lanelet (inside, outside, undecided) and the expected metres per bucket. Takes a scenario, a `PlannedRoute(start, goal, via, name)` or lanelet ids. Raises `RouteError` when the route cannot be planned. See [Checking a planned route](odd.md#checking-a-planned-route). |
| `combine_route_coverage(odd, routes)` | The expected coverage of several routes, and the buckets no route reaches. |
| `plan_route(route, lanelet_map, routing_graph)` | The lanelets of a `PlannedRoute`, with the metres driven on each. |
| `UNDECIDED` | What a probe's `on_lanelet` returns for a value only the run can tell. |
| `OddSampler(odd, knobs, *, controls, seed, strategy, coverage, max_tries, weights)` | Draws concrete cases inside the ODD: `.sample(count)` gives `OddSample(index, overrides, buckets, values, situation)`. `strategy="coverage"` draws the least covered buckets and the uncovered situations first; `weights={module: share}` (`strategy="weighted"`) puts that share of the cases in each module. See [Sampling scenarios from the ODD](odd.md#sampling-scenarios-from-the-odd). |
| `OddKnob(key, *, values, scale, offset, integer, range)`, `knobs_from_mapping(raw)`, `DEFAULT_KNOBS` | A control: the config key that sets a measure (or, as a sweep knob, an attribute), and what it can stage. Scenarios declare theirs under `controls` in their config, by measure key. |

## Conditions (`autoware_carla_scenario.conditions`)

All conditions inherit from `BaseCondition` and return a
`ScenarioResult` from `check(world, elapsed)` once triggered (or
`None` when not yet triggered). At the end of a run the runner calls
`close()` on every pass and fail condition and every action's trigger: a
condition that holds something in the simulator -- `CollisionCondition`'s
listening sensor -- stops and destroys it there, through its own handle.
`close()` does nothing by default, and composite conditions close their
children.

| Symbol | Description |
|--------|-------------|
| `BaseCondition` | Abstract base class. Subclasses override `_check()`. |
| `ScenarioResult` | Pass / fail outcome with message, elapsed time, and per-condition statuses. |
| `ConditionStatus` | Per-condition leaf record used for reporting. |
| `AlwaysTrueCondition` | Default trigger for actions. |
| `AndCondition`, `OrCondition`, `NotCondition` | Logical combinators. `AndCondition` and `OrCondition` check every child on every tick, so a stateful child (`PersistentCondition`, `StickyCondition`) stays current wherever it sits; an AND of `StickyCondition`s therefore says each state was visited, in any order. An AND fires only once every child has fired: a pending child holds back a failing one, so an AND holding a timeout or a collision fails the run only once its other children have fired. `NotCondition(child, holds_when_pending=True)` is the logical NOT -- it holds while the child has not fired -- where the default flips a fired child's verdict; with `OrCondition` over `EntityLanePositionCondition`s it says "on none of these lanelets" ([cut-in](scenarios/cut_in.md#staying-on-the-lanes-ahead)). |
| `StickyCondition`, `PersistentCondition` | Latch / persist a child condition's truth value. |
| `TimeoutCondition`, `ElapsedTimeCondition` | Time-based triggers. |
| `MetricCondition(metric, rule, value, tolerance, *, label)` | A built-in metric -- how the ego drives -- against a threshold; usually a fail condition. See [Metrics](metrics.md). |
| `EntityDistanceCondition`, `TimeToCollisionCondition` | Relative conditions between two entities: separation, and time to collision along the line joining them. |
| `CollisionCondition`, `EntityExistenceCondition` | Safety checks. |
| `OffRoadFractionCondition(fraction, entity_name=None, *, label)` | Fails once at least `fraction` (in `(0, 1]`) of the sample points of a 5 x 3 grid over an entity's (default: the ego's) bounding box -- edges and corners included, not an area -- are on no driving lane. The pass/fail counterpart of the `offroad` metric, with a tolerance. See [Metrics](metrics.md#off-the-road-as-a-fail-condition). |
| `TrafficSignalAction(..., route_junction=0, route_approach="same", route_turn="any")` | With `lanelet2_traffic_light_ids=TrafficLightTarget.ROUTE`: sets the lights of the paths through junction `route_junction` of the scenario's route that enter it from `route_approach` and turn `route_turn`, found on the map the run is on (`route.positions.junction_traffic_lights`). |
| `PlaceClearCondition(position, length, width, margin=0.3, *, label)` | Passes while no vehicle's or walker's bounding box comes within `margin` of a `length` x `width` rectangle at `position` (a `Lanelet2Pose`, facing along the lane turned by its `heading`; or any position a trajectory vertex takes -- a route pose is placed on the scenario's route when the place is first looked at). Actors parked under the map do not count; one put there on this tick does. The `place_clear` card; typically guards where an entity appears. |
| `TrafficSignalCondition` | Traffic-light state check: fires while every matching light shows the expected state, and is pending otherwise. |
| `RouteProgressCondition` | An entity's (default: the ego's) distance along the scenario's route, from an anchor (`junction:0:entry`, ...), compared with a rule; robust to lane changes. See [Logical Scenarios from Routes](logical_scenarios.md). |
| `ComparisonRule`, `ScalarComparisonRule`, `compare` | Numeric comparison primitives. `compare(actual, rule, value, tolerance)` is the underlying helper. |
| `find_actor_by_role_name`, `find_actor_in_list` | Helpers for locating CARLA actors by role. `find_actor_in_list` is reachable via `autoware_carla_scenario.conditions`. |

### Composition conditions (`autoware_carla_scenario.conditions.composition`)

These build on `CompositionCondition`, which composes a child condition
tree internally:

- `EntityLanePositionCondition`
- `SpeedCondition`, `SpeedDirection`, `SpeedCoordinateSystem`
- `StandstillCondition`
- `TurnIndicatorCondition`, `TurnIndicatorState` -- the turn indicators a vehicle shows (OFF / LEFT / RIGHT / HAZARD), read off its CARLA light state
- `TemporaryStopCondition`
- `WaypointCondition`, `WaypointCheckType`

## Actions (`autoware_carla_scenario.actions`)

| Symbol | Description |
|--------|-------------|
| `BaseAction` | Abstract base. Owns a trigger `BaseCondition`, a `TickTiming`, and an `execute(world)` side effect. |
| `TickTiming` | Enum: `PRE_TICK` / `POST_TICK`. |
| `TurnAction`, `TurnDirection` | Steer the ego through left / right turns via the CARLA TrafficManager route hints. |
| `TurnIndicatorAction` | Light a vehicle's turn indicators (`TurnIndicatorState`: `OFF` / `LEFT` / `RIGHT` / `HAZARD`), its other lights as they are; they stay so until set again. |
| `CommunicationLossAction` | Stop data between the HILS bridge and an ECU under test (`ego.entity=hils`) while the action runs (until `until` fires or `duration` passes, else to the scenario's end): `messages` by name (a CAN frame, a LiDAR's `scan`) or everything of a `device` (or of every device), `to_ecu` / `from_ecu` / `both`; no effect on any other ego. See [HILS](hils.md#communication-loss). |
| `LaneChangeAction`, `LaneChangeDirection` | Trigger a TrafficManager lane change. |
| `TrafficSignalAction`, `TrafficLightTarget` | Set traffic-light states (e.g. all RED, all GREEN, or a specific actor). |
| `RoutingAction`, `RoutingMode` | Send the ego, a vehicle or a pedestrian to a goal through waypoints (a goal with Lanelet2 waypoints, or a `trajectory`); `mode` is `DRIVER` (its own driver, Lanelet2 legs, paced by speed to the waypoints' times), `FOLLOW` or `POSITION` (straight legs, as OpenSCENARIO follows a trajectory); `speed` paces the segments no time does; `held_vertex` names the vertex a waypoint condition is holding it at; it routes an entity in the scene (bring a hidden one in with `SpawnAction`), starting where it stands. See [Trajectories and Recorded-Scene Replay](trajectory.md). |

### Adversarial actions (`autoware_carla_scenario.actions.adversarial`)

TeraSim's adversities, on a named NPC (exported from the top-level
package, not from `actions`, which would pull in the Lanelet2 map layer).
Each takes the entity and a required `location` (`AdversityLocation`: `ANY`,
`INTERSECTION`, `HIGHWAY`, `ROUNDABOUT`); its default trigger is an
`AdversityFeasibleCondition` of its kind there. When it fires, the command is
planned on the map and driven (`following_mode`: `FOLLOW` by default, or
`POSITION`), and the vehicle is handed back to its traffic backend at the
end. A `condition` given instead fires the action, and the adversity's
trigger is asked on that tick: when it does not hold, nothing is done.
`thresholds` overrides TeraSim's numbers.

| Symbol | Description |
|--------|-------------|
| `AdversarialBrakeAction` | Brake hard in front of a follower that keeps coming (`BRAKE_CHECK`). |
| `IgnoreLeaderAction` | Drive on as if the close, slower leader were not there (`IGNORE_LEADER`). |
| `AdversarialCutInAction` | Change lane in front of a vehicle too close behind in the next lane (`CUT_IN`). |
| `AdversarialMergeAction` | Beside a merging lane with traffic on it (`MapFeatures.merges`: a lane that ends into the lane on its left, there or within 100 m), cut a merging vehicle off: one lane over, speed up as if it were not there, when that beats giving way by more than 1.5 m/s², for 2 s; two lanes over, change into the lane between in 1 s when nobody there is within 2 s of headway (`MERGE`). |
| `LaneChangeAbortAction` | Start into the next lane in front of a vehicle too close behind there, drifting half a lane toward it, and swerve back, within 1 s (`LANE_CHANGE_ABORT`). |
| `HeadOnAction` | Drive on at a leader coming the other way: `IgnoreLeaderAction`'s command, where the lanelets the vehicle and its leader are on start more than 120 degrees apart (`HEAD_ON`). |
| `RunTrafficRuleAction` | Run the red or yellow light the vehicle is stopping for (`TRAFFIC_RULE`). |
| `JaywalkAction` | Walk a pedestrian straight across the nearest road, off any crosswalk (`JAYWALKING`); `seed` seeds its speed. |
| `PedestrianRunRedLightAction` | Walk a pedestrian waiting at a crosswalk (at most 0.2 m/s, within 3 m) across it, end to end, when the crosswalk's Lanelet2 traffic light is red (`RUNNING_RED_LIGHT`). |
| `StopCrossingAction` | Stand a pedestrian on a crosswalk still for N(2 s, 0.5 s), then walk it a crosswalk's length on along its heading (`STOP_CROSSING`); `seed` seeds the wait. |
| `SpawnAction`, `clear_props` | Put something into the scene mid-run (`autoware_carla_scenario.actions.spawn`, the counterpart of `DespawnAction`): a vehicle or pedestrian spawned hidden, brought in at `position` (any trajectory-vertex frame; `None`: where it was declared) moving at `speed`, or with `name` a prop (`blueprint`, `on_ground`; PropAction until 1.0); `shift_max` takes the nearest clear place that far on along its lane when its own is taken. `DespawnAction(name=...)` removes a prop; the runner clears every prop between runs (`clear_props`). |
| `StalledObjectAction`, `collision_objects` | TeraSim's scheduled static adversities (`autoware_carla_scenario.actions.stalled_object`): a vehicle the scenario spawned (out of the world, `spawn.hidden`, best) stands still at a pose -- a `Lanelet2Pose`, or an xy+yaw `MapPose` / `CarlaWorldPose` -- from `start_time` to `end_time` (scenario seconds; `None`: never removed), and is out of the world before and after. A trajectory run (the engine of `RoutingAction`'s `position` mode) of a held pose with `hidden_outside_trajectory`. `collision_objects` is two of them, for a collision's aftermath. |
| `ConstructionZoneAction`, `ConstructionSettings`, `URBAN_CONSTRUCTION` | TeraSim's construction zone (`autoware_carla_scenario.actions.construction_zone`): warning signs, an entry taper of cones, a buffer, the work zone, an exit taper and termination signs along a lane from `start_s` to `end_s`, as CARLA props from `start_time` to `end_time`; background traffic in it is taken off the road and the lane closed to the backend (SUMO). `layout="urban"` (preset: `URBAN_CONSTRUCTION`) lays out TeraSim's urban variant on an intersection approach instead: a short warning zone, cones along the curb and a merge taper that shuts the lane before the intersection. See [Adversarial Actions and NADE](adversity.md#construction-zones). |
| `AdversarialTrafficAction`, `NadeAdversity`, `NadeSettings` | TeraSim's NADE as an environment action that runs for the whole run. Each tick, the NPC or pedestrian near the ego whose adversity (the first of `adversities` whose trigger fires) would challenge the ego carries it out, with the importance-sampling probability `q = clip(p x magnitude, 0, max_is_probability)`, or a fixed one. One adversity at a time, then a cooldown. An adversity no victim can brake out of is unavoidable: its `p` is cut by 3e-4 and its magnitude multiplied by 40 (`adversity.avoidance`). With `challenge_npcs`, an NPC victim avoids it (emergency brake) or accepts it, with its own IS weight. `weight` keeps the run's likelihood ratio. It drives the registered entities and, with `drive_background` (default), the traffic backend's background vehicles (`TrafficBackend.background_actors`: TrafficManager background traffic, SUMO's ambient vehicles), each taken over while it carries out an adversity or responds to one and handed back after (#417). |
| `AdversityFeasibleCondition`, `AdversityKind`, `AdversityLocation` | Holds when the entity may carry out a `kind` adversity at `location` (`autoware_carla_scenario.conditions.adversity_feasible`). |

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
| `TrajectoryFollowingMode` | `POSITION` (kinematic replay) or `FOLLOW` (a controller tracks it): the adversarial actions' mode. |

Recorded scenes (T4) are transcribed into documents outside the framework, by
the separate `scene_to_scenario_transpiler` package (`scenario-import-t4`).

## Sensors (`autoware_carla_scenario.sensor`)

| Symbol | Description |
|--------|-------------|
| `CameraSensorBase`, `CameraSensorConfig` | Provider-agnostic camera sensor interface. |
| `CarlaCameraSensor`, `CarlaCameraSensorConfig` | CARLA RGB camera implementation, used by the driver rig. |

## External driver (`autoware_carla_scenario.driver`)

Connects the ego to a driving policy served over alpasim's
`egodriver.EgodriverService`. See [External Driver Interface](driver_interface.md).

| Symbol | Description |
|--------|-------------|
| `BaseEgoDriverClient` | Transport-agnostic client interface: session lifecycle, observation submission, `drive`. |
| `EgoDriverGrpcClient` | gRPC implementation against the vendored alpasim protos. |
| `DriverClientConfig`, `DriverCameraConfig` | Connection settings, policy cadence, and the camera rig streamed to the policy. |
| `ControlConfig`, `TrajectoryFollower`, `VehicleCommand` | Pure pursuit plus a port of Autoware's `autoware_pid_longitudinal_controller` (or the `time_indexed` tracker), turning a plan into `carla.VehicleControl`. |
| `Pose`, `Trajectory` | Rigid-transform primitives with protobuf conversions (right-handed frame). |
| `EgoObservation`, `DriveOutcome` | The state sent to the policy and the plan it returns. |
| `RendererDataBuilder` | Collects CARLA ground truth (governing traffic light, other vehicles, speed limit) into the `renderer_data` extension payload. |
| `driver.stop_lines.StopLines` | Which traffic light governs the ego (the lane graph walked ahead to the next junction) and the distance along its heading to where it stops for it (the stop waypoint walked to its junction's mouth); `RendererDataBuilder` reports both. |

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
| `lanelets_ahead` | A lanelet and the lanelets that follow it in the routing graph (no lane changes, every branch), up to a distance: the lanes a vehicle keeps to without changing lanes. |
| `get_stop_line_poses`, `get_stop_line_poses_with_following` | Resolve stop-line `Lanelet2Pose`s for a lanelet (optionally including following lanelets). |

## Entities (`autoware_carla_scenario.entity`)

| Symbol | Description |
|--------|-------------|
| `VehicleEntity`, `VehicleEntityConfig` | Generic vehicle actor with retry-aware spawn. |
| `EgoVehicle` | Subclass with the fixed `EGO_ROLE_NAME`, plus the `on_scenario_start` / `on_tick` / `on_scenario_end` lifecycle hooks `ScenarioRunner` calls. |
| `ExternallyDrivenEgo` | Base of every ego driven by a stack of its own rather than the TrafficManager: opts out of autopilot, refuses TrafficManager manoeuvres, and carries the termination request. |
| `BridgeHandshake` | The mission and readiness handshake with a stack over the `AutowareBridge`, optionally launching it per scenario; held by an ego (`entity.bridged_ego`). |
| `BridgedEgoEntity` | The ego that is nothing but a `BridgeHandshake`: the stack drives. |
| `AutowareEgoEntity` | The `BridgedEgoEntity` driven by Autoware through `autoware_carla_interface`. See [Autoware](autoware.md). |
| `CarlaDriverEntity` | Drives the ego from an external policy's plan over the `egodriver` gRPC contract; given a `bridge`, starts the policy once a stack serving its world is ready. See [External Driver Interface](driver_interface.md). |
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
| `Constraint` (ABC) | Base class of the constraints above, with an abstract `evaluate(lanelet)`. A constraint of your own may derive from it or only define `evaluate()`: the matching takes either. |
| `EqualsConstraint`, `InSetConstraint`, `OnRouteConstraint`, `LaneletLengthConstraint`, `HasStopLineConstraint`, `HasTrafficLightStopLineConstraint`, `HasAdjacentConstraint`, `IsJunctionConstraint`, `PreviousOfConstraint`, `FollowingOfConstraint` | Atomic constraints. |
| `AndConstraint`, `OrConstraint`, `NotConstraint` | Combinators. |
| `parse_constraint`, `find_matching_lanelets` | YAML-to-`Constraint` parsing and lanelet matching. The corresponding YAML `type:` keys for the atomics above are `equals`, `in_set`, `on_route`, `lanelet_length`, `has_stop_line`, `has_traffic_light_stop_line`, `has_adjacent`, `is_junction`, `previous_of`, `following_of`. |
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
| `ui.cases` | Which files are a session's cases: the stem each case's files share (a SLURM batch's manifest, or matched against `batch_results.json`). `session_dir` / `job_dir` / `contained` are the one check every path the viewer builds from a URL goes through: resolved, it must stay under `multirun/` or `outputs/`, else the route answers 404. |
| `ui.results` | A session's cases with their case, coverage and metrics (`load_results`, `case_metrics`), the cases in an ODD cell, and sessions compared case by case and cell by cell (`compare`). |
| `ui.case_paths` | The path guard (`contained`, `session_dir`, `job_dir`), `CaseFiles` and the stems of a directory's cases, which `ui.cases` re-exports; checked by the library check. |
| `ui.comparison` | Sessions compared case by case and ODD cell by cell (`compare`, `default_axes`, `varying_attributes`, `METRICS`) on typed `ComparedCase` records, which `ui.results` builds from a case's coverage document (`CaseResult.compared`) and re-exports; checked by the library check. |
| `ui.drive_metrics` | The box gaps and the running metrics of the ego's ticks (`DriveMetrics`) that `ui.results.case_metrics` reads a trajectory log with; checked by the library check. |
| `ui.slurm_launch` | Submitting a batch with `launcher=slurm` from the viewer (`build_command`, `start_submit`), a SLURM batch's progress (`batch_status`), and `scenario-slurm` run on it. |
| `ui.slurm_options` | The `launcher.*` keys the viewer sets (`SlurmOptions`), the checks of each field (`check_*`), the Hydra overrides they become (`SlurmOptions.overrides`) and the `scenario` command line that submits the batch (`build_command`, given how `scenario` is started); `ui.slurm_launch` reads them off the run panel (`SlurmOptions.from_request`) and re-exports them. |
| `ui.recordings` | A case's Rerun recording, made on demand with `autoware_carla_scenario_rerun` when installed, and where the Rerun web viewer is loaded from. |
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
