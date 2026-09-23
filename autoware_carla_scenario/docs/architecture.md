# Software Architecture

This document describes the software architecture of the `autoware-carla-scenario` package, with a focus on the **scenario execution system** and the **result viewer**.

## System Overview

The package consists of four major subsystems:

```mermaid
graph TB
    subgraph CLI["CLI Entry Points"]
        scenario["scenario command<br/>(Hydra)"]
        viewer_cmd["viewer command<br/>(FastAPI + Uvicorn)"]
        detect["detect-no-3d-model<br/>(argparse)"]
    end

    subgraph Execution["Scenario Execution System"]
        queue["ScenarioQueue"]
        runner["ScenarioRunner"]
        base["BaseScenario"]
        server["CarlaServerManager"]
    end

    subgraph Evaluation["Condition & Action System"]
        conditions["Conditions"]
        actions["Actions"]
    end

    subgraph Viewer["Result Viewer"]
        app["FastAPI App"]
        scan["Scanner"]
        ui_runner["Runner (subprocess)"]
        sweep_res["Sweep Resolver"]
    end

    subgraph Support["Support Modules"]
        coord["Coordinate Transforms"]
        entity["Entity Management"]
        kin["Kinematics"]
        sweeper["Lanelet Constraint Sweeper"]
        recorder["Camera Recorder"]
    end

    scenario --> queue
    viewer_cmd --> app
    queue --> runner
    queue --> server
    runner --> base
    runner --> conditions
    runner --> actions
    runner --> recorder
    base --> entity
    base --> coord
    app --> scan
    app --> ui_runner
    app --> sweep_res
    ui_runner -->|"uv run scenario<br/>(subprocess)"| scenario
```

---

## Scenario Execution System

The scenario execution system is responsible for the full lifecycle of a CARLA scenario test: server management, map loading, actor spawning, tick-loop execution, condition evaluation, recording, and cleanup.

### Architecture Diagram

```mermaid
graph TB
    subgraph Entry["Entry Layer"]
        main["main()<br/>examples/run.py"]
        hydra["Hydra Config<br/>Compose API"]
    end

    subgraph Orchestration["Orchestration Layer"]
        SQ["ScenarioQueue"]
        SR["ScenarioRunner"]
        CSM["CarlaServerManager"]
    end

    subgraph Scenario["Scenario Layer"]
        BS["BaseScenario (ABC)"]
        IP["IntersectionPassingScenario"]
        LC["LaneChangeScenario"]
        TLC["TrafficLightComplianceScenario"]
        TS["TemporaryStopScenario"]
    end

    subgraph Traffic["Traffic Backend"]
        tb["TrafficBackend<br/>(traffic_manager | none | third-party)"]
    end

    subgraph CARLA["CARLA Simulator"]
        world["carla.World"]
        tm["TrafficManager"]
        rec["Native Recorder"]
    end

    main --> hydra
    hydra --> SQ
    SQ --> CSM
    SQ --> SR
    SR --> BS
    BS --> IP & LC & TLC & TS
    SR --> world
    SR --> tb
    tb --> tm
    SR --> rec
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| **`CarlaServerManager`** | `server.py` | Start/stop/reuse the CARLA UE5 process. Manages lifecycle via process groups. |
| **`ScenarioQueue`** | `scenario_queue.py` | Batch execution of multiple scenarios. Owns server and runner lifecycle. Context manager. |
| **`ScenarioRunner`** | `scenario_runner.py` | Single-scenario execution: sync mode, tick loop, condition evaluation, recording, cleanup. |
| **`BaseScenario`** | `scenario_base.py` | Abstract base. Subclasses implement `setup()` and `is_done()`. Registers conditions, actions, entities. |
| **`TrafficBackend`** | `traffic/` | Owns the vehicles the scenario did not author and answers their manoeuvre intents. `traffic_manager` by default; see [Traffic Backends](traffic_backends.md). |

### Execution Lifecycle

The following sequence diagram shows the complete lifecycle of a single scenario execution:

```mermaid
sequenceDiagram
    participant CLI as CLI (main)
    participant SQ as ScenarioQueue
    participant CSM as CarlaServerManager
    participant SR as ScenarioRunner
    participant BS as BaseScenario
    participant W as carla.World
    participant TB as TrafficBackend
    participant TM as TrafficManager

    CLI->>SQ: add(scenario)
    CLI->>SQ: __enter__() / start()
    SQ->>CSM: start()
    Note over CSM: Reuse existing server<br/>or launch new process
    SQ->>SR: new ScenarioRunner(server, ...)
    SQ->>SR: load_map_by_name() or<br/>load_map_by_overwriting_xodr()
    SQ->>SR: Initialize MapManager<br/>(Lanelet2 + OpenDRIVE)

    CLI->>SQ: run_all()
    loop For each scenario
        SQ->>SR: run_scenario(scenario)

        rect rgb(240, 248, 255)
            Note over SR: Phase 1: Setup
            SR->>W: apply_settings(sync=True, dt=0.05)
            SR->>TB: prepare(TrafficContext)
            TB->>TM: set_synchronous_mode(True)
            TB->>TM: set_random_device_seed(seed)
            SR->>BS: set_client(client, tm_port)
            SR->>BS: set_traffic_backend(backend)
            SR->>BS: setup()
            Note over BS: Spawn NPCs, register<br/>conditions & actions
            SR->>W: Spawn ego vehicle
            SR->>BS: register_fail_condition(EntityExistence)
        end

        rect rgb(255, 248, 240)
            Note over SR: Phase 2: Warm-up (5 ticks)
            loop 5 times
                SR->>W: tick()
            end
            SR->>TB: start(world, skip_actor_ids)
            TB->>W: set_autopilot(True) on the vehicles it owns
            SR->>BS: set_initial_speed(ego)
        end

        rect rgb(240, 255, 240)
            Note over SR: Phase 3: Recording & Tick Loop
            SR->>W: start_recorder(path)
            SR->>BS: register_fail_condition(Timeout)
            loop Until done or condition met
                SR->>BS: pre_tick_actions.tick(world, elapsed)
                SR->>BS: pre_tick_callbacks(world)
                SR->>W: tick()
                SR->>BS: post_tick_actions.tick(world, elapsed)
                SR->>BS: post_tick_callbacks(world)
                SR->>BS: Check pass_conditions
                Note over SR: If any pass condition<br/>returns ScenarioResult → PASS
                SR->>BS: Check fail_conditions
                Note over SR: If any fail condition<br/>returns ScenarioResult → FAIL
                SR->>BS: is_done()?
                Note over SR: If True → PASS (default)
            end
        end

        rect rgb(255, 240, 240)
            Note over SR: Phase 4: Cleanup
            SR->>W: Destroy ego
            SR->>W: stop_recorder()
            SR->>TB: close()
            TB->>TM: shut_down()
            SR->>SR: Render video from recording
            SR->>W: reload_world()
        end

        SR-->>SQ: ScenarioResult
    end

    CLI->>SQ: __exit__() / stop()
    SQ->>CSM: stop() (if owned)
```

### Tick Loop Detail

The tick loop runs at a fixed 20 Hz (0.05 s per tick) in CARLA synchronous mode. Each tick follows a strict evaluation order:

```
┌───────────────────────────────────────────────────────────┐
│                     Single Tick Cycle                     │
├───────────────────────────────────────────────────────────┤
│  1. Pre-tick actions      → action.tick(world, elapsed)   │
│  2. Pre-tick callbacks    → callback(world)               │
│  3. world.tick()          → advance simulation by 0.05 s  │
│  4. Ego entity            → ego.on_tick(world, elapsed)   │
│  5. Traffic backend       → backend.tick(world, elapsed)  │
│  6. Post-tick actions     → action.tick(world, elapsed)   │
│  7. Post-tick callbacks   → callback(world)               │
│  8. Periodic logging      → ego OpenDRIVE position (1/s)  │
│  9. Pass conditions       → first satisfied → PASS & exit │
│  10. Fail conditions      → first triggered → FAIL & exit │
│  11. is_done() check      → True → PASS & exit            │
└───────────────────────────────────────────────────────────┘
```

**Evaluation semantics:**

- **Pass conditions** are checked first. The **first** condition to return a non-`None` `ScenarioResult` terminates the loop with a pass.
- **Fail conditions** are checked only if no pass condition was satisfied. The **first** triggered fail condition terminates the loop with a failure.
- If the loop exits via `is_done()` returning `True` with no condition triggered, the scenario is treated as **passed**.

### The scenario clock

`_ScenarioClock` supplies the `elapsed` every action, condition, entity and
backend receives, and the `elapsed_seconds` on a `ScenarioResult`. It reads
`world.get_snapshot().timestamp.elapsed_seconds` — **simulated** time, and
nothing else.

The world advances by `fixed_delta_seconds` per tick and the loop steps it as
fast as the slowest client allows (see `max_tick_rate_hz`), so how much
simulated time fits into a second of real time is a property of the host. A
scenario whose durations came off the wall clock would fire its triggers in
different places on a fast machine and a slow one — the determinism that
synchronous mode exists to provide.

That holds for `ScenarioRunner.timeout_seconds` as much as for a condition an
author wrote. It is registered as a default `TimeoutCondition` on every
scenario, so a scenario given sixty seconds is given sixty of *its own*
however fast the host runs.

Wall-clock protection against a **stuck** simulator lives outside the scenario
clock, where it belongs:

| Guard | Where | Covers |
|---|---|---|
| CARLA client RPC timeout (60 s) | `ScenarioRunner.__init__` | `world.tick()` stops returning |
| `sweep.job_timeout_seconds` (120 s) | sweeper config | a whole job overrunning |

The clock starts after the ego is ready, so an entity that needs an autonomy
stack to come up does not spend the scenario's timeout booting. Simulation
time does advance during that wait, which is why the start is measured rather
than assumed to be zero.

### ScenarioQueue: Batch Execution and Retry

`ScenarioQueue` wraps `ScenarioRunner` to support sequential execution of multiple scenarios:

```mermaid
graph LR
    subgraph ScenarioQueue
        direction TB
        A[Scenario 1] --> B[Cooldown]
        B --> C[Scenario 2]
        C --> D[Cooldown]
        D --> E[Scenario N]
    end

    subgraph Retry["Retry Logic"]
        direction TB
        R1["Attempt 1"] -->|fail| R2["Cooldown"]
        R2 --> R3["Attempt 2"]
        R3 -->|fail| R4["..."]
        R4 --> R5["Attempt N"]
        R5 -->|fail| R6["RuntimeError"]
    end
```

Key features:

- **Cooldown**: Configurable wait time (`cooldown_seconds`) between consecutive scenarios to allow CARLA server cleanup.
- **Retry**: Failed scenarios are retried up to `cooldown_max_retries` times before raising an error.
- **Server ownership**: Accepts an external `CarlaServerManager` or creates and owns one internally.
- **pytest integration**: `as_fixture()` generates a session-scoped pytest fixture with automatic skip when `CARLA_EXECUTABLE` is not set.

### Video Recording Pipeline

Video recording uses a two-pass approach:

```mermaid
graph LR
    subgraph Pass1["Pass 1: Native Recording (during scenario)"]
        T1["world.tick()"] --> R1["CARLA Native Recorder"]
        R1 --> L1["scenario.log file"]
    end

    subgraph Pass2["Pass 2: Video Rendering (after scenario)"]
        L1 --> RP["client.replay_file()"]
        RP --> CAM["RGB Camera Sensor<br/>(attached to ego)"]
        CAM --> FF["ffmpeg (H.264)"]
        FF --> MP4["scenario.mp4"]
    end
```

1. **During execution**: CARLA's native recorder captures all actor states to a `.log` file.
2. **After execution**: The recording is replayed in synchronous mode. An RGB camera sensor is attached to the ego vehicle at the same offset as the spectator camera. Frames are streamed to an `ffmpeg` subprocess for H.264 encoding via `CameraRecorder`.

`CameraRecorder` uses a frame queue (maxsize=2) to decouple sensor callbacks from the encoding thread, and writes frames synchronously after each `world.tick()` to avoid dropped frames.

---

## Condition and Action System

### Condition Architecture

Conditions evaluate scenario pass/fail criteria. All conditions inherit from `BaseCondition`:

```mermaid
classDiagram
    class BaseCondition {
        <<abstract>>
        +label: str
        +check(world, elapsed) ScenarioResult | None
        +get_details() dict
        +to_summary_dict() dict
    }

    class ScenarioResult {
        +passed: bool
        +message: str
        +elapsed_seconds: float
        +condition_statuses: list
    }

    class TimeoutCondition
    class ElapsedTimeCondition
    class CollisionCondition
    class EntityExistenceCondition
    class TrafficSignalCondition

    class EntityLanePositionCondition
    class EntityDistanceCondition
    class EntityPositionDistanceCondition
    class TimeToCollisionCondition
    class TimeHeadwayCondition
    class RelativeSpeedCondition
    class SpeedCondition
    class AccelerationCondition
    class StandstillCondition
    class TemporaryStopCondition
    class WaypointCondition

    class AndCondition
    class OrCondition
    class NotCondition
    class StickyCondition
    class PersistentCondition
    class AlwaysTrueCondition

    BaseCondition <|-- TimeoutCondition
    BaseCondition <|-- ElapsedTimeCondition
    BaseCondition <|-- CollisionCondition
    BaseCondition <|-- EntityExistenceCondition
    BaseCondition <|-- TrafficSignalCondition
    BaseCondition <|-- EntityLanePositionCondition
    BaseCondition <|-- EntityDistanceCondition
    BaseCondition <|-- EntityPositionDistanceCondition
    BaseCondition <|-- TimeToCollisionCondition
    BaseCondition <|-- TimeHeadwayCondition
    BaseCondition <|-- RelativeSpeedCondition
    BaseCondition <|-- SpeedCondition
    BaseCondition <|-- AccelerationCondition
    BaseCondition <|-- StandstillCondition
    BaseCondition <|-- TemporaryStopCondition
    BaseCondition <|-- WaypointCondition
    BaseCondition <|-- AndCondition
    BaseCondition <|-- OrCondition
    BaseCondition <|-- NotCondition
    BaseCondition <|-- StickyCondition
    BaseCondition <|-- PersistentCondition
    BaseCondition <|-- AlwaysTrueCondition

    BaseCondition --> ScenarioResult : returns
```

**Condition categories:**

| Category | Conditions | Description |
|----------|-----------|-------------|
| **Temporal** | `TimeoutCondition`, `ElapsedTimeCondition` | Time-based triggers |
| **Safety** | `CollisionCondition`, `EntityExistenceCondition` | Collision detection — with anything, or with a named entity or class of object — and actor alive checks |
| **Position** | `EntityLanePositionCondition`, `WaypointCondition` | Road/lane position, waypoint crossing |
| **Relative** | `EntityDistanceCondition`, `EntityPositionDistanceCondition`, `TimeToCollisionCondition`, `TimeHeadwayCondition`, `RelativeSpeedCondition` | Gap to another entity or to a place on the map, time to collision, following headway and speed difference |
| **Motion** | `SpeedCondition`, `AccelerationCondition`, `StandstillCondition`, `TemporaryStopCondition` | Speed and acceleration thresholds, standstill detection, stop-and-go |
| **Traffic** | `TrafficSignalCondition` | Traffic light state checks |
| **Composition** | `AndCondition`, `OrCondition`, `NotCondition` | Logical combinators |
| **Stateful** | `StickyCondition`, `PersistentCondition` | Latch once satisfied / persist across ticks |
| **Utility** | `AlwaysTrueCondition` | Unconditional trigger (default for actions) |

> **Distances are measured in a straight line, not along the lane.**
> `EntityDistanceCondition`, `TimeHeadwayCondition` and the TTC conditions all
> work in the entity coordinate system — a world-frame offset projected onto
> the subject's heading or direction of travel. OpenSCENARIO's
> `coordinateSystem: lane`, which `scenario_simulator_v2` measures by default,
> is not implemented: it needs the lanelet routing graph at run time, and the
> runtime holds none. On a curve the projection under-reads, and for
> `TimeHeadwayCondition` past a quarter turn it inverts and the condition stops
> firing. Tracked in
> [#62](https://github.com/hakuturu583/autoware_lanelet2_to_opendrive/issues/62).

**`check()` contract:**

- Returns `ScenarioResult` when the condition is triggered (satisfied or violated).
- Returns `None` when not yet triggered.
- `__init_subclass__` auto-wraps `check()` to track the latest result for UI display.

### Action Architecture

Actions execute side effects when their trigger condition is met:

```mermaid
classDiagram
    class BaseAction {
        <<abstract>>
        +label: str
        +timing: TickTiming
        +once: bool
        +execute(world) void
        +tick(world, elapsed) void
    }

    class BaseCondition {
        <<abstract>>
    }

    class TrafficSignalAction
    class EnvironmentAction
    class TurnAction
    class LaneChangeAction

    BaseAction <|-- TrafficSignalAction
    BaseAction <|-- EnvironmentAction
    BaseAction <|-- TurnAction
    BaseAction <|-- LaneChangeAction
    BaseAction --> BaseCondition : trigger condition
```

**Action lifecycle:**

1. Each tick, the runner calls `action.tick(world, elapsed)`.
2. `tick()` checks the internal `BaseCondition` via `condition.check(world, elapsed)`.
3. If the condition returns a non-`None` result, `execute(world)` is called.
4. If `once=True` (default), the action is marked as `done` and never re-evaluated.

**Tick timing:**

Actions can be registered as **pre-tick** (before `world.tick()`) or **post-tick** (after `world.tick()`) via `BaseScenario.register_pre_tick()` / `register_post_tick()`.

---

## Result Viewer

The result viewer is a web application for browsing, inspecting, and triggering scenario test results.

### Architecture Diagram

```mermaid
graph TB
    subgraph Browser["Browser"]
        HTML["HTML Pages<br/>(Jinja2 templates)"]
        SSE["SSE Client<br/>(EventSource)"]
        JS["JavaScript<br/>(htmx + vanilla)"]
    end

    subgraph FastAPI["FastAPI Application (app.py)"]
        direction TB
        pages["Page Routes<br/>GET /, /session/..., /scenario/..."]
        api["API Routes<br/>POST /api/run, /api/refresh<br/>GET /api/scenarios, /api/run/progress"]
        static["Static Files<br/>CSS, JavaScript"]
    end

    subgraph Services["Service Layer"]
        scanner_svc["Scanner<br/>(scanner.py)"]
        runner_svc["Runner<br/>(runner.py)"]
        sweep_svc["Sweep Resolver<br/>(sweep_resolver.py)"]
    end

    subgraph Storage["File System"]
        outputs["outputs/<br/>YYYY-MM-DD/HH-MM-SS/"]
        multirun["multirun/<br/>YYYY-MM-DD/HH-MM-SS/{0,1,2...}/"]
        conf["conf/scenario/<br/>*.yaml"]
    end

    subgraph External["External Process (uv run scenario)"]
        subprocess["subprocess.run()"]
        ext_main["main()<br/>(examples/run.py)"]
        ext_queue["ScenarioQueue"]
        ext_runner["ScenarioRunner"]
        ext_carla["CARLA Simulator"]
    end

    HTML --> pages
    JS --> api
    SSE --> api
    pages --> scanner_svc
    api --> scanner_svc
    api --> runner_svc
    api --> sweep_svc
    scanner_svc --> outputs
    scanner_svc --> multirun
    scanner_svc --> conf
    runner_svc --> subprocess
    subprocess --> ext_main
    ext_main --> ext_queue
    ext_queue --> ext_runner
    ext_runner --> ext_carla
    sweep_svc --> conf
    ext_runner -->|"*_result.json<br/>*.mp4"| outputs
    ext_runner -->|"*_result.json<br/>*.mp4"| multirun
```

### Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| **`app.py`** | `ui/app.py` | FastAPI routes, template rendering, SSE streaming |
| **`scanner.py`** | `ui/scanner.py` | Scan `outputs/` and `multirun/` directories for result JSON files. Build session lists and condition trees. |
| **`runner.py`** | `ui/runner.py` | Execute `uv run scenario` as subprocesses in a background thread. Track progress with thread-safe global state. |
| **`sweep_resolver.py`** | `ui/sweep_resolver.py` | Resolve sweep constraints into concrete override lists without launching CARLA. Lightweight Hydra Compose API usage. |
| **`models.py`** | `ui/models.py` | Pydantic data models for the viewer UI. |

### Data Models

```mermaid
classDiagram
    class SessionSummary {
        +date: str
        +time: str
        +session_type: str
        +scenario_name: str
        +passed_count: int
        +total_count: int
    }

    class SessionItem {
        +index: int
        +scenario_name: str
        +passed: bool | None
        +elapsed_seconds: float | None
        +overrides: list[str]
        +message: str
    }

    class ScenarioResultView {
        +passed: bool | None
        +message: str
        +elapsed_seconds: float | None
        +condition_statuses: list[ConditionNode]
        +overrides: list[str]
        +raw_log: str
        +video_filename: str | None
    }

    class ConditionNode {
        +label: str
        +satisfied: bool
        +message: str
        +condition_type: str
        +role: str
        +details: dict
        +children: list[ConditionNode]
    }

    class RunProgress {
        +current: int
        +total: int
        +scenario_name: str
        +status: Status
    }

    SessionSummary "1" --> "*" SessionItem : contains
    SessionItem "1" --> "1" ScenarioResultView : details
    ScenarioResultView "1" --> "*" ConditionNode : condition tree
    ConditionNode "1" --> "*" ConditionNode : children (recursive)
```

### Result Directory Structure

The scanner discovers results from two directory layouts:

```
base_path/
├── outputs/                          # Single and batch runs
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── ScenarioName_result.json
│           ├── ScenarioName.log        # CARLA native recording
│           ├── ScenarioName.mp4        # Rendered video
│           ├── batch_results.json      # Batch summary (batch mode only)
│           └── .hydra/
│               └── overrides.yaml      # Hydra CLI overrides
│
└── multirun/                         # Sweep / multi-scenario runs
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── 0/                    # Job index 0
            │   ├── ScenarioName_result.json
            │   ├── ScenarioName.mp4
            │   ├── raw_output.log    # Subprocess stdout/stderr
            │   └── .hydra/
            │       └── overrides.yaml
            ├── 1/                    # Job index 1
            │   └── ...
            └── N/
```

**Session types:**

| Type | Source | Identification |
|------|--------|----------------|
| `multirun` | `multirun/` directory | Numbered subdirectories (0, 1, 2, ...) |
| `batch` | `outputs/` directory | Contains `batch_results.json` |
| `single` | `outputs/` directory | Contains `*_result.json` (no batch file) |

### Scenario Execution from Viewer

The viewer runs scenarios via subprocess to isolate the CARLA Python API (which can cause SIGSEGV) from the web server process:

```mermaid
sequenceDiagram
    participant B as Browser
    participant A as FastAPI
    participant R as Runner (thread)
    participant S as Subprocess (uv run scenario)
    participant SQ as ScenarioQueue
    participant SR as ScenarioRunner
    participant C as CARLA Simulator

    B->>A: POST /api/run {scenario, overrides, sweeper}

    alt With sweeper
        A->>A: sweep_resolver.resolve_sweep()
        Note over A: Expand constraints into<br/>concrete override lists
    else Without sweeper
        A->>A: Expand glob to scenario names
    end

    A->>R: start_run(overrides_list)
    A-->>B: {"status": "started"}

    B->>A: GET /api/run/progress (SSE)

    loop For each job
        R->>R: _set_progress(running)
        R->>S: subprocess.run(["uv", "run", "scenario", ...overrides])

        rect rgb(240, 248, 255)
            Note over S,C: Isolated process — same execution<br/>pipeline as the scenario CLI command
            S->>SQ: main() → build_scenario() → queue.add()
            SQ->>SR: run_scenario(scenario)
            SR->>C: Setup, tick loop, recording, cleanup
            C-->>SR: ScenarioResult
            SR->>SR: Write *_result.json + *.mp4
        end

        S-->>R: exit code + stdout/stderr
        R->>R: Save raw_output.log
        R->>R: _set_progress(passed/failed)
        A-->>B: SSE event {current, total, status}
    end

    R->>R: _set_progress(done)
    A-->>B: SSE event {status: "done"}
```

**Key design decisions:**

- **Process isolation**: Each scenario runs in a separate `uv run scenario` subprocess. This prevents CARLA's C++ extension from crashing the web server.
- **Thread-safe progress**: A global `RunProgress` object protected by `threading.Lock` provides progress state.
- **SSE streaming**: The `/api/run/progress` endpoint uses Server-Sent Events for real-time progress updates (polled at 0.5 s intervals).
- **Multirun grouping**: When running multiple scenarios, jobs share a timestamped `multirun/` directory with numbered subdirectories.

### Condition Tree Display

The viewer reconstructs the condition hierarchy from the JSON result for display:

```
pass[0](all_roads_visited)         ← AndCondition (top-level)
├── StickyCondition(road_5)        ← Wrapper
│   └── EntityLanePositionCondition  ← Leaf
├── StickyCondition(road_12)
│   └── EntityLanePositionCondition
└── StickyCondition(road_8)
    └── EntityLanePositionCondition

fail[0](default_timeout)           ← TimeoutCondition
fail[1](ego_existence)             ← EntityExistenceCondition
```

The `_build_condition_tree()` function in `scanner.py` recursively parses:

- `children` key for composite conditions (`AndCondition`, `OrCondition`)
- `child` key for wrapper conditions (`StickyCondition`, `PersistentCondition`, `NotCondition`)

### Caching Strategy

The scanner uses a module-level dictionary cache (`_cache`) keyed by path and session identifiers. The cache is cleared explicitly via the `/api/refresh` endpoint (triggered by the "Update" button in the UI). This avoids repeated filesystem scans while keeping results fresh on demand.

---

## Coordinate Transform System

Three coordinate systems are unified through `MapManager`:

```mermaid
graph LR
    L2["Lanelet2Pose<br/>(lanelet_id, s)"]
    OD["OpenDrivePose<br/>(road_id, lane_id, s)"]
    CW["CarlaWorldPose<br/>(x, y, z, yaw)"]

    L2 -->|"to_opendrive()"| OD
    OD -->|"to_carla_location()"| CW
    CW -->|"to_lanelet2()"| L2
    OD -->|"snap_to_carla_road()"| CW
    CW -->|"to_opendrive()"| OD
```

**`MapManager`** is a singleton that holds the `LaneletMap`, `RoadNetwork`, and a precomputed z-offset (averaged from CARLA spawn points). It is initialized once per `ScenarioQueue.start()` when both `xodr_path` and `lanelet2_path` are provided.

**`snap_to_carla_road()`** uses CARLA's ray-cast API to project an OpenDRIVE position onto the actual 3D road surface, accounting for terrain height differences between the map definition and the rendered world.

---

## Entity Management

```mermaid
classDiagram
    class VehicleEntityConfig {
        +role_name: str
        +spawn_location: SpawnLocation
        +vehicle_type: str
        +initial_speed_kmh: float
        +spawn_retry_max_count: int
    }

    class EgoConfig {
        +role_name = "Ego" (fixed)
        +goal_pose: Lanelet2Pose | None
    }

    class SpawnLocation {
        <<interface>>
    }

    class SpawnTransform {
        +transform: carla.Transform
    }

    class SpawnPointIndex {
        +index: int
    }

    class VehicleEntity {
        +actor: carla.Actor | None
        +spawn(world) carla.Actor
        +destroy()
    }

    class EgoVehicle {
        +use_autopilot = true
        +requires_goal = false
        +actor: carla.Actor | None
        +spawn(world, config) carla.Actor
        +on_scenario_start(world)
        +on_tick(world, elapsed)
        +on_scenario_end(world)
        +destroy()
    }

    class AutowareEntity {
        +use_autopilot = false
        +requires_goal = true
    }

    class CarlaDriverEntity {
        +use_autopilot = false
        +termination_requested: bool
    }

    VehicleEntityConfig <|-- EgoConfig
    SpawnLocation <|.. SpawnTransform
    SpawnLocation <|.. SpawnPointIndex
    VehicleEntityConfig --> SpawnLocation
    VehicleEntity --> VehicleEntityConfig
    EgoVehicle <|-- AutowareEntity
    EgoVehicle <|-- CarlaDriverEntity
```

**Ego lifecycle hooks**: `ScenarioRunner` calls `on_scenario_start()` after the warm-up
ticks, `on_tick()` on every simulation tick, and `on_scenario_end()` during teardown.
They are no-ops on `EgoVehicle`, so a TrafficManager-driven ego costs nothing. An ego
that drives itself overrides them — see
[External Driver Interface](driver_interface.md).

**Who drives the traffic**: everything the scenario did not author belongs to a
[traffic backend](traffic_backends.md) — `traffic_manager` by default, `none` for
an empty road, or one from another package. `ScenarioRunner` calls
`prepare` / `start` / `tick` / `close` on it and never names a traffic simulator
itself, and an entity's manoeuvre intents (`change_lane`, `turn_at_junction`) are
delegated to whichever backend drives it.

**Who drives the ego**: `use_autopilot` decides whether the backend is given the
ego actor to drive or told to keep its hands off it. `EgoVehicle` opts in (TrafficManager drives);
`AutowareEntity` opts out and nothing drives the actor; `CarlaDriverEntity` opts out and
drives it itself from an external policy's plan.

**Where the ego is going**: `requires_goal` says whether an entity can start without a
destination. An `AutowareEntity` cannot — the goal is what makes it move at all — and
`BaseScenario.require_goal()`, asked by `register_route_to_goal` and by `ScenarioRunner`
once `setup()` returns, refuses one that has none. An ego driven by the TrafficManager or
a driver policy reads no goal, so a scenario may name none and the routing action is
simply not registered. The goal reaches the ego config from `ego.goal_lanelet_id` or from
the scenario itself — `IntersectionPassingScenario` derives it from the route it asserts,
through `BaseScenario.derive_goal_from_route()`.

**Spawn retry logic**: When a vehicle fails to spawn (e.g., collision with existing geometry), the spawn system retries with lateral (`t_step`) and vertical (`z_step`) offsets, up to `spawn_retry_max_count` attempts.

---

## Lanelet Constraint Sweeper

The sweeper enables automated parameter sweeps across lanelets that match specified constraints:

```mermaid
graph TB
    subgraph Input
        SC["Scenario Config<br/>(YAML)"]
        Constraints["Sweep Constraints<br/>(traffic_light, direction, ...)"]
        Bindings["Sweep Bindings<br/>(ego.spawn_s, npc.spawn_s, ...)"]
    end

    subgraph Processing
        Map["Load Lanelet2 Map"]
        RG["Create Routing Graph"]
        Match["Find Matching Lanelets"]
        Resolve["Resolve Bindings<br/>for each matched lanelet"]
    end

    subgraph Output
        Jobs["Override Lists<br/>[scenario=X, ego.spawn_lanelet_id=5, ...]"]
    end

    SC --> Map
    Constraints --> Match
    Map --> RG
    RG --> Match
    Match --> Resolve
    Bindings --> Resolve
    Resolve --> Jobs
```

The sweeper is registered as a Hydra plugin (`hydra_plugins/autoware_scenario_sweeper/`) and can also be invoked directly from the viewer via `sweep_resolver.py` (without CARLA, using only the Lanelet2 map).

---

## Ports and Environment Variables

### Network Ports

The system uses three TCP ports for inter-process communication with the CARLA simulator and the viewer web server:

```mermaid
graph LR
    subgraph Viewer["Viewer Process"]
        FastAPI["FastAPI<br/>(Uvicorn)"]
    end

    subgraph Scenario["Scenario Process"]
        Client["carla.Client"]
        TM_Client["TrafficManager<br/>Client"]
    end

    subgraph CARLA["CARLA Simulator"]
        RPC["RPC Server"]
        TM_Server["TrafficManager<br/>Server"]
    end

    subgraph Browser["Browser"]
        B["User"]
    end

    B -->|"TCP 9000<br/>(VIEWER_PORT)"| FastAPI
    Client -->|"TCP 2000<br/>(server.port)"| RPC
    TM_Client -->|"TCP 8100<br/>(traffic_manager.port)"| TM_Server
```

| Port | Default | Component | Protocol | Configured via | Description |
|------|---------|-----------|----------|----------------|-------------|
| **2000** | 2000 | CARLA RPC Server | TCP | `server.port` (Hydra), `--port` CLI arg | Main communication channel between `carla.Client` and the CARLA simulator. Used for world control, actor spawning, recording, etc. |
| **8100** | 8100 | CARLA TrafficManager | TCP | `traffic_manager.port` (Hydra), `tm_port` constructor arg | RPC port for the TrafficManager, which controls NPC vehicle autopilot behavior. The CARLA default is 8000, but this project uses 8100 to avoid conflicts with other services (e.g., VS Code). |
| **9000** | 9000 | Viewer Web Server | HTTP | `VIEWER_PORT` env var | FastAPI/Uvicorn server for the result viewer web UI. Serves HTML pages, REST API, and SSE progress stream. |

!!! note
    The CARLA RPC port and TrafficManager port are used within the **scenario subprocess**, not the viewer process. The viewer communicates with CARLA only indirectly through `subprocess.run(["uv", "run", "scenario", ...])`.

### Environment Variables

| Variable | Required | Default | Used by | Description |
|----------|----------|---------|---------|-------------|
| **`CARLA_EXECUTABLE`** | Yes (if launching server) | — | `CarlaServerManager` | Absolute path to the CARLA UE5 executable (`CarlaUE5.sh`). Required to launch a new CARLA server. When a server is already running and `reuse_if_running=True`, this variable is not needed. Also used as a gate for pytest: tests are skipped when this variable is unset. |
| **`NISHISHINJUKU_MAP_PATH`** | Yes (if using xodr overwrite) | — | `ScenarioRunner.load_map_by_overwriting_xodr()` | Path to the internal `.xodr` file inside the CARLA installation for the NishishinjukuMap. Used to overwrite the built-in OpenDRIVE file with a custom version while retaining full CARLA map assets (meshes, textures). The variable name is derived from the map name via CamelCase → `UPPER_SNAKE_CASE_PATH` conversion. |
| **`NISHISHINJUKU_XODR_PATH`** | No | `autoware_lanelet2_to_opendrive/test/data/nishishinjuku_carla.xodr` | Map config YAML (`${oc.env:...}`) | Path to the custom OpenDRIVE file for the Nishishinjuku map. Resolved by OmegaConf's `oc.env` interpolation in `conf/map/nishishinjuku.yaml`. |
| **`NISHISHINJUKU_LANELET2_PATH`** | No | `autoware_lanelet2_to_opendrive/test/data/nishishinjuku.osm` | Map config YAML (`${oc.env:...}`) | Path to the Lanelet2 `.osm` file for the Nishishinjuku map. Resolved by OmegaConf's `oc.env` interpolation. |
| **`VIEWER_BASE_PATH`** | No | Current working directory | Viewer (`ui/__init__.py`) | Base directory that the viewer scans for `outputs/` and `multirun/` result directories. |
| **`VIEWER_HOST`** | No | `0.0.0.0` | Viewer (`ui/__init__.py`) | Bind address for the Uvicorn HTTP server. |
| **`VIEWER_PORT`** | No | `9000` | Viewer (`ui/__init__.py`) | Listen port for the Uvicorn HTTP server. |
| **`SWEEP_RESUME_FROM`** | No | `0` | Sweeper (`lanelet_constraint_sweeper.py`) | Internal variable set by the `--resume-from N` CLI flag. Passed via environment because Hydra's CLI parser rejects unknown overrides. Tells the sweeper to skip the first N jobs. |

#### Dynamic Map Path Variables

`ScenarioRunner.load_map_by_overwriting_xodr()` derives the environment variable name from the CARLA map name using the `_map_name_to_env_var()` function:

| Map Name | Derived Variable |
|----------|-----------------|
| `NishishinjukuMap` | `NISHISHINJUKU_MAP_PATH` |
| `Town01` | `TOWN01_PATH` |
| `Town10HD_Opt` | `TOWN10_HD_OPT_PATH` |

The conversion rule is: insert `_` before each uppercase letter that follows a lowercase letter or digit, convert to UPPER_SNAKE_CASE, and append `_PATH`.

---

## Key Design Decisions

### Process Isolation for Viewer

The viewer deliberately avoids importing CARLA in its own process. All scenario execution happens via `subprocess.run(["uv", "run", "scenario", ...])`. This prevents CARLA's C++ extension from causing segmentation faults in the web server.

### Synchronous Simulation Mode

All scenario execution uses CARLA's synchronous mode at 20 Hz. This ensures:

- Deterministic tick ordering (actions, conditions, physics all advance together)
- Reproducible results with fixed random seeds
- No frame drops during recording

### Two-Pass Video Recording

Video is not rendered during execution but replayed afterwards. This avoids the overhead of RGB camera processing during the tick loop, keeping condition evaluation timing accurate.

### Condition Composition Pattern

Complex pass/fail criteria are built by composing atomic conditions:

```python
# Example: pass when ego visits all three road segments
pass_condition = AndCondition(
    label="all_roads_visited",
    children=[
        StickyCondition(EntityLanePositionCondition.anywhere_on_road("ego", "5", ...)),
        StickyCondition(EntityLanePositionCondition.anywhere_on_road("ego", "12", ...)),
        StickyCondition(EntityLanePositionCondition.anywhere_on_road("ego", "8", ...)),
    ],
)
```

`StickyCondition` latches once its child is satisfied, so each road only needs to be visited once. `AndCondition` requires all children to be satisfied simultaneously.

### World Reload for Clean State

After each scenario, `ScenarioRunner` calls `client.reload_world()` instead of manually destroying actors. This guarantees a completely clean state (actors, sensors, physics, Traffic Manager internal maps) and avoids "failed to destroy actor" errors from the CARLA server.
