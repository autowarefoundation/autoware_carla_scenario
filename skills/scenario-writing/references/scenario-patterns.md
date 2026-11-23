# Scenario Patterns Reference

The built-in scenarios under `autoware_carla_scenario/src/autoware_carla_scenario/examples/` are the patterns to copy: `traffic_light_compliance.py`, `temporary_stop.py`, `intersection_passing.py`, `lane_change.py`, `cut_in.py` (an NPC), `pedestrian_dart_out.py` (a pedestrian), `adversity.py`. Read the closest one before writing a new scenario. Everything below is imported from the top-level package (`from autoware_carla_scenario import ...`); the CARLA API is `import typesafe_carla.carla as carla` (never `import carla`).

## Scenario Class Anatomy

Every scenario inherits from `BaseScenario` and implements `setup()` and `is_done()`.

### Constructor Pattern

```python
class MyScenario(BaseScenario):
    """What the scenario verifies."""

    # Every attribute assigned on self, declared at class level: the static
    # check (Codon) refuses one it has not seen declared.
    _config: MyScenarioConfig

    def __init__(
        self,
        ego_config: EgoConfig,
        spawn_pose: Lanelet2Pose,
        config: MyScenarioConfig | None = None,
        ground_projection: GroundProjectionConfig | None = None,
    ) -> None:
        super().__init__(
            ego_config, spawn_pose=spawn_pose, ground_projection=ground_projection
        )
        self._config = config or MyScenarioConfig()
```

### Setup Pattern

```python
def setup(self) -> None:
    world = self.world
    cfg = self._config

    # 1. Ego spawn: Lanelet2 -> OpenDRIVE -> CARLA, snapped to the road
    self._setup_ego_spawn()

    # 2. Traffic lights: set once as the scenario starts (register_init runs an
    #    action once, before the tick loop). Without it the lights keep CARLA's
    #    own cycle.
    self.register_init(
        TrafficSignalAction(
            state=carla.TrafficLightState.Green,
            lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
            label="set_all_green",
        )
    )

    # 3. NPC vehicles (optional), placed in Lanelet2 coordinates
    npc_pose = Lanelet2Pose(lanelet_id=cfg.npc_lanelet_id, s=cfg.npc_s)
    npc_snapped = snap_to_carla_road(
        npc_pose, world, ground_projection=self._ground_projection
    )
    npc = VehicleEntity(
        VehicleEntityConfig(
            role_name=str(EntityRole.npc(1)),
            spawn_location=SpawnTransform(npc_snapped.to_carla_transform()),
            vehicle_type=cfg.npc_vehicle_type,
            initial_speed_kmh=cfg.npc_initial_speed_kmh,
            spawn_retry_max_count=self.ego_config.spawn_retry_max_count,
            spawn_retry_t_step=self.ego_config.spawn_retry_t_step,
            spawn_retry_z_step=self.ego_config.spawn_retry_z_step,
            od_pose=to_opendrive(npc_pose),
            ground_projection=self._ground_projection,
        )
    )
    npc.spawn(world)
    self.register_entity(npc)

    # 4. Actions, each fired when its condition holds
    self.register_pre_tick(some_action)

    # 5. Pass conditions: the scenario passes as soon as ANY of them fires
    self.register_pass_condition(pass_condition)

    # 6. Fail conditions (ALWAYS include a timeout). On a tick, pass
    #    conditions are checked before fail conditions.
    self.register_fail_condition(TimeoutCondition(cfg.timeout_seconds, label="timeout"))
```

## Common Condition Compositions

### Route Verification (Visit Several Lanelets)

```python
# Verify the ego reaches every expected lanelet (in any order). AndCondition
# checks every child on every tick, so each sticky child latches as soon as
# its lanelet is reached. AndCondition needs at least two children.
# To require an order, latch the later one on the earlier one:
# StickyCondition(AndCondition([StickyCondition(a), b])).
stickies: list[BaseCondition] = [
    StickyCondition(
        EntityLanePositionCondition(
            EGO_ROLE_NAME,
            Lanelet2Pose(lanelet_id=ll_id, s=0.0),
            label=f"ego_on_lanelet_{ll_id}",
        )
    )
    for ll_id in cfg.expected_lanelet_ids
]
visited_all: BaseCondition = (
    stickies[0] if len(stickies) == 1 else AndCondition(stickies)
)
self.register_pass_condition(visited_all)
```

### Speed Check with Grace Period

```python
# Fail if the ego drops below a minimum speed (after warm-up). Speeds in m/s.
self.register_fail_condition(
    AndCondition([
        ElapsedTimeCondition(grace_seconds, label="speed_grace"),
        SpeedCondition(
            EGO_ROLE_NAME,
            min_speed_kmh / 3.6,
            ComparisonRule.LESS_THAN,
            label="ego_min_speed",
        ),
    ])
)
```

### Temporary Stop Verification

```python
# Verify the ego stops, then restarts
stopped = StickyCondition(
    StandstillCondition(EGO_ROLE_NAME, stop_duration, label="ego_stopped")
)
restarted = StickyCondition(
    AndCondition([
        ElapsedTimeCondition(check_after, label="restart_check"),
        SpeedCondition(
            EGO_ROLE_NAME,
            restart_speed_kmh / 3.6,
            ComparisonRule.GREATER_THAN_OR_EQUAL,
            label="ego_restarted",
        ),
    ])
)
self.register_pass_condition(AndCondition([stopped, restarted]))
```

### Holding a State for a While

```python
# Pass once the condition has held for hold_seconds without interruption
self.register_pass_condition(
    PersistentCondition(npc_in_ego_lane, hold_seconds, label="npc_cut_in_held")
)
```

### Traffic Light Phase Verification

```python
# Red from the start: the ego must stop
self.register_init(
    TrafficSignalAction(
        state=carla.TrafficLightState.Red,
        lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
        label="set_red",
    )
)

# Green after a delay
self.register_pre_tick(
    TrafficSignalAction(
        state=carla.TrafficLightState.Green,
        lanelet2_traffic_light_ids=TrafficLightTarget.ALL,
        condition=ElapsedTimeCondition(delay, label="light_switch"),
        label="switch_green",
    )
)
```

### Time-Windowed Fail Condition

```python
# Fail if the ego moves during a specific time window
self.register_fail_condition(
    AndCondition([
        ElapsedTimeCondition(window_start, ComparisonRule.GREATER_THAN_OR_EQUAL, label="window_start"),
        ElapsedTimeCondition(window_end, ComparisonRule.LESS_THAN, label="window_end"),
        SpeedCondition(
            EGO_ROLE_NAME,
            threshold_kmh / 3.6,
            ComparisonRule.GREATER_THAN_OR_EQUAL,
            label="ego_moving_in_window",
        ),
    ])
)
```

### Collisions

```python
self.register_fail_condition(CollisionCondition(label="ego_collided"))
# or only with one entity:
self.register_fail_condition(CollisionCondition(target=npc_role, label="hit_npc"))
```

## Dataclass Config Pattern

```python
@dataclass
class MyScenarioConfig:
    """Parameters for my scenario."""

    name: str = "my_scenario"

    #: Fail-safe timeout in seconds.
    timeout_seconds: float = 10.0
    # Scenario-specific parameters with sensible defaults; every field is
    # overridable from YAML through Hydra.
```

The generated package's config refuses bad values in `__post_init__` (a `timeout_seconds` that is not finite and positive), naming the key: do the same for new fields.

## YAML Config Pattern

```yaml
# @package _global_
scenario:
  name: my_scenario
  timeout_seconds: 10.0

ego:
  initial_speed_kmh: 5.0
  spawn_lanelet_id: 242
  spawn_s: 25.0

# Optional: the sweep of the logical scenario (references/constraints-reference.md)
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: and
        constraints:
          - <constraints>
  bindings:
    ego.spawn_s:
      type: stop_line_offset
      offset: 10.0
```

`# @package _global_` must be the first line. `scenario.name` is the registered name.

## Registration

In a scenario package (`src/<package>/__init__.py`, generated by `scenario-new`):

```python
from pathlib import Path

from autoware_carla_scenario import register_conf_dir, register_scenario

from .configs import MyScenarioConfig
from .my_scenario import MyScenario


def register() -> None:
    register_scenario("my_scenario", MyScenario, MyScenarioConfig)
    register_conf_dir(Path(__file__).parent / "conf")
```

with the entry point in its `pyproject.toml`:

```toml
[project.entry-points."autoware_carla_scenario.scenarios"]
my_scenario_package = "my_scenario_package:register"
```

For a built-in scenario, see `code-generation.md`, "Registering a built-in scenario".

## NPC Position from the Ego's

When an NPC's position depends on the ego's, the ego's lanelet stays the one swept parameter and the NPC's is derived from it. Prefer a binding into a scenario config field (the built-in `cut_in` binds `scenario.npc_lanelet_id` with `type: adjacent`; `route_through`, `matched` and `stop_line_approach` cover other relations; see `references/constraints-reference.md`). Compute it in Python only when no binding expresses the relation:

```python
def setup(self) -> None:
    mm = MapManager.get_instance()
    spawn = self._spawn_pose  # the ego's Lanelet2Pose
    if spawn is None:
        raise ValueError("my_scenario needs an ego spawn pose")
    ego_lanelet = mm.lanelet_map.laneletLayer[spawn.lanelet_id]
    previous = mm.routing_graph.previous(ego_lanelet)
    if not previous:
        raise ValueError(f"lanelet {ego_lanelet.id} has no lanelet before it")
    npc_pose = Lanelet2Pose(lanelet_id=previous[0].id, s=0.0)
```

A sweep that relies on such a relation should constrain the pick so the relation exists (e.g. `following_of` a lanelet, so a predecessor exists).
