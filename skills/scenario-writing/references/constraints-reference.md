# Sweeper Constraint and Binding Reference

The lanelet constraint sweeper (`hydra/sweeper=lanelet_constraint`) expands one scenario config into one job per lanelet that matches the constraints, deriving the rest of each job's parameters from that lanelet with bindings. The user-facing description is `autoware_carla_scenario/docs/usage.md`, "Lanelet Constraint Sweeping".

## Discovery: Always Read Source First

**The tables below were current when written; the source is the definitive list.** Under `autoware_carla_scenario/src/autoware_carla_scenario/sweeper/`:

1. **Constraints** (`constraints.py`): leaf types are the `_LEAF_REGISTRY` mapping (`type:` string → class); the combinators are parsed in `parse_constraint()`. A class's dataclass fields are its YAML keys and its docstring shows its YAML.
2. **Bindings** (`bindings.py`): the `_BINDING_REGISTRY` mapping; a binding's dataclass fields other than `target_key` are its YAML keys.

An unknown `type`, a key a type does not take, a missing key or a bad value is refused when the sweep is expanded, naming the type and the key.

## The `sweep:` Section

```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:     # the ONE swept key
      - <constraint>          # a list, implicitly AND-ed
  bindings:
    <key>: {type: <binding>, ...}   # derived from the picked lanelet, per job
  exclude:
    <key>: [<values>]         # drop a case that ends with one of these on <key>
```

- **`constraints` holds exactly one key**: the swept parameter (usually `ego.spawn_lanelet_id`). A second key is refused. Everything else a case needs is derived from the pick with `bindings`.
- **`bindings`** maps a config key to a binding; it is applied to each pick. Picks that end as the same case once their bindings are applied run once.
- **`exclude`** drops a case by the values it *ends up with* after its bindings (a constraint only sees the pick, not where a walk-back moves a spawn):
  ```yaml
  exclude:
    ego.spawn_lanelet_id: ${map.no_3d_model_lanelet_ids}
  ```
- Other keys: `job_timeout_seconds` (per job, default 120), `resume_from`, `case_index`, and `odd_sample` (cases drawn from the ODD for scalar knobs such as speeds: `autoware_carla_scenario/docs/odd.md`, "Sampling scenarios from the ODD").

## Constraint Types

Leaves (`_LEAF_REGISTRY`):

| `type` | Keys | Matches |
|---|---|---|
| `has_stop_line` | — | Lanelets owning a stop-line regulatory element |
| `has_traffic_light_stop_line` | — | Lanelets with a traffic light's stop line |
| `is_junction` | — | Junction lanelets (those with a `turn_direction` tag) |
| `turn_direction` | `value`: `left` / `right` / `straight` | Junction lanelets with that `turn_direction` |
| `has_adjacent` | `value`: `left` / `right`, `include_forks` | Lanelets with a lane beside them a vehicle may change into |
| `lanelet_length` | `rule` (e.g. `greater_than_or_equal`, `less_than_or_equal`), `value` (m) | Lanelets whose length compares so |
| `equals` | `value`: a lanelet ID, or `any` | That lanelet (or every lanelet, as a debugging placeholder) |
| `in_set` | `values`: lanelet IDs | Those lanelets (an empty set matches nothing) |
| `on_route` | `lanelet_ids` | Those lanelets (an empty list matches everything) |

Combinators (`parse_constraint()`):

| `type` | Key | Matches |
|---|---|---|
| `and` / `or` | `constraints`: a list | All / any of them |
| `not` | `constraint`: one constraint | Lanelets it does not match |
| `previous_of` | `constraints`: a list | Lanelets just before (depth 1) a lanelet matching all of them |
| `following_of` | `constraints`: a list | Lanelets just after (depth 1) a lanelet matching all of them |

## Binding Types

| `type` | Keys | Value |
|---|---|---|
| `stop_line_offset` | `offset` (m, default 15) | `s` that is `offset` metres before the pick's stop line. If that does not fit on the pick it walks back through predecessors and moves its own spawn's lanelet key with it (`ego.spawn_s` moves `ego.spawn_lanelet_id`) |
| `stop_line_approach` | `offset` | The lanelet the same `stop_line_offset` lands on, by ID |
| `route_through` | `depth` (default 1), `last_only`, `through_junction` | The pick and the `depth` lanelets after it (a list); with `last_only`, only the last, as an ID (e.g. for `ego.goal_lanelet_id`) |
| `adjacent` | `side`: `left` / `right` | The lanelet beside the pick a vehicle may change into (a pick without one is dropped) |
| `matched` | — | The pick itself: another slot following it |

Two bindings that set one key to different values on a pick are refused.

## Examples

Traffic light compliance, trimmed from the built-in `traffic_light_compliance.yaml` (a lanelet just before a signalised junction lanelet -- on these maps the traffic light's stop line belongs to the junction lanelet --, the ego 10 m before that stop line, sent through the junction):

```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: and
        constraints:
          - type: previous_of
            constraints:
              - type: has_traffic_light_stop_line
              - type: is_junction
          - type: not
            constraint:
              type: in_set
              values: ${map.no_3d_model_lanelet_ids}
  bindings:
    ego.spawn_s:
      type: stop_line_offset
      offset: 10.0
    ego.goal_lanelet_id:
      type: route_through
      depth: 1
      through_junction: true
      last_only: true
  exclude:
    # A walk-back can land on a lanelet CARLA does not draw.
    ego.spawn_lanelet_id: ${map.no_3d_model_lanelet_ids}
```

An NPC beside the ego (as the built-in `cut_in/left.yaml` does): the NPC's lanelet is a scenario config field, bound from the pick rather than swept:

```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: and
        constraints:
          - type: has_adjacent
            value: left
          - type: not
            constraint:
              type: is_junction
          - type: lanelet_length
            rule: greater_than_or_equal
            value: 60.0
  bindings:
    scenario.npc_lanelet_id:
      type: adjacent
      side: left
```

The route a case asserts, moving with each pick (as `intersection_passing/straight.yaml` does):

```yaml
  bindings:
    scenario.expected_route_lanelet_ids:
      type: route_through
      depth: 1
```

**Pattern notes:**
- `not in_set ${map.no_3d_model_lanelet_ids}` keeps picks off lanelets the CARLA map draws no 3D model for (spawn failures); `exclude` does the same for where bindings move a spawn.
- Every key a case depends on must move with the pick: a value outside `sweep:` that only fits the fixed case (a `spawn_s` beyond a shorter lanelet's end, a route from another lanelet) stays as it is for every case.
- `type: equals, value: any` is a debugging placeholder: replace `any` with the lanelet ID of a failing case to rerun only that one.

Preview what a sweep expands to, without CARLA:

```bash
uv run scenario-expand scenario=<name>/<variant>
```
