# Sweeper Constraint and Binding Reference

> Paths: `<framework>/` (the installed framework's source) and `docs/<page>.md` (its documentation) as in `../SKILL.md`'s "Where Things Are".

The lanelet constraint sweeper (`hydra/sweeper=lanelet_constraint`) expands one scenario config into one job per lanelet that matches the constraints, deriving the rest of each job's parameters from that lanelet with bindings. The user-facing description is `docs/usage.md`, "Lanelet Constraint Sweeping".

## Discovery: Always Read the Source

The type lists live in the source, not here. Under `<framework>/sweeper/`:

1. **Constraints** (`constraints.py`): the leaf types are the `_LEAF_REGISTRY` mapping (`type:` string → class); the combinators `and`, `or`, `not` (one `constraint`) and `previous_of`, `following_of` (depth 1 through the routing graph) take nested constraints and are parsed in `parse_constraint()`.
2. **Bindings** (`bindings.py`): the `_BINDING_REGISTRY` mapping.

A class's docstring shows its YAML, and its dataclass fields are its keys (a binding's, except `target_key`). The user-facing description of the types and of the bindings' walk-back is `docs/usage.md`, "Lanelet Constraint Sweeping". An unknown `type`, a key a type does not take, a missing key or a bad value is refused when the sweep is expanded, naming the type and the key.

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

- **`constraints` holds exactly one key of lanelet constraints**: the swept parameter (usually `ego.spawn_lanelet_id`). A second such key is refused. Everything else a case needs is derived from the pick with `bindings`.
- **A scalar parameter can be drawn from a distribution** under a key of its own in `constraints`, one item: `- {type: uniform, low, high}` or `- {type: normal, mean, std, low?, high?}` (truncated to the bounds given). `samples: N` (default 1) runs each case N times, each with values of its own; each value's seed is made of `sweep.seed` (else the run's `seed`), the scenario's name, the case and the sample's number, so a case keeps its values whatever other cases the sweep has. A drawn key may not also be set by a binding or an override (`docs/usage.md`, "Drawing scalar parameters from distributions"):
  ```yaml
  constraints:
    ego.spawn_lanelet_id: [...]
    scenario.stop_line_distance_m:
      - {type: uniform, low: 0.5, high: 3.0}
  samples: 6
  ```
- **`relations`** relates drawn parameters to each other (or to numbers the case sets), solved with Z3: a list of expressions such as `ego.initial_speed_kmh > scenario.lead_speed_kmh + 5` (keys, numbers, `+ - * /`, `abs/min/max`, `< <= > >= == !=`, `and/or/not`). Each case's values are drawn from the distributions conditioned on them; a case they cannot hold for is dropped, and relations that conflict on their own are refused before the map loads (`docs/usage.md`, "Relations between drawn parameters (Z3)"). Prefer a relation over a derived parameter only when both parameters should vary; a value fully determined by others is better computed in the scenario:
  ```yaml
  constraints:
    ego.initial_speed_kmh: [{type: uniform, low: 20, high: 60}]
    scenario.lead_speed_kmh: [{type: uniform, low: 10, high: 50}]
  relations:
    - ego.initial_speed_kmh > scenario.lead_speed_kmh + 5
  samples: 10
  ```
- **`bindings`** maps a config key to a binding; it is applied to each pick. Picks that end as the same case once their bindings are applied run once.
- **`exclude`** drops a case by the values it *ends up with* after its bindings (a constraint only sees the pick, not where a walk-back moves a spawn):
  ```yaml
  exclude:
    ego.spawn_lanelet_id: ${map.no_3d_model_lanelet_ids}
  ```
- Other keys: `job_timeout_seconds` (per job, default 120), `resume_from`, `case_index`, `samples`, `seed`, and `odd_sample` (cases drawn from the ODD for what it describes, such as the weather: `docs/odd.md`, "Sampling scenarios from the ODD").

Two bindings that set one key to different values on a pick are refused. Map-specific exclusions (`no_3d_model_lanelet_ids`) are in `conf/map/<map_name>.yaml`, referenced as `${map.no_3d_model_lanelet_ids}`.

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
