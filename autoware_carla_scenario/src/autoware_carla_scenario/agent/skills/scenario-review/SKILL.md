---
name: scenario-review
description: This skill should be used when reviewing CARLA scenario parameters for orthogonality, analyzing parameter dependencies, deciding which values should be parameters vs. computed, validating the parameter design of a logical scenario, or checking sweep compatibility. Use when "review parameters", "check orthogonality", "parameter design", "review scenario parameters", "parameter independence", or discussing parameter selection for sweeper constraints.
---

# Scenario Parameter Review Guide

> Paths: `<framework>/` (the installed framework's source) and `docs/<page>.md` (its documentation) as in the `scenario-writing` skill's "Where Things Are".

## Purpose

Review and validate the parameter design of CARLA test scenarios before converting concrete scenarios to logical scenarios. Ensure parameters are orthogonal, minimal, and compatible with the lanelet constraint sweeper.

**How the sweeper works** (`scenario-writing`'s `references/constraints-reference.md`): `sweep.constraints` sweeps exactly **one** key -- a lanelet ID, usually `ego.spawn_lanelet_id` -- picking every lanelet that matches its constraints; `sweep.bindings` derives every other value a case needs from that pick; `sweep.exclude` drops cases by the values they end with. A scalar parameter of the scenario itself (a distance, a speed, a time) can be drawn from a distribution beside that key -- `- type: uniform` (`low`, `high`) or `- type: normal` (`mean`, `std`, optional `low`/`high`) under its own key in `sweep.constraints`, `sweep.samples` draws per case (`docs/usage.md`, "Drawing scalar parameters from distributions"); what the ODD describes (the weather, the traffic) `sweep.odd_sample` draws from the ODD (`docs/odd.md`, "Sampling scenarios from the ODD").

## Parameter Review Process

### Step 1: Identify Candidate Parameters

Examine the concrete scenario's YAML config and Python code to list all hardcoded values that could vary across test runs:

**Common candidates:**
- `ego.spawn_lanelet_id` — Ego vehicle spawn lanelet
- `ego.spawn_s` — Ego arc-length position along the lanelet
- NPC spawn lanelet and `s` (scenario config fields, e.g. `scenario.npc_lanelet_id`)
- The route or goal the scenario asserts (`scenario.expected_route_lanelet_ids`, `ego.goal_lanelet_id`)
- `scenario.timeout_seconds` — Scenario timeout
- Speed parameters (initial speed, target speed)
- Distance/timing thresholds
- Traffic light states or phases

### Step 2: Analyze Parameter Orthogonality

**Orthogonality principle:** Each parameter should be independently variable. If changing parameter A forces a change in parameter B, they are not orthogonal — one should be derived from the other.

**Checklist for each parameter pair:**
1. Can parameter A change without affecting parameter B's valid range? -> Orthogonal
2. Does parameter B's value depend on parameter A's value? -> Not orthogonal -> Derive B from A
3. Are both parameters constrained to the same lanelet topology? -> Likely not orthogonal

**Common non-orthogonal patterns:**

| Pattern | Problem | Solution |
|---------|---------|----------|
| Ego lanelet + NPC lanelet (NPC relative to ego) | NPC position depends on ego position | Sweep only the ego lanelet; derive the NPC lanelet with a binding (`adjacent`, `matched`, `stop_line_approach`, `route_through`), or in Python when no binding expresses it |
| Spawn lanelet + spawn s (s depends on lanelet geometry) | s valid range varies per lanelet | Use a binding to compute s from the lanelet (e.g., `stop_line_offset`) |
| Two parameters that must satisfy a topological relationship | The sweeper sweeps one key only | Make one the swept parameter, derive the other with a binding or via the routing graph |
| A route / goal the scenario asserts + the spawn lanelet | A fixed route only fits the original spawn | Bind it with `route_through` (`last_only: true` for a goal lanelet) |

### Step 3: Classify Each Candidate

For each candidate parameter, classify it as:

1. **Swept parameter** — The one lanelet ID the sweeper enumerates
   - Typically: `ego.spawn_lanelet_id`

2. **Binding** — Derived from the pick using a deterministic rule (`sweep.bindings`)
   - Typically: spawn `s` (from the stop line), NPC lanelets relative to the ego, the asserted route or goal

3. **Python-computed** — Derived in `setup()` from the pick when no binding expresses the relation

4. **ODD-sampled** — A scalar knob worth varying (speed, weather, ...), drawn by `sweep.odd_sample` rather than swept

5. **Constant** — Fixed value that should not vary
   - Typically: timeout values, speed thresholds, grace periods

**Decision tree:**
```
Is the value independently meaningful across different test runs?
+- YES -> Is it the lanelet the case is about?
|         +- YES -> SWEPT PARAMETER (the one key of sweep.constraints)
|         +- NO  -> A scalar knob? -> ODD-SAMPLED (sweep.odd_sample)
|                   Otherwise     -> derive it from the pick (below)
+- NO  -> Does it depend on the pick?
          +- YES -> BINDING (sweep.bindings), or compute in Python
          +- NO  -> CONSTANT (keep it as the dataclass default)
```

### Step 4: Validate NPC Dependency Design

**Critical rule:** When an NPC's spawn depends on the ego's position, **never** sweep the NPC's lanelet ID: derive it from the pick.

**Preferred: a binding into a scenario config field** (the built-in `cut_in/left.yaml`):
```yaml
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: has_adjacent      # only picks that have the lane the NPC needs
        value: left
  bindings:
    scenario.npc_lanelet_id:
      type: adjacent
      side: left
```

**Fallback: compute it in `setup()`** when no binding expresses the relation:
```python
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

Either way, constrain the pick so the relation exists for every case (`has_adjacent` for an adjacent lane, `following_of` for a predecessor).

**Why:** The sweeper sweeps one key; a second key in `sweep.constraints` is refused. Deriving the NPC from the pick keeps every case valid by construction and visible in `scenario-expand`'s output.

### Step 5: Verify Sweeper Compatibility

For each parameter that will be swept:

1. **Can a constraint express the valid set?**
   - Discover the constraint types (`scenario-writing`'s `references/constraints-reference.md`)
   - If no existing constraint fits, propose a new one to the user

2. **Can a binding compute derived values?**
   - Discover the binding types the same way
   - If no existing binding fits, propose a new one to the user

3. **Is the case set finite and reasonable?**
   - Lanelet IDs: finite per map (good); check the count with `uv run scenario-expand scenario=<name>/<variant>`
   - Continuous values (speed, distance): constants, bindings, or `sweep.odd_sample`
   - Does every case still make sense once its bindings are applied (walk-backs, lanes that split)? Use `sweep.exclude` for cases a constraint on the pick cannot see

### Step 6: Present Review Summary

Present the review results to the user in this format:

```
## Parameter Review Summary

### Swept parameter (the one key of sweep.constraints)
- `ego.spawn_lanelet_id` — [rationale]

### Bindings (derived from parameters)
- `ego.spawn_s` -> `stop_line_offset(offset=10.0)` — [rationale]

### Computed in Python (derived in setup())
- [only what no binding expresses] — [rationale]

### ODD-sampled (sweep.odd_sample)
- [scalar knobs, if any] — [rationale]

### Constants (fixed values)
- `timeout_seconds: 10.0` — [rationale]

### Orthogonality Check
- [pass/fail] One swept key; all other parameters are independent of it or derived from it
- [pass/fail] No NPC positions depend on ego without being derived
- [pass/fail] All derived values use bindings or Python computation, and move with the pick
- [pass/fail] Constraint and binding types exist, with the keys they take
```

## Anti-Patterns to Flag

### Anti-Pattern 1: Coupled Lanelet Parameters

```yaml
# BAD: a second swept key (refused: sweep.constraints holds one key)
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: has_adjacent
        value: left
    scenario.npc_lanelet_id:
      - type: equals
        value: ${ego.spawn_lanelet_id}
```

**Fix:** Sweep only `ego.spawn_lanelet_id` and bind the NPC's lanelet from the pick (`scenario.npc_lanelet_id: {type: adjacent, side: left}`), or compute it in Python.

### Anti-Pattern 2: Hardcoded s-values as Parameters

```yaml
# BAD: an s-value that only fits the original lanelet, kept for every pick
ego:
  spawn_lanelet_id: 242
  spawn_s: 25.0        # beyond the end of a shorter picked lanelet
sweep:
  constraints:
    ego.spawn_lanelet_id:
      - type: has_traffic_light_stop_line
```

**Fix:** Derive it from the pick with a binding (or constrain the picks with `lanelet_length` so the fixed value fits every one):
```yaml
sweep:
  bindings:
    ego.spawn_s:
      type: stop_line_offset
      offset: 10.0
```

### Anti-Pattern 3: A Fixed Route or Goal

A route the scenario asserts (`scenario.expected_route_lanelet_ids`) or a goal (`ego.goal_lanelet_id`) written for the original spawn is kept for every pick, so almost every case fails.

**Fix:** Bind it with `route_through` (`depth`, `through_junction`, `last_only: true` for a goal).

### Anti-Pattern 4: Non-Topological Parameter as Sweep Target

```yaml
# BAD: constraining a non-lanelet value (constraints pick lanelet IDs)
sweep:
  constraints:
    scenario.timeout_seconds:
      - type: equals
        value: 30
```

**Fix:** A timeout is a constant: keep it as the dataclass default (long enough for every case). A scalar worth varying is drawn: a parameter of the scenario (a speed, a distance) from a distribution in `sweep.constraints` (`type: uniform` / `type: normal`), what the ODD describes (the weather) with `sweep.odd_sample`.

## Integration with Write Workflow

This review should be performed at **Stage 5** of the `write-scenario` workflow (after the concrete scenario passes, before adding sweep constraints). The review ensures that the transition from concrete to logical scenario maintains correctness and efficiency.

After the review is confirmed by the user, proceed to Stage 6 (Discover Constraints/Bindings and Add Sweep Config) using the agreed parameter design.
