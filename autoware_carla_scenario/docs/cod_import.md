# Importing CoD Scenes

[CodSceneClassifier](https://github.com/tier4/CodSceneClassifier) labels Autoware
drive logs: it cuts a recording into scenes and gives each one the ego's driving
decision, the signal it faced, the junction and road shape, and up to five road
users described by categories and one relative position and speed sample.
`scenario-import-cod` turns each scene into a [Scenario IR](scenario_editor.md)
document:

```bash
uv run scenario-import-cod 2026-09-14.json -o cod_scenarios/
# cod_scenarios/cod_2026_09_14_09_12_33_scene_003.yaml ...
# cod_scenarios/import_report.json   -- what each document approximates
```

The documents open in the Scenario Editor, export as packages, and expand with
`scenario-expand` like any other.

## Logical, not a replay

The classifier output carries no lanelet IDs, no map name, no route and no
trajectories, and the logs come from maps CARLA does not have. So a scene is
imported as a **logical scenario**: the ego's spawn is a constraint search for
the kind of place the scene happened in, and everything else is derived from
the lanelet that search picks. The scenario then expands on whatever map it is
run on.

| Ego decision | Searched for | Goal | PASS |
| --- | --- | --- | --- |
| `turn_left` / `turn_right` | a junction lanelet turning that way with a stop line; the ego starts 25 m before it | the lane after the turn | the ego has been on the turn |
| `change_lane_left` / `_right` | a lane at least 60 m long with a neighbour on that side, outside a junction | the neighbouring lane | the ego has been on it |
| stopping at a red signal | a traffic-light stop line; the ego starts 25 m before it | two lanelets on | the ego has stood still for 2 s |
| anything else | a lane at least 50 m long, outside a junction | the next lanelet | the scene's length has elapsed |

Every search excludes the map's `no_3d_model_lanelet_ids`, and the scene's
scenery narrows it further:

| Scenery | Added to the search |
| --- | --- |
| `junction_type: T-junction` / `crossroad` | `junction_type` on a turn's junction lanelet; `previous_of: [junction_type]` for a lane approaching one (not for a lane change, which is searched outside junctions) |
| `horizontal_plane_type` | `road_shape` (straight, or bending left / right above 0.02 1/m) for lane following and lane changes |

A junction is the group of `turn_direction` lanelets that overlap, follow one
another or share an approach or exit lane; its type counts the roads meeting at
it, as CodSceneClassifier does: 3 is a T-junction, 4 a crossroad.

FAIL is a collision or a timeout of twice the scene's length plus 10 s (at least 20 s).

## Road users

Each road user is placed with the `route_offset` / `route_offset_s` bindings:
its recorded longitudinal offset from the ego, along the road, in the lane its
lateral category names.

| Classifier | IR |
| --- | --- |
| `in_the_same_lane` / `to_the_left` / `to_the_right` | `side: same` / `left` / `right` (and the search requires that neighbour) |
| `initial_direction: oncoming` | `side: opposite` |
| `initial_direction: crossing_from_left` / `_right` | `crossing` / `crossing_s`: the lane into the junction lanelet that crosses the ego's path from that side, as far before the junction as the vehicle was beside the ego's line (5-40 m); the search requires `has_crossing` |
| `velocity` | `initial_speed_kmh` |
| `standing_still` / `accelerating` / `decelerating` / `keeping_speed` | `set_speed` to 0 / +15 km/h / -15 km/h / the same speed |
| `change_lane_*`, or `context: cutting_in` | `lane_change` (towards the ego for a cut-in), 1 s in |
| `turn_left` / `turn_right` | `turn` |
| pedestrian on or near a crosswalk | `crosswalk` / `crosswalk_s` / `crosswalk_heading`: at the kerb of the first crosswalk ahead on the side it was on, facing across; the search requires `has_crosswalk_ahead` |
| other pedestrian | a walker at the recorded lateral offset; crossing contexts face across the road |
| walking contexts | `walk_straight` |
| `signal_color` | `traffic_signal` on every light |

## What is approximated

Each document's description lists what it leaves out, and
`import_report.json` collects the same notes per scene. The current gaps:

- ego decisions with no IR primitive -- pulling over or out, avoidance,
  emergency braking -- and `branching` are imported as lane following;
- motorcycles and bicycles are spawned as a car and a pedestrian;
- the road's shape is not searched for on a turn or a red-light approach,
  whose pick is a junction or the lane into one;
- when a lane change starts is not recorded, so it fires 1 s in;
- the signal colour is set on every traffic light, not only the ego's.
