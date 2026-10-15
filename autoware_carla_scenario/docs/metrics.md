# Metrics

A run reads two kinds of quantity off the world while it drives, under keys
the framework fixes:

| | Reads | Example | An ODD maps onto it | Watched by |
|---|---|---|---|---|
| **Measure** ([Scenario measures](odd.md#scenario-measures)) | A condition the ego drives in: what the scenario sets up | `vehicle_ahead_gap_m` | Yes (`measure:` in a binding file) | Coverage; a sweep draws a scenario's parameters from it |
| **Metric** (this page) | How the ego drives: an outcome | `collision_at_fault` | No | `MetricCondition` |

An ODD describes the conditions a system is built to operate in, whatever it
then does there. A metric reads what it did, so it has no place in one: were
an ODD to map onto `offroad`, the ego's own driving would decide whether a
run is inside the domain it is tested in, and an ODD excluding `offroad`
would report a run that left the road as *outside the ODD* rather than as
*failed*. `scenario_measure("offroad")`, `measure: offroad` in a binding file
and `register_measure("offroad", ...)` are all refused.

## The built-in metrics

They are the per-timestep driving-quality metrics of NVIDIA's
[AlpaSim](https://github.com/NVlabs/alpasim) (`src/eval/src/eval/scorers`),
under AlpaSim's own names so results read the same in both, computed from
CARLA's actors and map (`autoware_carla_scenario.metrics`,
`BUILT_IN_METRICS`). An event reads `1` while it holds and `0` while it does
not; every metric reads nothing without an ego, and those of the road nothing
without a map. A metric reads one frame; AlpaSim folds a run's frames into one
value with the aggregation in the last column (`Metric.aggregation`).

| Key | Unit | Reads | AlpaSim folds with |
|---|---|---|---|
| `collision_any` | | The ego's bounding box overlaps another vehicle's or a pedestrian's | max |
| `collision_front` | | ... and the other overlaps the ego's front edge | max |
| `collision_rear` | | ... its rear edge, and not its front | max |
| `collision_lateral` | | ... neither edge: a side swipe | max |
| `collision_at_fault` | | A front or a lateral collision | max |
| `offroad` | | A corner of the ego's box is on no driving lane | max |
| `offroad_or_collision` | | Off the road, or any collision | max |
| `offroad_or_collision_at_fault` | | Off the road, or a collision at fault | max |
| `wrong_lane` | | The ego heads more than 120 degrees away from the driving lane under its centre; nothing off the road | max |
| `min_distance_to_obstacle_m` | m | Box to box, to the nearest other vehicle or pedestrian; `0` touching, nothing when the ego is alone | min |
| `min_distance_to_lane_boundary_m` | m | From the ego's box to the edge of the driving lanes around it (lanes beside its own count while their near edge is within 6 m of its centre); `0` off the road | min |

Where AlpaSim's geometry is not CARLA's, the reading follows CARLA: AlpaSim
asks whether the union of the lane polygons around the ego covers its box,
CARLA whether a point is on a driving lane, so `offroad` asks it of the box's
corners; the lane boundary takes the lanes beside the ego's as parallel strips
as wide as CARLA says they are there, an approximation through a junction.
Boxes are those of vehicles (`vehicle.*`) and pedestrians (`walker.*`); props
are not road users. `collision_*` reads the boxes, not CARLA's collision
sensor (`CollisionCondition`): it sees a collision as AlpaSim does, and tells
the front from the rear.

AlpaSim's other metrics need more than the world, and are not metrics here:

| AlpaSim metric | Needs |
|---|---|
| `progress`, `progress_rel`, `dist_to_gt_trajectory`, `dist_to_gt_location`, `gt_dist_traveled_m` | The logged ground-truth trajectory of the ego |
| `min_ade@*`, `plan_deviation`, `open_loop_collision` | The driver's planned trajectories |
| `safety_monitor_triggered` | The driver's safety monitor |
| `img_is_black` | A camera image |
| `dist_traveled_m`, `duration_frac_20s`, `avg_dist_between_incidents` | The run, not a frame |

## Watching a metric

`MetricCondition(metric, rule, value, tolerance, *, label)` holds while the
metric satisfies the comparison. The rule defaults to
`GREATER_THAN_OR_EQUAL` and the value to `1`, so an event metric on its own
reads "it happens". A frame the metric reads nothing in does not satisfy it.
It is usually a fail condition:

```python
from autoware_carla_scenario import ComparisonRule, MetricCondition
from autoware_carla_scenario.metrics import (
    COLLISION_AT_FAULT,
    MIN_DISTANCE_TO_OBSTACLE_M,
    OFFROAD,
)


class MyScenario(BaseScenario):
    def setup(self) -> None:
        ...
        self.register_fail_condition(
            MetricCondition(COLLISION_AT_FAULT, label="collision at fault")
        )
        self.register_fail_condition(MetricCondition(OFFROAD, label="off the road"))
        self.register_fail_condition(
            MetricCondition(
                MIN_DISTANCE_TO_OBSTACLE_M,
                ComparisonRule.LESS_THAN,
                0.5,
                label="closer than half a metre",
            )
        )
```

It composes like any other condition: `PersistentCondition` to require a
metric to hold for a while (off the road for more than a second), `NotCondition`
to pass only while it does not. In the [Scenario Editor](scenario_editor.md) it
is the **Metric** card, under World.

To read a metric yourself -- in a probe, a test, a recorder --
`read_metric(key, world)` returns its value in the frame, or `None`.
