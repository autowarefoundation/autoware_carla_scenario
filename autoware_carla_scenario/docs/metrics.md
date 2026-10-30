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
| `importance_sampling_weight` | ratio | Not AlpaSim's: the run's importance-sampling weight so far, the product of NADE's `p/q` and `(1-p)/(1-q)` (`AdversarialTrafficAction`); `1` without NADE. Weight a batch of runs by its last value to estimate the naturalistic rate of what they count | last |
| `lateral_reversals` | count | Not AlpaSim's: how many times so far the ego's drift across its lane turned back by more than 0.1 m ([Weaving](#weaving)) | last |
| `lateral_wobble_m` | m | Not AlpaSim's: how far the ego strayed across its lane about its course over the last 4 s, RMS ([Weaving](#weaving)) | max |
| `steering_reversals` | count | Not AlpaSim's: how many times so far the steering applied to the ego turned back by more than 0.05 of full lock ([Weaving](#weaving)) | last |
| `turn_indicator_changes` | count | Not AlpaSim's: how many times so far the ego's turn indicators changed, between off, left, right and hazard ([Weaving](#weaving)) | last |

Where AlpaSim's geometry is not CARLA's, the reading follows CARLA: AlpaSim
asks whether the union of the lane polygons around the ego covers its box,
CARLA whether a point is on a driving lane, so `offroad` asks it of the box's
corners; the lane boundary takes the lanes beside the ego's as parallel strips
as wide as CARLA says they are there, an approximation through a junction.
Boxes are those of vehicles (`vehicle.*`) and pedestrians (`walker.*`); props
are not road users. `collision_*` reads the boxes, not CARLA's collision
sensor (`CollisionCondition`): it sees a collision as AlpaSim does, and tells
the front from the rear.

### Weaving

Three metrics read how the ego weaves from side to side, and a fourth how
often its turn indicators changed. Unlike the others they read the run so far,
not one frame: the runner hands the ego's motion to
`autoware_carla_scenario.metrics` every tick (`track_ego_motion`), and they read
what it has gathered.

The ego's drift is how far it has moved across its lanes since the run began:
its velocity across the lane under its centre, summed over time. The lane's
heading is taken anew every frame, so following a bend is not a drift, and
moving to the next lane is a drift of a lane's width with no jump at the lane
line. Off the driving lanes the drift holds until the ego is back on one.

- `lateral_reversals` counts the times the drift turned back by more than
  0.1 m: it went right, then came back left by 0.1 m or more. This is the
  reversal count of driver-workload studies (Markkula and Engström, 2006). A
  lane change is one drift, however long, and adds nothing; a weave adds two a
  cycle. Under 0.1 m it is the noise of holding a lane.
- `lateral_wobble_m` is how far the drift strayed about its course over the
  last 4 s: the RMS about the straight line that fits it best there. A weave of
  0.25 Hz or faster turns back within the window and reads nearly in full
(about 92% at 0.5 Hz); a lane
  change, slower, is mostly the course, and adds about a tenth of the lane's
  width while it lasts. It reads nothing before the third frame.
- In a junction, whose lanes overlap, the drift holds, as it does off the
  driving lanes; a lane the other way is crossed the ego's way.
- `steering_reversals` counts the times the steering applied to the ego
  (`get_control().steer`, `1` the vehicle's full lock) turned back by more
  than 0.05, about 3.5 degrees at the wheels of a car that locks at 70. It sees
  the controller's weaving before the vehicle shows it. Closed loop, a driver
  policy's controller moves its steer by a few hundredths while it holds a
  lane or a bend, which the gap leaves out; sawing through a turn is a tenth
  or more.

`turn_indicator_changes` counts the changes of the indicators the ego shows
(its CARLA light state: off, left, right, hazard), from the first frame on: a
turn signalled and cancelled is two. It is counted apart from the steering, so
an indicator that flickers through a turn and a steering that saws there each
read on their own count. A vehicle's light state holds while an indicator is
commanded, so its blinking is not a change.

The counts are of the run so far, so a longer run reads more: compare runs of
one scenario, or fail on a count, not on a rate:

```python
self.register_fail_condition(
    MetricCondition(LATERAL_REVERSALS, ComparisonRule.GREATER_THAN, 6, label="weaving")
)
self.register_fail_condition(
    MetricCondition(LATERAL_WOBBLE_M, ComparisonRule.GREATER_THAN, 0.5, label="weaving")
)
```

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

A run a metric condition fails is a failure of what the metric reads
([Why a run failed](usage.md#output-files)): a collision
metric `collision`, `offroad` and `min_distance_to_lane_boundary_m`
`lane_keeping`, `wrong_lane` `traffic_rule`, `min_distance_to_obstacle_m`
`safety_margin`, and the weaving metrics and `turn_indicator_changes`
`stability`. `offroad_or_collision` and its at-fault twin read two things and
say neither, nor does the importance-sampling weight: register a condition on
them with its `category`.

To read a metric yourself -- in a probe, a test, a recorder --
`read_metric(key, world)` returns its value in the frame, or `None`.

### Off the road as a fail condition

`offroad` is a measurement, and a strict one: it reads `1` as soon as any one
corner of the ego's box is on no driving lane. A corner over a gap between
lanes at a junction entry, or a small offset between the OpenDRIVE lanes and
where the vehicle drives, counts though the vehicle is in its lane, so
`MetricCondition(OFFROAD, ...)` fails such runs. To judge a run by how much of
the box is off instead, use `OffRoadFractionCondition(fraction, entity_name,
*, label)`, a condition rather than a metric:

```python
from autoware_carla_scenario import OffRoadFractionCondition

self.register_fail_condition(OffRoadFractionCondition(0.3, label="offroad"))
```

Every tick it samples a 5 x 3 grid of points over the entity's bounding box
(the ego's unless `entity_name` names another): 5 evenly spaced along its
length and 3 across its width, the box's edges and corners included. A point
is off when `get_waypoint(point, project_to_road=False,
lane_type=carla.LaneType.Driving)` finds no lane there, asked at the height of
the entity, so that where roads cross at different heights the lane is the one
it is on. It fails once at least `fraction` (in `(0, 1]`) of the 15 points are
off. The fraction is of points, not of area: one corner off is 1/15, the
front edge 3/15, one long side 5/15. `fraction=1/15` fails on any one point,
the way the metric does on a corner; `0.3` fails at 5 points, so it tolerates
a corner or the front edge over a lane gap and fails once a whole long side
of the box is off.
In the [Scenario Editor](scenario_editor.md) it is the **Off the road** card,
under Entity.

A scenario or a Python ODD may import any name of
`autoware_carla_scenario.metrics` -- the keys, `Metric`, `BUILT_IN_METRICS`,
`is_metric()` and `read_metric()` -- and is checked against them like any
other ([Static Check](typecheck.md)). A `Metric` is frozen: assigning to one of
its fields fails the check, as it fails the run. A scenario may build its own
(`Metric("my_metric", read)`); the check calls its `read` with a world and
refuses one that returns anything but a `float` or an `Optional[float]`.
