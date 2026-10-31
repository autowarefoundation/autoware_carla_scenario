# Pedestrian Motions (SOMA)

`PlayMotionAction` makes a pedestrian wave, beckon, or do anything else a
motion file holds: it plays a [SOMA-X](https://github.com/NVlabs/SOMA-X) motion
on the walker's skeleton.

```python
from autoware_carla_scenario import (
    ElapsedTimeCondition,
    PlayMotionAction,
    WalkStraightAction,
)

# Stop the pedestrian at the kerb, then wave at the ego for three seconds.
scenario.register_pre_tick(
    WalkStraightAction("walker1", speed_ms=0.0, condition=ElapsedTimeCondition(4.0))
)
scenario.register_pre_tick(
    PlayMotionAction(
        "walker1",
        "motions/wave.npz",
        condition=ElapsedTimeCondition(5.0),
        loop=True,
        duration_s=3.0,
    )
)
```

In the Scenario Editor it is the *Play Motion* card, under *Pedestrian*.

## The motion file

SOMA-X moves one skeleton, the SOMA rig, whatever identity model (MHR, SOMA,
Anny, SMPL) shapes the body, so a motion is the same file for every one of
them. The action reads the `.npz` that `soma.io.save_soma_npz` writes, which
SOMA-X's converters produce too:

```bash
# An AMASS clip as a SOMA motion
python -m tools.convert_amass_to_soma --input wave_amass.npz \
    --output-npz motions/wave.npz --no-render
```

| Key | What the action reads |
|---|---|
| `poses` | Each joint's rotation relative to its parent: `(N, J, 3)` axis-angle or `(N, J, 3, 3)` matrices |
| `joint_names` | The joints' names, in the order of `poses` (the SOMA rig's order if absent) |
| `absolute_pose`, `joint_orient` | Poses relative to the T-pose need its `joint_orient`; absolute poses need nothing more |
| `fps` | The frame rate, if the file stores one (`save_soma_npz` does not; give `fps=`, or 30 is assumed) |

`transl`, the identity coefficients and the units are not read: the walker
stays where CARLA puts it. A hand motion (`hand_type`) is refused. The file is
read when the action is built, so a missing or malformed file fails the
scenario before it runs. SOMA-X itself is not a dependency: the file is read
with numpy.

## How it is shown

A CARLA walker's mesh is its blueprint's and cannot be swapped from a client,
so the motion is **retargeted** onto the walker's skeleton (its `crl_*` bones)
rather than replayed joint for joint:

- Each SOMA joint's frame has its local X axis along the bone to its child,
  which gives the bone's direction in every frame without the rig's bone
  lengths. The first frame's facing is turned to the walker's.
- Each driven CARLA bone is swung, parents first, from where the walker's
  animation holds it onto that direction (the shortest rotation, so its twist
  stays the animation's), and the result is sent with `Walker.set_bones()` and
  shown with `blend_pose(1.0)`, one frame per tick at the motion's rate in
  simulated time.
- The driven bones are the spine, the neck and the arms (with
  `upper_body_only=False`, the legs too). The hips never are: where the walker
  stands and faces stays CARLA's.

When the motion ends (with `loop`, when `duration_s` is up) the walker is
handed back to its animation (`blend_pose(0.0)`).

!!! warning "A walker plays a motion standing"
    CARLA's custom pose is the whole skeleton: the bones the motion does not
    drive hold the animation frame the motion began on, so a walking
    pedestrian would glide with frozen legs. The action therefore fires only
    while the pedestrian stands (no faster than `STANDING_SPEED_MS`, 0.1 m/s)
    *and* its trigger holds: stop it first (`WalkStraightAction` at speed 0).
    A walking pedestrian's motion waits until it stands.

## Who controls the pedestrian

The motion takes the pedestrian's **animation**
[control domain](architecture.md): a second motion on it while one plays ends
the run in error, unless that one is built with `overrides=True` (*Override a
running action* on its card), which stops the first.

Driving the whole body (`upper_body_only=False`), the motion takes the
pedestrian's **movement** (longitudinal and lateral) too, since its legs are
the motion's: a `WalkStraightAction` or a route on it while it plays is an
error. With `upper_body_only=True` the movement stays free; a walk started
under the motion would glide the pedestrian with its legs frozen.

A bone the walker's skeleton lacks is left to the animation, with a warning
naming it; the bone names are those of CARLA's walkers (`crl_arm__L`,
`crl_foreArm__L`, ...).

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `entity_name` | | The pedestrian |
| `motion_path` | | The `.npz`; a relative path is taken from the working directory |
| `fps` | the file's, else 30 | Frames per second |
| `loop` | `False` | Play the motion over again when it ends |
| `duration_s` | `None` | Stop after this long; `None`: when the motion ends, or, looping, at the scenario's end |
| `upper_body_only` | `True` | Drive the spine, neck and arms only (animation domain); `False`: the whole body (animation and movement) |
| `overrides` | `False` | Take over a domain another action holds, stopping it |
