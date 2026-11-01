# Pedestrian Motions (SOMA, ARDY)

Two actions take over a pedestrian's skeleton:

- `PlayMotionAction` plays a motion file. It makes a pedestrian wave, beckon,
  or do anything else the file holds, using a
  [SOMA-X](https://github.com/NVlabs/SOMA-X) motion on the walker's skeleton.
- `SmartPedestrianControllerAction` has a model generate the motion while the
  run plays. The model is [ARDY](https://github.com/nv-tlabs/ardy), and it
  moves the whole body and where it goes, following text, waypoints and
  keyframes ([below](#smartpedestriancontrolleraction-ardy)).

Both share the core in `autoware_carla_scenario.pedestrian`:

- `skeleton` holds the walker's bones and how bone directions pose them.
- `soma` reads a SOMA file.
- `poser` takes a walker over, poses it, places it, and hands it back to its
  animation.

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
- The body is kept on the ground. The pose's lowest point is worked out from
  the walker's own bones, the directions it was given and the flesh around
  each bone (`CarlaBone.radius`), never from where a motion places its
  joints. The hips are then lowered so that point stays where the soles stood
  (`WalkerRetarget.lift`). A crouch, a sit or a fall lies on the ground
  instead of hanging from the hips.
- Only the mesh is lowered. The walker actor, its capsule and its physics
  stay where CARLA has them, so it moves, reports its velocity and collides
  as before. A vehicle that hits a pedestrian lying in its lane is seen by a
  collision sensor, and the Traffic Manager's vehicles stop for it. Its
  bounding box stays a standing one.

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

## SmartPedestrianControllerAction (ARDY)

`SmartPedestrianControllerAction` moves a pedestrian as
[ARDY](https://github.com/nv-tlabs/ardy) generates it. ARDY is NVIDIA's
autoregressive diffusion model of human motion. You tell it what the pedestrian
does in any of three ways:

- **Text.** The whole motion has a `prompt`. Each `TimedPrompt` takes over from
  its time into the motion.
- **Waypoints.** These are CARLA world points (`WalkWaypoint`) that the
  pedestrian passes in order, walking at `speed_ms` from where it stands.
- **Keyframes.** A JSON file of full-body poses that the motion passes through
  ([format](#keyframes)).

```python
from autoware_carla_scenario import (
    ElapsedTimeCondition,
    SmartPedestrianControllerAction,
    TimedPrompt,
    WalkWaypoint,
)

scenario.register_pre_tick(
    SmartPedestrianControllerAction(
        "walker1",
        "a man walks to the kerb and looks left and right",
        condition=ElapsedTimeCondition(2.0),
        prompts=[TimedPrompt(4.0, "he crosses the road in a hurry")],
        waypoints=[WalkWaypoint(104.0, -12.5), WalkWaypoint(104.0, -4.0)],
        speed_ms=1.4,
        seed=3,
    )
)
```

In the Scenario Editor it is the *Smart Pedestrian* card, under *Pedestrian*.

### How it runs

ARDY runs in its own container,
[`ghcr.io/hakuturu583/carla_ardy_backend`](https://github.com/hakuturu583/carla_ardy_backend),
on a GPU of the host that runs the scenario. The action declares that it needs
it (`@need_prepare(ArdyBackend)`). The queue then starts the container before
the run, after checking that the host has room for it, and stops it when the
batch ends. A host without room fails the run before it starts
([Backends a Run Starts](backends.md)).

When the action fires:

1. It opens a session on the generator with the pedestrian's pose and every
   condition.
2. It asks the generator for the next couple of seconds of motion.
3. Every tick, it places the walker at the motion's root and poses all of its
   bones, on the ground as above. The walker is told to walk only the way it
   faces, at the motion's speed that way: CARLA turns a walker towards where
   it is told to go, and a motion's root drifts and steps aside. A step
   sideways or back therefore reads as standing in the walker's velocity,
   though the walker goes where the motion takes it
   ([#957](https://github.com/autowarefoundation/autoware_carla_scenario/issues/957)).
4. Each time the motion it holds is about to run out, it asks for more,
   generated from the motion shown so far.

CARLA runs synchronously, so the simulation waits for each answer, and the same
`seed` replays the same motion.

The action takes the pedestrian's **movement and animation**
[control domains](architecture.md) for as long as it moves the walker. It runs
for `duration_s`. Without that, it runs until its last waypoint's time, and
without waypoints, until the scenario ends.

The generator and the action talk over
`proto/pedestrian_motion/v0/pedestrian_motion.proto`, which uses a right-handed,
z-up frame (CARLA's world with y negated). The generator receives joint
positions, and the action turns them into bone directions
(`pedestrian.skeleton.directions_from_joints`).

### Keyframes

A keyframes file lists full-body poses:

```json
{
  "keyframes": [
    {
      "time_s": 3.0,
      "x": 104.0, "y": -12.5, "yaw_deg": 90.0,
      "joints": {"Hips": [0.0, 0.0, 0.95], "Head": [0.05, 0.0, 1.62], "...": [0, 0, 0]}
    }
  ]
}
```

- `time_s` is the time into the motion.
- `x`, `y` and `yaw_deg` place the pose in CARLA's world. Without them, the pose
  is placed where the motion began and faces the way it did then.
- `joints` gives each joint the generator places (the `joint_names` it answers
  with; ARDY's SOMA model has 30) in metres. The axes are forward, left and up
  from the root's ground point, as the body faces.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `entity_name` | | The pedestrian |
| `prompt` | `""` | What it does, in words, from the start |
| `prompts` | `()` | `TimedPrompt(time_s, text)`: texts that take over later |
| `waypoints` | `()` | `WalkWaypoint(x, y)`: CARLA world points it passes |
| `speed_ms` | `1.3` | Walking speed along the waypoints, m/s |
| `keyframes` | `None` | A keyframes file (relative paths are taken from the working directory) |
| `duration_s` | `None` | Stop after this long |
| `seed` | `0` | The generator's draws |
| `overrides` | `False` | Take over a domain another action holds, stopping it |

The backend's own settings, including its image, GPU memory and Hugging Face
cache, are `backends.ardy` of the config ([Backends a Run Starts](backends.md)).
