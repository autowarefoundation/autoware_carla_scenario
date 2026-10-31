"""A SOMA motion played on a pedestrian's skeleton (PlayMotionAction).

The motion files are made here, the way ``soma.io.save_soma_npz`` writes
them, around a rig whose joint frames are laid out by hand: each joint's
local ``+X`` along its bone (``-X`` on the right side), as in SOMA's.  CARLA
is faked: a walker standing at attention, whose ``set_bones`` keeps what it
was given.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    PedestrianEntity,
    PedestrianEntityConfig,
    PlayMotionAction,
    SpawnTransform,
)
from autoware_carla_scenario.action_state import ActionState
from autoware_carla_scenario.entity._soma_motion import (
    CARLA_BONES,
    SOMA_JOINTS,
    Matrix,
    Vector,
    WalkerRetarget,
    load_soma_motion,
    rotation_matrix,
    rotator,
    swing,
)
from autoware_carla_scenario.entity.registry import register_entity, unregister_entity

FPS = 10.0
DT = 0.05


def _rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _t_pose_orient(joint: str) -> np.ndarray:
    """A T-pose world orientation for *joint*: its bone along local ``+X``
    (``-X`` on the right), the hips facing along their local ``+Y``."""
    if joint == "Hips":
        # X up, Y forward, Z left.
        return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if "Leg" in joint or "Shin" in joint:
        return _rz(-90.0) if joint.startswith("Left") else _rz(90.0)
    if "Foot" in joint or "Toe" in joint:
        return _ry(-90.0) if joint.startswith("Left") else _ry(90.0)
    if joint.startswith(("Left", "Right")):
        return np.eye(3)  # an arm: left along +X, right along -X
    return _rz(90.0)  # the spine and the head: up


def _rotvec(matrix: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(matrix) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Half a turn: the axis is the column of R + I that is not zero.
        column = (matrix + np.eye(3))[:, int(np.argmax(np.diag(matrix)))]
        return column / np.linalg.norm(column) * math.pi
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _write_motion(
    path: Path,
    frames: list[dict[str, np.ndarray]],
    *,
    absolute: bool = False,
    rotvec: bool = True,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """An ``.npz`` as ``save_soma_npz`` writes it (no Root), of *frames*: each
    a joint's rotation relative to the T-pose, by name (identity if absent)."""
    names = [name for name, _ in SOMA_JOINTS]
    orient = np.stack([np.eye(3)] + [_t_pose_orient(n) for n in names])  # Root first
    parents = dict(SOMA_JOINTS)
    poses = np.zeros((len(frames), len(names), 3, 3))
    for f, frame in enumerate(frames):
        for j, name in enumerate(names):
            relative = frame.get(name, np.eye(3))
            if absolute:
                # apply_joint_orient_local: orient_parent^T @ R @ orient.
                parent = parents[name]
                above = np.eye(3) if parent == "Root" else _t_pose_orient(parent)
                relative = above.T @ relative @ _t_pose_orient(name)
            poses[f, j] = relative
    stored: np.ndarray = poses
    if rotvec:
        stored = np.array([[_rotvec(m) for m in frame] for frame in poses])
    data: dict[str, Any] = {
        "poses": stored,
        "transl": np.zeros((len(frames), 3)),
        "joint_names": np.array(names),
        "identity_model_type": np.array("anny"),
        "identity_coeffs": np.zeros((1, 4)),
        "keep_root": np.bool_(False),
        "absolute_pose": np.bool_(absolute),
        "rotation_repr": np.array("rotvec" if rotvec else "matrix"),
        "unit": np.array("meters"),
    }
    if not absolute:
        data["joint_orient"] = orient
    data.update(extra or {})
    np.savez(path, **data)
    return path


#: The left arm raised straight up, a wave's top: the arm's rotation about the
#: forward axis carries the forearm and the hand with it.
RAISED: dict[str, np.ndarray] = {"LeftArm": _rz(90.0)}


def _close(a: Vector, b: Vector, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ----------------------------------------------------------------------
# Reading a motion
# ----------------------------------------------------------------------


class TestLoad:
    def test_the_t_pose_points_the_bones_as_the_body_holds_them(
        self, tmp_path: Path
    ) -> None:
        motion = load_soma_motion(_write_motion(tmp_path / "m.npz", [{}]))
        frame = motion.frame(0)
        # (forward, right, up)
        assert _close(frame["LeftArm"], (0.0, -1.0, 0.0))
        assert _close(frame["RightArm"], (0.0, 1.0, 0.0))
        assert _close(frame["Spine1"], (0.0, 0.0, 1.0))
        assert _close(frame["LeftShin"], (0.0, 0.0, -1.0))
        assert _close(frame["RightFoot"], (1.0, 0.0, 0.0))

    def test_a_raised_arm_carries_its_forearm_and_hand(self, tmp_path: Path) -> None:
        motion = load_soma_motion(_write_motion(tmp_path / "m.npz", [{}, RAISED]))
        frame = motion.frame(1)
        for joint in ("LeftArm", "LeftForeArm", "LeftHand"):
            assert _close(frame[joint], (0.0, 0.0, 1.0)), joint
        assert _close(frame["RightArm"], (0.0, 1.0, 0.0))

    @pytest.mark.parametrize("rotvec", [True, False])
    def test_absolute_and_relative_poses_read_the_same(
        self, tmp_path: Path, rotvec: bool
    ) -> None:
        relative = load_soma_motion(_write_motion(tmp_path / "r.npz", [{}, RAISED]))
        absolute = load_soma_motion(
            _write_motion(
                tmp_path / "a.npz", [{}, RAISED], absolute=True, rotvec=rotvec
            )
        )
        np.testing.assert_allclose(absolute.directions, relative.directions, atol=1e-9)

    def test_the_motion_faces_the_way_its_first_frame_does(
        self, tmp_path: Path
    ) -> None:
        turned = {"Hips": _ry(90.0)}
        plain = load_soma_motion(_write_motion(tmp_path / "p.npz", [{}, RAISED]))
        facing = load_soma_motion(
            _write_motion(
                tmp_path / "t.npz", [turned, {**RAISED, **turned}], rotvec=False
            )
        )
        np.testing.assert_allclose(facing.directions, plain.directions, atol=1e-9)

    def test_the_frame_rate_is_the_file_s_unless_given(self, tmp_path: Path) -> None:
        plain = _write_motion(tmp_path / "p.npz", [{}])
        timed = _write_motion(tmp_path / "t.npz", [{}], extra={"fps": np.float32(24)})
        assert load_soma_motion(plain).fps == 30.0
        assert load_soma_motion(timed).fps == 24.0
        assert load_soma_motion(timed, fps=12.0).fps == 12.0

    def test_a_relative_pose_needs_the_t_pose(self, tmp_path: Path) -> None:
        path = _write_motion(tmp_path / "m.npz", [{}])
        with np.load(path) as data:
            kept = {k: data[k] for k in data.files if k != "joint_orient"}
        np.savez(path, **kept)
        with pytest.raises(ValueError, match="joint_orient"):
            load_soma_motion(path)

    def test_a_hand_motion_is_refused(self, tmp_path: Path) -> None:
        path = _write_motion(
            tmp_path / "m.npz", [{}], extra={"hand_type": np.array("left")}
        )
        with pytest.raises(ValueError, match="hand motion"):
            load_soma_motion(path)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_soma_motion(tmp_path / "absent.npz")


# ----------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pitch", "yaw", "roll"),
    [(0.0, 0.0, 0.0), (30.0, -120.0, 45.0), (-70.0, 170.0, -100.0), (10.0, 90.0, 0.0)],
)
def test_a_rotation_s_matrix_turns_back_into_it(
    pitch: float, yaw: float, roll: float
) -> None:
    back = rotator(rotation_matrix(carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)))
    assert (back.pitch, back.yaw, back.roll) == pytest.approx(
        (pitch, yaw, roll), abs=1e-3
    )


def test_a_rotation_s_matrix_has_unreal_s_axes() -> None:
    m = rotation_matrix(carla.Rotation(yaw=90.0))
    assert _close((m[0][0], m[1][0], m[2][0]), (0.0, 1.0, 0.0))  # forward: +y
    assert _close((m[0][1], m[1][1], m[2][1]), (-1.0, 0.0, 0.0))  # right: -x
    pitched = rotation_matrix(carla.Rotation(pitch=90.0))
    assert _close((pitched[0][0], pitched[1][0], pitched[2][0]), (0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.6, 0.0, 0.8), (0.0, -0.6, 0.8)),
    ],
)
def test_a_swing_takes_one_direction_onto_the_other(a: Vector, b: Vector) -> None:
    m = swing(a, b)
    turned = tuple(sum(m[i][k] * a[k] for k in range(3)) for i in range(3))
    assert _close(turned, b)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    [
        "CARLA_BONES",
        "CarlaBone",
        "Matrix",
        "Vector",
        "WalkerRetarget",
        "rotation_matrix",
        "rotator",
        "swing",
    ],
)
def test_the_soma_motion_module_re_exports_the_walker_retarget(name: str) -> None:
    from autoware_carla_scenario.entity import _soma_motion, _walker_retarget

    assert name in _walker_retarget.__all__
    assert getattr(_soma_motion, name) is getattr(_walker_retarget, name)


# ----------------------------------------------------------------------
# A CARLA walker
# ----------------------------------------------------------------------

#: A walker at attention, facing +x (its right +y): where each bone is.
_AT_ATTENTION: dict[str, tuple[str, Vector]] = {
    "crl_root": ("", (0.0, 0.0, 0.0)),
    "crl_hips__C": ("crl_root", (0.0, 0.0, 1.0)),
    "crl_spine__C": ("crl_hips__C", (0.0, 0.0, 1.1)),
    "crl_spine01__C": ("crl_spine__C", (0.0, 0.0, 1.3)),
    "crl_neck__C": ("crl_spine01__C", (0.0, 0.0, 1.5)),
    "crl_Head__C": ("crl_neck__C", (0.0, 0.0, 1.6)),
    "crl_shoulder__L": ("crl_spine01__C", (0.0, -0.05, 1.45)),
    "crl_arm__L": ("crl_shoulder__L", (0.0, -0.2, 1.45)),
    "crl_foreArm__L": ("crl_arm__L", (0.0, -0.2, 1.15)),
    "crl_hand__L": ("crl_foreArm__L", (0.0, -0.2, 0.9)),
    "crl_handMiddle__L": ("crl_hand__L", (0.0, -0.2, 0.8)),
    "crl_shoulder__R": ("crl_spine01__C", (0.0, 0.05, 1.45)),
    "crl_arm__R": ("crl_shoulder__R", (0.0, 0.2, 1.45)),
    "crl_foreArm__R": ("crl_arm__R", (0.0, 0.2, 1.15)),
    "crl_hand__R": ("crl_foreArm__R", (0.0, 0.2, 0.9)),
    "crl_handMiddle__R": ("crl_hand__R", (0.0, 0.2, 0.8)),
    "crl_thigh__L": ("crl_hips__C", (0.0, -0.1, 0.95)),
    "crl_leg__L": ("crl_thigh__L", (0.0, -0.1, 0.5)),
    "crl_foot__L": ("crl_leg__L", (0.0, -0.1, 0.1)),
    "crl_toe__L": ("crl_foot__L", (0.15, -0.1, 0.0)),
    "crl_thigh__R": ("crl_hips__C", (0.0, 0.1, 0.95)),
    "crl_leg__R": ("crl_thigh__R", (0.0, 0.1, 0.5)),
    "crl_foot__R": ("crl_leg__R", (0.0, 0.1, 0.1)),
    "crl_toe__R": ("crl_foot__R", (0.15, 0.1, 0.0)),
}

#: The bones' world rotations: not the identity, so a bone's frame and its
#: direction are not the same thing.
_BONE_ROTATION = carla.Rotation(pitch=20.0, yaw=-35.0, roll=60.0)


def _bones() -> list[Any]:
    """``get_bones().bone_transforms`` of the walker at attention."""
    bones = []
    for name, (parent, at) in _AT_ATTENTION.items():
        offset = (
            at
            if not parent
            else tuple(a - b for a, b in zip(at, _AT_ATTENTION[parent][1]))
        )
        inverse = _transpose(rotation_matrix(_BONE_ROTATION))
        local = _apply(inverse, offset) if parent else offset  # type: ignore[arg-type]
        bones.append(
            SimpleNamespace(
                name=name,
                world=carla.Transform(carla.Location(*at), _BONE_ROTATION),
                relative=carla.Transform(
                    carla.Location(*local),
                    carla.Rotation() if parent else _BONE_ROTATION,
                ),
            )
        )
    return bones


def _transpose(m: Matrix) -> Matrix:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def _apply(m: Matrix, v: Vector) -> Vector:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _mul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _posed_positions(posed: dict[str, carla.Transform]) -> dict[str, Vector]:
    """Where each bone is once *posed* (relative transforms) is applied."""
    rest = {b.name: b for b in _bones()}
    world: dict[str, tuple[Matrix, Vector]] = {}
    for name, (parent, at) in _AT_ATTENTION.items():
        relative = posed.get(name, rest[name].relative)
        rotation = rotation_matrix(relative.rotation)
        if not parent:
            world[name] = (rotation, at)
            continue
        above, origin = world[parent]
        loc = relative.location
        step = _apply(above, (loc.x, loc.y, loc.z))
        world[name] = (
            _mul(above, rotation),
            (origin[0] + step[0], origin[1] + step[1], origin[2] + step[2]),
        )
    return {name: at for name, (_, at) in world.items()}


def _direction(positions: dict[str, Vector], bone: str, child: str) -> Vector:
    a, b = positions[bone], positions[child]
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = math.sqrt(sum(x * x for x in d))
    return (d[0] / n, d[1] / n, d[2] / n)


class TestRetarget:
    def test_a_frame_points_each_driven_bone_where_the_motion_does(
        self, tmp_path: Path
    ) -> None:
        motion = load_soma_motion(_write_motion(tmp_path / "m.npz", [{}, RAISED]))
        retarget = WalkerRetarget(
            _bones(), carla.Rotation(yaw=90.0), upper_body_only=False
        )
        assert retarget.missing == ()
        positions = _posed_positions(dict(retarget.pose(motion.frame(1))))
        # Facing +y, its right is -x, so its left is +x.
        assert _close(_direction(positions, "crl_arm__L", "crl_foreArm__L"), (0, 0, 1))
        assert _close(
            _direction(positions, "crl_hand__L", "crl_handMiddle__L"), (0, 0, 1)
        )
        assert _close(_direction(positions, "crl_arm__R", "crl_foreArm__R"), (-1, 0, 0))
        assert _close(_direction(positions, "crl_leg__L", "crl_foot__L"), (0, 0, -1))
        assert _close(_direction(positions, "crl_foot__R", "crl_toe__R"), (0, 1, 0))
        # The hips stay where the walker holds them.
        assert _close(positions["crl_hips__C"], (0.0, 0.0, 1.0))

    def test_upper_body_only_leaves_the_legs_to_the_animation(self) -> None:
        retarget = WalkerRetarget(_bones(), carla.Rotation())
        assert "crl_thigh__L" not in retarget.driven
        assert "crl_arm__L" in retarget.driven
        everything = WalkerRetarget(_bones(), carla.Rotation(), upper_body_only=False)
        assert set(everything.driven) == {bone.name for bone in CARLA_BONES}

    def test_a_bone_the_walker_lacks_is_left_out(self) -> None:
        bones = [b for b in _bones() if b.name != "crl_handMiddle__L"]
        retarget = WalkerRetarget(bones, carla.Rotation())
        assert retarget.missing == ("crl_hand__L",)
        assert "crl_hand__L" not in retarget.driven


# ----------------------------------------------------------------------
# The action
# ----------------------------------------------------------------------


class _Walker:
    """A walker that keeps the custom pose it is given."""

    def __init__(self) -> None:
        self.type_id = "walker.pedestrian.0001"
        self.blend: float = 0.0
        self.poses: list[dict[str, carla.Transform]] = []
        self.from_animation = 0

    def get_bones(self) -> Any:
        return SimpleNamespace(bone_transforms=_bones())

    def get_transform(self) -> carla.Transform:
        return carla.Transform(carla.Location(), carla.Rotation())

    def get_pose_from_animation(self) -> None:
        self.from_animation += 1

    def set_bones(self, control: carla.WalkerBoneControlIn) -> None:
        self.poses.append({b.name: b.transform for b in control.bone_transforms})

    def blend_pose(self, blend: float) -> None:
        self.blend = blend


class _Scene:
    def __init__(self) -> None:
        self.walker = _Walker()
        self.entity = PedestrianEntity(
            PedestrianEntityConfig(
                role_name="walker1", spawn_location=SpawnTransform(carla.Transform())
            )
        )
        self.entity._walker = self.walker  # type: ignore[assignment]
        self.world = MagicMock()
        self.elapsed = 0.0
        self.world.get_snapshot.side_effect = lambda: SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=self.elapsed)
        )

    def run(self, action: PlayMotionAction, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            action.tick(self.world, self.elapsed)
            self.elapsed += DT


@pytest.fixture
def scene() -> Iterator[_Scene]:
    built = _Scene()
    register_entity("walker1", built.entity)
    yield built
    unregister_entity("walker1")


@pytest.fixture
def wave(tmp_path: Path) -> Path:
    """Down, raised, down, raised: four frames at :data:`FPS`."""
    return _write_motion(
        tmp_path / "wave.npz", [{}, RAISED, {}, RAISED], extra={"fps": FPS}
    )


class TestPlayMotionAction:
    def test_the_motion_plays_frame_by_frame_then_hands_back(
        self, scene: _Scene, wave: Path
    ) -> None:
        action = PlayMotionAction("walker1", wave)
        scene.run(action, 0.05)
        assert scene.walker.blend == 1.0
        assert scene.walker.from_animation == 1
        assert action.playing
        scene.run(action, 0.6)
        assert scene.walker.blend == 0.0
        assert action.state is ActionState.COMPLETE
        # 0.4 s of motion at 0.05 s a tick: eight ticks, two of each frame.
        shown = [
            _direction(_posed_positions(p), "crl_arm__L", "crl_foreArm__L")[2] > 0.5
            for p in scene.walker.poses
        ]
        assert shown == [False, False, True, True, False, False, True, True]

    def test_a_looping_motion_stops_after_its_duration(
        self, scene: _Scene, wave: Path
    ) -> None:
        action = PlayMotionAction("walker1", wave, loop=True, duration_s=1.0)
        scene.run(action, 0.95)
        assert action.playing
        assert len(scene.walker.poses) == 19
        scene.run(action, 0.2)
        assert not action.playing
        assert scene.walker.blend == 0.0

    def test_a_pedestrian_that_is_not_there_plays_nothing(self, wave: Path) -> None:
        action = PlayMotionAction("nobody", wave)
        world = MagicMock()
        action.tick(world, 0.0)
        action.tick(world, DT)
        assert not action.playing
        assert action.state is not ActionState.RUNNING

    def test_a_bad_file_fails_when_the_scenario_is_built(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PlayMotionAction("walker1", tmp_path / "absent.npz")

    def test_a_rate_or_duration_must_be_positive(self, wave: Path) -> None:
        with pytest.raises(ValueError, match="fps"):
            PlayMotionAction("walker1", wave, fps=0.0)
        with pytest.raises(ValueError, match="duration_s"):
            PlayMotionAction("walker1", wave, duration_s=-1.0)
