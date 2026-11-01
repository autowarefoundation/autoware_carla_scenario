"""A walker's skeleton, posed by the directions of a motion's bones.

A CARLA walker's skeleton (``crl_*`` bones) is not the SOMA rig's, so a
motion is retargeted onto it by direction: :class:`WalkerRetarget` swings
each bone of :data:`CARLA_BONES` from where the walker's own animation holds
it onto the direction the motion gives the SOMA joint it follows, parents
first, and hands the result to ``Walker.set_bones()`` as transforms relative
to the parent.  The rotation helpers (:func:`rotation_matrix`,
:func:`rotator`, :func:`swing`) follow Unreal's conventions.

Where the directions come from is the caller's: a SOMA ``.npz``
(:mod:`~autoware_carla_scenario.pedestrian.soma`) or the joints a motion
generator places (:func:`directions_from_joints`).
This module needs no numpy, so it is checked by Codon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import typesafe_carla.carla as carla

__all__ = [
    "BONE_CHILDREN",
    "CARLA_BONES",
    "CarlaBone",
    "Matrix",
    "Vector",
    "WalkerRetarget",
    "directions_from_joints",
    "rotation_matrix",
    "rotator",
    "swing",
]


@dataclass(frozen=True)
class CarlaBone:
    """One bone of a CARLA walker that a motion drives.

    Attributes:
        name: The bone (``crl_*``).
        parent: Its parent bone.
        children: Bones it points at, the first the walker has wins: the
            bone's direction is the one from it to that child.
        joint: The SOMA joint whose bone gives its direction.
        upper_body: Whether it is above the hips (driven by default).
    """

    name: str
    parent: str
    children: tuple[str, ...]
    joint: str
    upper_body: bool = True


#: The CARLA walker bones a motion drives, parents first, and the SOMA joint
#: each follows.  The hips are not driven: the walker stands and faces where
#: CARLA puts it.
CARLA_BONES: tuple[CarlaBone, ...] = (
    CarlaBone("crl_spine__C", "crl_hips__C", ("crl_spine01__C",), "Spine1"),
    CarlaBone("crl_spine01__C", "crl_spine__C", ("crl_neck__C",), "Chest"),
    CarlaBone("crl_neck__C", "crl_spine01__C", ("crl_Head__C",), "Neck1"),
    CarlaBone("crl_shoulder__L", "crl_spine01__C", ("crl_arm__L",), "LeftShoulder"),
    CarlaBone("crl_arm__L", "crl_shoulder__L", ("crl_foreArm__L",), "LeftArm"),
    CarlaBone("crl_foreArm__L", "crl_arm__L", ("crl_hand__L",), "LeftForeArm"),
    CarlaBone(
        "crl_hand__L",
        "crl_foreArm__L",
        ("crl_handMiddle__L", "crl_handMiddle01__L", "crl_handIndex__L"),
        "LeftHand",
    ),
    CarlaBone("crl_shoulder__R", "crl_spine01__C", ("crl_arm__R",), "RightShoulder"),
    CarlaBone("crl_arm__R", "crl_shoulder__R", ("crl_foreArm__R",), "RightArm"),
    CarlaBone("crl_foreArm__R", "crl_arm__R", ("crl_hand__R",), "RightForeArm"),
    CarlaBone(
        "crl_hand__R",
        "crl_foreArm__R",
        ("crl_handMiddle__R", "crl_handMiddle01__R", "crl_handIndex__R"),
        "RightHand",
    ),
    CarlaBone("crl_thigh__L", "crl_hips__C", ("crl_leg__L",), "LeftLeg", False),
    CarlaBone("crl_leg__L", "crl_thigh__L", ("crl_foot__L",), "LeftShin", False),
    CarlaBone("crl_foot__L", "crl_leg__L", ("crl_toe__L",), "LeftFoot", False),
    CarlaBone("crl_thigh__R", "crl_hips__C", ("crl_leg__R",), "RightLeg", False),
    CarlaBone("crl_leg__R", "crl_thigh__R", ("crl_foot__R",), "RightShin", False),
    CarlaBone("crl_foot__R", "crl_leg__R", ("crl_toe__R",), "RightFoot", False),
)


#: A 3x3 matrix, row by row.
Matrix = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vector = tuple[float, float, float]


def rotation_matrix(rotation: "carla.Rotation") -> Matrix:
    """*rotation*'s matrix, as Unreal's ``FRotationMatrix``: its columns are
    the forward, right and up axes, in CARLA's coordinates."""
    p, y, r = (
        math.radians(rotation.pitch),
        math.radians(rotation.yaw),
        math.radians(rotation.roll),
    )
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    forward = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return (
        (forward[0], right[0], up[0]),
        (forward[1], right[1], up[1]),
        (forward[2], right[2], up[2]),
    )


def rotator(m: Matrix) -> "carla.Rotation":
    """The rotation whose :func:`rotation_matrix` is *m*, as Unreal's
    ``FMatrix::Rotator``."""
    fx, fy, fz = m[0][0], m[1][0], m[2][0]
    pitch = math.atan2(fz, math.hypot(fx, fy))
    yaw = math.atan2(fy, fx)
    zero_roll = rotation_matrix(
        carla.Rotation(pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=0.0)
    )
    sy_axis = (zero_roll[0][1], zero_roll[1][1], zero_roll[2][1])
    y_axis = (m[0][1], m[1][1], m[2][1])
    z_axis = (m[0][2], m[1][2], m[2][2])
    roll = math.atan2(_dot(z_axis, sy_axis), _dot(y_axis, sy_axis))
    return carla.Rotation(
        pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=math.degrees(roll)
    )


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalized(v: Vector) -> Optional[Vector]:
    norm = math.sqrt(_dot(v, v))
    if norm < 1e-9:
        return None
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _mul(a: Matrix, b: Matrix) -> Matrix:
    rows = [
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    ]
    return (
        (rows[0][0], rows[0][1], rows[0][2]),
        (rows[1][0], rows[1][1], rows[1][2]),
        (rows[2][0], rows[2][1], rows[2][2]),
    )


def _transposed(m: Matrix) -> Matrix:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def _apply(m: Matrix, v: Vector) -> Vector:
    return (_dot(m[0], v), _dot(m[1], v), _dot(m[2], v))


def swing(a: Vector, b: Vector) -> Matrix:
    """The shortest rotation taking unit vector *a* onto unit vector *b*."""
    c = max(-1.0, min(1.0, _dot(a, b)))
    axis = _normalized(_cross(a, b))
    if axis is None:
        if c > 0.0:
            return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        # Opposite: half a turn about any axis square to a.
        other: Vector = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = _normalized(_cross(a, other))
        assert axis is not None  # noqa: S101 -- other is not along a
    s = math.sqrt(max(0.0, 1.0 - c * c))
    x, y, z = axis
    t = 1.0 - c
    return (
        (c + x * x * t, x * y * t - z * s, x * z * t + y * s),
        (y * x * t + z * s, c + y * y * t, y * z * t - x * s),
        (z * x * t - y * s, z * y * t + x * s, c + z * z * t),
    )


@dataclass(frozen=True)
class _Rest:
    """A driven bone as the walker's animation held it when the motion began."""

    bone: CarlaBone
    world: Matrix
    parent_world: Matrix
    direction: Vector
    location: "carla.Location"


class WalkerRetarget:
    """Poses a walker's skeleton after a :class:`SomaMotion`'s frames.

    Built from the walker's bones as its animation holds them
    (``Walker.get_bones()``) and where it faces (``Walker.get_transform()``),
    both read once, as the motion begins: a frame is placed relative to them,
    so the walker may move on while it plays.

    Args:
        bones: ``get_bones().bone_transforms``.
        facing: The walker's rotation.
        upper_body_only: Drive only the bones above the hips; the legs keep
            the animation's pose.
    """

    _rests: list[_Rest]
    _body: Matrix
    missing: tuple[str, ...]

    def __init__(
        self,
        bones: Sequence["carla.bone_transform_out"],
        facing: "carla.Rotation",
        upper_body_only: bool = True,
    ) -> None:
        by_name = {str(b.name): b for b in bones}
        self._body = rotation_matrix(facing)
        self._rests = []
        missing: list[str] = []
        for bone in CARLA_BONES:
            if upper_body_only and not bone.upper_body:
                continue
            child = next((c for c in bone.children if c in by_name), None)
            if bone.name not in by_name or bone.parent not in by_name or child is None:
                missing.append(bone.name)
                continue
            here = by_name[bone.name].world
            there = by_name[child].world.location
            direction = _normalized(
                (
                    there.x - here.location.x,
                    there.y - here.location.y,
                    there.z - here.location.z,
                )
            )
            if direction is None:
                missing.append(bone.name)
                continue
            relative = by_name[bone.name].relative.location
            self._rests.append(
                _Rest(
                    bone=bone,
                    world=rotation_matrix(here.rotation),
                    parent_world=rotation_matrix(by_name[bone.parent].world.rotation),
                    direction=direction,
                    location=carla.Location(x=relative.x, y=relative.y, z=relative.z),
                )
            )
        self.missing = tuple(missing)

    @property
    def driven(self) -> tuple[str, ...]:
        """The bones a frame poses."""
        return tuple(rest.bone.name for rest in self._rests)

    def pose(self, frame: dict[str, Vector]) -> list[tuple[str, "carla.Transform"]]:
        """The driven bones' transforms, relative to their parents, for
        *frame*: each followed SOMA joint's bone direction in the body frame
        (forward, right, up) of the way the walker faces
        (:meth:`~autoware_carla_scenario.pedestrian.soma.SomaMotion.frame`)."""
        posed: dict[str, Matrix] = {}
        result: list[tuple[str, carla.Transform]] = []
        for rest in self._rests:
            parent = posed.get(rest.bone.parent, rest.parent_world)
            # The parent's turn from its rest carries the bone with it ...
            carried = _mul(parent, _transposed(rest.parent_world))
            world = _mul(carried, rest.world)
            # ... and the bone swings from there onto the motion's direction.
            target = frame.get(rest.bone.joint)
            if target is not None:
                world = _mul(
                    swing(_apply(carried, rest.direction), _apply(self._body, target)),
                    world,
                )
            posed[rest.bone.name] = world
            relative = rotator(_mul(_transposed(parent), world))
            result.append((rest.bone.name, carla.Transform(rest.location, relative)))
        return result


#: The joint each followed SOMA joint's bone runs to, the first a skeleton has
#: wins: a generator places joints, not their frames, so a bone's direction is
#: the one from its joint to this child.
BONE_CHILDREN: dict[str, list[str]] = {
    "Spine1": ["Spine2", "Chest"],
    "Chest": ["Neck1"],
    "Neck1": ["Neck2", "Head"],
    "LeftShoulder": ["LeftArm"],
    "LeftArm": ["LeftForeArm"],
    "LeftForeArm": ["LeftHand"],
    "LeftHand": ["LeftHandMiddle1", "LeftHandMiddleEnd"],
    "RightShoulder": ["RightArm"],
    "RightArm": ["RightForeArm"],
    "RightForeArm": ["RightHand"],
    "RightHand": ["RightHandMiddle1", "RightHandMiddleEnd"],
    "LeftLeg": ["LeftShin"],
    "LeftShin": ["LeftFoot"],
    "LeftFoot": ["LeftToeBase"],
    "RightLeg": ["RightShin"],
    "RightShin": ["RightFoot"],
    "RightFoot": ["RightToeBase"],
}


def directions_from_joints(
    joints: dict[str, Vector], heading: float
) -> dict[str, Vector]:
    """A frame of bone directions (:meth:`WalkerRetarget.pose`) from where a
    generator places the joints.

    Args:
        joints: Each SOMA joint's position, by name, in a right-handed, z-up
            frame (the pedestrian motion contract's).
        heading: The way the body faces, in radians counter-clockwise from
            ``+x`` of that frame.

    Returns:
        Each followed joint's bone direction in the body frame (forward,
        right, up) of *heading*; a joint whose bone the frame lacks, or whose
        bone has no length, is left out (the walker's animation keeps it).
    """
    cos, sin = math.cos(heading), math.sin(heading)
    frame: dict[str, Vector] = {}
    for joint, children in BONE_CHILDREN.items():
        here = joints.get(joint)
        if here is None:
            continue
        there: Optional[Vector] = None
        for child in children:
            if child in joints:
                there = joints[child]
                break
        if there is None:
            continue
        bone = _normalized((there[0] - here[0], there[1] - here[1], there[2] - here[2]))
        if bone is None:
            continue
        frame[joint] = (
            bone[0] * cos + bone[1] * sin,
            bone[0] * sin - bone[1] * cos,
            bone[2],
        )
    return frame
