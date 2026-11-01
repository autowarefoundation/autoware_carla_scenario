"""A SOMA motion (NVlabs SOMA-X) replayed on a CARLA walker's skeleton.

`SOMA-X <https://github.com/NVlabs/SOMA-X>`_ drives one skeleton -- the SOMA
rig -- whatever identity model (MHR, SOMA, Anny, SMPL) gives the body its
shape, so a motion is the same file for each.  ``soma.io.save_soma_npz``
writes it as an ``.npz``: ``poses`` (each joint's rotation relative to its
parent, ``(N, J, 3)`` axis-angle or ``(N, J, 3, 3)`` matrices), the
``joint_names`` they are in the order of, and, for poses relative to the
T-pose, the T-pose's ``joint_orient``.  :func:`load_soma_motion` reads one with
numpy alone: SOMA-X itself (torch, Warp, USD) is not a dependency.

A CARLA walker's mesh is the blueprint's, so the motion is retargeted onto its
skeleton (``crl_*`` bones) by direction rather than replayed joint for joint:
each SOMA joint's frame has its local X axis along the bone to its child (``-X``
on the right side), which gives the bone's direction in every frame without
the rig's bone lengths.  :class:`~autoware_carla_scenario.pedestrian.skeleton.WalkerRetarget` swings each CARLA bone of
:data:`CARLA_BONES` from where the walker's own animation holds it onto that
direction, parents first, and hands the result to ``Walker.set_bones()`` as
transforms relative to the parent.  The swing is the shortest rotation, so a
bone's twist about its own axis stays the animation's; the hips, and with
them where the walker stands and faces, stay the walker's.  That half lives in
:mod:`~autoware_carla_scenario.pedestrian.skeleton` (no numpy, so Codon
checks it).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from .skeleton import CARLA_BONES, Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FPS",
    "SOMA_JOINTS",
    "SomaMotion",
    "load_soma_motion",
]

#: Frames per second of a motion whose file does not say (``save_soma_npz``
#: stores none) and that is not given one.
DEFAULT_FPS: float = 30.0

#: The SOMA rig's 77 public joints below its virtual ``Root``, each with its
#: parent, in the order ``save_soma_npz`` writes them (``SOMA_template_rig.usda``,
#: twist joints left out).  A parent always comes before its children.
SOMA_JOINTS: tuple[tuple[str, str], ...] = (
    ("Hips", "Root"),
    ("Spine1", "Hips"),
    ("Spine2", "Spine1"),
    ("Chest", "Spine2"),
    ("Neck1", "Chest"),
    ("Neck2", "Neck1"),
    ("Head", "Neck2"),
    ("HeadEnd", "Head"),
    ("Jaw", "Head"),
    ("LeftEye", "Head"),
    ("RightEye", "Head"),
    ("LeftShoulder", "Chest"),
    ("LeftArm", "LeftShoulder"),
    ("LeftForeArm", "LeftArm"),
    ("LeftHand", "LeftForeArm"),
    ("LeftHandThumb1", "LeftHand"),
    ("LeftHandThumb2", "LeftHandThumb1"),
    ("LeftHandThumb3", "LeftHandThumb2"),
    ("LeftHandThumbEnd", "LeftHandThumb3"),
    ("LeftHandIndex1", "LeftHand"),
    ("LeftHandIndex2", "LeftHandIndex1"),
    ("LeftHandIndex3", "LeftHandIndex2"),
    ("LeftHandIndex4", "LeftHandIndex3"),
    ("LeftHandIndexEnd", "LeftHandIndex4"),
    ("LeftHandMiddle1", "LeftHand"),
    ("LeftHandMiddle2", "LeftHandMiddle1"),
    ("LeftHandMiddle3", "LeftHandMiddle2"),
    ("LeftHandMiddle4", "LeftHandMiddle3"),
    ("LeftHandMiddleEnd", "LeftHandMiddle4"),
    ("LeftHandRing1", "LeftHand"),
    ("LeftHandRing2", "LeftHandRing1"),
    ("LeftHandRing3", "LeftHandRing2"),
    ("LeftHandRing4", "LeftHandRing3"),
    ("LeftHandRingEnd", "LeftHandRing4"),
    ("LeftHandPinky1", "LeftHand"),
    ("LeftHandPinky2", "LeftHandPinky1"),
    ("LeftHandPinky3", "LeftHandPinky2"),
    ("LeftHandPinky4", "LeftHandPinky3"),
    ("LeftHandPinkyEnd", "LeftHandPinky4"),
    ("RightShoulder", "Chest"),
    ("RightArm", "RightShoulder"),
    ("RightForeArm", "RightArm"),
    ("RightHand", "RightForeArm"),
    ("RightHandThumb1", "RightHand"),
    ("RightHandThumb2", "RightHandThumb1"),
    ("RightHandThumb3", "RightHandThumb2"),
    ("RightHandThumbEnd", "RightHandThumb3"),
    ("RightHandIndex1", "RightHand"),
    ("RightHandIndex2", "RightHandIndex1"),
    ("RightHandIndex3", "RightHandIndex2"),
    ("RightHandIndex4", "RightHandIndex3"),
    ("RightHandIndexEnd", "RightHandIndex4"),
    ("RightHandMiddle1", "RightHand"),
    ("RightHandMiddle2", "RightHandMiddle1"),
    ("RightHandMiddle3", "RightHandMiddle2"),
    ("RightHandMiddle4", "RightHandMiddle3"),
    ("RightHandMiddleEnd", "RightHandMiddle4"),
    ("RightHandRing1", "RightHand"),
    ("RightHandRing2", "RightHandRing1"),
    ("RightHandRing3", "RightHandRing2"),
    ("RightHandRing4", "RightHandRing3"),
    ("RightHandRingEnd", "RightHandRing4"),
    ("RightHandPinky1", "RightHand"),
    ("RightHandPinky2", "RightHandPinky1"),
    ("RightHandPinky3", "RightHandPinky2"),
    ("RightHandPinky4", "RightHandPinky3"),
    ("RightHandPinkyEnd", "RightHandPinky4"),
    ("LeftLeg", "Hips"),
    ("LeftShin", "LeftLeg"),
    ("LeftFoot", "LeftShin"),
    ("LeftToeBase", "LeftFoot"),
    ("LeftToeEnd", "LeftToeBase"),
    ("RightLeg", "Hips"),
    ("RightShin", "RightLeg"),
    ("RightFoot", "RightShin"),
    ("RightToeBase", "RightFoot"),
    ("RightToeEnd", "RightToeBase"),
)

_SOMA_PARENT: dict[str, str] = dict(SOMA_JOINTS)
_SOMA_ORDER: dict[str, int] = {name: i for i, (name, _) in enumerate(SOMA_JOINTS)}

#: The SOMA joints :data:`CARLA_BONES` follow, in its order.
_FOLLOWED: tuple[str, ...] = tuple(bone.joint for bone in CARLA_BONES)


@dataclass(frozen=True)
class SomaMotion:
    """A SOMA motion, as the directions of the bones :data:`CARLA_BONES` follow.

    Attributes:
        fps: Frames per second.
        joints: The SOMA joints, in the order of :attr:`directions`' second axis.
        directions: ``(N, len(joints), 3)``: each joint's bone direction, a
            unit vector in the body frame (forward, right, up) of the
            character as it faces in the first frame.
    """

    fps: float
    joints: tuple[str, ...]
    directions: "NDArray[np.float64]"

    @property
    def frame_count(self) -> int:
        """How many frames the motion has."""
        return int(self.directions.shape[0])

    @property
    def duration_s(self) -> float:
        """How long the motion plays, in seconds."""
        return self.frame_count / self.fps

    def frame(self, index: int) -> dict[str, Vector]:
        """Frame *index*'s bone directions, by SOMA joint."""
        row = self.directions[index]
        return {
            joint: (float(row[i, 0]), float(row[i, 1]), float(row[i, 2]))
            for i, joint in enumerate(self.joints)
        }


def load_soma_motion(path: Union[str, Path], fps: Optional[float] = None) -> SomaMotion:
    """Read the ``.npz`` ``soma.io.save_soma_npz`` wrote at *path*.

    Args:
        path: The file.
        fps: Its frames per second.  ``None``: the file's ``fps`` (or
            ``mocap_framerate``, AMASS's name) if it has one, else
            :data:`DEFAULT_FPS`.

    Raises:
        FileNotFoundError: If there is no file at *path*.
        ValueError: If it is not a SOMA body motion: no ``poses``, poses of
            neither shape, a hand motion (``hand_type``), joints the SOMA rig
            does not have, or poses relative to the T-pose without the
            ``joint_orient`` that makes them absolute.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"no SOMA motion at {file}")
    with np.load(file, allow_pickle=False) as data:
        keys = set(data.files)
        if "poses" not in keys:
            raise ValueError(f"{file}: not a SOMA motion (no 'poses')")
        if "hand_type" in keys:
            raise ValueError(f"{file}: a hand motion; a walker plays a body motion")
        poses = np.asarray(data["poses"], dtype=np.float64)
        names = (
            [str(n) for n in data["joint_names"]]
            if "joint_names" in keys
            else _default_joint_names(poses.shape[1] if poses.ndim >= 2 else 0)
        )
        if "absolute_pose" in keys:
            absolute = bool(data["absolute_pose"])
        else:
            absolute = "joint_orient" not in keys
        orient = (
            np.asarray(data["joint_orient"], dtype=np.float64)
            if "joint_orient" in keys
            else None
        )
        rate = fps
        if rate is None:
            for key in ("fps", "mocap_framerate"):
                if key in keys:
                    rate = float(data[key])
                    break
    if rate is None:
        rate = DEFAULT_FPS
    if rate <= 0:
        raise ValueError(f"{file}: fps must be positive, not {rate}")
    rotations = _as_matrices(poses, file)
    if rotations.shape[0] == 0:
        raise ValueError(f"{file}: the motion has no frames")
    names = [n.rsplit("/", 1)[-1] for n in names]
    if len(names) != rotations.shape[1]:
        raise ValueError(
            f"{file}: {len(names)} joint names for {rotations.shape[1]} joints"
        )
    unknown = sorted(set(names) - set(_SOMA_PARENT) - {"Root"})
    if unknown:
        raise ValueError(f"{file}: joints the SOMA rig does not have: {unknown}")
    if not absolute:
        if orient is None:
            raise ValueError(
                f"{file}: poses relative to the T-pose need its 'joint_orient'"
            )
        if orient.shape[0] == len(names) + 1:
            orient = orient[1:]  # save_soma_npz strips Root from the poses only
        if orient.shape != (len(names), 3, 3):
            raise ValueError(
                f"{file}: joint_orient is {orient.shape}, not ({len(names)}, 3, 3)"
            )
    world = _forward_kinematics(rotations, names, orient)
    return SomaMotion(
        fps=float(rate), joints=_FOLLOWED, directions=_body_directions(world)
    )


def _default_joint_names(count: int) -> list[str]:
    """The SOMA rig's joints a file with *count* joints and no names holds."""
    names = [name for name, _ in SOMA_JOINTS]
    if count == len(names) + 1:
        return ["Root", *names]
    return names[:count]


def _as_matrices(poses: "NDArray[np.float64]", file: Path) -> "NDArray[np.float64]":
    """*poses* as ``(N, J, 3, 3)`` rotation matrices."""
    if poses.ndim == 4 and poses.shape[-2:] == (3, 3):
        return poses
    if poses.ndim == 3 and poses.shape[-1] == 3:
        return _rotvec_to_matrix(poses)
    raise ValueError(
        f"{file}: poses are {poses.shape}, neither (N, J, 3) axis-angle "
        "nor (N, J, 3, 3) matrices"
    )


def _rotvec_to_matrix(rotvec: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Rodrigues' formula over every ``(..., 3)`` axis-angle vector."""
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.divide(rotvec, angle, out=np.zeros_like(rotvec), where=angle > 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    sin = np.sin(angle)[..., None]
    cos = np.cos(angle)[..., None]
    eye = np.broadcast_to(np.eye(3), k.shape)
    result: "NDArray[np.float64]" = eye + sin * k + (1.0 - cos) * (k @ k)
    return result


def _forward_kinematics(
    rotations: "NDArray[np.float64]",
    names: Sequence[str],
    orient: "Optional[NDArray[np.float64]]",
) -> dict[str, "NDArray[np.float64]"]:
    """Each followed joint's world orientation, ``(N, 3, 3)``, by name.

    *rotations* are relative to the parent: absolute ones, or, given the
    T-pose's world orientations *orient*, relative to the T-pose, whose world
    orientation is then the T-pose-relative one times *orient*
    (``soma.geometry.rig_utils.apply_joint_orient_local``).
    """
    index = {name: i for i, name in enumerate(names)}
    frames = rotations.shape[0]
    root = rotations[:, index["Root"]] if "Root" in index else None
    identity = np.broadcast_to(np.eye(3), (frames, 3, 3))
    followed = {*_FOLLOWED, "Hips"}
    needed: set[str] = set()
    for joint in followed:
        while joint != "Root":
            needed.add(joint)
            joint = _SOMA_PARENT[joint]
    missing = sorted(needed - set(index))
    if missing:
        raise ValueError(f"the motion lacks joints a walker follows: {missing}")
    relative: dict[str, "NDArray[np.float64]"] = {}
    for joint in sorted(needed, key=_SOMA_ORDER.__getitem__):
        parent = _SOMA_PARENT[joint]
        if parent == "Root":
            above = identity if root is None else root
        else:
            above = relative[parent]
        relative[joint] = above @ rotations[:, index[joint]]
    world: dict[str, "NDArray[np.float64]"] = {}
    for joint in sorted(followed, key=_SOMA_ORDER.__getitem__):
        if orient is None:
            world[joint] = relative[joint]
        else:
            world[joint] = relative[joint] @ orient[index[joint]]
    return world


def _body_directions(
    world: dict[str, "NDArray[np.float64]"],
) -> "NDArray[np.float64]":
    """Each joint's bone direction, in the body frame of the first frame.

    SOMA is Y-up with the character facing ``+Z`` and its left ``+X``; a
    joint's bone runs along its local ``+X`` (``-X`` on the right side), and
    the hips face along their local ``+Y``.  The first frame's facing is
    turned to forward, so the motion plays the way the walker faces.
    """
    hips = world["Hips"][0]
    facing = hips[:, 1]
    yaw = math.atan2(float(facing[0]), float(facing[2]))
    cos, sin = math.cos(yaw), math.sin(yaw)
    # SOMA (x left, y up, z forward) -> unrotated by `yaw` about y -> the body
    # frame (forward, right, up).
    to_body = np.array(
        [
            [sin, 0.0, cos],  # forward: along the facing
            [-cos, 0.0, sin],  # right: away from the left, the facing's +X
            [0.0, 1.0, 0.0],  # up
        ]
    )
    columns = []
    for joint in _FOLLOWED:
        sign = -1.0 if joint.startswith("Right") else 1.0
        bone = sign * world[joint][:, :, 0]
        columns.append(bone @ to_body.T)
    stacked = np.stack(columns, axis=1)
    norms = np.linalg.norm(stacked, axis=-1, keepdims=True)
    result: "NDArray[np.float64]" = stacked / np.maximum(norms, 1e-12)
    return result
