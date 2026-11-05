"""t4_scene.frames: the frame math t4_scene.convert moved out (#1665),
compared with the code it replaced (copied here verbatim) on random inputs and
edge angles: the results must be bit for bit the same."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from autoware_carla_scenario.t4_scene import convert, frames
from autoware_carla_scenario.t4_scene.frames import (
    FLIP_Y,
    map_angles,
    map_positions,
    quaternions,
    rotations,
)

# --- the code t4_scene.convert held before #1665, verbatim ---------------

_OLD_FLIP_Y = np.array([1.0, -1.0, 1.0])


def _old_positions(
    offset: npt.NDArray[np.float64], xyz: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return xyz * _OLD_FLIP_Y + offset


def _old_angles(rpy_deg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.radians(rpy_deg) * np.array([1.0, -1.0, -1.0])


def _old_rotations(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    cr, cp, cw = np.cos(angles).T
    sr, sp, sw = np.sin(angles).T
    return np.stack(
        [
            np.stack(
                [cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr], axis=-1
            ),
            np.stack(
                [sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr], axis=-1
            ),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=-2,
    )


def _old_quaternions(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    cr, cp, cw = np.cos(angles / 2).T
    sr, sp, sw = np.sin(angles / 2).T
    return np.stack(
        [
            sr * cp * cw - cr * sp * sw,
            cr * sp * cw + sr * cp * sw,
            cr * cp * sw - sr * sp * cw,
            cr * cp * cw + sr * sp * sw,
        ],
        axis=-1,
    )


# --- inputs ---------------------------------------------------------------

_EDGES = (0.0, -0.0, math.pi, -math.pi, math.pi / 2, -math.pi / 2, 2 * math.pi)


def _edge_angles() -> npt.NDArray[np.float64]:
    """Every roll/pitch/yaw combination of the edge angles (radians): 0, ±π,
    the gimbal-lock pitch ±π/2 and a full turn."""
    grid = np.meshgrid(_EDGES, _EDGES, _EDGES, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=-1)


def _random_angles(seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2 * math.pi, 2 * math.pi, size=(200, 3))


_ANGLE_CASES = {
    "random-0": _random_angles(0),
    "random-1": _random_angles(1),
    "edges": _edge_angles(),
    "one-row": np.array([[0.1, -0.2, 0.3]]),
    "no-rows": np.zeros((0, 3)),
}


@pytest.mark.parametrize("angles", list(_ANGLE_CASES.values()), ids=list(_ANGLE_CASES))
def test_rotations_are_the_old_formula(angles: npt.NDArray[np.float64]) -> None:
    new = rotations(angles)
    assert new.shape == (len(angles), 3, 3)
    assert np.array_equal(new, _old_rotations(angles))


@pytest.mark.parametrize("angles", list(_ANGLE_CASES.values()), ids=list(_ANGLE_CASES))
def test_quaternions_are_the_old_formula(angles: npt.NDArray[np.float64]) -> None:
    new = quaternions(angles)
    assert new.shape == (len(angles), 4)
    assert np.array_equal(new, _old_quaternions(angles))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_angles_are_the_old_formula(seed: int) -> None:
    rpy_deg = np.random.default_rng(seed).uniform(-360.0, 360.0, size=(200, 3))
    rpy_deg = np.concatenate(
        [rpy_deg, np.degrees(_edge_angles()), [[90.0, -90.0, 180.0]]]
    )
    assert np.array_equal(map_angles(rpy_deg), _old_angles(rpy_deg))
    # Through the angles, as convert uses them.
    angles = map_angles(rpy_deg)
    assert np.array_equal(rotations(angles), _old_rotations(_old_angles(rpy_deg)))
    assert np.array_equal(quaternions(angles), _old_quaternions(_old_angles(rpy_deg)))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_positions_are_the_old_formula(seed: int) -> None:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1e5, 1e5, size=(200, 3))
    offset = rng.uniform(-1e6, 1e6, size=3)
    assert np.array_equal(map_positions(xyz, offset), _old_positions(offset, xyz))
    # One (3,) location, as convert passes an actor's state.
    assert np.array_equal(map_positions(xyz[0], offset), _old_positions(offset, xyz[0]))


def test_the_flip_is_the_old_one() -> None:
    assert np.array_equal(FLIP_Y, _OLD_FLIP_Y)


def test_convert_re_exports_and_calls_the_frames() -> None:
    assert convert._FLIP_Y is frames.FLIP_Y
    assert convert._rotations is frames.rotations
    assert convert._quaternions is frames.quaternions

    rng = np.random.default_rng(3)
    xyz = rng.uniform(-100.0, 100.0, size=(20, 3))
    rpy_deg = rng.uniform(-180.0, 180.0, size=(20, 3))
    frame = convert._MapFrame.of({"mgrs_offset": [81655.0, 50045.0], "z_offset": 4.5})
    assert np.array_equal(
        frame.positions(xyz), _old_positions(np.array([81655.0, 50045.0, 4.5]), xyz)
    )
    assert np.array_equal(convert._MapFrame.angles(rpy_deg), _old_angles(rpy_deg))


def test_convert_re_exports_the_checked_rear_axle_offset() -> None:
    """#1697: the check moved to frames; convert (and live, __main__) keep it."""
    assert convert.checked_rear_axle_offset is frames.checked_rear_axle_offset
