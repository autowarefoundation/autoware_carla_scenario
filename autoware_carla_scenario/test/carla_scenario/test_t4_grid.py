"""t4_scene.grid: the step, stem, ego, on-scene and turn-indicator logic that
t4_scene.convert, t4_scene.live and ``scenario-t4scene`` share, compared with
the code each replaced (copied here verbatim) on many inputs."""

from __future__ import annotations

import bisect
import itertools
import math
from pathlib import Path
from typing import Optional

import pytest

from autoware_carla_scenario.constants import EGO_ROLES, FIXED_DELTA_SECONDS
from autoware_carla_scenario.t4_scene.convert import (
    LOG_SUFFIX,
    _Log,
    trajectory_log_to_t4_scene,
)
from autoware_carla_scenario.t4_scene.grid import (
    indicator_at,
    is_ego,
    on_scene,
    scene_stem,
    step_frames,
)
from autoware_carla_scenario.t4_scene.live import LiveT4Scene

# --- The code grid replaced, as it was (convert.py, live.py, __main__.py) ---


def _original_step_frames(step_s: float) -> int:
    step_frames = round(step_s / FIXED_DELTA_SECONDS)
    if step_frames < 1 or not math.isclose(
        step_frames * FIXED_DELTA_SECONDS, step_s, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"step_s={step_s} is not a multiple of the log's tick ({FIXED_DELTA_SECONDS} s)"
        )
    return step_frames


def _original_convert_stem(path: Path) -> str:
    return (
        path.name[: -len(LOG_SUFFIX)] if path.name.endswith(LOG_SUFFIX) else path.stem
    )


def _original_main_stem(log: Path) -> str:
    return log.name[: -len(LOG_SUFFIX)] if log.name.endswith(LOG_SUFFIX) else log.stem


def _original_is_ego(role: str) -> bool:
    return role.lower() in EGO_ROLES


def _original_on_scene(
    spawned: Optional[int], despawned: Optional[int], frame: int
) -> bool:
    return (spawned is None or frame > spawned) and (
        despawned is None or frame < despawned
    )


def _original_indicator(
    changes: tuple[list[int], list[str]], carla_frame: int
) -> Optional[str]:
    last = bisect.bisect_right(changes[0], carla_frame) - 1
    return None if last < 0 else changes[1][last]


# --- The comparisons ---

_STEPS = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    10.0,
    0.1 + 0.2,
    0.05 * 3,
    0.07,
    0.03,
    0.024,
    0.026,
    0.075,
    0.1000000001,
    0.1 + 1e-8,
    0.0,
    -0.05,
    -0.1,
    1e-12,
    123.45,
    123.4,
    *(k * 0.01 for k in range(-3, 60)),
    *(k * FIXED_DELTA_SECONDS for k in range(0, 400, 7)),
]


def _outcome(function, *args):  # type: ignore[no-untyped-def]
    try:
        return ("value", function(*args))
    except Exception as error:  # noqa: BLE001 - the type is what is compared
        return ("raises", type(error))


@pytest.mark.parametrize("step_s", _STEPS)
def test_step_frames_matches_the_original(step_s: float) -> None:
    assert _outcome(step_frames, step_s) == _outcome(_original_step_frames, step_s)


@pytest.mark.parametrize("step_s", [math.nan, math.inf, -math.inf])
def test_step_frames_raises_as_the_original_on_non_finite(step_s: float) -> None:
    assert _outcome(step_frames, step_s) == _outcome(_original_step_frames, step_s)


def test_step_frames_returns_an_int() -> None:
    assert type(step_frames(0.1)) is int
    assert step_frames(0.1) == 2


def test_step_frames_message_is_the_unified_one() -> None:
    with pytest.raises(
        ValueError,
        match=r"^step_s=0\.07 is not a positive multiple of the 0\.05 s tick$",
    ):
        step_frames(0.07)


_NAMES = [
    "Run.trajectory.jsonl",
    "a.b.trajectory.jsonl",
    ".trajectory.jsonl",
    "trajectory.jsonl",
    "Run.jsonl",
    "Run",
    "Run.",
    ".hidden",
    "..",
    "a.b.c",
    "Run.trajectory.jsonl.bak",
    "Run.TRAJECTORY.JSONL",
    "x.trajectory.jsonl.trajectory.jsonl",
    "名前.trajectory.jsonl",
    "dots...",
    "a..b",
]


@pytest.mark.parametrize("name", _NAMES)
def test_scene_stem_matches_the_originals(name: str) -> None:
    for path in (Path(name), Path("some/dir") / name):
        assert scene_stem(path.name) == _original_convert_stem(path)
        assert scene_stem(path.name) == _original_main_stem(path)


@pytest.mark.parametrize(
    "role",
    ["ego", "hero", "EGO", "Hero", "hErO", "", "egos", "her", "npc", " ego", "ego "],
)
def test_is_ego_matches_the_original(role: str) -> None:
    assert is_ego(role) == _original_is_ego(role)


def test_on_scene_matches_the_original() -> None:
    marks: list[Optional[int]] = [None, -5, 0, 3, 7, 10]
    for spawned, despawned in itertools.product(marks, marks):
        for frame in range(-7, 13):
            assert on_scene(spawned, despawned, frame) == _original_on_scene(
                spawned, despawned, frame
            ), (spawned, despawned, frame)


def test_log_on_scene_matches_the_original() -> None:
    log = _Log(step_frames=1)
    log.spawned = {1: 3, 2: 5}
    log.despawned = {1: 8, 3: 4}
    for actor_id in range(5):
        for frame in range(-2, 12):
            assert log.on_scene(actor_id, frame) == _original_on_scene(
                log.spawned.get(actor_id), log.despawned.get(actor_id), frame
            )


_CHANGES: list[tuple[list[int], list[str]]] = [
    ([], []),
    ([5], ["left"]),
    ([0, 0], ["left", "right"]),
    ([2, 4, 4, 9], ["off", "left", "right", "hazard"]),
    ([1, 2, 3, 4, 5, 6, 7], ["a", "b", "c", "d", "e", "f", "g"]),
    ([-3, 10, 10, 10, 20], ["off", "left", "x", "hazard", "right"]),
]


@pytest.mark.parametrize("changes", _CHANGES)
def test_indicator_at_matches_the_original(
    changes: tuple[list[int], list[str]],
) -> None:
    for frame in range(-6, 25):
        assert indicator_at(changes[0], changes[1], frame) == _original_indicator(
            changes, frame
        ), (changes, frame)


# --- The callers' unified message ---


def test_live_and_convert_refuse_a_step_alike(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive multiple of the 0.05 s tick"):
        LiveT4Scene(1, step_s=0.07)
    with pytest.raises(ValueError, match="positive multiple of the 0.05 s tick"):
        trajectory_log_to_t4_scene(tmp_path / "Run.trajectory.jsonl", step_s=0.0)


def test_live_checks_history_rows_before_the_step() -> None:
    with pytest.raises(ValueError, match="history_rows"):
        LiveT4Scene(0, step_s=0.07)
