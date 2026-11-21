"""Trajectory logs as WOSAC rollouts, merged and packed as a submission
(``wosac``, docs/wosac.md) -- all from logs written here, without CARLA."""

from __future__ import annotations

import json
import logging
import math
import re
import tarfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from autoware_carla_scenario.wosac import (
    TARBALL_NAME,
    build_submission,
    check_runs,
    merge_rollouts,
    object_id_of,
    object_type_of,
    read_rollouts,
    read_run,
    read_submission,
    scenario_id_of,
    scenario_pb2,
    shard_name,
    sim_agents_submission_pb2 as pb,
    trajectory_log_to_rollout,
    write_rollout,
    write_submission,
)
from autoware_carla_scenario.wosac.__main__ import main

TRACK = scenario_pb2.Track

EGO = {
    "type": "actor",
    "id": 1,
    "type_id": "vehicle.lincoln.mkz",
    "role": "ego",
    "extent": [2.4, 1.0, 0.75],
    "center": [0.1, 0.2, 0.7],
}
CAR = {
    "type": "actor",
    "id": 2,
    "type_id": "vehicle.tesla.model3",
    "role": "npc1",
    "extent": [2.0, 0.9, 0.7],
    "center": [0.0, 0.0, 0.7],
}
BIKE = {
    "type": "actor",
    "id": 3,
    "type_id": "vehicle.bh.crossbike",
    "role": "cyclist",
    "extent": [0.8, 0.3, 0.8],
    "center": [0.0, 0.0, 0.8],
}
UNNAMED = {
    "type": "actor",
    "id": 4,
    "type_id": "walker.pedestrian.0001",
    "role": "",
    "extent": [0.3, 0.3, 0.9],
    "center": [0.0, 0.0, 0.0],
}

#: The log's first frame; a 10 Hz row is every second 20 Hz tick.
FIRST = 100


def _state(actor_id: int, x: float, y: float, yaw: float = 0.0, z: float = 0.5) -> list:
    return [actor_id, x, y, z, 0.0, 0.0, yaw, 0.0, 0.0, 0.0]


def _tick(row: int, *states: list) -> dict:
    frame = FIRST + 2 * row
    return {
        "type": "tick",
        "frame": frame,
        "t": round(row * 0.1, 3),
        "actors": list(states),
    }


def _write_log(
    path: Path,
    rows: int = 91,
    actors: tuple[dict, ...] = (EGO, CAR),
    state=None,
    extra: tuple[dict, ...] = (),
    result: Optional[dict] = None,
) -> Path:
    """A log of *rows* 10 Hz ticks; *state(actor, row)* is an actor's state
    on a row, or ``None`` where it is not in the tick."""
    if state is None:

        def state(actor: dict, row: int) -> Optional[list]:
            return _state(actor["id"], float(row), 3.0 * actor["id"])

    lines: list[dict[str, Any]] = [
        {"type": "world", "schema_version": 2, "map": "Carla/Maps/Test", "episode": 7},
        {
            "type": "map",
            "name": "Test",
            "source": None,
            "lanelet2": None,
            "opendrive": None,
            "projector_info": None,
            "projector_type": None,
            "mgrs_offset": [1000.0, 2000.0],
            "z_offset": 10.0,
        },
        *actors,
        *extra,
    ]
    for row in range(rows):
        states = [s for a in actors if (s := state(a, row)) is not None]
        if states:
            lines.append(_tick(row, *states))
    if result is not None:
        lines.append({"type": "result", **result})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _trajectories(rollouts: pb.ScenarioRollouts) -> dict[int, pb.SimulatedTrajectory]:
    (scene,) = rollouts.joint_scenes
    return {t.object_id: t for t in scene.simulated_trajectories}


# -- one rollout ---------------------------------------------------------------


def test_a_log_is_one_rollout_of_its_scenario(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "CutIn-3.trajectory.jsonl")
    rollout = trajectory_log_to_rollout(log)

    assert rollout.rollouts.scenario_id == "CutIn"
    assert len(rollout.rollouts.joint_scenes) == 1
    trajectories = _trajectories(rollout.rollouts)
    assert set(trajectories) == {object_id_of("Ego"), object_id_of("npc1")}
    for trajectory in trajectories.values():
        assert len(trajectory.center_x) == 80
        # The sim agents challenge reads the centre and heading only.
        assert not trajectory.valid and not trajectory.length
        assert not trajectory.HasField("object_type")
    assert rollout.current_frame == FIRST + 20
    assert rollout.current_t == pytest.approx(1.0)
    assert (rollout.steps, rollout.padded_steps, rollout.frame) == (80, 0, "local")


def test_the_rows_are_the_80_after_wosacs_current_step(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    car = _trajectories(trajectory_log_to_rollout(log).rollouts)[object_id_of("npc1")]

    # Row 11 is the first of the rollout: x = 11 there, 90 on the last.
    np.testing.assert_allclose(car.center_x, np.arange(11, 91), atol=1e-5)
    # CARLA's y is south: the map's is north.
    np.testing.assert_allclose(car.center_y, -6.0)
    np.testing.assert_allclose(car.center_z, 0.5 + 0.7, atol=1e-6)


def test_the_centre_is_the_boxs_in_the_maps_axes(tmp_path: Path) -> None:
    def state(actor: dict, row: int) -> Optional[list]:
        # Facing CARLA's +y (south), which is the map's -y.
        return _state(actor["id"], 5.0, 7.0, yaw=90.0)

    log = _write_log(tmp_path / "Run.trajectory.jsonl", actors=(EGO,), state=state)
    ego = _trajectories(trajectory_log_to_rollout(log).rollouts)[object_id_of("ego")]

    np.testing.assert_allclose(ego.heading, -math.pi / 2, atol=1e-6)
    # The box centre (0.1 ahead, 0.2 right in CARLA's actor frame) turned
    # with the actor: ahead is south, CARLA's right is west.
    np.testing.assert_allclose(ego.center_x, 5.0 - 0.2, atol=1e-5)
    np.testing.assert_allclose(ego.center_y, -7.0 - 0.1, atol=1e-5)


def test_the_frame_is_right_handed_as_waymos(tmp_path: Path) -> None:
    # Waymo's frame is right-handed, heading counterclockwise from x (east)
    # (scenario.proto, ObjectState.heading; sim_agents_submission.proto), and
    # its metrics turn a box with the standard yaw rotation. CARLA's is
    # left-handed: a left turn there decreases yaw. Drive one, forward along
    # its yaw, and the rollout must move along its heading and turn
    # counterclockwise -- a mirrored frame would flip the off-road metric.
    poses = []
    x, y, yaw = 0.0, 0.0, 30.0
    for _ in range(91):
        poses.append((x, y, yaw))
        x += math.cos(math.radians(yaw))
        y += math.sin(math.radians(yaw))
        yaw -= 1.5

    def state(actor: dict, row: int) -> Optional[list]:
        px, py, pyaw = poses[row]
        return _state(actor["id"], px, py, yaw=pyaw)

    log = _write_log(tmp_path / "Turn.trajectory.jsonl", actors=(EGO,), state=state)
    ego = _trajectories(trajectory_log_to_rollout(log).rollouts)[object_id_of("Ego")]
    heading = np.array(ego.heading)
    # The box centre is 0.1 ahead and 0.2 right of the origin: compare the
    # origins' motion, which the offset (turning with the box) shifts alike.
    steps = np.diff(np.c_[ego.center_x, ego.center_y], axis=0)
    motion = np.arctan2(steps[:, 1], steps[:, 0])
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (motion - heading[:-1]))), 0.0, atol=np.radians(0.5)
    )
    np.testing.assert_allclose(
        np.angle(np.exp(1j * np.diff(heading))), np.radians(1.5), atol=1e-4
    )
    turns = steps[:-1, 0] * steps[1:, 1] - steps[:-1, 1] * steps[1:, 0]
    assert (turns > 0).all()


def test_the_map_frame_adds_the_maps_offsets(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    rollout = trajectory_log_to_rollout(log, map_frame=True)
    car = _trajectories(rollout.rollouts)[object_id_of("npc1")]

    assert rollout.frame == "map"
    assert rollout.offset == (1000.0, 2000.0, 10.0)
    assert car.center_x[0] == pytest.approx(1011.0)
    assert car.center_y[0] == pytest.approx(1994.0)
    assert car.center_z[0] == pytest.approx(11.2, abs=1e-4)


def test_the_current_index_and_steps_are_options(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl", rows=40)
    rollout = trajectory_log_to_rollout(log, current_index=0, steps=None)
    car = _trajectories(rollout.rollouts)[object_id_of("npc1")]

    assert rollout.steps == 39
    np.testing.assert_allclose(car.center_x, np.arange(1, 40), atol=1e-5)


def test_a_log_too_short_is_refused_or_padded(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl", rows=50)
    with pytest.raises(ValueError, match="has 39 rows after row 10, not the 80"):
        trajectory_log_to_rollout(log)

    rollout = trajectory_log_to_rollout(log, pad=True, with_boxes=True)
    car = _trajectories(rollout.rollouts)[object_id_of("npc1")]
    assert rollout.padded_steps == 41
    assert len(car.center_x) == 80
    # The last pose is held, and marked invalid.
    np.testing.assert_allclose(car.center_x[39:], 49.0)
    assert list(car.valid) == [True] * 39 + [False] * 41
    assert {a.role: a.valid_steps for a in rollout.agents}["npc1"] == 39


def test_an_agent_gone_holds_its_last_pose(tmp_path: Path) -> None:
    def state(actor: dict, row: int) -> Optional[list]:
        if actor is CAR and row >= 30:
            return None
        return _state(actor["id"], float(row), 0.0)

    log = _write_log(tmp_path / "Run.trajectory.jsonl", state=state)
    car = _trajectories(trajectory_log_to_rollout(log, with_boxes=True).rollouts)[
        object_id_of("npc1")
    ]

    np.testing.assert_allclose(car.center_x[:19], np.arange(11, 30), atol=1e-5)
    np.testing.assert_allclose(car.center_x[19:], 29.0)
    assert list(car.valid) == [True] * 19 + [False] * 61


def test_the_agents_are_the_road_users_on_the_scene_at_the_current_row(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    late = {**CAR, "id": 5, "role": "late"}
    hidden = {**CAR, "id": 6, "role": "hidden"}

    def state(actor: dict, row: int) -> Optional[list]:
        if actor is late and row < 20:
            return None
        if actor is hidden:
            return _state(6, 0.0, 0.0, z=-500.0)
        return _state(actor["id"], float(row), 0.0)

    log = _write_log(
        tmp_path / "Run.trajectory.jsonl",
        actors=(EGO, CAR, BIKE, UNNAMED, late, hidden),
        state=state,
    )
    with caplog.at_level(logging.WARNING):
        rollout = trajectory_log_to_rollout(log, include_ego=False)

    # Not the ego (left out), the walker without a role (warned about), one
    # that comes later or one parked under the map.
    assert [agent.role for agent in rollout.agents] == ["npc1", "cyclist"]
    assert "leaving out actors 4" in caplog.text
    assert [agent.object_type for agent in rollout.agents] == [
        TRACK.TYPE_VEHICLE,
        TRACK.TYPE_CYCLIST,
    ]


def test_an_entity_spawned_later_is_not_an_agent(tmp_path: Path) -> None:
    spawn = {
        "type": "spawn",
        "frame": FIRST + 40,
        "t": 2.0,
        "label": "spawn",
        "entity": "npc1",
        "id": 2,
    }
    log = _write_log(tmp_path / "Run.trajectory.jsonl", extra=(spawn,))
    rollout = trajectory_log_to_rollout(log)
    assert [agent.role for agent in rollout.agents] == ["Ego"]


@pytest.mark.parametrize(
    ("type_id", "expected"),
    [
        ("walker.pedestrian.0030", TRACK.TYPE_PEDESTRIAN),
        ("vehicle.gazelle.omafiets", TRACK.TYPE_CYCLIST),
        ("vehicle.yamaha.yzf", TRACK.TYPE_VEHICLE),
        ("vehicle.carlamotors.firetruck", TRACK.TYPE_VEHICLE),
        ("static.prop.cone", TRACK.TYPE_OTHER),
    ],
)
def test_the_object_type_of_a_blueprint(type_id: str, expected: int) -> None:
    assert object_type_of(type_id) == expected


def test_the_boxes_are_written_when_asked(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    rollout = trajectory_log_to_rollout(log, with_boxes=True)
    ego = _trajectories(rollout.rollouts)[object_id_of("Ego")]

    np.testing.assert_allclose(ego.length, 4.8, atol=1e-6)
    np.testing.assert_allclose(ego.width, 2.0)
    np.testing.assert_allclose(ego.height, 1.5)
    assert ego.object_type == TRACK.TYPE_VEHICLE
    assert all(ego.valid)


def test_an_object_id_is_its_roles_whatever_the_actor_id(tmp_path: Path) -> None:
    first = _write_log(tmp_path / "A.trajectory.jsonl")
    renumbered = ({**EGO, "id": 41}, {**CAR, "id": 42})
    second = _write_log(tmp_path / "B.trajectory.jsonl", actors=renumbered)

    ids = [
        sorted(_trajectories(trajectory_log_to_rollout(log).rollouts))
        for log in (first, second)
    ]
    assert ids[0] == ids[1]
    # ``ego`` and ``hero`` are the framework's ``Ego``.
    assert object_id_of("hero") == object_id_of("Ego") == object_id_of("ego")
    assert 0 <= object_id_of("npc1") < 2**31


@pytest.mark.parametrize(
    ("role", "object_id"),
    [
        # Computed with int.from_bytes(digest[:4], "big") before wosac.ids
        # read them off the hex digest (#2158): a rollout's ids never change.
        ("ego", 1071303995),
        ("Ego", 1071303995),
        ("hero", 1071303995),
        ("npc_1", 705440643),
        ("pedestrian", 1963785820),
        ("CutInVehicle", 65601598),
        ("", 1672528962),
    ],
)
def test_object_ids_are_pinned(role: str, object_id: int) -> None:
    assert object_id_of(role) == object_id


def test_the_id_helpers_are_importable_where_they_were() -> None:
    """#2158 moved them into wosac.ids, which convert.py and pack.py import."""
    from autoware_carla_scenario import wosac
    from autoware_carla_scenario.wosac import convert, ids, pack

    for name in ("scenario_id_of", "object_id_of", "checked_count"):
        assert getattr(convert, name) is getattr(ids, name), name
        assert name in convert.__all__ and name in ids.__all__, name
    assert convert._checked_object_ids is ids._checked_object_ids
    assert pack.checked_count is ids.checked_count
    assert wosac.object_id_of is ids.object_id_of
    assert wosac.scenario_id_of is ids.scenario_id_of


def test_object_ids_can_be_given(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    rollout = trajectory_log_to_rollout(log, object_ids={"ego": 7, "npc1": 3})
    assert sorted(_trajectories(rollout.rollouts)) == [3, 7]

    with pytest.raises(ValueError, match="have the same object id, 3"):
        trajectory_log_to_rollout(log, object_ids={"ego": 3, "npc1": 3})
    with pytest.raises(ValueError, match="from 0 to 2147483647"):
        trajectory_log_to_rollout(log, object_ids={"ego": -1})


def test_two_object_ids_of_one_role_are_refused(tmp_path: Path) -> None:
    # `ego` and `hero` are both the ego: the last one won silently (#2265).
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    with pytest.raises(
        ValueError, match="the object ids 'ego' and 'hero' are both of the role 'Ego'"
    ):
        trajectory_log_to_rollout(log, object_ids={"ego": 5, "hero": 6})


@pytest.mark.parametrize("scenario_id", ["", 5, b"S"])
def test_a_scenario_id_that_is_no_text_is_refused(
    tmp_path: Path, scenario_id: Any
) -> None:
    # An empty id was stored as none, which pack refused later (#2265).
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    with pytest.raises(ValueError, match=f"scenario_id must be .*{scenario_id!r}"):
        trajectory_log_to_rollout(log, scenario_id=scenario_id)


def test_two_actors_of_one_role_are_refused(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run.trajectory.jsonl", actors=(EGO, CAR, {**CAR, "id": 9})
    )
    with pytest.raises(ValueError, match="actors 2 and 9 are both 'npc1'"):
        trajectory_log_to_rollout(log)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"current_index": -1}, "current_index must be a whole number >= 0"),
        ({"steps": 0}, "steps must be a whole number >= 1"),
        ({"steps": True}, "steps must be a whole number, got True"),
        ({"current_index": 200}, "no tick at row 200"),
    ],
)
def test_options_out_of_range_are_refused(
    tmp_path: Path, options: dict, message: str
) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    with pytest.raises(ValueError, match=message):
        trajectory_log_to_rollout(log, **options)


def test_scenario_id_of_drops_the_run_counter() -> None:
    assert scenario_id_of("CutInScenario") == "CutInScenario"
    assert scenario_id_of("CutInScenario-12") == "CutInScenario"
    assert scenario_id_of("Cut-In") == "Cut-In"


def test_write_rollout_writes_the_rollout_and_its_agents(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "Run-1.trajectory.jsonl",
        result={
            "passed": True,
            "message": "ok",
            "elapsed_seconds": 1.0,
            "conditions": [],
        },
    )
    rollouts_path, agents_path = write_rollout(log, tmp_path / "out", scenario_id="S")

    assert rollouts_path == tmp_path / "out" / "Run-1.wosac.pb"
    assert read_rollouts(rollouts_path).scenario_id == "S"
    agents = json.loads(agents_path.read_text())
    assert agents["scenario_id"] == "S"
    assert agents["result"] == {"passed": True, "message": "ok"}
    assert (agents["step_s"], agents["current_index"], agents["steps"]) == (0.1, 10, 80)
    # A log written before the run record: neither is known.
    assert (agents["seed"], agents["scenario_hash"]) == (None, None)
    current = agents["agents"][0].pop("current")
    assert len(current) == 4
    assert agents["agents"][0] == {
        "object_id": object_id_of("Ego"),
        "role": "Ego",
        "actor_id": 1,
        "type_id": "vehicle.lincoln.mkz",
        "object_type": "TYPE_VEHICLE",
        "ego": True,
        "valid_steps": 80,
    }


# -- many rollouts -------------------------------------------------------------


def _rollout(
    tmp_path: Path, name: str, scenario_id: str = "S", **options: Any
) -> tuple[pb.ScenarioRollouts, str]:
    actors = options.pop("actors", (EGO, CAR))
    rows = options.pop("rows", 91)
    log = _write_log(tmp_path / f"{name}.trajectory.jsonl", rows=rows, actors=actors)
    rollouts = trajectory_log_to_rollout(log, scenario_id=scenario_id, **options)
    return rollouts.rollouts, name


def test_rollouts_merge_by_scenario(tmp_path: Path) -> None:
    merged = merge_rollouts(
        [
            _rollout(tmp_path, "b1", "B"),
            _rollout(tmp_path, "a1", "A"),
            _rollout(tmp_path, "b2", "B"),
        ]
    )
    assert [(s.scenario_id, len(s.joint_scenes)) for s in merged] == [
        ("A", 1),
        ("B", 2),
    ]
    for scene in merged[1].joint_scenes:
        ids = [t.object_id for t in scene.simulated_trajectories]
        assert ids == sorted(ids)


def test_concatenated_files_are_one_scenarios_rollouts(tmp_path: Path) -> None:
    first, _ = _rollout(tmp_path, "r1")
    second, _ = _rollout(tmp_path, "r2")
    both = pb.ScenarioRollouts.FromString(
        first.SerializeToString() + second.SerializeToString()
    )
    assert both.scenario_id == "S"
    assert len(both.joint_scenes) == 2


def test_a_scenario_needs_the_rollouts_asked_for(tmp_path: Path) -> None:
    rollouts = [_rollout(tmp_path, f"r{n}") for n in range(3)]
    assert len(merge_rollouts(rollouts, num_rollouts=3)[0].joint_scenes) == 3
    with pytest.raises(
        ValueError, match="scenario 'S' has 3 rollouts, not 32: r0, r1, r2"
    ):
        merge_rollouts(rollouts, num_rollouts=32)
    with pytest.raises(ValueError, match="num_rollouts must be"):
        merge_rollouts(rollouts, num_rollouts=0)


def test_rollouts_of_different_agents_are_refused_or_cut(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rollouts = [
        _rollout(tmp_path, "r1"),
        _rollout(tmp_path, "r2", actors=(EGO, CAR, BIKE)),
    ]
    with pytest.raises(ValueError, match="r1 and r2 do not simulate the same agents"):
        merge_rollouts(rollouts)

    with caplog.at_level(logging.WARNING):
        (merged,) = merge_rollouts(rollouts, common_agents=True)
    assert f"leaving out object ids [{object_id_of('cyclist')}]" in caplog.text
    for scene in merged.joint_scenes:
        assert {t.object_id for t in scene.simulated_trajectories} == {
            object_id_of("Ego"),
            object_id_of("npc1"),
        }


def test_rollouts_of_different_lengths_are_refused(tmp_path: Path) -> None:
    rollouts = [_rollout(tmp_path, "r1"), _rollout(tmp_path, "r2", steps=40)]
    with pytest.raises(ValueError, match="40 rows long, where an earlier one is 80"):
        merge_rollouts(rollouts)


def test_a_trajectory_with_fields_of_different_lengths_is_refused() -> None:
    scene = pb.JointScene()
    scene.simulated_trajectories.add(
        object_id=1,
        center_x=[0.0, 1.0],
        center_y=[0.0, 1.0],
        center_z=[0.0],
        heading=[0, 0],
    )
    rollouts = pb.ScenarioRollouts(scenario_id="S", joint_scenes=[scene])
    with pytest.raises(
        ValueError, match="object 1 has 1 center_z values and 2 center_x"
    ):
        merge_rollouts([(rollouts, "f")])


def test_a_rollout_without_a_scenario_id_is_refused() -> None:
    with pytest.raises(ValueError, match="f has no scenario_id"):
        merge_rollouts([(pb.ScenarioRollouts(joint_scenes=[pb.JointScene()]), "f")])


def test_a_rollout_without_a_joint_scene_is_refused() -> None:
    """A ``TypeError`` from merging no scene before #2156."""
    with pytest.raises(ValueError, match="f has no joint scene"):
        merge_rollouts([(pb.ScenarioRollouts(scenario_id="a"), "f")])


def test_a_file_that_is_no_rollout_is_refused_naming_it(tmp_path: Path) -> None:
    """A ``DecodeError`` traceback before #2156."""
    bad = tmp_path / "Bad.wosac.pb"
    bad.write_bytes(b"\xff\xff\xff")
    with pytest.raises(ValueError, match=f"{bad}: not a ScenarioRollouts"):
        read_rollouts(bad)
    with pytest.raises(ValueError, match=f"{bad}: not a SimAgentsChallengeSubmission"):
        read_submission([bad])


# -- the submission ------------------------------------------------------------


def test_the_submission_carries_its_metadata(tmp_path: Path) -> None:
    scenarios = merge_rollouts([_rollout(tmp_path, "r1")])
    submission = build_submission(
        scenarios,
        {
            "account_name": "someone@example.com",
            "unique_method_name": "carla-sim",
            "authors": ["A", "B"],
            "uses_lidar_data": False,
            "num_model_parameters": "0",
            "acknowledge_complies_with_closed_loop_requirement": True,
        },
    )
    Submission = pb.SimAgentsChallengeSubmission
    assert submission.submission_type == Submission.SIM_AGENTS_SUBMISSION
    assert submission.unique_method_name == "carla-sim"
    assert list(submission.authors) == ["A", "B"]
    assert submission.HasField("uses_lidar_data")
    assert len(submission.scenario_rollouts) == 1

    with pytest.raises(ValueError, match="the submission's metadata"):
        build_submission(scenarios, {"no_such_field": 1})
    with pytest.raises(ValueError, match="cannot set scenario_rollouts"):
        build_submission(scenarios, {"scenario_rollouts": []})


def test_a_rollout_is_written_after_its_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file goes in one rename, the ``.wosac.json`` first, and an earlier
    export's ``.wosac.pb`` is removed before: a killed export never leaves a
    rollout without its sidecar (``pack`` would take it for a merged file) or
    beside an earlier export's (#2237)."""
    from autoware_carla_scenario import atomic_file

    log = _write_log(tmp_path / "Run-1.trajectory.jsonl")
    out = tmp_path / "out"
    rollouts_path, agents_path = write_rollout(log, out, scenario_id="S")
    written: list[str] = []
    real = atomic_file.os.replace

    def replace(src: str, dst: str) -> None:
        written.append(Path(dst).name)
        if dst.endswith(".pb"):
            raise OSError("killed before the rename")
        real(src, dst)

    monkeypatch.setattr(atomic_file.os, "replace", replace)
    with pytest.raises(OSError, match="killed"):
        write_rollout(log, out, scenario_id="S")
    assert written == [agents_path.name, rollouts_path.name]
    assert not rollouts_path.exists()
    assert sorted(p.name for p in out.iterdir()) == [agents_path.name]


def test_a_shard_killed_mid_write_leaves_no_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard goes in one rename: a killed pack leaves no truncated shard,
    which would parse as a submission short of scenarios (#2237)."""
    from autoware_carla_scenario import atomic_file

    scenarios = merge_rollouts([_rollout(tmp_path, f"r{n}", f"S{n}") for n in range(3)])
    submission = build_submission(scenarios, {"unique_method_name": "m"})

    def killed(src: str, dst: str) -> None:
        raise OSError("killed before the rename")

    monkeypatch.setattr(atomic_file.os, "replace", killed)
    out = tmp_path / "submission"
    with pytest.raises(OSError, match="killed"):
        write_submission(submission, out)
    assert list(out.iterdir()) == []


def test_the_submission_is_written_in_shards(tmp_path: Path) -> None:
    scenarios = merge_rollouts([_rollout(tmp_path, f"r{n}", f"S{n}") for n in range(5)])
    submission = build_submission(
        scenarios, {"unique_method_name": "m", "authors": ["A", "B"]}
    )
    out = tmp_path / "submission"
    paths = write_submission(submission, out, shards=2, tar=True)

    assert [p.name for p in paths] == [shard_name(0, 2), shard_name(1, 2), TARBALL_NAME]
    assert shard_name(0, 2) == "submission.binproto-00000-of-00002"
    shards = [
        pb.SimAgentsChallengeSubmission.FromString(p.read_bytes()) for p in paths[:2]
    ]
    assert [len(s.scenario_rollouts) for s in shards] == [2, 3]
    assert all(s.unique_method_name == "m" for s in shards)
    # Every shard holds the metadata; its repeated fields are read back
    # once, not once per shard (#2182).
    assert all(list(s.authors) == ["A", "B"] for s in shards)
    read = read_submission(paths[:2])
    assert list(read.authors) == ["A", "B"]
    assert read == submission
    with tarfile.open(paths[2]) as archive:
        assert archive.getnames() == [shard_name(0, 2), shard_name(1, 2)]

    with pytest.raises(ValueError, match="holds a submission already"):
        write_submission(submission, out)
    with pytest.raises(
        ValueError, match="shards must be a whole number from 1 to 5, not 6"
    ):
        write_submission(submission, tmp_path / "other", shards=6)


# -- the command line ----------------------------------------------------------


def test_the_command_exports_and_packs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "outputs"
    for n in range(3):
        directory = runs / str(n)
        directory.mkdir(parents=True)
        _write_log(directory / "CutIn.trajectory.jsonl")
    _write_log(runs / "0" / "CutIn-1.trajectory.jsonl")

    assert main(["export", str(runs), "--with-boxes"]) == 0
    assert len(list(runs.rglob("*.wosac.pb"))) == 4
    assert len(list(runs.rglob("*.wosac.json"))) == 4

    metadata = tmp_path / "metadata.yaml"
    metadata.write_text("unique_method_name: carla\nauthors: [someone]\n")
    out = tmp_path / "submission"
    code = main(
        [
            "pack",
            str(runs),
            "-o",
            str(out),
            "--rollouts",
            "4",
            "--metadata",
            str(metadata),
            "--tar",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out.splitlines()
    assert json.loads(printed[-3]) == {"rollouts": {"CutIn": 4}}
    submission = read_submission([out / shard_name(0, 1)])
    assert submission.unique_method_name == "carla"
    (scenario,) = submission.scenario_rollouts
    assert scenario.scenario_id == "CutIn"
    assert len(scenario.joint_scenes) == 4
    assert (out / TARBALL_NAME).is_file()


def test_the_command_reports_a_bad_pack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    assert main(["export", str(log)]) == 0
    capsys.readouterr()
    assert (
        main(["pack", str(tmp_path), "-o", str(tmp_path / "s"), "--rollouts", "2"]) == 1
    )
    assert "scenario 'Run' has 1 rollouts, not 2" in capsys.readouterr().err


def test_the_command_reports_a_file_that_is_no_rollout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "Bad.wosac.pb"
    bad.write_bytes(b"\xff\xff\xff")
    assert main(["pack", str(tmp_path), "-o", str(tmp_path / "s")]) == 1
    assert f"scenario-wosac pack: {bad}: not a ScenarioRollouts" in (
        capsys.readouterr().err
    )


def test_the_command_reports_a_log_it_cannot_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _write_log(tmp_path / "Good.trajectory.jsonl")
    short = _write_log(tmp_path / "Short.trajectory.jsonl", rows=30)
    assert main(["export", str(good), str(short)]) == 1
    captured = capsys.readouterr()
    assert "Good.wosac.pb" in captured.out
    assert f"{short}: " in captured.err and "not the 80 asked for" in captured.err
    assert main(["export", str(short), "--pad"]) == 0


def test_the_command_takes_object_ids_from_a_file(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"Ego": 0, "npc1": 1}))
    assert main(["export", str(log), "--object-ids", str(ids)]) == 0
    rollouts = read_rollouts(tmp_path / "Run.wosac.pb")
    assert sorted(_trajectories(rollouts)) == [0, 1]


def test_the_command_finds_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["pack", str(tmp_path), "-o", str(tmp_path / "s")])
    assert "no rollout (*.wosac.pb) found" in capsys.readouterr().err


# -- rollouts of one scenario --------------------------------------------------

HASH = "a" * 64


def _run_record(seed: int, digest: Optional[str] = HASH) -> dict:
    return {"type": "run", "seed": seed, "scenario_hash": digest}


def _export(
    tmp_path: Path,
    name: str,
    seed: Optional[int],
    digest: Optional[str] = HASH,
    **log: Any,
) -> dict:
    """Write a log of *name* with a run record, export it, and return its
    ``.wosac.json``."""
    extra = () if seed is None else (_run_record(seed, digest),)
    path = _write_log(tmp_path / f"{name}.trajectory.jsonl", extra=extra, **log)
    _, agents = write_rollout(path, scenario_id="S")
    return json.loads(agents.read_text())


def test_the_rollout_says_its_seed_scenario_hash_and_initial_state(
    tmp_path: Path,
) -> None:
    run = _export(tmp_path, "Run", 7)
    assert (run["seed"], run["scenario_hash"]) == (7, HASH)
    ego = next(agent for agent in run["agents"] if agent["ego"])
    rollout = trajectory_log_to_rollout(tmp_path / "Run.trajectory.jsonl")
    (agent,) = [a for a in rollout.agents if a.ego]
    assert ego["current"] == list(agent.current)
    # The current row: x is the row (10), the ego's box centre ahead of it.
    assert ego["current"][0] == pytest.approx(10.0 + 0.1)


def test_read_run_reads_the_json_beside_a_rollout(tmp_path: Path) -> None:
    _export(tmp_path, "Run", 3)
    run = read_run(tmp_path / "Run.wosac.pb")
    assert run is not None and run["seed"] == 3
    assert read_run(tmp_path / "Merged.wosac.pb") is None
    (tmp_path / "Bad.wosac.json").write_text("[1]")
    with pytest.raises(ValueError, match="not a JSON object"):
        read_run(tmp_path / "Bad.wosac.pb")


def test_runs_of_one_scenario_and_their_own_seeds_are_rollouts(tmp_path: Path) -> None:
    runs = [(_export(tmp_path, f"Run-{n}", n), f"Run-{n}") for n in range(3)]
    check_runs(runs)


def test_two_runs_of_one_seed_are_refused(tmp_path: Path) -> None:
    runs = [(_export(tmp_path, f"Run-{n}", 4), f"Run-{n}") for n in range(2)]
    with pytest.raises(ValueError, match="Run-0 and Run-1 both ran seed 4"):
        check_runs(runs)


def test_runs_of_different_scenarios_are_refused(tmp_path: Path) -> None:
    runs = [
        (_export(tmp_path, "A", 0, "a" * 64), "A"),
        (_export(tmp_path, "B", 1, "b" * 64), "B"),
    ]
    with pytest.raises(ValueError, match="A and B ran different scenarios"):
        check_runs(runs)


def test_runs_of_different_scenario_ids_are_checked_apart(tmp_path: Path) -> None:
    a = _export(tmp_path, "A", 0, "a" * 64)
    b = {**_export(tmp_path, "B", 0, "b" * 64), "scenario_id": "T"}
    check_runs([(a, "A"), (b, "B")])


def _moved(dx: float, dyaw: float = 0.0):
    def state(actor: dict, row: int) -> Optional[list]:
        return _state(actor["id"], float(row) + dx, 3.0 * actor["id"], dyaw)

    return state


def test_runs_that_start_their_future_apart_are_refused(tmp_path: Path) -> None:
    first = _export(tmp_path, "Run-0", 0)
    near = _export(tmp_path, "Run-1", 1, state=_moved(0.3))
    far = _export(tmp_path, "Run-2", 2, state=_moved(2.0))
    check_runs([(first, "Run-0"), (near, "Run-1")])
    with pytest.raises(
        ValueError, match=r"2\.000 m .* in Run-2 from where it does in Run-0"
    ):
        check_runs([(first, "Run-0"), (far, "Run-2")])
    check_runs([(first, "Run-0"), (far, "Run-2")], position_tolerance_m=2.5)


def test_runs_that_start_their_future_headed_apart_are_refused(tmp_path: Path) -> None:
    first = _export(tmp_path, "Run-0", 0)
    turned = _export(tmp_path, "Run-1", 1, state=_moved(0.0, 20.0))
    with pytest.raises(ValueError, match="rad away"):
        check_runs([(first, "Run-0"), (turned, "Run-1")])
    check_runs([(first, "Run-0"), (turned, "Run-1")], heading_tolerance_rad=0.5)


def test_headings_either_side_of_pi_are_close() -> None:
    def run(heading: float, seed: int) -> dict:
        agent = {"object_id": 1, "current": [0.0, 0.0, 0.0, heading]}
        return {
            "scenario_id": "S",
            "seed": seed,
            "scenario_hash": HASH,
            "agents": [agent],
        }

    check_runs([(run(math.pi - 0.01, 0), "a"), (run(-math.pi + 0.01, 1), "b")])


def test_runs_in_different_frames_are_refused(tmp_path: Path) -> None:
    # Their rollouts would be one scenario's joint scenes kilometres apart
    # (#2265).
    local = _export(tmp_path, "Run-0", 0)
    path = tmp_path / "Run-1.trajectory.jsonl"
    _write_log(path, extra=(_run_record(1),))
    _, agents = write_rollout(path, scenario_id="S", map_frame=True)
    mapped = json.loads(agents.read_text())
    with pytest.raises(
        ValueError, match=r"Run-0 and Run-1 differ in frame \('local' and 'map'\)"
    ):
        check_runs([(local, "Run-0"), (mapped, "Run-1")])
    check_runs([(mapped, "Run-1"), ({**mapped, "seed": 2}, "Run-2")])


@pytest.mark.parametrize(
    ("field", "value"),
    [("offset", [1.0, 0.0, 0.0]), ("step_s", 0.05), ("current_index", 11)],
)
def test_runs_on_different_grids_are_refused(
    tmp_path: Path, field: str, value: Any
) -> None:
    first = _export(tmp_path, "Run-0", 0)
    other = {**_export(tmp_path, "Run-1", 1), field: value}
    with pytest.raises(ValueError, match=f"Run-0 and Run-1 differ in {field} "):
        check_runs([(first, "Run-0"), (other, "Run-1")])


def test_a_null_offset_is_no_offset(tmp_path: Path) -> None:
    first = _export(tmp_path, "Run-0", 0)
    check_runs(
        [(first, "Run-0"), ({**_export(tmp_path, "Run-1", 1), "offset": None}, "Run-1")]
    )


@pytest.mark.parametrize(
    ("options", "field"),
    [(["--map-frame"], "frame"), (["--current-index", "12", "--pad"], "current_index")],
)
def test_the_command_refuses_to_pack_runs_exported_apart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list, field: str
) -> None:
    for n in range(2):
        _write_log(tmp_path / f"CutIn-{n}.trajectory.jsonl", extra=(_run_record(n),))
    assert main(["export", str(tmp_path / "CutIn-0.trajectory.jsonl")]) == 0
    one = tmp_path / "CutIn-1.trajectory.jsonl"
    assert main(["export", str(one), "--scenario-id", "CutIn", *options]) == 0
    capsys.readouterr()
    assert main(["pack", str(tmp_path), "-o", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err
    assert (
        f"CutIn-0.wosac.pb and {tmp_path / 'CutIn-1.wosac.pb'} differ in {field} "
        in err
    )
    assert not (tmp_path / "s").exists()


def test_a_run_that_does_not_say_its_seed_is_let_through_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs = [(_export(tmp_path, f"Run-{n}", None), f"Run-{n}") for n in range(2)]
    with caplog.at_level(logging.WARNING):
        check_runs(runs)
    assert "cannot tell that Run-0, Run-1 ran the scenario" in caplog.text


@pytest.mark.parametrize("tolerance", [-1.0, math.inf, math.nan])
def test_a_tolerance_out_of_range_is_refused(tolerance: float) -> None:
    with pytest.raises(ValueError, match="position_tolerance_m"):
        check_runs([], position_tolerance_m=tolerance)


def test_the_command_refuses_to_pack_one_rollout_run_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for n in range(2):
        _write_log(tmp_path / f"CutIn-{n}.trajectory.jsonl", extra=(_run_record(9),))
    assert main(["export", str(tmp_path), "--scenario-id", "CutIn"]) == 0
    capsys.readouterr()
    assert main(["pack", str(tmp_path), "-o", str(tmp_path / "s")]) == 1
    assert "both ran seed 9" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_the_command_packs_rollouts_of_their_own_seeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for n in range(2):
        _write_log(
            tmp_path / f"CutIn-{n}.trajectory.jsonl",
            extra=(_run_record(n),),
            state=_moved(0.4 * n),
        )
    assert main(["export", str(tmp_path), "--scenario-id", "CutIn"]) == 0
    capsys.readouterr()
    out = tmp_path / "s"
    assert (
        main(["pack", str(tmp_path), "-o", str(out), "--position-tolerance", "0.1"])
        == 1
    )
    assert "do not share an initial state" in capsys.readouterr().err
    assert main(["pack", str(tmp_path), "-o", str(out)]) == 0


_MALFORMED_SIDECARS = [
    ({"agents": "abc"}, "agents must be a list"),
    ({"agents": [1]}, r"agents\[0\] must be an object"),
    ({"agents": [{"current": [0, 0, 0, 0]}]}, r"agents\[0\]\.object_id"),
    ({"agents": [{"object_id": "x"}]}, r"agents\[0\]\.object_id"),
    ({"agents": [{"object_id": True}]}, r"agents\[0\]\.object_id"),
    ({"agents": [{"object_id": 1, "current": [0, 0]}]}, r"agents\[0\]\.current"),
    ({"agents": [{"object_id": 1, "current": 5}]}, r"agents\[0\]\.current"),
    (
        {"agents": [{"object_id": 1, "current": [0, 0, 0, "north"]}]},
        r"agents\[0\]\.current",
    ),
    ({"offset": [0, 0]}, "offset must be a list of 3 numbers"),
    ({"offset": "abc"}, "offset must be a list of 3 numbers"),
    ({"seed": [1]}, "seed must be a whole number"),
    ({"seed": "one"}, "seed must be a whole number"),
    ({"seed": 1.5}, "seed must be a whole number"),
    ({"agents": [{"object_id": 2.5}]}, r"agents\[0\]\.object_id"),
]


@pytest.mark.parametrize(("change", "message"), _MALFORMED_SIDECARS)
def test_read_run_refuses_a_sidecar_check_runs_cannot_read(
    tmp_path: Path, change: dict, message: str
) -> None:
    run = {**_export(tmp_path, "Run", 3), **change}
    sidecar = tmp_path / "Run.wosac.json"
    sidecar.write_text(json.dumps(run))
    with pytest.raises(ValueError, match=message) as refused:
        read_run(tmp_path / "Run.wosac.pb")
    assert str(refused.value).startswith(f"{sidecar}: ")


def test_read_run_lets_null_seed_offset_and_agents_through(tmp_path: Path) -> None:
    run = {**_export(tmp_path, "Run", 3), "seed": None, "offset": None, "agents": None}
    (tmp_path / "Run.wosac.json").write_text(json.dumps(run))
    assert read_run(tmp_path / "Run.wosac.pb") == run


@pytest.mark.parametrize(("change", "message"), _MALFORMED_SIDECARS)
def test_the_command_reports_a_malformed_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict, message: str
) -> None:
    # A KeyError, AttributeError or TypeError of check_runs() was a
    # traceback; it is the file and what is wrong with it (#2208).
    for n in range(2):
        _write_log(tmp_path / f"CutIn-{n}.trajectory.jsonl", extra=(_run_record(n),))
    assert main(["export", str(tmp_path), "--scenario-id", "CutIn"]) == 0
    capsys.readouterr()
    sidecar = tmp_path / "CutIn-1.wosac.json"
    sidecar.write_text(json.dumps({**json.loads(sidecar.read_text()), **change}))
    assert main(["pack", str(tmp_path), "-o", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"scenario-wosac pack: {sidecar}: ")
    assert re.search(message, err)
    assert "Traceback" not in err
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--position-tolerance", "-1"),
        ("--position-tolerance", "nan"),
        ("--heading-tolerance", "inf"),
        ("--heading-tolerance", "-0.5"),
    ],
)
def test_the_command_names_the_tolerance_flag_it_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    for n in range(2):
        _write_log(tmp_path / f"CutIn-{n}.trajectory.jsonl", extra=(_run_record(n),))
    assert main(["export", str(tmp_path), "--scenario-id", "CutIn"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as exited:
        main(["pack", str(tmp_path), "-o", str(tmp_path / "s"), flag, value])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} must be a finite number >= 0, not {float(value)!r}" in err
    assert "tolerance_m" not in err and "tolerance_rad" not in err
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize(
    ("command", "flag", "value"),
    [
        ("export", "--steps", "0"),
        ("export", "--current-index", "-1"),
        ("pack", "--rollouts", "0"),
        ("pack", "--shards", "0"),
    ],
)
def test_the_command_names_the_count_flag_it_refuses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    flag: str,
    value: str,
) -> None:
    # They were refused by their parameters' names, exit 1, and by export
    # once per log (#2265).
    for n in range(2):
        _write_log(tmp_path / f"CutIn-{n}.trajectory.jsonl", extra=(_run_record(n),))
    assert main(["export", str(tmp_path), "--scenario-id", "CutIn"]) == 0
    capsys.readouterr()
    argv = [command, str(tmp_path), flag, value]
    if command == "pack":
        argv += ["-o", str(tmp_path / "s")]
    with pytest.raises(SystemExit) as exited:
        main(argv)
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.count(f"{flag} must be a whole number >= ") == 1
    assert f"not {int(value)}" in err
    assert not (tmp_path / "s").exists()


def test_the_command_refuses_an_empty_scenario_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    with pytest.raises(SystemExit) as exited:
        main(["export", str(log), "--scenario-id", ""])
    assert exited.value.code == 2
    assert "--scenario-id must not be empty" in capsys.readouterr().err
    assert not (tmp_path / "Run.wosac.pb").exists()
