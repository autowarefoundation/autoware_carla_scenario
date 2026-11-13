"""Trajectory logs as WOSAC rollouts, merged and packed as a submission
(``wosac``, docs/wosac.md) -- all from logs written here, without CARLA."""

from __future__ import annotations

import json
import logging
import math
import tarfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from autoware_carla_scenario.wosac import (
    TARBALL_NAME,
    build_submission,
    merge_rollouts,
    object_id_of,
    object_type_of,
    read_rollouts,
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


def test_object_ids_can_be_given(tmp_path: Path) -> None:
    log = _write_log(tmp_path / "Run.trajectory.jsonl")
    rollout = trajectory_log_to_rollout(log, object_ids={"ego": 7, "npc1": 3})
    assert sorted(_trajectories(rollout.rollouts)) == [3, 7]

    with pytest.raises(ValueError, match="have the same object id, 3"):
        trajectory_log_to_rollout(log, object_ids={"ego": 3, "npc1": 3})
    with pytest.raises(ValueError, match="from 0 to 2147483647"):
        trajectory_log_to_rollout(log, object_ids={"ego": -1})


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
        ({"steps": True}, "steps must be a whole number >= 1"),
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


def test_the_submission_is_written_in_shards(tmp_path: Path) -> None:
    scenarios = merge_rollouts([_rollout(tmp_path, f"r{n}", f"S{n}") for n in range(5)])
    submission = build_submission(scenarios, {"unique_method_name": "m"})
    out = tmp_path / "submission"
    paths = write_submission(submission, out, shards=2, tar=True)

    assert [p.name for p in paths] == [shard_name(0, 2), shard_name(1, 2), TARBALL_NAME]
    assert shard_name(0, 2) == "submission.binproto-00000-of-00002"
    shards = [
        pb.SimAgentsChallengeSubmission.FromString(p.read_bytes()) for p in paths[:2]
    ]
    assert [len(s.scenario_rollouts) for s in shards] == [2, 3]
    assert all(s.unique_method_name == "m" for s in shards)
    assert read_submission(paths[:2]) == submission
    with tarfile.open(paths[2]) as archive:
        assert archive.getnames() == [shard_name(0, 2), shard_name(1, 2)]

    with pytest.raises(ValueError, match="holds a submission already"):
        write_submission(submission, out)
    with pytest.raises(ValueError, match="from 1 to the 5 scenarios, not 6"):
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
