"""A run's trajectory log as one WOSAC rollout (docs/wosac.md).

The Waymo Open Sim Agents Challenge (WOSAC) takes a simulation as
``sim_agents_submission.proto``'s messages: a ``JointScene`` -- one rollout,
every simulated agent's ``SimulatedTrajectory`` (box centre and heading, 10 Hz)
-- many of which, run from the same initial conditions, make a scenario's
``ScenarioRollouts``.  This module reads a run's trajectory log
(``{stem}.trajectory.jsonl``, docs/trajectory_log.md; no CARLA needed) and
writes it as a ``ScenarioRollouts`` of one ``JointScene``, so the runs of one
scenario -- its seeds -- are its rollouts: :mod:`.pack` merges them and builds
the submission.

The rows are WOSAC's: 10 Hz from the log's first tick, the current state at
``current_index`` (WOSAC's ``current_time_index``, 10: 1 s of history) and the
``steps`` rows after it (80, the 8 s WOSAC scores) in the rollout.  The agents
are the vehicles and walkers on the scene at the current row, as WOSAC
simulates the objects valid at its current step.  Each one's ``object_id`` is
derived from its role name (:func:`object_id_of`), which the scenario gives
it, so an entity has the same id in every run of a scenario, whatever CARLA
actor id it got.  A row where an agent is not in the log (it was despawned, or
the log skipped the tick) holds its last pose, and is marked invalid in
``valid`` when the boxes are written (``with_boxes``).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson

from ..constants import CARLA_BICYCLE_BLUEPRINTS
from ..entity_role import canonical_role_name
from ..run_outputs import WOSAC_AGENTS_SUFFIX, WOSAC_SUFFIX
from ..seed import check_whole_number

# The trajectory log's reader and its frame math are the T4Scene export's.
from ..t4_scene.convert import _Log, _MapFrame, _read_log, box_poses
from ..t4_scene.grid import is_ego, scene_stem, step_frames
from ._proto import scenario_pb2, sim_agents_submission_pb2

__all__ = [
    "CURRENT_INDEX",
    "FUTURE_STEPS",
    "STEP_S",
    "WosacAgent",
    "WosacRollout",
    "checked_count",
    "object_id_of",
    "object_type_of",
    "scenario_id_of",
    "trajectory_log_to_rollout",
    "write_rollout",
]

logger = logging.getLogger(__name__)

#: WOSAC's time step: its scenarios and rollouts are 10 Hz.
STEP_S = 0.1
#: WOSAC's ``current_time_index``: the rows 0-10 are the history, 10 the
#: current state the rollouts start from.
CURRENT_INDEX = 10
#: The rows after the current one a WOSAC rollout holds: 8 s.
FUTURE_STEPS = 80

_TRACK = scenario_pb2.Track
#: The run counter ``scenario_runner`` appends to a stem taken in the
#: directory already (``CutInScenario-1``).
_RUN_COUNTER = re.compile(r"-\d+$")
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class WosacAgent:
    """One simulated agent of a rollout: who its ``object_id`` stands for."""

    #: The ``SimulatedTrajectory.object_id``.
    object_id: int
    #: Its role name, as :func:`canonical_role_name` reads it (``Ego``).
    role: str
    #: Its CARLA actor id in this run.
    actor_id: int
    #: Its blueprint.
    type_id: str
    #: The rows of the rollout the log has it on (the others hold its pose).
    valid_steps: int

    @property
    def object_type(self) -> int:
        """Its ``Track.ObjectType`` (:func:`object_type_of`)."""
        return object_type_of(self.type_id)

    @property
    def ego(self) -> bool:
        """Whether it is the ego."""
        return is_ego(self.role)


@dataclass(frozen=True)
class WosacRollout:
    """A converted log: its ``ScenarioRollouts`` (one ``JointScene``), and
    what :func:`write_rollout` writes beside it."""

    rollouts: sim_agents_submission_pb2.ScenarioRollouts
    agents: tuple[WosacAgent, ...]
    #: The log's stem.
    stem: str
    #: The CARLA frame of the current row.
    current_frame: int
    #: Whether positions are in Autoware's ``map`` frame, offsets and all.
    map_frame: bool
    #: The offset added to every position: the map's MGRS and elevation
    #: offsets with :attr:`map_frame`, else zero.
    offset: tuple[float, float, float]
    current_index: int
    steps: int
    #: The rows after the end of the log, which hold every agent's last pose.
    padded_steps: int
    #: The run's verdict, the log's ``result`` record (``None`` without one).
    result: Optional[Mapping[str, Any]]

    @property
    def current_t(self) -> float:
        """The current row's time after the log's first tick, seconds."""
        return round(self.current_index * STEP_S, 6)

    @property
    def frame(self) -> str:
        """``map`` (Autoware's ``map`` frame) or ``local`` (its axes at
        CARLA's world origin: no MGRS or elevation offset)."""
        return "map" if self.map_frame else "local"

    def metadata(self) -> dict[str, Any]:
        """What ``{stem}.wosac.json`` holds: the rollout's grid, frame and
        agents, which the ``ScenarioRollouts`` has no field for."""
        result = None
        if self.result is not None:
            result = {
                "passed": self.result.get("passed"),
                "message": self.result.get("message"),
            }
        return {
            "scenario_id": self.rollouts.scenario_id,
            "stem": self.stem,
            "step_s": STEP_S,
            "current_index": self.current_index,
            "steps": self.steps,
            "padded_steps": self.padded_steps,
            "current_frame": self.current_frame,
            "current_t": self.current_t,
            "frame": self.frame,
            "offset": list(self.offset),
            "result": result,
            "agents": [
                {
                    "object_id": agent.object_id,
                    "role": agent.role,
                    "actor_id": agent.actor_id,
                    "type_id": agent.type_id,
                    "object_type": _TRACK.ObjectType.Name(agent.object_type),
                    "ego": agent.ego,
                    "valid_steps": agent.valid_steps,
                }
                for agent in self.agents
            ],
        }


def scenario_id_of(stem: str) -> str:
    """The scenario a run of *stem* is a rollout of, by default: the stem
    without the run counter a directory's later runs of one scenario get
    (``CutInScenario-2`` is a rollout of ``CutInScenario``)."""
    return _RUN_COUNTER.sub("", stem)


def object_id_of(role: str) -> int:
    """The ``object_id`` of an agent of *role*: the first 31 bits of the
    SHA-256 of its canonical role name (``ego`` and ``hero`` are ``Ego``).

    A run's actor ids are CARLA's, which differ from run to run; a role name
    is the scenario's, so the same entity has the same id in every rollout.
    """
    digest = hashlib.sha256(canonical_role_name(role).encode()).digest()
    return int.from_bytes(digest[:4], "big") & _INT32_MAX


def object_type_of(type_id: str) -> int:
    """The ``Track.ObjectType`` of a CARLA blueprint: a walker is a
    pedestrian, one of CARLA's bicycles a cyclist, any other vehicle a
    vehicle (motorcycles too, as in the Waymo Open Dataset), else other."""
    if type_id.startswith("walker."):
        return _TRACK.TYPE_PEDESTRIAN
    if type_id in CARLA_BICYCLE_BLUEPRINTS:
        return _TRACK.TYPE_CYCLIST
    if type_id.startswith("vehicle."):
        return _TRACK.TYPE_VEHICLE
    return _TRACK.TYPE_OTHER


def checked_count(
    value: int, key: str, minimum: int, maximum: Optional[int] = None
) -> int:
    """*value* as an int: a whole number (:func:`seed.check_whole_number`)
    from *minimum* to *maximum* (no limit when ``None``).

    Raises:
        ValueError: naming *key*.
    """
    number = check_whole_number(key, value)
    if number < minimum or (maximum is not None and number > maximum):
        bound = f">= {minimum}" if maximum is None else f"from {minimum} to {maximum}"
        raise ValueError(f"{key} must be a whole number {bound}, not {value!r}")
    return number


def _checked_object_ids(object_ids: Mapping[str, int]) -> dict[str, int]:
    """*object_ids* keyed by canonical role name, every id an int32 >= 0."""
    checked: dict[str, int] = {}
    for role, object_id in object_ids.items():
        checked[canonical_role_name(str(role))] = checked_count(
            object_id, f"the object id of {role!r}", 0, _INT32_MAX
        )
    return checked


def _agents_at(
    log: _Log, frame: int, include_ego: bool, path: Path
) -> list[tuple[int, str]]:
    """The (actor id, canonical role) of every road user on the scene at
    *frame*, in the order the log has them: the ones WOSAC would simulate.

    One without a role name has no id that holds across runs, and is left
    out with a warning; two of one role are refused.
    """
    agents: list[tuple[int, str]] = []
    unnamed: list[int] = []
    roles: dict[str, int] = {}
    states = log.states(frame)
    for state in states[log.present(states, frame)]:
        actor_id = int(state[0])
        actor = log.actors[actor_id]
        if is_ego(actor.role) and not include_ego:
            continue
        if not actor.role:
            unnamed.append(actor_id)
            continue
        role = canonical_role_name(actor.role)
        if role in roles:
            raise ValueError(
                f"{path}: actors {roles[role]} and {actor_id} are both {role!r}: "
                "an agent's object id is its role's, which has to be one actor's"
            )
        roles[role] = actor_id
        agents.append((actor_id, role))
    if unnamed:
        logger.warning(
            "%s: leaving out actors %s, which have no role name to give them "
            "an object id that holds across runs",
            path,
            ", ".join(str(i) for i in unnamed),
        )
    return agents


def trajectory_log_to_rollout(
    log_path: Path | str,
    *,
    scenario_id: Optional[str] = None,
    current_index: int = CURRENT_INDEX,
    steps: Optional[int] = FUTURE_STEPS,
    pad: bool = False,
    include_ego: bool = True,
    map_frame: bool = False,
    with_boxes: bool = False,
    object_ids: Optional[Mapping[str, int]] = None,
) -> WosacRollout:
    """Read the trajectory log at *log_path* as one WOSAC rollout.

    Args:
        log_path: A run's ``{stem}.trajectory.jsonl``.
        scenario_id: The ``ScenarioRollouts.scenario_id``: what its runs have
            in common.  Default: :func:`scenario_id_of` the log's stem.
        current_index: The current row, counted at 10 Hz from the log's first
            tick (WOSAC's ``current_time_index``, 10).
        steps: The rows after the current one in the rollout (WOSAC's 80);
            ``None`` takes every one up to the log's end.
        pad: Hold every agent's last pose on the rows after the end of the
            log (marked invalid), rather than refuse a log too short for
            *steps*.
        include_ego: Whether the ego is one of the agents.
        map_frame: Write positions in Autoware's ``map`` frame, the Lanelet2
            map's, offsets and all; by default in its axes at CARLA's world
            origin, where single precision keeps millimetres.
        with_boxes: Also write the optional ``width``, ``length``,
            ``height``, ``valid`` and ``object_type`` (what WOSAC's scenario
            generation challenge reads; the sim agents challenge ignores
            them).
        object_ids: The ``object_id`` of a role name, overriding
            :func:`object_id_of`'s: a scenario replaying a Waymo scenario
            gives its entities the tracks' ids.

    Raises:
        ValueError: If the log cannot be read, is too short for *steps* (and
            not *pad*) or has nothing on the current row, two agents share a
            role or an object id, or an option is out of range.
    """
    path = Path(log_path)
    current_index = checked_count(current_index, "current_index", 0)
    if steps is not None:
        steps = checked_count(steps, "steps", 1)
    given_ids = _checked_object_ids(object_ids or {})
    stride = step_frames(STEP_S)
    log = _read_log(path, stride)
    stem = scene_stem(path.name)

    current_frame = log.first_frame + current_index * stride
    available = (log.last_frame - current_frame) // stride
    if current_frame not in log.rows:
        raise ValueError(
            f"{path} has no tick at row {current_index} (frame {current_frame}, "
            f"{current_index * STEP_S:g} s after its first): it is "
            + ("too short" if available < 0 else "a tick without road users")
        )
    if steps is None:
        steps = available
        if steps < 1:
            raise ValueError(f"{path} ends at row {current_index}: no rows after it")
    elif available < steps and not pad:
        raise ValueError(
            f"{path} has {available} rows after row {current_index}, not the "
            f"{steps} asked for: ask for fewer steps, or pad the rest"
        )
    padded = max(steps - available, 0)

    roles = _agents_at(log, current_frame, include_ego, path)
    if not roles:
        raise ValueError(f"{path} has no road user on the scene at row {current_index}")
    ids = [given_ids.get(role, object_id_of(role)) for _, role in roles]
    seen: dict[int, str] = {}
    for object_id, (_, role) in zip(ids, roles):
        if object_id in seen:
            raise ValueError(
                f"{path}: {seen[object_id]!r} and {role!r} have the same object "
                f"id, {object_id}: give one another"
            )
        seen[object_id] = role

    frame_of = _MapFrame.of(log.map) if map_frame else _MapFrame(np.zeros(3))
    actor_ids = [actor_id for actor_id, _ in roles]
    actors = [log.actors[i] for i in actor_ids]

    # Row 0 is the current one, which every agent is on; the rest hold the
    # last pose where an agent is not.
    count = len(actors)
    positions = np.zeros((steps + 1, count, 3))
    headings = np.zeros((steps + 1, count))
    valid = np.zeros((steps + 1, count), dtype=bool)
    for row in range(steps + 1):
        frame = current_frame + row * stride
        if row > 0:
            positions[row] = positions[row - 1]
            headings[row] = headings[row - 1]
        states = log.states(frame)
        states = states[log.present(states, frame)]
        index = {int(state[0]): n for n, state in enumerate(states)}
        here = np.array([i in index for i in actor_ids], dtype=bool)
        if not here.any():
            continue
        picked = states[[index[i] for i in actor_ids if i in index]]
        positions[row, here], headings[row, here], _ = box_poses(
            frame_of, picked, [a for a, h in zip(actors, here) if h]
        )
        valid[row, here] = True

    scene = sim_agents_submission_pb2.JointScene()
    agents: list[WosacAgent] = []
    for n, actor in enumerate(actors):
        trajectory = scene.simulated_trajectories.add(object_id=ids[n])
        trajectory.center_x.extend(positions[1:, n, 0].tolist())
        trajectory.center_y.extend(positions[1:, n, 1].tolist())
        trajectory.center_z.extend(positions[1:, n, 2].tolist())
        trajectory.heading.extend(headings[1:, n].tolist())
        object_type = object_type_of(actor.type_id)
        if with_boxes:
            length, width, height = (2.0 * np.asarray(actor.extent)).tolist()
            trajectory.length.extend([length] * steps)
            trajectory.width.extend([width] * steps)
            trajectory.height.extend([height] * steps)
            trajectory.valid.extend(valid[1:, n].tolist())
            trajectory.object_type = object_type
        agents.append(
            WosacAgent(
                object_id=ids[n],
                role=roles[n][1],
                actor_id=actor_ids[n],
                type_id=actor.type_id,
                valid_steps=int(valid[1:, n].sum()),
            )
        )
    rollouts = sim_agents_submission_pb2.ScenarioRollouts(
        scenario_id=scenario_id_of(stem) if scenario_id is None else scenario_id,
        joint_scenes=[scene],
    )
    return WosacRollout(
        rollouts=rollouts,
        agents=tuple(agents),
        stem=stem,
        current_frame=current_frame,
        map_frame=map_frame,
        offset=(
            float(frame_of.offset[0]),
            float(frame_of.offset[1]),
            float(frame_of.offset[2]),
        ),
        current_index=current_index,
        steps=steps,
        padded_steps=padded,
        result=log.result,
    )


def write_rollout(
    log_path: Path | str,
    output_dir: Optional[Path | str] = None,
    **options: Any,
) -> tuple[Path, Path]:
    """Convert *log_path* and write ``{stem}.wosac.pb`` (the serialised
    ``ScenarioRollouts``) and ``{stem}.wosac.json`` (its agents and grid,
    :meth:`WosacRollout.metadata`) beside it, or in *output_dir*.

    *options* are :func:`trajectory_log_to_rollout`'s.

    Returns:
        The two paths.
    """
    path = Path(log_path)
    rollout = trajectory_log_to_rollout(path, **options)
    directory = path.parent if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rollouts_path = directory / f"{rollout.stem}{WOSAC_SUFFIX}"
    agents_path = directory / f"{rollout.stem}{WOSAC_AGENTS_SUFFIX}"
    rollouts_path.write_bytes(rollout.rollouts.SerializeToString(deterministic=True))
    agents_path.write_bytes(
        orjson.dumps(rollout.metadata(), option=orjson.OPT_INDENT_2) + b"\n"
    )
    return rollouts_path, agents_path
