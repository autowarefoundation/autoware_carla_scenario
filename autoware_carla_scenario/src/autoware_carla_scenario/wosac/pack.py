"""Rollouts merged by scenario, and packed as a WOSAC submission (docs/wosac.md).

Each run's ``{stem}.wosac.pb`` is a ``ScenarioRollouts`` of one
``JointScene`` (:mod:`.convert`).  The runs of one scenario are its rollouts:
:func:`merge_rollouts` gathers them by ``scenario_id`` into one
``ScenarioRollouts`` each, and checks they can be scored together -- the same
agents in every rollout, every trajectory the same length, and as many
rollouts as asked for (WOSAC's 32).  :func:`build_submission` puts the
scenarios into a ``SimAgentsChallengeSubmission`` with the submission's
metadata, and :func:`write_submission` writes it as WOSAC's shards
(``submission.binproto-00000-of-00001``), and the ``.tar.gz`` of them that
the challenge's server takes.

A ``ScenarioRollouts`` file merges as Protobuf merges any message: the
serialised bytes of two of one ``scenario_id``, concatenated, parse as one
whose ``joint_scenes`` are both files'.

Whether a scenario's runs are rollouts of it -- one initial state, futures
that differ only by chance -- is checked from what each run's
``{stem}.wosac.json`` says (:func:`check_runs`): they ran one concrete
scenario (one ``scenario_hash``), each with its own seed, and started its
future from the same state.
"""

from __future__ import annotations

import json
import logging
import math
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from google.protobuf import json_format
from google.protobuf.message import DecodeError

from ..atomic_file import write_chunks_atomically
from ..kinematics.angle import normalize_angle_rad
from ..run_outputs import WOSAC_SUFFIX, wosac_agents_path
from ..t4_scene.grid import files_under
from ._proto import sim_agents_submission_pb2 as pb
from .ids import checked_count

__all__ = [
    "SHARD_PREFIX",
    "TARBALL_NAME",
    "POSITION_TOLERANCE_M",
    "HEADING_TOLERANCE_RAD",
    "build_submission",
    "check_runs",
    "merge_rollouts",
    "read_rollouts",
    "read_run",
    "read_submission",
    "rollout_files",
    "shard_name",
    "write_submission",
]

logger = logging.getLogger(__name__)

#: The shards' names: ``submission.binproto-{index:05d}-of-{count:05d}``, as
#: WOSAC's submission tutorial writes them.
SHARD_PREFIX = "submission.binproto"
#: The archive of the shards the challenge's server takes.
TARBALL_NAME = "submission.tar.gz"

#: How far apart an agent may be, and how differently headed, on the current
#: row of two rollouts of one scenario (:func:`check_runs`): its history is
#: simulated too, and a stack that is not deterministic drifts a little.
POSITION_TOLERANCE_M = 0.5
HEADING_TOLERANCE_RAD = 0.1

#: The fields of a ``SimulatedTrajectory`` that hold a value per row: these
#: always, and the optional ones none or as many.
_REQUIRED = ("center_x", "center_y", "center_z", "heading")
_OPTIONAL = ("width", "length", "height", "valid")


def rollout_files(paths: list[Path]) -> list[Path]:
    """The rollout files *paths* name: a file itself, or every
    ``*.wosac.pb`` under a directory (sorted)."""
    return files_under(paths, WOSAC_SUFFIX)


def read_rollouts(path: Path | str) -> pb.ScenarioRollouts:
    """The ``ScenarioRollouts`` serialised in the file at *path*.

    Raises:
        ValueError: If the file does not hold one (a truncated export, a
            stray file), naming the file.
    """
    try:
        return pb.ScenarioRollouts.FromString(Path(path).read_bytes())
    except DecodeError as error:
        raise ValueError(f"{path}: not a ScenarioRollouts ({error})") from None


def read_run(path: Path | str) -> Optional[dict[str, Any]]:
    """What the run of the rollout file at *path* says of itself: the
    ``{stem}.wosac.json`` beside its ``{stem}.wosac.pb``, or ``None`` when
    there is none (a file merged from several, or renamed).

    Raises:
        ValueError: If the file is there but holds no JSON object, or one
            whose ``seed``, ``offset`` or ``agents`` :func:`check_runs` cannot
            read (:func:`_check_run`), naming the file.
    """
    if not Path(path).name.endswith(WOSAC_SUFFIX):
        return None
    sidecar = wosac_agents_path(Path(path))
    if not sidecar.is_file():
        return None
    try:
        run = json.loads(sidecar.read_text())
    except ValueError as error:
        raise ValueError(f"{sidecar}: not JSON ({error})") from None
    if not isinstance(run, dict):
        raise ValueError(f"{sidecar}: not a JSON object")
    problem = _check_run(run)
    if problem is not None:
        raise ValueError(f"{sidecar}: {problem}")
    return run


def _whole(value: Any) -> bool:
    """Whether ``int(value)`` reads *value* as a whole number: not a bool,
    which a sidecar never holds for an id or a seed, nor a fraction, which
    ``int()`` would truncate (seeds 1.2 and 1.7 would both read as 1)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, float) and not value.is_integer():
        return False
    try:
        int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _numbers(value: Any, count: int) -> bool:
    """Whether *value* is a list of *count* values ``float()`` reads."""
    if not isinstance(value, list) or len(value) != count:
        return False
    for item in value:
        if isinstance(item, bool):
            return False
        try:
            float(item)
        except (TypeError, ValueError):
            return False
    return True


def _check_run(run: Mapping[str, Any]) -> Optional[str]:
    """Why :func:`check_runs` cannot read the sidecar *run*
    (:func:`_check_scenario_runs`, :func:`_current_poses`), or ``None``: a
    ``seed`` that is no whole number, an ``offset`` that is not three
    numbers, ``agents`` that are not a list of objects, each with a whole
    ``object_id`` and a ``current`` pose of four numbers or null (#2208).
    Absent or null, ``seed``, ``offset`` and ``agents`` are let through, as
    :func:`check_runs` reads them."""
    seed = run.get("seed")
    if seed is not None and not _whole(seed):
        return f"seed must be a whole number or null, not {seed!r}"
    offset = run.get("offset")
    if offset is not None and not _numbers(offset, 3):
        return f"offset must be a list of 3 numbers or null, not {offset!r}"
    agents = run.get("agents")
    if agents is None:
        return None
    if not isinstance(agents, list):
        return f"agents must be a list or null, not {agents!r}"
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            return f"agents[{index}] must be an object, not {agent!r}"
        if not _whole(agent.get("object_id")):
            return (
                f"agents[{index}].object_id must be a whole number, not "
                f"{agent.get('object_id')!r}"
            )
        current = agent.get("current")
        if current is not None and not _numbers(current, 4):
            return (
                f"agents[{index}].current must be a list of 4 numbers "
                f"(x, y, z, heading) or null, not {current!r}"
            )
    return None


def _check_scenario_runs(
    scenario_id: str,
    runs: list[tuple[Mapping[str, Any], str]],
    position_tolerance_m: float,
    heading_tolerance_rad: float,
) -> None:
    """The runs of one scenario, checked as rollouts of it (:func:`check_runs`)."""
    first_hash: Optional[tuple[str, str]] = None
    seeds: dict[int, str] = {}
    unknown: list[str] = []
    for run, source in runs:
        digest = run.get("scenario_hash")
        seed = run.get("seed")
        if digest is None or seed is None:
            unknown.append(source)
        if digest is not None:
            if first_hash is None:
                first_hash = (str(digest), source)
            elif str(digest) != first_hash[0]:
                raise ValueError(
                    f"scenario {scenario_id!r}: {first_hash[1]} and {source} ran "
                    f"different scenarios (scenario_hash {first_hash[0][:12]}... "
                    f"and {str(digest)[:12]}...), which are not rollouts of one: "
                    "give them different scenario ids"
                )
        if seed is not None:
            seed = int(seed)
            if seed in seeds:
                raise ValueError(
                    f"scenario {scenario_id!r}: {seeds[seed]} and {source} both "
                    f"ran seed {seed}, so they are one rollout run twice: give "
                    "every rollout its own seed"
                )
            seeds[seed] = source
    if unknown:
        logger.warning(
            "scenario %r: cannot tell that %s ran the scenario with a seed of "
            "its own: no scenario_hash or seed (a log written before they were "
            "recorded, or a scenario not built by the scenario command)",
            scenario_id,
            ", ".join(unknown),
        )
    # The initial state: every agent where the first run has it.
    reference, reference_source = runs[0]
    poses = _current_poses(reference)
    for run, source in runs[1:]:
        for object_id, pose in _current_poses(run).items():
            start = poses.get(object_id)
            if start is None:
                continue
            distance = math.dist(pose[:3], start[:3])
            turn = abs(normalize_angle_rad(pose[3] - start[3]))
            if distance > position_tolerance_m or turn > heading_tolerance_rad:
                raise ValueError(
                    f"scenario {scenario_id!r}: object {object_id} starts its "
                    f"future {distance:.3f} m and {turn:.3f} rad away in "
                    f"{source} from where it does in {reference_source} (at "
                    f"most {position_tolerance_m:g} m and "
                    f"{heading_tolerance_rad:g} rad), so they do not share an "
                    "initial state: is something placed at random before the "
                    "current row?"
                )


def _current_poses(run: Mapping[str, Any]) -> dict[int, tuple[float, ...]]:
    """Each agent's pose on the current row of *run*, by object id, in the
    frame without the map's offsets (two runs may be written in either)."""
    offset = [float(v) for v in run.get("offset") or (0.0, 0.0, 0.0)]
    poses: dict[int, tuple[float, ...]] = {}
    for agent in run.get("agents") or ():
        current = agent.get("current")
        if current is None:
            continue
        x, y, z, heading = (float(v) for v in current)
        poses[int(agent["object_id"])] = (
            x - offset[0],
            y - offset[1],
            z - offset[2],
            heading,
        )
    return poses


def check_runs(
    runs: Iterable[tuple[Mapping[str, Any], str]],
    *,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    heading_tolerance_rad: float = HEADING_TOLERANCE_RAD,
) -> None:
    """Check that the runs of each scenario are rollouts of it.

    Whether runs are rollouts of one scenario cannot be told from what they
    did -- a future that differs may be chance, or a parameter that differs
    -- so it is told from what they were asked to run, and checked where it
    can be from where they started:

    * every run of a ``scenario_id`` ran one concrete scenario: one
      ``scenario_hash`` (:mod:`~autoware_carla_scenario.scenario_hash`);
    * each with a seed of its own: two of one seed are one rollout twice;
    * and each agent starts its future -- the current row -- where it does in
      the first run, within *position_tolerance_m* and
      *heading_tolerance_rad*: an agent placed at random before it would
      not.

    A run that does not say its hash or seed (a log written before they were
    recorded) is let through, with a warning.

    Args:
        runs: ``(metadata, source)`` pairs: what a run's ``{stem}.wosac.json``
            holds (:func:`read_run`), and where it came from, which an error
            names.

    Raises:
        ValueError: If a scenario's runs ran different scenarios, two ran
            one seed, or an agent starts two of them apart.
    """
    position_tolerance_m = float(position_tolerance_m)
    heading_tolerance_rad = float(heading_tolerance_rad)
    for name, value in (
        ("position_tolerance_m", position_tolerance_m),
        ("heading_tolerance_rad", heading_tolerance_rad),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0, not {value!r}")
    scenarios: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    for run, source in runs:
        scenario_id = run.get("scenario_id")
        if not scenario_id:
            raise ValueError(f"{source} has no scenario_id")
        scenarios.setdefault(str(scenario_id), []).append((run, source))
    for scenario_id in sorted(scenarios):
        _check_scenario_runs(
            scenario_id,
            scenarios[scenario_id],
            position_tolerance_m,
            heading_tolerance_rad,
        )


def read_submission(paths: Iterable[Path | str]) -> pb.SimAgentsChallengeSubmission:
    """The submission whose shards are at *paths*, as one message: its
    ``scenario_rollouts`` every shard's, its metadata the last shard's."""
    # Every shard holds the whole metadata: merging the shards would append
    # its repeated fields (``authors``) once per shard, so each shard is
    # parsed alone and only its scenarios are kept from the ones before.
    submission = pb.SimAgentsChallengeSubmission()
    rollouts: list[pb.ScenarioRollouts] = []
    for path in paths:
        try:
            shard = pb.SimAgentsChallengeSubmission.FromString(Path(path).read_bytes())
        except DecodeError as error:
            raise ValueError(
                f"{path}: not a SimAgentsChallengeSubmission ({error})"
            ) from None
        rollouts.extend(shard.scenario_rollouts)
        submission = shard
    del submission.scenario_rollouts[:]
    submission.scenario_rollouts.extend(rollouts)
    return submission


def _check_trajectory(trajectory: pb.SimulatedTrajectory, where: str) -> int:
    """The rows of *trajectory*, whose every per-row field holds that many
    values or none (the optional ones)."""
    rows = len(trajectory.center_x)
    for name in _REQUIRED + _OPTIONAL:
        values = getattr(trajectory, name)
        if (name in _REQUIRED or len(values)) and len(values) != rows:
            raise ValueError(
                f"{where}: object {trajectory.object_id} has {len(values)} "
                f"{name} values and {rows} center_x values"
            )
    return rows


def _merge_one(
    scenario_id: str,
    scenes: list[tuple[pb.JointScene, str]],
    num_rollouts: Optional[int],
    common_agents: bool,
) -> pb.ScenarioRollouts:
    """The rollouts *scenes* of one scenario as one ``ScenarioRollouts``,
    checked (:func:`merge_rollouts`)."""
    if num_rollouts is not None and len(scenes) != num_rollouts:
        raise ValueError(
            f"scenario {scenario_id!r} has {len(scenes)} rollouts, not "
            f"{num_rollouts}: " + ", ".join(source for _, source in scenes)
        )
    id_sets: list[set[int]] = []
    for scene, source in scenes:
        ids = [trajectory.object_id for trajectory in scene.simulated_trajectories]
        duplicated = sorted(i for i, n in Counter(ids).items() if n > 1)
        if duplicated:
            raise ValueError(f"{source}: object ids {duplicated} appear twice")
        id_sets.append(set(ids))
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if common != union:
        if not common_agents:
            first = id_sets[0]
            other = next(n for n, ids in enumerate(id_sets) if ids != first)
            raise ValueError(
                f"scenario {scenario_id!r}: {scenes[0][1]} and {scenes[other][1]} "
                f"do not simulate the same agents (only in the first: "
                f"{sorted(first - id_sets[other])}, only in the second: "
                f"{sorted(id_sets[other] - first)}); keep the agents common to "
                "every rollout, or give the scenarios different ids"
            )
        if not common:
            raise ValueError(f"scenario {scenario_id!r}: no agent is in every rollout")
        logger.warning(
            "scenario %r: leaving out object ids %s, which are not in every rollout",
            scenario_id,
            sorted(union - common),
        )
    merged = pb.ScenarioRollouts(scenario_id=scenario_id)
    rows: Optional[int] = None
    for scene, source in scenes:
        out = merged.joint_scenes.add()
        for trajectory in sorted(
            scene.simulated_trajectories, key=lambda t: t.object_id
        ):
            if trajectory.object_id not in common:
                continue
            length = _check_trajectory(trajectory, source)
            if rows is None:
                rows = length
            elif length != rows:
                raise ValueError(
                    f"scenario {scenario_id!r}: {source} has object "
                    f"{trajectory.object_id} {length} rows long, where an "
                    f"earlier one is {rows}: every trajectory of a scenario "
                    "has to be as long"
                )
            out.simulated_trajectories.append(trajectory)
    return merged


def merge_rollouts(
    rollouts: Iterable[tuple[pb.ScenarioRollouts, str]],
    *,
    num_rollouts: Optional[int] = None,
    common_agents: bool = False,
) -> list[pb.ScenarioRollouts]:
    """Gather *rollouts* by ``scenario_id``: one ``ScenarioRollouts`` per
    scenario, sorted by id, its ``joint_scenes`` every rollout of it in the
    order given, each one's trajectories sorted by ``object_id``.

    Args:
        rollouts: ``(ScenarioRollouts, source)`` pairs -- a run's file, as
            :mod:`.convert` writes it, and where it came from, which an
            error names.  A message of several joint scenes adds them all.
        num_rollouts: How many rollouts every scenario must have (WOSAC's
            32); ``None`` takes any number.
        common_agents: Keep, in every rollout of a scenario, only the agents
            all of them simulate (the rest are logged), rather than refuse a
            scenario whose rollouts differ: a seed can change which road
            users a run spawns.

    Raises:
        ValueError: If a message has no ``scenario_id`` or no joint scene,
            a scenario has not *num_rollouts* rollouts, its rollouts simulate
            different agents (and not *common_agents*), or two of its
            trajectories differ in length.
    """
    if num_rollouts is not None:
        num_rollouts = checked_count(num_rollouts, "num_rollouts", 1)
    scenarios: dict[str, list[tuple[pb.JointScene, str]]] = {}
    for message, source in rollouts:
        if not message.scenario_id:
            raise ValueError(f"{source} has no scenario_id")
        if not message.joint_scenes:
            raise ValueError(f"{source} has no joint scene")
        scenes = scenarios.setdefault(message.scenario_id, [])
        scenes.extend((scene, source) for scene in message.joint_scenes)
    # Each scenario's source messages are let go once it is merged.
    return [
        _merge_one(scenario_id, scenarios.pop(scenario_id), num_rollouts, common_agents)
        for scenario_id in sorted(scenarios)
    ]


def build_submission(
    scenario_rollouts: Iterable[pb.ScenarioRollouts],
    metadata: Optional[Mapping[str, Any]] = None,
) -> pb.SimAgentsChallengeSubmission:
    """A ``SimAgentsChallengeSubmission`` of *scenario_rollouts*.

    Args:
        scenario_rollouts: The scenarios (:func:`merge_rollouts`).
        metadata: The submission's other fields, by their names in the proto
            (``account_name``, ``unique_method_name``, ``authors``,
            ``uses_lidar_data``, ``num_model_parameters``, ...), as JSON
            would hold them.  ``submission_type`` is
            ``SIM_AGENTS_SUBMISSION`` unless it names another.

    Raises:
        ValueError: If *metadata* names a field the message lacks, gives one
            a value of the wrong type, or sets ``scenario_rollouts``.
    """
    submission = pb.SimAgentsChallengeSubmission(
        submission_type=pb.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION
    )
    if metadata:
        if "scenario_rollouts" in metadata or "scenarioRollouts" in metadata:
            raise ValueError("the metadata cannot set scenario_rollouts")
        try:
            json_format.ParseDict(dict(metadata), submission)
        except json_format.ParseError as error:
            raise ValueError(f"the submission's metadata: {error}") from error
    submission.scenario_rollouts.extend(scenario_rollouts)
    return submission


def shard_name(index: int, count: int) -> str:
    """The file name of shard *index* (from 0) of *count*."""
    return f"{SHARD_PREFIX}-{index:05d}-of-{count:05d}"


def _shard_chunks(head: bytes, scenarios: Sequence[Any]) -> Iterator[bytes]:
    """A shard's bytes: the metadata's *head*, then each of *scenarios* as a
    submission of its own, which concatenate into one message."""
    yield head
    for scenario in scenarios:
        one = pb.SimAgentsChallengeSubmission(scenario_rollouts=[scenario])
        yield one.SerializeToString(deterministic=True)


def write_submission(
    submission: pb.SimAgentsChallengeSubmission,
    output_dir: Path | str,
    *,
    shards: int = 1,
    tar: bool = False,
) -> list[Path]:
    """Write *submission* into *output_dir* as *shards* files, each a
    ``SimAgentsChallengeSubmission`` of its metadata and a contiguous share
    of its scenarios; with *tar*, also ``submission.tar.gz`` of them, which
    the challenge's server takes.

    Returns:
        The shards' paths, then the archive's with *tar*.

    Raises:
        ValueError: If *shards* is not a whole number from 1 to the number
            of scenarios, or *output_dir* holds shards already (which an
            archive of the new ones would be mixed up with).
    """
    count = len(submission.scenario_rollouts)
    shards = checked_count(shards, "shards", 1, max(count, 1))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stale = sorted(directory.glob(f"{SHARD_PREFIX}-*"))
    if stale:
        raise ValueError(
            f"{directory} holds a submission already ({stale[0].name}): write "
            "into another directory"
        )
    # Every shard is the metadata's bytes and then its scenarios', one at a
    # time: serialised fields concatenate into one message, so the
    # submission's rollouts are never copied whole.
    header = pb.SimAgentsChallengeSubmission()
    for field, value in submission.ListFields():
        if field.name == "scenario_rollouts":
            continue
        # The metadata are scalars and repeated scalars; told apart by the
        # value, as ``FieldDescriptor.label`` is gone from protobuf 7.
        if hasattr(value, "extend"):
            getattr(header, field.name).extend(value)
        else:
            setattr(header, field.name, value)
    head = header.SerializeToString(deterministic=True)
    paths: list[Path] = []
    for index in range(shards):
        start = index * count // shards
        stop = (index + 1) * count // shards
        path = directory / shard_name(index, shards)
        # In one rename: a run killed mid-shard leaves no truncated shard,
        # which would parse as a submission short of scenarios.
        write_chunks_atomically(
            path, _shard_chunks(head, submission.scenario_rollouts[start:stop])
        )
        paths.append(path)
    if tar:
        archive = directory / TARBALL_NAME
        with tarfile.open(archive, "w:gz") as stream:
            for path in paths:
                stream.add(path, arcname=path.name)
        paths.append(archive)
    return paths
