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
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from google.protobuf import json_format

from ..run_outputs import WOSAC_SUFFIX
from ._proto import sim_agents_submission_pb2 as pb

__all__ = [
    "SHARD_PREFIX",
    "TARBALL_NAME",
    "build_submission",
    "merge_rollouts",
    "read_rollouts",
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

#: The fields of a ``SimulatedTrajectory`` that hold a value per row.
_PER_ROW = (
    "center_x",
    "center_y",
    "center_z",
    "heading",
    "width",
    "length",
    "height",
    "valid",
)


def rollout_files(paths: Iterable[Path]) -> list[Path]:
    """The rollout files *paths* name: a file itself, or every
    ``*.wosac.pb`` under a directory (sorted)."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob(f"*{WOSAC_SUFFIX}")))
        else:
            found.append(path)
    return found


def read_rollouts(path: Path | str) -> pb.ScenarioRollouts:
    """The ``ScenarioRollouts`` serialised in the file at *path*."""
    return pb.ScenarioRollouts.FromString(Path(path).read_bytes())


def read_submission(paths: Iterable[Path | str]) -> pb.SimAgentsChallengeSubmission:
    """The submission whose shards are at *paths*, as one message: its
    ``scenario_rollouts`` every shard's, its metadata the last shard's."""
    submission = pb.SimAgentsChallengeSubmission()
    for path in paths:
        submission.MergeFromString(Path(path).read_bytes())
    return submission


def _length(trajectory: pb.SimulatedTrajectory) -> int:
    return len(trajectory.center_x)


def _check_trajectory(trajectory: pb.SimulatedTrajectory, where: str) -> int:
    """The rows of *trajectory*, whose every per-row field holds that many
    values or none (the optional ones)."""
    rows = _length(trajectory)
    for name in _PER_ROW:
        values = getattr(trajectory, name)
        required = name in ("center_x", "center_y", "center_z", "heading")
        if (required or len(values)) and len(values) != rows:
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
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(f"{source}: object ids {duplicated} appear twice")
        id_sets.append(set(ids))
    common = set.intersection(*id_sets)
    union = set.union(*id_sets)
    if common != union:
        if not common_agents:
            first = id_sets[0]
            other = next(
                (n for n, ids in enumerate(id_sets) if ids != first), len(id_sets)
            )
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
        ValueError: If a message has no ``scenario_id``, a scenario has not
            *num_rollouts* rollouts, its rollouts simulate different agents
            (and not *common_agents*), or two of its trajectories differ in
            length.
    """
    if num_rollouts is not None and (
        isinstance(num_rollouts, bool)
        or not isinstance(num_rollouts, int)
        or num_rollouts < 1
    ):
        raise ValueError(
            f"num_rollouts must be a whole number >= 1, not {num_rollouts!r}"
        )
    scenarios: dict[str, list[tuple[pb.JointScene, str]]] = {}
    for message, source in rollouts:
        if not message.scenario_id:
            raise ValueError(f"{source} has no scenario_id")
        scenes = scenarios.setdefault(message.scenario_id, [])
        scenes.extend((scene, source) for scene in message.joint_scenes)
    return [
        _merge_one(scenario_id, scenarios[scenario_id], num_rollouts, common_agents)
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
    if (
        isinstance(shards, bool)
        or not isinstance(shards, int)
        or not 1 <= shards <= max(count, 1)
    ):
        raise ValueError(
            f"shards must be a whole number from 1 to the {count} scenarios, not {shards!r}"
        )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stale = sorted(directory.glob(f"{SHARD_PREFIX}-*"))
    if stale:
        raise ValueError(
            f"{directory} holds a submission already ({stale[0].name}): write "
            "into another directory"
        )
    header = pb.SimAgentsChallengeSubmission()
    header.CopyFrom(submission)
    header.ClearField("scenario_rollouts")
    paths: list[Path] = []
    for index in range(shards):
        start = index * count // shards
        stop = (index + 1) * count // shards
        shard = pb.SimAgentsChallengeSubmission()
        shard.CopyFrom(header)
        shard.scenario_rollouts.extend(submission.scenario_rollouts[start:stop])
        path = directory / shard_name(index, shards)
        path.write_bytes(shard.SerializeToString(deterministic=True))
        paths.append(path)
    if tar:
        archive = directory / TARBALL_NAME
        with tarfile.open(archive, "w:gz") as stream:
            for path in paths:
                stream.add(path, arcname=path.name)
        paths.append(archive)
    return paths
