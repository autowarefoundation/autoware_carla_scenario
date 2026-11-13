"""Trajectory logs as Waymo Open Sim Agents Challenge (WOSAC) rollouts.

:func:`trajectory_log_to_rollout` reads a run's ``{stem}.trajectory.jsonl``
-- no CARLA needed -- as one rollout of its scenario: a ``ScenarioRollouts``
of one ``JointScene`` (``sim_agents_submission.proto``, vendored from
waymo-research/waymo-open-dataset), every agent's box centre and heading at
10 Hz after WOSAC's current step; :func:`write_rollout` writes it as
``{stem}.wosac.pb``, with its agents as ``{stem}.wosac.json``.  The runs of
one scenario are its rollouts: :func:`merge_rollouts` gathers them by
``scenario_id`` and checks they can be scored together, and
:func:`build_submission` and :func:`write_submission` pack them as a
``SimAgentsChallengeSubmission``'s shards.  From the shell::

    scenario-wosac export outputs/2026-11-13/
    scenario-wosac pack outputs/2026-11-13/ -o submission/ --rollouts 32 --tar

See docs/wosac.md for what goes where.
"""

from __future__ import annotations

from ._proto import scenario_pb2, sim_agents_submission_pb2
from .convert import (
    CURRENT_INDEX,
    FUTURE_STEPS,
    STEP_S,
    WosacAgent,
    WosacRollout,
    object_id_of,
    object_type_of,
    scenario_id_of,
    trajectory_log_to_rollout,
    write_rollout,
)
from .pack import (
    SHARD_PREFIX,
    TARBALL_NAME,
    build_submission,
    merge_rollouts,
    read_rollouts,
    read_submission,
    rollout_files,
    shard_name,
    write_submission,
)

__all__ = [
    "CURRENT_INDEX",
    "FUTURE_STEPS",
    "SHARD_PREFIX",
    "STEP_S",
    "TARBALL_NAME",
    "WosacAgent",
    "WosacRollout",
    "build_submission",
    "merge_rollouts",
    "object_id_of",
    "object_type_of",
    "read_rollouts",
    "read_submission",
    "rollout_files",
    "scenario_id_of",
    "scenario_pb2",
    "shard_name",
    "sim_agents_submission_pb2",
    "trajectory_log_to_rollout",
    "write_rollout",
    "write_submission",
]
