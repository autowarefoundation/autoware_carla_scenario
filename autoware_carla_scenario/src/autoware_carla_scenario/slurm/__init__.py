"""Running a batch's cases as SLURM array jobs: ``launcher=slurm`` (docs/slurm.md).

Off unless a run selects it.  The submitting side expands, labels, builds and
checks the batch exactly as a local batch does, then writes a manifest into the
batch's output directory and submits an array (:mod:`.submit`); each task runs
its cases on a CARLA of its own, on ports of its own (:mod:`.worker`,
:mod:`.ports`), into that directory; the outcomes are collected into the
summary and ``batch_results.json`` a local batch writes (:mod:`.collect`).
``scenario-slurm`` (:mod:`.cli`) follows, collects, resubmits and cancels.

Nothing here imports CARLA but the worker, which runs the cases.
"""

from .settings import SlurmSettings, launcher_type

__all__ = ["SlurmSettings", "launcher_type"]
