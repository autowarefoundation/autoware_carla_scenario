"""The state of one case of a SLURM batch, and the message the viewer shows for it.

A case is ``passed``, ``failed`` or ``errored`` once its result is recorded;
until then it is ``running`` or ``pending`` while ``squeue`` lists its task
as active, ``unknown`` when ``squeue`` could not be asked, and ``missing``
when it was never submitted or its task is gone without a result.  A case
whose submission has no job id (:func:`no_job_case_state`, #1791) is
``pending`` while its sbatch is in flight, ``unknown`` once that sbatch timed
out (the job may exist), and ``missing`` when the record is stale.

Split out of ``ui.slurm_launch`` so the library check compiles it (#839).
Reading the batch's JSON records (values of any type) and asking ``squeue``
stay there: ``_case_states()`` turns what it reads into the typed values
:func:`case_state` takes.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["case_state", "no_job_case_state"]


def case_state(
    recorded: bool,
    errored: bool,
    passed: bool,
    given: bool,
    queued: Optional[str],
    squeue_known: bool,
    at_case: bool,
    task: int,
    job_id: str,
    task_state: str,
    task_error: Optional[str],
    active_states: Tuple[str, ...],
) -> Tuple[str, str]:
    """A case's state and the message that says why.

    Args:
        recorded: Whether the case's result is recorded; its state is then
            ``errored``, else ``passed`` or ``failed``, and the message is
            ``""``: a recorded case's message is its record's.
        errored: Whether the record says the case errored.
        passed: Whether the recorded result passed.
        given: Whether a submission was given the case.
        queued: The state ``squeue`` lists for the case's task, if it lists
            one.
        squeue_known: Whether ``squeue`` could be asked about the case's
            submission.
        at_case: Whether the task's status names this case as the one it is
            running.
        task: The case's task in its submission's job array.
        job_id: The submission's job, as the message shows it.
        task_state: The task's state as its status records it (``"never
            started"`` without one).
        task_error: The error the task's status records, if any.
        active_states: The states of a task SLURM has not finished with.
    """
    if recorded:
        if errored:
            return "errored", ""
        if passed:
            return "passed", ""
        return "failed", ""
    if not given:
        return "missing", "never submitted"
    if queued is not None and queued in active_states:
        state = "pending"
        if queued == "RUNNING" and at_case:
            state = "running"
        return state, f"task {task} of job {job_id}: {queued.lower()}"
    if not squeue_known:
        return "unknown", "squeue could not be asked"
    why = task_state
    if task_error is not None:
        why += f": {task_error}"
    return "missing", f"no result: task {task} of job {job_id} {why}"


def no_job_case_state(
    submission: int, timed_out: bool, in_flight: bool, timed_out_at: str
) -> Tuple[str, str]:
    """The state and message of a case with no result whose submission has
    no job id (#1791): there is no job to ask ``squeue`` about, nor one to
    name.

    Args:
        submission: The index of the submission the case was given to.
        timed_out: Whether the submission's sbatch timed out: SLURM may have
            taken the job all the same, so the case is ``unknown``, not
            ``missing``.
        in_flight: Whether the submission is still being submitted (its
            sbatch has not returned yet): the case is ``pending``.
        timed_out_at: When the sbatch timed out, as its record says (``""``
            when it does not).
    """
    if timed_out:
        when = f" at {timed_out_at}" if timed_out_at else ""
        return (
            "unknown",
            f"no job id: the sbatch of submission {submission} timed out{when}; "
            "the job may exist all the same: check `squeue --me`",
        )
    if in_flight:
        return (
            "pending",
            f"no job id yet: submission {submission} is being submitted "
            "(sbatch in flight)",
        )
    return (
        "missing",
        f"no result: submission {submission} has no job id (its submitter "
        "stopped before sbatch returned)",
    )
