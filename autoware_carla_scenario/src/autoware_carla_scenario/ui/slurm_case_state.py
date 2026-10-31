"""The state of one case of a SLURM batch, and the message the viewer shows for it.

A case is ``passed``, ``failed`` or ``errored`` once its result is recorded;
until then it is ``running`` or ``pending`` while ``squeue`` lists its task
as active, ``unknown`` when ``squeue`` could not be asked, and ``missing``
when it was never submitted or its task is gone without a result.

Split out of ``ui.slurm_launch`` so the library check compiles it (#839).
Reading the batch's JSON records (values of any type) and asking ``squeue``
stay there: ``_case_states()`` turns what it reads into the typed values
:func:`case_state` takes.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["case_state"]


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
