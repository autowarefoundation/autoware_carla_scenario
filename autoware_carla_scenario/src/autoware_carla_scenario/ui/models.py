"""Pydantic view models for the scenario result viewer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Status = Literal["running", "passed", "failed", "done", "idle"]


class SessionSummary(BaseModel):
    """Summary of a single test session (multirun, batch, or single)."""

    date: str
    time: str
    session_type: str  # "multirun", "batch", or "single"
    scenario_name: str
    passed_count: int
    total_count: int
    #: Cases with no result to read (a SLURM batch's cases not collected, a
    #: job's missing or unreadable result file), which neither passed nor
    #: failed: the session page's "without a result" (#2285).
    pending_count: int = 0


class SessionItem(BaseModel):
    """One scenario within a session."""

    index: int
    scenario_name: str
    #: None when the case has no verdict: no result, or one whose ``passed``
    #: is not a bool.
    passed: bool | None
    elapsed_seconds: float | None
    overrides: list[str]
    message: str = ""
    #: The case has no result to read (#2285): its result file is missing or
    #: unreadable, or a batch lists it as pending (a SLURM case without an
    #: outcome yet, or with a stale one).  Neither passed nor failed, as the
    #: session list counts it (``SessionSummary.pending_count``); a result
    #: without a verdict is not pending, and counts as failed.
    pending: bool = False


class ConditionNode(BaseModel):
    """Recursive representation of a condition tree node."""

    label: str
    satisfied: bool
    message: str
    condition_type: str
    role: str
    details: dict[str, Any]
    children: list[ConditionNode] = []


class ScenarioResultView(BaseModel):
    """Full detail view of a single scenario result."""

    passed: bool | None
    message: str
    elapsed_seconds: float | None
    condition_statuses: list[ConditionNode]
    overrides: list[str]
    raw_log: str = ""
    #: The run's trajectory log, to make a Rerun recording of the run from.
    trajectory_path: str | None = None


class RunProgress(BaseModel):
    """SSE progress event payload."""

    current: int
    total: int
    scenario_name: str
    status: Status
    #: Why the run ended before any job ran (its multirun directory could not
    #: be created, #2183): set on its ``done``, which the viewer then shows as
    #: a failure, not as completed.
    message: str = ""
