"""The command that starts the CommonRoad worker, and the worker's error.

The command is, in order:

1. ``$AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER``, split as a shell would;
2. ``commonroad-metrics-worker`` on the ``PATH``;
3. in a checkout of the repository, ``uv run --project <checkout>/commonroad_metrics
   commonroad-metrics-worker``, which builds the environment the first time.

It needs neither ``select`` nor the worker's JSON: :mod:`.worker`, which
starts the worker and speaks to it, re-exports it.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path
from typing import Optional

__all__ = [
    "WORKER_ENV",
    "WORKER_SCRIPT",
    "CommonRoadWorkerError",
    "worker_command",
]

#: The environment variable that names the command starting the worker.
WORKER_ENV = "AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER"
#: The worker's console script.
WORKER_SCRIPT = "commonroad-metrics-worker"


class CommonRoadWorkerError(RuntimeError):
    """The worker could not be started, or failed to answer."""

    # Codon 0.19 raises no subclass of a built-in exception without a
    # constructor of its own (docs/typecheck.md). This one hands on what it
    # is given, so Python builds it as RuntimeError's would: the same args,
    # message and repr for any arguments, and the same TypeError for a
    # keyword.
    def __init__(self, *args: str, **kwargs: str) -> None:
        super().__init__(*args, **kwargs)


#: How much of an answer out of protocol an error quotes.
_EXCERPT_CHARS = 200


def _excerpt(line: str) -> str:
    """*line*, stripped and cut to :data:`_EXCERPT_CHARS`, quoted."""
    text = line.strip()
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + "..."
    return repr(text)


def _checkout_project() -> Optional[Path]:
    """``commonroad_metrics/`` of the repository this package was installed
    from in place (``uv sync`` installs the workspace editable), if any."""
    # src/autoware_carla_scenario/commonroad/worker_start.py -> the repository root.
    root = Path(__file__).resolve().parents[4]
    project = root / "commonroad_metrics"
    return project if (project / "pyproject.toml").is_file() else None


def worker_command() -> list[str]:
    """The command that starts the worker (the module's docstring says how it
    is chosen).

    Raises:
        CommonRoadWorkerError: No way to start it was found.
    """
    configured = os.environ.get(WORKER_ENV, "").strip()
    if configured:
        return shlex.split(configured)
    script = shutil.which(WORKER_SCRIPT)
    if script:
        # An annotated local, which Codon unwraps (docs/typecheck.md: Codon
        # does not narrow an Optional), for a list[str].
        found: str = script
        return [found]
    project = _checkout_project()
    uv = shutil.which("uv")
    if project is not None and uv:
        runner: str = uv
        return [runner, "run", "--quiet", "--project", str(project), WORKER_SCRIPT]
    raise CommonRoadWorkerError(
        f"the CommonRoad metrics need their worker ({WORKER_SCRIPT}): install the "
        f"commonroad_metrics project (Python 3.10 or 3.11) and put it on the PATH, "
        f"or name the command in ${WORKER_ENV}"
    )
