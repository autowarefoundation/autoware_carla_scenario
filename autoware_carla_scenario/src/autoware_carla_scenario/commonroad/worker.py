"""The CommonRoad worker, a process of its own, and how it is found.

CriMe and CommonRoad-Reach cannot share the framework's environment:
commonroad-io pins protobuf 3.20 where the framework's gRPC code needs 4 or
later, and CommonRoad-Reach builds for CPython 3.10 and 3.11 only.  They run
in the ``commonroad_metrics`` project at the root of the repository, with an
environment and a lock of its own, as ``commonroad-metrics-worker``: a process
that reads one JSON request a line on its standard input and answers each with
one line on its output (``commonroad_metrics/README.md`` has the protocol).

The command that starts it is, in order:

1. ``$AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER``, split as a shell would;
2. ``commonroad-metrics-worker`` on the ``PATH``;
3. in a checkout of the repository, ``uv run --project <checkout>/commonroad_metrics
   commonroad-metrics-worker``, which builds the environment the first time.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "WORKER_ENV",
    "CommonRoadWorker",
    "CommonRoadWorkerError",
    "worker_command",
]

logger = logging.getLogger(__name__)

#: The environment variable that names the command starting the worker.
WORKER_ENV = "AUTOWARE_CARLA_SCENARIO_COMMONROAD_WORKER"
#: The worker's console script.
WORKER_SCRIPT = "commonroad-metrics-worker"


class CommonRoadWorkerError(RuntimeError):
    """The worker could not be started, or failed to answer."""


#: How much of an answer out of protocol an error quotes.
_EXCERPT_CHARS = 200


#: How long :meth:`CommonRoadWorker.close` lets the worker exit before
#: killing it, in seconds.
_CLOSE_TIMEOUT_S = 5.0


def _excerpt(line: str) -> str:
    """*line*, stripped and cut to :data:`_EXCERPT_CHARS`, quoted."""
    text = line.strip()
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + "..."
    return repr(text)


def _checkout_project() -> Optional[Path]:
    """``commonroad_metrics/`` of the repository this package was installed
    from in place (``uv sync`` installs the workspace editable), if any."""
    # src/autoware_carla_scenario/commonroad/worker.py -> the repository root.
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
        return [script]
    project = _checkout_project()
    uv = shutil.which("uv")
    if project is not None and uv:
        return [uv, "run", "--quiet", "--project", str(project), WORKER_SCRIPT]
    raise CommonRoadWorkerError(
        f"the CommonRoad metrics need their worker ({WORKER_SCRIPT}): install the "
        f"commonroad_metrics project (Python 3.10 or 3.11) and put it on the PATH, "
        f"or name the command in ${WORKER_ENV}"
    )


class CommonRoadWorker:
    """A running worker, spoken to one request at a time.

    Args:
        command: The command starting it (:func:`worker_command`).
        map_path: The CommonRoad scenario of the map
            (:func:`~autoware_carla_scenario.commonroad.map.commonroad_map`).
        settings: The worker's evaluator settings (``dt``, ``horizon_steps``,
            ``radius_m``, ``reach_settings``).

    Raises:
        CommonRoadWorkerError: It could not be started or did not take the map.
    """

    def __init__(
        self,
        command: Sequence[str],
        map_path: Path,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        self.command = list(command)
        try:
            self._process: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise CommonRoadWorkerError(
                f"could not start the CommonRoad worker {self.command}: {exc}"
            ) from exc
        try:
            answer = self.request(
                {"op": "init", "map": str(map_path), "settings": settings or {}}
            )
        except CommonRoadWorkerError:
            # Not handed to the caller, so not left running.
            self.close()
            raise
        self.measures: list[str] = list(answer.get("measures", []))

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send *request*; its answer.

        Raises:
            CommonRoadWorkerError: The worker is gone, answered out of
                protocol (a line that is not a JSON object), or answered not
                ok.
        """
        process = self._process
        if (
            process.stdin is None
            or process.stdout is None
            or process.poll() is not None
        ):
            raise CommonRoadWorkerError(
                f"the CommonRoad worker has exited ({process.returncode})"
            )
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (OSError, ValueError) as exc:
            # ValueError: its pipes were closed (close()) meanwhile.
            raise CommonRoadWorkerError(f"the CommonRoad worker failed: {exc}") from exc
        if not line:
            raise CommonRoadWorkerError(
                f"the CommonRoad worker exited ({process.wait()}) without answering"
            )
        try:
            parsed = json.loads(line)
        except ValueError as exc:
            raise CommonRoadWorkerError(
                f"the CommonRoad worker answered {request.get('op')!r} out of "
                f"protocol, not with JSON: {_excerpt(line)}"
            ) from exc
        if not isinstance(parsed, dict):
            raise CommonRoadWorkerError(
                f"the CommonRoad worker answered {request.get('op')!r} out of "
                f"protocol, not with a JSON object: {_excerpt(line)}"
            )
        answer: dict[str, Any] = parsed
        if not answer.get("ok"):
            raise CommonRoadWorkerError(
                f"the CommonRoad worker refused {request.get('op')!r}: "
                f"{answer.get('error')}"
            )
        return answer

    def evaluate(
        self,
        measures: Sequence[str],
        ego: dict[str, Any],
        others: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, Optional[float]], dict[str, str]]:
        """The *measures* in a frame, and the error of each that failed."""
        answer = self.request(
            {
                "op": "evaluate",
                "measures": list(measures),
                "ego": ego,
                "others": list(others),
            }
        )
        values: dict[str, Optional[float]] = {}
        for key, value in dict(answer.get("values", {})).items():
            values[str(key)] = None if value is None else float(value)
        errors = {str(k): str(v) for k, v in dict(answer.get("errors", {})).items()}
        return values, errors

    def running(self) -> bool:
        """Whether the worker's process has not exited."""
        return self._process.poll() is None

    def close(self) -> None:
        """Ask the worker to exit, and make sure it does.

        The ``close`` request is sent, and the worker's input closed, without
        waiting for an answer: a worker deep in an evaluation (or stuck in
        one) would not give it, and a run interrupted with Ctrl-C would wait
        for it forever (#1775).  The worker exits on the request or on the end
        of its input; one that has not after :data:`_CLOSE_TIMEOUT_S` is
        killed.  Closing a worker that has exited, or closing it twice, does
        nothing more.
        """
        process = self._process
        stdin = process.stdin
        if process.poll() is None and stdin is not None and not stdin.closed:
            # A dead worker's pipe raises BrokenPipeError (an OSError); one
            # closed by another thread meanwhile, ValueError.
            with contextlib.suppress(OSError, ValueError):
                stdin.write(json.dumps({"op": "close"}) + "\n")
                stdin.flush()
        if stdin is not None:
            # The end of its input, should it not read the request.
            with contextlib.suppress(OSError, ValueError):
                stdin.close()
        try:
            process.wait(timeout=_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "the CommonRoad worker did not exit within %.0f s of being "
                "closed; killing it",
                _CLOSE_TIMEOUT_S,
            )
            process.kill()
            process.wait()
        if process.stdout is not None:
            # Its answer to ``close``, if any, is not read.
            with contextlib.suppress(OSError, ValueError):
                process.stdout.close()
