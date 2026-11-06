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
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

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

#: How long :meth:`CommonRoadWorker.close`, the worker gone, waits for a
#: request still holding a pipe to let go of it, in seconds.
_PIPE_RELEASE_S = 1.0


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
        on_start: Called with the worker as soon as its process runs, before
            it is given the map: whoever starts it can close it while it is
            still starting (an interrupt).  Should it raise, the worker is
            closed and the error raised.

    The worker runs in a session (and process group) of its own, so that
    :meth:`close` can kill all of it, the Python worker under a ``uv run``
    too.  So the terminal's Ctrl-C does not reach it: whoever starts it closes
    it (:func:`~autoware_carla_scenario.commonroad.session.close_commonroad_workers`
    does, at exit too).

    Raises:
        CommonRoadWorkerError: It could not be started or did not take the map.
    """

    def __init__(
        self,
        command: Sequence[str],
        map_path: Path,
        settings: Optional[dict[str, Any]] = None,
        *,
        on_start: Optional[Callable[["CommonRoadWorker"], None]] = None,
    ) -> None:
        self.command = list(command)
        # Held while a request writes to, or reads from, the worker: close()
        # touches a pipe only when no other thread is in it, as closing a
        # buffered pipe waits for the thread holding it (#1780).
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        # Set by close(): a request that fails from then on closes the pipe
        # it holds, which close() did not wait for.
        self._closing = False
        try:
            self._process: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise CommonRoadWorkerError(
                f"could not start the CommonRoad worker {self.command}: {exc}"
            ) from exc
        try:
            if on_start is not None:
                on_start(self)
            answer = self.request(
                {"op": "init", "map": str(map_path), "settings": settings or {}}
            )
        except BaseException:
            # Not handed to the caller, so not left running: in a session of
            # its own, nothing else would stop it (an interrupt included).
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
            with self._write_lock:
                try:
                    process.stdin.write(json.dumps(request) + "\n")
                    process.stdin.flush()
                except (OSError, ValueError):
                    self._close_if_closing(process.stdin)
                    raise
            with self._read_lock:
                try:
                    line = process.stdout.readline()
                except (OSError, ValueError):
                    self._close_if_closing(process.stdout)
                    raise
                if not line:
                    self._close_if_closing(process.stdout)
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

    def _close_if_closing(self, pipe: Any) -> None:
        """Close *pipe*, which the caller holds the lock of, if the worker is
        being closed: close() may have left it to this request (#1780)."""
        if self._closing:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()

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
        killed with its whole process group (a ``uv run`` and the Python
        worker under it, #1780).  It returns while another thread is blocked
        in :meth:`request`, which then raises.  Closing a worker that has
        exited, or closing it twice, does nothing more.
        """
        process = self._process
        self._closing = True
        self._ask_to_exit()
        try:
            process.wait(timeout=_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning(
                "the CommonRoad worker did not exit within %.0f s of being "
                "closed; killing it",
                _CLOSE_TIMEOUT_S,
            )
            # The worker leads its group (start_new_session), and the group
            # keeps that id while any of it lives, even once the leader is
            # reaped (by a request() that read the end of its output
            # meanwhile): the id cannot name another group then.
            # ProcessLookupError: all of it exited meanwhile.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                logger.warning(
                    "could not kill the CommonRoad worker's process group",
                    exc_info=True,
                )
                process.kill()
            process.wait()
        # Its answer to ``close``, if any, is not read.  A thread blocked in
        # request() holds a pipe's buffer, which closing would wait for.  Now
        # the worker is gone, that request fails (the end of the output, or a
        # broken input pipe) and, seeing _closing, closes the pipe itself;
        # the lock is waited for briefly, so that whichever of the two takes
        # it second finds the pipe closed (or closes it).  A pipe held open
        # by a process outside the worker's group is left to that request.
        self._close_pipe(process.stdin, self._write_lock)
        self._close_pipe(process.stdout, self._read_lock)

    @staticmethod
    def _close_pipe(pipe: Any, lock: threading.Lock) -> None:
        """Close *pipe* once no request holds *lock*, waiting at most
        :data:`_PIPE_RELEASE_S` for that."""
        if pipe is None or not lock.acquire(timeout=_PIPE_RELEASE_S):
            return
        try:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()
        finally:
            lock.release()

    def _ask_to_exit(self) -> None:
        """Send ``close`` and the end of the worker's input, without blocking.

        Skipped while another thread is writing a request: it is blocked on a
        worker that reads nothing, and closing the input would wait for it.
        The kill in :meth:`close` ends that write (``BrokenPipeError``).
        """
        process = self._process
        stdin = process.stdin
        if stdin is None or not self._write_lock.acquire(blocking=False):
            return
        try:
            if stdin.closed:
                return
            # A worker that reads nothing may have a full pipe: the request
            # is dropped (BlockingIOError) rather than waited for.
            with contextlib.suppress(OSError, ValueError):
                os.set_blocking(stdin.fileno(), False)
            if process.poll() is None:
                # A dead worker's pipe raises BrokenPipeError (an OSError).
                with contextlib.suppress(OSError, ValueError):
                    stdin.write(json.dumps({"op": "close"}) + "\n")
                    stdin.flush()
            # The end of its input, should it not read the request.  The
            # descriptor is closed even if the flush fails.
            with contextlib.suppress(OSError, ValueError):
                stdin.close()
        finally:
            self._write_lock.release()
