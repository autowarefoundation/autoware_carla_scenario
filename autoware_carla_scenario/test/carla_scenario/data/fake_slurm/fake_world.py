"""A batch's world faked for the SLURM launcher's tests: no CARLA, no scenario.

:func:`install` replaces what ``examples.run`` builds and runs a batch with
(``build_scenario``, ``ScenarioQueue``, ``resolve_map_paths``,
``build_traffic_backend``), in the test's process for a local batch and, through
the ``sitecustomize`` the tests put on ``PYTHONPATH``, in every SLURM task.

A case's outcome depends only on its config, as a real one's should: the same
case passes or fails the same way whichever task runs it.  The fake queue
writes the files a run writes under the stem the runner would choose, and on
start runs a stand-in for CARLA -- a process with ``-carla-rpc-port=<port>`` on
its command line, in a session of its own, as ``CarlaServerManager`` starts
CARLA -- which it stops on exit.

Environment knobs: ``FAKE_CASE_SECONDS`` (each case takes this long),
``FAKE_CRASH_DIGESTS`` (comma-separated prefixes of case digests: the process
dies, ``os._exit``, on such a case once -- a file in ``FAKE_RECORD_DIR``
remembers it did), ``FAKE_RECORD_DIR`` (where each process writes the ports
and stems it ran with).

A case whose digest is a multiple of 5 raises on every attempt (``ERROR``)
after its trajectory log was opened, as a real run that raises once it has
started does: its stem is taken, the next case's is the one after it.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from autoware_carla_scenario.conditions import ConditionStatus, ScenarioResult
from autoware_carla_scenario.scenario_runner import _run_file_stem


def _digest(cfg: Any) -> str:
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(cfg.scenario, resolve=True)
    env = OmegaConf.to_container(cfg.environment, resolve=True)
    return hashlib.sha256(
        json.dumps([data, env], sort_keys=True, default=str).encode()
    ).hexdigest()


class FakeScenario:
    """What a case built: its config's digest, which decides its outcome."""

    def __init__(self, digest: str, port: int) -> None:
        self.digest = digest
        self.port = port


def _scenario_class(name: str) -> type:
    title = "".join(part.title() for part in name.split("_")) + "Scenario"
    return type(title, (FakeScenario,), {})


_CLASSES: dict[str, type] = {}


def fake_build(cfg: Any, **_: object) -> tuple[None, FakeScenario]:
    name = str(cfg.scenario.name)
    cls = _CLASSES.setdefault(name, _scenario_class(name))
    return None, cls(_digest(cfg), int(cfg.server.port))


class FakeQueue:
    """Runs fake scenarios, writing a run's files, on a fake CARLA."""

    def __init__(self, **kwargs: Any) -> None:
        self.output_dir = Path(kwargs["output_dir"])
        self.port = int(kwargs["port"])
        self.tm_port = int(kwargs["tm_port"])
        self.extra = list(kwargs.get("server_extra_args") or [])
        self._pending: list[tuple[FakeScenario, Any, str | None]] = []
        self._errored: list[int] = []
        self._carla: subprocess.Popen[bytes] | None = None

    def add(
        self,
        scenario: FakeScenario,
        *,
        timeout_seconds: float | None = None,
        case: Any = None,
        file_stem: str | None = None,
    ) -> None:
        self._pending.append((scenario, case, file_stem))

    def __enter__(self) -> "FakeQueue":
        self._carla = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import time\nwhile True: time.sleep(1)",
                f"-carla-rpc-port={self.port}",
                *self.extra,
                # What the tests' clean-up looks for: never a real CARLA.
                "--fake-carla",
            ],
            start_new_session=True,
        )
        record = os.environ.get("FAKE_RECORD_DIR")
        if record:
            path = Path(record) / f"carla-{os.getpid()}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "pid": self._carla.pid,
                        "port": self.port,
                        "tm_port": self.tm_port,
                        "extra": self.extra,
                    }
                )
            )
        return self

    def __exit__(self, *_: object) -> None:
        if self._carla is not None:
            try:
                os.killpg(self._carla.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self._carla.wait()
            self._carla = None

    def errored_indices(self) -> list[int]:
        return list(self._errored)

    def run_all(self, **_: object) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        self._errored = []
        pending, self._pending = self._pending, []
        for i, (scenario, _case, file_stem) in enumerate(pending):
            name = type(scenario).__name__
            stem = file_stem or _run_file_stem(self.output_dir, name)
            seconds = float(os.environ.get("FAKE_CASE_SECONDS", "0"))
            if seconds:
                time.sleep(seconds)
            _maybe_crash(scenario)
            value = int(scenario.digest[:8], 16)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if value % 5 == 0:
                # Raised on every attempt, once its trajectory log was opened.
                (self.output_dir / f"{stem}.trajectory.jsonl").write_text("")
                self._errored.append(i)
                results.append(
                    ScenarioResult(
                        passed=False,
                        message="Errored after 1 attempt(s): RuntimeError: no ground",
                        elapsed_seconds=0.0,
                    )
                )
                continue
            passed = value % 3 != 0
            result = ScenarioResult(
                passed=passed,
                message="ok" if passed else f"failed at {value % 97}",
                elapsed_seconds=(value % 1000) / 10.0,
                condition_statuses=[
                    ConditionStatus(
                        label="pass[0](goal)",
                        satisfied=passed,
                        message=f"digest {scenario.digest[:12]}",
                        condition_type="FakeCondition",
                        role="pass",
                        details={"value": value % 1000},
                    )
                ],
            )
            (self.output_dir / f"{stem}.trajectory.jsonl").write_text(
                json.dumps({"type": "world", "digest": scenario.digest}) + "\n"
            )
            (self.output_dir / f"{stem}_result.json").write_text(
                result.to_json(indent=2)
            )
            _record_stem(stem, scenario)
            results.append(result)
        return results


def _record_stem(stem: str, scenario: FakeScenario) -> None:
    record = os.environ.get("FAKE_RECORD_DIR")
    if not record:
        return
    with open(Path(record) / "stems.jsonl", "a") as stream:
        stream.write(
            json.dumps(
                {
                    "stem": stem,
                    "digest": scenario.digest,
                    "port": scenario.port,
                    "pid": os.getpid(),
                    "task": os.environ.get("SLURM_ARRAY_TASK_ID"),
                }
            )
            + "\n"
        )


def _maybe_crash(scenario: FakeScenario) -> None:
    """Die (``os._exit``) once on a case whose digest starts with a marker."""
    markers = [m for m in os.environ.get("FAKE_CRASH_DIGESTS", "").split(",") if m]
    record = os.environ.get("FAKE_RECORD_DIR")
    for marker in markers:
        if scenario.digest.startswith(marker) and record:
            flag = Path(record) / f"crashed-{marker}"
            if not flag.exists():
                flag.write_text("1")
                os._exit(9)


def fake_map_paths(_map: Any) -> SimpleNamespace:
    return SimpleNamespace(
        install_xodr=None,
        overwrite_xodr=False,
        opendrive_path=None,
        lanelet2_path=None,
        name="M",
        projector_type=None,
    )


def install(run: Any = None) -> None:
    """Fake the world of ``examples.run`` (*run*, imported if not given)."""
    if run is None:
        from autoware_carla_scenario.examples import run as imported

        run = imported
    run.build_scenario = fake_build
    run.ScenarioQueue = FakeQueue
    run.resolve_map_paths = fake_map_paths
    run.build_traffic_backend = lambda _cfg: None
