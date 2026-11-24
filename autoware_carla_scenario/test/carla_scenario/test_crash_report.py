"""Crash reports (crash_report.py): what a run that raised or crashed leaves."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario import crash_report


@contextmanager
def _installed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str] | None = None,
) -> Iterator[None]:
    """A crash reporter installed for *argv*, *tmp_path* the working
    directory, with nothing of the environment's to change what it does."""
    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    for name in (crash_report.CRASH_REPORT_DIR_ENV, crash_report.DSN_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert crash_report.install(["scenario"] if argv is None else argv)
    try:
        yield
    finally:
        crash_report.uninstall()


@pytest.fixture
def reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A crash reporter installed in *tmp_path*, the working directory."""
    with _installed(tmp_path, monkeypatch, ["scenario", "scenario=demo", "seed=3"]):
        yield tmp_path


def _raise(message: str = "boom") -> None:
    local_value = 42  # noqa: F841 - the report keeps it
    raise ValueError(message)


def _caught(message: str = "boom") -> ValueError:
    try:
        _raise(message)
    except ValueError as exc:
        return exc
    raise AssertionError("unreachable")


def test_a_report_holds_what_it_takes_to_run_the_case_again(reporter: Path) -> None:
    logging.getLogger("demo").warning("the ego left its lane")
    run_dir = reporter / "run"
    with crash_report.run_context(
        run_dir, overrides=["scenario=demo", "seed=3"], config_yaml="seed: 3\n"
    ):
        report = crash_report.capture(_caught())

    assert report is not None
    assert report.parent == run_dir / crash_report.CRASH_REPORTS_DIRNAME
    assert "ValueError" in report.name
    assert sorted(p.name for p in report.iterdir()) == [
        "config.yaml",
        "context.json",
        "event.json",
        "log.txt",
        "repro.sh",
        "traceback.txt",
    ]
    event = json.loads((report / "event.json").read_text())
    (exception,) = event["exception"]["values"]
    assert exception["type"] == "ValueError"
    assert exception["value"] == "boom"
    frame = exception["stacktrace"]["frames"][-1]
    assert frame["function"] == "_raise"
    assert frame["vars"]["local_value"] == "42"
    assert event["contexts"]["run"]["overrides"] == ["scenario=demo", "seed=3"]
    assert "ValueError: boom" in (report / "traceback.txt").read_text()
    assert "the ego left its lane" in (report / "log.txt").read_text()
    assert (report / "config.yaml").read_text() == "seed: 3\n"
    context = json.loads((report / "context.json").read_text())
    assert context["argv"] == ["scenario", "scenario=demo", "seed=3"]
    assert context["output_dir"] == str(run_dir)
    script = (report / "repro.sh").read_text()
    assert script.splitlines()[-1] == 'exec scenario scenario=demo seed=3 "$@"'
    assert f"cd {reporter}" in script
    assert os.access(report / "repro.sh", os.X_OK)


def test_outside_a_run_a_report_goes_to_the_default_directory(reporter: Path) -> None:
    report = crash_report.capture(_caught())
    assert report is not None
    assert report.parent == Path("outputs") / crash_report.CRASH_REPORTS_DIRNAME
    # The command line the process was started with runs it again.
    script = (report / "repro.sh").read_text()
    assert script.splitlines()[-1] == 'exec scenario scenario=demo seed=3 "$@"'
    assert not (report / "config.yaml").exists()


def test_an_exception_is_reported_once(reporter: Path) -> None:
    exc = _caught()
    assert crash_report.capture(exc) is not None
    assert crash_report.capture(exc) is None
    # Nor is one raised from it: the queue's "failed after N attempts".
    try:
        raise RuntimeError("failed after 1 attempt(s)") from exc
    except RuntimeError as again:
        assert crash_report.capture(again) is None
    assert len(list((reporter / "outputs" / "crash_reports").iterdir())) == 1


def test_the_queue_s_failed_attempts_are_reported(reporter: Path) -> None:
    queue_logger = logging.getLogger("autoware_carla_scenario.scenario_queue")
    for _ in range(2):
        try:
            _raise("attempt")
        except ValueError:
            queue_logger.warning("Scenario attempt failed", exc_info=True)
    # A warning without an exception, and another logger's, are not.
    queue_logger.warning("no exception here")
    try:
        _raise("elsewhere")
    except ValueError:
        logging.getLogger("autoware_carla_scenario.other").warning("x", exc_info=True)
    reports = list((reporter / "outputs" / "crash_reports").iterdir())
    assert len(reports) == 2
    event = json.loads((reports[0] / "event.json").read_text())
    assert event["extra"]["message"] == "Scenario attempt failed"


def test_a_secret_in_the_environment_is_left_out(
    reporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARLA_SERVER_TOKEN", "hunter2")
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla")
    report = crash_report.capture(_caught())
    assert report is not None
    environment = json.loads((report / "context.json").read_text())["environment"]
    assert environment["CARLA_ROOT"] == "/opt/carla"
    assert "CARLA_SERVER_TOKEN" not in environment
    assert "hunter2" not in (report / "repro.sh").read_text()


def test_a_run_that_ends_normally_leaves_no_native_report(reporter: Path) -> None:
    run_dir = reporter / "run"
    run_dir.mkdir()
    with crash_report.run_context(run_dir, overrides=[]):
        (native,) = (run_dir / "crash_reports").iterdir()
        assert native.name.startswith("native-")
        assert sorted(p.name for p in native.iterdir()) == [
            "context.json",
            "faulthandler.log",
            "repro.sh",
        ]
    assert list(run_dir.iterdir()) == []


def test_reporting_captures_what_the_block_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    monkeypatch.chdir(tmp_path)

    class Refusal(Exception):
        pass

    with (
        pytest.raises(Refusal),
        crash_report.reporting(["scenario"], ignore=(Refusal,)),
    ):
        raise Refusal("a bad value")
    assert not (tmp_path / "outputs").exists()
    with pytest.raises(ValueError), crash_report.reporting(["scenario"]):
        _raise()
    assert len(list((tmp_path / "outputs" / "crash_reports").iterdir())) == 1
    assert not crash_report.installed()


def test_crash_reports_can_be_turned_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "off")
    monkeypatch.chdir(tmp_path)
    assert not crash_report.install(["scenario"])
    assert crash_report.capture(_caught()) is None
    assert list(tmp_path.iterdir()) == []


def _child_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "SWEEP_RESUME_FROM"}
    env.pop(crash_report.DSN_ENV, None)
    env[crash_report.CRASH_REPORT_ENV] = "1"
    env.update(extra)
    return env


def test_a_native_crash_leaves_the_stack_and_the_command(tmp_path: Path) -> None:
    script = tmp_path / "segfault.py"
    script.write_text(
        textwrap.dedent(
            """
            import faulthandler
            from pathlib import Path

            from autoware_carla_scenario import crash_report

            crash_report.install(["scenario", "scenario=demo"])
            with crash_report.run_context(Path("run"), overrides=["scenario=demo"]):
                faulthandler._sigsegv()
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env=_child_env(),
        check=False,
    )
    assert proc.returncode != 0
    (native,) = (tmp_path / "run" / "crash_reports").iterdir()
    assert "Segmentation fault" in (native / "faulthandler.log").read_text()
    assert (native / "repro.sh").read_text().splitlines()[-1] == (
        'exec scenario scenario=demo "$@"'
    )


#: ``scenario`` through Hydra, with the run of a scenario faked to raise.
_APP = """
import sys

from autoware_carla_scenario.examples import run


def fake_run_scenario(cfg, **_):
    raise RuntimeError("the CARLA client went away")


run.run_scenario = fake_run_scenario
sys.argv = ["scenario", *sys.argv[1:]]
run.main()
"""


def test_a_run_that_raises_leaves_a_report_in_its_directory(tmp_path: Path) -> None:
    script = tmp_path / "app.py"
    script.write_text(textwrap.dedent(_APP), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), "hydra.run.dir=run", "seed=7"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
        env=_child_env(),
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    reports = list((tmp_path / "run" / "crash_reports").iterdir())
    assert len(reports) == 1, proc.stderr
    (report,) = reports
    assert "RuntimeError" in report.name
    assert "the CARLA client went away" in (report / "traceback.txt").read_text()
    assert "seed: 7" in (report / "config.yaml").read_text()
    # The case's own overrides: run again, it writes to a directory of its own.
    assert (report / "repro.sh").read_text().splitlines()[-1] == (
        'exec scenario seed=7 "$@"'
    )
    # A single run's command line is kept too.
    context = json.loads((report / "context.json").read_text())
    assert context["argv"][1:] == ["hydra.run.dir=run", "seed=7"]


# -- the SLURM worker ---------------------------------------------------------


def test_a_slurm_case_is_run_again_without_the_launcher() -> None:
    from autoware_carla_scenario.slurm.worker import repro_overrides

    case = {
        "scenario": "cut_in",
        "overrides": [
            "seed=4",
            "launcher=slurm",
            "launcher.partition=gpu",
            "+launcher.extra=1",
            "server.port=2000",
        ],
    }
    assert repro_overrides(case) == ["scenario=cut_in", "seed=4", "server.port=2000"]


def _old_repro_overrides(case: dict[str, Any]) -> list[str]:
    """slurm.worker.repro_overrides() before #2386, verbatim."""
    return [
        f"scenario={case['scenario']}",
        *(
            str(o)
            for o in case["overrides"]
            if not str(o).lstrip("+~").startswith("launcher")
        ),
    ]


_REPRO_OVERRIDES = (
    "seed=4",
    "launcher=slurm",
    "launcher.partition=gpu",
    "+launcher.extra=1",
    "++launcher.x=2",
    "~launcher",
    "~launcher.partition",
    "+~launcher=odd",
    "launchers=3",
    "launcher",
    "my_launcher=1",
    " launcher=1",
    "hydra/launcher=basic",
    "scenario=other",
    "server.port=2000",
    "",
    "+",
    "~",
    "x='a b'",
)


def test_the_repro_overrides_are_the_same() -> None:
    import random

    from autoware_carla_scenario.crash_report_settings import (
        repro_overrides as settings_repro_overrides,
    )
    from autoware_carla_scenario.slurm.worker import repro_overrides

    rng = random.Random(2386)
    for _ in range(2000):
        overrides: list[Any] = [
            rng.choice(_REPRO_OVERRIDES) for _ in range(rng.randint(0, 6))
        ]
        if rng.random() < 0.2:
            overrides.append(rng.choice([7, 2.5, None]))
        scenario: Any = rng.choice(["cut_in", "", "a b", 3, "cut_in/*"])
        case = {"scenario": scenario, "overrides": overrides}
        want = _old_repro_overrides(case)
        assert repro_overrides(case) == want
        if all(isinstance(o, str) for o in overrides) and isinstance(scenario, str):
            assert settings_repro_overrides(scenario, overrides) == want
    assert settings_repro_overrides("s", ["launcher=x", "seed=1"]) == [
        "scenario=s",
        "seed=1",
    ]


def _slurm_batch(tmp_path: Path) -> Any:
    """A SLURM batch of one case (``cut_in``, seed 4) on disk, never run."""
    from autoware_carla_scenario.slurm.manifest import (
        MANIFEST_VERSION,
        BatchLayout,
        write_json,
    )

    layout = BatchLayout(tmp_path / "batch")
    case = {
        "index": 0,
        "label": "cut_in#1",
        "scenario": "cut_in",
        "overrides": ["seed=4", "launcher=slurm"],
        "stem": "CutIn",
        "scenario_class": "CutIn",
    }
    write_json(
        layout.manifest,
        {"version": MANIFEST_VERSION, "cases": [case], "launcher": {}},
    )
    write_json(layout.submission(0), {"index": 0, "job_id": "77", "chunks": [[0]]})
    return layout


@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The worker module, with what a task does around its cases faked, and a
    crash reporter installed as its main() installs it."""
    import signal

    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.slurm import worker

    monkeypatch.setattr(worker, "CarlaWatchdog", _no_watchdog)
    monkeypatch.setattr(worker, "kill_carla_on_port", lambda _port: [])
    monkeypatch.setattr(run, "load_scenario_plugins", lambda: None)
    monkeypatch.setattr(run, "_cap_malloc_arenas", lambda: None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    command = [
        sys.executable,
        "-m",
        "autoware_carla_scenario.slurm.worker",
        "m",
        "0",
        "0",
    ]
    try:
        with _installed(tmp_path, monkeypatch, command):
            yield worker
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _no_watchdog(*_args: object) -> None:
    raise OSError("no watchdog here")


def _slot() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        rpc=31500,
        streaming=31501,
        secondary=31502,
        to_dict=dict,
        release=lambda **_: None,
    )


def test_a_slurm_task_that_raises_leaves_a_report_in_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_env: Any
) -> None:
    layout = _slurm_batch(tmp_path)

    def acquire(*_args: object, **_kw: object) -> Any:
        raise RuntimeError("no port slot free")

    monkeypatch.setattr(worker_env, "acquire_slot", acquire)
    assert worker_env.run_task(layout.manifest, 0, 0) == 1

    status = layout.read_task(0, 0)
    assert status is not None and status.get("state") == "failed"
    report = Path(status.get("crash_report"))
    assert report.parent == layout.batch_dir / crash_report.CRASH_REPORTS_DIRNAME
    assert "no port slot free" in (report / "traceback.txt").read_text()
    # No case was running: the task again, as the job script runs it.
    assert (
        "autoware_carla_scenario.slurm.worker m 0 0"
        in ((report / "repro.sh").read_text().splitlines()[-1])
    )


def test_a_slurm_case_s_failed_attempt_leaves_a_report_of_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_env: Any
) -> None:
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from autoware_carla_scenario.examples import run

    layout = _slurm_batch(tmp_path)
    queue_logger = logging.getLogger("autoware_carla_scenario.scenario_queue")

    class CutIn:
        pass

    class FakeQueue:
        def __enter__(self) -> FakeQueue:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add(self, *_args: object, **_kw: object) -> None:
            return None

        def run_all(self, **_kw: object) -> list[Any]:
            try:
                _raise("the CARLA client went away")
            except ValueError:
                queue_logger.warning("Scenario attempt 1/1 failed", exc_info=True)
            return [SimpleNamespace(to_dict=lambda: {"passed": False})]

        def errored_indices(self) -> list[int]:
            return [0]

    cfg = OmegaConf.create({"seed": 4, "map": {}})
    monkeypatch.setattr(worker_env, "acquire_slot", lambda *_a, **_k: _slot())
    monkeypatch.setattr(worker_env, "worker_overrides", lambda *_a: {})
    monkeypatch.setattr(worker_env, "_check_same_config", lambda *_a: None)
    monkeypatch.setattr(worker_env, "run_timeout_seconds", lambda _cfg: 60.0)
    monkeypatch.setattr(worker_env, "_apply_metrics", lambda *_a: False)
    monkeypatch.setattr(run, "_compose_config", lambda *_a: cfg)
    monkeypatch.setattr(run, "check_scenario_maps", lambda *_a: None)
    monkeypatch.setattr(run, "build_scenario", lambda *_a, **_k: (None, CutIn()))
    monkeypatch.setattr(run, "build_rerun_converter", lambda *_a: None)
    monkeypatch.setattr(run, "apply_canvas", lambda *_a: None)
    monkeypatch.setattr(run, "check_batch_map", lambda *_a: None)
    monkeypatch.setattr(run, "resolve_map_paths", lambda _m: None)
    monkeypatch.setattr(run, "_batch_queue", lambda *_a, **_k: FakeQueue())
    monkeypatch.setattr(run, "_cooldown_seconds", lambda _cfg: 0.0)
    monkeypatch.setattr(run, "_odd_case", lambda _cfg: None)

    assert worker_env.run_task(layout.manifest, 0, 0) == 0
    status = layout.read_task(0, 0)
    assert status is not None and status.get("state") == "done"
    # The task's native-crash report goes when main() uninstalls the reporter.
    reports = [
        r
        for r in (layout.batch_dir / crash_report.CRASH_REPORTS_DIRNAME).iterdir()
        if not r.name.startswith("native-")
    ]
    assert [r.name.split("-")[1] for r in reports] == ["ValueError"]
    (report,) = reports
    assert "the CARLA client went away" in (report / "traceback.txt").read_text()
    assert "seed: 4" in (report / "config.yaml").read_text()
    # The case alone, here, without the batch's launcher.
    assert (report / "repro.sh").read_text().splitlines()[-1] == (
        'exec scenario scenario=cut_in seed=4 "$@"'
    )


# -- settings -----------------------------------------------------------------


def test_the_default_settings() -> None:
    settings = crash_report.read_settings({})
    assert settings == crash_report.Settings()
    assert settings.items == frozenset(crash_report.ITEMS)
    assert settings.send_items == settings.items
    assert settings.max_file_size == 1024**2
    assert settings.max_reports == 20
    assert settings.log_lines == 5000


def test_the_settings_read_from_the_environment() -> None:
    settings = crash_report.read_settings(
        {
            "SCENARIO_CRASH_REPORT_ITEMS": "all,-environment",
            "SCENARIO_CRASH_REPORT_SEND_ITEMS": "log, config, environment",
            "SCENARIO_CRASH_REPORT_MAX_FILE_SIZE": "512KiB",
            "SCENARIO_CRASH_REPORT_MAX_REPORTS": "0",
            "SCENARIO_CRASH_REPORT_LOG_LINES": "100",
        }
    )
    assert settings.items == frozenset(crash_report.ITEMS) - {"environment"}
    # Only what a report holds can be sent.
    assert settings.send_items == {"log", "config"}
    assert settings.max_file_size == 512 * 1024
    assert settings.max_reports is None
    assert settings.log_lines == 100
    assert crash_report.read_settings(
        {
            "SCENARIO_CRASH_REPORT_ITEMS": "none",
            "SCENARIO_CRASH_REPORT_MAX_FILE_SIZE": "0",
        }
    ) == crash_report.Settings(
        items=frozenset(), send_items=frozenset(), max_file_size=None
    )
    for size, expected in (("2MB", 2_000_000), ("1048576", 1024**2), ("1.5k", 1500)):
        settings = crash_report.read_settings(
            {"SCENARIO_CRASH_REPORT_MAX_FILE_SIZE": size}
        )
        assert settings.max_file_size == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SCENARIO_CRASH_REPORT_ITEMS", "locals,secrets"),
        ("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "a lot"),
        ("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "1 parsec"),
        ("SCENARIO_CRASH_REPORT_MAX_REPORTS", "-1"),
        ("SCENARIO_CRASH_REPORT_LOG_LINES", "0"),
    ],
)
def test_a_setting_that_cannot_be_read_keeps_its_default(
    name: str, value: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert crash_report.read_settings({name: value}) == crash_report.Settings()
    assert name in caplog.text


def test_a_report_holds_only_the_items_chosen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCENARIO_CRASH_REPORT_ITEMS", "none")
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla")
    with _installed(tmp_path, monkeypatch):
        logging.getLogger("demo").warning("the ego left its lane")
        with crash_report.run_context(
            tmp_path / "run", overrides=["seed=3"], config_yaml="seed: 3\n"
        ):
            report = crash_report.capture(_caught())
    assert report is not None
    assert sorted(p.name for p in report.iterdir()) == [
        "context.json",
        "event.json",
        "traceback.txt",
    ]
    event = json.loads((report / "event.json").read_text())
    frame = event["exception"]["values"][0]["stacktrace"]["frames"][-1]
    assert "vars" not in frame
    assert not (event.get("breadcrumbs") or {}).get("values")
    assert "modules" not in event
    assert "environment" not in json.loads((report / "context.json").read_text())


def test_a_long_file_is_cut_keeping_the_end_of_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "2KiB")
    with _installed(tmp_path, monkeypatch):
        for i in range(200):
            logging.getLogger("demo").warning("record %03d", i)
        config = "".join(f"key_{i}: {i}\n" for i in range(500))
        with crash_report.run_context(tmp_path / "run", config_yaml=config):
            report = crash_report.capture(_caught())
    assert report is not None
    log = (report / "log.txt").read_bytes()
    assert len(log) <= 2048
    assert log.startswith(b"[... ") and b"bytes cut" in log
    assert b"record 199" in log and b"record 000" not in log
    config_file = (report / "config.yaml").read_bytes()
    assert len(config_file) <= 2048
    assert config_file.startswith(b"key_0: 0\n")
    assert b"bytes cut" in config_file


def test_a_process_writes_at_most_max_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SCENARIO_CRASH_REPORT_MAX_REPORTS", "2")
    with _installed(tmp_path, monkeypatch):
        reports = [crash_report.capture(_caught(f"boom {i}")) for i in range(4)]
    assert [r is not None for r in reports] == [True, True, False, False]
    assert caplog.text.count("no more from this process") == 1


def test_only_the_send_items_are_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentry_sdk
    from sentry_sdk.transport import Transport

    sent: list[Any] = []

    class Recorder(Transport):
        def capture_envelope(self, envelope: Any) -> None:
            sent.append(envelope)

    monkeypatch.setenv("SCENARIO_CRASH_REPORT_SEND_ITEMS", "log,repro")
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla")
    with _installed(tmp_path, monkeypatch):
        sentry_sdk.get_client().transport.upstream = Recorder()  # type: ignore[union-attr]
        logging.getLogger("demo").warning("the ego left its lane")
        with crash_report.run_context(
            tmp_path / "run", overrides=["seed=3"], config_yaml="seed: 3\n"
        ):
            report = crash_report.capture(_caught())
    assert report is not None
    # Kept whole on disk.
    assert (report / "config.yaml").exists()
    local = json.loads((report / "event.json").read_text())
    assert "vars" in local["exception"]["values"][0]["stacktrace"]["frames"][-1]
    (envelope,) = sent
    event = envelope.get_event()
    assert "vars" not in event["exception"]["values"][0]["stacktrace"]["frames"][-1]
    assert "modules" not in event
    assert event["breadcrumbs"]["values"]
    files = {
        item.headers["filename"]: item.get_bytes()
        for item in envelope.items
        if item.type == "attachment"
    }
    assert sorted(files) == ["context.json", "log.txt", "repro.sh", "traceback.txt"]
    assert "environment" not in json.loads(files["context.json"])
    assert b"export " not in files["repro.sh"]
    assert b"exec scenario seed=3" in files["repro.sh"]


# -- sending ------------------------------------------------------------------


def test_nothing_is_sent_without_the_crash_report_dsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentry_sdk

    # Set for another project, or for the SDK's local Spotlight sidecar.
    monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.example.org/1")
    monkeypatch.setenv("SENTRY_SPOTLIGHT", "1")

    def no_http(*_args: object, **_kw: object) -> None:
        raise AssertionError("an HTTP transport was made")

    monkeypatch.setattr(crash_report, "HttpTransport", no_http)
    with _installed(tmp_path, monkeypatch):
        client = sentry_sdk.get_client()
        assert not client.options["dsn"]
        assert client.spotlight is None
        assert client.transport.upstream is None  # type: ignore[union-attr]
        assert crash_report.capture(_caught()) is not None


def test_reports_are_sent_with_the_crash_report_dsn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sentry_sdk
    from sentry_sdk.transport import HttpTransport

    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    monkeypatch.setenv(crash_report.DSN_ENV, "https://key@glitchtip.example.org/1")
    monkeypatch.chdir(tmp_path)
    assert crash_report.install(["scenario"])
    try:
        client = sentry_sdk.get_client()
        assert client.options["dsn"] == "https://key@glitchtip.example.org/1"
        upstream = client.transport.upstream  # type: ignore[union-attr]
        assert isinstance(upstream, HttpTransport)
    finally:
        crash_report.uninstall()


def test_crash_report_re_exports_the_checked_settings() -> None:
    # #2354: the settings, their parsers and the size cut are the checked
    # crash_report_settings's; crash_report's names are the same objects, and
    # no test patches them there.
    from autoware_carla_scenario import crash_report_settings

    for name in (
        "CRASH_REPORT_ENV",
        "CRASH_REPORT_DIR_ENV",
        "DSN_ENV",
        "ITEMS",
        "ITEMS_ENV",
        "SEND_ITEMS_ENV",
        "MAX_FILE_SIZE_ENV",
        "MAX_REPORTS_ENV",
        "LOG_LINES_ENV",
        "Settings",
        "_ENV_PATTERNS",
        "_SIZE_UNITS",
        "_parse_items",
        "_parse_size",
        "_parse_count",
        "_truncate",
        "_safe_name",
    ):
        assert getattr(crash_report, name) is getattr(crash_report_settings, name), name
    # Zero is no limit, as `or None` read it.
    assert crash_report_settings._parse_size("N", "0") is None
    assert crash_report_settings._parse_size("N", "0.0001k") is None
    assert crash_report_settings._parse_size("N", "1k") == 1000
    assert crash_report_settings._parse_count("N", "0", zero_is_none=True) is None
    assert crash_report_settings._parse_count("N", "3", zero_is_none=True) == 3


# ---------------------------------------------------------------------------
# The repro command and script, moved into the checked crash_report_settings
# (#2364)
# ---------------------------------------------------------------------------
#
# The ``_old_*`` functions are crash_report._Reporter.repro_command() and
# repro_script() before #2364, verbatim but for what they read off the
# reporter (argv, the context's overrides, cwd, the kept environment), which
# they now take.


def _old_repro_command(argv: list[str], overrides: list[str] | None) -> list[str]:
    if overrides is None:
        return list(argv)
    return ["scenario", *overrides]


def _old_repro_script(
    argv: list[str], cwd: Path, env: dict[str, str], overrides: list[str] | None
) -> bytes:
    import shlex  # noqa: PLC0415

    lines = [
        "#!/bin/sh",
        "# Runs the case this crash report comes from again.",
        # Each line of it a comment line since #2380: an argument holding a
        # newline ended the comment.
        "# Command line: " + shlex.join(argv).replace("\n", "\n# "),
        f"cd {shlex.quote(str(cwd))} || exit 1",
    ]
    lines += [f"export {name}={shlex.quote(value)}" for name, value in env.items()]
    lines.append(f'exec {shlex.join(_old_repro_command(argv, overrides))} "$@"')
    return ("\n".join(lines) + "\n").encode()


_WORDS = (
    "scenario",
    "scenario=demo",
    "seed=3",
    "",
    "it's",
    'say "hi"',
    "a b",
    "$HOME",
    "x;rm -rf y",
    "été",
    "tab\there",
    "new\nline",
    "~/dir",
    "+key=1",
)


def test_the_repro_command_and_script_are_as_before(tmp_path: Path) -> None:
    import random  # noqa: PLC0415

    from autoware_carla_scenario import crash_report_settings  # noqa: PLC0415

    rng = random.Random(2364)
    for _ in range(3000):
        argv = [rng.choice(_WORDS) for _ in range(rng.randrange(0, 5))]
        overrides = (
            [rng.choice(_WORDS) for _ in range(rng.randrange(0, 4))]
            if rng.random() < 0.6
            else None
        )
        cwd = tmp_path / rng.choice(("run", "a b", "it's", "é"))
        env = {f"SCENARIO_{i}": rng.choice(_WORDS) for i in range(rng.randrange(0, 3))}
        command = crash_report_settings.repro_command(argv, overrides)
        assert command == _old_repro_command(argv, overrides)
        assert crash_report_settings.repro_script(
            argv, str(cwd), env, command
        ) == _old_repro_script(argv, cwd, env, overrides)


@pytest.mark.parametrize("items", [crash_report.ITEMS, ("repro",)])
@pytest.mark.parametrize("overrides", [None, ["scenario=demo", "seed=it's"]])
def test_the_reporter_s_repro_script_is_as_before(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    items: tuple[str, ...],
    overrides: list[str] | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    env = {"SCENARIO_X": "a b", "CARLA_PORT": "2000"}
    monkeypatch.setattr(crash_report, "_kept_environment", lambda: dict(env))
    reporter = crash_report._Reporter(
        ["python", "-m", "x y"],
        crash_report.Settings(items=frozenset(items)),
    )
    reporter.context.overrides = overrides
    kept = env if "environment" in items else {}
    assert reporter.repro_command() == _old_repro_command(reporter.argv, overrides)
    assert reporter.repro_script() == _old_repro_script(
        reporter.argv, reporter.cwd, kept, overrides
    )


# ---------------------------------------------------------------------------
# #2380: the CLI's logging, a local batch's case, truncation, newlines
# ---------------------------------------------------------------------------


@contextmanager
def _fresh_root_logger() -> Iterator[logging.Logger]:
    """The root logger as a new process has it: no handler, at ``WARNING``
    (pytest's own handlers, which it adds once the test's fixtures are set
    up, taken off for the block, and put back)."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    for handler in handlers:
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)


def test_the_cli_logs_info_with_crash_reports_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from autoware_carla_scenario.examples import run

    def fake_main() -> None:
        # What a batch logs: its plan, its progress, its warnings.
        logging.getLogger("autoware_carla_scenario.examples.run").info(
            "Glob matched 2 scenario(s)"
        )
        logging.getLogger("autoware_carla_scenario.examples.run").warning(
            "hydra.sweep.dir=x is ignored"
        )
        raise RuntimeError("the CARLA client went away")

    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    for name in (crash_report.CRASH_REPORT_DIR_ENV, crash_report.DSN_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run, "_main", fake_main)
    monkeypatch.setattr(sys, "argv", ["scenario", "scenario=cut_in/*"])
    with _fresh_root_logger(), pytest.raises(RuntimeError):
        run.main()
    assert not crash_report.installed()
    err = capfd.readouterr().err
    # The console, as with crash reports off.
    assert "INFO autoware_carla_scenario.examples.run: Glob matched 2" in err, err
    assert "WARNING autoware_carla_scenario.examples.run: hydra.sweep.dir" in err
    # log.txt holds the INFO records too.
    (report,) = (tmp_path / "outputs" / crash_report.CRASH_REPORTS_DIRNAME).iterdir()
    log = (report / "log.txt").read_text()
    assert "INFO autoware_carla_scenario.examples.run: Glob matched 2" in log
    assert "hydra.sweep.dir=x is ignored" in log


def test_the_log_buffer_survives_a_dict_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging.config

    def hydra_job_logging() -> None:
        # What Hydra does at the start of each job: the root logger's
        # handlers are replaced.
        logging.config.dictConfig(
            {
                "version": 1,
                "handlers": {"null": {"class": "logging.NullHandler"}},
                "root": {"level": "INFO", "handlers": ["null"]},
                "disable_existing_loggers": False,
            }
        )

    def buffers() -> int:
        return sum(
            isinstance(h, crash_report._LogBuffer) for h in logging.getLogger().handlers
        )

    with _fresh_root_logger(), _installed(tmp_path, monkeypatch):
        logging.getLogger().setLevel(logging.INFO)
        assert buffers() == 1
        crash_report.set_output_dir(tmp_path / "batch")
        for case in range(2):
            hydra_job_logging()
            assert buffers() == 0
            with crash_report.run_context(tmp_path / f"run{case}"):
                assert buffers() == 1
                # Entered again, never added twice.
                with crash_report.run_context(tmp_path / f"run{case}"):
                    assert buffers() == 1
                logging.getLogger("demo").info("in case %d", case)
                report = crash_report.capture(_caught())
            assert report is not None
            assert f"INFO demo: in case {case}" in (report / "log.txt").read_text()
        # The batch's context, re-armed, takes it back after a job's.
        hydra_job_logging()
        with crash_report.run_context(tmp_path / "run2"):
            hydra_job_logging()
        assert buffers() == 1
        crash_report.uninstall()
        assert buffers() == 0


def test_a_local_batch_s_report_runs_the_case_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.examples import run
    from autoware_carla_scenario.scenario_runner import ScenarioResult

    monkeypatch.setenv("AUTOWARE_CARLA_SCENARIO_TYPECHECK", "off")
    queue_logger = logging.getLogger("autoware_carla_scenario.scenario_queue")

    class FakeQueue:
        def __init__(self) -> None:
            self.scenarios: list[Any] = []

        def __enter__(self) -> FakeQueue:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add(self, scenario: Any, **_kw: object) -> None:
            self.scenarios.append(scenario)

        def run_all(self, *, on_result: Any, **_kw: object) -> list[Any]:
            results = []
            for i, _scenario in enumerate(self.scenarios):
                if i == 1:
                    try:
                        _raise("the CARLA client went away")
                    except ValueError:
                        queue_logger.warning("Scenario attempt failed", exc_info=True)
                results.append(
                    ScenarioResult(passed=i != 1, message="", elapsed_seconds=0.0)
                )
                on_result(i, results[-1])
            return results

        def errored_indices(self) -> list[int]:
            return []

    monkeypatch.setattr(run, "build_scenario", lambda cfg, **_: (None, object()))
    monkeypatch.setattr(run, "resolve_map_paths", lambda _m: None)
    monkeypatch.setattr(run, "_batch_queue", lambda *_a, **_k: FakeQueue())
    with (
        _installed(tmp_path, monkeypatch, ["scenario", "scenario=*/left", "seed=5"]),
        pytest.raises(SystemExit) as raised,
    ):
        run.run_batch(
            ["lane_change/left", "cut_in/left"], ["seed=5"], output_dir=Path("b")
        )
    assert raised.value.code == 1
    reports = [
        r
        for r in (tmp_path / "b" / crash_report.CRASH_REPORTS_DIRNAME).iterdir()
        if not r.name.startswith("native-")
    ]
    assert len(reports) == 1
    (report,) = reports
    # The failed case alone, not the whole batch's command line.
    assert (report / "repro.sh").read_text().splitlines()[-1] == (
        'exec scenario scenario=cut_in/left seed=5 "$@"'
    )
    assert "seed: 5" in (report / "config.yaml").read_text()
    context = json.loads((report / "context.json").read_text())
    assert context["overrides"] == ["scenario=cut_in/left", "seed=5"]
    assert context["output_dir"] == str((tmp_path / "b").resolve())


def _sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool = False
) -> tuple[Path | None, list[Any]]:
    """A report captured with a DSN's transport recording (or failing on)
    what it sends, the environment not sent."""
    import sentry_sdk
    from sentry_sdk.transport import Transport

    sent: list[Any] = []

    class Recorder(Transport):
        def capture_envelope(self, envelope: Any) -> None:
            if fail:
                raise OSError("the server went away")
            sent.append(envelope)

    monkeypatch.setenv("SCENARIO_CRASH_REPORT_SEND_ITEMS", "all,-environment")
    with _installed(tmp_path, monkeypatch, ["scenario", "scenario.note=a\nb"]):
        sentry_sdk.get_client().transport.upstream = Recorder()  # type: ignore[union-attr]
        with crash_report.run_context(
            tmp_path / "run", overrides=["scenario.note=a\nb"], config_yaml="x: 1\n"
        ):
            report = crash_report.capture(_caught())
    return report, sent


def test_a_report_whose_context_exceeds_the_size_limit_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "300")
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla/" + "x" * 400)
    report, sent = _sending(tmp_path, monkeypatch)
    assert report is not None
    # Never cut: still JSON, and still a script that runs the case.
    context = json.loads((report / "context.json").read_text())
    assert context["environment"]["CARLA_ROOT"].endswith("x" * 400)
    script = (report / "repro.sh").read_text()
    assert len(script) > 300
    assert "\nexec scenario " in script and script.endswith('"$@"\n')
    (envelope,) = sent
    files = {
        item.headers["filename"]: item.get_bytes()
        for item in envelope.items
        if item.type == "attachment"
    }
    assert "environment" not in json.loads(files["context.json"])
    assert b"CARLA_ROOT" not in files["repro.sh"]


def test_a_send_that_fails_keeps_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, sent = _sending(tmp_path, monkeypatch, fail=True)
    assert report is not None and (report / "event.json").exists()
    assert not sent


def test_a_repro_script_with_a_newline_in_an_argument_is_a_script(
    tmp_path: Path,
) -> None:
    import shutil

    from autoware_carla_scenario import crash_report_settings

    argv = ["scenario", "scenario.note=a\nb", "x=it's"]
    command = crash_report_settings.repro_command(argv, None)
    script = tmp_path / "repro.sh"
    script.write_bytes(
        crash_report_settings.repro_script(
            argv, str(tmp_path), {"SCENARIO_NOTE": "one\ntwo"}, command
        )
    )
    sh = shutil.which("sh")
    assert sh is not None
    check = subprocess.run(
        [sh, "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert check.returncode == 0, check.stderr
    # Run with a `scenario` that prints what it was given.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "scenario"
    fake.write_text('#!/bin/sh\nprintf "%s|" "$SCENARIO_NOTE" "$@"\n')
    fake.chmod(0o755)
    proc = subprocess.run(
        [sh, str(script), "extra"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "one\ntwo|scenario.note=a\nb|x=it's|extra|"


def test_a_sent_repro_script_without_the_environment_is_written_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario import crash_report_settings

    monkeypatch.setenv("SCENARIO_NOTE", "first line\nsecond line")
    report, sent = _sending(tmp_path, monkeypatch)
    assert report is not None
    assert "second line" in (report / "repro.sh").read_text()
    (envelope,) = sent
    files = {
        item.headers["filename"]: item.get_bytes()
        for item in envelope.items
        if item.type == "attachment"
    }
    # Not a line left of the value that spans two.
    assert b"second line" not in files["repro.sh"]
    assert files["repro.sh"] == crash_report_settings.repro_script(
        ["scenario", "scenario.note=a\nb"],
        str(tmp_path),
        {},
        ["scenario", "scenario.note=a\nb"],
    )
