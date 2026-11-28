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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARLA_SERVER_TOKEN", "hunter2")
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla")
    with _installed(tmp_path, monkeypatch):
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


@pytest.mark.parametrize(
    ("launcher", "repro"),
    [
        (None, 'exec scenario scenario=cut_in seed=4 "$@"'),
        # #2616: the composed config selects SLURM, which a single run
        # refuses (a scenario's config may set it): run here.
        ("slurm", 'exec scenario scenario=cut_in seed=4 launcher=local "$@"'),
    ],
)
def test_a_slurm_case_s_failed_attempt_leaves_a_report_of_the_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worker_env: Any,
    launcher: str | None,
    repro: str,
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
    if launcher is not None:
        cfg.launcher = {"type": launcher}
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
    assert (report / "repro.sh").read_text().splitlines()[-1] == repro


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
        # #2616: under a byte is not "no limit", which only 0 is.
        ("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "0.5"),
        ("SCENARIO_CRASH_REPORT_MAX_FILE_SIZE", "0.0001KB"),
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
    # Zero is no limit, as `or None` read it; but only zero: a size under a
    # byte, which `or None` read as no limit too, is refused (#2616).
    assert crash_report_settings._parse_size("N", "0") is None
    assert crash_report_settings._parse_size("N", "0.0k") is None
    with pytest.raises(ValueError, match="less than a byte"):
        crash_report_settings._parse_size("N", "0.0001k")
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
    assert reporter.context_files()["repro.sh"] == _old_repro_script(
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


# ---------------------------------------------------------------------------
# #2460: credentials, the environment repro.sh exports, a batch's directory,
# the environment pin and the map's commit
# ---------------------------------------------------------------------------

_TOKEN = "glpat-SECRET123"
_SOURCE = f"https://oauth2:{_TOKEN}@gitlab.example.org/o/r"


def _recorded_sends(sent: list[Any]) -> Any:
    from sentry_sdk.transport import Transport

    class Recorder(Transport):
        def capture_envelope(self, envelope: Any) -> None:
            sent.append(envelope)

    return Recorder()


def _raise_with(source: str) -> None:
    map_source = source  # noqa: F841 - the report keeps it, redacted
    raise ValueError(f"cannot read {source}")


@pytest.mark.parametrize("send_items", ["none", "all"])
def test_map_credentials_are_redacted_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, send_items: str
) -> None:
    import sentry_sdk

    sent: list[Any] = []
    monkeypatch.setenv("SCENARIO_CRASH_REPORT_SEND_ITEMS", send_items)
    monkeypatch.setenv("SCENARIO_MAP", _SOURCE)
    argv = ["scenario", "scenario=demo", f"map.source={_SOURCE}"]
    with _installed(tmp_path, monkeypatch, argv):
        sentry_sdk.get_client().transport.upstream = _recorded_sends(sent)  # type: ignore[union-attr]
        logging.getLogger("demo").warning("cloning %s", _SOURCE)
        with crash_report.run_context(
            tmp_path / "run",
            overrides=argv[1:],
            config_yaml=f"map:\n  source: {_SOURCE}\n",
        ):
            try:
                _raise_with(_SOURCE)
            except ValueError as exc:
                report = crash_report.capture(exc)
    assert report is not None
    for path in report.iterdir():
        assert _TOKEN not in path.read_text(), path.name
    context = json.loads((report / "context.json").read_text())
    redacted = "https://oauth2:***@gitlab.example.org/o/r"
    assert context["argv"][-1] == f"map.source={redacted}"
    assert context["overrides"][-1] == f"map.source={redacted}"
    assert context["repro"][-1] == f"map.source={redacted}"
    assert context["environment"]["SCENARIO_MAP"] == redacted
    assert context["repro_redacted"] is True
    assert redacted in (report / "config.yaml").read_text()
    script = (report / "repro.sh").read_text()
    assert "Credentials in its URLs are redacted (***)" in script
    event = json.loads((report / "event.json").read_text())
    assert event["contexts"]["run"]["overrides"][-1] == f"map.source={redacted}"
    (envelope,) = sent
    assert _TOKEN not in json.dumps(envelope.get_event(), default=str)
    for item in envelope.items:
        assert _TOKEN.encode() not in item.get_bytes(), item.headers.get("filename")
    names = {item.headers.get("filename") for item in envelope.items}
    if send_items == "none":
        assert "repro.sh" not in names and "config.yaml" not in names


def test_a_report_without_credentials_says_nothing_of_them(reporter: Path) -> None:
    report = crash_report.capture(_caught())
    assert report is not None
    assert json.loads((report / "context.json").read_text())["repro_redacted"] is False
    assert "redacted" not in (report / "repro.sh").read_text()


def test_the_framework_s_own_variables_are_not_exported() -> None:
    from autoware_carla_scenario import crash_report_settings
    from autoware_carla_scenario.heavy_metrics import HEAVY_METRICS_DECIDED_ENV
    from autoware_carla_scenario.sweeper.isolation import SWEEPER_OWNS_RETRIES_ENV
    from autoware_carla_scenario.typecheck.mode import TYPECHECK_ENV

    assert set(crash_report_settings._FRAMEWORK_ENV_NAMES) == {
        HEAVY_METRICS_DECIDED_ENV,
        SWEEPER_OWNS_RETRIES_ENV,
    }
    assert TYPECHECK_ENV in crash_report_settings._SLURM_TASK_ENV_NAMES
    env = {
        "CARLA_ROOT": "/opt/carla",
        HEAVY_METRICS_DECIDED_ENV: "1",
        SWEEPER_OWNS_RETRIES_ENV: "1",
        TYPECHECK_ENV: "off",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    # The user's static check setting and GPUs, outside SLURM.
    assert crash_report_settings.repro_environment(env) == {
        "CARLA_ROOT": "/opt/carla",
        TYPECHECK_ENV: "off",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    # A SLURM task's: its worker's and SLURM's.
    assert crash_report_settings.repro_environment(
        {**env, "SLURM_JOB_ID": "7", "SLURM_PROCID": "0"}
    ) == {"CARLA_ROOT": "/opt/carla"}


def test_repro_sh_exports_the_environment_as_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario.sweeper.isolation import SWEEPER_OWNS_RETRIES_ENV
    from autoware_carla_scenario.typecheck.mode import TYPECHECK_ENV

    monkeypatch.delenv(SWEEPER_OWNS_RETRIES_ENV, raising=False)
    monkeypatch.delenv(TYPECHECK_ENV, raising=False)
    monkeypatch.setenv("CARLA_ROOT", "/opt/carla")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with _installed(tmp_path, monkeypatch):
        # Set by the framework once the reporter is installed: a forked
        # sweeper case, a SLURM task's worker.
        monkeypatch.setenv(SWEEPER_OWNS_RETRIES_ENV, "1")
        monkeypatch.setenv(TYPECHECK_ENV, "off")
        report = crash_report.capture(_caught())
    assert report is not None
    environment = json.loads((report / "context.json").read_text())["environment"]
    assert environment["SLURM_JOB_ID"] == "7"
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert SWEEPER_OWNS_RETRIES_ENV not in environment
    assert TYPECHECK_ENV not in environment
    exports = [
        line
        for line in (report / "repro.sh").read_text().splitlines()
        if line.startswith("export ")
    ]
    assert "export CARLA_ROOT=/opt/carla" in exports
    assert not any(line.startswith("export SLURM_") for line in exports)
    for name in ("CUDA_VISIBLE_DEVICES", SWEEPER_OWNS_RETRIES_ENV, TYPECHECK_ENV):
        assert not any(line.startswith(f"export {name}=") for line in exports), name


def test_a_batch_s_repro_runs_into_a_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "scenario",
        "scenario=cut_in/*",
        "hydra.run.dir=outputs/nightly",
        "+hydra.sweep.dir=outputs/sweep",
        "seed=3",
    ]
    with _installed(tmp_path, monkeypatch, argv):
        report = crash_report.capture(_caught())
    assert report is not None
    script = (report / "repro.sh").read_text()
    assert script.splitlines()[-1] == "exec scenario 'scenario=cut_in/*' seed=3 \"$@\""
    # The command line as it was, in the comment.
    assert "hydra.run.dir=outputs/nightly" in script.splitlines()[2]


def test_a_report_names_the_environment_pin(
    reporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario import environment

    class Env:
        id = "f" * 64
        by_name = {
            environment.FRAMEWORK: {
                "name": environment.FRAMEWORK,
                "version": "1.0",
                "source": {
                    "kind": "editable",
                    "path": "/src",
                    "commit": "c" * 40,
                    "dirty": True,
                },
            }
        }

        def scenario_package(self, scenario: Any) -> Any:
            return None

    monkeypatch.setattr(environment, "capture", lambda: Env())
    report = crash_report.capture(_caught())
    assert report is not None
    pin = json.loads((report / "context.json").read_text())["environment_pin"]
    assert pin == {
        "id": "f" * 64,
        "framework": {
            "name": environment.FRAMEWORK,
            "version": "1.0",
            "kind": "editable",
            "commit": "c" * 40,
            "dirty": True,
        },
        "scenario": None,
    }


def test_a_pin_that_cannot_be_read_leaves_the_report(
    reporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autoware_carla_scenario import environment

    def broken() -> Any:
        raise RuntimeError("no metadata")

    monkeypatch.setattr(environment, "capture", broken)
    report = crash_report.capture(_caught())
    assert report is not None
    assert json.loads((report / "context.json").read_text())["environment_pin"] is None


def test_the_noted_scenario_is_not_kept_alive(
    reporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gc
    import weakref

    from autoware_carla_scenario import environment

    class Scenario:
        pass

    seen: list[str] = []

    class Env:
        id = "f" * 64
        by_name: dict[str, Any] = {}

        def scenario_package(self, scenario: Any) -> Any:
            seen.append(type(scenario).__module__)
            return {"name": "pkg", "version": "1"}

    monkeypatch.setattr(environment, "capture", lambda: Env())
    scenario = Scenario()
    alive = weakref.ref(scenario)
    crash_report.note_run(scenario=scenario)
    del scenario
    gc.collect()
    assert alive() is None
    report = crash_report.capture(_caught())
    assert report is not None
    pin = json.loads((report / "context.json").read_text())["environment_pin"]
    assert pin["scenario"] == {"name": "pkg", "version": "1"}
    assert seen and set(seen) == {Scenario.__module__}


def _map_paths(uri: str, commit: str) -> Any:
    from types import SimpleNamespace

    from autoware_carla_scenario.maps.source import MapSource

    return SimpleNamespace(
        resolved=SimpleNamespace(source=MapSource.parse(uri), commit=commit)
    )


def test_a_case_s_repro_reads_the_map_at_the_commit_it_resolved_to(
    reporter: Path,
) -> None:
    commit = "a" * 40
    uri = f"git+{_SOURCE}@main#maps/T"
    run_dir = reporter / "run"
    # Noted before the batch's directory and its case's context: kept.
    crash_report.note_run(map_paths=_map_paths(uri, commit))
    crash_report.set_output_dir(reporter / "batch")
    with crash_report.run_context(
        run_dir, overrides=["scenario=demo", f"map.source={uri}"]
    ):
        report = crash_report.capture(_caught())
    assert report is not None
    context = json.loads((report / "context.json").read_text())
    redacted = "git+https://oauth2:***@gitlab.example.org/o/r"
    assert context["map"] == {
        "source": f"{redacted}@main#maps/T",
        "commit": commit,
        "pinned": True,
    }
    lines = (report / "repro.sh").read_text().splitlines()
    import shlex

    pinned = f"map.source='{redacted}@{commit}#maps/T'"
    assert lines[-1] == f'exec {shlex.join(["scenario", "scenario=demo", pinned])} "$@"'
    assert shlex.split(lines[-1])[3] == pinned
    assert any(f"pinned to commit {commit}" in line for line in lines)
    assert _TOKEN not in "\n".join(lines)


def test_a_pinned_map_source_is_named_with_its_commit(reporter: Path) -> None:
    commit = "b" * 40
    uri = f"git+https://huggingface.co/datasets/o/r@{commit}#T"
    with crash_report.run_context(reporter / "run", overrides=["scenario=demo"]):
        crash_report.note_run(map_paths=_map_paths(uri, commit))
        report = crash_report.capture(_caught())
    assert report is not None
    context = json.loads((report / "context.json").read_text())
    assert context["map"] == {"source": uri, "commit": commit, "pinned": False}
    script = (report / "repro.sh").read_text()
    assert script.splitlines()[-1] == 'exec scenario scenario=demo "$@"'
    assert f"# The map was read from {uri} at commit {commit}." in script
    # Outside the case, the process's context names no map.
    other = crash_report.capture(_caught())
    assert other is not None
    assert json.loads((other / "context.json").read_text())["map"] is None


def test_pin_map_source_replaces_the_case_s_map_source() -> None:
    from autoware_carla_scenario.crash_report_settings import pin_map_source

    assert pin_map_source(["scenario", "seed=1"], "u@c#p") == [
        "scenario",
        "seed=1",
        "map.source='u@c#p'",
    ]
    assert pin_map_source(
        ["scenario", "++map.source=x", "seed=1", "map.source=y"], "u"
    ) == ["scenario", "map.source='u'", "seed=1"]
    assert pin_map_source(["scenario"], "it's") == ["scenario", 'map.source="it\'s"']


def test_the_native_report_names_the_map_once_noted(reporter: Path) -> None:
    commit = "c" * 40
    uri = "git+https://huggingface.co/datasets/o/r@main#T"
    run_dir = reporter / "run"
    with crash_report.run_context(run_dir, overrides=["scenario=demo"]):
        (native,) = (run_dir / "crash_reports").iterdir()
        assert json.loads((native / "context.json").read_text())["map"] is None
        crash_report.note_run(map_paths=_map_paths(uri, commit))
        assert (
            json.loads((native / "context.json").read_text())["map"]["commit"] == commit
        )
        assert f"@{commit}#T" in (native / "repro.sh").read_text()


# ---------------------------------------------------------------------------
# #2616
# ---------------------------------------------------------------------------


def _raise_with_secrets() -> None:
    password = "hunter2-password"  # noqa: F841, S105 - the report filters it
    token = "hunter2-token"  # noqa: F841, S105
    secret = "hunter2-secret"  # noqa: F841, S105
    api_key = "hunter2-api-key"  # noqa: F841
    plain = 7  # noqa: F841 - the report keeps it
    raise ValueError("boom")


class _NoDataCollection:
    """A ClientConstructor of a sentry-sdk that takes no ``data_collection``
    (before 2.70): the ``include_local_variables`` path."""

    def __init__(self, include_local_variables: bool = True) -> None:
        pass


@pytest.mark.parametrize("path", ["data_collection", "include_local_variables"])
def test_a_local_named_like_a_secret_is_filtered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    if path == "include_local_variables":
        monkeypatch.setattr(crash_report, "ClientConstructor", _NoDataCollection)
    if path not in crash_report._locals_option(True):
        pytest.skip(f"the installed sentry-sdk takes no {path}")
    with _installed(tmp_path, monkeypatch):
        try:
            _raise_with_secrets()
        except ValueError as exc:
            report = crash_report.capture(exc)
    assert report is not None
    event = json.loads((report / "event.json").read_text())
    frames = event["exception"]["values"][0]["stacktrace"]["frames"]
    # In no frame's variables (the source lines around a frame show them).
    assert "hunter2" not in json.dumps([f.get("vars") for f in frames])
    frame = frames[-1]
    assert frame["function"] == "_raise_with_secrets"
    for name in ("password", "token", "secret", "api_key"):
        assert "Filtered" in frame["vars"][name]
    assert frame["vars"]["plain"] == "7"


@pytest.mark.parametrize("path", ["data_collection", "include_local_variables"])
def test_without_locals_a_frame_has_no_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    if path == "include_local_variables":
        monkeypatch.setattr(crash_report, "ClientConstructor", _NoDataCollection)
    monkeypatch.setenv(crash_report.ITEMS_ENV, "-locals")
    with _installed(tmp_path, monkeypatch):
        try:
            _raise_with_secrets()
        except ValueError as exc:
            report = crash_report.capture(exc)
    assert report is not None
    event = json.loads((report / "event.json").read_text())
    for frame in event["exception"]["values"][0]["stacktrace"]["frames"]:
        assert "vars" not in frame
    # -locals is all but locals.
    assert (report / "repro.sh").exists()


def test_a_removal_first_starts_from_all() -> None:
    every = frozenset(crash_report.ITEMS)
    for text, want in (
        ("-locals", every - {"locals"}),
        ("-locals,-log", every - {"locals", "log"}),
        ("log,-locals", frozenset({"log"})),
        ("none,-locals", frozenset()),
    ):
        settings = crash_report.read_settings({crash_report.ITEMS_ENV: text})
        assert settings.items == want, text


def test_a_normal_end_keeps_the_crash_report_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mine = tmp_path / "mine"
    mine.mkdir()
    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    monkeypatch.delenv(crash_report.DSN_ENV, raising=False)
    monkeypatch.setenv(crash_report.CRASH_REPORT_DIR_ENV, str(mine))
    monkeypatch.chdir(tmp_path)
    assert crash_report.install(["scenario"])
    try:
        (native,) = mine.iterdir()
        assert native.name.startswith("native-")
    finally:
        crash_report.uninstall()
    assert mine.is_dir() and list(mine.iterdir()) == []
    # One the reporter made is removed with its report.
    made = tmp_path / "made"
    monkeypatch.setenv(crash_report.CRASH_REPORT_DIR_ENV, str(made))
    assert crash_report.install(["scenario"])
    crash_report.uninstall()
    assert not made.exists()


def test_the_native_report_s_files_are_written_in_one_rename(
    reporter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []

    def atomically(path: Path, data: bytes) -> None:
        written.append(path.name)
        path.write_bytes(data)

    monkeypatch.setattr(crash_report, "write_bytes_atomically", atomically)
    run_dir = reporter / "run"
    with crash_report.run_context(
        run_dir, overrides=["scenario=demo"], config_yaml="seed: 3\n"
    ):
        assert sorted(written) == ["config.yaml", "context.json", "repro.sh"]
        (native,) = (run_dir / "crash_reports").iterdir()
        assert os.access(native / "repro.sh", os.X_OK)


def test_a_case_whose_config_selects_slurm_is_run_again_here() -> None:
    from omegaconf import OmegaConf

    from autoware_carla_scenario.crash_report_settings import (
        repro_overrides as settings_repro_overrides,
    )
    from autoware_carla_scenario.slurm.worker import repro_overrides

    case = {"scenario": "cut_in", "overrides": ["seed=4"]}
    slurm = OmegaConf.create({"seed": 4, "launcher": {"type": "slurm"}})
    local = OmegaConf.create({"seed": 4, "launcher": {"type": "local"}})
    assert repro_overrides(case, slurm) == [
        "scenario=cut_in",
        "seed=4",
        "launcher=local",
    ]
    assert repro_overrides(case, local) == ["scenario=cut_in", "seed=4"]
    assert repro_overrides(case, OmegaConf.create({"seed": 4})) == [
        "scenario=cut_in",
        "seed=4",
    ]
    assert settings_repro_overrides("s", ["launcher=slurm"], "slurm") == [
        "scenario=s",
        "launcher=local",
    ]
