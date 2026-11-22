"""Crash reports (crash_report.py): what a run that raised or crashed leaves."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from autoware_carla_scenario import crash_report


@pytest.fixture
def reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A crash reporter installed in *tmp_path*, the working directory."""
    monkeypatch.setenv(crash_report.CRASH_REPORT_ENV, "1")
    monkeypatch.delenv(crash_report.CRASH_REPORT_DIR_ENV, raising=False)
    monkeypatch.delenv(crash_report.DSN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert crash_report.install(["scenario", "scenario=demo", "seed=3"])
    try:
        yield tmp_path
    finally:
        crash_report.uninstall()


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
