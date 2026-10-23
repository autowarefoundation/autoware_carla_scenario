"""``scenario-check``: what a command line that does not compose prints (#532)."""

from __future__ import annotations

import pytest

from autoware_carla_scenario.typecheck.cli import main


def test_a_misspelled_override_key_is_named_with_a_suggestion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["scenario=intersection_passing/straight", "scenario.timout_seconds=3"])
    err = capsys.readouterr().err
    assert code == 2
    assert "could not override 'scenario.timout_seconds'" in err
    assert "did you mean 'scenario.timeout_seconds'?" in err
    assert "+scenario.timout_seconds=3" in err
    assert "Traceback" not in err


def test_an_unknown_scenario_config_lists_the_existing_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["scenario=intersection_passing/straigth"])
    captured = capsys.readouterr()
    assert code == 2
    assert "did you mean: intersection_passing/straight" in captured.err
    assert "intersection_passing/straight" in captured.err


def test_an_unknown_scenario_name_lists_the_known_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["scenario=intersection_passing/straight", "scenario.name=other"])
    out = capsys.readouterr().out
    assert code == 1
    assert "unknown scenario name 'other'" in out
    assert "known names:" in out


def test_hydra_full_error_shows_the_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra.errors import HydraException  # noqa: PLC0415

    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(HydraException):
        main(["scenario=intersection_passing/straight", "scenario.timout_seconds=3"])
