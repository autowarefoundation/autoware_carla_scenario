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


# `controls:` (#546): each is reported with its config path, by the check and
# by the expansion, never as a traceback.
_BAD_CONTROLS = [
    (
        "controls.vehicle_ahead_gap_m.range=abc",
        "controls.vehicle_ahead_gap_m.range: expected [low, high]",
    ),
    (
        "controls.vehicle_ahead_gap_m.range=[30,2]",
        "controls.vehicle_ahead_gap_m.range: expected [low, high] with low <= high",
    ),
    (
        "controls.vehicle_ahead_gap_m.target=bogus",
        "controls.vehicle_ahead_gap_m.target: expected one of",
    ),
    (
        "controls.vehicle_ahead_gap_m.key=scenario.nonexistent",
        "controls.vehicle_ahead_gap_m.key: 'scenario.nonexistent' is not a key",
    ),
    (
        "controls.vehicle_ahead_gap_m=5",
        "controls.vehicle_ahead_gap_m: expected a key or a mapping, got 5",
    ),
    (
        "+controls.vehicle_ahead_gap_m.rnge=[2,3]",
        "controls.vehicle_ahead_gap_m.rnge: unknown field (did you mean 'range'?)",
    ),
]


@pytest.mark.parametrize(("override", "expected"), _BAD_CONTROLS)
def test_a_bad_control_fails_the_check_with_its_path(
    override: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["scenario=cut_in/left", override])
    captured = capsys.readouterr()
    assert code == 1
    assert f"[FAILED] cut_in/left: {expected}" in captured.out
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(("override", "expected"), _BAD_CONTROLS)
def test_a_bad_control_fails_the_expansion_with_its_path(
    override: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.examples.expand import main as expand  # noqa: PLC0415

    code = expand(["scenario=cut_in/left", "+sweep.odd_sample.count=2", override])
    captured = capsys.readouterr()
    assert code == 1
    assert f"cut_in/left: {expected}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_an_unknown_measure_is_a_warning_with_a_suggestion() -> None:
    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    found = validate_controls({"vehicle_ahead_gap": {"key": "scenario.npc_gap_m"}})
    assert [(d.path, d.severity) for d in found] == [
        ("controls.vehicle_ahead_gap", "warning")
    ]
    assert "did you mean 'vehicle_ahead_gap_m'?" in found[0].message


def test_the_shipped_controls_are_valid() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415
    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    for name in run._resolve_scenario_glob("**/*"):
        cfg = run._compose_config(name, [])
        node = OmegaConf.select(cfg, "controls")
        controls = None if node is None else OmegaConf.to_container(node, resolve=True)
        assert validate_controls(controls, cfg) == [], name
