"""``scenario-check``: what a command line that does not compose prints (#532)."""

from __future__ import annotations

import math
from typing import Any

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
        "+controls.vehicle_ahead_gap_m.range=abc",
        "controls.vehicle_ahead_gap_m.range: expected [low, high]",
    ),
    (
        "+controls.vehicle_ahead_gap_m.range=[30,2]",
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
        "controls.vehicle_ahead_gap_m.quantity=5",
        "controls.vehicle_ahead_gap_m.quantity: expected the name of a quantity",
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


@pytest.mark.parametrize(
    ("field", "pair", "shown"),
    [
        ("range", [math.nan, 1.0], "[nan, 1.0]"),
        ("range", [0.0, math.nan], "[0.0, nan]"),
        ("limits", [math.nan, None], "[nan, None]"),
        ("limits", [None, math.nan], "[None, nan]"),
    ],
)
def test_a_nan_end_of_a_control_pair_is_refused(
    field: str, pair: list[Any], shown: str
) -> None:
    """A NaN end compares false with the other, so ``low > high`` let it
    through to fail when the knob is built (#2134)."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    cfg = OmegaConf.create({"scenario": {"npc_gap_m": 1.0}})
    spec = {"key": "scenario.npc_gap_m", field: pair}
    found = validate_controls({"vehicle_ahead_gap_m": spec}, cfg)
    assert [(d.path, d.message, d.severity) for d in found] == [
        (
            f"controls.vehicle_ahead_gap_m.{field}",
            f"expected [low, high] with low <= high, got {shown}",
            "error",
        )
    ]


@pytest.mark.parametrize(
    ("field", "pair"),
    [
        ("range", [0.0, 1.0]),
        ("range", [1.0, 1.0]),
        ("range", [0.0, math.inf]),
        ("limits", [0.0, None]),
        ("limits", [None, None]),
        ("limits", [-math.inf, 3.0]),
    ],
)
def test_a_control_pair_with_low_up_to_high_is_taken(
    field: str, pair: list[Any]
) -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    cfg = OmegaConf.create({"scenario": {"npc_gap_m": 1.0}})
    spec = {"key": "scenario.npc_gap_m", field: pair}
    assert validate_controls({"vehicle_ahead_gap_m": spec}, cfg) == []


def test_the_shipped_controls_are_valid() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.examples import run  # noqa: PLC0415
    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    for name in run._resolve_scenario_glob("**/*"):
        cfg = run._compose_config(name, [])
        node = OmegaConf.select(cfg, "controls")
        controls = None if node is None else OmegaConf.to_container(node, resolve=True)
        assert validate_controls(controls, cfg) == [], name


def test_a_key_with_a_null_or_mandatory_value_is_a_key_of_the_config() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    cfg = OmegaConf.create({"scenario": {"empty": None, "must": "???", "x": 1}})
    for key in ("scenario.empty", "scenario.must", "scenario.x"):
        assert validate_controls({"vehicle_ahead_gap_m": key}, cfg) == [], key
    found = validate_controls({"vehicle_ahead_gap_m": "scenario.nope"}, cfg)
    assert [d.path for d in found] == ["controls.vehicle_ahead_gap_m.key"]


def test_a_weak_resemblance_is_not_suggested() -> None:
    from omegaconf import OmegaConf  # noqa: PLC0415

    from autoware_carla_scenario.odd.controls import validate_controls  # noqa: PLC0415

    cfg = OmegaConf.create({"scenario": {"side": "left", "npc_gap_m": 1.0}})
    far = validate_controls({"vehicle_ahead_gap_m": "scenario.nonexistent"}, cfg)
    assert "did you mean" not in far[0].message
    near = validate_controls({"vehicle_ahead_gap_m": "scenario.npc_gap"}, cfg)
    assert "did you mean 'scenario.npc_gap_m'?" in near[0].message


# A check that raises (#2091) fails its config only: the others are still
# checked and reported, and the exit status is that of a failed check.
def _fake_toolchain() -> object:
    from pathlib import Path  # noqa: PLC0415

    from autoware_carla_scenario.typecheck.toolchain import Toolchain  # noqa: PLC0415

    return Toolchain(Path("/nonexistent/codon"), Path("/nonexistent"), "test")


def test_a_checker_error_fails_its_config_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.typecheck import cli  # noqa: PLC0415
    from autoware_carla_scenario.typecheck.check import (  # noqa: PLC0415
        TypeCheckResult,
    )

    def check(name: str, scenario: object, *, toolchain: object) -> TypeCheckResult:
        if dict(scenario)["side"] == "left":  # type: ignore[call-overload]
            raise RuntimeError("boom")
        return TypeCheckResult(name, ok=True)

    monkeypatch.setattr(cli, "find_supported_codon", _fake_toolchain)
    monkeypatch.setattr(cli, "typecheck_registered", check)
    code = main(["scenario=cut_in/*"])
    captured = capsys.readouterr()
    assert code == 1
    assert "[FAILED] cut_in/left: checker error: RuntimeError: boom" in captured.out
    assert "[ok] cut_in/right" in captured.out
    assert "1 of 2 scenario config(s) failed the static check" in captured.out
    assert "Traceback" not in captured.out
    assert "Traceback" in captured.err


def test_a_library_checker_error_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoware_carla_scenario.typecheck import cli  # noqa: PLC0415

    def library(modules: object, *, toolchain: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "find_supported_codon", _fake_toolchain)
    monkeypatch.setattr(cli, "typecheck_library", library)
    code = main(["--library"])
    captured = capsys.readouterr()
    assert code == 1
    assert "[FAILED] library check: checker error: RuntimeError: boom" in captured.out
    assert "Traceback" in captured.err
