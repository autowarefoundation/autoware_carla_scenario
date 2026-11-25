"""The rendering of the check's programs, without compiling them (#2511).

The generated driver (``typecheck/driver.py``), the library check's calls
(``typecheck/library_driver.py``), Codon's environment
(``typecheck/toolchain.py``) and the order ``scenario-check`` reports in
(``typecheck/cli.py``).
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pytest

from autoware_carla_scenario.typecheck import cli
from autoware_carla_scenario.typecheck.driver import render_driver
from autoware_carla_scenario.typecheck.library_driver import render_library_checks
from autoware_carla_scenario.typecheck.toolchain import (
    Toolchain,
    ToolchainError,
    codon_environment,
)

if TYPE_CHECKING:
    from not_installed_anywhere import Actor  # resolved by no get_type_hints()


class _Scenario:
    pass


def _by_key(driver: Any) -> dict[str, str]:
    lines = driver.source.splitlines()
    return {key: lines[line - 1].strip() for line, key in driver.config_lines.items()}


# ---------------------------------------------------------------------------
# 1. A rendered class is imported under a name of its own
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EgoConfig:  # the framework's driver text builds its own `EgoConfig`
    speed: float = 1.0


class Mode(enum.Enum):
    A = "a"


#: Another `Mode`, of another module.
OtherMode = enum.Enum("Mode", {"B": "b"}, module="other_package.modes")  # type: ignore[misc]


@dataclasses.dataclass
class _Shadowing:
    ego: EgoConfig = dataclasses.field(default_factory=EgoConfig)
    mode: Mode = Mode.A
    other: Any = OtherMode.B  # type: ignore[attr-defined]

    def __post_init__(self) -> None:
        if isinstance(self.ego, dict):
            self.ego = EgoConfig(**self.ego)


def test_a_class_named_as_another_is_imported_under_an_alias() -> None:
    driver = render_driver(
        _Scenario,
        _Shadowing,
        {"ego": {"speed": 2.0}, "mode": Mode.A, "other": OtherMode.B},  # type: ignore[attr-defined]
    )
    lines = driver.source.splitlines()
    # The framework's EgoConfig still builds the ego; the package's is aliased.
    assert "        EgoConfig(SpawnTransform(carla.Transform()))," in lines
    assert f"from {__name__} import EgoConfig as _acs_cls3_EgoConfig" in lines
    assert f"from {__name__} import Mode" in lines
    assert "from other_package.modes import Mode as _acs_cls5_Mode" in lines
    assert "    config.ego = _acs_cls3_EgoConfig(" in lines
    by_key = _by_key(driver)
    assert by_key["scenario.mode"] == "config.mode = Mode.A"
    assert by_key["scenario.other"] == "config.other = _acs_cls5_Mode.B"
    # The shape renders under the same names, line for line.
    assert len(driver.shape.splitlines()) == len(lines)
    assert "    config.other = <_acs_cls5_Mode>" in driver.shape.splitlines()


def test_an_unshadowed_class_keeps_its_own_name() -> None:
    @dataclasses.dataclass
    class Plain:
        mode: Mode = Mode.A

    driver = render_driver(_Scenario, Plain, {"mode": Mode.A})
    assert f"from {__name__} import Mode" in driver.source.splitlines()
    assert "_acs_cls" not in driver.source


# ---------------------------------------------------------------------------
# 2. A tuple value is rendered as the type the transform declares
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Tuples:
    pos: tuple[float, int] = (0.0, 0)
    pts: tuple[float, ...] = ()
    empty: tuple[()] = ()

    def __post_init__(self) -> None:
        self.pos = tuple(self.pos)  # type: ignore[assignment]
        self.pts = tuple(self.pts)
        self.empty = tuple(self.empty)  # type: ignore[assignment]


def test_a_variadic_tuple_is_a_list_and_a_fixed_one_is_typed_per_item() -> None:
    driver = render_driver(
        _Scenario, _Tuples, {"pos": [1.0, 2], "pts": [1, 2], "empty": []}
    )
    assert _by_key(driver) == {
        # tuple[float, int]: each item of its own type, not args[0].
        "scenario.pos": "config.pos = (1.0, 2,)",
        # tuple[float, ...] is the List[float] the transform declares.
        "scenario.pts": "config.pts = [1.0, 2.0]",
        # An empty tuple is a tuple.
        "scenario.empty": "config.empty = ()",
    }
    empty = render_driver(_Scenario, _Tuples, {"pts": []})
    assert _by_key(empty) == {"scenario.pts": "config.pts = []"}


# ---------------------------------------------------------------------------
# 3. One annotation that does not resolve drops that field's hint only
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Leg:
    speed: float = 1.0


@dataclasses.dataclass
class _TypeCheckingOnly:
    leg: _Leg = dataclasses.field(default_factory=_Leg)
    timeout_seconds: float = 30.0
    actor: Optional[Actor] = None

    def __post_init__(self) -> None:
        if isinstance(self.leg, dict):
            self.leg = _Leg(**self.leg)


def test_an_unresolvable_annotation_leaves_the_other_fields_typed() -> None:
    driver = render_driver(
        _Scenario, _TypeCheckingOnly, {"leg": {"speed": 2}, "timeout_seconds": 3}
    )
    lines = driver.source.splitlines()
    # The nested dataclass is built, not passed as a dict, and an int for a
    # float field is still a float, checked strictly.
    assert "    config.leg = _Leg(" in lines
    assert "speed=2.0," in lines
    assert "    config.timeout_seconds = 3.0" in lines
    assert "    _acs_as_float(3.0)" in lines


def test_a_dataclass_the_config_holds_is_rendered_as_its_type() -> None:
    @dataclasses.dataclass
    class Untyped:
        leg: Any = None

        def __post_init__(self) -> None:
            if isinstance(self.leg, dict):
                self.leg = _Leg(**self.leg)

    driver = render_driver(_Scenario, Untyped, {"leg": {"speed": 2.0}})
    assert "    config.leg = _Leg(" in driver.source.splitlines()


# ---------------------------------------------------------------------------
# 4. A bare ClassVar is no constructor argument
# ---------------------------------------------------------------------------


def test_a_bare_class_var_is_not_a_dataclass_field() -> None:
    checks = render_library_checks(
        ast.parse(
            "from dataclasses import dataclass\n"
            "from typing import ClassVar\n"
            "\n"
            "@dataclass\n"
            "class Point:\n"
            "    kind: ClassVar = 'p'\n"
            "    limit: ClassVar[int] = 3\n"
            "    x: float\n"
        )
    )
    calls = [c for c in checks.source.splitlines() if c.startswith("    Point(")]
    assert calls == ["    Point(_acs_value(float))"]


# ---------------------------------------------------------------------------
# 5. LD_LIBRARY_PATH gets no empty entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lib", "existing", "expected"),
    [
        (False, None, None),
        (False, "/usr/local/lib", "/usr/local/lib"),
        (True, None, "{lib}"),
        (True, "/usr/local/lib", "{lib}" + os.pathsep + "/usr/local/lib"),
        (True, "", "{lib}"),
    ],
)
def test_the_library_path_has_no_empty_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lib: bool,
    existing: Optional[str],
    expected: Optional[str],
) -> None:
    if lib:
        (tmp_path / "lib").mkdir()
    if existing is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", existing)
    env = codon_environment(Toolchain(tmp_path / "codon", tmp_path, "test"), tmp_path)
    want = None if expected is None else expected.format(lib=tmp_path / "lib")
    assert env.get("LD_LIBRARY_PATH") == want


# ---------------------------------------------------------------------------
# 6. A wrong control is reported without Codon
# ---------------------------------------------------------------------------


def _no_codon() -> Toolchain:
    raise ToolchainError("no Codon here")


def test_a_bad_control_fails_the_check_without_codon(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "find_supported_codon", _no_codon)
    code = cli.main(
        ["scenario=cut_in/left", "controls.vehicle_ahead_gap_m.target=bogus"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert (
        "[FAILED] cut_in/left: controls.vehicle_ahead_gap_m.target: expected one of"
        in captured.out
    )
    assert "Traceback" not in captured.out + captured.err


def test_without_codon_good_controls_still_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "find_supported_codon", _no_codon)
    code = cli.main(["scenario=cut_in/left"])
    assert code == 2
    assert "scenario-check: no Codon here" in capsys.readouterr().err
