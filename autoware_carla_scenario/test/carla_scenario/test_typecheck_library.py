"""The library check: the framework's own modules, compiled with Codon.

The manifest (``typecheck/library.py``) and the source the check generates
are tested without Codon; the compiles are skipped where no Codon compiler is
installed (typesafe-carla's toolchain, a dependency on Linux x86_64 and
aarch64).
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import (
    TypeCheckResult,
    available_toolchain,
    check,
    typecheck_library,
)
from autoware_carla_scenario.typecheck.cli import main as scenario_check
from autoware_carla_scenario.typecheck.library import (
    CHECKED,
    EXCLUDED,
    NOT_YET_CHECKED,
    package_modules,
)
from autoware_carla_scenario.typecheck.library_driver import (
    render_library_checks,
    render_library_driver,
    workspace_module,
)
from autoware_carla_scenario.typecheck.transform import redirect_imports

_PACKAGE = "autoware_carla_scenario"

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_every_module_is_either_checked_or_excluded() -> None:
    modules = set(package_modules())
    assert len(CHECKED) == len(set(CHECKED)), "a module is listed twice in CHECKED"
    both = set(CHECKED) & set(EXCLUDED)
    assert not both, f"in both CHECKED and EXCLUDED: {sorted(both)}"
    missing = modules - set(CHECKED) - set(EXCLUDED)
    assert not missing, (
        f"not in typecheck/library.py: {sorted(missing)}; add each to EXCLUDED "
        f"({NOT_YET_CHECKED!r}) or, if it compiles, to CHECKED"
    )
    unknown = (set(CHECKED) | set(EXCLUDED)) - modules
    assert (
        not unknown
    ), f"listed in typecheck/library.py but not modules: {sorted(unknown)}"


def test_every_excluded_module_has_a_reason() -> None:
    assert all(reason.strip() for reason in EXCLUDED.values())


def test_the_checker_itself_is_not_a_module_to_check() -> None:
    assert not [m for m in package_modules() if m.startswith(f"{_PACKAGE}.typecheck")]


# ---------------------------------------------------------------------------
# Imports of the framework, pointed into the workspace
# ---------------------------------------------------------------------------


_CHECKED = {f"{_PACKAGE}.coordinate.frames", f"{_PACKAGE}.kinematics.vector"}
_MODULES = _CHECKED | {
    _PACKAGE,
    f"{_PACKAGE}.coordinate",
    f"{_PACKAGE}.coordinate.poses",
    f"{_PACKAGE}.kinematics",
    f"{_PACKAGE}.utils",
    f"{_PACKAGE}.utils.config",
}


def _target(name: str) -> str | None:
    if name in _CHECKED:
        return workspace_module(name)
    if name.startswith(f"{_PACKAGE}.coordinate"):
        return f"{_PACKAGE}.coordinate"  # the model of the package
    return None


def _redirect(source: str, module: str = f"{_PACKAGE}.kinematics.velocity"):
    return redirect_imports(
        textwrap.dedent(source), module, False, _target, _MODULES.__contains__
    )


def test_an_import_of_a_checked_module_names_its_workspace_module() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.coordinate.frames import (
            CoordinateFrame,
            frame_of as of,
        )
        from .vector import Vector3
        import math
        """
    )
    assert problems == []
    lines = out.splitlines()
    assert lines[1] == (
        "from _acs_lib.autoware_carla_scenario__coordinate__frames "
        "import CoordinateFrame, frame_of as of"
    )
    assert (
        lines[5]
        == "from _acs_lib.autoware_carla_scenario__kinematics__vector import Vector3"
    )
    assert lines[6] == "import math"
    assert len(lines) == 7  # every statement kept its line


def test_an_import_of_an_unchecked_module_names_its_model() -> None:
    out, problems = _redirect(
        """
        if TYPE_CHECKING:
            from ..coordinate.poses import Lanelet2Pose
        from ..coordinate import frames, poses as p
        """
    )
    assert problems == []
    assert "    from autoware_carla_scenario.coordinate import Lanelet2Pose" in out
    assert (
        "from _acs_lib import autoware_carla_scenario__coordinate__frames as frames; "
        "from autoware_carla_scenario import coordinate as p"
    ) in out


def test_an_import_with_nothing_to_stand_in_is_reported() -> None:
    out, problems = _redirect(
        """
        from autoware_carla_scenario.utils.config import load
        import autoware_carla_scenario.coordinate.frames
        from .vector import *
        """
    )
    assert [line for line, _ in problems] == [2, 3, 4]
    assert "neither checked" in problems[0][1]
    assert "from autoware_carla_scenario.utils.config import load" in out


# ---------------------------------------------------------------------------
# The calls appended to a checked module
# ---------------------------------------------------------------------------


_MODULE = """
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


def angle(a: float, *, wrap: bool = True) -> float:
    return a


def where(pose: Lanelet2Pose | OpenDrivePose | None) -> float:
    return 0.0


def total(xs: Sequence[float]) -> float:
    return 0.0


def _private(x: Any) -> None:
    pass


class Vector:
    def __init__(self, x: float) -> None:
        self.x = x

    def __eq__(self, other: object) -> bool:
        return True

    @property
    def norm(self) -> float:
        return self.x

    @staticmethod
    def zero() -> Vector:
        return Vector(0.0)

    @classmethod
    def unit(cls, scale: float) -> Vector:
        return cls(scale)

    def _helper(self, anything: Any) -> None:
        pass


@dataclass
class Config:
    name: str
    ids: list[int] = field(default_factory=list)
    hidden: int = field(default=0, init=False)


class Level(Enum):
    LOW = 1

    def up(self) -> Level:
        return self


def generic(x: Any, y) -> None:
    pass
"""


def test_every_public_function_and_method_is_called_with_typed_values() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    calls = checks.source.splitlines()
    assert "    angle(_acs_value(float), wrap=_acs_value(bool))" in calls
    assert "    where(_acs_value(Lanelet2Pose))" in calls
    assert "    where(_acs_value(OpenDrivePose))" in calls
    assert "    total(_acs_value(Sequence[float]))" in calls
    assert "    Vector(_acs_value(float))" in calls
    assert "    _acs_value(Vector).__eq__(_acs_value(Vector))" in calls
    assert "    _acs_value(Vector).norm" in calls
    assert "    Vector.zero()" in calls
    assert "    Vector.unit(_acs_value(float))" in calls
    assert "    Config(_acs_value(str), _acs_value(list[int]))" in calls
    assert "    _acs_value(Level).up()" in calls
    assert not [c for c in calls if "_private" in c or "_helper" in c or "Level(" in c]
    label = checks.labels[calls.index("    where(_acs_value(OpenDrivePose))") + 1]
    assert label == "where(pose: OpenDrivePose)"


def test_a_parameter_the_check_cannot_call_with_is_a_problem_of_the_module() -> None:
    checks = render_library_checks(ast.parse(_MODULE))
    lines = _MODULE.splitlines()
    assert len(checks.problems) == 2
    for line, message in checks.problems:
        assert lines[line - 1].startswith("def generic(")
    assert any("`x`" in m and "Any" in m for _, m in checks.problems)
    assert any("`y` has no annotation" in m for _, m in checks.problems)


def test_the_driver_runs_the_calls_of_each_module() -> None:
    driver = render_library_driver([f"{_PACKAGE}.kinematics.angle"])
    assert (
        "from _acs_lib.autoware_carla_scenario__kinematics__angle import "
        "_acs_library_check as _acs_check_0"
    ) in driver.source
    assert "_acs_check_0()" in driver.source


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


@needs_codon
def test_the_checked_modules_compile() -> None:
    result = typecheck_library()
    assert result.skipped is None
    assert result.ok, result.format()


@needs_codon
def test_scenario_check_checks_the_library(capsys: pytest.CaptureFixture[str]) -> None:
    assert scenario_check(["--library", f"{_PACKAGE}.kinematics.angle"]) == 0
    assert "[ok]" in capsys.readouterr().out


_FAKE = f"{_PACKAGE}.zz_library_check_case"

_GOOD = """
from __future__ import annotations

from enum import Enum, auto

from autoware_carla_scenario.coordinate.poses import Lanelet2Pose, OpenDrivePose
from .entity_role import EntityRole
from .kinematics.vector import Vector3


class Turn(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Level(Enum):
    LOW = auto()
    HIGH = auto()


class LevelError(ValueError):
    level: Level

    def __init__(self, level: Level) -> None:
        super().__init__(f"bad level {level.name}")
        self.level = level


class Thing:
    size: float

    def __init__(self, size: float) -> None:
        self.size = size

    @classmethod
    def unit(cls) -> Thing:
        return cls(1.0)

    @property
    def doubled(self) -> float:
        return self.size * 2

    def scaled(self, v: Vector3) -> Vector3:
        return v * self.size


def lane_s(pose: Lanelet2Pose | OpenDrivePose) -> float:
    return pose.s


def describe(turn: Turn, level: Level, role: EntityRole) -> str:
    if level == Level.HIGH:
        raise LevelError(level)
    return turn.value + str(role)
"""


def _check_case(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TypeCheckResult:
    path = tmp_path / "zz_library_check_case.py"
    path.write_text(textwrap.dedent(source))
    modules = package_modules()
    modules[_FAKE] = path
    monkeypatch.setattr(check, "package_modules", lambda: modules)
    return typecheck_library(
        [_FAKE, f"{_PACKAGE}.entity_role", f"{_PACKAGE}.kinematics.vector"]
    )


@needs_codon
def test_enums_exceptions_classmethods_and_unions_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(_GOOD, tmp_path, monkeypatch)
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("wrong", "right"),
    [
        # A body of the wrong type, in each kind of callable.
        ("return turn.value + str(role)", "return level.value"),
        ("return cls(1.0)", "return cls('one')"),
        ("return self.size * 2", "return self.size.real()"),
        # Only one member of the union has the attribute.
        ("return pose.s", "return pose.road_id"),
    ],
    ids=["function", "classmethod", "property", "union-member"],
)
def test_a_wrong_module_is_refused_at_its_line(
    wrong: str, right: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _GOOD.replace(wrong, right)
    assert source != _GOOD
    result = _check_case(source, tmp_path, monkeypatch)
    assert not result.ok
    stripped = [text.strip() for text in textwrap.dedent(source).splitlines()]
    line = stripped.index(right) + 1
    located = [d for d in result.diagnostics if d.path and d.path.endswith(".py")]
    assert located, result.format()
    assert located[0].path == str(tmp_path / "zz_library_check_case.py")
    assert located[0].line == line, result.format()


@needs_codon
def test_a_module_importing_what_codon_cannot_compile_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _check_case(
        "import yaml\n\n\ndef f() -> int:\n    return 1\n", tmp_path, monkeypatch
    )
    assert not result.ok
    assert result.diagnostics[0].line == 1
    assert "yaml has no Codon model" in result.diagnostics[0].message


@needs_codon
def test_an_unknown_module_is_refused() -> None:
    result = typecheck_library([f"{_PACKAGE}.no_such_module"])
    assert not result.ok
    assert "not a module" in result.diagnostics[0].message
