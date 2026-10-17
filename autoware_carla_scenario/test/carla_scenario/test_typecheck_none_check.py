"""The None check of the static check: mypy's strict Optional, narrowed (#239).

No Codon needed: these run mypy alone on a module written for the test.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck import none_check
from autoware_carla_scenario.typecheck.none_check import (
    NoneCheckSkipped,
    find_none_misuses,
    mypy_available,
    run_none_check,
)

pytestmark = pytest.mark.skipif(not mypy_available(), reason="no mypy")

_MODULE = """
from __future__ import annotations


def maybe() -> float | None:
    return None


class Pose:
    lanelet_id: int = 0


def pose() -> Pose | None:
    return None


def half(x: float) -> float:
    return x / 2


def misuses() -> None:
    a = maybe() / 3.6
    b = pose().lanelet_id
    c = half(maybe())
    d: float = maybe()


def narrowed() -> float:
    x = maybe()
    if x is None:
        return 0.0
    p = pose()
    assert p is not None
    return x / 3.6 + p.lanelet_id + half(maybe() or 0.0)


def other_errors() -> int:
    y: int = "not None at all"
    return half("text")
"""


def test_only_a_value_that_may_be_none_misused_is_reported(tmp_path: Path) -> None:
    package = tmp_path / "acs_none_case"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_MODULE))

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    found = [(lines[f.line - 1].strip(), f.code) for f in findings]
    assert found == [
        ("a = maybe() / 3.6", "operator"),
        ("b = pose().lanelet_id", "union-attr"),
        ("c = half(maybe())", "arg-type"),
        ("d: float = maybe()", "assignment"),
    ]
    # Also when mypy replays them from its cache, cached at another path.
    assert all(f.path == str(module) for f in findings), findings
    assert findings[0].notes == ('Left operand is of type "float | None"',)


def test_a_finding_mypy_replays_from_its_cache_is_at_the_module_given(
    tmp_path: Path,
) -> None:
    # mypy replays the errors of a module its cache holds unchanged with the
    # path it was cached at: the same package in two places is reported at
    # the second place when it is checked there.
    modules = []
    for place in ("first", "second"):
        package = tmp_path / place / "acs_none_replayed"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        module = package / "probe.py"
        module.write_text(textwrap.dedent(_MODULE))
        modules.append(module)
    for module in modules:
        findings = find_none_misuses([module])
        assert findings, findings
        assert {f.path for f in findings} == {str(module)}


def test_no_module_gives_no_verdict() -> None:
    assert find_none_misuses([]) is None


_DECLARATIONS = """
from __future__ import annotations


class Plain:
    def __init__(self) -> None:
        self.count = None

    def later(self) -> None:
        self.count = 5


class Holder:
    value: int = None


def default(speed: float = None) -> float:
    return 1.0


def maybe() -> float | None:
    return None


def misuse() -> float:
    return maybe() * 2
"""


def test_an_assignment_that_only_declares_is_no_misuse(tmp_path: Path) -> None:
    # A None default or value for a non-Optional annotation, and a value for
    # an attribute mypy inferred as None, use no value that may be None.
    package = tmp_path / "acs_none_declarations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_DECLARATIONS))

    findings = find_none_misuses([module])

    assert findings is not None
    lines = module.read_text().splitlines()
    assert [lines[f.line - 1].strip() for f in findings] == ["return maybe() * 2"]


def test_an_unwritable_cache_directory_runs_mypy_without_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # mypy fails outright on a cache directory it cannot create (a
    # read-only home in a container); the check runs it without one.
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker / "cache"))
    assert none_check._cache_dir() == os.devnull
    package = tmp_path / "acs_none_no_cache"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "probe.py"
    module.write_text(textwrap.dedent(_MODULE))

    assert len(run_none_check([module])) == 4


def test_no_verdict_from_mypy_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "probe.py"
    module.write_text("x = 1\n")
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n")
    with pytest.raises(NoneCheckSkipped, match=r"exit status 2\): .*Invalid syntax"):
        run_none_check([broken])
    with pytest.raises(NoneCheckSkipped, match="no module"):
        run_none_check([])
    monkeypatch.setattr(none_check, "mypy_available", lambda: False)
    with pytest.raises(NoneCheckSkipped, match="not installed"):
        run_none_check([module])
