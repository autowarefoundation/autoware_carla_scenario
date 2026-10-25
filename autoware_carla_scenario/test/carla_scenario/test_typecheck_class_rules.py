"""The class rules of the static check: a framework base's ``__init__``
skipped (#587) and a declared attribute never assigned (#588).

Pure Python: the modules are written, imported and given to the rules, with
no Codon.
"""

from __future__ import annotations

import ast
import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from autoware_carla_scenario.typecheck.class_rules import (
    ClassProblem,
    skipped_base_inits,
    unassigned_attributes,
)

_HEADER = """
from __future__ import annotations

import typesafe_carla.carla as carla

from autoware_carla_scenario import (
    BaseAction,
    BaseCondition,
    BaseScenario,
    EgoConfig,
    Lanelet2Pose,
    ScenarioResult,
)
"""

_counter = 0


def _modules(
    tmp_path: Path, body: str, more: dict[str, str] | None = None
) -> tuple[dict[str, tuple[str, ast.Module]], Path]:
    """Write and import a package whose ``case`` module is *body*."""
    global _counter
    _counter += 1
    package = f"acs_class_rules_{_counter}"
    root = tmp_path / package
    root.mkdir()
    (root / "__init__.py").write_text("")
    files = {"case": _HEADER + textwrap.dedent(body)}
    for name, text in (more or {}).items():
        files[name] = _HEADER + textwrap.dedent(text).replace("{package}", package)
    out: dict[str, tuple[str, ast.Module]] = {}
    sys.path.insert(0, str(tmp_path))
    try:
        for name, text in files.items():
            path = root / f"{name}.py"
            path.write_text(text)
        for name, text in files.items():
            dotted = f"{package}.{name}"
            importlib.import_module(dotted)
            out[dotted] = (str(root / f"{name}.py"), ast.parse(text))
    finally:
        sys.path.remove(str(tmp_path))
    return out, root / "case.py"


def _line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} not in {path}")


def _only(problems: list[ClassProblem]) -> ClassProblem:
    assert len(problems) == 1, problems
    return problems[0]


# ---------------------------------------------------------------------------
# #587
# ---------------------------------------------------------------------------

_LATE = """
class LateCondition(BaseCondition):
    _after: float

    def __init__(self, after: float) -> None:
{init}

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        return None
"""


def _late(init: str) -> str:
    return _LATE.replace("{init}", textwrap.indent(textwrap.dedent(init), " " * 8))


def test_a_condition_init_without_super_is_refused_at_its_def(tmp_path: Path) -> None:
    modules, path = _modules(tmp_path, _late("self._after = after\n"))
    problem = _only(skipped_base_inits(modules))
    assert problem.path == str(path)
    assert problem.line == _line_of(path, "def __init__")
    assert "LateCondition.__init__ never calls BaseCondition.__init__" in (
        problem.message
    )
    assert "label" in problem.message
    assert "super().__init__(...)" in problem.message


@pytest.mark.parametrize(
    "init",
    [
        'super().__init__("late")\nself._after = after\n',
        'super(LateCondition, self).__init__(label="late")\nself._after = after\n',
        'BaseCondition.__init__(self, "late")\nself._after = after\n',
        'if after > 0:\n    super().__init__("late")\nelse:\n'
        '    super().__init__("never")\nself._after = after\n',
        'try:\n    super().__init__("late")\nexcept ValueError:\n    raise\n'
        "self._after = after\n",
    ],
    ids=["super", "super-two-args", "base-explicit", "in-if", "in-try"],
)
def test_a_call_of_the_base_init_anywhere_in_the_body_is_enough(
    tmp_path: Path, init: str
) -> None:
    modules, _ = _modules(tmp_path, _late(init))
    assert skipped_base_inits(modules) == []


def test_a_helper_calling_the_base_init_counts(tmp_path: Path) -> None:
    body = _late("self._start(after)\n") + textwrap.indent(
        textwrap.dedent(
            """
            def _start(self, after: float) -> None:
                super().__init__("late")
                self._after = after
            """
        ),
        " " * 4,
    )
    modules, _ = _modules(tmp_path, body)
    assert skipped_base_inits(modules) == []


def test_a_scenario_init_without_super_is_refused(tmp_path: Path) -> None:
    body = """
    class CaseScenario(BaseScenario):
        def __init__(self, ego_config: EgoConfig, spawn_pose: Lanelet2Pose) -> None:
            self.ego_spawn = spawn_pose

        def is_done(self) -> bool:
            return False

        def setup(self) -> None:
            pass
    """
    modules, path = _modules(tmp_path, body)
    problem = _only(skipped_base_inits(modules))
    assert "CaseScenario.__init__ never calls BaseScenario.__init__" in (
        problem.message
    )
    assert problem.line == _line_of(path, "def __init__")


def test_an_action_init_without_super_is_refused(tmp_path: Path) -> None:
    body = """
    class HonkAction(BaseAction):
        def __init__(self) -> None:
            pass

        def execute(self, world: carla.World) -> None:
            pass
    """
    modules, _ = _modules(tmp_path, body)
    assert "HonkAction.__init__ never calls BaseAction.__init__" in (
        _only(skipped_base_inits(modules)).message
    )


def test_a_subclass_of_a_subclass_that_calls_super_is_refused_too(
    tmp_path: Path,
) -> None:
    # The intermediate class calls the framework's __init__, but its own is
    # skipped, and the framework's with it.
    body = (
        _late('super().__init__("late")\nself._after = after\n')
        + """

class LaterCondition(LateCondition):
    def __init__(self) -> None:
        self._after = 9.0
"""
    )
    modules, path = _modules(tmp_path, body)
    problem = _only(skipped_base_inits(modules))
    assert "LaterCondition.__init__ never calls BaseCondition.__init__" in (
        problem.message
    )
    assert problem.line == _line_of(path, "self._after = 9.0") - 1


def test_a_subclass_without_its_own_init_is_left_alone(tmp_path: Path) -> None:
    body = (
        _late('super().__init__("late")\nself._after = after\n')
        + """

class LaterCondition(LateCondition):
    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        return None
"""
    )
    modules, _ = _modules(tmp_path, body)
    assert skipped_base_inits(modules) == []


def test_classes_of_no_framework_base_are_left_alone(tmp_path: Path) -> None:
    body = """
    class Base:
        def __init__(self) -> None:
            self.value = 1


    class Tracker(Base):
        def __init__(self) -> None:
            self.other = 2


    class Error(ValueError):
        def __init__(self, what: str) -> None:
            self.what = what
    """
    modules, _ = _modules(tmp_path, body)
    assert skipped_base_inits(modules) == []


def test_a_framework_base_with_a_trivial_init_is_left_alone(tmp_path: Path) -> None:
    framework = {"framework": "class Trivial:\n    def __init__(self) -> None:\n"}
    framework["framework"] += "        pass\n"
    body = """
    from .framework import Trivial


    class Mine(Trivial):
        def __init__(self) -> None:
            self.value = 1
    """
    modules, _ = _modules(tmp_path, body, framework)
    case = {k: v for k, v in modules.items() if k.endswith(".case")}
    every = dict(modules)

    def framework_cls(base: type) -> bool:
        return base.__module__.endswith(".framework")

    assert skipped_base_inits(case, framework=framework_cls) == []
    assert skipped_base_inits(every, framework=framework_cls) == []


def test_a_dataclass_base_of_the_framework_counts(tmp_path: Path) -> None:
    framework = {
        "framework": (
            "from dataclasses import dataclass\n\n\n@dataclass\n"
            "class Settings:\n    speed: float = 1.0\n"
        )
    }
    body = """
    from .framework import Settings


    class Mine(Settings):
        def __init__(self) -> None:
            self.extra = 1
    """
    modules, _ = _modules(tmp_path, body, framework)
    case = {k: v for k, v in modules.items() if k.endswith(".case")}
    problem = _only(
        skipped_base_inits(
            case, framework=lambda base: base.__module__.endswith(".framework")
        )
    )
    assert "never calls Settings.__init__, which sets speed" in problem.message


def test_a_dataclass_subclass_without_an_init_is_left_alone(tmp_path: Path) -> None:
    body = """
    from dataclasses import dataclass


    @dataclass
    class Settings:
        speed: float = 1.0


    @dataclass
    class More(Settings):
        extra: int = 0
    """
    modules, _ = _modules(tmp_path, body)
    assert skipped_base_inits(modules) == []


# ---------------------------------------------------------------------------
# #588
# ---------------------------------------------------------------------------

_COUNT = """
class CountCondition(BaseCondition):
    _count: int
    _limit: int

    def __init__(self, limit: int, *, label: str) -> None:
        super().__init__(label=label)
        self._limit = limit

    def check(self, world: carla.World, elapsed: float) -> ScenarioResult | None:
        self._count += 1
        if self._count > self._limit:
            return ScenarioResult(passed=True, message="c", elapsed_seconds=elapsed)
        return None
"""


def test_an_attribute_declared_and_never_assigned_is_refused(tmp_path: Path) -> None:
    modules, path = _modules(tmp_path, _COUNT)
    problem = _only(unassigned_attributes(modules))
    assert problem.path == str(path)
    assert problem.line == _line_of(path, "_count: int")
    assert "CountCondition._count is declared but never assigned" in (problem.message)
    assert "assign self._count in CountCondition.__init__" in problem.message


@pytest.mark.parametrize(
    "assignment",
    [
        "self._count = 0",
        "self._count: int = 0",
        "self._count, self._other = 0, 1",
        "setattr(self, '_count', 0)",
        "object.__setattr__(self, '_count', 0)",
        "[self._count] = [0]",
    ],
)
def test_an_assignment_of_any_shape_counts(tmp_path: Path, assignment: str) -> None:
    body = _COUNT.replace(
        "        self._limit = limit\n",
        f"        self._limit = limit\n        {assignment}\n",
    )
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("    _count: int\n", "    _count: int = 0\n"),
        ("    _count: int\n", "    _count: ClassVar[int]\n"),
        ("    _count: int\n", "    _count: int\n    _count = 0\n"),
        (
            "        self._count += 1\n",
            "        n = getattr(self, 'n', 0)\n        setattr(self, 'n' + '', n)\n",
        ),
        (
            "    def check(",
            "    def __getattr__(self, name: str) -> int:\n        return 0\n\n"
            "    def check(",
        ),
    ],
    ids=["default", "classvar", "class-attribute", "computed-setattr", "getattr"],
)
def test_what_the_rule_cannot_prove_unassigned_is_left_alone(
    tmp_path: Path, old: str, new: str
) -> None:
    body = _COUNT.replace(old, new)
    if "ClassVar" in new:
        body = "from typing import ClassVar\n" + body
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


def test_an_attribute_no_method_reads_is_left_alone(tmp_path: Path) -> None:
    body = _COUNT.replace("        self._count += 1\n", "").replace(
        "if self._count > self._limit", "if self._limit > 0"
    )
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


def test_a_subclass_or_another_module_assigning_it_counts(tmp_path: Path) -> None:
    body = (
        _COUNT
        + """

class Counting(CountCondition):
    def reset(self) -> None:
        self._count = 0
"""
    )
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []
    more = {"factory": "def make(c: object) -> None:\n    c._count = 0\n"}
    modules, _ = _modules(tmp_path, _COUNT, more)
    assert unassigned_attributes(modules) == []


def test_a_read_in_a_subclass_counts(tmp_path: Path) -> None:
    body = """
    class Base:
        _total: float

        def __init__(self) -> None:
            self._seen = 0


    class Sub(Base):
        def total(self) -> float:
            return self._total
    """
    modules, path = _modules(tmp_path, body)
    problem = _only(unassigned_attributes(modules))
    assert problem.line == _line_of(path, "_total: float")
    assert "Base._total" in problem.message


@pytest.mark.parametrize(
    "decorated",
    [
        "from dataclasses import dataclass\n\n\n@dataclass\nclass Holder:\n",
        "from typing import NamedTuple\n\n\nclass Holder(NamedTuple):\n",
        "from typing import Protocol\n\n\nclass Holder(Protocol):\n",
        "from typing import TypedDict\n\n\nclass Holder(TypedDict):\n",
    ],
    ids=["dataclass", "namedtuple", "protocol", "typeddict"],
)
def test_fields_of_dataclasses_and_typed_records_are_left_alone(
    tmp_path: Path, decorated: str
) -> None:
    body = decorated + (
        "    value: int\n\n    def doubled(self) -> int:\n"
        "        return self.value * 2\n"
    )
    if "TypedDict" in decorated:
        body = decorated + "    value: int\n"
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


def test_an_attribute_a_framework_base_assigns_is_left_alone(tmp_path: Path) -> None:
    # BaseCondition.__init__ assigns label; declaring it again is fine.
    body = (
        _COUNT.replace("    _count: int\n", "    label: str\n")
        .replace("self._count += 1", "print(self.label)")
        .replace("if self._count", "if self._limit")
    )
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


def test_a_class_of_a_third_party_base_is_left_alone(tmp_path: Path) -> None:
    body = """
    import threading


    class Worker(threading.Thread):
        _count: int

        def bump(self) -> None:
            self._count += 1
    """
    modules, _ = _modules(tmp_path, body)
    assert unassigned_attributes(modules) == []


# ---------------------------------------------------------------------------
# The framework itself
# ---------------------------------------------------------------------------


def test_the_framework_modules_pass_both_rules() -> None:
    # The library check gives the package's modules the same rules, with the
    # whole package as the framework: none of its classes may be refused.
    from autoware_carla_scenario.typecheck.check import (
        _class_problems,
        _read_module,
        _Sources,
    )
    from autoware_carla_scenario.typecheck.library import package_modules

    sources = _Sources()
    for name, path in sorted(package_modules().items()):
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - an optional dependency is missing
            continue
        sources.modules[name] = _read_module(path)
    assert len(sources.modules) > 100
    assert _class_problems(sources, library=True) == []
