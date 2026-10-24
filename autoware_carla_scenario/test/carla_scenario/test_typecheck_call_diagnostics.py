"""The static check's messages for a wrong call or ``return``, in the author's
terms (#527, #528, #529, #530)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any

import pytest

from autoware_carla_scenario.typecheck import available_toolchain, typecheck_scenario
from autoware_carla_scenario.typecheck.call_diagnostics import (
    codon_type,
    is_internal_frame,
    plain_literal_types,
)
from autoware_carla_scenario.typecheck.return_paths import return_problems

from .test_typecheck_codon import _HEADER, _line_of, _only_error, _write_case

_HEADER_WITH_ACTION = _HEADER.replace(
    "    BaseScenario,\n", "    BaseScenario,\n    SetSpeedAction,\n", 1
)

needs_codon = pytest.mark.skipif(
    available_toolchain() is None, reason="no Codon compiler"
)

_TIMEOUT = (
    'self.register_fail_condition(TimeoutCondition(cfg.timeout_seconds, label="t"))\n'
)


def _check(tmp_path: Path, setup: str, extra: str = "") -> tuple[Any, Path]:
    scenario, config, path = _write_case(
        tmp_path, setup + _TIMEOUT, extra, header=_HEADER_WITH_ACTION
    )
    return typecheck_scenario(scenario, config), path


# ---------------------------------------------------------------------------
# A wrong call to a framework class (#527)
# ---------------------------------------------------------------------------

_CALLS = [
    (
        'self.register_pre_tick(SetSpeedAction(EGO_ROLE_NAME, "fast", label="s"))\n',
        '"fast"',
        "SetSpeedAction(): target_speed_kmh: expected float, got str",
    ),
    (
        'self.register_pre_tick(SetSpeedAction(EGO_ROLE_NAME, label="s"))\n',
        "SetSpeedAction(EGO_ROLE_NAME",
        "SetSpeedAction(): missing required argument 'target_speed_kmh'",
    ),
    (
        'self.register_pass_condition(TimeoutCondition(3.0, lable="p"))\n',
        'lable="p"',
        "TimeoutCondition(): 'lable' is an invalid keyword argument "
        "(did you mean 'label'?)",
    ),
    (
        'self.register_pass_condition(TimeoutCondition("3", label="p"))\n',
        '"3"',
        "TimeoutCondition(): timeout_seconds: expected float, got str",
    ),
    (
        'Lanelet2Pose(lanelet_id="460", s=0.0)\n',
        '"460"',
        "Lanelet2Pose(): lanelet_id: expected int, got str",
    ),
    (
        "Lanelet2Pose(lanelet=460, s=0.0)\n",
        "lanelet=460",
        "Lanelet2Pose(): 'lanelet' is an invalid keyword argument "
        "(did you mean 'lanelet_id'?)",
    ),
    (
        'self.register_pass_condition(SpeedCondition(EGO_ROLE_NAME, 3.0, rule=">", label="s"))\n',
        'rule=">"',
        "SpeedCondition(): rule: expected ComparisonRule, got str",
    ),
    (
        "self.register_pass_condition(TimeoutCondition(3.0, label='p'), 1)\n",
        ", 1)",
        "register_pass_condition(): takes at most 1 positional argument, got 2",
    ),
]


@needs_codon
@pytest.mark.parametrize(("setup", "needle", "message"), _CALLS)
def test_a_wrong_call_names_the_parameter(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    result, path = _check(tmp_path, setup)
    error = _only_error(result)
    assert message in error.message, result.format()
    assert "self" not in error.message.split(";")[0].replace("self.", "")
    assert "no function" not in error.message
    assert error.line == _line_of(path, needle)
    # The column is the argument's, not the start of the call.
    source = path.read_text().splitlines()[error.line - 1]
    if "invalid keyword" in message:
        assert source[error.column - 1 :].startswith(needle), result.format()
    elif "expected" in message:
        # The value of the argument, not the call or the keyword.
        assert source[error.column - 1 :].startswith(
            needle.split("=", 1)[-1]
        ), result.format()


@needs_codon
def test_a_missing_label_shows_no_model_frames(tmp_path: Path) -> None:
    result, _ = _check(
        tmp_path, "self.register_pass_condition(TimeoutCondition(3.0))\n"
    )
    error = _only_error(result)
    assert error.message == "TimeoutCondition(): label is a required keyword argument"
    assert error.trace == ()
    assert ".codon" not in result.format()
    assert "_Required" not in result.format()


@needs_codon
def test_too_many_arguments_say_what_is_keyword_only(tmp_path: Path) -> None:
    setup = (
        "self.register_pre_tick(SetSpeedAction(EGO_ROLE_NAME, 10.0, None, None, "
        'None, "s", True))\n'
    )
    result, _ = _check(tmp_path, setup)
    error = _only_error(result)
    assert ".codon" not in result.format()
    assert "_keyword_only" not in result.format()
    assert "label" in error.message


@needs_codon
def test_a_helper_called_with_too_many_arguments_names_its_signature(
    tmp_path: Path,
) -> None:
    result, path = _check(
        tmp_path,
        'make_other(1, "a")\n',
        "\n\ndef make_other(n: int) -> str:\n    return str(n)\n",
    )
    error = _only_error(result)
    assert "make_other(): takes at most 1 positional argument, got 2" in error.message
    assert "make_other takes (n: int)" in error.message
    assert error.line == _line_of(path, "make_other(1")


# ---------------------------------------------------------------------------
# A return of the wrong type (#528)
# ---------------------------------------------------------------------------


@needs_codon
@pytest.mark.parametrize(
    ("declared", "value", "message"),
    [
        (
            "float",
            "'x'",
            "CaseScenario.val() returns str, but the function is declared -> float",
        ),
        (
            "int",
            "2.5",
            "CaseScenario.val() returns float, but the function is declared -> int",
        ),
        (
            "str",
            "5",
            "CaseScenario.val() returns int, but the function is declared -> str",
        ),
    ],
)
def test_a_wrong_return_says_what_is_returned_and_declared(
    tmp_path: Path, declared: str, value: str, message: str
) -> None:
    result, path = _check(
        tmp_path,
        "ok = self.val()\n",
        f"\n    def val(self) -> {declared}:\n        return {value}\n",
    )
    error = _only_error(result)
    assert error.message == message, result.format()
    assert error.line == _line_of(path, f"return {value}")
    assert "does not match expected type" not in error.message


@needs_codon
def test_setup_returning_a_value_is_refused_in_the_authors_terms(
    tmp_path: Path,
) -> None:
    result, path = _check(tmp_path, "return 5\n")
    error = _only_error(result)
    assert error.message == (
        "CaseScenario.setup() returns int, but the function is declared -> None"
    )
    assert error.line == _line_of(path, "return 5")


@needs_codon
def test_a_return_of_none_where_a_type_is_declared(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "ok = self.val()\n",
        "\n    def val(self) -> str:\n        return None\n",
    )
    messages = [d.message for d in result.diagnostics]
    assert (
        "CaseScenario.val() returns None, but the function is declared -> str"
        in messages
    )
    assert not any("test the value for None" in m for m in messages), result.format()


@needs_codon
def test_a_helper_function_with_a_wrong_return(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "make_other(1)\n",
        "\n\ndef make_other(n: int) -> str:\n    return n\n",
    )
    error = _only_error(result)
    assert error.message == (
        "make_other() returns int, but the function is declared -> str"
    )


# ---------------------------------------------------------------------------
# Codon's words in a message (#530)
# ---------------------------------------------------------------------------


@needs_codon
@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (
            'if cfg.timeout_seconds > "3":\n    pass\n',
            "unsupported operand type(s) for >: 'float' and 'str'",
        ),
        (
            'y = cfg.timeout_seconds + "s"\n',
            "unsupported operand type(s) for +: 'float' and 'str'",
        ),
    ],
)
def test_a_literal_string_type_is_not_quoted_twice(
    tmp_path: Path, setup: str, message: str
) -> None:
    result, _ = _check(tmp_path, setup)
    error = _only_error(result)
    assert error.message == message, result.format()


@needs_codon
def test_a_literal_none_is_not_told_to_be_tested(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path, 'self.register_pass_condition(TimeoutCondition(None, label="p"))\n'
    )
    error = _only_error(result)
    assert error.message == (
        "argument 1 of TimeoutCondition() does not take None, it takes float "
        "(mypy: arg-type)"
    )
    assert "test the value" not in error.message


# ---------------------------------------------------------------------------
# A declared return type Codon does not enforce (#529)
# ---------------------------------------------------------------------------


@needs_codon
def test_a_bool_function_returning_a_literal_str_is_refused(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "ok = self.flag()\n",
        "\n    def flag(self) -> bool:\n        return 'x'\n",
    )
    error = _only_error(result)
    assert error.message.startswith(
        "CaseScenario.flag() returns str, but the function is declared -> bool"
    )
    assert error.line == _line_of(path, "return 'x'")


@needs_codon
def test_a_float_function_with_a_path_returning_nothing_is_refused(
    tmp_path: Path,
) -> None:
    result, path = _check(
        tmp_path,
        "v = self.val()\n",
        "\n    def val(self) -> float:\n"
        "        if self._config.timeout_seconds > 1.0:\n"
        "            return 1.0\n",
    )
    error = _only_error(result)
    assert error.message.startswith(
        "CaseScenario.val() has a path that returns nothing, but the function is "
        "declared -> float"
    )
    assert error.line == _line_of(path, "def val")


_VALID = '''
import sys
from typing import NoReturn, Optional, Protocol, overload
from abc import abstractmethod


def fail() -> NoReturn:
    raise SystemExit(1)


def a(x: int) -> float:
    while True:
        if x:
            return 1.0


def b(x: int) -> float:
    if x:
        return 1.0
    else:
        raise ValueError(x)


def c(x: int) -> str:
    for i in range(x):
        if i:
            return "a"
    raise ValueError


def d(x: int) -> float:
    if x:
        return 1.0
    sys.exit(1)


def e(x: int) -> float:
    if x:
        return 1.0
    fail()


def f(x: int) -> Optional[float]:
    if x:
        return 1.0


def g(x: int) -> float | None:
    if x:
        return 1.0


def h(x: int) -> float:
    try:
        return 1.0
    except ValueError:
        raise
    finally:
        print(x)


def i(x: int) -> float:
    with open("f") as fh:
        return float(fh.read())


def j(x: int) -> int:
    match x:
        case 1:
            return 1
        case _:
            raise ValueError


def k(x: int) -> int:
    assert False, "never"


def l(x: int) -> bool:
    return x > 1


def m(x: int) -> None:
    return None


def n(x: int):
    if x:
        return 1


class P(Protocol):
    def q(self) -> float: ...

    @abstractmethod
    def r(self) -> float:
        """Doc."""

    def s(self) -> int:
        raise NotImplementedError


def t(x: int) -> bool:
    while x:
        x -= 1
        if x == 3:
            break
    else:
        return False
    return True


def u(x: int) -> int:
    try:
        return 1
    except ValueError:
        return 2


def v(x: int) -> "Optional[int]":
    pass
'''


def test_valid_returns_are_not_flagged() -> None:
    tree = ast.parse(textwrap.dedent(_VALID))
    assert return_problems({"m": ("m.py", tree)}) == []


_INVALID = """
def a(x: int) -> float:
    if x:
        return 1.0


def b(x: int) -> float:
    while x:
        return 1.0


def c(x: int) -> str:
    for i in range(x):
        return "a"


def d(x: int) -> float:
    while True:
        if x:
            break


def e(x: int) -> float:
    try:
        return 1.0
    except ValueError:
        pass


def f(x: int) -> bool:
    return f"{x}"


def g(x: int) -> bool:
    return 1


def h(x: int) -> int:
    print(x)
"""


def test_invalid_returns_are_flagged() -> None:
    tree = ast.parse(textwrap.dedent(_INVALID))
    found = return_problems({"m": ("m.py", tree)})
    assert [p.message.split("(")[0] for p in found] == [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
    ], found


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


def test_literal_string_types_read_as_str() -> None:
    assert plain_literal_types("for >: 'float' and ''3''") == "for >: 'float' and 'str'"
    assert plain_literal_types("'a' and 'b'") == "'a' and 'b'"
    assert (
        plain_literal_types("unsupported 'x' and ''''") == "unsupported 'x' and 'str'"
    )


@pytest.mark.parametrize(
    "frame",
    [
        "during the realization of _keyword_only(extra: Tuple[str], where: 'X') [actions.codon:454]",
        "during the realization of _required_str(value: _Required, where: 'T') [conditions.codon:636]",
        "during the realization of __init__(self: T, *_kw: Tuple, label: _Required) [m.py:28]",
    ],
)
def test_model_frames_are_internal(frame: str) -> None:
    assert is_internal_frame(frame)


def test_the_authors_frames_are_kept() -> None:
    assert not is_internal_frame("during the realization of val(self: Scn) [m.py:29]")


def test_codon_types_read_as_python_types() -> None:
    assert codon_type("Optional[NoneType]") == "None"
    assert codon_type("Optional[float]") == "float | None"
    assert codon_type("List[int]") == "list[int]"


_VALID_FLOW = """
import os
import pytest
from typing import Iterator


def nested_loop_break(x) -> int:
    while True:
        for i in x:
            if i:
                break
        return 1


def exhaustive_match(x) -> int:
    match x:
        case 1:
            return 1
        case 2:
            return 2


def pytest_fail() -> int:
    pytest.fail("no")


def os_exit() -> int:
    os._exit(1)


def nested_helpers() -> bool:
    def inner():
        return "s"

    def gen():
        yield 1

    return True
"""


def test_flow_probes_of_valid_code_are_not_flagged() -> None:
    tree = ast.parse(_VALID_FLOW)
    assert return_problems({"m": ("m.py", tree)}) == []


def test_a_return_of_a_nested_function_is_not_the_outer_functions(
    tmp_path: Path,
) -> None:
    source = (
        "def outer() -> float:\n"
        "    def inner():\n"
        '        return "s"\n'
        "    return 1.0\n"
        "g = lambda: 1\n"
    )
    path = tmp_path / "m.py"
    path.write_text(source)
    from autoware_carla_scenario.typecheck.call_diagnostics import (
        _Files,
        return_message,
    )

    message = "'float' does not match expected type 'str'"
    files = _Files()
    assert return_message(message, str(path), 3, 9, files) is None
    assert "outer() returns str" in (
        return_message(message, str(path), 4, 5, files) or ""
    )


def test_an_annotated_assignment_is_reported_declared_type_last(
    tmp_path: Path,
) -> None:
    from autoware_carla_scenario.typecheck.call_diagnostics import (
        _Files,
        annotated_message,
        json_hint,
    )

    path = tmp_path / "m.py"
    path.write_text(
        "def f(doc):\n    speed: float = doc['speed']\n    n: int = 'x'\n"
        "    m: int | None = 1.5\n    k: int\n"
    )
    files = _Files()
    got = annotated_message(
        "'float' does not match expected type 'JSONValue'", str(path), 2, 5, files
    )
    assert got == "speed is assigned a JSONValue, but is declared float"
    assert "convert it where you use it" in json_hint(got)
    assert annotated_message(
        "'int' does not match expected type 'str'", str(path), 3, 5, files
    ) == ("n is assigned a str, but is declared int")
    assert annotated_message(
        "'Optional[int]' does not match expected type 'float'", str(path), 4, 5, files
    ) == ("m is assigned a float, but is declared int | None")
    # Another type on that line, or not an assignment: Codon's text stays.
    assert (
        annotated_message(
            "'str' does not match expected type 'int'", str(path), 2, 5, files
        )
        is None
    )
    assert annotated_message("not a mismatch", str(path), 2, 5, files) is None
    assert json_hint("an unrelated message") == "an unrelated message"
    assert "read-only" in json_hint("'JSONValue' object has no attribute '__setitem__'")


# ---------------------------------------------------------------------------
# Own callables, `X | None` returns, a forgotten `self` (#539, #540, #542):
# the rewrites on their own, with Codon's messages written out
# ---------------------------------------------------------------------------

_OWN = """\
from dataclasses import dataclass


def f(a: int, b: int) -> int:
    return a


def g(a: int, *, b: int = 2) -> int:
    return a + b


def m(a: int, b: int, c: str) -> int:
    return a


def opt(a: int | None = None) -> int:
    return 0


class U:
    @staticmethod
    def k(a: int, b: float) -> float:
        return a + b


class P:
    def __init__(self, a: int, b: float = 0.0) -> None:
        self.a = a


@dataclass
class D:
    a: int
    b: float = 0.0


class S:
    flag_v: bool = True

    def h(self, a: int) -> int:
        return a

    def flag(self) -> bool:
        return True

    @staticmethod
    def st() -> bool:
        return flag()

    def run(self, n: int, s: str) -> None:
        self.h("x")
        f(n, s)
        f(
            1,
            "x",
        )
        g(1, b="s")
        U.k(1, "x")
        P("x", 1.0)
        D(a="s")
        D(a=1, c=2)
        D(a=1, bb=2)
        m(1, 2, "ok")
        m("x", "y", 3)
        m("x", 2, "c")
        f(*[1, 2])
        self.h(a=-1.5)
        opt("s")
        if flag():
            pass
        if self.flag:
            pass
        if not self.flag and n:
            pass
        x = flag_v
"""

_INT = "'str' does not match expected type 'int'"


def _refiner(tmp_path: Path, source: str) -> tuple[Any, Path, list[str]]:
    import importlib.util

    from autoware_carla_scenario.typecheck.call_diagnostics import Refiner

    path = tmp_path / "own_module.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("own_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["own_module"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules["own_module"]
    refiner = Refiner(
        lambda name, _path: getattr(module, name, None),
        lambda name, _path: getattr(module, name, None),
    )
    return refiner, path, source.splitlines()


def _at(lines: list[str], needle: str, occurrence: int = 1) -> tuple[int, int]:
    seen = 0
    for number, text in enumerate(lines, start=1):
        if needle in text:
            seen += 1
            if seen == occurrence:
                return number, text.index(needle) + 1
    raise AssertionError(needle)


def _refine(
    tmp_path: Path, source: str, message: str, needle: str, occurrence: int = 1
) -> tuple[str, int | None, int | None]:
    refiner, path, lines = _refiner(tmp_path, source)
    line, column = _at(lines, needle, occurrence)
    return refiner.refine(message, str(path), line, column)


@pytest.mark.parametrize(
    ("call", "occurrence", "message", "argument", "got_at"),
    [
        (
            'f(\n            1,\n            "x",',
            3,
            "f(): b: expected int, got str",
            '"x"',
            2,
        ),
        ('g(1, b="s")', 1, "g(): b: expected int, got str", '"s"', 1),
        ('U.k(1, "x")', 1, "U.k(): b: expected float, got str", '"x"', 1),
        ('P("x", 1.0)', 1, "P(): a: expected int, got str", '"x"', 1),
        ('D(a="s")', 1, "D(): a: expected int, got str", '"s"', 1),
        ('self.h("x")', 1, "S.h(): a: expected int, got str", '"x"', 1),
        ("f(n, s)", 1, "f(): b: expected int, got str", "s)", 1),
    ],
)
def test_a_wrong_argument_of_an_own_callable_is_named(
    tmp_path: Path,
    call: str,
    occurrence: int,
    message: str,
    argument: str,
    got_at: int,
) -> None:
    del got_at
    first = call.splitlines()[0]
    raw = "'float' does not match expected type 'int'" if "U.k" in call else _INT
    if "U.k" in call:
        raw = "'str' does not match expected type 'float'"
    refined, line, column = _refine(tmp_path, _OWN, raw, first, occurrence)
    lines = _OWN.splitlines()
    assert refined == message
    assert line is not None and column is not None
    assert lines[line - 1][column - 1 :].startswith(argument.rstrip(")"))


def test_a_multi_line_call_points_at_the_argument_line(tmp_path: Path) -> None:
    refiner, path, lines = _refiner(tmp_path, _OWN)
    line, column = _at(lines, "f(", 3)
    assert lines[line - 1].strip() == "f("
    message, at_line, at_column = refiner.refine(_INT, str(path), line, column)
    assert message == "f(): b: expected int, got str"
    assert (at_line, at_column) == (line + 2, column + 4)


@pytest.mark.parametrize(
    ("needle", "occurrence", "message"),
    [
        # nothing tells which argument: two certain strings for two ints
        ('m("x", "y", 3)', 1, _INT),
        # a variable of unknown type next to a parameter of the same type
        ("f(n, s)", 1, "'float' does not match expected type 'int'"),
        # the starred call, the unannotated callee, a call that is not ours
        ("f(*[1, 2])", 1, _INT),
        # the types in the message are not the argument's
        ('opt("s")', 1, "'float' does not match expected type 'str'"),
        ("flag()", 1, _INT),
    ],
)
def test_an_argument_that_cannot_be_told_keeps_codons_message(
    tmp_path: Path, needle: str, occurrence: int, message: str
) -> None:
    refined = _refine(tmp_path, _OWN, message, needle, occurrence)
    assert refined[0] == message


def test_a_mismatch_off_the_start_of_a_call_is_not_rewritten(tmp_path: Path) -> None:
    refiner, path, lines = _refiner(tmp_path, _OWN)
    line, column = _at(lines, '"x", 1.0)')
    assert refiner.refine(_INT, str(path), line, column) == (_INT, line, column)


def test_two_arguments_of_the_expected_type_are_not_guessed(tmp_path: Path) -> None:
    source = (
        "def f(a: int, b: int) -> int:\n    return a\n\n\n"
        "def run(n) -> None:\n    f(n, 'x')\n    f('x', 'y')\n"
    )
    refined = _refine(tmp_path, source, _INT, "f(n,")
    assert refined[0] == _INT  # n may be the one
    refined = _refine(tmp_path, source, _INT, "f('x', 'y')")
    assert refined[0] == _INT  # both are strings


def test_a_parameter_that_is_assigned_is_not_a_certain_type(tmp_path: Path) -> None:
    source = (
        "def f(a: int) -> int:\n    return a\n\n\n"
        "def run(s: str) -> None:\n    s = 3\n    f(s)\n"
    )
    assert _refine(tmp_path, source, _INT, "f(s)")[0] == _INT
    source = source.replace("    s = 3\n", "")
    assert _refine(tmp_path, source, _INT, "f(s)")[0] == (
        "f(): a: expected int, got str"
    )


def test_an_unknown_keyword_names_the_class_and_the_signature(tmp_path: Path) -> None:
    refiner, path, lines = _refiner(tmp_path, _OWN)
    line, column = _at(lines, "c=2")
    message, at_line, at_column = refiner.refine(
        "'c' is an invalid keyword argument for __init__()", str(path), line, column
    )
    assert message == (
        "D(): 'c' is an invalid keyword argument; D takes (a: int, b: float = 0.0)"
    )
    assert (at_line, at_column) == (line, column)


def test_an_unknown_keyword_close_to_a_parameter_gets_a_suggestion(
    tmp_path: Path,
) -> None:
    line, column = _at(_OWN.splitlines(), "bb=2")
    refined = _refine(
        tmp_path, _OWN, "'bb' is an invalid keyword argument for __init__()", "bb=2"
    )
    assert refined[0].startswith("D(): 'bb' is an invalid keyword argument (did you")
    assert (refined[1], refined[2]) == (line, column)


@pytest.mark.parametrize(
    ("annotation", "value", "message", "shown"),
    [
        (
            "float | None",
            "'s'",
            "'Optional[float]' does not match expected type 'Optional[str]'",
            "str",
        ),
        (
            "int | None",
            "1.5",
            "'Optional[int]' does not match expected type 'Optional[float]'",
            "float",
        ),
        (
            "list[int] | None",
            "4",
            "'Optional[List[int]]' does not match expected type 'Optional[int]'",
            "int",
        ),
        (
            "Optional[float]",
            "'s'",
            "'Optional[float]' does not match expected type 'Optional[str]'",
            "str",
        ),
        (
            "typing.Optional[float]",
            "'s'",
            "'Optional[float]' does not match expected type 'Optional[str]'",
            "str",
        ),
        (
            "Union[float, None]",
            "'s'",
            "'Optional[float]' does not match expected type 'Optional[str]'",
            "str",
        ),
        (
            "None | float",
            "'s'",
            "'Optional[float]' does not match expected type 'Optional[str]'",
            "str",
        ),
        ("float", "'s'", "'float' does not match expected type 'str'", "str"),
    ],
)
def test_an_optional_return_is_told_in_the_authors_terms(
    tmp_path: Path, annotation: str, value: str, message: str, shown: str
) -> None:
    source = (
        "from typing import Optional, Union\nimport typing\n\n\n"
        f"def f(x: int) -> {annotation}:\n    return {value}\n"
    )
    refined = _refine(tmp_path, source, message, f"return {value}")
    assert (
        refined[0]
        == f"f() returns {shown}, but the function is declared -> {annotation}"
    )
    assert (refined[1], refined[2]) == (6, 5)


def test_a_method_return_names_its_class(tmp_path: Path) -> None:
    source = (
        "class C:\n    def check(self, w: int) -> int | None:\n        return True\n"
    )
    refined = _refine(
        tmp_path,
        source,
        "'Optional[int]' does not match expected type 'Optional[bool]'",
        "return True",
    )
    assert refined[0] == (
        "C.check() returns bool, but the function is declared -> int | None"
    )
    refined = _refine(
        tmp_path,
        source.replace("int | None", "str"),
        "cannot typecheck 'return None'",
        "return True",
    )
    assert refined[0] == "C.check() returns None, but the function is declared -> str"


def test_an_optional_return_of_a_variable_keeps_its_optionality(
    tmp_path: Path,
) -> None:
    source = "def f(x: int | None) -> float | None:\n    return x\n"
    refined = _refine(
        tmp_path,
        source,
        "'Optional[float]' does not match expected type 'Optional[int]'",
        "return x",
    )
    assert (
        refined[0]
        == "f() returns int | None, but the function is declared -> float | None"
    )


def test_another_mismatch_on_a_return_line_is_left_alone(tmp_path: Path) -> None:
    source = "def f(x: int) -> float | None:\n    return g(x, 's')\n"
    message = "'Optional[int]' does not match expected type 'str'"
    assert _refine(tmp_path, source, message, "return g")[0] == message


def test_a_missing_return_names_the_class() -> None:
    tree = ast.parse(
        "class C:\n    def val(self) -> float:\n        x = 1\n\n\n"
        "def val() -> float:\n    x = 1\n\n\n"
        "class D:\n    def flag(self) -> bool:\n        return 'x'\n"
    )
    messages = [p.message for p in return_problems({"m": ("m.py", tree)})]
    assert any(
        m.startswith("C.val() has a path that returns nothing") for m in messages
    )
    assert any(m.startswith("val() has a path that returns nothing") for m in messages)
    assert any(m.startswith("D.flag() returns str") for m in messages)


def test_a_bare_call_of_a_method_gets_the_self_hint(tmp_path: Path) -> None:
    refined = _refine(tmp_path, _OWN, "name 'flag' is not defined", "flag()", 2)
    assert refined[0] == (
        "name 'flag' is not defined: S has a method flag(), did you mean `self.flag()`?"
    )
    line, column = _at(_OWN.splitlines(), "flag()", 2)
    assert (refined[1], refined[2]) == (line, column)


def test_a_bare_attribute_gets_the_self_hint(tmp_path: Path) -> None:
    refined = _refine(tmp_path, _OWN, "name 'flag_v' is not defined", "flag_v", 2)
    assert refined[0] == (
        "name 'flag_v' is not defined: S has an attribute flag_v, "
        "did you mean `self.flag_v`?"
    )


def test_a_static_method_and_a_module_function_get_no_self_hint(
    tmp_path: Path,
) -> None:
    message = "name 'flag' is not defined"
    # `return flag()` of the static method: there is no self
    assert _refine(tmp_path, _OWN, message, "return flag()")[0] == message
    # nothing called `nothing` in the class
    assert _refine(tmp_path, _OWN, "name 'nothing' is not defined", "flag()", 2)[0] == (
        "name 'nothing' is not defined"
    )


@pytest.mark.parametrize("statement", ["if self.flag:", "if not self.flag and n:"])
def test_a_method_used_as_a_value_is_told_to_be_called(
    tmp_path: Path, statement: str
) -> None:
    refined = _refine(
        tmp_path, _OWN, "'flag(S)' object has no attribute '__bool__'", statement
    )
    assert refined[0] == (
        "`self.flag` is a method, not its result (a method is always true): "
        "call it, `self.flag()`"
    )
    line, column = _at(_OWN.splitlines(), statement)
    assert refined[1] == line
    assert refined[2] == column + statement.index("self.flag")


def test_a_method_that_is_called_is_not_told_to_be_called(tmp_path: Path) -> None:
    message = "'flag(S)' object has no attribute '__bool__'"
    source = _OWN.replace("if self.flag:", "if self.flag():").replace(
        "if not self.flag and n:", "if not self.flag() and n:"
    )
    assert _refine(tmp_path, source, message, "if self.flag()")[0] == message


def test_a_method_value_of_an_unknown_class_keeps_codons_message(
    tmp_path: Path,
) -> None:
    message = "'flag(Other)' object has no attribute '__bool__'"
    assert _refine(tmp_path, _OWN, message, "if self.flag:")[0] == message


_HELPERS = """

def f(a: int, b: int) -> int:
    return a


def g(a: int, *, b: int = 2) -> int:
    return a + b


class U:
    @staticmethod
    def k(a: int, b: float) -> float:
        return a + b


class P:
    def __init__(self, a: int, b: float = 0.0) -> None:
        self.a = a


@dataclass
class D:
    a: int
    b: float = 0.0
"""


@needs_codon
@pytest.mark.parametrize(
    ("setup", "needle", "message"),
    [
        ("f(\n    1,\n    'x',\n)\n", "'x',", "f(): b: expected int, got str"),
        ("g(1, b='s')\n", "'s'", "g(): b: expected int, got str"),
        ("U.k(1, 'x')\n", "'x'", "U.k(): b: expected float, got str"),
        ("P('x', 1.0)\n", "'x'", "P(): a: expected int, got str"),
        ("D(a='s')\n", "'s'", "D(): a: expected int, got str"),
        (
            "D(a=1, c=2)\n",
            "c=2",
            "D(): 'c' is an invalid keyword argument; D takes (a: int, b: float = 0.0)",
        ),
    ],
)
def test_a_wrong_argument_of_an_own_callable_is_named_by_codon_check(
    tmp_path: Path, setup: str, needle: str, message: str
) -> None:
    result, path = _check(
        tmp_path, setup, "from dataclasses import dataclass\n" + _HELPERS
    )
    error = _only_error(result)
    assert error.message == message, result.format()
    assert error.line == _line_of(path, needle)
    source = path.read_text().splitlines()[error.line - 1]
    assert source[error.column - 1 :].startswith(needle.rstrip(","))


@needs_codon
def test_a_wrong_argument_of_an_own_method_names_the_class(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "self.h('x')\n",
        "\n    def h(self, a: int) -> int:\n        return a\n",
    )
    error = _only_error(result)
    assert error.message == "CaseScenario.h(): a: expected int, got str"
    assert error.line == _line_of(path, "self.h('x')")


@needs_codon
def test_right_arguments_of_own_callables_are_accepted(tmp_path: Path) -> None:
    result, _ = _check(
        tmp_path,
        "f(1, 2)\ng(1, b=3)\nU.k(1, 2.5)\nP(1)\nD(a=1, b=2.0)\n",
        "from dataclasses import dataclass\n" + _HELPERS,
    )
    assert result.ok, result.format()


@needs_codon
@pytest.mark.parametrize(
    ("declared", "body", "message"),
    [
        (
            "float | None",
            "return 's'",
            "returns str, but the function is declared -> float | None",
        ),
        (
            "int | None",
            "if x:\n            return None\n        return 1.5",
            "returns float, but the function is declared -> int | None",
        ),
        (
            "list[int] | None",
            "return 4",
            "returns int, but the function is declared -> list[int] | None",
        ),
        (
            "Optional[float]",
            "return 's'",
            "returns str, but the function is declared -> Optional[float]",
        ),
    ],
)
def test_a_wrong_return_of_an_optional_function(
    tmp_path: Path, declared: str, body: str, message: str
) -> None:
    result, path = _check(
        tmp_path,
        "ok = self.val(1)\n",
        f"\n    def val(self, x: int) -> {declared}:\n        {body}\n",
    )
    error = _only_error(result)
    assert error.message == f"CaseScenario.val() {message}", result.format()
    assert "does not match expected type" not in error.message


@needs_codon
def test_a_forgotten_self_before_a_method_call(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "if flag():\n    pass\n",
        "\n    def flag(self) -> bool:\n        return True\n",
    )
    error = _only_error(result)
    assert error.message == (
        "name 'flag' is not defined: CaseScenario has a method flag(), "
        "did you mean `self.flag()`?"
    )
    assert error.line == _line_of(path, "if flag()")


@needs_codon
def test_a_method_used_as_a_condition_is_told_to_be_called(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "if self.flag:\n    pass\n",
        "\n    def flag(self) -> bool:\n        return True\n",
    )
    error = _only_error(result)
    assert error.message.startswith("`self.flag` is a method, not its result")
    assert error.line == _line_of(path, "if self.flag")
    source = path.read_text().splitlines()[error.line - 1]
    assert source[error.column - 1 :].startswith("self.flag")


def test_a_local_that_shadows_the_callee_is_not_the_callee(tmp_path: Path) -> None:
    source = (
        "def f(a: int) -> int:\n    return a\n\n\n"
        "def run() -> None:\n    def f(a: str) -> int:\n        return 1\n"
        "    f('s')\n"
    )
    assert _refine(tmp_path, source, _INT, "f('s')")[0] == _INT
    source = source.replace(
        "    def f(a: str)", "    f = len\n    def h(a: str)"
    ).replace("        return 1\n    f('s')", "        return 1\n    f('s')")
    assert _refine(tmp_path, source, _INT, "f('s')")[0] == _INT


def test_a_lambda_parameter_is_not_the_enclosing_functions_parameter(
    tmp_path: Path,
) -> None:
    source = (
        "def f(a: int) -> int:\n    return a\n\n\n"
        "def run(s: str) -> None:\n    g = lambda s: f(s)\n    g(1)\n"
    )
    assert _refine(tmp_path, source, _INT, "f(s)")[0] == _INT


@pytest.mark.parametrize("decorator", ["staticmethod", "classmethod"])
def test_a_bare_call_of_a_static_or_class_method_keeps_its_parentheses(
    tmp_path: Path, decorator: str
) -> None:
    first = "cls" if decorator == "classmethod" else ""
    source = (
        "class S:\n"
        f"    @{decorator}\n    def flag({first}) -> bool:\n        return True\n\n"
        "    def run(self) -> None:\n        if flag():\n            pass\n"
    )
    assert _refine(tmp_path, source, "name 'flag' is not defined", "flag():")[0] == (
        "name 'flag' is not defined: S has a method flag(), did you mean `self.flag()`?"
    )


@pytest.mark.parametrize(
    ("expr", "flagged"),
    [
        ("'yes' if x else 'no'", True),
        ("[1] if x else [2]", True),
        ("'yes' if x else 1", False),
        ("'yes' if x else y", False),
        ("'yes' if x else None", False),
        ("'a' if x else ('b' if y else 'c')", True),
        ("True if x else False", False),
    ],
)
def test_a_conditional_of_literals_is_a_certain_return_type(
    expr: str, flagged: bool
) -> None:
    tree = ast.parse(f"def f(x: int, y: int) -> bool:\n    return {expr}\n")
    problems = return_problems({"m": ("m.py", tree)})
    assert bool(problems) is flagged, problems
