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
        ("float", "'x'", "val() returns str, but the function is declared -> float"),
        ("int", "2.5", "val() returns float, but the function is declared -> int"),
        ("str", "5", "val() returns int, but the function is declared -> str"),
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
    assert error.message == "setup() returns int, but the function is declared -> None"
    assert error.line == _line_of(path, "return 5")


@needs_codon
def test_a_return_of_none_where_a_type_is_declared(tmp_path: Path) -> None:
    result, path = _check(
        tmp_path,
        "ok = self.val()\n",
        "\n    def val(self) -> str:\n        return None\n",
    )
    messages = [d.message for d in result.diagnostics]
    assert "val() returns None, but the function is declared -> str" in messages
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
        "flag() returns str, but the function is declared -> bool"
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
        "val() has a path that returns nothing, but the function is declared -> float"
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
