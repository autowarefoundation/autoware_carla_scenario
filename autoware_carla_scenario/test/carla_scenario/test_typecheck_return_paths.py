"""The flow analysis of :mod:`.return_paths` (#529, #2513): pure Python, no
Codon.  Every function the tests expect refused, mypy refuses too ("Missing
return statement"); every one they expect accepted, mypy accepts."""

from __future__ import annotations

import ast
import sys
import textwrap

import pytest

from autoware_carla_scenario.typecheck.return_paths import return_problems


def _refused(source: str, **others: str) -> list[str]:
    """The functions of *source* (module ``m``) refused for a path that
    returns nothing; *others* are more modules, by name."""
    modules = {
        name: (f"{name.replace('.', '/')}.py", ast.parse(textwrap.dedent(text)))
        for name, text in {"m": source, **others}.items()
    }
    return [
        p.message.split("()")[0]
        for p in return_problems(modules)
        if p.path == "m.py" and "returns nothing" in p.message
    ]


# ---------------------------------------------------------------------------
# match: a guard narrows nothing
# ---------------------------------------------------------------------------


def test_a_guarded_wildcard_is_not_exhaustive() -> None:
    assert _refused(
        """
        def f(x: int, y: bool) -> int:
            match x:
                case _ if y:
                    return 1
        """
    ) == ["f"]


def test_a_guarded_capture_after_a_literal_of_an_int_is_not_exhaustive() -> None:
    assert _refused(
        """
        def f(x: int, y: int) -> int:
            match x:
                case 1 | 2:
                    return 1
                case n if n > y:
                    return 2
        """
    ) == ["f"]


def test_an_unguarded_wildcard_after_a_guarded_one_is_exhaustive() -> None:
    assert (
        _refused(
            """
            def f(x: int, y: bool) -> int:
                match x:
                    case _ if y:
                        return 1
                    case _:
                        return 2
            """
        )
        == []
    )


@pytest.mark.parametrize(
    "source",
    [
        # a class pattern may exhaust the subject's type (mypy accepts)
        """
        def f(x: int, y: bool) -> int:
            match x:
                case int():
                    return 1
                case _ if y:
                    return 2
        """,
        # literals may exhaust a Literal / an enum / a bool
        """
        from typing import Literal
        def f(x: Literal[1, 2], y: bool) -> int:
            match x:
                case 1 | 2:
                    return 1
                case _ if y:
                    return 2
        """,
        """
        def f(x: bool, y: bool) -> int:
            match x:
                case True:
                    return 1
                case False:
                    return 0
                case _ if y:
                    return 2
        """,
        # an unannotated subject: not known
        """
        def f(x, y: bool) -> int:
            match x:
                case 1:
                    return 1
                case _ if y:
                    return 2
        """,
    ],
)
def test_a_match_that_may_be_exhaustive_is_not_refused(source: str) -> None:
    assert _refused(source) == []


# ---------------------------------------------------------------------------
# Calls that never return: by what they stand for, not by their name
# ---------------------------------------------------------------------------


def test_a_method_of_another_class_declared_noreturn_ends_no_path() -> None:
    assert _refused(
        """
        from typing import NoReturn
        class A:
            def stop(self) -> NoReturn:
                raise SystemExit
        class B:
            def stop(self) -> None:
                pass
        def f(b: B, c: bool) -> int:
            if c:
                return 1
            b.stop()
        def g(a: A, c: bool) -> int:
            if c:
                return 1
            a.stop()
        """
    ) == ["f"]


def test_a_method_of_the_class_named_like_pytest_fail_ends_no_path() -> None:
    assert _refused(
        """
        class S:
            def fail(self, why: str) -> None:
                self.why = why
            def judge(self, x: int) -> bool:
                if x:
                    return True
                self.fail("no")
        """
    ) == ["S.judge"]


def test_noreturn_methods_of_self_and_its_bases_end_a_path() -> None:
    assert (
        _refused(
            """
            from typing import NoReturn
            class Base:
                def abort(self, why: str) -> NoReturn:
                    raise RuntimeError(why)
            class S(Base):
                def judge(self, x: int) -> bool:
                    if x:
                        return True
                    self.abort("no")
                def again(self, x: int) -> bool:
                    if x:
                        return True
                    super().abort("no")
                @classmethod
                def build(cls, x: int) -> int:
                    if x:
                        return 1
                    cls.abort(cls(), "no")
            """
        )
        == []
    )


def test_a_method_of_a_class_from_elsewhere_is_judged_by_its_name() -> None:
    # The base is not one of the modules: its `fail` may be unittest's.
    assert (
        _refused(
            """
            import unittest
            class T(unittest.TestCase):
                def check(self, x: int) -> int:
                    if x:
                        return 1
                    self.fail("no")
            def g(parser, x: int) -> int:
                if x:
                    return 1
                parser.exit(2)
            """
        )
        == []
    )


def test_a_module_level_noreturn_function_ends_a_path_through_imports() -> None:
    assert (
        _refused(
            """
            from .helpers import die
            from . import helpers
            def f(x: int) -> int:
                if x:
                    return 1
                die()
            def g(x: int) -> int:
                if x:
                    return 1
                helpers.die()
            """,
            helpers="""
            from typing import NoReturn
            def die() -> NoReturn:
                raise SystemExit
            """,
        )
        == []
    )


def test_a_function_of_the_same_name_in_another_module_ends_no_path() -> None:
    assert _refused(
        """
        def stop() -> None:
            pass
        def f(x: int) -> int:
            if x:
                return 1
            stop()
        """,
        other="""
        from typing import NoReturn
        def stop() -> NoReturn:
            raise SystemExit
        """,
    ) == ["f"]


@pytest.mark.parametrize(
    "call",
    [
        "sys.exit(1)",
        "os._exit(1)",
        "os.abort()",
        "os.execv('a', ['a'])",
        "pytest.fail('no')",
        "pytest.skip('no')",
        "exit(1)",
        "quit()",
        "assert_never(x)",
        "typing.assert_never(x)",
        "sys_exit(1)",
    ],
)
def test_the_calls_that_never_return_end_a_path(call: str) -> None:
    source = f"""
    import os
    import sys
    import typing
    import pytest
    from typing import assert_never
    from sys import exit as sys_exit
    def f(x: int) -> int:
        if x:
            return 1
        {call}
    """
    assert _refused(source) == []


# ---------------------------------------------------------------------------
# Aliases of an Optional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias",
    [
        "MaybeF = Optional[float]",
        "MaybeF = float | None",
        'MaybeF = "Optional[float]"',
        "MaybeF: TypeAlias = Optional[float]",
        "Inner = Optional[float]\nMaybeF = Inner",
    ],
)
def test_an_alias_of_an_optional_is_an_optional(alias: str) -> None:
    source = (
        "from typing import Optional, TypeAlias\n"
        + alias
        + textwrap.dedent(
            """
            def f(x: int) -> MaybeF:
                if x:
                    return 1.0
            def g(x: int) -> "MaybeF":
                if x:
                    return 1.0
            def h(x: int) -> MaybeF | int:
                if x:
                    return 1.0
            """
        )
    )
    assert _refused(source) == []


def test_an_alias_imported_from_another_module_is_followed() -> None:
    assert (
        _refused(
            """
            from .types import MaybeF
            def f(x: int) -> MaybeF:
                if x:
                    return 1.0
            """,
            types="""
            from typing import Optional
            MaybeF = Optional[float]
            """,
        )
        == []
    )


def test_an_alias_of_a_plain_type_is_still_checked() -> None:
    assert _refused(
        """
        Speed = float
        def f(x: int) -> Speed:
            if x:
                return 1.0
        """
    ) == ["f"]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="`type X = ...` is 3.12+")
def test_a_type_statement_alias_of_an_optional_is_an_optional() -> None:
    assert (
        _refused(
            """
            type MaybeF = float | None
            type Opt[T] = T | None
            def f(x: int) -> MaybeF:
                if x:
                    return 1.0
            def g(x: int) -> Opt[float]:
                if x:
                    return 1.0
            """
        )
        == []
    )
    assert _refused(
        """
        type Speed = float
        def f(x: int) -> Speed:
            if x:
                return 1.0
        """
    ) == ["f"]


# ---------------------------------------------------------------------------
# contextlib.suppress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "import contextlib\n",
        "from contextlib import suppress as contextlib_suppress\n",
    ],
)
def test_a_return_inside_suppress_may_fall_through(header: str) -> None:
    manager = (
        "contextlib.suppress" if header.startswith("import") else "contextlib_suppress"
    )
    source = (
        header
        + "def f(x: str) -> int:\n"
        + f"    with {manager}(ValueError):\n"
        + "        return int(x)\n"
        + "def g(x: str) -> int:\n"
        + f"    with {manager}(ValueError):\n"
        + "        return int(x)\n"
        + "    return 0\n"
    )
    assert _refused(source) == ["f"]
